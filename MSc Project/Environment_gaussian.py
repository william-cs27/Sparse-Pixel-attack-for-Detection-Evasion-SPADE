# Environment.py
import torch
import torch.nn as nn
import numpy as np
from utils import normalize
import torch.nn.functional as F
from PIL import Image

class Env:
    def __init__(self, classification_model, config):
        super().__init__()
        self.classification_model = classification_model
        self.pixel = config["attack_pixel"]
        self.config = config
        self.ori_prob = []
        self.ori_cls = []
        self.ori_box_num = []
        self.s = None
        self.init = False
        self.query_counts = np.zeros(config["ni"])

    def make_transformed_images(self, original_images, actions, allowed_pixels=None):
        """
        allowed_pixels: List of allowed (x, y) coords to attack in the image.
                        For batch, should be a list of lists (one per image).
        actions: For each image, a 6-dim vector: [mean_r, mean_g, mean_b, logstd_r, logstd_g, logstd_b]
        """
        RGB = original_images.shape[1]
        arr = []
        batch = original_images.shape[0]
        # No sigmoid needed for noise parameters
        scale = 255 if self.config["classifier"] in ["yolo", "ddq"] else 1

        # For batch mode, we handle allowed_pixels as a list of lists
        for i in range(batch):
            changed_image = original_images[i].float().clone()
            # Apply Gaussian noise to all allowed pixels
            if allowed_pixels is not None:
                coords_this_img = allowed_pixels[i]  # List of (x, y)
                if len(coords_this_img) > 0:
                    # Extract noise parameters
                    noise_means = actions[i, 0, :3]  # [r, g, b]  # Added 0 for compatibility with extra dim
                    noise_stds = F.softplus(actions[i, 0, 3:]) + 1e-6  # Ensure positive [r, g, b]  # Added 0

                    # Vectorize coordinates
                    xs = torch.tensor([xy[0] for xy in coords_this_img], device=self.config["device"])
                    ys = torch.tensor([xy[1] for xy in coords_this_img], device=self.config["device"])
                    num_pixels = len(coords_this_img)

                    # Generate noise: [3, num_pixels]
                    noise = torch.normal(
                        mean=noise_means.unsqueeze(1).expand(-1, num_pixels),
                        std=noise_stds.unsqueeze(1).expand(-1, num_pixels)
                    )

                    # Add noise to each channel
                    for c in range(3):
                        changed_image[c, ys, xs] += noise[c]

                    # Clip to valid range
                    changed_image = torch.clamp(changed_image, 0, scale)

            else:
                # Original fallback: but for noise, we could apply to whole image, but skip for consistency
                pass  # Or implement whole-image noise if desired

            arr.append(changed_image.unsqueeze(0))
        changed_images = torch.cat(arr, 0)
        return changed_images

    def yolo_step_not_sub(self, original_images, actions, bt, labels=None, probs=None, allowed_pixels=None, batch_indices=None):
        # Pass allowed_pixels from your calling code!
        changed_images = self.make_transformed_images(original_images.to(self.config["device"]), actions, allowed_pixels=allowed_pixels)
        with torch.no_grad():
            if self.init == True:
                labels = []
                probs = []
                for n, img in enumerate(original_images):
                    if batch_indices is not None:
                        self.query_counts[batch_indices[n]] += 1
                    result = self.classification_model(
                        img.detach().cpu().numpy().transpose(1,2,0),
                        imgsz=640,
                        conf=self.config["yolo_conf"]
                    )
                    probs.append(result[0].boxes.conf)
                    labels.append(result[0].boxes.cls)
                self.ori_prob = self.ori_prob + probs
                self.ori_cls = self.ori_cls + labels

            changed_images = changed_images.detach().cpu().numpy().transpose(0,2,3,1)
            prob_list = []
            cls_list = []
            temp_list = []
            for n, i in enumerate(changed_images):
                if batch_indices is not None:
                    self.query_counts[batch_indices[n]] += 1
                temp_list.append(i)
                results = self.classification_model(i, conf=self.config["yolo_conf"])
                prob_list.append(results[0].boxes.conf)
                cls_list.append(results[0].boxes.cls)

            rewards = []
            dif_list = []

            for i in range(len(cls_list)):
                size = max(torch.bincount(labels[i].long()).shape[0], torch.bincount(cls_list[i].long()).shape[0])
                temp = torch.bincount(labels[i].long(), minlength=size) - torch.bincount(cls_list[i].long(), minlength=size)
                dif = temp.sum()
                dif_list.append(dif)
                # Compute reward
                if (temp != 0).any():
                    indices = list(filter(lambda x: temp[x] > 0, range(len(temp))))
                    values = list(filter(lambda x: x > 0, temp))
                    for indice, value in zip(indices, values):
                        add_cls = torch.LongTensor([indice for _ in range(value)]).to(self.config["device"])
                        add_prob = torch.zeros(value).float().to(self.config["device"])
                        cls_list[i] = torch.cat((cls_list[i].long(), add_cls), dim=0)
                        prob_list[i] = torch.cat((prob_list[i], add_prob), dim=0)
                    indices = list(filter(lambda x: temp[x] < 0, range(len(temp))))
                    values = list(filter(lambda x: x < 0, temp))
                    for indice, value in zip(indices, values):
                        add_cls = torch.LongTensor([indice for _ in range(abs(value))]).to(self.config["device"])
                        add_prob = torch.zeros(abs(value)).float().to(self.config["device"])
                        labels[i] = torch.cat((labels[i].long(), add_cls), dim=0)
                        probs[i] = torch.cat((probs[i], add_prob), dim=0)
                reward = (probs[i][labels[i].sort()[1]].to('cpu') - prob_list[i][cls_list[i].sort()[1]].to('cpu')).sum() + dif
                rewards.append(reward)

        return torch.tensor(rewards).to(self.config["device"]), torch.tensor(dif_list), changed_images

    def yolo_step_disunity_not_sub(self, original_images, actions, bt, hws, labels=None, probs=None, allowed_pixels=None, batch_indices=None):
        # Similar to yolo_step_not_sub but handles different shapes by cropping before model inference
        changed_images = self.make_transformed_images(original_images.to(self.config["device"]), actions, allowed_pixels=allowed_pixels)
        with torch.no_grad():
            if self.init == True:
                labels = []
                probs = []
                for n, img in enumerate(original_images):
                    if batch_indices is not None:
                        self.query_counts[batch_indices[n]] += 1
                    cropped_img = img[:, :hws[n,0], :hws[n,1]].detach().cpu().numpy().transpose(1,2,0)
                    result = self.classification_model(
                        cropped_img,
                        imgsz=640,
                        conf=self.config["yolo_conf"]
                    )
                    probs.append(result[0].boxes.conf)
                    labels.append(result[0].boxes.cls)
                self.ori_prob = self.ori_prob + probs
                self.ori_cls = self.ori_cls + labels

            changed_images_full = changed_images.detach().cpu().numpy().transpose(0,2,3,1)
            prob_list = []
            cls_list = []
            changed_images = []  # Will return full perturbed
            for n, i in enumerate(changed_images_full):
                if batch_indices is not None:
                    self.query_counts[batch_indices[n]] += 1
                cropped_i = i[:hws[n,0], :hws[n,1], :]
                results = self.classification_model(cropped_i, conf=self.config["yolo_conf"])
                prob_list.append(results[0].boxes.conf)
                cls_list.append(results[0].boxes.cls)
                changed_images.append(i)  # Append full

            rewards = []
            dif_list = []

            for i in range(len(cls_list)):
                size = max(torch.bincount(labels[i].long()).shape[0], torch.bincount(cls_list[i].long()).shape[0])
                temp = torch.bincount(labels[i].long(), minlength=size) - torch.bincount(cls_list[i].long(), minlength=size)
                dif = temp.sum()
                dif_list.append(dif)
                # Compute reward
                if (temp != 0).any():
                    indices = list(filter(lambda x: temp[x] > 0, range(len(temp))))
                    values = list(filter(lambda x: x > 0, temp))
                    for indice, value in zip(indices, values):
                        add_cls = torch.LongTensor([indice for _ in range(value)]).to(self.config["device"])
                        add_prob = torch.zeros(value).float().to(self.config["device"])
                        cls_list[i] = torch.cat((cls_list[i].long(), add_cls), dim=0)
                        prob_list[i] = torch.cat((prob_list[i], add_prob), dim=0)
                    indices = list(filter(lambda x: temp[x] < 0, range(len(temp))))
                    values = list(filter(lambda x: x < 0, temp))
                    for indice, value in zip(indices, values):
                        add_cls = torch.LongTensor([indice for _ in range(abs(value))]).to(self.config["device"])
                        add_prob = torch.zeros(abs(value)).float().to(self.config["device"])
                        labels[i] = torch.cat((labels[i].long(), add_cls), dim=0)
                        probs[i] = torch.cat((probs[i], add_prob), dim=0)
                reward = (probs[i][labels[i].sort()[1]].to('cpu') - prob_list[i][cls_list[i].sort()[1]].to('cpu')).sum() + dif
                rewards.append(reward)

        return torch.tensor(rewards).to(self.config["device"]), torch.tensor(dif_list), changed_images