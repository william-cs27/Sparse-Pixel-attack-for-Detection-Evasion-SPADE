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

    def make_transformed_images(self, original_images, actions, allowed_pixels=None):
        """
        allowed_pixels: List of allowed (x, y) coords to attack in the image.
                        For batch, should be a list of lists (one per image).
        actions: For each image, an index or set of indices into allowed_pixels.
        """
        RGB = original_images.shape[1]
        arr = []
        batch = original_images.shape[0]
        actions = torch.sigmoid(actions)

        # For batch mode, we handle allowed_pixels as a list of lists
        for i in range(batch):
            changed_image = original_images[i].clone()
            # For each pixel to attack
            if allowed_pixels is not None:
                coords_this_img = allowed_pixels[i] # List of (x, y)
                # actions shape: [batch, n_pixels, action_dim], flatten if needed
                # Let's assume actions[i] is shape [n_pixels, action_dim]
                if len(coords_this_img) > 0:
                    for j in range(actions.shape[1]):
                        idx = int(actions[i, j, 0] * (len(coords_this_img) - 1))
                        x, y = coords_this_img[idx]
                        # For RGB: actions[...,2:5] are used for r,g,b
                        if self.config["classifier"] == "yolo" or self.config["classifier"] == "ddq":
                            r = 255 if actions[i, j, 2] > 0.5 else 0
                            g = 255 if actions[i, j, 3] > 0.5 else 0
                            b = 255 if actions[i, j, 4] > 0.5 else 0
                        else:
                            r = float(actions[i, j, 2] > 0.5)
                            g = float(actions[i, j, 3] > 0.5)
                            b = float(actions[i, j, 4] > 0.5)
                        changed_image[0, y, x] = r
                        changed_image[1, y, x] = g
                        changed_image[2, y, x] = b
            else:
                # Original fallback: act on full image
                x_bound = original_images.shape[3]  # W
                y_bound = original_images.shape[2]  # H
                x = (actions[i, :, 0] * (x_bound - 1)).long()
                y = (actions[i, :, 1] * (y_bound - 1)).long()
                if self.config["classifier"] == "yolo" or self.config["classifier"] == "ddq":
                    r = (actions[i, :, 2] > 0.5).float() * 255
                    g = (actions[i, :, 3] > 0.5).float() * 255
                    b = (actions[i, :, 4] > 0.5).float() * 255
                else:
                    r = (actions[i, :, 2] > 0.5).float()
                    g = (actions[i, :, 3] > 0.5).float()
                    b = (actions[i, :, 4] > 0.5).float()
                for j in range(x.shape[0]):
                    changed_image[0, y[j], x[j]] = r[j]
                    changed_image[1, y[j], x[j]] = g[j]
                    changed_image[2, y[j], x[j]] = b[j]
            arr.append(changed_image.unsqueeze(0))
        changed_images = torch.cat(arr, 0)
        return changed_images

    def yolo_step_not_sub(self, original_images, actions, bt, labels=None, probs=None, allowed_pixels=None):
        # Pass allowed_pixels from your calling code!
        changed_images = self.make_transformed_images(original_images.to(self.config["device"]), actions, allowed_pixels=allowed_pixels)
        with torch.no_grad():
            if self.init == True:
                labels = []
                probs = []
                for img in original_images:
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

    def yolo_step_disunity_not_sub(self, original_images, actions, bt, hws, labels=None, probs=None, allowed_pixels=None):
        # Similar to yolo_step_not_sub but handles different shapes by cropping before model inference
        changed_images = self.make_transformed_images(original_images.to(self.config["device"]), actions, allowed_pixels=allowed_pixels)
        with torch.no_grad():
            if self.init == True:
                labels = []
                probs = []
                for n, img in enumerate(original_images):
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