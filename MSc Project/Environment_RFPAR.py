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

    def make_transformed_images(self, original_images, actions):
        RGB = original_images.shape[1]
        x_bound = original_images.shape[2]
        y_bound = original_images.shape[3]

        actions = torch.sigmoid(actions)
        arr = []

        x = (actions[:, 0] * x_bound - 1).long()
        y = (actions[:, 1] * y_bound - 1).long()

        if RGB == 1:  # Grayscale
            brightness = ((actions[:, 2] * 255).int()).float() / 255
            for i in range(self.config["batch_size"]):
                changed_image = original_images[i].squeeze().squeeze().detach().cpu().numpy()
                changed_image[x[i], y[i]] = brightness[i]
                arr.append(changed_image)
        elif RGB == 3:  # RGB
            if self.config["classifier"] == "yolo" or self.config["classifier"] == "ddq":
                r = (actions[:, 2] > 0.5).float() * 255
                g = (actions[:, 3] > 0.5).float() * 255
                b = (actions[:, 4] > 0.5).float() * 255
            else:
                r = (actions[:, 2] > 0.5).float()
                g = (actions[:, 3] > 0.5).float()
                b = (actions[:, 4] > 0.5).float()
            batch = original_images.shape[0]
            for i in range(batch):
                changed_image = original_images[i].clone()
                if self.pixel == 1:
                    changed_image[0, x[i], y[i]] = r[i]
                    changed_image[1, x[i], y[i]] = g[i]
                    changed_image[2, x[i], y[i]] = b[i]
                else:
                    idx = torch.tensor([j for j in range(self.pixel)]) * batch + i
                    if self.config["classifier"] == "yolo" or self.config["classifier"] == "ddq":
                        changed_image[0, x[idx], y[idx]] = r[idx].type(torch.uint8)
                        changed_image[1, x[idx], y[idx]] = g[idx].type(torch.uint8)
                        changed_image[2, x[idx], y[idx]] = b[idx].type(torch.uint8)
                    else:
                        changed_image[0, x[idx], y[idx]] = r[idx].type(torch.float32)
                        changed_image[1, x[idx], y[idx]] = g[idx].type(torch.float32)
                        changed_image[2, x[idx], y[idx]] = b[idx].type(torch.float32)
                arr.append(changed_image.unsqueeze(0))
        changed_images = torch.cat(arr, 0)
        return changed_images

    # (No logic changes needed for step()—this is for classification, not used in your detection flow)

    def yolo_step_not_sub(self, original_images, actions, bt, labels=None, probs=None):
        # Used in the paper for object detection with YOLO. (For BDD100K, works as is.)
        changed_images = self.make_transformed_images(original_images.to(self.config["device"]), actions)
        with torch.no_grad():
            if self.init == True:
                labels = []
                probs = []
                for img in original_images:
                    # This works with Ultralytics YOLOv8n on BDD100K as well.
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

    # All other YOLO step variants remain unchanged for your case
    # The core logic is compatible with BDD100K as long as you call YOLOv8n and pass correct config

# No further changes needed for other step variants unless you want to filter to specific BDD100K classes or change reward logic.

