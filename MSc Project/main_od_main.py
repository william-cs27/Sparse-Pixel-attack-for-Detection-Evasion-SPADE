# main_od.py
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import glob
import numpy as np
from config import config
import torchvision.transforms as transforms
import Environment_main
import Adversarial_RL_simple as Adversarial_RL_simple
import torch
from utils import seed_all, sample_action, early_stopping, MyBaseDataset, L0_norm, L2_norm
import math
import copy
import os

from tqdm import tqdm

def attack(model, train_data, config, hw_array):
    env = Environment_main.Env(model, config=config)
    h_max, w_max = hw_array.max(axis=0)[0], hw_array.max(axis=0)[1]
    if ((hw_array[:,0] != h_max).sum()!=0 or (hw_array[:,1] != w_max).sum()!=0):
        config["shape_unity"] = False
        for _ in range(len(train_data)):
            train_data[_] = np.pad(train_data[_], ((0, h_max - hw_array[_,0]), (0, w_max - hw_array[_,1]), (0,0)), 'constant', constant_values=0)
        env.hw_array = torch.tensor(hw_array).long()
    else:
        config["shape_unity"] = True
    print(f'shape_unity : {config["shape_unity"]}')

    # Object detection results for clean images 
    i = 1
    print("Analyzing clean images with YOLOv8n and saving results...")
    image_allowed_pixels = {}
    for n, _ in tqdm(enumerate(train_data), desc="Clean image analysis", total=len(train_data)):
        if config["shape_unity"]:
            results = model(_, conf=config["yolo_conf"], verbose=False)
            img_h, img_w = _.shape[:2]
        else:
            results = model(_[:hw_array[n,0], :hw_array[n,1]], conf=config["yolo_conf"], verbose=False)
            img_h, img_w = hw_array[n,0], hw_array[n,1]
        boxes = results[0].boxes.xyxy.cpu().numpy()
        confs = results[0].boxes.conf.cpu().numpy()
        if len(confs) >= 2:
            chosen_indices = confs.argsort()[:2]  # 2 lowest confidence
        else:
            chosen_indices = np.arange(len(confs))
        chosen_boxes = boxes[chosen_indices]

        # Build allowed pixels inside chosen boxes
        allowed_pixels = []
        for box in chosen_boxes:
            x1, y1, x2, y2 = map(int, box)
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(img_w - 1, x2)
            y2 = min(img_h - 1, y2)
            for x in range(x1, x2 + 1):
                for y in range(y1, y2 + 1):
                    allowed_pixels.append((x, y))
        image_allowed_pixels[n] = allowed_pixels 

        env.ori_prob.append(results[0].boxes.conf)
        env.ori_cls.append(results[0].boxes.cls)
        env.ori_box_num.append(results[0].boxes.shape[0])
        im_array = results[0].plot()
        im = Image.fromarray(im_array)
        im.save(f'{result_path}ori_'+'{0:04}'.format(i)+'.jpg')
        i += 1

    env.ori_box_num = torch.tensor(env.ori_box_num).long()
    print("The Average number of Detected Object :", env.ori_box_num.float().mean().item())


    update_images = torch.tensor(np.array(train_data)).clone().to(config["device"])
    metric_images = torch.tensor(np.array(train_data)).clone()
    trick_element = torch.zeros(len(train_data))
    yolo_list = torch.tensor(np.array(train_data)).clone()
    cls_list = []

    torchvision_transform = transforms.Compose([transforms.Resize((256, 256)), transforms.CenterCrop(224)])
    train_data = MyBaseDataset(train_data, trick_element, transform=True)

    it = 0
    total = 0
    iteration = torch.zeros(update_images.shape[0]).to(config["device"])
    change_idx = []
    L0 = []
    L2 = []
    box_count = torch.zeros(update_images.shape[0]).to(config["device"])

    for p in range(config["bound"]):
        agent = Adversarial_RL_simple.REINFORCE(config).to(config["device"])
        flag = False
        stop_count = 0
        prev_change_list = torch.zeros(update_images.shape[0]).to(config["device"])
        update_rewards = torch.zeros(update_images.shape[0]).to(config["device"])
        temp = torch.zeros(update_images.shape[0]).to(config["device"])
        inner_rl_episode = 0
        while 1:
            inner_rl_episode += 1
            bts = math.ceil(train_data.x_data.shape[0] / config["batch_size"])
            train_x = []
            change_train_x = []
            it += 1
            total_rewards_list = []
            total_change_list = torch.tensor([]).to(config["device"])

            for bt in tqdm(range(bts), desc=f"RL steps (Cycle {p+1}, Episode {inner_rl_episode})", leave=False):
                if bt != (bts-1):
                    s = train_data.x_data[bt*config["batch_size"]:bt*config["batch_size"]+config["batch_size"]]
                    if env.init == False:
                        labels = env.ori_cls[bt*config["batch_size"]:bt*config["batch_size"]+config["batch_size"]]
                        probs = env.ori_prob[bt*config["batch_size"]:bt*config["batch_size"]+config["batch_size"]]
                    if config["shape_unity"] == False:
                        hws = env.hw_array[bt*config["batch_size"]:bt*config["batch_size"]+config["batch_size"]]
                else:
                    s = train_data.x_data[bt*config["batch_size"]:]
                    if env.init == False:
                        probs = env.ori_prob[bt*config["batch_size"]:]
                        labels = env.ori_cls[bt*config["batch_size"]:]
                    if config["shape_unity"] == False:
                        hws = env.hw_array[bt*config["batch_size"]:]

                s = s.permute(0,3,1,2)
                action_means, action_stds = agent(torchvision_transform(s.to(config["device"])) / 255)
                action_stds = torch.clamp(action_stds, 0.1, 10)
                actions, actions_logprob = sample_action(action_means, action_stds, config)
                actions = actions.reshape(s.shape[0], config["attack_pixel"], 5)  # Always reshape to [batch, n_pixels, 5]
                actions_logprob = actions_logprob.reshape(s.shape[0], -1).sum(dim=1)  # Sum logprobs per sample

                allowed_pixels_batch = [image_allowed_pixels[idx] for idx in range(bt * config["batch_size"], bt * config["batch_size"] + s.shape[0])]
                if config["shape_unity"]:
                    rewards, dif_list, changed_images = env.yolo_step_not_sub(s, actions, bt, labels, probs, allowed_pixels_batch)
                else:
                    rewards, dif_list, changed_images = env.yolo_step_disunity_not_sub(s, actions, bt, hws, labels, probs, allowed_pixels_batch)

                change_list = dif_list
                s = s.permute(0,2,3,1)
                for _ in range(len(changed_images)):
                    change_train_x.append(changed_images[_])
                total_change_list = torch.cat((total_change_list, change_list.to(config["device"])), dim=0).long()
                total_rewards_list += rewards.tolist()

                agent.r = rewards
                agent.prob = actions_logprob
                agent.train_net()

            standard = update_rewards.mean()
            total_rewards_list = torch.tensor(total_rewards_list).to(config["device"])
            update_rewards = torch.cat((update_rewards.unsqueeze(dim=0), total_rewards_list.unsqueeze(dim=0)), dim=0)
            update_rewards_indices = torch.max(update_rewards, axis=0).indices
            update_rewards = update_rewards[update_rewards_indices, np.arange(total_rewards_list.shape[0])]
            update_images = torch.cat((update_images.unsqueeze(dim=0), torch.tensor(np.array(change_train_x)).to(config["device"]).unsqueeze(dim=0)), dim=0)
            update_images = update_images[update_rewards_indices, np.arange(total_rewards_list.shape[0]), :, :]

            temp = torch.max(torch.cat((prev_change_list.unsqueeze(dim=0), total_change_list.unsqueeze(dim=0)), dim=0), axis=0).values
            delta_box = temp - prev_change_list
            prev_change_list = temp.clone()
            update_rewards_sum = update_rewards.mean()

            if delta_box.sum() > 0:
                change_idx = list(filter(lambda x: delta_box[x] > 0, range(len(delta_box))))
                for _ in change_idx:
                    yolo_list[_] = torch.tensor(change_train_x[_]).clone()
                    iteration[_] = it
            box_count += delta_box

            stop_count, flag = early_stopping(((update_rewards_sum - standard) / standard) + delta_box.sum(), stop_count, limit=config["limit"], patient=config["patient"])

            if flag:
                env.init = True
                train_x = update_images
                train_data = MyBaseDataset(train_x, cls_list)
                env.ori_cls = []
                env.ori_prob = []
                it += 1
                break
            else:
                env.init = False

        total = box_count.mean()
        print(f'Forget:{p}, eliminated box: {total}')
        if env.ori_box_num.float().mean() - total == 0:
            print("all Images are deceived")
            break

    total = len(prev_change_list)
    if config["shape_unity"]:
        L0 = L0_norm(metric_images, yolo_list)
        L2 = L2_norm(metric_images.float(), yolo_list)
    else:
        for _ in range(total):
            L0.extend(L0_norm(metric_images[_, :hw_array[_,0], :hw_array[_,1]], yolo_list[_, :hw_array[_,0], :hw_array[_,1]]))
            L2.extend(L2_norm(metric_images.float()[_, :hw_array[_,0], :hw_array[_,1]], yolo_list[_, :hw_array[_,0], :hw_array[_,1]]))

    # Save results
    for _ in range(len(yolo_list)):
        if config["shape_unity"]:
            img = Image.fromarray(yolo_list[_].numpy())
            delta_img = Image.fromarray(torch.abs(metric_images[_] - yolo_list[_]).numpy())
            result = model(yolo_list[_].numpy(), conf=config["yolo_conf"], verbose=False)
        else:
            result = model(yolo_list[_, :hw_array[_,0], :hw_array[_,1]].numpy(), conf=config["yolo_conf"], verbose=False)
            delta_img = Image.fromarray(255 - torch.abs(metric_images[_, :hw_array[_,0], :hw_array[_,1]] - yolo_list[_, :hw_array[_,0], :hw_array[_,1]]).numpy())
            img = Image.fromarray(yolo_list[_, :hw_array[_,0], :hw_array[_,1]].numpy())

        img.save(adv_path + f'adv_' + '{0:04}'.format(_+1) + '.png', 'PNG')
        delta_img.save(delta_path + f'delta_' + '{0:04}'.format(_+1) + '.png', 'PNG')
        im_array = result[0].plot()
        im = Image.fromarray(im_array)
        im.save(f'{adv_result_path}adv_' + '{0:04}'.format(_+1) + '.png', 'PNG')

if __name__ == '__main__':
    seed = 2
    seed_all(seed)
    config["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config["bound"] = 100
    config["limit"] = 5e-2
    config["attack_pixel"] = 0.05
    config["yolo_conf"] = 0.50
    config["patient"] = 20
    config["classifier"] = "yolo"
    config["dataset"] = "BDD100K"  # <-- Set for your project

    fix_path = os.path.abspath(os.getcwd())

    # Model
    model = YOLO('/content/drive/MyDrive/Project_RLAB/runs/detect/train/weights/best.pt').to('cuda') 

    # Dataset path (update this to your actual validation images location for BDD100K)
    file_path = "/content/drive/MyDrive/Project_RLAB/Dataset/BD100K/images/test/"

    # Result path
    result_path = "/content/drive/MyDrive/Project_RLAB/RFPAR/Results/original_result"
    adv_path = "/content/drive/MyDrive/Project_RLAB/RFPAR/Results/adv_images"
    adv_result_path = "/content/drive/MyDrive/Project_RLAB/RFPAR/Results/adv_result"
    delta_path = "/content/drive/MyDrive/Project_RLAB/RFPAR/Results/delta_images"

    

    os.makedirs(result_path, exist_ok=True)
    os.makedirs(adv_path, exist_ok=True)
    os.makedirs(adv_result_path, exist_ok=True)
    os.makedirs(delta_path, exist_ok=True)

    result_path = result_path + "/"
    adv_path = adv_path + "/"
    adv_result_path = adv_result_path + "/"
    delta_path = delta_path + "/"

    list_images = glob.glob(file_path + '*.jpg')
    list_images = sorted(list_images)
    list_images = list_images[:50]

    img_list = []
    img_hw_list = []
    
    from tqdm import tqdm
    #for image in list_images:
        #img = np.array(Image.open(image))
        #img_hw_list.append([img.shape[0], img.shape[1]])
        #img_list.append(img)
    for image in tqdm(list_images, desc="Loading images"):
        img = np.array(Image.open(image))
        img_hw_list.append([img.shape[0], img.shape[1]])
        img_list.append(img)

    img_hw_list = np.array(img_hw_list)
    config["ni"] = len(img_list)
    img_array = img_list
    if len(img_list) == 0:
      raise RuntimeError("No images found in the specified directory! Check your file_path and images.")

# Now it is safe:
    config["attack_pixel"] = int((img_hw_list.max(axis=0)[0] + img_hw_list.max(axis=0)[1]) / 2 * config["attack_pixel"])

    # Run attack
    attack(model, img_array, config, img_hw_list)