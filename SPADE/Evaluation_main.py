# Evaluation.py
import glob
import numpy as np
from PIL import Image
import torch
from ultralytics import YOLO
from torchmetrics.detection import MeanAveragePrecision
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from sklearn.preprocessing import label_binarize

# Hardcoded paths and configs (match from main_od.py)
MODEL_PATH = '/content/drive/MyDrive/Project_RLAB/Main/runs/detect/train2/weights/best.pt'
ORIG_DIR = "/content/drive/MyDrive/Project_RLAB/Dataset/BD100K/images/test/"
ADV_DIR = "/content/drive/MyDrive/Project_RLAB/Main/Results/adv_images/"
RESULTS_DIR = "/content/drive/MyDrive/Project_RLAB/Main/Results/evaluated_results/"
VIS_DIR = os.path.join(RESULTS_DIR, "visuals/")
QUERY_FILE = os.path.join(ADV_DIR, 'queries.npy')  # Assume main_od.py saves np.save(QUERY_FILE, iteration.cpu().numpy())
YOLO_CONF = 0.50
REDUCTION_THRESHOLD = 0.5  # 50% reduction for ASR
IOU_THRESHOLD = 0.5

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(VIS_DIR, exist_ok=True)

def compute_iou(box1, box2):
    """Compute IoU between two boxes (xyxy format)."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = (box1[2] - box1[0]) * (box1[3] - box1[1]) + (box2[2] - box2[0]) * (box2[3] - box2[1]) - inter
    return inter / union if union > 0 else 0

def evaluate():
    model = YOLO(MODEL_PATH)
    class_names = model.names
    num_classes = len(class_names)
    
    # Load image paths (assuming first 50 sorted as in main)
    orig_paths = sorted(glob.glob(ORIG_DIR + '*.jpg'))[:50]
    adv_paths = sorted(glob.glob(ADV_DIR + 'adv_*.png'))
    
    print(f"Found {len(orig_paths)} original images and {len(adv_paths)} adversarial images.")
    if len(orig_paths) != len(adv_paths):
        print("Warning: Mismatch in number of images. Evaluating on the minimum available pairs.")
    num_images = min(len(orig_paths), len(adv_paths))
    
    # Initialize metrics
    asr_count = 0  # global
    recall_drops = []  # global
    delta_ious = []  # global
    num_drops = []  # global
    precision_drops = []  # global (new)
    fprs = []  # global (new)
    misclass_rates = []  # global (new)
    conf_drops = []  # global (new)
    l1_norms = []  # global (new)
    asr_per_class = np.zeros(num_classes)
    image_count_per_class = np.zeros(num_classes)
    recall_drops_per_class = [[] for _ in range(num_classes)]
    delta_ious_per_class = [[] for _ in range(num_classes)]
    precision_drops_per_class = [[] for _ in range(num_classes)]  # new
    fprs_per_class = [[] for _ in range(num_classes)]  # new
    misclass_rates_per_class = [[] for _ in range(num_classes)]  # new
    conf_drops_per_class = [[] for _ in range(num_classes)]  # new
    l0_percents = []
    rmses = []
    linf_norms = []
    ssims = []
    psnrs = []
    
    preds_list = []
    targets_list = []
    
    metric = MeanAveragePrecision(class_metrics=True)
    
    # For PR curves (aggregate scores and labels)
    all_orig_confs = []
    all_adv_confs = []
    all_tp_orig = []
    all_tp_adv = []
    
    for i in range(num_images):
        orig_img = np.array(Image.open(orig_paths[i]))
        adv_img = np.array(Image.open(adv_paths[i]))
        
        orig_results = model(orig_img, conf=YOLO_CONF, verbose=False)[0]
        adv_results = model(adv_img, conf=YOLO_CONF, verbose=False)[0]
        
        orig_boxes = orig_results.boxes.xyxy.cpu().numpy()
        orig_conf = orig_results.boxes.conf.cpu().numpy()
        orig_cls = orig_results.boxes.cls.cpu().numpy().astype(int)
        
        adv_boxes = adv_results.boxes.xyxy.cpu().numpy()
        adv_conf = adv_results.boxes.conf.cpu().numpy()
        adv_cls = adv_results.boxes.cls.cpu().numpy().astype(int)
        
        num_orig_total = len(orig_boxes)
        num_adv_total = len(adv_boxes)
        
        # Global ASR
        if num_orig_total > 0 and num_adv_total <= (1 - REDUCTION_THRESHOLD) * num_orig_total:
            asr_count += 1
        
        # Global drop in number of detections
        num_drop = (num_orig_total - num_adv_total) / num_orig_total if num_orig_total > 0 else 0
        num_drops.append(num_drop)
        
        # Global recall, ΔIoU, and new metrics
        matched_total = 0
        misclassified_matched = 0
        total_iou = 0.0
        total_conf_drop = 0.0
        matched_adv_to_orig = 0  # For precision
        if num_orig_total > 0:
            for j, o_box in enumerate(orig_boxes):
                o_c = orig_cls[j]
                o_conf_val = orig_conf[j]
                max_iou = 0.0
                best_k = -1
                for k, a_box in enumerate(adv_boxes):
                    if adv_cls[k] == o_c:
                        iou_val = compute_iou(o_box, a_box)
                        if iou_val > max_iou:
                            max_iou = iou_val
                            best_k = k
                total_iou += max_iou
                if max_iou > IOU_THRESHOLD:
                    matched_total += 1
                    if adv_cls[best_k] != o_c:  # Should be == o_c, but check for misclass
                        misclassified_matched += 1
                    total_conf_drop += (o_conf_val - adv_conf[best_k])
        
            # Precision: matched adv to orig
            for k, a_box in enumerate(adv_boxes):
                a_c = adv_cls[k]
                for j, o_box in enumerate(orig_boxes):
                    if orig_cls[j] == a_c and compute_iou(a_box, o_box) > IOU_THRESHOLD:
                        matched_adv_to_orig += 1
                        break
        
            recall_total = matched_total / num_orig_total
            avg_iou_total = total_iou / num_orig_total
            precision_adv = matched_adv_to_orig / num_adv_total if num_adv_total > 0 else 0
            precision_drop = (1 - precision_adv) * 100  # Assuming orig precision ~1
            fp = num_adv_total - matched_adv_to_orig
            fpr = fp / num_adv_total if num_adv_total > 0 else 0
            misclass_rate = misclassified_matched / matched_total if matched_total > 0 else 0
            avg_conf_drop = total_conf_drop / matched_total if matched_total > 0 else 0
        else:
            recall_total = 1.0
            avg_iou_total = 1.0
            precision_drop = 0
            fpr = 0
            misclass_rate = 0
            avg_conf_drop = 0
        recall_drops.append(1 - recall_total)
        delta_ious.append(1 - avg_iou_total)
        precision_drops.append(precision_drop)
        fprs.append(fpr)
        misclass_rates.append(misclass_rate)
        conf_drops.append(avg_conf_drop)
        
        # Per class metrics
        for c in range(num_classes):
            orig_mask = (orig_cls == c)
            num_orig_c = np.sum(orig_mask)
            if num_orig_c == 0:
                continue
            image_count_per_class[c] += 1
            
            adv_mask = (adv_cls == c)
            num_adv_c = np.sum(adv_mask)
            
            # Class ASR
            if num_adv_c <= (1 - REDUCTION_THRESHOLD) * num_orig_c:
                asr_per_class[c] += 1
            
            # Class recall, ΔIoU, and new metrics
            matched_c = 0
            misclassified_matched_c = 0
            total_iou_c = 0.0
            total_conf_drop_c = 0.0
            matched_adv_c = 0
            orig_boxes_c = orig_boxes[orig_mask]
            orig_conf_c = orig_conf[orig_mask]
            adv_boxes_c = adv_boxes[adv_mask]
            adv_conf_c = adv_conf[adv_mask]
            for j, o_box in enumerate(orig_boxes_c):
                max_iou = 0.0
                best_k = -1
                for k, a_box in enumerate(adv_boxes_c):
                    iou_val = compute_iou(o_box, a_box)
                    if iou_val > max_iou:
                        max_iou = iou_val
                        best_k = k
                total_iou_c += max_iou
                if max_iou > IOU_THRESHOLD:
                    matched_c += 1
                    if adv_cls[adv_mask][best_k] != c:  # Misclass check
                        misclassified_matched_c += 1
                    total_conf_drop_c += (orig_conf_c[j] - adv_conf_c[best_k])
            
            # Precision for class
            for a_box in adv_boxes_c:
                for o_box in orig_boxes_c:
                    if compute_iou(a_box, o_box) > IOU_THRESHOLD:
                        matched_adv_c += 1
                        break
            
            recall_c = matched_c / num_orig_c if num_orig_c > 0 else 1.0
            avg_iou_c = total_iou_c / num_orig_c if num_orig_c > 0 else 1.0
            precision_c = matched_adv_c / num_adv_c if num_adv_c > 0 else 0
            fp_c = num_adv_c - matched_adv_c
            fpr_c = fp_c / num_adv_c if num_adv_c > 0 else 0
            misclass_rate_c = misclassified_matched_c / matched_c if matched_c > 0 else 0
            avg_conf_drop_c = total_conf_drop_c / matched_c if matched_c > 0 else 0
            
            recall_drops_per_class[c].append(1 - recall_c)
            delta_ious_per_class[c].append(1 - avg_iou_c)
            precision_drops_per_class[c].append((1 - precision_c) * 100)
            fprs_per_class[c].append(fpr_c)
            misclass_rates_per_class[c].append(misclass_rate_c)
            conf_drops_per_class[c].append(avg_conf_drop_c)
        
        # For mAP
        preds_dict = {
            "boxes": torch.tensor(adv_boxes),
            "scores": torch.tensor(adv_conf),
            "labels": torch.tensor(adv_cls)
        }
        target_dict = {
            "boxes": torch.tensor(orig_boxes),
            "labels": torch.tensor(orig_cls)
        }
        preds_list.append(preds_dict)
        targets_list.append(target_dict)
        
        # For PR curve (aggregate confs and match labels)
        for j in range(num_orig_total):
            all_orig_confs.append(orig_conf[j])
            all_tp_orig.append(1)
        
        for k in range(num_adv_total):
            all_adv_confs.append(adv_conf[k])
            matched = False
            for j in range(num_orig_total):
                if orig_cls[j] == adv_cls[k] and compute_iou(orig_boxes[j], adv_boxes[k]) > IOU_THRESHOLD:
                    matched = True
                    break
            all_tp_adv.append(1 if matched else 0)
    
    # Update metric
    metric.update(preds_list, targets_list)
    map_results = metric.compute()
    delta_map_global = 1.0 - map_results['map'].item()
    
    # Collect unique classes from targets
    unique_classes = set()
    for target in targets_list:
        unique_classes.update(target['labels'].tolist())
    sorted_unique = sorted(unique_classes)
    
    # Create full delta_map_per_class with NaNs
    delta_map_per_class_full = np.full(num_classes, np.nan)
    for i, cls in enumerate(sorted_unique):
        delta_map_per_class_full[cls] = 1.0 - map_results['map_per_class'][i].item()
    
    # Aggregate global metrics
    asr_global = asr_count / num_images * 100
    avg_recall_drop_global = np.mean(recall_drops) * 100
    avg_delta_iou_global = np.mean(delta_ious)
    avg_num_drop = np.mean(num_drops) * 100
    avg_precision_drop_global = np.mean(precision_drops)
    avg_fpr_global = np.mean(fprs)
    avg_misclass_rate_global = np.mean(misclass_rates)
    avg_conf_drop_global = np.mean(conf_drops)
    
    # Aggregate per-class metrics
    asr_per_class = (asr_per_class / np.maximum(image_count_per_class, 1)) * 100
    avg_recall_drop_per_class = [np.mean(recall_drops_per_class[c]) * 100 if recall_drops_per_class[c] else np.nan for c in range(num_classes)]
    avg_delta_iou_per_class = [np.mean(delta_ious_per_class[c]) if delta_ious_per_class[c] else np.nan for c in range(num_classes)]
    avg_precision_drop_per_class = [np.mean(precision_drops_per_class[c]) if precision_drops_per_class[c] else np.nan for c in range(num_classes)]
    avg_fpr_per_class = [np.mean(fprs_per_class[c]) if fprs_per_class[c] else np.nan for c in range(num_classes)]
    avg_misclass_rate_per_class = [np.mean(misclass_rates_per_class[c]) if misclass_rates_per_class[c] else np.nan for c in range(num_classes)]
    avg_conf_drop_per_class = [np.mean(conf_drops_per_class[c]) if conf_drops_per_class[c] else np.nan for c in range(num_classes)]
    
    # Perturbation metrics (loop continuation)
    for i in range(num_images):
        orig_img = np.array(Image.open(orig_paths[i]))
        adv_img = np.array(Image.open(adv_paths[i]))
        diff = np.abs(orig_img.astype(np.float32) - adv_img.astype(np.float32))
        
        # L0%
        changed_pixels = np.any(diff > 0, axis=-1)
        l0 = np.sum(changed_pixels)
        l0_percent = (l0 / (orig_img.shape[0] * orig_img.shape[1])) * 100 if orig_img.size > 0 else 0
        l0_percents.append(l0_percent)
        
        # RMSE (normalized L2)
        rmse = np.sqrt(np.mean(diff ** 2))
        rmses.append(rmse)
        
        # L∞
        linf = np.max(diff)
        linf_norms.append(linf)
        
        # L1 Norm (new)
        l1 = np.sum(diff)
        l1_norms.append(l1)
        
        # SSIM
        ssim_val = ssim(orig_img, adv_img, channel_axis=2, data_range=255)
        ssims.append(ssim_val)
        
        # PSNR
        psnr_val = psnr(orig_img, adv_img, data_range=255)
        psnrs.append(psnr_val)
    
    avg_l0_percent = np.mean(l0_percents)
    avg_rmse = np.mean(rmses)
    avg_linf = np.mean(linf_norms)
    avg_l1 = np.mean(l1_norms)
    avg_ssim = np.mean(ssims)
    finite_psnrs = [p for p in psnrs if np.isfinite(p)]
    avg_psnr = np.mean(finite_psnrs) if finite_psnrs else np.nan
    median_psnr = np.median(finite_psnrs) if finite_psnrs else np.nan
    
    # Load query counts (assume saved as npy from main_od.py)
    if os.path.exists(QUERY_FILE):
        queries = np.load(QUERY_FILE)
        avg_query = np.mean(queries)
        # Save per-image queries to results path (as CSV)
        query_df = pd.DataFrame({'Image_Index': range(num_images), 'Queries': queries})
        query_df.to_csv(os.path.join(RESULTS_DIR, 'per_image_queries.csv'), index=False)
    else:
        avg_query = 0  # Placeholder if file not found
    avg_time = 0.0  # Placeholder
    
    # Prepare data for table
    metrics = ["ASR (%)", "ΔmAP", "Recall Drop (%)", "ΔIoU", "Avg Detection Drop (%)", 
               "Precision Drop (%)", "FPR", "Misclassification Rate", "Avg Confidence Drop"]
    global_values = [asr_global, delta_map_global, avg_recall_drop_global, avg_delta_iou_global, avg_num_drop,
                     avg_precision_drop_global, avg_fpr_global, avg_misclass_rate_global, avg_conf_drop_global]
    
    data = {"Metric": [], "Average": []}
    for c in range(num_classes):
        data[class_names[c]] = []
    
    for m, g_val in zip(metrics, global_values):
        data["Metric"].append(f"{m} - Average")
        data["Average"].append(g_val)
        for c in range(num_classes):
            if m == "ASR (%)":
                val = asr_per_class[c]
            elif m == "ΔmAP":
                val = delta_map_per_class_full[c]
            elif m == "Recall Drop (%)":
                val = avg_recall_drop_per_class[c]
            elif m == "ΔIoU":
                val = avg_delta_iou_per_class[c]
            elif m == "Avg Detection Drop (%)":
                val = np.nan  # Not computed per class
            elif m == "Precision Drop (%)":
                val = avg_precision_drop_per_class[c]
            elif m == "FPR":
                val = avg_fpr_per_class[c]
            elif m == "Misclassification Rate":
                val = avg_misclass_rate_per_class[c]
            elif m == "Avg Confidence Drop":
                val = avg_conf_drop_per_class[c]
            data[class_names[c]].append(val)
    
    # Add perturbation metrics (global only)
    data["Metric"].extend(["L0 (%)", "RMSE", "L∞ Norm", "L1 Norm", "SSIM", "PSNR", "Median PSNR", "Avg Query Count", "Avg Time per Attack (s)"])
    data["Average"].extend([avg_l0_percent, avg_rmse, avg_linf, avg_l1, avg_ssim, avg_psnr, median_psnr, avg_query, avg_time])
    for c in range(num_classes):
        data[class_names[c]].extend([np.nan] * 9)
    
    df = pd.DataFrame(data)
    print(df.to_string(index=False))
    df.to_csv(os.path.join(RESULTS_DIR, 'metrics.csv'), index=False)
    
    # Visual Graphs
    valid_classes = [c for c in range(num_classes) if image_count_per_class[c] > 0]
    class_labels = [class_names[c] for c in valid_classes]
    x = np.arange(len(class_labels))
    width = 0.15
    
    # Bar Chart: Per-Class Metrics
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.bar(x - 2*width, asr_per_class[valid_classes], width, label='ASR (%)')
    ax.bar(x - width, [avg_recall_drop_per_class[c] for c in valid_classes], width, label='Recall Drop (%)')
    ax.bar(x, delta_map_per_class_full[valid_classes], width, label='ΔmAP')
    ax.bar(x + width, [avg_precision_drop_per_class[c] for c in valid_classes], width, label='Precision Drop (%)')
    ax.bar(x + 2*width, [avg_fpr_per_class[c] for c in valid_classes], width, label='FPR')
    ax.set_xticks(x)
    ax.set_xticklabels(class_labels, rotation=45)
    ax.legend()
    plt.title('Per-Class Attack Metrics')
    plt.savefig(os.path.join(VIS_DIR, 'per_class_metrics_bar.png'))
    plt.close()
    
    # Histogram: Perturbation Distributions
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    axs[0,0].hist(l0_percents, bins=20, color='blue')
    axs[0,0].axvline(avg_l0_percent, color='red', linestyle='--')
    axs[0,0].set_title('L0% Distribution')
    axs[0,1].hist(rmses, bins=20, color='green')
    axs[0,1].axvline(avg_rmse, color='red', linestyle='--')
    axs[0,1].set_title('RMSE Distribution')
    axs[1,0].hist(linf_norms, bins=20, color='purple')
    axs[1,0].axvline(avg_linf, color='red', linestyle='--')
    axs[1,0].set_title('L∞ Distribution')
    axs[1,1].hist(ssims, bins=20, color='orange')
    axs[1,1].axvline(avg_ssim, color='red', linestyle='--')
    axs[1,1].set_title('SSIM Distribution')
    plt.suptitle('Perturbation Metric Distributions')
    plt.savefig(os.path.join(VIS_DIR, 'pert_dist_hist.png'))
    plt.close()
    
    # Scatter: L0% vs Recall Drop
    asr_flags = [1 if num_drops[i] >= REDUCTION_THRESHOLD else 0 for i in range(num_images)]  # Approximate success
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(l0_percents, [rd * 100 for rd in recall_drops], c=asr_flags, cmap='viridis')
    plt.colorbar(scatter, label='ASR Success (1=Yes)')
    plt.xlabel('L0%')
    plt.ylabel('Recall Drop (%)')
    plt.title('Perturbation Size vs Attack Success')
    plt.savefig(os.path.join(VIS_DIR, 'l0_vs_recall_scatter.png'))
    plt.close()
    
    # Precision-Recall Curve (Global, using aggregated data)
    precision_orig, recall_orig, _ = precision_recall_curve(all_tp_orig, all_orig_confs)
    precision_adv, recall_adv, _ = precision_recall_curve(all_tp_adv, all_adv_confs)
    
    # Scale recall for adv to account for FN
    total_gt = len(all_tp_orig)
    tp_adv = sum(all_tp_adv)
    if total_gt > 0 and tp_adv > 0:
        scale = tp_adv / total_gt
        recall_adv *= scale
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall_orig, precision_orig, label='Original')
    plt.plot(recall_adv, precision_adv, label='Adversarial')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve (Before/After Attack)')
    plt.legend()
    plt.savefig(os.path.join(VIS_DIR, 'pr_curve.png'))
    plt.close()

if __name__ == "__main__":
    evaluate()