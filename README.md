# Now You See, Now You Don't — RL-Based Sparse Pixel Attacks for Detection Evasion

**Author:** William Cherian Sam (Student ID: 2869850)  
**Programme:** MSc Artificial Intelligence and Machine Learning  
**Supervisor:** Dr. Kashif Rajpoot  
**Institution:** University of Birmingham, School of Computer Science  
**Academic Year:** 2024–25

---

## Overview

This project proposes and evaluates an **object-specific reinforcement learning (RL) adversarial pixel attack** against the YOLOv8n object detection model, applied to the BDD100K autonomous driving dataset.

Unlike conventional adversarial attacks that scatter perturbations across the entire image, this method constrains modifications exclusively to **pixels within detected object bounding boxes** (cars, buses, trucks, and persons). The attack is framed as a **Markov Decision Process (MDP)**, where an RL agent iteratively perturbs pixels, observes changes in detection outcomes, and refines its strategy — all in a **black-box setting** with no access to model gradients or parameters.

The key insight is that targeting object-region pixels dramatically improves query efficiency and perceptual stealth, since background pixels are irrelevant to detection confidence. The agent uses a **Remember–Forget cycle** (inspired by RFPAR) to escape local optima and sustain long-term optimisation.

---

## Key Features

- **Object-specific perturbations** — perturbations restricted to bounding box regions of detected objects, avoiding wasted queries on background pixels.
- **Black-box attack** — interacts with YOLOv8 through detection outputs only (bounding boxes, confidence scores, class labels); no gradient access required.
- **Reinforcement learning agent** — a CNN-based REINFORCE policy network samples pixel-level write/erase actions and learns from detection feedback.
- **Remember–Forget mechanism** — resets agent state when reward plateaus, reinitialising from the best perturbation found so far to escape local optima.
- **Dual reward signal** — rewards both bounding box elimination and confidence reduction for surviving detections.
- **Imperceptible perturbations** — only ~0.11% of pixels are modified on average, with SSIM ≈ 0.98 and PSNR ≈ 35 dB.
- **Query-efficient** — achieves strong attack performance in ~736 queries per image on average, substantially fewer than the RFPAR baseline (1,736 queries).

---

## Results

The proposed attack was evaluated on the BDD100K validation set using YOLOv8n, targeting four safety-critical object classes: car, bus, truck, and person.

### Detection Degradation

| Metric             | Average | Car    | Bus    | Truck | Person |
|--------------------|---------|--------|--------|-------|--------|
| ASR (%)            | 68.0    | 72.09  | 66.67  | 100   | 100    |
| ΔmAP               | 0.814   | 0.594  | 0.663  | 1.0   | 1.0    |
| ΔIoU               | 0.629   | 0.679  | 0.681  | 1.0   | 1.0    |
| Recall Drop (%)    | 62.6    | 67.6   | 66.7   | 100   | 100    |

### Perceptual Quality

| Metric       | Value  |
|--------------|--------|
| L0 (%)       | 0.1121 |
| RMSE         | 4.32   |
| SSIM         | 0.9806 |
| PSNR (dB)    | 34.53  |

### Comparison with Baselines

| Metric              | Ours    | Gaussian | RFPAR  |
|---------------------|---------|----------|--------|
| ASR (%)             | 68.0    | 76.0     | 48.0   |
| ΔmAP                | 0.814   | 0.724    | 0.740  |
| Recall Drop (%)     | 62.6    | 62.0     | 44.95  |
| SSIM                | 0.9806  | 0.9737   | 0.9699 |
| PSNR (dB)           | 34.53   | 29.06    | 33.47  |
| Avg Query Count     | 736     | 633      | 1,736  |

The proposed method achieves the best balance: stronger detection degradation than RFPAR, significantly better perceptual quality than Gaussian noise, and far fewer queries than the RFPAR baseline.

---

## Repository Contents

- `main_od_main.py` — End-to-end attack execution pipeline with RL training loop and checkpointing.
- `Environment_main.py` — Custom reinforcement learning environment with YOLO evaluation and pixel perturbation functions.
- `Adversarial_RL_simple.py` — Policy network and reinforcement learning agent.
- `adv_images/` — Folder for saving generated adversarial images and intermediate outputs.
- `checkpoints/` — Stores RL agent checkpoints for resuming experiments.
- `requirements.txt` — List of required Python packages and dependencies.
- `README.md` — Setup and usage instructions.

---

## How to Run the Software

To reproduce the experiments:

**1. Clone the repository:**
```
git clone https://github.com/william-cs27/Sparse-Pixel-attack-for-Detection-Evasion-SPADE.git
cd SPADE
```

**2. Install dependencies:**
```
pip install -r requirements.txt
```

**3. Dataset download:** https://bair.berkeley.edu/blog/2018/05/30/bdd/

**4. Prepare the dataset** (BDD100K in YOLOv8 format). Place it in the `datasets/` directory.
```
git clone https://github.com/william-cs27/Sparse-Pixel-attack-for-Detection-Evasion-SPADE.git
cd Bdd100k_to_yolo
Configure config.py
#Set IMAGES_ROOT = Path("Images/root/directory/path")
#Set LABELS_ROOT = Path("Labels/root/path")
#Set OUTPUT_DATASET_DIR = Path("Output/directory/path")
#Run
python config.py
```

**5. Run Yolov8n on BDD100K:**
```
Set paths
dataset_path = "dataset/path" # root folder containing images and labels
train_images_dir = f"{dataset_path}train/images/path"
val_images_dir = f"{dataset_path}val/images/path"
test_images_dir = f"{dataset_path}test/images/path"
train_labels_dir = f"{dataset_path}train/labels/path"
val_labels_dir = f"{dataset_path}val/labels/path"
test_labels_dir = f"{dataset_path}test/labels/path"
python yolo_bdd100k.py
```

**6. Set the paths before the Attack:**
```
model = YOLO('Path/to/best/model/trained').to('cuda')
file_path = "Dataset/path"
#Result path
result_path = "results/path"
adv_path = "adversarial/path"
adv_result_path = "adversarial/results/path"
delta_path = "delta/images/path"
```

**7. Run the attack pipeline:**
```
python main_od_main.py
```
- Adversarial images saved in `adv_images/`.
- Perturbation heatmaps and logs saved in `results/`.

**8. Outputs:**
- Adversarial images saved in `adv_images/`.
- Perturbation heatmaps and logs saved in `results/`.

**9. For Evaluation:**
```
#Set the paths in the file to output the results
python Evaluation_main.py
```
