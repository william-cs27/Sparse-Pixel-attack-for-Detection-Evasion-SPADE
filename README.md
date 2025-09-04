

# Repository Contents
The repository contains the following files and directories:
- `main_od_main.py` -- End-to-end attack execution pipeline with RL training loop and checkpointing.
- `Environment_main.py` -- Custom reinforcement learning environment with YOLO evaluation and pixel perturbation functions.
- `Adversarial_RL_simple.py` -- Policy network and reinforcement learning agent.
- `adv_images/` -- Folder for saving generated adversarial images and intermediate outputs.
- `checkpoints/` -- Stores RL agent checkpoints for resuming experiments.
- `requirements.txt` -- List of required Python packages and dependencies.
- `README.md` -- Setup and usage instructions.

# How to Run the Software
To reproduce the experiments:

1. Clone the repository:
   ```
   git clone https://git.cs.bham.ac.uk/projects-2024-25/wxc450
   cd SPADE
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Dataset download: https://bair.berkeley.edu/blog/2018/05/30/bdd/

4. Prepare the dataset (BDD100K in YOLOv8 format). Place it in the `datasets/` directory.
   ```
   git clone https://git.cs.bham.ac.uk/projects-2024-25/wxc450
   cd Bdd100k_to_yolo
   Configure config.py
   #Set IMAGES_ROOT = Path("Images/root/directory/path")
   #Set LABELS_ROOT = Path("Labels/root/path")
   #Set OUTPUT_DATASET_DIR = Path("Output/directory/path")
   #Run
   python config.py
   ```

5. Run Yolov8n on BDD100K:
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

6. Set the paths before the Attack.
   ```
   model = YOLO('Path/to/best/model/trained').to('cuda')
   file_path = "Dataset/path"
   #Result path
   result_path = "results/path"
   adv_path = "adversarial/path"
   adv_result_path = "adversarial/results/path"
   delta_path = "delta/images/path"
   ```

7. Run the attack pipeline:
   ```
   python main_od_main.py
   ```
   - Adversarial images saved in `adv_images/`.
   - Perturbation heatmaps and logs saved in `results/`.

8. Outputs:
   - Adversarial images saved in `adv_images/`.
   - Perturbation heatmaps and logs saved in `results/`.

9. For Evaluation:
   ```
   #Set the paths in the file to output the results
   python Evaluation_main.py
   ```
