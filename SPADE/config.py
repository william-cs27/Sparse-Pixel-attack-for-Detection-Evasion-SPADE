# Hyperparameters
config = {
    #  Fix config
    "batch_size": 16,  # smaller batch for large BDD100K images/GPU
    "RL_learning_rate": 0.0001,
    "RGB": 3,
    "shape_unity": True,
    "subtract": True,
    "std": "learn",

    #  Variate config
    "classifier": "yolo",    # CHANGE for YOLO
    "attack_pixel": 0.05,   
    "dataset": "BDD100K",    # CHANGE for your dataset
    "bound": 50,
    "patient": 20,
    "limit": 5e-2,

    #  Additional for YOLOv8n/BDD100K
    "yolo_conf": 0.6,        # Detection confidence threshold
}

# Dataset-specific image size and params
if config["dataset"] == "ImageNet":
    config["img_size_x"] = 224
    config["img_size_y"] = 224
    config["RGB"] = 3
    config["patient"] = 3
    config["bound"] = 100
    config["limit"] = 5e-2

elif config["dataset"] == "COCO":
    config["img_size_x"] = 224
    config["img_size_y"] = 224
    config["RGB"] = 3
    config["patient"] = 20
    config["bound"] = 100
    config["limit"] = 5e-2

elif config["dataset"] == "BDD100K":
    # You should set these to match your actual input image resolution to YOLOv8n
    config["img_size_x"] = 640
    config["img_size_y"] = 384   # or 720, or whatever your actual input size is
    config["RGB"] = 3
    config["patient"] = 20
    config["bound"] = 50
    config["limit"] = 5e-2
    config["pert_area_fraction"] = 0.05  # 5% of image pixels; tune as needed (e.g., 0.01 for stricter minimization)

# Always check config["img_size_x"/"y"] match the real image input for the model!
