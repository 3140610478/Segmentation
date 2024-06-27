import os
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# reload_data = True
reload_data = False
splitting = (0.8, 0.1, 0.1)
seed = 42

batch_size = 1

sam = {
    # "use_sam": True,
    "use_sam": False,
    "model_type": "vit_h",
    "grid_size": 8,
    "gaussian_blur_sigma": 2,
}

DownscaleFactor = 16

loss_weights = 1, 4


base_folder = os.path.dirname(os.path.abspath(__file__))
if sam["use_sam"]:
    save_path = os.path.abspath(os.path.join(
        base_folder, "./Networks/save/model_sam.tar"
    ))
else:
    save_path = os.path.abspath(os.path.join(
        base_folder, "./Networks/save/model.tar"
    ))
if not os.path.exists(os.path.dirname(save_path)):
    os.mkdir(os.path.dirname(save_path))
