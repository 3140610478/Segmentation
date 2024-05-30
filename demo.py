import os
import sys
import torch
from tqdm import tqdm
import plotly.graph_objects as go
from plotly.offline import plot

base_folder = os.path.dirname(os.path.abspath(__file__))
if base_folder not in sys.path:
    sys.path.append(base_folder)
if True:
    from Networks.UNet import UNet
    from Data import MSRC
    from Data.utils import show_masks
    import config

@torch.no_grad()
def demo(model, data, name=None):
    data_loader = data.test_loader
    for sample in tqdm(data_loader):
        x, y = sample
        x, y = x.to(config.device), y.to(config.device)

        h = model(x)
        x, h = x.squeeze(0), h.squeeze(0)
        show_masks(x, h)
        pass
        


if __name__ == "__main__":
    checkpoint = torch.load(config.save_path)
    unet = UNet(3, MSRC.NUM_CLASSES)
    unet.load_state_dict(checkpoint["state_dict"])
    demo(unet.to(config.device), MSRC)
    pass
