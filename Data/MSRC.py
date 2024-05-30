import os
import sys
import pickle
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.functional import to_tensor, gaussian_blur
from PIL import Image


base_folder = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), ".."))
if base_folder not in sys.path:
    sys.path.append(base_folder)
if True:
    import config
    from Data.utils import train_transform, val_transform, test_transform, join_path, save_masks, erode, dilate
    from Log.Logger import getLogger


# The file structure was reorganized as follows:
#     Segmentation/Data/:
#         original:
#             msrc2_seg:
#                 images:
#                     xxx.bmp
#                     xxx.bmp
#                     .
#                     .
#                     .
#                     xxx.bmp
#                 gt:
#                     xxx_GT.bmp
#                     xxx_GT.bmp
#                     .
#                     .
#                     .
#                     xxx_GT.bmp
#                 Train.txt       (name of images "xxx.bmp" written in lines)
#                 Validation.txt  (name of images "xxx.bmp" written in lines)
#                 Test.txt        (name of images "xxx.bmp" written in lines)
#                 ReadMe.txt
#             msrc2_seg.zip
#         preprocessed:
#             pseudo_gt:
#                 xxx.bmp
#                 xxx.bmp
#                 .
#                 .
#                 .
#                 xxx.bmp
#             train.pickle    {"imgs": List[torch.Tensor] of RGB images, "gts": List[torch.Tensor] in one_hot encoding}
#             val.pickle      {"imgs": List[torch.Tensor] of RGB images, "gts": List[torch.Tensor] in one_hot encoding}
#             test.pickle     {"imgs": List[torch.Tensor] of RGB images, "gts": List[torch.Tensor] in one_hot encoding}


original_folder = join_path(base_folder, "./Data/original/msrc2_seg")
if config.sam["use_sam"]:
    preprocessed_folder = join_path(base_folder, "./Data/preprocessed_sam")
else:
    preprocessed_folder = join_path(base_folder, "./Data/preprocessed")
train_path = join_path(preprocessed_folder, "./train.pickle")
val_path = join_path(preprocessed_folder, "./val.pickle")
test_path = join_path(preprocessed_folder, "./test.pickle")

NUM_CLASSES = 22
GT_COLORS = torch.tensor(
    (
        (0x00, 0x00, 0x00, ),
        (0x00, 0x00, 0x80, ),
        (0x00, 0x40, 0x00, ),
        (0x00, 0x80, 0x00, ),
        (0x00, 0x80, 0x80, ),
        (0x00, 0xc0, 0x00, ),
        (0x00, 0xc0, 0x80, ),
        (0x40, 0x00, 0x80, ),
        (0x40, 0x40, 0x00, ),
        (0x40, 0x80, 0x00, ),
        (0x40, 0x80, 0x80, ),
        (0x80, 0x00, 0x00, ),
        (0x80, 0x40, 0x00, ),
        (0x80, 0x40, 0x80, ),
        (0x80, 0x80, 0x00, ),
        (0x80, 0x80, 0x80, ),
        (0x80, 0xc0, 0x80, ),
        (0xc0, 0x00, 0x00, ),
        (0xc0, 0x00, 0x80, ),
        (0xc0, 0x40, 0x00, ),
        (0xc0, 0x80, 0x00, ),
        (0xc0, 0x80, 0x80, ),
    ),
    dtype=torch.uint8,
    device=config.device,
)


def _preprocess():
    os.makedirs(join_path(preprocessed_folder, "./pseudo_gt"), exist_ok=True)

    if config.sam["use_sam"]:
        from Data.SAM import SEM
        sem = SEM().to(config.device)
        # from Data.SAM import SAM
        # sam = SAM().to(config.device)
    groups = ("./Train.txt", "./Validation.txt", "./Test.txt")
    outputs = (train_path, val_path, test_path)
    for index in (0, 1, 2):
        with open(join_path(original_folder, groups[index])) as f:
            files = [i.rstrip()[:-4] for i in f.readlines()]

        imgs, gts = [], []
        print(f"Processing {groups[index]}")
        for file in tqdm(files):
            img = join_path(original_folder, f"./images/{file}.bmp")
            gt = join_path(original_folder, f"./gt/{file}_GT.bmp")

            img = to_tensor(Image.open(img))
            gt = (to_tensor(Image.open(gt))*255).to(torch.uint8)
            img, gt = img.to(config.device), gt.to(config.device)

            masks = torch.empty(
                (NUM_CLASSES, gt.shape[-2], gt.shape[-1]),
                dtype=torch.bool,
                device=config.device,
            )
            for i, color in enumerate(GT_COLORS):
                mask = (gt == color.reshape((-1, 1, 1)))
                masks[i] = mask.all(dim=0)
            masks = masks.float()

            scaling_rate = 256 / max(img.shape[-2:])
            shape = torch.tensor(img.shape[-2:], device=config.device)
            shape = (shape * scaling_rate).int() + config.DownscaleFactor // 2
            shape = shape - shape % config.DownscaleFactor
            shape = shape.cpu().tolist()
            img = torch.cat((img, masks), dim=0).unsqueeze(0)
            img = F.interpolate(img, shape, mode='bilinear').squeeze(0)
            img, masks = img[0:3], img[3:]

            masks = gaussian_blur(masks, 11, config.sam["gaussian_blur_sigma"])
            masks = F.one_hot(masks.argmax(0), NUM_CLASSES)
            masks = masks.permute(2, 0, 1).float()

            if config.sam["use_sam"]:
                masks = sem(img, masks, plot=False)
                # masks = sam(img, masks)
                masks = gaussian_blur(
                    masks, 11, config.sam["gaussian_blur_sigma"])
                masks = F.one_hot(masks.argmax(0), NUM_CLASSES)
                masks = masks.permute(2, 0, 1).float()
            masks[0] = masks[0] + \
                torch.logical_not(torch.any(masks.bool(), dim=0)).float()

            filename = join_path(
                preprocessed_folder, f"./pseudo_gt/{file}.bmp",
            )
            save_masks(masks, GT_COLORS, filename)

            imgs.append(img.cpu())
            gts.append(masks.cpu())

        with open(outputs[index], "wb") as f:
            pickle.dump({"imgs": imgs, "gts": gts}, f)


if config.reload_data or not os.path.exists(preprocessed_folder):
    _preprocess()


class _MSRC_Dataset(Dataset):
    def __init__(self, path, transform=None):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.imgs, self.gts = data["imgs"], data["gts"]
        self.LEN = len(self.imgs)
        for i in range(self.LEN):
            self.imgs[i].requires_grad_(False)
            self.gts[i].requires_grad_(False)
        self.transform = transform

    def __len__(self):
        return self.LEN

    def __getitem__(self, index):
        img, gt = self.imgs[index].clone(), self.gts[index].clone()
        img, gt = img.to(config.device), gt.to(config.device)
        if self.transform is not None:
            item = torch.cat((img, gt), dim=0)
            item = self.transform(item)
            img, gt = item[0:3], item[3:]
            gt = (gt > 0.5).float()
        return img, gt


try:
    train_set = _MSRC_Dataset(train_path, train_transform)
    val_set = _MSRC_Dataset(val_path, val_transform)
    test_set = _MSRC_Dataset(test_path, test_transform)
except:
    _preprocess()
    train_set = _MSRC_Dataset(train_path, train_transform)
    val_set = _MSRC_Dataset(val_path, val_transform)
    test_set = _MSRC_Dataset(test_path, test_transform)
len_train, len_val, len_test = \
    len(train_set), len(val_set), len(test_set)

train_loader = DataLoader(
    train_set, batch_size=config.batch_size, shuffle=True)
val_loader = DataLoader(
    val_set, batch_size=config.batch_size, shuffle=True)
test_loader = DataLoader(
    test_set, batch_size=config.batch_size, shuffle=True)

if __name__ == "__main__":
    for img, gt in train_loader:
        print(img.shape, gt.shape)
