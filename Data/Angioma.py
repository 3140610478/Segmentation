import os
import sys
import random
import pickle
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.transforms.functional import to_tensor, gaussian_blur, resize
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
#             Angioma:
#                 img:
#                     xxx.png
#                     xxx.png
#                     .
#                     .
#                     .
#                     xxx.png
#                 mask:
#                     xxx.png
#                     xxx.png
#                     .
#                     .
#                     .
#                     xxx.png
#             Angioma.zip
#         preprocessed:
#             pseudo_gt:
#                 xxx.png
#                 xxx.png
#                 .
#                 .
#                 .
#                 xxx.png
#             train.pickle    {"imgs": List[torch.Tensor] of RGB images, "gts": List[torch.Tensor] in one_hot encoding}
#             val.pickle      {"imgs": List[torch.Tensor] of RGB images, "gts": List[torch.Tensor] in one_hot encoding}
#             test.pickle     {"imgs": List[torch.Tensor] of RGB images, "gts": List[torch.Tensor] in one_hot encoding}


original_folder = join_path(base_folder, "./Data/original")
if config.sam["use_sam"]:
    preprocessed_folder = join_path(base_folder, "./Data/preprocessed_sam")
else:
    preprocessed_folder = join_path(base_folder, "./Data/preprocessed")
train_path = join_path(preprocessed_folder, "./train.pickle")
val_path = join_path(preprocessed_folder, "./val.pickle")
test_path = join_path(preprocessed_folder, "./test.pickle")


NUM_CLASSES = 2
GT_COLORS = torch.tensor(
    ((0xFF, 0xFF, 0xFF, ),),
    dtype=torch.uint8,
    device=config.device,
)


def _preprocess():
    os.makedirs(join_path(preprocessed_folder, "./pseudo_gt"), exist_ok=True)

    files = os.listdir(join_path(original_folder, "./img"))
    files = [i for i in files if i.endswith(".png")]
    files = random_split(
        files,
        config.splitting,
        torch.Generator().manual_seed(config.seed),
    )

    outputs = (train_path, val_path, test_path)
    groups = ("Train", "Val", "Test",)
    for index in range(len(outputs)):
        imgs, gts = [], []
        print(f"Processing {groups[index]}")
        for file in tqdm(files[index]):
            img = join_path(original_folder, f"./img/{file}")
            gt = join_path(original_folder, f"./mask/{file}")

            img = to_tensor(Image.open(img))
            gt = to_tensor(Image.open(gt))
            img, gt = img.to(config.device), gt.to(config.device)
            masks = (resize(gt, img.shape[-2:]) > 0.5).to(torch.float)

            filename = join_path(
                preprocessed_folder, f"./pseudo_gt/{file}",
            )
            save_masks(masks, GT_COLORS, filename)

            imgs.append(img.cpu())
            gts.append(masks.cpu())

        with open(outputs[index], "wb") as f:
            pickle.dump({"imgs": imgs, "gts": gts}, f)


if config.reload_data or not os.path.exists(preprocessed_folder):
    _preprocess()


class _Angioma_Dataset(Dataset):
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
    train_set = _Angioma_Dataset(train_path, train_transform)
    val_set = _Angioma_Dataset(val_path, val_transform)
    test_set = _Angioma_Dataset(test_path, test_transform)
except:
    _preprocess()
    train_set = _Angioma_Dataset(train_path, train_transform)
    val_set = _Angioma_Dataset(val_path, val_transform)
    test_set = _Angioma_Dataset(test_path, test_transform)
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
