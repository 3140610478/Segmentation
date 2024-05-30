import os
import sys
import torch
import torch.nn.functional as F
from segment_anything import SamPredictor, SamAutomaticMaskGenerator, sam_model_registry
base_folder = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), ".."))
if base_folder not in sys.path:
    sys.path.append(base_folder)
if True:
    import config
    from Data.utils import join_path

_sam_checkpoints = {
    "vit_b": join_path(base_folder, "./Data/sam_weights/sam_vit_b_01ec64.pth"),
    "vit_h": join_path(base_folder, "./Data/sam_weights/sam_vit_h_4b8939.pth")
}


class SAM(torch.nn.Module):
    from cv2 import connectedComponents

    def __init__(self):
        super().__init__()
        sam = sam_model_registry[config.sam["model_type"]](
            checkpoint=_sam_checkpoints[config.sam["model_type"]],
        ).to(config.device)
        sam.requires_grad_(False)
        self._predictor = SamPredictor(sam)

    def forward(self, img: torch.Tensor, gt: torch.Tensor, prompt_mode="point", plot=False):
        device = img.device
        output_shape = gt.shape

        p_channel = gt.float().mean(dim=(1, 2))

        img_np = img.permute(1, 2, 0).cpu().numpy()
        self._predictor.set_image(img_np)

        if prompt_mode == "point":
            gt_channels = gt.to(torch.uint8)
            gt_out = -torch.ones(output_shape,
                                 dtype=torch.float32, device=device)
            gt_out = gt_out * torch.inf

            grid = torch.zeros(
                output_shape[-2:], dtype=torch.bool, device=device)
            grid[::config.sam["grid_size"],
                 ::config.sam["grid_size"]] = 1

            for channel in range(len(gt_out)):
                if not p_channel[channel]:
                    continue
                mask_channel = (gt_channels[channel] * 255).numpy()
                _, mask_channel = self.connectedComponents(mask_channel)
                mask_channel = torch.from_numpy(
                    mask_channel).to(torch.int64).to(device)
                mask_channel = mask_channel - mask_channel.min()
                num_instance = mask_channel.max()
                mask_instance = F.one_hot(mask_channel, num_instance+1)
                mask_instance = mask_instance.permute(
                    2, 0, 1)[1:].bool()
                point_coords = torch.nonzero(grid).numpy()
                point_labels_instance = torch.masked_select(
                    mask_instance, grid,
                ).reshape(num_instance, -1).numpy()

                for point_labels in point_labels_instance:
                    mask = torch.from_numpy(
                        self._predictor.predict(
                            point_coords=point_coords,
                            point_labels=point_labels,
                            multimask_output=True,
                            return_logits=True,
                        )[0],
                    ).to(device)
                    mask_bin = mask > self._predictor.model.mask_threshold
                    miou = torch.div(
                        (torch.logical_and(mask_bin, gt[channel]).float().sum(
                            dim=(-2, -1)) + 1e-5),
                        (torch.logical_or(mask_bin, gt[channel]).float().sum(
                            dim=(-2, -1)) + 1e-5),
                    )
                    mask = mask[torch.argmax(miou)]
                    mask = F.sigmoid(mask)
                    mask_bin = mask_bin[torch.argmax(miou)]

                    gt_out[channel] = torch.max(gt_out[channel], mask.clone())
        elif prompt_mode == "box":
            pass

        gt_out = gt_out.permute(1, 2, 0)
        # gt_out = gt_out / gt_out.std(dim=(0, 1)) - gt_out.mean(dim=(0, 1))
        gt_out = F.softmax(gt_out, dim=-1)
        # gt_out = gt_out * p_channel
        gt_out = gt_out.argmax(dim=-1)
        gt_out = F.one_hot(gt_out, num_classes=output_shape[0])
        gt_out = gt_out.permute(2, 0, 1)

        if plot:
            from Data.utils import show_masks
            show_masks(img, gt_out)
        return gt_out


class SEM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        sem = sam_model_registry[config.sam["model_type"]](
            checkpoint=_sam_checkpoints[config.sam["model_type"]],
        ).to(config.device)
        sem.requires_grad_(False)
        self._predictor = SamAutomaticMaskGenerator(
            sem,
            pred_iou_thresh=0.75,
            stability_score_thresh=0.75,
            crop_n_layers=1,
            crop_n_points_downscale_factor=2,
        )

    def forward(self, img: torch.Tensor, gt: torch.Tensor, plot=False):
        device = img.device
        output_shape = gt.shape

        gt_out = torch.zeros_like(gt)
        imgnp = img.permute(1, 2, 0).cpu().numpy()
        pred = self._predictor.generate(imgnp)
        masks = [torch.from_numpy(mask["segmentation"]).float().to(device)
                 for mask in pred]

        for mask in masks:
            intersection = torch.logical_and(mask, gt).float()
            intersection = intersection.sum(dim=(1, 2)) / mask.sum()
            index = torch.argmax(intersection, dim=0)
            gt_out[index] = gt_out[index] + mask

        gt_out = gt_out.bool()
        gt_out[0] = torch.logical_or(
            gt_out[0],
            torch.logical_not(torch.any(gt_out, dim=0)),
        )
        gt_out = gt_out.float()

        if plot:
            from Data.utils import show_masks
            show_masks(img, gt_out)
        return gt_out
    
    def to(self, *args, **kwargs):
        self._predictor.predictor.model.to(*args, **kwargs)
        return super().to(*args, **kwargs)
