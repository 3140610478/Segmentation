import os
import sys
import torch
import torch.nn.functional as F
from tqdm import tqdm, trange
from logging import Logger
from typing import Iterable
base_folder = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), ".."))
if base_folder not in sys.path:
    sys.path.append(base_folder)
if True:
    import config


class CrossEntropy(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input: torch.Tensor, target: torch.Tensor, eps=1e-12) -> torch.Tensor:
        output = \
            - target * torch.log(input+eps) \
            - (1-target) * torch.log(1-input+eps)
        output = output.mean()
        return output


class ConfusionMatrix(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input: torch.Tensor, target: torch.Tensor, eps=1e-8) -> torch.Tensor:
        # return in Matrix[n, c_target, c_predicted]
        input_bin = F.one_hot(
            torch.argmax(input, dim=1),
            num_classes=input.shape[1],
        ).permute(0, 3, 1, 2)
        y = target.flatten(start_dim=2)
        h = input_bin.flatten(start_dim=2).float()
        cm = y @ h.permute(0, 2, 1)
        cm = (cm / (y.sum(dim=2, keepdim=True) + eps)).mean(dim=0)
        return cm


class MIoU(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input: torch.Tensor, target: torch.Tensor, eps=1e-5) -> torch.Tensor:
        input_bin = F.one_hot(
            torch.argmax(input, dim=1),
            num_classes=input.shape[1],
        )
        input_bin = input_bin.permute(3, 0, 1, 2).flatten(start_dim=1)
        target_bin = target.bool().permute(1, 0, 2, 3).flatten(start_dim=1)
        intersection = torch.logical_and(input_bin, target_bin)
        intersection = intersection.count_nonzero(dim=1).float()
        union = torch.logical_or(input_bin, target_bin)
        union = union.count_nonzero(dim=1).float()

        mask = torch.logical_or(input_bin, target_bin).any(dim=1)
        miou = (intersection[mask] / union[mask]).mean()
        return miou


ce_fun = CrossEntropy()
miou_fun = MIoU()
cm_fun = ConfusionMatrix()


def get_optimizers(
    model: torch.nn.Module,
    learning_rate: Iterable[float] = (0.1, 0.01, 0.001, 0.0001)
):
    return {
        lr: torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=0.9,
            weight_decay=0.0001,
        )
        for lr in learning_rate
    }


def train_epoch(
    model: torch.nn.Module,
    data,
    logger: Logger,
    epoch: int,
    optimizer: torch.optim.Optimizer,
    best: float = 0,
) -> tuple[float]:
    message = "[loss]\ttrain:{:.8f},\tval:{:.8f}\n[miou]\ttrain:{:.8f},\tval:{:.8f}\n"

    logger.info("\n[Epoch {:0>4d}]".format(epoch+1))
    train_loss, val_loss, train_miou, val_miou = 0, 0, 0, 0

    model.train()
    print("\nTraining:")
    for sample in tqdm(data.train_loader):
        x, y = sample
        x, y = x.to(config.device), y.to(config.device)

        optimizer.zero_grad()

        h = model(x)
        miou = miou_fun(h, y)
        loss = ce_fun(h, y)
        backward = \
            config.loss_weights[0] * ce_fun(h, y) + \
            config.loss_weights[1] * (1 - miou)
        backward.backward()
        optimizer.step()

        train_loss += len(y)*(float(loss))
        train_miou += len(y)*(float(miou))
    train_loss /= data.len_train
    train_miou /= data.len_train

    model.eval()
    print("\nValidating:")
    with torch.no_grad():
        for sample in tqdm(data.val_loader):
            x, y = sample
            x, y = x.to(config.device), y.to(config.device)

            h = model(x)
            miou = miou_fun(h, y)
            loss = ce_fun(h, y)

            val_loss += len(y)*(float(loss))
            val_miou += len(y)*(float(miou))
        val_loss /= data.len_val
        val_miou /= data.len_val

    if val_miou > best:
        torch.save(
            {
                "epoch": epoch,
                "best": best,
                "state_dict": model.state_dict(),
            },
            config.save_path,
        )

    best = max(best, val_miou)
    result = train_loss, val_loss, train_miou, val_miou, best
    print("")
    logger.info(message.format(*result))

    return result


def train_epoch_range(
    model: torch.nn.Module,
    data,
    logger: Logger,
    start: int,
    stop: int,
    optimizer: torch.optim.Optimizer,
    best=0,
) -> None:
    for epoch in trange(start, stop):
        best = train_epoch(model, data, logger, epoch, optimizer, best)[-1]
    return best


def train_until(
    model: torch.nn.Module,
    data,
    logger: Logger,
    threshold: float,
    optimizer: torch.optim.Optimizer,
    best=0,
) -> int:
    epoch = 0
    train_loss, val_loss, train_miou, val_miou, best = \
        (0, 0, 0, 0, best)
    while train_miou <= threshold or val_miou <= threshold:
        train_loss, val_loss, train_miou, val_miou, best = \
            train_epoch(model, data, logger, epoch, optimizer, best)
        epoch += 1
    return best, epoch


@torch.no_grad()
def test(
    model: torch.nn.Module,
    data,
    logger: Logger,
) -> None:
    logger.info("\nTesting: ")

    checkpoint = torch.load(config.save_path)
    model.load_state_dict(checkpoint["state_dict"])

    message = "[loss]\ttest:{:.8f}\n[miou]\ttest:{:.8f}\n"
    test_loss, test_miou = 0, 0
    test_cm = torch.zeros(
        (data.NUM_CLASSES, data.NUM_CLASSES),
        device=config.device
    )
    test_cnt = torch.zeros(
        (data.NUM_CLASSES,),
        device=config.device
    )

    model.eval()
    print("\nTesting:")
    for sample in tqdm(data.test_loader):
        x, y = sample
        x, y = x.to(config.device), y.to(config.device)

        h = model(x)
        miou = miou_fun(h, y)
        loss = ce_fun(h, y)
        cm = cm_fun(h, y)

        test_loss += len(y)*(float(loss))
        test_miou += len(y)*(float(miou))
        test_cm += len(y)*cm
        test_cnt += len(y)*cm.sum(dim=1)
    test_loss /= data.len_test
    test_miou /= data.len_test
    test_cm /= test_cnt.unsqueeze(1)

    print("")
    logger.info(message.format(test_loss, test_miou))
    logger.info(f"Best Epoch: {checkpoint['epoch']+1}")

    test_cm = test_cm.cpu().numpy()
    logger.info("[confusion matrix]")
    logger.info(
        "\n".join(
            [
                "\t".join(
                    [
                        "{:.8f}".format(j) for j in i
                    ]
                ) for i in test_cm
            ]
        )
    )

    return test_cm
