import os
import sys
import torch
import plotly.graph_objects as go
from plotly.offline import plot

base_folder = os.path.dirname(os.path.abspath(__file__))
if base_folder not in sys.path:
    sys.path.append(base_folder)
if True:
    from Networks import Training
    from Networks.UNet import UNet
    from Data import MSRC
    from Log.Logger import getLogger
    import config

def run(model, data, name=None):
    if not isinstance(name, str):
        name = f"{model.__class__.__name__}_on_{data.__name__.rsplit('.')[-1]}"
    if config.sam["use_sam"]:
        name += "_SAM"

    logger = getLogger(name)

    optimizers = Training.get_optimizers(
        model,
        (0.1, 0.01, 0.001, 0.0001)
    )

    logger.info(f"{name}\n")
    start_epoch = 0
    best = 0
    # warm-up epoch
    # logger.info("Warm-up")
    # best, start_epoch = Training.train_until(
    #     model, data, logger, 0.1, optimizers[0.01], 0
    # )
    logger.info("learning_rate = 0.1")
    best = Training.train_epoch_range(
        model, data, logger, start_epoch, 80, optimizers[0.1], best
    )
    logger.info("learning_rate = 0.01")
    best = Training.train_epoch_range(
        model, data, logger, 80, 120, optimizers[0.01], best
    )
    # logger.info("learning_rate = 0.001")
    # best = Training.train_epoch_range(
    #     model, data, logger, 100, 150, optimizers[0.001], best
    # )
    # logger.info("learning_rate = 0.0001")
    # best = Training.train_epoch_range(
    #     model, data, logger, 150, 200, optimizers[0.0001], best
    # )
    cm = Training.test(model, data, logger)

    fig = go.Figure(
        data=go.Heatmap(
            z=cm,
            x=tuple(range(data.NUM_CLASSES)),
            y=tuple(range(data.NUM_CLASSES)),
            text=cm,
            texttemplate="%{z:.4f}",
            hoverongaps=False,
        )
    )
    fig.update_layout(
        width=800,
        height=800,
        title="Confusion Matrix",
        xaxis_title="Predicted",
        yaxis_title="Actual",
    )
    plot(
        fig,
        filename=os.path.abspath(os.path.join(
            base_folder, f"./ConfusionMatrix_{name}.html"
        )),
    )


if __name__ == "__main__":
    run(UNet(3, 22).to(config.device), MSRC)
    pass
