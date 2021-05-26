import numpy as np


def get_loss_func(
    y_pred: np.ndarray, y_true: np.ndarray, loss: str="mse", mean = False
) -> np.ndarray:
    if loss == "mse":
        if mean:
            return 0.5 * np.sum(((y_pred - y_true) ** 2).ravel()) / y_pred.shape[-1]
        else:
            return 0.5 * np.sum((y_pred - y_true) ** 2).ravel()
    else:
        raise Exception(f"loss func: {loss} is not supported!")