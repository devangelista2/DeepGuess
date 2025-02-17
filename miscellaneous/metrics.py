import torch
import numpy as np

from skimage.metrics import structural_similarity as ssim

def d(x1, x2):
    """Compute the Euclidean distance between x1 and x2"""
    return np.linalg.norm(x1.flatten() - x2.flatten(), 2)

def np_RE(x_pred, x_true):
    """Compute relative error between numpy arrays."""
    return np.linalg.norm(x_pred.flatten() - x_true.flatten(), 2) / np.linalg.norm(x_true.flatten(), 2)

def torch_RE(x_pred, x_true):
    """Compute the relative error between torch tensors."""
    # Assume x_pred and x_true to be arrays of shape (N, c, m, n)
    N = x_pred.shape[0]

    x_pred = x_pred.reshape((N, -1))
    x_true = x_true.reshape((N, -1))

    return torch.linalg.norm(x_pred - x_true, 2, dim=-1) / torch.linalg.norm(x_true, 2, dim=-1)

def SSIM(x_pred, x_true):
    """Compute the SSIM between two input images x_pred and x_true. Both are assumed to be in the range [0, 1]."""
    return ssim(x_pred, x_true, data_range=x_true.max() - x_true.min())

def batch_SSIM(x_pred, x_true):
    """Compute the SSIM between two batches of images x_pred and x_true. Both are assumed to be in the range [0, 1]."""
    # Convert into numpy arrays
    x_pred = x_pred.detach().cpu().numpy()
    x_true = x_true.detach().cpu().numpy()

    ssim_total = 0
    for i in range(len(x_pred)):
        ssim_total = ssim_total + SSIM(x_pred[i, 0], x_true[i, 0])
    return ssim_total / len(x_pred)

def RMSE(x_pred, x_true):
    """Compute the Root Mean Squared Error (RMSE) between the two input imaged x_pred and x_true."""
    return np.sqrt(np.mean(np.square(x_pred.flatten() - x_true.flatten())))

def RMSE_loss(x_pred, x_true):
    """Compute the Root Mean Squared Error (RMSE) between the two input imaged x_pred and x_true. We assume both of them to be
    pytorch tensors."""
    return torch.sqrt(torch.mean(torch.square(x_pred - x_true)))


def TpV(x, p):
    """Compute the p-norm of the gradient of the input image x."""
    # Normalize x
    if np.linalg.norm(x) != 0:
        x = (x - x.min()) / (x.max() - x.min())
    Dh_x = np.diff(x, n=1, axis=1, prepend=0)
    Dv_x = np.diff(x, n=1, axis=0, prepend=0)

    D_x = np.sqrt(np.square(Dh_x) + np.square(Dv_x))
    return np.power(np.sum(np.power(D_x, p)), 1 / p) / np.prod(D_x.shape)


def rTpV(x_true, x_pred, p=1):
    TpV_true = TpV(x_true, p)
    TpV_pred = TpV(x_pred, p)

    return np.abs(TpV_pred - TpV_true) / np.abs(TpV_true)

