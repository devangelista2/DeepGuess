import glob
import os

import matplotlib.pyplot as plt
import numpy as np


def create_path_if_not_exists(path):
    """
    Check if the path exists. If this is not the case, it creates the required folders.
    """
    if not os.path.isdir(path):
        os.makedirs(path)

def viz_and_compare(img, title=None, save_path=None):
    """
    Takes as input a tuple of images and (optionally) titles, and returns
    a sequence of visualization of images with the given title. 
    The tuple of images is assumed to have shape (m, n) or (1, m, n).
    If a single image is given, it is visualized.
    """
    if type(img) is not tuple:
        if len(img.shape) == 3:
            img = img[0]
            
        plt.imshow(img, cmap='gray')
        plt.axis('off')
        if title is not None:
            plt.title(title)
        if save_path is not None:
            plt.savefig(save_path)
        plt.tight_layout()
        plt.show()

    else:
        plt.figure(figsize=(len(img)*2, 2))
        for i, x in enumerate(img):
            if len(x.shape) == 3:
                x = x[0]

            plt.subplot(1, len(img), i+1)
            plt.imshow(x, cmap='gray')
            plt.axis('off')
            if title is not None:
                plt.title(title[i])
        if save_path is not None:
            plt.savefig(save_path)
        plt.tight_layout()
        plt.show()

# Noise is added by noise level
def gaussian_noise(y, noise_level, seed=42):
    np.random.seed(seed)
    e = np.random.randn(*y.shape)
    return e / np.linalg.norm(e.flatten()) * np.linalg.norm(y.flatten()) * noise_level

def load_toml(data_name):
    """Load TOML data from file, given dataset name."""
    from pip._vendor import tomli

    with open(f"./config/{data_name}.toml", 'rb') as f:
        tomo_file = tomli.load(f)
        return tomo_file

def get_save_folder(dig_cfg, is_cfg):
    """
    Return the output folder name, based on the algorithm used to compute the DIR solution.
    """
    save_folder = f"{dig_cfg['algorithm']}/{is_cfg['algorithm']}_p_{is_cfg['p']}"
    return save_folder

def normalize(x):
    """Given an array x, returns its normalized version (i.e. the linear projection into [0, 1])."""
    return (x - x.min()) / (x.max() - x.min())

def corrupt_multiple(K, x, noise_level=0.0):
    """
    Given an array of multiple data x, with shape: (N, c, h, w), return an array of shape (N, m) containing the elements y_delta,
    where:
    
    y_delta = K(x) + e
    """
    N = x.shape[0]
    m = K.shape[0]

    y_delta = np.zeros((N, m))
    for i in range(N):
        x_i = x[i].numpy().flatten()
        y_i = K(x_i)
        y_delta[i] = y_i + gaussian_noise(y_i, noise_level)
    return y_delta

def reconstruct_multiple(solver, y_delta, alg_cfg, x_true):
    """
    Given an array of multiple corrupted data y_delta, with shape: (N, m), 
    return an array of shape (N, n) containing the elements x_sol = solver(y_delta), with specified config values
    """
    N = x_true.shape[0]
    n = np.prod(x_true.shape[1:])

    x_sol = np.zeros((N, n))
    for i in range(N):
        y_delta_i = y_delta[i]
        x_sol[i] = solver(y_delta_i, epsilon=alg_cfg["epsilon_scale"] * np.max(y_delta_i) * np.sqrt(len(y_delta_i)),
                          lmbda=alg_cfg["lmbda"], x_true=x_true[i].numpy().flatten(), p=alg_cfg["p"], starting_point=None,
                          maxiter=alg_cfg['maxit'], verbose=False)[0].flatten()
    return x_sol