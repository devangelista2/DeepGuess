import time

import matplotlib.pyplot as plt
import numpy as np

import miscellaneous.datasets as datasets
import miscellaneous.metrics as metrics
import miscellaneous.utilities as utilities
from miscellaneous import metrics
from variational import operators, solvers

###############################
# PARAMETERS
###############################
DATASET = "Mayo360_360" # in {"COULE", "Mayo180_60", "Mayo360_360"}
MODE = "train" # in {"train", "test"}
IS_ALGORITHM = "CP-Unconstrained" # {"CP-Constrained", "CP-Unconstrained"}

#### IS configuration
IS_cfg = {
    "p": 0.5,
    "algorithm": IS_ALGORITHM,
    "epsilon_scale": 3e-2, # 3e-5 per Mayo, 5e-3 per Coule
    "lmbda": 1e-3, # 1e-3 per Mayo, 6e-3 per Coule
    "maxit": 15,   # 500 per Mayo, 200 per Coule
    "verbose": False,
}

############################### Load informations ################
# Get general configuration path.
cfg = utilities.load_toml(DATASET)

# Define paths
gt_path = f'./data/{DATASET}/{MODE}/gt/'
IS_path = f'./data/{DATASET}/{MODE}/{IS_ALGORITHM}_p_{IS_cfg["p"]}_RIS/'

# Since we save the FBP solutions, generate the correspoinding results folder.
FBP_path = f'./results/{DATASET}/{MODE}/FBP/'

# Create output folders if not exists
utilities.create_path_if_not_exists(IS_path)
utilities.create_path_if_not_exists(FBP_path)

gt_data = datasets.ImageDataset(gt_path,
                                numpy=False,
                                )

# Get data shape.
m, n = cfg['m'], cfg['n']

######################## Define forward problem parameters
# Define the operator K
angles = np.linspace(np.deg2rad(cfg['start_angle']), np.deg2rad(cfg['end_angle']), 
                     cfg['n_angles'], endpoint=False)

K = operators.CTProjector((m, n), angles, det_size=cfg['det_size'], geometry=cfg['geometry'])

# Initialize the IS solver
if IS_cfg["algorithm"] == "CP-Constrained":
    solver = solvers.ChambollePockTpVConstrained(K)
elif IS_cfg["algorithm"] == "CP-Unconstrained":
    solver = solvers.ChambollePockTpVUnconstrained(K)

######################## OPTIMIZATION ###########################
RMSE_FBP_vec = np.zeros((len(gt_data), ))
RMSE_vec = np.zeros((len(gt_data), ))
for i in range(len(gt_data)):
    # Compute IS solution
    x_true = gt_data[i].numpy().flatten()

    # Compute the sinogram and add noise
    y = K(x_true)
    y_delta = y + utilities.gaussian_noise(y, cfg["noise_level"])

    # Get FBP reconstruction (for comparison).
    x_fbp = K.FBP(y_delta).reshape((m, n))
    plt.imsave(f'{FBP_path}/{gt_data.get_name(i)}', x_fbp, cmap='gray')

    # Compute x_is
    start_time = time.time()
    x_is, niter = solver(y_delta, epsilon=IS_cfg["epsilon_scale"] * np.max(y_delta) * np.sqrt(len(y_delta)),
                        lmbda=IS_cfg["lmbda"], x_true=x_true, p=IS_cfg["p"], starting_point=None,
                        maxiter=IS_cfg['maxit'], verbose=False)

    # Save the result
    plt.imsave(f'{IS_path}/{gt_data.get_name(i)}', x_is.reshape((m, n)), cmap='gray')
    
    # Compute (and save) metrics
    RMSE_FBP_vec[i] = metrics.SSIM(x_fbp.reshape((m, n)), x_true.reshape((m, n)))
    RMSE_vec[i] = metrics.SSIM(x_is.reshape((m, n)), x_true.reshape((m, n)))

    # Verbose
    print(f"{i+1}/{len(gt_data)} (in {time.time() - start_time:0.4f}s). FBP RMSE: {RMSE_FBP_vec[i]:0.4f}, TpV RMSE: {RMSE_vec[i]:0.4f}.")