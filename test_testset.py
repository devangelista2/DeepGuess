import time

import matplotlib.pyplot as plt
import numpy as np

import miscellaneous.datasets as datasets
import miscellaneous.utilities as utilities
from miscellaneous import metrics
from variational import DIG, operators, solvers

###############################
# PARAMETERS
###############################
DATASET = "COULE" # in {"COULE", "Mayo180_60", "Mayo360_360"}
MODE = "test" # in {"train", "test"}
# IDX = 6 # 10 per Mayo, 6 per COULE

DIG_ALGORITHM = "zeros" # in {"zeros", "FBP-LPP", "SGP-LPP", "SGP-RISING", "CP-LPP", "CP-RISING"}
IS_ALGORITHM = "CP-Unconstrained" # {"CP-Constrained", "CP-Unconstrained"}
print(f"Test... Dataset {DATASET}, Algorithm {DIG_ALGORITHM}.")

#### DIG configuration
DIG_cfg = {
    "algorithm": DIG_ALGORITHM,
    "epsilon_scale": 3e-5, # 3e-5 per Mayo, 5e-3 per Coule
    "lmbda": 0.001, # 1e-5 per SGP con Mayo180_60, senno 1e-3
    "k": 10,   # 15 per SGP con Mayo180_60, senno 10
}

#### IS configuration
IS_cfg = {
    "p": 0.5,
    "algorithm": IS_ALGORITHM,
    "epsilon_scale": 5e-3, # 3e-5 per Mayo, 5e-3 per Coule
    "lmbda": 6e-3, # 1e-3 per Mayo p=0.5, 5e-4 per Mayo p=0.2, 6e-3 per Coule
    "maxit": 200,   # 500 per Mayo, 200 per Coule
    "verbose": False,
}

############################### Load informations ################
# Get general configuration path.
cfg = utilities.load_toml(DATASET)

# Define paths
gt_path = f'./data/{DATASET}/{MODE}/gt/'

gt_data = datasets.ImageDataset(gt_path,
                                numpy=False
                                )
# Get data shape.
m, n = cfg['m'], cfg['n']


######################## Define forward problem parameters
# Define the operator A
angles = np.linspace(np.deg2rad(cfg['start_angle']), np.deg2rad(cfg['end_angle']), 
                     cfg['n_angles'], endpoint=False)

K = operators.CTProjector((m, n), angles, det_size=cfg['det_size'], geometry=cfg['geometry'])

# Initialize the DIG and IS solver
DIG_solver = DIG.DeepInitialGuess(K, DATASET)
DIG_solver.set_mode(DIG_cfg["algorithm"])
if IS_cfg["algorithm"] == "CP-Constrained":
    solver = solvers.ChambollePockTpVConstrained(K)
elif IS_cfg["algorithm"] == "CP-Unconstrained":
    solver = solvers.ChambollePockTpVUnconstrained(K)

# Define save path (for DIG and IS data)
folder_name = utilities.get_save_folder(DIG_cfg, IS_cfg)
save_path = f"./results/{DATASET}/{MODE}/{folder_name}/"
utilities.create_path_if_not_exists(save_path)


nimages = 25

ssim_pre = np.zeros((nimages, ))
ssim_DIG = np.zeros((nimages, ))
ssim_IS = np.zeros((nimages, ))

RE_pre = np.zeros((nimages, ))
RE_DIG = np.zeros((nimages, ))
RE_IS = np.zeros((nimages, ))


for i in range(nimages):  # range( len(gt_data) )
    IDX = i
    print(f"******   IMAGE {IDX}")     
    x_true = gt_data[IDX].numpy().flatten()

    # Compute the sinogram and add noise
    y = K(x_true)
    y_delta = y + utilities.gaussian_noise(y, cfg["noise_level"], seed=42)

    # Choose initial guess
    # print(f"Computing DIG...", end=" ")
    x_pre, x_DIG = DIG_solver(y_delta, epsilon=DIG_cfg["epsilon_scale"] * np.max(y_delta) * np.sqrt(len(y_delta)),
                            lmbda=DIG_cfg["lmbda"], x_true=x_true, k=DIG_cfg["k"])

    # From x_DIG, compute x_is
    start_time = time.time()
    x_is, niter = solver(y_delta, epsilon=IS_cfg["epsilon_scale"] * np.max(y_delta) * np.sqrt(len(y_delta)),
                        lmbda=IS_cfg["lmbda"], x_true=x_true, p=IS_cfg["p"], starting_point=x_DIG,
                        maxiter=IS_cfg['maxit'], verbose=IS_cfg['verbose'])
    #print(f"IS -> Done ({niter} iter, in {time.time() - start_time:0.3f}s)")

    print(f"SSIM = ", metrics.SSIM(x_is.reshape((m, n)), x_true.reshape((m, n))), "RE = ",  metrics.np_RE(x_is.reshape((m, n)), x_true.reshape((m, n))))
    # Compute metrics
    ssim_pre[i] = metrics.SSIM(x_pre.reshape((m, n)), x_true.reshape((m, n)))
    ssim_DIG[i] = metrics.SSIM(x_DIG.reshape((m, n)), x_true.reshape((m, n)))
    ssim_IS[i] = metrics.SSIM(x_is.reshape((m, n)), x_true.reshape((m, n)))

    RE_pre[i] = metrics.np_RE(x_pre.reshape((m, n)), x_true.reshape((m, n)))
    RE_DIG[i] = metrics.np_RE(x_DIG.reshape((m, n)), x_true.reshape((m, n)))
    RE_IS[i] = metrics.np_RE(x_is.reshape((m, n)), x_true.reshape((m, n)))

print("\n")
print(f"SSIM PRE -> Mean: {ssim_pre.mean()}, std: {ssim_pre.std()}.")
print(f"SSIM DG -> Mean: {ssim_DIG.mean()}, std: {ssim_DIG.std()}.")
print(f"SSIM IS -> Mean: {ssim_IS.mean()}, std: {ssim_IS.std()}.")

print(f"RE PRE -> Mean: {RE_pre.mean()}, std: {RE_pre.std()}.")
print(f"RE DG -> Mean: {RE_DIG.mean()}, std: {RE_DIG.std()}.")
print(f"RE IS -> Mean: {RE_IS.mean()}, std: {RE_IS.std()}.")

print(ssim_IS)
print(RE_IS)
'''
# Save in png
plt.imsave(save_path + "PRE_" + gt_data.get_name(IDX), x_pre.reshape((m, n)), cmap='gray')
plt.imsave(save_path + "DIG_" + gt_data.get_name(IDX), x_DIG.reshape((m, n)), cmap='gray')
plt.imsave(save_path + "IS_" + gt_data.get_name(IDX), x_is.reshape((m, n)), cmap='gray')

RMSE_pre = metrics.RMSE(x_pre.reshape((m, n)), x_true.reshape((m, n)))
RMSE_DIG = metrics.RMSE(x_DIG.reshape((m, n)), x_true.reshape((m, n)))
RMSE_IS = metrics.RMSE(x_is.reshape((m, n)), x_true.reshape((m, n)))

rTpV_pre = metrics.rTpV(x_pre.reshape((m, n)), x_true.reshape((m, n)))
rTpV_DIG = metrics.rTpV(x_DIG.reshape((m, n)), x_true.reshape((m, n)))
rTpV_IS = metrics.rTpV(x_is.reshape((m, n)), x_true.reshape((m, n)))

# Verbose
print(f"SSIM -> PRE: {ssim_pre:0.4f}, DIG: {ssim_DIG:0.4f}, IS: {ssim_IS:0.4f}.")
print(f"RE -> PRE: {RE_pre:0.4f}, DIG: {RE_DIG:0.4f}, IS: {RE_IS:0.4f}.")
#print(f"RMSE -> PRE: {RMSE_pre:0.4f}, DIG: {RMSE_DIG:0.4f}, IS: {RMSE_IS:0.4f}.")
#print(f"rTpV -> PRE: {rTpV_pre:0.4f}, DIG: {rTpV_DIG:0.4f}, IS: {rTpV_IS:0.4f}.")
'''

