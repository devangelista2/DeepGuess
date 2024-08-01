import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import miscellaneous.datasets as datasets
import miscellaneous.utilities as utilities
from miscellaneous import metrics
from models.architectures import ResUNet
from variational import operators, solvers

###############################
# PARAMETERS
###############################
DATASET = "Mayo180_30" # in {"COULE", "Mayo180_60", "Mayo360_360"}
MODE = "train"

ALGORITHM = "CP-Unconstrained" # {"CP-Constrained", "CP-Unconstrained"}
print(f"Test... Dataset {DATASET}.")

BATCH_SIZE = 4
N_EPOCHS = 100
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

#### Algorithm configuration
ALG_cfg = {
    "p": 0.5,
    "algorithm": ALGORITHM,
    "epsilon_scale": 3e-2, # 3e-5 per Mayo, 5e-3 per Coule
    "lmbda": 1e-3, # 1e-3 per Mayo, 6e-3 per Coule
    "maxit": 10,   # 500 per Mayo, 200 per Coule
    "verbose": False,
}

############################### Load informations ################
# Get general configuration path.
cfg = utilities.load_toml(DATASET)

# Define paths
in_path = f'./data/{DATASET}/{MODE}/gt/'
out_path = f'./data/{DATASET}/{MODE}/{ALGORITHM}_p_{ALG_cfg["p"]}/'

train_data = datasets.ImageDataset(in_path, 
                                   label_path=out_path,
                                   numpy=False
                                   )

train_loader = DataLoader(train_data,
                          batch_size=BATCH_SIZE,
                          shuffle=True
                          )

# Get data shape.
m, n = cfg['m'], cfg['n']

######################## Define forward problem parameters
# Define the operator A
angles = np.linspace(np.deg2rad(cfg['start_angle']), np.deg2rad(cfg['end_angle']), 
                     cfg['n_angles'], endpoint=False)

K = operators.CTProjector((m, n), angles, det_size=cfg['det_size'], geometry=cfg['geometry'])

# Initialize the DIG and IS solver
if ALG_cfg["algorithm"] == "CP-Constrained":
    solver = solvers.ChambollePockTpVConstrained(K)
elif ALG_cfg["algorithm"] == "CP-Unconstrained":
    solver = solvers.ChambollePockTpVUnconstrained(K)

######################## Define train parameters
# Get NN model
model_suffix = "ResUNet"
model = ResUNet(img_ch=1, output_ch=1).to(DEVICE)

# Loss function
loss_fn = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Cycle over the epochs
print(f"Training ResUNet model for {N_EPOCHS} epochs and batch size of {BATCH_SIZE}.")
loss_total = np.zeros((N_EPOCHS,))
ssim_total = np.zeros((N_EPOCHS,))
for epoch in range(N_EPOCHS):

    # Cycle over the batches
    epoch_loss = 0.0
    ssim_loss = 0.0
    start_time = time.time()
    for t, data in enumerate(train_loader):
        x_true, x_is = data


        y_delta = utilities.corrupt_multiple(K, x_true, noise_level=cfg['noise_level'])
        x_ris = torch.tensor(utilities.reconstruct_multiple(solver, y_delta, ALG_cfg, x_true).reshape((x_true.shape[0], 1, m, n)),
                             requires_grad=True, dtype=torch.float32)

        # Send x and y to gpu
        x_ris = x_ris.to(DEVICE)
        x_is = x_is.to(DEVICE)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        x_rising = model(x_ris)
        loss = loss_fn(x_rising, x_is)
        loss.backward()
        optimizer.step()

        # update loss
        epoch_loss += loss.item()
        ssim_loss += metrics.batch_SSIM(x_rising, x_is)

        # print the value of loss
        print(f"({t+1}, {epoch+1}) in {time.time() - start_time:0.4f}s - RMSE: {epoch_loss / (t+1)} - SSIM: {ssim_loss / (t+1)}", end="\r")
    print("")

    # Update the history
    loss_total[epoch] = epoch_loss / (t+1)
    ssim_total[epoch] = ssim_loss / (t+1)

# Save the weights
weights_path = f'./model_weights/{DATASET}/RISING_p_{ALG_cfg["p"]}/'

# Create folder is not exists
utilities.create_path_if_not_exists(weights_path)

# Save the weights of the model
torch.save(model.state_dict(), weights_path+'ResUNet.pt')