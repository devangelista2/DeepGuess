
import numpy as np
import matplotlib.pyplot as plt
import miscellaneous.utilities as utilities

# Constants
DATASET = "COULE" # in {"COULE", "Mayo180_60", "Mayo360_360"}
MODE = "test" # in {"train", "test"}
IDX = 6 # 10 per Mayo, 6 per COULE

# Define save path
# folder_name = utilities.get_save_folder(DIG_cfg, IS_cfg)
save_path = f"./results/plots/"
utilities.create_path_if_not_exists(save_path)


# APPROXIMATION_legend = ["MBIR", "FBP", "FBP-LPP", "TV-LPP", "TV-RISING"]
# colors = ['darkviolet', 'limegreen', 'green', 'orange', 'red']

'''
x =               [0,       0.010,   0.015,  0.020,    0.030,  0.050,    0.100]
ssim_mbir =       [99.45,   0.5,      90.42,   88.06,   87.04,   85.92,  83.03]
ssim_fbp =        [61.65,   0.5,      50.21,   44.48,   34.85,   23.58,  13.69]
ssim_fbplpp =     [93.91,   0.5,      93.74,   92.77,   86.24,   56.29,  15.73]
ssim_fbplppis =   [99.89,   0.5,      97.62,   97.36,   94.59,   82.34,  80.66]
ssim_tvlpp =      [88.94,   0.5,      86.78,   84.31,   75.75,   58.32,  34.29]
ssim_tvlppis =    [99.36,   0.5,      98.13,   97.77,   93.23,   89.63,  68.36]
ssim_tvrising =   [92.66,   0.5,      91.46,   90.39,   86.97,   75.74,  49.98]
ssim_tvrisingis=  [99.40,   0.5,      97.37,   96.56,   93.60,   88.20,  76.32]
'''

x =               [0,       0.010,   0.015,  0.020,    0.030,  0.050 ]
ssim_mbir =       [99.45,   92.73,    90.42,   88.06,   87.04,   82.92]
ssim_fbp =        [61.65,   55.81,    50.21,   44.48,   34.85,   23.58]
ssim_fbplpp =     [93.91,   93.97,    93.74,   92.27,   86.24,   56.29]
ssim_fbplppis =   [99.89,   98.67,    97.62,   97.36,   94.59,   82.34]
ssim_tvlpp =      [88.94,   88.12,    86.78,   84.31,   75.75,   58.32]
ssim_tvlppis =    [99.36,   98.69,    98.13,   97.77,   93.23,   89.63]
ssim_tvrising =   [92.66,   92.57,    91.46,   90.39,   86.97,   75.74]
ssim_tvrisingis=  [99.40,   98.05,    97.37,   96.56,   93.60,   88.20]



# plot lines 
plt.figure(figsize=(8, 4))
plt.plot(x, ssim_mbir,      marker='o',  label = "CP from zeros", color="darkviolet",
    linestyle='dashed', linewidth=1.5, markersize=5) 
# plt.plot(x, ssim_fbp,       'o', label = "FBP",      color="limegreen", linestyle='dashed', linewidth=1.5, markersize=5) 

plt.plot(x, ssim_fbplpp,    ':*', label = "DG by FBP-LPP",  color="green", markersize=5) 
plt.plot(x, ssim_fbplppis,  'o', label = "CP from DG by FBP-LPP",  color="green", linestyle='dashed', linewidth=1.5, markersize=5) 

plt.plot(x, ssim_tvlpp,    ':*', label = "DG by TV-LPP",  color="orange", markersize=5) 
plt.plot(x, ssim_tvlppis,  'o', label = "CP from DG by TV-LPP",  color="orange", linestyle='dashed', linewidth=1.5, markersize=5) 

plt.plot(x, ssim_tvrising,    ':*', label = "DG by TV-RISING",  color="red", markersize=5) 
plt.plot(x, ssim_tvrisingis,  'o', label = "CP from DG by TV-RISING",  color="red", linestyle='dashed', linewidth=1.5, markersize=5) 

plt.legend() 
plt.xlabel('noise level')
plt.ylabel('SSIM')  
# plt.xlim ([-0.007, 0.052])
plt.ylim ([48, 103])
# plt.show()
plt.savefig(f"{save_path}/{DATASET}_{IDX}_ssim.png", bbox_inches="tight", dpi=400)


'''
#Objective Functions
plt.figure()
for i, approx in enumerate(APPROXIMATION_MODE):
    # Read data
    objf =  np.load(f"{RESULTS_PATH}/obj_f_{APPROXIMATION_MODE[i]}.npy")
    # Plot
    plt.semilogy(iters[:len(objf)], objf, c=colors[i])

plt.legend(APPROXIMATION_legend)
plt.grid()
plt.ylim([3e3,6e3])
plt.xlabel('Iterations')
plt.ylabel('Objective Function')  
plt.savefig(f"{RESULTS_PATH}/obj_f.png", bbox_inches="tight", dpi=400)

# Relative Errors
plt.figure()
for i, approx in enumerate(APPROXIMATION_MODE):
    # Read data
    err =  np.load(f"{RESULTS_PATH}/rel_err_{APPROXIMATION_MODE[i]}.npy")
    # Plot
    plt.plot(iters[:len(err)], err, c=colors[i])

plt.legend(APPROXIMATION_legend)
plt.grid()
plt.ylim([0.07, 0.20])
plt.xlabel('Iterations')
plt.ylabel('Relative Error')  
plt.savefig(f"{RESULTS_PATH}/rel_err.png", bbox_inches="tight", dpi=400)
'''