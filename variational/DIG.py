
import time

import numpy as np

from models import tf_models
from models.architectures import ResUNet
from variational import solvers


class DeepInitialGuess:
    def __init__(self, K, dataset, device=None):
        self.K = K
        self.dataset = dataset
        self.device = device

        # Get image size by dataset
        if self.dataset == "COULE":
            self.nx, self.ny = 256, 256
        elif self.dataset == "Mayo180_60":
            self.nx, self.ny = 512, 512
        elif self.dataset == "Mayo360_360":
            self.nx, self.ny = 512, 512
        elif self.dataset == "Mayo180_120":
            self.nx, self.ny = 512, 512
        elif self.dataset == "Mayo180_30":
            self.nx, self.ny = 512, 512
        else:
            raise NotImplementedError("Dataset not described yet.")

    def set_mode(self, mode: str) -> None:
        """
        Set the mode for DeepInitalGuess class.
        Available choices: 
            zeros -> Returns zero starting point
            SGP-LPP -> Computes MBIR SGP with p=1 + LPP
            SGP-RISING -> Computes RISING SGP with p=1
            CP-LPP -> Computes MBIR CP with p<1 + LPP
            CP-RISING -> Computes RISING CP with p<1
        """
        self.mode = mode

        if mode == "zeros":
            self.solver = self.zeros
        elif mode == "FBP":
            self.solver = self.FBP
        elif mode == "SGP":
            self.solver = self.SGP
        elif mode == "FBP-LPP":
            self.solver = self.FBP_LPP
        elif mode == "SGP-LPP":
            self.solver = self.SGP_LPP
        elif mode == "SGP-RISING":
            self.solver = self.SGP_RISING
        elif mode == "CP-LPP":
            self.solver = self.CP_LPP
        elif mode == "CP-RISING":
            self.solver = self.CP_RISING
        else:
            raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.solver(*args, **kwargs)

    def zeros(self, *args, **kwargs):
        """
        Returns the zero starting point for both x_pre and x_DIG.
        """
        start_time = time.time()
        x_pre = np.zeros((self.K.shape[-1],))
        print(f"Pre -> Done (in {time.time() - start_time:0.3f}s)", end=" ")

        start_time = time.time()
        x_DIG = np.zeros((self.K.shape[-1],))
        print(f"DIG -> Done (in {time.time() - start_time:0.3f}s)", end=" ")

        return x_pre, x_DIG

    def FBP(self, y_delta, *args, **kwargs):
        """
        Returns the FBP starting point for both x_pre and x_DIG.
        """
        start_time = time.time()
        x_pre = self.K.FBP(y_delta).flatten()
        print(f"Pre -> Done (in {time.time() - start_time:0.3f}s)", end=" ")

        start_time = time.time()
        x_DIG = x_pre
        print(f"DIG -> Done (in {time.time() - start_time:0.3f}s)", end=" ")

        return x_pre, x_DIG

    def SGP(self,
                y_delta,
                lmbda, 
                x_true=None, 
                starting_point=None, 
                m_bb=3, 
                tau_bb=0.5, 
                tol_grad=1e-6, 
                tol_x = 1e-6, 
                k=20, 
                alpha=1,
                verbose=True,
                *args, **kwargs):
        """
        Computes a pre solution by running a few iterations of SGP algorithm for TV-smooth minimization,
        then applies an LPP neural network trained to learn the GT solution to obtain the DIG.
        """
        # This is in Tensorflow
        import os
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

        # Initialize the solver
        solver = solvers.SGPReconstructor(self.K)

        # Computes pre solution
        start_time = time.time()
        x_pre, niter = solver(y_delta, lmbda, x_true, starting_point, m_bb, tau_bb, tol_grad,
                              tol_x, k, alpha, verbose)[:2]
        x_pre = x_pre.flatten()
       
        # Compute DIG solution
        start_time = time.time()
        x_DIG = x_pre
        print(f"DIG -> Done (in {time.time() - start_time:0.3f}s)", end=" ")

        return x_pre, x_DIG
    
    def FBP_LPP(self, y_delta, *args, **kwargs):
        """
        Computes the FBP reconstruction and then applies an LPP neural network.
        """
        # This is in Tensorflow
        import os
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

        # Computes pre solution
        start_time = time.time()
        x_pre = self.K.FBP(y_delta).flatten()
        print(f"Pre -> Done (in {time.time() - start_time:0.3f}s)", end=" ")

        # Get NN model
        model_suffix = "ResUNet"
        model = tf_models.ResUNet((self.nx, self.ny, 1), 4, 2, 64, final_relu=False)

        # Get the weights
        weights_path = f'./model_weights/{self.dataset}/FBP_LPP/'
        model.load_weights(weights_path+model_suffix+'.h5')
        
        # Compute DIG solution
        start_time = time.time()
        x_DIG = model.predict(x_pre.reshape((1, self.nx, self.ny, 1)), verbose=0).flatten()
        print(f"DIG -> Done (in {time.time() - start_time:0.3f}s)", end=" ")

        return x_pre, x_DIG

    def CP_LPP(self, 
               y_delta, 
               epsilon, 
               lmbda, 
               x_true=None, 
               starting_point=None, 
               eta=2e-3, 
               k=10, 
               p=1.0,
               *args, **kwargs):
        """
        Computes a pre solution by running a few iterations of ChambollePock algorithm for TpV minimization,
        then applies an LPP neural network trained to learn the GT solution to obtain the DIG.
        """
        # This is in PyTorch
        import torch

        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Initialize the solver
        solver = solvers.ChambollePockTpVConstrained(self.K)

        # Computes pre solution
        start_time = time.time()
        x_pre, niter = solver(y_delta, epsilon, lmbda, x_true, starting_point, eta, k, p)
        x_pre = x_pre.flatten()
        print(f"Pre -> Done ({niter} iter, in {time.time() - start_time:0.3f}s)", end=" ")

        # Get NN model
        model_suffix = "ResUNet"
        model = ResUNet(img_ch=1, output_ch=1).to(self.device)

        # Get the weights
        weights_path = f'./model_weights/{self.dataset}/LPP_p_0.5/'

        # Load the weights of the model
        model.load_state_dict(torch.load(weights_path+model_suffix+'.pt'))
        model = model.to(self.device)

        # Reshape x_pre and convert to torch tensor
        x_pre_torch = torch.tensor(x_pre, dtype=torch.float32, requires_grad=False).to(self.device)
        x_pre_torch = torch.reshape(x_pre_torch, (1, 1, self.nx, self.ny))

        # Compute DIG solution
        with torch.no_grad():
            start_time = time.time()
            x_DIG = model(x_pre_torch).flatten().cpu().numpy()
            print(f"DIG -> Done (in {time.time() - start_time:0.3f}s)", end=" ")

        return x_pre, x_DIG

    def CP_RISING(self, 
                  y_delta, 
                  epsilon, 
                  lmbda, 
                  x_true=None, 
                  starting_point=None, 
                  eta=2e-3, 
                  k=10, 
                  p=1.0,
                  *args, **kwargs):
        """
        Computes a pre solution by running a few iterations of ChambollePock algorithm for TpV minimization,
        then applies a RISING neural network trained to learn the convergence solution to obtain the DIG.
        """
        # This is in PyTorch
        import torch

        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Initialize the solver
        solver = solvers.ChambollePockTpVConstrained(self.K)

        # Computes pre solution
        start_time = time.time()
        x_pre, niter = solver(y_delta, epsilon, lmbda, x_true, starting_point, eta, k, p)
        x_pre = x_pre.flatten()
        print(f"Pre -> Done ({niter} iter, in {time.time() - start_time:0.3f}s)", end=" ")

        # Get NN model
        model_suffix = "ResUNet"
        model = ResUNet(img_ch=1, output_ch=1).to(self.device)

        # Get the weights
        weights_path = f'./model_weights/{self.dataset}/RISING_p_0.5/'

        # Load the weights of the model
        model.load_state_dict(torch.load(weights_path+model_suffix+'.pt'))
        model = model.to(self.device)

        # Reshape x_pre and convert to torch tensor
        x_pre_torch = torch.tensor(x_pre, dtype=torch.float32, requires_grad=False).to(self.device)
        x_pre_torch = torch.reshape(x_pre_torch, (1, 1, self.nx, self.ny))

        # Compute DIG solution
        with torch.no_grad():
            start_time = time.time()
            x_DIG = model(x_pre_torch).flatten().cpu().numpy()
            print(f"DIG -> Done (in {time.time() - start_time:0.3f}s)", end=" ")

        return x_pre, x_DIG
    
    def SGP_LPP(self,
                y_delta,
                lmbda, 
                x_true=None, 
                starting_point=None, 
                m_bb=3, 
                tau_bb=0.5, 
                tol_grad=1e-6, 
                tol_x = 1e-6, 
                k=100, 
                alpha=1,
                verbose=False,
                *args, **kwargs):
        """
        Computes a pre solution by running a few iterations of SGP algorithm for TV-smooth minimization,
        then applies an LPP neural network trained to learn the GT solution to obtain the DIG.
        """
        # This is in Tensorflow
        import os
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

        # Initialize the solver
        solver = solvers.SGPReconstructor(self.K)

        # Computes pre solution
        start_time = time.time()
        x_pre, niter = solver(y_delta, lmbda, x_true, starting_point, m_bb, tau_bb, tol_grad,
                              tol_x, k, alpha, verbose)[:2]
        x_pre = x_pre.flatten()
        print(f"Pre -> Done ({niter} iter, in {time.time() - start_time:0.3f}s)", end=" ")

        # Get NN model
        model_suffix = "ResUNet"
        model = tf_models.ResUNet((self.nx, self.ny, 1), 4, 2, 64, final_relu=False)

        # Get the weights
        weights_path = f'./model_weights/{self.dataset}/LPP_p_1.0/'
        model.load_weights(weights_path+model_suffix+'.h5')
        
        # Compute DIG solution
        start_time = time.time()
        x_DIG = model.predict(x_pre.reshape((1, self.nx, self.ny, 1)), verbose=0).flatten()
        print(f"DIG -> Done (in {time.time() - start_time:0.3f}s)", end=" ")

        return x_pre, x_DIG

    def SGP_RISING(self,
                y_delta,
                lmbda, 
                x_true=None, 
                starting_point=None, 
                m_bb=3, 
                tau_bb=0.5, 
                tol_grad=1e-6, 
                tol_x = 1e-6, 
                k=100, 
                alpha=1,
                verbose=False,
                *args, **kwargs):
        """
        Computes a pre solution by running a few iterations of SGP algorithm for TV-smooth minimization,
        then applies an RISING neural network trained to learn the convergence solution to obtain the DIG.
        """
        # This is in Tensorflow
        import os
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

        # Initialize the solver
        solver = solvers.SGPReconstructor(self.K)

        # Computes pre solution
        start_time = time.time()
        x_pre, niter = solver(y_delta, lmbda, x_true, starting_point, m_bb, tau_bb, tol_grad,
                              tol_x, k, alpha, verbose)[:2]
        x_pre = x_pre.flatten()
        print(f"Pre -> Done ({niter} iter, in {time.time() - start_time:0.3f}s)", end=" ")

        # Get NN model
        model_suffix = "ResUNet"
        model = tf_models.ResUNet((self.nx, self.ny, 1), 4, 2, 64, final_relu=False)

        # Get the weights
        weights_path = f'./model_weights/{self.dataset}/RISING_p_1.0/'
        model.load_weights(weights_path+model_suffix+'.h5')
        
        # Compute DIG solution
        start_time = time.time()
        x_DIG = model.predict(x_pre.reshape((1, self.nx, self.ny, 1)), verbose=0).flatten()
        print(f"DIG -> Done (in {time.time() - start_time:0.3f}s)", end=" ")

        return x_pre, x_DIG