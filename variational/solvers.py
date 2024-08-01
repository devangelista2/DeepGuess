import numba as nb
import numpy as np

from . import operators, utils


##################################
# CP-TpV Constrained
##################################
class ChambollePockTpVConstrained:
    def __init__(self, A):
        self.A = A

        self.m, self.n = A.shape
        
        # Generate Gradient operators
        self.grad = operators.Gradient(1, (int(np.sqrt(self.n)), int(np.sqrt(self.n))))

    def __call__(self, b, epsilon, lmbda, x_true=None, starting_point=None, eta=2e-3, maxiter=100, p=1, verbose=False):
        # Compute the approximation to || A ||_2
        nu = np.sqrt(self.power_method(self.A, num_iterations=10) / self.power_method(self.grad, num_iterations=10))

        # Generate concatenate operator
        K = operators.ConcatenateOperator(self.A, self.grad)

        Gamma = np.sqrt(self.power_method(K, num_iterations=10))

        # Compute the parameters given Gamma
        tau = 1 / Gamma
        sigma = 1 / Gamma
        theta = 1
        
        # Iteration counter
        k = 0

        # Initialization
        if starting_point is None:
            x = np.zeros((self.n, 1))
        else:
            x = np.expand_dims(starting_point, -1)
        y = np.zeros((self.m, 1))
        w = np.zeros((2 * self.n, 1))

        xx = x

        # Initialize errors
        rel_err = np.zeros((maxiter+1, 1))
        residues = np.zeros((maxiter+1, 1))

        # Stopping conditions
        con = True
        while con and (k < maxiter):
            # Update y
            yy = y + sigma * np.expand_dims(self.A(xx) - b, -1)
            y = max(np.linalg.norm(yy) - (sigma*epsilon), 0) * yy / np.linalg.norm(yy)

            # Compute the magnitude of the gradient
            grad_x = self.grad(xx)
            grad_mag = np.square(grad_x[:len(grad_x)//2]) + np.square(grad_x[len(grad_x)//2:])

            # Compute the reweighting factor
            W = np.expand_dims(np.power(np.sqrt(eta**2 + grad_mag) / eta, p-1), -1)
            WW = np.concatenate((W, W), axis=0)

            # Update w
            x_grad = np.expand_dims(self.grad(xx), -1)
            ww = w + sigma * x_grad

            abs_ww = np.zeros((self.n, 1))
            abs_ww = np.square(ww[:self.n]) + np.square(ww[self.n:])
            abs_ww = np.concatenate((abs_ww, abs_ww), axis=0)
            
            lmbda_vec_over_nu = lmbda * WW / nu
            w = lmbda_vec_over_nu * ww / np.maximum(lmbda_vec_over_nu, abs_ww)

            # Save the value of x
            xtmp = x

            # Update x
            x = xtmp - tau * (np.expand_dims(self.A.T(y), -1) + nu * np.expand_dims(self.grad.T(w), -1))

            # Project x to (x>0)
            x[x<0] = 0

            # Compte signed x
            xx = x + theta * (x - xtmp)

            # Compute relative error
            if x_true is not None:
                rel_err[k] = np.linalg.norm(xx.flatten() - x_true.flatten()) / np.linalg.norm(x_true.flatten())

            # Compute the magnitude of the gradient of the actual iterate
            grad_x = self.grad(xx)
            grad_mag = np.expand_dims(np.sqrt(np.square(grad_x[:len(grad_x)//2]) + np.square(grad_x[len(grad_x)//2:])), -1)

            # Compute the value of TpV by reweighting
            ftpv = np.sum(np.abs(W * grad_mag))
            res = np.linalg.norm(self.A(xx) - b, 2)**2
            residues[k] = 0.5 * res + lmbda * ftpv

            # Stopping criteria
            c = np.sqrt(res) / (np.max(b) * np.sqrt(self.m))
            d_abs = np.linalg.norm(x.flatten() - xtmp.flatten(), 2)

            if (c>= 9e-6) and (c<=1.1e-5):
                con = False

            if d_abs < 1e-3*(1 + np.linalg.norm(xtmp.flatten(), 2)):  
                con = False

            # Update k
            k = k + 1
            if verbose:
                print(f"Iteration {k}, RE: {rel_err[k-1]}.")

        return x, k
    
    def power_method(self, A, num_iterations: int):
        b_k = np.random.rand(A.shape[1])

        for _ in range(num_iterations):
            # calculate the matrix-by-vector product Ab
            b_k1 = A.T(A(b_k))

            # calculate the norm
            b_k1_norm = np.linalg.norm(b_k1)

            # re normalize the vector
            b_k = b_k1 / b_k1_norm

        return b_k1_norm
    
    def compute_obj_value(self, x, y, lmbda, p, eta):
        # Compute the value of the objective function TpV by reweighting
        grad_x = np.expand_dims(self.grad(x), -1)
        grad_mag = np.square(grad_x[:len(grad_x)//2]) + np.square(grad_x[len(grad_x)//2:])
        W = np.power(np.sqrt(eta**2 + grad_mag) / eta, p-1)

        ftpv = np.sum(np.abs(W * np.sqrt(grad_mag)))
        return 0.5 * np.linalg.norm(self.A(x) - y, 2)**2 + lmbda * ftpv
    

##################################
# CP-TpV Unconstrained
##################################
class ChambollePockTpVUnconstrained:
    def __init__(self, A):
        self.A = A

        self.m, self.n = A.shape
        
        # Generate Gradient operators
        self.grad = operators.Gradient(1, (int(np.sqrt(self.n)), int(np.sqrt(self.n))))

    def __call__(
        self,
        b,
        lmbda,
        x_true=None,
        starting_point=None,
        eta=2e-3,
        maxiter=200,
        p=1,
        verbose=False,
        *args, **kwargs
    ):
        """
        Chambolle-Pock algorithm for the minimization of the objective function
            ||K*x - d||_2^2 + Lambda*TpV(x)
        by reweighting

        K : projection operator
        Lambda : weight of the TV penalization (the higher Lambda, the more sparse is the solution)
        L : norm of the operator [P, Lambda*grad] (see power_method)
        maxit : number of iterations
        """
            # Compute the approximation to || A ||_2
        nu = np.sqrt(self.power_method(self.A, num_iterations=10) / self.power_method(self.grad, num_iterations=10))

        # Generate concatenate operator
        K = operators.ConcatenateOperator(self.A, self.grad)

        Gamma = np.sqrt(self.power_method(K, num_iterations=10))

        # Compute the parameters given Gamma
        tau = 1 / Gamma
        sigma = 1 / Gamma
        theta = 1
        
        from miscellaneous import metrics

        # Iteration counter
        k = 0

        # Initialization
        if starting_point is None:
            x = np.zeros((self.n, 1))
        else:
            x = np.expand_dims(starting_point, -1)
        y = np.zeros((self.m, 1))
        w = np.zeros((2 * self.n, 1))

        xx = x

        # Initialize errors
        rel_err = np.zeros((maxiter+1, 1))
        residues = np.zeros((maxiter+1, 1))

        # Stopping conditions
        con = True
        while con and (k < maxiter):
            # Update y
            y = (y + sigma * np.expand_dims(self.A(xx) - b, -1)) / (1 + lmbda * sigma)

            # Compute the magnitude of the gradient
            grad_x = self.grad(xx)
            grad_mag = np.square(grad_x[:len(grad_x)//2]) + np.square(grad_x[len(grad_x)//2:])

            # Compute the reweighting factor
            W = np.expand_dims(np.power(np.sqrt(eta**2 + grad_mag) / eta, p-1), -1)
            WW = np.concatenate((W, W), axis=0)

            # Update w
            x_grad = np.expand_dims(self.grad(xx), -1)
            ww = w + sigma * x_grad

            abs_ww = np.zeros((self.n, 1))
            abs_ww = np.square(ww[:self.n]) + np.square(ww[self.n:])
            abs_ww = np.concatenate((abs_ww, abs_ww), axis=0)
            
            lmbda_vec_over_nu = lmbda * WW / nu
            w = lmbda_vec_over_nu * ww / np.maximum(lmbda_vec_over_nu, abs_ww)

            # Save the value of x
            xtmp = x

            # Update x
            x = xtmp - tau * (np.expand_dims(self.A.T(y), -1) + nu * np.expand_dims(self.grad.T(w), -1))

            # Project x to (x>0)
            x[x<0] = 0

            # Compte signed x
            xx = x + theta * (x - xtmp)
        

            # Compute relative error
            if x_true is not None:
                rel_err[k] = np.linalg.norm(xx.flatten() - x_true.flatten()) / np.linalg.norm(x_true.flatten())
                ssim_it = metrics.SSIM(xx.flatten(), x_true.flatten())

            # Compute the magnitude of the gradient of the actual iterate
            grad_x = self.grad(xx)
            grad_mag = np.expand_dims(np.sqrt(np.square(grad_x[:len(grad_x)//2]) + np.square(grad_x[len(grad_x)//2:])), -1)

            # Compute the value of TpV by reweighting
            ftpv = np.sum(np.abs(W * grad_mag))
            res = np.linalg.norm(self.A(xx) - b, 2)**2
            residues[k] = 0.5 * res + lmbda * ftpv

            # Stopping criteria
            c = np.sqrt(res) / (np.max(b) * np.sqrt(self.m))
            d_abs = np.linalg.norm(x.flatten() - xtmp.flatten(), 2)

            if (c>= 9e-6) and (c<=1.1e-5):
                con = False

            #if d_abs < 1e-3*(1 + np.linalg.norm(xtmp.flatten(), 2)):  
            #    con = False

            # Update k
            k = k + 1
            if verbose:
                print(f"Iteration {k}, RE: {rel_err[k-1]}, SSIM: {ssim_it}")

        return x, k
    
    def power_method(self, A, num_iterations: int):
        b_k = np.random.rand(A.shape[1])

        for _ in range(num_iterations):
            # calculate the matrix-by-vector product Ab
            b_k1 = A.T(A(b_k))

            # calculate the norm
            b_k1_norm = np.linalg.norm(b_k1)

            # re normalize the vector
            b_k = b_k1 / b_k1_norm

        return b_k1_norm

    def obj_function(self, x, y_delta, lmbda):
        # Compute reweighted gradient of x
        grad_x = self.gradient(x)
        grad_mag = utils.gradient_magnitude(grad_x)

        # Compute the residual and the regularization term
        res = np.linalg.norm(self.K(x).flatten() - y_delta.flatten(), 2)
        tpv = np.sum(np.power(grad_mag, self.p))

        return 0.5 * res**2 + lmbda * tpv



##################################
# CGLS
##################################
class CGLS:
    def __init__(self, A):
        self.A = A

        self.m, self.n = A.shape
    
    def __call__(self, b, x_true, starting_point, maxiter=100, tolf=1e-6, tolx=1e-6):
        # Initialization
        if starting_point is None:
            x = np.zeros((self.n,))
        else:
            x = starting_point.flatten()
        d  = b
        r0 = self.A.T(b)
        p = r0
        t = self.A @ p
        
        x = x0
        r = r0 
        k = 0

        err_vec = np.zeros((maxiter, 1))
        err_vec[k] = np.linalg.norm(x - x_true) / np.linalg.norm(x_true)
        while np.linalg.norm(r) > tolf and err_vec[k] > tolx and k < maxiter -1:
            x0 = x

            alpha = np.linalg.norm(r0, 2)**2 / np.linalg.norm(t, 2)**2
            x = x0 + alpha * p
            d = d - alpha * t
            r = self.A.T(d)
            beta = np.linalg.norm(r, 2)**2 / np.linalg.norm(r0, 2)**2
            p = r + beta * p
            t = self.A @ p
            k = k + 1

            r0 = r
            err_vec[k] = np.linalg.norm(x - x_true) / np.linalg.norm(x_true)
        err_vec = err_vec[:k+1]

        return x



##################################
# SGP TV
##################################
class SGPReconstructor():
    def __init__(self, A):
        self.A = A

        self.m, self.n = A.shape

    def __call__(self, 
                 b, 
                 lmbda, 
                 x_true=None, 
                 starting_point=None, 
                 m_bb=3, 
                 tau_bb=0.5, 
                 tol_grad=1e-6, 
                 tol_x = 1e-6, 
                 maxiter=100, 
                 alpha=1,
                 verbose=True):
        
        # Initialization
        if starting_point is None:
            x = np.zeros((self.n,))
        else:
            x = starting_point.flatten()
        self.b = b
        self.lmbda = lmbda

        # SGP additional parameters
        self.alpha = alpha
        self.m_bb = m_bb
        self.tau_bb = tau_bb
        self.alpha_bb2_vec = np.array([self.alpha] * self.m_bb)
        
        # SGP stopping criteria
        self.tol_grad = tol_grad
        self.tol_x = tol_x
        self.maxiter = maxiter

        err_list = np.zeros(self.maxiter+1)
        grad_list = np.zeros(self.maxiter+1)
        fun_list = np.zeros(self.maxiter+1)

        u_ls = self.U_LS()
        v_ls = self.V_LS(x) 
        u_tv = self.U_TV(x)
        v_tv = self.V_TV(x)

        grad = v_ls - u_ls + self.lmbda * (v_tv - u_tv)

        # Compute errors first step
        grad_list[0] = np.linalg.norm(grad, 2)
        fun_list[0] = self.f(x)
        if x_true is not None:
            err_list[0] = np.linalg.norm(x - x_true, 2) / np.linalg.norm(x_true, 2)

        rho = np.sqrt(1 + 1e15)
        s = self.compute_scaling(x, v_ls, v_tv, rho)

        k = 0
        CONTINUE = True
        while CONTINUE:            
            desc_direction = self.Proj_positive(x - self.alpha * s * grad) - x
            step_length = self.backtracking(x, self.f, grad, desc_direction)

            x0 = x
            x = x + step_length * desc_direction

            k += 1
            if x_true is not None:
                err_list[k] = np.linalg.norm(x - x_true, 2) / np.linalg.norm(x_true, 2)
            grad_list[k] = np.linalg.norm(grad, 2)
            fun_list[k] = self.f(x)

            CONTINUE = (np.linalg.norm(grad) > self.tol_grad*grad_list[0]) and (k < self.maxiter) and (np.linalg.norm(x0 - x) / np.linalg.norm(x) > self.tol_x)
            if CONTINUE:
                # Update v_ls, u_tv, v_tv
                v_ls = self.V_LS(x)
                u_tv = self.U_TV(x)
                v_tv = self.V_TV(x)
                grad_0 = grad
                grad = v_ls - u_ls + self.lmbda * (v_tv - u_tv)

                rho = np.sqrt(1 + 1e15 / (k**2.1))
                s = self.compute_scaling(x, v_ls, v_tv, rho)
                self.alpha = self.bb(s, x0, x, grad_0, grad, self.alpha)

                if verbose:
                    print('************* Iteration: ', k)
                    #print('Step Length: ', step_length)
                    print('Rel Err: ', err_list[k])

        err_list = err_list[:k+1]
        grad_list = grad_list[:k+1]
        fun_list = fun_list[:k+1]

        return x, k, grad_list, err_list, fun_list

    def f(self, x):
        return 0.5 * (np.linalg.norm(self.A(x) - self.b, 2)**2 + self.lmbda * self.TV(x))

    def grad_f(self, x):
        grad = self.A.T*(self.A(x) - self.b)
        return grad + self.lmbda * x
        
    def V_LS(self, x):
        return self.A.T(self.A(x))

    def U_LS(self):
        return self.A.T(self.b)

    def diff(self, x, axis, append=0):
        return _diff(int(np.sqrt(self.n)), x, axis, append)

    def D_h(self, x):
        return _D_h(int(np.sqrt(self.n)), x)

    def D_v(self, x):
        return _D_v(int(np.sqrt(self.n)), x)

    def phi(self, t, beta=1e-3):
        return 2 * np.sqrt(t + beta ** 2)

    def dphi(self, t, beta=1e-3):
        return _dphi(t, beta)
    
    def delta(self, x):
        return _delta(int(np.sqrt(self.n)), x)

    def V_TV(self, x, beta=1e-3): # Zero padding
        return _V_TV(int(np.sqrt(self.n)), _delta, _dphi, x, beta)

    def U_TV(self, x, beta=1e-3): # Zero padding
        return _U_TV(int(np.sqrt(self.n)), _delta, _dphi, x, beta)

    def TV(self, x, beta=1e-3):
        Dh = self.D_h(x)
        Dv = self.D_v(x)

        return np.sum(np.sqrt(np.square(Dh) + np.square(Dv) + beta ** 2)) 
     
    def compute_scaling(self, x, V_LS, V_TV, rho):
            return _compute_scaling(int(np.sqrt(self.n)), self.lmbda, x, V_LS, V_TV, rho)

    def backtracking(self, x, f, grad, d):
        alpha = 1
        rho = 0.8
        c1 = 0.25

        fx = f(x)
        while f(x + alpha * d) > fx + alpha * c1 * grad.dot(d):
            alpha *= rho

        return alpha

    def bb(self, s, x0, x, grad_0, grad, alpha_old):
        alpha_min = 1e-10
        alpha_max = 1e5

        s_k = x - x0
        z_k = grad - grad_0

        Dz = z_k * s

        alpha_bb1_denom = np.dot(s_k, z_k / s)
        alpha_bb2_num = np.dot(s_k, Dz)

        if alpha_bb1_denom <= 0:
            alpha_bb1 = min(10 * alpha_old, alpha_max)
        else:
            alpha_bb1 = np.linalg.norm(s_k / s, 2)**2 / alpha_bb1_denom
            alpha_bb1 = max(min(alpha_bb1, alpha_max), alpha_min)

        if alpha_bb2_num <= 0:
            alpha_bb2 = min(10 * alpha_old, alpha_max)
        else:
            alpha_bb2 = alpha_bb2_num / (np.linalg.norm(Dz, 2) ** 2)
            alpha_bb2 = max(min(alpha_bb2, alpha_max), alpha_min)

        self.alpha_bb2_vec = np.append(self.alpha_bb2_vec, alpha_bb2)

        if alpha_bb2 / alpha_bb1 < self.tau_bb:
            alpha = np.min(self.alpha_bb2_vec[-self.m_bb-1:])
            self.tau_bb = self.tau_bb * 0.9
        else:
            alpha = alpha_bb1
            self.tau_bb = self.tau_bb * 1.1

        return alpha

    def Proj_positive(self, x):
        return _Proj_positive(x)

    def gradient_descent(self, x0, x_true):
        x = x0
        grad = self.grad_f(x)
        err_list = np.zeros(self.maxiter)
        grad_list = np.zeros(self.maxiter)

        k = 0
        while (np.linalg.norm(grad) > self.tol_grad) and (k < self.maxiter):
            alpha = self.backtracking(x, self.f, grad)

            print('Iter: ', k, '\t Alpha: ', alpha, '\n')
            x = x - alpha * grad
            grad = self.grad_f(x)

            if x_true is not None:
                err_list[k] = np.linalg.norm(x - x_true, 2) / np.linalg.norm(x_true, 2)
            grad_list[k] = np.linalg.norm(grad, 2)
            k += 1

        err_list = err_list[0:k+1]
        grad_list = grad_list[0:k+1]

        return(x, k, grad_list, err_list)


"""
Jit wrappers
"""
@nb.njit()
def _diff(n, x, axis, append=0):
        xout = np.empty((n+1, n+1))
        xout[:-1, :-1] = x

        for i in range(n+1):
            xout[-1, i] = append
        for j in range(n+1):
            xout[j, -1] = append

        if axis == 0:
            xpart = xout[1:, :] - xout[:-1, :]
            return xpart[:, :-1].flatten()
        elif axis == 1:
            xpart = xout[:, 1:] - xout[:, :-1]
            return xpart[:-1, :].flatten()

@nb.njit()
def _D_h(n, x):
    x = np.reshape(x, (n, n))
    return _diff(n, x, axis=0)

@nb.njit()
def _D_v(n, x):
    x = np.reshape(x, (n, n))
    return _diff(n, x, axis=1)

@nb.njit()
def _V_TV(n, delta, dphi, x, beta=1e-3): # Zero padding
    x = np.reshape(x, (n, n))
    V_tv = np.empty((n, n))
    D = np.reshape(delta(n, x), (n, n))

    V_tv[0, 0] = (2 * dphi(D[0, 0]) + dphi(x[0, 0]) + dphi(x[0, 0])) * x[0, 0]
    for j in range(1, n):
        V_tv[0, j] = (2 * dphi(D[0, j]) + dphi(x[0, j]) + dphi(D[0, j-1])) * x[0, j]
    for i in range(1, n):
        V_tv[i, 0] = (2 * dphi(D[i, 0]) + dphi(D[i-1, 0]) + dphi(x[i, 0])) * x[i, 0]

    for i in range(1, n):
        for j in range(1, n):
            V_tv[i, j] = (2 * dphi(D[i, j]) + dphi(D[i-1, j]) + dphi(D[i, j-1])) * x[i, j]

    return V_tv.flatten()

@nb.njit(parallel=True)
def _U_TV(n, delta, dphi, x, beta=1e-3): # Zero padding
    x = np.reshape(x, (n, n))
    U_tv = np.empty((n, n), dtype=np.float32)
    D = np.reshape(delta(n, x), (n, n))

    U_tv[0, 0] = dphi(D[0, 0]) * (x[1, 0] + x[0, 1])
    for j in range(1, n-1):
        U_tv[0, j] = dphi(D[0, j]) * (x[1, j] + x[0, j+1]) + dphi(D[0, j-1]) * x[0, j-1]
    U_tv[0, n-1] = dphi(D[0, n-1]) * x[1, n-1] + dphi(D[0, n-2]) * x[0, n-2] 
    for i in range(1, n-1):
        U_tv[i, 0] = dphi(D[i, 0]) * (x[i+1, 0] + x[i, 1]) + dphi(D[i-1, 0]) * x[i-1, 0]
    U_tv[n-1, 0] = dphi(D[n-1, 0]) * x[n-1, 1] + dphi(D[n-2, 0]) * x[n-2, 0]

    for i in range(1, n-1):
        for j in range(1, n-1):
            U_tv[i, j] = dphi(D[i, j]) * (x[i+1, j] + x[i, j+1]) + dphi(D[i-1, j]) * x[i-1, j] + dphi(D[i, j-1]) * x[i, j-1]

    U_tv[1:, n-1] = 0
    U_tv[n-1, 1:] = 0
    
    return U_tv.flatten()

@nb.njit()
def _compute_scaling(n, lmbda, x, V_LS, V_TV, rho):
    V = V_LS + lmbda * V_TV

    d = np.zeros(n**2)
    for i in range(n**2):
        if V[i] < 1e-4:
            V[i] = 1e-4
        if rho == 0:
            d[i] = x[i]/V[i]
        else:
            d[i] = min(rho, max(1/rho, x[i]/V[i]))
    return d

@nb.njit()
def _Proj_positive(x):
    for i in range(len(x)):
        x[i] = max(0, x[i])
    return x

@nb.njit(fastmath=True)
def _dphi(t, beta=1e-3):
    return 1 / np.sqrt(t + beta ** 2)
    
@nb.njit(fastmath=True)
def _delta(n, x):
    return np.square(_D_h(n, x)) + np.square(_D_v(n, x))