import numpy as np
import matplotlib.image as mpimg 
import matplotlib.pyplot as plt
import torch
from decimal import Decimal

tensor_type = torch.DoubleTensor


def _differences(x, y):
    
    """ 
    x is of shape (n, 2)
    y is of shape (m, 2)
    --------
    returns the difference between each element of x and y in a (2,n,m) tensor
    
    """
    x_col = x.t().unsqueeze(2)  # (M,D) -> (D,M,1)
    y_lin = y.t().unsqueeze(1)  # (N,D) -> (D,1,N)
    return x_col - y_lin

def _squared_distances(x, y):
    
    """ 
    x is of shape (n, 2)
    y is of shape (m, 2)
    
    --------
    returns the squared euclidean distance between each element of x and y in a (n,m) tensor
    
    """
    
    x_norm = (x ** 2).sum(1).view(-1, 1)
    y_norm = (y ** 2).sum(1).view(1, -1)

    dist = x_norm + y_norm - 2.0 * torch.mm(x, torch.transpose(y, 0, 1))
    return dist


def gaussian_kernel(x, y, kernel_width):
    """ 
    x is of shape (n, 2)
    y is of shape (m, 2)
    kernel_width is a value
    
    --------
    returns the gaussian kernel value between each element of x and y in a (n,m) tensor
    
    """

    squared_dist = _squared_distances(x, y)
    return torch.exp(- squared_dist / kernel_width **2 )

def h_gradx(cp, alpha, kernel_width):
    
    '''
    This function computes derivative of the kernel for each couple (cp_i,alpha_i), with cp_i a control point(landmark).
    ---------
    
    cp is of shape (n_landmarks, 2)
    alpha is of shape (n_landmarks, 2)
    kernel_width is a value
    
    --------
    returns a tensor of shape (n_landmarks, 2)
    '''
    
    sq = _squared_distances(cp, cp)
    A = torch.exp(-sq / kernel_width **2)
    B = _differences(cp, cp) * A
    return (- 2 * torch.sum(alpha * (torch.matmul(B, alpha)), 2) / (kernel_width ** 2)).t()
    
    
def discretisation_step(cp, alpha, dt, kernel_width):
    
    '''
   
    This function computes a step of discretized equations for both alpha and control points on one step. 
    Compute here a displacement step  of control points an alpha, from discretized system seen in class.
    ---------
    
    cp is of shape (n_landmarks, 2)
    alpha is of shape (n_landmarks, 2)
    dt is your time step 
    kernel_width is a value
    
    --------
    
    returns resulting control point and alpha displacements in tensors of size (n_landmarks,2) both.
    
    '''

    res_cp = cp + dt * torch.matmul(gaussian_kernel(cp, cp, kernel_width), alpha)
    res_alpha = alpha - dt / 2  * h_gradx(cp, alpha, kernel_width)
    return res_cp, res_alpha

def mean_image(dico):
    image_arrays = []
    for key in dico.keys():
        image_arrays.append(dico[key])
    image_arrays = np.array(image_arrays)
    return np.mean(image_arrays,axis=0)

def save_momentum(array):
    with open('momentum.npy', 'wb') as f:
        np.save(f, array)

def save_control(array):
    with open('control.npy', 'wb') as f:
        np.save(f, array)


def load_weights(paths):
    with open('momentum.npy', 'rb') as f:
        a = np.load(f)

    with open('control.npy', 'rb') as g:
        b = np.load(g)

    return a,b
