import numpy as np
import matplotlib.image as mpimg 
import matplotlib.pyplot as plt
import torch
from decimal import Decimal

tensor_type = torch.DoubleTensor

def _swap_colums(ar, i, j):
    aux = np.copy(ar[:, i])
    ar[:, i] = np.copy(ar[:, j])
    ar[:, j] = np.copy(aux)
    return np.copy(ar)



def interpolate_image(intensities, deformed_pixels, padding_width=1):
    '''
    This function, given original image in intensities tensor, 
    deformed pixels : coordinates of registered pixels in original image 
    returns the final registered image in deformed_intensities
    ------- 
    intensities : (nr,nc)
    deformed_pixels : (nr*nc,d)
    -------
    returns registered image in deformed_intensities, of shape (nr,nc)
    
    '''
    padding_color = 1.
    if intensities.ndim == 2:
        i,j = intensities.size()
        k = 1
        intensities_cp = intensities[...,np.newaxis]
    else :
        i,j,k = intensities.size()
        intensities_cp = intensities
        
    deformed_pixels += torch.from_numpy(np.array([float(padding_width), float(padding_width)])).view(1, 2).expand(i*j, 2).type(torch.DoubleTensor)

    padded_intensities = torch.ones((i + 2 * padding_width, j + 2 * padding_width, k)).type(torch.DoubleTensor) * padding_color
    padded_intensities[padding_width:padding_width + i, padding_width:padding_width + j] = intensities_cp

    u, v = deformed_pixels[:, 0], deformed_pixels[:, 1]

    u1 = torch.floor(u)
    v1 = torch.floor(v)

    u1 = torch.clamp(u1, 0, i - 1 + 2 * padding_width)
    v1 = torch.clamp(v1, 0, j - 1 + 2 * padding_width)
    u2 = torch.clamp(u1 + 1, 0, i - 1 + 2 * padding_width)
    v2 = torch.clamp(v1 + 1, 0, j - 1 + 2 * padding_width)

    fu = (u - u1).view(i * j, 1).expand(i*j, k)
    fv = (v - v1).view(i * j, 1).expand(i*j, k)
    gu = ((u1 + 1) - u).view(i * j, 1).expand(i*j, k)
    gv = ((v1 + 1) - v).view(i * j, 1).expand(i*j, k)

    deformed_intensities = (padded_intensities[u1.type(torch.LongTensor), v1.type(torch.LongTensor)] * gu * gv +
                            padded_intensities[u1.type(torch.LongTensor), v2.type(torch.LongTensor)] * gu * fv +
                            padded_intensities[u2.type(torch.LongTensor), v1.type(torch.LongTensor)] * fu * gv +
                            padded_intensities[u2.type(torch.LongTensor), v2.type(torch.LongTensor)] * fu * fv).view(i, j, k)
    deformed_intensities = torch.clamp(deformed_intensities, 0., 1.)

    return deformed_intensities.reshape(intensities.size())


