import numpy as np
import matplotlib.image as mpimg 
import matplotlib.pyplot as plt
import torch
from decimal import Decimal

tensor_type = torch.DoubleTensor


def shoot(cp, alpha, kernel_width, n_steps=10):
     
    """
    This is the trajectory of a Hamiltonian dynamic, with system seen in lecture notes. 
    Compute here trajectories of control points and alpha from t=0 to t=1.
    ------------
    cp is of shape (n_landmarks, 2)
    alpha is of shape (n_landmarks, 2)
    n_step : number of steps in your hamiltonian trajectory, use to define your time step
    kernel_width is a value
    --------
    returns traj_cp and traj_alpha trajectories of control points and alpha in lists. 
    The length of a list is equal to n_step. 
    In each element of the list, you have a tensor of size (n_landmarks,2) returned by discretisation_step() function.
    """
    
    
    traj_cp, traj_alpha = [], []
    traj_cp.append(cp)
    traj_alpha.append(alpha)
    dt = 1. / float(n_steps-1)
    
    for _ in range(n_steps-1):
        new_cp, new_alpha = discretisation_step(traj_cp[-1], traj_alpha[-1], dt, kernel_width)
        traj_cp.append(new_cp)
        traj_alpha.append(new_alpha)
        
    return traj_cp, traj_alpha

def register_points(traj_cp, traj_alpha, y, kernel_width):
    """
    This is the application of the computed trajectories on a set of points (landmarks or new points).
    ------------
    
    traj_cp is the list containing the trajectory of your landmarks 
    traj_alpha is is the list containing the trajectory of your alpha 
    y : points you want to register (landmarks or other points), size (n,2)
    kernel_width is a value
    
    --------
    
    returns traj_y,  the trajectory of points y, in a list of length n_step. 
    In each element of the list, you should have an array of dimension (n,2) (same dimensions as y)
    
    """
    # We now flow the points
    traj_y = [y]
    n_steps = len(traj_cp) - 1
    dt = 1. / float(n_steps)
    
    for i in range(len(traj_cp)-1):
        new_y = traj_y[-1] + dt * torch.matmul(gaussian_kernel(traj_y[-1], traj_cp[i], kernel_width), traj_alpha[i])
        traj_y.append(new_y)
    
    return traj_y



def register_image(traj_cp, traj_alpha, image, kernel_width):
    """
    This is the application of the computed trajectories on an image, by computation of inversed phi_1.
    ------------
    traj_cp is the list containing the trajectory of your landmarks 
    traj_alpha is the list containing the trajectory of your alpha 
    image : image to register, of size (nr,nc)
    kernel_width is a value
    --------
    returns the registered image, of same dimensions as image, (nr,nc)
    
    """
    
    if image.ndim==2:
        i,j = image.shape
        k = 1
    else :
        i,j,k = image.shape
        
    points = np.array(np.meshgrid(range(i), range(j)))
    points = np.swapaxes(points, 0, 2).reshape(i * j, 2) 
    points = torch.from_numpy(points).type(tensor_type)
    traj_cp_inverse = traj_cp[::-1]
    traj_alpha_inverse = [-1 * elt for elt in traj_alpha[::-1]]
    deformed_points = register_points(traj_cp_inverse, traj_alpha_inverse, points, kernel_width)[-1]
    
    return interpolate_image(image, deformed_points)

def compute_attachment_regularity_gradient(cp,momenta,template_data,subjects,kernel_width,gamma):
    '''
    TO DO
    ATTENTION : you only use torch tensors here, no numpy objects
    ----------
    This function compute attachments and regularities in order to compute the loss and optimize momenta, template with gradient descent.
    In order to do this, you have to deform control points and template as with LDDMM, for each image/subject.
    Then compute attachment and regularity for each deformed template and points, and finally the total attachment and total regularity.
    cp : tensor (n_landmarks,2)
    momenta : tensor (n_landmarks,2,n_images)
    template_data : tensor (nr,nc)
    subjects : tensor (nr,nc,n_images)
    kernel_width and gamma : parameters - values
    ----- 
    returns 4 objects :
    attachement is a tensor with only one value inside, ex. tensor(10.0)
    regularity is a tensor with only one value inside
    deformed template according to each subject : numpy array (nr,nc,n_images) 
    deformed control points according to each subject : numpy array (n_landmarks,2,n_images)
    '''
    
    attachment, regularity = 0.,0.
    deformed_template_list, deformed_points_list = [], []
    #we loop over all the alphas of each image/subject to deform our template
    #### Compute an estimation of control points and alpha trajectories
    for idx in range(momenta.shape[2]):
        alpha = momenta[:,:,idx]
        traj_cp, traj_alpha = shoot(cp, alpha, kernel_width, n_steps=10)
        
        def_pts = register_points(traj_cp, traj_alpha, cp, kernel_width)[-1]
        img = register_image(traj_cp, traj_alpha, template_data,kernel_width)
        
        deformed_points_list.append(def_pts)
        deformed_template_list.append(img)
        
        attachment += torch.sum((img-subjects[:,:,idx])**2)
        regularity += gamma * torch.sum(torch.mm(alpha.T,torch.mm(gaussian_kernel(cp,cp,kernel_width), alpha)))
        
    #transform the lists into the correct shape tensors
    deformed_points = deformed_points_list[0]
    deformed_template = deformed_template_list[0]
    
    deformed_points = torch.reshape(deformed_points,(deformed_points.shape[0],deformed_points.shape[1],1))
    deformed_template = torch.reshape(deformed_template,(deformed_template.shape[0],deformed_template.shape[1],1))
    
    for idx in range(1,momenta.shape[2]):
        tmp1 = torch.reshape(deformed_points_list[idx],(deformed_points.shape[0],deformed_points.shape[1],1))
        deformed_points = torch.cat((deformed_points,tmp1),axis=-1)
        
        tmp2 = torch.reshape(deformed_template_list[idx],(deformed_template.shape[0],deformed_template.shape[1],1))
        deformed_template = torch.cat((deformed_template,tmp2),axis=-1)
    
    ### here is computed the total loss and  gradient with torch.backward()
    total = attachment + regularity
    total.backward()

    return attachment.detach(),regularity.detach(),deformed_template.detach(),deformed_points.detach()