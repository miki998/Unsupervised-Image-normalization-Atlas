import numpy as np
import matplotlib.image as mpimg 
import matplotlib.pyplot as plt
import torch
from decimal import Decimal

tensor_type = torch.DoubleTensor


def atlas_learn_template(dico_images,niter,kernel_width,gamma,eps,template=None,landmarks=None):
    
    '''
    This is the principal function, which computes gradient descent to minimize cost function,
    find optimal trajectories for control points, alpha and deformed template
    Take a dictionary with images, niter number of iterations, kernel_width and gamma parameters, eps for step size
    template and landmarks are optionals.
    -------
    returns 
    cp : deformation of  control points to match with each subject : numpy array (n_landmarks,2,n_images)
    alpha : optimized momenta,  numpy array (n_landmarks,2,n_images)
    template_data : the optimized template, numpy array (nr,nc) 
    deformed_template : deformation of template according to each image,  numpy array (nr,nc,n_images)

    '''

  
    ## Convert from dico of images to 3 dimensional array (nr,nc,n_images)
    
    Images = dico_images[1]
    for key, value in dico_images.items():
        if key==2:
            Images = np.stack((Images,value),axis=-1)   
        elif key !=1 :
            Images = np.concatenate((Images,value[...,np.newaxis]),axis=-1)   

    Images_base = torch.from_numpy(Images.copy()).type(tensor_type)
    
    
    ### Convert template, if given, to a tensor (nx,ny)
    ### If no template is provided, take first image of set, but it's not the optimal solution
    if template is None :
        template_data = torch.from_numpy(Images[...,0].copy()).type(tensor_type)  
    else :
        template_data = torch.from_numpy(template).type(tensor_type)
        
    #### Initialize control points and momenta for template
    #### Here, with digits database, no control points are provided, 
    #### So we build a regular grid with interval equals to kernel_width.
    #### You can plot control points to see where they are exactly.
    #### cp tensor is of  shape (n_landmarks,2) with y-axis coordinates in first column and x-axis coordinates in second column
    if landmarks is not None :
        cp = torch.from_numpy(landmarks1).type(tensor_type)
    else :
        cp = np.array(np.meshgrid(np.arange(0,template.shape[0]-2,kernel_width), np.arange(0,template.shape[1]-2,kernel_width)))
        cp = np.swapaxes(cp, 0, 2).reshape(-1, 2) 
        cp = torch.from_numpy(cp).type(tensor_type)
        

    
    ##### Plot template at the beginning and first original digits
    
    plt.figure()
    plt.subplot(1,5,1)
    plt.imshow(template_data.detach().numpy(),cmap='gray')
    plt.subplot(1,5,2)
    plt.imshow(np.clip(Images[...,0],0,1),cmap='gray')
    plt.subplot(1,5,3)
    plt.imshow(np.clip(Images[...,1],0,1),cmap='gray')
    plt.subplot(1,5,4)
    plt.imshow(np.clip(Images[...,2],0,1),cmap='gray')
    plt.subplot(1,5,5)
    plt.imshow(np.clip(Images[...,3],0,1),cmap='gray')
    plt.title('Template at the beginning and first four digits')
    plt.show()
      
    #######################    
    #### Iterations    
            
    number_of_subjects = Images_base.size(-1)
    alpha = torch.zeros((cp.size(0),cp.size(1),number_of_subjects)).type(tensor_type)
    alpha.requires_grad_(True)
    template_data.requires_grad_(True)
    
    
    for it in range(niter):
        
        current_attachment, current_regularity,deformed_template,deformed_points = compute_attachment_regularity_gradient(cp,alpha,template_data,Images_base,kernel_width,gamma)
  
        gradient = {}
        gradient['alpha'] = alpha.grad.detach()
        gradient['template_data'] = template_data.grad.detach()

        eps_mom = eps/np.sqrt(np.sum(gradient['alpha'].numpy() ** 2))
        eps_template = eps/np.sqrt(np.sum(gradient['template_data'].numpy() ** 2)+10**-5)
        
        with torch.no_grad():
            alpha -=   alpha.grad * eps_mom
            template_data -=   template_data.grad * eps_template
            
        alpha.grad.zero_()
        template_data.grad.zero_()
        
                                                                                
       
  
        attach_val = current_attachment.numpy()
        regul_val = current_regularity.numpy()

        if it % 20 == 0:
            print('------------------------------------- Iteration: ' + str(it)  + ' -------------------------------------')
            print('>> [ attachment = %.5E ; regularity = %.5E ]' %
              (Decimal(str(attach_val)),
               Decimal(str(regul_val))))

   
            ##### Plot template and deformed template according to first digits

            plt.figure()
            plt.subplot(1,5,1)
            plt.imshow(np.clip(template_data.detach().numpy(),0,1),cmap='gray')
            plt.title('Template')
            plt.subplot(1,5,2)
            plt.imshow(np.clip(deformed_template.detach().numpy()[...,0],0,1),cmap='gray')
            plt.subplot(1,5,3)
            plt.imshow(np.clip(deformed_template.detach().numpy()[...,1],0,1),cmap='gray')
            plt.subplot(1,5,4)
            plt.imshow(np.clip(deformed_template.detach().numpy()[...,2],0,1),cmap='gray')
            plt.subplot(1,5,5)
            plt.imshow(np.clip(deformed_template.detach().numpy()[...,3],0,1),cmap='gray')
            plt.show()
    

    
    return cp.detach().numpy(),alpha.detach().numpy(),np.clip(template_data.detach().numpy(),0,1),deformed_template.detach().numpy()


    