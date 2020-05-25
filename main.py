#Imported libs
import numpy as np
import matplotlib.image as mpimg 
import matplotlib.pyplot as plt
import torch
from decimal import Decimal
import os

#helper scripts
from atlas_helpers import *
from helper_plot import *
from elementary import *
from atlas_templating import *
import cv2


#From population
images = os.listdir("population")
dico_images = {}
assert len(images) != 0;

for i in range(1,len(images)+1):
    dico_images[i] = np.array(mpimg.imread("./population/{}".format(images[i-1]))) 

#mean image to compute
digit_template = mean_image(dico_images)

#Param
eps = 1
kernel_width = 2
niter = 100
gamma = 10

## Execution of the algorithm
control_points, momenta,optim_template,deformed_digits = atlas_learn_template(dico_images,
                                                                          niter,kernel_width,
                                                                          gamma,eps,
                                                                          template=digit_template,
                                                                          landmarks=None)
plt.imshow(optim_template)
plt.show()
cv2.imwrite("normalized.png",optim_template)

#DEMO: uncomment for usage
# dico_images = {}
# for i in range(1,21):
#     dico_images[i] = np.array(mpimg.imread("./Images/digits/digit_2_sample_{}.png".format(i))) 
# digit_template =  np.array(mpimg.imread("./Images/digits/digit_2_mean.png")) 

# #Param
# eps = 1
# kernel_width = 2
# niter = 100
# gamma = 10

# ## Execution of the algorithm
# control_points, momenta,optim_template,deformed_digits = atlas_learn_template(dico_images,
# 																			niter,kernel_width,
# 																			gamma,eps,
# 																			template=digit_template,
# 																			landmarks=None,verbose=True)
# plt.imshow(optim_template)
# plt.show()
# cv2.imwrite("normalized.png",optim_template)


#Reconstruct: uncomment to have inverse map of population
#############################################################################


# Reconstruction using learned momentum and active control points and template
# n_subjects = deformed_digits.shape[-1]
# plt.figure(figsize=(5,30))
# number_image = 1

# for i in dico_images.keys():
#     plt.subplot(n_subjects,2,number_image)
#     plt.imshow(dico_images[i],cmap='gray')
#     if i==1:
#         plt.title('Original digit')

#     plt.subplot(n_subjects,2,number_image+1)
#     plt.imshow(deformed_digits[...,i-1],cmap='gray')
#     if i==1:
#         plt.title('Reconstructed digit from atlas')

#     number_image += 2
# plt.show()