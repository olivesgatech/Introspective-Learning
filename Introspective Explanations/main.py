# This is an implementation of Explanatory Paradigms in Neural Networks.
# The code is built on top of Grad-CAM and Contrast-CAM explanations.
#
# The base implementation of Grad-CAM is derived from :
# https://github.com/1Konny/gradcam_plus_plus-pytorch

# The base implementation of Contrast-CAM is derived from :
# https://github.com/olivesgatech/Contrastive-Explanations

import os
import PIL
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.models as models
from torchvision.utils import make_grid, save_image
from matplotlib import pyplot as plt
from PIL import Image
import time
import glob
import math

from utils import visualize_cam, Normalize
from methods import GradCAM, ContrastCAM, CounterfactualCAM


img_dir = 'Images'
#img_name = 'water-bird.JPEG'
#img_name = 'Flamingo6.jpg'
img_name = 'cat_dog.png'

img_path = os.path.join(img_dir, img_name)

pil_img = PIL.Image.open(img_path)

#plt.imshow(pil_img)
#plt.show()

normalizer = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

torch_img = torch.from_numpy(np.asarray(pil_img)).permute(2, 0, 1).unsqueeze(0).float().div(255).cuda()
torch_img = F.upsample(torch_img, size=(224, 224), mode='bilinear', align_corners=False)

normed_torch_img = normalizer(torch_img)


output_dir = 'Results'

img_folder,_ = img_name.split('.')
os.makedirs(output_dir + '/' + img_folder + '/', exist_ok=True)

cam_dict = dict()

# Please select any of the networks. Your own networks and data can also be used. In that case, the layer specifications
# should be provided below (layer_name), and extraction should be done in utils.py
# As an example we demo on VGG-16, layer 29

model = models.vgg16(pretrained=True)
model.eval(), model.cuda()
model_dict = dict(type='vgg', arch=model, layer_name='features_29', input_size=(224, 224))


# Example configurations of other architectures are shown below

#model_dict = dict(type='resnet', arch=model, layer_name='layer4', input_size=(224, 224))
#model_dict = dict(type='alexnet', arch=model, layer_name='features_11', input_size=(224, 224))
#model_dict = dict(type='densenet', arch=model, layer_name='features_norm5', input_size=(224, 224))
#model_dict = dict(type='squeezenet', arch=squeezenet, layer_name='features_12_expand3x3_activation', input_size=(224, 224))

gradcam = GradCAM(model_dict)
contrast = ContrastCAM(model_dict)
counterfactualCAM = CounterfactualCAM(model_dict)

mask_gradcam, logit = gradcam(normed_torch_img)
heatmap_gradcam, result_gradcam = visualize_cam(mask_gradcam.cpu(), torch_img)

output_path = os.path.join(output_dir + '/' + img_folder + '/' + 'GradCAM.png')
save_image(result_gradcam, output_path)


mask_contrast, _ = contrast(normed_torch_img, 242)  # Your choice of contrast; The Q in `Why P, rather than Q?'. Class 130 is flamingo. Class 242 is Boxer
heatmap_contrast, result_contrast = visualize_cam(mask_contrast.cpu(), torch_img)

output_path = os.path.join(output_dir + '/' + img_folder + '/' + 'ContrastCAM.png')
save_image(result_contrast, output_path)

mask_couterfactual, _ = counterfactualCAM(normed_torch_img)
heatmap_counterfactual, result_counterfactual = visualize_cam(mask_couterfactual.cpu(), torch_img)

output_path = os.path.join(output_dir + '/' + img_folder + '/' + 'Counterfactual.png')
save_image(result_counterfactual, output_path)