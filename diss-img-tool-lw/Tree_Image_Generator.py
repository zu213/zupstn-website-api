#!/usr/bin/env python
# coding: utf-8

### THIS FILE IS VERY SIMILAR TO GUI, THEREFORE MANY COMMENTS HAVE BEEN OMITTED

# Imports
import sys
from util import util
from models import create_model
import matplotlib.pyplot as plt
from PIL import Image

from Tree_Image_Generator_opts import bicycle_gan_opts
from Tree_Image_Generator_util import images_to_input, pad_images, rescale, get_darkest

# Important constants
name = 'high_ren'
model_nom = 'bicycle_gan'
num_epochs = ''
data_set_path = ''
# constant for count of iamges produced
image_count = 3

# Constants for testing purposes
auto_loop = 0 #0 for no auto, 1 for simple auto, 2 for dark auto
display_popups = True
save_frequency = 5
loop_limit = 5


def generate_images(img_model, input_sketch_jpg, input_art_jpg = None, z = None, chosen_image = None):

    noises_temp = []
    imgs = []

    # if its the first loop we have an image to pass in otherwise we pass in a z vector
    if input_art_jpg != None:

        # Pad the images
        input_sketch_padded, input_art_padded = pad_images(input_sketch_jpg, input_art_jpg)
        input_sketch_padded, input_art_padded = rescale([input_sketch_jpg, input_art_jpg])

        # put them in the correct form to be processed
        processed_input = images_to_input(input_sketch_padded, input_art_padded)
    else:
        input_sketch_padded, _= pad_images(input_sketch_jpg)
        [input_sketch_padded] = rescale([input_sketch_jpg])
        # special z style vector
        processed_input = images_to_input(input_sketch_padded, z = z)
    img_model.set_input(processed_input)
    
    for i in range(image_count):
        # Run them through the network
        img_model.forward()
        # this gets us the style vectors our randomness created for network updates
        noises_temp.append(img_model.get_current_noise())

        visuals = img_model.get_current_visuals() 

        # Loop to work through generated fluff
        count = 0
        for _, value in visuals.items():
            count += 1
            if count < 4:
                # If its the generated image convert it from a tensor to show
                img = Image.fromarray(util.tensor2im(value))
                if count == 3:
                    imgs.append(img)

    return imgs, noises_temp
    

