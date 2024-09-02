#!/usr/bin/env python
# coding: utf-8

### THIS FILE IS VERY SIMILAR TO GUI, THEREFORE MANY COMMENTS HAVE BEEN OMITTED

# Imports
import shutil
import sys
import os
from models import create_model
from PIL import Image
from Tree_Image_Generator_opts import bicycle_gan_opts
from Tree_Image_Generator import generate_images
import requests


# Important constants
name = 'high_ren'
model_nom = 'bicycle_gan'
num_epochs = ''
data_set_path = ''
# constant for count of iamges produced
image_count = 3


image_path_sketch = 'diss-img-tool-lw/imgs/sketch.png'
image_path_art = 'diss-img-tool-lw/imgs/style.png'

def cleanup_models():
    folder = './diss-img-tool-lw/pretrained_models/high_ren'
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        print(filename)
        if (os.path.isfile(file_path) or os.path.islink(file_path)) and filename.endswith('.pth'):
            os.unlink(file_path)


# Main loop where we take options
def main(image_path_sketch, image_path_art):
    # Setup the model ready for human feedback loop   
        # open images up
        try:
            image_sketch_jpg = Image.open(image_path_sketch)
            image_art_jpg = Image.open(image_path_art)

            # Run once with input sketch and art
            options, noises = generate_images(model, image_sketch_jpg, image_art_jpg)
            #print(options, noises)
            options[0].save('./saved/option1.png')
            options[1].save('./saved/option2.png')
            options[2].save('./saved/option3.png')

            #files = [('uploadedImages', open('./saved/option1.png', 'rb')), ('uploadedImages', open('./saved/option2.png', 'rb')), ('uploadedImages', open('./saved/option3.png', 'rb'))]

            #x = requests.post("http://zupstn.com:4000", files=files, timeout=20)
            #print(x)
        except Exception as e:
             print(e)
    

# normal main run
try:
    cleanup_models()
except Exception as e:
        print(e)

opt = bicycle_gan_opts(name)
model = create_model(opt)
model.setup(opt) 
main(image_path_sketch, image_path_art)
# must cleanup even if program fails
try:
    cleanup_models()
except Exception as e:
        print(e)
sys.stdout.flush()
