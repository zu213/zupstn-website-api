from util import util
import io
from PIL import Image, ImageOps
import torch
from torchvision import transforms
import numpy as np
import random

# image mix up stuff
def random_noise(image_input):
    #add random noise to an iamge returning the image and noise arrray as a tuple
    # https://stackoverflow.com/questions/22937589/how-to-add-noise-gaussian-salt-and-pepper-etc-to-image-in-python-with-opencv
    row,col = image_input.size
    # three channels for rgb
    ch = 3
    mean = 0
    var = 0.1
    sigma = var**0.5
    gauss = np.random.normal(mean,sigma,(row,col,ch))
    gauss = gauss.reshape(col,row,ch)
    gauss = Image.fromarray(np.uint8(gauss*255))
    image_input = Image.blend(image_input, gauss, 0.6)
    return (image_input, gauss)

def slice_up(image_input, slices = 4):
    #rearrange the iamge randomly
    # Change to tensor and make it divide evenly
    # may lose 1 to slices-1 pixels on the edge here
    subtensor_list = []
    # This currently scales the colours of iamge needs fix
    image_to_tensor = transforms.ToTensor()
    img = image_to_tensor(image_input)
    _, row,col = img.size()
    len_mod = row % slices
    width_mod = col % slices
    img = img[:, len_mod:, width_mod:]
    row = row - len_mod

    # split iamge up into aprts
    sliced_input = torch.tensor_split(img, slices, dim = 1)
    for i in sliced_input:
        sliced_slices = torch.tensor_split(i, slices, dim = 2)
        for j in sliced_slices:
            subtensor_list.append(j)

    # put parts back into iamge randomly
    index_list = []
    img = torch.Tensor([])
    sub_image = torch.Tensor([])
    count = len(subtensor_list)

    # iterate through lsit picking random part
    while count > 0:
        choice = random.randint(0, count - 1)
        chosen = subtensor_list[choice]
        index_list.append(choice)#index)
        del subtensor_list[choice]
        # attach part to horizontal sub image
        if sub_image.size()[0] < 3:
            sub_image = chosen
        else:
            sub_image = torch.hstack((sub_image, chosen))

        # When sub iamge is long enough attach it to final image vetically
        if sub_image.size()[1] >= row:
            if img.size()[0] < 3:
                img = sub_image
            else:
                img = torch.dstack((img, sub_image))
            sub_image = torch.Tensor([])
        count = len(subtensor_list)

    img = img.unsqueeze(0)
    out_img = Image.fromarray(util.tensor2im(img))
    return (out_img, index_list)

# Image display stuff
#https://www.tutorialspoint.com/pysimplegui/pysimplegui_working_with_pil.htm
def convert_to_bytes(img, resize=None):
   [img] = rescale([img])
   with io.BytesIO() as bio:
      img.save(bio, format="PNG")
      del img
      return bio.getvalue()


#function to pad iamges
def pad_images(image_sketch_jpg, image_art_jpg = None):
    if image_art_jpg != None:
        image_art_jpg = image_art_jpg.resize((256,256))
    image_sketch_jpg.thumbnail((256,256))
    image_sketch_jpg = ImageOps.pad(image_sketch_jpg, (256,256), color='White')
    return(image_sketch_jpg, image_art_jpg)

def rescale(input_imgs, load_width=256, load_height=256):
    osize = [load_height, load_width]
    resize_tr = transforms.Resize(osize, transforms.InterpolationMode.BICUBIC)
    return_imgs = []
    for i in input_imgs:
        return_imgs.append(resize_tr(i))
    return return_imgs

# function to change input to correct format
def images_to_input(image_sketch_jpg, image_art_jpg = None, image_path_sketch=None, image_path_art=None, z = None):
    #https://www.projectpro.io/recipes/convert-image-tensor-pytorch
    image_to_tensor = transforms.ToTensor()
    image_sketch = image_to_tensor(image_sketch_jpg)
    f_input = {'A':image_sketch.unsqueeze(0),  'A_paths':[image_path_sketch]}
    if image_art_jpg != None:
        image_art = image_to_tensor(image_art_jpg)
        f_input['B'] =image_art.unsqueeze(0)
        f_input['B_paths'] =[image_path_art]
    if z != None:
        f_input['z'] = z
    return f_input

def get_darkest(options):
    image_to_tensor = transforms.ToTensor()
    darkest = -1
    darkness_darkest = 1000000000
    for i in range(0,len(options)):
        temp_torch = image_to_tensor(options[i])
        #temp_torch = torch.exp(temp_torch)
        temp_mean = torch.mean(temp_torch)
        if temp_mean < darkness_darkest:
            darkest = i
            darkness_darkest = temp_mean
    return darkest + 1