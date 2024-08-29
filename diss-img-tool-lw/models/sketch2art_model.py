import torch
from torch import nn
from torch.nn.utils import spectral_norm

from PIL import Image
import torch
import numpy as np
def tensor2im(input_image, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        # altered this line
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)

class IDN(nn.Module):
    # take a tensor as input, predict out the:
    # 1. grey-scale sketch image
    # 2. style feature vector

    # in adaIN: result_feature = [(content_feature - mu_c) / sigma_c ] * sigma_s + mu_s
    # so we can first get mu_s by a conv layer
    # then we can calculate the sigma_s from (result_feature - mu_s)
    # then we get the content feature using the (result_feature - mu_s)/sigma_s

    def __init__(self, in_channel, style_dim, content_channel=1):
        print(style_dim)
        super().__init__()
        
        self.style_mu_conv = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channel, in_channel, 4, 2, 1)), nn.LeakyReLU(0.1),
            spectral_norm(nn.Conv2d(in_channel, in_channel, 4, 2, 1)), nn.LeakyReLU(0.1),
            nn.AdaptiveAvgPool2d(1))
        self.to_style = nn.Linear(in_channel*2, style_dim)

        self.to_content = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channel, in_channel, 3, 1, 1)), nn.LeakyReLU(0.1),
            spectral_norm(nn.Conv2d(in_channel, content_channel, 3, 1, 1)), nn.LeakyReLU(0.1),
            nn.AdaptiveAvgPool2d(16*16))

    def forward(self, feat):
        b, c, _, _ = feat.size()

        style_mu = self.style_mu_conv(feat)

        feat_no_style_mu = feat - style_mu
        style_sigma = feat_no_style_mu.view(b, c, -1).std(-1)
        feat_content = feat_no_style_mu / style_sigma.view(b,c,1,1)
        style = self.to_style( torch.cat([style_mu.view(b,-1), style_sigma.view(b,-1)], dim=1) )
        content = self.to_content(feat_content)
        
        return content, style


class FeatMapTransfer(nn.Module):
    '''
    step 1: Get the Edge feature and Plain-area feature
    use the style-image's sketch as a mask M, so we can filter the values in the style-image's feature maps F
    with M*F, we can get a feature-map M_E, that only have the feature values around the edges, 
    and with (1-M)*F, we can get the M_P with features only around the plain areas of the image.
    then we use a max-pooling layer to extract the filtered feature values, because the filtered feature-map will be sparse
    finally, we will have two 4x4 feature-maps, one for the edge and one for the rest part.
    step 2:
    Then we impose the features into the content feature map. To begin with, we fill a blank feature-map with same size 
    of the content feature-map with the Edge feature-map, then we use the content-sketch as mask to filter this feature map.
    Do the same thing with Plain feature-map. Then we can add the 2 feature-map into one.
    '''

    def __init__(self, hw=64, divisor=8):
        super().__init__()
        self.ad_pool = nn.AdaptiveAvgPool2d(hw)
        self.feat_pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=5, stride=2),
            nn.MaxPool2d(kernel_size=5, stride=2),
            nn.AdaptiveAvgPool2d(hw//divisor),
        )
        self.divisor = divisor
        
    def forward(self, style_feat, content_skt, style_skt=None, return_feat = False):
        if style_skt == None:
            style_skt = FeatMapPreProcess(style_feat)[1]
        style_feat = rescale(style_feat, (-1,1))
        style_skt = rescale(style_skt, (-1,1))
        content_skt = rescale(content_skt, (-1,1))
        style_feat = self.ad_pool(style_feat)

        style_skt = (self.ad_pool(style_skt))
        content_skt = (self.ad_pool(content_skt))
        style_feat = torch.flatten(style_feat, 0, 1).unsqueeze(0)
        content_skt = torch.flatten(content_skt, 0, 1).unsqueeze(0)

        edge_feat = style_feat * style_skt
        plain_feat = style_feat * (-style_skt)

        edge_feat_unrepeated = self.feat_pool(edge_feat)
        plain_feat_unrepeated = self.feat_pool(plain_feat)
        edge_feat = edge_feat_unrepeated.repeat(1,1,self.divisor,self.divisor)
        plain_feat = plain_feat_unrepeated.repeat(1,1,self.divisor,self.divisor)
        if not return_feat:
            return rescale(torch.add(edge_feat*content_skt , plain_feat*(-content_skt)), (-1,1))
        processed_edge_feat = torch.mean(edge_feat_unrepeated, dim=1)
        print(processed_edge_feat.size())
        return (edge_feat*content_skt + plain_feat*(1-content_skt), edge_feat_unrepeated + plain_feat_unrepeated)




class DualMaskInjection(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        
        self.weight_a = nn.Parameter(torch.ones(1, in_channels, 1, 1)*1.1)
        self.weight_b = nn.Parameter(torch.ones(1, in_channels, 1, 1)*0.9)

        self.bias_a = nn.Parameter(torch.zeros(1, in_channels, 1, 1)+0.01)
        self.bias_b = nn.Parameter(torch.zeros(1, in_channels, 1, 1)+0.01)

    def forward(self, feat, mask):
        feat_a = (self.weight_a * feat * (mask) + self.bias_a)
        feat_b = (self.weight_b * feat * (1-mask) + self.bias_b)

        return feat_a + feat_b
    
def rescale(tensor, range=(0, 1)):
        return ((tensor - tensor.min()) / (tensor.max() - tensor.min()))*(range[1]-range[0]) + range[0]

def FeatMapPreProcess(input_image):
    #return input_image
    # convert to greyscale and then convert to black and white
    feat_map = input_image
    feat_map_grey = greyscale(feat_map)
    average = torch.mean(feat_map_grey)
    style_sketch = (((feat_map_grey<average).float()) - 0.5) * 2

    return (feat_map, style_sketch)

def greyscale(input_image):
    feat_map = (torch.mean(input_image, 1) / 2)
    feat_map = torch.unsqueeze(feat_map.repeat(3, 1 , 1), 0)
    feat_map = rescale(feat_map)
    return feat_map