import torch
from .base_model import BaseModel
from . import networks
import numpy as np
from PIL import Image
from torchvision import transforms
from .sketch2art_model import FeatMapTransfer


class BiCycleGANModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        return parser

    def __init__(self, opt):
        if opt.isTrain:
            assert opt.batch_size % 2 == 0  # load two images at one time.

        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = ['G_GAN', 'D', 'G_GAN2', 'D2', 'G_L1', 'z_L1', 'kl']
        # some modifications i've added
        self.current_noise = ()
        self.using_z = 0
        self.image_count = 0
        self.loss_E_RLHF = 0
        self.loss_G_RLHF = 0
        self.loss_GE_RLHF = 0
        self.loss_G_L1 = 0
        self.count = 0
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        self.visual_names = ['real_A_encoded', 'real_B_encoded', 'fake_B_random', 'fake_B_encoded']
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        use_D = opt.isTrain and opt.lambda_GAN > 0.0
        use_D2 = opt.isTrain and opt.lambda_GAN2 > 0.0 and not opt.use_same_D
        use_E = opt.isTrain or not opt.no_encode
        use_vae = True
        self.fake_B_list = []
        self.model_names = ['G']
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.nz, opt.ngf, netG=opt.netG,
                                      norm=opt.norm, nl=opt.nl, use_dropout=opt.use_dropout, init_type=opt.init_type, init_gain=opt.init_gain,
                                      gpu_ids=self.gpu_ids, where_add=opt.where_add, upsample=opt.upsample)
        D_output_nc = opt.input_nc + opt.output_nc if opt.conditional_D else opt.output_nc
        if use_D:
            self.model_names += ['D']
            self.netD = networks.define_D(D_output_nc, opt.ndf, netD=opt.netD, norm=opt.norm, nl=opt.nl,
                                          init_type=opt.init_type, init_gain=opt.init_gain, num_Ds=opt.num_Ds, gpu_ids=self.gpu_ids)
        if use_D2:
            self.model_names += ['D2']
            self.netD2 = networks.define_D(D_output_nc, opt.ndf, netD=opt.netD2, norm=opt.norm, nl=opt.nl,
                                           init_type=opt.init_type, init_gain=opt.init_gain, num_Ds=opt.num_Ds, gpu_ids=self.gpu_ids)
        else:
            self.netD2 = None
        if use_E:
            self.model_names += ['E']
            if self.opt.mod_fmt:
                self.model_names.append('FMT')
                self.netFMT = networks.init_net(FeatMapTransfer(hw=opt.load_size, divisor=4),  opt.init_type, opt.init_gain, self.gpu_ids)
            self.netE = networks.define_E(opt.output_nc, opt.nz, opt.nef, netE=opt.netE, norm=opt.norm, nl=opt.nl,
                                          init_type=opt.init_type, init_gain=opt.init_gain, gpu_ids=self.gpu_ids, vaeLike=use_vae)

        if opt.isTrain:
            self.criterionGAN = networks.GANLoss(gan_mode=opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionZ = torch.nn.L1Loss()
            # initialize optimizers
            self.optimizers = []
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            if use_E:
                self.optimizer_E = torch.optim.Adam(self.netE.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizers.append(self.optimizer_E)

            if use_D:
                self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizers.append(self.optimizer_D)
            if use_D2:
                self.optimizer_D2 = torch.optim.Adam(self.netD2.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizers.append(self.optimizer_D2)

    def is_train(self):
        """check if the current batch is good for training."""
        return self.opt.isTrain and self.real_A.size(0) == self.opt.batch_size

    def set_input(self, input):
        if 'B' in input:
            AtoB = self.opt.direction == 'AtoB'
            self.real_A = input['A' if AtoB else 'B'].to(self.device)
            self.real_B = input['B' if AtoB else 'A'].to(self.device)
            self.image_paths = input['A_paths' if AtoB else 'B_paths']
            self.using_z = 0
        elif 'z' in input:
            # only works atob for now
            self.real_A = input['A'].to(self.device)
            self.real_B = None
            self.image_paths = input['A_paths']
            self.noise_data = input['z']
            self.using_z += 1
        else:
            #throw error
            raise Exception('Error: neither input image or z value entered into forward')

    def get_z_random(self, batch_size, nz, random_type='gauss'):
        if random_type == 'uni':
            z = torch.rand(batch_size, nz) * 2.0 - 1.0
        elif random_type == 'gauss':
            z = torch.randn(batch_size, nz)
        return z.detach().to(self.device)

    def encode(self, input_image):
        if self.opt.mod_fmt:
            preprocessed = self.netFMT.forward(input_image, self.real_A_encoded)
        else:
            preprocessed = input_image
        mu, logvar = self.netE.forward(preprocessed)
        std = logvar.mul(0.5).exp_()
        eps = self.get_z_random(std.size(0), std.size(1))
        z = eps.mul(std).add_(mu)
        return z, mu, logvar
    
    def get_current_noise(self):
        return self.current_noise

    def test(self, z0=None, encode=False):
        with torch.no_grad():
            if encode:  # use encoded z
                if self.opt.mod_fmt:
                    preprocessed = self.netFMT.forward(self.real_B, self.real_A_encoded)
                else:
                    preprocessed = self.real_B
                z0, _ = self.netE(preprocessed)
            if z0 is None:
                z0 = self.get_z_random(self.real_A.size(0), self.opt.nz)
            self.fake_B = self.netG(self.real_A, z0)
            return self.real_A, self.fake_B, self.real_B
        
    def greyscale(self, input_image):
        feat_map = (torch.mean(input_image, 1, keepdim=True) / 2)
        feat_map = feat_map.repeat(1, 3, 1 , 1)
        feat_map = self.rescale(feat_map, (-1,1))
        return feat_map
    
    def rescale(self, tensor, range=(0, 1)):
        return ((tensor - tensor.min()) / (tensor.max() - tensor.min()))*(range[1]-range[0]) + range[0]

    def forward(self):
        # get real images
        half_size = self.opt.batch_size // 2
        # A1, B1 for encoded; A2, B2 for random
        # Put flip in here for if sketch edges are black instead
        self.real_A_flip = self.greyscale(-self.real_A)
        self.real_A_encoded = self.real_A_flip[0:half_size]
        self.real_A_random = self.real_A_flip[half_size:]
        # If normal training z will never go over 0 so this if will always run otherwise it runs just for first loop
        if self.using_z < 1:
            self.real_B_encoded = self.real_B[0:half_size]
            self.real_B_random = self.real_B[half_size:]
            # If doing the RLHF stage remove randomness from z at this part
            if self.opt.mod_rlhf:
                if self.opt.mod_fmt:
                    preprocessed = self.netFMT.forward(self.real_B_encoded, self.real_A_encoded )
                else:
                    preprocessed = self.real_B_encoded

                # workaround for certain images
                if preprocessed.size()[1] > 3:
                   preprocessed = preprocessed[:, :3, :, :]

                mu, logvar = self.netE.forward(preprocessed)
                std = logvar.mul(0.5).exp_()
                z = std.add_(mu)
                # Careful with z here can cause bugs
                z = mu + logvar
                z = mu
                #z = logvar
                self.z_encoded, self.mu, self.logvar = z, mu, logvar
            else:
                self.z_encoded, self.mu, self.logvar = self.encode(self.real_B_encoded)

        else:
            self.z_encoded = self.noise_data[1]
        # At this part we generate our controlled randomness for z
        if self.opt.mod_rlhf:
            self.z_random = self.get_z_random(self.real_A_encoded.size(0), self.opt.nz)
            # Detach z network in rlhf case as it is looped
            self.fake_B_encoded = self.netG(self.real_A_encoded, self.z_encoded.detach())
        else:
            self.z_random = self.get_z_random(self.real_A_encoded.size(0), self.opt.nz)
            self.fake_B_encoded = self.netG(self.real_A_encoded, self.z_encoded)

        # Now we combine our fixed z with our known random            
        if self.opt.mod_rlhf:
            # produce an image from the previous style plus a decreasing amount of randomness 
            ratio = self.opt.ratio * (self.opt.ratio_decay ** (self.using_z + 1))
            
            # This parts enables you to keep one iamge the same if you opt too, mainly for testing
            if self.opt.fixed_choice:
                self.count += 1
                if self.count == 1:
                    print(round(ratio * 100, 2),'% Random')
                    ratio = 0
                if self.count == 3:
                    self.count = 0
                    
            # below the different varaitions tried for adding randomness
            #self.encoded_random =2 * (self.z_encoded * self.z_random * ratio  +  self.z_encoded * (1-ratio))
            self.encoded_random = self.z_random * ratio + self.z_encoded * (1-ratio)
            #self.encoded_random = self.z_random
            #self.encoded_random = self.forward_image_variance(self.z_encoded, ratio)

            self.fake_B_random = self.netG(self.real_A_encoded, self.encoded_random)
            self.fake_B_list.append(self.fake_B_random)
            # save all noises for later use
            self.current_noise = (self.fake_B_random, self.encoded_random, self.z_encoded, self.z_random)

        else:
            self.fake_B_random = self.netG(self.real_A_encoded, self.z_random)

        if self.opt.conditional_D:
            self.fake_data_encoded = torch.cat([self.real_A_encoded, self.fake_B_encoded], 1)
            self.real_data_encoded = torch.cat([self.real_A_encoded, self.real_B_encoded], 1)
            self.fake_data_random = torch.cat([self.real_A_encoded, self.fake_B_random], 1)
            self.real_data_random = torch.cat([self.real_A_random, self.real_B_random], 1)
        else:
            self.fake_data_encoded = self.fake_B_encoded
            self.fake_data_random = self.fake_B_random
            if self.using_z < 1:
                # if using z we don't have a b to input here ! 
                self.real_data_encoded = self.real_B_encoded
                self.real_data_random = self.real_B_random

        # compute z_predict
        if self.opt.lambda_z > 0.0:
            if self.opt.mod_fmt:
                preprocessed = self.netFMT.forward(self.fake_B_random, self.real_A_encoded)
            else:
                preprocessed = self.fake_B_random
            self.mu2, logvar2 = self.netE(preprocessed)  # mu2 is a point estimate

    def forward_image_variance(self, encoded_image, ratio):
        ratio = ratio *2
        if self.image_count == 0:
            new_encoded = encoded_image * (1 + ratio)
            self.image_count += 1
        elif self.image_count == 1:
            new_encoded = encoded_image
            self.image_count += 1
        else:
            new_encoded = encoded_image * (1 - ratio)
            self.image_count = 0
        return new_encoded

    def backward_D(self, netD, real, fake):
        # Fake, stop backprop to the generator by detaching fake_B
        pred_fake = netD(fake.detach())
        # real
        pred_real = netD(real)
        loss_D_fake, _ = self.criterionGAN(pred_fake, False)
        loss_D_real, _ = self.criterionGAN(pred_real, True)
        # Combined loss
        loss_D = loss_D_fake + loss_D_real
        loss_D.backward()
        return loss_D, [loss_D_fake, loss_D_real]

    def backward_G_GAN(self, fake, netD=None, ll=0.0):
        if ll > 0.0:
            pred_fake = netD(fake)
            loss_G_GAN, _ = self.criterionGAN(pred_fake, True)
        else:
            loss_G_GAN = 0
        return loss_G_GAN * ll

    def backward_EG(self):
        # 1, G(A) should fool D
        self.loss_G_GAN = self.backward_G_GAN(self.fake_data_encoded, self.netD, self.opt.lambda_GAN)
        if self.opt.use_same_D:
            self.loss_G_GAN2 = self.backward_G_GAN(self.fake_data_random, self.netD, self.opt.lambda_GAN2)
        else:
            self.loss_G_GAN2 = self.backward_G_GAN(self.fake_data_random, self.netD2, self.opt.lambda_GAN2)
        # 2. KL loss
        if self.opt.lambda_kl > 0.0:
            self.loss_kl = torch.sum(1 + self.logvar - self.mu.pow(2) - self.logvar.exp()) * (-0.5 * self.opt.lambda_kl)
        else:
            self.loss_kl = 0
        # 3, reconstruction |fake_B-real_B|
        if self.opt.lambda_L1 > 0.0:
            self.loss_G_L1 = self.criterionL1(self.fake_B_encoded, self.real_B_encoded) * self.opt.lambda_L1
        else:
            self.loss_G_L1 = 0.0

        self.loss_G = self.loss_G_GAN + self.loss_G_GAN2 + self.loss_G_L1 + self.loss_kl
        self.loss_G.backward(retain_graph=True)

    def update_D(self):
        self.set_requires_grad([self.netD, self.netD2], True)
        # update D1
        if self.opt.lambda_GAN > 0.0:
            self.optimizer_D.zero_grad()
            self.loss_D, self.losses_D = self.backward_D(self.netD, self.real_data_encoded, self.fake_data_encoded)
            if self.opt.use_same_D:
                self.loss_D2, self.losses_D2 = self.backward_D(self.netD, self.real_data_random, self.fake_data_random)
            self.optimizer_D.step()

        if self.opt.lambda_GAN2 > 0.0 and not self.opt.use_same_D:
            self.optimizer_D2.zero_grad()
            self.loss_D2, self.losses_D2 = self.backward_D(self.netD2, self.real_data_random, self.fake_data_random)
            self.optimizer_D2.step()

    def backward_G_alone(self):
        # 3, reconstruction |(E(G(A, z_random)))-z_random|
        if self.opt.lambda_z > 0.0:
            self.loss_z_L1 = self.criterionZ(self.mu2, self.z_random) * self.opt.lambda_z
            self.loss_z_L1.backward()
        else:
            self.loss_z_L1 = 0.0

    def update_G_and_E(self):
        # update G and E
        self.set_requires_grad([self.netD, self.netD2], False)
        self.optimizer_E.zero_grad()
        self.optimizer_G.zero_grad()
        self.backward_EG()

        # update G alone
        if self.opt.lambda_z > 0.0:
            self.set_requires_grad([self.netE], False)
            self.backward_G_alone()
            self.set_requires_grad([self.netE], True)

        self.optimizer_E.step()
        self.optimizer_G.step()

    def optimize_parameters(self):
        self.forward()
        self.update_G_and_E()
        self.update_D()
        
    def update_rlhf(self, choice, noises, options):
        # Loses are calulated at each choice but steps are only taken at the end
        self.set_requires_grad([self.netD, self.netD2, self.netE], False)
        self.set_requires_grad([self.netG], True)

        # Loss from g network
        self.loss_GE_RLHF += self.update_GE_rlhf(choice, options)
        self.loss_G_RLHF += self.update_G_rlhf(choice, options)
        self.loss_E_RLHF += self.update_E_rlhf(choice, noises)
        
        print('Loss for E: ', self.loss_E_RLHF)
        print('Loss for G: ', self.loss_G_RLHF)
        print('Loss for GE: ', self.loss_GE_RLHF)
        self.fake_B_list = []
        
    def update_E_rlhf(self, choice, noises):
        # loss is how far we are from what we predicted to chosen one
        return self.criterionL1(noises[choice - 1][1], self.z_encoded) * self.opt.lambda_L1
        
    def update_GE_rlhf(self, choice, options):
        self.loss_G_GAN = 0
        # First gan loss for unchosen images
        for i in range(0,len(self.fake_B_list)):
            if i != choice-1:
                self.loss_G_GAN += self.backward_G_GAN(self.fake_B_list[choice - 1], self.netD, self.opt.lambda_GAN2)
                
        # Second gan loss is for chosen image
        if self.opt.use_same_D:
            self.loss_G_GAN2 = self.backward_G_GAN(self.fake_B_list[choice - 1], self.netD, self.opt.lambda_GAN)
        else:
            self.loss_G_GAN2 = self.backward_G_GAN(self.fake_B_list[choice - 1], self.netD2, self.opt.lambda_GAN)
                
        return (self.loss_G_GAN + self.loss_G_GAN2)
    
    def update_G_rlhf(self, choice, options):
        # This loss we take unchosen and give it more points the more variance is present
            self.loss_G_RLHF_negative = 0
            for i in range(0,len(options)):
                if i != choice - 1:
                    # for negative compare unchosen to chosen
                    self.loss_G_RLHF_negative -= self.criterionL1(self.fake_B_list[i], self.fake_B_list[choice - 1]) * self.opt.lambda_L1
            if self.loss_G_RLHF_negative < -20:
                return self.loss_G_RLHF_negative * 0.001
            else:
                return self.loss_G_RLHF_negative
    
    def update_E_final_rlhf(self):
        # Combined loss backward first, general improvement
        self.set_requires_grad([self.netD, self.netD2], False)
        self.set_requires_grad([self.netG, self.netE], True)
        self.loss_GE_RLHF.backward(retain_graph=True)
        
        # Backward just G loss, improve varaition from style
        self.set_requires_grad([self.netD, self.netD2, self.netE], False)
        self.set_requires_grad([self.netG], True)
        self.loss_G_RLHF.backward(retain_graph=True)
        
        # Backward just E loss, improve accuracy of extracted style
        self.set_requires_grad([self.netG, self.netD, self.netD2], False)
        self.set_requires_grad([self.netE], True)
        self.loss_E_RLHF.backward()
        
        self.optimizer_G.step()
        self.optimizer_E.step()
        
        # RESET everything
        self.loss_G_RLHF = 0
        self.loss_GE_RLHF = 0
        self.loss_E_RLHF = 0
        self.optimizer_G.zero_grad()
        self.optimizer_E.zero_grad()

