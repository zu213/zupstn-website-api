from options.train_options import TrainOptions

def pix2pix_opts(name):
    # Setting up a bunch of options

    opt_pix2pix = TrainOptions()

    opt_pix2pix.model = 'pix2pix'
    opt_pix2pix.name = name
    opt_pix2pix.gpu_ids = [0]
    # true since rlhf
    opt_pix2pix.isTrain = True
    opt_pix2pix.checkpoints_dir = './checkpoints/rlhf'
    opt_pix2pix.preprocess = 'resize_and_crop'
    opt_pix2pix.input_nc = 3
    opt_pix2pix.output_nc = 3
    opt_pix2pix.ngf = 64
    opt_pix2pix.ndf = 64
    opt_pix2pix.netD = 'basic'
    opt_pix2pix.netG = 'resnet_9blocks'
    opt_pix2pix.netG = 'unet_256' #'resnet_9blocks'
    opt_pix2pix.n_layers_D = 3
    opt_pix2pix.norm = 'batch'
    opt_pix2pix.init_type = 'normal'
    opt_pix2pix.init_gain = 0.02
    opt_pix2pix.no_dropout = False
    opt_pix2pix.load_iter = 0
    opt_pix2pix.epoch = 'latest'
    opt_pix2pix.batch_size = 1
    opt_pix2pix.load_size = 256
    opt_pix2pix.crop_size = 256
    opt_pix2pix.display_winsize = 256
    opt_pix2pix.use_wandb = False
    opt_pix2pix.verbose = False
    opt_pix2pix.suffix = ''
    opt_pix2pix.wandb_project_name = 'CycleGAN-and-pix2pix'
    opt_pix2pix.direction = 'AtoB'

    opt_pix2pix.epoch_count = 1
    opt_pix2pix.save_latest_freq = 10
    opt_pix2pix.save_epoch_freq = 1
    opt_pix2pix.save_by_iter = True
    opt_pix2pix.continue_train = True
    opt_pix2pix.phase = 'train'

    opt_pix2pix.lr = 0.0002
    opt_pix2pix.beta1 = 0.5
    opt_pix2pix.lr_policy = 'linear'
    opt_pix2pix.n_epochs = 1
    opt_pix2pix.n_epochs_decay = 0
    opt_pix2pix.lr_decay_iters = 1

    opt_pix2pix.lambda_A = 10.0
    opt_pix2pix.lambda_B = 10.0
    opt_pix2pix.lambda_identity = 0.5
    opt_pix2pix.sketch2art = False

    opt_pix2pix.num_threads = 0   # test code only supports num_threads = 0
    opt_pix2pix.batch_size = 1    # test code only supports batch_size = 1
    opt_pix2pix.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt_pix2pix.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt_pix2pix.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    opt_pix2pix.lambda_identity = -1
    opt_pix2pix.pool_size = 0
    opt_pix2pix.gan_mode = 'vanilla'


# In[6]: bicycle gan opts

def bicycle_gan_opts(name):
    opt = TrainOptions()

    opt.dataroot = "./datasets/data"
    opt.name = name
    opt.gpu_ids = []
    opt.model = "bicycle_gan"
    opt.input_nc = 3
    opt.output_nc = 3
    opt.lambda_kl = 0.01
    opt.lambda_L1 =10.0
    opt.ngf = 64
    opt.ndf = 64
    opt.nef  = 64
    opt.netD = 'basic_256_multi'
    opt.netD2 = 'basic_256_multi'
    opt.netG = 'unet_256'
    opt.netE = 'resnet_256'
    opt.n_layers_D = 3
    opt.norm = "instance"
    opt.init_type = 'xavier'
    opt.init_gain = 0.02
    opt.no_dropout = True
    opt.dataset_mode = "unaligned"
    opt.direction = "AtoB"
    opt.serial_batches = False
    opt.num_threads = 4
    opt.batch_size = 2
    opt.load_size = 256
    opt.crop_size = 256
    opt.preprocess = "resize_and_crop"
    opt.no_flip = False
    opt.display_winsize = 256
    opt.load_iter = 0
    opt.verbose = False
    opt.suffix = ""
    opt.use_wandb = False
    opt.wandb_project_name = "CycleGAN-and-pix2pix"
    opt.dataset_mode = "unaligned"
    opt.basic_encode = True
    opt.display_freq = 100
    opt.display_ncols = 4
    opt.display_id = 1
    opt.display_env = "main"
    opt.display_port = 8097
    opt.update_html_freq = 1000
    opt.print_freq = 100
    opt.no_html = False
    opt.save_latest_freq = 10
    opt.save_epoch_freq = 1
    opt.save_by_iter = False
    opt.continue_train = True
    opt.epoch_count = 1
    opt.phase = "train"
    opt.niter = 1
    opt.niter_decay = 0
    opt.beta1 = 0.5
    opt.lr = 0.0002
    opt.gan_mode = "lsgan"
    opt.pool_size = 50
    opt.lr_policy = "linear"
    opt.lr_decay_iters = 50
    opt.lambda_A = 10.0
    opt.lambda_B = 10.0
    opt.lambda_identity = 0.0
    opt.isTrain = True
    opt.lambda_GAN = 1.0
    opt.lambda_GAN2 = 1.0
    opt.use_same_D = True
    opt.nl = 'relu'
    opt.use_dropout = False
    opt.where_add = 'all'
    opt.upsample = 'basic'
    opt.num_Ds = 2
    opt.center_crop = False
    opt.conditional_D = False
    opt.lambda_z = 0.5

    opt.mod_rlhf=True
    opt.mod_fmt =False
    opt.fixed_choice = True

    # Epoch to load from
    opt.checkpoints_dir = "./diss-img-tool-lw/pretrained_models"
    opt.epoch = '1675'
    opt.nz = 32

    # extra add ons
    opt.mod = True
    opt.ratio = 0.9
    opt.ratio_decay = 0.95

    opt.nz = 32

    # extra add ons
    opt.mod = True

    opt.ratio = 0.75
    opt.ratio_decay = 0.95

    return opt