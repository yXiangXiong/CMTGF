import torch
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks3d
import torch.nn.functional as F

class PairedPix2Pix3dModel(BaseModel):
    def name(self):
        return 'PairedPix2Pix3dModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):

        # changing the default values to match the pix2pix paper
        # (https://phillipi.github.io/pix2pix/)
        parser.set_defaults(pool_size=0, no_lsgan=True, norm='batch')
        parser.set_defaults(dataset_mode='aligned')
        parser.set_defaults(netG='unet_256')

        return parser

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = ['G_GAN', 'G_L1', 'G_CE', 'D_real', 'D_fake', 'C_acc']
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        self.visual_names = ['real_A', 'fake_B', 'real_B', 'real_label', 'Pred_label_prob', 'Pred_label_index']
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        if self.isTrain:
            self.model_names = ['G', 'C', 'D']
        else:  # during test time, only load Gs
            self.model_names = ['G', 'C']

        last_fc = True
        if opt.mode == 'score':
            last_fc = True
        elif opt.mode == 'feature':
            last_fc = False

        # load/define networks
        self.netG = networks3d.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        self.netC = networks3d.define_C(opt.n_classes, opt.resnet_shortcut, opt.sample_size1, opt.sample_size2,
                                        opt.sample_duration, opt.netC, last_fc, self.gpu_ids)

        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            self.netD = networks3d.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                    opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            self.fake_AB_pool = ImagePool(opt.pool_size)
            # define loss functions
            self.criterionGAN = networks3d.GANLoss(use_lsgan=not opt.no_lsgan).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionC = torch.nn.CrossEntropyLoss()

            # initialize optimizers
            self.optimizers = []
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_C = torch.optim.Adam(self.netC.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            self.optimizers.append(self.optimizer_C)

    def set_input(self, input):
        AtoB = self.opt.which_direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.real_label = input['L'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward_G(self):
        self.fake_B = self.netG(self.real_A)

    def forward_C(self):
        pred_AB = torch.cat((self.fake_B[:, 1, :, :, :].unsqueeze(1), self.real_A), 1)
        outputs = self.netC(pred_AB.detach())
        prob = torch.softmax(outputs, dim=1)[:, 1]
        _, pred = outputs.topk(k=1, dim=1, largest=True)
        self.Pred_label_prob = prob
        self.Pred_label_index = pred

    def calculate_accuracy(self, outputs, labels):
        batch_size = labels.size(0)

        _, pred = outputs.topk(k=1, dim=1, largest=True)
        pred = pred.t()
        correct = pred.eq(labels.view(1, -1))
        n_correct_elems = correct.float().sum().data

        return n_correct_elems / batch_size

    def backward_C(self):
        pred_AB = torch.cat((self.fake_B[:, 1, :, :, :].unsqueeze(1), self.real_A), 1)
        pred_label = self.netC(pred_AB.detach())
        self.loss_C = self.criterionC(pred_label, self.real_label)
        self.loss_C_acc = self.calculate_accuracy(pred_label, self.real_label)
        self.loss_C.backward()

    def backward_D(self):
        # Fake
        # stop backprop to the generator by detaching fake_B
        fake_AB = self.fake_AB_pool.query(torch.cat((self.real_A, self.fake_B), 1))
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)

        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)

        # Combined loss
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5

        self.loss_D.backward()

    def backward_G(self):
        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)

        pred_AB = torch.cat((self.real_A, self.fake_B[:, 1, :, :, :].unsqueeze(1)), 1)
        pre_label = self.netC(pred_AB)
        self.loss_G_CE = self.criterionC(pre_label, self.real_label) * self.opt.lambda_C

        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1
        self.loss_G = self.loss_G_GAN + self.loss_G_L1 + self.loss_G_CE

        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward_G()

        # update D
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

        # update C
        self.set_requires_grad(self.netC, True)
        self.optimizer_C.zero_grad()
        self.backward_C()
        self.optimizer_C.step()

        # update G
        self.set_requires_grad(self.netD, False)
        self.set_requires_grad(self.netC, False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()