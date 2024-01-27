import logging
from collections import OrderedDict

import torch
import torch.nn as nn
import os
import model.networks as networks
from .base_model import BaseModel
logger = logging.getLogger('base')
from pytorch_msssim import ssim
import torch
import torch.nn.functional as F
from torchvision import transforms, utils
from network import resnext as Resnet_Deep
#from itertools import cycle
#from CoRRN import CoRRN
#from network import resnet_d as Resnet_Deep
#from network import resnext as Resnet_Deep
from network.nn.mynn import Upsample, Norm2d
import torch.nn.functional as F
from network.vgg import Vgg19
def compute_gradient(img):
    gradx=img[...,1:,:]-img[...,:-1,:]
    grady=img[...,1:]-img[...,:-1]
    return gradx,grady
class GradientLoss(nn.Module):
    def __init__(self):
        super(GradientLoss, self).__init__()
        self.loss = nn.L1Loss()
    
    def forward(self, predict, target):
        predict_gradx, predict_grady = compute_gradient(predict)
        target_gradx, target_grady = compute_gradient(target) 
        
        return self.loss(predict_gradx, target_gradx) + self.loss(predict_grady, target_grady)

class MeanShift(nn.Conv2d):
    def __init__(self, data_mean, data_std, data_range=1, norm=True):
        """norm (bool): normalize/denormalize the stats"""
        c = len(data_mean)
        super(MeanShift, self).__init__(c, c, kernel_size=1)
        std = torch.Tensor(data_std)
        self.weight.data = torch.eye(c).view(c, c, 1, 1)
        if norm:
            self.weight.data.div_(std.view(c, 1, 1, 1))
            self.bias.data = -1 * data_range * torch.Tensor(data_mean)
            self.bias.data.div_(std)
        else:
            self.weight.data.mul_(std.view(c, 1, 1, 1))
            self.bias.data = data_range * torch.Tensor(data_mean)
        self.requires_grad = False
class VGGLoss(nn.Module):
    def __init__(self, vgg=None, weights=None, indices=None, normalize=True):
        super(VGGLoss, self).__init__()
        if vgg is None:
            self.vgg = Vgg19().cuda()
        else:
            self.vgg = vgg
        self.criterion = nn.L1Loss()
        self.weights = weights or [1.0 / 2.6, 1.0 / 4.8, 1.0 / 3.7, 1.0 / 5.6, 10 / 1.5]
        self.indices = indices or [2, 7, 12, 21, 30]
        if normalize:
            self.normalize = MeanShift([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], norm=True).cuda()
        else:
            self.normalize = None

    def forward(self, x, y):
        if self.normalize is not None:
            x = self.normalize(x)
            y = self.normalize(y)
        x_vgg, y_vgg = self.vgg(x, self.indices), self.vgg(y, self.indices)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())

        return loss
def reflect_strength(raw_imgs, reflect_imgs):
    with torch.no_grad():
        raw_imgs=(raw_imgs+1)/2.0
        reflect_imgs=(reflect_imgs+1)/2.0
        raw_imgs = raw_imgs.permute(0, 2, 3, 1)#* 255
        reflect_imgs = reflect_imgs.permute(0, 2, 3, 1)#* 255
        raw_imgs_gray = (0.2989 * raw_imgs[:,:,:,0] + 0.5870 * raw_imgs[:,:,:,1] + 0.1140 * raw_imgs[:,:,:,2]).unsqueeze(-1)
        reflect_imgs_gray = (0.2989 * reflect_imgs[:,:,:,0] + 0.5870 * reflect_imgs[:,:,:,1] + 0.1140 * reflect_imgs[:,:,:,2]).unsqueeze(-1)
        raw_imgs_gray = F.adaptive_avg_pool2d(raw_imgs_gray.permute(0, 3, 1, 2), (14, 14)).view(-1,196)
        reflect_imgs_gray = F.adaptive_avg_pool2d(reflect_imgs_gray.permute(0, 3, 1, 2), (14, 14)).view(-1,196)
        means=reflect_imgs_gray/(raw_imgs_gray+reflect_imgs_gray+ 1e-6)
        # means = torch.flatten(xx, start_dim=1)
        #xxx=xx-means
    return means
class _AtrousSpatialPyramidPoolingModule(nn.Module):
    """
    operations performed:
      1x1 x depth
      3x3 x depth dilation 6
      3x3 x depth dilation 12
      3x3 x depth dilation 18
      image pooling
      concatenate all together
      Final 1x1 conv
    """

    def __init__(self, in_dim, reduction_dim=256, output_stride=16, rates=(6, 12, 18)):
        super(_AtrousSpatialPyramidPoolingModule, self).__init__()

        if output_stride == 8:
            rates = [2 * r for r in rates]
        elif output_stride == 16:
            pass
        else:
            raise 'output stride of {} not supported'.format(output_stride)

        self.features = []
        # 1x1
        self.features.append(
            nn.Sequential(nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                          Norm2d(reduction_dim), nn.ReLU(inplace=True)))
        # other rates
        for r in rates:
            self.features.append(nn.Sequential(
                nn.Conv2d(in_dim, reduction_dim, kernel_size=3,
                          dilation=r, padding=r, bias=False),
                Norm2d(reduction_dim),
                nn.ReLU(inplace=True)
            ))
        self.features = torch.nn.ModuleList(self.features)

        # img level features
        self.img_pooling = nn.AdaptiveAvgPool2d(1)
        self.img_conv = nn.Sequential(
            nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
            Norm2d(reduction_dim), nn.ReLU(inplace=True))

    def forward(self, x):
        x_size = x.size()

        img_features = self.img_pooling(x)
        img_features = self.img_conv(img_features)
        #img_features = Upsample(img_features, x_size[2:])
        img_features=F.interpolate(img_features, size=x_size[2:],mode='bilinear', align_corners=True)
        out = img_features

        for f in self.features:
            y = f(x)
            out = torch.cat((out, y), 1)
        return out

class Res_step(nn.Module):
    def __init__(self):
        super().__init__()
        #han-satst
        #self.resnet = Resnet_Deep.resnet50()
        self.resnet = Resnet_Deep.resnext101_32x8()
        #resnet = Resnet_Deep.resnet101()
        self.resnet.layer0 = nn.Sequential(self.resnet.conv1, self.resnet.bn1, self.resnet.relu, self.resnet.maxpool)
        self.aspp = _AtrousSpatialPyramidPoolingModule(in_dim=2048, reduction_dim=256,
                                                       output_stride=8)
        self.resnet_transform=transforms.Compose([
        transforms.Resize(size=224),
        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
    ])
        self.fc1 = nn.Linear(2048, 1)
        self.downs = nn.Conv2d(1280, 1, kernel_size=1)
    def forward(self,x):
        x=self.resnet_transform(x)
        #x_t=self.resnet(x_t)
        x=self.resnet.layer0(x)  # 200
        x=self.resnet.layer1(x)
        x=self.resnet.layer2(x)
        x=self.resnet.layer3(x)
        x_res_oyut=self.resnet.layer4(x)


        x_aspp=self.aspp(x_res_oyut)
        x_aspp=self.downs(x_aspp)
        x_aspp=x_aspp.squeeze(1)
        return x_aspp.view(-1,49)

class DDPM(BaseModel):
    def __init__(self, opt):
        super(DDPM, self).__init__(opt)
        # define network and load pretrained models
        self.netG = self.set_device(networks.define_G(opt))
        self.schedule_phase = None
        self.loss_func = nn.L1Loss(reduction='sum')
        
        self.Res_step=self.set_device(Res_step())
        self.Res_step.eval()
        self.loss_func = nn.L1Loss(reduction='sum')
        self.vgg = self.set_device(Vgg19(requires_grad=False))
        self.loss_vgg=VGGLoss(self.vgg)
        self.gradient_loss=GradientLoss()
        
        if self.opt['phase'] == 'train':
            self.netG.train()
            # find the parameters to optimize
            if opt['model']['finetune_norm']:
                optim_params = []
                for k, v in self.netG.named_parameters():
                    v.requires_grad = False
                    if k.find('transformer') >= 0:
                        v.requires_grad = True
                        v.data.zero_()
                        optim_params.append(v)
                        logger.info(
                            'Params [{:s}] initialized to 0 and will optimize.'.format(k))
            else:
                optim_params = list(self.netG.parameters())

            self.optG = torch.optim.Adam(
                optim_params, lr=opt['train']["optimizer"]["lr"])
            self.log_dict = OrderedDict()
        self.load_network()
        self.print_network()

    def feed_data(self, data):
        self.data = self.set_device(data)

    def optimize_parameters(self):
        self.optG.zero_grad()
        with torch.no_grad():
           x_mean,x_time=self.Res_step((self.data['SR']+1)/2.0)
        x_time=x_time.clamp(0,1)
        #x_time = x_time.unsqueeze(1)
        #b,n=x_time.shape
        #x_pre = self.netG(self.data,x_time)
        ground=self.data['HR']
        x_pre = self.netG(self.data,x_time)
        #ground=self.data['HR']
        ground0_1=(self.data['HR']+1)/2.0
        loss_ssim = (1- ssim((x_pre+1)/2.0, ground0_1, data_range=1, size_average=True))*0.2#.abs().mean()/10
        b, c, h, w = ground.size()
        loss_pix = F.mse_loss(x_pre, ground)*0.4
        
        loss_icnn_vgg = self.loss_vgg((x_pre+1)/2.0,ground0_1)*0.1
        loss_gradient=self.gradient_loss((x_pre+1)/2.0,ground0_1)*0.6
        loss2=loss_pix.sum()+loss_ssim.sum()+loss_icnn_vgg+loss_gradient
       
        loss2.backward()
        self.optG.step()

        # set log
        self.log_dict['l_pix'] = loss2.item()
        self.log_dict['l_pixxx'] = loss_pix.item()
        self.log_dict['l_vgg'] = loss_icnn_vgg.item()
        self.log_dict['l_grad'] = loss_gradient.item()
        self.log_dict['l_ssim'] = loss_ssim.item()

    def test(self, continous=False):
        self.netG.eval()
        self.Res_step.eval()
        with torch.no_grad():
            #x_mean,x_time=self.Res_step((self.data['SR']+1)/2.0)
            x_time=self.Res_step((self.data['SR']+1)/2.0)
            b,n=x_time.shape
            x_time=x_time.clamp(0,1)
            if isinstance(self.netG, nn.DataParallel):      
                self.SR= self.netG.module(
                    self.data['SR'], x_time)
            else:
                self.SR= self.netG(
                    self.data['SR'], x_time)
        
        self.netG.train()

    def sample(self, batch_size=1, continous=False):
        self.netG.eval()
        with torch.no_grad():
            if isinstance(self.netG, nn.DataParallel):
                self.SR = self.netG.module.sample(batch_size, continous)
            else:
                self.SR = self.netG.sample(batch_size, continous)
        self.netG.train()

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_LR=True, sample=False):
        out_dict = OrderedDict()
        if sample:
            out_dict['SAM'] = self.SR.detach().float().cpu()
        else:
            out_dict['SR'] = self.SR.detach().float().cpu()
            out_dict['INF'] = self.data['SR'].detach().float().cpu()
            out_dict['HR'] = self.data['HR'].detach().float().cpu()
            if need_LR and 'LR' in self.data:
                out_dict['LR'] = self.data['LR'].detach().float().cpu()
            else:
                out_dict['LR'] = out_dict['INF']
        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
                                             self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)

        logger.info(
            'Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
        logger.info(s)

    def save_network(self, epoch, iter_step):
        gen_path = os.path.join(
            self.opt['path']['checkpoint'], 'I{}_E{}_gen.pth'.format(iter_step, epoch))
        # gen
        network = self.netG
        if isinstance(self.netG, nn.DataParallel):
            network = network.module                    
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, gen_path)
        # opt
        
        logger.info(
            'Saved model in [{:s}] ...'.format(gen_path))

    def load_network(self):
        load_path = self.opt['path']['resume_state']
        self.Res_step.load_state_dict(torch.load('./priors_ckp/Res20000_E48_gen.pth'), strict=True)
        if load_path is not None:
            logger.info(
                'Loading pretrained model for G [{:s}] ...'.format(load_path))
            gen_path = '{}_gen.pth'.format(load_path)            # gen
            network = self.netG
            if isinstance(self.netG, nn.DataParallel):
                network = network.module
            network.load_state_dict(torch.load(
                gen_path), strict=(not self.opt['model']['finetune_norm']))
