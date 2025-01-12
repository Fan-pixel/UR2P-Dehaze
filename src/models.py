import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from .networks import Discriminator, HazeRemovalNet,HazeProduceNet, DepthEstimationNet
from .loss import AdversarialLoss
from .net import net
from .utils import C_loss,R_loss,P_loss

import sys


class BaseModel(nn.Module):
    def __init__(self, config):
        super(BaseModel, self).__init__()

        if config.MODEL == 1:
            self.name = 'reconstruct'
        elif config.MODEL == 2:
            self.name = 'feature_process'

        self.config = config
        self.iteration = 0

        self.gen_weights_path = os.path.join(config.PATH, 'weights.pth')
        self.gen_optimizer_path = os.path.join(config.PATH, 'optimizer_'+self.name + '.pth')
        self.dis_weights_path = os.path.join(config.PATH, self.name + '_dis.pth')
        self.transformer_weights_path = os.path.join(config.PATH, self.name + '.pth')
        self.transformer_discriminator_weights_path = os.path.join(config.PATH, self.name + '_dis.pth')
        self.reconstructor_weights_path = os.path.join(config.PATH, self.name + '.pth')

    def load(self):
        pass

    def save(self):
        pass


model_net= net().cuda()

class Model(BaseModel):
    def __init__(self, config):
        super(Model, self).__init__(config)
        self.l2_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        self.adversarial_loss = AdversarialLoss(type='lsgan')
        self.use_dc_A = False if self.config.DATASET == 'SOTS-indoor' else True



        self.depth_estimator = DepthEstimationNet(config.BASE_CHANNEL_NUM // 2, min_d=config.MIN_D, max_d=config.MAX_D, path=self.gen_weights_path[:-4]+'_'+self.name+'.pth')
        self.net_h2c = HazeRemovalNet(config.BASE_CHANNEL_NUM // 2 , min_beta=config.MIN_BETA, max_beta=config.MAX_BETA, min_d=config.MIN_D, max_d=config.MAX_D, path=self.gen_weights_path[:-4]+'_'+self.name+'.pth', use_dc_A=self.use_dc_A)
        self.net_c2h = HazeProduceNet(config.BASE_CHANNEL_NUM // 2, in_channels=3, out_channels=3, min_beta=config.MIN_BETA, max_beta=config.MAX_BETA)


        self.epoch = 0

        if config.MODE == 1:

            self.discriminator_h2c = Discriminator(in_channels=3, use_spectral_norm=True, use_sigmoid=True)
            self.discriminator_c2h = Discriminator(in_channels=3, use_spectral_norm=True, use_sigmoid=True)


            self.optimizer = optim.Adam(
                [
                    {'params': self.net_c2h.parameters()},
                    {'params': self.net_h2c.parameters()},

                ],

                lr=float(config.LR),
                betas=(config.BETA1, config.BETA2),
                weight_decay=config.WEIGHT_DECAY
            )

            self.optimizer_depth = optim.Adam(
                [
                    {'params': self.depth_estimator.parameters()},
                ],

                lr=float(config.LR),
                betas=(config.BETA1, config.BETA2),
                weight_decay=config.WEIGHT_DECAY
            )

            self.optimizer_dis = optim.Adam(
                [
                    {'params': self.discriminator_h2c.parameters()},
                    {'params': self.discriminator_c2h.parameters()},
                ],

                lr=float(config.LR * config.D2G_LR),
                betas=(config.BETA1, config.BETA2)
            )



            self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer,
                                                            milestones=[100], gamma=0.5,
                                last_epoch = self.epoch - 1)

#对干净图像进行深度估计
    def forward_depth(self, clean_imgs): ## out: e^(-d(x)), N,1,H,W

        n,c,h,w = clean_imgs.shape
        input = F.interpolate(clean_imgs, mode='bilinear', size=[self.config.CROP_SIZE,self.config.CROP_SIZE])
        result = self.depth_estimator(input)
        result = F.interpolate(result, mode='bilinear', size=[h,w])
        return result

#将清晰图像转换为模糊图像，并使用给定的透射率参数。
    def forward_c2h_given_parameters(self, clean_imgs, ex, beta):
        x = self.net_c2h.forward(clean_imgs,ex,beta)
        return x
#将清晰图像输入模型，并使用随机生成的透射率参数生成对应的模糊图像。
    def forward_c2h_random_parameters(self, clean_imgs, ex, requires_paras=False): # random beta
        x, beta = self.net_c2h.forward_random_parameters(clean_imgs, ex)
        if requires_paras:
            return x, beta
        else:
            return x
#用于将模糊图像转换为清晰图像
    def forward_h2c(self, hazy_imgs, require_paras=False, use_guided_filter=False):
        # if requires_paras: return clean, ed, beta
        return self.net_h2c(hazy_imgs, require_paras, use_guided_filter=use_guided_filter)


    def process(self, clean_images, hazy_images):
        self.iteration += 1
        self.optimizer_dis.zero_grad()
        self.discriminator_h2c.zero_grad()
        self.discriminator_c2h.zero_grad()


        L1, R1, X1 = model_net(hazy_images)

        clean_images_h2c, gt_ed_h2c, pred_beta_h2c = self.forward_h2c(hazy_images, require_paras=True)
        pred_ed_h2c = self.forward_depth(clean_images_h2c)
        hazy_images_h2c2h = self.forward_c2h_given_parameters(clean_images_h2c, pred_ed_h2c, pred_beta_h2c)


        pred_ed_clean = self.forward_depth(clean_images)
        hazy_images_c2h, beta_gt_c2h = self.forward_c2h_random_parameters(clean_images, pred_ed_clean, requires_paras=True)
        clean_images_c2h2c, ed_pred_c2h, beta_pred_c2h = self.forward_h2c(hazy_images_c2h,require_paras=True)


        gen_loss = 0
        dis_loss = 0

        #### dis loss ####


        dis_real_clean, _ = self.discriminator_h2c(clean_images)
        dis_fake_clean, _ = self.discriminator_h2c(
            clean_images_h2c.detach())

        dis_clean_real_loss = self.adversarial_loss((dis_real_clean), is_real=True, is_disc=True)
        dis_clean_fake_loss = self.adversarial_loss((dis_fake_clean), is_real=False, is_disc=True)

        dis_clean_loss = (dis_clean_real_loss + dis_clean_fake_loss) / 2
        dis_clean_loss.backward()

        dis_real_haze, _ = self.discriminator_c2h(
            (hazy_images))
        dis_fake_haze, _ = self.discriminator_c2h(
            hazy_images_c2h.detach())

        dis_haze_real_loss = self.adversarial_loss((dis_real_haze), is_real=True, is_disc=True)
        dis_haze_fake_loss = self.adversarial_loss((dis_fake_haze), is_real=False, is_disc=True)
        dis_haze_loss = (dis_haze_real_loss + dis_haze_fake_loss) / 2
        dis_haze_loss.backward()

        dis_loss += (dis_clean_fake_loss + dis_clean_real_loss + dis_haze_real_loss + dis_haze_fake_loss) / 4

        self.optimizer_dis.step()


        ############################################net
        L2, R2, X2 = model_net(clean_images_h2c)     
        loss1 = C_loss(R1, R2)
        loss2 = R_loss(L1, R1, hazy_images, X1)
        loss3 = P_loss(hazy_images, X1)
        
        loss_net =  loss1 * 1 + loss2 * 1 + loss3 * 500
        loss_net = 0.1 * loss_net



        ### gen loss ####
        self.optimizer.zero_grad()
        self.net_h2c.zero_grad()
        self.net_c2h.zero_grad()

        ### Parameter loss ###3





        ### cycle reconstruction loss###
        cycle_loss_c2h2c = self.l1_loss(clean_images,
                                        clean_images_c2h2c)
        cycle_loss_h2c2h = self.l1_loss(hazy_images, hazy_images_h2c2h)
        cycle_loss = (cycle_loss_c2h2c + cycle_loss_h2c2h)


        ### para loss ###
        para_beta_loss = self.l2_loss(beta_pred_c2h, beta_gt_c2h.detach())
        para_loss = para_beta_loss

        ### global ###
        gen_fake_haze, _ = self.discriminator_c2h(
            (hazy_images_c2h))
        gen_fake_clean, _ = self.discriminator_h2c(
            clean_images_h2c)


        gen_fake_haze_ganloss = self.adversarial_loss((gen_fake_haze), is_real=True, is_disc=False)
        gen_fake_clean_ganloss = self.adversarial_loss((gen_fake_clean), is_real=True, is_disc=False)
        gen_gan_loss = (gen_fake_clean_ganloss + gen_fake_haze_ganloss) / 2


        # contur loss#
        def compute_contour_loss(self, images):
            # 计算灰度图像

            gray_images = torch.mean(images, dim=1, keepdim=True)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            gray_images = gray_images.to(device)

            # 计算水平和垂直方向的梯度
            gradient_x = F.conv2d(
                gray_images,
                torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
                .unsqueeze(0).unsqueeze(0)
                .to(device)  # 将权重转换为GPU tensor
            )
            gradient_y = F.conv2d(
                gray_images,
                torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
                .unsqueeze(0).unsqueeze(0)
                .to(device)  # 将权重转换为GPU tensor
            )


            # 计算梯度幅值
            gradient_magnitude = torch.sqrt(torch.pow(gradient_x, 2) + torch.pow(gradient_y, 2))

            # 根据梯度幅值计算轮廓损失
            contour_loss = torch.mean(gradient_magnitude)

            return contour_loss
        #contour_loss_c2h2c = compute_contour_loss(clean_images_c2h2c,clean_images)
        contour_loss_h2c = compute_contour_loss(clean_images_h2c,hazy_images)

        #contour_loss = 0.2 * (contour_loss_c2h2c + contour_loss_h2c2h)
        contour_loss= 0.5 * contour_loss_h2c


        #===================================
        r_total_l1_loss = 0
        g_total_l1_loss = 0
        b_total_l1_loss = 0
        # 将图像分成三个通道
        clean_images_r = clean_images[:, 0, :, :]
        clean_images_g = clean_images[:, 1, :, :]
        clean_images_b = clean_images[:, 2, :, :]

        clean_images_h2c_r = hazy_images[:, 0, :, :]
        clean_images_h2c_g = hazy_images[:, 1, :, :]
        clean_images_h2c_b = hazy_images[:, 2, :, :]

        # 计算每个通道的L1损失
        l1_loss_r = self.l1_loss(clean_images_r, clean_images_h2c_r)
        l1_loss_g = self.l1_loss(clean_images_g, clean_images_h2c_g)
        l1_loss_b = self.l1_loss(clean_images_b, clean_images_h2c_b)

        # 将三个通道的损失加起来
        #rgb_total_l1_loss = l1_loss_r + l1_loss_g + l1_loss_b
        r_total_l1_loss += l1_loss_r
        g_total_l1_loss += l1_loss_g
        b_total_l1_loss += l1_loss_b



        #===================================

        ### total loss ###

        gen_loss += self.config.GAN_LOSS_WEIGHT * gen_gan_loss
        gen_loss += self.config.CYCLE_LOSS_WEIGHT * cycle_loss
        gen_loss += self.config.PARA_LOSS_WEIGHT * para_loss
        gen_loss +=contour_loss
        gen_loss +=r_total_l1_loss
        gen_loss +=g_total_l1_loss
        gen_loss +=b_total_l1_loss
        gen_loss += loss_net
        gen_loss.backward()
        self.optimizer.step()

        self.optimizer_depth.zero_grad()
        self.depth_estimator.zero_grad()


        depth_net_loss = 0
        pred_ed_h2c = self.forward_depth(clean_images_h2c.detach())
        depth_loss = self.l1_loss(gt_ed_h2c.detach(), pred_ed_h2c)
        depth_net_loss += depth_loss
        depth_net_loss.backward()

        self.optimizer_depth.step()


        logs = [
            ("g_cyc", cycle_loss.item()),
            ("g_para", para_loss.item()),
            ("g_contur", contour_loss.item()),
            ("g_depth", depth_net_loss.item()),
            ("r_total_l1_loss", r_total_l1_loss.item()),
            ("g_total_l1_loss", g_total_l1_loss.item()),
            ("b_total_l1_loss", b_total_l1_loss.item()),
            ("loss_net", loss_net.item()),
            ("g_gan", gen_gan_loss.item()),
            ("g_total", gen_loss.item()),
            ("d_dis", dis_loss.item()),
            ("lr", self.get_current_lr()),
        ]
        return clean_images_c2h2c, gen_loss, dis_loss, logs


    def get_current_lr(self):
        return self.optimizer.param_groups[0]["lr"]


    def save(self,iteration):

        if self.config.MODEL == 1:
            torch.save({
                'net_h2c':self.net_h2c.state_dict(),
                'net_c2h':self.net_c2h.state_dict(),
                'net_depth':self.depth_estimator.state_dict(),
            },self.gen_weights_path[:-4]+'_'+self.name+str(iteration)+'.pth', _use_new_zipfile_serialization=False)
            torch.save({'discriminator_c2h': self.discriminator_c2h.state_dict(),
                        'discriminator_h2c': self.discriminator_h2c.state_dict(),
                        }, self.gen_weights_path[
                           :-4] + '_' + self.name + '_dis.pth' , _use_new_zipfile_serialization=False)

            torch.save({
                'iteration': self.iteration,
                'epoch': self.epoch,
                'scheduler': self.scheduler.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'optimizer_dis': self.optimizer_dis.state_dict(),
                'optimizer_depth':self.optimizer_depth.state_dict()

            }, self.gen_optimizer_path , _use_new_zipfile_serialization=False)


    def load(self):
        if os.path.exists(self.gen_weights_path[:-4] + '_reconstruct' + '.pth'):
            print('Loading %s weights...' % 'reconstruct')

            if torch.cuda.is_available():
                weights = torch.load(self.gen_weights_path[:-4] + '_reconstruct' + '.pth')
            else:
                weights = torch.load(self.gen_weights_path[:-4] + '_reconstruct' + '.pth',
                                     lambda storage, loc: storage)

            self.net_h2c.load_state_dict(weights['net_h2c'])
            self.net_c2h.load_state_dict(weights['net_c2h'])
            self.depth_estimator.load_state_dict(weights['net_depth'])


            print('Loading %s weights...' % 'reconstruct complete!')

        if os.path.exists(self.gen_weights_path[:-4] + '_' + self.name + '_dis.pth') and self.config.MODE == 1:
            print('Loading discriminator weights...')

            if torch.cuda.is_available():
                weights = torch.load(self.gen_weights_path[:-4] + '_' + self.name + '_dis.pth')
            else:
                weights = torch.load(self.gen_weights_path[:-4] + '_' + self.name + '_dis.pth',
                                     lambda storage, loc: storage)

            self.discriminator_c2h.load_state_dict(weights['discriminator_c2h'])
            self.discriminator_h2c.load_state_dict(weights['discriminator_h2c'])

        if os.path.exists(self.gen_optimizer_path) and self.config.MODE == 1:
            print('Loading %s optimizer...' % self.name)
            if torch.cuda.is_available():
                data = torch.load(self.gen_optimizer_path)
            else:
                data = torch.load(self.gen_optimizer_path, lambda storage, loc: storage)

            self.optimizer.load_state_dict(data['optimizer'])
            self.scheduler.load_state_dict(data['scheduler'])
            self.epoch = data['epoch']
            self.iteration = data['iteration']
            self.optimizer_dis.load_state_dict(data['optimizer_dis'])
            self.optimizer_depth.load_state_dict(data['optimizer_depth'])

    def backward(self, gen_loss):
        gen_loss.backward()
        self.optimizer.step()


    def update_scheduler(self):
        self.scheduler.step()








