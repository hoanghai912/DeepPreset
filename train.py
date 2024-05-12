import torch
import torch.nn as nn
import torchvision
import os
import sys
import time
import numpy as np
import json
from PIL import Image
from writer import LossWriter
from math import log10
import dataset.dataloader as db
import dataset.custom_transform as tr
from torchvision import transforms
from networks.network import get_model
from parameters import get_params
from utils import *
import lpips_models
from torchsummary import summary


# import torch.multiprocessing as mp
# mp.set_start_method('spawn', force=True) 

class Trainer(object):
    def __init__(self, args):

        self.args = self.preprocess_args(args)
        print(self.args)
        self.build_G(self.args)
        # self.load_ckpt(self.args)
        self.build_loader(self.args)
        self.build_preset_handler(self.args)
        self.build_loss()
        self.save_args()

        print("Model Name: " + self.args.model_name)
        self.summary()
    
    def summary(self):
        summary(self.G, [(3,self.args.crop_size[1],self.args.crop_size[0]), (3,self.args.crop_size[1],self.args.crop_size[0])])

    @staticmethod
    def preprocess_args(in_args):
        args = in_args
        if type(args.crop_size) == int:
            args.crop_size = (args.crop_size,args.crop_size)
        if args.d_ckpt is not None:
            ckpt_args = torch.load(args.d_ckpt)['opts']
            for arg_key in ['d_net', 'd_in_channels', 'd_nchannels', 'd_norm']:
                setattr(args, arg_key, getattr(ckpt_args, arg_key))

        if args.g_ckpt is not None:
            ckpt_args = torch.load(args.g_ckpt)['opts']
            args.old_ckpts = ckpt_args.old_ckpts if hasattr(ckpt_args, 'old_ckpts') else []
            args.old_ckpts.append(os.path.basename(args.g_ckpt))
            if args.mode == 'train':
                for arg_key in [
                    'g_depth', 'g_in_channels', 'g_out_channels', 'g_upsampler',
                    'g_downsampler','g_norm'
                    ]:
                    setattr(args, arg_key, getattr(ckpt_args, arg_key))

            elif args.mode in ['resume', 'test']:
                for arg_key in [
                    'model_name', 'g_net', 'g_depth', 'g_in_channels', 'g_out_channels',
                    'g_upsampler', 'g_downsampler', 'g_norm'
                    ]:
                    setattr(args, arg_key, getattr(ckpt_args, arg_key))

        if args.mode == 'train':
            _timestamp = str(time.time()).replace('.', '')
            args.model_name += "{}-{}-{}".format(
                args.trainer_type,
                args.g_net.lower(),
                _timestamp
                )

        args.checkpoint_dir = os.path.join(args.checkpoint_dir, args.model_name)
        args.board_dir = os.path.join(args.board_dir, args.model_name)
        args.val_out = os.path.join(args.val_out, args.model_name)
        if not os.path.exists(args.checkpoint_dir):
            os.makedirs(args.checkpoint_dir)
        if not os.path.exists(args.val_out):
            os.makedirs(args.val_out)
        print(args)

        return args
    
    def reset_grad(self):
        pass

    def set_train_mode(self, flag):
        pass

    def build_preset_handler(self, args):
        self.preset_handler = PresetHandler()

    def build_G(self, args):
        self.mvars = ['G']
        self.G = get_model(args.g_net)(args).cuda()
        # print(list(self.G.modules()))
        self.g_optimizer = torch.optim.Adam(list(self.G.parameters()), args.g_lr, [args.beta1, args.beta2])

    def load_ckpt(self, args):
        if args.g_ckpt is not None:
            args.old_ckpts += [os.path.basename(args.g_ckpt)]
            g_ckpt = torch.load(args.g_ckpt)

            self.G.load_state_dict(g_ckpt['G'])
            self.actual_iters = g_ckpt['iters'] if args.mode != "train" else 0
            print("g_ckpt at: " + str(self.actual_iters))

            # Free
            g_ckpt = None
        else:
            self.actual_iters = 0
        
        if args.d_ckpt is not None:
            d_ckpt = torch.load(args.d_ckpt)
            self.D.load_state_dict(d_ckpt['D'])
            d_ckpt = None

    def build_loader(self, args):
        composed_transforms = transforms.Compose([
            tr.RandomCrop(cropsize=args.crop_size),
            tr.ToTensor(),
            tr.TensorRandomFlip(),
            tr.TensorRandomRotation(size=args.crop_size),
            ])
        train_dataset = db.PhotoSet(args.db_root_dir, random_diff=args.random_diff, mode="train", transform=composed_transforms)
        self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=16, pin_memory=True)

        val_folder = 'test' if args.mode == 'test' else 'val'
        val_dataset = db.PhotoSet(args.db_root_dir, random_diff=0, mode=val_folder, transform=transforms.Compose([tr.ResizeImages(size=args.crop_size), tr.ToTensor()]))
        self.val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.val_batch_size, shuffle=False, num_workers=16, pin_memory=True)

    def build_loss(self):
        pass

    def save_args(self):
        with open('./opts/{}.json'.format(self.args.model_name), 'w') as f:
            json.dump(self.args.__dict__, f, indent=2)
    
    def train_iter(self):
        pass

    def print_train_log(self, iters, i):
        with open('logs/{}_losses.txt'.format(self.args.model_name), 'a') as f:
            f.write(
                "[Iter {}][Train] Losses at {}/{}: {}\n".format(
                    iters+1,
                    i,
                    self.args.iters_interval, #len(self.train_loader),
                    ','.join([str(k) for k in np.around(self.writer.get_mean_losses(), 6).tolist()])
                    )
                )

    def train(self):
        max_psnrs = 0
        min_p = 999
        iters = self.actual_iters

        for epoch in range(self.args.epochs):
            self.set_train_mode(True)
            for i, data_sample in enumerate(self.train_loader):

                self.train_iter(data_sample)
                self.print_train_log(iters, self.writer.nof_samples)

                if self.writer.nof_samples % self.args.iters_interval == 0:
                    self.writer.finish(iters)

                    val_psnrs, val_p = self.validate(iters)
                    self.writer.add_scalar('val_simulating_lr', val_psnrs, iters + 1)
                    self.writer.add_scalar('val_learning_preset', val_p, iters + 1)

                    if max_psnrs < val_psnrs or min_p > val_p:
                        max_psnrs = val_psnrs if max_psnrs < val_psnrs else max_psnrs
                        min_p = val_p if min_p > val_p else min_p
                        print("Found psnr_s={} (max: {}), min_p={} (min: {}) at {}x{}".format(round(val_psnrs,2),round(max_psnrs,2),round(val_p,6),round(min_p,6), iters + 1, self.args.iters_interval))
                    self.save_ckpts(iters)

                    # reset
                    iters += 1

        print('Train Completed')
        self.writer.close()

    def save_ckpts(self, iters):
        for k in self.mvars:
            torch.save(
                {
                    'iters': iters + 1,
                    k : getattr(self, k).state_dict(),
                    'opts': self.args
                },
                os.path.join(self.args.checkpoint_dir, "{}_{}_{}x{}.pth.tar".format(self.args.model_name, k, iters + 1, self.args.iters_interval))
            )

    def validate(self, iters):
        self.set_train_mode(False)
        with torch.no_grad():
            avg_spsnr = 0
            avg_perror = 0
            for i, data_sample in enumerate(self.val_loader):

                reference = torch.autograd.Variable(data_sample['reference']).cuda()
                img = torch.autograd.Variable(data_sample['img']).cuda()
                gth_img = torch.autograd.Variable(data_sample['gth_img']).cuda()
                gth_preset = torch.autograd.Variable(data_sample['gth_preset']).cuda()

                pairs = data_sample['pairs']

                _stime = time.time()
                retouched_img, preset, _ = self.G(img, reference) if self.args.mode != 'test' else self.G.stylize(img, reference)
                loss_s = self.criterionMSE((retouched_img+1)/2, (gth_img+1)/2)
                loss_p = self.criterionL1(preset, gth_preset).item() if preset is not None else 0

                psnr = 10 * log10(1 / loss_s.item())
                avg_spsnr += psnr
                avg_perror += loss_p

                for j, refer_name in enumerate(pairs[1]):
                    if refer_name in ["0862.jpg", "0803.jpg"] and pairs[2][j] in ["501"]:
                        valout = (retouched_img[j,:,:,:] + 1)/2
                        valout = np.array(valout.clamp(0,1).cpu().numpy().transpose(1,2,0) * 255.0, dtype=np.uint8)
                        strs = [pairs[k][j] for k in [0,1,2]] + [str(iters+1)]
                        img_dir = os.path.join(self.args.val_out, '-'.join(strs).replace('.jpg', '').replace('IMG_', '') + '.png')
                        Image.fromarray(valout).save(img_dir)
                        if preset is not None:
                            self.preset_handler.save_numpy_preset(img_dir.replace('.png', '.json'), preset[j].cpu().numpy())

            avg_spsnr = avg_spsnr/len(self.val_loader)
            avg_perror = avg_perror/len(self.val_loader)

        return avg_spsnr, avg_perror

class Trainer_LPIPS(Trainer):
    def __init__(self, args):
        super().__init__(args)
        self.writer = LossWriter(
            self.args.board_dir,
            board_names=['total', 'learning_preset', 'simulating_lr' , 'lpips', 'pos_pairwise']
            )
        self.load_ckpt(self.args)

    def reset_grad(self):
        self.g_optimizer.zero_grad()

    def set_train_mode(self, _flag):
        if _flag:
            self.G.train()
        else:
            self.G.eval()

    def build_loss(self):
        self.criterionL1 = nn.L1Loss().cuda()
        self.criterionMSE = nn.MSELoss().cuda()
        self.criterionLPIPS = lpips_models.PerceptualLoss(model='net-lin', net='alex', use_gpu=True)

    def train_iter(self, data_sample):
        reference = torch.autograd.Variable(data_sample['reference']).cuda()
        img = torch.autograd.Variable(data_sample['img']).cuda()
        real_images = torch.autograd.Variable(data_sample['gth_img']).cuda()
        gth_preset = torch.autograd.Variable(data_sample['gth_preset']).cuda()
        positive_reference = torch.autograd.Variable(data_sample['positive_reference']).cuda()


        retouched_images, preset, preset_emb = self.G(img, reference)
        _, _, positive_ref_emb = self.G.estimate_preset(img, positive_reference)


        loss_p = self.criterionL1(preset, gth_preset)
        loss_s = self.criterionMSE(retouched_images, real_images)
        loss_pos = self.criterionL1(preset_emb, positive_ref_emb)
        loss_lpips = torch.mean(self.criterionLPIPS.forward(retouched_images, real_images, normalize=False))

        loss = self.args.p_weight*loss_p + self.args.s_weight*loss_s + self.args.lpips_weight*loss_lpips + self.args.pw_weight*loss_pos
        self.writer.update([
            loss.item(),
            loss_p.item(),
            loss_s.item(),
            loss_lpips.item(),
            loss_pos.item(),
        ])

        self.reset_grad()
        loss.backward()
        self.g_optimizer.step()

def get_trainer(_name):
    return {
        'lpips': Trainer_LPIPS,
    }[_name]

if __name__  == "__main__":
    args = get_params()
    print(args)
    trainer = get_trainer(args.trainer_type)(args)
    if args.mode != 'test':
        trainer.train()
    else:
        print(trainer.args)
        psnr_img, l1_preset = trainer.validate(trainer.actual_iters)
        print('PSNR: {}, Preset L1 Dist: {}'.format(psnr_img, l1_preset))

# main()