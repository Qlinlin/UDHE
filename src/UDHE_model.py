import torch.nn as nn
from torch.optim import Adam
from utils import *
import os
import json
import pytorch_ssim
from UDHE_arch import LightweightRestorationNet as Net
from loss.Perceptual_our import PerceptualLoss
from loss.UCR import UnContrastLoss,mosaic_module

class UDHE(object):


    def __init__(self, params, trainable):
        """Initializes model."""

        self.p = params
        self.trainable = trainable
        self._compile()
        self.top_models = []  # Each element is a tuple (PSNR, model_filename)




    def _compile(self):
        """Compiles model (architecture, loss function, optimizers, etc.)."""




        self.model=Net(32).cuda()
        # Set optimizer and loss, if in training mode
        if self.trainable:
            self.optim = Adam(self.model.parameters(),
                              lr=self.p.learning_rate,
                              betas=self.p.adam[:2],
                              eps=self.p.adam[2])


            self.scheduler =torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optim,T_0=2,T_mult=2) #CosineAnnealingLR





        # CUDA support
        self.L1 = nn.L1Loss()
        self.loss_p = PerceptualLoss()
        self.UCR = UnContrastLoss()
        self.use_cuda = torch.cuda.is_available() and self.p.cuda
        if self.use_cuda:
            self.model = self.model.cuda()
            if self.trainable:
                self.L1 = self.L1.cuda()
                self.loss_p = self.loss_p.cuda()
                self.UCR = self.UCR.cuda()
        self.model = torch.nn.DataParallel(self.model)


    def _print_params(self):
        """Formats parameters to print when training."""

        print('Training parameters: ')
        self.p.cuda = self.use_cuda
        param_dict = vars(self.p)
        pretty = lambda x: x.replace('_', ' ').capitalize()
        print('\n'.join('  {} = {}'.format(pretty(k), str(v)) for k, v in param_dict.items()))
        print()

    def save_model(self, epoch, stats, first=False):
        """Saves model to files; can be overwritten at every epoch to save disk space."""

        # Create directory for model checkpoints, if nonexistent
        if first:
            ckpt_dir_name = f'{datetime.now():{self.p.dataset_name}-%m%d-%H%M}'
            if self.p.ckpt_overwrite:
                ckpt_dir_name = self.p.dataset_name

            self.ckpt_dir = os.path.join(self.p.ckpt_save_path, ckpt_dir_name)
            if not os.path.isdir(self.p.ckpt_save_path):
                os.mkdir(self.p.ckpt_save_path)
            if not os.path.isdir(self.ckpt_dir):
                os.mkdir(self.ckpt_dir)

        current_psnr = stats['valid_psnr'][epoch]  # Get current PSNR
        current_model_filename = f'UDHE-epoch{epoch + 1}-{current_psnr:.5f}.pt'
        current_model_path = os.path.join(self.ckpt_dir, current_model_filename)

        # Always save the current model temporarily
        torch.save(self.model.state_dict(), current_model_path)

        if len(self.top_models) < 10:
            # If fewer than 5 models are saved, just add the current model
            self.top_models.append((current_psnr, current_model_filename))
        else:
            # Find the model with the lowest PSNR
            min_psnr, min_psnr_model_filename = min(self.top_models, key=lambda x: x[0])
            if current_psnr > min_psnr:
                # Replace the model with the lowest PSNR
                self.top_models.remove((min_psnr, min_psnr_model_filename))
                self.top_models.append((current_psnr, current_model_filename))
                # Delete the lowest PSNR model file
                os.remove(os.path.join(self.ckpt_dir, min_psnr_model_filename))
                print(f'Removed: {min_psnr_model_filename}')
            else:
                # If the current model is not in the top 5, delete its file
                os.remove(current_model_path)
                print(f'Removed: {current_model_filename}')

        print(f'Saving model with PSNR {current_psnr:.5f} to: {current_model_filename}')
        # Save stats to JSON
        fname_dict = '{}/UHDFour-stats.json'.format(self.ckpt_dir)
        with open(fname_dict, 'w') as fp:
            json.dump(stats, fp, indent=2)



    def load_model(self, ckpt_fname):
        """Loads model from checkpoint file."""

        print('Loading checkpoint from: {}'.format(ckpt_fname))
        if self.use_cuda:
            self.model.load_state_dict(torch.load(ckpt_fname))
        else:
            self.model.load_state_dict(torch.load(ckpt_fname, map_location='cpu'))


    def _on_epoch_end(self, stats, train_loss, epoch, epoch_start, valid_loader):
        """Tracks and saves starts after each epoch."""
        # import pdb;pdb.set_trace()
        # Evaluate model on validation set
        print('\rTesting model on validation set... ', end='')
        epoch_time = time_elapsed_since(epoch_start)[0]
        valid_loss, valid_time, valid_psnr = self.eval(valid_loader)
        show_on_epoch_end(epoch_time, valid_time, valid_loss, valid_psnr)

        # Decrease learning rate if plateau
        self.scheduler.step(valid_loss)

        # Save checkpoint
        stats['train_loss'].append(train_loss)
        stats['valid_loss'].append(valid_loss)
        stats['valid_psnr'].append(valid_psnr)
        self.save_model(epoch, stats, epoch == 0)

        # Plot stats
        if self.p.plot_stats:
            plot_per_epoch(self.ckpt_dir, 'Valid loss', stats['valid_loss'], 'L1_Loss')
            plot_per_epoch(self.ckpt_dir, 'Valid PSNR', stats['valid_psnr'], 'PSNR (dB)')


    @torch.no_grad()
    def eval(self, valid_loader):

        self.model.train(False)

        valid_start = datetime.now()
        loss_meter = AvgMeter()
        psnr_meter = AvgMeter()

        for batch_idx, ([degrad_name,_],source, target1,) in enumerate(valid_loader):

            if True:
                source = source.cuda()
                #source_down = source_down.cuda()
                target1 = target1.cuda()

            final_result = self.model(source)

            # Update loss
            loss = self.L1(final_result, target1)
            loss_meter.update(loss.item())

            # Compute PSRN
            for i in range(1):
            #import pdb;pdb.set_trace()
                final_result = final_result.cpu()
                target1 = target1.cpu()
                psnr_meter.update(psnr(final_result[i], target1[i]).item())

        valid_loss = loss_meter.avg
        valid_time = time_elapsed_since(valid_start)[0]
        psnr_avg = psnr_meter.avg

        return valid_loss, valid_time, psnr_avg


    def train(self, train_loader, valid_loader):
        """Trains UHDNet on training set."""


        self.model.train(True)
        if self.p.ckpt_load_path is not None:
            print('load pretrained model')
            print(self.p.ckpt_load_path)
            self.model.load_state_dict(torch.load(self.p.ckpt_load_path), strict=False)
            print('The pretrain model is loaded.')
        self._print_params()
        num_batches = len(train_loader)
        assert num_batches % self.p.report_interval == 0, 'Report interval must divide total number of batches'





        # Dictionaries of tracked stats
        stats = {'train_loss': [],
                 'valid_loss': [],
                 'valid_psnr': []}





        # Main training loop
        train_start = datetime.now()
        for epoch in range(self.p.nb_epochs):
            print('EPOCH {:d} / {:d}'.format(epoch + 1, self.p.nb_epochs))


            # Some stats trackers
            epoch_start = datetime.now()
            train_loss_meter = AvgMeter()
            loss_meter = AvgMeter()
            time_meter = AvgMeter()

            # Minibatch SGD
            for batch_idx, ([_,_],source, target) in enumerate(train_loader):
                batch_start = datetime.now()
                progress_bar(batch_idx, num_batches, self.p.report_interval, loss_meter.val)
                #factor=torch.mean(target)/torch.mean(source)
                # import pdb;pdb.set_trace()
                if True:
                    source = source.cuda()
                    target = target.cuda()

                # import pdb;pdb.set_trace()
                final_result = self.model(source)

                # Loss function

                loss_ssim=(1-pytorch_ssim.ssim(final_result, target))
                l1_r = 1*self.L1(final_result[:, 0, :, :], target[:, 0, :, :])
                l1_g = 1.5*self.L1(final_result[:, 1, :, :], target[:, 1, :, :])
                l1_b = 2*self.L1(final_result[:, 2, :, :], target[:, 2, :, :])
                loss_l1 = l1_r+l1_g+l1_b
                loss_p = self.loss_p(final_result,target)
                x3 = mosaic_module(source,16,16)
                UCR = self.UCR(final_result,target,x3)
                loss_final = loss_l1+0.03*loss_p+0.2*loss_ssim+0.1*UCR
                loss_meter.update(loss_final.item())

                # Zero gradients, perform a backward pass, and update the weights
                self.optim.zero_grad()
                loss_final.backward()
                self.optim.step()

                # Report/update statistics
                time_meter.update(time_elapsed_since(batch_start)[1])
                if (batch_idx + 1) % self.p.report_interval == 0 and batch_idx:
                    #show_on_report(batch_idx, num_batches, loss_meter.avg, time_meter.avg)
                    train_loss_meter.update(loss_meter.avg)
                    loss_meter.reset()
                    time_meter.reset()
                #if batch_idx==10:
                #    break

                #print("total", ":", loss_final.item(),  "loss_l1", ":", loss_l1.item(),"loss_ssim", ":", loss_ssim.item())

            self._on_epoch_end(stats, train_loss_meter.avg, epoch, epoch_start, valid_loader)
            train_loss_meter.reset()
        train_elapsed = time_elapsed_since(train_start)[0]
        print('Training done! Total elapsed time: {}\n'.format(train_elapsed))




