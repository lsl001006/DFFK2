from dataset import cifar
import engine  

from mosaic_core import registry 
import numpy as np 
import config
import torch 
import os 
import logging
import torch.nn as nn
from utils.utils import DataIter 
from torch.utils.data import DataLoader
import torch.optim as optim
import utils.utils as utils
import copy

from tqdm import tqdm

class OneTeacher:
    def __init__(self, student, teacher, generator, discriminator, args,writer):
        self.writer=writer
        self.args = args
        self.val_loader, self.priv_data,  self.local_datanum, self.local_cls_datanum = self.gen_dataset(args)
        self.local_datanum = torch.FloatTensor(self.local_datanum).cuda()
        self.local_cls_datanum = torch.FloatTensor(self.local_cls_datanum).cuda()
        self.netG = generator
        self.normalizer = utils.Normalizer(args.dataset)
        self.netDS = utils.copy_parties(config.N_PARTIES, discriminator)
        #teacher/student
        self.netS = student
        self.init_netTS(teacher)
    
    def gen_dataset(self,args):
        num_classes, ori_training_dataset, val_dataset = registry.get_dataset(name=args.dataset, data_root=args.data_root)
        _, train_dataset, _ = registry.get_dataset(name=args.unlabeled, data_root=args.data_root)
        # _, ood_dataset, _ = registry.get_dataset(name=args.unlabeled, data_root=args.data_root)
        # see Appendix Sec 2, ood data is also used for training
        # ood_dataset.transforms = ood_dataset.transform = train_dataset.transform # w/o augmentation
        train_dataset.transforms = train_dataset.transform = val_dataset.transform # w/ augmentation
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
             shuffle=False,
            batch_size=config.DIS_BATCHSIZE,  num_workers=config.NUM_WORKERS)
        # return train_dataset,ood_dataset,num_classes,ori_training_dataset,val_dataset
        local_datanum = np.zeros(config.N_PARTIES)
        local_cls_datanum = np.zeros((config.N_PARTIES, args.N_class))
        for localid in range(config.N_PARTIES):
            #count
            local_datanum[localid] = 50000
            #class specific count
            for cls in range(args.N_class):
                local_cls_datanum[localid, cls] = 5000
        return val_loader,[train_dataset], local_datanum, local_cls_datanum 
    
    def init_netTS(self,teacher):
        epochs = config.INIT_EPOCHS
        self.netTS = utils.copy_parties(config.N_PARTIES, teacher) #can be different
        ckpt_dir = f'{config.LOCAL_CKPTPATH}/{self.args.dataset}/a{self.args.alpha}+sd{self.args.seed}+e{epochs}+b{config.BATCHSIZE}' 
        if not os.path.isdir(ckpt_dir):
            os.mkdir(ckpt_dir)
         
    def init_training(self):
        self.local_dataloader = []
        self.local_ood_dataloader=[]
        for n in range(config.N_PARTIES):
            tr_dataset = self.priv_data[n]
            local_loader = DataLoader(
                        dataset=tr_dataset, batch_size=config.DIS_BATCHSIZE, shuffle=True, num_workers=config.NUM_WORKERS, sampler=None) 
            self.local_dataloader.append(DataIter(local_loader))
             

        self.iters_per_round = min([len(local_loader.dataloader) for local_loader in self.local_dataloader])#4~9
        # import ipdb; ipdb.set_trace()
        steps_all = self.args.epochs*self.iters_per_round
        #init optim
        if config.OPTIMIZER == 'SGD':
            self.optim_s = optim.SGD(
                self.netS.parameters(), lr=config.DIS_LR, momentum=0.9, weight_decay=config.DIS_WD)
        else:    
            self.optim_s = optim.Adam(
                self.netS.parameters(), lr=config.DIS_LR,  betas=(0.9, 0.999), weight_decay=config.DIS_WD)
        self.sched_s = optim.lr_scheduler.CosineAnnealingLR(
                self.optim_s, steps_all)#TODO, eta_min=config.DIS_LR_MIN,
        #for gen
        self.optim_g = torch.optim.Adam(self.netG.parameters(), lr=config.GEN_LR, betas=[0.5, 0.999])
        self.sched_g = torch.optim.lr_scheduler.CosineAnnealingLR(self.optim_g, T_max=steps_all )
        param_ds = []
        for n in range(config.N_PARTIES):
            param_ds += list(self.netDS[n].parameters()) 
        self.optim_d = torch.optim.Adam(param_ds, lr=config.GEN_LR, betas=[0.5, 0.999])
        self.sched_d = torch.optim.lr_scheduler.CosineAnnealingLR(self.optim_d, T_max=steps_all ) 
        ##TODO: not all complete steps_all if local_percetnt<1
        #
        # import ipdb; ipdb.set_trace()

        #criterion
        # self.criterion_distill = nn.L1Loss(reduce='mean')
        self.criterion=nn.CrossEntropyLoss().cuda()
        self.criterion_bce = nn.functional.binary_cross_entropy_with_logits

        #netS training records
        self.bestacc = 0
        self.best_statdict = copy.deepcopy(self.netS.state_dict())
        #save path
        self.savedir = f'{config.CKPTPATH}/{self.args.logfile}'
        self.savedir_gen = f'{config.CKPTPATH}/{self.args.logfile}/gen'
        if not os.path.isdir(self.savedir_gen):
            os.makedirs(self.savedir_gen)
            
    def update(self):
        self.init_training()
        self.global_step = 0
        #default: local_percent =1
        selectN = list(range(0, config.N_PARTIES))
        self.localweight = self.local_datanum/self.local_datanum.sum()#nlocal*nclass
        self.localclsweight = self.local_cls_datanum/self.local_cls_datanum.sum(dim=0)#nlocal*nclass
        # resume trainning if args.resume exists
        self.resume_trainning()
        for round in range(self.args.start_epoch, self.args.epochs): 
            logging.info(f'************Start Round {round} -->> {self.args.epochs}***************')
            self.update_round(round, selectN)
        #save G,D
        torch.save(self.netG.state_dict(), f'{self.savedir_gen}/generator.pt') 
        for n in range(config.N_PARTIES):
             torch.save(self.netDS[n].state_dict(), f'{self.savedir_gen}/discrim{n}.pt')

    def resume_trainning(self):
        ############################################
        # Resume Train from checkpoints
        ############################################
        if self.args.resume:
            if os.path.isfile(self.args.resume):
                print("=> loading checkpoint '{}'".format(self.args.resume))
                if self.args.gpu is None:
                    checkpoint = torch.load(self.args.resume)
                else:
                    # Map model to be loaded to specified single gpu.
                    loc = 'cuda:{}'.format(self.args.gpu)
                    checkpoint = torch.load(self.args.resume, map_location=loc)
                
                try:
                    self.netS.module.load_state_dict(checkpoint['s_state_dict'])
                    self.netG.module.load_state_dict(checkpoint['g_state_dict'])
                    for n in range(config.N_PARTIES):
                        self.netDS[n].module.load_state_dict(checkpoint['ds_state_dict'][n])
                except:
                    self.netS.load_state_dict(checkpoint['s_state_dict'])
                    self.netG.load_state_dict(checkpoint['g_state_dict'])
                    for n in range(config.N_PARTIES):
                        self.netDS[n].load_state_dict(checkpoint['ds_state_dict'][n])
                best_acc1 = checkpoint['best_acc']
                try: 
                    self.args.start_epoch = checkpoint['epoch']
                    self.optim_g.load_state_dict(checkpoint['optim_g'])
                    self.sched_g.load_state_dict(checkpoint['sched_g'])
                    self.optim_s.load_state_dict(checkpoint['optim_s'])
                    self.sched_s.load_state_dict(checkpoint['sched_s'])
                    self.sched_d.load_state_dict(checkpoint['sched_d'])
                    self.optim_d.load_state_dict(checkpoint['optim_d'])
                except: 
                    print("Fails to load additional model information")
                print("[!] loaded checkpoint '{}' (epoch {} acc {})"
                    .format(self.args.resume, checkpoint['epoch'], best_acc1))
            else:
                print("[!] no checkpoint found at '{}'".format(self.args.resume)) 
             
             
    def change_status_to_train(self):
        
        self.netG.train() 
        for n in range(config.N_PARTIES):
            self.netDS[n].train() 
            self.netTS[n].eval()
        self.netS.train() 
        
    def update_round(self, roundd, selectN):
        #
        self.change_status_to_train()
        bestacc_round = self.bestacc
        for iter in tqdm(range(self.iters_per_round)):
            #1. update D,G
            z = torch.randn(size=(config.DIS_BATCHSIZE, config.GEN_Z_DIM)).cuda()
            syn_img = self.netG(z)
            syn_img = self.normalizer(syn_img)
            self.update_netDS_batch(syn_img, selectN)
            self.update_netG_batch(syn_img, selectN)
            #2. Distill, update S
            self.update_netS_batch(selectN)
            
            self.sched_d.step()
            self.sched_g.step()
            self.sched_s.step()
            #val
        acc = validate(self.optim_s.param_groups[0]['lr'], self.val_loader, self.netS, self.criterion, self.args,roundd)
        self.global_step += 1
        is_best = False
        if acc>bestacc_round:
            logging.info(f'Iter{iter}, best for now:{acc}')
            self.best_statdict = copy.deepcopy(self.netS.state_dict())
            bestacc_round = acc
            is_best = True
            # if iter % config.PRINT_FREQ == 0:
            #     logging.info(f'===R{roundd}, {iter}/{self.iters_per_round}, acc{acc}, best{self.bestacc}')
                
        #reload G,D?
        # self.netS.load_state_dict(self.best_statdict, strict=True)
        logging.info(f'=============Round{roundd}, BestAcc originate: {(self.bestacc):.2f}, to{(bestacc_round):.2f}====================')
        
        if bestacc_round>self.bestacc:
            savename = os.path.join(self.savedir, f'r{roundd}_{(bestacc_round):.2f}.pt')
            torch.save(self.best_statdict, savename)
            self.bestacc = bestacc_round
        
        if self.writer is not None:
            self.writer.add_scalar('BestACC', self.bestacc, roundd)
        
        checkpoints = {
            "epoch":roundd+1,
            "arch":self.args.student,
            "s_state_dict":self.best_statdict,
            "g_state_dict":self.netG.state_dict(),
            "ds_state_dict":[self.netDS[i].state_dict() for i in range(config.N_PARTIES)],
            "best_acc":float(bestacc_round),
            "optim_s":self.optim_s.state_dict(),
            "optim_g":self.optim_g.state_dict(),
            "optim_d":self.optim_d.state_dict(),
            "sched_d":self.sched_d.state_dict(),
            "sched_s":self.sched_s.state_dict(),
            "sched_g":self.sched_g.state_dict()
        }
        save_checkpoint(checkpoints, 
                        is_best, 
                        self.args.ckpt_path, 
                        filename=f'E{roundd}-Dbs{config.DIS_BATCHSIZE}-Ts{config.N_PARTIES}-ACC{round(float(self.bestacc),2)}.pth')
        
            
    def update_netDS_batch(self, syn_img, selectN):
        loss = 0.
        for localid in selectN:
            d_out_fake = self.netDS[localid](syn_img.detach())
            real_img = self.local_dataloader[localid].next()[0].cuda()#list [img, label, ?]
            d_out_real = self.netDS[localid](real_img.detach())
            loss_d = (self.criterion_bce(d_out_fake, torch.zeros_like(d_out_fake), reduction='sum') + \
                self.criterion_bce(d_out_real, torch.ones_like(d_out_real), reduction='sum'))/  (2*len(d_out_fake))  
            loss += loss_d
        loss *= self.args.w_disc

        self.optim_d.zero_grad()
        loss.backward()
        self.optim_d.step()
        
        if self.writer is not None:
            self.writer.add_scalar('LossDiscriminator', loss.item(), self.global_step)
            
    def update_netG_batch(self, syn_img, selectN):
        #1. gan loss
        loss_gan = []
        for localid in selectN:
            d_out_fake = self.netDS[localid](syn_img)#gradients in syn
            loss_gan.append( self.criterion_bce(d_out_fake, torch.ones_like(d_out_fake), reduction='sum') / len(d_out_fake))
        if self.args.is_emsember_generator_GAN_loss=="y":
            loss_gan= self.ensemble_locals(torch.stack(loss_gan))
        else:
            loss_gan= torch.sum(torch.stack(loss_gan ))
        
        #2. adversarial distill loss
        logits_T = self.forward_teacher_outs(syn_img, selectN)
        ensemble_logits_T = self.ensemble_locals(logits_T)
        logits_S = self.netS(syn_img)
        loss_adv = -engine.criterions.kldiv(logits_S, ensemble_logits_T)  # - self.criterion_distill(logits_S, ensemble_logits_T) #(bs, ncls) 
        #3.regularization for each t_out (not ensembled) #TO DISCUSS
        loss_align = []
        loss_balance = []
        for n in range(len(selectN)):
            t_out = logits_T[n]
            pyx = torch.nn.functional.softmax(t_out, dim = 1) # p(y|G(z)
            log_softmax_pyx = torch.nn.functional.log_softmax(t_out, dim=1)
            py = pyx.mean(0)
            loss_align.append( -(pyx * log_softmax_pyx).sum(1).mean()) #To generate distinguishable imgs
            loss_balance.append( (py * torch.log2(py)).sum()) #Alleviating Mode Collapse for unconditional GAN
        
        loss_align = self.ensemble_locals(torch.stack(loss_align))
        loss_balance = self.ensemble_locals(torch.stack(loss_balance))

        loss_gan = self.args.w_gan*loss_gan
        loss_adv = self.args.w_adv*loss_adv
        loss_align = self.args.w_algn*loss_align
        loss_balance = self.args.w_baln*loss_balance

        # Final loss: L_align + L_local + L_adv (DRO) + L_balance
        loss = loss_gan + loss_adv + loss_align + loss_balance

        self.optim_g.zero_grad()
        loss.backward()
        self.optim_g.step()
        
        if self.writer is not None:
            self.writer.add_scalars('LossGen', {'loss_gan': loss_gan.item(), 
                                                'loss_adv': loss_adv.item(), 
                                                'loss_align': loss_align.item(), 
                                                'loss_balance': loss_balance.item()}, self.global_step)


    def forward_teacher_outs(self, images, localN=None):
        if localN is None: #use central as teacher
            total_logits = self.netS(images).detach()
        else: #update student
            #get local
            total_logits = []
            for n in localN:
                logits=self.netTS[n](images)
                total_logits.append(logits)
            total_logits = torch.stack(total_logits) #nlocal*batch*ncls       
        return total_logits
    
    def  ensemble_locals(self, locals):
        """
        locals: (nlocal, batch, ncls) or (nlocal, batch/ncls) or (nlocal)
        """
        if len(locals.shape)==3:
            localweight = self.localclsweight.unsqueeze(dim=1)#nlocal*1*ncls
            ensembled = (locals*localweight).sum(dim=0) #batch*ncls
        elif len(locals.shape)==2:
            localweight = self.localweight[:,None]#nlocal*1
            ensembled = (locals*localweight).sum(dim=0) #batch/ncls
        elif len(locals.shape)==1:
            ensembled = (locals*self.localweight).sum() #1
        return ensembled


    def update_netS_batch(self, selectN):
        for _ in range(5): 
            with torch.no_grad():
                z = torch.randn(size=(config.DIS_BATCHSIZE, config.GEN_Z_DIM)).cuda()
                syn_img = self.netG(z)
                syn_img = self.normalizer(syn_img)
                logits_T = self.ensemble_locals(self.forward_teacher_outs(syn_img, selectN))
                 
            #
            loigts_S = self.netS(syn_img.detach())
            
            # loss =   self.criterion_distill(loigts_S, logits_T) #
            loss=engine.criterions.kldiv(loigts_S, logits_T.detach() )
            loss *= self.args.w_dist

            self.optim_s.zero_grad()
            loss.backward()
            self.optim_s.step()
            

            if self.writer is not None:
                self.writer.add_scalar('LossDistill', loss.item(), self.global_step)


def save_checkpoint(state, is_best, save_dir, filename='checkpoint.pth'):
    if is_best:
        torch.save(state, os.path.join(save_dir, filename))  
        print(f'[saved] ckpt saved to {os.path.join(save_dir, filename)}')

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res    

def validate(current_lr, val_loader, model, criterion, args,current_epoch):
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    model.eval()
    with torch.no_grad():
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            if torch.cuda.is_available():
                target = target.cuda(args.gpu, non_blocking=True)
            output = model(images)
            loss = criterion(output, target)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))
        if args.rank<=0:
            args.logger.info(' [Eval] Epoch={current_epoch} Acc@1={top1.avg:.4f} Acc@5={top5.avg:.4f} Loss={losses.avg:.4f} Lr={lr:.4f}'
                .format(current_epoch= current_epoch, top1=top1, top5=top5, losses=losses, lr=current_lr))
    return top1.avg
    
    

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

    
class MultiTeacher(OneTeacher):
    # def __init__(self,  args):
    #     pass 
    
    def gen_dataset(self,args):
        
        priv_train_data,ori_training_dataset, test_dataset = cifar.dirichlet_datasplit(
            args, privtype=args.dataset)
        local_dataset = []
        for n in range(config.N_PARTIES):
            tr_dataset  = cifar.Dataset_fromarray(priv_train_data[n]['x'], priv_train_data[n]['y'], train=True, verbose=False)
            local_dataset.append(tr_dataset)
        
        num_classes, ori_training_dataset, val_dataset = registry.get_dataset(name=args.dataset, data_root=args.data_root)
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
             shuffle=False,
            batch_size=config.DIS_BATCHSIZE,  num_workers=config.NUM_WORKERS)
        local_datanum = np.zeros(config.N_PARTIES)
        local_cls_datanum = np.zeros((config.N_PARTIES, self.args.N_class))
        for localid in range(config.N_PARTIES):
            #count
            local_datanum[localid] = priv_train_data[localid]['x'].shape[0]
            #class specific count
            for cls in range(args.N_class):
                local_cls_datanum[localid, cls] = (priv_train_data[localid]['y']==cls).sum()
            
            
        assert sum(local_datanum) == 50000
        assert sum(sum(local_cls_datanum)) == 50000
        return val_loader,local_dataset, local_datanum, local_cls_datanum 
        
         