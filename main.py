import argparse
import os
from tensorboardX import SummaryWriter
import logging
from datetime import datetime
import torch 
import models.resnet8 as resnet8
from models.generator import *
import utils.fedkd as fedkd 
import config
from utils import registry
import copy
import random
import torch.backends.cudnn as cudnn
import warnings
def get_args():
    parser = argparse.ArgumentParser()
    # path
    parser.add_argument('--dataset', type=str, default = 'cifar10')
    parser.add_argument('--data_root', default='data')
    parser.add_argument('--unlabeled', default='cifar10')
    parser.add_argument('--N_class', type=int, default = 10)#10
    parser.add_argument('--logfile', default='', type=str)
    # training
    parser.add_argument('--gpu', type=str, default ='0')  
    parser.add_argument('--teacher', type=str, default ='wrn40_2') #TODO resnet8
    parser.add_argument('--student', type=str, default ='wrn16_1')  #TODO resnet8
    # parser.add_argument('--mode', type=str, default ='train_with_pretrained_teacher')  #train_from_scratch
    # parser.add_argument('--training_arguments_enum', type=str, default ='Data_Free_KD')  
    # parser.add_argument('--betas_min', type=float, default =0.5)  
    # fed choice
    parser.add_argument('--alpha', type=float, default = 1.0)
    parser.add_argument('--seed', type=int, default = 20220819)
    #loss weight
    parser.add_argument('--w_disc', type=float, default = 1.0)
    parser.add_argument('--w_gan', type=float, default = 1.0)
    parser.add_argument('--w_adv', type=float, default = 1.0)
    parser.add_argument('--w_algn', type=float, default = 1.0)
    parser.add_argument('--w_baln', type=float, default = 10.0)    
    parser.add_argument('--w_dist', type=float, default = 1.0)
    return parser.parse_args()


def setup_modal(args):
    if args.teacher=="resnet8":
        student = resnet8.ResNet8(num_classes=args.N_class).cuda()
        teacher=copy.deepcopy(student)
    else:
        student = registry.get_model(args.student, num_classes=args.N_class).cuda()
        teacher = registry.get_model(args.teacher, num_classes=args.N_class, pretrained=True).cuda()
        # teacher.load_state_dict(torch.load('ckpt/pretrained/%s_%s.pth'%(args.dataset, args.teacher), map_location='cpu')['state_dict'])
    return teacher, student

if __name__ == "__main__":
    args = get_args()
    os.environ['CUDA_VISIBLE_DEVICES']=args.gpu
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')
    handlers = [logging.StreamHandler()]
    if not os.path.isdir('./logs'):
        os.mkdir('./logs')
    if args.logfile:
        args.logfile = f'{datetime.now().strftime("%m%d%H%M")}'+args.logfile
        writer = SummaryWriter(comment=args.logfile)
        handlers.append(logging.FileHandler(f'./logs/{args.logfile}.txt', mode='a'))
    else:
        args.logfile = 'debug'
        writer = None
        handlers.append(logging.FileHandler(f'./logs/debug.txt', mode='a'))

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s', handlers=handlers,     
    )
    logging.info(args)
    for key in dir(config):
        if not key.startswith('_'):
            value = getattr(config, key)
            logging.info(f'{key}:{value}' )

    # 1. Model
    logging.info("CREATE MODELS.......")
    gpu = [int(i) for i in range(torch.cuda.device_count())]
    logging.info(f'GPU: {args.gpu}')
    
    teacher,student=setup_modal(args)
     
    generator = Generator(nz=config.GEN_Z_DIM).cuda()
    discriminator = PatchDiscriminator().cuda()#TODO
    if len(gpu)>1:
        student = torch.nn.DataParallel(student, device_ids=gpu)
        generator = torch.nn.DataParallel(generator, device_ids=gpu)
        discriminator = torch.nn.DataParallel(discriminator, device_ids=gpu)

    # import torchsummary
    # torchsummary.summary(model, (3, 32, 32))
    # print('parameters_count:',count_parameters(model))
    # logging.info("totally {} paramerters".format(sum(x.numel() for x in model.parameters())))
    # logging.info("Param size {}".format(np.sum(np.prod(x.size()) for name,x in model.named_parameters() if 'linear2' not in name)))
    # import ipdb; ipdb.set_trace()
    
    # 2. fed
    fed = fedkd.myFed(student,teacher, generator, discriminator, writer, args)
    fed.update()

    if writer is not None:
        writer.close()

