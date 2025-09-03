import argparse
import os
import random 
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import pdb
import wandb
from datetime import datetime, timezone, timedelta

from dataset.CramedDataset import CramedDataset
from models.basic_model import AVClassifier, UNIMODALClassifier
from utils.utils import setup_seed, weight_init

os.environ.setdefault("WANDB_CONSOLE", "wrap")

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='CREMAD', type=str,
                        help='CREMAD')
    parser.add_argument('--modulation', default='Normal', type=str,

                        choices=['Normal', 'OGM', 'OGM_GE'])
    parser.add_argument('--fusion_method', default='concat', type=str,
                        choices=['sum', 'concat']) #gated, film은 잠시 버려
    parser.add_argument('--fps', default=1, type=int)
    parser.add_argument('--use_video_frames', default=3, type=int)
    parser.add_argument('--audio_path', default='/workspace/datasets/CREMA-D/AudioWAV', type=str)
    parser.add_argument('--visual_path', default='/workspace/datasets/CREMA-D/', type=str)

    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=200, type=int)

    parser.add_argument('--optimizer', default='sgd', type=str, choices=['sgd', 'adam'])
    parser.add_argument('--learning_rate', default=0.001, type=float, help='initial learning rate')
    parser.add_argument('--lr_decay_step', default=70, type=int, help='where learning rate decays')
    parser.add_argument('--lr_decay_ratio', default=0.1, type=float, help='decay coefficient')

    parser.add_argument('--modulation_starts', default=0, type=int, help='where modulation begins')
    parser.add_argument('--modulation_ends', default=1, type=int, help='where modulation ends')
    parser.add_argument('--alpha', default=None, type=float, help='alpha in OGM-GE')

    parser.add_argument('--ckpt_path', type=str, default="/workspace/emotion_KD250628/MKD/CREMA_D_MKD/MKD/ckpt", help='path to save trained models')
    parser.add_argument('--train', action='store_true', required=True, help='turn on train mode')

    # parser.add_argument('--use_tensorboard', default=False, type=bool, help='whether to visualize')
    # parser.add_argument('--tensorboard_path', type=str, help='path to save tensorboard logs')

    # parser.add_argument('--random_seed', default=0, type=int)
    parser.add_argument('--gpu_ids', default='0, 1', type=str, help='GPU ids')
    parser.add_argument("--small_dataset", action='store_true', help="use a small subset for debugging")
    
    parser.add_argument("--unimodal", action='store_true')
    parser.add_argument("--unimodal_modality", type=str, choices=['audio', 'visual'], help="which modality to use in unimodal training")

    return parser.parse_args()



# Modulation starts here !
def ogm_modulation(args, epoch, model, out_a, out_v, label):
    softmax = nn.Softmax(dim=1)
    relu = nn.ReLU(inplace=True)
    tanh = nn.Tanh()
    score_v = sum([softmax(out_v)[i][label[i]] for i in range(out_v.size(0))])
    score_a = sum([softmax(out_a)[i][label[i]] for i in range(out_a.size(0))])

    ratio_v = score_v / score_a
    ratio_a = 1 / ratio_v

    """
    Below is the Eq.(10) in our CVPR paper:
            1 - tanh(alpha * rho_t_u), if rho_t_u > 1
    k_t_u =
            1,                         else
    coeff_u is k_t_u, where t means iteration steps and u is modality indicator, either a or v.
    """

    if ratio_v > 1:
        coeff_v = 1 - tanh(args.alpha * relu(ratio_v))
        coeff_a = 1
    else:
        coeff_a = 1 - tanh(args.alpha * relu(ratio_a))
        coeff_v = 1



    if args.modulation_starts <= epoch <= args.modulation_ends: # bug fixed
        for name, parms in model.named_parameters():
            layer = str(name).split('.')[1]
            # last_name = str(name).split('.')[-2]
            # print(name)
            if 'audio' in layer and len(parms.grad.size()) == 4:
                
                if args.modulation == 'OGM_GE':  # bug fixed
                    parms.grad = parms.grad * coeff_a + \
                                torch.zeros_like(parms.grad).normal_(0, parms.grad.std().item() + 1e-8)
                elif args.modulation == 'OGM':
                    parms.grad *= coeff_a

            if 'visual' in layer and len(parms.grad.size()) == 4:
                if args.modulation == 'OGM_GE':  # bug fixed
                    parms.grad = parms.grad * coeff_v + \
                                torch.zeros_like(parms.grad).normal_(0, parms.grad.std().item() + 1e-8)
                elif args.modulation == 'OGM':
                    parms.grad *= coeff_v
        return coeff_a, coeff_v
    else:
        return None, None


def train_epoch(args, epoch, model, device, dataloader, optimizer, scheduler, global_step=0):
    criterion = nn.CrossEntropyLoss()
    softmax = nn.Softmax(dim=1)
    relu = nn.ReLU(inplace=True)
    tanh = nn.Tanh()

    model.train()
    
    
    
    print("Start training ... ")

    _loss = 0
    _loss_a = 0
    _loss_v = 0

    for step, (spec, image, label) in enumerate(dataloader):

        #pdb.set_trace()
        spec = spec.to(device)
        image = image.to(device)
        label = label.to(device)

        optimizer.zero_grad()

        # TODO: make it simpler and easier to extend
        
        if args.unimodal:
            feature, out= model(spec.unsqueeze(1).float(), image.float())
            loss = criterion(out, label)
    
            loss.backward()
            _loss += loss.item()
            optimizer.step()
            
            wandb.log({
                'train/loss': loss.item(),
                'train/epoch': epoch,
                'train/iter_in_epoch': step,
                'train/global_step': global_step,
                'train/lr': optimizer.param_groups[0]['lr'],
            }, step=global_step)

            
        else:
            a, v, out, out_a, out_v = model(spec.unsqueeze(1).float(), image.float())
            if args.fusion_method == 'sum':
                out_v = (torch.mm(v, torch.transpose(model.module.fusion_module.fc_y.weight, 0, 1)) +
                        model.module.fusion_module.fc_y.bias)
                out_a = (torch.mm(a, torch.transpose(model.module.fusion_module.fc_x.weight, 0, 1)) +
                        model.module.fusion_module.fc_x.bias)
            elif args.fusion_method == 'concat':
                weight_size = model.module.fusion_module.fc_out.weight.size(1)
                out_v = (torch.mm(v, torch.transpose(model.module.fusion_module.fc_out.weight[:, weight_size // 2:], 0, 1))
                        + model.module.fusion_module.fc_out.bias / 2)

                out_a = (torch.mm(a, torch.transpose(model.module.fusion_module.fc_out.weight[:, :weight_size // 2], 0, 1))
                        + model.module.fusion_module.fc_out.bias / 2)

            loss = criterion(out, label)
            loss_v = criterion(out_v, label)
            loss_a = criterion(out_a, label)
            loss.backward()
            ###############################################################################
            ############### OGM - GE 코드 작성 부분 ########################################
            ###############################################################################
            if args.modulation == 'Normal':
                # no modulation, regular optimization
                pass
            else:
                # Modulation starts here !
                coeff_a, coeff_v = ogm_modulation(args, epoch, model, out_a, out_v, label)


            

            _loss += loss.item()
            _loss_a += loss_a.item()
            _loss_v += loss_v.item()
            
            
            optimizer.step()
            if args.modulation in ("OGM", "OGM_GE"):
                log_dict = {
                    'train/loss': loss.item(),
                    'train/loss_a': loss_a.item(),
                    'train/loss_v': loss_v.item(),
                    'train/epoch': epoch,
                    'train/iter_in_epoch': step,
                    'train/global_step': global_step,
                    'train/lr': optimizer.param_groups[0]['lr'],
                }
                if coeff_a is not None:
                    log_dict['train/coeff_a'] = float(coeff_a)
                    log_dict['train/coeff_v'] = float(coeff_v)
                wandb.log(log_dict, step=global_step)
            elif args.modulation == "Normal":
                wandb.log({
                    'train/loss': loss.item(),
                    'train/loss_a': loss_a.item(),
                    'train/loss_v': loss_v.item(),
                    'train/epoch': epoch,
                    'train/iter_in_epoch': step,
                    'train/global_step': global_step,
                    'train/lr': optimizer.param_groups[0]['lr'],
                }, step=global_step)
        global_step += 1
    scheduler.step()
    
    if args.unimodal:
        return _loss / len(dataloader), global_step
    else:
        return _loss / len(dataloader), _loss_a / len(dataloader), _loss_v / len(dataloader), global_step


def valid(args, model, device, dataloader):
    softmax = nn.Softmax(dim=1)


    n_classes = 6
    if args.unimodal:
        with torch.no_grad():
            
            
            model.eval()
            # TODO: more flexible
            num = [0.0 for _ in range(n_classes)]
            acc = [0.0 for _ in range(n_classes)]
            

            for step, (spec, image, label) in enumerate(dataloader):

                spec = spec.to(device)
                image = image.to(device)
                label = label.to(device)

                feature, out = model(spec.unsqueeze(1).float(), image.float())

                

                prediction = softmax(out)
                

                for i in range(image.shape[0]):

                    ma = np.argmax(prediction[i].cpu().data.numpy())
                
                
                    num[label[i]] += 1.0

                    #pdb.set_trace()
                    if np.asarray(label[i].cpu()) == ma:
                        acc[label[i]] += 1.0
        return sum(acc) / sum(num)
    else:
        with torch.no_grad():
            
            
            model.eval()
            # TODO: more flexible
            num = [0.0 for _ in range(n_classes)]
            acc = [0.0 for _ in range(n_classes)]
            acc_a = [0.0 for _ in range(n_classes)]
            acc_v = [0.0 for _ in range(n_classes)]

            for step, (spec, image, label) in enumerate(dataloader):

                spec = spec.to(device)
                image = image.to(device)
                label = label.to(device)

                a, v, out, out_a, out_v = model(spec.unsqueeze(1).float(), image.float())

                if args.fusion_method == 'sum':
                    out_v = (torch.mm(v, torch.transpose(model.module.fusion_module.fc_y.weight, 0, 1)) +
                            model.module.fusion_module.fc_y.bias / 2)
                    out_a = (torch.mm(a, torch.transpose(model.module.fusion_module.fc_x.weight, 0, 1)) +
                            model.module.fusion_module.fc_x.bias / 2)
                else:
                    out_v = (torch.mm(v, torch.transpose(model.module.fusion_module.fc_out.weight[:, 512:], 0, 1)) +
                            model.module.fusion_module.fc_out.bias / 2)
                    out_a = (torch.mm(a, torch.transpose(model.module.fusion_module.fc_out.weight[:, :512], 0, 1)) +
                            model.module.fusion_module.fc_out.bias / 2)

                prediction = softmax(out)
                pred_v = softmax(out_v)
                pred_a = softmax(out_a)

                for i in range(image.shape[0]):

                    ma = np.argmax(prediction[i].cpu().data.numpy())
                    v = np.argmax(pred_v[i].cpu().data.numpy())
                    a = np.argmax(pred_a[i].cpu().data.numpy())
                    num[label[i]] += 1.0

                    #pdb.set_trace()
                    if np.asarray(label[i].cpu()) == ma:
                        acc[label[i]] += 1.0
                    if np.asarray(label[i].cpu()) == v:
                        acc_v[label[i]] += 1.0
                    if np.asarray(label[i].cpu()) == a:
                        acc_a[label[i]] += 1.0

        return sum(acc) / sum(num), sum(acc_a) / sum(num), sum(acc_v) / sum(num)


def main():
    args = get_arguments()
    print(args)
    random_seed = random.randint(1, 10000)
    setup_seed(random_seed)
    args.random_seed = random_seed 
    def make_run_name(args, seed: int) -> str:
        """
        wandb run name 규칙 정의
        예) 'CREMAD_OGM_GE_concat_optadam_e200_seed1234'
            'unimodal_audio_optadam_e200_seed42'
        """
        # 한국 시간
        now_kst = datetime.now(timezone(timedelta(hours=9))).strftime("%Y%m%d%H%M%S")

        if args.unimodal:
            base = f"unimodal_{args.unimodal_modality}_opt{args.optimizer}_e{args.epochs}_seed{seed}"
        else:
            base = f"{args.dataset}_{args.modulation}_{args.fusion_method}_opt{args.optimizer}_e{args.epochs}_seed{seed}"

        return f"{base}_{now_kst}", now_kst
    
    
    run_name, now_kst = make_run_name(args, random_seed)
    wandb.init(
        project="MKD_gaja",   # 프로젝트 고정
        name=run_name,         # 규칙 기반 run name
        config=vars(args)      # 모든 args 기록
    )
    if args.unimodal:
        assert args.unimodal_modality in ['audio', 'visual'], "Please specify which modality to use in unimodal training"
        print("Training unimodal model using {} modality".format(args.unimodal_modality))
        random_seed = random.randint(1, 10000)
        setup_seed(random_seed)
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
        gpu_ids = list(range(torch.cuda.device_count()))
        device = torch.device('cuda:0')
        model = UNIMODALClassifier(args)
        model.apply(weight_init)
        model.to(device)
        model = torch.nn.DataParallel(model, device_ids=gpu_ids)
        model.cuda()
        if args.optimizer == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=args.learning_rate,  weight_decay=1e-4)
        elif args.optimizer == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer, args.lr_decay_step, args.lr_decay_ratio)
    
    else:
        random_seed = random.randint(1, 10000)
        setup_seed(random_seed)
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
        gpu_ids = list(range(torch.cuda.device_count()))

        device = torch.device('cuda:0')

        model = AVClassifier(args)

        model.apply(weight_init)
        model.to(device)

        model = torch.nn.DataParallel(model, device_ids=gpu_ids)

        model.cuda()

        if args.optimizer == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=args.learning_rate,weight_decay=1e-4)
        elif args.optimizer == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer, args.lr_decay_step, args.lr_decay_ratio)
        
        

    train_dataset = CramedDataset(args, mode='train')
    test_dataset = CramedDataset(args, mode='test')
   

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,
                                  shuffle=True, num_workers=0, pin_memory=True)

    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size,
                                 shuffle=False, num_workers=0, pin_memory=True)

    if args.train:
        if args.modulation in ("OGM", "OGM_GE"):
            
            assert not args.unimodal, "Modulation is not applicable in unimodal training"
            assert args.alpha is not None, "--alpha must be specified when --modulation is OGM or OGM_GE"
            assert args.alpha > 0, "--alpha must be a positive float"
            
            
        

        best_acc = 0.0
        global_step = 0

        for epoch in range(args.epochs):

            print('Epoch: {}: '.format(epoch))

            if args.unimodal:
                batch_loss, global_step = train_epoch(args, epoch, model, device,
                                                                        train_dataloader, optimizer, scheduler,global_step)
                acc = valid(args, model, device, test_dataloader)
                wandb.log({'valid/acc': float(acc), 'epoch': epoch}, step=global_step)
            else:
                batch_loss, batch_loss_a, batch_loss_v, global_step = train_epoch(args, epoch, model, device,
                                                                        train_dataloader, optimizer, scheduler,global_step)
                acc, acc_a, acc_v = valid(args, model, device, test_dataloader)
                wandb.log({'valid/acc': float(acc),
                           'valid/acc_a': float(acc_a),
                           'valid/acc_v': float(acc_v),
                           'epoch': epoch}, step=global_step)

            if acc > best_acc:
                if args.unimodal:
                    best_acc = float(acc)
                    if not os.path.exists(args.ckpt_path):
                        os.mkdir(args.ckpt_path)

                    model_name = 'best_model_of_dataset_{}_unimodal_{}_' \
                                'optimizer_{}_' \
                                'epoch_{}_acc_{}_{}.pth'.format(args.dataset,
                                                             args.unimodal_modality,
                                                            args.optimizer,
                                                            epoch, acc, now_kst)

                    saved_dict = {'saved_epoch': epoch,
                                'acc': acc,
                                'model': model.state_dict(),
                                'optimizer': optimizer.state_dict(),
                                'scheduler': scheduler.state_dict(),
                                'args': vars(args)}

                    save_dir = os.path.join(args.ckpt_path, model_name)

                    torch.save(saved_dict, save_dir)
                    print('The best model has been saved at {}.'.format(save_dir))
                    print("Loss: {:.3f}, Acc: {:.3f}".format(batch_loss, acc))
                    
                else:
                    best_acc = float(acc)

                    if not os.path.exists(args.ckpt_path):
                        os.mkdir(args.ckpt_path)

                    model_name = 'best_model_of_dataset_{}_{}_{}_alpha_{}_' \
                                'optimizer_{}_modulate_starts_{}_ends_{}_' \
                                'epoch_{}_acc_{}_{}.pth'.format(args.dataset,
                                                            args.modulation,
                                                            args.fusion_method,
                                                            args.alpha,
                                                            args.optimizer,
                                                            args.modulation_starts,
                                                            args.modulation_ends,
                                                            epoch, acc, now_kst)

                    saved_dict = {'saved_epoch': epoch,
                                'modulation': args.modulation,
                                'alpha': args.alpha,
                                'fusion': args.fusion_method,
                                'acc': acc,
                                'model': model.state_dict(),
                                'optimizer': optimizer.state_dict(),
                                'scheduler': scheduler.state_dict(),
                                'args': vars(args)}

                    save_dir = os.path.join(args.ckpt_path, model_name)

                    torch.save(saved_dict, save_dir)
                    print('The best model has been saved at {}.'.format(save_dir))
                    print("Loss: {:.3f}, Acc: {:.3f}".format(batch_loss, acc))
                    print("Audio Acc: {:.3f}， Visual Acc: {:.3f} ".format(acc_a, acc_v))
                wandb.log({"checkpoint_path": save_dir}, step=global_step)
                wandb.run.summary["best_model_path"] = save_dir
            else:
                if args.unimodal:
                    print("Loss: {:.3f}, Acc: {:.3f}, Best Acc: {:.3f}".format(batch_loss, acc, best_acc))
                else:
                    print("Loss: {:.3f}, Acc: {:.3f}, Best Acc: {:.3f}".format(batch_loss, acc, best_acc))
                    print("Audio Acc: {:.3f}， Visual Acc: {:.3f} ".format(acc_a, acc_v))
                wandb.log({"checkpoint_path": None}, step=global_step)
    else:
        # first load trained model
        loaded_dict = torch.load(args.ckpt_path)
        # epoch = loaded_dict['saved_epoch']
        modulation = loaded_dict['modulation']
        # alpha = loaded_dict['alpha']
        fusion = loaded_dict['fusion']
        state_dict = loaded_dict['model']
        # optimizer_dict = loaded_dict['optimizer']
        # scheduler = loaded_dict['scheduler']

        assert modulation == args.modulation, 'inconsistency between modulation method of loaded model and args !'
        assert fusion == args.fusion_method, 'inconsistency between fusion method of loaded model and args !'

        model = model.load_state_dict(state_dict)
        print('Trained model loaded!')

        acc, acc_a, acc_v = valid(args, model, device, test_dataloader)
        print('Accuracy: {}, accuracy_a: {}, accuracy_v: {}'.format(acc, acc_a, acc_v))


if __name__ == "__main__":
    main()
