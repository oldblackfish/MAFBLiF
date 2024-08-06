import sys
import time
import torch
import argparse
from tqdm import tqdm

from Val import val
from Utils import *
from Model import Network
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate')
    parser.add_argument('--n_steps', type=int, default=15, help='number of epochs to update learning rate')
    parser.add_argument('--n_epochs', type=int, default=30, help='number of epochs to train')
    parser.add_argument('--gamma', type=float, default=0.1, help='learning rate decaying factor')
    parser.add_argument('--trainset_dir', type=str, default='./Datasets/Win5_MLI_7x32x32/')

    return parser.parse_args()


def train(train_loader, cfg, test_scene_id):
    os.makedirs('./log/' + str(test_scene_id[0]) + '_' + str(test_scene_id[1]))
    sys.stdout = open('./log/' + str(test_scene_id[0]) + '_' + str(test_scene_id[1])
                      + '/' + str(test_scene_id[0]) + '_' + str(test_scene_id[1]) + '.txt', 'a')
    print(cfg)
    print(test_scene_id)

    net = Network().to(cfg.device)
    cudnn.benchmark = True
    optimizer = torch.optim.SGD([paras for paras in net.parameters() if paras.requires_grad == True],
                                lr=cfg.lr, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.n_steps, gamma=cfg.gamma)

    for idx_epoch in range(0, cfg.n_epochs):

        loss_epoch = []
        loss_list = []
        start_time = time.time()

        for idx_iter, (data, score_label) in tqdm(enumerate(train_loader), total=len(train_loader), leave=False):
            data, score_label = Variable(data).to(cfg.device), Variable(score_label).to(cfg.device)
            score_label = score_label.view(score_label.size()[0], -1)
            score_out = net(data)
            loss = torch.nn.MSELoss().to(cfg.device)(score_out, score_label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_epoch.append(loss.data.cpu())

        loss_list.append(float(np.array(loss_epoch).mean()))
        end_time = time.time()
        print('Test Epoch----%5d,'
              ' loss---%f,'
              ' Time---%f s'
              ' lr---%7f s'
              % (idx_epoch + 1,
                 float(np.array(loss_epoch).mean()),
                 end_time - start_time,
                 scheduler.get_last_lr()[0]))

        save_ckpt({'epoch': idx_epoch + 1, 'state_dict': net.state_dict(), 'loss': loss_list, },
                  save_path='./log/' + str(test_scene_id[0]) + '_' + str(test_scene_id[1]) + '/',
                  filename='MAFBLiF_epoch' + str(idx_epoch + 1) + '.pth.tar')
        load_model_path = './log/' + str(test_scene_id[0]) + '_' + str(test_scene_id[1]) + \
                          '/MAFBLiF_epoch' + str(idx_epoch + 1) + '.pth.tar'

        start_time_val = time.time()
        val(valset_dir=cfg.trainset_dir, test_scene_id=test_scene_id, load_model_path=load_model_path)
        end_time_val = time.time()
        print('Val_Time----    %f s'
              % (end_time_val - start_time_val)
              )

        if (idx_epoch + 1) != cfg.n_epochs:
            os.system('rm -r ./log/' + str(test_scene_id[0]) + '_' + str(test_scene_id[1]) + '/'
                      + 'MAFBLiF_epoch' + str(idx_epoch + 1) + '.pth.tar')
        scheduler.step()


def save_ckpt(state, save_path='./log', filename='checkpoint.pth.tar'):
    torch.save(state, os.path.join(save_path, filename))


def main(cfg):
    scene_num = 10  # for Win5 dataset
    full_dataset_dir = cfg.trainset_dir
    for i in range(0, scene_num):
        for j in range(i + 1, scene_num):
            test_scene_id = [i, j]
            train_set = MyTrainSetLoader_Kfold(dataset_dir=full_dataset_dir, test_scene_id=test_scene_id)
            train_loader = DataLoader(dataset=train_set, num_workers=cfg.num_workers, batch_size=cfg.batch_size, shuffle=True)
            train(train_loader, cfg, test_scene_id)


if __name__ == '__main__':
    cfg = parse_args()
    main(cfg)
