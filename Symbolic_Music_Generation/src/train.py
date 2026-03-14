import os
import glob
import pickle
import argparse
import numpy as np
from tqdm import tqdm
from torch import nn
import torch
from torch.utils.data.dataloader import DataLoader
from dataset import NewsDataset
from model import Model
from transformers import GPT2Config, GPT2Model

def parse_opt():
    parser = argparse.ArgumentParser()
    ####################################################
    # you can define your arguments here. there is a example below.
    # parser.add_argument('--device', type=str, help='gpu device.', default='cuda')
    ####################################################
    parser.add_argument('--device', type=str, help='gpu device.', default='cuda')
    # read and save configs
    parser.add_argument('--dict_path', type=str, help='the dictionary path.', default='dict/basic_event_dictionary.pkl')
    parser.add_argument('--ckp_folder', type=str, help='checkpoint folder.', default='checkpoint')
    # training configs
    parser.add_argument('--epoch', type=int, help='epoch.', default=100)
    parser.add_argument('--batch_size', type=int, help='batch size.', default=4)
    parser.add_argument('--lr', type=float, help='learning rate.', default=0.0002)
    # chord
    parser.add_argument('--chord', type=int, default=0)
    args = parser.parse_args()
    return args

def train(opt, is_continue = False, checkpoints_path = ''):
    epochs = opt.epoch

    # create data list
    # use glob to get all midi file path
    train_list = glob.glob('Pop1K7/midi_analyzed/**/*.mid', recursive=True)
    # train_list = train_list[:20]
    print('train list len =', len(train_list))

    # dataset
    if opt.chord:
        train_dataset = NewsDataset(opt.dict_path, midi_l = train_list, chord=True)
    else:
        train_dataset = NewsDataset(opt.dict_path, midi_l = train_list, chord=False)
    # dataloader
    BATCH_SIZE = opt.batch_size
    train_dataloader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle=True)
    print('Dataloader is created')

    if torch.cuda.is_available():
        print("Training on GPU")
        device = torch.device(opt.device)
    else:
        device = torch.device("cpu")
    
    # create model
    if not is_continue:
        start_epoch = 1
        cfg = GPT2Config()
        model = Model(cfg).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr = opt.lr)
    else:
        # wheather checkpoint_path is exist
        if os.path.isfile(checkpoints_path):
            checkpoint = torch.load(checkpoints_path)
        else:
            os._exit()
        start_epoch = checkpoint['epoch'] + 1

        model = Model().to(device)
        model.load_state_dict(checkpoint['model'])

        optimizer = torch.optim.Adam(model.parameters(), lr = 0.0002)
        optimizer.load_state_dict(checkpoint['optimizer'])

    print('Model is created \nStart training')
    
    model.train()
    losses = []
    try:
        os.makedirs(opt.ckp_folder, exist_ok=True)
        print("dir is created")
    except:
        pass
    
    for epoch in range(start_epoch, epochs+1):
        single_epoch = []
        for i in tqdm(train_dataloader):
            # x, y shape = (batch_size, length)
            x = i[:, 0, :].to(device).long()
            y = i[:, 1, :].to(device).long()
            output_logit = model(x)['last_hidden_state']
            # print(output_logit)
            loss = nn.CrossEntropyLoss()(output_logit.permute(0,2,1), y)
            loss.backward()
            single_epoch.append(loss.to('cpu').mean().item())
            optimizer.step()
            optimizer.zero_grad()
        single_epoch = np.array(single_epoch)
        losses.append(single_epoch.mean())
        print('>>> Epoch: {}, Loss: {:.5f}'.format(epoch,losses[-1]))
        torch.save({'epoch': epoch,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'loss': losses[-1],
                    }, os.path.join(opt.ckp_folder, 'epoch_%03d.pkl'%epoch))
        np.save(os.path.join(opt.ckp_folder, 'training_loss'), np.array(losses))

def main():
    opt = parse_opt()
    train(opt)

if __name__ == "__main__":
    main()