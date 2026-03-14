# coding: utf-8
import torch
import torch.nn as nn
import torchaudio
from dataloader import *

class Conv_2d(nn.Module):
    def __init__(self, input_channels, output_channels, shape=3, stride=1, pooling=2):
        super(Conv_2d, self).__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, shape, stride=stride, padding=shape//2)
        self.bn = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU()
        self.mp = nn.MaxPool2d(pooling)
    def forward(self, x):
        out = self.mp(self.relu(self.bn(self.conv(x))))
        return out

class ShortChunkCNN(nn.Module):
    '''
    Short-chunk CNN architecture.
    So-called vgg-ish model with a small receptive field.
    Deeper layers, smaller pooling (2x2).
    '''
    def __init__(self,
                n_channels=128,
                sample_rate=16000,
                n_mfcc=20,
                n_fft=512,
                f_min=0.0,
                f_max=8000.0,
                n_mels=128,
                n_class=20):
        super(ShortChunkCNN, self).__init__()

        # Spectrogram
        self.spec = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate,
                                                         n_fft=n_fft,
                                                         f_min=f_min,
                                                         f_max=f_max,
                                                         n_mels=n_mels)
        self.to_db = torchaudio.transforms.AmplitudeToDB()

        self.mfcc = torchaudio.transforms.MFCC(
            sample_rate=sample_rate,
            n_mfcc=n_mfcc,
            melkwargs={
                'n_fft': n_fft,
                'n_mels': n_mels,
                'f_min': f_min,
                'f_max': f_max
            }
        )
        self.spec_bn = nn.BatchNorm2d(1)

        # CNN
        self.layer1 = Conv_2d(1, n_channels, pooling=2)
        self.layer2 = Conv_2d(n_channels, n_channels, pooling=2)
        self.layer3 = Conv_2d(n_channels, n_channels*2, pooling=2)
        self.layer4 = Conv_2d(n_channels*2, n_channels*2, pooling=2)
        self.layer5 = Conv_2d(n_channels*2, n_channels*2, pooling=2)
        self.layer6 = Conv_2d(n_channels*2, n_channels*2, pooling=2)
        self.layer7 = Conv_2d(n_channels*2, n_channels*4, pooling=2)

        # Dense
        self.dense1 = nn.Linear(n_channels*4, n_channels*4)
        self.bn = nn.BatchNorm1d(n_channels*4)
        self.dense2 = nn.Linear(n_channels*4, n_class)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()

    def forward(self, x):
        # # Spectrogram
        # x = self.spec(x)
        # x = self.to_db(x)
        # # x = self.mfcc(x)
        # x = x.unsqueeze(1)
        # print(x.shape)
        x = self.spec_bn(x)

        # CNN
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = x.squeeze(2)

        # Global Max Pooling
        if x.size(-1) != 1:
            x = nn.MaxPool1d(x.size(-1))(x)
        x = x.squeeze(2)

        # Dense
        x = self.dense1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.dense2(x)
        x = nn.Sigmoid()(x)

        return x

if __name__ == "__main__":
    # # x = torch.randn((1, 480000))
    # x = torch.randn((8, 1, 128, 1876))
    # model = ShortChunkCNN()
    # model.eval()
    # y = model(x)
    # print(x.shape, y.shape)

    # for testing dataloader
    root = "dataset"
    train_dataset = Artist20Dataset(root, mode="train")
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)

    print(len(train_dataset))
    # test dataloader
    for wav, melspec, mfcc, label in train_loader:
        model = ShortChunkCNN()
        model.eval()
        y = model(melspec)
        print(melspec.shape, y.shape)
        break