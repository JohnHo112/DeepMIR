import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio
from tqdm import tqdm
from torchaudio_augmentations import (
    RandomResizedCrop,
    RandomApply,
    PolarityInversion,
    Noise,
    Gain,
    HighLowPass,
    Delay,
    PitchShift,
    Reverb,
    Compose,
)

class Artist20Dataset(Dataset):
    def __init__(self, root, mode="train", transform=True):
        # artist to idx table
        self.artist2idx = {'aerosmith': 0, 'beatles': 1, 'creedence_clearwater_revival': 2, 'cure': 3, 'dave_matthews_band': 4, 'depeche_mode': 5, 'fleetwood_mac': 6, 'garth_brooks': 7, 'green_day': 8, 'led_zeppelin': 9, 'madonna': 10, 'metallica': 11, 'prince': 12, 'queen': 13, 'radiohead': 14, 'roxette': 15, 'steely_dan': 16, 'suzanne_vega': 17, 'tori_amos': 18, 'u2': 19}
        self.transform = transform
        
        if transform:
            self._get_augmentations()
        
        if mode == "train":
            # load the train json
            with open(os.path.join(root, mode, f"{mode}.json"), 'r') as f:
                self.datas = json.load(f)
        elif mode == "val":
            # load the valid json
            with open(os.path.join(root, mode, f"{mode}.json"), 'r') as f:
                self.datas = json.load(f)
        # print(len(self.datas))
    
    def _get_augmentations(self):
        transforms = [
            RandomResizedCrop(n_samples=15*16000),
            RandomApply([PolarityInversion()], p=0.8),
            RandomApply([Noise(min_snr=0.3, max_snr=0.5)], p=0.3),
            RandomApply([Gain()], p=0.2),
            RandomApply([HighLowPass(sample_rate=16000)], p=0.8),
            RandomApply([Delay(sample_rate=16000)], p=0.5),
            RandomApply([PitchShift(n_samples=15*16000, sample_rate=16000)], p=0.4),
            RandomApply([Reverb(sample_rate=16000)], p=0.3),
        ]
        self.augmentation = Compose(transforms=transforms)

    def __len__(self):
        return len(self.datas)
    
    def __getitem__(self, idx):
        data = self.datas[idx]
        org_path, wav_path, melspec_path, mfcc_path, artist = data["org"], data["wav"], data["melspec"], data["mfcc"], data["artist"]
        label = self.artist2idx[artist]

        wav, sr = torchaudio.load(wav_path)
        # melspec = torch.load(melspec_path, weights_only=False)  # weight_only=False avoid warning
        mfcc = torch.load(mfcc_path, weights_only=False)  # weight_only=False avoid warning

        if self.transform:
            wav_arg = self.augmentation(wav)

        # save melspectrogram
        melspec_transform = torchaudio.transforms.MelSpectrogram(sample_rate=sr,
                                                        n_fft=2048,
                                                        n_mels=128,
                                                        hop_length=512)
        to_db = torchaudio.transforms.AmplitudeToDB()
        melspec = melspec_transform(wav_arg)
        melspec = to_db(melspec)

        
        return wav, melspec, mfcc, label
    
    
if __name__ == "__main__":
    # for testing dataloader
    root = "dataset"
    train_dataset = Artist20Dataset(root, mode="train")
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)

    print(len(train_dataset))
    # test dataloader
    for wav, melspec, mfcc, label in train_loader:
        print(wav.shape)
        print(melspec.shape)
        print(mfcc.shape)
        print(label)
        break