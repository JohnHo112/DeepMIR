import os
import shutil
import json
import torch
import torchaudio
import torchaudio.transforms as T
from tqdm import tqdm
import argparse

import demucs.separate

def preprocessing(root, save_dir, mode="train", segment_sec=15, source_sep=True):
    # read data
    with open(os.path.join(root, f"{mode}.json"), 'r') as f:
        data_paths = json.load(f)
    data_paths = [os.path.join(root, i[2:]) for i in data_paths]
    # print(data_paths)

    # delete and create save directory
    wav_dir = os.path.join(save_dir, "wav")
    melspec_dir = os.path.join(save_dir, "melspec")
    mfcc_dir = os.path.join(save_dir, "mfcc")

    if not os.path.exists(wav_dir):
        os.mkdir(wav_dir)
    if not os.path.exists(melspec_dir):
        os.mkdir(melspec_dir)
    if not os.path.exists(mfcc_dir):
        os.mkdir(mfcc_dir)

    # segment to slice
    data_json = []
    i = 0
    for data_path in tqdm(data_paths):
        artist_name = data_path.split("/")[3]

        # read waveform and sample rate
        waveform, sr = torchaudio.load(data_path)

        if source_sep:
            # source separete
            demucs.separate.main(["--mp3", "--two-stems", "vocals", "-n", "mdx_extra", data_path])
            waveform, vocal_sr = torchaudio.load(f"separated/mdx_extra/{data_path.split('/')[-1][:-4]}/vocals.mp3")

            # resample
            resampler = T.Resample(orig_freq=vocal_sr, new_freq=sr)
            waveform = resampler(waveform)

        segment_len = segment_sec * sr

        # count number segment
        num_segments = waveform.shape[1] // segment_len

        for n in range(num_segments):
            start = n * segment_len
            waveform_clip = waveform[:, start:start+segment_len]

            # save waveform
            # print(f"waveform shape: {waveform_clip.shape}")
            torchaudio.save(os.path.join(wav_dir, f"{i}.mp3"), waveform_clip, sample_rate=sr)
            
            # save melspectrogram
            melspec_transform = torchaudio.transforms.MelSpectrogram(sample_rate=sr,
                                                         n_fft=2048,
                                                         n_mels=128,
                                                         hop_length=512)
            to_db = torchaudio.transforms.AmplitudeToDB()
            melspec = melspec_transform(waveform_clip)
            melspec_db = to_db(melspec)
            # print(f"melspec shape: {melspec_db.shape}")
            torch.save(melspec_db, os.path.join(melspec_dir, f"{i}.pt"))

            # save mfcc
            mfcc_transform = torchaudio.transforms.MFCC(
                sample_rate=sr,
                n_mfcc=20,               # MFCC
                melkwargs={
                    "n_fft": 2048,
                    "hop_length": 512,
                    "n_mels": 40
                }
            )

            mfcc = mfcc_transform(waveform_clip)   # shape: (channels, n_mfcc, time)
            # print(f"mfcc shape: {mfcc.shape}")
            torch.save(mfcc, os.path.join(mfcc_dir, f"{i}.pt"))

            # create data dict
            data = {
                "id": i,
                "org": data_path,
                "wav": os.path.join(wav_dir, f"{i}.mp3"),
                "melspec": os.path.join(melspec_dir, f"{i}.pt"),
                "mfcc": os.path.join(mfcc_dir, f"{i}.pt"),
                "artist": artist_name,
                "segment_sec": segment_sec,
                "start": start,
            }
            data_json.append(data)

            i+=1

    # write json file
    with open(os.path.join(save_dir, f"{mode}.json"), "w", encoding="utf-8") as f:
        json.dump(data_json, f, ensure_ascii=False, indent=4)
    
def main(arg):
    save_root = arg.save_root
    segment_sec = arg.segment_sec
    source_sep = arg.source_sep

    # create save root
    if os.path.exists(save_root):
        shutil.rmtree(save_root)
    if not os.path.exists(save_root):
        os.mkdir(save_root)

    root = "hw1/artist20"

    # make train dataset
    save_dir = os.path.join(save_root, "train")
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    preprocessing(root, save_dir, mode="train", segment_sec=segment_sec, source_sep=source_sep)

    # make val dataset
    save_dir = os.path.join(save_root, "val")
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    preprocessing(root, save_dir, mode="val", segment_sec=segment_sec, source_sep=source_sep)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_root', type=str, default='dataset')
    parser.add_argument('--segment_sec', type=int, default='15')
    parser.add_argument('--source_sep', type=bool, default=False, choices=[True, False])
    args = parser.parse_args()
    main(args)
