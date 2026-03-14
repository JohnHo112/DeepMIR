import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import demucs.separate
import argparse

from model import ShortChunkCNN
from ulits import artist2idx, idx2artist

def predict_song_softvote(model, song_segments, device):
    model.eval()
    all_probs = []
    with torch.no_grad():
        for seg in song_segments:
            seg = seg.to(device).unsqueeze(0)
            output = model(seg)
            probs = F.softmax(output, dim=1)  # (1, num_classes)
            all_probs.append(probs)
    avg_probs = torch.mean(torch.cat(all_probs, dim=0), dim=0)  # (num_classes,)
    # final_pred = avg_probs.argmax().item()
    return avg_probs

def val(args):
    # get args
    root = args.root
    checkpoint_path = args.checkpoint_path
    segment_sec = args.segment_sec
    source_sep = args.source_sep

    # model setting
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ShortChunkCNN().to(device)
    model.load_state_dict(torch.load(checkpoint_path, weights_only=True))

    # load the valid json
    with open(os.path.join(root, "val.json"), 'r') as f:
        data_paths = json.load(f)
    data_paths = [os.path.join(root, i[2:]) for i in data_paths]
    # print(data_paths)

    labels = []
    preds = []
    preds_json = {}

    i = 1
    for path in tqdm(data_paths):
        artist_name = path.split("/")[3]
        labels.append(artist2idx[artist_name])

        # read waveform and sample rate
        waveform, sr = torchaudio.load(path)
        if source_sep:
            # source separete
            demucs.separate.main(["--mp3", "--two-stems", "vocals", "-n", "mdx_extra", path])
            waveform, vocal_sr = torchaudio.load(f"separated/mdx_extra/{path.split('/')[-1][:-4]}/vocals.mp3")

            # resample
            resampler = T.Resample(orig_freq=vocal_sr, new_freq=sr)
            waveform = resampler(waveform)

        segment_len = segment_sec * sr
        # count number segment
        num_segments = waveform.shape[1] // segment_len

        song_segments = []

        overlap = 0.5   # 50% overlap
        hop_len = int(segment_len * (1 - overlap))  # 每次移動的長度

        for n in range(0, waveform.shape[1]-segment_len+1, hop_len):
            start = n * segment_len
            # print(path)
            waveform_clip = waveform[:, n:n+segment_len]
            # save melspectrogram
            melspec_transform = torchaudio.transforms.MelSpectrogram(sample_rate=sr,
                                                         n_fft=2048,
                                                         n_mels=128,
                                                         hop_length=512)
            to_db = torchaudio.transforms.AmplitudeToDB()
            melspec = melspec_transform(waveform_clip)
            melspec_db = to_db(melspec)
            melspec_db = torch.sum(melspec_db, dim=0).unsqueeze(0)
            
            song_segments.append(melspec_db)

        avg_probs = predict_song_softvote(model, song_segments, device)
        preds.append(avg_probs.argmax().item())
        topk_probs, topk_indices = torch.topk(avg_probs, k=3)
        preds_json[f"{i:03d}"] = [idx2artist[idx] for idx in topk_indices.tolist()]
        i += 1
        
    # write json file
    with open("task2/results/val_pred.json", "w", encoding="utf-8") as f:
        json.dump(preds_json, f, ensure_ascii=False, indent=4)
    with open("task2/results/val_ans.json", "w", encoding="utf-8") as f:
        json.dump([idx2artist[l] for l in labels], f, ensure_ascii=False, indent=4)

    # conjdfusion matrix
    cm = confusion_matrix(labels, preds)

    plt.figure(figsize=(10, 8))
    # sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    sns.heatmap(cm, annot=True, fmt="d")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.savefig("task2/results/confusion_matrix.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='hw1/artist20')
    parser.add_argument('--checkpoint_path', type=str, default='task2/results/model.pth')
    parser.add_argument('--segment_sec', type=int, default=15)
    parser.add_argument('--source_sep', type=bool, default=True)
    args = parser.parse_args()
    val(args)