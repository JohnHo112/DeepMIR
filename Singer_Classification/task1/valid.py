import os
import numpy as np
import json
from sklearn.metrics import accuracy_score
import torchaudio
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from train import train
from ulits import artist2idx, idx2artist

def predict_song_softvote(model, song_segments):
    song_segments = np.stack((song_segments))  # (seg num, feature num)
    probs = model.predict_proba(song_segments)
    avg_probs = np.mean(probs, axis=0)
    return avg_probs

def load_json(root, path):
    # load the valid json
    with open(os.path.join(root, path), 'r') as f:
        data_paths = json.load(f)
    data_paths = [os.path.join(root, i[2:]) for i in data_paths]
    return data_paths

def val(dataset_root, segment_sec=15):
    # Train classifier
    print("Training...")
    clf, X, y = train(dataset_root)
    y_pred = clf.predict(X)
    print("Train Acc:", accuracy_score(y, y_pred))

    print("Validating...")
    # load val json
    val_json_root = "hw1/artist20"
    val_json_path = "val.json"
    data_paths = load_json(val_json_root, val_json_path)
    # print(f"val data paths: \n {data_paths}")

    labels = []
    preds = []
    preds_json = {}

    i = 1
    for path in tqdm(data_paths):
        artist_name = path.split("/")[3]
        labels.append(artist2idx[artist_name])  # get artists name label list

        # read waveform and sample rate
        waveform, sr = torchaudio.load(path)
        segment_len = segment_sec * sr

        song_segments = []

        overlap = 0.5   # 50% overlap
        hop_len = int(segment_len * (1 - overlap))  # shift window step

        for n in range(0, waveform.shape[1]-segment_len+1, hop_len):
            waveform_clip = waveform[:, n:n+segment_len]

            mfcc_transform = torchaudio.transforms.MFCC(
                sample_rate=sr,
                n_mfcc=20,               # MFCC dim
                melkwargs={
                    "n_fft": 2048,
                    "hop_length": 512,
                    "n_mels": 40
                }
            )
            mfccs = mfcc_transform(waveform_clip).squeeze()
            mfccs_mean = mfccs.mean(axis=1)
            mfccs_std  = mfccs.std(axis=1)
            mfccs_vec = np.concatenate([mfccs_mean, mfccs_std])
            song_segments.append(mfccs_vec)
        
        avg_probs = predict_song_softvote(clf, song_segments)
        preds.append(avg_probs.argmax())
        top3 = np.flip(np.argsort(avg_probs)[-3:])
        preds_json[f"{i:03d}"] = [idx2artist[idx] for idx in top3.tolist()]
        i += 1

    print("Val Acc:", accuracy_score(labels, preds))

    if not os.path.exists("task1/results"):
        os.mkdir("task1/results")

    # write json file
    with open("task1/results/val_pred.json", "w", encoding="utf-8") as f:
        json.dump(preds_json, f, ensure_ascii=False, indent=4)
    with open("task1/results/val_ans.json", "w", encoding="utf-8") as f:
        json.dump([idx2artist[l] for l in labels], f, ensure_ascii=False, indent=4)
    
    # conjdfusion matrix
    cm = confusion_matrix(labels, preds)

    plt.figure(figsize=(10, 8))
    # sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    sns.heatmap(cm, annot=True, fmt="d")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.savefig("task1/results/confusion_matrix.png")
    

if __name__ == "__main__":
    dataset_root = "dataset"
    val(dataset_root)