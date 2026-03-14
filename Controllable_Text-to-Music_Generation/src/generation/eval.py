import os
import json
import random
import argparse
import scipy.signal as signal
import numpy as np
import torch
import torch.nn.functional as F
from torchaudio import transforms as T
import librosa
import torchaudio
import laion_clap
from audiobox_aesthetics.infer import initialize_predictor

def set_random_seeds(seed=1234):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def CLAP(tars, gens, texts):
    model = laion_clap.CLAP_Module(enable_fusion=False)
    model.load_ckpt() # download the default pretrained checkpoint.
    model.eval()

    tar_text_scores = []
    text_gen_scores = []
    gen_tar_scores = []
    for tar, gen, text in zip(tars, gens, texts):
        tar_waveform, sr = torchaudio.load(tar)
        tar_embed = model.get_audio_embedding_from_data(x=tar_waveform[:, :47*44100].tolist(), use_tensor=False).squeeze()
        tar_embed = tar_embed[0]
        # tar_embed = model.get_audio_embedding_from_filelist(x=[tar], use_tensor=False).squeeze()
        gen_embed = model.get_audio_embedding_from_filelist(x=[gen], use_tensor=False).squeeze()
        text_embed = model.get_text_embedding(x=[text], use_tensor=False).squeeze()

        tar_text_scores.append(F.cosine_similarity(torch.tensor(tar_embed), torch.tensor(text_embed), dim=0))
        text_gen_scores.append(F.cosine_similarity(torch.tensor(text_embed), torch.tensor(gen_embed), dim=0))
        gen_tar_scores.append(F.cosine_similarity(torch.tensor(gen_embed), torch.tensor(tar_embed), dim=0))
        
    return tar_text_scores, text_gen_scores, gen_tar_scores

def MetaAudioboxAesthetics(datas):
    predictor = initialize_predictor()
    refs = [{"path": v} for v in datas]

    return predictor.forward(refs)

def extract_melody_one_hot(audio_path,
                           sr=44100,
                           cutoff=261.2, 
                           win_length=2048,
                           hop_length=256):
    """
    Extract a one-hot chromagram-based melody from an audio file (mono).
    
    Parameters:
    -----------
    audio_path : str
        Path to the input audio file.
    sr : int
        Target sample rate to resample the audio (default: 44100).
    cutoff : float
        The high-pass filter cutoff frequency in Hz (default: Middle C ~ 261.2 Hz).
    win_length : int
        STFT window length for the chromagram (default: 2048).
    hop_length : int
        STFT hop length for the chromagram (default: 256).
    
    Returns:
    --------
    one_hot_chroma : np.ndarray, shape=(12, n_frames)
        One-hot chromagram of the most prominent pitch class per frame.
    """
    # ---------------------------------------------------------
    # 1. Load audio (Torchaudio => shape: (channels, samples))
    # ---------------------------------------------------------
    audio, in_sr = torchaudio.load(audio_path)

    # Convert to mono by averaging channels: shape => (samples,)
    audio_mono = audio.mean(dim=0)

    # Resample if necessary
    if in_sr != sr:
        resample_tf = T.Resample(orig_freq=in_sr, new_freq=sr)
        audio_mono = resample_tf(audio_mono)

    # Convert torch.Tensor => NumPy array: shape (samples,)
    y = audio_mono.numpy()

    # ---------------------------------------------------------
    # 2. Design & apply a high-pass filter (Butterworth, order=2)
    # ---------------------------------------------------------
    nyquist = 0.5 * sr
    norm_cutoff = cutoff / nyquist
    b, a = signal.butter(N=2, Wn=norm_cutoff, btype='high', analog=False)
    
    # filtfilt expects shape (n_samples,) for 1D
    y_hp = signal.filtfilt(b, a, y)

    # ---------------------------------------------------------
    # 3. Compute the chromagram (librosa => shape: (12, n_frames))
    # ---------------------------------------------------------
    chroma = librosa.feature.chroma_stft(
        y=y_hp,
        sr=sr,
        n_fft=win_length,      # Usually >= win_length
        win_length=win_length,
        hop_length=hop_length
    )

    # ---------------------------------------------------------
    # 4. Convert chromagram to one-hot via argmax along pitch classes
    # ---------------------------------------------------------
    # pitch_class_idx => shape=(n_frames,)
    pitch_class_idx = np.argmax(chroma, axis=0)

    # Make a zero array of the same shape => (12, n_frames)
    one_hot_chroma = np.zeros_like(chroma)

    # For each frame (column in chroma), set the argmax row to 1
    one_hot_chroma[pitch_class_idx, np.arange(chroma.shape[1])] = 1.0
    
    return one_hot_chroma

def MelodyAcc(tars, gens):
    melody_accs = []
    for target_audio_path, generated_audio_path in zip(tars, gens):
        gt_melody = extract_melody_one_hot(target_audio_path)      
        gen_melody = extract_melody_one_hot(generated_audio_path)
        min_len_melody = min(gen_melody.shape[1], gt_melody.shape[1])
        matches = ((gen_melody[:, :min_len_melody] == gt_melody[:, :min_len_melody]) & (gen_melody[:, :min_len_melody] == 1)).sum()
        accuracy = matches / min_len_melody
        melody_accs.append(accuracy)

    return melody_accs

def main(args):
    # set random seeds
    set_random_seeds()

    # read results json file
    gen_infos = args.gen_infos
    eval_results = args.output

    with open(gen_infos, "r", encoding="utf-8") as f:
        datas = json.load(f)
    # print(datas)

    tars = []
    gens = []
    texts = []
    for key, val in datas.items():
        gen = os.path.join("MuseControlLite", val["gen"])
        text = val["text"]
        tars.append(key[3:])
        gens.append(gen)
        texts.append(text)

    # compute CLAP sorce
    tar_text_scores, text_gen_scores, gen_tar_scores = CLAP(tars, gens, texts)
    tar_text_scores = [s.item() for s in tar_text_scores]
    text_gen_scores = [s.item() for s in text_gen_scores]
    gen_tar_scores = [s.item() for s in gen_tar_scores]
    # print(tar_text_scores)
    # print(text_gen_scores)
    # print(gen_tar_scores)

    # compute meta audiobox aesthetics score
    aesthetics = MetaAudioboxAesthetics(gens)
    # print(aesthetics)

    # compute melody acc
    melody_accs = MelodyAcc(tars, gens)
    # print(melody_accs)

    eval_results_dict = {}
    for i, items in enumerate(datas.items()):
        temp = {}
        temp["Ref"] = items[1]
        temp["tar_text"] = tar_text_scores[i]
        temp["text_gen"] = text_gen_scores[i]
        temp["gen_tar"] = gen_tar_scores[i]
        temp["Meta Audiobox Aesthetics"] = aesthetics[i]
        temp["Melody similarity"] = melody_accs[i]
        eval_results_dict[items[0]] = temp
    
    with open(eval_results, "w", encoding="utf-8") as f:
        json.dump(eval_results_dict, f, ensure_ascii=False, indent=4)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gen_infos', type=str, default='MuseControlLite/gen_infos.json')
    parser.add_argument('--output', type=str, default='gen_eval_results.json')
    args = parser.parse_args()
    main(args)