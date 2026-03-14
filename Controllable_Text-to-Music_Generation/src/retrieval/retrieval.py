import os
import argparse    
import laion_clap
import torch
import torch.nn.functional as F
from tqdm import tqdm
import json
import random
import numpy as np

from utils import find_close
from CLAP import clap
from MusicLatent import music_latent

def set_random_seeds(seed=1234):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main(args):
    set_random_seeds()
    tars_root = args.target
    refs_root = args.reference
    output = args.output
    mode = args.mode
    
    tars = os.listdir(tars_root)
    refs = os.listdir(refs_root)
    # print(tars)

    tars_path = [os.path.join(tars_root, tar_path) for tar_path in tars]
    refs_path = [os.path.join(refs_root, ref_root) for ref_root in refs]
    # print(tars_path)
    
    if mode == "CLAP":
        print(mode)
        tars_audio_embed, refs_audio_embed = clap(tars_path, refs_path)
    elif mode == "M2L":
        print(mode)
        tars_audio_embed, refs_audio_embed = music_latent(tars_path, refs_path)

    results = find_close(tars_path, refs_path, tars_audio_embed, refs_audio_embed)

    with open(output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=str, default='../../home/fundwotsai/Deep_MIR_hw2/target_music_list_60s')
    parser.add_argument('--reference', type=str, default='../../home/fundwotsai/Deep_MIR_hw2/referecne_music_list_60s')
    parser.add_argument('--output', type=str, default='tar_ref.json')
    parser.add_argument('--mode', type=str, default='CLAP')
    args = parser.parse_args()
    main(args)