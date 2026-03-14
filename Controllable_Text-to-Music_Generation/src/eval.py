import json
import random
import argparse
import numpy as np
import torch
from audiobox_aesthetics.infer import initialize_predictor
import os

def set_random_seeds(seed=1234):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def MetaAudioboxAesthetics(tar_paths):
    predictor = initialize_predictor()
    refs = [{"path": v} for v in tar_paths]

    return predictor.forward(refs)

def main(args):
    # set random seeds
    set_random_seeds()

    # read results json file
    tar_dir = args.tar_dir
    eval_results = args.output

    tar_paths = os.listdir(tar_dir)
    tar_paths = [os.path.join(tar_dir, p) for p in tar_paths]


    # compute meta audiobox aesthetics score
    aesthetics = MetaAudioboxAesthetics(tar_paths)
    # print(aesthetics)

    eval_results_dict = {}
    for i, items in enumerate(tar_paths):
        eval_results_dict[items] = aesthetics[i]
    
    with open(eval_results, "w", encoding="utf-8") as f:
        json.dump(eval_results_dict, f, ensure_ascii=False, indent=4)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--tar_dir', type=str, default='../home/fundwotsai/Deep_MIR_hw2/target_music_list_60s')
    parser.add_argument('--output', type=str, default='target_eval_results.json')
    args = parser.parse_args()
    main(args)