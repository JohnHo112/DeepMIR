from tqdm import tqdm
import torch
import torch.nn.functional as F

def find_close(tars_path, refs_path, tars_audio_embed, refs_audio_embed):
    results = {}
    for i in tqdm(range(len(tars_audio_embed))):
        best = 0
        for j in range(len(refs_audio_embed)):
            cos_sim = F.cosine_similarity(torch.tensor(tars_audio_embed[i]), torch.tensor(refs_audio_embed[j]), dim=0)
            if cos_sim > best:
                results[tars_path[i]] = refs_path[j]
                best = cos_sim
    return results