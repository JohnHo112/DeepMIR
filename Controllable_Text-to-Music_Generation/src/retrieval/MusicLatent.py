from music2latent import EncoderDecoder
from tqdm import tqdm
import torchaudio

def music_latent(tars_path, refs_path):
    encdec = EncoderDecoder()
    tars = []
    refs = []
    for p in tars_path:
        wav, sr = torchaudio.load(p)
        tars.append(wav)
    for p in refs_path:
        wav, sr = torchaudio.load(p)
        refs.append(wav)

    tars_audio_embed = []
    for i in tqdm(tars):
        tars_audio_embed.append(encdec.encode(i).mean(dim=0).mean(dim=1))
    # print(tars_audio_embed)

    refs_audio_embed = []
    for i in tqdm(refs):
        refs_audio_embed.append(encdec.encode(i).mean(dim=0).mean(dim=1))
    # print(refs_audio_embed)
    
    return tars_audio_embed, refs_audio_embed
