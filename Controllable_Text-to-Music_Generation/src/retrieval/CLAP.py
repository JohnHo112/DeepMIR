import laion_clap
from tqdm import tqdm

def clap(tars_path, refs_path):
    model = laion_clap.CLAP_Module(enable_fusion=False)
    model.load_ckpt() # download the default pretrained checkpoint.

    tars_audio_embed = []
    for i in tqdm(tars_path):
        tars_audio_embed.append(model.get_audio_embedding_from_filelist(x=[i], use_tensor=False).squeeze())
    # print(tars_audio_embed)

    refs_audio_embed = []
    for i in tqdm(refs_path):
        refs_audio_embed.append(model.get_audio_embedding_from_filelist(x=[i], use_tensor=False).squeeze())
    # print(refs_audio_embed)
    
    return tars_audio_embed, refs_audio_embed