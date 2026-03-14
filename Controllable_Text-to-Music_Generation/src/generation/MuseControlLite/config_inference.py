def get_config():
    return {
        "condition_type": ["melody_stereo"], #  you can choose any combinations in the two sets: ["dynamics", "rhythm", "melody_mono", "audio"],  ["melody_stereo", "audio"]
                                    # When using audio, is recommend to use empty string "" as prompt
        "output_dir": "./generated_audio/output",

        "GPU_id": "0",

        "apadapter": True, # True for MuseControlLite, False for original Stable-audio

        "ap_scale": 1.0, # recommend 1.0 for MuseControlLite, other values are not tested

        "guidance_scale_text": 7.0,

        "guidance_scale_con": 1.5, # The separated guidance for Musical attribute condition
        
        "guidance_scale_audio": 1.0,
        
        "denoise_step": 50,

        "sigma_min": 0.3, # sigma_min and sigma_max are for the scheduler.

        "sigma_max": 500,  # Note that if sigma_max is too large or too small, the "audio condition generation" will be bad.

        "weight_dtype": "fp16", # fp16 and fp32 sounds quiet the same.

        "negative_text_prompt": "",

        ###############

        "audio_mask_start_seconds": 14, # Apply mask to musical attributes choose only one mask to use, it automatically generates a complemetary mask to the other condition

        "audio_mask_end_seconds": 47, 

        "musical_attribute_mask_start_seconds": 0, # 'Apply mask to audio condition, choose only one mask to use, it automatically generates a complemetary mask to the other condition'

        "musical_attribute_mask_end_seconds": 0,

        ###############

        "no_text": False, # Optional, set to true if no text prompt is needed (possible for audio inpainting or outpainting)

        "show_result_and_plt": True,

        "audio_files": [
            "/home/disp-ho/DeepMIR/hw2/home/fundwotsai/Deep_MIR_hw2/target_music_list_60s/【楽譜あり】IRIS OUT⧸米津玄師（ピアノソロ上級）劇場版『チェンソーマン レゼ篇』主題歌【ピアノアレンジ楽譜】.mp3",
            "/home/disp-ho/DeepMIR/hw2/home/fundwotsai/Deep_MIR_hw2/target_music_list_60s/【菊花台-周杰倫（D調）】附伴奏⧸鋼琴伴奏(竹笛Bamboo flute、Roland Aerophone AE-10) 演奏：蘇俊琪(PSR-S970)audio-technica AT-2035_60s.mp3",
            "/home/disp-ho/DeepMIR/hw2/home/fundwotsai/Deep_MIR_hw2/target_music_list_60s/4_jazz_120_beat_3-4.wav",
            "/home/disp-ho/DeepMIR/hw2/home/fundwotsai/Deep_MIR_hw2/target_music_list_60s/6_rock_102_beat_3-4.wav",
            "/home/disp-ho/DeepMIR/hw2/home/fundwotsai/Deep_MIR_hw2/target_music_list_60s/10_country_114_beat_4-4.wav",
            "/home/disp-ho/DeepMIR/hw2/home/fundwotsai/Deep_MIR_hw2/target_music_list_60s/Hedwig’s theme x dizi ｜from Harry Potter ｜竹笛也能施魔法！Bamboo flute_60s.mp3",
            "/home/disp-ho/DeepMIR/hw2/home/fundwotsai/Deep_MIR_hw2/target_music_list_60s/Mussorgsky： Pictures at an Exhibition (Pletnev, Andsnes)_60s.mp3",
            "/home/disp-ho/DeepMIR/hw2/home/fundwotsai/Deep_MIR_hw2/target_music_list_60s/Spirited Away OST「Always With Me ⧸ Itsumo Nando Demo」Ru's Piano Cover [Sheet Music]_60s.mp3",
            "/home/disp-ho/DeepMIR/hw2/home/fundwotsai/Deep_MIR_hw2/target_music_list_60s/竹笛｜这世界那么多人_cover 莫文蔚_60s.mp3",
        ],
        # "audio_files": [
        #     "SDD_nosinging/SDD_audio/34/1004034.mp3",
        #     "original_15s/original_9.wav",
        #     "original_15s/original_10.wav",
        #     "original_15s/original_11.wav",
        #     "original_15s/original_15.wav",
        #     "original_15s/original_16.wav",
        #     "original_15s/original_21.wav",
        #     "original_15s/original_25.wav",
        # ],

        "text": [
                "metal music that has a constant melody throughout with accompanying instruments used to supplement the melody which can be heard in possibly a casual setting",
                "metal music that has a constant melody throughout with accompanying instruments used to supplement the melody which can be heard in possibly a casual setting",
                "metal music that has a constant melody throughout with accompanying instruments used to supplement the melody which can be heard in possibly a casual setting",
                "metal music that has a constant melody throughout with accompanying instruments used to supplement the melody which can be heard in possibly a casual setting",
                "metal music that has a constant melody throughout with accompanying instruments used to supplement the melody which can be heard in possibly a casual setting",
                "Electronic music that has a constant melody throughout with accompanying instruments used to supplement the melody which can be heard in possibly a casual setting",
                "Electronic music that has a constant melody throughout with accompanying instruments used to supplement the melody which can be heard in possibly a casual setting",
                "Electronic music that has a constant melody throughout with accompanying instruments used to supplement the melody which can be heard in possibly a casual setting",
                "Electronic music that has a constant melody throughout with accompanying instruments used to supplement the melody which can be heard in possibly a casual setting",
                ],

        ########## adapters avilable ############
        # We trained 4 set of adapters:
        # 1. with conditions ["melody_mono", "dynamics", "rhythm"]
        # 2. with conditions ["melody_mono"]
        # 3. with conditions ["melody_stereo"]
        # 3. with conditions ["audio"]
        # MuseControlLite_inference_all.py will automaticaly choose the most suitable model according to the condition type:
        ###############
        # Works for condition ["dynamics", "rhythm", "melody_mono"]
        "transformer_ckpt_musical": "./checkpoints/woSDD-all/model_3.safetensors",
        
        "extractor_ckpt_musical": {
            "dynamics": "./checkpoints/woSDD-all/model_1.safetensors",
            "melody": "./checkpoints/woSDD-all/model.safetensors",
            "rhythm": "./checkpoints/woSDD-all/model_2.safetensors",
        },
        ###############

        # Works for ['audio], it works without a feature extractor, and could cooperate with other adapters
        #################
        "audio_transformer_ckpt": "./checkpoints/70000_Audio/model.safetensors",

        # Specialized for ['melody_stereo']
        ###############
        "transformer_ckpt_melody_stero": "./checkpoints/70000_Melody_stereo/model_1.safetensors",

        "extractor_ckpt_melody_stero": {
            "melody": "./checkpoints/70000_Melody_stereo/model.safetensors",
        },
        ###############

        # Specialized for ['melody_mono']
        ###############
        "transformer_ckpt_melody_mono": "./checkpoints/40000_Melody_mono/model_1.safetensors",

        "extractor_ckpt_melody_mono": {
            "melody": "./checkpoints/40000_Melody_mono/model.safetensors",
        },
        ###############
    }