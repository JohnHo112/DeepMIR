# HW2: Controllable text-to-music generation
## Dataset
Download and unzip the HW2 dataset in the project root directory:
```bash
unzip Deep_MIR_hw2.zip
```

## Retrieval
### Environment Setup
run the following command to create envirements and  install all required dependencies
```bash
conda create --name DeepMIR python=3.10
conda activate DeepMIR
pip install -r requirements.txt
```

### Retrieval
Navigate to the retrieval directory:
```bash
cd src/retrieval
```

Run the following command to perform music retrieval based on latent similarity.
You can specify the latent space mode with `--mode`, choosing between `CLAP` and `M2L (Music2Latent)`:
```bash
python retrieval.py --mode "CLAP"
```

The target–reference music pairs are saved in: `src/retrieval/tar_ref.json`

### Evaluation
Run the following command to evaluate the similarity between target and reference music using:
* CLAP similarity
* Meta Audiobox Aesthetics (CE, CU, PC, PQ)
* Melody similarity
```bash
python eval.py
```
The evaluation results are saved in: `src/retrieval/ret_eval_results.json`

## Generation (Controllable text-to-music)
First, return to the project root and navigate to the generation directory:
```bash
cd src/generation
```

### Music Captioning
Follow the setup instructions in the [audio-flamingo3](https://github.com/NVIDIA/audio-flamingo/tree/audio_flamingo_3) repository.
Use the following commands to create the environment:
```bash
cd audio-flamingo
./environment_setup.sh af3
conda activate af3
```
Run the following command to perform music captioning:
```bash
./mtt.sh
```

The generated captions are saved in: `src/generation/audio-flamingo/caption.json`

### Text-to-Music
Follow the setup instructions in the [MuseControlLit](https://github.com/fundwotsai2001/MuseControlLite) repository.
`cd ..` and Run the following commands to set up the environment:
```bash
cd MuseControlLite
conda create -n MuseControlLite python=3.11
conda activate MuseControlLite
pip install -r requirements.txt
sudo apt install ffmpeg # For Linux
gdown 1Q9B333jcq1czA11JKTbM-DHANJ8YqGbP --folder
```

Since `Stable Audio Open 1.0` requires authentication, you’ll need a Hugging Face `token` for model access. 
run the following command login:
```bash
huggingface-cli login
```

Run the following script to perform text-to-music generation:
```bash
./ttm.sh
```

The generated music information is saved in:`src/generation/MuseControlLite/gen_infos.json`

### Evaluation
Before running the evaluation, make sure to switch back to the `DeepMIR` environment:
```bash
conda activate DeepMIR
```

Navigate to the generation directory `src/generation` and run the following command to evaluation the target music and generated music:
```bash
python eval.py
```
The evaluation results are saved in:`src/generation/gen_eval_results.json`

## Evaluation target music:
Navigate to the src directory `src`.
Before running the evaluation, make sure to switch back to the `DeepMIR` environment:
```bash
conda activate DeepMIR
```

Run the following command to evalution the target music Meta Audiobox Aesthetics.
```bash
python eval.py
```
The evaluation results are saved in:`src/target_eval_results.json`

## References
* https://github.com/LAION-AI/CLAP
* https://github.com/facebookresearch/audiobox-aesthetics
* https://github.com/NVIDIA/audio-flamingo/tree/audio_flamingo_3
* https://github.com/fundwotsai2001/MuseControlLite
