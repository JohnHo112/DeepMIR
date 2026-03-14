# HW3: Symbolic Music Generation
## Environment
- **OS:** Ubuntu 22.04  
- **CUDA:** 12.2  
- **GPU:** RTX 2080 Ti  
- **Python:** 3.10

## Environment Setup
Create and activate the conda environment:
```bash
conda create --name deepmir_hw3 python=3.10
conda activate deepmir_hw3
```

Install all required dependencies:
```bash
pip install -r requirements.txt
```

To enable MIDI-to-WAV conversion, install:
```bash
sudo apt-get install fluidsynth
```

## Dataset
Download and unzip the HW3 dataset in the project root directory:
```bash
unzip Pop1K7.zip
```

## Prompt song for Continuation generation
Download and unzip the provided prompt songs:
```bash
unzip prompt_song.zip
```

## Directory Structure
```bash
root/
│-- dict/        
│   ├── basic_event_dictionary.pkl    # Dictionary without chord tokens             
│   ├── chord_event_dictionary.pkl    # Dictionary with chord tokens              
│-- MusDr/                            # Evaluation tools             
│-- Pop1K7/                           # Dataset                          
│-- prompt_song/                      # Prompt songs for continuation        
│-- src/                              # Source code       
│-- .gitignore       
│-- eval_metrics.py                   # Evaluation metrics (H1, H4, GS)      
│-- README.md      
│-- requirements.txt
```

## Training
* Training Without Chords
```bash
python src/train.py
```

* Training With Chords
```bash
python src/train.py --chord 1 --dict_path dict/chord_event_dictionary.pkl
```

Argument description:
* `--chord`: 0 = without chord extraction, 1 = with chord extraction
* `--dict_path`: path to the event2word and word2event dictionary

After training, a `checkpoint/` directory will be created containing model checkpoints.

## Inference
### Task1 Unconditional generation
* Without Chords
```bash
python src/test_task1.py --dict_path dict/basic_event_dictionary.pkl --model_path checkpoint/epoch_050.pkl --output_dir results_task1 --temperature 1.2 --topk 5 --chord 0
```

* With Chords
```bash
python src/test_task1.py --dict_path dict/chord_event_dictionary.pkl --model_path checkpoint/chord_epoch_050.pkl --output_dir results_task1 --temperature 1.2 --topk 5 --chord 1
```

Argument description:
* `--dict_path`: path to the event2word and word2event dictionary
* `--model_path`: model checkpoint
* `--output_dir`: directory for generated outputs
* `--temperature`: sampling temperature
* `--topk`: topk sampling
* `--chord`: 0 = without chord, 1 = with chord

After inference, a `results_task1/` directory will be created containing `*.mid` and corresponding `*.wav` files.

### Task2 Continuation generation
* Without Chords
```bash
python src/test_task2.py --dict_path dict/basic_event_dictionary.pkl --model_path checkpoint/epoch_100.pkl --output_dir results_task2 --prompt_dir prompt_song --config 1 --chord 0
```

* With Chord
```bash
python src/test_task2.py --dict_path dict/chord_event_dictionary.pkl --model_path checkpoint/chord_epoch_050.pkl --output_dir results_task2 --prompt_dir prompt_song --config 3 --chord 1
```

Argument description:
* `--dict_path`: path to the event2word and word2event dictionary
* `--model_path`: model checkpoint
* `--output_dir`: directory for generated outputs
* `--prompt_dir`: directory containing prompt MIDI files
* `--config`: identifier appended to output filenames
* `--chord`: 0 = without chord, 1 = with chord

After inference, a `results_task2/` directory will be created containing `*.mid` and `*.wav` files.

## Evaluation
* Evaluate Generated Songs (Without Chords)
```bash
python eval_metrics.py --dict_path dict/basic_event_dictionary.pkl --output_file_path results_task1 --real_data 0 --chord 0
```

* Evaluate Generated Songs (With Chords)
```bash
python eval_metrics.py --dict_path dict/chord_event_dictionary.pkl --output_file_path results_task1 --real_data 0 --chord 1
```

* Evaluate Real Dataset
Run the following command to evaluation real data:
```bash
python eval_metrics.py --dict_path dict/basic_event_dictionary.pkl --real_data 1
```

Argument description:
* `--dict_path`: path to the event2word and word2event dictionary
* `--output_file_path`: directory containing generated results
* `--real_data`: 0 = evaluate generated songs, 1 = evaluate real Pop1K7 dataset (directory structure is automatically handled)