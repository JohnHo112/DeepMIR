import os
import pickle
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from transformers import GPT2Config, GPT2Model
import utils
from model import Model
from midi2audio import FluidSynth
import chord_recognition

def parse_opt():
    parser = argparse.ArgumentParser()
    ####################################################
    # you can define your arguments here. there is a example below.
    ####################################################
    parser.add_argument('--device', type=str, help='gpu device.', default='cuda')
    # dict and model checkpoint
    parser.add_argument('--dict_path', type=str, default='src/basic_event_dictionary.pkl')
    parser.add_argument('--model_path', type=str, default='checkpoint_base/epoch_100.pkl')
    parser.add_argument('--output_dir', type=str, default='results_task1')
    # hyparameter
    parser.add_argument('--temperature', type=float, default=1.2)
    parser.add_argument('--topk', type=int, default=5)
    # chord
    parser.add_argument('--chord', type=int, default=0)
    args = parser.parse_args()
    return args

opt = parse_opt()
event2word, word2event = pickle.load(open(opt.dict_path, 'rb'))
print(word2event)



def temperature_sampling(logits, temperature, topk):
    #################################################
    # 1. adjust softmax with the temperature parameter
    # 2. choose top-k highest probs
    # 3. normalize the topk highest probs
    # 4. random choose one from the top-k highest probs as result by the probs after normalize
    #################################################
    # 1. 調整溫度：logits / temperature
    logits = logits / temperature

    # 2. 選出 top-k 的 logit 和對應索引
    topk_logits, topk_indices = torch.topk(logits, k=topk)

    # 3. 對 topk logits 做 softmax 正規化
    topk_probs = F.softmax(topk_logits, dim=-1)

    # 4. 依據機率分布隨機取樣一個結果
    sampled_index = torch.multinomial(topk_probs, num_samples=1)

    # 回傳實際對應的 token id（從 topk_indices 中取出）
    return topk_indices[sampled_index]

def chord_extract(items):
        ####################################################
        # add your chord extraction method here if you want
        ####################################################
        method = chord_recognition.MIDIChord()
        chords = method.extract(notes=items)
        output = []
        for chord in chords:
            output.append(utils.Item(
                name='Chord',
                start=chord[0],
                end=chord[1],
                velocity=None,
                pitch=chord[2].split('/')[0]))
        return output

def extract_events(input_path, chord):
        note_items, tempo_items = utils.read_items(input_path)
        note_items = utils.quantize_items(note_items)
        max_time = note_items[-1].end


        # if you use chord items, you need to add chord_items into "items"
        # e.g. items = tempo_items + note_items + chord_items
        if chord:
            print("test with chord")
            chord_items = chord_extract(note_items)
            items = tempo_items + note_items + chord_items
        else:
            print("test without chord")
            items = tempo_items + note_items

        groups = utils.group_items(items, max_time)
        events = utils.item2event(groups)
        return events

def test(opt, n_target_bar = 32, prompt = False):
    # check path folder
    try:
        os.makedirs(f'{opt.output_dir.split("/")[0]}', exist_ok=True)
    except:
        pass

    event2word, word2event = pickle.load(open(opt.dict_path, 'rb'))
    with torch.no_grad():
        # load model
        checkpoint = torch.load(opt.model_path)
        cfg = GPT2Config()
        model = Model(cfg).to(opt.device)
        model.load_state_dict(checkpoint['model'])
        model.eval()

        batch_size = 1

        if prompt:  
            # If prompt, load prompt file, extract events, create tokens. (similar to dataset preparation)
            events = extract_events(os.path.join(opt.prompt_dir, prompt), opt.chord)
            words = [[event2word['{}_{}'.format(e.name, e.value)] for e in events]]
            words[0].append(event2word['Bar_None'])
        else:  
            # Or, random select prompt to start
            words = []
            for _ in range(batch_size):
                ws = [event2word['Bar_None']]
                if 'chord' in opt.dict_path:
                    print("with chord")
                    tempo_classes = [v for k, v in event2word.items() if 'Tempo Class' in k]
                    tempo_values = [v for k, v in event2word.items() if 'Tempo Value' in k]
                    chords = [v for k, v in event2word.items() if 'Chord' in k]
                    ws.append(event2word['Position_1/16'])
                    ws.append(np.random.choice(chords))
                    ws.append(event2word['Position_1/16'])
                    ws.append(np.random.choice(tempo_classes))
                    ws.append(np.random.choice(tempo_values))
                else:
                    print("without chord")
                    tempo_classes = [v for k, v in event2word.items() if 'Tempo Class' in k]
                    tempo_values = [v for k, v in event2word.items() if 'Tempo Value' in k]
                    ws.append(event2word['Position_1/16'])
                    ws.append(np.random.choice(tempo_classes))
                    ws.append(np.random.choice(tempo_values))
                words.append(ws)

        # generate
        original_length = len(words[0])
        initial_flag = 1
        if prompt:
            current_generated_bar = 7
        else:
            current_generated_bar = 0
        print('Start generating')
        while current_generated_bar < n_target_bar:
            print("\r", current_generated_bar, end="")
            # input
            if initial_flag:
                temp_x = np.zeros((batch_size, original_length))
                for b in range(batch_size):
                    for z, t in enumerate(words[b]):
                        temp_x[b][z] = t
                initial_flag = 0
            else:
                temp_x_new = np.zeros((batch_size, 1))
                for b in range(batch_size):
                    temp_x_new[b][0] = words[b][-1]
                temp_x = np.array([np.append(temp_x[0], temp_x_new[0])])
            
            temp_x = torch.Tensor(temp_x).long()

             # —— 新增這段 —— 
            max_len = model.model.config.n_positions  # 通常是 1024
            if temp_x.size(1) > max_len:
                temp_x = temp_x[:, -max_len:]  # 只保留最後 max_len 個 token
            
            output_logits = model(temp_x.to(opt.device))['last_hidden_state']
            # print(output_logits.shape)

            # sampling
            _logit = output_logits[0, -1].to('cpu').detach()
            word = temperature_sampling(
                logits=_logit, 
                temperature=opt.temperature,
                topk=opt.topk)

            words[0].append(int(word))

            if word == event2word['Bar_None']:
                current_generated_bar += 1
        utils.write_midi(
            words=words[0],
            word2event=word2event,
            output_path=opt.output_dir,
            prompt_path=None)
        
def main():
    opt = parse_opt()

    # generate midi
    for i in range(1, 21):
        opt.output_dir = os.path.join(opt.output_dir, f"{i:02d}.mid")
        test(opt)
        opt.output_dir = opt.output_dir.split("/")[0]
    print("task1 finish generate midi")

    # convert to wav
    midi_list = os.listdir(opt.output_dir)
    print(f"generated midi_list: {midi_list}")
    for midi in midi_list:
        print(os.path.join(opt.output_dir, f"{midi}.mid"))
        FluidSynth().midi_to_audio(os.path.join(opt.output_dir, f"{midi}"), 
                                os.path.join(opt.output_dir, f"{midi[:-4]}.wav"))
    print("finish midi to wav")

    

if __name__ == "__main__":
    main()