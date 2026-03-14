import numpy as np
import pickle
from torch.utils.data.dataloader import Dataset
import utils
import os
import chord_recognition

class NewsDataset(Dataset):
    def __init__(self, dict_path, midi_l = [], X_LEN=1024, chord=False):
        self.chord = chord
        self.midi_l = midi_l
        self.x_len = X_LEN
        self.dictionary_path = dict_path
        self.event2word, self.word2event = pickle.load(open(self.dictionary_path, 'rb'))
        self.parser = self.prepare_data(self.midi_l)
    
    def __len__(self):
        return len(self.parser)  
    
    def __getitem__(self, index):
        return self.parser[index]
    
    def chord_extract(self, items):
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
    
    def extract_events(self, input_path):
        note_items, tempo_items = utils.read_items(input_path)
        note_items = utils.quantize_items(note_items)

        # chord

        max_time = note_items[-1].end

        # if you use chord items, you need to add chord_items into "items"
        # e.g. items = tempo_items + note_items + chord_items
        if self.chord:
            print("train with chord")
            chord_items = self.chord_extract(note_items)
            items = tempo_items + note_items + chord_items
        else:
            print("train without chord")
            items = tempo_items + note_items

        groups = utils.group_items(items, max_time)
        events = utils.item2event(groups)
        return events
        
    def prepare_data(self, midi_paths):
        # extract events
        all_events = []
        for path in midi_paths:
            events = self.extract_events(path)
            all_events.append(events)
        # event to word
        all_words = []
        for events in all_events:
            words = []
            for event in events:
                e = '{}_{}'.format(event.name, event.value)
                if e in self.event2word:
                    words.append(self.event2word[e])
                else:
                    # OOV
                    if event.name == 'Note Velocity':
                        # replace with max velocity based on our training data
                        words.append(self.event2word['Note Velocity_21'])
                    else:
                        # something is wrong
                        # you should handle it for your own purpose
                        print('something is wrong! {}'.format(e))
            all_words.append(words)
        
        # all_words is a list containing words list of all midi files
        # all_words = [[tokens of midi], [tokens of midi], ...]

        # you can cut the data into what you want to feed into model
        # Warning : this example cannot use in transformer_XL, you must implement group segments by yourself
        segments = []
        for words in all_words:
            pairs = []
            for i in range(0, len(words)-self.x_len-1, self.x_len):
                x = words[i:i+self.x_len]
                y = words[i+1:i+self.x_len+1]
                pairs.append([x, y])
            # abandon last segments in a midi
            pairs = pairs[0:len(pairs)-(len(pairs)%5)]
            segments = segments + pairs
        segments = np.array(segments)
        print(segments.shape)
        return segments
    
if __name__ == "__main__":
    dict_path = "src/basic_event_dictionary.pkl"
    midi_paths = os.listdir("Pop1K7/midi_analyzed/src_001")
    midi_paths = [os.path.join("Pop1K7/midi_analyzed/src_001", p) for p in midi_paths]
    dataset = NewsDataset(dict_path=dict_path, midi_l=midi_paths)
