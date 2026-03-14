import pickle

event2word, word2event = pickle.load(open("src/dictionary.pkl", 'rb'))
print(word2event)