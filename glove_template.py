import torch
# import torchtext
import nltk
from spacy.lang.en import English
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
# pytorch pad_sequence is not very flexible, you cannot specify the max len nor the position you want to pad to.
# Therefore we use keras pad_sequence or huggingface tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def load_file(file_path):
    labels, text = [], []
    nlp = English()
    tokenizer = nlp.tokenizer
    with open(file_path, 'r') as f:
        for line in f:
            if line == '':
                continue
            else:
                labels.append(line.split('\t')[0])
                tokens = tokenizer(line.split('\t')[1].strip())
                text.append([tok.text for tok in list(tokens)])
    f.close()
    return text, labels

def load_pretrained_embedding(file_path):
    vocab,embeddings = [],[]
    with open(file_path, 'rt') as f:
        print('Load pretrained embeddings: ')
        for line in tqdm(f):
            word = line.split(' ')[0]
            embedding = [float(val) for val in line.split(' ')[1:]]
            vocab.append(word)
            embeddings.append(embedding)
    return vocab, embeddings

def text_tokenizing(text, vocab_lookup, maxlen=20):
    tokenized_text = []
    for line in text:
        tokenized_line = []
        for word in line:
            tok = vocab_lookup[word] if word in vocab_lookup else vocab_lookup['<unk>']
            tokenized_line.append(tok)
        tokenized_text.append(tokenized_line)
        
    tokenized_text = pad_sequences(tokenized_text, maxlen=maxlen, padding='post', value=-1)
    return tokenized_text


vocab, embeddings = load_pretrained_embedding('glove.6B.100d.txt')
vocab_lookup = {w:i for i, w in enumerate(vocab)}

text, labels = load_file('train.txt')


tokenized_text = text_tokenizing(text, vocab_lookup)
