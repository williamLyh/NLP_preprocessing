import torch
from torch import nn
import torch.optim as optim
# import torchtext
import nltk
from spacy.lang.en import English
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
# pytorch pad_sequence is not very flexible, you cannot specify the max len nor the position you want to pad to.
# Therefore we use keras pad_sequence or huggingface tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from collections import Counter

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
    unk_token_idx = vocab_lookup['<unk>']
    for line in text:
        tokenized_line = []
        for word in line:
            tok = vocab_lookup[word] if word in vocab_lookup else unk_token_idx
            tokenized_line.append(tok)
        tokenized_text.append(tokenized_line)
        
    tokenized_text = pad_sequences(tokenized_text, maxlen=maxlen, padding='post', value=unk_token_idx)
    return tokenized_text

def processing_data(data_path):
    text, labels = load_file(data_path)
    tokenized_text = text_tokenizing(text, vocab_lookup)
    y_labels = [0 if label==list(set(labels))[0] else 1 for label in labels]
    return tokenized_text, y_labels

# Building vocabulary and loading embedding matrix 
vocab, embedding_matrix = load_pretrained_embedding('glove.6B.100d.txt')
    # The official version of glove has the '<unk>' token in its vocab. 
    # Its embedding is the same as the last existing word in the vocab.
vocab_lookup = {w:i for i, w in enumerate(vocab)}

# Tokenizing the data.
tokenized_text_train, y_labels_train = processing_data('train.txt')
print('labels composition:', Counter(y_labels_train))
tokenized_text_test, y_labels_test = processing_data('test.txt')
print('labels composition:', Counter(y_labels_train))
print(tokenized_text_train.shape, tokenized_text_test.shape)


def acc(l1, l2):
    cnt = 0
    for a, b in zip(l1, l2):
        if a==b:
            cnt += 1
    return cnt/len(l1)

class ExampleModel(nn.Module):
    def __init__(self, embedding_matrix):
        super(ExampleModel, self).__init__()
        self._embedding = nn.Embedding.from_pretrained(embedding_matrix)
        self._embedding.requires_grad = False
        self._lstm = nn.LSTM(100,100, batch_first=True)
        self._ffn = nn.Linear(100,1)

    def forward(self, x):
        x = self._embedding(x) # batch_size * sequence_len * embedding_dim
        x,(h_n,c_n) = self._lstm(x)
        x = self._ffn(h_n)
        return x

model = ExampleModel(torch.Tensor(embedding_matrix))
loss_func = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.2, weight_decay=0.001)

# model training 
model.train()
for epoch in range(200):
    model.train()
    model.zero_grad()
    y_pred = model(torch.tensor(tokenized_text_train))
    loss = loss_func(y_pred.squeeze(), torch.tensor(y_labels_train).type_as(y_pred))
    
    loss.backward()
    optimizer.step()

    if epoch%20 ==0:
        print('=====================================')
        print('loss: ', loss.item())
        y_pred_labels = [0 if y<0 else 1 for y in y_pred.squeeze()]
        print('train acc: ', acc(y_pred_labels, y_labels_train))
        model.eval()
        y_pred = model(torch.tensor(tokenized_text_test))
        y_pred_labels = [0 if y<0 else 1 for y in y_pred.squeeze()]
        print('test acc: ', acc(y_pred_labels, y_labels_test))

# model evaluating
model.eval()
y_pred = model(torch.tensor(tokenized_text_test))
y_pred_labels = [0 if y<0 else 1 for y in y_pred.squeeze()]
print(acc(y_pred_labels, y_labels_test))



