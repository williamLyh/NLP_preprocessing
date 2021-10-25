# coding: utf-8
import gensim
import gensim.downloader as api
import torch
import torch.nn as nn


# Load word2vec pre-train model
# model = gensim.models.Word2Vec.load('./word2vec_pretrain_v300.model')
model = api.load('glove-wiki-gigaword-100')
weights = torch.FloatTensor(model.vectors)

## OOV token
oov_index = weights.shape(0)+1
weights = torch.cat((weights, torch.randn(1,100)))

# Build nn.Embedding() layer
embedding = nn.Embedding.from_pretrained(weights)
embedding.requires_grad = False


# Query
word = 'nightmare'
query_id = torch.tensor(model.key_to_index[word])

gensim_vector = torch.tensor(model[word])
embedding_vector = embedding(query_id)

print(gensim_vector==embedding_vector)