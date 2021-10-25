# Huggingface has convenient API for preprocessing,
# but can only be used on transformer-based models, such as BERT or XLNet.
# https://huggingface.co/transformers/preprocessing.html

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')

batch_sentences = ["Hello I'm a single sentence",
                    "And another sentence",
                    "And the very very last one"]

encoded_input = tokenizer(batch_sentences, padding='max_length', max_length=20, truncation=True )
print(encoded_input)
