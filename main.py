import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import requests
import tiktoken

"""
 get the dataset
         
"""
if not os.path.exists('data/sales_textbook.txt'):
    url =
    with open('data/sales_textbook.txt', 'w') as f:
        f.write(requests.get(url).text)

with open('data/sales_textbook.txt', 'r', encoding = 'UTF-8') as f:
    text = f.read()

"""
 set hyperparameters
 get token:text and data of train, valid
"""
# parameters
batch_size = 4
context_length = 16
d_model = 64


# method of OpenAi
encoding = tiktoken.get_encoding("cl100k_base")
tokenized_text = encoding.encode(text)
tokenized_text = torch.tensor(tokenized_text, dtype=torch.long)
max_token_value = tokenized_text.max().item()

# split text into train and validation
train_idex = int(len(tokenized_text) * 0.9)
train_data = tokenized_text[:train_idex]
valid_data = tokenized_text[train_idex:]

"""
 switch the data to inputs
"""
data = train_data
idxs = torch.randint(low=0, high=len(data) - context_length, size=(batch_size,))
x_batch = torch.stack([data[idx:idx+context_length] for idx in idxs])
y_batch = torch.stack([data[idx+1:idx+context_length+1] for idx in idxs])

# embedding table
input_embedding_lookup_table = nn.Embedding(max_token_value+1, d_model)

x_batch_embedding = input_embedding_lookup_table(x_batch)
y_batch_embedding = input_embedding_lookup_table(y_batch)

# positional encoding
# PE 2i   sin
# PE 2i+1 cos
position_encoding_lookup_table = torch.zeros(context_length, d_model)
position = torch.arange(0, context_length, dtype=torch.float).unsqueeze(1)
div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
position_encoding_lookup_table[:, 0::2] = torch.sin(position * div_term)
position_encoding_lookup_table[:, 1::2] = torch.cos(position * div_term)
position_encoding_lookup_table = position_encoding_lookup_table.unsqueeze(0).expend(batch_size, -1, -1)

x = x_batch_embedding + position_encoding_lookup_table
y = y_batch_embedding + position_encoding_lookup_table
