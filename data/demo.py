# import math
#
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import os
# import requests
# import tiktoken
#
# """
#  get the dataset
#
# """
# if not os.path.exists('data/sales_textbook.txt'):
#     url = ("https://github.com/lycorisadiata/learn-LLM/blob/8759b448f678dab51abec58be3890e9fbb542460/data"
#            "/sales_textbook.txt")
#     with open('data/sales_textbook.txt', 'w') as f:
#         f.write(requests.get(url).text)
#
# with open('data/sales_textbook.txt', 'r', encoding='UTF-8') as f:
#     text = f.read()
#
# """
#  set hyperparameters
#  get token:text and data of train, valid
# """
# # parameters
# batch_size = 4
# context_length = 16
# d_model = 64
# num_heads = 4
#
# # method of OpenAi
# encoding = tiktoken.get_encoding("cl100k_base")
# tokenized_text = encoding.encode(text)
# tokenized_text = torch.tensor(tokenized_text, dtype=torch.long)
# max_token_value = tokenized_text.max().item()
#
# # split text into train and validation
# train_idex = int(len(tokenized_text) * 0.9)
# train_data = tokenized_text[:train_idex]
# valid_data = tokenized_text[train_idex:]
#
# """
#  switch the data to inputs
# """
# data = train_data
# idxs = torch.randint(low=0, high=len(data) - context_length, size=(batch_size,))
# x_batch = torch.stack([data[idx:idx+context_length] for idx in idxs])
# y_batch = torch.stack([data[idx+1:idx+context_length+1] for idx in idxs])
#
# # embedding table
# input_embedding_lookup_table = nn.Embedding(max_token_value+1, d_model)
#
# x_batch_embedding = input_embedding_lookup_table(x_batch)
# y_batch_embedding = input_embedding_lookup_table(y_batch)
#
# # positional encoding
# # PE 2i   sin
# # PE 2i+1 cos
# position_encoding_lookup_table = torch.zeros(context_length, d_model)
# position = torch.arange(0, context_length, dtype=torch.float).unsqueeze(1)
# div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
# position_encoding_lookup_table[:, 0::2] = torch.sin(position * div_term)
# position_encoding_lookup_table[:, 1::2] = torch.cos(position * div_term)
# position_encoding_lookup_table = position_encoding_lookup_table.unsqueeze(0).expend(batch_size, -1, -1)
#
# x = x_batch_embedding + position_encoding_lookup_table
# y = y_batch_embedding + position_encoding_lookup_table
#
# """
#  Multi-Head Attention part
# """
# # process Q K V
# Wq = nn.Linear(d_model, d_model)
# Wk = nn.Linear(d_model, d_model)
# Wv = nn.Linear(d_model, d_model)
#
# Q = Wq(x)
# K = Wk(x)
# V = Wv(x)
#
# Q = Q.view(batch_size, context_length, num_heads, d_model//num_heads).permute(0, 2, 1, 3)
# K = K.view(batch_size, context_length, num_heads, d_model//num_heads).permute(0, 2, 1, 3)
# V = V.view(batch_size, context_length, num_heads, d_model//num_heads).permute(0, 2, 1, 3)
#
# # Attention(Q,K,V)
# output = Q @ K.transpose(-2, -1) / math.sqrt(d_model//num_heads)
#
# # set Mask mat
# mask = torch.triu(torch.ones(batch_size, batch_size), diagonal=1).bool()
# output = output.masked_fill(mask, float('-inf'))
#
# # softmax
# attention_score = F.softmax(output, dim=-1)
#
# # attention @ V
# A = attention_score @ V
#
# # concatenate
# A = A.transpose(1, 2).reshape(batch_size, -1, d_model)
# Wo = nn.Linear(d_model, d_model)
# output = Wo(A)
#
# # residual connection and layer normalization
# output = output + x
# layer_norm = nn.LayerNorm(d_model)
# layer_norm_output = layer_norm(output)
#
# # feedforward network
# output = nn.Linear(d_model, d_model * 4)(layer_norm_output)
# output = nn.ReLU()(output)
# output = nn.Linear(d_model * 4, d_model)(output)
# output = output + layer_norm_output
#
# # last Linear
# output = layer_norm(output)
#
# """
#  final layer linear
# """
# output = nn.Linear(d_model, max_token_value+1)(output)
# logits = F.softmax(output, dim=-1)
#
