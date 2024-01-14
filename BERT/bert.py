#https://github.com/google-research/bert

import numpy as np
import pandas as pd
import torch

from pytorch_transformers import BertTokenizer, BertForSequenceClassification, BertConfig

device = torch.device('cuda')
model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased')
model.to(device)
model.train()

# for epoch in range(100):
#     #Dataset X, Y
#     loss, logits = model(X, labels = Y) #X: 두개의 문장, Y: 두 문장이 연속인가? (1/0)
#     pred = torch.argmax(F.softmax(logits), dim = 1)
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
