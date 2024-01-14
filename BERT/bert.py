#https://github.com/google-research/bert

import numpy as np
import pandas as pd
import torch

from pytorch_transformers import BertTokenizer, BertForSequenceClassification, BertConfig

model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased')
