import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from glob import glob
from tqdm.notebook import tqdm
import gc
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset
from transformers import DebertaTokenizer, DebertaForSequenceClassification
from sklearn.model_selection import train_test_split
import wandb  # Import wandb library for experiment tracking
import wandb_addons
import math
from torch.optim.lr_scheduler import LambdaLR

cmap = mpl.cm.get_cmap('coolwarm')

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Rest of your code...

class CFG:
    verbose = 0
    wandb = True
    competition = 'llm-detect-ai-generated-text'
    _wandb_kernel = 'awsaf49'
    comment = 'DebertaV3-MaxSeq_200-ext_s-torch'
    preset = "deberta_v3_base_en"
    sequence_length = 200
    device = 'GPU' if str(device) == 'cuda' else 'CPU'  # Set to 'GPU' or 'CPU'
    seed = 42
    num_folds = 5
    selected_folds = [0, 1, 2]
    epochs = 3
    batch_size = 3
    drop_remainder = True
    cache = True
    scheduler = 'cosine'
    class_names = ["real", "fake"]
    num_classes = len(class_names)
    class_labels = list(range(num_classes))
    label2name = dict(zip(class_labels, class_names))
    name2label = {v: k for k, v in label2name.items()}


def set_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_random_seed(CFG.seed)


def get_device():
    "Detect and initializes GPU/TPU automatically"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if str(device) == "cuda":
        ngpu = torch.cuda.device_count()
        print(f"> Running on GPU | Num of GPUs: {ngpu}")
        strategy = "GPU"
    else:
        print("> Running on CPU")
        strategy = "CPU"

    return strategy, device

strategy, CFG.device = get_device()
CFG.replicas = 1

# Set the base path
BASE_PATH = 'C:/Users/gks12/PycharmProjects/keras_LLM_exam_1/llm-detect-ai-generated-text'

# TRAIN
# Read CSV file into a DataFrame
df = pd.read_csv(f'{BASE_PATH}/train_essays.csv')

df['name'] = df.generated.map(CFG.label2name)  # Map answer labels using name-to-label mapping

# Display information about the train data
# print("# Train Data: {:,}".format(len(df)))
# print("# Sample:")
# print(df.head(2))
#
# # Show distribution of answers using a bar plot
# plt.figure(figsize=(8, 4))
# df['name'].value_counts().plot.bar(color=['blue', 'orange'])  # Adjust colors as needed
# plt.xlabel("Class")
# plt.ylabel("Count")
# plt.title("Class distribution for Train Data")
# plt.show()

# External Datasets
ext_df1 = pd.read_csv(f'{BASE_PATH}/train_drcat_04.csv')
ext_df2 = pd.read_csv(f'{BASE_PATH}/argugpt.csv')[['id', 'text', 'model']]

ext_df2.rename(columns={'model': 'source'}, inplace=True)
ext_df2['label'] = 1

ext_df = pd.concat([
    ext_df1[ext_df1.source == 'persuade_corpus'].sample(10000),
    ext_df1[ext_df1.source != 'persuade_corpus'],
])
ext_df['name'] = ext_df.label.map(CFG.label2name)
# Display information about the external data
# print("# External Data: {:,}".format(len(ext_df)))
# print("# Sample:")
ext_df.head(2)

# Show distribution of answers using a bar plot
# plt.figure(figsize=(8, 4))
# ext_df.name.value_counts().plot.bar(color=[cmap(0.0), cmap(0.65)])
# plt.xlabel("Class")
# plt.ylabel("Count")
# plt.title("Answer distribution for External Data")
# plt.show()

# Combine External and Train Data
df = ext_df.copy().reset_index(drop=True)  # pd.concat([ext_df, df], axis=0)
df.head()

# Data Split
skf = StratifiedKFold(n_splits=CFG.num_folds, shuffle=True, random_state=CFG.seed)  # Initialize K-Fold
df = df.reset_index(drop=True)  # Reset dataframe index
df['stratify'] = df.label.astype(str) + df.source.astype(str)
df["fold"] = -1  # New 'fold' column
# Assign folds using StratifiedKFold
for fold, (train_idx, val_idx) in enumerate(skf.split(df, df['stratify'])):
    df.loc[val_idx, 'fold'] = fold
# Display label distribution for each fold
df.groupby(["fold", "name", "source"]).size()

# Preprocessing
tokenizer = DebertaTokenizer.from_pretrained(CFG.preset)
model = DebertaForSequenceClassification.from_pretrained(CFG.preset)


def preprocess_fn(text, label=None):
    inputs = tokenizer(text, truncation=True, padding='max_length', max_length=CFG.sequence_length, return_tensors='pt')
    inputs = {key: value.squeeze(0) for key, value in inputs.items()}
    return (inputs, torch.tensor(label)) if label is not None else inputs


# Dataset class
class TextDataset(Dataset):
    def __init__(self, texts, labels=None):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx] if self.labels is not None else None
        return preprocess_fn(text, label)


# DataLoader function
def build_dataset(texts, labels=None, batch_size=32, cache=False, drop_remainder=True, repeat=False, shuffle=True):
    dataset = TextDataset(texts, labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, drop_last=drop_remainder)
    return loader


# Fetch Train/Valid Dataset
def get_datasets(fold):
    train_df, valid_df = train_test_split(df, test_size=0.2, random_state=CFG.seed, stratify=df.label)

    train_texts = train_df.text.tolist()
    train_labels = train_df.label.tolist()

    train_loader = build_dataset(train_texts, train_labels, batch_size=CFG.batch_size * CFG.replicas,
                                 cache=CFG.cache, shuffle=True, drop_remainder=True, repeat=True)

    valid_texts = valid_df.text.tolist()
    valid_labels = valid_df.label.tolist()

    valid_loader = build_dataset(valid_texts, valid_labels,
                                 batch_size=min(CFG.batch_size * CFG.replicas, len(valid_df)),
                                 cache=CFG.cache, shuffle=False, drop_remainder=True, repeat=False)

    return train_loader, valid_loader


try:
    from kaggle_secrets import UserSecretsClient  # Import UserSecretsClient

    user_secrets = UserSecretsClient()  # Create secrets client instance
    api_key = user_secrets.get_secret("WANDB")  # Get API key from Kaggle secrets
    wandb.login(key=api_key)  # Login to wandb with the API key
    anonymous = None  # Set anonymous mode to None
except:
    anonymous = 'must'  # Set anonymous mode to 'must'
    wandb.login(anonymous=anonymous, relogin=True)


# Logger
def wandb_init(fold):
    config = {k: v for k, v in dict(vars(CFG)).items() if '__' not in k}
    config.update({"fold": int(fold)})
    run = wandb.init(project="llm-fake-text",
                     name=f"fold-{fold}|max_seq-{CFG.sequence_length}|model-{CFG.preset}",
                     config=config,
                     group=CFG.comment,
                     save_code=True)
    return run


def log_wandb():
    wandb.log({'best_auc': best_auc, 'best_loss': best_loss, 'best_epoch': best_epoch})


def get_wb_callbacks(fold):
    return []  # PyTorch does not have direct equivalent for WandB callbacks in Keras


# LR Schedule
def get_lr_scheduler(optimizer, batch_size=8, mode='cos', epochs=10, plot=False):
    lr_start, lr_max, lr_min = 0.6e-6, 0.5e-6 * batch_size, 0.3e-6
    lr_ramp_ep, lr_sus_ep, lr_decay = 1, 0, 0.75

    def lrfn(epoch):  # Learning rate update function
        if epoch < lr_ramp_ep: lr = (lr_max - lr_start) / lr_ramp_ep * epoch + lr_start
        elif epoch < lr_ramp_ep + lr_sus_ep: lr = lr_max
        elif mode == 'exp': lr = (lr_max - lr_min) * lr_decay**(epoch - lr_ramp_ep - lr_sus_ep) + lr_min
        elif mode == 'step': lr = lr_max * lr_decay**((epoch - lr_ramp_ep - lr_sus_ep) // 2)
        elif mode == 'cos':
            decay_total_epochs, decay_epoch_index = epochs - lr_ramp_ep - lr_sus_ep + 3, epoch - lr_ramp_ep - lr_sus_ep
            phase = math.pi * decay_epoch_index / decay_total_epochs
            lr = (lr_max - lr_min) * 0.5 * (1 + math.cos(phase)) + lr_min
        return lr

    lr_lambda = lambda epoch: lrfn(epoch)
    lr_scheduler = LambdaLR(optimizer, lr_lambda)

    if plot:  # Plot lr curve if plot is True
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 5))
        plt.plot(range(epochs), [lrfn(epoch) for epoch in range(epochs)], marker='o')
        plt.xlabel('epoch'); plt.ylabel('lr')
        plt.title('LR Scheduler')
        plt.show()

    return lr_scheduler


# Example usage of LR scheduler
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
lr_scheduler = get_lr_scheduler(optimizer, CFG.batch_size * CFG.replicas, mode='cos', epochs=10, plot=True)
