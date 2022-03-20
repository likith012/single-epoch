from augmentations import *
from loss import loss_fn
from model import sleep_model
from train import *
from utils import *

from braindecode.util import set_random_seeds

import os
import numpy as np
import copy
import wandb
import torch
from torch.utils.data import DataLoader, Dataset


PATH = '/scratch/sleepkfold'

# Params
SAVE_PATH = "moco-single-epoch.pth"
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 128
lr = 5e-4
n_epochs = 250
NUM_WORKERS = 5
N_DIM = 256
PROJ_DIM = N_DIM // 2
EPOCH_LEN = 7
m = 0.9995
TEMPERATURE = 1

####################################################################################################

random_state = 1234
sfreq = 100

# Seeds
rng = np.random.RandomState(random_state)
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False
    print(f"GPU available: {torch.cuda.device_count()}")

set_random_seeds(seed=random_state, cuda=device == "cuda")


##################################################################################################


# Extract number of channels and time steps from dataset
n_channels, input_size_samples = (2, 3000)
model_q = sleep_model(n_channels, input_size_samples, n_dim = N_DIM)
model_k = sleep_model(n_channels, input_size_samples, n_dim = N_DIM)

q_encoder = model_q.to(device)
k_encoder = model_k.to(device)

for param_q, param_k in zip(q_encoder.parameters(), k_encoder.parameters()):
    param_k.data.copy_(param_q.data) 
    param_k.requires_grad = False  # not update by gradient

optimizer = torch.optim.Adam(q_encoder.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
criterion = loss_fn(device,T=TEMPERATURE).to(device)

#####################################################################################################


class pretext_data(Dataset):

    def __init__(self, filepath):
        
        self.file_path = filepath
        self.idx = np.array(range(len(self.file_path)))

    def __len__(self):
        return len(self.file_path)

    def __getitem__(self, index):
        
        path = self.file_path[index]
        data = np.load(path)
        pos = data['pos'] #(7, 2, 3000)
        pos_len = len(pos) # 7
        pos = pos[pos_len // 2] # (2, 3000)
        neg = data['neg'][pos_len // 2] # (2, 3000)
        anc = copy.deepcopy(pos)
        
        # augment
        pos = augment(pos)
        anc = augment(anc)
        neg = augment(neg)
       
        return anc, pos, neg
    
class train_data(Dataset):

    def __init__(self, filepath):
        
        self.file_path = filepath
        self.idx = np.array(range(len(self.file_path)))

    def __len__(self):
        return len(self.file_path)

    def __getitem__(self, index):
        
        path = self.file_path[index]
        data = np.load(path)
        
        return data['x'], data['y']
    
    

PRETEXT_FILE = os.listdir(os.path.join(PATH, "pretext"))
PRETEXT_FILE.sort(key=natural_keys)
PRETEXT_FILE = [os.path.join(PATH, "pretext", f) for f in PRETEXT_FILE]

TEST_FILE = os.listdir(os.path.join(PATH, "test"))
TEST_FILE.sort(key=natural_keys)
TEST_FILE = [os.path.join(PATH, "test", f) for f in TEST_FILE]

print(f'Number of pretext files: {len(PRETEXT_FILE)}')
print(f'Number of test records: {len(TEST_FILE)}')

pretext_loader = DataLoader(pretext_data(PRETEXT_FILE), batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

test_records = [np.load(f) for f in TEST_FILE]
test_subjects = dict()

for i, rec in enumerate(test_records):
    if rec['_description'][0] not in test_subjects.keys():
        test_subjects[rec['_description'][0]] = [rec]
    else:
        test_subjects[rec['_description'][0]].append(rec)

test_subjects = list(test_subjects.values())


##############################################################################################################################


wb = wandb.init(
        project="single-epoch-contrastive-loss",
        notes="single-epoch, symmetric loss, 1000 samples, using same projection heads and no batch norm, original simclr",
        save_code=True,
        entity="sleep-staging",
        name="MOCO t-1, same_pos_anc",
    )
wb.save('single-epoch/moco/*.py')
wb.watch([q_encoder],log='all',log_freq=500)

n_queue = 4096 # SIZE of the dictionary queue
queue = torch.rand((n_queue, PROJ_DIM), dtype = torch.float).to(device)
Pretext(q_encoder, k_encoder, m, queue, n_queue, optimizer, n_epochs, criterion, pretext_loader, test_subjects, wb, device, SAVE_PATH, BATCH_SIZE)

wb.finish()
