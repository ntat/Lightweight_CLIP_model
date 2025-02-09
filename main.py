import os
import re
import torch
import datetime
import transformers

import matplotlib.pyplot as plt

import pandas as pd
import torch.nn as nn
import torch.optim as optim

from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, SequentialLR, CosineAnnealingWarmRestarts
from torch.utils.data import Subset, DataLoader, Dataset, random_split

from transformers import BertTokenizer, BertModel
from transformers import ViTImageProcessor, ViTModel
from PIL import Image

from tqdm import tqdm
import torchvision.datasets as dset
import torchvision.transforms as transforms
import random

from accelerate import Accelerator
accelerator = Accelerator()

device = accelerator.device

image_processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')


def get_parameter_groups(encoder):
    # we r just gonna skip gains & biases from L2 regularization, as per clip
    decay_params = []
    no_decay_params = []
    
    for name, param in encoder.named_parameters():
        if (
            "bias" in name
            or ("layernorm" in name.lower() and "weight" in name)
            or "embeddings" in name
        ):
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    
    return decay_params, no_decay_params


# custom wrapper for coco
class CocoCaptionDataset(torch.utils.data.Dataset):
    def __init__(self, coco_dataset, mode="train"):
        self.coco_dataset = coco_dataset
        if mode not in ["train", "val"]:
            raise ValueError("Mode must be either 'train' or 'val'")
        self.mode = mode

    def __len__(self):
        return len(self.coco_dataset)

    def __getitem__(self, idx):
        image, captions = self.coco_dataset[idx]
        image = image_processor(images=image, return_tensors="pt")
        
        # mode selection
        if isinstance(captions, list) and len(captions) > 0:
            if self.mode == "train":
                # we pick in random one of the 5 availble captions for training
                caption = random.choice(captions)
            else:
                # for validation just check against the first caption
                caption = captions[0]
        else:
            caption = ""
        
        return image['pixel_values'].squeeze(0), caption



coco_dataset_tr = dset.CocoCaptions(
    root='/cephyr/NOBACKUP/groups/naiss2024-6-186/nikos/train2017',
    annFile='/mimer/NOBACKUP/groups/snic2022-6-127/nikos/ViTClip/coco/annotations/captions_train2017.json'
)


coco_dataset_val = dset.CocoCaptions(
    root='/mimer/NOBACKUP/groups/snic2022-6-127/nikos/ViTClip/coco/val2017',
    annFile='/mimer/NOBACKUP/groups/snic2022-6-127/nikos/ViTClip/coco/annotations/captions_val2017.json'
)


def make_image_from_mat(matrix, epoch):
    ct = datetime.datetime.now()     
    temp = matrix
    matrix_norm = temp.detach().cpu().numpy()
    matrix_norm = (matrix_norm - matrix_norm.min()) / (matrix_norm.max() - matrix_norm.min()) #(0-1 range)
    plt.figure(figsize=(8, 8))
    plt.imshow(matrix_norm, cmap='viridis', interpolation='none')
    plt.title (f'Epoch = {epoch}')
    plt.colorbar(label='Normalized Similarity')
    plt.savefig(f'/proj/berzelius-2024-205/nikos/res_pics//img{ct}.png')



class Transformer_One(nn.Module):
    def __init__(self, vit_model_name, embed_dim, device="cuda"):
        super(Transformer_One, self).__init__()
        self.device = device
        self.model = ViTModel.from_pretrained(vit_model_name).to(self.device)

        hidden_size = self.model.config.hidden_size

        self.projection = nn.Sequential(
            nn.Linear(hidden_size, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )

        for param in self.model.parameters():
            param.requires_grad = True

    def forward(self, images):
        outputs = self.model(images)

        cls_embeddings = outputs.last_hidden_state[:, 0, :]
        embeddings = self.projection(cls_embeddings)  # size=(batch_size, embed_dim)
        return embeddings


class Transformer_Two(nn.Module):
    def __init__(self, bert_model_name, embed_dim, device="cuda"):
        super(Transformer_Two, self).__init__()
        self.device = device 
        self.tokenizer =  BertTokenizer.from_pretrained(bert_model_name) # "bert-base-uncased"
        self.bert = BertModel.from_pretrained(bert_model_name).to(self.device)

        hidden_size = self.bert.config.hidden_size

        self.projection = nn.Sequential(
            nn.Linear(hidden_size, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )

        for param in self.bert.parameters():
            param.requires_grad = True

    def forward(self, input_text):

        encoded_inputs = self.tokenizer(
            input_text, padding=True,
            truncation=True,
            return_tensors="pt").to(self.device)

        outputs = self.bert(**encoded_inputs)

        cls_embeddings = outputs.last_hidden_state[:, 0, :] # cls
        embeddings = self.projection(cls_embeddings)  
        return embeddings


class ContrastiveLoss(nn.Module):
    def __init__(self, temperature_init=0.07):
        super(ContrastiveLoss, self).__init__()
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1.0 / temperature_init)))

    def forward(self, feats_one, feats_two, epoch, valid):

        # normalize for vecotrs for cos similarity (possibly equivalent: add a layer norm after projection)
        feats_one = nn.functional.normalize(feats_one, dim=1, p=2.0)
        feats_two = nn.functional.normalize(feats_two, dim=1, p=2.0)

        # similarity matrix
        similarity_matrix = torch.matmul(feats_one, feats_two.T) * self.logit_scale.exp() #/ self.temperature
        # print (similarity_matrix.shape)

        if epoch % 2 == 0 and valid:
            make_image_from_mat(similarity_matrix, epoch)

        # constrative learning labels
        labels = torch.arange(similarity_matrix.size(0)).to(device)
        
        # cel loss
        loss_i = nn.CrossEntropyLoss()(similarity_matrix, labels)
        loss_t = nn.CrossEntropyLoss()(similarity_matrix.T, labels)

        return (loss_i + loss_t) / 2


def load_model(checkpoint_path, embed_dim, device, vit_trans_name, bert_model_name):

    encoder_1 = Transformer_One(vit_trans_name, embed_dim, device=device).to(device)
    encoder_2 = Transformer_Two(bert_model_name, embed_dim, device=device).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # distributed trainining adds 'module.' keyword --> we remove it
    checkpoint['encoder_1'] = {key.replace("module.", ""): value for key, value in checkpoint['encoder_1'].items()}
    checkpoint['encoder_2'] = {key.replace("module.", ""): value for key, value in checkpoint['encoder_2'].items()}
    encoder_1.load_state_dict(checkpoint['encoder_1'])
    encoder_2.load_state_dict(checkpoint['encoder_2'])

    optimizer = optim.Adam(
        list(encoder_1.parameters()) + list(encoder_2.parameters()), 
        lr=1e-5
    )
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']#+1 # todo fix this during the saving part+1
    loss = checkpoint['loss']

    return encoder_1, encoder_2, optimizer, epoch, loss[-1]

###### Trainer

embed_dim = 128

criterion = ContrastiveLoss(temperature_init=0.07)
criterion = criterion.to(device)

vit_trans_name = 'google/vit-base-patch16-224'
bert_model_name = 'bert-base-uncased'


## data
train_dataset = CocoCaptionDataset(coco_dataset_tr, mode="train")
val_dataset = CocoCaptionDataset(coco_dataset_val, mode="val")


dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True, num_workers=16, pin_memory=True)

## check performnce in vld
validloader = DataLoader(val_dataset, batch_size=32, shuffle=False, drop_last=True, num_workers=16, pin_memory=True)


# Train here
continue_from_check = False ## if true resume from previous state

if continue_from_check == True:
    checkpoint_path ='/cephyr/NOBACKUP/groups/naiss2024-6-186/nikos/modelSave/model_epoch_50.pt'
    encoder_1, encoder_2, optimizer, epoch, loss = load_model(checkpoint_path, embed_dim, device, vit_trans_name, bert_model_name)
    accelerator.print(f"Resuming training the model from epoch {epoch}, with previous loss: {loss:.10f}")
else:
    accelerator.print(f"Starting a Fresh Training Session")
    encoder_1 = Transformer_One(vit_trans_name, embed_dim, device=device)
    encoder_2 = Transformer_Two(bert_model_name, embed_dim, device=device)

image_decay, image_no_decay = get_parameter_groups(encoder_1)
text_decay, text_no_decay = get_parameter_groups(encoder_2)


image_lr = 4e-5       # bit higher LR for image encoder
text_lr = 1e-5        # bit lower LR for text encoder
logit_lr = 1e-4       # highest LR for temperature
weight_decay_im = 0.2 
weight_decay_tx = 0.1

optimizer_groups = [
        # image encoder groups
        {"params": image_decay, "lr": image_lr, "weight_decay": weight_decay_im, "eps": 1e-06},
        {"params": image_no_decay, "lr": image_lr, "weight_decay": 0.0, "eps": 1e-06},

        # text encoder groups
        {"params": text_decay, "lr": text_lr, "weight_decay": weight_decay_tx, "eps": 1e-08},
        {"params": text_no_decay, "lr": text_lr, "weight_decay": 0.0, "eps": 1e-08},

        # temperature parameter (logit_scale)
        {"params": [criterion.logit_scale], "lr": logit_lr, "weight_decay": 0.0, "eps": 1e-08},
]

optimizer = optim.AdamW(
        optimizer_groups,
        betas=(0.9, 0.98)  # clip-style momentum
)


scheduler = CosineAnnealingWarmRestarts(optimizer, 1, 2)
iters = len(dataloader)


# for multi node / multi gpu object prep
dataloader, encoder_1, encoder_2, optimizer, scheduler = accelerator.prepare(dataloader,  encoder_1, encoder_2, optimizer, scheduler)

num_epochs = 32
check_vld = True
ls_store = []
vld_store = []
plt_pics = False

output_dir = '/cephyr/NOBACKUP/groups/naiss2024-6-186/nikos/modelSaveFull_32WLux_Cos2_clip_rand_clmp/'

encoder_1.to(device)
encoder_2.to(device)

for epoch in range(num_epochs):
    encoder_1.train()
    encoder_2.train()
    total_loss = 0.0
    num_batches = 0
    for img, text in dataloader: #tqdm(dataloader, desc="Processing Images", total=len(dataloader)):

        img = img.to(device)

        optimizer.zero_grad()

        img_embeddings = encoder_1(img)
        text_embeddings = encoder_2(text)

        loss = criterion(img_embeddings, text_embeddings, epoch, plt_pics)
        accelerator.backward(loss)

        optimizer.step()

        with torch.no_grad(): # temperature clipping max=ln(100)
            criterion.logit_scale.clamp_(0.0, 4.6052)

        scheduler.step(epoch + num_batches / iters)

#        accelerator.clip_grad_norm_(encoder_1.parameters(), 1.0)
#        accelerator.clip_grad_norm_(encoder_2.parameters(), 1.0)

        total_loss += loss.item()
        num_batches += 1

    avg_loss = total_loss / num_batches
    ls_store.append(avg_loss)
    accelerator.print(f'Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.10f}, Temperature: {criterion.logit_scale}')

    if check_vld: #accelerator.is_main_process:
        encoder_1.eval()
        encoder_2.eval()
        total_valid_loss = 0.0
        num_batches_vld = 0
        with torch.no_grad():
            for tst_img1, tst_txt2 in validloader:
                tst_img1 = tst_img1.to(device)
                img1_embeddings_tst = encoder_1(tst_img1)
                txt2_embeddings_tst = encoder_2(tst_txt2)
                valid_loss = criterion(img1_embeddings_tst, txt2_embeddings_tst, epoch, plt_pics) # over all items in tst
                total_valid_loss += valid_loss.item()
                num_batches_vld += 1

        avg_loss_vld = total_valid_loss / num_batches_vld
        vld_store.append(avg_loss_vld)
        accelerator.print(f'Epoch [{epoch+1}/{num_epochs}], Vld Loss: {avg_loss_vld:.10f}')

        accelerator.wait_for_everyone()

        if accelerator.is_main_process:
               save_path = f"{output_dir}/model_epoch_{epoch+1}.pt"
               torch.save({
                    'epoch': epoch,
                    'encoder_1': encoder_1.state_dict(),
                    'encoder_2': encoder_2.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': ls_store,
                    'vld_loss': vld_store,
                     }, save_path)


print ('finished')


