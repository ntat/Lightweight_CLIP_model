import os
import re
import torch
import datetime
import transformers

import matplotlib.pyplot as plt

import pandas as pd
import torch.nn as nn
import torch.optim as optim

from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import Subset, DataLoader, Dataset, random_split

from transformers import BertTokenizer, BertModel
from transformers import ViTImageProcessor, ViTModel
from PIL import Image

from tqdm import tqdm
import torchvision.datasets as dset
import torchvision.transforms as transforms

from accelerate import Accelerator
accelerator = Accelerator()

device = accelerator.device

image_processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')

def get_parameter_groups(encoder):
    ## fix parameter groups: no decay for layernorm, biases & embd layers
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


# custom wrapper
class SingleCaptionDataset(torch.utils.data.Dataset):
    def __init__(self, coco_dataset):
        self.coco_dataset = coco_dataset

    def __len__(self):
        return len(self.coco_dataset)

    def __getitem__(self, idx):
        image, captions = self.coco_dataset[idx]
        image = image_processor(images=image, return_tensors="pt")
        caption = captions[0] if isinstance(captions, list) and len(captions) > 0 else ""
        return image['pixel_values'].squeeze(0), caption

coco_dataset_tr = dset.CocoCaptions(
    root='/mimer/NOBACKUP/groups/snic2022-6-127/nikos/ViTClip/coco/train2017',
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
    plt.savefig(f'/mimer/NOBACKUP/groups/snic2022-6-127/nikos/ViTClip/res_pics//img{ct}.png')



class Transformer_One(nn.Module):
    def __init__(self, vit_model_name, embed_dim):
        super(Transformer_One, self).__init__()
        self.device = device
        self.model = ViTModel.from_pretrained(vit_model_name).to(self.device)

        self.projection = nn.Sequential(
            nn.Linear(768, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )

        for param in self.model.parameters():
            param.requires_grad = True

    def forward(self, images):
        outputs = self.model(images)

        cls_embeddings = outputs.last_hidden_state[:, 0, :]
        embeddings = self.projection(cls_embeddings)  # shape=(batch_size, embed_dim)
        return embeddings


class Transformer_Two(nn.Module):
    def __init__(self, bert_model_name, embed_dim):
        super(Transformer_Two, self).__init__()

        self.device = device 
        self.tokenizer =  BertTokenizer.from_pretrained(bert_model_name) # "bert-base-uncased"
        self.bert = BertModel.from_pretrained(bert_model_name).to(self.device)

        self.projection = nn.Sequential(
            nn.Linear(768, embed_dim),
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

        cls_embeddings = outputs.last_hidden_state[:, 0, :] 
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
        similarity_matrix = torch.matmul(feats_one, feats_two.T) * self.logit_scale.exp()
        # print (similarity_matrix.shape)

        if epoch % 2 == 0 and valid:
            make_image_from_mat(similarity_matrix, epoch)

        # constrative learning labels
        labels = torch.arange(similarity_matrix.size(0)).to(device)

        # cel loss
        loss_i = nn.CrossEntropyLoss()(similarity_matrix, labels)
        loss_t = nn.CrossEntropyLoss()(similarity_matrix.T, labels)
#        print (self.logit_scale)

        return (loss_i + loss_t) / 2


def load_model(checkpoint_path, embed_dim, device, vit_trans_name, bert_model_name):

    encoder_1 = Transformer_One(vit_trans_name, embed_dim).to(device)
    encoder_2 = Transformer_Two(bert_model_name, embed_dim).to(device)

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
train_dataset = SingleCaptionDataset(coco_dataset_tr)
val_dataset = SingleCaptionDataset(coco_dataset_val)


dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True, num_workers=16, pin_memory=True)

## check performnce in vld
validloader = DataLoader(val_dataset, batch_size=32, shuffle=False, drop_last=True, num_workers=16, pin_memory=True)


# Train here
continue_from_check = False ## if true resume from previous state

if continue_from_check == True:
    checkpoint_path ='/mimer/NOBACKUP/groups/snic2022-6-127/nikos/ViTClip/modelSave/model_epoch_50.pt'
    encoder_1, encoder_2, optimizer, epoch, loss = load_model(checkpoint_path, embed_dim, device, vit_trans_name, bert_model_name)
    accelerator.print(f"Resuming training the model from epoch {epoch}, with previous loss: {loss:.10f}")
else:
    accelerator.print(f"Starting a Fresh Training Session")
    encoder_1 = Transformer_One(vit_trans_name, embed_dim)
    encoder_2 = Transformer_Two(bert_model_name, embed_dim)

    encoder_1.to(device)
    encoder_2.to(device)


    image_decay, image_no_decay = get_parameter_groups(encoder_1)
    text_decay, text_no_decay = get_parameter_groups(encoder_2)

    image_lr = 3e-5      
    text_lr = 1e-5        
    logit_lr = 1e-4       
    weight_decay_im = 0.2 
    weight_decay_tx = 0.1

    optimizer_groups = [
        # ViT encoder groups
        {"params": image_decay, "lr": image_lr, "weight_decay": weight_decay_im},
        {"params": image_no_decay, "lr": image_lr, "weight_decay": 0.0},

        # bert uncased encoder groups
        {"params": text_decay, "lr": text_lr, "weight_decay": weight_decay_tx},
        {"params": text_no_decay, "lr": text_lr, "weight_decay": 0.0},

        # temp parameter (logit_scale)
        {"params": [criterion.logit_scale], "lr": logit_lr, "weight_decay": 0.0},
    ]

    optimizer = optim.AdamW(
        optimizer_groups,
        betas=(0.9, 0.98)  # CLIP-style momentum
    )


def lr_lambda(step):
    warmup_steps = 10000  # warmup steps (batch size quite small)
    if step < warmup_steps:
        return float(step) / warmup_steps
    else:
        return 1.0  # no change after warmup

scheduler = LambdaLR(optimizer, lr_lambda)

# for multi node / multi gpu object prep
dataloader, encoder_1, encoder_2, optimizer, scheduler = accelerator.prepare(dataloader,  encoder_1, encoder_2, optimizer, scheduler)

num_epochs = 20
check_vld = True
ls_store = []
vld_store = []

output_dir = '/mimer/NOBACKUP/groups/snic2022-6-127/nikos/ViTClip/modelSave/'

for epoch in range(num_epochs):
    encoder_1.to(device)
    encoder_2.to(device)

    encoder_1.train()
    encoder_2.train()
    total_loss = 0.0
    num_batches = 0
    for img, text in dataloader: #tqdm(dataloader, desc="Processing Images", total=len(dataloader)):
        # get embbs
        img = img.to(device)
        img_embeddings = encoder_1(img)
        text_embeddings = encoder_2(text)

        loss = criterion(img_embeddings, text_embeddings, epoch, False)
        # print (loss)
        optimizer.zero_grad()
        scheduler.step()
        accelerator.backward(loss)
        #loss.backward()
        # updt
        optimizer.step()
        # for printing
        total_loss += loss.item()
        num_batches += 1

    avg_loss = total_loss / num_batches
    ls_store.append(avg_loss)
    accelerator.print(f'Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.10f}')

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
                valid_loss = criterion(img1_embeddings_tst, txt2_embeddings_tst, epoch, False) # over all items in vld
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