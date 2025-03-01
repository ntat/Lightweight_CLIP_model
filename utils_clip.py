import os
import re
import yaml
import torch
import random
import datetime
import torch.nn as nn

import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1 import make_axes_locatable
from transformers import ViTImageProcessor
from torch.utils.data import Dataset
from tqdm import tqdm
from PIL import Image

# custom wrapper for coco
class CocoCaptionDataset(torch.utils.data.Dataset):
    def __init__(self, coco_dataset, mode="train", processor_name="google/vit-base-patch16-224"):
        self.coco_dataset = coco_dataset
        if mode not in ["train", "val", "test"]:
            raise ValueError("Mode must be either 'train' or 'val' or 'test'")
        self.mode = mode
        self.image_processor = ViTImageProcessor.from_pretrained(processor_name)

    def __len__(self):
        return len(self.coco_dataset)

    def __getitem__(self, idx):
        image, captions = self.coco_dataset[idx]
        image = self.image_processor(images=image, return_tensors="pt")
        
        # mode selection
        if isinstance(captions, list) and len(captions) > 0:
            if self.mode == "train":
                # we pick in random one of the 5 availble captions for training (super important for better generalization!)
                caption = random.choice(captions)
            else:
                # for validation just check against the first caption
                # also test set just has a dummy placeholder
                caption = captions[0]
        else:
            caption = ""
        
        return image["pixel_values"].squeeze(0), caption

class MatrixVisualizer:
    def __init__(self, save_dir, total_iterations, percentage=5):
        ## handles the ploting of similarity matrixes at a predefined set of training steps
        ## we always include 1st and last step, percentage should be (0 < percentage <= 100)
        ## if all steps needed --> percentage=100 (not recommended)
        self.__save_dir = save_dir
        os.makedirs(self.__save_dir, exist_ok=True)
        self.__total_iterations = total_iterations
        self.__percentage = percentage
        self.__plot_indices = self.__calculate_plot_indices()

    def __calculate_plot_indices(self):
        ## pre-calculates which indexes should triger a matrix plot
        ## based on percentage of total iterations.
        num_plots = max(2, int(round(self.__total_iterations * (self.__percentage / 100.0))))
        indices = np.linspace(0, self.__total_iterations - 1, num=num_plots, dtype=int)
        unique_indices = sorted(set(indices)) # just in case of duplicates ie very small iters
        return unique_indices

    def should_plot(self, current_iteration): 
        return current_iteration in self.__plot_indices

    def plot_matrix(self, matrix, current_iteration):
        ct = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        matrix_norm = matrix.detach().cpu().numpy()
        min_val, max_val = matrix_norm.min(), matrix_norm.max()

        if min_val == max_val:
            matrix_norm = np.zeros_like(matrix_norm) # just in case of a 0 div
        else:
            matrix_norm = (matrix_norm - min_val) / (max_val - min_val) # norm to (0-1 range)

        fig, ax = plt.subplots()
        im = ax.imshow(matrix_norm, cmap="viridis", interpolation="none")
        ax.set_title(f'Iteration: {current_iteration}')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.1)
        cbar = fig.colorbar(im, cax=cax, label=f'Normalized Similarity - Iteration:{current_iteration}')

        save_path = os.path.join(self.__save_dir, f'img_{current_iteration:09d}_{ct}.png')
        plt.savefig(save_path)
        plt.close()

def load_config(config_path='config.yaml'):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def get_parameter_groups(encoder):
    # we r just gonna skip gains & biases from L2 regularization, as per clip paper
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

def precompute_img_emb(encoder, dataloader, path):
    ## extracts and saves img embeddings in smaller chunks
    ## to the disk in case of low gpu mem
    img_embeddings = []
    idx = 1
    encoder.eval()
    with torch.no_grad():
        for img, _ in tqdm(dataloader, desc="Extracting Image Embeddings..."):
            img_embed = encoder(img)
            img_embeddings.append(img_embed)
            if idx % 5 == 0:
                img_embeddings = torch.cat(img_embeddings, dim=0)
                torch.save(img_embeddings, os.path.join(path, f'test_set_img_{idx}.pt'))
                img_embeddings = []
            idx+=1

    if len(img_embeddings)>0: # left overs from dataloader
        img_embeddings = torch.cat(img_embeddings, dim=0)
        torch.save(img_embeddings, os.path.join(path, f'test_set_img_{idx}.pt'))

def precompute_class_emb(encoder, path):
    ## labels are very small (80 unique) / fit easily on mem
    class_embed = []
    labels = []
    with open(path, 'r') as file:
        for line in file:
            labels.append(line.strip())
    print (f'Total classes: {len(labels)}')

    encoder.eval()
    with torch.no_grad():
        for txt in labels:
            txt_embed = encoder(txt)
            class_embed.append(txt_embed)

    return torch.cat(class_embed, dim=0), labels

def compute_query_embedding(encoder, query, device): # the text prompt
    with torch.no_grad():
        text_embedding = encoder(query)
    return text_embedding

def find_top_k_matches(precom_embbs, query_embedding, logit, k=5, device='cuda'):
    # nroms for cos
    query_embedding = nn.functional.normalize(query_embedding, p=2, dim=1).to(device)
    precom_embbs = nn.functional.normalize(precom_embbs, p=2, dim=1).to(device)

    similarities = torch.matmul(precom_embbs, query_embedding.T) * logit # logit must be in exponated
    similarities = torch.nn.functional.softmax(similarities, dim=0) # turn it into pseudo probs
    similarities = similarities.squeeze()
    top_k_indices = similarities.topk(k).indices
    return top_k_indices, similarities[top_k_indices]

def load_model(checkpoint_path, encoder_1, encoder_2, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # distributed trainining adds 'module.' keyword --> we remove it (it also messes up inference)
    checkpoint['encoder_1'] = {key.replace("module.", ""): value for key, value in checkpoint['encoder_1'].items()}
    checkpoint['encoder_2'] = {key.replace("module.", ""): value for key, value in checkpoint['encoder_2'].items()}
    encoder_1.load_state_dict(checkpoint['encoder_1'])
    encoder_2.load_state_dict(checkpoint['encoder_2'])

    opt_state_dct = checkpoint['optimizer_state_dict']
    sch_state_dct = checkpoint['scheduler_state_dict']
    loss_state_dct = checkpoint['loss_state_dict']
    epoch = checkpoint['epoch']
    tr_loss = checkpoint['tr_loss']
    vld_loss = checkpoint['vld_loss']

    return encoder_1, encoder_2, opt_state_dct, sch_state_dct, loss_state_dct, epoch, tr_loss, vld_loss

def numeric_sort(files):
    return sorted(files, key=lambda x: int(re.search(r'(\d+)', x).group()))

def load_embeddings(save_dir):
    embeddings = []
    files = [f for f in os.listdir(save_dir) if f.endswith(".pt")]
    for file in numeric_sort(files):
        print (file)
        batch_embeddings = torch.load(os.path.join(save_dir, file))
        embeddings.append(batch_embeddings)
    return torch.cat(embeddings, dim=0)

#### plotter functions

def plot_images_and_attentions(images, attentions, title="Images retrieved for prompt"):
    ## 1) resizes all images and attention maps (maps in PIL) to a fixed size of 224x224 
    ## 2) plots the images in the first row and their attention map overlay in the second row
    plt.close('all')
    num_images = len(images)
    
    # resize images and attention maps to 224x224
    target_size = (224, 224)
    images_resized = [image.resize(target_size, Image.LANCZOS) for image in images]
    attentions_resized = [attn.resize(target_size, Image.LANCZOS) for attn in attentions]

    # convert resized images and attentions to numpy for plotting
    images_resized = [np.array(image) for image in images_resized]
    attentions_resized = [np.array(attn) for attn in attentions_resized]
 
    fig, axes = plt.subplots(2, num_images, figsize=(num_images * 4 + 1, 10)) 

    # plot orig images (1st row)
    for j in range(num_images):
        ax = axes[0, j]
        ax.imshow(images_resized[j])
        ax.axis('off')

    # attention map overlays (2nd row)
    attn_images = []
    for j in range(num_images):
        ax = axes[1, j]
        ax.imshow(images_resized[j])
        im = ax.imshow(attentions_resized[j], cmap="viridis", alpha=0.75)  # overlay
        ax.axis('off')
        attn_images.append(im)

    # shared colorbar (last column)
    cax = fig.add_axes([0.91, 0.2, 0.02, 0.6])  # manual colorbar positioning
    cbar = fig.colorbar(attn_images[0], cax=cax, orientation='vertical')
    cbar.set_label('Attention Intensity', fontsize=14, labelpad=15)
    cbar.ax.tick_params(labelsize=12)

    plt.subplots_adjust(hspace=0.001)
    fig.suptitle(title, fontsize=24, y=0.85, va='bottom')
    plt.savefig("output.png", bbox_inches='tight', pad_inches=0.2)
    plt.show()

def plot_images_and_bars(images, texts, probs, title="Zero Shot Classification on CoCo test2017"):
    ## 1) resizes all images to a fixed size of 224x224
    ## 2) plots images in the first row and barplots in the second row.
    plt.close('all')
    num_images = len(images)

    # resize images to 224x224
    target_size = (224, 224)
    images_resized = [image.resize(target_size, Image.LANCZOS) for image in images]

    # convert to numpy for plotting
    images_resized = [np.array(image) for image in images_resized]
    fig, axes = plt.subplots(2, num_images, figsize=(num_images * 4 + 1, 10))

    # plot original images (1st row)
    for j in range(num_images):
        ax = axes[0, j]
        ax.imshow(images_resized[j])
        ax.axis('off')

    # bar plots (2nd row)
    bar_plots = []
    for j in range(num_images):
        ax = axes[1, j]
        bar = ax.barh(texts[j][::-1], probs[j].flip(0).cpu(), color='skyblue')
        bar_plots.append(bar)

    # adjust the space between the rows 
    plt.subplots_adjust(hspace=0.01) 
    plt.tight_layout()

    fig.suptitle(title, fontsize=24, y=0.94, va='bottom')
    plt.savefig("zero_out.png", bbox_inches='tight', pad_inches=0.2)
    plt.show()

def plot_loss(train, vld):
    epochs = list(range(1, len(train) + 1))
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train, label="Training Loss", marker='o')
    plt.plot(epochs, vld, label="Validation Loss", marker='o', linestyle='--')
    plt.title("Training and Validation Loss Over Epochs", fontsize=14)
    plt.xlabel("Epochs", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.legend(fontsize=12)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
