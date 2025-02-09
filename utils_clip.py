import os
import torch
import random
import datetime

import numpy as np
import matplotlib.pyplot as plt

from transformers import ViTImageProcessor
from torch.utils.data import Dataset

# custom wrapper for coco
class CocoCaptionDataset(torch.utils.data.Dataset):
    def __init__(self, coco_dataset, mode="train", processor_name="google/vit-base-patch16-224"):
        self.coco_dataset = coco_dataset
        if mode not in ["train", "val"]:
            raise ValueError("Mode must be either 'train' or 'val'")
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
                # we pick in random one of the 5 availble captions for training
                caption = random.choice(captions)
            else:
                # for validation just check against the first caption
                caption = captions[0]
        else:
            caption = ""
        
        return image["pixel_values"].squeeze(0), caption


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


def make_image_from_mat(matrix, epoch):
    ct = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]

    matrix_norm = matrix.detach().cpu().numpy()
    min_val, max_val = matrix_norm.min(), matrix_norm.max()

    if min_val == max_val:
        matrix_norm = np.zeros_like(matrix_norm)  # just in case of a 0 div
    else:
        matrix_norm = (matrix_norm - min_val) / (max_val - min_val)  # norm to (0-1 range)

    plt.figure(figsize=(8, 8))
    plt.imshow(matrix_norm, cmap="viridis", interpolation="none")
    plt.title(f'Epoch = {epoch}')
    plt.colorbar(label="Normalized Similarity")

    # save the sim matrix
    save_path = os.path.join("/proj/berzelius-2024-205/nikos/res_pics", f'img_{ct}.png')
    plt.savefig(save_path)
    plt.close()


def load_model(checkpoint_path, encoder_1, encoder_2, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # distributed trainining adds 'module.' keyword --> we remove it (it also messes up inference)
    checkpoint['encoder_1'] = {key.replace("module.", ""): value for key, value in checkpoint['encoder_1'].items()}
    checkpoint['encoder_2'] = {key.replace("module.", ""): value for key, value in checkpoint['encoder_2'].items()}
    encoder_1.load_state_dict(checkpoint['encoder_1'])
    encoder_2.load_state_dict(checkpoint['encoder_2'])

    loss_state_dct = checkpoint['loss_state_dict']
    opt_state_dct = checkpoint['optimizer_state_dict']
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    vld_loss = checkpoint['vld_loss']

    return encoder_1, encoder_2, opt_state_dct, loss_state_dct, epoch, loss, vld_loss
