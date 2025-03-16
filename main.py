import os
import torch
import torch.optim as optim

import torchvision.datasets as dset

from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from accelerate import DistributedDataParallelKwargs
from accelerate import Accelerator
from itertools import count

from encoder_models import *
from loss_functions import *
from utils_clip import *

def train(encoder_1, encoder_2, criterion, dataloader, validloader, learning_rates, device, accelerator, config, resume_training = False):

    if resume_training:
        checkpoint_path = config['paths']['checkpoint']
        encoder_1, encoder_2, opt_state_dct, sch_state_dct, loss_state_dct, epoch_load, _, _ = load_model(checkpoint_path, encoder_1, encoder_2, device)
        start_epoch = epoch_load+1
    else:
        epoch_load = 0
        start_epoch = 0


    image_decay, image_no_decay = get_parameter_groups(encoder_1)
    text_decay, text_no_decay = get_parameter_groups(encoder_2)

    optimizer_groups = [
        # image encoder groups
        {"params": image_decay, "lr": learning_rates["image_lr"], "weight_decay": learning_rates["weight_decay_im"], "eps": 1e-06},
        {"params": image_no_decay, "lr": learning_rates["image_lr"], "weight_decay": 0.0, "eps": 1e-06},

        # text encoder groups
        {"params": text_decay, "lr": learning_rates["text_lr"], "weight_decay": learning_rates["weight_decay_tx"], "eps": 1e-08},
        {"params": text_no_decay, "lr": learning_rates["text_lr"], "weight_decay": 0.0, "eps": 1e-08},

        # temperature parameter (logit_scale)
        {"params": [criterion.logit_scale], "lr": learning_rates["logit_lr"], "weight_decay": 0.0, "eps": 1e-08},

    ]

    optimizer = optim.AdamW(
        optimizer_groups,
        betas=(0.9, 0.98)  # clip-style momentum
    )

    scheduler = CosineAnnealingWarmRestarts(optimizer, 1, 2)

    if resume_training:
        optimizer.load_state_dict(opt_state_dct)
        scheduler.load_state_dict(sch_state_dct)
        criterion.load_state_dict(loss_state_dct)

    accelerator.print (f'Starting from Epoch: {start_epoch}')

    # for multi node / multi gpu object prep
    dataloader, encoder_1, encoder_2, optimizer, scheduler = accelerator.prepare(dataloader,  encoder_1, encoder_2, optimizer, scheduler)


    num_epochs = epoch_load+31
    iters = len(dataloader)
    iteration_counter = count(start=0)
    check_vld = True
    plt_pics = True
    tr_store = []
    vld_store = []
    output_dir = config['paths']['output_dir']

    visualizer = MatrixVisualizer(config['paths']['mat_similarity_plots'], (num_epochs-start_epoch)*iters, percentage=5)

    for epoch in range(start_epoch, num_epochs):
        encoder_1.train()
        encoder_2.train()
        total_loss = 0.0
        num_batches = 0
        for img, text in dataloader:

            current_iteration = next(iteration_counter)
            optimizer.zero_grad()

            img_embeddings = encoder_1(img)
            text_embeddings = encoder_2(text)

            loss, similarity_matrix = criterion(img_embeddings, text_embeddings)
            accelerator.backward(loss)

            #accelerator.clip_grad_norm_(encoder_1.parameters(), 1.0)
            #accelerator.clip_grad_norm_(encoder_2.parameters(), 1.0)

            optimizer.step()

            with torch.no_grad(): # temperature clipping max=ln(100)
                criterion.logit_scale.clamp_(0.0, 4.6052)

            scheduler.step(epoch + num_batches / iters)

            if accelerator.is_main_process and plt_pics and visualizer.should_plot(current_iteration):
                visualizer.plot_matrix(similarity_matrix, current_iteration)

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        tr_store.append(avg_loss)
        accelerator.print(f'Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.10f}, Temperature: {criterion.logit_scale}')

        if check_vld: # and accelerator.is_main_process:
            avg_loss_vld = run_validation(encoder_1, encoder_2, validloader, criterion)
            vld_store.append(avg_loss_vld)
            accelerator.print(f'Epoch [{epoch+1}/{num_epochs}], Vld Loss: {avg_loss_vld:.10f}')

        accelerator.wait_for_everyone()

        if accelerator.is_main_process and avg_loss_vld <= min(vld_store):
            save_checkpoint(output_dir, epoch, encoder_1, encoder_2, optimizer, scheduler, criterion, tr_store, vld_store)

    accelerator.print("finished")
    return 0

def main():

    accelerator = Accelerator(kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)])
    device = accelerator.device

    config = load_config()

    ## CoCo dataset
    coco_dataset_tr = dset.CocoCaptions(
        root=config['data']['train_root'],
        annFile=config['data']['train_ann']
    )

    coco_dataset_val = dset.CocoCaptions(
        root=config['data']['val_root'],
        annFile=config['data']['val_ann']
    )

    train_dataset = CocoCaptionDataset(coco_dataset_tr, mode="train")
    val_dataset = CocoCaptionDataset(coco_dataset_val, mode="val")

    # train & valid sets
    dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True, num_workers=16, pin_memory=True)
    validloader = DataLoader(val_dataset, batch_size=32, shuffle=False, drop_last=True, num_workers=16, pin_memory=True)

    ## Model hypers

    embed_dim = 128
    # criterion = SigLipLoss(temperature_init=0.1, device=device)
    criterion = ContrastiveLoss(temperature_init=0.07, device=device)
    criterion = criterion.to(device)

    vit_trans_name = "google/vit-base-patch16-224"
    bert_model_name = "bert-base-uncased"

    learning_rates = {
        "image_lr": 4e-5,       # bit higher LR for image encoder
        "text_lr": 1e-5,        # bit lower LR for text encoder
        "logit_lr": 1e-4,       # highest LR for temperature
        "weight_decay_im": 0.2, # weight decay for image encoder
        "weight_decay_tx": 0.1  # weight decay for text encoder
    }

    encoder_1 = Transformer_One(vit_trans_name, embed_dim, device=device)
    encoder_2 = Transformer_Two(bert_model_name, embed_dim, device=device)

    resume_training = False

    # just rain the model
    train(encoder_1, encoder_2, criterion, dataloader, validloader, learning_rates, device, accelerator, config, resume_training)


if __name__ == '__main__':
    main()
