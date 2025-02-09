import torch
import torch.optim as optim

import torchvision.datasets as dset

from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from accelerate import Accelerator

from encoder_models import *
from loss_functions import *
from utils_clip import *

# from tqdm import tqdm


def train(encoder_1, encoder_2, criterion, dataloader, validloader, learning_rates, device, accelerator, resume_training = False):

    if resume_training:
        checkpoint_path ='/cephyr/NOBACKUP/groups/naiss2024-6-186/nikos/modelSaveFull_32WLux_Cos2_clip_rand_clmp_proj/model_epoch_5.pt' ## todo: put this somewhere else
        encoder_1, encoder_2, opt_state_dct, loss_state_dct, epoch_load, loss, vld_loss = load_model(checkpoint_path, encoder_1, encoder_2, device)


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

    # print(f"Un restored logit_scale: {criterion.logit_scale.item()}")

    optimizer = optim.AdamW(
        optimizer_groups,
        betas=(0.9, 0.98)  # clip-style momentum
    )

    if resume_training:
        optimizer.load_state_dict(opt_state_dct)
        criterion.load_state_dict(loss_state_dct)
        start_epoch = epoch_load
    else:
        start_epoch = 0

    scheduler = CosineAnnealingWarmRestarts(optimizer, 1, 2)
    iters = len(dataloader)
    print (f'Starting from Epoch: {start_epoch}')
    # print(f"Restored logit_scale: {criterion.logit_scale.item()}")


    # for multi node / multi gpu object prep
    dataloader, encoder_1, encoder_2, optimizer, scheduler = accelerator.prepare(dataloader,  encoder_1, encoder_2, optimizer, scheduler)


    num_epochs = 31
    check_vld = True
    ls_store = []
    vld_store = []
    plt_pics = False
    output_dir = '/cephyr/NOBACKUP/groups/naiss2024-6-186/nikos/modelSaveFull_32WLux_Cos2_clip_rand_clmp_proj/'

    for epoch in range(start_epoch, num_epochs):
        encoder_1.train()
        encoder_2.train()
        total_loss = 0.0
        num_batches = 0
        for img, text in dataloader: #tqdm(dataloader, desc="Processing Images", total=len(dataloader)):

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
                   save_path = f"{output_dir}/model_epoch_{epoch}.pt"
                   torch.save({
                        'epoch': epoch,
                        'loss_state_dict': criterion.state_dict(),
                        'encoder_1': encoder_1.state_dict(),
                        'encoder_2': encoder_2.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': ls_store,
                        'vld_loss': vld_store,
                         }, save_path)

    print ("finished")
    return 0

def main():

    accelerator = Accelerator()
    device = accelerator.device

    ## DATA

    coco_dataset_tr = dset.CocoCaptions(
        root='/cephyr/NOBACKUP/groups/naiss2024-6-186/nikos/train2017',
        annFile='/mimer/NOBACKUP/groups/snic2022-6-127/nikos/ViTClip/coco/annotations/captions_train2017.json'
    )

    coco_dataset_val = dset.CocoCaptions(
        root='/mimer/NOBACKUP/groups/snic2022-6-127/nikos/ViTClip/coco/val2017',
        annFile='/mimer/NOBACKUP/groups/snic2022-6-127/nikos/ViTClip/coco/annotations/captions_val2017.json'
    )

    train_dataset = CocoCaptionDataset(coco_dataset_tr, mode="train")
    val_dataset = CocoCaptionDataset(coco_dataset_val, mode="val")

    # train & valid sets
    dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True, num_workers=16, pin_memory=True)
    validloader = DataLoader(val_dataset, batch_size=32, shuffle=False, drop_last=True, num_workers=16, pin_memory=True)

    ## Model hypers

    embed_dim = 128
    criterion = ContrastiveLoss(device, temperature_init=0.07)
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
    train(encoder_1, encoder_2, criterion, dataloader, validloader, learning_rates, device, accelerator, resume_training)


if __name__ == '__main__':
    main()
