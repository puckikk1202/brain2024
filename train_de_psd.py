import torch
import torch.nn as nn
import numpy as np
from dataset import EEGDataset
from model.vit import ViT, SeqTransformer, cnn_classifier
from model.eeg_mae import MAEforEEG, eeg_encoder
from torch.utils.data import DataLoader, Subset
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from cfg import config
from tqdm import tqdm
import os
import json
import argparse
import wandb
import matplotlib.pyplot as plt


    
wandb.init(project="mca_clip", entity="ohicarip")

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
data_dir = './datasets'
anno_dir = os.path.join(data_dir, 'annotation_order.json')
eeg_dir = data_dir

def log_image_to_wandb(data, inout="input", train_or_test='train', epoch=0):
    """
    Function to plot the input data and log the image to WandB.
    
    Args:
    - data: numpy array of shape [ch, seq]
    - title: Title of the plot (optional)
    
    Returns:
    - None
    """
    seq, ch = data.shape
    
    # Plot the image
    fig = plt.figure(figsize=(15, 5))
    plt.imshow(data, cmap='viridis', aspect='auto')
    plt.colorbar()
    plt.xlabel(f'Seq (Length: {seq})')
    plt.ylabel(f'Ch (Channels: {ch})')
    
    # Save the plot to a file buffer
    plt.savefig("temp_image.png")
    
    # Log the image to WandB
    wandb.log({f"epoch {train_or_test} {inout}": wandb.Image(fig)})
    
    # Close the plot to free resources
    plt.close()

def calc_vq_loss(pred, target, quant_loss, quant_loss_weight=1.0, alpha=1.0):
    """ function that computes the various components of the VQ loss """
    rec_loss = nn.L1Loss()(pred, target)
    ## loss is VQ reconstruction + weighted pre-computed quantization loss
    quant_loss = quant_loss.mean()
    return quant_loss * quant_loss_weight + rec_loss, [rec_loss, quant_loss]

# Load the dataset
eeg_dataset = EEGDataset(eeg_dir, anno_dir, is_staring=True, is_classification=False, is_binary=True, is_clip=True, compute_de_psd=True)
split_ratio = 0.8
train_size = int(split_ratio * len(eeg_dataset))
test_size = len(eeg_dataset) - train_size
indices = list(range(len(eeg_dataset)))
train_indices = indices[:train_size]  
test_indices = indices[train_size:]

train_dataset = Subset(eeg_dataset, train_indices)
test_dataset = Subset(eeg_dataset, test_indices)
# train_dataset, test_dataset = torch.utils.data.split(eeg_dataset, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
print("train_set:", len(train_dataset), "test_set:", len(test_dataset))
# Define the model
# model = ViT(patch_size=64, emb_dim=768, num_classes=8).to(device)
# model = SeqTransformer(num_classes=8, ch=5, seq_len=64).to(device)
model = cnn_classifier().to(device)
# model = eeg_encoder().to(device)
# state_dict = torch.load('./checkpoints/mae/model_latest.pth')
# model.load_checkpoint(state_dict)

def get_parser():
    parser = argparse.ArgumentParser(description=' ')
    parser.add_argument('--config', type=str, default='./model/cfg.yaml', help='config file')
    parser.add_argument('opts', help=' ', default=None,
                        nargs=argparse.REMAINDER)
    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
    return cfg

def get_model(cfg):
    if cfg.arch == 'eeg_vqvae':
        from model.vqvae import VQAutoEncoder as Model
        model = Model(args=cfg)
    else:
        raise Exception('architecture not supported yet'.format(cfg.arch))
    return model

# args = get_parser()
# vae = get_model(args).to(device)
# vae.load_state_dict(torch.load('./checkpoints/vae/model_latest.pth'))
# model = SeqTransformer(num_classes=8).to(device)
optimizer = AdamW(model.parameters(), lr=1e-4)
scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
# criterion = nn.CrossEntropyLoss()
criterion = nn.MSELoss()
# Training loop
num_epochs = 100
best_loss = float('inf')
for epoch in range(num_epochs):
    model.train()
    # vae.eval()
    train_loss = 0.0
    correct = 0
    total = 0
    for i, data in tqdm(enumerate(train_loader)):
        data['eeg_sample'], data['mca_feat'], data['clip_feat'], data['label'] = data['eeg_sample'].to(device, dtype=torch.float32), data['mca_feat'].to(device, dtype=torch.float32), data['clip_feat'].to(device, dtype=torch.float32), data['label'].to(device, dtype=torch.float32)
        data['de_feat'], data['psd_feat'] = data['de_feat'].to(device, dtype=torch.float32), data['psd_feat'].to(device, dtype=torch.float32)

        optimizer.zero_grad()

        # print(mca_feat.shape)
        output = model(data['de_feat'], data['psd_feat'])
        loss = criterion(output, data['label'])
        # loss = criterion(output, data['clip_feat'])

        if i % 100 == 0:
            log_image_to_wandb(data['eeg_sample'][0].cpu().detach().numpy(), epoch=epoch, inout="input", train_or_test='train')
            # log_image_to_wandb(eeg_feat[0].cpu().detach().numpy(), epoch=epoch, inout="feature", train_or_test='train')

        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()

        _, predicted = output.max(1)
        _, label = data['label'].max(1)
        total += label.size(0)
        correct += predicted.eq(label).sum().item()
    
    # train_accuracy = 100 * (correct / total)
    train_loss /= len(train_loader)
    # Validation loop
    model.eval()
    # vae.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for i, data in tqdm(enumerate(test_loader)):
            data['eeg_sample'], data['mca_feat'], data['clip_feat'], data['label'] = data['eeg_sample'].to(device, dtype=torch.float32), data['mca_feat'].to(device, dtype=torch.float32), data['clip_feat'].to(device, dtype=torch.float32), data['label'].to(device, dtype=torch.float32)
            data['de_feat'], data['psd_feat'] = data['de_feat'].to(device, dtype=torch.float32), data['psd_feat'].to(device, dtype=torch.float32)
  

            output = model(data['de_feat'], data['psd_feat'])
            loss = criterion(output, data['label'])
            # loss = criterion(output, data['clip_feat'])

            if i % 10 == 0:
                log_image_to_wandb(data['eeg_sample'][0].cpu().detach().numpy(), epoch=epoch, inout="input", train_or_test='test')
                # log_image_to_wandb(eeg_feat[0].cpu().detach().numpy(), epoch=epoch, inout="feature", train_or_test='test')
            
            test_loss += loss.item()
            _, predicted = output.max(1)
            _, label = data['label'].max(1)
            total += label.size(0)
            correct += predicted.eq(label).sum().item()
        
    # test_accuracy = 100 * (correct / total)
    test_loss /= len(test_loader)
    if epoch > 30 and test_loss < best_loss:
        best_loss = test_loss
        print(f'best model saved at epoch {epoch}.')
        torch.save(model.state_dict(), './checkpoints/mcaformer/model_best.pth')
    torch.save(model.state_dict(), './checkpoints/mcaformer/model_latest.pth')

    # print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')
    wandb.log({'train_loss': train_loss, 'test_loss': test_loss})
    # wandb.log({'train_accuracy': train_accuracy, 'test_accuracy': test_accuracy})
    # print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, Test Loss: {test_loss:.4f}, Test Acc: {test_accuracy:.2f}%')
    print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')
    scheduler.step()
