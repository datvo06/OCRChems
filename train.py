import os
import time
import numpy as np
import torch
import torch.nn as nn
import Levenshtein
import pandas as pd
import pickle
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader, Dataset
from apex import amp
from utils import AverageMeter, CFG, time_since, LOGGER
from dataset import TrainDataset, TestDataset
from tokenizer import tokenizer, train
from albumentations import (
    Compose, OneOf, Normalize, Resize, RandomResizedCrop, RandomCrop,
    HorizontalFlip, VerticalFlip, ShiftScaleRotate, Transpose
)
from albumentations.pytorch import ToTensorV2
from dataset import TrainDataset, bms_collate

from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts,\
    CosineAnnealingLR, ReduceLROnPlateau
from torch.optim import Adam, SGD
from model import Encoder, DecoderWithAttention, Attention
from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUTPUT_DIR = 'saved_model/'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)



def get_transforms(*, data):
    if data == 'train':
        return Compose([
            Resize(CFG.size, CFG.size),
            HorizontalFlip(p=0.5),
            Transpose(p=0.5),
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.5),
            ShiftScaleRotate(p=0.5),
            Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2()
        ])
    elif data == 'valid':
        return Compose([
            Resize(CFG.size, CFG.size),
            Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2()
        ])

def get_score(y_true, y_pred):
    scores = []
    for true, pred in zip(y_true, y_pred):
        score = Levenshtein.distance(true, pred)
        scores.append(score)
    avg_score = np.mean(scores)
    return avg_score

def train_fn(train_loader, encoder, decoder, criterion,
             encoder_optimizer, decoder_optimizer, epoch,
             encoder_scheduler, decoder_scheduler, device):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    encoder.train()
    decoder.train()
    start = end = time.time()
    global_step = 0
    for step, (images, labels, label_lengths) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        images = images.to(device)
        labels = labels.to(device)
        label_lengths = label_lengths.to(device)
        batch_size = images.size(0)
        features = encoder(images)
        predictions, caps_sorted, decode_lengths, alphas, sort_ind = decoder(
            features, labels, label_lengths)
        targets = caps_sorted[:, 1:]
        predictions = pack_padded_sequence(predictions, decode_lengths,
                                           batch_first=True).data
        targets = pack_padded_sequence(targets, decode_lengths,
                                       batch_first=True).data
        loss = criterion(predictions, targets)
        # record losses
        losses.update(loss.item(), batch_size)
        if CFG.gradient_accumulation_steps > 1:
            loss = loss / CFG.gradient_accumulation_steps

        if CFG.apex:
            with amp.scale_loss(loss, decoder_optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        encoder_grad_norm = torch.nn.utils.clip_grad_norm_(
            encoder.parameters(), CFG.max_grad_norm)
        decoder_grad_norm = torch.nn.utils.clip_grad_norm_(
            decoder.parameters(), CFG.max_grad_norm)
        if (step + 1) % CFG.gradient_accumulation_steps == 0:
            encoder_optimizer.step()
            decoder_optimizer.step()
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            global_step += 1

        batch_time.update(time.time() - end)
        end = time.time()
        if step % CFG.print_freq == 0 or step == (len(train_loader) - 1):
            print('Epoch: [{0}][{1}/{2}] '
                  'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                  'Elapsed {remain:s} '
                  'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                  'Encoder Grad: {encoder_grad_norm:.4f}  '
                  'Decoder Grad: {decoder_grad_norm:.4f}  '
                  #'Encoder LR: {encoder_lr:.6f}  '
                  #'Decoder LR: {decoder_lr:.6f}  '
                  .format(
                   epoch+1, step, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses,
                   remain=time_since(start, float(step+1)/len(train_loader)),
                   encoder_grad_norm=encoder_grad_norm,
                   decoder_grad_norm=decoder_grad_norm,
                   #encoder_lr=encoder_scheduler.get_lr()[0],
                   #decoder_lr=decoder_scheduler.get_lr()[0],
                   ))
    return losses.avg


def valid_fn(valid_loader, encoder, decoder, tokenizer, criterion, device):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    encoder.eval()
    decoder.eval()
    text_preds = []
    start = end = time.time()
    for step, (images) in enumerate(valid_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        images = images.to(device)
        batch_size = images.size(0)
        with torch.no_grad():
            features = encoder(images)
            predictions = decoder.predict(features, CFG.max_len, tokenizer)
        predicted_sequence = torch.argmax(predictions.detach().cpu(),
                                          -1).numpy()
        _text_preds = tokenizer.predict_captions(predicted_sequence)
        text_preds.append(_text_preds)
        batch_time.update(time.time() - end)
        end = time.time()
        if step % CFG.print_freq == 0 or step == (len(valid_loader) -1):
            print('EVAL:[{0}/{1}] '
                  'Data {data_time.val: .3f} ({data_time.avg: .3f})'
                  'Elapsed {remain:s} '
                  .format(
                      step, len(valid_loader), batch_time=batch_time,
                      data_time=data_time,
                      remain=time_since(start, float(step+1)/len(valid_loader))
                  ))
    text_preds = np.concatenate(text_preds)
    return text_preds


def train_loop(folds, fold):
    LOGGER.info(f"========= fold: {fold} training =============")

    # =========================================================
    # loader
    # =========================================================
    trn_idxs = folds[folds['fold'] != fold].index
    val_idxs = folds[folds['fold'] == fold].index

    train_folds = folds.loc[trn_idxs].reset_index(drop=True)
    valid_folds = folds.loc[val_idxs].reset_idndex(drop=True)
    valid_labels = valid_folds['InChI'].values
    train_dataset = TrainDataset(train_folds, tokenizer,
                                 transform=get_transforms(data='train'))
    valid_dataset = TestDataset(valid_folds,
                                transform=get_transforms(data='valid'))

    train_loader = DataLoader(train_dataset,
                              batch_size=CFG.batch_size,
                              shuffle=True,
                              num_workers=CFG.num_workers,
                              pin_memory=True,
                              drop_last=True,
                              collate_fn=bms_collate)
    valid_loader = DataLoader(valid_dataset,
                              batch_size=CFG.batch_size,
                              shuffle=False,
                              num_workers=CFG.num_workers,
                              pin_memory=True,
                              drop_last=False)

    def get_scheduler(optimizer):
        if CFG.scheduler == 'ReduceLROnPlateau':
            scheduler = ReduceLROnPlateau(optimizer, mode='min',
                                          factor=CFG.factor,
                                          patience=CFG.patience,
                                          verbose=True,
                                          eps=CFG.eps)
        elif CFG.scheduler == 'CosineAnnealingLR':
            scheduler = CosineAnnealingLR(optimizer,
                                          T_max=CFG.T_max,
                                          eta_min=CFG.min_lr,
                                          last_epoch=-1)
        elif CFG.scheduler == 'CosineAnnealingWarmRestarts':
            scheduler = CosineAnnealingWarmRestarts(
                optimizer, T_0=CFG.T_0, T_mult=1, eta_min=CFG.min_lr,
                last_epoch=-1)
        return scheduler

    encoder = Encoder(CFG.model_name, pretrained=True)
    encoder.to(device)
    encoder_optimizer = Adam(encoder.parameters(), lr=CFG.encoder_lr,
                             weight_decay=CFG.weight_, amsgrad=False)
    encoder_scheduler = get_scheduler(encoder_optimizer)

    decoder = DecoderWithAttention(attention_dim=CFG.attention_dim,
                                   embed_dim=CFG.embed_dim,
                                   decoder_dim=CFG.decoder_dim,
                                   vocab_size=len(tokenizer),
                                   dropout=CFG.dropout,
                                   device=device)
    decoder.to(device)
    decoder_optimizer = Adam(decoder.parameters(),
                             lr=CFG.decoder_lr,
                             weight_decay=CFG.weight_decay,
                             amsgrad=False)
    decoder_scheduler = get_scheduler(decoder_optimizer)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.stoi["<pad>"])

    best_score = np.inf
    best_loss = np.inf

    for epoch in range(CFG.epochs):
        start_time = time.time()
        # train
        avg_loss = train_fn(train_loader, encoder, decoder, criterion,
                            encoder_optimizer, decoder_optimizer,
                            epoch, encoder_scheduler, decoder_scheduler,
                            device)

        # eval
        text_preds = valid_fn(valid_loader, encoder, decoder, tokenizer,
                              criterion, device)
        text_preds = [f"InChI=1S/{text}" for text in text_preds]
        LOGGER.info(f"labels: {valid_labels[:5]}")
        LOGGER.info(f"preds: {text_preds[:5]}")

        # scoring
        score = get_score(valid_labels, text_preds)
        if isinstance(encoder_scheduler, ReduceLROnPlateau):
            encoder_scheduler.step(score)
        elif isinstance(encoder_scheduler, CosineAnnealingLR):
            encoder_scheduler.step()
        elif isinstance(encoder_scheduler, CosineAnnealingWarmRestarts):
            encoder_scheduler.step()

        if isinstance(decoder_scheduler, ReduceLROnPlateau):
            decoder_scheduler.step(score)
        elif isinstance(decoder_scheduler, CosineAnnealingLR):
            decoder_scheduler.step()
        elif isinstance(decoder_scheduler, CosineAnnealingWarmRestarts):
            decoder_scheduler.step()

        elapsed = time.time() - start_time

        LOGGER.info(f'Epoch {epoch + 1} - avg_train_loss: {avg_loss: .4f} time: {elapsed: .0f}s')
        LOGGER.info(f'Epoch {epoch + 1} - Score: {score: .4f}')

        if score < best_score:
            best_score = score
            LOGGER.info(f'Epoch {epoch+1} - Save Best Score: {best_score:.4f} Model')
            torch.save({'encoder': encoder.state_dict(),
                        'encoder_optimizer': encoder_optimizer.state_dict(),
                        'encoder_scheduler': encoder_scheduler.state_dict(),
                        'decoder': decoder.state_dict(),
                        'decoder_optimizer': decoder_optimizer.state_dict(),
                        'decoder_scheduler': decoder_scheduler.state_dict(),
                        'text_preds': text_preds,
                        },
                       OUTPUT_DIR + f'{CFG.model_name}_fold{fold}_best.pth')


def get_train_file_path(image_id):
    return CFG.train_path + '{}/{}/{}/{}.png'.format(
        image_id[0], image_id[1], image_id[2], image_id
    )


print(f'train.shape: {train.shape}')

'''
if CFG.debug:
    CFG.epochs = 1
    train = train.sample(
        n= CFG.samp_size, random_state=CFG.seed).reset_index(drop=True)
    '''

train_dataset = TrainDataset(train, tokenizer,
                             transform=get_transforms(data='train'))
folds = train.copy()
Fold = StratifiedKFold(n_splits=CFG.n_fold, shuffle=True,
                       random_state=CFG.seed)
for n, (train_index, val_index) in enumerate(
        Fold.split(folds, folds['InChI_length'])):
    folds.loc[val_index, 'fold'] = int(n)
folds['fold'] = folds['fold'].astype(int)
train_loop(folds, CFG.trn_fold)
