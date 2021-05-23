from tokenizer import tokenizer
from utils import read_pickle_from_file, CFG_tnt as CFG
import pandas as pd
import nump as np
import os
from torch.utils.data import DataLoader, Dataset
import torch
from patch import uncompress_array, resize_image, repad_image
import cv2


import collections
from collections import defaultdict


def make_fold(mode='train-1'):
    if 'train' in mode:
        df = read_pickle_from_file(CFG.data_dir+'/df_train.more.csv.pickle')
        df_fold = pd.read_csv(CFG.data_dir+'/df_fold.csv')
        # df_fold = pd.read_csv(data_dir+'/df_fold.fine.csv')
        df_meta = pd.read_csv(CFG.data_dir+'/df_train_image_meta.csv')
        df = df.merge(df_fold, on='image_id')
        df = df.merge(df_meta, on='image_id')
        df.loc[:,'path']=f'train_patch16_s{CFG.pixel_scale:.3f}'

        df['fold'] = df['fold'].astype(int)
        #print(df.groupby(['fold']).size()) #404_031
        #print(df.columns)

        fold = int(mode[-1])
        df_train = df[df.fold != fold].reset_index(drop=True)
        df_valid = df[df.fold == fold].reset_index(drop=True)
        return df_train, df_valid

    # Index(['image_id', 'InChI'], dtype='object')
    if 'test' in mode:
        df = pd.read_csv(CFG.data_dir+'/sample_submission.csv')
        # df = pd.read_csv(CFG.data_dir+'/submit_lb3.80.csv')
        df_meta = pd.read_csv(CFG.data_dir+'/df_test_image_meta.csv')
        df = df.merge(df_meta, on='image_id')

        df.loc[:, 'path'] = 'test'
        #df.loc[:, 'InChI'] = '0'
        df.loc[:, 'formula'] = '0'
        df.loc[:, 'text'] =  '0'
        df.loc[:, 'sequence'] = pd.Series([[0]] * len(df))
        df.loc[:, 'length'] = df.InChI.str.len()

        df_test = df
        return df_test

# tokenization, padding, ...
def pad_sequence_to_max_length(sequence, max_length, padding_value):
    batch_size =len(sequence)
    pad_sequence = np.full((batch_size,max_length), padding_value, np.int32)
    for b, s in enumerate(sequence):
        L = len(s)
        pad_sequence[b, :L, ...] = s
    return pad_sequence

def null_augment(r):
    return r


class BmsDatasetNoCache(Dataset):
    def __init__(self, df, tokenizer, transform=None):
        super().__init__()
        self.df = df
        self.tokenizer = tokenizer
        self.file_paths = df['file_path'].values
        self.labels = df['InChI_text'].values
        self.save_dir = 'preprocessed'
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        image = resize_image(image, scale)
        image = repad_image(image, CFG.patch_size)  # remove border and repad
        # Make patches out of it
        label = self.labels[idx]
        label = self.tokenizer.text_to_sequence(label)
        label_length = len(label)
        label_length = torch.LongTensor([label_length])
        return image, torch.LongTensor(label), label_length

class BmsDatasetNoCache(Dataset):
    def __init__(self, df, tokenizer, augment=null_augment):
        super().__init__()
        self.tokenizer = tokenizer
        self.df = df
        self.augment = augment
        self.length = len(self.df)

    def __str__(self):
        string  = ''
        string += '\tlen = %d\n'%len(self)
        string += '\tdf  = %s\n'%str(self.df.shape)

        g = self.df['length'].values.astype(np.int32)//20
        g = np.bincount(g, minlength=14)
        string += '\tlength distribution\n'
        for n in range(14):
            string += '\t\t %3d = %8d (%0.4f)\n'%((n+1)*20,g[n], g[n]/g.sum() )
        return string

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        d = self.df.iloc[index]
        token = d.sequence

        # Have to preprocess all files to patches first
        patch_file = os.path.join(CFG.patch_dump_dir,
                                  f'train_patch16_s{CFG.pixel_scale:.3f}/%s/%s/%s/%s.pickle'%(d.image_id[0],
                                                                                          d.image_id[1], d.image_id[2], d.image_id))
        k = read_pickle_from_file(patch_file)

        patch = uncompress_array(k['patch'])
        patch = np.concatenate([
            np.zeros((1, CFG.patch_size+2*CFG.pixel_pad,
                      CFG.patch_size+2*CFG.pixel_pad), np.uint8),
            patch],0) #cls token

        coord  = k['coord']
        w = k['width' ]
        h = k['height']

        h = h // CFG.patch_size -1
        w = w // CFG.patch_size -1
        coord = np.insert(coord, 0, [h, w], 0) #cls token

        r = {
            'index'    : index,
            'image_id' : d.image_id,
            'InChI'    : d.InChI,
            'd' : d,
            'token' : token,
            'patch' : patch,
            'coord' : coord,
        }
        if self.augment is not None: r = self.augment(r)
        return r

class TestBmsDatasetNoCache(Dataset):
    def __init__(self, df, tokenizer, mode='valid', augment=null_augment):
        super().__init__()
        self.tokenizer = tokenizer
        self.df = df
        self.mode = mode
        self.augment = augment
        self.length = len(self.df)

    def __str__(self):
        string  = ''
        string += '\tlen = %d\n'%len(self)
        string += '\tdf  = %s\n'%str(self.df.shape)

        g = self.df['length'].values.astype(np.int32)//20
        g = np.bincount(g, minlength=14)
        string += '\tlength distribution\n'
        for n in range(14):
            string += '\t\t %3d = %8d (%0.4f)\n'%((n+1)*20,g[n], g[n]/g.sum() )
        return string

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        d = self.df.iloc[index]
        token = d.sequence

        if self.mode == 'test':
            patch_file = os.path.join(CFG.patch_dump_dir,
                                      'test_patch16_s{CFG.pixel_scale:.1f}/%s/%s/%s/%s.pickle'%(d.image_id[0],
                                      d.image_id[1], d.image_id[2], d.image_id))
        elif self.mode == 'valid':
            patch_file = os.path.join(CFG.patch_dump_dir,
                                      'train_patch16_s{CFG.pixel_scale:.3f}/%s/%s/%s/%s.pickle'%(
                                          d.image_id[0],
                                          d.image_id[1], d.image_id[2], d.image_id))
        k = read_pickle_from_file(patch_file)

        patch = uncompress_array(k['patch'])
        # Adding class token
        patch = np.concatenate([
            np.zeros((1, CFG.patch_size+2*CFG.pixel_pad,
                      CFG.patch_size+2*CFG.pixel_pad), np.uint8),
            patch],0) #cls token

        coord  = k['coord']
        w = k['width' ]
        h = k['height']

        h = h // CFG.patch_size -1
        w = w // CFG.patch_size -1
        coord = np.insert(coord, 0, [h, w], 0) #cls token

        r = {
            'index'    : index,
            'image_id' : d.image_id,
            'InChI'    : d.InChI,
            'd' : d,
            'token' : token,
            'patch' : patch,
            'coord' : coord,
        }
        if self.augment is not None: r = self.augment(r)
        return r



class BmsDataset(Dataset):
    def __init__(self, df, tokenizer, augment=null_augment):
        super().__init__()
        self.tokenizer = tokenizer
        self.df = df
        self.augment = augment
        self.length = len(self.df)

    def __str__(self):
        string  = ''
        string += '\tlen = %d\n'%len(self)
        string += '\tdf  = %s\n'%str(self.df.shape)

        g = self.df['length'].values.astype(np.int32)//20
        g = np.bincount(g, minlength=14)
        string += '\tlength distribution\n'
        for n in range(14):
            string += '\t\t %3d = %8d (%0.4f)\n'%((n+1)*20,g[n], g[n]/g.sum() )
        return string

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        d = self.df.iloc[index]
        token = d.sequence

        # Have to preprocess all files to patches first
        patch_file = os.path.join(CFG.patch_dump_dir,
                                  f'train_patch16_s{CFG.pixel_scale:.3f}/%s/%s/%s/%s.pickle'%(d.image_id[0],
                                                                                          d.image_id[1], d.image_id[2], d.image_id))
        k = read_pickle_from_file(patch_file)

        patch = uncompress_array(k['patch'])
        patch = np.concatenate([
            np.zeros((1, CFG.patch_size+2*CFG.pixel_pad,
                      CFG.patch_size+2*CFG.pixel_pad), np.uint8),
            patch],0) #cls token

        coord  = k['coord']
        w = k['width' ]
        h = k['height']

        h = h // CFG.patch_size -1
        w = w // CFG.patch_size -1
        coord = np.insert(coord, 0, [h, w], 0) #cls token

        r = {
            'index'    : index,
            'image_id' : d.image_id,
            'InChI'    : d.InChI,
            'd' : d,
            'token' : token,
            'patch' : patch,
            'coord' : coord,
        }
        if self.augment is not None: r = self.augment(r)
        return r

class TestBmsDataset(Dataset):
    def __init__(self, df, tokenizer, mode='valid', augment=null_augment):
        super().__init__()
        self.tokenizer = tokenizer
        self.df = df
        self.mode = mode
        self.augment = augment
        self.length = len(self.df)

    def __str__(self):
        string  = ''
        string += '\tlen = %d\n'%len(self)
        string += '\tdf  = %s\n'%str(self.df.shape)

        g = self.df['length'].values.astype(np.int32)//20
        g = np.bincount(g, minlength=14)
        string += '\tlength distribution\n'
        for n in range(14):
            string += '\t\t %3d = %8d (%0.4f)\n'%((n+1)*20,g[n], g[n]/g.sum() )
        return string

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        d = self.df.iloc[index]
        token = d.sequence

        if self.mode == 'test':
            patch_file = os.path.join(CFG.patch_dump_dir,
                                      'test_patch16_s{CFG.pixel_scale:.1f}/%s/%s/%s/%s.pickle'%(d.image_id[0],
                                      d.image_id[1], d.image_id[2], d.image_id))
        elif self.mode == 'valid':
            patch_file = os.path.join(CFG.patch_dump_dir,
                                      'train_patch16_s{CFG.pixel_scale:.3f}/%s/%s/%s/%s.pickle'%(
                                          d.image_id[0],
                                          d.image_id[1], d.image_id[2], d.image_id))
        k = read_pickle_from_file(patch_file)

        patch = uncompress_array(k['patch'])
        # Adding class token
        patch = np.concatenate([
            np.zeros((1, CFG.patch_size+2*CFG.pixel_pad,
                      CFG.patch_size+2*CFG.pixel_pad), np.uint8),
            patch],0) #cls token

        coord  = k['coord']
        w = k['width' ]
        h = k['height']

        h = h // CFG.patch_size -1
        w = w // CFG.patch_size -1
        coord = np.insert(coord, 0, [h, w], 0) #cls token

        r = {
            'index'    : index,
            'image_id' : d.image_id,
            'InChI'    : d.InChI,
            'd' : d,
            'token' : token,
            'patch' : patch,
            'coord' : coord,
        }
        if self.augment is not None: r = self.augment(r)
        return r

def null_collate(batch, is_sort_decreasing_length=True):
    collate = defaultdict(list)

    if is_sort_decreasing_length: #sort by decreasing length
        sort  = np.argsort([-len(r['token']) for r in batch])
        batch = [batch[s] for s in sort]

    for r in batch:
        for k, v in r.items():
            collate[k].append(v)
    #----

    batch_size = len(batch)
    collate['length'] = [len(l) for l in collate['token']]

    token  = [np.array(t,np.int32) for t in collate['token']]
    token  = pad_sequence_to_max_length(token, max_length=CFG.max_len, padding_value=tokenizer.stoi['<pad>'])
    collate['token'] = torch.from_numpy(token).long()

    max_of_length = max(collate['length'])
    token_pad_mask  = np.zeros((batch_size, max_of_length, max_of_length))
    for b in range(batch_size):
        L = collate['length'][b]
        token_pad_mask [b, :L, :L] = 1 #+1 for cls_token

    collate['token_pad_mask'] = torch.from_numpy(token_pad_mask).byte()

    collate['num_patch'] = [len(l) for l in collate['patch']]

    max_of_num_patch = max(collate['num_patch'])
    patch_pad_mask  = np.zeros((batch_size, max_of_num_patch, max_of_num_patch))
    patch = np.full((batch_size, max_of_num_patch,
                     CFG.patch_size+2*CFG.pixel_pad,
                     CFG.patch_size+2*CFG.pixel_pad),255) #pad as 255
    coord = np.zeros((batch_size, max_of_num_patch, 2))
    for b in range(batch_size):
        N = collate['num_patch'][b]
        patch[b, :N] = collate['patch'][b]
        coord[b, :N] = collate['coord'][b]
        patch_pad_mask [b, :N, :N] = 1 #+1 for cls_token

    collate['patch'] = torch.from_numpy(patch).float() / 255
    collate['coord'] = torch.from_numpy(coord).long()
    collate['patch_pad_mask' ] = torch.from_numpy(patch_pad_mask).byte()
    return collate
