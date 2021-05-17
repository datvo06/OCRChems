from include import * 
from configure import *

import os
import pickle
import sys
import pandas as pd
import shutil

import builtins
import re

def normalize_inchi(inchi):
    try:
        mol = Chem.MolFromInchi(inchi)
        if mol is not None:
            try:
                inchi = Chem.MolToInchi(mol)
            except:
                pass
    except:
        pass
    return inchi

class YNakamaTokenizer(object):

    def __init__(self, is_load=True):
        self.stoi = {}
        self.itos = {}

        if is_load:
            self.stoi = read_pickle_from_file(data_dir+'/tokenizer.stoi.pickle')
            self.itos = {k: v for v, k in self.stoi.items()}

    def __len__(self):
        return len(self.stoi)

    def build_vocab(self, text):
        vocab = set()
        for t in text:
            vocab.update(t.split(' '))
        vocab = sorted(vocab)
        vocab.append('<sos>')
        vocab.append('<eos>')
        vocab.append('<pad>')
        for i, s in enumerate(vocab):
            self.stoi[s] = i
        self.itos = {k: v for v, k in self.stoi.items()}

    def one_text_to_sequence(self, text):
        sequence = []
        sequence.append(self.stoi['<sos>'])
        for s in text.split(' '):
            sequence.append(self.stoi[s])
        sequence.append(self.stoi['<eos>'])
        return sequence

    def one_sequence_to_text(self, sequence):
        return ''.join(list(map(lambda i: self.itos[i], sequence)))

    def one_predict_to_inchi(self, predict):
        inchi = 'InChI=1S/'
        for p in predict:
            if p == self.stoi['<eos>'] or p == self.stoi['<pad>']:
                break
            inchi += self.itos[p]
        return inchi

    # ---
    def text_to_sequence(self, text):
        sequence = [
            self.one_text_to_sequence(t)
            for t in text
        ]
        return sequence

    def sequence_to_text(self, sequence):
        text = [
            self.one_sequence_to_text(s)
            for s in sequence
        ]
        return text

    def predict_to_inchi(self, predict):
        inchi = [
            self.one_predict_to_inchi(p)
            for p in predict
        ]
        return inchi

def compute_lb_score(predict, truth):
    score = []
    for p, t in zip(predict, truth):
        s = Levenshtein.distance(p, t)
        score.append(s)
    score = np.array(score)
    return score

class Struct(object):
    def __init__(self, is_copy=False, **kwargs):
        self.add(is_copy, **kwargs)

    def add(self, is_copy=False, **kwargs):
        #self.__dict__.update(kwargs)

        if is_copy == False:
            for key, value in kwargs.items():
                setattr(self, key, value)
        else:
            for key, value in kwargs.items():
                try:
                    setattr(self, key, copy.deepcopy(value))
                    #setattr(self, key, value.copy())
                except Exception:
                    setattr(self, key, value)

    def drop(self,  missing=None, **kwargs):
        drop_value = []
        for key, value in kwargs.items():
            try:
                delattr(self, key)
                drop_value.append(value)
            except:
                drop_value.append(missing)
        return drop_value

    def __str__(self):
        text =''
        for k,v in self.__dict__.items():
            text += '\t%s : %s\n'%(k, str(v))
        return text



# log ------------------------------------
def remove_comments(lines, token='#'):
    """ Generator. Strips comments and whitespace from input lines.
    """

    l = []
    for line in lines:
        s = line.split(token, 1)[0].strip()
        if s != '':
            l.append(s)
    return l


def open(file, mode=None, encoding=None):
    if mode == None: mode = 'r'

    if '/' in file:
        if 'w' or 'a' in mode:
            dir = os.path.dirname(file)
            if not os.path.isdir(dir):  os.makedirs(dir)

    f = builtins.open(file, mode=mode, encoding=encoding)
    return f


def remove(file):
    if os.path.exists(file): os.remove(file)


def empty(dir):
    if os.path.isdir(dir):
        shutil.rmtree(dir, ignore_errors=True)
    else:
        os.makedirs(dir)


# http://stackoverflow.com/questions/34950201/pycharm-print-end-r-statement-not-working
class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout  #stdout
        self.file = None

    def open(self, file, mode=None):
        if mode is None: mode ='w'
        self.file = open(file, mode)

    def write(self, message, is_terminal=1, is_file=1 ):
        if '\r' in message: is_file=0

        if is_terminal == 1:
            self.terminal.write(message)
            self.terminal.flush()
            #time.sleep(1)

        if is_file == 1:
            self.file.write(message)
            self.file.flush()

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass

# io ------------------------------------
def write_list_to_file(list_file, strings):
    with open(list_file, 'w') as f:
        for s in strings:
            f.write('%s\n'%str(s))
    pass


def read_list_from_file(list_file, comment='#'):
    with open(list_file) as f:
        lines  = f.readlines()
    strings=[]
    for line in lines:
        if comment is not None:
            s = line.split(comment, 1)[0].strip()
        else:
            s = line.strip()
        if s != '':
            strings.append(s)
    return strings

def read_pickle_from_file(pickle_file):
    with open(pickle_file,'rb') as f:
        x = pickle.load(f)
    return x

def write_pickle_to_file(pickle_file, x):
    with open(pickle_file, 'wb') as f:
        pickle.dump(x, f, pickle.HIGHEST_PROTOCOL)

# etc ------------------------------------
def time_to_str(t, mode='min'):
    if mode=='min':
        t  = int(t)/60
        hr = t//60
        min = t%60
        return '%2d hr %02d min'%(hr,min)

    elif mode=='sec':
        t   = int(t)
        min = t//60
        sec = t%60
        return '%2d min %02d sec'%(min,sec)

    else:
        raise NotImplementedError


def np_float32_to_uint8(x, scale=255):
    return (x*scale).astype(np.uint8)

def np_uint8_to_float32(x, scale=255):
    return (x/scale).astype(np.float32)


def int_tuple(x):
    return tuple( [int(round(xx)) for xx in x] )


def df_loc_by_list(df, key, values):
    df = df.loc[df[key].isin(values)]
    df = df.assign(sort = pd.Categorical(df[key], categories=values, ordered=True))
    df = df.sort_values('sort')
    #df = df.reset_index()
    df = df.drop('sort', axis=1)
    return  df
