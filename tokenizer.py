import os
import pickle as pkl
import pandas as pd
import re
from tqdm.auto import tqdm
from utils import CFG
tqdm.pandas()

def split_form(form):
    string = ''
    for i in re.findall(r"[A-Z][^A-Z]*", form):
        elem = re.match(r"\D+", i).group()
        num = i.replace(elem, "")
        if num == "":
            string += f"{elem} "
        else:
            string += f"{elem} {str(num)} "
    return string.rstrip(' ')

def split_form2(form):
    string = ''
    for i in re.findall(r"[a-z][^a-z]*", form):
        elem = i[0]
        num = i.replace(elem, "").replace('/', "")
        num_string = ''
        for j in re.findall(r"[0-9]+[^0-9]*", num):
            num_list = list(re.findall(r'\d+', j))
            assert len(num_list) == 1, f"len(num_list) != 1"
            _num = num_list[0]
            if j == _num:
                num_string += f"{_num} "
            else:
                extra = j.replace(_num, "")
                num_string += f"{_num} {' '.join(list(extra))} "
        string += f"/{elem} {num_string}"
    return string.rstrip(' ')




class Tokenizer(object):
    def __init__(self):
        self.save_path = 'tokenizer.pkl'
        self.save_dir = 'preprocessed'
        self.stoi = {}
        self.itos = {}

    def has_cache(self):
        return os.path.exists(os.path.join(self.save_dir, self.save_path))

    def save(self):
        if not (os.path.exists(self.save_dir)):
            os.makedirs(self.save_dir)
        pkl.dump((self.stoi, self.itos), open(os.path.join(
            self.save_dir, self.save_path), 'wb'))

    def load(self):
        self.stoi, self.itos = pkl.load(
            open(os.path.join(self.save_dir, self.save_path), 'rb'))

    def __len__(self):
        return len(self.stoi)

    def fit_on_texts(self, texts):
        vocab = set()
        for text in texts:
            vocab.update(text.split(' '))
        vocab = sorted(vocab)
        vocab.append('<sos>')
        vocab.append('<eos>')
        vocab.append('<pad>')
        for i, s in enumerate(vocab):
            self.stoi[s] = i
        self.itos = dict([(v, k) for k, v in self.stoi.items()])

    def text_to_sequence(self, text):
        sequence = []
        sequence.append(self.stoi['<sos>'])
        for s in text.split(' '):
            sequence.append(self.stoi[s])
        sequence.append(self.stoi['<eos>'])
        return sequence

    def txts_to_sqncs(self, texts):
        sequences = []
        for text in texts:
            sequence = self.text_to_sequence(text)
            sequences.append(sequence)
        return sequences

    def sequence_to_text(self, sequence):
        return ''.join(list(map(lambda i: self.itos[i], sequence)))

    def sqncs_to_txts(self, sequences):
        texts = []
        for sequence in sequences:
            text = self.sequence_to_text(sequence)
            texts.append(text)
        return texts

    def predict_caption(self, sequence):
        caption = ''
        for i in sequence:
            if i == self.stoi['<eos>'] or i == self.stoi['<pad>']:
                break
            caption += self.itos[i]
        return caption

    def predict_captions(self, sequences):
        captions = []
        for sequence in sequences:
            caption = self.predict_caption(sequence)
            captions.append(caption)
        return captions


tokenizer = Tokenizer()
if os.path.exists(CFG.prep_path + 'train2.pkl') and tokenizer.has_cache():
    train = pd.read_pickle(CFG.prep_path + 'train2.pkl')
    tokenizer.load()
else:
    train = pd.read_csv('train_labels.csv')
    train['InChI_1'] = train['InChI'].progress_apply(lambda x: x.split('/')[1])
    train['InChI_text'] = train['InChI_1'].progress_apply(split_form) + ' ' + \
                            train['InChI'].apply(lambda x: '/'.join(x.split('/')[2:])).progress_apply(split_form2).values
    # ====================================================
    # create tokenizer
    # ====================================================
    tokenizer.fit_on_texts(train['InChI_text'].values)
    tokenizer.save()
    print('Saved tokenizer')
    # ====================================================
    # preprocess train.csv
    # ====================================================
    lengths = []
    tk0 = tqdm(train['InChI_text'].values, total=len(train))
    for text in tk0:
        seq = tokenizer.text_to_sequence(text)
        length = len(seq) - 2       # removed sos and eos
        lengths.append(length)
    train['InChI_length'] = lengths
    train.to_pickle(CFG.prep_path + 'train2.pkl')
    print('Saved preprocessed train.pkl')
