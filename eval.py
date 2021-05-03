from __future__ import print_function
import tqdm
import torch
from utils import CFG
from Tokenizer import tokenizer
import numpy as np
from model import Encoder, DecoderWithAttention
from dataset import TestDataset, get_transforms
from torch.utils.data import DataLoader, Dataset
import pandas as pd

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def inference(test_loader, encoder, decoder, tokenizer, device):
    encoder.eval()
    decoder.eval()
    text_preds = []
    tk0 = tqdm(test_loader, total=len(test_loader))
    for images in tk0:
        images = images.to(device)
        with torch.no_grad():
            features = encoder(images)
            predictions = decoder.predict(features, CFG.max_len, tokenizer)
        predicted_sequence = torch.argmax(predictions.detach().cpu(), -1).numpy()
        _text_preds = tokenizer.predict_captions(predicted_sequence)
        text_preds.append(_text_preds)
    text_preds = np.concatenate(text_preds)
    return text_preds


def get_test_file_path(image_id):
    return CFG.train_path + "{}/{}/{}/{}.png".format(
        image_id[0], image_id[1], image_id[2], image_id
    )

if __name__ == '__main__':
    test = pd.read_csv('sample_submission.csv')
    test['file_path'] = test['image_id'].apply(get_test_file_path)
    print(f'test.shape: {test.shape}')
    states = torch.load(CFG.pred_model, map_location=torch.device('cpu'))
    encoder = Encoder(CFG.model_name, pretrained=False)
    encoder.load_state_dict(states['encoder'])
    decoder = DecoderWithAttention(attention_dim=CFG.attention_dim,
                                   embed_dim=CFG.embed_dim,
                                   decoder_dim=CFG.decoder_dim,
                                   vocab_size=len(tokenizer),
                                   dropout=CFG.dropout,
                                   device=device)
    decoder.load_state_dict(states['decoder'])
    decoder.to(device)
    test_dataset = TestDataset(test, get_transforms(data='valid'))
    test_loader = DataLoader(test_dataset, batch_size= 256,
                             shuffle=False, num_workers=CFG.num_workers)
    predictions = inference(test_loader, encoder, decoder, tokenizer, device)
    test['InChI'] = [f"InChI=1S/{text}" for text in predictions]
    test[['image_id', 'InChI']].to_csv('submission.csv', index=False)
