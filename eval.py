from __future__ import print_function
from tqdm import tqdm
import torch
from utils import CFG
from tokenizer import tokenizer
import numpy as np
from model import Encoder, DecoderWithAttention
from dataset import TestDataset, get_transforms
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import sys

from top_k_decoder import TopKDecoder

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def inference(test_loader, encoder, decoder, tokenizer, device):
    encoder.eval()
    decoder.eval()
    text_preds = []

    # k = 2
    topk_decoder = TopKDecoder(decoder, 5, CFG.decoder_dim, CFG.max_len, tokenizer)

    tk0 = tqdm(test_loader, total=len(test_loader))
    for images in tk0:
        images = images.to(device)
        predictions = []
        with torch.no_grad():
            encoder_out = encoder(images)
            batch_size = encoder_out.size(0)
            encoder_dim = encoder_out.size(-1)
            encoder_out = encoder_out.view(batch_size, -1, encoder_dim)
            h, c = decoder.init_hidden_state(encoder_out)
            hidden = (h.unsqueeze(0), c.unsqueeze(0))

            decoder_outputs, decoder_hidden, other = topk_decoder(None, hidden, encoder_out)

            for b in range(batch_size):
                length = other['topk_length'][b][0]
                tgt_id_seq = [other['topk_sequence'][di][b, 0, 0].item() for di in range(length)]
                predictions.append(tgt_id_seq)
            assert len(predictions) == batch_size

        predictions = tokenizer.predict_captions(predictions)
        predictions = ['InChI=1S/' + p.replace('<sos>', '') for p in predictions]
        text_preds.append(predictions)
    text_preds = np.concatenate(text_preds)
    return text_preds

def inference_old(test_loader, encoder, decoder, tokenizer, device):
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
        _text_preds = [f"InChI=1S/{text}" for text in _text_preds]
        text_preds.append(_text_preds)
    text_preds = np.concatenate(text_preds)
    return text_preds

def get_test_file_path(image_id):
    return CFG.test_path + "{}/{}/{}/{}.png".format(
        image_id[0], image_id[1], image_id[2], image_id
    )

if __name__ == '__main__':
    test = pd.read_csv('sample_submission.csv')
    test['file_path'] = test['image_id'].apply(get_test_file_path)
    print(f'test.shape: {test.shape}')
    model_path = CFG.pred_model
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    states = torch.load(model_path, map_location=torch.device('cpu'))
    encoder = Encoder(CFG.model_name, pretrained=False, use_coord_net=CFG.use_coord)
    encoder.load_state_dict(states['encoder'])
    encoder = encoder.to(device)
    decoder = DecoderWithAttention(attention_dim=CFG.attention_dim,
                                   embed_dim=CFG.embed_dim,
                                   decoder_dim=CFG.decoder_dim,
                                   vocab_size=len(tokenizer),
                                   dropout=CFG.dropout,
                                   device=device)
    decoder.load_state_dict(states['decoder'])
    decoder.to(device)
    test_dataset = TestDataset(test, get_transforms(data='valid'))
    test_loader = DataLoader(test_dataset, batch_size=180,
                             shuffle=False, num_workers=CFG.num_workers)
    predictions = inference(test_loader, encoder, decoder, tokenizer, device)
    test['InChI'] = [text for text in predictions]
    # test['InChI'] = [f"InChI=1S/{text}" for text in predictions]
    test[['image_id', 'InChI']].to_csv('submission_' + CFG.model_name + '_288_bs5.csv', index=False)
