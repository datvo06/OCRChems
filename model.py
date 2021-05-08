import numpy as np
import torch
from torch import nn
import timm
import warnings
from tqdm import tqdm
from utils import CFG
from layers import CoordConvNet

class Encoder(nn.Module):
    def __init__(self, model_name='resnet18', use_coord_net=False,
                 pretrained=False):
        super().__init__()
        self.cnn = timm.create_model(model_name, pretrained=pretrained)
        self.used_model = model_name

        self.use_coord_net = use_coord_net
        if self.use_coord_net:
            if not self.used_model.startswith('vgg'):
                raise NotImplementedError
            self.cnn.head = nn.Identity()
            self.cnn = CoordConvNet(self.cnn)
        '''
        self.n_features = self.cnn.fc.in_features
        self.cnn.global_pool = nn.Identity()
        self.cnn.fc = nn.Identity()
        '''

    def forward(self, x):
        bs = x.size(0)

        if not self.use_coord_net:
            if self.used_model.startswith('efficientnet') or self.used_model.startswith('vgg'):
                features = self.cnn.forward_features(x)
        else:
            features = self.cnn(x)[-1]

        # B H W C
        features = features.permute(0, 2, 3, 1)
        return features


class  Attention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        '''
        :param encoder_dim: input size of encoder network
        :param decoder_dim: input size of decoder network
        :param attention_dim: input size of attention network
        '''
        super(Attention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)
        self.full_att = nn.Linear(attention_dim, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, encoder_out, decoder_hidden):
        att1 = self.encoder_att(encoder_out)    # b, hw, a
        att2 = self.decoder_att(decoder_hidden) # b, a
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)
        # b, hw, a
        alpha = self.softmax(att)
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(
            dim=1)
        return attention_weighted_encoding, alpha


class DecoderWithAttention(nn.Module):
    def __init__(self, attention_dim, embed_dim, decoder_dim, vocab_size,
                 device, encoder_dim=CFG.enc_size, dropout=0.5):
        '''
        :param attention_dim: input size of attention network
        :param embed_dim: input size of embeding network
        :param decoder_dim: input size of decoder network
        :param vocab_size: total number of character used in training
        :param encoder_dim: input size of encoder network
        :param dropout: dropout rate
        '''
        super(DecoderWithAttention, self).__init__()
        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.dropout = dropout
        self.device = device
        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.dropout = nn.Dropout(p=self.dropout)

        self.decode_step = nn.LSTMCell(embed_dim + encoder_dim,
                                       decoder_dim, bias=True)

        self.init_h = nn.Linear(encoder_dim, decoder_dim)   # init hidden
        self.init_c = nn.Linear(encoder_dim, decoder_dim)   # init cell state
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)   # sigmoid gate
        self.sigmoid = nn.Sigmoid()

        self.fc = nn.Linear(decoder_dim, vocab_size)    # find score over vcab

        self.init_weights()

    def init_weights(self):
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def load_pretrained_embeddings(self, embeddings):
        self.embedding.weight = nn.Parameter(embeddings)

    def fine_tune_embeddings(self, fine_tune=True):
        for p in self.embedding.parameters():
            p.requires_grad = fine_tune

    def init_hidden_state(self, encoder_out):
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)
        c = self.init_c(mean_encoder_out)
        return h, c


    def forward(self, encoder_out, encoded_captions, caption_lengths):
        '''
        :param encoder_out: output of encoder network
        :param encoder_captions: transformed sequence from s to i
        :param caption_lengths: length of transformed sequence
        '''
        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        vocab_size = self.vocab_size
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim) # b, n, c
        num_pixels = encoder_out.size(1)

        # This function is used for training too, so...
        # Encoded captions: labels
        # Caption_lengths: labels_lengths

        # Sort the labels from top down
        caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(
            dim=0, descending=True)

        # Sort the images in that order
        encoder_out = encoder_out[sort_ind]

        # Sort the labels in that order too
        encoded_captions = encoded_captions[sort_ind]

        embeddings = self.embedding(encoded_captions)   # b, max_c_l, embed_dim
        h, c = self.init_hidden_state(encoder_out)

        # set decode length by caption length -1, because of omitting start
        # token
        decode_lengths = (caption_lengths - 1).tolist()

        predictions = torch.zeros(
            batch_size, max(decode_lengths), vocab_size).to(self.device)
        alphas = torch.zeros(
            batch_size, max(decode_lengths), num_pixels).to(self.device)

        # predict sequence
        for t in range(max(decode_lengths)):
            # Only get the sequences that is longer than t
            batch_size_t = sum([l > t for l in decode_lengths])
            # Calculate for that sequence
            attention_weighted_encoding, alpha = self.attention(
                encoder_out[:batch_size_t], h[:batch_size_t])
            gate = self.sigmoid(self.f_beta(h[:batch_size_t]))
            # gating scalar, (batch_size_t, encoder_dim)
            attention_weighted_encoding = gate * attention_weighted_encoding
            # Then LSTM
            h, c = self.decode_step(
                torch.cat([embeddings[:batch_size_t, t, :],
                           attention_weighted_encoding],
                          dim=1),
                (h[:batch_size_t], c[:batch_size_t])) # batch_size_t, decoder_dim
            preds = self.fc(self.dropout(h))
            predictions[:batch_size_t, t, :] = preds
            alphas[:batch_size_t, t, :] = alpha
        return predictions, encoded_captions, decode_lengths, alphas, sort_ind

    def predict(self, encoder_out, decode_lengths, tokenizer):
        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        vocab_size = self.vocab_size
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)
        num_pixels = encoder_out.size(1)
        start_tokens = torch.ones(
            batch_size, dtype=torch.long).to(self.device) * tokenizer.stoi[
                '<sos>']
        embeddings = self.embedding(start_tokens)
        h, c = self.init_hidden_state(encoder_out)
        # decode length is also max length
        predictions = torch.zeros(batch_size, decode_lengths, vocab_size
                                  ).to(self.device)
        # end condition denotes whether and eos has been found for each
        # of element in batchsize
        end_condition = torch.zeros(batch_size, dtype=torch.long).to(
            encoder_out.device)
        for t in range(decode_lengths):
            attention_weighted_encoding, alpha = self.attention(encoder_out, h)
            gate = self.sigmoid(self.f_beta(h))
            attention_weighted_encoding = gate * attention_weighted_encoding
            h, c = self.decode_step(
                torch.cat([embeddings, attention_weighted_encoding], dim =1),
                (h, c))     # batch_size_t, decoder_dim
            preds = self.fc(self.dropout(h))
            predictions[:, t, :] = preds

            end_condition |= (torch.argmax(preds, -1) == tokenizer.stoi['<eos>'])
            if end_condition.sum() == batch_size:
                break
            embeddings = self.embedding(torch.argmax(preds, -1))
        return predictions
