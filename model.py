import os
import pytorch_lightning as pl
import torch
import torchmetrics
import torchtext

from pathlib import Path
from torch import nn, optim
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def load_glove_embeddings(vocab: torchtext.vocab.Vocab, data_dir='./data') -> torch.Tensor:
    """Returns an embedding matrix for the given vocabulary, initialized with GloVe embeddings.

    Args:
        vocab (torchtext.vocab.Vocab): the vocabulary object
        cache (str): the dir where the GloVe embeddings are stored. Defaults to './.cache'.

    Returns:
        torch.Tensor: the embedding matrix
    """
    if Path(f'{data_dir}/embeddings.pt').exists():
        return torch.load(f'{data_dir}/embeddings.pt')
    else:
        print('Downloading GloVe embeddings...')
        
        # Download and load the 300-dimensional glove embeddings trained on Common Crawl 840B
        embeddings = torchtext.vocab.GloVe(name='840B', dim=300, cache=data_dir)
        
        # Get a list of the words in the vocabulary, in increasing index order
        words_in_vocab = vocab.get_itos()
        
        # Get the embedding matrix for the words in the vocabulary
        W_vocab = embeddings.get_vecs_by_tokens(words_in_vocab)
        
        # Initialize embeddings for special symbols
        W_pad = torch.zeros(1, 300)
        W_unk = torch.randn(1, 300) * 0.6 # should follow the same distribution as W_vocab
        
        # Obtain and save embedding matrix W
        W = torch.cat([W_pad, W_unk, W_vocab])
        torch.save(W, f'{data_dir}/embeddings.pt')
        
        # Remove cached files (~10GB)
        os.remove(f'{data_dir}/glove.840B.300d.txt')
        os.remove(f'{data_dir}/glove.840B.300d.txt.pt')
        os.remove(f'{data_dir}/glove.840B.300d.zip')
    
    return W


class AWE(nn.Module):
    """Average Word Embeddings (AWE) model.
    
    It encodes a sentence by averaging the embeddings of each word in the sentence.
    
    Args:
        vocab (torchtext.vocab.Vocab): the vocabulary object
        
    Inputs:
        input (torch.Tensor): the input sentences, shape (B x T)
        
    Outputs:
        output (torch.Tensor): the encoded sentences, shape (B x D)
    """
    def __init__(self, vocab: torchtext.vocab.Vocab, data_dir: str):
        super().__init__()
        
        # model architecture
        self.embedding = nn.Embedding.from_pretrained(load_glove_embeddings(vocab, data_dir))
        
    def forward(self, input):
        # input shape is B x T
        B, T = input.shape
        
        # Get the lengths of the sentences in the input
        lengths = input.argmin(dim=1)
        lengths = torch.where(lengths > 0, lengths, T)
        
        # Pass the sentences through the embedding layer: (B x T) -> (B x T x D)
        output = self.embedding(input)
        
        # Get average embedding of each sentence: (B x T x D) -> (B x D)
        output = output.sum(dim=1) / lengths.view(-1 , 1)
        
        return output


class LSTM(nn.Module):
    """LSTM model.
    
    It encodes a sentence by passing each word through a single LSTM layer. The sentence
    representation is the last hidden state of the LSTM.
    
    Args:
        vocab (torchtext.vocab.Vocab): the vocabulary object
        hidden_dim (int): the dimension of the hidden state of the LSTM. Defaults to 2048.
        
    Inputs:
        input (torch.Tensor): the input sentences, shape (B x T)
        
    Outputs:
        output (torch.Tensor): the encoded sentences, shape (B x H)
    """
    def __init__(self, vocab: torchtext.vocab.Vocab, data_dir: str, hidden_dim=2048):
        super().__init__()
        
        # model architecture
        self.embedding = nn.Embedding.from_pretrained(load_glove_embeddings(vocab, data_dir))
        self.rnn = nn.LSTM(300, hidden_dim, batch_first=True)
        
    def forward(self, input):
        # input shape is B x T
        B, T = input.shape
        
        # Get the lengths of the sentences in input
        lengths = input.argmin(dim=1)
        lengths = torch.where(lengths > 0, lengths, T)
        
        # Pass the sentences through the embedding layer: (B x T) -> (B x T x D)
        embeddings = self.embedding(input)
        
        # Pass the sentences through the LSTM layer: (B x T x D) -> (B x T x H)
        packed_embeddings = pack_padded_sequence(embeddings, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h, _) = self.rnn(packed_embeddings)
        
        # output is the last hidden state of the LSTM: (B x T x H) -> (B x H)
        output = h[-1, :, :]
        
        return output
    
    
class BiLSTM(nn.Module):
    """BiLSTM model.
    
    It encodes a sentence by passing each word through a single Bidirectional LSTM layer. The 
    sentence representation is either:
        - the concatentation of the last hidden states of the forward and backward LSTMs.
        - the maximum over each dimension of the concatenated hidden states of the forward
            and backward LSTMs (max pooling).
    
    Args:
        vocab (torchtext.vocab.Vocab): the vocabulary object
        hidden_dim (int): the dimension of the hidden state of the LSTM. Defaults to 2048.
        max_pooling (bool): whether to use max pooling. Defaults to True.
        
    Inputs:
        input (torch.Tensor): the input sentences, shape (B x T)
        
    Outputs:
        output (torch.Tensor): the encoded sentences, shape (B x 2H)
    """
    def __init__(self, vocab: torchtext.vocab.Vocab, data_dir: str, hidden_dim=2048, max_pooling=True):
        super().__init__()
        
        self.max_pooling = max_pooling
        
        # model architecture
        self.embedding = nn.Embedding.from_pretrained(load_glove_embeddings(vocab, data_dir))
        self.rnn = nn.LSTM(300, hidden_dim, bidirectional=True, batch_first=True)
        
    def forward(self, input):
        # input shape is B x T
        B, T = input.shape
        
        # Get the lengths of the sentences in input
        lengths = input.argmin(dim=1)
        lengths = torch.where(lengths > 0, lengths, T)
        
        # Pass the sentences through the embedding layer: (B x T) -> (B x T x D)
        embeddings = self.embedding(input)
        
        # Pass the sentences through the LSTM layer: (B x T x D) -> (B x T x 2H)
        packed_embeddings = pack_padded_sequence(embeddings, lengths.cpu(), batch_first=True, enforce_sorted=False)
        output, (h, _) = self.rnn(packed_embeddings)
        
        # Get setence representation: (B x T x 2H) -> (B x 2H)
        if not self.max_pooling:
            # output is the concatenation of the last hidden states of the BiLSTM
            output = torch.cat([h[-2, :, :], h[-1, :, :]], dim=1)
        else:
            output, _ = pad_packed_sequence(output, batch_first=True)

            # mask-out hidden states after the end of each sentence.
            mask = torch.zeros_like(output)
            for i in range(B):
               mask[i, lengths[i]:] = -float('inf')
            
            # output is the max over each dimension of the concatenated hidden states of the BiLSTM
            output, _ = torch.max(output + mask, dim = 1)
        
        return output


class NLI(pl.LightningModule):
    """A Lightning Module for Natural Language Inference (NLI).
    
    This module first encodes the premise (u) and hypothesis (v) sentences using an (LSTM) sentence
    encoder. Then, it transforms the sentence encodings into a feature vector [u, v, |u-v|, u*v] 
    and passes it through an MLP with a softmax output layer, to classify the sentence pair as 
    entailment, contradiction or neutral.
    
    For more details see "Supervised  learning of universal sentence representations from natural 
    language inference data" by A. Conneau et al.

    Args:
        vocab (torchtext.vocab.Vocab): the vocabulary object
        encoder (str): the encoder model. One of 'AWE', 'LSTM', 'BiLSTM', 'BiLSTM-MaxPool'.
        hidden_dim (int): the dimension of the hidden state of the LSTM, if applicable.
    """
    def __init__(self, vocab: torchtext.vocab.Vocab, encoder, hidden_dim=None, data_dir='./data'):
        super().__init__()
        
        match encoder:
            case 'AWE':
                encoder = AWE(vocab, data_dir)
                encoder_output_dim = 300
            case 'LSTM':
                encoder = LSTM(vocab, data_dir, hidden_dim=hidden_dim)
                encoder_output_dim = hidden_dim
            case 'BiLSTM':
                encoder = BiLSTM(vocab, data_dir, hidden_dim=hidden_dim, max_pooling=False)
                encoder_output_dim = 2*hidden_dim
            case 'BiLSTM-MaxPool':
                encoder = BiLSTM(vocab, data_dir, hidden_dim=hidden_dim, max_pooling=True)
                encoder_output_dim = 2*hidden_dim
            case _:
                raise NotImplementedError(f'Model {encoder} not implemented.')
        
        # model architecture
        self.encoder = encoder
        self.MLP = nn.Sequential(
            nn.Linear(4*encoder_output_dim, 512),
            nn.Linear(512, 3),
        )
        
        # lightning metics
        self._val_acc = torchmetrics.Accuracy()
        self._test_acc = torchmetrics.Accuracy()
        
    def forward(self, u, v):
        # Encode the sentences u and v.
        u = self.encoder(u)
        v = self.encoder(v)
                
        # Create the feature vector and pass it through the MLP classifier.
        x = torch.cat([u, v, torch.abs(u-v), u*v], dim=1)
        x = self.MLP(x)
        
        return x
    
    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=0.1)
        
        lr_scheduler1_config = {
            'scheduler': StepLR(optimizer, step_size=1, gamma=0.99)
        }
        lr_scheduler2_config = {
            'scheduler': ReduceLROnPlateau(optimizer, mode='max', factor=0.2, patience=2),
            'monitor': 'accuracy/validation',
        }
        
        return [optimizer], [lr_scheduler1_config, lr_scheduler2_config]
    
    def training_step(self, batch, batch_idx):
        # forward pass
        premise, hypothesis, target = batch
        pred = self(premise, hypothesis)
        
        # metrics
        loss = nn.functional.cross_entropy(pred, target)
        
        # logging
        self.log('loss/train', loss)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        # forward pass
        premise, hypothesis, target = batch
        pred = self(premise, hypothesis)
        
        # metrics
        loss = nn.functional.cross_entropy(pred, target)
        acc = self._val_acc(pred, target)
        
        # logging
        self.log('loss/validation', loss)
        
        return acc
    
    def validation_epoch_end(self, outputs):
        self.log('accuracy/validation', self._val_acc)
    
    def test_step(self, batch, batch_idx):
        # forward pass
        premise, hypothesis, target = batch
        pred = self(premise, hypothesis)
        
        # metrics
        acc = self._test_acc(pred, target)
        
        return acc

    def test_epoch_end(self, outputs):
        self.log('accuracy/test', self._test_acc)
