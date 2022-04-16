import nltk
import pickle
import pytorch_lightning as pl
import torch
import torchtext

from collections import Counter, OrderedDict
from datasets import load_dataset, load_from_disk
from nltk.tokenize import word_tokenize
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm.auto import tqdm


class SNLIDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=64, data_dir='./data'):
        super().__init__()
        
        self.data_dir = data_dir
        self.batch_size = batch_size
        
    def prepare_data(self):
        nltk.download('punkt')
        load_dataset('snli', cache_dir=f'{self.data_dir}/.cache')
        
    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.vocab = self._load_vocab()
        
            self.train_dataset = self._load_dataset('train')
            self.validation_dataset = self._load_dataset('validation')
        
        if stage == 'test' or stage is None:
            self.train_dataset = self._load_dataset('test')
        
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=SNLIDataModule._collate_fn,
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.validation_dataset,
            batch_size=self.batch_size,
            collate_fn=SNLIDataModule._collate_fn,
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            collate_fn=SNLIDataModule._collate_fn,
        )
        
    def _preprocess(self, text):
        return self.vocab(word_tokenize(text.lower()))
    
    def _encode(self, samples):
        return {
            'premise': [self._preprocess(sentence) for sentence in samples['premise']],
            'hypothesis': [self._preprocess(sentence) for sentence in samples['hypothesis']],
            'label': samples['label']
        }
        
    def _load_dataset(self, split):
        if Path(f'{self.data_dir}/{split}').is_dir():
            dataset = load_from_disk(f'{self.data_dir}/{split}')
        else:            
            dataset = load_dataset('snli', split=split, cache_dir=f'{self.data_dir}/.cache')
            print(f'Pre-processing {split} split...')
            
            # process each sentence in the dataset and map it to a list of indices
            dataset = dataset.map(self._encode, batched=True)
            
            # discard samples with invalid labels
            dataset = dataset.filter(lambda sample: 0 <= sample['label'] and sample['label'] <= 2)
            
            # save for future speed-up
            dataset.save_to_disk(f'{self.data_dir}/{split}')
            
        dataset.set_format(type='torch')
        
        return dataset
    
    def _load_vocab(self):
        if Path(f'{self.data_dir}/vocab.pkl').exists():
            vocab = pickle.load(open(f'{self.data_dir}/vocab.pkl', 'rb'))
        else:            
            train_dataset = load_dataset('snli', split='train', cache_dir=f'{self.data_dir}/.cache')
            print('Building vocabulary...')
            
            # extract premise and hypothesis from the train dataset
            premise = train_dataset['premise']
            hypothesis = train_dataset['hypothesis']
            
            # count word frequencies
            counter = Counter()
            for sentence in tqdm(premise + hypothesis):
                counter.update(word_tokenize(sentence.lower()))
            
            # sort words according to their frequencies
            counter = sorted(counter.items(), key=lambda x: x[1], reverse=True)

            # build a torchtext vocabulary
            vocab = torchtext.vocab.vocab(OrderedDict(counter), specials=['<pad>', '<unk>'], special_first=True)
            vocab.set_default_index(vocab['<unk>'])

            # save vocab            
            pickle.dump(vocab, open(f'{self.data_dir}/vocab.pkl', 'wb'))

        return vocab
    
    @staticmethod
    def _collate_fn(batch):
        premise = [torch.LongTensor(sample['premise']) for sample in batch]
        hypothesis = [torch.LongTensor(sample['hypothesis']) for sample in batch]
        label = torch.LongTensor([sample['label'] for sample in batch])
        
        premise = pad_sequence(premise, batch_first=True)
        hypothesis = pad_sequence(hypothesis, batch_first=True)

        return premise, hypothesis, label


class SNLIDataset(Dataset):
    def __init__(self, split, vocab=None, data_dir='./data', cache_dir='./.cache'):
        super().__init__()
        
        if Path(f'{data_dir}/snli_{split}.pt').exists():
            dataset = torch.load(f'{data_dir}/snli_{split}.pt')
            
            self.premise = dataset['premise'][:1000]
            self.hypothesis = dataset['hypothesis'][:1000]
            self.label = dataset['label'][:1000]
        else:
            dataset = load_dataset('snli', split=split, cache_dir=cache_dir)
            
            # extract premise, hypothesis and label from the dataset
            premise = dataset['premise']
            hypothesis = dataset['hypothesis']
            label = dataset['label']
            
            # pre-process the premise and hypothesis (lowercase + tokenize)
            premise = [word_tokenize(s.lower()) for s in premise]
            hypothesis = [word_tokenize(s.lower()) for s in hypothesis]
            
            # convert words to indices
            premise = [vocab(s) for s in premise]
            hypothesis = [vocab(s) for s in hypothesis]
            
            # convert to list of tensors
            premise = [torch.LongTensor(p) for p in premise]
            hypothesis = [torch.LongTensor(h) for h in hypothesis]
            label = torch.LongTensor(label)
            
            # pad premise and hypothesis sequences
            premise = pad_sequence(premise, batch_first=True)
            hypothesis = pad_sequence(hypothesis, batch_first=True)
            
            # remove corrupt data
            idx = torch.logical_and(0 <= label, label <= 2)
            
            self.premise = premise[idx]
            self.hypothesis = hypothesis[idx]
            self.label = label[idx]
            
            torch.save(
                {'premise': self.premise, 'hypothesis': self.hypothesis, 'label': self.label},
                open(f'{data_dir}/snli_{split}.pt', 'wb')
            )
        
    def __len__(self):
        return len(self.label)
        
    def __getitem__(self, index):
        return self.premise[index], self.hypothesis[index], self.label[index]
