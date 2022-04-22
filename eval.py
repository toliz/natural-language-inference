import argparse
import logging
import pickle 
import sys
import torch

from model import NLI
from torch.nn.utils.rnn import pad_sequence
from tabulate import tabulate

# Set paths
PATH_TO_SENTEVAL = '../'
PATH_TO_DATA = '../SentEval/data/'
PATH_TO_CKPT = {
    'AWE': 'pretrained/AWE/version_0/AWE-epoch=6.ckpt',
    'LSTM': 'pretrained/LSTM/version_0/LSTM-epoch=8.ckpt',
    'BiLSTM': 'pretrained/BiLSTM/version_0/BiLSTM-epoch=4.ckpt',
    'BiLSTM-MaxPool': 'pretrained/BiLSTM-MaxPool/version_0/BiLSTM-MaxPool-epoch=2.ckpt',
}

# import SentEval
sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval


# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.INFO)


def prepare(params, samples):
    """
    Here you will initialize your model.
    Remember to add what you model needs into the params dictionary
    """
    params.vocab = pickle.load(open('data/vocab.pkl', 'rb'))
    params.model = NLI.load_from_checkpoint(PATH_TO_CKPT[params.encoder]).encoder
    
    if params.cuda:
        params.model.cuda()
    
    return


def batcher(params, batch):
    """
    Here you can process each sentence of the batch, or a complete batch (you may need masking for that).
    """
    # If a sentence is empty, dot is set to be the only token
    batch = [sentence if sentence != [] else ['.'] for sentence in batch]
    
    # Pre-process each sentence
    batch = [[word.lower() for word in sentence] for sentence in batch]
    
    # convert it to list of indices
    batch = [params.vocab(sentence) for sentence in batch]
    
    # Convert to torch tensors
    batch = [torch.LongTensor(sentence) for sentence in batch]
    
    # Pad the batch
    batch = pad_sequence(batch, batch_first=True)
    
    # Encode the batch
    with torch.no_grad():
        if params.cuda:
            embeddings = params.model(batch.cuda())
        else:
            embeddings = params.model(batch)
    
    return embeddings.detach().cpu().numpy()


def main(args):
    params_senteval = {
        'encoder': args.encoder,
        'cuda': torch.cuda.is_available(),
        'task_path': PATH_TO_DATA,
        'usepytorch': False,
        'kfold': 5
    }
    
    transfer_tasks = ['MR', 'CR', 'SUBJ', 'MPQA', 'SST2', 'TREC', 'MRPC', 'SICKRelatedness', 'SICKEntailment', 'STS14']
    
    # Evaluate on SentEval        
    se = senteval.engine.SE(params_senteval, batcher, prepare)
    results = se.eval(transfer_tasks)
    
    # Save results
    pickle.dump(results, open(f'senteval/{args.encoder}.pkl', 'wb'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--encoder', type=str, help='The model to be evaluated.')
    
    args = parser.parse_args()
    
    main(args)
