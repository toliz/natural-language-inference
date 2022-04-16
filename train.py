import argparse
import pytorch_lightning as pl

from data import SNLIDataModule
from model import NLI
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping


def main(args):
    # Set up the SNLI data module
    dm = SNLIDataModule(batch_size=args.batch_size, data_dir=args.data_dir); dm.setup()
    
    # Set up the NLI model
    model = NLI(dm.vocab, args.data_dir, args.encoder, args.lstm_hidden_dim)
    
    # Set up the model checkpointing and logging
    tb_logger = TensorBoardLogger(
        save_dir='./tb_logs',
        name=args.encoder,
    )
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=f'./pretrained/{args.encoder}',
        filename=f'{args.encoder}' + '-epoch={epoch}',
        monitor='accuracy/validation',
        mode='max',
        save_weights_only=True,
        auto_insert_metric_name=False,
        every_n_epochs=1,
    )
    
    early_stopping_callback = EarlyStopping(
        monitor='accuracy/validation',
        min_delta=0.1,
        mode='max',
    )
    
    # Set up the training loop
    trainer = pl.Trainer(
        accelerator='auto',
        callbacks=[checkpoint_callback, early_stopping_callback],
        logger=tb_logger,
        max_epochs=args.num_epochs,
    )
    
    # Train on SNLI
    trainer.fit(model, dm)
    
    # Test on SNLI
    trainer.test(model, dm)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Directory to store datasets, vocabulary and embeddings.')
    
    # model hyperparameters
    parser.add_argument('--encoder', type=str, choices=['AWE', 'LSTM', 'BiLSTM', 'BiLSTM-MaxPool'],
                        help='Sentence encoder to use.')
    parser.add_argument('--lstm_hidden_dim', type=int, default=2048,
                        help='Number of hidden units in the LSTM.')
    
    # training hyperparameters
    parser.add_argument('--batch_size', type=int, default=64, 
                        help='Batch size.')
    parser.add_argument('--num_epochs', type=int, default=20,
                        help='Number of training epochs.')
    
    args = parser.parse_args()
    
    main(args)
