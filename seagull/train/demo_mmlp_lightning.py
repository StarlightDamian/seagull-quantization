# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 23:59:11 2024

@author: awei
"""

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from torchmetrics import AUC
import tensorboardX
import argparse
import pandas as pd
from seagull.settings import PATH
from seagull.utils import utils_database, utils_character, utils_log, utils_math

class CustomDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class AutoencoderMLP(pl.LightningModule):
    def __init__(self, num_columns, num_labels, hidden_units, dropout_rates, ls=1e-2, lr=1e-3):
        super(AutoencoderMLP, self).__init__()
        self.num_columns = num_columns
        self.num_labels = num_labels
        self.hidden_units = hidden_units
        self.dropout_rates = dropout_rates
        self.ls = ls
        self.lr = lr
        
        # Autoencoder part
        self.encoder = nn.Sequential(
            nn.BatchNorm1d(self.num_columns),
            nn.GaussianNoise(self.dropout_rates[0]),
            nn.Linear(self.num_columns, self.hidden_units[0]),
            nn.BatchNorm1d(self.hidden_units[0]),
            nn.SiLU(),
        )
        
        self.decoder = nn.Sequential(
            nn.Dropout(self.dropout_rates[1]),
            nn.Linear(self.hidden_units[0], self.num_columns),
        )
        
        self.fc = nn.Sequential(
            nn.Linear(self.num_columns + self.hidden_units[0], self.hidden_units[1]),
            nn.BatchNorm1d(self.hidden_units[1]),
            nn.SiLU(),
            nn.Dropout(self.dropout_rates[2]),
            nn.Linear(self.hidden_units[1], self.hidden_units[2]),
            nn.BatchNorm1d(self.hidden_units[2]),
            nn.SiLU(),
            nn.Dropout(self.dropout_rates[3]),
            nn.Linear(self.hidden_units[2], self.num_labels),
            nn.Sigmoid(),
        )

        # Multi-label Classification Loss
        self.loss_fn = nn.BCEWithLogitsLoss(reduction='none')
        
    def forward(self, x):
        encoder_output = self.encoder(x)
        decoder_output = self.decoder(encoder_output)
        concatenated = torch.cat([x, encoder_output], dim=1)
        final_output = self.fc(concatenated)
        return decoder_output, final_output

    def training_step(self, batch, batch_idx):
        x, y, sample_weights = batch
        decoder_output, final_output = self(x)
        # Compute loss for decoder and final output
        decoder_loss = nn.MSELoss()(decoder_output, x)
        classification_loss = self.loss_fn(final_output, y)
        weighted_loss = torch.mean(classification_loss * sample_weights)
        total_loss = decoder_loss + weighted_loss
        self.log("train_loss", total_loss)
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        decoder_output, final_output = self(x)
        decoder_loss = nn.MSELoss()(decoder_output, x)
        classification_loss = self.loss_fn(final_output, y)
        total_loss = decoder_loss + classification_loss
        self.log("val_loss", total_loss, on_epoch=True)
        return total_loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def on_epoch_end(self):
        # Optionally, log additional metrics to TensorBoard
        auc = AUC()
        auc.update(self.y_true, self.y_pred)
        self.log("val_auc", auc.compute())
        
if __name__ == "__main__":
    # TensorBoard Integration
    from pytorch_lightning.loggers import TensorBoardLogger
    
    logger = TensorBoardLogger("tb_logs", name="autoencoder_mlp")
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--date_start', type=str, default='2023-10-20', help='When to start feature engineering')
    parser.add_argument('--date_end', type=str, default='2023-12-27', help='End time for feature engineering')
    args = parser.parse_args()

    logger.info(f"""task: feature_engineering
                    date_start: {args.date_start}
                    date_end: {args.date_end}""")

    # 获取日期段数据
    with utils_database.engine_conn("POSTGRES") as conn:
        df = pd.read_sql(f"SELECT * FROM dwd_freq_incr_stock_daily WHERE date BETWEEN '{args.date_start}' AND '{args.date_end}'", con=conn.engine)

    features = ['open_rate', 'high_rate', 'low_rate', 'close_rate', 'volume',
               'value_traded', 'turnover', 'chg_rel', 'pe_ttm',
              'ps_ttm', 'pcf_ttm', 'pb_mrq','board_type', 'price_limit_rate']
    x = df[features]
    
    # 划分训练和测试集
    x_train, x_test, y_class_train, y_class_test, y_reg_train, y_reg_test = train_test_split(
        x, y_class, y_reg, test_size=0.2, random_state=42)
    
    # Example usage with a DataLoader
    train_dataset = CustomDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    val_dataset = CustomDataset(X_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=32)
    
    # Trainer with early stopping and checkpointing
    checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor='val_auc', mode='max', save_top_k=1)
    early_stop_callback = pl.callbacks.EarlyStopping(monitor='val_auc', patience=10, mode='max')
    
    trainer = pl.Trainer(
        max_epochs=100,
        gpus=1,
        logger=logger,
        callbacks=[checkpoint_callback, early_stop_callback],
        progress_bar_refresh_rate=30
    )
    
    # Train the model
    model = AutoencoderMLP(num_columns=len(features), num_labels=5, hidden_units=[96, 96, 896, 448, 448, 256], dropout_rates=[0.035, 0.038, 0.42, 0.1, 0.49, 0.32, 0.27, 0.44])
    trainer.fit(model, train_loader, val_loader)
