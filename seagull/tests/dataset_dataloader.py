# -*- coding: utf-8 -*-
"""
Created on Mon May 27 14:50:16 2024

@author: awei
"""
from sentence_transformers import SentenceTransformer
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class CustomDataset(Dataset):
    def __init__(self, sentences):
        self.sentences = sentences

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return self.sentences[idx]

if __name__ == '__main__':
    # Set params
    data_stream_size = 16384  # Size of the data that is loaded into memory at once
    chunk_size = 1024  # Size of the chunks that are sentiment to each process
    encode_batch_size = 128  # Batch size of the model
    
    # Your sentences
    sentences = ["sentence"] * 1024  # Replace with your actual sentences

    # Instantiate custom dataset
    custom_dataset = CustomDataset(sentences)

    # Create DataLoader
    dataloader = DataLoader(custom_dataset, batch_size=data_stream_size, shuffle=True)

    # Define the model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Start the multi-process pool on all available CUDA devices
    pool = model.start_multi_process_pool()

    for i, batch in enumerate(tqdm(dataloader)):
        # Compute the embeddings using the multi-process pool
        batch_emb = model.encode_multi_process(batch, pool, chunk_size=chunk_size, batch_size=encode_batch_size)
        print("Embeddings computed for 1 batch. Shape:", batch_emb.shape)

    # Optional: Stop the processes in the pool
    model.stop_multi_process_pool(pool)
