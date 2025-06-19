Converting your TensorFlow code into PyTorch involves several steps, including adapting the model architecture, training loop, and the way PyTorch handles optimizers, losses, and batch normalization. I'll outline the structure in PyTorch, which mirrors your existing approach in TensorFlow.

Here's a conversion to PyTorch:

### Step-by-Step PyTorch Model Translation

1.  **Imports and Setup**: You need to import necessary libraries such as PyTorch, torchvision (if needed), and other utilities.

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.nn import functional as F
import numpy as np
import pandas as pd
```

1.  **Data Preparation**: In PyTorch, datasets are typically wrapped in a `Dataset` class. You can use a `DataLoader` for efficient mini-batching.

```python
class JaneStreetDataset(Dataset):
    def __init__(self, X, y, sample_weight=None):
        self.X = X
        self.y = y
        self.sample_weight = sample_weight

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx], dtype=torch.float32)
        y = torch.tensor(self.y[idx], dtype=torch.float32)
        if self.sample_weight is not None:
            sw = torch.tensor(self.sample_weight[idx], dtype=torch.float32)
            return x, y, sw
        return x, y
```

1.  **Model Architecture**: Now, let's define the model in PyTorch, including the autoencoder (AE) and the MLP part, as well as the loss functions.

```python
class AE_MLP(nn.Module):
    def __init__(self, num_columns, num_labels, hidden_units, dropout_rates, ls=1e-2, lr=1e-3):
        super(AE_MLP, self).__init__()
        
        self.num_columns = num_columns
        self.num_labels = num_labels
        
        # Encoder (Autoencoder part)
        self.encoder = nn.Sequential(
            nn.BatchNorm1d(num_columns),
            nn.GaussianNoise(dropout_rates[0]),
            nn.Linear(num_columns, hidden_units[0]),
            nn.BatchNorm1d(hidden_units[0]),
            nn.SiLU(),  # Swish activation
        )
        
        self.decoder = nn.Sequential(
            nn.Dropout(dropout_rates[1]),
            nn.Linear(hidden_units[0], num_columns),
        )

        # Supervised Autoencoder output (regression)
        self.regression = nn.Sequential(
            nn.Linear(num_columns, hidden_units[1]),
            nn.BatchNorm1d(hidden_units[1]),
            nn.SiLU(),
            nn.Dropout(dropout_rates[2]),
        )
        
        # Main MLP classifier
        self.mlp = nn.Sequential(
            nn.Linear(num_columns + hidden_units[0], hidden_units[2]),
            nn.BatchNorm1d(hidden_units[2]),
            nn.SiLU(),
            nn.Dropout(dropout_rates[3]),
            nn.Linear(hidden_units[2], hidden_units[3]),
            nn.BatchNorm1d(hidden_units[3]),
            nn.SiLU(),
            nn.Dropout(dropout_rates[4]),
            nn.Linear(hidden_units[3], num_labels),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoder_out = self.encoder(x)
        decoder_out = self.decoder(encoder_out)
        regression_out = self.regression(decoder_out)
        mlp_out = self.mlp(torch.cat([x, encoder_out], dim=1))
        
        return decoder_out, regression_out, mlp_out
```

### Key Points:

1.  **GaussianNoise Layer**: PyTorch does not have a built-in `GaussianNoise` layer, but we can define it manually as follows:

```python
class GaussianNoise(nn.Module):
    def __init__(self, stddev):
        super(GaussianNoise, self).__init__()
        self.stddev = stddev

    def forward(self, x):
        if self.training:
            noise = torch.randn_like(x) * self.stddev
            return x + noise
        return x
```

1.  **Loss Functions**: We'll use PyTorch's built-in loss functions: `BCELoss` for binary classification and `MSELoss` for the regression task.

```python
criterion_ae = nn.MSELoss()
criterion_bce = nn.BCEWithLogitsLoss()  # For multi-label classification
```

1.  **Training Loop**: The training loop will follow the usual PyTorch pattern where we loop through the epochs, perform backpropagation, and optimize the weights.

```python
def train_model(model, train_loader, val_loader, epochs, lr, sample_weights=None):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()

    for epoch in range(epochs):
        running_loss = 0.0
        for i, (inputs, targets, sw) in enumerate(train_loader):
            optimizer.zero_grad()
            decoder_out, regression_out, mlp_out = model(inputs)
            
            loss_decoder = criterion_ae(decoder_out, inputs)  # Autoencoder loss
            loss_ae = criterion_bce(regression_out, targets)  # Supervised AE loss
            loss_mlp = criterion_bce(mlp_out, targets)  # MLP loss
            
            # Weighted loss
            loss = loss_decoder + loss_ae + loss_mlp
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss / len(train_loader)}")

# Example usage
# Assume X_train, y_train, and sample_weights are already defined
train_dataset = JaneStreetDataset(X_train, y_train, sample_weight=sample_weights)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
```

1.  **Cross-Validation**: Implementing the `PurgedGroupTimeSeriesSplit` is outside the scope of this translation, but PyTorch can handle such splitting just as TensorFlow does, using `GroupKFold` or custom splits based on date.
2.  **Model Evaluation**: After training, evaluate the model on the validation set.

```python
def evaluate_model(model, val_loader):
    model.eval()
    with torch.no_grad():
        total_auc = 0.0
        for inputs, targets, _ in val_loader:
            decoder_out, regression_out, mlp_out = model(inputs)
            auc = roc_auc_score(targets.numpy(), mlp_out.numpy())
            total_auc += auc

    avg_auc = total_auc / len(val_loader)
    print(f"Validation AUC: {avg_auc}")
```

1.  **Hyperparameter Tuning**: Hyperopt can be used for hyperparameter tuning, but the implementation is not significantly different from TensorFlow; you'll optimize the hyperparameters within the same PyTorch training loop.

```python
from hyperopt import hp, fmin, tpe, Trials

def objective(params):
    model = AE_MLP(**params)
    # Train the model here using the `train_model` function
    # Return the validation loss or AUC for optimization
```

------

### Conclusion

This PyTorch implementation follows the structure you provided in TensorFlow, with a few differences related to PyTorch's approach (e.g., how batch normalization, dropout, and other layers are defined). The key adjustments involve:

-   Replacing TensorFlow's `Model` class with PyTorch's `nn.Module`.
-   Adapting the training loop and optimization process.
-   Customizing the `GaussianNoise` layer for PyTorch.

You can now extend the code with more advanced features (like group-based splits or hyperparameter tuning with `hyperopt`) as per your requirements.