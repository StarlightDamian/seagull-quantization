在 PyTorch 中，遇到 "CUDA out of memory" 错误时，通常是因为训练过程中使用了太多 GPU 内存，导致超出了显存的限制。为了解决这个问题，你可以尝试以下几种方法：

### 1. 减少批次大小（Batch Size）

减小批次大小可以直接减少每次训练所需的显存。你可以将 `batch_size=64` 降低到一个更小的值，例如 `batch_size=32` 或更小，看看是否能够缓解显存不足的问题。

```python
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)  # 降低批次大小
```

### 2. 使用 `torch.cuda.empty_cache()`

在每个训练周期结束时调用 `torch.cuda.empty_cache()` 可以释放未使用的显存，帮助缓解显存溢出的情况。

```python
import torch

# 在每个epoch结束时清理缓存
torch.cuda.empty_cache()
```

### 3. 使用更少的模型参数

你可以尝试减少模型的参数，例如减小隐藏层的大小或减少层数。

```python
# 修改模型的结构，减少隐藏层的数量或神经元数量
hidden_units = [64, 32]  # 减小隐藏层的神经元数目
```

### 4. 更改计算精度

如果使用的模型不需要过高的精度，你可以考虑将模型的计算精度设置为 `float16`，这样会大大减少显存的使用。

```python
model = model.half()  # 将模型转换为float16
x_tensor = x_tensor.half()  # 将输入张量转换为float16
y_tensor = y_tensor.half()  # 将目标张量转换为float16
```

### 5. 增加显存限制

你可以限制每个GPU的最大内存使用量，防止显存过度分配，具体操作如下：

```bash
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
```

这将限制每个分配的内存块的最大大小为 128MB，有助于减少内存碎片。

### 6. 使用梯度累积（Gradient Accumulation）

如果你想使用较小的批次，但又希望模型在多个小批次上积累梯度（相当于使用一个较大的批次），你可以使用梯度累积。

```python
accumulation_steps = 4  # 每次更新前累积4个小批次的梯度

for epoch in range(epochs):
    model.train()
    running_loss_reg = 0.0
    optimizer.zero_grad()  # 清空梯度
    for idx, (x_batch, y_batch) in enumerate(train_loader):
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        # 前向传播
        y_output = model(x_batch)

        # 计算损失
        loss_reg = criterion_reg(y_output, y_batch)

        # 反向传播（累积梯度）
        loss_reg.backward()

        # 每积累一定次数的梯度，进行一次优化
        if (idx + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        running_loss_reg += loss_reg.item()

    logger.info(f"Epoch {epoch+1}/{epochs}, Regression Loss: {running_loss_reg / len(train_loader):.7f}")
    torch.cuda.empty_cache()  # 清理缓存
```

### 7. 使用 `DataParallel` 或 `DistributedDataParallel`

如果你有多个GPU，可以使用 `DataParallel` 或 `DistributedDataParallel` 来分担显存的压力。

```python
model = nn.DataParallel(model)  # 如果有多个GPU
model.to(device)
```

### 8. 修改优化器

优化器的状态也占用显存，Adam等优化器可能会占用更多内存。你可以尝试使用`SGD`或`AdamW`优化器，它们通常占用较少的内存。

```python
optimizer = optim.SGD(model.parameters(), lr=0.001)  # 使用SGD优化器代替Adam
```

### 9. 在 CPU 上进行训练

如果上述方法都无法解决问题，你可以考虑暂时将训练从 GPU 转移到 CPU，尽管这样会慢一些，但可以避免显存不足的问题。

```python
device = torch.device('cpu')  # 切换到 CPU
```

通过上述方法，你可以逐步解决显存不足的问题，并确保训练可以继续进行。如果问题仍然存在，可以尝试通过减少数据规模或模型复杂度来进一步减少内存占用。

显然，即使你已经尝试使用 `half()` 来减少显存占用，但仍然面临显存不足的问题。这是因为，即使你使用了 `half()`（即 float16 精度），但在训练过程中，仍然可能会因为多次梯度计算、缓存、或者批量数据较大等原因，占用过多的显存。

为了解决这个问题，以下是一些可能的优化方案：

### 1. **降低批次大小（Batch Size）**

这是最常见的解决显存溢出的方法。减小批次大小会直接减少每次训练中需要处理的数据量，从而减少显存占用。你可以尝试减小 `batch_size` 例如从 64 降到 32 或 16。

```python
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
```

### 2. **释放缓存：`torch.cuda.empty_cache()`**

在每个 epoch 或训练步骤完成后，调用 `torch.cuda.empty_cache()` 来清理没有使用的缓存，释放内存。

```python
import torch

torch.cuda.empty_cache()
```

可以在每个训练步骤或者每个 epoch 之后调用该函数，特别是在循环结束后清理。

### 3. **使用 `gradient_checkpointing`**

`gradient_checkpointing` 是一种技术，它允许在前向传播时保存更少的中间结果，减少内存的占用。这样虽然会牺牲一定的计算速度，但能有效地减少显存占用。

你可以通过在 `torch` 中开启 `checkpoint` 来实现这一点：

```python
from torch.utils.checkpoint import checkpoint

# 在模型中使用checkpoint
def checkpointed_forward(model, input):
    return checkpoint(model, input)
```

在使用 `checkpoint` 时，你可以根据实际需要选择性地为部分计算进行检查点操作，以减少显存占用。

### 4. **增加 `max_split_size_mb` 参数**

PyTorch 允许配置 `max_split_size_mb` 来解决内存碎片问题。你可以通过设置 `PYTORCH_CUDA_ALLOC_CONF` 环境变量来避免显存碎片。

```bash
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
```

### 5. **使用混合精度训练**

虽然你已经使用了 `.half()` 将模型转换为 `float16`，但 PyTorch 还提供了一个 `torch.cuda.amp`（Automatic Mixed Precision）模块，可以在保证精度的情况下更加高效地管理显存。你可以使用 `amp` 来训练模型，进一步减小内存消耗。

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for epoch in range(epochs):
    model.train()
    running_loss_reg = 0.0
    for idx, (x_batch, y_batch) in enumerate(train_loader):
        optimizer.zero_grad()

        # 使用autocast进行混合精度训练
        with autocast():
            y_output = model(x_batch)
            loss_reg = criterion_reg(y_output, y_batch)

        # 缩放梯度并进行反向传播
        scaler.scale(loss_reg).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss_reg += loss_reg.item()

    torch.cuda.empty_cache()  # 清理缓存
```

这种方法会自动在适当的地方选择使用 `float16` 和 `float32` 来计算，以提高效率并减少显存占用。

### 6. **减小模型规模**

如果你仍然面临显存溢出问题，可以考虑减小模型规模，比如减少隐藏层单元的数量或减少层数。

```python
hidden_units = [64, 32]  # 减少隐藏层神经元的数量
```

### 7. **分布式训练**

如果你有多个 GPU，可以通过分布式训练将模型和数据分配到多个设备上，减轻每个 GPU 的负担。

```python
model = nn.DataParallel(model)  # 如果你有多个 GPU
model.to(device)
```

### 8. **使用 CPU 进行训练**

如果显卡显存不足，作为临时解决方案，你可以将模型训练转移到 CPU 上进行，尽管会比较慢，但可以避免显存不足的问题。

```python
device = torch.device('cpu')  # 切换到 CPU
```

### 9. **增加系统的显存**

如果上述方法无法解决问题，可以考虑升级显卡或者使用更大显存的机器。

------

### 总结

首先，尝试减小批次大小并释放缓存。如果问题仍然存在，可以考虑通过 `gradient_checkpointing`、混合精度训练等方法来进一步优化显存的使用。如果有多个 GPU，使用 `DataParallel` 或 `DistributedDataParallel` 也是一个有效的选择。

确保在使用这些方法时，可以根据实际需求逐步调整，来找到最适合的显存优化策略。