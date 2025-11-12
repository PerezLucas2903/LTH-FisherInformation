import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune
from torch.utils.data import DataLoader, TensorDataset


# -------------------------------------------------------------
# Example CNN
# -------------------------------------------------------------
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 8 * 8, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x


# -------------------------------------------------------------
# Random Unstructured Pruning (still optional)
# -------------------------------------------------------------
def random_unstructured_pruning(model, amount=0.2):
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            prune.random_unstructured(module, name='weight', amount=amount)
            print(f"[Random Unstructured Pruning] Pruned {amount*100}% of {name}.weight")
    return model


# -------------------------------------------------------------
# L1 Channel Pruning (structured)
# -------------------------------------------------------------
def l1_channel_prune_layer(conv_layer, amount=0.2):
    """Returns a new Conv2d layer with smallest-L1 filters removed."""
    weight = conv_layer.weight.data.abs().mean(dim=(1, 2, 3))
    num_prune = int(amount * conv_layer.out_channels)
    if num_prune == 0:
        return conv_layer, torch.ones(conv_layer.out_channels, dtype=torch.bool)

    prune_indices = torch.argsort(weight)[:num_prune]
    mask = torch.ones(conv_layer.out_channels, dtype=torch.bool)
    mask[prune_indices] = False

    new_conv = nn.Conv2d(
        in_channels=conv_layer.in_channels,
        out_channels=mask.sum().item(),
        kernel_size=conv_layer.kernel_size,
        stride=conv_layer.stride,
        padding=conv_layer.padding,
        dilation=conv_layer.dilation,
        bias=(conv_layer.bias is not None)
    )
    new_conv.weight.data = conv_layer.weight.data[mask].clone()
    if conv_layer.bias is not None:
        new_conv.bias.data = conv_layer.bias.data[mask].clone()

    print(f"[L1 Channel Pruning] Removed {num_prune}/{conv_layer.out_channels} filters.")
    return new_conv, mask


def apply_l1_pruning_during_training(model, amount=0.1):
    """Applies channel pruning to all Conv2d layers in-place."""
    for name, module in model.named_children():
        if isinstance(module, nn.Conv2d):
            new_conv, _ = l1_channel_prune_layer(module, amount)
            setattr(model, name, new_conv)
        else:
            apply_l1_pruning_during_training(module, amount)
    return model


# -------------------------------------------------------------
# Training loop with pruning applied every few epochs
# -------------------------------------------------------------
def train_with_pruning(model, dataloader, optimizer, criterion, device,
                       num_epochs=10, prune_every=3, prune_amount=0.1):
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {avg_loss:.4f}")

        # Apply structured pruning periodically
        if (epoch + 1) % prune_every == 0 and epoch + 1 < num_epochs:
            print(f"\n--- Applying L1 Channel Pruning (epoch {epoch+1}) ---")
            model = apply_l1_pruning_during_training(model, prune_amount)
            model.to(device)  # move to GPU again if needed

    return model


# -------------------------------------------------------------
# Example usage
# -------------------------------------------------------------
if __name__ == "__main__":
    # Fake dataset
    X = torch.randn(256, 3, 32, 32)
    y = torch.randint(0, 10, (256,))
    dataloader = DataLoader(TensorDataset(X, y), batch_size=32, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    pruned_model = train_with_pruning(
        model,
        dataloader,
        optimizer,
        criterion,
        device,
        num_epochs=9,
        prune_every=3,
        prune_amount=0.2
    )

    print("\nTraining complete. Final model:")
    print(pruned_model)
