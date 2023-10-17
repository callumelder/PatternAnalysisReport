import torch
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from tqdm import tqdm, trange

from dataset import load_datasets
from modules import ViT

# device config
def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        print("WARNING: CUDA not found, using CPU")
    return device

def create_model(image_size, in_channels, patch_size, embedding_dims, num_heads, num_classes, patches):
    return ViT(img_size=image_size, 
               in_channels=in_channels,
               patch_size=patch_size,
               embedding_dims=embedding_dims,
               num_heads=num_heads,
               num_classes=num_classes,
               patches=patches)

def train_model(model, root, image_size, crop_size, batch_size, learning_rate, weight_decay, epochs):
    device = get_device()
    train_loader, _, _ = load_datasets(root, image_size, crop_size, batch_size)
    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = CrossEntropyLoss()
    model.train()
    
    for epoch in trange(epochs, desc="Training"):
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1} in training", leave=False):
            x, y = batch
            x, y = x.to(device), y.to(device)
            y_hat = model(x)
            loss = criterion(y_hat, y)

            train_loss += loss.detach().cpu().item() / len(train_loader)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}/{epochs} loss: {train_loss:.2f}")

def evaluate_model(model, root, image_size, crop_size, batch_size):
    device = get_device()
    _, _, validation_dataloader = load_datasets(root, image_size, crop_size, batch_size)
    criterion = CrossEntropyLoss()
    model.eval()
    
    with torch.no_grad():
        correct, total = 0, 0
        test_loss = 0.0
        for batch in tqdm(validation_dataloader, desc="Testing"):
            x, y = batch
            x, y = x.to(device), y.to(device)
            y_hat = model(x)
            loss = criterion(y_hat, y)
            test_loss += loss.detach().cpu().item() / len(validation_dataloader)

            correct += torch.sum(torch.argmax(y_hat, dim=1) == y).detach().cpu().item()
            total += len(x)
        print(f"Test loss: {test_loss:.2f}")
        print(f"Test accuracy: {correct / total * 100:.2f}%")
    return

def test_model():
    
    return

def save_model(model, file):
    """saves model to current working directory

    Args:
        model (<class 'modules.ViT'>): the vision transformer model to be saved
        file (string): file name of the model to be saved (needs .pth)
    """
    torch.save(model, file)
        