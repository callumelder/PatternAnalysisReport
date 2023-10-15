from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


def crop_images():
    """
    crops the dataset images to only be of the brain
    """
    return

def filter_datasets():
    """filters dataset so patient images are kept together, preventing data leakage and overfitting

    Returns:
        _type_: _description_
    """
    return

def load_dataset(image_size, crop_size, batch_size, path):
    # transform
    train_transform = transforms.Compose([
        transforms.CenterCrop((crop_size, crop_size)),
        transforms.Resize((image_size, image_size)),
        transforms.Grayscale(),
        transforms.ToTensor(),
    ])
    
    test_transform = transforms.Compose([
        transforms.CenterCrop((crop_size, crop_size)),
        transforms.Resize((image_size, image_size)),
        transforms.Grayscale(),
        transforms.ToTensor(),
    ])

    # data    
    train_dataset = ImageFolder(path, transform=train_transform)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    test_dataset = ImageFolder(path, transform=test_transform)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    return train_dataloader, test_dataloader

# KEEP PATIENTS DATA TOGETHER TO PREVENT DATA LEAKAGE AND OVERFITTING