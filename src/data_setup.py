import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

def get_dataloaders(data_dir="./data", batch_size=64):

    transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)) 
    ])

    train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

    test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)

    return train_loader, test_loader

if __name__ == "__main__":
    # Get loaders
    train_loader, test_loader = get_dataloaders()

    # Print dataset sizes
    print("Number of training batches:", len(train_loader))
    print("Number of testing batches:", len(test_loader))

    # Pull one batch
    images, labels = next(iter(train_loader))
    print("Batch shape:", images.shape)
    print("Labels:", labels[:10].tolist())

    # Show the first image
    plt.imshow(images[0][0], cmap="gray")
    plt.title(f"Label: {labels[0].item()}")
    plt.show()