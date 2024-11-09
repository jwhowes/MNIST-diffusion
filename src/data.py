from torch.utils.data import Dataset
from datasets import load_dataset
from torchvision import transforms


class MNISTDataset(Dataset):
    def __init__(self, split):
        self.dataset = load_dataset("ylecun/mnist", split=split)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.1307,), std=(0.3081,)
            )
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx].values()

        return self.transform(image), label
