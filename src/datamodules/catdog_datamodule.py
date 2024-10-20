from pathlib import Path
from typing import Union, Tuple
import os
import gdown
import zipfile
import os
import lightning as L
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.datasets.utils import download_and_extract_archive


class CatDogImageDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: Union[str, Path] = "data",
        num_workers: int = 0,
        batch_size: int = 8,
        splits: Tuple[float, float, float] = (0.8, 0.1, 0.1),
        pin_memory: bool = False,
    ):
        super().__init__()
        self._data_dir = Path(data_dir)
        self._num_workers = num_workers
        self._batch_size = batch_size
        self._splits = splits
        self._pin_memory = pin_memory
        self._dataset = None

    def prepare_data(self):
        
        print('prepare_data' + str(self.data_path.exists()))
        if not self.data_path.exists():
            
            # Google Drive file ID (from the shareable link)

            # URL to download the file
            download_url = f"https://drive.google.com/uc?id=1lylGUWzGUK9GKIIid3Pm0pEdKN7POYzh"

            # Name of the output zip file (it will be downloaded in the current directory)
            output_zip = 'data.zip'

            # Download the zip file
            gdown.download(download_url, output_zip, quiet=False)

            # Extract the zip file in the current directory
            with zipfile.ZipFile(output_zip, 'r') as zip_ref:
               zip_ref.extractall(".")  # Extract to the current directory

            # Optionally, remove the downloaded zip file after extraction
            os.remove(output_zip)

            print("Cat images downloaded and extracted successfully.")
            
    @property
    def data_path(self):
        return self._data_dir

    @property
    def normalize_transform(self):
        return transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    @property
    def train_transform(self):
        return transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                self.normalize_transform,
            ]
        )

    @property
    def valid_transform(self):
        return transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                self.normalize_transform,
            ]
        )

    def create_dataset(self, root, transform):
        return ImageFolder(root=root, transform=transform)

    def setup(self, stage: str = None):
        if self._dataset is None:
            self.train_dataset = self.create_dataset(
                self.data_path  / "test",
                self.train_transform,
            )
            self.val_dataset = self.create_dataset(
                self.data_path  / "validation",
                self.train_transform,
            )
            self.test_dataset = self.create_dataset(
                self.data_path  / "test",
                self.train_transform,
            )

    def __dataloader(self, dataset, shuffle: bool = False):
        return DataLoader(
            dataset=dataset,
            batch_size=self._batch_size,
            num_workers=self._num_workers,
            shuffle=shuffle,
            pin_memory=self._pin_memory,
        )

    def train_dataloader(self):
        return self.__dataloader(self.train_dataset, shuffle=True)

    def val_dataloader(self):
        return self.__dataloader(self.val_dataset)

    def test_dataloader(self):
        return self.__dataloader(self.test_dataset)