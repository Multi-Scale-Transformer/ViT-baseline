from typing import Optional, Tuple
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from lightning import LightningDataModule

class NetteDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = "/root/SharedData/datasets/imagenette2",
        batch_size: int = 64,
        num_workers: int = 8,
        pin_memory: bool = True,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.train_transforms = transforms.Compose([
            transforms.Resize((256, 256)),  # 先将图片放大到稍微比目标尺寸大一点的尺寸
            transforms.RandomCrop((224, 224)),  # 随机裁剪到目标尺寸
            transforms.RandomHorizontalFlip(p=0.3),  # 随机水平翻转图片
            transforms.RandomRotation(8),  # 随机旋转图片±15度
            transforms.ColorJitter(brightness=0.03, contrast=0.03, saturation=0.03, hue=0.03),  # 随机调整亮度、对比度、饱和度和色调
            transforms.ToTensor(),  # 将图片转换为PyTorch张量
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 标准化
        ])        
        # Define transforms here
        self.val_transforms = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize images to common size
            transforms.ToTensor(),  # Convert images to PyTorch tensors
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize images
        ])

        self.data_train: Optional[ImageFolder] = None
        self.data_val: Optional[ImageFolder] = None
        self.batch_size_per_device = batch_size

    def num_classes(self) -> int:
        """Get the number of classes.

        :return: The number of Imagenette2 classes (10).
        """
        return 10

    def prepare_data(self) -> None:
        # No need to implement this for ImageNet as we expect data to be pre-downloaded
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size
                
            
        # Assuming data is already downloaded and located at self.hparams.data_dir
        self.data_train = ImageFolder(root=f"{self.hparams.data_dir}/train", transform=self.train_transforms)
        self.data_val = ImageFolder(root=f"{self.hparams.data_dir}/val", transform=self.val_transforms)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )
        
    def test_dataloader(self) -> DataLoader:
        return self.val_dataloader()

    def teardown(self, stage: Optional[str] = None) -> None:
        # Clean up if necessary
        pass
    
    


if __name__ == "__main__":
    _ = NetteDataModule()
