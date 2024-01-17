from typing import Optional, Tuple
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from lightning import LightningDataModule
from timm.data.transforms_factory import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
try:
    from torchvision.transforms import InterpolationMode


    def _pil_interp(method):
        if method == 'bicubic':
            return InterpolationMode.BICUBIC
        elif method == 'lanczos':
            return InterpolationMode.LANCZOS
        elif method == 'hamming':
            return InterpolationMode.HAMMING
        else:
            # default bilinear, do we want to allow nearest?
            return InterpolationMode.BILINEAR


    import timm.data.transforms as timm_transforms

    timm_transforms._pil_interp = _pil_interp
except:
    from timm.data.transforms import _pil_interp



def build_transform(is_train=True):
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=224,
            is_training=True,
            color_jitter=0.4,
            auto_augment='rand-m9-mstd0.5-inc1',
            re_prob=0.25,
            re_mode='pixel',
            re_count=1,
            interpolation='bicubic',
        )

        return transform

    t = []

    size = int((256 / 224) * 224)
    t.append(
        transforms.Resize(size, interpolation=_pil_interp('bicubic')),
        # to maintain same ratio w.r.t. 224 images
    )
    t.append(transforms.CenterCrop(224))


    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)


class NetteDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = "/root/SharedData/datasets/imagenette2",
        batch_size: int = 128,
        num_workers: int = 16,
        pin_memory: bool = True,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)
        
        self.train_transforms = build_transform(is_train=True)
        
        
        # Define transforms here
        self.val_transforms = build_transform(is_train=False)

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
