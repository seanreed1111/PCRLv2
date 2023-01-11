"""
# A datamodule encapsulates the five steps involved in data processing in PyTorch:

# -- Download / tokenize / process.

# -- Clean and (maybe) save to disk.

# -- Load inside Dataset.

# -- Apply transforms (rotate, tokenize, etc…).

# -- Wrap inside a DataLoader.

Why do I need a DataModule?
In normal PyTorch code, the data cleaning/preparation is usually scattered across many files. This makes sharing and reusing the exact splits and transforms across projects impossible.

Datamodules are for you if you ever asked the questions:

what splits did you use?

what transforms did you use?

what normalization did you use?

how did you prepare/tokenize the data?
"""
# https://github.com/Lightning-AI/lightning-bolts/tree/master/pl_bolts/datamodules
# https://pytorch-lightning.readthedocs.io/en/latest/data/datamodule.html#prepare-data-per-node

import sys, functools, operator, logging, os, json
from argparse import ArgumentParser
from pathlib import Path
from datetime import datetime
from typing import Any, Callable, Optional
from pytorch_lightning import LightningDataModule

from monai.data import (
    PersistentDataset,
    DataLoader,
    Dataset,
    load_decathlon_datalist,
)
from monai.transforms import (
    AddChanneld,
    AsChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    RandCropByPosNegLabeld,
    RandSpatialCropSamplesd,
    ScaleIntensityRanged,
    Spacingd,
    SpatialPadd,
    ToTensord,
)


from loggers import create_python_logger

logger = create_python_logger(__name__, level=logging.INFO)


class MARSDataModule(LightningDataModule):
    """_summary_

    Args:
        LightningDataModule (_type_): _description_

    expects input 3d inputs of size 500x500x1
    Example::
        from pl_bolts.datamodules import MARSCTDataModule
        dm = ImagenetDataModule(RAW_CT_SRC_PATH)
        model = LitModel()
        Trainer().fit(model, datamodule=dm)
    """

    def __init__(self, args, *super_args, **super_kwargs) -> None:
        """
        Args:

        """
        super().__init__(*super_args, **super_kwargs)
        self.args = args
        self.base_data_dir = args.base_data_dir
        self.num_classes = args.num_classes
        self.unsupervised = True if not self.args.num_classes else False
        if not self.unsupervised:
            raise ("Supervised Datamodule not yet implemented")
        self.train_ds = None  # datasets assigned within setup
        self.val_ds = None  # datasets assigned within setup
        self.save_hyperparameters()

    def val_transforms(self):
        return Compose(
            [
                LoadImaged(keys=["image"]),
                AddChanneld(keys=["image"]),
                Orientationd(keys=["image"], axcodes="RAS"),
                ScaleIntensityRanged(
                    keys=["image"],
                    a_min=self.args.a_min,
                    a_max=self.args.a_max,
                    b_min=self.args.b_min,
                    b_max=self.args.b_max,
                    clip=True,
                ),
                SpatialPadd(
                    keys="image",
                    spatial_size=[self.args.roi_x, self.args.roi_y, self.args.roi_z],
                ),
                CropForegroundd(
                    keys=["image"],
                    source_key="image",
                    k_divisible=[self.args.roi_x, self.args.roi_y, self.args.roi_z],
                ),
                RandSpatialCropSamplesd(
                    keys=["image"],
                    roi_size=[self.args.roi_x, self.args.roi_y, self.args.roi_z],
                    num_samples=self.args.sw_batch_size,
                    random_center=True,
                    random_size=False,
                ),
                ToTensord(keys=["image"]),
            ]
        )

    def train_transforms(self):
        return Compose(
            [
                LoadImaged(keys=["image"]),
                AddChanneld(keys=["image"]),
                # Orientationd(keys=["image"], axcodes="RAS"),
                ScaleIntensityRanged(
                    keys=["image"],
                    a_min=self.args.a_min,
                    a_max=self.args.a_max,
                    b_min=self.args.b_min,
                    b_max=self.args.b_max,
                    clip=True,
                ),
                # Spacingd(
                #     keys=["image"],
                #     pixdim=(2.0),
                #     # mode=("bilinear", "nearest"),
                # ),
                SpatialPadd(
                    keys="image",
                    spatial_size=[self.args.roi_x, self.args.roi_y, self.args.roi_z],
                ),
                CropForegroundd(
                    keys=["image"],
                    source_key="image",
                    k_divisible=[self.args.roi_x, self.args.roi_y, self.args.roi_z],
                ),
                RandSpatialCropSamplesd(
                    keys=["image"],
                    roi_size=[self.args.roi_x, self.args.roi_y, self.args.roi_z],
                    num_samples=self.args.sw_batch_size,
                    random_center=True,
                    random_size=False,
                ),
                ToTensord(keys=["image"]),
            ]
        )

    def prepare_data(self):
        pass
        #######     self.prepare_data_per_node ##########
        # If set to True will call prepare_data() on LOCAL_RANK=0 for every node. If set to False will only call from NODE_RANK=0, LOCAL_RANK=0.
        # class LitDataModule(LightningDataModule):
        #     def __init__(self):
        #         super().__init__()
        #         self.prepare_data_per_node = True

    # setup is called from every process across all the nodes. Setting state here is recommended.
    # see this link for how to split data among different devices in the setup and dataloaders
    ################ https://developer.habana.ai/tutorials/pytorch-lightning/finetune-transformers-models-with-pytorch-lightning/
    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        dataset_json_path = self.generate_dataset_json()
        with open(dataset_json_path, "r") as f:
            contents = json.load(f)
        train_list = load_decathlon_datalist(
            dataset_json_path, False, "training", base_dir=self.base_data_dir
        )
        val_list = load_decathlon_datalist(
            dataset_json_path, False, "validation", base_dir=self.base_data_dir
        )
        if stage == "fit" or stage is None:
            if self.args.persistent_dataset:
                print("Using MONAI Persistent Dataset")
                self.train_ds = PersistentDataset(
                    data=train_list,
                    num_workers=args.num_dataloader_workers,
                    transform=self.train_transforms(),
                    cache_dir=Path.cwd() / "train_persistent_dir",
                )
            elif self.args.cache_dataset:
                print("Using MONAI CacheDataset")
                # num_workers = None means use default of os.cpu_count()
                self.train_ds = CacheDataset(
                    data=train_list,
                    transform=self.train_transforms(),
                    runtime_cache=True,
                    cache_num=2 * args.batch_size * args.sw_batch_size,
                    num_workers=args.cache_dataset_workers,
                    progress=False,
                )
            # elif self.args.smartcache_dataset:
            #     assert False, "Do not use smartcache_dataset w/o setting up the appropriate handler"
            #     print("Using MONAI SmartCacheDataset")
            #     #num_workers = None means use default of os.cpu_count()
            #     self.train_ds = SmartCacheDataset(
            #         data=train_list, transform=self.train_transforms(), cache_num=2 * args.batch_size * args.sw_batch_size, replace_rate=1.0, num_workers=args.smartcache_dataset_workers
            #     )
            else:
                print("Using generic monai dataset")
                self.train_ds = Dataset(
                    data=train_list, transform=self.train_transforms()
                )

            self.val_ds = Dataset(data=val_list, transform=self.val_transforms())

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_dataloader_workers,
            persistent_workers=False,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_dataloader_workers,
            shuffle=False,
            drop_last=True,
        )

    def generate_dataset_json(self):
        # base_datadir is the dir that contains imagesTr, imagesTs, and imagesVal dirs
        # logger = make_python_logger(__name__)

        # helper functions

        def make_list_of_dicts_from_path_list_and_key(path_list, key="image"):
            result = []
            for path in path_list:
                result.append({key: str(path)})
            return result

        def create_image_path_list_from_dir(image_dir_path):
            """image_dir_path should be an absolute path"""
            generators = [
                child.glob("**/*.nii.gz") for child in image_dir_path.iterdir()
            ]
            # convert generators into a list of lists, one for each dir
            image_path_list = [list(generator) for generator in generators]
            # return the flattened list
            return functools.reduce(operator.iconcat, image_path_list, [])

        path_to_output_json_file = Path(f"{self.args.default_dataset_json_filepath}")
        path_to_output_json_file.parent.mkdir(exist_ok=True, parents=True)

        # expects to have studies under the base_data_dir on Azure storage
        # already separated into folders "imagesTr", "imagesVal", and "imagesTs"
        base_data_dir = Path(self.args.base_data_dir).absolute()
        training_dir = base_data_dir / "imagesTr"
        validation_dir = base_data_dir / "imagesVal"
        test_dir = base_data_dir / "imagesTs"

        if not base_data_dir.exists():
            raise RuntimeError("base data directory does not exist")
            sys.exit(1)
        if not training_dir.exists():
            logger.WARNING("'imagesTr' directory does not exist")
        if not validation_dir.exists():
            logger.WARNING("'imagesVal' directory does not exist")
        if not test_dir.exists():
            logger.WARNING("'imagesTs' directory does not exist")

        result = {}
        result["modality"] = {"0": "CT"}
        result["date_generated"] = f"{datetime.now()}"
        result["baseDataDir"] = str(base_data_dir.absolute())
        result["numTraining"] = None
        result["numValidation"] = None
        result["numTest"] = None
        result["training"] = []  # list of dicts, all having key="image"
        result["validation"] = []  # list of dicts, all having key="image"
        result["test"] = []  # list of dicts, all having key="image"

        if training_dir.exists():
            training_path_list = create_image_path_list_from_dir(training_dir)
            if self.args.debug_get_loader:
                training_path_list = training_path_list[-5:].copy()
            result["training"] = make_list_of_dicts_from_path_list_and_key(
                training_path_list, key="image"
            )
            result["numTraining"] = len(result["training"])
            logger.debug(f"training_path_list={training_path_list}\n")
            print(f"training_path_list={training_path_list}\n")

        if validation_dir.exists():
            val_path_list = create_image_path_list_from_dir(validation_dir)
            if self.args.debug_get_loader:
                val_path_list = val_path_list[-5:].copy()
            result["validation"] = make_list_of_dicts_from_path_list_and_key(
                val_path_list, key="image"
            )
            result["numValidation"] = len(result["validation"])
            logger.debug(f"val_path_list={val_path_list}\n")
            print(f"val_path_list={val_path_list}\n")
        if test_dir.exists():
            test_path_list = create_image_path_list_from_dir(test_dir)
            if self.args.debug_get_loader:
                test_path_list = test_path_list[-5:].copy()
            result["test"] = make_list_of_dicts_from_path_list_and_key(
                test_path_list, key="image"
            )
            result["numTest"] = len(result["test"])
            logger.debug(f"test_path_list={test_path_list}\n")
            print(f"test_path_list={test_path_list}\n")
        logger.info(f"path_to_output_json_file = {path_to_output_json_file}")
        print(f"path_to_output_json_file = {path_to_output_json_file}")
        with open(path_to_output_json_file, "w") as f:
            json.dump(result, f, indent=4)
        logger.info("json file created successfully")
        print("json file created successfully")
        return path_to_output_json_file


if __name__ == "__main__":
    pass
