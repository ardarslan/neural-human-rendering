"""Dataset class template

This module provides a template for users to implement custom datasets.
You can specify '--dataset_mode template' to use this dataset.
The class name should be consistent with both the filename and its dataset_mode option.
The filename should be <dataset_mode>_dataset.py
The class name should be <Dataset_mode>Dataset.py
You need to implement the following functions:
    -- <modify_commandline_options>:　Add dataset-specific options and rewrite default values for existing options.
    -- <__init__>: Initialize this dataset class.
    -- <__getitem__>: Return a data point and its metadata information.
    -- <__len__>: Return the number of images.
"""
import os
from os import listdir
from os.path import isfile, join

from PIL import Image
from data.base_dataset import BaseDataset, get_transform, get_params


# from data.image_folder import make_dataset
# from PIL import Image


class TemplateDataset(BaseDataset):
    """A template dataset class for you to implement custom datasets."""
    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        parser.add_argument('--datasplit', type=str, default='train', help='directory')
        # parser.set_defaults(max_dataset_size=10)  # specify dataset-specific default values
        return parser

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions

        A few things can be done here.
        - save the options (have been done in BaseDataset)
        - get image paths and meta information of the dataset.
        - define the image transformation.
        """
        # save the option and dataset root
        BaseDataset.__init__(self, opt)
        # get the image paths of your dataset;
        root = opt.dataroot + opt.datasplit
        paths = []
        for path, dir, files in os.walk(root):
            if 'input' in path:
                for file in files:
                    paths.extend([(path + '/' + file).replace('\\', '/')])
        self.image_paths = paths  # You can call sorted(make_dataset(self.root, opt.max_dataset_size)) to get all the image paths under the directory self.root
        print(len(self.image_paths), self.image_paths[0])
        # define the default transform function. You can use <base_dataset.get_transform>; You can also define your custom transform function

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index -- a random integer for data indexing

        Returns:
            a dictionary of data with their names. It usually contains the data itself and its metadata information.

        Step 1: get a random image path: e.g., path = self.image_paths[index]
        Step 2: load your data from the disk: e.g., image = Image.open(path).convert('RGB').
        Step 3: convert your data to a PyTorch tensor. You can use helpder functions such as self.transform. e.g., data = self.transform(image)
        Step 4: return a data point as a dictionary.
        """
        path = self.image_paths[index]    # needs to be a string
        transform_params = get_params(self.opt, (self.opt.crop_size, self.opt.crop_size))
        self.transform_g = get_transform(self.opt, transform_params, grayscale=True)
        self.transform_rgb = get_transform(self.opt, transform_params, grayscale=False, normalize=True)
        data_A = self.transform_g(Image.open(path))    # needs to be a tensor
        data_B = self.transform_rgb(Image.open(path.replace('input', 'output')))    # needs to be a tensor
        return {'A': data_A, 'B': data_B, 'path': path}

    def __len__(self):
        """Return the total number of images."""
        return len(self.image_paths)
