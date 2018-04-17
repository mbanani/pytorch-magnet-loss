from PIL import Image
import os
import numpy as np
import time
import pandas
import copy
import random
from torchvision import transforms
import torch.utils.data as data

from IPython import embed


class oxford_flowers(data.Dataset):
    def __init__(self, datasplit, dataset_dir, transform = None):
        """
        Initialize oxford_flowers dataset class
        Args:
            datasplit:         String indicating train/test data split being chosen
            dataset_dir:       Full path to directory containing data
            image_size:        Integer indicating size of image after preprocessing
            transform:         Pytorch transforms object to be applied to data
        Return:
            Nothing
        """
        curr_time = time.time()

        # Load dataset files
        if datasplit == 'train':
            csv_path = os.path.join(dataset_dir, 'train_labels.csv')
        elif datasplit == 'valid':
            csv_path = os.path.join(dataset_dir, 'valid_labels.csv')
        elif datasplit == 'test':
            csv_path = os.path.join(dataset_dir, 'test_labels.csv')
        else:
            print("Error: Invalid dataset choice.")
        # END IF

        # Load dataset annotations
        image_file, flower_cls  = self.pandas_csv_to_info(csv_path)
        size_dataset            = len(image_file)

        flower_cls = [fc -1 for fc in flower_cls]  # change range of classes

        print("csv file length: ", size_dataset)
        print("Dataset loaded in ", time.time() - curr_time, " secs.")
        print("Dataset size: "    , len(image_file))

        if len(image_file) == 0:  raise RuntimeError

        self.image_root     = os.path.join(dataset_dir, "jpg")
        self.num_classes    = 102
        self.classes        = flower_cls
        self.image_paths    = image_file
        self.loader         = self.pil_loader
        self.num_instances  = len(self.image_paths)
        self.read_order     = range(0, len(self.image_paths))

        # Normalization as instructed from pyTorch documentation
        self.transform = transform or transforms.Compose([transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225))]
        )


    def __getitem__(self, read_index):
        """
        Function that returns images from oxfordiiit dataset
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """

        index       = self.read_order[read_index]
        # Load and transform image
        path        = os.path.join(self.image_root, self.image_paths[index])
        img         = self.loader(path)

        if self.transform is not None:
            img = self.transform(img)

        return img, self.classes[index], index


    def update_read_order(self, new_order):
        self.read_order = new_order


    def default_read_order(self):
        self.read_order     = range(0, len(self.image_paths))

    def __len__(self):
        """
        Function that returns the value of num_instances variable
        Args:
            Nothing
        Returns:
            Number of instances properties
        """
        return len(self.read_order)

    def pil_loader(self, path):
        """
        Function that returns resized/cropped images from oxfordiiit dataset
        Args:
            path: Full path to name of image file
            bbox: Bounding box values for chosen image
        Returns:
            Cropped/Resized image from oxfordiiit dataset
        """
        # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
        with open(path, 'rb') as f:
            with Image.open(f) as img:

                img = img.convert('RGB')

                return img
            # END WITH
        # END WITH

    def pandas_csv_to_info(self, csv_path):
        """
        Function that parses the annotations text file into pandas csv format
        Args:
            csv_path: Full path to name of annotations text file
        Returns:
            tuple: (image_file, props, bbox, coarse_class, fine_class)
        """
        df   = pandas.read_csv(csv_path, sep=',')
        data = df.values

        # image, pose, truncated, occluded, difficult, xmin, ymin, xmax, ymax, cgrain, fgran
        data_split = np.split(data, [0, 1, 2], axis=1)

        del(data_split[0])

        image_file  = np.squeeze(data_split[0]).tolist()
        classes     = np.squeeze(data_split[1]).tolist()


        return image_file, classes
