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


class oxford_iiit_pet(data.Dataset):
    def __init__(self, datasplit, dataset_dir, transform = None):
        """
        Initialize oxfordiiit dataset class
        Args:
            datasplit:         String indicating train/test data split being chosen
            dataset_dir:       Full path to directory containing data
            preprocessed_root: Full path to directory containing preprocessed data
            image_size:        Integer indicating size of image after preprocessing
            transform:         Pytorch transforms object to be applied to data
        Return:
            Nothing
        """
        curr_time = time.time()

        # Load dataset files
        if datasplit == 'train':
            csv_path = os.path.join(dataset_dir, 'oxford_pet_train.txt')

        elif datasplit == 'test':
            csv_path = os.path.join(dataset_dir, 'oxford_pet_test.txt')

        else:
            print("Error: Invalid dataset choice.")

        # END IF

        # Load dataset annotations
        image_file, props, bbs, c_cls, f_cls = self.pandas_csv_to_info(csv_path)
        size_dataset                         = len(image_file)

        print("csv file length: ", size_dataset)

        #images          = image_file
        #fine_class      = f_cls
        #coarse_class    = c_cls

        bboxes          = bbs
        props           = props
        bboxes          = [tuple(bbox) for bbox in bboxes]

        print("Dataset loaded in ", time.time() - curr_time, " secs.")
        print("Dataset size: "    , len(image_file))

        if len(image_file) == 0:
            raise RuntimeError

        # END IF


        self.image_root     = os.path.join(dataset_dir, "images")
        self.num_classes    = 37
        self.bboxes         = bboxes
        self.fine_class     = f_cls
        self.coarse_class   = c_cls
        self.image_paths    = image_file
        self.loader         = self.pil_loader
        self.num_instances  = len(self.image_paths)

        # Generate heirarchical labels
        self.heirarchy = self.generate_heirarchy()

        assert np.sum(self.heirarchy[1,:]) == 25, 'Error: number of dog classes is not 25 (README is incorrect) .. heiarchy miscalculated'

        # Normalization as instructed from pyTorch documentation
        self.transform = transform or transforms.Compose([transforms.ToTensor(),
                                                          transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                          std=(0.229, 0.224, 0.225))])

        print("NOTE: Bounding box information not provided due to missing xml files. Full images shown not object bounded (bbox cropping in pil_loader is commented).")

    def __getitem__(self, index):
        """
        Function that returns images from oxfordiiit dataset
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        # Load and transform image
        path        = os.path.join(self.image_root, self.image_paths[index])

        bbox        = self.bboxes[index]
        fine_cls    = self.fine_class[index]

        img         = self.loader(path, bbox)

        if self.transform is not None:
            img = self.transform(img)

        # onehot_label           = np.zeros(self.num_classes)
        # onehot_label[fine_cls] = 1

        return img, self.fine_class[index]

    def __len__(self):
        """
        Function that returns the value of num_instances variable
        Args:
            Nothing
        Returns:
            Number of instances properties
        """
        return self.num_instances

    def pil_loader(self, path, bbox):
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
        data_split = np.split(data, [0, 1, 2, 3, 4, 5, 9, 10, 11], axis=1)

        del(data_split[0])

        image_file   = np.squeeze(data_split[0]).tolist()
        pose         = np.squeeze(data_split[1]).tolist()
        truncated    = data_split[2].tolist()
        occluded     = data_split[3].tolist()
        difficult    = data_split[4].tolist()
        bbox         = data_split[5].tolist()
        coarse_class = np.squeeze(data_split[6]).tolist()
        fine_class   = np.squeeze(data_split[7]).tolist()
        props        = [ (pose[i], truncated[i], occluded[i], difficult[i]) for i in range(0, len(pose)) ]

        return image_file, props, bbox, coarse_class, fine_class

    def generate_heirarchy(self):
        """
        Function that generate heirarchical labels
        Args:
            Nothing
        Returns:
            Heirarchical labels connection array
        """
        heirarchy = np.zeros( (3, 37) )

        for i in range(self.num_instances):
            heirarchy[2, self.fine_class[i]] = self.fine_class[i]
            heirarchy[1, self.fine_class[i]] = self.coarse_class[i]

        # END FOR

        return heirarchy


    def generate_validation(self, ratio = 0.1):
        """
        Function that generates validation dataset from training split of data
        Args:
            ratio: Float value to indicate ratio of training data to be used as validation set
        Returns:
            validation set data loader
        """

        assert ratio > (2.*self.num_classes/float(self.num_instances)) and ratio < 0.5

        random.seed(a = 2741998)

        valid_class     = copy.deepcopy(self)

        valid_size      = int(ratio * self.num_instances)
        train_size      = self.num_instances - valid_size
        train_instances = list(range(0, self.num_instances))
        valid_instances = random.sample(train_instances, valid_size)
        train_instances = [x for x in train_instances if x not in valid_instances]

        assert train_size == len(train_instances) and valid_size == len(valid_instances)

        valid_class.image_paths     = [ self.image_paths[i]     for i in sorted(valid_instances) ]
        valid_class.bboxes          = [ self.bboxes[i]          for i in sorted(valid_instances) ]
        valid_class.fine_class      = [ self.fine_class[i]     for i in sorted(valid_instances) ]
        valid_class.coarse_class    = [ self.coarse_class[i]  for i in sorted(valid_instances) ]
        valid_class.num_instances   = valid_size

        self.image_paths     = [ self.image_paths[i]    for i in sorted(train_instances) ]
        self.bboxes          = [ self.bboxes[i]         for i in sorted(train_instances) ]
        self.fine_class      = [ self.fine_class[i]    for i in sorted(train_instances) ]
        self.coarse_class    = [ self.coarse_class[i] for i in sorted(train_instances) ]
        self.num_instances   = train_size

        return valid_class
