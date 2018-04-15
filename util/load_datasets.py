import os,sys, math
import torch
import numpy                    as np
import torchvision.transforms   as transforms

root_dir     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_data_loaders(dataset, batch_size, num_workers, model, valid = 0.0):

    image_size = 227
    train_transform   = transforms.Compose([transforms.ToTensor(),
                        transforms.Normalize(   mean=(0., 0., 0.),
                                                 std=(1./255., 1./255., 1./255.)
                                             ),
                        transforms.Normalize(   mean=(104, 116.668, 122.678),
                                                std=(1., 1., 1.)
                                            )
                        ])

    test_transform   = transforms.Compose([transforms.ToTensor(),
                        transforms.Normalize(   mean=(0., 0., 0.),
                                                 std=(1./255., 1./255., 1./255.)
                                             ),
                        transforms.Normalize(   mean=(104, 116.668, 122.678),
                                                std=(1., 1., 1.)
                                            )
                        ])


    if dataset == "pascal":
        csv_train = os.path.join(root_dir, 'data/pascal3d_train.csv')
        csv_test  = os.path.join(root_dir, 'data/pascal3d_valid.csv')

        train_set = pascal3d(csv_train, dataset_root= dataset_root, transform = train_transform, im_size = image_size)
        test_set  = pascal3d(csv_test,  dataset_root= dataset_root, transform = test_transform,  im_size = image_size)
    else:
        print("Error in load_datasets: Dataset name not defined.")


    # Generate validation dataset
    if valid > 0.0:
        valid_set   = train_set.generate_validation(valid)


    # Generate data loaders
    train_loader = torch.utils.data.DataLoader( dataset=train_set,
                                                batch_size=batch_size,
                                                shuffle=True,
                                                num_workers=num_workers,
                                                drop_last = True)

    test_loader  = torch.utils.data.DataLoader( dataset=test_set,
                                                batch_size=batch_size,
                                                num_workers=num_workers,
                                                drop_last = False)

    if valid > 0.0:
        print("Generated Validation Dataset - size : ", valid_set.num_instances)
        valid_loader = torch.utils.data.DataLoader( dataset     = valid_set,
                                                    batch_size  = batch_size,
                                                    shuffle     = False,
                                                    pin_memory  = True,
                                                    num_workers = num_workers,
                                                    drop_last = False)
    else:
        valid_loader = None

    return train_loader, valid_loader, test_loader
