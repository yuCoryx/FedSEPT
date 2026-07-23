"""
Domain-skew data utilities for federated learning with domain adaptation.

本文件只负责**跨域数据集**（Office / Office31 / Office-Home / PACS / DomainNet 等）的处理，
即：给定若干域（domains），先按域划分数据，再在每个域内部平均切分为若干个客户端。

"""

import os
import sys
import numpy as np
from collections import Counter
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from PIL import Image


base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)






class Datum:
    """Data instance which defines the basic attributes.

    Args:
        impath (str): Image file path.
        label (int): Class label.
        domain (int): Domain label.
        classname (str): Class name.
    """

    def __init__(self, impath, label=0, domain=0, classname=""):
        self._impath = impath
        self._label = label
        self._domain = domain
        self._classname = classname

    @property
    def impath(self):
        return self._impath

    @property
    def label(self):
        return self._label

    @property
    def domain(self):
        return self._domain

    @property
    def classname(self):
        return self._classname






class OfficeDataset(Dataset):
    """Office-Caltech-10 dataset loader.
    
    Args:
        base_path (str): Base path to data directory.
        site (str): Domain name (amazon, caltech, dslr, webcam).
        net_dataidx_map (list, optional): Indices for data subset.
        train (bool): Whether to load training set.
        transform: Image transformations.
        subset_train_num (int): Number of training samples per subset.
        subset_capacity (int): Capacity of each subset.
    """
    
    def __init__(self, base_path, site, net_dataidx_map=None, train=True, 
                 transform=None, subset_train_num=7, subset_capacity=10):
        self.base_path = base_path
        self.site = site
        
        
        self.imagefolder_obj = ImageFolder(
            os.path.join(self.base_path, 'Office-Caltech-10', site), 
            transform
        )
        self.train = train
        
        
        all_data = self.imagefolder_obj.samples
        self.train_data_list = []
        self.test_data_list = []
        for i in range(len(all_data)):
            if i % subset_capacity <= subset_train_num:
                self.train_data_list.append(all_data[i])
            else:
                self.test_data_list.append(all_data[i])

        
        if train:
            data_list = np.array(self.train_data_list)
        else:
            data_list = np.array(self.test_data_list)

        
        self.site_domain = {'amazon': 0, 'caltech': 1, 'dslr': 2, 'webcam': 3}
        self.domain = self.site_domain[site]

        
        self.imgs = data_list[:, 0]
        self.label = (data_list[:, 1]).astype('int32')

        
        self.classnames = self.imagefolder_obj.classes
        self.lab2cname = {i: self.classnames[i] for i in range(len(self.classnames))}
        
        
        if train:
            print('Counter({}_train data:)'.format(site), Counter(self.label))
        else:
            print('Counter({}_test data:)'.format(site), Counter(self.label))
        
        
        if net_dataidx_map is not None:
            self.imgs = self.imgs[net_dataidx_map]
            self.label = self.label[net_dataidx_map]

        self.transform = transform
        self.data_detailed = self._convert()

    def __len__(self):
        return len(self.label)

    def _convert(self):
        """Convert to Datum objects."""
        data_with_label = []
        for i in range(len(self.label)):
            img_path = self.imgs[i]
            data_idx = img_path
            target_idx = int(self.label[i])
            notation_idx = self.imagefolder_obj.classes[target_idx]
            item = Datum(
                impath=data_idx, 
                label=int(target_idx), 
                domain=int(self.domain), 
                classname=notation_idx
            )
            data_with_label.append(item)
        return data_with_label

    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        label = self.label[idx]
        image = Image.open(img_path)

        if len(image.split()) != 3:
            image = transforms.Grayscale(num_output_channels=3)(image)

        if self.transform is not None:
            image = self.transform(image)

        return image, label


class Office31Dataset(Dataset):
    """Office-31 dataset loader.
    
    Args:
        base_path (str): Base path to data directory.
        site (str): Domain name (amazon, dslr, webcam).
        net_dataidx_map (list, optional): Indices for data subset.
        train (bool): Whether to load training set.
        transform: Image transformations.
        subset_train_num (int): Number of training samples per subset.
        subset_capacity (int): Capacity of each subset.
    """
    
    def __init__(self, base_path, site, net_dataidx_map=None, train=True, 
                 transform=None, subset_train_num=7, subset_capacity=10):
        site = site.lower()
        self.base_path = base_path
        self.site = site
        
        
        self.imagefolder_obj = ImageFolder(
            os.path.join(self.base_path, 'office31', site), 
            transform
        )
        self.train = train
        
        
        all_data = self.imagefolder_obj.samples
        self.train_data_list = []
        self.test_data_list = []
        for i in range(len(all_data)):
            if i % subset_capacity <= subset_train_num:
                self.train_data_list.append(all_data[i])
            else:
                self.test_data_list.append(all_data[i])

        
        if train:
            data_list = np.array(self.train_data_list)
        else:
            data_list = np.array(self.test_data_list)

        
        self.site_domain = {'amazon': 0, 'dslr': 1, 'webcam': 2}

        
        self.imgs = data_list[:, 0]
        self.label = (data_list[:, 1]).astype('int32')

        
        if net_dataidx_map is not None:
            self.imgs = self.imgs[net_dataidx_map]
            self.label = self.label[net_dataidx_map]

        self.transform = transform
        self.data_detailed = self._convert()

    def __len__(self):
        return len(self.label)

    def _convert(self):
        """Convert to Datum objects."""
        data_with_label = []
        for i in range(len(self.label)):
            img_path = self.imgs[i]
            data_idx = img_path
            target_idx = int(self.label[i])
            notation_idx = self.imagefolder_obj.classes[target_idx]
            item = Datum(
                impath=data_idx, 
                label=int(target_idx), 
                domain=self.site, 
                classname=notation_idx
            )
            data_with_label.append(item)
        return data_with_label

    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        label = self.label[idx]
        image = Image.open(img_path)

        if len(image.split()) != 3:
            image = transforms.Grayscale(num_output_channels=3)(image)

        if self.transform is not None:
            image = self.transform(image)

        return image, label


class OfficeHomeDataset(Dataset):
    """Office-Home dataset loader.
    
    Args:
        base_path (str): Base path to data directory.
        site (str): Domain name (Art, Clipart, Product, Real_World).
        net_dataidx_map (list, optional): Indices for data subset.
        train (bool): Whether to load training set.
        transform: Image transformations.
        subset_train_num (int): Number of training samples per subset.
        subset_capacity (int): Capacity of each subset.
    """
    
    def __init__(self, base_path, site, net_dataidx_map=None, train=True, 
                 transform=None, subset_train_num=7, subset_capacity=10):
        import logging
        logger = logging.getLogger(__name__)
        
        self.base_path = base_path
        self.site = site
        
        
        
        domain_name_map = {
            'Real_World': 'Real World',
            'Real World': 'Real World',  
        }
        actual_site = domain_name_map.get(site, site)
        
        
        data_path = os.path.join(self.base_path, 'Office-Home', actual_site)
        logger.info(f"Loading OfficeHome dataset from {data_path} (train={train}, site={site})")
        self.imagefolder_obj = ImageFolder(
            data_path, 
            transform=None  
        )
        logger.info(f"ImageFolder loaded: {len(self.imagefolder_obj.samples)} samples, {len(self.imagefolder_obj.classes)} classes")
        self.train = train
        
        
        all_data = self.imagefolder_obj.samples
        self.train_data_list = []
        self.test_data_list = []
        for i in range(len(all_data)):
            if i % subset_capacity <= subset_train_num:
                self.train_data_list.append(all_data[i])
            else:
                self.test_data_list.append(all_data[i])
        
        logger.info(f"Split: {len(self.train_data_list)} train, {len(self.test_data_list)} test")
        
        
        if train:
            data_list = np.array(self.train_data_list)
        else:
            data_list = np.array(self.test_data_list)
        
        
        self.site_domain = {'Art': 0, 'Clipart': 1, 'Product': 2, 'Real_World': 3}
        
        
        self.imgs = data_list[:, 0]
        self.label = (data_list[:, 1]).astype('int32')
        
        
        if net_dataidx_map is not None:
            self.imgs = self.imgs[net_dataidx_map]
            self.label = self.label[net_dataidx_map]
        
        self.transform = transform
        logger.info(f"Converting {len(self.label)} samples to Datum objects...")
        self.data_detailed = self._convert()
        logger.info(f"Conversion complete: {len(self.data_detailed)} Datum objects")

    def __len__(self):
        return len(self.label)

    def _convert(self):
        """Convert to Datum objects."""
        data_with_label = []
        for i in range(len(self.label)):
            img_path = self.imgs[i]
            data_idx = img_path
            target_idx = int(self.label[i])
            notation_idx = self.imagefolder_obj.classes[target_idx]
            item = Datum(
                impath=data_idx, 
                label=int(target_idx), 
                domain=self.site, 
                classname=notation_idx
            )
            data_with_label.append(item)
        return data_with_label

    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        label = self.label[idx]
        image = Image.open(img_path)

        if len(image.split()) != 3:
            image = transforms.Grayscale(num_output_channels=3)(image)

        if self.transform is not None:
            image = self.transform(image)

        return image, label


class PACSDataset(Dataset):
    """PACS dataset loader.
    
    Args:
        base_path (str): Base path to data directory.
        site (str): Domain name (art_painting, cartoon, photo, sketch).
        net_dataidx_map (list, optional): Indices for data subset.
        train (bool): Whether to load training set.
        transform: Image transformations.
        subset_train_num (int): Number of training samples per subset.
        subset_capacity (int): Capacity of each subset.
    """
    
    def __init__(self, base_path, site, net_dataidx_map=None, train=True, 
                 transform=None, subset_train_num=7, subset_capacity=10):
        site = site.lower()
        self.base_path = base_path
        self.site = site
        
        
        self.imagefolder_obj = ImageFolder(
            os.path.join(self.base_path, 'PACS', site), 
            transform=None  
        )
        self.train = train
        
        
        all_data = self.imagefolder_obj.samples
        self.train_data_list = []
        self.test_data_list = []
        for i in range(len(all_data)):
            if i % subset_capacity <= subset_train_num:
                self.train_data_list.append(all_data[i])
            else:
                self.test_data_list.append(all_data[i])

        
        if train:
            data_list = np.array(self.train_data_list)
        else:
            data_list = np.array(self.test_data_list)

        
        self.site_domain = {'art_painting': 0, 'cartoon': 1, 'photo': 2, 'sketch': 3}

        
        self.imgs = data_list[:, 0]
        self.label = (data_list[:, 1]).astype('int32')
        
        
        if net_dataidx_map is not None:
            self.imgs = self.imgs[net_dataidx_map]
            self.label = self.label[net_dataidx_map]
        
        self.transform = transform
        self.data_detailed = self._convert()

    def __len__(self):
        return len(self.label)

    def _convert(self):
        """Convert to Datum objects."""
        data_with_label = []
        for i in range(len(self.label)):
            img_path = self.imgs[i]
            data_idx = img_path
            target_idx = int(self.label[i])
            notation_idx = self.imagefolder_obj.classes[target_idx]
            item = Datum(
                impath=data_idx, 
                label=int(target_idx), 
                domain=self.site, 
                classname=notation_idx
            )
            data_with_label.append(item)
        return data_with_label

    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        label = self.label[idx]
        image = Image.open(img_path)

        if len(image.split()) != 3:
            image = transforms.Grayscale(num_output_channels=3)(image)

        if self.transform is not None:
            image = self.transform(image)

        return image, label


class DomainNetDataset(Dataset):
    """DomainNet dataset loader.
    
    Args:
        base_path (str): Base path to data directory.
        site (str): Domain name (clipart, infograph, painting, quickdraw, real, sketch).
        net_dataidx_map (list, optional): Indices for data subset.
        train (bool): Whether to load training set.
        transform: Image transformations.
    """
    
    def __init__(
        self,
        base_path,
        site,
        net_dataidx_map=None,
        train=True,
        transform=None,
        num_classes=None,
    ):
        site = site.lower()
        self.base_path = base_path
        self.site = site
        self.num_classes = int(num_classes) if num_classes is not None else None
        
        
        if train:
            self.split_file = os.path.join(
                self.base_path, 'DomainNet/{}_train.txt'.format(site)
            )
        else:
            self.split_file = os.path.join(
                self.base_path, 'DomainNet/{}_test.txt'.format(site)
            )
        
        self.imgs, self.notation, self.label = DomainNetDataset.read_txt(
            self.split_file, 
            os.path.join(self.base_path, 'DomainNet')
        )
        
        
        self.site_domain = {
            'clipart': 0, 'infograph': 1, 'painting': 2, 
            'quickdraw': 3, 'real': 4, 'sketch': 5
        }
        
        
        self.imgs = np.asarray(self.imgs)
        self.label = np.asarray(self.label)
        self.notation = np.asarray(self.notation)

        
        
        if self.num_classes is not None:
            keep_mask = self.label < self.num_classes
            self.imgs = self.imgs[keep_mask]
            self.label = self.label[keep_mask]
            
            self.notation = self.notation[keep_mask]
        
        
        if net_dataidx_map is not None:
            self.imgs = self.imgs[net_dataidx_map]
            self.label = self.label[net_dataidx_map]
            self.notation = self.notation[net_dataidx_map].tolist()
        
        self.transform = transform
        self.data_detailed = self._convert()

    @staticmethod
    def read_txt(txt_path, root_path):
        """Read image paths and labels from text file."""
        imgs = []
        notations = []
        targets = []
        with open(txt_path, 'r') as f:
            txt_component = f.readlines()
        for line_txt in txt_component:
            label_name = line_txt.split('/')[1]
            line_txt = line_txt.replace('\n', '').split(' ')
            img_path = os.path.join(root_path, line_txt[0])
            
            
            
            if not os.path.exists(img_path):
                
                path_parts = line_txt[0].split('/')
                if len(path_parts) > 0:
                    domain_name = path_parts[0]
                    
                    nested_path = os.path.join(domain_name, line_txt[0])
                    nested_img_path = os.path.join(root_path, nested_path)
                    if os.path.exists(nested_img_path):
                        img_path = nested_img_path
            
            imgs.append(img_path)
            notations.append(label_name)
            targets.append(int(line_txt[1]))
        return imgs, notations, targets

    def __len__(self):
        return len(self.label)

    def _convert(self):
        """Convert to Datum objects."""
        data_with_label = []
        for i in range(len(self.label)):
            img_path = self.imgs[i]
            data_idx = img_path
            target_idx = self.label[i]
            notation_idx = self.notation[i]
            item = Datum(
                impath=data_idx, 
                label=int(target_idx), 
                domain=self.site, 
                classname=notation_idx
            )
            data_with_label.append(item)
        return data_with_label

    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        label = self.label[idx]
        image = Image.open(img_path)

        if len(image.split()) != 3:
            image = transforms.Grayscale(num_output_channels=3)(image)

        if self.transform is not None:
            image = self.transform(image)

        return image, label






def get_domain_client_num(domain_num, total_client_num):
    """Distribute total clients across domains.
    
    Args:
        domain_num (int): Number of domains.
        total_client_num (int): Total number of clients.
    
    Returns:
        np.ndarray: Number of clients per domain.
    """
    
    
    n_clients = total_client_num // domain_num
    domain_client_num = np.ones(domain_num, dtype=int) * n_clients
    return domain_client_num


def record_net_data_stats(y_train, net_dataidx_map):
    """Record class distribution for each client.
    
    Args:
        y_train (np.ndarray): Training labels.
        net_dataidx_map (dict): Mapping from client ID to data indices.
    
    Returns:
        dict: Class counts for each client.
    """
    net_cls_counts = {}
    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp
    return net_cls_counts


def Dataset_partition_domain(global_domain_trainset, global_domain_testset, beta, K=None, 
                            n_parties=5, min_require_size=2):
    """Partition domain dataset for federated learning.
    
    Args:
        global_domain_trainset: Training dataset object.
        global_domain_testset: Test dataset object.
        beta (float): Dirichlet distribution parameter (0 for IID, >0 for non-IID).
        K (int, optional): Number of classes. If None, will be inferred from dataset.
        n_parties (int): Number of clients.
        min_require_size (int): Minimum samples per client.
    
    Returns:
        tuple: (train_dataidx_map, test_dataidx_map) dictionaries mapping client ID to indices.
    """
    min_size = 0

    train_path = global_domain_trainset.imgs
    train_labels = global_domain_trainset.label
    train_labels = np.array(train_labels)
    test_path = global_domain_testset.imgs
    test_labels = global_domain_testset.label
    test_labels = np.array(test_labels)

    N_train = len(train_labels)
    
    
    if K is None:
        if hasattr(global_domain_trainset, 'classnames'):
            K = len(global_domain_trainset.classnames)
        elif hasattr(global_domain_trainset, 'imagefolder_obj'):
            K = len(global_domain_trainset.imagefolder_obj.classes)
        else:
            K = len(np.unique(train_labels))
    
    net_dataidx_map_train = {}
    net_dataidx_map_test = {}

    
    while min_size < min_require_size:
        idx_batch_train = [[] for _ in range(n_parties)]
        idx_batch_test = [[] for _ in range(n_parties)]
        
        for k in range(K):
            train_idx_k = np.where(train_labels == k)[0]
            test_idx_k = np.where(test_labels == k)[0]
            train_idx_k = np.array(train_idx_k)
            test_idx_k = np.array(test_idx_k)
            
            
            np.random.seed(0)
            np.random.shuffle(train_idx_k)
            np.random.shuffle(test_idx_k)
            
            if beta == 0:
                
                idx_batch_train = [
                    idx_j + idx.tolist() 
                    for idx_j, idx in zip(idx_batch_train, np.array_split(train_idx_k, n_parties))
                ]
                idx_batch_test = [
                    idx_j + idx.tolist() 
                    for idx_j, idx in zip(idx_batch_test, np.array_split(test_idx_k, n_parties))
                ]
            else:
                
                np.random.seed(0)  
                proportions = np.random.dirichlet(np.repeat(beta, n_parties))
                proportions = np.array([
                    p * (len(idx_j) < N_train / n_parties) 
                    for p, idx_j in zip(proportions, idx_batch_train)
                ])
                proportions = proportions / proportions.sum()
                
                proportions_train = (np.cumsum(proportions) * len(train_idx_k)).astype(int)[:-1]
                proportions_test = (np.cumsum(proportions) * len(test_idx_k)).astype(int)[:-1]
                train_part_list = np.split(train_idx_k, proportions_train)
                test_part_list = np.split(test_idx_k, proportions_test)
                
                idx_batch_train = [
                    idx_j + idx.tolist() 
                    for idx_j, idx in zip(idx_batch_train, train_part_list)
                ]
                idx_batch_test = [
                    idx_j + idx.tolist() 
                    for idx_j, idx in zip(idx_batch_test, test_part_list)
                ]

            min_size_train = min([len(idx_j) for idx_j in idx_batch_train])
            min_size_test = min([len(idx_j) for idx_j in idx_batch_test])
            min_size = min(min_size_test, min_size_train)

    
    for j in range(n_parties):
        np.random.seed(0)  
        np.random.shuffle(idx_batch_train[j])
        np.random.shuffle(idx_batch_test[j])
        net_dataidx_map_train[j] = idx_batch_train[j]
        net_dataidx_map_test[j] = idx_batch_test[j]

    
    traindata_cls_counts = record_net_data_stats(train_labels, net_dataidx_map_train)
    
    testdata_cls_counts = record_net_data_stats(test_labels, net_dataidx_map_test)
    
    
    return net_dataidx_map_train, net_dataidx_map_test






def prepare_data_domain_partition_train(cfg, data_base_path):
    """Prepare and partition data for federated learning training.
    
    Args:
        cfg: Configuration object with dataset settings.
        data_base_path (str): Base path to data directory.
    
    Returns:
        tuple: (train_set, test_set, global_test_set, classnames, lab2cname)
    """
    
    transform_train = transforms.Compose([
        transforms.Resize([256, 256]),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation((-30, 30)),
        transforms.ToTensor(),
    ])
    
    transform_test = transforms.Compose([
        transforms.Resize([256, 256]),
        transforms.ToTensor(),
    ])
    
    
    total_client_num = cfg.DATASET.USERS
    domain_name = cfg.DATASET.SOURCE_DOMAINS
    domain_num = len(domain_name)
    min_pic_require_size = 2
    domain_client_num = get_domain_client_num(domain_num, total_client_num)
    
    all_domain_trainset = []
    all_domain_testset = []
    global_test_set = []
    
    
    for domain_name_index in range(domain_num):
        current_domain_name = domain_name[domain_name_index]
        domain_n_clients = int(domain_client_num[domain_name_index])
        
        
        if cfg.DATASET.NAME == 'Office':
            global_domain_trainset = OfficeDataset(
                data_base_path, current_domain_name, 
                transform=transform_train, train=True
            )
            global_domain_testset = OfficeDataset(
                data_base_path, current_domain_name, 
                transform=transform_test, train=False
            )
            net_dataidx_map_train, net_dataidx_map_test = Dataset_partition_domain(
                global_domain_trainset, global_domain_testset,
                beta=cfg.DATASET.BETA,
                K=None,  
                n_parties=domain_n_clients,
                min_require_size=min_pic_require_size
            )
        elif cfg.DATASET.NAME == 'DomainNet':
            dn_num_classes = getattr(cfg.DATASET, "DOMAINNET_NUM_CLASSES", None)
            global_domain_trainset = DomainNetDataset(
                data_base_path, current_domain_name, 
                transform=transform_train, train=True,
                num_classes=dn_num_classes
            )
            global_domain_testset = DomainNetDataset(
                data_base_path, current_domain_name, 
                transform=transform_test, train=False,
                num_classes=dn_num_classes
            )
            net_dataidx_map_train, net_dataidx_map_test = Dataset_partition_domain(
                global_domain_trainset, global_domain_testset,
                beta=cfg.DATASET.BETA,
                K=int(dn_num_classes) if dn_num_classes is not None else 365,
                n_parties=domain_n_clients,
                min_require_size=min_pic_require_size
            )
        elif cfg.DATASET.NAME == 'PACS':
            global_domain_trainset = PACSDataset(
                data_base_path, current_domain_name, 
                transform=transform_train, train=True
            )
            global_domain_testset = PACSDataset(
                data_base_path, current_domain_name, 
                transform=transform_test, train=False
            )
            net_dataidx_map_train, net_dataidx_map_test = Dataset_partition_domain(
                global_domain_trainset, global_domain_testset,
                beta=cfg.DATASET.BETA,
                K=7, n_parties=domain_n_clients,
                min_require_size=min_pic_require_size
            )
        elif cfg.DATASET.NAME == 'Office31':
            global_domain_trainset = Office31Dataset(
                data_base_path, current_domain_name, 
                transform=transform_train, train=True
            )
            global_domain_testset = Office31Dataset(
                data_base_path, current_domain_name, 
                transform=transform_test, train=False
            )
            net_dataidx_map_train, net_dataidx_map_test = Dataset_partition_domain(
                global_domain_trainset, global_domain_testset,
                beta=cfg.DATASET.BETA,
                K=len(global_domain_trainset.imagefolder_obj.classes),
                n_parties=domain_n_clients,
                min_require_size=min_pic_require_size
            )
        elif cfg.DATASET.NAME == 'OfficeHome':
            global_domain_trainset = OfficeHomeDataset(
                data_base_path, current_domain_name, 
                transform=transform_train, train=True
            )
            global_domain_testset = OfficeHomeDataset(
                data_base_path, current_domain_name, 
                transform=transform_test, train=False
            )
            net_dataidx_map_train, net_dataidx_map_test = Dataset_partition_domain(
                global_domain_trainset, global_domain_testset,
                beta=cfg.DATASET.BETA,
                K=len(global_domain_trainset.imagefolder_obj.classes),
                n_parties=domain_n_clients,
                min_require_size=min_pic_require_size
            )
        
        
        if hasattr(global_domain_testset, 'imagefolder_obj'):
            classnames = global_domain_trainset.imagefolder_obj.classes
            lab2cname = {i: classnames[i] for i in range(len(classnames))}
        else:
            lab2cname = dict(zip(global_domain_testset.label, global_domain_testset.notation))
            classnames = [lab2cname[key] for key in sorted(lab2cname.keys())]
        
        global_domain_trainset = global_domain_trainset.data_detailed
        global_domain_testset = global_domain_testset.data_detailed
        
        
        domain_trainset = [[] for i in range(domain_n_clients)]
        domain_testset = [[] for i in range(domain_n_clients)]
        
        for i in range(domain_n_clients):
            if cfg.DATASET.NAME == 'Office':
                domain_trainset[i] = OfficeDataset(
                    data_base_path, current_domain_name, 
                    net_dataidx_map_train[i], transform=transform_train
                )
                domain_testset[i] = OfficeDataset(
                    data_base_path, current_domain_name, 
                    net_dataidx_map_test[i], train=False, transform=transform_test
                ).data_detailed
            elif cfg.DATASET.NAME == 'DomainNet':
                dn_num_classes = getattr(cfg.DATASET, "DOMAINNET_NUM_CLASSES", None)
                domain_trainset[i] = DomainNetDataset(
                    data_base_path, current_domain_name, 
                    net_dataidx_map_train[i], transform=transform_train,
                    num_classes=dn_num_classes
                )
                domain_testset[i] = DomainNetDataset(
                    data_base_path, current_domain_name, 
                    net_dataidx_map_test[i], train=False, transform=transform_test,
                    num_classes=dn_num_classes
                ).data_detailed
            elif cfg.DATASET.NAME == 'PACS':
                domain_trainset[i] = PACSDataset(
                    data_base_path, current_domain_name, 
                    net_dataidx_map_train[i], transform=transform_train
                )
                domain_testset[i] = PACSDataset(
                    data_base_path, current_domain_name, 
                    net_dataidx_map_test[i], train=False, transform=transform_test
                ).data_detailed
            elif cfg.DATASET.NAME == 'Office31':
                domain_trainset[i] = Office31Dataset(
                    data_base_path, current_domain_name, 
                    net_dataidx_map_train[i], transform=transform_train
                )
                domain_testset[i] = Office31Dataset(
                    data_base_path, current_domain_name, 
                    net_dataidx_map_test[i], train=False, transform=transform_test
                ).data_detailed
            elif cfg.DATASET.NAME == 'OfficeHome':
                domain_trainset[i] = OfficeHomeDataset(
                    data_base_path, current_domain_name, 
                    net_dataidx_map_train[i], transform=transform_train
                )
                domain_testset[i] = OfficeHomeDataset(
                    data_base_path, current_domain_name, 
                    net_dataidx_map_test[i], train=False, transform=transform_test
                ).data_detailed
            
            domain_trainset[i] = domain_trainset[i].data_detailed
        
        all_domain_trainset.append(domain_trainset)
        all_domain_testset.append(domain_testset)
        global_test_set.append(global_domain_testset)
    
    
    train_data_num_list = []
    test_data_num_list = []
    train_set = []
    test_set = []
    
    for dataset in all_domain_trainset:
        for i in range(len(dataset)):
            train_data_num_list.append(len(dataset[i]))
            train_set.append(dataset[i])
    
    for dataset in all_domain_testset:
        for i in range(len(dataset)):
            test_data_num_list.append(len(dataset[i]))
            test_set.append(dataset[i])
    
    print("train_data_num_list:", train_data_num_list)
    print("test_data_num_list:", test_data_num_list)
    
    return train_set, test_set, global_test_set, classnames, lab2cname
