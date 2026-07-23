import os
import random
import os.path as osp
import tarfile
import zipfile
import logging
from collections import defaultdict
import gdown

from Dassl.dassl.utils import check_isfile


class Datum:
    """Data instance which defines the basic attributes.

    Args:
        impath (str): image path.
        label (int): class label.
        domain (int): domain label.
        classname (str): class name.
    """

    def __init__(self, impath="", label=0, domain=0, classname=""):
        assert isinstance(impath, str)
        assert check_isfile(impath)

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


class DatasetBase:
    """A unified dataset class for
    1) domain adaptation
    2) domain generalization
    3) semi-supervised learning
    """

    dataset_dir = ""  # the directory where the dataset is stored
    domains = []  # string names of all domains

    def __init__(self, train_x=None, federated_train_x=None, train_u=None, val=None, federated_test_x=None, test=None):
        self._train_x = train_x  # labeled training data
        self._federated_train_x = federated_train_x # federated labeled training data (optional)
        self._train_u = train_u  # unlabeled training data (optional)
        self._val = val  # validation data (optional)
        self._federated_test_x = federated_test_x  # federated labeled test data
        self._test = test # test data
        self._num_classes = self.get_num_classes(train_x)
        self._lab2cname, self._classnames = self.get_lab2cname(train_x)

    @property
    def train_x(self):
        return self._train_x

    @property
    def federated_train_x(self):
        return self._federated_train_x

    @property
    def train_u(self):
        return self._train_u

    @property
    def val(self):
        return self._val

    @property
    def federated_test_x(self):
        return self._federated_test_x

    @property
    def test(self):
        return self._test

    @property
    def lab2cname(self):
        return self._lab2cname

    @property
    def classnames(self):
        return self._classnames

    @property
    def num_classes(self):
        return self._num_classes

    def get_num_classes(self, data_source):
        """Count number of classes.

        Args:
            data_source (list): a list of Datum objects.
        """
        label_set = set()
        for item in data_source:
            label_set.add(item.label)
        return max(label_set) + 1

    def get_lab2cname(self, data_source):
        """Get a label-to-classname mapping (dict).

        Args:
            data_source (list): a list of Datum objects.
        """
        container = set()
        for item in data_source:
            container.add((item.label, item.classname))
        mapping = {label: classname for label, classname in container}
        labels = list(mapping.keys())
        labels.sort()
        classnames = [mapping[label] for label in labels]
        return mapping, classnames

    def check_input_domains(self, source_domains, target_domains):
        assert len(source_domains) > 0, "source_domains (list) is empty"
        assert len(target_domains) > 0, "target_domains (list) is empty"
        self.is_input_domain_valid(source_domains)
        self.is_input_domain_valid(target_domains)

    def is_input_domain_valid(self, input_domains):
        for domain in input_domains:
            if domain not in self.domains:
                raise ValueError(
                    "Input domain must belong to {}, "
                    "but got [{}]".format(self.domains, domain)
                )

    def download_data(self, url, dst, from_gdrive=True):
        if not osp.exists(osp.dirname(dst)):
            os.makedirs(osp.dirname(dst))

        if from_gdrive:
            gdown.download(url, dst, quiet=False)
        else:
            raise NotImplementedError

        print("Extracting file ...")

        if dst.endswith(".zip"):
            zip_ref = zipfile.ZipFile(dst, "r")
            zip_ref.extractall(osp.dirname(dst))
            zip_ref.close()

        elif dst.endswith(".tar"):
            tar = tarfile.open(dst, "r:")
            tar.extractall(osp.dirname(dst))
            tar.close()

        elif dst.endswith(".tar.gz"):
            tar = tarfile.open(dst, "r:gz")
            tar.extractall(osp.dirname(dst))
            tar.close()

        else:
            raise NotImplementedError

        print("File extracted to {}".format(osp.dirname(dst)))

    def generate_fewshot_dataset(
        self, *data_sources, num_shots=-1, repeat=False
    ):
        """Generate a few-shot dataset (typically for the training set).

        This function is useful when one wants to evaluate a model
        in a few-shot learning setting where each class only contains
        a small number of images.

        Args:
            data_sources: each individual is a list containing Datum objects.
            num_shots (int): number of instances per class to sample.
            repeat (bool): repeat images if needed (default: False).
        """
        if num_shots < 1:
            if len(data_sources) == 1:
                return data_sources[0]
            return data_sources

        logging.info(f"Creating a {num_shots}-shot dataset")

        output = []
        logging.info(f"data_sources len {len(data_sources)}")

        for data_source in data_sources:
            logging.info(f"data_source len {len(data_source)}")
            tracker = self.split_dataset_by_label(data_source)
            logging.info(f"tracker len {len(tracker)}")
            dataset = []

            for label, items in tracker.items():
                if len(items) >= num_shots:
                    sampled_items = random.sample(items, num_shots)
                else:
                    if repeat:
                        sampled_items = random.choices(items, k=num_shots)
                    else:
                        sampled_items = items
                dataset.extend(sampled_items)

            output.append(dataset)

        if len(output) == 1:
            return output[0]

        return output


    def generate_federated_fewshot_dataset(
        self, *data_sources, num_shots=-1, num_users=5, is_iid=False, repeat_rate = 0.0, repeat=False
    ):
        """Generate a federated few-shot dataset (typically for the federated training set).

        This function is useful when one wants to evaluate a model
        in a few-shot learning setting where each class only contains
        a small number of images.

        Args:
            data_sources: each individual is a list containing Datum objects.
            num_shots (int): number of instances per class to sample.
            num_users (int): number of users
            repeat (bool): repeat images if needed (default: False).
        Return:
            Directory[list]:list of data for each user
        """
        logging.info(f"Creating a {num_shots}-shot federated dataset")
        output_dict = defaultdict(list)
        if num_shots < 1:
            for idx in range(num_users):
                if len(data_sources) == 1:
                    output_dict[idx] = data_sources[0]
                output_dict[idx].append(data_sources)

        else:
            user_class_dict = defaultdict(list)
            class_num = self.get_num_classes(data_sources[0])
            logging.info(f"class_num {class_num}")
            class_per_user = int(round(class_num/num_users))
            class_list = list(range(0,class_num))
            random.seed(2023)
            random.shuffle(class_list)
            if repeat_rate > 0:
                repeat_num = int(repeat_rate*class_num)
                class_repeat_list = class_list[0:repeat_num]
                class_norepeat_list = class_list[repeat_num:class_num]
                class_per_user = int(round((class_num-repeat_num)/num_users))
                fold = int(num_users/num_shots)
                print("repeat_num",repeat_num)
                print("class_repeat_list", class_repeat_list)
                print("class_norepeat_list",class_norepeat_list)
                print("fold", fold)
                if fold > 0:
                    client_idx_fold = defaultdict(list)
                    client_per_fold = int(round(num_users/fold))
                    repeat_per_fold = int(round(repeat_num/fold))
                    client_list = list(range(0,num_users))
                    random.shuffle(client_list)
                    for i in range(fold):
                        client_idx_fold[i] = client_list[i*client_per_fold:min((i + 1) * client_per_fold, num_users)]


            for data_source in data_sources:
                tracker = self.split_dataset_by_label(data_source)

                for idx in range(num_users):
                    if is_iid:
                        user_class_dict[idx] = list(range(0, class_num))
                    else:
                        if repeat_rate == 0.0:
                            # 修复：当 class_per_user = 0 时（类别数 < 客户端数），使用循环分配确保每个客户端至少有一个类别
                            if class_per_user == 0:
                                # 循环分配：每个客户端分配一个类别，类别不足时循环使用
                                assigned_class = class_list[idx % class_num]
                                user_class_dict[idx] = [assigned_class]
                            else:
                                # 原有逻辑：每个客户端分配多个类别
                                if idx == num_users-1:
                                    user_class_dict[idx] = class_list[idx * class_per_user:  class_num]
                                else:
                                    user_class_dict[idx] = class_list[idx * class_per_user: (idx + 1) * class_per_user]
                        else:
                            user_class_dict[idx] = []
                            if fold > 0:
                                for k, v in client_idx_fold.items():
                                    if idx in v:
                                        if k == len(client_idx_fold) - 1:
                                            user_class_dict[idx].extend(class_repeat_list[k * repeat_per_fold: repeat_num])
                                        else:
                                            user_class_dict[idx].extend(class_repeat_list[k * repeat_per_fold:(k + 1) * repeat_per_fold])
                            else:
                                user_class_dict[idx].extend(class_repeat_list)
                            logging.info(f"user_class_dict repeat part {user_class_dict[idx]}")

                            # 修复：当 class_per_user = 0 时（类别数 < 客户端数），使用循环分配
                            if class_per_user == 0:
                                # 循环分配：每个客户端从 nonrepeat_list 中分配一个类别
                                if len(class_norepeat_list) > 0:
                                    assigned_class = class_norepeat_list[idx % len(class_norepeat_list)]
                                    user_class_dict[idx].append(assigned_class)
                                    logging.info(f"user_class_dict nonrepeat part (circular) {[assigned_class]}")
                            else:
                                if idx == num_users - 1:
                                    user_class_dict[idx].extend(class_norepeat_list[idx * class_per_user:  class_num - repeat_num])
                                    logging.info(f"user_class_dict nonrepeat part {class_norepeat_list[idx * class_per_user:  class_num - repeat_num]}")
                                else:
                                    user_class_dict[idx].extend(class_norepeat_list[idx * class_per_user: (idx + 1) * class_per_user])
                                    logging.info(f"user_class_dict nonrepeat part {class_norepeat_list[idx * class_per_user: (idx + 1) * class_per_user]}")

                    logging.info(f"user class dict total {user_class_dict[idx]}")

                    dataset = []

                    for label, items in tracker.items():
                        if label in user_class_dict[idx]:
                            if repeat_rate == 0.0:
                                if len(items) >= num_shots:
                                    sampled_items = random.sample(items, num_shots)
                                else:
                                    if repeat:
                                        sampled_items = random.choices(items, k=num_shots)
                                    else:
                                        sampled_items = items
                                dataset.extend(sampled_items)
                            else:
                                if label in class_repeat_list:
                                    if int(num_shots/num_users) >0:
                                        tmp_num_shots = int(num_shots/num_users)
                                    else:
                                        tmp_num_shots = 1
                                    sampled_items = random.sample(items, tmp_num_shots)
                                else:
                                    sampled_items = random.sample(items, num_shots)
                                dataset.extend(sampled_items)

                    output_dict[idx] = dataset
                    logging.info(f"idx: {idx}, output_dict_len: {len(output_dict[idx])}")

        return output_dict


    def generate_federated_dataset(
        self, *data_sources, num_shots=-1, num_users=5, is_iid=False, repeat_rate = 0.0, repeat=False
    ):
        """Generate a federated dataset (typically for the federated baseline training set).
        Every client owns total number of class per client

        This function is useful when one wants to evaluate a model
        in a few-shot learning setting where each class only contains
        a small number of images.

        Args:
            data_sources: each individual is a list containing Datum objects.
            num_shots (int): number of instances per class to sample.
            num_users (int): number of users
            repeat (bool): repeat images if needed (default: False).
        Return:
            Directory[list]:list of data for each user
        """
        logging.info(f"Creating a baseline federated dataset")
        output_dict = defaultdict(list)
        user_class_dict = defaultdict(list)
        sample_per_user = defaultdict(int)
        sample_order = defaultdict(list)
        class_num = self.get_num_classes(data_sources[0])
        logging.info(f"class_num {class_num}")
        class_per_user = int(round(class_num/num_users))
        class_list = list(range(0, class_num))
        random.seed(2023)
        random.shuffle(class_list)
        if repeat_rate > 0:
            repeat_num = int(repeat_rate * class_num)
            class_repeat_list = class_list[0:repeat_num]
            class_norepeat_list = class_list[repeat_num:class_num]
            class_per_user = int(round((class_num - repeat_num) / num_users))
            if repeat_rate > 0:
                repeat_num = int(repeat_rate*class_num)
                class_repeat_list = class_list[0:repeat_num]
                class_norepeat_list = class_list[repeat_num:class_num]
                class_per_user = int(round((class_num-repeat_num)/num_users))
                fold = int(num_users/num_shots)
                print("repeat_num",repeat_num)
                print("class_repeat_list", class_repeat_list)
                print("class_norepeat_list",class_norepeat_list)
                print("fold", fold)
                if fold > 0:
                    client_idx_fold = defaultdict(list)
                    client_per_fold = int(round(num_users/fold))
                    repeat_per_fold = int(round(repeat_num/fold))
                    client_list = list(range(0,num_users))
                    random.shuffle(client_list)
                    for i in range(fold):
                        client_idx_fold[i] = client_list[i*client_per_fold:min((i + 1) * client_per_fold, num_users)]

        for data_source in data_sources:
            tracker = self.split_dataset_by_label(data_source)
            for label, items in tracker.items():
                sample_order[label] = list(range(0, len(items)))
                sample_per_user[label] = int(round(len(items) / num_users))
                # print("label, sample_per_user",label, sample_per_user[label])
                random.shuffle(sample_order[label])
                if repeat_rate > 0 and fold > 0:
                    sample_per_user[label] = int(round(len(items) /(num_users/fold)))

            # 记录每个 client 的样本数和类别数，由调用方说明是 train/test
            for idx in range(num_users):
                if is_iid:
                    user_class_dict[idx] = list(range(0, class_num))
                else:
                    if repeat_rate == 0.0:
                        # 修复：当 class_per_user = 0 时（类别数 < 客户端数），使用循环分配确保每个客户端至少有一个类别
                        if class_per_user == 0:
                            # 循环分配：每个客户端分配一个类别，类别不足时循环使用
                            assigned_class = class_list[idx % class_num]
                            user_class_dict[idx] = [assigned_class]
                        else:
                            # 原有逻辑：每个客户端分配多个类别
                            if idx == num_users - 1:
                                user_class_dict[idx] = class_list[idx * class_per_user:  class_num]
                            else:
                                user_class_dict[idx] = class_list[idx * class_per_user: (idx + 1) * class_per_user]
                    else:
                        user_class_dict[idx] = []
                        if fold > 0:
                            for k,v in client_idx_fold.items():
                                if idx in v:
                                    if k == len(client_idx_fold)-1:
                                        user_class_dict[idx].extend(class_repeat_list[k*repeat_per_fold: repeat_num])
                                    else:
                                        user_class_dict[idx].extend(class_repeat_list[k * repeat_per_fold:(k + 1) * repeat_per_fold])
                        else:
                            user_class_dict[idx].extend(class_repeat_list)
                        logging.info(f"user_class_dict repeat part {user_class_dict[idx]}")

                        # 修复：当 class_per_user = 0 时（类别数 < 客户端数），使用循环分配
                        if class_per_user == 0:
                            # 循环分配：每个客户端从 nonrepeat_list 中分配一个类别
                            if len(class_norepeat_list) > 0:
                                assigned_class = class_norepeat_list[idx % len(class_norepeat_list)]
                                user_class_dict[idx].append(assigned_class)
                                logging.info(f"user_class_dict nonrepeat part (circular) {[assigned_class]}")
                        else:
                            if idx == num_users - 1:
                                user_class_dict[idx].extend(class_norepeat_list[idx * class_per_user:  class_num-repeat_num])
                                logging.info(f"user_class_dict nonrepeat part {class_norepeat_list[idx * class_per_user:  class_num-repeat_num]}")
                            else:
                                user_class_dict[idx].extend(class_norepeat_list[idx * class_per_user: (idx + 1) * class_per_user])
                                logging.info(f"user_class_dict nonrepeat part {class_norepeat_list[idx * class_per_user: (idx + 1) * class_per_user]}")

                # client 的类别集合（保留原有打印但默认关闭）
                # print("user class dict total",user_class_dict[idx])
                # 统计该 client 的样本数量和类别数量
                num_samples = 0
                for label, items in tracker.items():
                    if label in user_class_dict[idx]:
                        num_samples += len(items)
                num_classes = len(user_class_dict[idx])
                # 由上游在调用前打印 "Creating a baseline federated dataset for train/test set"
                # logging.info(f"[Federated for set] C{idx}: #samples={num_samples}, #classes={num_classes}")  # 已禁用：减少日志冗余

                # 为每个客户端分配数据（修复：将数据分配逻辑移到for idx循环内部）
                dataset = []

                for label, items in tracker.items():
                    if label in user_class_dict[idx]:
                        if is_iid:
                            sampled_items=[]
                            for k,v in enumerate(items):
                                if k in sample_order[label][idx * sample_per_user[label]: min((idx + 1) * sample_per_user[label], len(items))]:
                                    # print(idx,label,sample_order[label][idx * sample_per_user[label]: min((idx + 1) * sample_per_user[label], len(items))])
                                    sampled_items.append(v)
                            dataset.extend(sampled_items)
                        else:
                            if repeat_rate == 0.0:
                                sampled_items = items
                                dataset.extend(sampled_items)
                            else:
                                if label in user_class_dict[idx][0:repeat_num]:
                                    sampled_items = []
                                    for k, v in enumerate(items):
                                        if k in sample_order[label][idx * sample_per_user[label]: min((idx + 1) * sample_per_user[label],len(items))]:
                                            # print(idx,label,sample_order[label][idx * sample_per_user[label]: min((idx + 1) * sample_per_user[label], len(items))])
                                            sampled_items.append(v)
                                    dataset.extend(sampled_items)
                                else:
                                    sampled_items = items
                                    dataset.extend(sampled_items)

                output_dict[idx] = dataset
                logging.info(f"idx: {idx}, output_dict_len: {len(output_dict[idx])}")
                
                # 验证：确保每个客户端都有数据
                # 如果数据为空，说明数据分配逻辑有问题，需要修复而不是改变分配策略
                if len(output_dict[idx]) == 0:
                    # 详细调试信息
                    assigned_classes = user_class_dict[idx]
                    tracker_classes = list(tracker.keys())
                    missing_in_tracker = [c for c in assigned_classes if c not in tracker]
                    empty_in_tracker = [c for c in assigned_classes if c in tracker and len(tracker[c]) == 0]
                    
                    logging.error(f"ERROR: Client {idx} has empty dataset!")
                    logging.error(f"  Assigned classes: {assigned_classes}")
                    logging.error(f"  Tracker classes: {tracker_classes}")
                    logging.error(f"  Missing in tracker: {missing_in_tracker}")
                    logging.error(f"  Empty in tracker: {empty_in_tracker}")
                    
                    # 尝试修复：从已分配的类别中重新分配数据（不改变分配策略）
                    if len(tracker) > 0:
                        # 优先从已分配的类别中找数据（如果那些类别在tracker中存在且有数据）
                        fallback_label = None
                        for assigned_label in assigned_classes:
                            if assigned_label in tracker and len(tracker[assigned_label]) > 0:
                                fallback_label = assigned_label
                                break
                        
                        if fallback_label is not None:
                            # 从已分配的类别中分配数据（保持分配策略不变）
                            # 使用该类别的所有数据，因为这是原本应该分配给该客户端的
                            fallback_items = tracker[fallback_label]
                            output_dict[idx] = fallback_items
                            logging.warning(f"WARNING: Client {idx} - Fixed empty dataset by using assigned class {fallback_label} (strategy unchanged)")
                            logging.info(f"idx: {idx}, output_dict_len after fix: {len(output_dict[idx])}")
                        else:
                            # 如果已分配的类别都不在tracker中，这是数据分配逻辑的严重问题
                            # 这种情况下，从tracker中找一个有数据的类别（会改变分配策略，但至少保证有数据）
                            available_labels = [label for label in tracker.keys() if len(tracker[label]) > 0]
                            if available_labels:
                                fallback_label = available_labels[0]
                                fallback_items = tracker[fallback_label]
                                # 只分配少量数据，并明确警告这会改变分配策略
                                num_fallback = max(10, min(len(fallback_items) // 5, len(fallback_items)))
                                output_dict[idx] = fallback_items[:num_fallback]
                                logging.error(f"CRITICAL: Client {idx} assigned classes {assigned_classes} not in tracker! Using class {fallback_label} instead (THIS CHANGES ASSIGNMENT STRATEGY - please fix the root cause!)")
                                logging.info(f"idx: {idx}, output_dict_len after fallback: {len(output_dict[idx])}")
                
                # 验证：确保每个客户端都有数据
                if len(output_dict[idx]) == 0:
                    logging.error(f"ERROR: Client {idx} has empty dataset! Assigned classes: {user_class_dict[idx]}, Available classes in tracker: {list(tracker.keys())}")
                    # 如果数据为空，尝试从其他类别分配数据以确保每个客户端都有数据
                    if len(tracker) > 0:
                        # 找到有数据的类别
                        available_labels = [label for label in tracker.keys() if len(tracker[label]) > 0]
                        if available_labels:
                            # 从第一个可用类别分配一些数据
                            fallback_label = available_labels[0]
                            fallback_items = tracker[fallback_label]
                            # 分配少量数据给空客户端
                            num_fallback = min(10, len(fallback_items))
                            output_dict[idx] = fallback_items[:num_fallback]
                            logging.warning(f"WARNING: Assigned {num_fallback} samples from class {fallback_label} to empty client {idx}")
                            logging.info(f"idx: {idx}, output_dict_len after fallback: {len(output_dict[idx])}")

        return output_dict


    def split_dataset_by_label(self, data_source):
        """Split a dataset, i.e. a list of Datum objects,
        into class-specific groups stored in a dictionary.

        Args:
            data_source (list): a list of Datum objects.
        """
        output = defaultdict(list)

        for item in data_source:
            output[item.label].append(item)

        return output

    def split_dataset_by_domain(self, data_source):
        """Split a dataset, i.e. a list of Datum objects,
        into domain-specific groups stored in a dictionary.

        Args:
            data_source (list): a list of Datum objects.
        """
        output = defaultdict(list)

        for item in data_source:
            output[item.domain].append(item)

        return output
