import os

from utils.domain_skew import prepare_data_domain_partition_train

class DomainNet():
    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        train_set, test_set, global_test_set, classnames, lab2cname = prepare_data_domain_partition_train(cfg, root)
        
        self.num_classes = len(classnames) if classnames is not None else int(getattr(cfg.DATASET, "DOMAINNET_NUM_CLASSES", 365))
        self.data_test = global_test_set
        self.federated_train_x = train_set
        self.federated_test_x = test_set
        self.lab2cname = lab2cname
        self.classnames = classnames



