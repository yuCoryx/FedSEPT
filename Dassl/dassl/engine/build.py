import logging
from Dassl.dassl.utils import Registry, check_availability
from trainers.PROMPTFL import PROMPTFL
from trainers.FEDPGP import FEDPGP
from trainers.FEDOTP import FEDOTP
from trainers.FEDPHA import FEDPHA
from trainers.PFEDMOAP import PFEDMOAP
from trainers.DPFPL import DPFPL
from trainers.FEDSEPT import FEDSEPT

TRAINER_REGISTRY = Registry("TRAINER")
TRAINER_REGISTRY.register(PROMPTFL)
TRAINER_REGISTRY.register(FEDPGP)
TRAINER_REGISTRY.register(FEDOTP)
TRAINER_REGISTRY.register(FEDPHA)
TRAINER_REGISTRY.register(PFEDMOAP)
TRAINER_REGISTRY.register(DPFPL)
TRAINER_REGISTRY.register(FEDSEPT)

def build_trainer(args,cfg):
    avai_trainers = TRAINER_REGISTRY.registered_names()
    trainer_name = cfg.TRAINER.NAME
    if trainer_name not in avai_trainers:
        # Allow case-insensitive trainer selection from CLI/configs
        lower_name_map = {name.lower(): name for name in avai_trainers}
        matched = lower_name_map.get(trainer_name.lower())
        if matched:
            trainer_name = matched
    check_availability(trainer_name, avai_trainers)
    if cfg.VERBOSE:
        logging.info("Loading trainer: {}".format(trainer_name))
    return TRAINER_REGISTRY.get(trainer_name)(args,cfg)
