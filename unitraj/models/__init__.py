from unitraj.models.autobot.autobot import AutoBotEgo
from unitraj.models.mtr.MTR import MotionTransformer
from unitraj.models.wayformer.wayformer import Wayformer
from unitraj.models.qcnet.qcnet import QCNet

__all__ = {
    'autobot': AutoBotEgo,
    'wayformer': Wayformer,
    'MTR': MotionTransformer,
    "QCNet": QCNet,
}


def build_model(config):
    model = __all__[config.method.model_name](
        config=config
    )

    return model
