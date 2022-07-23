from fairseq.optim import register_optimizer, FairseqOptimizer
from fairseq.optim.adam import FairseqAdam, FairseqAdamConfig
import torch
from dataclasses import dataclass, field
from typing import Any, List
from omegaconf import II, OmegaConf
from fairseq.dataclass import FairseqDataclass
from collections.abc import Collection

import pdb
@register_optimizer("bnb", dataclass=FairseqAdamConfig)
class FairseqBnb(FairseqAdam):
    """Adam optimizer for fairseq.
    Important note: this optimizer corresponds to the "AdamW" variant of
    Adam in its weight decay behavior. As such, it is most closely
    analogous to torch.optim.AdamW from PyTorch.
    """

    def __init__(self, cfg: FairseqAdamConfig, params):
        super().__init__(cfg, params)
        if torch.cuda.is_available():
            import bitsandbytes as bnb
            self._optimizer = bnb.optim.Adam8bit(params, **self.optimizer_config)
        else:
            raise Exception('bnb only works when cuda is available')




@dataclass
class FairseqQAdamConfig(FairseqDataclass):
    adam_betas: Any = field(
        default=(0.9, 0.999), metadata={"help": "betas for Adam optimizer"}
    )
    adam_eps: float = field(
        default=1e-8, metadata={"help": "epsilon for Adam optimizer"}
    )
    weight_decay: float = field(default=0.0, metadata={"help": "weight decay"})
    use_old_adam: bool = field(
        default=False, metadata={"help": "Use fairseq.optim.adam.Adam"}
    )
    fp16_adam_stats: bool = field(
        default=False, metadata={"help": "use FP16 stats (with automatic scaling)"}
    )
    exp: int = field(
        default=5, metadata={"help": "significant"}
    )
    man: int = field(
        default=2, metadata={"help": "man"}
    )
    representation: str = field(
        default='fp', metadata = {"help": "int or float"}
    )
    rounding: str = field(
        default='nearest', metadata= {"help": "rounding stochasting rounding or nearest"}
    )
    scaling: str = field(
        default='simple', metadata={"help": "scaling technique"} 
    )
    onlyqfm: bool = field(
        default= False, metadata={"help": "only quantize first moument if set True"}
    )
    onlyqsm: bool = field(
        default= False, metadata={"help": "only quantize first moument if set True"}
    )
    
    # TODO common vars below in parent
    tpu: bool = II("common.tpu")
    lr: List[float] = II("optimization.lr")

@register_optimizer("QAdam", dataclass=FairseqQAdamConfig)
class Fairseqsq(FairseqOptimizer):
    """Adam optimizer for fairseq.
    Important note: this optimizer corresponds to the "AdamW" variant of
    Adam in its weight decay behavior. As such, it is most closely
    analogous to torch.optim.AdamW from PyTorch.
    """

    def __init__(self, cfg: FairseqQAdamConfig, params):
        super().__init__(cfg)
        if torch.cuda.is_available():
            # pdb.set_trace()
            from opt_quant.schemes.optim.QAdam import QAdam 

            
            self._optimizer = QAdam(params, **self.optimizer_config)
        else:
            raise Exception('cpu version is not available')
        
    
    @property
    def optimizer_config(self):
        """
        Return a kwarg dictionary that will be used to override optimizer
        args stored in checkpoints. This allows us to load a checkpoint and
        resume training using a different set of optimizer args, e.g., with a
        different learning rate.
        """
        return {
            "lr": self.cfg.lr[0]
            if isinstance(self.cfg.lr, Collection)
            else self.cfg.lr,
            "betas": eval(self.cfg.adam_betas)
            if isinstance(self.cfg.adam_betas, str)
            else OmegaConf.to_container(self.cfg.adam_betas),
            "eps": self.cfg.adam_eps,
            "weight_decay": self.cfg.weight_decay,
            "exp": self.cfg.exp,
            "man": self.cfg.man,
            "rounding": self.cfg.rounding,
            "representation": self.cfg.representation,
            "scaling": self.cfg.scaling,
            "onlyqfm": self.cfg.onlyqfm,
            "onlyqsm": self.cfg.onlyqsm
            
        }

                 
