from collections import OrderedDict
import sys
sys.path.append('/root/xtlv/lxt/ai_build_ai/train_sandbox')
from MLPs.classic_mlp import ClassicMLP
from MLPs.llama_mlp import LlamaMLP


MLP_MAPPING = OrderedDict(
    [
        ('ClassicMLP', ClassicMLP),
        ('LlamaMLP', LlamaMLP),
    ]
)

class AutoMLP:
    @classmethod
    def get(self, mlp_name, hidden_size, intermediate_size):
        if mlp_name in MLP_MAPPING:
            return MLP_MAPPING[mlp_name](hidden_size, intermediate_size)
        raise ValueError(
            "Unrecognized MLP {} for AutoMLP Model: {}.\n"
            "MLP name should be one of {}.".format(
                ", ".join(c for c in MLP_MAPPING.keys())
            )
        )