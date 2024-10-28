# SFT of ``AI builds AI``
## Implementation
git clone https://github.com/InternLM/xtuner
cd xtuner
pip install -U 'xtuner[deepspeed]'

## Dataset Conversion
Convert the dataset to the format that XTuner can read.
"""Data format:

[{
    "messages": [
        { "role": "system", "content": "xxx." },
        { "role": "user", "content": "xxx." },
        { "role": "assistant", "content": "xxx.", "loss": false},
        { "role": "user", "content": "xxx." },
        { "role": "assistant", "content": "xxx.", "loss": true}
    ]
},
...
]
"""

## SFT
xtuner train ./internlm2_1_8b_full_custom_pretrain_e1_copy.py # single GPU
MKL_SERVICE_FORCE_INTEL=1 MKL_THREADING_LAYER=GNU NPROC_PER_NODE=2 xtuner train ./internlm2_chat_1_8b_qlora_alpaca_e3_copy.py --deepspeed deepspeed_zero3 # multi-GPU
**Note**:

## Conversion
由于时间限制，还没尝试conversion
参考 https://github.com/InternLM/Tutorial/blob/camp3/docs/L1/XTuner/xtuner_finetune_advance.md 即可


# Evaluation of ``AI builds AI``

## Benchmark Selection
- SST-2 
- CoLA 
- MRPC
- STS-B
- ... other glue tasks
- SQuAD
- ARC-Easy 
- ARC-Challenge
- MMLU

## Usage
from eval import pipe
pipe(args)

args = {
    'tasks': 'sst-2' 'cola' 'mrpc' 'sts-b' 'squad' 'arc-easy' 'arc-challenge' 'mmlu'
    'models': '' # list of models
    'batch_size': 32
    'device': 'cuda'
}

## Time Costs (Without multi-GPU and vLLM) 
- 10 min for GLUE tasks on 1 3090 GPU with 32 batch size
- 5 min for Arc-Easy and Arc-Challenge and MMLU on 1 3090 GPU with 32 batch size
- 30 min for SQuAD on 1 3090 GPU with 32 batch size (Too large)
