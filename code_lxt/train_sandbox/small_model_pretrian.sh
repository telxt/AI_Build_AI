torchrun --nproc_per_node=8 /root/xtlv/lxt/OLMo-main/scripts/train.py configs/official/OLMo-1B.yaml
torchrun --nproc_per_node=8 /root/xtlv/lxt/OLMo-main/olmo/Ltrain1.py configs/official/OLMo-1B.yaml
torchrun --nproc_per_node=8 /root/xtlv/lxt/OLMo-main/olmo/Ltrain2.py configs/official/OLMo-1B.yaml