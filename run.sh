#!/bin/sh

source env.sh
conda activate pyg


# python scripts/heptrx_nnconv.py -c -m=EdgeNetWithCategories -l=nll_loss -d=char_EC --forcecats --cats=2 --hidden_dim=64 --lr 1e-4 -o AdamW |& tee char_EC.log
# python scripts/heptrx_nnconv.py -c -m=InteractionNetwork -l=binary_cross_entropy -d=char_IN --forcecats --cats=2 --hidden_dim=64 --lr 1e-4 -o AdamW |& tee char_IN.log


python scripts/heptrx_nnconv.py -c -m=InteractionNetwork -l=binary_cross_entropy -d=test_run --forcecats --cats=2 --hidden_dim=64 --lr 1e-4 -o AdamW
