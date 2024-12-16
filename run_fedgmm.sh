#!/bin/bash
source ~/anaconda3/bin/activate; conda activate fl_data_dist

# for i in 0 1 2 3; do
#     python run_experiment_domainbed.py domainbed FedGMM \
#         --data_dir /data/enyij2/domainbed/data \
#         --dataset VLCS \
#         --uda_holdout_fraction 0 \
#         --n_learners 3 \
#         --n_gmm 2 \
#         --test_envs $i \
#         --n_rounds 50 \
#         --log_freq 5 \
#         --optimizer sgd \
#         --device cuda \
#         --logs_dir logs/domainbed/FedGMM/VLCS/$i \
#         --seed 1234 \
#         --trial_seed 1234 \
#         --verbose 1 > log_fedgmm_VLCS_$i.txt
# done

for i in 0 1 2 3; do
    python run_experiment_domainbed.py domainbed FedGMM \
        --data_dir /data/enyij2/domainbed/data \
        --dataset PACS \
        --uda_holdout_fraction 0 \
        --n_learners 3 \
        --n_gmm 2 \
        --test_envs $i \
        --n_rounds 50 \
        --log_freq 5 \
        --optimizer sgd \
        --device cuda \
        --logs_dir logs/domainbed/FedGMM/PACS/$i \
        --seed 1234 \
        --trial_seed 1234 \
        --verbose 1 > log_fedgmm_PACS_$i.txt
done

# for i in 1 ; do
#     python run_experiment_domainbed.py domainbed FedEM --n_learners 3 --n_rounds 150 --bz 128 --lr 0.01 \
#  --lr_scheduler multi_step --log_freq 5 --device cuda --optimizer sgd --seed 1234 --verbose 1 \
#  --data_dir /data/enyij2/domainbed/data \
#         --dataset VLCS \
#         --uda_holdout_fraction 0 \
#         --test_envs $i > log_fedEM_VLCS_$i.txt
# done