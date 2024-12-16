#!/bin/bash
source ~/anaconda3/bin/activate; conda activate fl_data_dist

for i in 0 1 2 3; do
    python run_experiment_domainbed.py domainbed FedEM --n_learners 3 --n_rounds 50 --bz 128 --lr 0.01 \
 --lr_scheduler multi_step --log_freq 5 --device cuda --optimizer sgd --seed 1234 --verbose 1 \
 --data_dir /data/enyij2/domainbed/data \
 --logs_dir logs/domainbed/FedEM/VLCS/$i \
        --dataset VLCS \
        --uda_holdout_fraction 0 \
        --test_envs $i > log_fedEM_VLCS_$i.txt
done

for i in 0 1 2 3; do
    python run_experiment_domainbed.py domainbed FedEM --n_learners 3 --n_rounds 50 --bz 128 --lr 0.01 \
 --lr_scheduler multi_step --log_freq 5 --device cuda --optimizer sgd --seed 1234 --verbose 1 \
 --data_dir /data/enyij2/domainbed/data \
 --logs_dir logs/domainbed/FedEM/PACS/$i \
        --dataset PACS \
        --uda_holdout_fraction 0 \
        --test_envs $i > log_fedEM_PACS_$i.txt
done


# for i in 0; do
#     python run_experiment_flamby.py flamby FedAvg --n_learners 1 --n_rounds 150 --bz 128 --lr 0.001 \
#  --lr_scheduler multi_step --log_freq 5 --device cuda --optimizer sgd --seed 1234 --verbose 1 \
#         --dataset heart \
#         --uda_holdout_fraction 0 \
#         --test_envs $i 
#         #> log_fedEM_heart_$i.txt
# done