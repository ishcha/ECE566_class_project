# Instruction to run the code

## Running baselines
To run FedAvg and FedADG, following the instructions below.

The code to run these baselines for VLCS is in ```baselines/FedADG/src```. The command to run the FedAvg is:

```python trainFedLear.py --lr0 0.001 --label_smoothing 0.2 --lr-threshold 0.0001 --factor 0.2 --epochs 10 --patience 20 --ite-warmup 500 --global_epochs 20```

The command to run FedADG is:

```python train.py --lr0 0.001 --lr1 0.0007 --label_smoothing 0.2 --lr-threshold 0.0001 --factor 0.2 --epochs 10 --rp-size 1024 --patience 20 --ite-warmup 500 --global_epochs 20```

The code for PACS is in ```baselines/FedDG_Benchmark```.

The code to run these baselines for VLCS is in ```baselines/FedADG/src```. The command to run the FedAvg is:

```./run_fedavg.sh```

The command to run FedADG is:

```./run_fedadg.sh```
