#!/bin/bash
#------- qsub option -----------
#PBS -q SQUID-H
#PBS --group=jh220009
#PBS -N iVON
#PBS -j o
#PBS -b 1
#PBS -l gpunum_job=1
#PBS -l elapstim_req=100:00:00

module load BaseGPU/2021
source /sqfs/home/z6b038/vir/py3/bin/activate

export PYTHONPATH=$PYTHONPATH:/sqfs/home/z6b038/.local/lib/python3.8/site-packages
export PYTHONPATH=$PYTHONPATH:/sqfs/home/z6b038/common
export PYTHONPATH=$PYTHONPATH:/sqfs/home/z6b038/asdfghjkl
export PYTHONPATH=$PYTHONPATH:/sqfs/home/z6b038/sam
export PYTHONPATH=$PYTHONPATH:/sqfs/home/z6b038/vit-pytorch

export http_proxy="http://ibgw1f-ib0:3128"
export https_proxy="https://ibgw1f-ib0:3128"

export WANDB_DIR=/sqfs/work/jh220009/z6b038/wandb

cd $PBS_O_WORKDIR

# wandb agent --count 50 2nd-order-opt-survey/ivon/6g2v5741 #bayes
wandb agent --count 4 2nd-order-opt-survey/ivon/63tjr46w # lr
