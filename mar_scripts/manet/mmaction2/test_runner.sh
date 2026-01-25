#!/bin/bash
#OAR -p esterel37
#OAR -l host=1/gpu=1,walltime=72:00:00
#OAR --name unique_experts_with_videomae_average_poll_cross_entropy_test
#OAR --stdout nef_logs/%jobname%.%jobid%.out
#OAR --stderr nef_logs/%jobname%.%jobid%.err


source ~/.bashrc
module load conda/2020.48-python3.8 cuda/12.2 gcc/9.2.0

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
source activate openmmlab|| { echo "Conda environment not found"; exit 1; }

# Display python version and path
python --version
which python

# Add to PATH
export PATH="/home/npoddar/:$PATH"

# Run ffmpeg and nvidia-smi to check availability
ffmpeg
nvidia-smi || { echo "NVIDIA driver issue"; exit 1; }

export PATH=/pytorch_env/bin:$PATH

export CUBLAS_WORKSPACE_CONFIG=:4096:8

python -u tools/test.py configs/recognition/manet/manet.py  /srv/storage/stars@storage3.sophia.grid5000.fr/npoddar/MANET_original_six_classes/Micro-Action/mar_scripts/manet/mmaction2/work_dirs/unique_experts_with_average_pooling_cross_entropy_final/best_top1_acc_epoch_93.pth --out online_evaluation/unique_experts_with_videomae_average_pool_cross_entropy_ma52_final.pickle