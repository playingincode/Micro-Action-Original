#!/bin/bash
#OAR -p esterel40
#OAR -l host=1/gpu=1,walltime=72:00:00
#OAR --name timnesformer_for_Social_gestures_final_corrected_num_classes_testing
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

python -u tools/test.py configs/recognition/timesformer/timesformer_jointST_8x32x1_15e_kinetics400_rgb.py /srv/storage/stars@storage3.sophia.grid5000.fr/npoddar/MANET_original_six_classes/Micro-Action/mar_scripts/manet/mmaction2/work_dirs/timnesformer_for_Social_gestures_final_corrected_num_classes/best_top1_acc_epoch_17.pth --out online_evaluation/timesformer_Social_gesture.pickle