#!/bin/bash
#SBATCH -N 1
#SBATCH -p GPU-shared
#SBATCH -t 24:00:00
#SBATCH --gpus=v100-32:1
#SBATCH --output=machine_translation_%j.log      # Standard output file (%j will be replaced with job ID)
#SBATCH --error=machine_translation_%j.log        # Standard error file (%j will be replaced with job ID)

source .venv/bin/activate
module load cuda
python project/run_machine_translation.py