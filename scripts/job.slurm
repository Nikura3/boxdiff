#!/bin/bash
#SBATCH --job-name="GLIGEN_boxdiff"
#SBATCH --output=output.out
#SBATCH --error=err.log
#SBATCH --time=04:00:00  # Maximum time limit (hh:mm:ss)
#SBATCH --partition=testing  # Partition name, e.g., general, gpu, etc.
#SBATCH --ntasks=1  # Number of tasks
#SBATCH --cpus-per-task=4  # Number of CPU cores per task
#SBATCH --mem=24G  # Memory per node
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --constraint=A6000

# Load required modules (if any)

# Activate your virtual environment
source activate /home/dvezzaro/.conda/envs/hf_boxdiff

# Run your Python script
python /home/dvezzaro/hf_boxdiff/run_gligen_boxdiff.py

# Deactivate the virtual environment after the job is done
conda deactivate
