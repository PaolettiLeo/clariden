#!/bin/bash -l
#SBATCH --job-name=model_download
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=normal
#SBATCH --cpus-per-task=16
#SBATCH --account=a-a145
#SBATCH --output=slurm-%x.out

srun --environment=demo python model_downloader.py --output-dir /capstor/scratch/cscs/leoplt/models --use-auth --token "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"