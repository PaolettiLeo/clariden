#!/bin/bash
#SBATCH --job-name=model_download   
#SBATCH --time=00:40:00                  
#SBATCH --nodes=1       
#SBATCH --ntasks-per-core=1                      
#SBATCH --ntasks-per-node=4                      
#SBATCH --cpus-per-task=16                
#SBATCH --account=a-a07             

srun python python model_downloader.py --output-dir /capstor/scratch/cscs/leoplt/models --use-auth --token "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"          
