#!/bin/bash
#SBATCH --job-name=sbd.qdPyjobs
#SBATCH --output=out.qdPyjobs.log                                                         
#SBATCH --error=err.qdPyjobs.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=40
#SBATCH --mem=190G
#SBATCH --time=02:00:00
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=sbdas@princeton.edu

module load anaconda3
conda activate helio

./qdpyjobs.sh
