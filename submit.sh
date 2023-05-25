#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --partition=gpu-2xA100 # request node from a specific partition
#SBATCH --nodes=1 # number of nodes
#SBATCH --output=slurm_job_output.log
#SBATCH --error=slurm_job_errors.log # Logs if job crashes

. ~/.program_env_bash

# Location of scratch directory on the compute nodes
scratchlocation=/scratch/users

# Create a user directory if it does not exist
if [ ! -d $scratchlocation/$USER ]; then
    mkdir -p $scratchlocation/$USER
fi

# Create a temporary directory with a unique identifier associated with your jobid
tdir=$(mktemp -d $scratchlocation/$USER/$SLURM_JOB_ID-XXXX)

# Go to the temporary directory
cd $tdir

# Exit if tdir does not exist
if [ ! -d $tdir ]; then
    echo "Temporary scratch directory does not exist ..."
    echo "Something is wrong, contact support."
    exit
fi

# Copy the necessary input files to run your job
cp -R $SLURM_SUBMIT_DIR/* $tdir/

# If the program needs many input files you can add a separate line for each file.

# If your job requires a directory of input files
# cp -r $SLURM_SUBMIT_DIR/myinputdir $tdir/

# Now the run the job from the temporary directory e.g.
python model_sar.py

# After the job is completed make sure to copy the output to your submit directory.
cp -R $tdir/* $SLURM_SUBMIT_DIR/

# If the program produces many output files you can add a separate line for each file.
# Please try to only copy the files that you need.

# IMPORTANT. Delete the temporary directory and all of its content
rm -rf $tdir
