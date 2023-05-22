#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=<Your E-mail> # for example uname@hi.is
#SBATCH --partition=48cpu_192mem  # request node from a specific partition
#SBATCH --nodes=2                 # number of nodes
#SBATCH --ntasks-per-node=48      # 48 cores per node (96 in total)
#SBATCH --mem-per-cpu=3900        # MB RAM per cpu core
#SBATCH --time=0-04:00:00         # run for 4 hours maximum (DD-HH:MM:SS)
#SBATCH --hint=nomultithread      # Suppress multithread
#SBATCH --output=slurm_job_output.log
#SBATCH --error=slurm_job_errors.log   # Logs if job crashes

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
cp $SLURM_SUBMIT_DIR/* $tdir/

# If the program needs many input files you can add a separate line for each file.

# If your job requires a directory of input files
# cp -r $SLURM_SUBMIT_DIR/myinputdir $tdir/

# Now the run the job from the temporary directory e.g.
python main.py

# After the job is completed make sure to copy the output to your submit directory.
cp $tdir/* $SLURM_SUBMIT_DIR/

# If the program produces many output files you can add a separate line for each file.
# Please try to only copy the files that you need.

# IMPORTANT. Delete the temporary directory and all of its content
rm -rf $tdir
