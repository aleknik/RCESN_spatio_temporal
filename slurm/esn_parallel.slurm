#!/bin/bash
#----------------------------------------------------
# Sample Slurm job script
#   for TACC Stampede2 KNL nodes
#
#   *** MPI Job on Normal Queue ***
#
# Last revised: 20 Oct 2017
#
# Notes:
#
#   -- Launch this script by executing
#      "sbatch knl.mpi.slurm" on Stampede2 login node.
#
#   -- Use ibrun to launch MPI codes on TACC systems.
#      Do not use mpirun or mpiexec.
#
#   -- Max recommended MPI tasks per KNL node: 64-68
#      (start small, increase gradually).
#
#   -- If you're running out of memory, try running
#      fewer tasks per node to give each task more memory.
#
#----------------------------------------------------

#SBATCH -J esn_parallel              # Job name
#SBATCH -p normal                    # Queue (partition) name
#SBATCH -o ./log/esn-%j.out          # Name of stdout output file
#SBATCH -e ./log/esn-%j.err          # Name of stderr error file
#SBATCH -N 5                         # Total # of nodes
#SBATCH -n 11
#SBATCH -t 48:00:00                  # Run time (hh:mm:ss)
#SBATCH --mail-user=nikolic95@gmail.com
#SBATCH --mail-type=end    # Send email at begin and end of job

# Other commands must follow all #SBATCH directives...

module load python2

# Launch MPI code...

ibrun python ./parallel_run.py g 11 l 7 r 1000 s 0.05 rad 0.95 b 0.003        # Use ibrun instead of mpirun or mpiexec

# ---------------------------------------------------
