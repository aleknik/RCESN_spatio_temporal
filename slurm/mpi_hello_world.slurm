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
#SBATCH -N 2                         # Total # of nodes
#SBATCH --ntasks-per-node 1
#SBATCH -t 24:00:00                  # Run time (hh:mm:ss)
#SBATCH --mail-user=nikolic95@gmail.com
#SBATCH --mail-type=end    # Send email at begin and end of job

# Other commands must follow all #SBATCH directives...

module load python2

# Launch MPI code...

ibrun python ./mpi_hello_world.py       # Use ibrun instead of mpirun or mpiexec

# ---------------------------------------------------
