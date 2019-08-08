import datetime
import sys

from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()


def print_with_rank(string):
    print(string + ' Rank: ' + str(rank) + " Time: " + str(
        datetime.datetime.now()) + " Processor: " + MPI.Get_processor_name())
    sys.stdout.flush()
