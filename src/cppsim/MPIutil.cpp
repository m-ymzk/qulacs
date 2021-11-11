#include <bitset>
#include <cassert>
#include <mpi.h>
#include "MPIutil.hpp"

MPIutil::MPIutil(){}
MPIutil::~MPIutil(){}

int MPIutil::get_rank(MPI_Comm comm){
    this->comm = comm;
    MPI_Comm_rank(comm, &this->rank);
    return this->rank;
}

int MPIutil::get_size(MPI_Comm comm){
    this->comm = comm;
    MPI_Comm_size(this->comm, &this->size);
    return this->size;
}

void MPIutil::run(CommRequest_t op, int peer_rank){
    int my_rank;
    MPI_Comm_rank(comm, &my_rank);
    switch (comm_op) {
        case TypeISendRecv:
            MPI_Isend(sendbuf, sendcount, MPI_DOUBLE,
                      peer_rank, mpi_tag, comm, & (send_request));
            MPI_Irecv(recvbuf, recvcount, MPI_DOUBLE,
                      peer_rank, mpi_tag ^ 0x1, comm, & (recv_request));
            break;
        case TypeISend:
            MPI_Isend(sendbuf, sendcount, MPI_DOUBLE,
                      peer_rank, mpi_tag, comm, & (send_request));
            break;
        case TypeIRecv:
            MPI_Irecv(recvbuf, recvcount, MPI_DOUBLE,
                      peer_rank, mpi_tag ^ 0x1, comm, & (recv_request));
            break;
        default:
            assert(false); // "Undefined communication type is given"
            break;
    }
}
