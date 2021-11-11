#pragma once

#include <cstddef>
//#include <new>

#include <mpi.h>

typedef enum { Undefined        =  0,
               TypeISend        =  1,
               TypeIRecv        =  2,
               TypeISendRecv    =  3,
} CommRequest_t;

class MPIutil {
public:
    bool is_completed();
    void check_mpi_complete();
    void run(CommRequest_t, int);

    MPIutil();
    ~MPIutil();
    int get_rank(MPI_Comm);
    int get_size(MPI_Comm);
    int get_uniq_seed();

protected:
    MPI_Comm comm;
    CommRequest_t comm_op;
    MPI_Request send_request;
    MPI_Request recv_request;
    int  rank;
    int  size;
    int  recvcount;
    int  sendcount;
    int  mpi_tag;
    //int  peer_rank;
    void *sendbuf;
    void *recvbuf;
    int  state;
};

