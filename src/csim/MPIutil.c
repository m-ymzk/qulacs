#include <assert.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include "MPIutil.h"

static MPI_Comm mpicomm;
static int mpirank = 0;
static int mpisize = 0;
static int mpitag = 0;
static MPIutil mpiutil;
static MPI_Status mpistat;
static pthread_mutex_t mutex= PTHREAD_MUTEX_INITIALIZER;

static void set_comm(MPI_Comm c) {
    if (mpicomm == 0) {
        mpicomm = c;
        MPI_Comm_rank(mpicomm, &mpirank);
        MPI_Comm_size(mpicomm, &mpisize);
    } else if (mpicomm != c) { 
        fprintf(stderr, "## warn: MPI_Comm is conflicted!! %x %x\n", (uint)mpicomm, (uint)c);
    }
}

static int get_rank() {
    return mpirank;
}

static int get_size() {
    return mpisize;
}

static int get_tag() {
    pthread_mutex_lock(&mutex);
    mpitag += 1<<20; // max 1M-ranks
    pthread_mutex_unlock(&mutex);
    return mpitag;
}

static void mpisendrecv(void *sendbuf, void *recvbuf, int count, int peer_rank) {
    int tag0 = get_tag();
    //if (peer_rank
    int mpi_tag1 = tag0 + ((mpirank & peer_rank)<<1) + (mpirank > peer_rank);
    int mpi_tag2 = mpi_tag1 ^ 1;
    //int mpi_tag1 = tag0 + (mpirank & 0xFFFE);
    //int mpi_tag2 = mpi_tag1 ^ 1;
    printf("#%d: mpisendrecv: %d, %d, %d, %d, %d\n", mpirank, count, mpirank, peer_rank, mpi_tag1, mpi_tag2);
    MPI_Sendrecv(sendbuf, count, MPI_DOUBLE, peer_rank, mpi_tag1,
                 recvbuf, count, MPI_DOUBLE, peer_rank, mpi_tag2,
                 mpicomm, &mpistat);
    /*
    MPI_Isend(sendbuf, sendcount, MPI_DOUBLE,
              peer_rank, mpi_tag, comm, & (send_request));
    MPI_Irecv(recvbuf, recvcount, MPI_DOUBLE,
              peer_rank, mpi_tag ^ 0x1, comm, & (recv_request));
    */
        //case TypeISend:
        //    MPI_Isend(sendbuf, sendcount, MPI_DOUBLE,
        //              peer_rank, mpi_tag, comm, & (send_request));
        //    break;
        //case TypeIRecv:
        //    MPI_Irecv(recvbuf, recvcount, MPI_DOUBLE,
        //              peer_rank, mpi_tag ^ 0x1, comm, & (recv_request));
    //        break;
    //    default:
    //        //assert(0); // "Undefined communication type is given"
    //        break;
}

MPIutil get_instance() {
    static int entered;
    int flag = (entered == 1);
    if (flag) {
      return mpiutil;
    }

    pthread_mutex_lock(&mutex);
    entered = 1;
    pthread_mutex_unlock(&mutex);

    mpiutil = malloc(sizeof(*mpiutil));
    mpiutil->set_comm = set_comm;
    mpiutil->get_rank = get_rank;
    mpiutil->get_size = get_size;
    mpiutil->get_tag = get_tag;
    mpiutil->mpisendrecv = mpisendrecv;
    mpitag = 0;
    return mpiutil;
}

