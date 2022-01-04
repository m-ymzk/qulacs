#include <assert.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include "MPIutil.h"

static MPI_Comm mpicomm = 0;
static int mpirank = 0;
static int mpisize = 0;
static int mpitag = 0;
static int initialized = 0; // initialized = 1;
static MPIutil mpiutil;
static MPI_Status mpistat;
static pthread_mutex_t mutex= PTHREAD_MUTEX_INITIALIZER;

static void set_comm(MPI_Comm c) {
    if (mpicomm == 0) {
        mpicomm = c;
        initialized = 1;
        MPI_Comm_rank(mpicomm, &mpirank);
        MPI_Comm_size(mpicomm, &mpisize);
    } else if (mpicomm != c) { 
        fprintf(stderr, "## warn: MPI_Comm is conflicted!! %x %x\n", (uint)mpicomm, (uint)c);
    }
}

static int usempi() {
    return initialized; // if mpi didn't initialized, return false(0)
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

static void barrier() {
    MPI_Barrier(mpicomm);
}

static void m_DC_sendrecv(void *sendbuf, void *recvbuf, int count, int pair_rank) {
    int tag0 = get_tag();
    int mpi_tag1 = tag0 + ((mpirank & pair_rank)<<1) + (mpirank > pair_rank);
    int mpi_tag2 = mpi_tag1 ^ 1;
    //int mpi_tag1 = tag0 + (mpirank & 0xFFFE);
    //int mpi_tag2 = mpi_tag1 ^ 1;
    //printf("#%d: m_DC_sendrecv: %d, %d, %d, %d, %d\n", mpirank, count, mpirank, pair_rank, mpi_tag1, mpi_tag2);
    MPI_Sendrecv(sendbuf, count, MPI_DOUBLE_COMPLEX, pair_rank, mpi_tag1,
                 recvbuf, count, MPI_DOUBLE_COMPLEX, pair_rank, mpi_tag2,
                 mpicomm, &mpistat);
    }

/*
static void m_DC_isendrecv(void *sendbuf, void *recvbuf, int count, int pair_rank) {
    MPI_Request send_request;
    int tag0 = get_tag();
    int mpi_tag1 = tag0 + ((mpirank & pair_rank)<<1) + (mpirank > pair_rank);
    int mpi_tag2 = mpi_tag1 ^ 1;

    MPI_Isend(sendbuf, sendcount, MPI_DOUBLE,
              pair_rank, mpi_tag1, comm, & (send_request));
    MPI_recv(recvbuf, recvcount, MPI_DOUBLE,
              pair_rank, mpi_tag2, comm, &mpistat);

    return send_request;
}
*/

static void m_I_allreduce(void* buf, UINT count) {
    MPI_Allreduce(MPI_IN_PLACE, buf, count, MPI_LONG_LONG_INT, MPI_SUM, mpicomm);
}

static void s_D_allgather(double a, void* recvbuf) {
    MPI_Allgather(&a, 1, MPI_DOUBLE, recvbuf, 1, MPI_DOUBLE, mpicomm);
}

static void s_D_allreduce(void* buf) {
    MPI_Allreduce(MPI_IN_PLACE, buf, 1, MPI_DOUBLE, MPI_SUM, mpicomm);
}

/*
static double s_D_send_next_rank(double a) {
    int tag0 = get_tag();
    int pair_rank = (mpirank + 1) % mpisize;
    int mpi_tag1 = tag0 + ((mpirank & pair_rank)<<1) + (mpirank > pair_rank);
    int mpi_tag2 = mpi_tag1 ^ 1;
    double ret;
    printf("#%d: s_D_send_next_rank: %f, %d, %d, %d\n", mpirank, a, pair_rank, mpi_tag1, mpi_tag2);
    MPI_Sendrecv(&a, 1, MPI_DOUBLE, pair_rank, mpi_tag1,
                 &ret, 1, MPI_DOUBLE, pair_rank, mpi_tag2,
                 mpicomm, &mpistat);
    return ret;
}
*/

static int s_i_bcast(int a) {
    int ret = a;
    //if (mpirank == 0) printf("#%d: s_ui_bcast(result): %d\n", mpirank, a);
    MPI_Bcast(&ret, 1, MPI_INT, 0, mpicomm);
    //printf("#%d: s_ui_bcast: %d\n", mpirank, ret);
    return ret;
}
/*
static void send_osstr(char *sendbuf, int len) {
}

static void recv_osstr(char *recvbuf, int len) {
}
*/

MPIutil get_mpiutil() {
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
    mpiutil->usempi = usempi;
    mpiutil->barrier = barrier;
    mpiutil->m_DC_sendrecv = m_DC_sendrecv; // multi, Double Complex, SendRecv
    mpiutil->m_I_allreduce = m_I_allreduce; // multi, Double, Allreduce
    mpiutil->s_D_allgather = s_D_allgather; // single, Double, Allgather
    mpiutil->s_D_allreduce = s_D_allreduce; // single, Double, Allreduce
    //mpiutil->s_D_send_next_rank = s_D_send_next_rank; // single, Double, Send_Next_Rank
    mpiutil->s_i_bcast = s_i_bcast; // single, Int, Bcast
    //mpiutil->recv_osstr = recv_osstr;
    //mpiutil->send_osstr = send_osstr;
    mpitag = 0;
    return mpiutil;
}

