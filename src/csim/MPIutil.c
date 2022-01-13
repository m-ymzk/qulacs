//
#include <assert.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include "utility.h"
#include "MPIutil.h"

#define _NQUBIT_WORK 5 // small buffer(5 qubit/proc.) for test
//#define _NQUBIT_WORK 22 // 4 Mi x 16 Byte(CTYPE)

static MPI_Comm mpicomm = 0;
static int mpirank = 0;
static int mpisize = 0;
static int mpitag = 0;
static MPIutil mpiutil = NULL;
static MPI_Status mpistat;
//static pthread_mutex_t mutex= PTHREAD_MUTEX_INITIALIZER;
static CTYPE* workarea = NULL;

static int get_rank() {
    return mpirank;
}

static int get_size() {
    return mpisize;
}

static int get_tag() {
    //pthread_mutex_lock(&mutex);
    mpitag ^= 1<<20; // max 1M-ranks
    //pthread_mutex_unlock(&mutex);
    return mpitag;
}

static CTYPE* get_workarea(ITYPE *dim_work, ITYPE *num_work) {
    ITYPE dim = *dim_work;
    *dim_work = get_min_ll(1 << _NQUBIT_WORK, dim);
    *num_work = get_max_ll(1, dim >> _NQUBIT_WORK);
    if (workarea == NULL) {
        workarea = (CTYPE *)malloc(sizeof(CTYPE) * (1 << _NQUBIT_WORK));
        if (workarea == NULL) {
            fprintf(stderr, "Can't malloc for variable, %s, %d\n", __FILE__,
                    __LINE__);
        exit(1);
        }
    }
    return workarea;
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

#define REGISTER_METHOD_POINTER( M ) mpiutil->M = M;

MPIutil get_mpiutil() {
    if (mpiutil != NULL) {
      return mpiutil;
    }

    mpicomm = MPI_COMM_WORLD;
    //printf("# MPI_COMM_WORLD %p\n", mpicomm);
    MPI_Comm_rank(mpicomm, &mpirank);
    MPI_Comm_size(mpicomm, &mpisize);
    mpitag = 0;

    mpiutil = malloc(sizeof(*mpiutil));
    REGISTER_METHOD_POINTER(get_rank)
    REGISTER_METHOD_POINTER(get_size)
    REGISTER_METHOD_POINTER(get_tag)
    REGISTER_METHOD_POINTER(get_workarea)
    REGISTER_METHOD_POINTER(barrier)
    REGISTER_METHOD_POINTER(m_DC_sendrecv)
    REGISTER_METHOD_POINTER(m_I_allreduce)
    REGISTER_METHOD_POINTER(s_D_allgather)
    REGISTER_METHOD_POINTER(s_D_allreduce)
    REGISTER_METHOD_POINTER(s_i_bcast)
    //mpiutil->s_D_send_next_rank = s_D_send_next_rank; // single, Double, Send_Next_Rank
    //mpiutil->recv_osstr = recv_osstr;
    //mpiutil->send_osstr = send_osstr;
    return mpiutil;
}

#undef REGISTER_METHOD_POINTER
