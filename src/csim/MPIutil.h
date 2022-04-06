//
#pragma once

#include <mpi.h>
#include <pthread.h>
#include <stdlib.h>

#include "type.h"

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

// typedef enum { Undefined        =  0,
//                TypeISend        =  1,
//                TypeIRecv        =  2,
//                TypeISendRecv    =  3,
// } CommRequest_t;

typedef struct {
    int (*get_rank)();
    int (*get_size)();
    int (*get_tag)();
    CTYPE *(*get_workarea)(ITYPE *dim_work, ITYPE *num_work);
    void (*release_workarea)();
    void (*barrier)();
    void (*mpi_wait)(UINT count);
    void (*m_DC_allgather)(void *sendbuf, void *recvbuf, int count);
    void (*m_DC_send)(void *sendbuf, int count, int pair_rank);
    void (*m_DC_recv)(void *recvbuf, int count, int pair_rank);
    void (*m_DC_sendrecv)(
        void *sendbuf, void *recvbuf, int count, int pair_rank);
    void (*m_DC_sendrecv_replace)(void *buf, int count, int pair_rank);
    void (*m_DC_isendrecv)(
        void *sendbuf, void *recvbuf, int count, int pair_rank);
    void (*m_I_allreduce)(void *buf, UINT count);
    void (*s_D_allgather)(double a, void *recvbuf);
    void (*s_D_allreduce)(void *buf);
    void (*s_u_bcast)(UINT *a);
    void (*s_D_bcast)(double *a);
    // void (*send_osstr)(char* sendbuf, int len);
    // void (*recv_osstr)(char* recvbuf, int len);
    // double (*s_D_send_next_rank)(double a);
} MPIutil_;
typedef MPIutil_ *MPIutil;

MPIutil get_mpiutil(void);

#ifdef __cplusplus
}
#endif /* __cplusplus */
