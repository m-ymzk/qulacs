//
#pragma once

#include <pthread.h>
#include <stdlib.h>
#include "type.h"

#include <mpi.h>

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

//typedef enum { Undefined        =  0,
//               TypeISend        =  1,
//               TypeIRecv        =  2,
//               TypeISendRecv    =  3,
//} CommRequest_t;

typedef struct {
    int (*get_rank)();
    int (*get_size)();
    int (*get_tag)();
    CTYPE* (*get_workarea)(ITYPE *dim_work, ITYPE *num_work);
    void (*release_workarea)();
    void (*barrier)();
    void (*m_DC_sendrecv)(void *sendbuf, void *recvbuf, int count, int pair_rank);
    void (*m_DC_sendrecv_replace)(void *buf, int count, int pair_rank);
    void (*m_I_allreduce)(void *buf, UINT count);
    void (*s_D_allgather)(double a, void *recvbuf);
    void (*s_D_allreduce)(void *buf);
    int (*s_i_bcast)(int a);
    //void (*send_osstr)(char* sendbuf, int len);
    //void (*recv_osstr)(char* recvbuf, int len);
    //double (*s_D_send_next_rank)(double a);
} *MPIutil;

MPIutil get_mpiutil(void);

#ifdef __cplusplus
}
#endif /* __cplusplus */

