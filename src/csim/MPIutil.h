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
    /* int a; */
    void (*set_comm)(MPI_Comm c);
    int (*usempi)();
    int (*get_rank)();
    int (*get_size)();
    int (*get_tag)();
    void (*barrier)();
    double (*s_D_allreduce)();
    //void (*send_osstr)(char* sendbuf, int len);
    //void (*recv_osstr)(char* recvbuf, int len);
    void (*m_DC_sendrecv)(void *sendbuf, void *recvbuf, int count, int peer_rank);
} *MPIutil;

MPIutil get_mpiutil(void);

#ifdef __cplusplus
}
#endif /* __cplusplus */

