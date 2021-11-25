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
    void (*mpisendrecv)(void *sendbuf, void *recvbuf, int count, int peer_rank);
} *MPIutil;

MPIutil get_instance(void);

#ifdef __cplusplus
}
#endif /* __cplusplus */

