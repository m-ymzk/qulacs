// #include <stdio.h>
#include <stdlib.h>

#include "utility.h"

#ifdef _OPENMP
static OMPutil omputil = NULL;

static UINT qulacs_num_default_thread_max = 1;
static UINT qulacs_num_thread_max = 0;

static void set_qulacs_num_threads(ITYPE dim, UINT threshold) {
    if (dim < (((ITYPE)1) << threshold)) {
        omp_set_num_threads(1);
        // printf("# set omp_num_thread = 1\n");
    } else {
        omp_set_num_threads(qulacs_num_thread_max);
        // printf("# set omp_num_thread = %d\n", qulacs_num_thread_max);
        // printf("# omp_max_thread = %d\n", omp_get_max_threads());
    }
}

static void reset_qulacs_num_threads() {
    omp_set_num_threads(qulacs_num_default_thread_max);
    // printf("# reset omp_num_thread = %d\n", qulacs_num_default_thread_max);
}

#define REGISTER_METHOD_POINTER(M) omputil->M = M;

OMPutil get_omputil() {
    if (omputil != NULL) {
        return omputil;
    }

    // printf("# set omputil initializer entry, %d\n", qulacs_num_thread_max);
    qulacs_num_thread_max = omp_get_max_threads();
    const char *tmp = getenv("QULACS_NUM_THREADS");
    if (tmp) {
        const UINT tmp_val = atoi(tmp);
        if (0 < tmp_val && tmp_val < 1025) qulacs_num_thread_max = tmp_val;
    }
    // printf("# set qulacs_num_thread_max = %d\n", qulacs_num_thread_max);

    qulacs_num_default_thread_max = omp_get_max_threads();
    // printf("# set qulacs_num_default_thread_max = %d\n",
    //    qulacs_num_default_thread_max);

    omputil = malloc(sizeof(*omputil));
    REGISTER_METHOD_POINTER(set_qulacs_num_threads)
    REGISTER_METHOD_POINTER(reset_qulacs_num_threads)

    return omputil;
}

#undef REGISTER_METHOD_POINTER
#endif
