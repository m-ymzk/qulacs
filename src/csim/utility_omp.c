// #include <stdio.h>
#include <stdlib.h>
#include <errno.h>

#include "utility.h"

#ifdef _OPENMP
static OMPutil omputil = NULL;
static UINT qulacs_num_default_thread_max = 1;
static UINT qulacs_num_thread_max = 0;
static UINT qulacs_force_threshold = 0;

static void set_qulacs_num_threads(ITYPE dim, UINT para_threshold) {
    UINT threshold = para_threshold;
    if (qulacs_force_threshold) threshold = qulacs_force_threshold;
    if (dim < (((ITYPE)1) << threshold)) {
        omp_set_num_threads(1);
    } else {
        omp_set_num_threads(qulacs_num_thread_max);
    }
}

static void reset_qulacs_num_threads() {
    omp_set_num_threads(qulacs_num_default_thread_max);
}

#define REGISTER_METHOD_POINTER(M) omputil->M = M;

OMPutil get_omputil() {
    if (omputil != NULL) {
        return omputil;
    }

    omputil = malloc(sizeof(*omputil));

    errno = 0;
    qulacs_num_thread_max = omp_get_max_threads();
    char* endp;
    char* tmp = getenv("QULACS_NUM_THREADS");
    if (tmp) {
        const UINT tmp_val = strtol(tmp, &endp, 0);
        if (0 < tmp_val && tmp_val < 1025) qulacs_num_thread_max = tmp_val;
    }

    qulacs_force_threshold = 0;
    tmp = getenv("QULACS_FORCE_THRESHOLD");
    if (tmp) {
        const UINT tmp_val = strtol(tmp, &endp, 0);
        if (0 < tmp_val && tmp_val < 65) qulacs_force_threshold = tmp_val;
    }

    qulacs_num_default_thread_max = omp_get_max_threads();

    REGISTER_METHOD_POINTER(set_qulacs_num_threads)
    REGISTER_METHOD_POINTER(reset_qulacs_num_threads)

    return omputil;
}

#undef REGISTER_METHOD_POINTER
#endif
