#include "utility.hpp"

#include <assert.h>
#include <errno.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "constant.hpp"

void get_Pauli_masks_partial_list(const UINT* target_qubit_index_list,
    const UINT* Pauli_operator_type_list, UINT target_qubit_index_count,
    ITYPE* bit_flip_mask, ITYPE* phase_flip_mask,
    UINT* global_phase_90rot_count, UINT* pivot_qubit_index) {
    (*bit_flip_mask) = 0;
    (*phase_flip_mask) = 0;
    (*global_phase_90rot_count) = 0;
    (*pivot_qubit_index) = 0;
    for (UINT cursor = 0; cursor < target_qubit_index_count; ++cursor) {
        UINT target_qubit_index = target_qubit_index_list[cursor];
        switch (Pauli_operator_type_list[cursor]) {
            case 0:  // I
                break;
            case 1:  // X
                (*bit_flip_mask) ^= 1ULL << target_qubit_index;
                (*pivot_qubit_index) = target_qubit_index;
                break;
            case 2:  // Y
                (*bit_flip_mask) ^= 1ULL << target_qubit_index;
                (*phase_flip_mask) ^= 1ULL << target_qubit_index;
                (*global_phase_90rot_count)++;
                (*pivot_qubit_index) = target_qubit_index;
                break;
            case 3:  // Z
                (*phase_flip_mask) ^= 1ULL << target_qubit_index;
                break;
            default:
                fprintf(stderr, "Invalid Pauli operator ID called");
                assert(0);
        }
    }
}

void get_Pauli_masks_whole_list(const UINT* Pauli_operator_type_list,
    UINT target_qubit_index_count, ITYPE* bit_flip_mask, ITYPE* phase_flip_mask,
    UINT* global_phase_90rot_count, UINT* pivot_qubit_index) {
    (*bit_flip_mask) = 0;
    (*phase_flip_mask) = 0;
    (*global_phase_90rot_count) = 0;
    (*pivot_qubit_index) = 0;
    for (UINT target_qubit_index = 0;
         target_qubit_index < target_qubit_index_count; ++target_qubit_index) {
        switch (Pauli_operator_type_list[target_qubit_index]) {
            case 0:  // I
                break;
            case 1:  // X
                (*bit_flip_mask) ^= 1ULL << target_qubit_index;
                (*pivot_qubit_index) = target_qubit_index;
                break;
            case 2:  // Y
                (*bit_flip_mask) ^= 1ULL << target_qubit_index;
                (*phase_flip_mask) ^= 1ULL << target_qubit_index;
                (*global_phase_90rot_count)++;
                (*pivot_qubit_index) = target_qubit_index;
                break;
            case 3:  // Z
                (*phase_flip_mask) ^= 1ULL << target_qubit_index;
                break;
            default:
                fprintf(stderr, "Invalid Pauli operator ID called");
                assert(0);
        }
    }
}

ITYPE* create_matrix_mask_list(
    const UINT* qubit_index_list, UINT qubit_index_count) {
    const ITYPE matrix_dim = 1ULL << qubit_index_count;
    ITYPE* mask_list = (ITYPE*)calloc((size_t)matrix_dim, sizeof(ITYPE));
    ITYPE cursor = 0;

    for (cursor = 0; cursor < matrix_dim; ++cursor) {
        for (UINT bit_cursor = 0; bit_cursor < qubit_index_count;
             ++bit_cursor) {
            if ((cursor >> bit_cursor) % 2) {
                UINT bit_index = qubit_index_list[bit_cursor];
                mask_list[cursor] ^= (1ULL << bit_index);
            }
        }
    }
    return mask_list;
}

ITYPE
create_control_mask(
    const UINT* qubit_index_list, const UINT* value_list, UINT size) {
    ITYPE mask = 0;
    for (UINT cursor = 0; cursor < size; ++cursor) {
        mask ^= (1ULL << qubit_index_list[cursor]) * value_list[cursor];
    }
    return mask;
}

static int compare_ui(const void* a, const void* b) {
    return (*((UINT*)a)) - (*((UINT*)b));
}
void sort_ui(UINT* array, size_t size) {
    qsort(array, size, sizeof(UINT), compare_ui);
}
UINT* create_sorted_ui_list(const UINT* array, size_t size) {
    UINT* new_array = (UINT*)calloc(size, sizeof(UINT));
    memcpy(new_array, array, size * sizeof(UINT));
    sort_ui(new_array, size);
    return new_array;
}
UINT* create_sorted_ui_list_value(const UINT* array, size_t size, UINT value) {
    UINT* new_array = (UINT*)calloc(size + 1, sizeof(UINT));
    memcpy(new_array, array, size * sizeof(UINT));
    new_array[size] = value;
    sort_ui(new_array, size + 1);
    return new_array;
}
UINT* create_sorted_ui_list_list(
    const UINT* array1, size_t size1, const UINT* array2, size_t size2) {
    UINT* new_array = (UINT*)calloc(size1 + size2, sizeof(UINT));
    memcpy(new_array, array1, size1 * sizeof(UINT));
    memcpy(new_array + size1, array2, size2 * sizeof(UINT));
    sort_ui(new_array, size1 + size2);
    return new_array;
}

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

    omputil = (OMPutil)malloc(sizeof(*omputil));

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
