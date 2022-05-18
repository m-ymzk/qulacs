#include "constant.h"
#include "type.h"
#include "omp.h"

#ifdef _OPENMP 
typedef struct {
    void (*set_qulacs_num_threads)(ITYPE dim, UINT threshold);
    void (*reset_qulacs_num_threads)();
} OMPutil_;
typedef OMPutil_ *OMPutil;

OMPutil get_omputil(void);
#endif

