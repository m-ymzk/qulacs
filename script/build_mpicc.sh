#!/bin/sh

set -eux

export C_COMPILER="mpicc"
export CXX_COMPILER="mpic++"
unset QULACS_OPT_FLAGS
export USE_MPI=Yes
export USE_TEST=Yes

./script/build_gcc.sh

