#!/bin/bash

#-ex
. ../../setenv
#export QHOME=${HOME}/mpi-qulacs

. ${VENV_PATH}/bin/activate

cd ${QHOME}/ict/python

#NP=$1; shift
NP=${OMPI_COMM_WORLD_LOCAL_SIZE}
LSIZE=${OMPI_COMM_WORLD_LOCAL_SIZE}
LRANK=${OMPI_COMM_WORLD_LOCAL_RANK}

COMM=

if [ $NP -eq 1 ]; then
    numactl -m 0-3 -N 0-3 ${COMM} ${@}
elif [ $NP -eq 4 ]; then
    numactl -N ${LRANK} -m ${LRANK} ${COMM} ${@}
else
  ${COMM} ${@}
fi

#eof
