#!/bin/bash

#-ex
export QHOME=/home/yamazaki/mpi-qulacs

. ${QHOME}/setenv
#ulimit -s 1048576
#export OMP_STACKSIZE=1G
#export FLIB_BARRIER=HARD

cd ${QHOME}/ict/python

#NP=$1; shift
NP=${OMPI_COMM_WORLD_LOCAL_SIZE}

COMM="python qulacsbench.py"

if [ $NP -eq 1 ]; then
  export OMP_NUM_THREADS=48
  export OMP_PROC_BIND=TRUE
  export GOMP_CPU_AFFINITY=0-47

  numactl -m 0-3 -N 0-3 ${COMM} ${@} 2>&1
  #${COMM} ${@} 2>&1
elif [ $NP -eq 4 ]; then
  export OMP_NUM_THREADS=12
  export OMP_PROC_BIND=TRUE
  case "${OMPI_COMM_WORLD_LOCAL_RANK}" in
  0)
    export GOMP_CPU_AFFINITY=0-11
    exec numactl -N 0 -m 0 ${COMM} ${@}
    ;;
  1)
    export GOMP_CPU_AFFINITY=12-23
    exec numactl -N 1 -m 1 ${COMM} ${@}
    ;;
  2)
    export GOMP_CPU_AFFINITY=24-35
    exec numactl -N 2 -m 2 ${COMM} ${@}
    ;;
  3)
    export GOMP_CPU_AFFINITY=36-47
    exec numactl -N 3 -m 3 ${COMM} ${@}
    ;;
  *)
    echo "Local Rank ERROR: ${OMPI_COMM_WORLD_LOCAL_RANK}"
    exit 1
    ;;
  esac
elif [ $NP -eq 48 ]; then
  ${COMM} ${@}
else
  echo "unexpected the number of Core slots"
fi

#eof
