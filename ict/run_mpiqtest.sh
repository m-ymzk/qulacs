#!/bin/bash -e

if [ $# -ge 1 ]; then
  if [ $1 = "-h" -o $1 = "--help" ]; then
    echo " Usage: $0 TargetBit NQubit NumThreads FappFlag"
    echo " Four arguments are optional."
    exit
  fi
fi


TargetBit=${1:-1}
NQubit=${2:-20}
NT=${3:-1}
FappFlag=${4:-0}
PerfStatFlag=0

DebugFlag=-1
NumLoops=100

Test=mpiqtest 
NP=1

export OMP_NUM_THREADS=${NT}
export OMP_PROC_BIND=TRUE

echo "OMP_NUM_THREADS: ${OMP_NUM_THREADS}"
echo "NumLoops: ${NumLoops}"

case ${NT} in
  1)
    export GOMP_CPU_AFFINITY=0
    Cmd="numactl -N 0 -m 0 ./${Test} ${DebugFlag} ${NQubit} ${TargetBit} ${NumLoops}"
    ;;
  12)
    export OMP_PROC_BIND=TRUE
    export GOMP_CPU_AFFINITY=0-11
    Cmd="numactl -N 0 ./${Test} ${DebugFlag} ${NQubit} ${TargetBit} ${NumLoops}"
    ;;
  48)
    export OMP_PROC_BIND=TRUE
    export GOMP_CPU_AFFINITY=0-47
    Cmd="numactl -N 0-3 ./${Test} ${DebugFlag} ${NQubit} ${TargetBit} ${NumLoops}"
    ;;
  *)
    echo "NT must be 1, 12, or 48."
    exit
esac

if [ ${FappFlag} -ne 0 ]; then
  for i in `seq 1 17`
  do 
    echo "Loop: ${i}"
    fapp -C -d ./rep${i} -Hevent=pa${i} mpirun -n 1 ${Cmd}
  done

elif [ ${PerfStatFlag} -ne 0 ]; then
  perf stat mpirun -np ${NP} ${Cmd}
else
  mpirun -np ${NP} ${Cmd}
fi

