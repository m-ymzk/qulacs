#!/bin/bash
OPT_MPI="-x UCX_IB_MLX5_DEVX=no"

rm -f _log_regression_n*
mpirun -n 2 ${OPT_MPI} ../bin/cppsim_test --gtest_filter=*multicpu* 2>&1 |tee -a _log_regression_n2
mpirun -n 4 ${OPT_MPI} ../bin/cppsim_test --gtest_filter=*multicpu* 2>&1 |tee -a _log_regression_n4
mpirun -n 8 ${OPT_MPI} ../bin/cppsim_test --gtest_filter=*multicpu* 2>&1 |tee -a _log_regression_n8
mpirun -n 16 ${OPT_MPI} ../bin/cppsim_test --gtest_filter=*multicpu* 2>&1 |tee -a _log_regression_n16
mpirun -n 32 ${OPT_MPI} ../bin/cppsim_test --gtest_filter=*multicpu* 2>&1 |tee -a _log_regression_n32

echo Num of FAIL, PASSED, log_file
for f in _log_regression_n*; do
   	fn=`grep FAIL $f | wc -l`
	ps=`grep PASSED $f | wc -l`
	echo $fn $ps $f
done
