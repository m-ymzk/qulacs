#!/bin/sh
#export TCSDS_PATH="/opt/FJSVstclanga/v1.1.0"
#export CC="${TCSDS_PATH}/bin/fcc -Nclang -Kfast -Knolargepage -lpthread"
#export CXX="${TCSDS_PATH}/bin/FCC -Nclang -Kfast -Knolargepage -lpthread"

set -ex

GCC_COMMAND="mpifcc"
GXX_COMMAND="mpiFCC"
unset QULACS_OPT_FLAGS
export USE_MPI=Yes
export USE_TEST=Yes

mkdir -p ./build
cd ./build
cmake -G "Unix Makefiles" -D CMAKE_C_COMPILER=$GCC_COMMAND -D CMAKE_CXX_COMPILER=$GXX_COMMAND -D CMAKE_BUILD_TYPE=Release \
       	-D USE_GPU=$USE_GPU -D USE_TEST=$USE_TEST -D USE_MPI=$USE_MPI .. 2>&1 |tee -a build_cmake.log
make -j $(nproc) 2>&1 |tee -a build_make.log
#make python 2>&1 |tee -a build_make2.log
cd ../

