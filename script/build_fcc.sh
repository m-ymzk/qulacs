#!/bin/sh
#export TCSDS_PATH="/opt/FJSVstclanga/v1.1.0"
#export CC="${TCSDS_PATH}/bin/fcc -Nclang -Kfast -Knolargepage -lpthread"
#export CXX="${TCSDS_PATH}/bin/FCC -Nclang -Kfast -Knolargepage -lpthread"

GCC_COMMAND="mpifcc"
GXX_COMMAND="mpiFCC"

mkdir ./build
cd ./build
cmake -G "Unix Makefiles" -D CMAKE_C_COMPILER=$GCC_COMMAND -D CMAKE_CXX_COMPILER=$GXX_COMMAND -D CMAKE_BUILD_TYPE=Release .. .. 2>&1 |tee -a build_cmake.log
make 2>&1 |tee -a build_make.log
make python 2>&1 |tee -a build_make2.log
cd ../

