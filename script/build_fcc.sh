#!/bin/sh
export TCSDS_PATH="/opt/FJSVstclanga/v1.1.0"
export CC="${TCSDS_PATH}/bin/fcc -Nclang -Kfast -Knolargepage -lpthread"
export CXX="${TCSDS_PATH}/bin/FCC -Nclang -Kfast -Knolargepage -lpthread"

GCC_COMMAND="fcc"
GXX_COMMAND="FCC"

mkdir ./build
cd ./build
cmake -G "Unix Makefiles" -D CMAKE_C_COMPILER=$GCC_COMMAND -D CMAKE_CXX_COMPILER=$GXX_COMMAND -D CMAKE_BUILD_TYPE=Release ..
make
make python
cd ../

