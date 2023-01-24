#!/bin/bash

rm CryptoFL_lib/CryptoFL*.so
echo "rm CryptoFL.so!"
cd build 
#make clean 
cmake -DCMAKE_POSITION_INDEPENDENT_CODE=ON ..
make -j8

cp lib/CryptoFL_lib*.so ../CryptoFL_lib/
echo "cp CryptoFL.so!"
cd ../
# python $1
