TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
TF_LIB=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')

#/bin/bash
/usr/local/cuda-8.0/bin/nvcc DFCN.cu -o DFCN_g.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC

# TF1.4
g++ -std=c++11 main.cpp DFCN_g.cu.o -o tf_DFCN_so.so -shared -fPIC -I$TF_INC -I /usr/local/cuda-8.0/include -I$TF_INC/external/nsync/public -lcudart -L /usr/local/cuda-8.0/lib64/ -L$TF_LIB -ltensorflow_framework -O2 -D_GLIBCXX_USE_CXX11_ABI=0