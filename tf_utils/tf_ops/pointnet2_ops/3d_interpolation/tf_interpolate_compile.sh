# TF1.2
# g++ -std=c++11 tf_interpolate.cpp -o tf_interpolate_so.so -shared -fPIC -I /usr/local/lib/python2.7/dist-packages/tensorflow/include -I /usr/local/cuda-8.0/include -lcudart -L /usr/local/cuda-8.0/lib64/ -O2 -D_GLIBCXX_USE_CXX11_ABI=0

# # TF1.4

# # /home/mmvc/anaconda2/envs/Xiang_Li/lib/python2.7/site-packages/tensorflow/include 

TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
TF_LIB=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')
    
g++ -std=c++11 tf_interpolate.cpp -o tf_interpolate_so.so -shared -fPIC -I /home/mmvc/anaconda2/envs/Xiang_Li/lib/python2.7/site-packages/tensorflow/include -I /usr/local/cuda-8.0/include -I /home/mmvc/anaconda2/envs/Xiang_Li/lib/python2.7/site-packages/tensorflow/include/external/nsync/public -lcudart -L /usr/local/cuda-8.0/lib64/ -L//home/mmvc/anaconda2/envs/Xiang_Li/lib/python2.7/site-packages/tensorflow -ltensorflow_framework -O2 -D_GLIBCXX_USE_CXX11_ABI=0 
# -I$TF_INC/external/nsync/public -L$TF_LIB -ltensorflow_framework
