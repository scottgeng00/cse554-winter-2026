rm -rf build
mkdir build
cd build
cmake .. && make && ./rms_norm_vector
cd ..
