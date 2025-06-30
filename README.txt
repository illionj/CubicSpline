编译测试
make && ./test && python3 show.py

观察png图片

清除结果
make clean

CC=clang CXX=clang++ cmake -B build -DCMAKE_BUILD_TYPE=Release
CC=gcc CXX=g++ cmake -B build -DCMAKE_BUILD_TYPE=Release
CC=clang CXX=clang++ cmake -B build -DCMAKE_BUILD_TYPE=RELWITHDEBINFO
cmake --build build  
cmake --build build && cp ./build/src/libtrajectory_smooth.so  ../pure_pursuit/example/