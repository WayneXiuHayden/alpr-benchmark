# alpr-benchmark

build
```
mkdir build && cd build
cmake ..
make -j$(nproc)
```

run
```
# gpu
./benchmark --use_gpu  --batch_size 1 --stream_queue_size 200
# cpu
./benchmark --batch_size 10 --stream_queue_size 200
```