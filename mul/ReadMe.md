Readme for GPU midterm exam:

## Part 1:
#### Case 1: ./mul 1024 1024 1024
CPU executation is **4.1862**
GPU executation for A2 is **0.0849**
Part A1 Test PASSED
GPU executation for A2 is **0.0387**
Part A2 Test PASSED
A1 speedup **49.3**
A2 Speedup **108.17**

#### Case 2: ./mul 2048 2048 2048
CPU executation is 52.8143
GPU executation for A2 is 0.5763
Part A1 Test PASSED

GPU executation for A2 is 0.1827
Part A2 Test PASSED
A1 speedup **91.64**
A2 Speedup **289.08**

#### Case 3: ./mul 1024 2048 1024
>>> ./mul 1024 2048 1024

CPU executation is 8.2365
The execution time of GPU is :135.629028
GPU executation for A2 is 0.1357
Part A1 Test PASSED

GPU executation for A2 is 0.0557
Part A2 Test PASSED

#### Case 4 ./mul 1234 2345 1256
>>> ./mul 1234 2345 1256

CPU executation is 11.7326
The execution time of GPU is :233.147202
GPU executation for A2 is 0.2332
Part A1 Test PASSED

GPU executation for A2 is 0.0905
Part A2 Test PASSED

## Part 2: Non-square-tiled Matrix Multipilication
#### case 1
 bm 16
 bk bm
 bn 64
 >>> ./mul 1024 2048 1024

 GPU executation for B is **0.2423**
 Part B Test PASSED

#### case 2
  bm 16
  bk bm
  bn 32

./mul 1024 2048 1024

GPU executation for B is **0.1353**
Part B Test PASSED

#### case 3
bm 16
bk bm
bn 16
./mul 1024 2048 1024

GPU executation for B is **0.0749**
Part B Test PASSED

#### case 4

#define bm 32
#define bk bm
#define bn 32

./mul 1024 2048 1024

GPU executation for B is **0.2033**
Part B Test PASSED
