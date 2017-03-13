# Makefile
#
# author      : Weijun Xiao <wxiao@umn.edu>
# version     : 0.3
# date        : 20170122
# discription : generic Makefile for making cuda programs
#

BIN               := simplematrixmul 
CXXFLAGS          := -O3
LIBSUFFIX	  := _x86_64

CUDA_INSTALL_PATH ?= /usr/local/cuda
CUDA_SDK_PATH ?= /usr/local/cuda/samples


NVCC ?= $(CUDA_INSTALL_PATH)/bin/nvcc
INCD = -I"$(CUDA_SDK_PATH)/C/common/inc" -I"$(CUDA_INSTALL_PATH)/include" -I"./"
LIBS = -L"$(CUDA_INSTALL_PATH)/lib64" -lcuda -lcudart  -lstdc++ -lpthread
CUDA_SDK?=4
COMMONFLAGS = -DCUDA_SDK=$(CUDA_SDK)
NVCCFLAGS := --ptxas-options=-v -O3 


# files
CPP_SOURCES       := simplematrixmul_gold.cpp 
CU_SOURCES        := simplematrixmul.cu
HEADERS           := $(wildcard *.h)
CPP_OBJS          := $(patsubst %.cpp, %.o, $(CPP_SOURCES))
CU_OBJS           := $(patsubst %.cu, %.cu_o, $(CU_SOURCES))

%.cu_o : %.cu
	$(NVCC) $(NVCCFLAGS) -c $(INCD) -o $@ $<

%.o: %.cpp
	$(CXX) -c $(CXXFLAGS) $(INCD) -o $@ $<

$(BIN): $(CPP_OBJS) $(CU_OBJS)
	$(CXX) -o $(BIN) $(CU_OBJS) $(CPP_OBJS) $(LDFLAGS) $(INCD) $(LIBS)

clean:
	rm -f $(BIN) *.o *.cu_o


