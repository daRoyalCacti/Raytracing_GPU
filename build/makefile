CUDA_PATH     ?= /opt/cuda
HOST_COMPILER  = g++
NVCC           = $(CUDA_PATH)/bin/nvcc -ccbin $(HOST_COMPILER)

# select one of these for Debug vs. Release
#NVCC_DBG       = -g -G
NVCC_DBG       =

NVCCFLAGS      = $(NVCC_DBG) -m64 -std=c++17 -lassimp -lpng
GENCODE_FLAGS  = -gencode arch=compute_60,code=sm_60

#Generate: Generate.o
#	$(NVCC) $(NVCCFLAGS) $(GENCODE_FLAGS) -o Generate Generate.o

#Generate.o: ../main.cu
#	$(NVCC) $(NVCCFLAGS) $(GENCODE_FLAGS) -o Generate.o -c ../main.cu

Generate: ../main.cu
	$(NVCC) `libpng-config --cflags --ldflags` $(NVCCFLAGS) $(GENCODE_FLAGS) -o Generate ../main.cu -I.

profile_basic: Generate
	nvprof ./Generate

# use nvprof --query-metrics
profile_metrics: Generate
	nvprof --metrics achieved_occupancy,inst_executed,inst_fp_32,inst_fp_64,inst_integer ./Generate

clean:
	rm -f Generate Generate.o
