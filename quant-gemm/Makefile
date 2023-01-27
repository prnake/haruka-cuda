NVCC=nvcc
OPTIONS=--std=c++11 --gpu-architecture=compute_80 -lcublas

TARGETS=$(patsubst %.cu, %, $(wildcard *.cu))

all: $(TARGETS)

%: %.cu %.cpp
	$(NVCC) $^ $(OPTIONS) -o $@

run: quantization
	./quantization

.PHONY : clean, copy
clean:
	rm $(TARGETS)
