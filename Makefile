# `make` builds bin/sgemm for the GPU in this machine. With no GPU (CI): make SM=sm_89
NVCC      ?= nvcc
CUDA_HOME ?= /usr/local/cuda
SM        ?= native
BIN       := bin

# Every binary records the commit it was built from, with "-dirty" when the tree had
# uncommitted changes, and prints it at the top of every log. The stamp file rebuilds the
# binary whenever that string changes, so a log can't name a commit its binary didn't come from.
GIT_SHA   := $(shell git describe --always --dirty --abbrev=7 2>/dev/null || echo unknown)

# -lineinfo maps SASS back to source lines in Nsight Compute.
NVCCFLAGS := -O3 -std=c++17 -arch=$(SM) -lineinfo -Iharness -Isgemm -DGIT_SHA=\"$(GIT_SHA)\"
# NVML's stub library lets the link succeed without a driver (CI); at run time the driver's
# libnvidia-ml.so.1 is loaded instead.
LDLIBS    := -L$(CUDA_HOME)/lib64/stubs -lnvidia-ml -lcublas

SGEMM_DEPS := $(wildcard sgemm/*.cuh sgemm/driver/*.cuh sgemm/[0-9]*/kernel.cuh harness/*.cuh)

.PHONY: all clean FORCE
all: $(BIN)/sgemm

$(BIN)/sgemm: sgemm/driver/main.cu $(SGEMM_DEPS) $(BIN)/.git-sha | $(BIN)
	$(NVCC) $(NVCCFLAGS) $< -o $@ $(LDLIBS)

$(BIN)/.git-sha: FORCE | $(BIN)
	@echo '$(GIT_SHA)' | cmp -s - $@ || echo '$(GIT_SHA)' > $@

$(BIN):
	mkdir -p $@

clean:
	rm -rf $(BIN)

FORCE:
