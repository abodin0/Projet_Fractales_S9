TARGET 		= fractal

CC			= g++
NVCC 		= nvcc
LINKER		= g++

CUDA_INSTALL_PATH = /usr/local/cuda

INCD 		= -I$(CUDA_INSTALL_PATH)/include
LIBS 		= -L$(CUDA_INSTALL_PATH)/lib64 -lcudart

CFLAGS		= -O3 -std=c++11 -march=native -mtune=native -fopenmp -I/opt/local/include -I./src 
NFLAGS 		= -O3 -std=c++11 
LFLAGS		= -Wall -I. -L/opt/local/lib -lm -fopenmp -lsfml-graphics -lsfml-window -lsfml-system

SRCDIR		= src
OBJDIR		= obj
BINDIR		= bin

SOURCES		=	src/main.cpp \
				src/Mandelbrot.cpp \
				src/Color/ColorSmooth/ColorSmooth.cpp \
				src/Convergence/Convergence.cpp \
				src/Utils/FileHandler.cpp \
				src/Utils/Settings.cpp \
				src/Utils/StringUtils.cpp \
				src/Utils/Utils.cpp \
				src/Convergence/double/Convergence_dp_x86.cpp

SOURCES_CU 	:= src/Convergence/double_gpu/kernel_GPU.cu \
			   src/Convergence/double_gpu/Convergence_GPU.cu \
			   src/Convergence/float_gpu/kernel_GPU_float.cu \
			   src/Convergence/float_gpu/Convergence_GPU_float.cu \
			   src/Convergence/double_gpu_julia/kernel_GPU_julia.cu \
			   src/Convergence/double_gpu_julia/Convergence_GPU_julia.cu \
			   src/Convergence/double_gpu_multibrot/kernel_GPU_multibrot.cu \
			   src/Convergence/double_gpu_multibrot/Convergence_GPU_multibrot.cu \
			   src/Convergence/double_gpu_mme/kernel_GPU_mme.cu \
			   src/Convergence/double_gpu_mme/Convergence_GPU_mme.cu \
			   src/Convergence/double_gpu_ship/kernel_GPU_ship.cu \
			   src/Convergence/double_gpu_ship/Convergence_GPU_ship.cu \
			   src/Convergence/double_gpu_mr/kernel_GPU_mr.cu \
			   src/Convergence/double_gpu_mr/Convergence_GPU_mr.cu 
#			   $(wildcard $(SRCDIR)/*.cu)
INCLUDES	:= $(wildcard $(SRCDIR)/*.h)
INCLUDES_CU := $(wildcard $(SRCDIR)/*.cuh)
OBJECTS		:= $(SOURCES:$(SRCDIR)/%.cpp=$(OBJDIR)/%.o)
OBJECTS_CU 	:= $(SOURCES_CU:$(SRCDIR)/%.cu=$(OBJDIR)/%.cuo)
rm			= rm -f

all: $(BINDIR)/$(TARGET)

$(BINDIR)/$(TARGET): $(OBJECTS_CU) $(OBJECTS)
	$(LINKER) -o $@ $(OBJECTS_CU) $(OBJECTS) $(INCD) $(LIBS) $(LFLAGS)

$(OBJECTS): $(OBJDIR)/%.o : $(SRCDIR)/%.cpp
	$(CC) $(CFLAGS) $(INCD) -c $< -o $@

$(OBJECTS_CU): $(OBJDIR)/%.cuo : $(SRCDIR)/%.cu
	$(NVCC) $(NFLAGS) $(INCD) -c $< -o $@

.PHONY: clean
clean:
	@$(rm) $(OBJECTS_CU) $(OBJECTS)

.PHONY: remove
remove: clean
	@$(rm) $(BINDIR)/$(TARGET)

