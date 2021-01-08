TARGET 		= fractal

CC			= g++
#NVCC 		= nvcc
LINKER		= g++

#CUDA_INSTALL_PATH ?= /usr/local/cuda

#INCD 		= -I$(CUDA_INSTALL_PATH)/include
#LIBS 		= -L$(CUDA_INSTALL_PATH)/lib -lcudart

CFLAGS		= -O3 -std=c++11 -march=native -mtune=native -fopenmp -I/opt/local/include -I./src
#NFLAGS 		= -O3 -arch=compute_32 -std=c++11
LFLAGS		= -Wall -I. -L/opt/local/lib -lm -fopenmp -lsfml-graphics -lsfml-window -lsfml-system

SRCDIR		= src
OBJDIR		= obj
BINDIR		= bin

SOURCES		=	src/main.cpp \
				src/Mandelbrot.cpp \
				src/Color/ColorSmooth/ColorSmooth.cpp \
				src/Color/Colorblue/Colorblue.cpp \
				src/Color/Colorgreen/Colorgreen.cpp \
				src/Color/Colorgrey/Colorgrey.cpp \
				src/Convergence/Convergence.cpp \
				src/Convergence/double/Convergence_dp_x86.cpp \
				src/Convergence/doublem/Convergence_dpm_x86.cpp \
				src/Convergence/m256d/Convergence_m256d_x86.cpp \
				src/Convergence/m256dm/Convergence_m256dm_x86.cpp \
				src/Convergence/doublej/Convergence_dpj_x86.cpp \
				src/Convergence/m128/Convergence_m128_x86.cpp \
				src/Convergence/m128j/Convergence_m128j_x86.cpp \
				src/Convergence/m256_float/Convergence_m256_float_x86.cpp \
				src/Convergence/m256j_float/Convergence_m256j_float_x86.cpp \
				src/Convergence/m256dj/Convergence_m256dj_x86.cpp \
				src/Convergence/doublen/Convergence_dpn_x86.cpp \
				src/Convergence/double_n/Convergence_dp_n_x86.cpp \
				src/Convergence/doublebs/Convergence_dpbs_x86.cpp \
				src/Convergence/doublemr/Convergence_dpmr_x86.cpp \
				src/Convergence/doublemme/Convergence_dpmme_x86.cpp \
				src/Utils/FileHandler.cpp \
				src/Utils/Settings.cpp \
				src/Utils/StringUtils.cpp \
				src/Utils/Utils.cpp

#SOURCES_CU 	:= $(wildcard $(SRCDIR)/*.cu)
INCLUDES	:= $(wildcard $(SRCDIR)/*.h)
#INCLUDES_CU 	:= $(wildcard $(SRCDIR)/*.cuh)
OBJECTS		:= $(SOURCES:$(SRCDIR)/%.cpp=$(OBJDIR)/%.o)
#OBJECTS_CU 	:= $(SOURCES_CU:$(SRCDIR)/%.cu=$(OBJDIR)/%.cuo)
rm			= rm -f

all: $(BINDIR)/$(TARGET)

$(BINDIR)/$(TARGET): $(OBJECTS_CU) $(OBJECTS)
	$(LINKER) -o $@ $(OBJECTS_CU) $(OBJECTS) $(INCD) $(LIBS) $(LFLAGS)

$(OBJECTS): $(OBJDIR)/%.o : $(SRCDIR)/%.cpp
	@$(CC) $(CFLAGS) $(INCD) -c $< -o $@

#$(OBJECTS_CU): $(OBJDIR)/%.cuo : $(SRCDIR)/%.cu
#	@$(NVCC) $(NFLAGS) $(INCD) -c $< -o $@

.PHONY: clean
clean:
	@$(rm) $(OBJECTS_CU) $(OBJECTS)

.PHONY: remove
remove: clean
	@$(rm) $(BINDIR)/$(TARGET)
