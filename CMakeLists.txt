cmake_minimum_required(VERSION 3.7)

project(mandelbrot)

SET(CMAKE_BUILD_TYPE Release)

SET (CMAKE_C_COMPILER   /usr/bin/gcc)
SET (CMAKE_CXX_COMPILER /usr/bin/g++)
SET (CMAKE_CXX_STANDARD 14)
SET (CMAKE_CXX_FLAGS "")
SET (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -std=c++11 -march=native -mtune=native -fopenmp -I/usr/local/include")
SET (CMAKE_EXE_LINKER_FLAGS "-I. -L/usr/include/SFML -lm -fopenmp -lsfml-graphics -lsfml-window -lsfml-system")

# Generate the source files list
file (GLOB_RECURSE source_files src/*)

# Add includes
include_directories (src)

# Object
add_library (dec-obj OBJECT ${source_files})

add_executable(mandelbrot $<TARGET_OBJECTS:dec-obj>)
