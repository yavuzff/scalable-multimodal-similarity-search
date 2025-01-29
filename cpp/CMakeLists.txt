cmake_minimum_required(VERSION 3.29)
project(scalable-multimodal-similarity-search)

set(CMAKE_CXX_STANDARD 20)

# Release adds -O3 and -DNDEBUG, Debug adds -g
# set(CMAKE_BUILD_TYPE Debug)
# set(CMAKE_BUILD_TYPE Release)

# Set all warning flags for the compiler.
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic")
# Add ASan, UBSan (not MSan: -fsanitize=memory not supported for mac clang and ubuntu(docker) compiler?) Do not add these when looking for runtime performance.
# set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} -fsanitize=address -fsanitize=undefined")
# can use LeakSanitiser as well if using ubuntu through docker:
# set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} -fsanitize=leak")

# release flags
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} -O3 -DNDEBUG")
#set(CMAKE_CXX_FLAGS_RELEASE  "${CMAKE_CXX_FLAGS} -flto -fomit-frame-pointer -march=native")
#set(CMAKE_CXX_FLAGS_RELEASE  "${CMAKE_CXX_FLAGS} -flto")
# compiler warnings
#set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic")

# -pg flag for gprof profiling, -DNDEBUG for disabling assertions (which get enabled by -pg?)
#set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} -pg -DNDEBUG")

# for std::execution::par_unseq in macos
# set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -fexperimental-library")

# for openmp #pragma
#set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -fopenmp")

# for compilation to target the native architecture - setting this leads to Valgrind Illegal Instruction.
# set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -march=native")

# path for the eigen library
set(EIGEN_PATH /usr/local/include/eigen3)

# library for simple exact knn search
add_library(simple-exact STATIC
        index/simple-knn/ExactKNNIndex.cpp
        index/DistanceMetrics.cpp
)
set_target_properties(simple-exact PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(simple-exact PRIVATE ${EIGEN_PATH})

# library for multi exact search
add_library(multi-index STATIC
        index/utils.cpp
        index/DistanceMetrics.cpp
        index/AbstractMultiIndex.cpp
        index/ExactMultiIndex.cpp
        index/MultiHNSW.cpp
        index/MultiHNSWBuilder.cpp
)
set_target_properties(multi-index PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(multi-index PUBLIC ${EIGEN_PATH})

# unit testing with Catch2
find_package(Catch2 3 REQUIRED)
# These tests can use the Catch2-provided main
add_executable(tests tests/test_metrics.cpp
        tests/test_utils.cpp
        tests/test_exact.cpp
        tests/test_multi_hnsw.cpp)
target_link_libraries(tests PRIVATE Catch2::Catch2WithMain multi-index)

# performance testing - uses cnpy and z libraries
set(CNPY_PATH /usr/local/lib/libcnpy.a) # this is the default location for macos and ubuntu
add_executable(performance tests/test_performance.cpp)
target_link_libraries(performance PRIVATE Catch2::Catch2WithMain multi-index ${CNPY_PATH} z)

# benchmarking different distance calculation vectorisation implementations
set(CNPY_PATH /usr/local/lib/libcnpy.a) # this is the default location for macos and ubuntu
add_executable(compare-dist tests/benchmark_dist_calculations.cpp)
# add compiled libraries to link against
target_link_libraries(compare-dist PRIVATE Catch2::Catch2WithMain multi-index ${CNPY_PATH} z )
# include Eigen headers
target_include_directories(compare-dist PRIVATE ${EIGEN_PATH})

# create executable with target main and link to library:
add_executable(main main.cpp)
target_link_libraries(main PRIVATE simple-exact multi-index)

# Find pybind11 installed with conda
find_package(pybind11 REQUIRED)
pybind11_add_module(multimodal_index pybinding/binder.cpp)
target_link_libraries(multimodal_index PRIVATE multi-index)
