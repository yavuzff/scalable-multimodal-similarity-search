cmake_minimum_required(VERSION 3.29)
project(scalable-multimodal-similarity-search)

set(CMAKE_CXX_STANDARD 20)

# Release adds -O3 and -DNDEBUG, Debug adds -g
set(CMAKE_BUILD_TYPE Debug)

# Set all warning flags for the compiler.
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic")
# Add ASan, UBSan (not MSan: -fsanitize=memory not supported for mac?) Do not add these when looking for runtime performance.
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -fsanitize=undefined")

# library for simple exact knn search
add_library(exact STATIC
        index/ExactKNNIndex.cpp
        index/DistanceMetrics.cpp
)
set_target_properties(exact PROPERTIES POSITION_INDEPENDENT_CODE ON)

#library for multi exact search
add_library(multiExact STATIC
        index/ExactMultiIndex.cpp
        index/DistanceMetrics.cpp
)
set_target_properties(multiExact PROPERTIES POSITION_INDEPENDENT_CODE ON)

# can also create executable for unit testing

# create executable with target main and link to library:
add_executable(main main.cpp)
target_link_libraries(main PRIVATE exact multiExact)

# Find pybind11 installed with conda
find_package(pybind11 REQUIRED)
pybind11_add_module(cppindex pybinding/binder.cpp)
target_link_libraries(cppindex PRIVATE exact)
