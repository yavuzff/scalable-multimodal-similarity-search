cmake_minimum_required(VERSION 3.29)
project(CppTest)

set(CMAKE_CXX_STANDARD 20)

# Release adds -O3 and -DNDEBUG, Debug adds -g
set(CMAKE_BUILD_TYPE Debug)

# Set all warning flags for the compiler.
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic")
# Add ASan, UBSan (not MSan: -fsanitize=memory not supported for mac?) Do not add these when looking for runtime performance.
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -fsanitize=undefined")

add_library(exact STATIC
        index/ExactKNNIndex.cpp
        index/ExactKNNIndex.hpp)
set_target_properties(exact PROPERTIES POSITION_INDEPENDENT_CODE ON)

# can deploy add as library, create executable for unit testing and create separate executables

# create executable with target main and link to library:
add_executable(main main.cpp)
target_link_libraries(main PRIVATE exact)


# Find pybind11 installed with conda
find_package(pybind11 REQUIRED)
pybind11_add_module(cppindex pybinding/binder.cpp)
target_link_libraries(cppindex PRIVATE exact)
