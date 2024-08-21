# Set the minimum required CMake version
cmake_minimum_required(VERSION 3.12)

# Set the project name
project(benchmark)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Werror -Wno-error=deprecated-declarations")

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)


set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIBRARIES_DIRS})

find_package(ALPR REQUIRED)
find_package(GFLAGS REQUIRED)
find_package(Boost COMPONENTS system filesystem REQUIRED)

if (NOT GFLAGS_CONFLICT)
    set(GFLAGS_LIB gflags)
endif ()

set(EXTERNAL_LIBRARIES
        ${OpenCV_LIBS}
        ${Boost_LIBRARIES}
        pthread
        ${ALPR_LIBRARIES}
        ${GFLAGS_LIB}
        )


# Add the executable target
add_executable(benchmark main.cpp)
target_link_libraries(benchmark ${EXTERNAL_LIBRARIES})
