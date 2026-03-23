cmake_minimum_required(VERSION 3.20)

project(milestone06_cmake LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(MPI REQUIRED)
find_package(Kokkos REQUIRED)

set(PROJECT_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../..")

add_library(lbm_m6
    "${PROJECT_ROOT}/src/lbm_d2q9.cpp"
    "${PROJECT_ROOT}/src/lbm_d2q9_mpi.cpp"
)

target_include_directories(lbm_m6
    PUBLIC
        "${PROJECT_ROOT}/src"
)

target_compile_definitions(lbm_m6
    PUBLIC
        WITH_MPI
)

target_link_libraries(lbm_m6
    PUBLIC
        Kokkos::kokkos
        MPI::MPI_CXX
)

add_executable(milestone06
    "${PROJECT_ROOT}/executables/milestone06/main.cpp"
)

target_link_libraries(milestone06
    PRIVATE
        lbm_m6
)

set_target_properties(milestone06 PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/executables/milestone06"
)
