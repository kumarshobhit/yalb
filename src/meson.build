lib_sources = [  # All source files (excluding headers)
    'hello.cpp',
    'lbm_d2q9.cpp'
]

lib = static_library(
    'lib',  # Unique name
    lib_sources,
    dependencies : [kokkos, mpi]
)

lib_incdirs = include_directories('.')
