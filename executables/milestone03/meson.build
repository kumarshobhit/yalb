fs = import('fs')

executable(
    'milestone03',
    'main.cpp',
    include_directories : [lib_incdirs],
    link_with : [lib],
    dependencies : [kokkos, mpi],
    install : true
)

fs.copyfile('plot.py')
