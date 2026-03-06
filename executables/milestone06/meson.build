if mpi.found()
    fs = import('fs')

    executable(
        'milestone06',
        'main.cpp',
        include_directories : [lib_incdirs],
        link_with : [lib],
        dependencies : [kokkos, mpi],
        install : true
    )

    fs.copyfile('plot.py')
    fs.copyfile('measure_scaling.py')
else
    message('MPI not found: skipping milestone06 target')
endif
