#/usr/local/bin/cmake
##
## ./build_visit generated host.cmake
## created: Fri Nov  1 15:03:16 EDT 2019

## system: Linux whoopingcough 4.15.0-65-generic #74~16.04.1-Ubuntu SMP Wed Sep 18 09:51:44 UTC 2019 x86_64 x86_64 x86_64 GNU/Linux
## by: dpn

##
## Setup VISITHOME & VISITARCH variables.
##

SET(VISITHOME /mnt/apps/visit/3.0RC/TP/third_party)
SET(VISITARCH linux-x86_64_gcc-5.4)

## Compiler flags.
##
VISIT_OPTION_DEFAULT(VISIT_C_COMPILER /usr/bin/gcc TYPE FILEPATH)
VISIT_OPTION_DEFAULT(VISIT_CXX_COMPILER /usr/bin/g++ TYPE FILEPATH)
VISIT_OPTION_DEFAULT(VISIT_FORTRAN_COMPILER no TYPE FILEPATH)
VISIT_OPTION_DEFAULT(VISIT_C_FLAGS " -m64 -fPIC" TYPE STRING)
VISIT_OPTION_DEFAULT(VISIT_CXX_FLAGS " -m64 -fPIC" TYPE STRING)

##
## Parallel Build Setup.
##
VISIT_OPTION_DEFAULT(VISIT_PARALLEL ON TYPE BOOL)
## (configured w/ mpi compiler wrapper)
VISIT_OPTION_DEFAULT(VISIT_MPI_COMPILER /usr/bin/mpicc TYPE FILEPATH)

##
## VisIt Thread Option
##
VISIT_OPTION_DEFAULT(VISIT_THREAD OFF TYPE BOOL)

##############################################################
##
## Database reader plugin support libraries
##
## The HDF4, HDF5 and NetCDF libraries must be first so that
## their libdeps are defined for any plugins that need them.
##
## For libraries with LIBDEP settings, order matters.
## Libraries with LIBDEP settings that depend on other
## Library's LIBDEP settings must come after them.
##############################################################
##

##
## OPENSSL 
##
VISIT_OPTION_DEFAULT(VISIT_OPENSSL_DIR ${VISITHOME}/openssl/1.0.2j/${VISITARCH})

##
## Python
##
VISIT_OPTION_DEFAULT(VISIT_PYTHON_DIR ${VISITHOME}/python/2.7.14/${VISITARCH})

##
## Qt
##
SETUP_APP_VERSION(QT 5.10.1)
VISIT_OPTION_DEFAULT(VISIT_QT_DIR ${VISITHOME}/qt/${QT_VERSION}/${VISITARCH})

##
## QWT
##
SETUP_APP_VERSION(QWT 6.1.2)
VISIT_OPTION_DEFAULT(VISIT_QWT_DIR ${VISITHOME}/qwt/${QWT_VERSION}/${VISITARCH})

##
## VTK
##
SETUP_APP_VERSION(VTK 8.1.0)
VISIT_OPTION_DEFAULT(VISIT_VTK_DIR ${VISITHOME}/vtk/${VTK_VERSION}/${VISITARCH})

##
## SZIP
##
VISIT_OPTION_DEFAULT(VISIT_SZIP_DIR ${VISITHOME}/szip/2.1/${VISITARCH})

##
## HDF5
##
VISIT_OPTION_DEFAULT(VISIT_HDF5_DIR ${VISITHOME}/hdf5/1.8.14/${VISITARCH})
VISIT_OPTION_DEFAULT(VISIT_HDF5_LIBDEP ${VISITHOME}/szip/2.1/${VISITARCH}/lib sz /usr/lib/x86_64-linux-gnu z TYPE STRING)

##
## Conduit
##
SETUP_APP_VERSION(ADIOS2 2.5.0)
VISIT_OPTION_DEFAULT(VISIT_ADIOS2_DIR ${VISITHOME}/adios2-ser/${ADIOS2_VERSION}/${VISITARCH})
## (configured w/ mpi compiler wrapper)
VISIT_OPTION_DEFAULT(VISIT_ADIOS2_PAR_DIR ${VISITHOME}/adios2-par/${ADIOS2_VERSION}/${VISITARCH})

##
## Conduit
##
VISIT_OPTION_DEFAULT(VISIT_CONDUIT_DIR ${VISITHOME}/conduit/v0.4.0/${VISITARCH})
VISIT_OPTION_DEFAULT(VISIT_CONDUIT_LIBDEP HDF5_LIBRARY_DIR hdf5 ${VISIT_HDF5_LIBDEP} TYPE STRING)

##
## MFEM 
##

VISIT_OPTION_DEFAULT(VISIT_MFEM_DIR ${VISITHOME}/mfem/3.4/${VISITARCH})
VISIT_OPTION_DEFAULT(VISIT_MFEM_INCDEP CONDUIT_INCLUDE_DIR TYPE STRING)
VISIT_OPTION_DEFAULT(VISIT_MFEM_LIBDEP ${VISIT_CONDUIT_LIBDEP} /usr/lib/x86_64-linux-gnu z TYPE STRING)
##

##
## NetCDF
##
VISIT_OPTION_DEFAULT(VISIT_NETCDF_DIR ${VISITHOME}/netcdf/4.1.1/${VISITARCH})
VISIT_OPTION_DEFAULT(VISIT_NETCDF_LIBDEP HDF5_LIBRARY_DIR hdf5_hl HDF5_LIBRARY_DIR hdf5 ${VISIT_HDF5_LIBDEP} TYPE STRING)

##
## Silo
##
VISIT_OPTION_DEFAULT(VISIT_SILO_DIR ${VISITHOME}/silo/4.10.2/${VISITARCH})
VISIT_OPTION_DEFAULT(VISIT_SILO_LIBDEP HDF5_LIBRARY_DIR hdf5 ${VISIT_HDF5_LIBDEP} /usr/lib/x86_64-linux-gnu z TYPE STRING)

