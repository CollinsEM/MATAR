module purge
### Load environment modules here
module load cmake
module load clang/13.0.0
module load rocm/4.3.1
module list

export basedir=`pwd`
export srcdir=${basedir}
export builddir=${basedir}/build
export installdir=${basedir}/src/install
export KOKKOS_SOURCE_DIR=${srcdir}/src/kokkos
export KOKKOS_BUILD_DIR=${builddir}/kokkos
export KOKKOS_INSTALL_DIR=${installdir}/kokkos

export MATAR_SOURCE_DIR=${srcdir}
export MATAR_BUILD_DIR=${builddir}

#export OMP_NUM_THREADS=4
#export OMP_PROC_BIND=true
#export OMP_PROC_BIND=spread
#export OMP_NUM_THREADS=4
