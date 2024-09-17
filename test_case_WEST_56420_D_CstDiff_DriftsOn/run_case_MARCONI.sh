#!/bin/bash

# To have a bit of verbose from the shell
set -vx
 
# Load modules
module purge
module load env-skl
module load intel/pe-xe-2018--binary \
    intelmpi/2018--binary \
    mkl/2018--binary \
    szip/2.1--gnu--6.1.0 \
    zlib/1.2.8--gnu--6.1.0 \
    hdf5/1.10.4--intelmpi--2018--binary \
    petsc/3.13.3--intelmpi--2018--binary
module load python/3.9.4

# GKS/GR libraries options
export GLI_HOME='/marconi/home/userexternal/ptamain0/libs/libs_gateway/gks'
export GRSOFT_DEVICE=62

# To increase the stack size
ulimit -s unlimited


# Setting-up directories and files
##################################

# Location of executable file, mesh files, parameter files...
LOCDIR=`pwd`

# Executable
EXECDIR=<DIRECTORY WHERE EXECUTABLE FILE IS LOCATED> # !!! CHANGE HERE!!!
EXECNAME=<EXECUTABLE FILE NAME>  # !!! CHANGE HERE!!!

# Simulation directories
SIMDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )" # Path of the directory containing this script, no matter where it is called from
RUNDIR=$SIMDIR/run_dir
PARAMDIR=$SIMDIR
MESHDIR=$SIMDIR/mesh  # !!! CHANGE HERE!!! (if necessary)
MESHNAME=mesh # !!! CHANGE HERE!!! (if necessary)

# Database directories
S3XDIR=<PATH TO LOCAL SOLEDGE3X GIT REPO> # !!! CHANGE HERE!!!
STYXDIR=<PATH TO LOCAL STYX GIT REPO>  # !!! CHANGE HERE!!!
DBDIR_S3X=$S3XDIR/database
DBDIR_STYX=$STYXDIR/Database

# Create and clean execution directory
mkdir $RUNDIR
cd $RUNDIR
rm -rf *.dat
rm -rf plasma_*.h5
rm -rf emergencySave*.h5
ls
cd $LOCDIR

# Copy exec file to the execution dir
cp $EXECDIR/$EXECNAME $RUNDIR/.


# ------ SOLEDGE3X FILES ------

# SOLEDGE3X input files
cp $MESHDIR/$MESHNAME.h5 $RUNDIR/mesh.h5
cp $MESHDIR/meshEIRENE.h5 $RUNDIR/.
cp $MESHDIR/soledge3x.* $RUNDIR/.
#cp $MESHDIR/diffusion.h5 $RUNDIR/diffusion.h5
cp $PARAMDIR/param_raptorX.txt $RUNDIR/.
#cp $PARAMDIR/param_geom.txt $RUNDIR/.
#cp $PARAMDIR/source_raptorX.txt $RUNDIR/.

# SOLEDGE3X database for AM rates between ionized species
mkdir -v $RUNDIR/AMSplines
rsync -tv $DBDIR_S3X/* $RUNDIR/AMSplines/.


# ------ EIRENE FILES ------

# STYX-EIRENE input files
cp $PARAMDIR/eirene_coupling.txt $RUNDIR/.
cp $PARAMDIR/input_vacuum_cleaner.txt $RUNDIR/.
cp $PARAMDIR/cx_setup $RUNDIR/.

# EIRENE database for AM rates for neutrals
cd $RUNDIR
# amjuel, h2vibr, hydel, methane, spectral databases (remove .tex extension and capitalize)
rsync -tv $DBDIR_STYX/AMdata/amjuel.tex ./AMJUEL
rsync -tv $DBDIR_STYX/AMdata/h2vibr.tex ./H2VIBR
rsync -tv $DBDIR_STYX/AMdata/hydhel.tex ./HYDHEL
rsync -tv $DBDIR_STYX/AMdata/methane.tex ./METHANE
rsync -tv $DBDIR_STYX/AMdata/spectral.tex ./SPECTRAL
rsync -rtv $DBDIR_STYX/AMdata/Adas_Eirene_2010/* ./ADAS # ADAS database folder
rsync -rtv $DBDIR_STYX/Surfacedata/TRIM . # TRIM databases
rsync -tv $DBDIR_STYX/Surfacedata/SPUTER . # Sputtering database file
rsync -tv $DBDIR_STYX/PHOTON . # PHOTON database file
rsync -tv $DBDIR_STYX/POLARI . 

# EIRENE atomic and molecular physics models templates
rsync -tv $STYXDIR/input-files/atomic-and-molecular-models/* .

cd $LOCDIR

# Move eventually to execution dir
cd $RUNDIR


# Optimization options
######################

# MPI tasks options and binding
export SRUN_CPU_BIND_OPTIONS="--cpu_bind=verbose --distribution=block:block"

# OpenMP threads options and binding
export OMP_SCHEDULE="GUIDED"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export AGMG_NUM_THREAD=$SLURM_CPUS_PER_TASK
export OMP_NESTED="FALSE" # Old OpenMP norm
export OMP_MAX_ACTIVE_LEVELS=1 # New OpenMP norm
#export KMP_AFFINITY="verbose,granularity=core,compact,1,0"
export KMP_AFFINITY="granularity=core,compact,1,0"
export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK # Needed for new version of SLURM in which srun does not read the SLURM_CPUS_PER_TASK environment variable

# Forcing MKL to run with only 1 thread (for thread compatibility with PASTIX)
export MKL_DOMAIN_NUM_THREADS="MKL_BLAS=1"
export MKL_NUM_THREADS="1"
export MKL_DYNAMIC="FALSE"
export MKL_ENABLE_INSTRUCTIONS=AVX2

# PETSC solver options
# For direct solver
PETSC_OPTIONS="-ksp_type preonly -pc_type lu"
# For iterative solver
#PETSC_OPTIONS="-ksp_type bcgs -pc_type gamg"


# Launch the job
################
time srun --mpi=pmi2 -K1 $SRUN_CPU_BIND_OPTIONS $EXECNAME $PETSC_OPTIONS


# Finalize
##########

rm -rf Plasma
mkdir Plasma/
cp plasma_*.h5  Plasma/

cd $LOCDIR
