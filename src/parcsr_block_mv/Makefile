# Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
# NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

include ../config/Makefile.config

CINCLUDES = ${INCLUDES} ${MPIINCLUDE}

C_COMPILE_FLAGS = \
 -I..\
 -I$(srcdir)\
 -I$(srcdir)/..\
 -I$(srcdir)/../utilities\
 -I$(srcdir)/../multivector\
 -I$(srcdir)/../seq_mv\
 -I$(srcdir)/../parcsr_mv\
 -I$(srcdir)/../parcsr_ls\
 -I$(srcdir)/../IJ_mv\
 -I$(srcdir)/../krylov\
 $(SUPERLU_INCLUDE)\
 $(DSUPERLU_INCLUDE)\
 ${CINCLUDES}

LFLAGS = \
  -L. -lNALU_HYPRE_parcsr_block_mv -L../hypre/lib -lNALU_HYPRE_krylov \
      -lNALU_HYPRE_parcsr_ls -lNALU_HYPRE_ParaSails -lNALU_HYPRE_Euclid \
      -lNALU_HYPRE_DistributedMatrixPilutSolver -lNALU_HYPRE_IJ_mv \
      -lNALU_HYPRE_parcsr_mv -lNALU_HYPRE_seq_mv -lNALU_HYPRE_DistributedMatrix \
      -lNALU_HYPRE_MatrixMatrix -lNALU_HYPRE_utilities -lm
# -L. -lNALU_HYPRE_parcsr_block_mv -L../hypre/lib -lNALU_HYPRE_LSI -lNALU_HYPRE_superlu -lm

HEADERS =\
 csr_block_matrix.h\
 par_csr_block_matrix.h

FILES =\
 csr_block_matrix.c\
 csr_block_matvec.c\
 par_csr_block_matrix.c\
 par_csr_block_matvec.c\
 par_csr_block_comm.c\
 par_csr_block_rap.c\
 par_csr_block_rap_communication.c\
 par_csr_block_interp.c\
 par_csr_block_relax.c\
 par_block_nodal_systems.c

EXTRA_FILES =

DRIVER_FILES =

OBJS = ${FILES:.c=.o}
DRIVER_OBJS = ${DRIVER_FILES:.c=.o}

SONAME = libNALU_HYPRE_parcsr_block_mv-${NALU_HYPRE_RELEASE_VERSION}${NALU_HYPRE_LIB_SUFFIX}

##################################################################
# Targets
##################################################################

all: libNALU_HYPRE_parcsr_block_mv${NALU_HYPRE_LIB_SUFFIX}
	cp -fR $(srcdir)/par_csr_block_matrix.h $(NALU_HYPRE_BUILD_DIR)/include	
	cp -fR $(srcdir)/csr_block_matrix.h $(NALU_HYPRE_BUILD_DIR)/include
#	cp -fR libNALU_HYPRE_* $(NALU_HYPRE_BUILD_DIR)/lib/.

driver: driver.o libNALU_HYPRE_parcsr_block_mv${NALU_HYPRE_LIB_SUFFIX}
	@echo  "Linking" $@ "... "
	${CC} -o driver driver.o ${LFLAGS} 

driver_matvec: driver_matvec.o
	@echo  "Linking" $@ "... "
	${CC} -o driver_matvec driver_matvec.o ${LFLAGS} 

driver_matmul: driver_matmul.o libNALU_HYPRE_parcsr_mv${NALU_HYPRE_LIB_SUFFIX}
	@echo  "Linking" $@ "... "
	${CC} -o driver_matmul driver_matmul.o ${LFLAGS} 

driver_boolmatmul: driver_boolmatmul.o libNALU_HYPRE_parcsr_mv${NALU_HYPRE_LIB_SUFFIX}
	@echo  "Linking" $@ "... "
	${CC} -o driver_boolmatmul driver_boolmatmul.o ${LFLAGS} 

driver_aat: driver_aat.o libNALU_HYPRE_parcsr_mv${NALU_HYPRE_LIB_SUFFIX}
	@echo  "Linking" $@ "... "
	${CC} -o driver_aat driver_aat.o ${LFLAGS} 

driver_boolaat: driver_boolaat.o libNALU_HYPRE_parcsr_mv${NALU_HYPRE_LIB_SUFFIX}
	@echo  "Linking" $@ "... "
	${CC} -o driver_boolaat driver_boolaat.o ${LFLAGS} 

install: libNALU_HYPRE_parcsr_block_mv${NALU_HYPRE_LIB_SUFFIX}
	cp -fR $(srcdir)/par_csr_block_matrix.h $(NALU_HYPRE_INC_INSTALL)/.
	cp -fR $(srcdir)/csr_block_matrix.h $(NALU_HYPRE_INC_INSTALL)/.
#	cp -fR libNALU_HYPRE_* $(NALU_HYPRE_LIB_INSTALL)/.

clean:
	rm -rf *.o libNALU_HYPRE_*
	rm -f driver driver_matvec driver_matmul

distclean: clean

##################################################################
# Rules
##################################################################

libNALU_HYPRE_parcsr_block_mv.a: ${OBJS}
	@echo  "Building $@ ... "
	${AR} $@ ${OBJS}
	${RANLIB} $@

libNALU_HYPRE_parcsr_block_mv.so libNALU_HYPRE_parcsr_block_mv.dylib: ${OBJS}
	@echo  "Building $@ ... "
	${BUILD_CC_SHARED} -o ${SONAME} ${OBJS} ${SHARED_SET_SONAME}${SONAME}
	ln -s -f ${SONAME} $@

${OBJS}: ${HEADERS}

${DRIVER_OBJS}: ${HEADERS}

