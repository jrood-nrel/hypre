# Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
# NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

include ../config/Makefile.config

CINCLUDES = ${INCLUDES} ${MPIINCLUDE}

C_COMPILE_FLAGS =\
 -I..\
 -I$(srcdir)/..\
 -I$(srcdir)/../utilities\
 -I$(srcdir)/../struct_mv\
 -I$(srcdir)/../seq_mv\
 -I$(srcdir)/../parcsr_mv\
 -I$(srcdir)/../parcsr_ls\
 ${CINCLUDES}

HEADERS =\
 aux_parcsr_matrix.h\
 aux_par_vector.h\
 NALU_HYPRE_IJ_mv.h\
 _nalu_hypre_IJ_mv.h\
 IJ_matrix.h\
 IJ_vector.h

EXTRA_FILES =\
 IJMatrix_isis.c\
 IJMatrix_petsc.c

FILES =\
 aux_parcsr_matrix.c\
 aux_par_vector.c\
 F90_NALU_HYPRE_IJMatrix.c\
 F90_NALU_HYPRE_IJVector.c\
 F90_IJMatrix.c\
 NALU_HYPRE_IJMatrix.c\
 NALU_HYPRE_IJVector.c\
 IJ_assumed_part.c\
 IJMatrix.c\
 IJMatrix_parcsr.c\
 IJVector.c\
 IJVector_parcsr.c

CUFILES =\
 IJMatrix_parcsr_device.c\
 IJVector_parcsr_device.c

COBJS = ${FILES:.c=.o}
CUOBJS = ${CUFILES:.c=.obj}
OBJS = ${COBJS} ${CUOBJS}

SONAME = libNALU_HYPRE_IJ_mv-${NALU_HYPRE_RELEASE_VERSION}${NALU_HYPRE_LIB_SUFFIX}

##################################################################
# Targets
##################################################################

all: libNALU_HYPRE_IJ_mv${NALU_HYPRE_LIB_SUFFIX}
	cp -fR $(srcdir)/NALU_HYPRE_*.h $(NALU_HYPRE_BUILD_DIR)/include
	cp -fR $(srcdir)/_nalu_hypre_IJ_mv.h $(NALU_HYPRE_BUILD_DIR)/include
#	cp -fR libHYPRE* $(NALU_HYPRE_BUILD_DIR)/lib

install: libNALU_HYPRE_IJ_mv${NALU_HYPRE_LIB_SUFFIX}
	cp -fR $(srcdir)/NALU_HYPRE_*.h $(NALU_HYPRE_INC_INSTALL)
	cp -fR $(srcdir)/_nalu_hypre_IJ_mv.h $(NALU_HYPRE_INC_INSTALL)
#	cp -fR libHYPRE* $(NALU_HYPRE_LIB_INSTALL)

clean:
	rm -f *.o *.obj libHYPRE*
	rm -rf pchdir tca.map *inslog*

distclean: clean

##################################################################
# Rules
##################################################################

libNALU_HYPRE_IJ_mv.a: ${OBJS}
	@echo  "Building $@ ... "
	${AR} $@ ${OBJS}
	${RANLIB} $@

libNALU_HYPRE_IJ_mv.so libNALU_HYPRE_IJ_mv.dylib: ${OBJS}
	@echo  "Building $@ ... "
	${BUILD_CC_SHARED} -o ${SONAME} ${OBJS} ${SHARED_SET_SONAME}${SONAME}
	ln -s -f ${SONAME} $@

${OBJS}: ${HEADERS}
