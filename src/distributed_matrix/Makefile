# Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

include ../config/Makefile.config

CINCLUDES = ${INCLUDES} ${MPIINCLUDE}

C_COMPILE_FLAGS = \
 -I..\
 -I$(srcdir)/..\
 -I$(srcdir)/../utilities\
 -I$(srcdir)/../parcsr_mv\
 -I$(srcdir)/../seq_mv\
 ${CINCLUDES}

EXTRA_HEADERS = internal_protos.h

HEADERS =\
 distributed_matrix.h\
 NALU_HYPRE_distributed_matrix_mv.h\
 NALU_HYPRE_distributed_matrix_protos.h\
 NALU_HYPRE_distributed_matrix_types.h

FILES =\
 distributed_matrix.c\
 NALU_HYPRE_distributed_matrix.c\
 distributed_matrix_ISIS.c\
 distributed_matrix_PETSc.c\
 distributed_matrix_parcsr.c

OBJS = ${FILES:.c=.o}

SONAME = libNALU_HYPRE_DistributedMatrix-${NALU_HYPRE_RELEASE_VERSION}${NALU_HYPRE_LIB_SUFFIX}

##################################################################
# Targets
##################################################################

all: libNALU_HYPRE_DistributedMatrix${NALU_HYPRE_LIB_SUFFIX}
	cp -fR $(srcdir)/distributed_matrix.h $(NALU_HYPRE_BUILD_DIR)/include
#	cp -fR libHYPRE* $(NALU_HYPRE_BUILD_DIR)/lib

install: libNALU_HYPRE_DistributedMatrix${NALU_HYPRE_LIB_SUFFIX}
	cp -fR $(srcdir)/distributed_matrix.h $(NALU_HYPRE_INC_INSTALL)
#	cp -fR libHYPRE* $(NALU_HYPRE_LIB_INSTALL)

clean:
	rm -f *.o libHYPRE*
	rm -rf pchdir tca.map *inslog*

distclean: clean

##################################################################
# Rules
##################################################################

libNALU_HYPRE_DistributedMatrix.a: ${OBJS}
	@echo  "Building $@ ... "
	${AR} $@ ${OBJS}
	${RANLIB} $@

libNALU_HYPRE_DistributedMatrix.so libNALU_HYPRE_DistributedMatrix.dylib: ${OBJS}
	@echo  "Building $@ ... "
	${BUILD_CC_SHARED} -o ${SONAME} ${OBJS} ${SHARED_SET_SONAME}${SONAME}
	ln -s -f ${SONAME} $@

${OBJS}: ${HEADERS}
