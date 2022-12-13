# Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

include ../../config/Makefile.config

CINCLUDES = ${INCLUDES} ${MPIINCLUDE}

C_COMPILE_FLAGS = \
 -I../..\
 -I$(srcdir)\
 -I$(srcdir)/../..\
 -I$(srcdir)/../../blas\
 -I$(srcdir)/../../lapack\
 -I$(srcdir)/../../utilities\
 -I$(srcdir)/../../distributed_matrix\
 ${CINCLUDES}

HEADERS =\
 Common.h\
 ConjGrad.h\
 DiagScale.h\
 Hash.h\
 hypre_ParaSails.h\
 LoadBal.h\
 Matrix.h\
 Mem.h\
 Numbering.h\
 OrderStat.h\
 ParaSails.h\
 PrunedRows.h\
 RowPatt.h\
 StoredRows.h

FILES =\
 ConjGrad.c\
 DiagScale.c\
 FGmres.c\
 Hash.c\
 hypre_ParaSails.c\
 LoadBal.c\
 Matrix.c\
 Mem.c\
 Numbering.c\
 OrderStat.c\
 ParaSails.c\
 PrunedRows.c\
 RowPatt.c\
 StoredRows.c

OBJS = ${FILES:.c=.o}

SONAME = libNALU_HYPRE_ParaSails-${NALU_HYPRE_RELEASE_VERSION}${NALU_HYPRE_LIB_SUFFIX}

##################################################################
# Targets
##################################################################

all: libNALU_HYPRE_ParaSails${NALU_HYPRE_LIB_SUFFIX}
#	cp -fR libHYPRE* $(NALU_HYPRE_BUILD_DIR)/lib

install: libNALU_HYPRE_ParaSails${NALU_HYPRE_LIB_SUFFIX}
#	cp -fR libHYPRE* $(NALU_HYPRE_LIB_INSTALL)

clean:
	rm -f *.o libHYPRE*
	rm -rf pchdir tca.map *inslog*

distclean: clean

##################################################################
# Rules
##################################################################

libNALU_HYPRE_ParaSails.a: ${OBJS}
	@echo  "Building $@ ... "
	${AR} $@ ${OBJS}
	${RANLIB} $@

libNALU_HYPRE_ParaSails.so libNALU_HYPRE_ParaSails.dylib: ${OBJS}
	@echo  "Building $@ ... "
	${BUILD_CC_SHARED} -o ${SONAME} ${OBJS} ${SHARED_SET_SONAME}${SONAME}
	ln -s -f ${SONAME} $@

${OBJS}: ${HEADERS}
