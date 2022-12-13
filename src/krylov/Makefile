# Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

include ../config/Makefile.config

CINCLUDES = ${INCLUDES} ${MPIINCLUDE}
 
C_COMPILE_FLAGS = \
 -I..\
 -I$(srcdir)/..\
 -I$(srcdir)/../blas\
 -I$(srcdir)/../lapack\
 -I$(srcdir)/../multivector\
 -I$(srcdir)/../utilities\
 ${CINCLUDES}

HEADERS =\
 NALU_HYPRE_krylov.h\
 krylov.h\
 bicgstab.h\
 cgnr.h\
 gmres.h\
 cogmres.h\
 flexgmres.h\
 lgmres.h\
 pcg.h\
 NALU_HYPRE_lobpcg.h\
 NALU_HYPRE_MatvecFunctions.h\
 lobpcg.h

FILES =\
 bicgstab.c\
 cgnr.c\
 gmres.c\
 cogmres.c\
 flexgmres.c\
 lgmres.c\
 NALU_HYPRE_bicgstab.c\
 NALU_HYPRE_cgnr.c\
 NALU_HYPRE_gmres.c\
 NALU_HYPRE_cogmres.c\
 NALU_HYPRE_lgmres.c\
 NALU_HYPRE_flexgmres.c\
 NALU_HYPRE_pcg.c\
 pcg.c\
 NALU_HYPRE_lobpcg.c\
 lobpcg.c

OBJS = ${FILES:.c=.o}

SONAME = libNALU_HYPRE_krylov-${NALU_HYPRE_RELEASE_VERSION}${NALU_HYPRE_LIB_SUFFIX}

##################################################################
# Targets
##################################################################

all: libNALU_HYPRE_krylov${NALU_HYPRE_LIB_SUFFIX}
	cp -fR $(srcdir)/NALU_HYPRE_*.h $(NALU_HYPRE_BUILD_DIR)/include
	cp -fR $(srcdir)/krylov.h $(NALU_HYPRE_BUILD_DIR)/include
	cp -fR $(srcdir)/lobpcg.h $(NALU_HYPRE_BUILD_DIR)/include
#	cp -fR libNALU_HYPRE* $(NALU_HYPRE_BUILD_DIR)/lib

install: all
	cp -fR $(srcdir)/NALU_HYPRE_*.h $(NALU_HYPRE_INC_INSTALL)
	cp -fR $(srcdir)/krylov.h $(NALU_HYPRE_INC_INSTALL)
	cp -fR $(srcdir)/lobpcg.h $(NALU_HYPRE_INC_INSTALL)
#	cp -fR libNALU_HYPRE* $(NALU_HYPRE_LIB_INSTALL)

clean:
	rm -f *.o libNALU_HYPRE*
	rm -rf pchdir tca.map *inslog*

distclean: clean

##################################################################
# Rules
##################################################################

libNALU_HYPRE_krylov.a: ${OBJS}
	@echo  "Building $@ ... "
	${AR} $@ ${OBJS}
	${RANLIB} $@

libNALU_HYPRE_krylov.so libNALU_HYPRE_krylov.dylib: ${OBJS}
	@echo  "Building $@ ... "
	${BUILD_CC_SHARED} -o ${SONAME} ${OBJS} ${SHARED_SET_SONAME}${SONAME}
	ln -s -f ${SONAME} $@

${OBJS}: ${HEADERS}
