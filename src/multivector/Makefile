# Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

include ../config/Makefile.config

CINCLUDES = ${INCLUDES} ${MPIINCLUDE}

C_COMPILE_FLAGS = \
 -I.\
 -I$(srcdir)/..\
 -I$(srcdir)/../utilities\
 ${CINCLUDES}

HEADERS =\
 multivector.h\
 temp_multivector.h

FILES =\
 multivector.c\
 temp_multivector.c

OBJS = ${FILES:.c=.o}

SONAME = libNALU_HYPRE_multivector-${NALU_HYPRE_RELEASE_VERSION}${NALU_HYPRE_LIB_SUFFIX}

##################################################################
# Targets
##################################################################

all: libNALU_HYPRE_multivector${NALU_HYPRE_LIB_SUFFIX}
	cp -fR *.h $(NALU_HYPRE_BUILD_DIR)/include
#	cp -fR libNALU_HYPRE* $(NALU_HYPRE_BUILD_DIR)/lib

install: libNALU_HYPRE_multivector${NALU_HYPRE_LIB_SUFFIX}
	cp -fR *.h $(NALU_HYPRE_INC_INSTALL)
#	cp -fR libNALU_HYPRE* $(NALU_HYPRE_LIB_INSTALL)

clean:
	rm -f *.o libNALU_HYPRE*
	rm -rf pchdir tca.map *inslog*

distclean: clean

##################################################################
# Rules
##################################################################

libNALU_HYPRE_multivector.a: ${OBJS}
	@echo  "Building $@ ... "
	${AR} $@ ${OBJS}
	${RANLIB} $@

libNALU_HYPRE_multivector.so libNALU_HYPRE_multivector.dylib: ${OBJS}
	@echo  "Building $@ ... "
	${BUILD_CC_SHARED} -o ${SONAME} ${OBJS} ${SHARED_SET_SONAME}${SONAME}
	ln -s -f ${SONAME} $@

${OBJS}: ${HEADERS}
