# Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
# NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

include ../../config/Makefile.config

CINCLUDES = ${INCLUDES} ${MPIINCLUDE}

C_COMPILE_FLAGS = \
 -I../..\
 -I$(srcdir)\
 -I$(srcdir)/../..\
 -I$(srcdir)/../../utilities\
 -I$(srcdir)/../../parcsr_mv\
 -I$(srcdir)/../../seq_mv\
 -I$(srcdir)/../../distributed_matrix\
 -I$(srcdir)/../../hypre/include\
 ${CINCLUDES}

HEADERS =\
 blas_dh.h\
 euclid_common.h\
 euclid_config.h\
 Euclid_dh.h\
 ExternalRows_dh.h\
 Factor_dh.h\
 getRow_dh.h\
 guards_dh.h\
 Hash_dh.h\
 Hash_i_dh.h\
 ilu_dh.h\
 io_dh.h\
 krylov_dh.h\
 macros_dh.h\
 Mat_dh.h\
 mat_dh_private.h\
 MatGenFD.h\
 Mem_dh.h\
 Numbering_dh.h\
 Parser_dh.h\
 shellSort_dh.h\
 sig_dh.h\
 SortedList_dh.h\
 SortedSet_dh.h\
 SubdomainGraph_dh.h\
 TimeLog_dh.h\
 Timer_dh.h\
 Vec_dh.h

FILES =\
 blas_dh.c\
 Euclid_apply.c\
 Euclid_dh.c\
 ExternalRows_dh.c\
 Factor_dh.c\
 getRow_dh.c\
 globalObjects.c\
 Hash_dh.c\
 Hash_i_dh.c\
 ilu_mpi_bj.c\
 ilu_mpi_pilu.c\
 ilu_seq.c\
 io_dh.c\
 krylov_dh.c\
 Mat_dh.c\
 mat_dh_private.c\
 MatGenFD.c\
 Mem_dh.c\
 Numbering_dh.c\
 Parser_dh.c\
 shellSort_dh.c\
 sig_dh.c\
 SortedList_dh.c\
 SortedSet_dh.c\
 SubdomainGraph_dh.c\
 TimeLog_dh.c\
 Timer_dh.c\
 Vec_dh.c

OBJS = ${FILES:.c=.o}

SONAME = libNALU_HYPRE_Euclid-${NALU_HYPRE_RELEASE_VERSION}${NALU_HYPRE_LIB_SUFFIX}

##################################################################
# Targets
##################################################################

all: libNALU_HYPRE_Euclid${NALU_HYPRE_LIB_SUFFIX}
#	cp -fR libNALU_HYPRE* $(NALU_HYPRE_BUILD_DIR)/lib

install: libNALU_HYPRE_Euclid${NALU_HYPRE_LIB_SUFFIX}
#	cp -fR libNALU_HYPRE* $(NALU_HYPRE_LIB_INSTALL)

clean:
	rm -f *.o libNALU_HYPRE*
	rm -rf pchdir tca.map *inslog*

distclean: clean

##################################################################
# Rules
##################################################################

libNALU_HYPRE_Euclid.a: ${OBJS}
	@echo  "Building $@ ... "
	${AR} $@ ${OBJS}
	${RANLIB} $@

libNALU_HYPRE_Euclid.so libNALU_HYPRE_Euclid.dylib: ${OBJS}
	@echo  "Building $@ ... "
	${BUILD_CC_SHARED} -o ${SONAME} ${OBJS} ${SHARED_SET_SONAME}${SONAME}
	ln -s -f ${SONAME} $@

${OBJS}: ${HEADERS}
