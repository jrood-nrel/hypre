# Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

include ../config/Makefile.config

CINCLUDES = ${INCLUDES} ${MPIINCLUDE}

C_COMPILE_FLAGS = \
 -I..\
 -I$(srcdir)\
 -I$(srcdir)/..\
 -I$(srcdir)/../blas\
 -I$(srcdir)/../lapack\
 -I$(srcdir)/../utilities\
 -I$(srcdir)/../seq_mv\
 ${CINCLUDES}

HEADERS =\
 NALU_HYPRE_parcsr_mv.h\
 _nalu_hypre_parcsr_mv.h\
 new_commpkg.h\
 numbers.h\
 par_chord_matrix.h\
 par_csr_communication.h\
 par_csr_matrix.h\
 par_vector.h

FILES =\
 communicationT.c\
 F90_NALU_HYPRE_parcsr_matrix.c\
 F90_NALU_HYPRE_parcsr_vector.c\
 F90_parcsr_matrix.c\
 F90_par_vector.c\
 NALU_HYPRE_parcsr_matrix.c\
 NALU_HYPRE_parcsr_vector.c\
 gen_fffc.c\
 new_commpkg.c\
 numbers.c\
 par_csr_aat.c\
 par_csr_assumed_part.c\
 par_csr_bool_matop.c\
 par_csr_bool_matrix.c\
 par_csr_communication.c\
 par_csr_matop.c\
 par_csr_matrix.c\
 par_csr_matvec.c\
 par_csr_matop_marked.c\
 par_csr_triplemat.c\
 par_make_system.c\
 par_vector.c\
 par_vector_batched.c

CUFILES =\
 par_csr_matvec_device.c\
 par_csr_fffc_device.c\
 par_csr_matop_device.c\
 par_csr_triplemat_device.c\
 par_vector_device.c

EXTRA_FILES =\
 parchord_to_parcsr.c

DRIVER_FILES =\
 driver_aat.c\
 driver_boolaat.c\
 driver_boolmatmul.c\
 driver.c\
 driver_matmul.c\
 driver_mat_multivec.c\
 driver_matvec.c\
 driver_multivec.c

COBJS = ${FILES:.c=.o}
CUOBJS = ${CUFILES:.c=.obj}
OBJS = ${COBJS} ${CUOBJS}

DRIVER_OBJS = ${DRIVER_FILES:.c=.o}

SONAME = libNALU_HYPRE_parcsr_mv-${NALU_HYPRE_RELEASE_VERSION}${NALU_HYPRE_LIB_SUFFIX}

##################################################################
# Targets
##################################################################

all: libNALU_HYPRE_parcsr_mv${NALU_HYPRE_LIB_SUFFIX}
	cp -fR $(srcdir)/NALU_HYPRE_*.h $(NALU_HYPRE_BUILD_DIR)/include
	cp -fR $(srcdir)/_nalu_hypre_parcsr_mv.h $(NALU_HYPRE_BUILD_DIR)/include
#	cp -fR libHYPRE* $(NALU_HYPRE_BUILD_DIR)/lib

driver: driver.o libNALU_HYPRE_parcsr_mv${NALU_HYPRE_LIB_SUFFIX}
	@echo  "Linking" $@ "... "
	${CC} -o driver driver.o ${LFLAGS}

driver_matvec: driver_matvec.o libNALU_HYPRE_parcsr_mv${NALU_HYPRE_LIB_SUFFIX}
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

install: libNALU_HYPRE_parcsr_mv${NALU_HYPRE_LIB_SUFFIX}
	cp -fR $(srcdir)/NALU_HYPRE_*.h $(NALU_HYPRE_INC_INSTALL)
	cp -fR $(srcdir)/_nalu_hypre_parcsr_mv.h $(NALU_HYPRE_INC_INSTALL)
#	cp -fR libHYPRE* $(NALU_HYPRE_LIB_INSTALL)

clean:
	rm -f *.o *.obj libHYPRE*
	rm -f driver driver_matvec driver_matmul
	rm -rf pchdir tca.map *inslog*

distclean: clean

##################################################################
# Rules
##################################################################

libNALU_HYPRE_parcsr_mv.a: ${OBJS}
	@echo  "Building $@ ... "
	${AR} $@ ${OBJS}
	${RANLIB} $@

libNALU_HYPRE_parcsr_mv.so libNALU_HYPRE_parcsr_mv.dylib: ${OBJS}
	@echo  "Building $@ ... "
	${BUILD_CC_SHARED} -o ${SONAME} ${OBJS} ${SHARED_SET_SONAME}${SONAME}
	ln -s -f ${SONAME} $@

${OBJS}: ${HEADERS}

${DRIVER_OBJS}: ${HEADERS}
