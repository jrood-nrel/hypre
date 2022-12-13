# Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

include ../../config/Makefile.config

BASE_DIR = ${NALU_HYPRE_FEI_BASE_DIR}

#SUPERLU_INCLUDE = -I$(srcdir)/../SuperLU/SRC
#SUPERLU_LIB     = -L$(srcdir)/../SuperLU

CINCLUDES=${INCLUDES} ${MPIINCLUDE} -I../ml/src/Include
CXXINCLUDES=${INCLUDES} ${MPIINCLUDE} -I../ml/src/Include
CDEFS = -DBOOL_NOT_SUPPORTED
CXXDEFS = ${CDEFS} -DMPICH_SKIP_MPICXX

C_COMPILE_FLAGS = \
 ${CDEFS}\
 -I../..\
 -I$(BASE_DIR)\
 -I$(srcdir)\
 -I$(srcdir)/../..\
 -I$(srcdir)/../../lapack\
 -I$(srcdir)/../../IJ_mv\
 -I$(srcdir)/../../utilities\
 -I$(srcdir)/../../multivector\
 -I$(srcdir)/../../krylov\
 -I$(srcdir)/../../parcsr_block_mv\
 -I$(srcdir)/../../parcsr_mv\
 -I$(srcdir)/../../parcsr_ls\
 -I$(srcdir)/../../seq_mv\
 -I$(srcdir)/../../distributed_matrix\
 -I$(srcdir)/../../distributed_ls\
 -I$(srcdir)/../fei-base\
 ${SUPERLU_INCLUDE}\
 ${DSUPERLU_INCLUDE}\
 ${CINCLUDES}

CXX_COMPILE_FLAGS = \
 ${CXXDEFS}\
 -I../..\
 -I$(BASE_DIR)\
 -I$(srcdir)/../..\
 -I$(srcdir)/../../lapack\
 -I$(srcdir)/../../IJ_mv\
 -I$(srcdir)/../../utilities\
 -I$(srcdir)/../../multivector\
 -I$(srcdir)/../../krylov\
 -I$(srcdir)/../../parcsr_block_mv\
 -I$(srcdir)/../../parcsr_mv\
 -I$(srcdir)/../../parcsr_ls\
 -I$(srcdir)/../../seq_mv\
 -I$(srcdir)/../../distributed_matrix\
 -I$(srcdir)/../../distributed_ls\
 -I$(srcdir)/../femli\
 -I$(srcdir)/../fei-base\
 ${SUPERLU_INCLUDE}\
 ${DSUPERLU_INCLUDE}\
 ${CXXINCLUDES}

HEADERS =\
 cfei-hypre.h\
 cfei_hypre.h\
 hypre_cfei.h\
 NALU_HYPRE_LinSysCore.h\
 NALU_HYPRE_FEI_includes.h \
 $(BASE_DIR)/fei_Data.hpp  \
 $(BASE_DIR)/fei_Lookup.hpp  \
 $(BASE_DIR)/fei_LinearSystemCore.hpp \
 $(BASE_DIR)/fei_defs.h\
 LLNL_FEI_Impl.h\
 LLNL_FEI_Fei.h\
 LLNL_FEI_LSCore.h\
 LLNL_FEI_Solver.h\
 LLNL_FEI_Matrix.h\
 FEI_NALU_HYPRE_Impl.h\
 FEI_NALU_HYPRE_include.h\
 NALU_HYPRE_Builder.h\
 NALU_HYPRE_LSI_Dsuperlu.h\
 NALU_HYPRE_LSI_UZAWA.h\
 NALU_HYPRE_LSI_Uzawa_c.h\
 NALU_HYPRE_LSI_blkprec.h\
 NALU_HYPRE_LSI_block.h\
 NALU_HYPRE_LSI_ddict.h\
 NALU_HYPRE_LSI_ddilut.h\
 NALU_HYPRE_LSI_mli.h\
 NALU_HYPRE_LSI_poly.h\
 NALU_HYPRE_LSI_schur.h\
 NALU_HYPRE_LSI_schwarz.h\
 NALU_HYPRE_MHMatrix.h\
 NALU_HYPRE_MLMatrix.h\
 NALU_HYPRE_MLMaxwell.h\
 NALU_HYPRE_SlideReduction.h\
 NALU_HYPRE_parcsr_TFQmr.h\
 NALU_HYPRE_parcsr_bicgs.h\
 NALU_HYPRE_parcsr_bicgstabl.h\
 NALU_HYPRE_parcsr_fgmres.h\
 NALU_HYPRE_parcsr_lsicg.h\
 NALU_HYPRE_parcsr_symqmr.h\
 fei_mv.h

FILESC = \
 NALU_HYPRE_LSI_Dsuperlu.c\
 NALU_HYPRE_LSI_ddict.c\
 NALU_HYPRE_LSI_ddilut.c\
 NALU_HYPRE_LSI_ml.c\
 NALU_HYPRE_LSI_mlmaxwell.c\
 NALU_HYPRE_LSI_poly.c\
 NALU_HYPRE_LSI_schwarz.c\
 NALU_HYPRE_parcsr_TFQmr.c\
 NALU_HYPRE_parcsr_bicgs.c\
 NALU_HYPRE_parcsr_bicgstabl.c\
 NALU_HYPRE_parcsr_fgmres.c\
 NALU_HYPRE_parcsr_lsicg.c\
 NALU_HYPRE_parcsr_symqmr.c\
 NALU_HYPRE_parcsr_maxwell.c\
 SymQMR.c\
 TFQmr.c\
 bicgs.c\
 bicgstabl.c\
 fgmres.c\
 hypre_lsi_amge.c\
 hypre_lsi_ddamg.c\
 hypre_lsi_misc.c\
 lsicg.c

FILESCXX = \
 FEI_NALU_HYPRE_Impl.cxx\
 LLNL_FEI_Impl.cxx\
 LLNL_FEI_Fei.cxx\
 LLNL_FEI_LSCore.cxx\
 LLNL_FEI_Solver.cxx\
 LLNL_FEI_Matrix.cxx\
 NALU_HYPRE_LSC_aux.cxx\
 NALU_HYPRE_LSI_UZAWA.cxx\
 NALU_HYPRE_LSI_blkprec.cxx\
 NALU_HYPRE_LSI_mli.cxx\
 NALU_HYPRE_LinSysCore.cxx\
 NALU_HYPRE_SlideReduction.cxx\
 cfei_hypre.cxx\
 hypre_cfei.cxx\
 hypre_schur_reduce.cxx\
 hypre_slide_reduce.cxx\
 NALU_HYPRE_fei_mesh.cxx\
 NALU_HYPRE_fei_matrix.cxx\
 NALU_HYPRE_fei_vector.cxx

OBJSC = ${FILESC:.c=.o}
OBJSCXX = ${FILESCXX:.cxx=.o}
OBJS = ${OBJSC} ${OBJSCXX}

SONAME = libNALU_HYPRE_FEI-${NALU_HYPRE_RELEASE_VERSION}${NALU_HYPRE_LIB_SUFFIX}

##################################################################
# Targets
##################################################################

all: libNALU_HYPRE_FEI${NALU_HYPRE_LIB_SUFFIX}
	cp -fR $(srcdir)/cfei-hypre.h $(NALU_HYPRE_BUILD_DIR)/include
	cp -fR $(srcdir)/cfei_hypre.h $(NALU_HYPRE_BUILD_DIR)/include
	cp -fR $(srcdir)/hypre_cfei.h $(NALU_HYPRE_BUILD_DIR)/include
	cp -fR $(srcdir)/NALU_HYPRE_LinSysCore.h $(NALU_HYPRE_BUILD_DIR)/include
	cp -fR $(srcdir)/NALU_HYPRE_FEI*.h $(NALU_HYPRE_BUILD_DIR)/include
	cp -fR $(srcdir)/LLNL_FEI*.h $(NALU_HYPRE_BUILD_DIR)/include
	cp -fR $(BASE_DIR)/fei_Data.hpp $(NALU_HYPRE_BUILD_DIR)/include
	cp -fR $(BASE_DIR)/fei_Lookup.hpp $(NALU_HYPRE_BUILD_DIR)/include
	cp -fR $(BASE_DIR)/fei_LinearSystemCore.hpp $(NALU_HYPRE_BUILD_DIR)/include
	cp -fR $(BASE_DIR)/fei_defs.h $(NALU_HYPRE_BUILD_DIR)/include

install: libNALU_HYPRE_FEI${NALU_HYPRE_LIB_SUFFIX}
	cp -fR $(srcdir)/cfei-hypre.h $(NALU_HYPRE_INC_INSTALL)
	cp -fR $(srcdir)/cfei_hypre.h $(NALU_HYPRE_INC_INSTALL)
	cp -fR $(srcdir)/hypre_cfei.h $(NALU_HYPRE_BUILD_DIR)/include
	cp -fR $(srcdir)/NALU_HYPRE_LinSysCore.h $(NALU_HYPRE_INC_INSTALL)
	cp -fR $(srcdir)/NALU_HYPRE_FEI*.h $(NALU_HYPRE_INC_INSTALL)
	cp -fR $(srcdir)/LLNL_FEI*.h $(NALU_HYPRE_INC_INSTALL)
	cp -fR $(BASE_DIR)/fei_Data.hpp $(NALU_HYPRE_BUILD_DIR)/include
	cp -fR $(BASE_DIR)/fei_Lookup.hpp $(NALU_HYPRE_BUILD_DIR)/include
	cp -fR $(BASE_DIR)/fei_LinearSystemCore.hpp $(NALU_HYPRE_BUILD_DIR)/include
	cp -fR $(BASE_DIR)/fei_defs.h $(NALU_HYPRE_BUILD_DIR)/include
	@echo  ""

clean:
	rm -rf *.o libHYPRE* 
	rm -rf pchdir tca.map *inslog*

distclean: clean

##################################################################
# Rules
##################################################################

libNALU_HYPRE_FEI.a: ${OBJS}
	@echo  "Building $@ ... "
	${AR} $@ ${OBJS}
	${RANLIB} $@

libNALU_HYPRE_FEI.so libNALU_HYPRE_FEI.dylib: ${OBJS}
	@echo  "Building $@ ... "
	${BUILD_CC_SHARED} -o ${SONAME} ${OBJS} ${SHARED_SET_SONAME}${SONAME}
	ln -s -f ${SONAME} $@

${OBJS}: ${HEADERS}

