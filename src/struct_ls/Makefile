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
 -I$(srcdir)/../multivector\
 -I$(srcdir)/../utilities\
 -I$(srcdir)/../krylov\
 -I$(srcdir)/../struct_mv\
 ${CINCLUDES}

HEADERS =\
 NALU_HYPRE_struct_ls.h\
 _hypre_struct_ls.h\
 pfmg.h\
 red_black_gs.h\
 smg.h\
 sparse_msg.h

FILES =\
 coarsen.c\
 F90_NALU_HYPRE_struct_bicgstab.c\
 F90_NALU_HYPRE_struct_cycred.c\
 F90_NALU_HYPRE_struct_flexgmres.c\
 F90_NALU_HYPRE_struct_gmres.c\
 F90_NALU_HYPRE_struct_lgmres.c\
 F90_NALU_HYPRE_struct_hybrid.c\
 F90_NALU_HYPRE_struct_int.c\
 F90_NALU_HYPRE_struct_jacobi.c\
 F90_NALU_HYPRE_struct_pcg.c\
 F90_NALU_HYPRE_struct_pfmg.c\
 F90_NALU_HYPRE_struct_smg.c\
 F90_NALU_HYPRE_struct_sparse_msg.c\
 hybrid.c\
 NALU_HYPRE_struct_bicgstab.c\
 NALU_HYPRE_struct_cycred.c\
 NALU_HYPRE_struct_flexgmres.c\
 NALU_HYPRE_struct_gmres.c\
 NALU_HYPRE_struct_hybrid.c\
 NALU_HYPRE_struct_jacobi.c\
 NALU_HYPRE_struct_lgmres.c\
 NALU_HYPRE_struct_pfmg.c\
 NALU_HYPRE_struct_smg.c\
 NALU_HYPRE_struct_sparse_msg.c\
 jacobi.c\
 pcg_struct.c\
 pfmg.c\
 pfmg_relax.c\
 pfmg_setup_rap.c\
 pfmg_solve.c\
 semi.c\
 smg_relax.c\
 smg_setup.c\
 smg_setup_rap.c\
 smg_setup_restrict.c\
 smg_solve.c\
 sparse_msg.c\
 sparse_msg_setup.c\
 sparse_msg_setup_rap.c\
 sparse_msg_solve.c

CUFILES =\
 cyclic_reduction.c\
 NALU_HYPRE_struct_int.c\
 NALU_HYPRE_struct_pcg.c\
 pfmg2_setup_rap.c\
 pfmg3_setup_rap.c\
 pfmg_setup.c\
 pfmg_setup_interp.c\
 pfmg_setup_rap5.c\
 pfmg_setup_rap7.c\
 point_relax.c\
 red_black_constantcoef_gs.c\
 red_black_gs.c\
 semi_interp.c\
 semi_restrict.c\
 semi_setup_rap.c\
 smg2_setup_rap.c\
 smg3_setup_rap.c\
 smg.c\
 smg_axpy.c\
 smg_residual.c\
 smg_setup_interp.c\
 sparse_msg2_setup_rap.c\
 sparse_msg3_setup_rap.c\
 sparse_msg_filter.c\
 sparse_msg_interp.c\
 sparse_msg_restrict.c

COBJS = ${FILES:.c=.o}
CUOBJS = ${CUFILES:.c=.obj}
OBJS = ${COBJS} ${CUOBJS}

SONAME = libNALU_HYPRE_struct_ls-${NALU_HYPRE_RELEASE_VERSION}${NALU_HYPRE_LIB_SUFFIX}

##################################################################
# Targets
##################################################################

all: libNALU_HYPRE_struct_ls${NALU_HYPRE_LIB_SUFFIX}
	cp -fR $(srcdir)/NALU_HYPRE_*.h $(NALU_HYPRE_BUILD_DIR)/include
	cp -fR $(srcdir)/_hypre_struct_ls.h $(NALU_HYPRE_BUILD_DIR)/include
#	cp -fR libNALU_HYPRE* $(NALU_HYPRE_BUILD_DIR)/lib

install: libNALU_HYPRE_struct_ls${NALU_HYPRE_LIB_SUFFIX}
	cp -fR $(srcdir)/NALU_HYPRE_*.h $(NALU_HYPRE_INC_INSTALL)
	cp -fR $(srcdir)/_hypre_struct_ls.h $(NALU_HYPRE_INC_INSTALL)
#	cp -fR libNALU_HYPRE* $(NALU_HYPRE_LIB_INSTALL)

clean:
	rm -f *.o *.obj libNALU_HYPRE*
	rm -rf pchdir tca.map *inslog*

distclean: clean

##################################################################
# Rules
##################################################################

libNALU_HYPRE_struct_ls.a: ${OBJS}
	@echo  "Building $@ ... "
	${AR} $@ ${OBJS}
	${RANLIB} $@

libNALU_HYPRE_struct_ls.so libNALU_HYPRE_struct_ls.dylib: ${OBJS}
	@echo  "Building $@ ... "
	${BUILD_CC_SHARED} -o ${SONAME} ${OBJS} ${SHARED_SET_SONAME}${SONAME}
	ln -s -f ${SONAME} $@

${OBJS}: ${HEADERS}
