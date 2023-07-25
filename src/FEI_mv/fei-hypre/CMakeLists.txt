# Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
# NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)


set(NALU_HYPRE_fei_HEADERS
  cfei-hypre.h
  cfei_hypre.h
  nalu_hypre_cfei.h
  NALU_HYPRE_LinSysCore.h
  NALU_HYPRE_FEI_includes.h
  LLNL_FEI_Fei.h
  LLNL_FEI_Impl.h
  LLNL_FEI_LSCore.h
  LLNL_FEI_Matrix.h
  LLNL_FEI_Solver.h
)

set(NALU_HYPRE_fei_SRCS 
  NALU_HYPRE_LSI_Dsuperlu.c
  NALU_HYPRE_LSI_ddict.c
  NALU_HYPRE_LSI_ddilut.c
  NALU_HYPRE_LSI_ml.c
  NALU_HYPRE_LSI_mlmaxwell.c
  NALU_HYPRE_LSI_poly.c
  NALU_HYPRE_LSI_schwarz.c
  NALU_HYPRE_parcsr_TFQmr.c
  NALU_HYPRE_parcsr_bicgs.c
  NALU_HYPRE_parcsr_bicgstabl.c
  NALU_HYPRE_parcsr_fgmres.c
  NALU_HYPRE_parcsr_lsicg.c
  NALU_HYPRE_parcsr_symqmr.c
  NALU_HYPRE_parcsr_maxwell.c
  SymQMR.c
  TFQmr.c
  bicgs.c
  bicgstabl.c
  fgmres.c
  nalu_hypre_lsi_amge.c
  nalu_hypre_lsi_ddamg.c
  nalu_hypre_lsi_misc.c
  lsicg.c
  FEI_NALU_HYPRE_Impl.cxx
  LLNL_FEI_Impl.cxx
  LLNL_FEI_Fei.cxx
  LLNL_FEI_LSCore.cxx
  LLNL_FEI_Solver.cxx
  LLNL_FEI_Matrix.cxx
  NALU_HYPRE_LSC_aux.cxx
  NALU_HYPRE_LSI_UZAWA.cxx
  NALU_HYPRE_LSI_blkprec.cxx
  NALU_HYPRE_LSI_mli.cxx
  NALU_HYPRE_LinSysCore.cxx
  NALU_HYPRE_SlideReduction.cxx
  cfei_hypre.cxx
  nalu_hypre_cfei.cxx
  nalu_hypre_schur_reduce.cxx
  nalu_hypre_slide_reduce.cxx
  NALU_HYPRE_fei_mesh.cxx
  NALU_HYPRE_fei_matrix.cxx
  NALU_HYPRE_fei_vector.cxx 
)

install (FILES ${NALU_HYPRE_fei_HEADERS} DESTINATION include)

target_sources(${PROJECT_NAME}
  PRIVATE ${NALU_HYPRE_fei_SRCS}
)
