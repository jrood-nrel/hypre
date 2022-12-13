# Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

set(HDRS
  NALU_HYPRE_krylov.h
  NALU_HYPRE_lobpcg.h
  NALU_HYPRE_MatvecFunctions.h
  krylov.h
  lobpcg.h
)

set(SRCS
  bicgstab.c
  cgnr.c
  cogmres.c
  gmres.c
  flexgmres.c
  lgmres.c
  NALU_HYPRE_bicgstab.c
  NALU_HYPRE_cgnr.c
  NALU_HYPRE_gmres.c
  NALU_HYPRE_cogmres.c
  NALU_HYPRE_lgmres.c
  NALU_HYPRE_flexgmres.c
  NALU_HYPRE_pcg.c
  pcg.c
  NALU_HYPRE_lobpcg.c
  lobpcg.c
)

target_sources(${PROJECT_NAME}
  PRIVATE ${SRCS}
          ${HDRS}
)

convert_filenames_to_full_paths(HDRS)
set(NALU_HYPRE_HEADERS ${NALU_HYPRE_HEADERS} ${HDRS} PARENT_SCOPE)
