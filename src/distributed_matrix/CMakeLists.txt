# Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
# NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

set(HDRS
  distributed_matrix.h
  NALU_HYPRE_distributed_matrix_mv.h
  NALU_HYPRE_distributed_matrix_protos.h
  NALU_HYPRE_distributed_matrix_types.h
)

set(SRCS
  distributed_matrix.c
  distributed_matrix_ISIS.c
  distributed_matrix_parcsr.c
  distributed_matrix_PETSc.c
  NALU_HYPRE_distributed_matrix.c
)

target_sources(${PROJECT_NAME}
  PRIVATE ${SRCS}
          ${HDRS}
)

convert_filenames_to_full_paths(HDRS)
set(NALU_HYPRE_HEADERS ${NALU_HYPRE_HEADERS} ${HDRS} PARENT_SCOPE)
