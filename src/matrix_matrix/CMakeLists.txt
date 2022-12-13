# Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

set(HDRS
  NALU_HYPRE_matrix_matrix_protos.h
)

set(SRCS
  NALU_HYPRE_ConvertParCSRMatrixToDistributedMatrix.c
  NALU_HYPRE_ConvertPETScMatrixToDistributedMatrix.c
)

target_sources(${PROJECT_NAME}
  PRIVATE ${SRCS}
          ${HDRS}
)

convert_filenames_to_full_paths(HDRS)
set(NALU_HYPRE_HEADERS ${NALU_HYPRE_HEADERS} ${HDRS} PARENT_SCOPE)
