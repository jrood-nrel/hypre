# Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

set(HDRS
  NALU_HYPRE_sstruct_mv.h
  _hypre_sstruct_mv.h
)

set(SRCS
  F90_NALU_HYPRE_sstruct_graph.c
  F90_NALU_HYPRE_sstruct_grid.c
  F90_NALU_HYPRE_sstruct_matrix.c
  F90_NALU_HYPRE_sstruct_stencil.c
  F90_NALU_HYPRE_sstruct_vector.c
  NALU_HYPRE_sstruct_graph.c
  NALU_HYPRE_sstruct_grid.c
  NALU_HYPRE_sstruct_matrix.c
  NALU_HYPRE_sstruct_stencil.c
  NALU_HYPRE_sstruct_vector.c
  sstruct_axpy.c
  sstruct_copy.c
  sstruct_graph.c
  sstruct_grid.c
  sstruct_innerprod.c
  sstruct_matrix.c
  sstruct_matvec.c
  sstruct_scale.c
  sstruct_stencil.c
  sstruct_vector.c
)

target_sources(${PROJECT_NAME}
  PRIVATE ${SRCS}
          ${HDRS}
)

if (NALU_HYPRE_USING_CUDA OR NALU_HYPRE_USING_SYCL)
  set(GPU_SRCS
    sstruct_matrix.c
    sstruct_vector.c
  )
  convert_filenames_to_full_paths(GPU_SRCS)
  set(NALU_HYPRE_GPU_SOURCES ${NALU_HYPRE_GPU_SOURCES} ${GPU_SRCS} PARENT_SCOPE)
endif ()

convert_filenames_to_full_paths(HDRS)
set(NALU_HYPRE_HEADERS ${NALU_HYPRE_HEADERS} ${HDRS} PARENT_SCOPE)
