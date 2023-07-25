# Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
# NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

set(HDRS
  aux_parcsr_matrix.h
  aux_par_vector.h
  NALU_HYPRE_IJ_mv.h
  _nalu_hypre_IJ_mv.h
  IJ_matrix.h
  IJ_vector.h
)

set(SRCS
  aux_parcsr_matrix.c
  aux_par_vector.c
  F90_NALU_HYPRE_IJMatrix.c
  F90_NALU_HYPRE_IJVector.c
  F90_IJMatrix.c
  NALU_HYPRE_IJMatrix.c
  NALU_HYPRE_IJVector.c
  IJ_assumed_part.c
  IJMatrix.c
  IJMatrix_parcsr.c
  IJVector.c
  IJVector_parcsr.c
  IJMatrix_parcsr_device.c
  IJVector_parcsr_device.c
)

target_sources(${PROJECT_NAME}
  PRIVATE ${SRCS}
          ${HDRS}
)

if (NALU_HYPRE_USING_CUDA OR NALU_HYPRE_USING_SYCL)
  set(GPU_SRCS
    IJMatrix_parcsr_device.c
    IJVector_parcsr_device.c
  )
  convert_filenames_to_full_paths(GPU_SRCS)
  set(NALU_HYPRE_GPU_SOURCES ${NALU_HYPRE_GPU_SOURCES} ${GPU_SRCS} PARENT_SCOPE)
endif ()

convert_filenames_to_full_paths(HDRS)
set(NALU_HYPRE_HEADERS ${NALU_HYPRE_HEADERS} ${HDRS} PARENT_SCOPE)
