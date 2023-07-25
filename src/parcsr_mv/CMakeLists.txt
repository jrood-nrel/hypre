# Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

set(HDRS
  NALU_HYPRE_parcsr_mv.h
  _nalu_hypre_parcsr_mv.h
)

set(SRCS
  communicationT.c
  F90_NALU_HYPRE_parcsr_matrix.c
  F90_NALU_HYPRE_parcsr_vector.c
  F90_parcsr_matrix.c
  F90_par_vector.c
  gen_fffc.c
  NALU_HYPRE_parcsr_matrix.c
  NALU_HYPRE_parcsr_vector.c
  new_commpkg.c
  numbers.c
  par_csr_aat.c
  par_csr_assumed_part.c
  par_csr_bool_matop.c
  par_csr_bool_matrix.c
  par_csr_communication.c
  par_csr_matop.c
  par_csr_matrix.c
  par_csr_matop_marked.c
  par_csr_matvec.c
  par_csr_matvec_device.c
  par_vector.c
  par_vector_batched.c
  par_make_system.c
  par_csr_triplemat.c
  par_csr_fffc_device.c
  par_csr_matop_device.c
  par_csr_triplemat_device.c
  par_vector_device.c
)

target_sources(${PROJECT_NAME}
  PRIVATE ${SRCS}
          ${HDRS}
)

if (NALU_HYPRE_USING_CUDA OR NALU_HYPRE_USING_SYCL)
  set(GPU_SRCS
    par_csr_matvec_device.c
    par_csr_fffc_device.c
    par_csr_matop_device.c
    par_csr_triplemat_device.c
    par_vector_device.c
  )
  convert_filenames_to_full_paths(GPU_SRCS)
  set(NALU_HYPRE_GPU_SOURCES ${NALU_HYPRE_GPU_SOURCES} ${GPU_SRCS} PARENT_SCOPE)
endif ()

convert_filenames_to_full_paths(HDRS)
set(NALU_HYPRE_HEADERS ${NALU_HYPRE_HEADERS} ${HDRS} PARENT_SCOPE)
