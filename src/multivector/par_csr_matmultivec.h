/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Matvec functions for nalu_hypre_CSRMatrix class.
 *
 *****************************************************************************/

#ifndef PAR_CSR_MATMULTIVEC_HEADER
#define PAR_CSR_MATMULTIVEC_HEADER

#include "_nalu_hypre_parcsr_mv.h"
#include "par_multivector.h"

#ifdef __cplusplus
extern "C" {
#endif

NALU_HYPRE_Int nalu_hypre_ParCSRMatrixMatMultiVec(NALU_HYPRE_Complex, nalu_hypre_ParCSRMatrix*,
                                        nalu_hypre_ParMultiVector*,
                                        NALU_HYPRE_Complex, nalu_hypre_ParMultiVector*);


NALU_HYPRE_Int nalu_hypre_ParCSRMatrixMatMultiVecT(NALU_HYPRE_Complex, nalu_hypre_ParCSRMatrix*,
                                         nalu_hypre_ParMultiVector*,
                                         NALU_HYPRE_Complex, nalu_hypre_ParMultiVector*);

#ifdef __cplusplus
}
#endif

#endif  /* PAR_CSR_MATMULTIVEC_HEADER */
