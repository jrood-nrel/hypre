/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Matvec functions for hypre_CSRMatrix class.
 *
 *****************************************************************************/

#ifndef PAR_CSR_MATMULTIVEC_HEADER
#define PAR_CSR_MATMULTIVEC_HEADER

#include "_hypre_parcsr_mv.h"
#include "par_multivector.h"

#ifdef __cplusplus
extern "C" {
#endif

NALU_HYPRE_Int hypre_ParCSRMatrixMatMultiVec(NALU_HYPRE_Complex, hypre_ParCSRMatrix*,
                                        hypre_ParMultiVector*,
                                        NALU_HYPRE_Complex, hypre_ParMultiVector*);


NALU_HYPRE_Int hypre_ParCSRMatrixMatMultiVecT(NALU_HYPRE_Complex, hypre_ParCSRMatrix*,
                                         hypre_ParMultiVector*,
                                         NALU_HYPRE_Complex, hypre_ParMultiVector*);

#ifdef __cplusplus
}
#endif

#endif  /* PAR_CSR_MATMULTIVEC_HEADER */
