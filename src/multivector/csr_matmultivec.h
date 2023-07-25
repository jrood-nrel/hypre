/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#ifndef CSR_MULTIMATVEC_H
#define CSR_MULTIMATVEC_H

#include "seq_mv.h"
#include "seq_multivector.h"

#ifdef __cplusplus
extern "C" {
#endif
/*--------------------------------------------------------------------------
 * nalu_hypre_CSRMatrixMatMultivec
 *--------------------------------------------------------------------------*/
NALU_HYPRE_Int
nalu_hypre_CSRMatrixMatMultivec(NALU_HYPRE_Complex alpha, nalu_hypre_CSRMatrix *A,
                           nalu_hypre_Multivector *x, NALU_HYPRE_Complex beta,
                           nalu_hypre_Multivector *y);


/*--------------------------------------------------------------------------
 * nalu_hypre_CSRMatrixMultiMatvecT
 *
 *   Performs y <- alpha * A^T * x + beta * y
 *
 *   From Van Henson's modification of nalu_hypre_CSRMatrixMatvec.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_CSRMatrixMatMultivecT(NALU_HYPRE_Complex alpha, nalu_hypre_CSRMatrix *A,
                            nalu_hypre_Multivector *x, NALU_HYPRE_Complex beta,
                            nalu_hypre_Multivector *y);

#ifdef __cplusplus
}
#endif

#endif /* CSR_MATMULTIVEC_H */
