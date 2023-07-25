/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Header info for CSR Block Matrix data structures
 *
 * Note: this matrix currently uses 0-based indexing.
 * Note: everything is in terms of blocks (ie. num_rows is the number
 *       of block rows)
 *
 *****************************************************************************/

#ifndef nalu_hypre_CSR_BLOCK_MATRIX_HEADER
#define nalu_hypre_CSR_BLOCK_MATRIX_HEADER

#include "seq_mv.h"
#include "_nalu_hypre_utilities.h"

#ifdef __cplusplus
extern "C" {
#endif

/*--------------------------------------------------------------------------
 * CSR Block Matrix
 *--------------------------------------------------------------------------*/

typedef struct
{
   NALU_HYPRE_Complex    *data;
   NALU_HYPRE_Int        *i;
   NALU_HYPRE_Int        *j;
   NALU_HYPRE_BigInt     *big_j;
   NALU_HYPRE_Int         block_size;
   NALU_HYPRE_Int         num_rows;
   NALU_HYPRE_Int         num_cols;
   NALU_HYPRE_Int         num_nonzeros;
   NALU_HYPRE_Int         owns_data;

} nalu_hypre_CSRBlockMatrix;

/*--------------------------------------------------------------------------
 * Accessor functions for the CSR Block Matrix structure
 *--------------------------------------------------------------------------*/

#define nalu_hypre_CSRBlockMatrixData(matrix)         ((matrix) -> data)
#define nalu_hypre_CSRBlockMatrixI(matrix)            ((matrix) -> i)
#define nalu_hypre_CSRBlockMatrixJ(matrix)            ((matrix) -> j)
#define nalu_hypre_CSRBlockMatrixBigJ(matrix)         ((matrix) -> big_j)
#define nalu_hypre_CSRBlockMatrixBlockSize(matrix)    ((matrix) -> block_size)
#define nalu_hypre_CSRBlockMatrixNumRows(matrix)      ((matrix) -> num_rows)
#define nalu_hypre_CSRBlockMatrixNumCols(matrix)      ((matrix) -> num_cols)
#define nalu_hypre_CSRBlockMatrixNumNonzeros(matrix)  ((matrix) -> num_nonzeros)
#define nalu_hypre_CSRBlockMatrixOwnsData(matrix)     ((matrix) -> owns_data)

/*--------------------------------------------------------------------------
 * other functions for the CSR Block Matrix structure
 *--------------------------------------------------------------------------*/

nalu_hypre_CSRBlockMatrix
*nalu_hypre_CSRBlockMatrixCreate(NALU_HYPRE_Int, NALU_HYPRE_Int, NALU_HYPRE_Int, NALU_HYPRE_Int);
NALU_HYPRE_Int nalu_hypre_CSRBlockMatrixDestroy(nalu_hypre_CSRBlockMatrix *);
NALU_HYPRE_Int nalu_hypre_CSRBlockMatrixInitialize(nalu_hypre_CSRBlockMatrix *);
NALU_HYPRE_Int nalu_hypre_CSRBlockMatrixBigInitialize(nalu_hypre_CSRBlockMatrix *);
NALU_HYPRE_Int nalu_hypre_CSRBlockMatrixSetDataOwner(nalu_hypre_CSRBlockMatrix *, NALU_HYPRE_Int);
nalu_hypre_CSRMatrix
*nalu_hypre_CSRBlockMatrixCompress(nalu_hypre_CSRBlockMatrix *);
nalu_hypre_CSRMatrix
*nalu_hypre_CSRBlockMatrixConvertToCSRMatrix(nalu_hypre_CSRBlockMatrix *);
nalu_hypre_CSRBlockMatrix
*nalu_hypre_CSRBlockMatrixConvertFromCSRMatrix(nalu_hypre_CSRMatrix *, NALU_HYPRE_Int);
NALU_HYPRE_Int nalu_hypre_CSRBlockMatrixBlockAdd(NALU_HYPRE_Complex *, NALU_HYPRE_Complex *, NALU_HYPRE_Complex*, NALU_HYPRE_Int);

NALU_HYPRE_Int nalu_hypre_CSRBlockMatrixBlockMultAdd(NALU_HYPRE_Complex *, NALU_HYPRE_Complex *, NALU_HYPRE_Complex,
                                           NALU_HYPRE_Complex *, NALU_HYPRE_Int);
NALU_HYPRE_Int nalu_hypre_CSRBlockMatrixBlockMultAddDiag(NALU_HYPRE_Complex *, NALU_HYPRE_Complex *, NALU_HYPRE_Complex,
                                               NALU_HYPRE_Complex *, NALU_HYPRE_Int);
NALU_HYPRE_Int
nalu_hypre_CSRBlockMatrixBlockMultAddDiag2(NALU_HYPRE_Complex* i1, NALU_HYPRE_Complex* i2, NALU_HYPRE_Complex beta,
                                      NALU_HYPRE_Complex* o, NALU_HYPRE_Int block_size);
NALU_HYPRE_Int
nalu_hypre_CSRBlockMatrixBlockMultAddDiag3(NALU_HYPRE_Complex* i1, NALU_HYPRE_Complex* i2, NALU_HYPRE_Complex beta,
                                      NALU_HYPRE_Complex* o, NALU_HYPRE_Int block_size);


NALU_HYPRE_Int nalu_hypre_CSRBlockMatrixBlockInvMult(NALU_HYPRE_Complex *, NALU_HYPRE_Complex *, NALU_HYPRE_Complex *,
                                           NALU_HYPRE_Int);
NALU_HYPRE_Int nalu_hypre_CSRBlockMatrixBlockInvMultDiag(NALU_HYPRE_Complex *, NALU_HYPRE_Complex *, NALU_HYPRE_Complex *,
                                               NALU_HYPRE_Int);

NALU_HYPRE_Int
nalu_hypre_CSRBlockMatrixBlockInvMultDiag2(NALU_HYPRE_Complex* i1, NALU_HYPRE_Complex* i2, NALU_HYPRE_Complex* o,
                                      NALU_HYPRE_Int block_size);

NALU_HYPRE_Int
nalu_hypre_CSRBlockMatrixBlockInvMultDiag3(NALU_HYPRE_Complex* i1, NALU_HYPRE_Complex* i2, NALU_HYPRE_Complex* o,
                                      NALU_HYPRE_Int block_size);




NALU_HYPRE_Int nalu_hypre_CSRBlockMatrixBlockMultInv(NALU_HYPRE_Complex *, NALU_HYPRE_Complex *, NALU_HYPRE_Complex *,
                                           NALU_HYPRE_Int);
NALU_HYPRE_Int nalu_hypre_CSRBlockMatrixBlockTranspose(NALU_HYPRE_Complex *, NALU_HYPRE_Complex *, NALU_HYPRE_Int);

NALU_HYPRE_Int nalu_hypre_CSRBlockMatrixTranspose(nalu_hypre_CSRBlockMatrix *A,
                                        nalu_hypre_CSRBlockMatrix **AT, NALU_HYPRE_Int data);

NALU_HYPRE_Int nalu_hypre_CSRBlockMatrixBlockCopyData(NALU_HYPRE_Complex*, NALU_HYPRE_Complex*, NALU_HYPRE_Complex,
                                            NALU_HYPRE_Int);
NALU_HYPRE_Int nalu_hypre_CSRBlockMatrixBlockCopyDataDiag(NALU_HYPRE_Complex*, NALU_HYPRE_Complex*, NALU_HYPRE_Complex,
                                                NALU_HYPRE_Int);

NALU_HYPRE_Int nalu_hypre_CSRBlockMatrixBlockAddAccumulate(NALU_HYPRE_Complex*, NALU_HYPRE_Complex*, NALU_HYPRE_Int);
NALU_HYPRE_Int nalu_hypre_CSRBlockMatrixBlockAddAccumulateDiag(NALU_HYPRE_Complex* i1, NALU_HYPRE_Complex* o,
                                                     NALU_HYPRE_Int block_size);



NALU_HYPRE_Int
nalu_hypre_CSRBlockMatrixMatvec(NALU_HYPRE_Complex alpha, nalu_hypre_CSRBlockMatrix *A,
                           nalu_hypre_Vector *x, NALU_HYPRE_Complex beta, nalu_hypre_Vector *y);


NALU_HYPRE_Int
nalu_hypre_CSRBlockMatrixMatvecT( NALU_HYPRE_Complex alpha, nalu_hypre_CSRBlockMatrix *A, nalu_hypre_Vector  *x,
                             NALU_HYPRE_Complex beta, nalu_hypre_Vector *y );

NALU_HYPRE_Int
nalu_hypre_CSRBlockMatrixBlockInvMatvec(NALU_HYPRE_Complex* mat, NALU_HYPRE_Complex* v,
                                   NALU_HYPRE_Complex* ov, NALU_HYPRE_Int block_size);

NALU_HYPRE_Int
nalu_hypre_CSRBlockMatrixBlockMatvec(NALU_HYPRE_Complex alpha, NALU_HYPRE_Complex* mat, NALU_HYPRE_Complex* v,
                                NALU_HYPRE_Complex beta,
                                NALU_HYPRE_Complex* ov, NALU_HYPRE_Int block_size);


NALU_HYPRE_Int nalu_hypre_CSRBlockMatrixBlockNorm(NALU_HYPRE_Int norm_type, NALU_HYPRE_Complex* data, NALU_HYPRE_Real* out,
                                        NALU_HYPRE_Int block_size);

NALU_HYPRE_Int nalu_hypre_CSRBlockMatrixBlockSetScalar(NALU_HYPRE_Complex* o, NALU_HYPRE_Complex beta,
                                             NALU_HYPRE_Int block_size);

NALU_HYPRE_Int nalu_hypre_CSRBlockMatrixComputeSign(NALU_HYPRE_Complex *i1, NALU_HYPRE_Complex *o,
                                          NALU_HYPRE_Int block_size);
NALU_HYPRE_Int nalu_hypre_CSRBlockMatrixBlockAddAccumulateDiagCheckSign(NALU_HYPRE_Complex* i1, NALU_HYPRE_Complex* o,
                                                              NALU_HYPRE_Int block_size, NALU_HYPRE_Real *sign);
NALU_HYPRE_Int nalu_hypre_CSRBlockMatrixBlockMultAddDiagCheckSign(NALU_HYPRE_Complex* i1, NALU_HYPRE_Complex* i2,
                                                        NALU_HYPRE_Complex beta, NALU_HYPRE_Complex* o, NALU_HYPRE_Int block_size, NALU_HYPRE_Real *sign);

#ifdef __cplusplus
}
#endif
#endif
