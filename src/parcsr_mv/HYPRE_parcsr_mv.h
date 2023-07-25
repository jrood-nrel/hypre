/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Header file for NALU_HYPRE_parcsr_mv library
 *
 *****************************************************************************/

#ifndef NALU_HYPRE_PARCSR_MV_HEADER
#define NALU_HYPRE_PARCSR_MV_HEADER

#include "NALU_HYPRE_utilities.h"
#include "NALU_HYPRE_seq_mv.h"

#ifdef __cplusplus
extern "C" {
#endif

/*--------------------------------------------------------------------------
 * Structures
 *--------------------------------------------------------------------------*/

struct nalu_hypre_ParCSRMatrix_struct;
typedef struct nalu_hypre_ParCSRMatrix_struct *NALU_HYPRE_ParCSRMatrix;
struct nalu_hypre_ParVector_struct;
typedef struct nalu_hypre_ParVector_struct *NALU_HYPRE_ParVector;

/*--------------------------------------------------------------------------
 * Prototypes
 *--------------------------------------------------------------------------*/

/* NALU_HYPRE_parcsr_matrix.c */
NALU_HYPRE_Int NALU_HYPRE_ParCSRMatrixCreate( MPI_Comm comm, NALU_HYPRE_BigInt global_num_rows,
                                    NALU_HYPRE_BigInt global_num_cols, NALU_HYPRE_BigInt *row_starts, NALU_HYPRE_BigInt *col_starts,
                                    NALU_HYPRE_Int num_cols_offd, NALU_HYPRE_Int num_nonzeros_diag, NALU_HYPRE_Int num_nonzeros_offd,
                                    NALU_HYPRE_ParCSRMatrix *matrix );
NALU_HYPRE_Int NALU_HYPRE_ParCSRMatrixDestroy( NALU_HYPRE_ParCSRMatrix matrix );
NALU_HYPRE_Int NALU_HYPRE_ParCSRMatrixInitialize( NALU_HYPRE_ParCSRMatrix matrix );
NALU_HYPRE_Int NALU_HYPRE_ParCSRMatrixRead( MPI_Comm comm, const char *file_name,
                                  NALU_HYPRE_ParCSRMatrix *matrix );
NALU_HYPRE_Int NALU_HYPRE_ParCSRMatrixPrint( NALU_HYPRE_ParCSRMatrix matrix, const char *file_name );
NALU_HYPRE_Int NALU_HYPRE_ParCSRMatrixGetComm( NALU_HYPRE_ParCSRMatrix matrix, MPI_Comm *comm );
NALU_HYPRE_Int NALU_HYPRE_ParCSRMatrixGetDims( NALU_HYPRE_ParCSRMatrix matrix, NALU_HYPRE_BigInt *M, NALU_HYPRE_BigInt *N );
NALU_HYPRE_Int NALU_HYPRE_ParCSRMatrixGetRowPartitioning( NALU_HYPRE_ParCSRMatrix matrix,
                                                NALU_HYPRE_BigInt **row_partitioning_ptr );
NALU_HYPRE_Int NALU_HYPRE_ParCSRMatrixGetColPartitioning( NALU_HYPRE_ParCSRMatrix matrix,
                                                NALU_HYPRE_BigInt **col_partitioning_ptr );
NALU_HYPRE_Int NALU_HYPRE_ParCSRMatrixGetLocalRange( NALU_HYPRE_ParCSRMatrix matrix, NALU_HYPRE_BigInt *row_start,
                                           NALU_HYPRE_BigInt *row_end, NALU_HYPRE_BigInt *col_start, NALU_HYPRE_BigInt *col_end );
NALU_HYPRE_Int NALU_HYPRE_ParCSRMatrixGetRow( NALU_HYPRE_ParCSRMatrix matrix, NALU_HYPRE_BigInt row, NALU_HYPRE_Int *size,
                                    NALU_HYPRE_BigInt **col_ind, NALU_HYPRE_Complex **values );
NALU_HYPRE_Int NALU_HYPRE_ParCSRMatrixRestoreRow( NALU_HYPRE_ParCSRMatrix matrix, NALU_HYPRE_BigInt row,
                                        NALU_HYPRE_Int *size, NALU_HYPRE_BigInt **col_ind, NALU_HYPRE_Complex **values );
NALU_HYPRE_Int NALU_HYPRE_CSRMatrixToParCSRMatrix( MPI_Comm comm, NALU_HYPRE_CSRMatrix A_CSR,
                                         NALU_HYPRE_BigInt *row_partitioning, NALU_HYPRE_BigInt *col_partitioning, NALU_HYPRE_ParCSRMatrix *matrix );
NALU_HYPRE_Int NALU_HYPRE_ParCSRMatrixMatvec( NALU_HYPRE_Complex alpha, NALU_HYPRE_ParCSRMatrix A, NALU_HYPRE_ParVector x,
                                    NALU_HYPRE_Complex beta, NALU_HYPRE_ParVector y );
NALU_HYPRE_Int NALU_HYPRE_ParCSRMatrixMatvecT( NALU_HYPRE_Complex alpha, NALU_HYPRE_ParCSRMatrix A, NALU_HYPRE_ParVector x,
                                     NALU_HYPRE_Complex beta, NALU_HYPRE_ParVector y );

/* NALU_HYPRE_parcsr_vector.c */
NALU_HYPRE_Int NALU_HYPRE_ParVectorCreate( MPI_Comm comm, NALU_HYPRE_BigInt global_size,
                                 NALU_HYPRE_BigInt *partitioning, NALU_HYPRE_ParVector *vector );
NALU_HYPRE_Int NALU_HYPRE_ParVectorDestroy( NALU_HYPRE_ParVector vector );
NALU_HYPRE_Int NALU_HYPRE_ParVectorInitialize( NALU_HYPRE_ParVector vector );
NALU_HYPRE_Int NALU_HYPRE_ParVectorRead( MPI_Comm comm, const char *file_name, NALU_HYPRE_ParVector *vector );
NALU_HYPRE_Int NALU_HYPRE_ParVectorPrint( NALU_HYPRE_ParVector vector, const char *file_name );
NALU_HYPRE_Int NALU_HYPRE_ParVectorSetConstantValues( NALU_HYPRE_ParVector vector, NALU_HYPRE_Complex value );
NALU_HYPRE_Int NALU_HYPRE_ParVectorSetRandomValues( NALU_HYPRE_ParVector vector, NALU_HYPRE_Int seed );
NALU_HYPRE_Int NALU_HYPRE_ParVectorCopy( NALU_HYPRE_ParVector x, NALU_HYPRE_ParVector y );
NALU_HYPRE_Int NALU_HYPRE_ParVectorScale( NALU_HYPRE_Complex value, NALU_HYPRE_ParVector x );
NALU_HYPRE_Int NALU_HYPRE_ParVectorInnerProd( NALU_HYPRE_ParVector x, NALU_HYPRE_ParVector y, NALU_HYPRE_Real *prod );
NALU_HYPRE_Int NALU_HYPRE_VectorToParVector( MPI_Comm comm, NALU_HYPRE_Vector b, NALU_HYPRE_BigInt *partitioning,
                                   NALU_HYPRE_ParVector *vector );
NALU_HYPRE_Int NALU_HYPRE_ParVectorGetValues( NALU_HYPRE_ParVector vector, NALU_HYPRE_Int num_values,
                                    NALU_HYPRE_BigInt *indices, NALU_HYPRE_Complex *values );

#ifdef __cplusplus
}
#endif

#endif
