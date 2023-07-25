/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Header info for Parallel CSR Matrix data structures
 *
 * Note: this matrix currently uses 0-based indexing.
 *
 *****************************************************************************/

#ifndef nalu_hypre_PAR_CSR_BLOCK_MATRIX_HEADER
#define nalu_hypre_PAR_CSR_BLOCK_MATRIX_HEADER

#include "_nalu_hypre_utilities.h"
#include "csr_block_matrix.h"
#include "_nalu_hypre_parcsr_mv.h"

#ifdef __cplusplus
extern "C" {
#endif

/*--------------------------------------------------------------------------
 * Parallel CSR Block Matrix
 *--------------------------------------------------------------------------*/

typedef struct
{
   MPI_Comm             comm;

   NALU_HYPRE_BigInt         global_num_rows;
   NALU_HYPRE_BigInt         global_num_cols;
   NALU_HYPRE_BigInt         first_row_index;
   NALU_HYPRE_BigInt         first_col_diag;

   /* need to know entire local range in case row_starts and col_starts
      are null */
   NALU_HYPRE_BigInt         last_row_index;
   NALU_HYPRE_BigInt         last_col_diag;

   nalu_hypre_CSRBlockMatrix *diag;
   nalu_hypre_CSRBlockMatrix *offd;
   NALU_HYPRE_BigInt         *col_map_offd;
   /* maps columns of offd to global columns */
   NALU_HYPRE_BigInt          row_starts[2];
   /* row_starts[0] is start of local rows
      row_starts[1] is start of next processor's rows */
   NALU_HYPRE_BigInt          col_starts[2];
   /* col_starts[0] is start of local columns
      col_starts[1] is start of next processor's columns */

   nalu_hypre_ParCSRCommPkg  *comm_pkg;
   nalu_hypre_ParCSRCommPkg  *comm_pkgT;

   /* Does the ParCSRBlockMatrix create/destroy `diag', `offd', `col_map_offd'? */
   NALU_HYPRE_Int      owns_data;

   NALU_HYPRE_BigInt   num_nonzeros;
   NALU_HYPRE_Real     d_num_nonzeros;

   /* Buffers used by GetRow to hold row currently being accessed. AJC, 4/99 */
   NALU_HYPRE_Int     *rowindices;
   NALU_HYPRE_Complex *rowvalues;
   NALU_HYPRE_Int      getrowactive;

   nalu_hypre_IJAssumedPart *assumed_partition; /* only populated if no_global_partition option
                                              is used (compile-time option)*/
} nalu_hypre_ParCSRBlockMatrix;

/*--------------------------------------------------------------------------
 * Accessor functions for the Parallel CSR Block Matrix structure
 *--------------------------------------------------------------------------*/

#define nalu_hypre_ParCSRBlockMatrixComm(matrix)            ((matrix)->comm)
#define nalu_hypre_ParCSRBlockMatrixGlobalNumRows(matrix)   ((matrix)->global_num_rows)
#define nalu_hypre_ParCSRBlockMatrixGlobalNumCols(matrix)   ((matrix)->global_num_cols)
#define nalu_hypre_ParCSRBlockMatrixFirstRowIndex(matrix)   ((matrix)->first_row_index)
#define nalu_hypre_ParCSRBlockMatrixFirstColDiag(matrix)    ((matrix)->first_col_diag)
#define nalu_hypre_ParCSRBlockMatrixLastRowIndex(matrix)    ((matrix) -> last_row_index)
#define nalu_hypre_ParCSRBlockMatrixLastColDiag(matrix)     ((matrix) -> last_col_diag)
#define nalu_hypre_ParCSRBlockMatrixBlockSize(matrix)       ((matrix)->diag->block_size)
#define nalu_hypre_ParCSRBlockMatrixDiag(matrix)            ((matrix) -> diag)
#define nalu_hypre_ParCSRBlockMatrixOffd(matrix)            ((matrix) -> offd)
#define nalu_hypre_ParCSRBlockMatrixColMapOffd(matrix)      ((matrix) -> col_map_offd)
#define nalu_hypre_ParCSRBlockMatrixRowStarts(matrix)       ((matrix) -> row_starts)
#define nalu_hypre_ParCSRBlockMatrixColStarts(matrix)       ((matrix) -> col_starts)
#define nalu_hypre_ParCSRBlockMatrixCommPkg(matrix)         ((matrix) -> comm_pkg)
#define nalu_hypre_ParCSRBlockMatrixCommPkgT(matrix)        ((matrix) -> comm_pkgT)
#define nalu_hypre_ParCSRBlockMatrixOwnsData(matrix)        ((matrix) -> owns_data)
#define nalu_hypre_ParCSRBlockMatrixNumRows(matrix) \
nalu_hypre_CSRBlockMatrixNumRows(nalu_hypre_ParCSRBlockMatrixDiag(matrix))
#define nalu_hypre_ParCSRBlockMatrixNumCols(matrix) \
nalu_hypre_CSRBlockMatrixNumCols(nalu_hypre_ParCSRBlockMatrixDiag(matrix))
#define nalu_hypre_ParCSRBlockMatrixNumNonzeros(matrix)     ((matrix) -> num_nonzeros)
#define nalu_hypre_ParCSRBlockMatrixDNumNonzeros(matrix)    ((matrix) -> d_num_nonzeros)
#define nalu_hypre_ParCSRBlockMatrixRowindices(matrix)      ((matrix) -> rowindices)
#define nalu_hypre_ParCSRBlockMatrixRowvalues(matrix)       ((matrix) -> rowvalues)
#define nalu_hypre_ParCSRBlockMatrixGetrowactive(matrix)    ((matrix) -> getrowactive)
#define nalu_hypre_ParCSRBlockMatrixAssumedPartition(matrix) ((matrix) -> assumed_partition)


nalu_hypre_CSRBlockMatrix *
nalu_hypre_ParCSRBlockMatrixExtractBExt(nalu_hypre_ParCSRBlockMatrix *B,
                                   nalu_hypre_ParCSRBlockMatrix *A, NALU_HYPRE_Int data);

nalu_hypre_ParCSRBlockMatrix *
nalu_hypre_ParCSRBlockMatrixCreate(MPI_Comm comm, NALU_HYPRE_Int block_size, NALU_HYPRE_BigInt global_num_rows,
                              NALU_HYPRE_BigInt global_num_cols, NALU_HYPRE_BigInt *row_starts, NALU_HYPRE_BigInt *col_starts,
                              NALU_HYPRE_Int num_cols_offd, NALU_HYPRE_Int num_nonzeros_diag,
                              NALU_HYPRE_Int num_nonzeros_offd);

NALU_HYPRE_Int
nalu_hypre_ParCSRBlockMatrixDestroy( nalu_hypre_ParCSRBlockMatrix *matrix );



NALU_HYPRE_Int
nalu_hypre_BoomerAMGBuildBlockInterp( nalu_hypre_ParCSRBlockMatrix    *A,
                                 NALU_HYPRE_Int                  *CF_marker,
                                 nalu_hypre_ParCSRMatrix         *S,
                                 NALU_HYPRE_BigInt               *num_cpts_global,
                                 NALU_HYPRE_Int                   num_functions,
                                 NALU_HYPRE_Int                  *dof_func,
                                 NALU_HYPRE_Int                   debug_flag,
                                 NALU_HYPRE_Real                  trunc_factor,
                                 NALU_HYPRE_Int                   max_elmts,
                                 NALU_HYPRE_Int                   add_weak_to_diag,
                                 nalu_hypre_ParCSRBlockMatrix   **P_ptr);



NALU_HYPRE_Int
nalu_hypre_BoomerAMGBuildBlockInterpRV( nalu_hypre_ParCSRBlockMatrix   *A,
                                   NALU_HYPRE_Int                 *CF_marker,
                                   nalu_hypre_ParCSRMatrix        *S,
                                   NALU_HYPRE_BigInt              *num_cpts_global,
                                   NALU_HYPRE_Int                  num_functions,
                                   NALU_HYPRE_Int                 *dof_func,
                                   NALU_HYPRE_Int                  debug_flag,
                                   NALU_HYPRE_Real                 trunc_factor,
                                   NALU_HYPRE_Int                  max_elmts,
                                   nalu_hypre_ParCSRBlockMatrix  **P_ptr);

NALU_HYPRE_Int
nalu_hypre_BoomerAMGBuildBlockInterpRV2( nalu_hypre_ParCSRBlockMatrix    *A,
                                    NALU_HYPRE_Int                  *CF_marker,
                                    nalu_hypre_ParCSRMatrix         *S,
                                    NALU_HYPRE_BigInt               *num_cpts_global,
                                    NALU_HYPRE_Int                   num_functions,
                                    NALU_HYPRE_Int                  *dof_func,
                                    NALU_HYPRE_Int                   debug_flag,
                                    NALU_HYPRE_Real                  trunc_factor,
                                    NALU_HYPRE_Int                   max_elmts,
                                    nalu_hypre_ParCSRBlockMatrix   **P_ptr);
NALU_HYPRE_Int
nalu_hypre_BoomerAMGBuildBlockInterpDiag( nalu_hypre_ParCSRBlockMatrix  *A,
                                     NALU_HYPRE_Int                *CF_marker,
                                     nalu_hypre_ParCSRMatrix       *S,
                                     NALU_HYPRE_BigInt             *num_cpts_global,
                                     NALU_HYPRE_Int                 num_functions,
                                     NALU_HYPRE_Int                *dof_func,
                                     NALU_HYPRE_Int                 debug_flag,
                                     NALU_HYPRE_Real                trunc_factor,
                                     NALU_HYPRE_Int                 max_elmts,
                                     NALU_HYPRE_Int                 add_weak_to_diag,
                                     nalu_hypre_ParCSRBlockMatrix  **P_ptr);

NALU_HYPRE_Int nalu_hypre_BoomerAMGBlockInterpTruncation( nalu_hypre_ParCSRBlockMatrix *P,
                                                NALU_HYPRE_Real trunc_factor,
                                                NALU_HYPRE_Int max_elements);


NALU_HYPRE_Int
nalu_hypre_BoomerAMGBuildBlockDirInterp( nalu_hypre_ParCSRBlockMatrix   *A,
                                    NALU_HYPRE_Int                 *CF_marker,
                                    nalu_hypre_ParCSRMatrix        *S,
                                    NALU_HYPRE_BigInt              *num_cpts_global,
                                    NALU_HYPRE_Int                  num_functions,
                                    NALU_HYPRE_Int                 *dof_func,
                                    NALU_HYPRE_Int                  debug_flag,
                                    NALU_HYPRE_Real                 trunc_factor,
                                    NALU_HYPRE_Int                  max_elmts,
                                    nalu_hypre_ParCSRBlockMatrix  **P_ptr);


NALU_HYPRE_Int  nalu_hypre_BoomerAMGBlockRelaxIF( nalu_hypre_ParCSRBlockMatrix *A,
                                        nalu_hypre_ParVector    *f,
                                        NALU_HYPRE_Int          *cf_marker,
                                        NALU_HYPRE_Int           relax_type,
                                        NALU_HYPRE_Int           relax_order,
                                        NALU_HYPRE_Int           cycle_type,
                                        NALU_HYPRE_Real          relax_weight,
                                        NALU_HYPRE_Real          omega,
                                        nalu_hypre_ParVector    *u,
                                        nalu_hypre_ParVector    *Vtemp );


NALU_HYPRE_Int  nalu_hypre_BoomerAMGBlockRelax( nalu_hypre_ParCSRBlockMatrix *A,
                                      nalu_hypre_ParVector    *f,
                                      NALU_HYPRE_Int          *cf_marker,
                                      NALU_HYPRE_Int           relax_type,
                                      NALU_HYPRE_Int           relax_points,
                                      NALU_HYPRE_Real          relax_weight,
                                      NALU_HYPRE_Real          omega,
                                      nalu_hypre_ParVector    *u,
                                      nalu_hypre_ParVector    *Vtemp );

NALU_HYPRE_Int
nalu_hypre_GetCommPkgBlockRTFromCommPkgBlockA( nalu_hypre_ParCSRBlockMatrix *RT,
                                          nalu_hypre_ParCSRBlockMatrix *A,
                                          NALU_HYPRE_Int *tmp_map_offd,
                                          NALU_HYPRE_BigInt *fine_to_coarse_offd);


nalu_hypre_ParCSRCommHandle *
nalu_hypre_ParCSRBlockCommHandleCreate(NALU_HYPRE_Int job, NALU_HYPRE_Int bnnz, nalu_hypre_ParCSRCommPkg *comm_pkg,
                                  void *send_data, void *recv_data );


NALU_HYPRE_Int
nalu_hypre_ParCSRBlockCommHandleDestroy(nalu_hypre_ParCSRCommHandle *comm_handle);



NALU_HYPRE_Int
nalu_hypre_BlockMatvecCommPkgCreate(nalu_hypre_ParCSRBlockMatrix *A);


NALU_HYPRE_Int
nalu_hypre_ParCSRBlockMatrixCreateAssumedPartition( nalu_hypre_ParCSRBlockMatrix *matrix);

NALU_HYPRE_Int
nalu_hypre_ParCSRBlockMatrixDestroyAssumedPartition(nalu_hypre_ParCSRBlockMatrix *matrix );



nalu_hypre_ParCSRMatrix *
nalu_hypre_ParCSRBlockMatrixConvertToParCSRMatrix(nalu_hypre_ParCSRBlockMatrix *matrix);


nalu_hypre_ParCSRBlockMatrix *
nalu_hypre_ParCSRBlockMatrixConvertFromParCSRMatrix(nalu_hypre_ParCSRMatrix *matrix,
                                               NALU_HYPRE_Int matrix_C_block_size );


NALU_HYPRE_Int
nalu_hypre_ParCSRBlockMatrixRAP(nalu_hypre_ParCSRBlockMatrix  *RT,
                           nalu_hypre_ParCSRBlockMatrix  *A,
                           nalu_hypre_ParCSRBlockMatrix  *P,
                           nalu_hypre_ParCSRBlockMatrix **RAP_ptr );

NALU_HYPRE_Int
nalu_hypre_ParCSRBlockMatrixSetNumNonzeros( nalu_hypre_ParCSRBlockMatrix *matrix);

NALU_HYPRE_Int
nalu_hypre_ParCSRBlockMatrixSetDNumNonzeros( nalu_hypre_ParCSRBlockMatrix *matrix);

NALU_HYPRE_Int
nalu_hypre_BoomerAMGBlockCreateNodalA(nalu_hypre_ParCSRBlockMatrix    *A,
                                 NALU_HYPRE_Int  option, NALU_HYPRE_Int diag_option,
                                 nalu_hypre_ParCSRMatrix   **AN_ptr);

nalu_hypre_ParVector *
nalu_hypre_ParVectorCreateFromBlock(MPI_Comm comm,
                               NALU_HYPRE_BigInt p_global_size,
                               NALU_HYPRE_BigInt *p_partitioning, NALU_HYPRE_Int block_size);

NALU_HYPRE_Int
nalu_hypre_ParCSRBlockMatrixMatvec(NALU_HYPRE_Complex alpha, nalu_hypre_ParCSRBlockMatrix *A,
                              nalu_hypre_ParVector *x, NALU_HYPRE_Complex beta,
                              nalu_hypre_ParVector *y);
NALU_HYPRE_Int
nalu_hypre_ParCSRBlockMatrixMatvecT( NALU_HYPRE_Complex           alpha,
                                nalu_hypre_ParCSRBlockMatrix *A,
                                nalu_hypre_ParVector         *x,
                                NALU_HYPRE_Complex            beta,
                                nalu_hypre_ParVector          *y);






void nalu_hypre_block_qsort( NALU_HYPRE_Int *v,
                        NALU_HYPRE_Complex *w,
                        NALU_HYPRE_Complex *blk_array,
                        NALU_HYPRE_Int block_size,
                        NALU_HYPRE_Int  left,
                        NALU_HYPRE_Int  right );


void nalu_hypre_swap_blk( NALU_HYPRE_Complex *v,
                     NALU_HYPRE_Int block_size,
                     NALU_HYPRE_Int  i,
                     NALU_HYPRE_Int  j );


#ifdef __cplusplus
}
#endif
#endif
