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

#ifndef hypre_PAR_CSR_BLOCK_MATRIX_HEADER
#define hypre_PAR_CSR_BLOCK_MATRIX_HEADER

#include "_hypre_utilities.h"
#include "csr_block_matrix.h"
#include "_hypre_parcsr_mv.h"

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

   hypre_CSRBlockMatrix *diag;
   hypre_CSRBlockMatrix *offd;
   NALU_HYPRE_BigInt         *col_map_offd;
   /* maps columns of offd to global columns */
   NALU_HYPRE_BigInt          row_starts[2];
   /* row_starts[0] is start of local rows
      row_starts[1] is start of next processor's rows */
   NALU_HYPRE_BigInt          col_starts[2];
   /* col_starts[0] is start of local columns
      col_starts[1] is start of next processor's columns */

   hypre_ParCSRCommPkg  *comm_pkg;
   hypre_ParCSRCommPkg  *comm_pkgT;

   /* Does the ParCSRBlockMatrix create/destroy `diag', `offd', `col_map_offd'? */
   NALU_HYPRE_Int      owns_data;

   NALU_HYPRE_BigInt   num_nonzeros;
   NALU_HYPRE_Real     d_num_nonzeros;

   /* Buffers used by GetRow to hold row currently being accessed. AJC, 4/99 */
   NALU_HYPRE_Int     *rowindices;
   NALU_HYPRE_Complex *rowvalues;
   NALU_HYPRE_Int      getrowactive;

   hypre_IJAssumedPart *assumed_partition; /* only populated if no_global_partition option
                                              is used (compile-time option)*/
} hypre_ParCSRBlockMatrix;

/*--------------------------------------------------------------------------
 * Accessor functions for the Parallel CSR Block Matrix structure
 *--------------------------------------------------------------------------*/

#define hypre_ParCSRBlockMatrixComm(matrix)            ((matrix)->comm)
#define hypre_ParCSRBlockMatrixGlobalNumRows(matrix)   ((matrix)->global_num_rows)
#define hypre_ParCSRBlockMatrixGlobalNumCols(matrix)   ((matrix)->global_num_cols)
#define hypre_ParCSRBlockMatrixFirstRowIndex(matrix)   ((matrix)->first_row_index)
#define hypre_ParCSRBlockMatrixFirstColDiag(matrix)    ((matrix)->first_col_diag)
#define hypre_ParCSRBlockMatrixLastRowIndex(matrix)    ((matrix) -> last_row_index)
#define hypre_ParCSRBlockMatrixLastColDiag(matrix)     ((matrix) -> last_col_diag)
#define hypre_ParCSRBlockMatrixBlockSize(matrix)       ((matrix)->diag->block_size)
#define hypre_ParCSRBlockMatrixDiag(matrix)            ((matrix) -> diag)
#define hypre_ParCSRBlockMatrixOffd(matrix)            ((matrix) -> offd)
#define hypre_ParCSRBlockMatrixColMapOffd(matrix)      ((matrix) -> col_map_offd)
#define hypre_ParCSRBlockMatrixRowStarts(matrix)       ((matrix) -> row_starts)
#define hypre_ParCSRBlockMatrixColStarts(matrix)       ((matrix) -> col_starts)
#define hypre_ParCSRBlockMatrixCommPkg(matrix)         ((matrix) -> comm_pkg)
#define hypre_ParCSRBlockMatrixCommPkgT(matrix)        ((matrix) -> comm_pkgT)
#define hypre_ParCSRBlockMatrixOwnsData(matrix)        ((matrix) -> owns_data)
#define hypre_ParCSRBlockMatrixNumRows(matrix) \
hypre_CSRBlockMatrixNumRows(hypre_ParCSRBlockMatrixDiag(matrix))
#define hypre_ParCSRBlockMatrixNumCols(matrix) \
hypre_CSRBlockMatrixNumCols(hypre_ParCSRBlockMatrixDiag(matrix))
#define hypre_ParCSRBlockMatrixNumNonzeros(matrix)     ((matrix) -> num_nonzeros)
#define hypre_ParCSRBlockMatrixDNumNonzeros(matrix)    ((matrix) -> d_num_nonzeros)
#define hypre_ParCSRBlockMatrixRowindices(matrix)      ((matrix) -> rowindices)
#define hypre_ParCSRBlockMatrixRowvalues(matrix)       ((matrix) -> rowvalues)
#define hypre_ParCSRBlockMatrixGetrowactive(matrix)    ((matrix) -> getrowactive)
#define hypre_ParCSRBlockMatrixAssumedPartition(matrix) ((matrix) -> assumed_partition)


hypre_CSRBlockMatrix *
hypre_ParCSRBlockMatrixExtractBExt(hypre_ParCSRBlockMatrix *B,
                                   hypre_ParCSRBlockMatrix *A, NALU_HYPRE_Int data);

hypre_ParCSRBlockMatrix *
hypre_ParCSRBlockMatrixCreate(MPI_Comm comm, NALU_HYPRE_Int block_size, NALU_HYPRE_BigInt global_num_rows,
                              NALU_HYPRE_BigInt global_num_cols, NALU_HYPRE_BigInt *row_starts, NALU_HYPRE_BigInt *col_starts,
                              NALU_HYPRE_Int num_cols_offd, NALU_HYPRE_Int num_nonzeros_diag,
                              NALU_HYPRE_Int num_nonzeros_offd);

NALU_HYPRE_Int
hypre_ParCSRBlockMatrixDestroy( hypre_ParCSRBlockMatrix *matrix );



NALU_HYPRE_Int
hypre_BoomerAMGBuildBlockInterp( hypre_ParCSRBlockMatrix    *A,
                                 NALU_HYPRE_Int                  *CF_marker,
                                 hypre_ParCSRMatrix         *S,
                                 NALU_HYPRE_BigInt               *num_cpts_global,
                                 NALU_HYPRE_Int                   num_functions,
                                 NALU_HYPRE_Int                  *dof_func,
                                 NALU_HYPRE_Int                   debug_flag,
                                 NALU_HYPRE_Real                  trunc_factor,
                                 NALU_HYPRE_Int                   max_elmts,
                                 NALU_HYPRE_Int                   add_weak_to_diag,
                                 hypre_ParCSRBlockMatrix   **P_ptr);



NALU_HYPRE_Int
hypre_BoomerAMGBuildBlockInterpRV( hypre_ParCSRBlockMatrix   *A,
                                   NALU_HYPRE_Int                 *CF_marker,
                                   hypre_ParCSRMatrix        *S,
                                   NALU_HYPRE_BigInt              *num_cpts_global,
                                   NALU_HYPRE_Int                  num_functions,
                                   NALU_HYPRE_Int                 *dof_func,
                                   NALU_HYPRE_Int                  debug_flag,
                                   NALU_HYPRE_Real                 trunc_factor,
                                   NALU_HYPRE_Int                  max_elmts,
                                   hypre_ParCSRBlockMatrix  **P_ptr);

NALU_HYPRE_Int
hypre_BoomerAMGBuildBlockInterpRV2( hypre_ParCSRBlockMatrix    *A,
                                    NALU_HYPRE_Int                  *CF_marker,
                                    hypre_ParCSRMatrix         *S,
                                    NALU_HYPRE_BigInt               *num_cpts_global,
                                    NALU_HYPRE_Int                   num_functions,
                                    NALU_HYPRE_Int                  *dof_func,
                                    NALU_HYPRE_Int                   debug_flag,
                                    NALU_HYPRE_Real                  trunc_factor,
                                    NALU_HYPRE_Int                   max_elmts,
                                    hypre_ParCSRBlockMatrix   **P_ptr);
NALU_HYPRE_Int
hypre_BoomerAMGBuildBlockInterpDiag( hypre_ParCSRBlockMatrix  *A,
                                     NALU_HYPRE_Int                *CF_marker,
                                     hypre_ParCSRMatrix       *S,
                                     NALU_HYPRE_BigInt             *num_cpts_global,
                                     NALU_HYPRE_Int                 num_functions,
                                     NALU_HYPRE_Int                *dof_func,
                                     NALU_HYPRE_Int                 debug_flag,
                                     NALU_HYPRE_Real                trunc_factor,
                                     NALU_HYPRE_Int                 max_elmts,
                                     NALU_HYPRE_Int                 add_weak_to_diag,
                                     hypre_ParCSRBlockMatrix  **P_ptr);

NALU_HYPRE_Int hypre_BoomerAMGBlockInterpTruncation( hypre_ParCSRBlockMatrix *P,
                                                NALU_HYPRE_Real trunc_factor,
                                                NALU_HYPRE_Int max_elements);


NALU_HYPRE_Int
hypre_BoomerAMGBuildBlockDirInterp( hypre_ParCSRBlockMatrix   *A,
                                    NALU_HYPRE_Int                 *CF_marker,
                                    hypre_ParCSRMatrix        *S,
                                    NALU_HYPRE_BigInt              *num_cpts_global,
                                    NALU_HYPRE_Int                  num_functions,
                                    NALU_HYPRE_Int                 *dof_func,
                                    NALU_HYPRE_Int                  debug_flag,
                                    NALU_HYPRE_Real                 trunc_factor,
                                    NALU_HYPRE_Int                  max_elmts,
                                    hypre_ParCSRBlockMatrix  **P_ptr);


NALU_HYPRE_Int  hypre_BoomerAMGBlockRelaxIF( hypre_ParCSRBlockMatrix *A,
                                        hypre_ParVector    *f,
                                        NALU_HYPRE_Int          *cf_marker,
                                        NALU_HYPRE_Int           relax_type,
                                        NALU_HYPRE_Int           relax_order,
                                        NALU_HYPRE_Int           cycle_type,
                                        NALU_HYPRE_Real          relax_weight,
                                        NALU_HYPRE_Real          omega,
                                        hypre_ParVector    *u,
                                        hypre_ParVector    *Vtemp );


NALU_HYPRE_Int  hypre_BoomerAMGBlockRelax( hypre_ParCSRBlockMatrix *A,
                                      hypre_ParVector    *f,
                                      NALU_HYPRE_Int          *cf_marker,
                                      NALU_HYPRE_Int           relax_type,
                                      NALU_HYPRE_Int           relax_points,
                                      NALU_HYPRE_Real          relax_weight,
                                      NALU_HYPRE_Real          omega,
                                      hypre_ParVector    *u,
                                      hypre_ParVector    *Vtemp );

NALU_HYPRE_Int
hypre_GetCommPkgBlockRTFromCommPkgBlockA( hypre_ParCSRBlockMatrix *RT,
                                          hypre_ParCSRBlockMatrix *A,
                                          NALU_HYPRE_Int *tmp_map_offd,
                                          NALU_HYPRE_BigInt *fine_to_coarse_offd);


hypre_ParCSRCommHandle *
hypre_ParCSRBlockCommHandleCreate(NALU_HYPRE_Int job, NALU_HYPRE_Int bnnz, hypre_ParCSRCommPkg *comm_pkg,
                                  void *send_data, void *recv_data );


NALU_HYPRE_Int
hypre_ParCSRBlockCommHandleDestroy(hypre_ParCSRCommHandle *comm_handle);



NALU_HYPRE_Int
hypre_BlockMatvecCommPkgCreate(hypre_ParCSRBlockMatrix *A);


NALU_HYPRE_Int
hypre_ParCSRBlockMatrixCreateAssumedPartition( hypre_ParCSRBlockMatrix *matrix);

NALU_HYPRE_Int
hypre_ParCSRBlockMatrixDestroyAssumedPartition(hypre_ParCSRBlockMatrix *matrix );



hypre_ParCSRMatrix *
hypre_ParCSRBlockMatrixConvertToParCSRMatrix(hypre_ParCSRBlockMatrix *matrix);


hypre_ParCSRBlockMatrix *
hypre_ParCSRBlockMatrixConvertFromParCSRMatrix(hypre_ParCSRMatrix *matrix,
                                               NALU_HYPRE_Int matrix_C_block_size );


NALU_HYPRE_Int
hypre_ParCSRBlockMatrixRAP(hypre_ParCSRBlockMatrix  *RT,
                           hypre_ParCSRBlockMatrix  *A,
                           hypre_ParCSRBlockMatrix  *P,
                           hypre_ParCSRBlockMatrix **RAP_ptr );

NALU_HYPRE_Int
hypre_ParCSRBlockMatrixSetNumNonzeros( hypre_ParCSRBlockMatrix *matrix);

NALU_HYPRE_Int
hypre_ParCSRBlockMatrixSetDNumNonzeros( hypre_ParCSRBlockMatrix *matrix);

NALU_HYPRE_Int
hypre_BoomerAMGBlockCreateNodalA(hypre_ParCSRBlockMatrix    *A,
                                 NALU_HYPRE_Int  option, NALU_HYPRE_Int diag_option,
                                 hypre_ParCSRMatrix   **AN_ptr);

hypre_ParVector *
hypre_ParVectorCreateFromBlock(MPI_Comm comm,
                               NALU_HYPRE_BigInt p_global_size,
                               NALU_HYPRE_BigInt *p_partitioning, NALU_HYPRE_Int block_size);

NALU_HYPRE_Int
hypre_ParCSRBlockMatrixMatvec(NALU_HYPRE_Complex alpha, hypre_ParCSRBlockMatrix *A,
                              hypre_ParVector *x, NALU_HYPRE_Complex beta,
                              hypre_ParVector *y);
NALU_HYPRE_Int
hypre_ParCSRBlockMatrixMatvecT( NALU_HYPRE_Complex           alpha,
                                hypre_ParCSRBlockMatrix *A,
                                hypre_ParVector         *x,
                                NALU_HYPRE_Complex            beta,
                                hypre_ParVector          *y);






void hypre_block_qsort( NALU_HYPRE_Int *v,
                        NALU_HYPRE_Complex *w,
                        NALU_HYPRE_Complex *blk_array,
                        NALU_HYPRE_Int block_size,
                        NALU_HYPRE_Int  left,
                        NALU_HYPRE_Int  right );


void hypre_swap_blk( NALU_HYPRE_Complex *v,
                     NALU_HYPRE_Int block_size,
                     NALU_HYPRE_Int  i,
                     NALU_HYPRE_Int  j );


#ifdef __cplusplus
}
#endif
#endif
