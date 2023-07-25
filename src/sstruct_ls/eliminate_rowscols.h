/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#ifndef nalu_hypre_PARCSR_ELIMINATE_ROWSCOLS
#define nalu_hypre_PARCSR_ELIMINATE_ROWSCOLS

#ifdef __cplusplus
extern "C" {
#endif

/*
  Function:  nalu_hypre_ParCSRMatrixEliminateRowsCols

  This function eliminates the global rows and columns of a matrix
  A corresponding to given lists of sorted (!) local row numbers.

  The elimination is done as follows:

                / A_ii | A_ib \          / A_ii |  0   \
    (input) A = | -----+----- |   --->   | -----+----- | (output)
                \ A_bi | A_bb /          \   0  |  I   /
*/
NALU_HYPRE_Int nalu_hypre_ParCSRMatrixEliminateRowsCols (nalu_hypre_ParCSRMatrix *A,
                                               NALU_HYPRE_Int nrows_to_eliminate,
                                               NALU_HYPRE_Int *rows_to_eliminate);


/*
  Function:  nalu_hypre_CSRMatrixEliminateRowsColsDiag

  Eliminate the rows and columns of Adiag corresponding to the
  given sorted (!) list of rows. Put I on the eliminated diagonal.
*/
NALU_HYPRE_Int nalu_hypre_CSRMatrixEliminateRowsColsDiag (nalu_hypre_ParCSRMatrix *A,
                                                NALU_HYPRE_Int nrows_to_eliminate,
                                                NALU_HYPRE_Int *rows_to_eliminate);

/*
  Function:  nalu_hypre_CSRMatrixEliminateRowsOffd

  Eliminate the given list of rows of Aoffd.
*/
NALU_HYPRE_Int nalu_hypre_CSRMatrixEliminateRowsOffd (nalu_hypre_ParCSRMatrix *A,
                                            NALU_HYPRE_Int nrows_to_eliminate,
                                            NALU_HYPRE_Int *rows_to_eliminate);

/*
  Function:  nalu_hypre_CSRMatrixEliminateColsOffd

  Eliminate the given sorted (!) list of columns of Aoffd.
*/
NALU_HYPRE_Int nalu_hypre_CSRMatrixEliminateColsOffd (nalu_hypre_CSRMatrix *Aoffd,
                                            NALU_HYPRE_Int ncols_to_eliminate,
                                            NALU_HYPRE_Int *cols_to_eliminate);

#ifdef __cplusplus
}
#endif

#endif
