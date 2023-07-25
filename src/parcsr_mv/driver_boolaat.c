/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_nalu_hypre_parcsr_mv.h"

/*--------------------------------------------------------------------------
 * Test driver for unstructured Boolean matrix interface , A * A^T
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
main( NALU_HYPRE_Int   argc,
      char *argv[] )
{
   nalu_hypre_ParCSRBooleanMatrix     *A;
   nalu_hypre_ParCSRBooleanMatrix     *C;
   nalu_hypre_CSRBooleanMatrix *As;
   NALU_HYPRE_BigInt *row_starts, *col_starts;
   NALU_HYPRE_Int num_procs, my_id;

   /* Initialize MPI */
   nalu_hypre_MPI_Init(&argc, &argv);

   nalu_hypre_MPI_Comm_size(nalu_hypre_MPI_COMM_WORLD, &num_procs);
   nalu_hypre_MPI_Comm_rank(nalu_hypre_MPI_COMM_WORLD, &my_id);
   row_starts = NULL;
   col_starts = NULL;

   if (my_id == 0)
   {
      As = nalu_hypre_CSRBooleanMatrixRead("inpr");
      nalu_hypre_printf(" read input A\n");
   }
   A = nalu_hypre_CSRBooleanMatrixToParCSRBooleanMatrix(nalu_hypre_MPI_COMM_WORLD, As, row_starts,
                                                   col_starts);
   row_starts = nalu_hypre_ParCSRBooleanMatrix_Get_RowStarts(A);
   col_starts = nalu_hypre_ParCSRBooleanMatrix_Get_ColStarts(A);

   nalu_hypre_ParCSRBooleanMatrixPrint(A, "echo_A" );
   nalu_hypre_ParCSRBooleanMatrixPrintIJ(A, "echo_AIJ" );
   C = nalu_hypre_ParBooleanAAt( A );
   nalu_hypre_ParCSRBooleanMatrixPrint(C, "result");
   nalu_hypre_ParCSRBooleanMatrixPrintIJ(C, "resultIJ");

   if (my_id == 0)
   {
      nalu_hypre_CSRBooleanMatrixDestroy(As);
   }
   nalu_hypre_ParCSRBooleanMatrixDestroy(A);
   nalu_hypre_ParCSRBooleanMatrixDestroy(C);

   nalu_hypre_MPI_Finalize();

   return 0;
}

