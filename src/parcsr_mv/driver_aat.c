/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_parcsr_mv.h"

/*--------------------------------------------------------------------------
 * Test driver for unstructured matrix interface , A * A^T
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
main( NALU_HYPRE_Int   argc,
      char *argv[] )
{
   hypre_ParCSRMatrix     *A;
   hypre_ParCSRMatrix     *C;
   hypre_CSRMatrix *As;
   NALU_HYPRE_BigInt *row_starts, *col_starts;
   NALU_HYPRE_Int num_procs, my_id;

   /* Initialize MPI */
   hypre_MPI_Init(&argc, &argv);

   hypre_MPI_Comm_size(hypre_MPI_COMM_WORLD, &num_procs);
   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &my_id);
   row_starts = NULL;
   col_starts = NULL;

   if (my_id == 0)
   {
      As = hypre_CSRMatrixRead("inpr");
      hypre_printf(" read input A\n");
   }
   A = hypre_CSRMatrixToParCSRMatrix(hypre_MPI_COMM_WORLD, As, row_starts,
                                     col_starts);
   row_starts = hypre_ParCSRMatrixRowStarts(A);
   col_starts = hypre_ParCSRMatrixColStarts(A);

   hypre_ParCSRMatrixPrint(A, "echo_A" );
   hypre_ParCSRMatrixPrintIJ(A, 0, 0, "echo_AIJ" );
   C = hypre_ParCSRAAt( A );
   hypre_ParCSRMatrixPrint(C, "result");
   hypre_ParCSRMatrixPrintIJ(C, 0, 0, "resultIJ");

   if (my_id == 0)
   {
      hypre_CSRMatrixDestroy(As);
   }
   hypre_ParCSRMatrixDestroy(A);
   hypre_ParCSRMatrixDestroy(C);

   hypre_MPI_Finalize();

   return 0;
}

