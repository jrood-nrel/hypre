/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_nalu_hypre_parcsr_mv.h"

/*--------------------------------------------------------------------------
 * Test driver for unstructured matrix interface , A * A^T
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
main( NALU_HYPRE_Int   argc,
      char *argv[] )
{
   nalu_hypre_ParCSRMatrix     *A;
   nalu_hypre_ParCSRMatrix     *C;
   nalu_hypre_CSRMatrix *As;
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
      As = nalu_hypre_CSRMatrixRead("inpr");
      nalu_hypre_printf(" read input A\n");
   }
   A = nalu_hypre_CSRMatrixToParCSRMatrix(nalu_hypre_MPI_COMM_WORLD, As, row_starts,
                                     col_starts);
   row_starts = nalu_hypre_ParCSRMatrixRowStarts(A);
   col_starts = nalu_hypre_ParCSRMatrixColStarts(A);

   nalu_hypre_ParCSRMatrixPrint(A, "echo_A" );
   nalu_hypre_ParCSRMatrixPrintIJ(A, 0, 0, "echo_AIJ" );
   C = nalu_hypre_ParCSRAAt( A );
   nalu_hypre_ParCSRMatrixPrint(C, "result");
   nalu_hypre_ParCSRMatrixPrintIJ(C, 0, 0, "resultIJ");

   if (my_id == 0)
   {
      nalu_hypre_CSRMatrixDestroy(As);
   }
   nalu_hypre_ParCSRMatrixDestroy(A);
   nalu_hypre_ParCSRMatrixDestroy(C);

   nalu_hypre_MPI_Finalize();

   return 0;
}

