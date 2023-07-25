/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_nalu_hypre_parcsr_mv.h"

/*--------------------------------------------------------------------------
 * Test driver for unstructured matrix interface: matvec with multivectors
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
main( NALU_HYPRE_Int   argc,
      char *argv[] )
{
   nalu_hypre_CSRMatrix     *matrix;
   nalu_hypre_CSRMatrix     *matrix1;
   nalu_hypre_ParCSRMatrix  *par_matrix;
   nalu_hypre_Vector        *x_local;
   nalu_hypre_Vector        *y_local;
   nalu_hypre_Vector        *y2_local;
   nalu_hypre_ParVector     *x;
   nalu_hypre_ParVector     *x2;
   nalu_hypre_ParVector     *y;
   nalu_hypre_ParVector     *y2;

   NALU_HYPRE_Int          vecstride_x, idxstride_x, vecstride_y, idxstride_y;
   NALU_HYPRE_Int          num_procs, my_id;
   NALU_HYPRE_Int            local_size;
   NALU_HYPRE_Int          num_vectors;
   NALU_HYPRE_BigInt         global_num_rows, global_num_cols;
   NALU_HYPRE_BigInt         first_index;
   NALU_HYPRE_Int            i, j, ierr = 0;
   NALU_HYPRE_Complex        *data, *data2;
   NALU_HYPRE_BigInt         *row_starts, *col_starts;
   char         file_name[80];
   /* Initialize MPI */
   nalu_hypre_MPI_Init(&argc, &argv);

   nalu_hypre_MPI_Comm_size(nalu_hypre_MPI_COMM_WORLD, &num_procs);
   nalu_hypre_MPI_Comm_rank(nalu_hypre_MPI_COMM_WORLD, &my_id);

   nalu_hypre_printf(" my_id: %d num_procs: %d\n", my_id, num_procs);

   if (my_id == 0)
   {
      matrix = nalu_hypre_CSRMatrixRead("input");
      nalu_hypre_printf(" read input\n");
   }
   row_starts = NULL;
   col_starts = NULL;
   par_matrix = nalu_hypre_CSRMatrixToParCSRMatrix(nalu_hypre_MPI_COMM_WORLD, matrix,
                                              row_starts, col_starts);
   nalu_hypre_printf(" converted\n");

   matrix1 = nalu_hypre_ParCSRMatrixToCSRMatrixAll(par_matrix);

   nalu_hypre_sprintf(file_name, "matrix1.%d", my_id);

   if (matrix1) { nalu_hypre_CSRMatrixPrint(matrix1, file_name); }

   nalu_hypre_ParCSRMatrixPrint(par_matrix, "matrix");
   nalu_hypre_ParCSRMatrixPrintIJ(par_matrix, 0, 0, "matrixIJ");

   par_matrix = nalu_hypre_ParCSRMatrixRead(nalu_hypre_MPI_COMM_WORLD, "matrix");

   global_num_cols = nalu_hypre_ParCSRMatrixGlobalNumCols(par_matrix);
   nalu_hypre_printf(" global_num_cols %d\n", global_num_cols);
   global_num_rows = nalu_hypre_ParCSRMatrixGlobalNumRows(par_matrix);

   col_starts = nalu_hypre_ParCSRMatrixColStarts(par_matrix);
   first_index = col_starts[my_id];
   local_size = col_starts[my_id + 1] - first_index;

   num_vectors = 3;

   x = nalu_hypre_ParMultiVectorCreate( nalu_hypre_MPI_COMM_WORLD, global_num_cols,
                                   col_starts, num_vectors );
   nalu_hypre_ParVectorInitialize(x);
   x_local = nalu_hypre_ParVectorLocalVector(x);
   data = nalu_hypre_VectorData(x_local);
   vecstride_x = nalu_hypre_VectorVectorStride(x_local);
   idxstride_x = nalu_hypre_VectorIndexStride(x_local);
   for ( j = 0; j < num_vectors; ++j )
      for (i = 0; i < local_size; i++)
      {
         data[i * idxstride_x + j * vecstride_x] = (NALU_HYPRE_Int)first_index + i + 1 + 100 * j;
      }

   x2 = nalu_hypre_ParMultiVectorCreate( nalu_hypre_MPI_COMM_WORLD, global_num_cols,
                                    col_starts, num_vectors );
   nalu_hypre_ParVectorInitialize(x2);
   nalu_hypre_ParVectorSetConstantValues(x2, 2.0);

   row_starts = nalu_hypre_ParCSRMatrixRowStarts(par_matrix);
   first_index = row_starts[my_id];
   local_size = (NALU_HYPRE_Int)(row_starts[my_id + 1] - first_index);
   y = nalu_hypre_ParMultiVectorCreate( nalu_hypre_MPI_COMM_WORLD, global_num_rows,
                                   row_starts, num_vectors );
   nalu_hypre_ParVectorInitialize(y);
   y_local = nalu_hypre_ParVectorLocalVector(y);

   y2 = nalu_hypre_ParMultiVectorCreate( nalu_hypre_MPI_COMM_WORLD, global_num_rows,
                                    row_starts, num_vectors );
   nalu_hypre_ParVectorInitialize(y2);
   y2_local = nalu_hypre_ParVectorLocalVector(y2);
   data2 = nalu_hypre_VectorData(y2_local);
   vecstride_y = nalu_hypre_VectorVectorStride(y2_local);
   idxstride_y = nalu_hypre_VectorIndexStride(y2_local);

   for ( j = 0; j < num_vectors; ++j )
      for (i = 0; i < local_size; i++)
      {
         data2[i * idxstride_y + j * vecstride_y] = (NALU_HYPRE_Int)first_index + i + 1 + 100 * j;
      }

   nalu_hypre_ParVectorSetConstantValues(y, 1.0);
   nalu_hypre_printf(" initialized vectors, first_index=%i\n", first_index);

   nalu_hypre_ParVectorPrint(x, "vectorx");
   nalu_hypre_ParVectorPrint(y, "vectory");

   nalu_hypre_MatvecCommPkgCreate(par_matrix);

   nalu_hypre_ParCSRMatrixMatvec ( 1.0, par_matrix, x, 1.0, y);
   nalu_hypre_printf(" did matvec\n");

   nalu_hypre_ParVectorPrint(y, "result");

   ierr = nalu_hypre_ParCSRMatrixMatvecT ( 1.0, par_matrix, y2, 1.0, x2);
   nalu_hypre_printf(" did matvecT %d\n", ierr);

   nalu_hypre_ParVectorPrint(x2, "transp");

   nalu_hypre_ParCSRMatrixDestroy(par_matrix);
   nalu_hypre_ParVectorDestroy(x);
   nalu_hypre_ParVectorDestroy(x2);
   nalu_hypre_ParVectorDestroy(y);
   nalu_hypre_ParVectorDestroy(y2);
   if (my_id == 0) { nalu_hypre_CSRMatrixDestroy(matrix); }
   if (matrix1) { nalu_hypre_CSRMatrixDestroy(matrix1); }

   /* Finalize MPI */
   nalu_hypre_MPI_Finalize();

   return 0;
}
