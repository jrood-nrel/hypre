/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_nalu_hypre_parcsr_mv.h"

/*--------------------------------------------------------------------------
 * Test driver for PAR multivectors (under construction)
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
main( NALU_HYPRE_Int   argc,
      char *argv[] )
{
   nalu_hypre_ParVector   *vector1;
   nalu_hypre_ParVector   *vector2;
   nalu_hypre_ParVector   *tmp_vector;

   NALU_HYPRE_Int          num_procs, my_id;
   NALU_HYPRE_BigInt         global_size = 20;
   NALU_HYPRE_Int            local_size;
   NALU_HYPRE_BigInt         first_index;
   NALU_HYPRE_Int          num_vectors, vecstride, idxstride;
   NALU_HYPRE_Int            i, j;
   NALU_HYPRE_BigInt         *partitioning;
   NALU_HYPRE_Real           prod;
   NALU_HYPRE_Complex        *data, *data2;
   nalu_hypre_Vector *vector;
   nalu_hypre_Vector *local_vector;
   nalu_hypre_Vector *local_vector2;

   /* Initialize MPI */
   nalu_hypre_MPI_Init(&argc, &argv);

   nalu_hypre_MPI_Comm_size(nalu_hypre_MPI_COMM_WORLD, &num_procs );
   nalu_hypre_MPI_Comm_rank(nalu_hypre_MPI_COMM_WORLD, &my_id );

   nalu_hypre_printf(" my_id: %d num_procs: %d\n", my_id, num_procs);

   partitioning = NULL;
   num_vectors = 3;
   vector1 = nalu_hypre_ParMultiVectorCreate
             ( nalu_hypre_MPI_COMM_WORLD, global_size, partitioning, num_vectors );
   partitioning = nalu_hypre_ParVectorPartitioning(vector1);

   nalu_hypre_ParVectorInitialize(vector1);
   local_vector = nalu_hypre_ParVectorLocalVector(vector1);
   data = nalu_hypre_VectorData(local_vector);
   local_size = nalu_hypre_VectorSize(local_vector);
   vecstride = nalu_hypre_VectorVectorStride(local_vector);
   idxstride = nalu_hypre_VectorIndexStride(local_vector);
   first_index = partitioning[my_id];

   nalu_hypre_printf("vecstride=%i idxstride=%i local_size=%i num_vectors=%i",
                vecstride, idxstride, local_size, num_vectors );
   for (j = 0; j < num_vectors; ++j )
      for (i = 0; i < local_size; i++)
      {
         data[ j * vecstride + i * idxstride ] = (NALU_HYPRE_Int)first_index + i + 100 * j;
      }

   nalu_hypre_ParVectorPrint(vector1, "Vector");

   local_vector2 = nalu_hypre_SeqMultiVectorCreate( global_size, num_vectors );
   nalu_hypre_SeqVectorInitialize(local_vector2);
   data2 = nalu_hypre_VectorData(local_vector2);
   vecstride = nalu_hypre_VectorVectorStride(local_vector2);
   idxstride = nalu_hypre_VectorIndexStride(local_vector2);
   for (j = 0; j < num_vectors; ++j )
      for (i = 0; i < global_size; i++)
      {
         data2[ j * vecstride + i * idxstride ] = i + 100 * j;
      }

   /*   partitioning = nalu_hypre_CTAlloc(NALU_HYPRE_Int,4);
      partitioning[0] = 0;
      partitioning[1] = 10;
      partitioning[2] = 10;
      partitioning[3] = 20;
   */
   partitioning = nalu_hypre_CTAlloc(NALU_HYPRE_BigInt, 1 + num_procs, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_GeneratePartitioning( global_size, num_procs, &partitioning );

   vector2 = nalu_hypre_VectorToParVector(nalu_hypre_MPI_COMM_WORLD, local_vector2, partitioning);

   nalu_hypre_ParVectorPrint(vector2, "Convert");

   vector = nalu_hypre_ParVectorToVectorAll(vector2);

   /*-----------------------------------------------------------
    * Copy the vector into tmp_vector
    *-----------------------------------------------------------*/

   /* Read doesn't work for multivectors yet...
      tmp_vector = nalu_hypre_ParVectorRead(nalu_hypre_MPI_COMM_WORLD, "Convert");*/
   tmp_vector = nalu_hypre_ParMultiVectorCreate
                ( nalu_hypre_MPI_COMM_WORLD, global_size, partitioning, num_vectors );
   nalu_hypre_ParVectorInitialize( tmp_vector );
   nalu_hypre_ParVectorCopy( vector2, tmp_vector );
   /*
      tmp_vector = nalu_hypre_ParVectorCreate(nalu_hypre_MPI_COMM_WORLD,global_size,partitioning);
      nalu_hypre_ParVectorInitialize(tmp_vector);
      nalu_hypre_ParVectorCopy(vector1, tmp_vector);

      nalu_hypre_ParVectorPrint(tmp_vector,"Copy");
   */
   /*-----------------------------------------------------------
    * Scale tmp_vector
    *-----------------------------------------------------------*/

   nalu_hypre_ParVectorScale(2.0, tmp_vector);
   nalu_hypre_ParVectorPrint(tmp_vector, "Scale");

   /*-----------------------------------------------------------
    * Do an Axpy (2*vector - vector) = vector
    *-----------------------------------------------------------*/

   nalu_hypre_ParVectorAxpy(-1.0, vector1, tmp_vector);
   nalu_hypre_ParVectorPrint(tmp_vector, "Axpy");

   /*-----------------------------------------------------------
    * Do an inner product vector* tmp_vector
    *-----------------------------------------------------------*/

   prod = nalu_hypre_ParVectorInnerProd(vector1, tmp_vector);

   nalu_hypre_printf (" prod: %8.2f \n", prod);

   /*-----------------------------------------------------------
    * Finalize things
    *-----------------------------------------------------------*/

   nalu_hypre_ParVectorDestroy(vector1);
   nalu_hypre_ParVectorDestroy(vector2);
   nalu_hypre_ParVectorDestroy(tmp_vector);
   nalu_hypre_SeqVectorDestroy(local_vector2);
   if (vector) { nalu_hypre_SeqVectorDestroy(vector); }

   /* Finalize MPI */
   nalu_hypre_MPI_Finalize();

   return 0;
}
