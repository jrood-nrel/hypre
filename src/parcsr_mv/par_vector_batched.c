/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Member functions for hypre_Vector class.
 *
 *****************************************************************************/

#include "_hypre_parcsr_mv.h"

/*--------------------------------------------------------------------------
 * hypre_ParVectorMassAxpy
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
hypre_ParVectorMassAxpy( NALU_HYPRE_Complex    *alpha,
                         hypre_ParVector **x,
                         hypre_ParVector  *y,
                         NALU_HYPRE_Int         k,
                         NALU_HYPRE_Int         unroll )
{
   NALU_HYPRE_Int i;
   hypre_Vector **x_local;
   hypre_Vector *y_local = hypre_ParVectorLocalVector(y);
   x_local = hypre_TAlloc(hypre_Vector *, k, NALU_HYPRE_MEMORY_HOST);

   for (i = 0; i < k; i++)
   {
      x_local[i] = hypre_ParVectorLocalVector(x[i]);
   }

   hypre_SeqVectorMassAxpy( alpha, x_local, y_local, k, unroll);

   hypre_TFree(x_local, NALU_HYPRE_MEMORY_HOST);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ParVectorMassInnerProd
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
hypre_ParVectorMassInnerProd( hypre_ParVector  *x,
                              hypre_ParVector **y,
                              NALU_HYPRE_Int         k,
                              NALU_HYPRE_Int         unroll,
                              NALU_HYPRE_Real       *result )
{
   MPI_Comm      comm    = hypre_ParVectorComm(x);
   hypre_Vector *x_local = hypre_ParVectorLocalVector(x);
   NALU_HYPRE_Real *local_result;
   NALU_HYPRE_Int i;
   hypre_Vector **y_local;
   y_local = hypre_TAlloc(hypre_Vector *, k, NALU_HYPRE_MEMORY_HOST);

   for (i = 0; i < k; i++)
   {
      y_local[i] = (hypre_Vector *) hypre_ParVectorLocalVector(y[i]);
   }

   local_result = hypre_CTAlloc(NALU_HYPRE_Real, k, NALU_HYPRE_MEMORY_HOST);

   hypre_SeqVectorMassInnerProd(x_local, y_local, k, unroll, local_result);

#ifdef NALU_HYPRE_PROFILE
   hypre_profile_times[NALU_HYPRE_TIMER_ID_ALL_REDUCE] -= hypre_MPI_Wtime();
#endif
   hypre_MPI_Allreduce(local_result, result, k, NALU_HYPRE_MPI_REAL,
                       hypre_MPI_SUM, comm);
#ifdef NALU_HYPRE_PROFILE
   hypre_profile_times[NALU_HYPRE_TIMER_ID_ALL_REDUCE] += hypre_MPI_Wtime();
#endif

   hypre_TFree(y_local, NALU_HYPRE_MEMORY_HOST);
   hypre_TFree(local_result, NALU_HYPRE_MEMORY_HOST);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ParVectorMassDotpTwo
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
hypre_ParVectorMassDotpTwo ( hypre_ParVector  *x,
                             hypre_ParVector  *y,
                             hypre_ParVector **z,
                             NALU_HYPRE_Int         k,
                             NALU_HYPRE_Int         unroll,
                             NALU_HYPRE_Real       *result_x,
                             NALU_HYPRE_Real       *result_y )
{
   MPI_Comm      comm    = hypre_ParVectorComm(x);
   hypre_Vector *x_local = hypre_ParVectorLocalVector(x);
   hypre_Vector *y_local = hypre_ParVectorLocalVector(y);
   NALU_HYPRE_Real *local_result, *result;
   NALU_HYPRE_Int i;
   hypre_Vector **z_local;
   z_local = hypre_TAlloc(hypre_Vector*, k, NALU_HYPRE_MEMORY_HOST);

   for (i = 0; i < k; i++)
   {
      z_local[i] = (hypre_Vector *) hypre_ParVectorLocalVector(z[i]);
   }

   local_result = hypre_CTAlloc(NALU_HYPRE_Real, 2 * k, NALU_HYPRE_MEMORY_HOST);
   result = hypre_CTAlloc(NALU_HYPRE_Real, 2 * k, NALU_HYPRE_MEMORY_HOST);

   hypre_SeqVectorMassDotpTwo(x_local, y_local, z_local, k, unroll, &local_result[0],
                              &local_result[k]);

#ifdef NALU_HYPRE_PROFILE
   hypre_profile_times[NALU_HYPRE_TIMER_ID_ALL_REDUCE] -= hypre_MPI_Wtime();
#endif
   hypre_MPI_Allreduce(local_result, result, 2 * k, NALU_HYPRE_MPI_REAL,
                       hypre_MPI_SUM, comm);
#ifdef NALU_HYPRE_PROFILE
   hypre_profile_times[NALU_HYPRE_TIMER_ID_ALL_REDUCE] += hypre_MPI_Wtime();
#endif

   for (i = 0; i < k; i++)
   {
      result_x[i] = result[i];
      result_y[i] = result[k + i];
   }
   hypre_TFree(z_local, NALU_HYPRE_MEMORY_HOST);
   hypre_TFree(local_result, NALU_HYPRE_MEMORY_HOST);
   hypre_TFree(result, NALU_HYPRE_MEMORY_HOST);

   return hypre_error_flag;
}

