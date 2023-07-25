/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Member functions for nalu_hypre_Vector class.
 *
 *****************************************************************************/

#include "_nalu_hypre_parcsr_mv.h"

/*--------------------------------------------------------------------------
 * nalu_hypre_ParVectorMassAxpy
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ParVectorMassAxpy( NALU_HYPRE_Complex    *alpha,
                         nalu_hypre_ParVector **x,
                         nalu_hypre_ParVector  *y,
                         NALU_HYPRE_Int         k,
                         NALU_HYPRE_Int         unroll )
{
   NALU_HYPRE_Int i;
   nalu_hypre_Vector **x_local;
   nalu_hypre_Vector *y_local = nalu_hypre_ParVectorLocalVector(y);
   x_local = nalu_hypre_TAlloc(nalu_hypre_Vector *, k, NALU_HYPRE_MEMORY_HOST);

   for (i = 0; i < k; i++)
   {
      x_local[i] = nalu_hypre_ParVectorLocalVector(x[i]);
   }

   nalu_hypre_SeqVectorMassAxpy( alpha, x_local, y_local, k, unroll);

   nalu_hypre_TFree(x_local, NALU_HYPRE_MEMORY_HOST);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ParVectorMassInnerProd
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ParVectorMassInnerProd( nalu_hypre_ParVector  *x,
                              nalu_hypre_ParVector **y,
                              NALU_HYPRE_Int         k,
                              NALU_HYPRE_Int         unroll,
                              NALU_HYPRE_Real       *result )
{
   MPI_Comm      comm    = nalu_hypre_ParVectorComm(x);
   nalu_hypre_Vector *x_local = nalu_hypre_ParVectorLocalVector(x);
   NALU_HYPRE_Real *local_result;
   NALU_HYPRE_Int i;
   nalu_hypre_Vector **y_local;
   y_local = nalu_hypre_TAlloc(nalu_hypre_Vector *, k, NALU_HYPRE_MEMORY_HOST);

   for (i = 0; i < k; i++)
   {
      y_local[i] = (nalu_hypre_Vector *) nalu_hypre_ParVectorLocalVector(y[i]);
   }

   local_result = nalu_hypre_CTAlloc(NALU_HYPRE_Real, k, NALU_HYPRE_MEMORY_HOST);

   nalu_hypre_SeqVectorMassInnerProd(x_local, y_local, k, unroll, local_result);

#ifdef NALU_HYPRE_PROFILE
   nalu_hypre_profile_times[NALU_HYPRE_TIMER_ID_ALL_REDUCE] -= nalu_hypre_MPI_Wtime();
#endif
   nalu_hypre_MPI_Allreduce(local_result, result, k, NALU_HYPRE_MPI_REAL,
                       nalu_hypre_MPI_SUM, comm);
#ifdef NALU_HYPRE_PROFILE
   nalu_hypre_profile_times[NALU_HYPRE_TIMER_ID_ALL_REDUCE] += nalu_hypre_MPI_Wtime();
#endif

   nalu_hypre_TFree(y_local, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(local_result, NALU_HYPRE_MEMORY_HOST);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ParVectorMassDotpTwo
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ParVectorMassDotpTwo ( nalu_hypre_ParVector  *x,
                             nalu_hypre_ParVector  *y,
                             nalu_hypre_ParVector **z,
                             NALU_HYPRE_Int         k,
                             NALU_HYPRE_Int         unroll,
                             NALU_HYPRE_Real       *result_x,
                             NALU_HYPRE_Real       *result_y )
{
   MPI_Comm      comm    = nalu_hypre_ParVectorComm(x);
   nalu_hypre_Vector *x_local = nalu_hypre_ParVectorLocalVector(x);
   nalu_hypre_Vector *y_local = nalu_hypre_ParVectorLocalVector(y);
   NALU_HYPRE_Real *local_result, *result;
   NALU_HYPRE_Int i;
   nalu_hypre_Vector **z_local;
   z_local = nalu_hypre_TAlloc(nalu_hypre_Vector*, k, NALU_HYPRE_MEMORY_HOST);

   for (i = 0; i < k; i++)
   {
      z_local[i] = (nalu_hypre_Vector *) nalu_hypre_ParVectorLocalVector(z[i]);
   }

   local_result = nalu_hypre_CTAlloc(NALU_HYPRE_Real, 2 * k, NALU_HYPRE_MEMORY_HOST);
   result = nalu_hypre_CTAlloc(NALU_HYPRE_Real, 2 * k, NALU_HYPRE_MEMORY_HOST);

   nalu_hypre_SeqVectorMassDotpTwo(x_local, y_local, z_local, k, unroll, &local_result[0],
                              &local_result[k]);

#ifdef NALU_HYPRE_PROFILE
   nalu_hypre_profile_times[NALU_HYPRE_TIMER_ID_ALL_REDUCE] -= nalu_hypre_MPI_Wtime();
#endif
   nalu_hypre_MPI_Allreduce(local_result, result, 2 * k, NALU_HYPRE_MPI_REAL,
                       nalu_hypre_MPI_SUM, comm);
#ifdef NALU_HYPRE_PROFILE
   nalu_hypre_profile_times[NALU_HYPRE_TIMER_ID_ALL_REDUCE] += nalu_hypre_MPI_Wtime();
#endif

   for (i = 0; i < k; i++)
   {
      result_x[i] = result[i];
      result_y[i] = result[k + i];
   }
   nalu_hypre_TFree(z_local, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(local_result, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(result, NALU_HYPRE_MEMORY_HOST);

   return nalu_hypre_error_flag;
}

