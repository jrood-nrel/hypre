/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_nalu_hypre_utilities.h"

void nalu_hypre_prefix_sum(NALU_HYPRE_Int *in_out, NALU_HYPRE_Int *sum, NALU_HYPRE_Int *workspace)
{
#ifdef NALU_HYPRE_USING_OPENMP
   NALU_HYPRE_Int my_thread_num = nalu_hypre_GetThreadNum();
   NALU_HYPRE_Int num_threads = nalu_hypre_NumActiveThreads();
   nalu_hypre_assert(1 == num_threads || omp_in_parallel());

   workspace[my_thread_num + 1] = *in_out;

   #pragma omp barrier
   #pragma omp master
   {
      NALU_HYPRE_Int i;
      workspace[0] = 0;
      for (i = 1; i < num_threads; i++)
      {
         workspace[i + 1] += workspace[i];
      }
      *sum = workspace[num_threads];
   }
   #pragma omp barrier

   *in_out = workspace[my_thread_num];
#else /* !NALU_HYPRE_USING_OPENMP */
   *sum = *in_out;
   *in_out = 0;

   workspace[0] = 0;
   workspace[1] = *sum;
#endif /* !NALU_HYPRE_USING_OPENMP */
}

void nalu_hypre_prefix_sum_pair(NALU_HYPRE_Int *in_out1, NALU_HYPRE_Int *sum1, NALU_HYPRE_Int *in_out2, NALU_HYPRE_Int *sum2,
                           NALU_HYPRE_Int *workspace)
{
#ifdef NALU_HYPRE_USING_OPENMP
   NALU_HYPRE_Int my_thread_num = nalu_hypre_GetThreadNum();
   NALU_HYPRE_Int num_threads = nalu_hypre_NumActiveThreads();
   nalu_hypre_assert(1 == num_threads || omp_in_parallel());

   workspace[(my_thread_num + 1) * 2] = *in_out1;
   workspace[(my_thread_num + 1) * 2 + 1] = *in_out2;

   #pragma omp barrier
   #pragma omp master
   {
      NALU_HYPRE_Int i;
      workspace[0] = 0;
      workspace[1] = 0;

      for (i = 1; i < num_threads; i++)
      {
         workspace[(i + 1) * 2] += workspace[i * 2];
         workspace[(i + 1) * 2 + 1] += workspace[i * 2 + 1];
      }
      *sum1 = workspace[num_threads * 2];
      *sum2 = workspace[num_threads * 2 + 1];
   }
   #pragma omp barrier

   *in_out1 = workspace[my_thread_num * 2];
   *in_out2 = workspace[my_thread_num * 2 + 1];
#else /* !NALU_HYPRE_USING_OPENMP */
   *sum1 = *in_out1;
   *sum2 = *in_out2;
   *in_out1 = 0;
   *in_out2 = 0;

   workspace[0] = 0;
   workspace[1] = 0;
   workspace[2] = *sum1;
   workspace[3] = *sum2;
#endif /* !NALU_HYPRE_USING_OPENMP */
}

void nalu_hypre_prefix_sum_triple(NALU_HYPRE_Int *in_out1, NALU_HYPRE_Int *sum1, NALU_HYPRE_Int *in_out2,
                             NALU_HYPRE_Int *sum2, NALU_HYPRE_Int *in_out3, NALU_HYPRE_Int *sum3, NALU_HYPRE_Int *workspace)
{
#ifdef NALU_HYPRE_USING_OPENMP
   NALU_HYPRE_Int my_thread_num = nalu_hypre_GetThreadNum();
   NALU_HYPRE_Int num_threads = nalu_hypre_NumActiveThreads();
   nalu_hypre_assert(1 == num_threads || omp_in_parallel());

   workspace[(my_thread_num + 1) * 3] = *in_out1;
   workspace[(my_thread_num + 1) * 3 + 1] = *in_out2;
   workspace[(my_thread_num + 1) * 3 + 2] = *in_out3;

   #pragma omp barrier
   #pragma omp master
   {
      NALU_HYPRE_Int i;
      workspace[0] = 0;
      workspace[1] = 0;
      workspace[2] = 0;

      for (i = 1; i < num_threads; i++)
      {
         workspace[(i + 1) * 3] += workspace[i * 3];
         workspace[(i + 1) * 3 + 1] += workspace[i * 3 + 1];
         workspace[(i + 1) * 3 + 2] += workspace[i * 3 + 2];
      }
      *sum1 = workspace[num_threads * 3];
      *sum2 = workspace[num_threads * 3 + 1];
      *sum3 = workspace[num_threads * 3 + 2];
   }
   #pragma omp barrier

   *in_out1 = workspace[my_thread_num * 3];
   *in_out2 = workspace[my_thread_num * 3 + 1];
   *in_out3 = workspace[my_thread_num * 3 + 2];
#else /* !NALU_HYPRE_USING_OPENMP */
   *sum1 = *in_out1;
   *sum2 = *in_out2;
   *sum3 = *in_out3;
   *in_out1 = 0;
   *in_out2 = 0;
   *in_out3 = 0;

   workspace[0] = 0;
   workspace[1] = 0;
   workspace[2] = 0;
   workspace[3] = *sum1;
   workspace[4] = *sum2;
   workspace[5] = *sum3;
#endif /* !NALU_HYPRE_USING_OPENMP */
}

void nalu_hypre_prefix_sum_multiple(NALU_HYPRE_Int *in_out, NALU_HYPRE_Int *sum, NALU_HYPRE_Int n, NALU_HYPRE_Int *workspace)
{
   NALU_HYPRE_Int i;
#ifdef NALU_HYPRE_USING_OPENMP
   NALU_HYPRE_Int my_thread_num = nalu_hypre_GetThreadNum();
   NALU_HYPRE_Int num_threads = nalu_hypre_NumActiveThreads();
   nalu_hypre_assert(1 == num_threads || omp_in_parallel());

   for (i = 0; i < n; i++)
   {
      workspace[(my_thread_num + 1)*n + i] = in_out[i];
   }

   #pragma omp barrier
   #pragma omp master
   {
      NALU_HYPRE_Int t;
      for (i = 0; i < n; i++)
      {
         workspace[i] = 0;
      }

      // assuming n is not so big, we don't parallelize this loop
      for (t = 1; t < num_threads; t++)
      {
         for (i = 0; i < n; i++)
         {
            workspace[(t + 1)*n + i] += workspace[t * n + i];
         }
      }

      for (i = 0; i < n; i++)
      {
         sum[i] = workspace[num_threads * n + i];
      }
   }
   #pragma omp barrier

   for (i = 0; i < n; i++)
   {
      in_out[i] = workspace[my_thread_num * n + i];
   }
#else /* !NALU_HYPRE_USING_OPENMP */
   for (i = 0; i < n; i++)
   {
      sum[i] = in_out[i];
      in_out[i] = 0;

      workspace[i] = 0;
      workspace[n + i] = sum[i];
   }
#endif /* !NALU_HYPRE_USING_OPENMP */
}
