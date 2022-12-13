/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include "_hypre_utilities.h"

#ifdef NALU_HYPRE_USING_OPENMP

NALU_HYPRE_Int
hypre_NumThreads( )
{
   NALU_HYPRE_Int num_threads;

   num_threads = omp_get_max_threads();

   return num_threads;
}

/* This next function must be called from within a parallel region! */

NALU_HYPRE_Int
hypre_NumActiveThreads( )
{
   NALU_HYPRE_Int num_threads;

   num_threads = omp_get_num_threads();

   return num_threads;
}

/* This next function must be called from within a parallel region! */

NALU_HYPRE_Int
hypre_GetThreadNum( )
{
   NALU_HYPRE_Int my_thread_num;

   my_thread_num = omp_get_thread_num();

   return my_thread_num;
}

void
hypre_SetNumThreads(NALU_HYPRE_Int nt)
{
   omp_set_num_threads(nt);
}

#endif

/* This next function must be called from within a parallel region! */

void
hypre_GetSimpleThreadPartition( NALU_HYPRE_Int *begin, NALU_HYPRE_Int *end, NALU_HYPRE_Int n )
{
   NALU_HYPRE_Int num_threads = hypre_NumActiveThreads();
   NALU_HYPRE_Int my_thread_num = hypre_GetThreadNum();

   NALU_HYPRE_Int n_per_thread = (n + num_threads - 1) / num_threads;

   *begin = hypre_min(n_per_thread * my_thread_num, n);
   *end = hypre_min(*begin + n_per_thread, n);
}
