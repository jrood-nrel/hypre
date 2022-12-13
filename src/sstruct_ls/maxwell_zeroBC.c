/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_sstruct_ls.h"

NALU_HYPRE_Int
hypre_ParVectorZeroBCValues(hypre_ParVector *v,
                            NALU_HYPRE_Int       *rows,
                            NALU_HYPRE_Int        nrows)
{
   NALU_HYPRE_Int   ierr = 0;

   hypre_Vector *v_local = hypre_ParVectorLocalVector(v);

   hypre_SeqVectorZeroBCValues(v_local, rows, nrows);

   return ierr;
}

NALU_HYPRE_Int
hypre_SeqVectorZeroBCValues(hypre_Vector *v,
                            NALU_HYPRE_Int    *rows,
                            NALU_HYPRE_Int     nrows)
{
   NALU_HYPRE_Real  *vector_data = hypre_VectorData(v);
   NALU_HYPRE_Int      i;
   NALU_HYPRE_Int      ierr  = 0;

#if defined(NALU_HYPRE_USING_OPENMP)
   #pragma omp parallel for private(i) NALU_HYPRE_SMP_SCHEDULE
#endif
   for (i = 0; i < nrows; i++)
   {
      vector_data[rows[i]] = 0.0;
   }

   return ierr;
}

