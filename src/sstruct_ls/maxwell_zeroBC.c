/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_nalu_hypre_sstruct_ls.h"

NALU_HYPRE_Int
nalu_hypre_ParVectorZeroBCValues(nalu_hypre_ParVector *v,
                            NALU_HYPRE_Int       *rows,
                            NALU_HYPRE_Int        nrows)
{
   NALU_HYPRE_Int   ierr = 0;

   nalu_hypre_Vector *v_local = nalu_hypre_ParVectorLocalVector(v);

   nalu_hypre_SeqVectorZeroBCValues(v_local, rows, nrows);

   return ierr;
}

NALU_HYPRE_Int
nalu_hypre_SeqVectorZeroBCValues(nalu_hypre_Vector *v,
                            NALU_HYPRE_Int    *rows,
                            NALU_HYPRE_Int     nrows)
{
   NALU_HYPRE_Real  *vector_data = nalu_hypre_VectorData(v);
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

