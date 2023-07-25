/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_nalu_hypre_onedpl.hpp"
#include "seq_mv.h"
#include "_nalu_hypre_utilities.hpp"

#if defined(NALU_HYPRE_USING_GPU)

/* This function effectively does (in Matlab notation)
 *              C := alpha * A(:, a_colmap)
 *              C(num_b, :) += beta * B(:, b_colmap)
 *
 * if num_b != NULL: A is ma x n and B is mb x n. len(num_b) == mb.
 *                   All numbers in num_b must be in [0,...,ma-1]
 *
 * if num_b == NULL: C = alpha * A + beta * B. ma == mb
 *
 * if d_ja_map/d_jb_map == NULL, it is [0:n)
 */
NALU_HYPRE_Int
hypreDevice_CSRSpAdd( NALU_HYPRE_Int       ma, /* num of rows of A */
                      NALU_HYPRE_Int       mb, /* num of rows of B */
                      NALU_HYPRE_Int       n,  /* not used actually */
                      NALU_HYPRE_Int       nnzA,
                      NALU_HYPRE_Int       nnzB,
                      NALU_HYPRE_Int      *d_ia,
                      NALU_HYPRE_Int      *d_ja,
                      NALU_HYPRE_Complex   alpha,
                      NALU_HYPRE_Complex  *d_aa,
                      NALU_HYPRE_Int      *d_ja_map,
                      NALU_HYPRE_Int      *d_ib,
                      NALU_HYPRE_Int      *d_jb,
                      NALU_HYPRE_Complex   beta,
                      NALU_HYPRE_Complex  *d_ab,
                      NALU_HYPRE_Int      *d_jb_map,
                      NALU_HYPRE_Int      *d_num_b,
                      NALU_HYPRE_Int      *nnzC_out,
                      NALU_HYPRE_Int     **d_ic_out,
                      NALU_HYPRE_Int     **d_jc_out,
                      NALU_HYPRE_Complex **d_ac_out)
{
   /* trivial case */
   if (nnzA == 0 && nnzB == 0)
   {
      *d_ic_out = nalu_hypre_CTAlloc(NALU_HYPRE_Int, ma + 1, NALU_HYPRE_MEMORY_DEVICE);
      *d_jc_out = nalu_hypre_CTAlloc(NALU_HYPRE_Int,      0, NALU_HYPRE_MEMORY_DEVICE);
      *d_ac_out = nalu_hypre_CTAlloc(NALU_HYPRE_Complex,  0, NALU_HYPRE_MEMORY_DEVICE);
      *nnzC_out = 0;

      return nalu_hypre_error_flag;
   }

#ifdef NALU_HYPRE_PROFILE
   nalu_hypre_profile_times[NALU_HYPRE_TIMER_ID_SPADD] -= nalu_hypre_MPI_Wtime();
#endif

   /* expansion size */
   NALU_HYPRE_Int nnzT = nnzA + nnzB, nnzC;
   NALU_HYPRE_Int *d_it, *d_jt, *d_it_cp, *d_jt_cp, *d_ic, *d_jc;
   NALU_HYPRE_Complex *d_at, *d_at_cp, *d_ac;

   /* some trick here for memory alignment. maybe not worth it at all */
   NALU_HYPRE_Int align = 32;
   NALU_HYPRE_Int nnzT2 = (nnzT + align - 1) / align * align;
   char *work_mem = nalu_hypre_TAlloc(char, (4 * sizeof(NALU_HYPRE_Int) + 2 * sizeof(NALU_HYPRE_Complex)) * nnzT2,
                                 NALU_HYPRE_MEMORY_DEVICE);
   char *work_mem_saved = work_mem;

   //d_it = nalu_hypre_TAlloc(NALU_HYPRE_Int, nnzT, NALU_HYPRE_MEMORY_DEVICE);
   //d_jt = nalu_hypre_TAlloc(NALU_HYPRE_Int, nnzT, NALU_HYPRE_MEMORY_DEVICE);
   //d_at = nalu_hypre_TAlloc(NALU_HYPRE_Complex, nnzT, NALU_HYPRE_MEMORY_DEVICE);
   d_it = (NALU_HYPRE_Int *) work_mem;
   work_mem += sizeof(NALU_HYPRE_Int) * nnzT2;
   d_jt = (NALU_HYPRE_Int *) work_mem;
   work_mem += sizeof(NALU_HYPRE_Int) * nnzT2;
   d_at = (NALU_HYPRE_Complex *) work_mem;
   work_mem += sizeof(NALU_HYPRE_Complex) * nnzT2;

   /* expansion: j */
   if (d_ja_map)
   {
#if defined(NALU_HYPRE_USING_SYCL)
      hypreSycl_gather(d_ja, d_ja + nnzA, d_ja_map, d_jt);
#else
      NALU_HYPRE_THRUST_CALL(gather, d_ja, d_ja + nnzA, d_ja_map, d_jt);
#endif
   }
   else
   {
      nalu_hypre_TMemcpy(d_jt, d_ja, NALU_HYPRE_Int, nnzA, NALU_HYPRE_MEMORY_DEVICE, NALU_HYPRE_MEMORY_DEVICE);
   }
   if (d_jb_map)
   {
#if defined(NALU_HYPRE_USING_SYCL)
      hypreSycl_gather(d_jb, d_jb + nnzB, d_jb_map, d_jt + nnzA);
#else
      NALU_HYPRE_THRUST_CALL(gather, d_jb, d_jb + nnzB, d_jb_map, d_jt + nnzA);
#endif
   }
   else
   {
      nalu_hypre_TMemcpy(d_jt + nnzA, d_jb, NALU_HYPRE_Int, nnzB, NALU_HYPRE_MEMORY_DEVICE, NALU_HYPRE_MEMORY_DEVICE);
   }

   /* expansion: a */
   if (alpha == 1.0)
   {
      nalu_hypre_TMemcpy(d_at, d_aa, NALU_HYPRE_Complex, nnzA, NALU_HYPRE_MEMORY_DEVICE, NALU_HYPRE_MEMORY_DEVICE);
   }
   else
   {
      hypreDevice_ComplexScalen( d_aa, nnzA, d_at, alpha );
   }

   if (beta == 1.0)
   {
      nalu_hypre_TMemcpy(d_at + nnzA, d_ab, NALU_HYPRE_Complex, nnzB, NALU_HYPRE_MEMORY_DEVICE, NALU_HYPRE_MEMORY_DEVICE);
   }
   else
   {
      hypreDevice_ComplexScalen( d_ab, nnzB, d_at + nnzA, beta );
   }

   /* expansion: i */
   hypreDevice_CsrRowPtrsToIndices_v2(ma, nnzA, d_ia, d_it);
   if (d_num_b || mb <= 0)
   {
      hypreDevice_CsrRowPtrsToIndicesWithRowNum(mb, nnzB, d_ib, d_num_b, d_it + nnzA);
   }
   else
   {
      nalu_hypre_assert(ma == mb);
      hypreDevice_CsrRowPtrsToIndices_v2(mb, nnzB, d_ib, d_it + nnzA);
   }

   /* make copy of (it, jt, at), since reduce cannot be done in-place */
   //d_it_cp = nalu_hypre_TAlloc(NALU_HYPRE_Int,     nnzT, NALU_HYPRE_MEMORY_DEVICE);
   //d_jt_cp = nalu_hypre_TAlloc(NALU_HYPRE_Int,     nnzT, NALU_HYPRE_MEMORY_DEVICE);
   //d_at_cp = nalu_hypre_TAlloc(NALU_HYPRE_Complex, nnzT, NALU_HYPRE_MEMORY_DEVICE);
   d_it_cp = (NALU_HYPRE_Int *) work_mem;
   work_mem += sizeof(NALU_HYPRE_Int) * nnzT2;
   d_jt_cp = (NALU_HYPRE_Int *) work_mem;
   work_mem += sizeof(NALU_HYPRE_Int) * nnzT2;
   d_at_cp = (NALU_HYPRE_Complex *) work_mem;
   work_mem += sizeof(NALU_HYPRE_Complex) * nnzT2;

   nalu_hypre_assert( (size_t) (work_mem - work_mem_saved) == (4 * sizeof(NALU_HYPRE_Int) + 2 * sizeof(
                                                             NALU_HYPRE_Complex)) * ((size_t)nnzT2) );

   /* sort: lexicographical order (row, col): hypreDevice_StableSortByTupleKey */
   hypreDevice_StableSortByTupleKey(nnzT, d_it, d_jt, d_at, 0);

   /* compress */
   /* returns end: so nnz = end - start */
   nnzC = hypreDevice_ReduceByTupleKey(nnzT, d_it, d_jt, d_at, d_it_cp, d_jt_cp, d_at_cp);

   /* allocate final C */
   d_jc = nalu_hypre_TAlloc(NALU_HYPRE_Int,     nnzC, NALU_HYPRE_MEMORY_DEVICE);
   d_ac = nalu_hypre_TAlloc(NALU_HYPRE_Complex, nnzC, NALU_HYPRE_MEMORY_DEVICE);

   nalu_hypre_TMemcpy(d_jc, d_jt_cp, NALU_HYPRE_Int,     nnzC, NALU_HYPRE_MEMORY_DEVICE, NALU_HYPRE_MEMORY_DEVICE);
   nalu_hypre_TMemcpy(d_ac, d_at_cp, NALU_HYPRE_Complex, nnzC, NALU_HYPRE_MEMORY_DEVICE, NALU_HYPRE_MEMORY_DEVICE);

   /* convert into ic: row idx --> row ptrs */
   d_ic = hypreDevice_CsrRowIndicesToPtrs(ma, nnzC, d_it_cp);

#ifdef NALU_HYPRE_DEBUG
   NALU_HYPRE_Int tmp_nnzC;
   nalu_hypre_TMemcpy(&tmp_nnzC, &d_ic[ma], NALU_HYPRE_Int, 1, NALU_HYPRE_MEMORY_HOST, NALU_HYPRE_MEMORY_DEVICE);
   nalu_hypre_assert(nnzC == tmp_nnzC);
#endif

   /*
   nalu_hypre_TFree(d_it,    NALU_HYPRE_MEMORY_DEVICE);
   nalu_hypre_TFree(d_jt,    NALU_HYPRE_MEMORY_DEVICE);
   nalu_hypre_TFree(d_at,    NALU_HYPRE_MEMORY_DEVICE);
   nalu_hypre_TFree(d_it_cp, NALU_HYPRE_MEMORY_DEVICE);
   nalu_hypre_TFree(d_jt_cp, NALU_HYPRE_MEMORY_DEVICE);
   nalu_hypre_TFree(d_at_cp, NALU_HYPRE_MEMORY_DEVICE);
   */
   nalu_hypre_TFree(work_mem_saved, NALU_HYPRE_MEMORY_DEVICE);

   *nnzC_out = nnzC;
   *d_ic_out = d_ic;
   *d_jc_out = d_jc;
   *d_ac_out = d_ac;

#ifdef NALU_HYPRE_PROFILE
   nalu_hypre_profile_times[NALU_HYPRE_TIMER_ID_SPADD] += nalu_hypre_MPI_Wtime();
#endif

   return nalu_hypre_error_flag;
}

#endif // defined(NALU_HYPRE_USING_GPU)
