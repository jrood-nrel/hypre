/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Member functions for nalu_hypre_AuxParVector class.
 *
 *****************************************************************************/

#include "_nalu_hypre_IJ_mv.h"
#include "aux_par_vector.h"

/*--------------------------------------------------------------------------
 * nalu_hypre_AuxParVectorCreate
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_AuxParVectorCreate( nalu_hypre_AuxParVector **aux_vector_ptr)
{
   nalu_hypre_AuxParVector  *aux_vector;

   aux_vector = nalu_hypre_CTAlloc(nalu_hypre_AuxParVector, 1, NALU_HYPRE_MEMORY_HOST);

   /* set defaults */
   nalu_hypre_AuxParVectorMaxOffProcElmts(aux_vector)     = 0;
   nalu_hypre_AuxParVectorCurrentOffProcElmts(aux_vector) = 0;

   /* stash for setting or adding off processor values */
   nalu_hypre_AuxParVectorOffProcI(aux_vector)            = NULL;
   nalu_hypre_AuxParVectorOffProcData(aux_vector)         = NULL;
   nalu_hypre_AuxParVectorMemoryLocation(aux_vector)      = NALU_HYPRE_MEMORY_HOST;

#if defined(NALU_HYPRE_USING_CUDA) || defined(NALU_HYPRE_USING_HIP) || defined(NALU_HYPRE_USING_SYCL)
   nalu_hypre_AuxParVectorMaxStackElmts(aux_vector)       = 0;
   nalu_hypre_AuxParVectorCurrentStackElmts(aux_vector)   = 0;
   nalu_hypre_AuxParVectorStackI(aux_vector)              = NULL;
   nalu_hypre_AuxParVectorStackVoff(aux_vector)           = NULL;
   nalu_hypre_AuxParVectorStackData(aux_vector)           = NULL;
   nalu_hypre_AuxParVectorStackSorA(aux_vector)           = NULL;
   nalu_hypre_AuxParVectorUsrOffProcElmts(aux_vector)     = -1;
   nalu_hypre_AuxParVectorInitAllocFactor(aux_vector)     = 1.5;
   nalu_hypre_AuxParVectorGrowFactor(aux_vector)          = 2.0;
#endif

   *aux_vector_ptr = aux_vector;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_AuxParVectorDestroy
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_AuxParVectorDestroy( nalu_hypre_AuxParVector *aux_vector )
{
   if (aux_vector)
   {
      nalu_hypre_TFree(nalu_hypre_AuxParVectorOffProcI(aux_vector),    NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(nalu_hypre_AuxParVectorOffProcData(aux_vector), NALU_HYPRE_MEMORY_HOST);

#if defined(NALU_HYPRE_USING_CUDA) || defined(NALU_HYPRE_USING_HIP) || defined(NALU_HYPRE_USING_SYCL)
      NALU_HYPRE_MemoryLocation  memory_location = nalu_hypre_AuxParVectorMemoryLocation(aux_vector);

      nalu_hypre_TFree(nalu_hypre_AuxParVectorStackI(aux_vector),    memory_location);
      nalu_hypre_TFree(nalu_hypre_AuxParVectorStackVoff(aux_vector), memory_location);
      nalu_hypre_TFree(nalu_hypre_AuxParVectorStackData(aux_vector), memory_location);
      nalu_hypre_TFree(nalu_hypre_AuxParVectorStackSorA(aux_vector), memory_location);
#endif

      nalu_hypre_TFree(aux_vector, NALU_HYPRE_MEMORY_HOST);
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_AuxParVectorInitialize_v2
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_AuxParVectorInitialize_v2( nalu_hypre_AuxParVector   *aux_vector,
                                 NALU_HYPRE_MemoryLocation  memory_location )
{
   nalu_hypre_AuxParVectorMemoryLocation(aux_vector) = memory_location;

   if (memory_location == NALU_HYPRE_MEMORY_HOST)
   {
      /* CPU assembly */
      /* allocate stash for setting or adding off processor values */
      NALU_HYPRE_Int max_off_proc_elmts = nalu_hypre_AuxParVectorMaxOffProcElmts(aux_vector);
      if (max_off_proc_elmts > 0)
      {
         nalu_hypre_AuxParVectorOffProcI(aux_vector)    = nalu_hypre_CTAlloc(NALU_HYPRE_BigInt,  max_off_proc_elmts,
                                                                   NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_AuxParVectorOffProcData(aux_vector) = nalu_hypre_CTAlloc(NALU_HYPRE_Complex, max_off_proc_elmts,
                                                                   NALU_HYPRE_MEMORY_HOST);
      }
   }

   return nalu_hypre_error_flag;
}
