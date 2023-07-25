/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_nalu_hypre_utilities.h"

#if defined(NALU_HYPRE_USING_MAGMA)

/*--------------------------------------------------------------------------
 * nalu_hypre_MagmaInitialize
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_MagmaInitialize(void)
{
   /* Initialize MAGMA */
   magma_init();

   /* Create device queue */
#if defined(NALU_HYPRE_USING_CUDA) || defined(NALU_HYPRE_USING_HIP)
   nalu_hypre_int device_id;

   nalu_hypre_GetDevice(&device_id);
   magma_queue_create((magma_int_t) device_id, &nalu_hypre_HandleMagmaQueue(nalu_hypre_handle()));
#endif

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_MagmaFinalize
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_MagmaFinalize(void)
{
   /* Finalize MAGMA */
   magma_finalize();

   return nalu_hypre_error_flag;
}

#endif /* NALU_HYPRE_USING_MAGMA */
