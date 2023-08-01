/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * NALU_HYPRE.handle utility functions
 *
 *****************************************************************************/

#include "_nalu_hypre_utilities.h"

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SetSpTransUseVendor
 *--------------------------------------------------------------------------*/
NALU_HYPRE_Int
NALU_HYPRE_SetSpTransUseVendor( NALU_HYPRE_Int use_vendor )
{
   return nalu_hypre_SetSpTransUseVendor(use_vendor);
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SetSpMVUseVendor
 *--------------------------------------------------------------------------*/
NALU_HYPRE_Int
NALU_HYPRE_SetSpMVUseVendor( NALU_HYPRE_Int use_vendor )
{
   return nalu_hypre_SetSpMVUseVendor(use_vendor);
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SetSpGemmUseVendor
 *--------------------------------------------------------------------------*/
NALU_HYPRE_Int
NALU_HYPRE_SetSpGemmUseVendor( NALU_HYPRE_Int use_vendor )
{
   return nalu_hypre_SetSpGemmUseVendor(use_vendor);
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SetUseGpuRand
 *--------------------------------------------------------------------------*/
NALU_HYPRE_Int
NALU_HYPRE_SetUseGpuRand( NALU_HYPRE_Int use_gpu_rand )
{
   return nalu_hypre_SetUseGpuRand(use_gpu_rand);
}

