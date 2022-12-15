/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * NALU_HYPRE.handle utility functions
 *
 *****************************************************************************/

#include "_nalu_hypre_utilities.h"
#include "_nalu_hypre_utilities.hpp"

/* GPU SpTrans */
NALU_HYPRE_Int
nalu_hypre_SetSpTransUseVendor( NALU_HYPRE_Int use_vendor )
{
#if defined(NALU_HYPRE_USING_GPU)
   nalu_hypre_HandleSpTransUseVendor(nalu_hypre_handle()) = use_vendor;
#endif
   return nalu_hypre_error_flag;
}

/* GPU SpMV */
NALU_HYPRE_Int
nalu_hypre_SetSpMVUseVendor( NALU_HYPRE_Int use_vendor )
{
#if defined(NALU_HYPRE_USING_GPU)
   nalu_hypre_HandleSpMVUseVendor(nalu_hypre_handle()) = use_vendor;
#endif
   return nalu_hypre_error_flag;
}

/* GPU SpGemm */
NALU_HYPRE_Int
nalu_hypre_SetSpGemmUseVendor( NALU_HYPRE_Int use_vendor )
{
#if defined(NALU_HYPRE_USING_GPU)
   nalu_hypre_HandleSpgemmUseVendor(nalu_hypre_handle()) = use_vendor;
#endif
   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_SetSpGemmAlgorithm( NALU_HYPRE_Int value )
{
#if defined(NALU_HYPRE_USING_GPU)
   if (value >= 1 && value <= 3)
   {
      nalu_hypre_HandleSpgemmAlgorithm(nalu_hypre_handle()) = value;
   }
   else
   {
      nalu_hypre_error_in_arg(1);
   }
#endif
   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_SetSpGemmBinned( NALU_HYPRE_Int value )
{
#if defined(NALU_HYPRE_USING_GPU)
   nalu_hypre_HandleSpgemmBinned(nalu_hypre_handle()) = value;
#endif
   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_SetSpGemmRownnzEstimateMethod( NALU_HYPRE_Int value )
{
#if defined(NALU_HYPRE_USING_GPU)
   if (value >= 1 && value <= 3)
   {
      nalu_hypre_HandleSpgemmRownnzEstimateMethod(nalu_hypre_handle()) = value;
   }
   else
   {
      nalu_hypre_error_in_arg(1);
   }
#endif
   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_SetSpGemmRownnzEstimateNSamples( NALU_HYPRE_Int value )
{
#if defined(NALU_HYPRE_USING_GPU)
   nalu_hypre_HandleSpgemmRownnzEstimateNsamples(nalu_hypre_handle()) = value;
#endif
   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_SetSpGemmRownnzEstimateMultFactor( NALU_HYPRE_Real value )
{
#if defined(NALU_HYPRE_USING_GPU)
   if (value > 0.0)
   {
      nalu_hypre_HandleSpgemmRownnzEstimateMultFactor(nalu_hypre_handle()) = value;
   }
   else
   {
      nalu_hypre_error_in_arg(1);
   }
#endif
   return nalu_hypre_error_flag;
}

/* GPU Rand */
NALU_HYPRE_Int
nalu_hypre_SetUseGpuRand( NALU_HYPRE_Int use_gpurand )
{
#if defined(NALU_HYPRE_USING_GPU)
   nalu_hypre_HandleUseGpuRand(nalu_hypre_handle()) = use_gpurand;
#endif
   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_SetGaussSeidelMethod( NALU_HYPRE_Int gs_method )
{
#if defined(NALU_HYPRE_USING_GPU)
   nalu_hypre_HandleDeviceGSMethod(nalu_hypre_handle()) = gs_method;
#endif
   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_SetUserDeviceMalloc(GPUMallocFunc func)
{
#if defined(NALU_HYPRE_USING_GPU)
   nalu_hypre_HandleUserDeviceMalloc(nalu_hypre_handle()) = func;
#endif
   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_SetUserDeviceMfree(GPUMfreeFunc func)
{
#if defined(NALU_HYPRE_USING_GPU)
   nalu_hypre_HandleUserDeviceMfree(nalu_hypre_handle()) = func;
#endif
   return nalu_hypre_error_flag;
}
