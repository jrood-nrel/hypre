/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * NALU_NALU_HYPRE.handle utility functions
 *
 *****************************************************************************/

#include "_hypre_utilities.h"
#include "_hypre_utilities.hpp"

/* GPU SpTrans */
NALU_HYPRE_Int
hypre_SetSpTransUseVendor( NALU_HYPRE_Int use_vendor )
{
#if defined(NALU_HYPRE_USING_GPU)
   hypre_HandleSpTransUseVendor(hypre_handle()) = use_vendor;
#endif
   return hypre_error_flag;
}

/* GPU SpMV */
NALU_HYPRE_Int
hypre_SetSpMVUseVendor( NALU_HYPRE_Int use_vendor )
{
#if defined(NALU_HYPRE_USING_GPU)
   hypre_HandleSpMVUseVendor(hypre_handle()) = use_vendor;
#endif
   return hypre_error_flag;
}

/* GPU SpGemm */
NALU_HYPRE_Int
hypre_SetSpGemmUseVendor( NALU_HYPRE_Int use_vendor )
{
#if defined(NALU_HYPRE_USING_GPU)
   hypre_HandleSpgemmUseVendor(hypre_handle()) = use_vendor;
#endif
   return hypre_error_flag;
}

NALU_HYPRE_Int
hypre_SetSpGemmAlgorithm( NALU_HYPRE_Int value )
{
#if defined(NALU_HYPRE_USING_GPU)
   if (value >= 1 && value <= 3)
   {
      hypre_HandleSpgemmAlgorithm(hypre_handle()) = value;
   }
   else
   {
      hypre_error_in_arg(1);
   }
#endif
   return hypre_error_flag;
}

NALU_HYPRE_Int
hypre_SetSpGemmBinned( NALU_HYPRE_Int value )
{
#if defined(NALU_HYPRE_USING_GPU)
   hypre_HandleSpgemmBinned(hypre_handle()) = value;
#endif
   return hypre_error_flag;
}

NALU_HYPRE_Int
hypre_SetSpGemmRownnzEstimateMethod( NALU_HYPRE_Int value )
{
#if defined(NALU_HYPRE_USING_GPU)
   if (value >= 1 && value <= 3)
   {
      hypre_HandleSpgemmRownnzEstimateMethod(hypre_handle()) = value;
   }
   else
   {
      hypre_error_in_arg(1);
   }
#endif
   return hypre_error_flag;
}

NALU_HYPRE_Int
hypre_SetSpGemmRownnzEstimateNSamples( NALU_HYPRE_Int value )
{
#if defined(NALU_HYPRE_USING_GPU)
   hypre_HandleSpgemmRownnzEstimateNsamples(hypre_handle()) = value;
#endif
   return hypre_error_flag;
}

NALU_HYPRE_Int
hypre_SetSpGemmRownnzEstimateMultFactor( NALU_HYPRE_Real value )
{
#if defined(NALU_HYPRE_USING_GPU)
   if (value > 0.0)
   {
      hypre_HandleSpgemmRownnzEstimateMultFactor(hypre_handle()) = value;
   }
   else
   {
      hypre_error_in_arg(1);
   }
#endif
   return hypre_error_flag;
}

/* GPU Rand */
NALU_HYPRE_Int
hypre_SetUseGpuRand( NALU_HYPRE_Int use_gpurand )
{
#if defined(NALU_HYPRE_USING_GPU)
   hypre_HandleUseGpuRand(hypre_handle()) = use_gpurand;
#endif
   return hypre_error_flag;
}

NALU_HYPRE_Int
hypre_SetGaussSeidelMethod( NALU_HYPRE_Int gs_method )
{
#if defined(NALU_HYPRE_USING_GPU)
   hypre_HandleDeviceGSMethod(hypre_handle()) = gs_method;
#endif
   return hypre_error_flag;
}

NALU_HYPRE_Int
hypre_SetUserDeviceMalloc(GPUMallocFunc func)
{
#if defined(NALU_HYPRE_USING_GPU)
   hypre_HandleUserDeviceMalloc(hypre_handle()) = func;
#endif
   return hypre_error_flag;
}

NALU_HYPRE_Int
hypre_SetUserDeviceMfree(GPUMfreeFunc func)
{
#if defined(NALU_HYPRE_USING_GPU)
   hypre_HandleUserDeviceMfree(hypre_handle()) = func;
#endif
   return hypre_error_flag;
}
