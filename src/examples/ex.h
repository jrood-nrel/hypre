/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/*--------------------------------------------------------------------------
 * Header file for examples
 *--------------------------------------------------------------------------*/

#ifndef NALU_HYPRE_EXAMPLES_INCLUDES
#define NALU_HYPRE_EXAMPLES_INCLUDES

#include <NALU_HYPRE_config.h>

#if defined(NALU_HYPRE_EXAMPLE_USING_CUDA)

#include <cuda_runtime.h>

#ifndef NALU_HYPRE_USING_UNIFIED_MEMORY
#error *** Running the examples on GPUs requires Unified Memory. Please reconfigure and rebuild with --enable-unified-memory ***
#endif

static inline void*
gpu_malloc(size_t size)
{
   void *ptr = NULL;
   cudaMallocManaged(&ptr, size, cudaMemAttachGlobal);
   return ptr;
}

static inline void*
gpu_calloc(size_t num, size_t size)
{
   void *ptr = NULL;
   cudaMallocManaged(&ptr, num * size, cudaMemAttachGlobal);
   cudaMemset(ptr, 0, num * size);
   return ptr;
}

#define malloc(size) gpu_malloc(size)
#define calloc(num, size) gpu_calloc(num, size)
#define free(ptr) ( cudaFree(ptr), ptr = NULL )
#endif /* #if defined(NALU_HYPRE_EXAMPLE_USING_CUDA) */
#endif /* #ifndef NALU_HYPRE_EXAMPLES_INCLUDES */

