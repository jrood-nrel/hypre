/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#ifndef DEVICE_ALLOCATOR_H
#define DEVICE_ALLOCATOR_H

#if defined(NALU_HYPRE_USING_CUDA) || defined(NALU_HYPRE_USING_HIP)

/* C++ style memory allocator for the device using the abstract memory model */
struct nalu_hypre_device_allocator
{
   typedef char value_type;

   nalu_hypre_device_allocator()
   {
      // constructor
   }

   ~nalu_hypre_device_allocator()
   {
      // destructor
   }

   char *allocate(std::ptrdiff_t num_bytes)
   {
      return nalu_hypre_TAlloc(char, num_bytes, NALU_HYPRE_MEMORY_DEVICE);
   }

   void deallocate(char *ptr, size_t n)
   {
      nalu_hypre_TFree(ptr, NALU_HYPRE_MEMORY_DEVICE);
   }
};

#endif /* #if defined(NALU_HYPRE_USING_CUDA) || defined(NALU_HYPRE_USING_HIP) */

#endif
