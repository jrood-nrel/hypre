/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Memory management utilities
 *
 *****************************************************************************/

#include "_nalu_hypre_utilities.h"
#include "_nalu_hypre_utilities.hpp"

#if defined(NALU_HYPRE_USE_UMALLOC)
#undef NALU_HYPRE_USE_UMALLOC
#endif

/******************************************************************************
 *
 * Helper routines
 *
 *****************************************************************************/

NALU_HYPRE_Int
nalu_hypre_GetMemoryLocationName(nalu_hypre_MemoryLocation  memory_location,
                            char                 *memory_location_name)
{
   if (memory_location == nalu_hypre_MEMORY_HOST)
   {
      sprintf(memory_location_name, "%s", "HOST");
   }
   else if (memory_location == nalu_hypre_MEMORY_HOST_PINNED)
   {
      sprintf(memory_location_name, "%s", "HOST PINNED");
   }
   else if (memory_location == nalu_hypre_MEMORY_DEVICE)
   {
      sprintf(memory_location_name, "%s", "DEVICE");
   }
   else if (memory_location == nalu_hypre_MEMORY_UNIFIED)
   {
      sprintf(memory_location_name, "%s", "UNIFIED");
   }
   else
   {
      sprintf(memory_location_name, "%s", "");
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_OutOfMemory
 *--------------------------------------------------------------------------*/
static inline void
nalu_hypre_OutOfMemory(size_t size)
{
   nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_MEMORY, "Out of memory trying to allocate too many bytes\n");
   nalu_hypre_assert(0);
   fflush(stdout);
}

static inline void
nalu_hypre_WrongMemoryLocation()
{
   nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_MEMORY, "Unrecognized nalu_hypre_MemoryLocation\n");
   nalu_hypre_assert(0);
   fflush(stdout);
}

void
nalu_hypre_CheckMemoryLocation(void *ptr, nalu_hypre_MemoryLocation location)
{
#if defined(NALU_HYPRE_DEBUG)
#if defined(NALU_HYPRE_USING_CUDA) || defined(NALU_HYPRE_USING_HIP) || defined(NALU_HYPRE_USING_SYCL)
   if (!ptr)
   {
      return;
   }

   nalu_hypre_MemoryLocation location_ptr;
   nalu_hypre_GetPointerLocation(ptr, &location_ptr);
   /* do not use nalu_hypre_assert, which has alloc and free;
    * will create an endless loop otherwise */
   assert(location == location_ptr);
#endif
#endif
}

/*==========================================================================
 * Physical memory location (nalu_hypre_MemoryLocation) interface
 *==========================================================================*/

/*--------------------------------------------------------------------------
 * Memset
 *--------------------------------------------------------------------------*/
static inline void
nalu_hypre_HostMemset(void *ptr, NALU_HYPRE_Int value, size_t num)
{
   memset(ptr, value, num);
}

static inline void
nalu_hypre_DeviceMemset(void *ptr, NALU_HYPRE_Int value, size_t num)
{
#if defined(NALU_HYPRE_USING_DEVICE_OPENMP)
#if defined(NALU_HYPRE_DEVICE_OPENMP_ALLOC)
   #pragma omp target teams distribute parallel for is_device_ptr(ptr)
   for (size_t i = 0; i < num; i++)
   {
      ((unsigned char *) ptr)[i] = (unsigned char) value;
   }
#else
   memset(ptr, value, num);
   NALU_HYPRE_OMPOffload(nalu_hypre__offload_device_num, ptr, num, "update", "to");
#endif
   /* NALU_HYPRE_CUDA_CALL( cudaDeviceSynchronize() ); */
#endif

#if defined(NALU_HYPRE_USING_CUDA)
   NALU_HYPRE_CUDA_CALL( cudaMemset(ptr, value, num) );
#endif

#if defined(NALU_HYPRE_USING_HIP)
   NALU_HYPRE_HIP_CALL( hipMemset(ptr, value, num) );
#endif

#if defined(NALU_HYPRE_USING_SYCL)
   NALU_HYPRE_SYCL_CALL( (nalu_hypre_HandleComputeStream(nalu_hypre_handle()))->memset(ptr, value, num).wait() );
#endif
}

static inline void
nalu_hypre_UnifiedMemset(void *ptr, NALU_HYPRE_Int value, size_t num)
{
#if defined(NALU_HYPRE_USING_DEVICE_OPENMP)
#if defined(NALU_HYPRE_DEVICE_OPENMP_ALLOC)
   #pragma omp target teams distribute parallel for is_device_ptr(ptr)
   for (size_t i = 0; i < num; i++)
   {
      ((unsigned char *) ptr)[i] = (unsigned char) value;
   }
#else
   memset(ptr, value, num);
   NALU_HYPRE_OMPOffload(nalu_hypre__offload_device_num, ptr, num, "update", "to");
#endif
   /* NALU_HYPRE_CUDA_CALL( cudaDeviceSynchronize() ); */
#endif

#if defined(NALU_HYPRE_USING_CUDA)
   NALU_HYPRE_CUDA_CALL( cudaMemset(ptr, value, num) );
#endif

#if defined(NALU_HYPRE_USING_HIP)
   NALU_HYPRE_HIP_CALL( hipMemset(ptr, value, num) );
#endif

#if defined(NALU_HYPRE_USING_SYCL)
   NALU_HYPRE_SYCL_CALL( (nalu_hypre_HandleComputeStream(nalu_hypre_handle()))->memset(ptr, value, num).wait() );
#endif
}

/*--------------------------------------------------------------------------
 * Memprefetch
 *--------------------------------------------------------------------------*/
static inline void
nalu_hypre_UnifiedMemPrefetch(void *ptr, size_t size, nalu_hypre_MemoryLocation location)
{
   if (!size)
   {
      return;
   }

   nalu_hypre_CheckMemoryLocation(ptr, nalu_hypre_MEMORY_UNIFIED);

#if defined(NALU_HYPRE_USING_CUDA)
   if (location == nalu_hypre_MEMORY_DEVICE)
   {
      NALU_HYPRE_CUDA_CALL( cudaMemPrefetchAsync(ptr, size, nalu_hypre_HandleDevice(nalu_hypre_handle()),
                                            nalu_hypre_HandleComputeStream(nalu_hypre_handle())) );
   }
   else if (location == nalu_hypre_MEMORY_HOST)
   {
      NALU_HYPRE_CUDA_CALL( cudaMemPrefetchAsync(ptr, size, cudaCpuDeviceId,
                                            nalu_hypre_HandleComputeStream(nalu_hypre_handle())) );
   }
#endif

#if defined(NALU_HYPRE_USING_HIP)
   // Not currently implemented for HIP, but leaving place holder
   /*
    *if (location == nalu_hypre_MEMORY_DEVICE)
    *{
    *  NALU_HYPRE_HIP_CALL( hipMemPrefetchAsync(ptr, size, nalu_hypre_HandleDevice(nalu_hypre_handle()),
    *                   nalu_hypre_HandleComputeStream(nalu_hypre_handle())) );
    *}
    *else if (location == nalu_hypre_MEMORY_HOST)
    *{
    *   NALU_HYPRE_CUDA_CALL( hipMemPrefetchAsync(ptr, size, cudaCpuDeviceId,
    *                    nalu_hypre_HandleComputeStream(nalu_hypre_handle())) );
    *}
    */
#endif

#if defined(NALU_HYPRE_USING_SYCL)
   if (location == nalu_hypre_MEMORY_DEVICE)
   {
      /* WM: todo - the call below seems like it may occasionally result in an error: */
      /*     Native API returns: -997 (The plugin has emitted a backend specific error) */
      /*     or a seg fault. On the other hand, removing this line can also cause the code 
       *     to hang (or run excessively slow?). */
      /* NALU_HYPRE_SYCL_CALL( nalu_hypre_HandleComputeStream(nalu_hypre_handle())->prefetch(ptr, size).wait() ); */
   }
#endif
}

/*--------------------------------------------------------------------------
 * Malloc
 *--------------------------------------------------------------------------*/
static inline void *
nalu_hypre_HostMalloc(size_t size, NALU_HYPRE_Int zeroinit)
{
   void *ptr = NULL;

#if defined(NALU_HYPRE_USING_UMPIRE_HOST)
   nalu_hypre_umpire_host_pooled_allocate(&ptr, size);
   if (zeroinit)
   {
      memset(ptr, 0, size);
   }
#else
   if (zeroinit)
   {
      ptr = calloc(size, 1);
   }
   else
   {
      ptr = malloc(size);
   }
#endif

   return ptr;
}

static inline void *
nalu_hypre_DeviceMalloc(size_t size, NALU_HYPRE_Int zeroinit)
{
   void *ptr = NULL;

   if ( nalu_hypre_HandleUserDeviceMalloc(nalu_hypre_handle()) )
   {
      nalu_hypre_HandleUserDeviceMalloc(nalu_hypre_handle())(&ptr, size);
   }
   else
   {
#if defined(NALU_HYPRE_USING_UMPIRE_DEVICE)
      nalu_hypre_umpire_device_pooled_allocate(&ptr, size);
#else

#if defined(NALU_HYPRE_USING_DEVICE_OPENMP)
#if defined(NALU_HYPRE_DEVICE_OPENMP_ALLOC)
      ptr = omp_target_alloc(size, nalu_hypre__offload_device_num);
#else
      ptr = malloc(size + sizeof(size_t));
      size_t *sp = (size_t*) ptr;
      sp[0] = size;
      ptr = (void *) (&sp[1]);
      NALU_HYPRE_OMPOffload(nalu_hypre__offload_device_num, ptr, size, "enter", "alloc");
#endif
#endif

#if defined(NALU_HYPRE_USING_CUDA)
#if defined(NALU_HYPRE_USING_DEVICE_POOL)
      NALU_HYPRE_CUDA_CALL( nalu_hypre_CachingMallocDevice(&ptr, size) );
#elif defined(NALU_HYPRE_USING_DEVICE_MALLOC_ASYNC)
      NALU_HYPRE_CUDA_CALL( cudaMallocAsync(&ptr, size, NULL) );
#else
      NALU_HYPRE_CUDA_CALL( cudaMalloc(&ptr, size) );
#endif
#endif

#if defined(NALU_HYPRE_USING_HIP)
      NALU_HYPRE_HIP_CALL( hipMalloc(&ptr, size) );
#endif

#if defined(NALU_HYPRE_USING_SYCL)
      ptr = (void *)sycl::malloc_device(size, *(nalu_hypre_HandleComputeStream(nalu_hypre_handle())));
#endif

#endif /* #if defined(NALU_HYPRE_USING_UMPIRE_DEVICE) */
   }

   if (ptr && zeroinit)
   {
      nalu_hypre_DeviceMemset(ptr, 0, size);
   }

   return ptr;
}

static inline void *
nalu_hypre_UnifiedMalloc(size_t size, NALU_HYPRE_Int zeroinit)
{
   void *ptr = NULL;

#if defined(NALU_HYPRE_USING_UMPIRE_UM)
   nalu_hypre_umpire_um_pooled_allocate(&ptr, size);
#else

#if defined(NALU_HYPRE_USING_DEVICE_OPENMP)
#if defined(NALU_HYPRE_DEVICE_OPENMP_ALLOC)
   ptr = omp_target_alloc(size, nalu_hypre__offload_device_num);
#else
   ptr = malloc(size + sizeof(size_t));
   size_t *sp = (size_t*) ptr;
   sp[0] = size;
   ptr = (void *) (&sp[1]);
   NALU_HYPRE_OMPOffload(nalu_hypre__offload_device_num, ptr, size, "enter", "alloc");
#endif
#endif

#if defined(NALU_HYPRE_USING_CUDA)
#if defined(NALU_HYPRE_USING_DEVICE_POOL)
   NALU_HYPRE_CUDA_CALL( nalu_hypre_CachingMallocManaged(&ptr, size) );
#else
   NALU_HYPRE_CUDA_CALL( cudaMallocManaged(&ptr, size, cudaMemAttachGlobal) );
#endif
#endif

#if defined(NALU_HYPRE_USING_HIP)
   NALU_HYPRE_HIP_CALL( hipMallocManaged(&ptr, size, hipMemAttachGlobal) );
#endif

#if defined(NALU_HYPRE_USING_SYCL)
   NALU_HYPRE_SYCL_CALL( ptr = (void *)sycl::malloc_shared(size,
                                                      *(nalu_hypre_HandleComputeStream(nalu_hypre_handle()))) );
#endif

#endif /* #if defined(NALU_HYPRE_USING_UMPIRE_UM) */

   /* prefecth to device */
   if (ptr)
   {
      nalu_hypre_UnifiedMemPrefetch(ptr, size, nalu_hypre_MEMORY_DEVICE);
   }

   if (ptr && zeroinit)
   {
      nalu_hypre_UnifiedMemset(ptr, 0, size);
   }

   return ptr;
}

static inline void *
nalu_hypre_HostPinnedMalloc(size_t size, NALU_HYPRE_Int zeroinit)
{
   void *ptr = NULL;

#if defined(NALU_HYPRE_USING_UMPIRE_PINNED)
   nalu_hypre_umpire_pinned_pooled_allocate(&ptr, size);
#else

#if defined(NALU_HYPRE_USING_CUDA)
   NALU_HYPRE_CUDA_CALL( cudaMallocHost(&ptr, size) );
#endif

#if defined(NALU_HYPRE_USING_HIP)
   NALU_HYPRE_HIP_CALL( hipHostMalloc(&ptr, size) );
#endif

#if defined(NALU_HYPRE_USING_SYCL)
   NALU_HYPRE_SYCL_CALL( ptr = (void *)sycl::malloc_host(size,
                                                    *(nalu_hypre_HandleComputeStream(nalu_hypre_handle()))) );
#endif

#endif /* #if defined(NALU_HYPRE_USING_UMPIRE_PINNED) */

   if (ptr && zeroinit)
   {
      nalu_hypre_HostMemset(ptr, 0, size);
   }

   return ptr;
}

static inline void *
nalu_hypre_MAlloc_core(size_t size, NALU_HYPRE_Int zeroinit, nalu_hypre_MemoryLocation location)
{
   if (size == 0)
   {
      return NULL;
   }

   void *ptr = NULL;

   switch (location)
   {
      case nalu_hypre_MEMORY_HOST :
         ptr = nalu_hypre_HostMalloc(size, zeroinit);
         break;
      case nalu_hypre_MEMORY_DEVICE :
         ptr = nalu_hypre_DeviceMalloc(size, zeroinit);
         break;
      case nalu_hypre_MEMORY_UNIFIED :
         ptr = nalu_hypre_UnifiedMalloc(size, zeroinit);
         break;
      case nalu_hypre_MEMORY_HOST_PINNED :
         ptr = nalu_hypre_HostPinnedMalloc(size, zeroinit);
         break;
      default :
         nalu_hypre_WrongMemoryLocation();
   }

   if (!ptr)
   {
      nalu_hypre_OutOfMemory(size);
      nalu_hypre_MPI_Abort(nalu_hypre_MPI_COMM_WORLD, -1);
   }

   return ptr;
}

void *
_nalu_hypre_MAlloc(size_t size, nalu_hypre_MemoryLocation location)
{
   return nalu_hypre_MAlloc_core(size, 0, location);
}

/*--------------------------------------------------------------------------
 * Free
 *--------------------------------------------------------------------------*/
static inline void
nalu_hypre_HostFree(void *ptr)
{
#if defined(NALU_HYPRE_USING_UMPIRE_HOST)
   nalu_hypre_umpire_host_pooled_free(ptr);
#else
   free(ptr);
#endif
}

static inline void
nalu_hypre_DeviceFree(void *ptr)
{
   if ( nalu_hypre_HandleUserDeviceMfree(nalu_hypre_handle()) )
   {
      nalu_hypre_HandleUserDeviceMfree(nalu_hypre_handle())(ptr);
   }
   else
   {
#if defined(NALU_HYPRE_USING_UMPIRE_DEVICE)
      nalu_hypre_umpire_device_pooled_free(ptr);
#else

#if defined(NALU_HYPRE_USING_DEVICE_OPENMP)
#if defined(NALU_HYPRE_DEVICE_OPENMP_ALLOC)
      omp_target_free(ptr, nalu_hypre__offload_device_num);
#else
      NALU_HYPRE_OMPOffload(nalu_hypre__offload_device_num, ptr, ((size_t *) ptr)[-1], "exit", "delete");
#endif
#endif

#if defined(NALU_HYPRE_USING_CUDA)
#if defined(NALU_HYPRE_USING_DEVICE_POOL)
      NALU_HYPRE_CUDA_CALL( nalu_hypre_CachingFreeDevice(ptr) );
#elif defined(NALU_HYPRE_USING_DEVICE_MALLOC_ASYNC)
      NALU_HYPRE_CUDA_CALL( cudaFreeAsync(ptr, NULL) );
#else
      NALU_HYPRE_CUDA_CALL( cudaFree(ptr) );
#endif
#endif

#if defined(NALU_HYPRE_USING_HIP)
      NALU_HYPRE_HIP_CALL( hipFree(ptr) );
#endif

#if defined(NALU_HYPRE_USING_SYCL)
      NALU_HYPRE_SYCL_CALL( sycl::free(ptr, *(nalu_hypre_HandleComputeStream(nalu_hypre_handle()))) );
#endif

#endif /* #if defined(NALU_HYPRE_USING_UMPIRE_DEVICE) */
   }
}

static inline void
nalu_hypre_UnifiedFree(void *ptr)
{
#if defined(NALU_HYPRE_USING_UMPIRE_UM)
   nalu_hypre_umpire_um_pooled_free(ptr);
#else

#if defined(NALU_HYPRE_USING_DEVICE_OPENMP)
#if defined(NALU_HYPRE_DEVICE_OPENMP_ALLOC)
   omp_target_free(ptr, nalu_hypre__offload_device_num);
#else
   NALU_HYPRE_OMPOffload(nalu_hypre__offload_device_num, ptr, ((size_t *) ptr)[-1], "exit", "delete");
#endif
#endif

#if defined(NALU_HYPRE_USING_CUDA)
#if defined(NALU_HYPRE_USING_DEVICE_POOL)
   NALU_HYPRE_CUDA_CALL( nalu_hypre_CachingFreeManaged(ptr) );
#else
   NALU_HYPRE_CUDA_CALL( cudaFree(ptr) );
#endif
#endif

#if defined(NALU_HYPRE_USING_HIP)
   NALU_HYPRE_HIP_CALL( hipFree(ptr) );
#endif

#if defined(NALU_HYPRE_USING_SYCL)
   NALU_HYPRE_SYCL_CALL( sycl::free(ptr, *(nalu_hypre_HandleComputeStream(nalu_hypre_handle()))) );
#endif

#endif /* #if defined(NALU_HYPRE_USING_UMPIRE_UM) */
}

static inline void
nalu_hypre_HostPinnedFree(void *ptr)
{
#if defined(NALU_HYPRE_USING_UMPIRE_PINNED)
   nalu_hypre_umpire_pinned_pooled_free(ptr);
#else

#if defined(NALU_HYPRE_USING_CUDA)
   NALU_HYPRE_CUDA_CALL( cudaFreeHost(ptr) );
#endif

#if defined(NALU_HYPRE_USING_HIP)
   NALU_HYPRE_HIP_CALL( hipHostFree(ptr) );
#endif

#if defined(NALU_HYPRE_USING_SYCL)
   NALU_HYPRE_SYCL_CALL( sycl::free(ptr, *(nalu_hypre_HandleComputeStream(nalu_hypre_handle()))) );
#endif

#endif /* #if defined(NALU_HYPRE_USING_UMPIRE_PINNED) */
}

static inline void
nalu_hypre_Free_core(void *ptr, nalu_hypre_MemoryLocation location)
{
   if (!ptr)
   {
      return;
   }

   nalu_hypre_CheckMemoryLocation(ptr, location);

   switch (location)
   {
      case nalu_hypre_MEMORY_HOST :
         nalu_hypre_HostFree(ptr);
         break;
      case nalu_hypre_MEMORY_DEVICE :
         nalu_hypre_DeviceFree(ptr);
         break;
      case nalu_hypre_MEMORY_UNIFIED :
         nalu_hypre_UnifiedFree(ptr);
         break;
      case nalu_hypre_MEMORY_HOST_PINNED :
         nalu_hypre_HostPinnedFree(ptr);
         break;
      default :
         nalu_hypre_WrongMemoryLocation();
   }
}

void
_nalu_hypre_Free(void *ptr, nalu_hypre_MemoryLocation location)
{
   nalu_hypre_Free_core(ptr, location);
}


/*--------------------------------------------------------------------------
 * Memcpy
 *--------------------------------------------------------------------------*/
static inline void
nalu_hypre_Memcpy_core(void *dst, void *src, size_t size, nalu_hypre_MemoryLocation loc_dst,
                  nalu_hypre_MemoryLocation loc_src)
{
#if defined(NALU_HYPRE_USING_SYCL)
   sycl::queue* q = nalu_hypre_HandleComputeStream(nalu_hypre_handle());
#endif

   if (dst == NULL || src == NULL)
   {
      if (size)
      {
         nalu_hypre_printf("nalu_hypre_Memcpy warning: copy %ld bytes from %p to %p !\n", size, src, dst);
         nalu_hypre_assert(0);
      }

      return;
   }

   if (dst == src)
   {
      return;
   }

   if (size > 0)
   {
      nalu_hypre_CheckMemoryLocation(dst, loc_dst);
      nalu_hypre_CheckMemoryLocation(src, loc_src);
   }

   /* Totally 4 x 4 = 16 cases */

   /* 4: Host   <-- Host, Host   <-- Pinned,
    *    Pinned <-- Host, Pinned <-- Pinned.
    */
   if ( loc_dst != nalu_hypre_MEMORY_DEVICE && loc_dst != nalu_hypre_MEMORY_UNIFIED &&
        loc_src != nalu_hypre_MEMORY_DEVICE && loc_src != nalu_hypre_MEMORY_UNIFIED )
   {
      memcpy(dst, src, size);
      return;
   }


   /* 3: UVM <-- Device, Device <-- UVM, UVM <-- UVM */
   if ( (loc_dst == nalu_hypre_MEMORY_UNIFIED && loc_src == nalu_hypre_MEMORY_DEVICE)  ||
        (loc_dst == nalu_hypre_MEMORY_DEVICE  && loc_src == nalu_hypre_MEMORY_UNIFIED) ||
        (loc_dst == nalu_hypre_MEMORY_UNIFIED && loc_src == nalu_hypre_MEMORY_UNIFIED) )
   {
#if defined(NALU_HYPRE_USING_DEVICE_OPENMP)
      omp_target_memcpy(dst, src, size, 0, 0, nalu_hypre__offload_device_num, nalu_hypre__offload_device_num);
#endif

#if defined(NALU_HYPRE_USING_CUDA)
      NALU_HYPRE_CUDA_CALL( cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice) );
#endif

#if defined(NALU_HYPRE_USING_HIP)
      NALU_HYPRE_HIP_CALL( hipMemcpy(dst, src, size, hipMemcpyDeviceToDevice) );
#endif

#if defined(NALU_HYPRE_USING_SYCL)
      NALU_HYPRE_SYCL_CALL( q->memcpy(dst, src, size).wait() );
#endif
      return;
   }


   /* 2: UVM <-- Host, UVM <-- Pinned */
   if (loc_dst == nalu_hypre_MEMORY_UNIFIED)
   {
#if defined(NALU_HYPRE_USING_DEVICE_OPENMP)
      omp_target_memcpy(dst, src, size, 0, 0, nalu_hypre__offload_device_num, nalu_hypre__offload_host_num);
#endif

#if defined(NALU_HYPRE_USING_CUDA)
      NALU_HYPRE_CUDA_CALL( cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice) );
#endif

#if defined(NALU_HYPRE_USING_HIP)
      NALU_HYPRE_HIP_CALL( hipMemcpy(dst, src, size, hipMemcpyHostToDevice) );
#endif

#if defined(NALU_HYPRE_USING_SYCL)
      NALU_HYPRE_SYCL_CALL( q->memcpy(dst, src, size).wait() );
#endif
      return;
   }


   /* 2: Host <-- UVM, Pinned <-- UVM */
   if (loc_src == nalu_hypre_MEMORY_UNIFIED)
   {
#if defined(NALU_HYPRE_USING_DEVICE_OPENMP)
      omp_target_memcpy(dst, src, size, 0, 0, nalu_hypre__offload_host_num, nalu_hypre__offload_device_num);
#endif

#if defined(NALU_HYPRE_USING_CUDA)
      NALU_HYPRE_CUDA_CALL( cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost) );
#endif

#if defined(NALU_HYPRE_USING_HIP)
      NALU_HYPRE_HIP_CALL( hipMemcpy(dst, src, size, hipMemcpyDeviceToHost) );
#endif

#if defined(NALU_HYPRE_USING_SYCL)
      NALU_HYPRE_SYCL_CALL( q->memcpy(dst, src, size).wait() );
#endif
      return;
   }


   /* 2: Device <-- Host, Device <-- Pinned */
   if ( loc_dst == nalu_hypre_MEMORY_DEVICE && (loc_src == nalu_hypre_MEMORY_HOST ||
                                           loc_src == nalu_hypre_MEMORY_HOST_PINNED) )
   {
#if defined(NALU_HYPRE_USING_DEVICE_OPENMP)
#if defined(NALU_HYPRE_DEVICE_OPENMP_ALLOC)
      omp_target_memcpy(dst, src, size, 0, 0, nalu_hypre__offload_device_num, nalu_hypre__offload_host_num);
#else
      memcpy(dst, src, size);
      NALU_HYPRE_OMPOffload(nalu_hypre__offload_device_num, dst, size, "update", "to");
#endif
#endif

#if defined(NALU_HYPRE_USING_CUDA)
      NALU_HYPRE_CUDA_CALL( cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice) );
#endif

#if defined(NALU_HYPRE_USING_HIP)
      NALU_HYPRE_HIP_CALL( hipMemcpy(dst, src, size, hipMemcpyHostToDevice) );
#endif

#if defined(NALU_HYPRE_USING_SYCL)
      NALU_HYPRE_SYCL_CALL( q->memcpy(dst, src, size).wait() );
#endif
      return;
   }


   /* 2: Host <-- Device, Pinned <-- Device */
   if ( (loc_dst == nalu_hypre_MEMORY_HOST || loc_dst == nalu_hypre_MEMORY_HOST_PINNED) &&
        loc_src == nalu_hypre_MEMORY_DEVICE )
   {
#if defined(NALU_HYPRE_USING_DEVICE_OPENMP)
#if defined(NALU_HYPRE_DEVICE_OPENMP_ALLOC)
      omp_target_memcpy(dst, src, size, 0, 0, nalu_hypre__offload_host_num, nalu_hypre__offload_device_num);
#else
      NALU_HYPRE_OMPOffload(nalu_hypre__offload_device_num, src, size, "update", "from");
      memcpy(dst, src, size);
#endif
#endif

#if defined(NALU_HYPRE_USING_CUDA)
      NALU_HYPRE_CUDA_CALL( cudaMemcpy( dst, src, size, cudaMemcpyDeviceToHost) );
#endif

#if defined(NALU_HYPRE_USING_HIP)
      NALU_HYPRE_HIP_CALL( hipMemcpy(dst, src, size, hipMemcpyDeviceToHost) );
#endif

#if defined(NALU_HYPRE_USING_SYCL)
      NALU_HYPRE_SYCL_CALL( q->memcpy(dst, src, size).wait() );
#endif
      return;
   }


   /* 1: Device <-- Device */
   if (loc_dst == nalu_hypre_MEMORY_DEVICE && loc_src == nalu_hypre_MEMORY_DEVICE)
   {
#if defined(NALU_HYPRE_USING_DEVICE_OPENMP)
#if defined(NALU_HYPRE_DEVICE_OPENMP_ALLOC)
      omp_target_memcpy(dst, src, size, 0, 0, nalu_hypre__offload_device_num, nalu_hypre__offload_device_num);
#else
      NALU_HYPRE_OMPOffload(nalu_hypre__offload_device_num, src, size, "update", "from");
      memcpy(dst, src, size);
      NALU_HYPRE_OMPOffload(nalu_hypre__offload_device_num, dst, size, "update", "to");
#endif
#endif

#if defined(NALU_HYPRE_USING_CUDA)
      NALU_HYPRE_CUDA_CALL( cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice) );
#endif

#if defined(NALU_HYPRE_USING_HIP)
      NALU_HYPRE_HIP_CALL( hipMemcpy(dst, src, size, hipMemcpyDeviceToDevice) );
#endif

#if defined(NALU_HYPRE_USING_SYCL)
      NALU_HYPRE_SYCL_CALL( q->memcpy(dst, src, size).wait() );
#endif
      return;
   }

   nalu_hypre_WrongMemoryLocation();
}

/*--------------------------------------------------------------------------*
 * ExecPolicy
 *--------------------------------------------------------------------------*/
static inline NALU_HYPRE_ExecutionPolicy
nalu_hypre_GetExecPolicy1_core(nalu_hypre_MemoryLocation location)
{
   NALU_HYPRE_ExecutionPolicy exec = NALU_HYPRE_EXEC_UNDEFINED;

   switch (location)
   {
      case nalu_hypre_MEMORY_HOST :
      case nalu_hypre_MEMORY_HOST_PINNED :
         exec = NALU_HYPRE_EXEC_HOST;
         break;
      case nalu_hypre_MEMORY_DEVICE :
         exec = NALU_HYPRE_EXEC_DEVICE;
         break;
      case nalu_hypre_MEMORY_UNIFIED :
#if defined(NALU_HYPRE_USING_GPU)
         exec = nalu_hypre_HandleDefaultExecPolicy(nalu_hypre_handle());
#endif
         break;
      default :
         nalu_hypre_WrongMemoryLocation();
   }

   nalu_hypre_assert(exec != NALU_HYPRE_EXEC_UNDEFINED);

   return exec;
}

/* for binary operation */
static inline NALU_HYPRE_ExecutionPolicy
nalu_hypre_GetExecPolicy2_core(nalu_hypre_MemoryLocation location1,
                          nalu_hypre_MemoryLocation location2)
{
   NALU_HYPRE_ExecutionPolicy exec = NALU_HYPRE_EXEC_UNDEFINED;

   /* HOST_PINNED has the same exec policy as HOST */
   if (location1 == nalu_hypre_MEMORY_HOST_PINNED)
   {
      location1 = nalu_hypre_MEMORY_HOST;
   }

   if (location2 == nalu_hypre_MEMORY_HOST_PINNED)
   {
      location2 = nalu_hypre_MEMORY_HOST;
   }

   /* no policy for these combinations */
   if ( (location1 == nalu_hypre_MEMORY_HOST && location2 == nalu_hypre_MEMORY_DEVICE) ||
        (location2 == nalu_hypre_MEMORY_HOST && location1 == nalu_hypre_MEMORY_DEVICE) )
   {
      exec = NALU_HYPRE_EXEC_UNDEFINED;
   }

   /* this should never happen */
   if ( (location1 == nalu_hypre_MEMORY_UNIFIED && location2 == nalu_hypre_MEMORY_DEVICE) ||
        (location2 == nalu_hypre_MEMORY_UNIFIED && location1 == nalu_hypre_MEMORY_DEVICE) )
   {
      exec = NALU_HYPRE_EXEC_UNDEFINED;
   }

   if (location1 == nalu_hypre_MEMORY_UNIFIED && location2 == nalu_hypre_MEMORY_UNIFIED)
   {
#if defined(NALU_HYPRE_USING_GPU)
      exec = nalu_hypre_HandleDefaultExecPolicy(nalu_hypre_handle());
#endif
   }

   if (location1 == nalu_hypre_MEMORY_HOST || location2 == nalu_hypre_MEMORY_HOST)
   {
      exec = NALU_HYPRE_EXEC_HOST;
   }

   if (location1 == nalu_hypre_MEMORY_DEVICE || location2 == nalu_hypre_MEMORY_DEVICE)
   {
      exec = NALU_HYPRE_EXEC_DEVICE;
   }

   nalu_hypre_assert(exec != NALU_HYPRE_EXEC_UNDEFINED);

   return exec;
}

/*==========================================================================
 * Conceptual memory location (NALU_HYPRE_MemoryLocation) interface
 *==========================================================================*/

/*--------------------------------------------------------------------------
 * nalu_hypre_Memset
 * "Sets the first num bytes of the block of memory pointed by ptr to the specified value
 * (*** value is interpreted as an unsigned char ***)"
 * http://www.cplusplus.com/reference/cstring/memset/
 *--------------------------------------------------------------------------*/
void *
nalu_hypre_Memset(void *ptr, NALU_HYPRE_Int value, size_t num, NALU_HYPRE_MemoryLocation location)
{
   if (num == 0)
   {
      return ptr;
   }

   if (ptr == NULL)
   {
      if (num)
      {
         nalu_hypre_printf("nalu_hypre_Memset warning: set values for %ld bytes at %p !\n", num, ptr);
      }
      return ptr;
   }

   nalu_hypre_CheckMemoryLocation(ptr, nalu_hypre_GetActualMemLocation(location));

   switch (nalu_hypre_GetActualMemLocation(location))
   {
      case nalu_hypre_MEMORY_HOST :
      case nalu_hypre_MEMORY_HOST_PINNED :
         nalu_hypre_HostMemset(ptr, value, num);
         break;
      case nalu_hypre_MEMORY_DEVICE :
         nalu_hypre_DeviceMemset(ptr, value, num);
         break;
      case nalu_hypre_MEMORY_UNIFIED :
         nalu_hypre_UnifiedMemset(ptr, value, num);
         break;
      default :
         nalu_hypre_WrongMemoryLocation();
   }

   return ptr;
}

/*--------------------------------------------------------------------------
 * Memprefetch
 *--------------------------------------------------------------------------*/
void
nalu_hypre_MemPrefetch(void *ptr, size_t size, NALU_HYPRE_MemoryLocation location)
{
   nalu_hypre_UnifiedMemPrefetch( ptr, size, nalu_hypre_GetActualMemLocation(location) );
}

/*--------------------------------------------------------------------------*
 * nalu_hypre_MAlloc, nalu_hypre_CAlloc
 *--------------------------------------------------------------------------*/

void *
nalu_hypre_MAlloc(size_t size, NALU_HYPRE_MemoryLocation location)
{
   return nalu_hypre_MAlloc_core(size, 0, nalu_hypre_GetActualMemLocation(location));
}

void *
nalu_hypre_CAlloc( size_t count, size_t elt_size, NALU_HYPRE_MemoryLocation location)
{
   return nalu_hypre_MAlloc_core(count * elt_size, 1, nalu_hypre_GetActualMemLocation(location));
}

/*--------------------------------------------------------------------------
 * nalu_hypre_Free
 *--------------------------------------------------------------------------*/

void
nalu_hypre_Free(void *ptr, NALU_HYPRE_MemoryLocation location)
{
   nalu_hypre_Free_core(ptr, nalu_hypre_GetActualMemLocation(location));
}

/*--------------------------------------------------------------------------
 * nalu_hypre_Memcpy
 *--------------------------------------------------------------------------*/

void
nalu_hypre_Memcpy(void *dst, void *src, size_t size, NALU_HYPRE_MemoryLocation loc_dst,
             NALU_HYPRE_MemoryLocation loc_src)
{
   nalu_hypre_Memcpy_core( dst, src, size, nalu_hypre_GetActualMemLocation(loc_dst),
                      nalu_hypre_GetActualMemLocation(loc_src) );
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ReAlloc
 *--------------------------------------------------------------------------*/
void *
nalu_hypre_ReAlloc(void *ptr, size_t size, NALU_HYPRE_MemoryLocation location)
{
   if (size == 0)
   {
      nalu_hypre_Free(ptr, location);
      return NULL;
   }

   if (ptr == NULL)
   {
      return nalu_hypre_MAlloc(size, location);
   }

   if (nalu_hypre_GetActualMemLocation(location) != nalu_hypre_MEMORY_HOST)
   {
      nalu_hypre_printf("nalu_hypre_TReAlloc only works with NALU_HYPRE_MEMORY_HOST; Use nalu_hypre_TReAlloc_v2 instead!\n");
      nalu_hypre_assert(0);
      nalu_hypre_MPI_Abort(nalu_hypre_MPI_COMM_WORLD, -1);
      return NULL;
   }

#if defined(NALU_HYPRE_USING_UMPIRE_HOST)
   ptr = nalu_hypre_umpire_host_pooled_realloc(ptr, size);
#else
   ptr = realloc(ptr, size);
#endif

   if (!ptr)
   {
      nalu_hypre_OutOfMemory(size);
   }

   return ptr;
}

void *
nalu_hypre_ReAlloc_v2(void *ptr, size_t old_size, size_t new_size, NALU_HYPRE_MemoryLocation location)
{
   if (new_size == 0)
   {
      nalu_hypre_Free(ptr, location);
      return NULL;
   }

   if (ptr == NULL)
   {
      return nalu_hypre_MAlloc(new_size, location);
   }

   if (old_size == new_size)
   {
      return ptr;
   }

   void *new_ptr = nalu_hypre_MAlloc(new_size, location);
   size_t smaller_size = new_size > old_size ? old_size : new_size;
   nalu_hypre_Memcpy(new_ptr, ptr, smaller_size, location, location);
   nalu_hypre_Free(ptr, location);
   ptr = new_ptr;

   if (!ptr)
   {
      nalu_hypre_OutOfMemory(new_size);
   }

   return ptr;
}

/*--------------------------------------------------------------------------*
 * nalu_hypre_GetExecPolicy: return execution policy based on memory locations
 *--------------------------------------------------------------------------*/
/* for unary operation */
NALU_HYPRE_ExecutionPolicy
nalu_hypre_GetExecPolicy1(NALU_HYPRE_MemoryLocation location)
{

   return nalu_hypre_GetExecPolicy1_core(nalu_hypre_GetActualMemLocation(location));
}

/* for binary operation */
NALU_HYPRE_ExecutionPolicy
nalu_hypre_GetExecPolicy2(NALU_HYPRE_MemoryLocation location1,
                     NALU_HYPRE_MemoryLocation location2)
{
   return nalu_hypre_GetExecPolicy2_core(nalu_hypre_GetActualMemLocation(location1),
                                    nalu_hypre_GetActualMemLocation(location2));
}

/*--------------------------------------------------------------------------
 * Query the actual memory location pointed by ptr
 *--------------------------------------------------------------------------*/
NALU_HYPRE_Int
nalu_hypre_GetPointerLocation(const void *ptr, nalu_hypre_MemoryLocation *memory_location)
{
   NALU_HYPRE_Int ierr = 0;

#if defined(NALU_HYPRE_USING_GPU)
   *memory_location = nalu_hypre_MEMORY_UNDEFINED;

#if defined(NALU_HYPRE_USING_CUDA)
   struct cudaPointerAttributes attr;

#if (CUDART_VERSION >= 10000)
#if (CUDART_VERSION >= 11000)
   NALU_HYPRE_CUDA_CALL( cudaPointerGetAttributes(&attr, ptr) );
#else
   cudaError_t err = cudaPointerGetAttributes(&attr, ptr);
   if (err != cudaSuccess)
   {
      ierr = 1;
      /* clear the error */
      cudaGetLastError();
   }
#endif
   if (attr.type == cudaMemoryTypeUnregistered)
   {
      *memory_location = nalu_hypre_MEMORY_HOST;
   }
   else if (attr.type == cudaMemoryTypeHost)
   {
      *memory_location = nalu_hypre_MEMORY_HOST_PINNED;
   }
   else if (attr.type == cudaMemoryTypeDevice)
   {
      *memory_location = nalu_hypre_MEMORY_DEVICE;
   }
   else if (attr.type == cudaMemoryTypeManaged)
   {
      *memory_location = nalu_hypre_MEMORY_UNIFIED;
   }
#else
   cudaError_t err = cudaPointerGetAttributes(&attr, ptr);
   if (err != cudaSuccess)
   {
      ierr = 1;

      /* clear the error */
      cudaGetLastError();

      if (err == cudaErrorInvalidValue)
      {
         *memory_location = nalu_hypre_MEMORY_HOST;
      }
   }
   else if (attr.isManaged)
   {
      *memory_location = nalu_hypre_MEMORY_UNIFIED;
   }
   else if (attr.memoryType == cudaMemoryTypeDevice)
   {
      *memory_location = nalu_hypre_MEMORY_DEVICE;
   }
   else if (attr.memoryType == cudaMemoryTypeHost)
   {
      *memory_location = nalu_hypre_MEMORY_HOST_PINNED;
   }
#endif // CUDART_VERSION >= 10000
#endif // defined(NALU_HYPRE_USING_CUDA)

#if defined(NALU_HYPRE_USING_HIP)

   struct hipPointerAttribute_t attr;
   *memory_location = nalu_hypre_MEMORY_UNDEFINED;

   hipError_t err = hipPointerGetAttributes(&attr, ptr);
   if (err != hipSuccess)
   {
      ierr = 1;

      /* clear the error */
      hipGetLastError();

      if (err == hipErrorInvalidValue)
      {
         *memory_location = nalu_hypre_MEMORY_HOST;
      }
   }
   else if (attr.isManaged)
   {
      *memory_location = nalu_hypre_MEMORY_UNIFIED;
   }
   else if (attr.memoryType == hipMemoryTypeDevice)
   {
      *memory_location = nalu_hypre_MEMORY_DEVICE;
   }
   else if (attr.memoryType == hipMemoryTypeHost)
   {
      *memory_location = nalu_hypre_MEMORY_HOST_PINNED;
   }
#endif // defined(NALU_HYPRE_USING_HIP)

#if defined(NALU_HYPRE_USING_SYCL)
   /* If the device is not setup, then all allocations are assumed to be on the host */
   *memory_location = nalu_hypre_MEMORY_HOST;
   if (nalu_hypre_HandleDeviceData(nalu_hypre_handle()))
   {
      if (nalu_hypre_HandleDevice(nalu_hypre_handle()))
      {
         sycl::usm::alloc allocType;
         allocType = sycl::get_pointer_type(ptr, (nalu_hypre_HandleComputeStream(nalu_hypre_handle()))->get_context());

         if (allocType == sycl::usm::alloc::unknown)
         {
            *memory_location = nalu_hypre_MEMORY_HOST;
         }
         else if (allocType == sycl::usm::alloc::host)
         {
            *memory_location = nalu_hypre_MEMORY_HOST_PINNED;
         }
         else if (allocType == sycl::usm::alloc::device)
         {
            *memory_location = nalu_hypre_MEMORY_DEVICE;
         }
         else if (allocType == sycl::usm::alloc::shared)
         {
            *memory_location = nalu_hypre_MEMORY_UNIFIED;
         }
      }
   }
#endif //NALU_HYPRE_USING_SYCL

#else /* #if defined(NALU_HYPRE_USING_GPU) */
   *memory_location = nalu_hypre_MEMORY_HOST;
#endif

   return ierr;
}

/*--------------------------------------------------------------------------*
 * Memory Pool
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SetCubMemPoolSize(nalu_hypre_uint cub_bin_growth,
                        nalu_hypre_uint cub_min_bin,
                        nalu_hypre_uint cub_max_bin,
                        size_t     cub_max_cached_bytes)
{
#if defined(NALU_HYPRE_USING_CUDA)
#if defined(NALU_HYPRE_USING_DEVICE_POOL)
   nalu_hypre_HandleCubBinGrowth(nalu_hypre_handle())      = cub_bin_growth;
   nalu_hypre_HandleCubMinBin(nalu_hypre_handle())         = cub_min_bin;
   nalu_hypre_HandleCubMaxBin(nalu_hypre_handle())         = cub_max_bin;
   nalu_hypre_HandleCubMaxCachedBytes(nalu_hypre_handle()) = cub_max_cached_bytes;

   //TODO XXX RL: cub_min_bin, cub_max_bin are not (re)set
   if (nalu_hypre_HandleCubDevAllocator(nalu_hypre_handle()))
   {
      nalu_hypre_HandleCubDevAllocator(nalu_hypre_handle()) -> SetMaxCachedBytes(cub_max_cached_bytes);
   }

   if (nalu_hypre_HandleCubUvmAllocator(nalu_hypre_handle()))
   {
      nalu_hypre_HandleCubUvmAllocator(nalu_hypre_handle()) -> SetMaxCachedBytes(cub_max_cached_bytes);
   }
#endif
#endif

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
NALU_HYPRE_SetGPUMemoryPoolSize(NALU_HYPRE_Int bin_growth,
                           NALU_HYPRE_Int min_bin,
                           NALU_HYPRE_Int max_bin,
                           size_t    max_cached_bytes)
{
   return nalu_hypre_SetCubMemPoolSize(bin_growth, min_bin, max_bin, max_cached_bytes);
}

#if defined(NALU_HYPRE_USING_DEVICE_POOL)
cudaError_t
nalu_hypre_CachingMallocDevice(void **ptr, size_t nbytes)
{
   if (!nalu_hypre_HandleCubDevAllocator(nalu_hypre_handle()))
   {
      nalu_hypre_HandleCubDevAllocator(nalu_hypre_handle()) =
         nalu_hypre_DeviceDataCubCachingAllocatorCreate( nalu_hypre_HandleCubBinGrowth(nalu_hypre_handle()),
                                                    nalu_hypre_HandleCubMinBin(nalu_hypre_handle()),
                                                    nalu_hypre_HandleCubMaxBin(nalu_hypre_handle()),
                                                    nalu_hypre_HandleCubMaxCachedBytes(nalu_hypre_handle()),
                                                    false,
                                                    false,
                                                    false );
   }

   return nalu_hypre_HandleCubDevAllocator(nalu_hypre_handle()) -> DeviceAllocate(ptr, nbytes);
}

cudaError_t
nalu_hypre_CachingFreeDevice(void *ptr)
{
   return nalu_hypre_HandleCubDevAllocator(nalu_hypre_handle()) -> DeviceFree(ptr);
}

cudaError_t
nalu_hypre_CachingMallocManaged(void **ptr, size_t nbytes)
{
   if (!nalu_hypre_HandleCubUvmAllocator(nalu_hypre_handle()))
   {
      nalu_hypre_HandleCubUvmAllocator(nalu_hypre_handle()) =
         nalu_hypre_DeviceDataCubCachingAllocatorCreate( nalu_hypre_HandleCubBinGrowth(nalu_hypre_handle()),
                                                    nalu_hypre_HandleCubMinBin(nalu_hypre_handle()),
                                                    nalu_hypre_HandleCubMaxBin(nalu_hypre_handle()),
                                                    nalu_hypre_HandleCubMaxCachedBytes(nalu_hypre_handle()),
                                                    false,
                                                    false,
                                                    true );
   }

   return nalu_hypre_HandleCubUvmAllocator(nalu_hypre_handle()) -> DeviceAllocate(ptr, nbytes);
}

cudaError_t
nalu_hypre_CachingFreeManaged(void *ptr)
{
   return nalu_hypre_HandleCubUvmAllocator(nalu_hypre_handle()) -> DeviceFree(ptr);
}

nalu_hypre_cub_CachingDeviceAllocator *
nalu_hypre_DeviceDataCubCachingAllocatorCreate(nalu_hypre_uint bin_growth,
                                          nalu_hypre_uint min_bin,
                                          nalu_hypre_uint max_bin,
                                          size_t     max_cached_bytes,
                                          bool       skip_cleanup,
                                          bool       debug,
                                          bool       use_managed_memory)
{
   nalu_hypre_cub_CachingDeviceAllocator *allocator =
      new nalu_hypre_cub_CachingDeviceAllocator( bin_growth,
                                            min_bin,
                                            max_bin,
                                            max_cached_bytes,
                                            skip_cleanup,
                                            debug,
                                            use_managed_memory );

   return allocator;
}

void
nalu_hypre_DeviceDataCubCachingAllocatorDestroy(nalu_hypre_DeviceData *data)
{
   delete nalu_hypre_DeviceDataCubDevAllocator(data);
   delete nalu_hypre_DeviceDataCubUvmAllocator(data);
}

#endif // #if defined(NALU_HYPRE_USING_DEVICE_POOL)

#if defined(NALU_HYPRE_USING_UMPIRE_HOST)
NALU_HYPRE_Int
nalu_hypre_umpire_host_pooled_allocate(void **ptr, size_t nbytes)
{
   nalu_hypre_Handle *handle = nalu_hypre_handle();
   const char *resource_name = "HOST";
   const char *pool_name = nalu_hypre_HandleUmpireHostPoolName(handle);

   umpire_resourcemanager *rm_ptr = &nalu_hypre_HandleUmpireResourceMan(handle);
   umpire_allocator pooled_allocator;

   if ( umpire_resourcemanager_is_allocator_name(rm_ptr, pool_name) )
   {
      umpire_resourcemanager_get_allocator_by_name(rm_ptr, pool_name, &pooled_allocator);
   }
   else
   {
      umpire_allocator allocator;
      umpire_resourcemanager_get_allocator_by_name(rm_ptr, resource_name, &allocator);
      nalu_hypre_umpire_resourcemanager_make_allocator_pool(rm_ptr, pool_name, allocator,
                                                       nalu_hypre_HandleUmpireHostPoolSize(handle),
                                                       nalu_hypre_HandleUmpireBlockSize(handle), &pooled_allocator);
      nalu_hypre_HandleOwnUmpireHostPool(handle) = 1;
   }

   *ptr = umpire_allocator_allocate(&pooled_allocator, nbytes);

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_umpire_host_pooled_free(void *ptr)
{
   nalu_hypre_Handle *handle = nalu_hypre_handle();
   const char *pool_name = nalu_hypre_HandleUmpireHostPoolName(handle);
   umpire_allocator pooled_allocator;

   umpire_resourcemanager *rm_ptr = &nalu_hypre_HandleUmpireResourceMan(handle);

   nalu_hypre_assert(umpire_resourcemanager_is_allocator_name(rm_ptr, pool_name));

   umpire_resourcemanager_get_allocator_by_name(rm_ptr, pool_name, &pooled_allocator);
   umpire_allocator_deallocate(&pooled_allocator, ptr);

   return nalu_hypre_error_flag;
}

void *
nalu_hypre_umpire_host_pooled_realloc(void *ptr, size_t size)
{
   nalu_hypre_Handle *handle = nalu_hypre_handle();
   const char *pool_name = nalu_hypre_HandleUmpireHostPoolName(handle);
   umpire_allocator pooled_allocator;

   umpire_resourcemanager *rm_ptr = &nalu_hypre_HandleUmpireResourceMan(handle);

   nalu_hypre_assert(umpire_resourcemanager_is_allocator_name(rm_ptr, pool_name));

   umpire_resourcemanager_get_allocator_by_name(rm_ptr, pool_name, &pooled_allocator);
   ptr = umpire_resourcemanager_reallocate_with_allocator(rm_ptr, ptr, size, pooled_allocator);

   return ptr;
}
#endif

#if defined(NALU_HYPRE_USING_UMPIRE_DEVICE)
NALU_HYPRE_Int
nalu_hypre_umpire_device_pooled_allocate(void **ptr, size_t nbytes)
{
   nalu_hypre_Handle *handle = nalu_hypre_handle();
   const nalu_hypre_int device_id = nalu_hypre_HandleDevice(handle);
   char resource_name[16];
   const char *pool_name = nalu_hypre_HandleUmpireDevicePoolName(handle);

   nalu_hypre_sprintf(resource_name, "%s::%d", "DEVICE", device_id);

   umpire_resourcemanager *rm_ptr = &nalu_hypre_HandleUmpireResourceMan(handle);
   umpire_allocator pooled_allocator;

   if ( umpire_resourcemanager_is_allocator_name(rm_ptr, pool_name) )
   {
      umpire_resourcemanager_get_allocator_by_name(rm_ptr, pool_name, &pooled_allocator);
   }
   else
   {
      umpire_allocator allocator;
      umpire_resourcemanager_get_allocator_by_name(rm_ptr, resource_name, &allocator);
      nalu_hypre_umpire_resourcemanager_make_allocator_pool(rm_ptr, pool_name, allocator,
                                                       nalu_hypre_HandleUmpireDevicePoolSize(handle),
                                                       nalu_hypre_HandleUmpireBlockSize(handle), &pooled_allocator);

      nalu_hypre_HandleOwnUmpireDevicePool(handle) = 1;
   }

   *ptr = umpire_allocator_allocate(&pooled_allocator, nbytes);

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_umpire_device_pooled_free(void *ptr)
{
   nalu_hypre_Handle *handle = nalu_hypre_handle();
   const char *pool_name = nalu_hypre_HandleUmpireDevicePoolName(handle);
   umpire_allocator pooled_allocator;

   umpire_resourcemanager *rm_ptr = &nalu_hypre_HandleUmpireResourceMan(handle);

   nalu_hypre_assert(umpire_resourcemanager_is_allocator_name(rm_ptr, pool_name));

   umpire_resourcemanager_get_allocator_by_name(rm_ptr, pool_name, &pooled_allocator);
   umpire_allocator_deallocate(&pooled_allocator, ptr);

   return nalu_hypre_error_flag;
}
#endif

#if defined(NALU_HYPRE_USING_UMPIRE_UM)
NALU_HYPRE_Int
nalu_hypre_umpire_um_pooled_allocate(void **ptr, size_t nbytes)
{
   nalu_hypre_Handle *handle = nalu_hypre_handle();
   const char *resource_name = "UM";
   const char *pool_name = nalu_hypre_HandleUmpireUMPoolName(handle);

   umpire_resourcemanager *rm_ptr = &nalu_hypre_HandleUmpireResourceMan(handle);
   umpire_allocator pooled_allocator;

   if ( umpire_resourcemanager_is_allocator_name(rm_ptr, pool_name) )
   {
      umpire_resourcemanager_get_allocator_by_name(rm_ptr, pool_name, &pooled_allocator);
   }
   else
   {
      umpire_allocator allocator;
      umpire_resourcemanager_get_allocator_by_name(rm_ptr, resource_name, &allocator);
      nalu_hypre_umpire_resourcemanager_make_allocator_pool(rm_ptr, pool_name, allocator,
                                                       nalu_hypre_HandleUmpireUMPoolSize(handle),
                                                       nalu_hypre_HandleUmpireBlockSize(handle), &pooled_allocator);

      nalu_hypre_HandleOwnUmpireUMPool(handle) = 1;
   }

   *ptr = umpire_allocator_allocate(&pooled_allocator, nbytes);

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_umpire_um_pooled_free(void *ptr)
{
   nalu_hypre_Handle *handle = nalu_hypre_handle();
   const char *pool_name = nalu_hypre_HandleUmpireUMPoolName(handle);
   umpire_allocator pooled_allocator;

   umpire_resourcemanager *rm_ptr = &nalu_hypre_HandleUmpireResourceMan(handle);

   nalu_hypre_assert(umpire_resourcemanager_is_allocator_name(rm_ptr, pool_name));

   umpire_resourcemanager_get_allocator_by_name(rm_ptr, pool_name, &pooled_allocator);
   umpire_allocator_deallocate(&pooled_allocator, ptr);

   return nalu_hypre_error_flag;
}
#endif

#if defined(NALU_HYPRE_USING_UMPIRE_PINNED)
NALU_HYPRE_Int
nalu_hypre_umpire_pinned_pooled_allocate(void **ptr, size_t nbytes)
{
   nalu_hypre_Handle *handle = nalu_hypre_handle();
   const char *resource_name = "PINNED";
   const char *pool_name = nalu_hypre_HandleUmpirePinnedPoolName(handle);

   umpire_resourcemanager *rm_ptr = &nalu_hypre_HandleUmpireResourceMan(handle);
   umpire_allocator pooled_allocator;

   if ( umpire_resourcemanager_is_allocator_name(rm_ptr, pool_name) )
   {
      umpire_resourcemanager_get_allocator_by_name(rm_ptr, pool_name, &pooled_allocator);
   }
   else
   {
      umpire_allocator allocator;
      umpire_resourcemanager_get_allocator_by_name(rm_ptr, resource_name, &allocator);
      nalu_hypre_umpire_resourcemanager_make_allocator_pool(rm_ptr, pool_name, allocator,
                                                       nalu_hypre_HandleUmpirePinnedPoolSize(handle),
                                                       nalu_hypre_HandleUmpireBlockSize(handle), &pooled_allocator);

      nalu_hypre_HandleOwnUmpirePinnedPool(handle) = 1;
   }

   *ptr = umpire_allocator_allocate(&pooled_allocator, nbytes);

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_umpire_pinned_pooled_free(void *ptr)
{
   nalu_hypre_Handle *handle = nalu_hypre_handle();
   const char *pool_name = nalu_hypre_HandleUmpirePinnedPoolName(handle);
   umpire_allocator pooled_allocator;

   umpire_resourcemanager *rm_ptr = &nalu_hypre_HandleUmpireResourceMan(handle);

   nalu_hypre_assert(umpire_resourcemanager_is_allocator_name(rm_ptr, pool_name));

   umpire_resourcemanager_get_allocator_by_name(rm_ptr, pool_name, &pooled_allocator);
   umpire_allocator_deallocate(&pooled_allocator, ptr);

   return nalu_hypre_error_flag;
}
#endif
