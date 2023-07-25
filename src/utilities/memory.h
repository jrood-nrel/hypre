/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Header file for memory management utilities
 *
 * The abstract memory model has a Host (think CPU) and a Device (think GPU) and
 * three basic types of memory management utilities:
 *
 *    1. Malloc(..., location)
 *             location=LOCATION_DEVICE - malloc memory on the device
 *             location=LOCATION_HOST   - malloc memory on the host
 *    2. MemCopy(..., method)
 *             method=HOST_TO_DEVICE    - copy from host to device
 *             method=DEVICE_TO_HOST    - copy from device to host
 *             method=DEVICE_TO_DEVICE  - copy from device to device
 *    3. SetExecutionMode
 *             location=LOCATION_DEVICE - execute on the device
 *             location=LOCATION_HOST   - execute on the host
 *
 * Although the abstract model does not explicitly reflect a managed memory
 * model (i.e., unified memory), it can support it.  Here is a summary of how
 * the abstract model would be mapped to specific hardware scenarios:
 *
 *    Not using a device, not using managed memory
 *       Malloc(..., location)
 *             location=LOCATION_DEVICE - host malloc          e.g., malloc
 *             location=LOCATION_HOST   - host malloc          e.g., malloc
 *       MemoryCopy(..., locTo,locFrom)
 *             locTo=LOCATION_HOST,   locFrom=LOCATION_DEVICE  - copy from host to host e.g., memcpy
 *             locTo=LOCATION_DEVICE, locFrom=LOCATION_HOST    - copy from host to host e.g., memcpy
 *             locTo=LOCATION_DEVICE, locFrom=LOCATION_DEVICE  - copy from host to host e.g., memcpy
 *       SetExecutionMode
 *             location=LOCATION_DEVICE - execute on the host
 *             location=LOCATION_HOST   - execute on the host
 *
 *    Using a device, not using managed memory
 *       Malloc(..., location)
 *             location=LOCATION_DEVICE - device malloc        e.g., cudaMalloc
 *             location=LOCATION_HOST   - host malloc          e.g., malloc
 *       MemoryCopy(..., locTo,locFrom)
 *             locTo=LOCATION_HOST,   locFrom=LOCATION_DEVICE  - copy from device to host e.g., cudaMemcpy
 *             locTo=LOCATION_DEVICE, locFrom=LOCATION_HOST    - copy from host to device e.g., cudaMemcpy
 *             locTo=LOCATION_DEVICE, locFrom=LOCATION_DEVICE  - copy from device to device e.g., cudaMemcpy
 *       SetExecutionMode
 *             location=LOCATION_DEVICE - execute on the device
 *             location=LOCATION_HOST   - execute on the host
 *
 *    Using a device, using managed memory
 *       Malloc(..., location)
 *             location=LOCATION_DEVICE - managed malloc        e.g., cudaMallocManaged
 *             location=LOCATION_HOST   - host malloc          e.g., malloc
 *       MemoryCopy(..., locTo,locFrom)
 *             locTo=LOCATION_HOST,   locFrom=LOCATION_DEVICE  - copy from device to host e.g., cudaMallocManaged
 *             locTo=LOCATION_DEVICE, locFrom=LOCATION_HOST    - copy from host to device e.g., cudaMallocManaged
 *             locTo=LOCATION_DEVICE, locFrom=LOCATION_DEVICE  - copy from device to device e.g., cudaMallocManaged
 *       SetExecutionMode
 *             location=LOCATION_DEVICE - execute on the device
 *             location=LOCATION_HOST   - execute on the host
 *
 *****************************************************************************/

#ifndef nalu_hypre_MEMORY_HEADER
#define nalu_hypre_MEMORY_HEADER

#include <stdio.h>
#include <stdlib.h>

#if defined(NALU_HYPRE_USING_UNIFIED_MEMORY) && defined(NALU_HYPRE_USING_DEVICE_OPENMP)
//#pragma omp requires unified_shared_memory
#endif

#if defined(NALU_HYPRE_USING_UMPIRE)
#include "umpire/config.hpp"
#if UMPIRE_VERSION_MAJOR >= 2022
#include "umpire/interface/c_fortran/umpire.h"
#define nalu_hypre_umpire_resourcemanager_make_allocator_pool umpire_resourcemanager_make_allocator_quick_pool
#else
#include "umpire/interface/umpire.h"
#define nalu_hypre_umpire_resourcemanager_make_allocator_pool umpire_resourcemanager_make_allocator_pool
#endif
#define NALU_HYPRE_UMPIRE_POOL_NAME_MAX_LEN 1024
#endif

/* stringification:
 * _Pragma(string-literal), so we need to cast argument to a string
 * The three dots as last argument of the macro tells compiler that this is a variadic macro.
 * I.e. this is a macro that receives variable number of arguments.
 */
#define NALU_HYPRE_STR(...) #__VA_ARGS__
#define NALU_HYPRE_XSTR(...) NALU_HYPRE_STR(__VA_ARGS__)

#ifdef __cplusplus
extern "C" {
#endif

typedef enum _nalu_hypre_MemoryLocation
{
   nalu_hypre_MEMORY_UNDEFINED = -1,
   nalu_hypre_MEMORY_HOST,
   nalu_hypre_MEMORY_HOST_PINNED,
   nalu_hypre_MEMORY_DEVICE,
   nalu_hypre_MEMORY_UNIFIED,
   nalu_hypre_NUM_MEMORY_LOCATION
} nalu_hypre_MemoryLocation;

/*-------------------------------------------------------
 * nalu_hypre_GetActualMemLocation
 *   return actual location based on the selected memory model
 *-------------------------------------------------------*/
static inline nalu_hypre_MemoryLocation
nalu_hypre_GetActualMemLocation(NALU_HYPRE_MemoryLocation location)
{
   if (location == NALU_HYPRE_MEMORY_HOST)
   {
      return nalu_hypre_MEMORY_HOST;
   }

   if (location == NALU_HYPRE_MEMORY_DEVICE)
   {
#if defined(NALU_HYPRE_USING_HOST_MEMORY)
      return nalu_hypre_MEMORY_HOST;
#elif defined(NALU_HYPRE_USING_DEVICE_MEMORY)
      return nalu_hypre_MEMORY_DEVICE;
#elif defined(NALU_HYPRE_USING_UNIFIED_MEMORY)
      return nalu_hypre_MEMORY_UNIFIED;
#else
#error Wrong NALU_HYPRE memory setting.
#endif
   }

   return nalu_hypre_MEMORY_UNDEFINED;
}


#if !defined(NALU_HYPRE_USING_MEMORY_TRACKER)

#define nalu_hypre_TAlloc(type, count, location) \
( (type *) nalu_hypre_MAlloc((size_t)(sizeof(type) * (count)), location) )

#define _nalu_hypre_TAlloc(type, count, location) \
( (type *) _nalu_hypre_MAlloc((size_t)(sizeof(type) * (count)), location) )

#define nalu_hypre_CTAlloc(type, count, location) \
( (type *) nalu_hypre_CAlloc((size_t)(count), (size_t)sizeof(type), location) )

#define nalu_hypre_TReAlloc(ptr, type, count, location) \
( (type *) nalu_hypre_ReAlloc((char *)ptr, (size_t)(sizeof(type) * (count)), location) )

#define nalu_hypre_TReAlloc_v2(ptr, old_type, old_count, new_type, new_count, location) \
( (new_type *) nalu_hypre_ReAlloc_v2((char *)ptr, (size_t)(sizeof(old_type)*(old_count)), (size_t)(sizeof(new_type)*(new_count)), location) )

#define nalu_hypre_TMemcpy(dst, src, type, count, locdst, locsrc) \
(nalu_hypre_Memcpy((void *)(dst), (void *)(src), (size_t)(sizeof(type) * (count)), locdst, locsrc))

#define nalu_hypre_TFree(ptr, location) \
( nalu_hypre_Free((void *)ptr, location), ptr = NULL )

#define _nalu_hypre_TFree(ptr, location) \
( _nalu_hypre_Free((void *)ptr, location), ptr = NULL )

#endif /* #if !defined(NALU_HYPRE_USING_MEMORY_TRACKER) */


/*--------------------------------------------------------------------------
 * Prototypes
 *--------------------------------------------------------------------------*/

/* memory.c */
NALU_HYPRE_Int nalu_hypre_GetMemoryLocationName(nalu_hypre_MemoryLocation memory_location,
                                      char *memory_location_name);
void   nalu_hypre_CheckMemoryLocation(void *ptr, nalu_hypre_MemoryLocation location);
void * nalu_hypre_Memset(void *ptr, NALU_HYPRE_Int value, size_t num, NALU_HYPRE_MemoryLocation location);
void   nalu_hypre_MemPrefetch(void *ptr, size_t size, NALU_HYPRE_MemoryLocation location);
void * nalu_hypre_MAlloc(size_t size, NALU_HYPRE_MemoryLocation location);
void * nalu_hypre_CAlloc( size_t count, size_t elt_size, NALU_HYPRE_MemoryLocation location);
void   nalu_hypre_Free(void *ptr, NALU_HYPRE_MemoryLocation location);
void   nalu_hypre_Memcpy(void *dst, void *src, size_t size, NALU_HYPRE_MemoryLocation loc_dst,
                    NALU_HYPRE_MemoryLocation loc_src);
void * nalu_hypre_ReAlloc(void *ptr, size_t size, NALU_HYPRE_MemoryLocation location);
void * nalu_hypre_ReAlloc_v2(void *ptr, size_t old_size, size_t new_size, NALU_HYPRE_MemoryLocation location);

void * _nalu_hypre_MAlloc(size_t size, nalu_hypre_MemoryLocation location);
void   _nalu_hypre_Free(void *ptr, nalu_hypre_MemoryLocation location);

NALU_HYPRE_ExecutionPolicy nalu_hypre_GetExecPolicy1(NALU_HYPRE_MemoryLocation location);
NALU_HYPRE_ExecutionPolicy nalu_hypre_GetExecPolicy2(NALU_HYPRE_MemoryLocation location1,
                                           NALU_HYPRE_MemoryLocation location2);

NALU_HYPRE_Int nalu_hypre_GetPointerLocation(const void *ptr, nalu_hypre_MemoryLocation *memory_location);
NALU_HYPRE_Int nalu_hypre_SetCubMemPoolSize( nalu_hypre_uint bin_growth, nalu_hypre_uint min_bin, nalu_hypre_uint max_bin,
                                   size_t max_cached_bytes );
NALU_HYPRE_Int nalu_hypre_umpire_host_pooled_allocate(void **ptr, size_t nbytes);
NALU_HYPRE_Int nalu_hypre_umpire_host_pooled_free(void *ptr);
void *nalu_hypre_umpire_host_pooled_realloc(void *ptr, size_t size);
NALU_HYPRE_Int nalu_hypre_umpire_device_pooled_allocate(void **ptr, size_t nbytes);
NALU_HYPRE_Int nalu_hypre_umpire_device_pooled_free(void *ptr);
NALU_HYPRE_Int nalu_hypre_umpire_um_pooled_allocate(void **ptr, size_t nbytes);
NALU_HYPRE_Int nalu_hypre_umpire_um_pooled_free(void *ptr);
NALU_HYPRE_Int nalu_hypre_umpire_pinned_pooled_allocate(void **ptr, size_t nbytes);
NALU_HYPRE_Int nalu_hypre_umpire_pinned_pooled_free(void *ptr);

/* memory_dmalloc.c */
NALU_HYPRE_Int nalu_hypre_InitMemoryDebugDML( NALU_HYPRE_Int id );
NALU_HYPRE_Int nalu_hypre_FinalizeMemoryDebugDML( void );
char *nalu_hypre_MAllocDML( NALU_HYPRE_Int size, char *file, NALU_HYPRE_Int line );
char *nalu_hypre_CAllocDML( NALU_HYPRE_Int count, NALU_HYPRE_Int elt_size, char *file, NALU_HYPRE_Int line );
char *nalu_hypre_ReAllocDML( char *ptr, NALU_HYPRE_Int size, char *file, NALU_HYPRE_Int line );
void nalu_hypre_FreeDML( char *ptr, char *file, NALU_HYPRE_Int line );

/* GPU malloc prototype */
typedef void (*GPUMallocFunc)(void **, size_t);
typedef void (*GPUMfreeFunc)(void *);

#ifdef __cplusplus
}
#endif

#endif /* nalu_hypre_MEMORY_HEADER */
