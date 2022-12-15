/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * General structures and values
 *
 *****************************************************************************/

#ifndef NALU_HYPRE_HANDLE_H
#define NALU_HYPRE_HANDLE_H

struct nalu_hypre_DeviceData;
typedef struct nalu_hypre_DeviceData nalu_hypre_DeviceData;

typedef struct
{
   NALU_HYPRE_Int              nalu_hypre_error;
   NALU_HYPRE_MemoryLocation   memory_location;
   NALU_HYPRE_ExecutionPolicy  default_exec_policy;
#if defined(NALU_HYPRE_USING_GPU)
   nalu_hypre_DeviceData      *device_data;
   /* device G-S options */
   NALU_HYPRE_Int              device_gs_method;
#endif
#if defined(NALU_HYPRE_USING_UMPIRE)
   char                   umpire_device_pool_name[NALU_HYPRE_UMPIRE_POOL_NAME_MAX_LEN];
   char                   umpire_um_pool_name[NALU_HYPRE_UMPIRE_POOL_NAME_MAX_LEN];
   char                   umpire_host_pool_name[NALU_HYPRE_UMPIRE_POOL_NAME_MAX_LEN];
   char                   umpire_pinned_pool_name[NALU_HYPRE_UMPIRE_POOL_NAME_MAX_LEN];
   size_t                 umpire_device_pool_size;
   size_t                 umpire_um_pool_size;
   size_t                 umpire_host_pool_size;
   size_t                 umpire_pinned_pool_size;
   size_t                 umpire_block_size;
   NALU_HYPRE_Int              own_umpire_device_pool;
   NALU_HYPRE_Int              own_umpire_um_pool;
   NALU_HYPRE_Int              own_umpire_host_pool;
   NALU_HYPRE_Int              own_umpire_pinned_pool;
   umpire_resourcemanager umpire_rm;
#endif
   /* user malloc/free function pointers */
   GPUMallocFunc          user_device_malloc;
   GPUMfreeFunc           user_device_free;
} nalu_hypre_Handle;

/* accessor macros to nalu_hypre_Handle */
#define nalu_hypre_HandleMemoryLocation(nalu_hypre_handle)                 ((nalu_hypre_handle) -> memory_location)
#define nalu_hypre_HandleDefaultExecPolicy(nalu_hypre_handle)              ((nalu_hypre_handle) -> default_exec_policy)
#define nalu_hypre_HandleDeviceData(nalu_hypre_handle)                     ((nalu_hypre_handle) -> device_data)
#define nalu_hypre_HandleDeviceGSMethod(nalu_hypre_handle)                 ((nalu_hypre_handle) -> device_gs_method)

#define nalu_hypre_HandleCurandGenerator(nalu_hypre_handle)                nalu_hypre_DeviceDataCurandGenerator(nalu_hypre_HandleDeviceData(nalu_hypre_handle))
#define nalu_hypre_HandleCublasHandle(nalu_hypre_handle)                   nalu_hypre_DeviceDataCublasHandle(nalu_hypre_HandleDeviceData(nalu_hypre_handle))
#define nalu_hypre_HandleCusparseHandle(nalu_hypre_handle)                 nalu_hypre_DeviceDataCusparseHandle(nalu_hypre_HandleDeviceData(nalu_hypre_handle))
#define nalu_hypre_HandleComputeStream(nalu_hypre_handle)                  nalu_hypre_DeviceDataComputeStream(nalu_hypre_HandleDeviceData(nalu_hypre_handle))
#define nalu_hypre_HandleCubBinGrowth(nalu_hypre_handle)                   nalu_hypre_DeviceDataCubBinGrowth(nalu_hypre_HandleDeviceData(nalu_hypre_handle))
#define nalu_hypre_HandleCubMinBin(nalu_hypre_handle)                      nalu_hypre_DeviceDataCubMinBin(nalu_hypre_HandleDeviceData(nalu_hypre_handle))
#define nalu_hypre_HandleCubMaxBin(nalu_hypre_handle)                      nalu_hypre_DeviceDataCubMaxBin(nalu_hypre_HandleDeviceData(nalu_hypre_handle))
#define nalu_hypre_HandleCubMaxCachedBytes(nalu_hypre_handle)              nalu_hypre_DeviceDataCubMaxCachedBytes(nalu_hypre_HandleDeviceData(nalu_hypre_handle))
#define nalu_hypre_HandleCubDevAllocator(nalu_hypre_handle)                nalu_hypre_DeviceDataCubDevAllocator(nalu_hypre_HandleDeviceData(nalu_hypre_handle))
#define nalu_hypre_HandleCubUvmAllocator(nalu_hypre_handle)                nalu_hypre_DeviceDataCubUvmAllocator(nalu_hypre_HandleDeviceData(nalu_hypre_handle))
#define nalu_hypre_HandleDevice(nalu_hypre_handle)                         nalu_hypre_DeviceDataDevice(nalu_hypre_HandleDeviceData(nalu_hypre_handle))
#define nalu_hypre_HandleDeviceMaxWorkGroupSize(nalu_hypre_handle)         nalu_hypre_DeviceDataDeviceMaxWorkGroupSize(nalu_hypre_HandleDeviceData(nalu_hypre_handle))
#define nalu_hypre_HandleDeviceMaxShmemPerBlock(nalu_hypre_handle)         nalu_hypre_DeviceDataDeviceMaxShmemPerBlock(nalu_hypre_HandleDeviceData(nalu_hypre_handle))
#define nalu_hypre_HandleComputeStreamNum(nalu_hypre_handle)               nalu_hypre_DeviceDataComputeStreamNum(nalu_hypre_HandleDeviceData(nalu_hypre_handle))
#define nalu_hypre_HandleReduceBuffer(nalu_hypre_handle)                   nalu_hypre_DeviceDataReduceBuffer(nalu_hypre_HandleDeviceData(nalu_hypre_handle))
#define nalu_hypre_HandleStructCommRecvBuffer(nalu_hypre_handle)           nalu_hypre_DeviceDataStructCommRecvBuffer(nalu_hypre_HandleDeviceData(nalu_hypre_handle))
#define nalu_hypre_HandleStructCommSendBuffer(nalu_hypre_handle)           nalu_hypre_DeviceDataStructCommSendBuffer(nalu_hypre_HandleDeviceData(nalu_hypre_handle))
#define nalu_hypre_HandleStructCommRecvBufferSize(nalu_hypre_handle)       nalu_hypre_DeviceDataStructCommRecvBufferSize(nalu_hypre_HandleDeviceData(nalu_hypre_handle))
#define nalu_hypre_HandleStructCommSendBufferSize(nalu_hypre_handle)       nalu_hypre_DeviceDataStructCommSendBufferSize(nalu_hypre_HandleDeviceData(nalu_hypre_handle))
#define nalu_hypre_HandleSpgemmUseVendor(nalu_hypre_handle)                nalu_hypre_DeviceDataSpgemmUseVendor(nalu_hypre_HandleDeviceData(nalu_hypre_handle))
#define nalu_hypre_HandleSpMVUseVendor(nalu_hypre_handle)                  nalu_hypre_DeviceDataSpMVUseVendor(nalu_hypre_HandleDeviceData(nalu_hypre_handle))
#define nalu_hypre_HandleSpTransUseVendor(nalu_hypre_handle)               nalu_hypre_DeviceDataSpTransUseVendor(nalu_hypre_HandleDeviceData(nalu_hypre_handle))
#define nalu_hypre_HandleSpgemmAlgorithm(nalu_hypre_handle)                nalu_hypre_DeviceDataSpgemmAlgorithm(nalu_hypre_HandleDeviceData(nalu_hypre_handle))
#define nalu_hypre_HandleSpgemmBinned(nalu_hypre_handle)                   nalu_hypre_DeviceDataSpgemmBinned(nalu_hypre_HandleDeviceData(nalu_hypre_handle))
#define nalu_hypre_HandleSpgemmNumBin(nalu_hypre_handle)                   nalu_hypre_DeviceDataSpgemmNumBin(nalu_hypre_HandleDeviceData(nalu_hypre_handle))
#define nalu_hypre_HandleSpgemmHighestBin(nalu_hypre_handle)               nalu_hypre_DeviceDataSpgemmHighestBin(nalu_hypre_HandleDeviceData(nalu_hypre_handle))
#define nalu_hypre_HandleSpgemmBlockNumDim(nalu_hypre_handle)              nalu_hypre_DeviceDataSpgemmBlockNumDim(nalu_hypre_HandleDeviceData(nalu_hypre_handle))
#define nalu_hypre_HandleSpgemmRownnzEstimateMethod(nalu_hypre_handle)     nalu_hypre_DeviceDataSpgemmRownnzEstimateMethod(nalu_hypre_HandleDeviceData(nalu_hypre_handle))
#define nalu_hypre_HandleSpgemmRownnzEstimateNsamples(nalu_hypre_handle)   nalu_hypre_DeviceDataSpgemmRownnzEstimateNsamples(nalu_hypre_HandleDeviceData(nalu_hypre_handle))
#define nalu_hypre_HandleSpgemmRownnzEstimateMultFactor(nalu_hypre_handle) nalu_hypre_DeviceDataSpgemmRownnzEstimateMultFactor(nalu_hypre_HandleDeviceData(nalu_hypre_handle))
#define nalu_hypre_HandleDeviceAllocator(nalu_hypre_handle)                nalu_hypre_DeviceDataDeviceAllocator(nalu_hypre_HandleDeviceData(nalu_hypre_handle))
#define nalu_hypre_HandleUseGpuRand(nalu_hypre_handle)                     nalu_hypre_DeviceDataUseGpuRand(nalu_hypre_HandleDeviceData(nalu_hypre_handle))

#define nalu_hypre_HandleUserDeviceMalloc(nalu_hypre_handle)               ((nalu_hypre_handle) -> user_device_malloc)
#define nalu_hypre_HandleUserDeviceMfree(nalu_hypre_handle)                ((nalu_hypre_handle) -> user_device_free)

#define nalu_hypre_HandleUmpireResourceMan(nalu_hypre_handle)              ((nalu_hypre_handle) -> umpire_rm)
#define nalu_hypre_HandleUmpireDevicePoolSize(nalu_hypre_handle)           ((nalu_hypre_handle) -> umpire_device_pool_size)
#define nalu_hypre_HandleUmpireUMPoolSize(nalu_hypre_handle)               ((nalu_hypre_handle) -> umpire_um_pool_size)
#define nalu_hypre_HandleUmpireHostPoolSize(nalu_hypre_handle)             ((nalu_hypre_handle) -> umpire_host_pool_size)
#define nalu_hypre_HandleUmpirePinnedPoolSize(nalu_hypre_handle)           ((nalu_hypre_handle) -> umpire_pinned_pool_size)
#define nalu_hypre_HandleUmpireBlockSize(nalu_hypre_handle)                ((nalu_hypre_handle) -> umpire_block_size)
#define nalu_hypre_HandleUmpireDevicePoolName(nalu_hypre_handle)           ((nalu_hypre_handle) -> umpire_device_pool_name)
#define nalu_hypre_HandleUmpireUMPoolName(nalu_hypre_handle)               ((nalu_hypre_handle) -> umpire_um_pool_name)
#define nalu_hypre_HandleUmpireHostPoolName(nalu_hypre_handle)             ((nalu_hypre_handle) -> umpire_host_pool_name)
#define nalu_hypre_HandleUmpirePinnedPoolName(nalu_hypre_handle)           ((nalu_hypre_handle) -> umpire_pinned_pool_name)
#define nalu_hypre_HandleOwnUmpireDevicePool(nalu_hypre_handle)            ((nalu_hypre_handle) -> own_umpire_device_pool)
#define nalu_hypre_HandleOwnUmpireUMPool(nalu_hypre_handle)                ((nalu_hypre_handle) -> own_umpire_um_pool)
#define nalu_hypre_HandleOwnUmpireHostPool(nalu_hypre_handle)              ((nalu_hypre_handle) -> own_umpire_host_pool)
#define nalu_hypre_HandleOwnUmpirePinnedPool(nalu_hypre_handle)            ((nalu_hypre_handle) -> own_umpire_pinned_pool)

#endif
