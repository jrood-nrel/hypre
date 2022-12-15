/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_nalu_hypre_utilities.h"
#include "_nalu_hypre_utilities.hpp"

#ifdef NALU_HYPRE_USING_MEMORY_TRACKER
nalu_hypre_MemoryTracker *_nalu_hypre_memory_tracker = NULL;

/* accessor to the global ``_nalu_hypre_memory_tracker'' */
nalu_hypre_MemoryTracker*
nalu_hypre_memory_tracker()
{
   if (!_nalu_hypre_memory_tracker)
   {
      _nalu_hypre_memory_tracker = nalu_hypre_MemoryTrackerCreate();
   }

   return _nalu_hypre_memory_tracker;
}
#endif

/* global variable _nalu_hypre_handle:
 * Outside this file, do NOT access it directly,
 * but use nalu_hypre_handle() instead (see handle.h) */
nalu_hypre_Handle *_nalu_hypre_handle = NULL;

/* accessor to the global ``_nalu_hypre_handle'' */
nalu_hypre_Handle*
nalu_hypre_handle()
{
   if (!_nalu_hypre_handle)
   {
      _nalu_hypre_handle = nalu_hypre_HandleCreate();
   }

   return _nalu_hypre_handle;
}

nalu_hypre_Handle*
nalu_hypre_HandleCreate()
{
   nalu_hypre_Handle *nalu_hypre_handle_ = nalu_hypre_CTAlloc(nalu_hypre_Handle, 1, NALU_HYPRE_MEMORY_HOST);

   nalu_hypre_HandleMemoryLocation(nalu_hypre_handle_) = NALU_HYPRE_MEMORY_DEVICE;

#if defined(NALU_HYPRE_USING_GPU)
   nalu_hypre_HandleDefaultExecPolicy(nalu_hypre_handle_) = NALU_HYPRE_EXEC_DEVICE;
   nalu_hypre_HandleDeviceData(nalu_hypre_handle_) = nalu_hypre_DeviceDataCreate();
   /* Gauss-Seidel: SpTrSV */
   nalu_hypre_HandleDeviceGSMethod(nalu_hypre_handle_) = 1; /* CPU: 0; Cusparse: 1 */
#endif

   return nalu_hypre_handle_;
}

NALU_HYPRE_Int
nalu_hypre_HandleDestroy(nalu_hypre_Handle *nalu_hypre_handle_)
{
   if (!nalu_hypre_handle_)
   {
      return nalu_hypre_error_flag;
   }

#if defined(NALU_HYPRE_USING_GPU)
   nalu_hypre_DeviceDataDestroy(nalu_hypre_HandleDeviceData(nalu_hypre_handle_));
   nalu_hypre_HandleDeviceData(nalu_hypre_handle_) = NULL;
#endif

   nalu_hypre_TFree(nalu_hypre_handle_, NALU_HYPRE_MEMORY_HOST);

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_SetDevice(nalu_hypre_int device_id, nalu_hypre_Handle *nalu_hypre_handle_)
{

#if defined(NALU_HYPRE_USING_DEVICE_OPENMP)
   omp_set_default_device(device_id);
#endif

#if defined(NALU_HYPRE_USING_CUDA)
   NALU_HYPRE_CUDA_CALL( cudaSetDevice(device_id) );
#endif

#if defined(NALU_HYPRE_USING_HIP)
   NALU_HYPRE_HIP_CALL( hipSetDevice(device_id) );
#endif

#if defined(NALU_HYPRE_USING_GPU)
   if (nalu_hypre_handle_)
   {
#if defined(NALU_HYPRE_USING_SYCL)
      if (!nalu_hypre_HandleDevice(nalu_hypre_handle_))
      {
         /* Note: this enforces "explicit scaling," i.e. we treat each tile of a multi-tile GPU as a separate device */
         sycl::platform platform(sycl::gpu_selector{});
         auto gpu_devices = platform.get_devices(sycl::info::device_type::gpu);
         NALU_HYPRE_Int n_devices = 0;
         nalu_hypre_GetDeviceCount(&n_devices);
         if (device_id >= n_devices)
         {
            nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC,
                              "ERROR: SYCL device-ID exceed the number of devices on-node\n");
         }

         NALU_HYPRE_Int local_n_devices = 0;
         NALU_HYPRE_Int i;
         for (i = 0; i < gpu_devices.size(); i++)
         {
            /* WM: commenting out multi-tile GPU stuff for now as it is not yet working */
            // multi-tile GPUs
            /* if (gpu_devices[i].get_info<sycl::info::device::partition_max_sub_devices>() > 0) */
            /* { */
            /*    auto subDevicesDomainNuma = */
            /*       gpu_devices[i].create_sub_devices<sycl::info::partition_property::partition_by_affinity_domain> */
            /*       (sycl::info::partition_affinity_domain::numa); */
            /*    for (auto &tile : subDevicesDomainNuma) */
            /*    { */
            /*       if (local_n_devices == device_id) */
            /*       { */
            /*          nalu_hypre_HandleDevice(nalu_hypre_handle_) = new sycl::device(tile); */
            /*       } */
            /*       local_n_devices++; */
            /*    } */
            /* } */
            /* // single-tile GPUs */
            /* else */
            {
               if (local_n_devices == device_id)
               {
                  nalu_hypre_HandleDevice(nalu_hypre_handle_) = new sycl::device(gpu_devices[i]);
               }
               local_n_devices++;
            }
         }
      }
      nalu_hypre_DeviceDataDeviceMaxWorkGroupSize(nalu_hypre_HandleDeviceData(nalu_hypre_handle_)) =
         nalu_hypre_DeviceDataDevice(nalu_hypre_HandleDeviceData(
                                   nalu_hypre_handle_))->get_info<sycl::info::device::max_work_group_size>();
#else
      nalu_hypre_HandleDevice(nalu_hypre_handle_) = device_id;
#endif // #if defined(NALU_HYPRE_USING_SYCL)
   }
#endif // # if defined(NALU_HYPRE_USING_GPU)

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_GetDeviceMaxShmemSize(nalu_hypre_int device_id, nalu_hypre_Handle *nalu_hypre_handle_)
{
#if defined(NALU_HYPRE_USING_GPU)
   nalu_hypre_int max_size = 0, max_size_optin = 0;
#endif

#if defined(NALU_HYPRE_USING_CUDA)
   cudaDeviceGetAttribute(&max_size, cudaDevAttrMaxSharedMemoryPerBlock, device_id);
   cudaDeviceGetAttribute(&max_size_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, device_id);
#endif

#if defined(NALU_HYPRE_USING_HIP)
   hipDeviceGetAttribute(&max_size, hipDeviceAttributeMaxSharedMemoryPerBlock, device_id);
#endif

#if defined(NALU_HYPRE_USING_GPU)
   nalu_hypre_HandleDeviceMaxShmemPerBlock(nalu_hypre_handle_)[0] = max_size;
   nalu_hypre_HandleDeviceMaxShmemPerBlock(nalu_hypre_handle_)[1] = max_size_optin;
#endif

   return nalu_hypre_error_flag;
}

/* Note: it doesn't return device_id in nalu_hypre_Handle->nalu_hypre_DeviceData,
 *       calls API instead. But these two should match at all times
 */
NALU_HYPRE_Int
nalu_hypre_GetDevice(nalu_hypre_int *device_id)
{
#if defined(NALU_HYPRE_USING_DEVICE_OPENMP)
   *device_id = omp_get_default_device();
#endif

#if defined(NALU_HYPRE_USING_CUDA)
   NALU_HYPRE_CUDA_CALL( cudaGetDevice(device_id) );
#endif

#if defined(NALU_HYPRE_USING_HIP)
   NALU_HYPRE_HIP_CALL( hipGetDevice(device_id) );
#endif

#if defined(NALU_HYPRE_USING_SYCL)
   /* WM: note - no sycl call to get which device is setup for use (if the user has already setup a device at all)
    * Assume the rank/device binding below */
   NALU_HYPRE_Int n_devices, my_id;
   nalu_hypre_MPI_Comm_rank(nalu_hypre_MPI_COMM_WORLD, &my_id);
   nalu_hypre_GetDeviceCount(&n_devices);
   (*device_id) = my_id % n_devices;
#endif

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_GetDeviceCount(nalu_hypre_int *device_count)
{
#if defined(NALU_HYPRE_USING_DEVICE_OPENMP)
   *device_count = omp_get_num_devices();
#endif

#if defined(NALU_HYPRE_USING_CUDA)
   NALU_HYPRE_CUDA_CALL( cudaGetDeviceCount(device_count) );
#endif

#if defined(NALU_HYPRE_USING_HIP)
   NALU_HYPRE_HIP_CALL( hipGetDeviceCount(device_count) );
#endif

#if defined(NALU_HYPRE_USING_SYCL)
   (*device_count) = 0;
   sycl::platform platform(sycl::gpu_selector{});
   auto const& gpu_devices = platform.get_devices(sycl::info::device_type::gpu);
   NALU_HYPRE_Int i;
   for (i = 0; i < gpu_devices.size(); i++)
   {
      /* WM: commenting out multi-tile GPU stuff for now as it is not yet working */
      /* if (gpu_devices[i].get_info<sycl::info::device::partition_max_sub_devices>() > 0) */
      /* { */
      /*    auto subDevicesDomainNuma = */
      /*       gpu_devices[i].create_sub_devices<sycl::info::partition_property::partition_by_affinity_domain> */
      /*       (sycl::info::partition_affinity_domain::numa); */
      /*    (*device_count) += subDevicesDomainNuma.size(); */
      /* } */
      /* else */
      {
         (*device_count)++;
      }
   }
#endif

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_GetDeviceLastError()
{
#if defined(NALU_HYPRE_USING_CUDA)
   NALU_HYPRE_CUDA_CALL( cudaGetLastError() );
#endif

#if defined(NALU_HYPRE_USING_HIP)
   NALU_HYPRE_HIP_CALL( hipGetLastError() );
#endif

#if defined(NALU_HYPRE_USING_SYCL)
   try
   {
      nalu_hypre_HandleComputeStream(nalu_hypre_handle())->wait_and_throw();
   }
   catch (sycl::exception const& e)
   {
      std::cout << "Caught synchronous SYCL exception:\n"
                << e.what() << std::endl;
   }
#endif

   return nalu_hypre_error_flag;
}

/******************************************************************************
 *
 * hypre initialization
 *
 *****************************************************************************/

NALU_HYPRE_Int
NALU_HYPRE_Init()
{
#ifdef NALU_HYPRE_USING_MEMORY_TRACKER
   if (!_nalu_hypre_memory_tracker)
   {
      _nalu_hypre_memory_tracker = nalu_hypre_MemoryTrackerCreate();
   }
#endif

   if (!_nalu_hypre_handle)
   {
      _nalu_hypre_handle = nalu_hypre_HandleCreate();
   }

#if defined(NALU_HYPRE_USING_GPU)
#if !defined(NALU_HYPRE_USING_SYCL)
   /* With sycl, cannot call nalu_hypre_GetDeviceLastError() until after device and queue setup */
   nalu_hypre_GetDeviceLastError();
#endif

   /* Notice: the cudaStream created is specific to the device
    * that was in effect when you created the stream.
    * So, we should first set the device and create the streams
    */
   nalu_hypre_int device_id;
   nalu_hypre_GetDevice(&device_id);
   nalu_hypre_SetDevice(device_id, _nalu_hypre_handle);
   nalu_hypre_GetDeviceMaxShmemSize(device_id, _nalu_hypre_handle);

#if defined(NALU_HYPRE_USING_DEVICE_MALLOC_ASYNC)
   cudaMemPool_t mempool;
   cudaDeviceGetDefaultMemPool(&mempool, device_id);
   uint64_t threshold = UINT64_MAX;
   cudaMemPoolSetAttribute(mempool, cudaMemPoolAttrReleaseThreshold, &threshold);
#endif

   /* To include the cost of creating streams/cudahandles in NALU_HYPRE_Init */
   /* If not here, will be done at the first use */
#if defined(NALU_HYPRE_USING_CUDA_STREAMS)
   nalu_hypre_HandleComputeStream(_nalu_hypre_handle);
#endif

   /* A separate stream for prefetching */
   //nalu_hypre_HandleCudaPrefetchStream(_nalu_hypre_handle);
#endif // NALU_HYPRE_USING_GPU

#if defined(NALU_HYPRE_USING_CUBLAS)
   nalu_hypre_HandleCublasHandle(_nalu_hypre_handle);
#endif

#if defined(NALU_HYPRE_USING_CUSPARSE) || defined(NALU_HYPRE_USING_ROCSPARSE)
   nalu_hypre_HandleCusparseHandle(_nalu_hypre_handle);
#endif

#if defined(NALU_HYPRE_USING_CURAND) || defined(NALU_HYPRE_USING_ROCRAND)
   nalu_hypre_HandleCurandGenerator(_nalu_hypre_handle);
#endif

   /* Check if cuda arch flags in compiling match the device */
#if defined(NALU_HYPRE_USING_CUDA) && defined(NALU_HYPRE_DEBUG)
   nalu_hypre_CudaCompileFlagCheck();
#endif

#if defined(NALU_HYPRE_USING_DEVICE_OPENMP)
   NALU_HYPRE_OMPOffloadOn();
#endif

#if defined(NALU_HYPRE_USING_DEVICE_POOL)
   /* Keep this check here at the end of NALU_HYPRE_Init()
    * Make sure that device pool allocator has not been setup in NALU_HYPRE_Init,
    * otherwise users are not able to set all the parameters
    */
   if ( nalu_hypre_HandleCubDevAllocator(_nalu_hypre_handle) ||
        nalu_hypre_HandleCubUvmAllocator(_nalu_hypre_handle) )
   {
      char msg[256];
      nalu_hypre_sprintf(msg, "%s %s", "ERROR: device pool allocators have been created in", __func__);
      nalu_hypre_fprintf(stderr, "%s\n", msg);
      nalu_hypre_error_w_msg(-1, msg);
   }
#endif

#if defined(NALU_HYPRE_USING_UMPIRE)
   nalu_hypre_UmpireInit(_nalu_hypre_handle);
#endif

   return nalu_hypre_error_flag;
}

/******************************************************************************
 *
 * hypre finalization
 *
 *****************************************************************************/

NALU_HYPRE_Int
NALU_HYPRE_Finalize()
{
#if defined(NALU_HYPRE_USING_UMPIRE)
   nalu_hypre_UmpireFinalize(_nalu_hypre_handle);
#endif

   nalu_hypre_HandleDestroy(_nalu_hypre_handle);

   _nalu_hypre_handle = NULL;

#if !defined(NALU_HYPRE_USING_SYCL)
   /* With sycl, cannot call nalu_hypre_GetDeviceLastError() after destroying the handle */
   nalu_hypre_GetDeviceLastError();
#endif

#ifdef NALU_HYPRE_USING_MEMORY_TRACKER
   nalu_hypre_PrintMemoryTracker(nalu_hypre_total_bytes, nalu_hypre_peak_bytes, nalu_hypre_current_bytes,
                            nalu_hypre_memory_tracker_print, nalu_hypre_memory_tracker_filename);

   nalu_hypre_MemoryTrackerDestroy(_nalu_hypre_memory_tracker);
#endif

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
NALU_HYPRE_PrintDeviceInfo()
{
#if defined(NALU_HYPRE_USING_CUDA)
   nalu_hypre_int dev;
   struct cudaDeviceProp deviceProp;

   NALU_HYPRE_CUDA_CALL( cudaGetDevice(&dev) );
   NALU_HYPRE_CUDA_CALL( cudaGetDeviceProperties(&deviceProp, dev) );
   nalu_hypre_printf("Running on \"%s\", major %d, minor %d, total memory %.2f GB\n", deviceProp.name,
                deviceProp.major, deviceProp.minor, deviceProp.totalGlobalMem / 1e9);
#endif

#if defined(NALU_HYPRE_USING_HIP)
   nalu_hypre_int dev;
   hipDeviceProp_t deviceProp;

   NALU_HYPRE_HIP_CALL( hipGetDevice(&dev) );
   NALU_HYPRE_HIP_CALL( hipGetDeviceProperties(&deviceProp, dev) );
   nalu_hypre_printf("Running on \"%s\", major %d, minor %d, total memory %.2f GB\n", deviceProp.name,
                deviceProp.major, deviceProp.minor, deviceProp.totalGlobalMem / 1e9);
#endif

#if defined(NALU_HYPRE_USING_SYCL)
   auto device = *nalu_hypre_HandleDevice(nalu_hypre_handle());
   auto p_name = device.get_platform().get_info<sycl::info::platform::name>();
   nalu_hypre_printf("Platform Name: %s\n", p_name.c_str());
   auto p_version = device.get_platform().get_info<sycl::info::platform::version>();
   nalu_hypre_printf("Platform Version: %s\n", p_version.c_str());
   auto d_name = device.get_info<sycl::info::device::name>();
   nalu_hypre_printf("Device Name: %s\n", d_name.c_str());
   auto max_work_group = device.get_info<sycl::info::device::max_work_group_size>();
   nalu_hypre_printf("Max Work Groups: %d\n", max_work_group);
   auto max_compute_units = device.get_info<sycl::info::device::max_compute_units>();
   nalu_hypre_printf("Max Compute Units: %d\n", max_compute_units);
#endif

#if defined(NALU_HYPRE_USING_GPU)
   nalu_hypre_printf("MaxSharedMemoryPerBlock %d, MaxSharedMemoryPerBlockOptin %d\n",
                nalu_hypre_HandleDeviceMaxShmemPerBlock(nalu_hypre_handle())[0],
                nalu_hypre_HandleDeviceMaxShmemPerBlock(nalu_hypre_handle())[1]);
#endif

   return nalu_hypre_error_flag;
}

/******************************************************************************
 *
 * hypre Umpire
 *
 *****************************************************************************/

#if defined(NALU_HYPRE_USING_UMPIRE)
NALU_HYPRE_Int
nalu_hypre_UmpireInit(nalu_hypre_Handle *nalu_hypre_handle_)
{
   umpire_resourcemanager_get_instance(&nalu_hypre_HandleUmpireResourceMan(nalu_hypre_handle_));

   nalu_hypre_HandleUmpireDevicePoolSize(nalu_hypre_handle_) = 4LL * 1024 * 1024 * 1024;
   nalu_hypre_HandleUmpireUMPoolSize(nalu_hypre_handle_)     = 4LL * 1024 * 1024 * 1024;
   nalu_hypre_HandleUmpireHostPoolSize(nalu_hypre_handle_)   = 4LL * 1024 * 1024 * 1024;
   nalu_hypre_HandleUmpirePinnedPoolSize(nalu_hypre_handle_) = 4LL * 1024 * 1024 * 1024;

   nalu_hypre_HandleUmpireBlockSize(nalu_hypre_handle_) = 512;

   strcpy(nalu_hypre_HandleUmpireDevicePoolName(nalu_hypre_handle_), "NALU_HYPRE_DEVICE_POOL");
   strcpy(nalu_hypre_HandleUmpireUMPoolName(nalu_hypre_handle_),     "NALU_HYPRE_UM_POOL");
   strcpy(nalu_hypre_HandleUmpireHostPoolName(nalu_hypre_handle_),   "NALU_HYPRE_HOST_POOL");
   strcpy(nalu_hypre_HandleUmpirePinnedPoolName(nalu_hypre_handle_), "NALU_HYPRE_PINNED_POOL");

   nalu_hypre_HandleOwnUmpireDevicePool(nalu_hypre_handle_) = 0;
   nalu_hypre_HandleOwnUmpireUMPool(nalu_hypre_handle_)     = 0;
   nalu_hypre_HandleOwnUmpireHostPool(nalu_hypre_handle_)   = 0;
   nalu_hypre_HandleOwnUmpirePinnedPool(nalu_hypre_handle_) = 0;

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_UmpireFinalize(nalu_hypre_Handle *nalu_hypre_handle_)
{
   umpire_resourcemanager *rm_ptr = &nalu_hypre_HandleUmpireResourceMan(nalu_hypre_handle_);
   umpire_allocator allocator;

#if defined(NALU_HYPRE_USING_UMPIRE_HOST)
   if (nalu_hypre_HandleOwnUmpireHostPool(nalu_hypre_handle_))
   {
      const char *pool_name = nalu_hypre_HandleUmpireHostPoolName(nalu_hypre_handle_);
      umpire_resourcemanager_get_allocator_by_name(rm_ptr, pool_name, &allocator);
      umpire_allocator_release(&allocator);
   }
#endif

#if defined(NALU_HYPRE_USING_UMPIRE_DEVICE)
   if (nalu_hypre_HandleOwnUmpireDevicePool(nalu_hypre_handle_))
   {
      const char *pool_name = nalu_hypre_HandleUmpireDevicePoolName(nalu_hypre_handle_);
      umpire_resourcemanager_get_allocator_by_name(rm_ptr, pool_name, &allocator);
      umpire_allocator_release(&allocator);
   }
#endif

#if defined(NALU_HYPRE_USING_UMPIRE_UM)
   if (nalu_hypre_HandleOwnUmpireUMPool(nalu_hypre_handle_))
   {
      const char *pool_name = nalu_hypre_HandleUmpireUMPoolName(nalu_hypre_handle_);
      umpire_resourcemanager_get_allocator_by_name(rm_ptr, pool_name, &allocator);
      umpire_allocator_release(&allocator);
   }
#endif

#if defined(NALU_HYPRE_USING_UMPIRE_PINNED)
   if (nalu_hypre_HandleOwnUmpirePinnedPool(nalu_hypre_handle_))
   {
      const char *pool_name = nalu_hypre_HandleUmpirePinnedPoolName(nalu_hypre_handle_);
      umpire_resourcemanager_get_allocator_by_name(rm_ptr, pool_name, &allocator);
      umpire_allocator_release(&allocator);
   }
#endif

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
NALU_HYPRE_SetUmpireDevicePoolSize(size_t nbytes)
{
   nalu_hypre_HandleUmpireDevicePoolSize(nalu_hypre_handle()) = nbytes;

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
NALU_HYPRE_SetUmpireUMPoolSize(size_t nbytes)
{
   nalu_hypre_HandleUmpireUMPoolSize(nalu_hypre_handle()) = nbytes;

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
NALU_HYPRE_SetUmpireHostPoolSize(size_t nbytes)
{
   nalu_hypre_HandleUmpireHostPoolSize(nalu_hypre_handle()) = nbytes;

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
NALU_HYPRE_SetUmpirePinnedPoolSize(size_t nbytes)
{
   nalu_hypre_HandleUmpirePinnedPoolSize(nalu_hypre_handle()) = nbytes;

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
NALU_HYPRE_SetUmpireDevicePoolName(const char *pool_name)
{
   if (strlen(pool_name) > NALU_HYPRE_UMPIRE_POOL_NAME_MAX_LEN)
   {
      nalu_hypre_error_in_arg(1);

      return nalu_hypre_error_flag;
   }

   strcpy(nalu_hypre_HandleUmpireDevicePoolName(nalu_hypre_handle()), pool_name);

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
NALU_HYPRE_SetUmpireUMPoolName(const char *pool_name)
{
   if (strlen(pool_name) > NALU_HYPRE_UMPIRE_POOL_NAME_MAX_LEN)
   {
      nalu_hypre_error_in_arg(1);

      return nalu_hypre_error_flag;
   }

   strcpy(nalu_hypre_HandleUmpireUMPoolName(nalu_hypre_handle()), pool_name);

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
NALU_HYPRE_SetUmpireHostPoolName(const char *pool_name)
{
   if (strlen(pool_name) > NALU_HYPRE_UMPIRE_POOL_NAME_MAX_LEN)
   {
      nalu_hypre_error_in_arg(1);

      return nalu_hypre_error_flag;
   }

   strcpy(nalu_hypre_HandleUmpireHostPoolName(nalu_hypre_handle()), pool_name);

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
NALU_HYPRE_SetUmpirePinnedPoolName(const char *pool_name)
{
   if (strlen(pool_name) > NALU_HYPRE_UMPIRE_POOL_NAME_MAX_LEN)
   {
      nalu_hypre_error_in_arg(1);

      return nalu_hypre_error_flag;
   }

   strcpy(nalu_hypre_HandleUmpirePinnedPoolName(nalu_hypre_handle()), pool_name);

   return nalu_hypre_error_flag;
}

#endif /* #if defined(NALU_HYPRE_USING_UMPIRE) */

/******************************************************************************
 *
 * HYPRE memory location
 *
 *****************************************************************************/

NALU_HYPRE_Int
NALU_HYPRE_SetMemoryLocation(NALU_HYPRE_MemoryLocation memory_location)
{
   nalu_hypre_HandleMemoryLocation(nalu_hypre_handle()) = memory_location;

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
NALU_HYPRE_GetMemoryLocation(NALU_HYPRE_MemoryLocation *memory_location)
{
   *memory_location = nalu_hypre_HandleMemoryLocation(nalu_hypre_handle());

   return nalu_hypre_error_flag;
}

/******************************************************************************
 *
 * HYPRE execution policy
 *
 *****************************************************************************/

NALU_HYPRE_Int
NALU_HYPRE_SetExecutionPolicy(NALU_HYPRE_ExecutionPolicy exec_policy)
{
   nalu_hypre_HandleDefaultExecPolicy(nalu_hypre_handle()) = exec_policy;

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
NALU_HYPRE_GetExecutionPolicy(NALU_HYPRE_ExecutionPolicy *exec_policy)
{
   *exec_policy = nalu_hypre_HandleDefaultExecPolicy(nalu_hypre_handle());

   return nalu_hypre_error_flag;
}

