/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_utilities.h"

#if defined(NALU_HYPRE_USING_DEVICE_OPENMP)

/* global variables for device OpenMP */
NALU_HYPRE_Int hypre__global_offload = 0;
NALU_HYPRE_Int hypre__offload_device_num;
NALU_HYPRE_Int hypre__offload_host_num;

/* stats */
size_t hypre__target_allc_count = 0;
size_t hypre__target_free_count = 0;
size_t hypre__target_allc_bytes = 0;
size_t hypre__target_free_bytes = 0;

size_t hypre__target_htod_count = 0;
size_t hypre__target_dtoh_count = 0;
size_t hypre__target_htod_bytes = 0;
size_t hypre__target_dtoh_bytes = 0;

/* num: number of bytes */
NALU_HYPRE_Int
NALU_HYPRE_OMPOffload(NALU_HYPRE_Int device, void *ptr, size_t num,
                 const char *type1, const char *type2)
{
   hypre_omp_device_offload(device, ptr, char, 0, num, type1, type2);

   return 0;
}

NALU_HYPRE_Int
NALU_HYPRE_OMPPtrIsMapped(void *p, NALU_HYPRE_Int device_num)
{
   if (hypre__global_offload && !omp_target_is_present(p, device_num))
   {
      printf("HYPRE mapping error: %p has not been mapped to device %d!\n", p, device_num);
      return 1;
   }
   return 0;
}

/* OMP offloading switch */
NALU_HYPRE_Int
NALU_HYPRE_OMPOffloadOn()
{
   hypre__global_offload = 1;
   hypre__offload_device_num = omp_get_default_device();
   hypre__offload_host_num   = omp_get_initial_device();

   /*
   NALU_HYPRE_Int myid, nproc;
   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid);
   hypre_MPI_Comm_size(hypre_MPI_COMM_WORLD, &nproc);
   hypre_fprintf(stdout, "Proc %d: Hypre OMP 4.5 offloading has been turned on. Device %d\n",
                 myid, hypre__offload_device_num);
   */

   return 0;
}

NALU_HYPRE_Int
NALU_HYPRE_OMPOffloadOff()
{
   /*
   NALU_HYPRE_Int myid, nproc;
   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid);
   hypre_MPI_Comm_size(hypre_MPI_COMM_WORLD, &nproc);
   fprintf(stdout, "Proc %d: Hypre OMP 4.5 offloading has been turned off\n", myid);
   */

   hypre__global_offload = 0;
   hypre__offload_device_num = omp_get_initial_device();
   hypre__offload_host_num   = omp_get_initial_device();

   return 0;
}

NALU_HYPRE_Int
NALU_HYPRE_OMPOffloadStatPrint()
{
   hypre_printf("Hypre OMP target memory stats:\n"
                "      ALLOC   %ld bytes, %ld counts\n"
                "      FREE    %ld bytes, %ld counts\n"
                "      HTOD    %ld bytes, %ld counts\n"
                "      DTOH    %ld bytes, %ld counts\n",
                hypre__target_allc_bytes, hypre__target_allc_count,
                hypre__target_free_bytes, hypre__target_free_count,
                hypre__target_htod_bytes, hypre__target_htod_count,
                hypre__target_dtoh_bytes, hypre__target_dtoh_count);

   return 0;
}

#endif /* #if defined(NALU_HYPRE_USING_DEVICE_OPENMP) */

