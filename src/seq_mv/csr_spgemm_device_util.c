/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_nalu_hypre_onedpl.hpp"
#include "seq_mv.h"
#include "csr_spgemm_device.h"

#if defined(NALU_HYPRE_USING_GPU)

#if defined(NALU_HYPRE_USING_SYCL)
struct row_size
#else
struct row_size : public thrust::unary_function<NALU_HYPRE_Int, NALU_HYPRE_Int>
#endif
{
   NALU_HYPRE_Int SHMEM_HASH_SIZE;

   row_size(NALU_HYPRE_Int SHMEM_HASH_SIZE_ = NALU_HYPRE_Int()) { SHMEM_HASH_SIZE = SHMEM_HASH_SIZE_; }

   __device__ NALU_HYPRE_Int operator()(const NALU_HYPRE_Int &x) const
   {
      // RL: ???
      return next_power_of_2(x - SHMEM_HASH_SIZE) + x;
   }
};

/* Assume d_c is of length m and contains the size of each row
 *        d_i has size (m+1) on entry
 * generate (i,j,a) with d_c */
void
nalu_hypre_create_ija( NALU_HYPRE_Int       m,
                  NALU_HYPRE_Int      *row_id, /* length of m, row indices; if null, it is [0,1,2,3,...] */
                  NALU_HYPRE_Int      *d_c,    /* d_c[row_id[i]] is the size of ith row */
                  NALU_HYPRE_Int      *d_i,
                  NALU_HYPRE_Int     **d_j,
                  NALU_HYPRE_Complex **d_a,
                  NALU_HYPRE_Int
                  *nnz_ptr /* in/out: if input >= 0, it must be the sum of d_c, remain unchanged in output
                                                     if input <  0, it is computed as the sum of d_c and output */)
{
   NALU_HYPRE_Int nnz = 0;

   nalu_hypre_Memset(d_i, 0, sizeof(NALU_HYPRE_Int), NALU_HYPRE_MEMORY_DEVICE);

#if defined(NALU_HYPRE_USING_SYCL)
   if (row_id)
   {
      NALU_HYPRE_ONEDPL_CALL(std::inclusive_scan,
                        oneapi::dpl::make_permutation_iterator(d_c, row_id),
                        oneapi::dpl::make_permutation_iterator(d_c, row_id) + m,
                        d_i + 1);
   }
   else
   {
      NALU_HYPRE_ONEDPL_CALL(std::inclusive_scan,
                        d_c,
                        d_c + m,
                        d_i + 1);
   }
#else
   if (row_id)
   {
      NALU_HYPRE_THRUST_CALL(inclusive_scan,
                        thrust::make_permutation_iterator(d_c, row_id),
                        thrust::make_permutation_iterator(d_c, row_id) + m,
                        d_i + 1);
   }
   else
   {
      NALU_HYPRE_THRUST_CALL(inclusive_scan,
                        d_c,
                        d_c + m,
                        d_i + 1);
   }
#endif

   if (*nnz_ptr >= 0)
   {
#if defined(NALU_HYPRE_DEBUG)
      nalu_hypre_TMemcpy(&nnz, d_i + m, NALU_HYPRE_Int, 1, NALU_HYPRE_MEMORY_HOST, NALU_HYPRE_MEMORY_DEVICE);
      nalu_hypre_assert(nnz == *nnz_ptr);
#endif
      nnz = *nnz_ptr;
   }
   else
   {
      nalu_hypre_TMemcpy(&nnz, d_i + m, NALU_HYPRE_Int, 1, NALU_HYPRE_MEMORY_HOST, NALU_HYPRE_MEMORY_DEVICE);
      *nnz_ptr = nnz;
   }

   if (d_j)
   {
      *d_j = nalu_hypre_TAlloc(NALU_HYPRE_Int, nnz, NALU_HYPRE_MEMORY_DEVICE);
   }

   if (d_a)
   {
      *d_a = nalu_hypre_TAlloc(NALU_HYPRE_Complex, nnz, NALU_HYPRE_MEMORY_DEVICE);
   }
}

/* Assume d_c is of length m and contains the size of each row
 *        d_i has size (m+1) on entry
 * generate (i,j,a) with row_size(d_c) see above (over allocation) */
void
nalu_hypre_create_ija( NALU_HYPRE_Int       SHMEM_HASH_SIZE,
                  NALU_HYPRE_Int       m,
                  NALU_HYPRE_Int      *row_id,        /* length of m, row indices; if null, it is [0,1,2,3,...] */
                  NALU_HYPRE_Int      *d_c,           /* d_c[row_id[i]] is the size of ith row */
                  NALU_HYPRE_Int      *d_i,
                  NALU_HYPRE_Int     **d_j,
                  NALU_HYPRE_Complex **d_a,
                  NALU_HYPRE_Int      *nnz_ptr )
{
   NALU_HYPRE_Int nnz = 0;

   nalu_hypre_Memset(d_i, 0, sizeof(NALU_HYPRE_Int), NALU_HYPRE_MEMORY_DEVICE);

#if defined(NALU_HYPRE_USING_SYCL)
   if (row_id)
   {
      NALU_HYPRE_ONEDPL_CALL( std::inclusive_scan,
                         oneapi::dpl::make_transform_iterator(oneapi::dpl::make_permutation_iterator(d_c, row_id),
                                                              row_size(SHMEM_HASH_SIZE)),
                         oneapi::dpl::make_transform_iterator(oneapi::dpl::make_permutation_iterator(d_c, row_id),
                                                              row_size(SHMEM_HASH_SIZE)) + m,
                         d_i + 1 );
   }
   else
   {
      NALU_HYPRE_ONEDPL_CALL( std::inclusive_scan,
                         oneapi::dpl::make_transform_iterator(d_c, row_size(SHMEM_HASH_SIZE)),
                         oneapi::dpl::make_transform_iterator(d_c, row_size(SHMEM_HASH_SIZE)) + m,
                         d_i + 1 );
   }
#else
   if (row_id)
   {
      NALU_HYPRE_THRUST_CALL( inclusive_scan,
                         thrust::make_transform_iterator(thrust::make_permutation_iterator(d_c, row_id),
                                                         row_size(SHMEM_HASH_SIZE)),
                         thrust::make_transform_iterator(thrust::make_permutation_iterator(d_c, row_id),
                                                         row_size(SHMEM_HASH_SIZE)) + m,
                         d_i + 1 );
   }
   else
   {
      NALU_HYPRE_THRUST_CALL( inclusive_scan,
                         thrust::make_transform_iterator(d_c, row_size(SHMEM_HASH_SIZE)),
                         thrust::make_transform_iterator(d_c, row_size(SHMEM_HASH_SIZE)) + m,
                         d_i + 1 );
   }
#endif

   nalu_hypre_TMemcpy(&nnz, d_i + m, NALU_HYPRE_Int, 1, NALU_HYPRE_MEMORY_HOST, NALU_HYPRE_MEMORY_DEVICE);

   if (nnz_ptr)
   {
      *nnz_ptr = nnz;
   }

   if (d_j)
   {
      *d_j = nalu_hypre_TAlloc(NALU_HYPRE_Int, nnz, NALU_HYPRE_MEMORY_DEVICE);
   }

   if (d_a)
   {
      *d_a = nalu_hypre_TAlloc(NALU_HYPRE_Complex, nnz, NALU_HYPRE_MEMORY_DEVICE);
   }
}

__global__ void
nalu_hypre_SpGemmGhashSize( nalu_hypre_DeviceItem &item,
                       NALU_HYPRE_Int  num_rows,
                       NALU_HYPRE_Int *row_id,
                       NALU_HYPRE_Int  num_ghash,
                       NALU_HYPRE_Int *row_sizes,
                       NALU_HYPRE_Int *ghash_sizes,
                       NALU_HYPRE_Int  SHMEM_HASH_SIZE )
{
   const NALU_HYPRE_Int global_thread_id = nalu_hypre_gpu_get_grid_thread_id<1, 1>(item);

   if (global_thread_id >= num_ghash)
   {
      return;
   }

   NALU_HYPRE_Int j = 0;

   for (NALU_HYPRE_Int i = global_thread_id; i < num_rows; i += num_ghash)
   {
      const NALU_HYPRE_Int rid = row_id ? read_only_load(&row_id[i]) : i;
      const NALU_HYPRE_Int rnz = read_only_load(&row_sizes[rid]);
      const NALU_HYPRE_Int j1 = next_power_of_2(rnz - SHMEM_HASH_SIZE);
      j = nalu_hypre_max(j, j1);
   }

   ghash_sizes[global_thread_id] = j;
}

NALU_HYPRE_Int
nalu_hypre_SpGemmCreateGlobalHashTable( NALU_HYPRE_Int       num_rows,        /* number of rows */
                                   NALU_HYPRE_Int      *row_id,          /* row_id[i] is index of ith row; i if row_id == NULL */
                                   NALU_HYPRE_Int       num_ghash,       /* number of hash tables <= num_rows */
                                   NALU_HYPRE_Int      *row_sizes,       /* row_sizes[rowid[i]] is the size of ith row */
                                   NALU_HYPRE_Int       SHMEM_HASH_SIZE,
                                   NALU_HYPRE_Int     **ghash_i_ptr,     /* of length num_ghash + 1 */
                                   NALU_HYPRE_Int     **ghash_j_ptr,
                                   NALU_HYPRE_Complex **ghash_a_ptr,
                                   NALU_HYPRE_Int      *ghash_size_ptr )
{
   nalu_hypre_assert(num_ghash <= num_rows);

   NALU_HYPRE_Int *ghash_i, ghash_size;
   dim3 bDim = nalu_hypre_GetDefaultDeviceBlockDimension();

   ghash_i = nalu_hypre_TAlloc(NALU_HYPRE_Int, num_ghash + 1, NALU_HYPRE_MEMORY_DEVICE);
   nalu_hypre_Memset(ghash_i + num_ghash, 0, sizeof(NALU_HYPRE_Int), NALU_HYPRE_MEMORY_DEVICE);
   dim3 gDim = nalu_hypre_GetDefaultDeviceGridDimension(num_ghash, "thread", bDim);
   NALU_HYPRE_GPU_LAUNCH( nalu_hypre_SpGemmGhashSize, gDim, bDim,
                     num_rows, row_id, num_ghash, row_sizes, ghash_i, SHMEM_HASH_SIZE );

   hypreDevice_IntegerExclusiveScan(num_ghash + 1, ghash_i);

   nalu_hypre_TMemcpy(&ghash_size, ghash_i + num_ghash, NALU_HYPRE_Int, 1, NALU_HYPRE_MEMORY_HOST,
                 NALU_HYPRE_MEMORY_DEVICE);

   if (!ghash_size)
   {
      nalu_hypre_TFree(ghash_i, NALU_HYPRE_MEMORY_DEVICE);  nalu_hypre_assert(ghash_i == NULL);
   }

   if (ghash_i_ptr)
   {
      *ghash_i_ptr = ghash_i;
   }

   if (ghash_j_ptr)
   {
      *ghash_j_ptr = nalu_hypre_TAlloc(NALU_HYPRE_Int, ghash_size, NALU_HYPRE_MEMORY_DEVICE);
   }

   if (ghash_a_ptr)
   {
      *ghash_a_ptr = nalu_hypre_TAlloc(NALU_HYPRE_Complex, ghash_size, NALU_HYPRE_MEMORY_DEVICE);
   }

   if (ghash_size_ptr)
   {
      *ghash_size_ptr = ghash_size;
   }

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int nalu_hypre_SpGemmCreateBins( NALU_HYPRE_Int  m,
                                  char       s,
                                  char       t,
                                  char       u,
                                  NALU_HYPRE_Int *d_rc,
                                  bool       d_rc_indice_in,
                                  NALU_HYPRE_Int *d_rc_indice,
                                  NALU_HYPRE_Int *h_bin_ptr )
{
#ifdef NALU_HYPRE_SPGEMM_TIMING
   nalu_hypre_ForceSyncComputeStream(nalu_hypre_handle());
   NALU_HYPRE_Real t1 = nalu_hypre_MPI_Wtime();
#endif

   NALU_HYPRE_Int  num_bins = nalu_hypre_HandleSpgemmNumBin(nalu_hypre_handle());
   NALU_HYPRE_Int *d_bin_ptr = nalu_hypre_TAlloc(NALU_HYPRE_Int, num_bins + 1, NALU_HYPRE_MEMORY_DEVICE);

   /* assume there are no more than 127 = 2^7-1 bins, which should be enough */
   char *d_bin_key = nalu_hypre_TAlloc(char, m, NALU_HYPRE_MEMORY_DEVICE);

#if defined(NALU_HYPRE_USING_SYCL)
   NALU_HYPRE_ONEDPL_CALL( std::transform,
                      d_rc,
                      d_rc + m,
                      d_bin_key,
                      spgemm_bin_op<NALU_HYPRE_Int>(s, t, u) );

   if (!d_rc_indice_in)
   {
      hypreSycl_sequence(d_rc_indice, d_rc_indice + m, 0);
   }

   hypreSycl_stable_sort_by_key(d_bin_key, d_bin_key + m, d_rc_indice);

   NALU_HYPRE_ONEDPL_CALL( oneapi::dpl::lower_bound,
                      d_bin_key,
                      d_bin_key + m,
                      oneapi::dpl::counting_iterator<NALU_HYPRE_Int>(1),
                      oneapi::dpl::counting_iterator<NALU_HYPRE_Int>(num_bins + 2),
                      d_bin_ptr );
#else
   NALU_HYPRE_THRUST_CALL( transform,
                      d_rc,
                      d_rc + m,
                      d_bin_key,
                      spgemm_bin_op<NALU_HYPRE_Int>(s, t, u) );

   if (!d_rc_indice_in)
   {
      NALU_HYPRE_THRUST_CALL( sequence, d_rc_indice, d_rc_indice + m);
   }

   NALU_HYPRE_THRUST_CALL( stable_sort_by_key, d_bin_key, d_bin_key + m, d_rc_indice );

   NALU_HYPRE_THRUST_CALL( lower_bound,
                      d_bin_key,
                      d_bin_key + m,
                      thrust::make_counting_iterator(1),
                      thrust::make_counting_iterator(num_bins + 2),
                      d_bin_ptr );
#endif

   nalu_hypre_TMemcpy(h_bin_ptr, d_bin_ptr, NALU_HYPRE_Int, num_bins + 1, NALU_HYPRE_MEMORY_HOST,
                 NALU_HYPRE_MEMORY_DEVICE);

   nalu_hypre_assert(h_bin_ptr[num_bins] == m);

   nalu_hypre_TFree(d_bin_key, NALU_HYPRE_MEMORY_DEVICE);
   nalu_hypre_TFree(d_bin_ptr, NALU_HYPRE_MEMORY_DEVICE);

#ifdef NALU_HYPRE_SPGEMM_TIMING
   nalu_hypre_ForceSyncComputeStream(nalu_hypre_handle());
   NALU_HYPRE_Real t2 = nalu_hypre_MPI_Wtime() - t1;
   NALU_HYPRE_SPGEMM_PRINT("%s[%d]: Binning time %f\n", __FILE__, __LINE__, t2);
#endif

   return nalu_hypre_error_flag;
}

#endif // #if defined(NALU_HYPRE_USING_GPU)

