/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "seq_mv.h"
#include "csr_spgemm_device.h"

/*- - - - - - - - - - - - - - - - - - - - - - - - - - *
                Symbolic Multiplication
 *- - - - - - - - - - - - - - - - - - - - - - - - - - */

#if defined(NALU_HYPRE_USING_CUDA) || defined(NALU_HYPRE_USING_HIP)

/* HashKeys: assumed to be initialized as all -1's
 * Key:      assumed to be nonnegative
 * increase by 1 if is a new entry
 */
template <NALU_HYPRE_Int SHMEM_HASH_SIZE, char HASHTYPE, NALU_HYPRE_Int UNROLL_FACTOR>
static __device__ __forceinline__
NALU_HYPRE_Int
nalu_hypre_spgemm_hash_insert_symbl( volatile NALU_HYPRE_Int *HashKeys,
                                NALU_HYPRE_Int           key,
                                NALU_HYPRE_Int          &count )
{
   NALU_HYPRE_Int j = 0;

#pragma unroll UNROLL_FACTOR
   for (NALU_HYPRE_Int i = 0; i < SHMEM_HASH_SIZE; i++)
   {
      /* compute the hash value of key */
      if (i == 0)
      {
         j = key & (SHMEM_HASH_SIZE - 1);
      }
      else
      {
         j = HashFunc<SHMEM_HASH_SIZE, HASHTYPE>(key, i, j);
      }

      /* try to insert key+1 into slot j */
      NALU_HYPRE_Int old = atomicCAS((NALU_HYPRE_Int*)(HashKeys + j), -1, key);

      if (old == -1)
      {
         count++;
         return j;
      }
      if (old == key)
      {
         return j;
      }
   }
   return -1;
}

template <char HASHTYPE>
static __device__ __forceinline__
NALU_HYPRE_Int
nalu_hypre_spgemm_hash_insert_symbl( NALU_HYPRE_Int           HashSize,
                                volatile NALU_HYPRE_Int *HashKeys,
                                NALU_HYPRE_Int           key,
                                NALU_HYPRE_Int          &count )
{
   NALU_HYPRE_Int j = 0;

   for (NALU_HYPRE_Int i = 0; i < HashSize; i++)
   {
      /* compute the hash value of key */
      if (i == 0)
      {
         j = key & (HashSize - 1);
      }
      else
      {
         j = HashFunc<HASHTYPE>(HashSize, key, i, j);
      }

      /* try to insert key+1 into slot j */
      NALU_HYPRE_Int old = atomicCAS((NALU_HYPRE_Int*)(HashKeys + j), -1, key);

      if (old == -1)
      {
         count++;
         return j;
      }
      if (old == key)
      {
         return j;
      }
   }
   return -1;
}

template <NALU_HYPRE_Int SHMEM_HASH_SIZE, char HASHTYPE, NALU_HYPRE_Int GROUP_SIZE, bool HAS_GHASH, bool IA1, NALU_HYPRE_Int UNROLL_FACTOR>
static __device__ __forceinline__
NALU_HYPRE_Int
nalu_hypre_spgemm_compute_row_symbl( NALU_HYPRE_Int           istart_a,
                                NALU_HYPRE_Int           iend_a,
                                const NALU_HYPRE_Int    *ja,
                                const NALU_HYPRE_Int    *ib,
                                const NALU_HYPRE_Int    *jb,
                                volatile NALU_HYPRE_Int *s_HashKeys,
                                NALU_HYPRE_Int           g_HashSize,
                                NALU_HYPRE_Int          *g_HashKeys,
                                char               &failed )
{
   NALU_HYPRE_Int num_new_insert = 0;

   /* load column idx and values of row i of A */
   for (NALU_HYPRE_Int i = istart_a + threadIdx.y; __any_sync(NALU_HYPRE_WARP_FULL_MASK, i < iend_a);
        i += blockDim.y)
   {
      NALU_HYPRE_Int rowB = -1;

      if (threadIdx.x == 0 && i < iend_a)
      {
         rowB = read_only_load(ja + i);
      }

#if 0
      //const NALU_HYPRE_Int ymask = get_mask<4>(...);
      // TODO: need to confirm the behavior of __ballot_sync, leave it here for now
      //const NALU_HYPRE_Int num_valid_rows = __popc(__ballot_sync(ymask, valid_i));
      //for (NALU_HYPRE_Int j = 0; j < num_valid_rows; j++)
#endif

      /* threads in the same ygroup work on one row together */
      rowB = __shfl_sync(NALU_HYPRE_WARP_FULL_MASK, rowB, 0, blockDim.x);
      /* open this row of B, collectively */
      NALU_HYPRE_Int tmp = 0;
      if (rowB != -1 && threadIdx.x < 2)
      {
         tmp = read_only_load(ib + rowB + threadIdx.x);
      }
      const NALU_HYPRE_Int rowB_start = __shfl_sync(NALU_HYPRE_WARP_FULL_MASK, tmp, 0, blockDim.x);
      const NALU_HYPRE_Int rowB_end   = __shfl_sync(NALU_HYPRE_WARP_FULL_MASK, tmp, 1, blockDim.x);

      for (NALU_HYPRE_Int k = rowB_start + threadIdx.x; __any_sync(NALU_HYPRE_WARP_FULL_MASK, k < rowB_end);
           k += blockDim.x)
      {
         if (k < rowB_end)
         {
            if (IA1)
            {
               num_new_insert ++;
            }
            else
            {
               const NALU_HYPRE_Int k_idx = read_only_load(jb + k);
               /* first try to insert into shared memory hash table */
               NALU_HYPRE_Int pos = nalu_hypre_spgemm_hash_insert_symbl<SHMEM_HASH_SIZE, HASHTYPE, UNROLL_FACTOR>
                               (s_HashKeys, k_idx, num_new_insert);

               if (HAS_GHASH && -1 == pos)
               {
                  pos = nalu_hypre_spgemm_hash_insert_symbl<HASHTYPE>
                        (g_HashSize, g_HashKeys, k_idx, num_new_insert);
               }
               /* if failed again, both hash tables must have been full
                  (hash table size estimation was too small).
                  Increase the counter anyhow (will lead to over-counting) */
               if (pos == -1)
               {
                  num_new_insert ++;
                  failed = 1;
               }
            }
         }
      }
   }

   return num_new_insert;
}

template <NALU_HYPRE_Int NUM_GROUPS_PER_BLOCK, NALU_HYPRE_Int GROUP_SIZE, NALU_HYPRE_Int SHMEM_HASH_SIZE, bool HAS_RIND,
          bool CAN_FAIL, char HASHTYPE, bool HAS_GHASH>
__global__ void
nalu_hypre_spgemm_symbolic( nalu_hypre_DeviceItem                   &item,
                       const NALU_HYPRE_Int               M,
                       const NALU_HYPRE_Int* __restrict__ rind,
                       const NALU_HYPRE_Int* __restrict__ ia,
                       const NALU_HYPRE_Int* __restrict__ ja,
                       const NALU_HYPRE_Int* __restrict__ ib,
                       const NALU_HYPRE_Int* __restrict__ jb,
                       const NALU_HYPRE_Int* __restrict__ ig,
                       NALU_HYPRE_Int*       __restrict__ jg,
                       NALU_HYPRE_Int*       __restrict__ rc,
                       char*            __restrict__ rf )
{
   /* number of groups in the grid */
   volatile const NALU_HYPRE_Int grid_num_groups = get_num_groups() * gridDim.x;
   /* group id inside the block */
   volatile const NALU_HYPRE_Int group_id = get_group_id();
   /* group id in the grid */
   volatile const NALU_HYPRE_Int grid_group_id = blockIdx.x * get_num_groups() + group_id;
   /* lane id inside the group */
   volatile const NALU_HYPRE_Int lane_id = get_group_lane_id(item);
   /* shared memory hash table */
#if defined(NALU_HYPRE_SPGEMM_DEVICE_USE_DSHMEM)
   extern __shared__ volatile NALU_HYPRE_Int shared_mem[];
   volatile NALU_HYPRE_Int *s_HashKeys = shared_mem;
#else
   __shared__ volatile NALU_HYPRE_Int s_HashKeys[NUM_GROUPS_PER_BLOCK * SHMEM_HASH_SIZE];
#endif
   /* shared memory hash table for this group */
   volatile NALU_HYPRE_Int *group_s_HashKeys = s_HashKeys + group_id * SHMEM_HASH_SIZE;

   const NALU_HYPRE_Int UNROLL_FACTOR = nalu_hypre_min(NALU_HYPRE_SPGEMM_SYMBL_UNROLL, SHMEM_HASH_SIZE);

   nalu_hypre_device_assert(blockDim.x * blockDim.y == GROUP_SIZE);

   for (NALU_HYPRE_Int i = grid_group_id; __any_sync(NALU_HYPRE_WARP_FULL_MASK, i < M); i += grid_num_groups)
   {
      NALU_HYPRE_Int ii = -1;
      char failed = 0;

      if (HAS_RIND)
      {
         group_read<GROUP_SIZE>(item, rind + i, GROUP_SIZE >= NALU_HYPRE_WARP_SIZE || i < M, ii);
      }
      else
      {
         ii = i;
      }

      /* start/end position of global memory hash table */
      NALU_HYPRE_Int istart_g = 0, iend_g = 0, ghash_size = 0;

      if (HAS_GHASH)
      {
         group_read<GROUP_SIZE>(item, ig + grid_group_id, GROUP_SIZE >= NALU_HYPRE_WARP_SIZE || i < M,
                                istart_g, iend_g);

         /* size of global hash table allocated for this row
           (must be power of 2 and >= the actual size of the row of C - shmem hash size) */
         ghash_size = iend_g - istart_g;

         /* initialize group's global memory hash table */
         for (NALU_HYPRE_Int k = lane_id; k < ghash_size; k += GROUP_SIZE)
         {
            jg[istart_g + k] = -1;
         }
      }

      /* initialize group's shared memory hash table */
      if (GROUP_SIZE >= NALU_HYPRE_WARP_SIZE || i < M)
      {
#pragma unroll UNROLL_FACTOR
         for (NALU_HYPRE_Int k = lane_id; k < SHMEM_HASH_SIZE; k += GROUP_SIZE)
         {
            group_s_HashKeys[k] = -1;
         }
      }

      group_sync<GROUP_SIZE>();

      /* start/end position of row of A */
      NALU_HYPRE_Int istart_a = 0, iend_a = 0;

      /* load the start and end position of row ii of A */
      group_read<GROUP_SIZE>(item, ia + ii, GROUP_SIZE >= NALU_HYPRE_WARP_SIZE || i < M, istart_a, iend_a);

      /* work with two hash tables */
      NALU_HYPRE_Int jsum;

      if (iend_a == istart_a + 1)
      {
         jsum = nalu_hypre_spgemm_compute_row_symbl<SHMEM_HASH_SIZE, HASHTYPE, GROUP_SIZE, HAS_GHASH, true, UNROLL_FACTOR>
                (istart_a, iend_a, ja, ib, jb, group_s_HashKeys, ghash_size, jg + istart_g, failed);
      }
      else
      {
         jsum = nalu_hypre_spgemm_compute_row_symbl<SHMEM_HASH_SIZE, HASHTYPE, GROUP_SIZE, HAS_GHASH, false, UNROLL_FACTOR>
                (istart_a, iend_a, ja, ib, jb, group_s_HashKeys, ghash_size, jg + istart_g, failed);
      }

#if defined(NALU_HYPRE_DEBUG)
      nalu_hypre_device_assert(CAN_FAIL || failed == 0);
#endif

      /* num of nonzeros of this row (an upper bound)
       * use s_HashKeys as shared memory workspace */
      if (GROUP_SIZE <= NALU_HYPRE_WARP_SIZE)
      {
         jsum = group_reduce_sum<NALU_HYPRE_Int, NUM_GROUPS_PER_BLOCK, GROUP_SIZE>(item, jsum);
      }
      else
      {
         group_sync<GROUP_SIZE>();

         jsum = group_reduce_sum<NALU_HYPRE_Int, NUM_GROUPS_PER_BLOCK, GROUP_SIZE>(item, jsum, s_HashKeys);
      }

      /* if this row failed */
      if (CAN_FAIL)
      {
         if (GROUP_SIZE <= NALU_HYPRE_WARP_SIZE)
         {
            failed = (char) group_reduce_sum<nalu_hypre_int, NUM_GROUPS_PER_BLOCK, GROUP_SIZE>(item,
                                                                                          (nalu_hypre_int) failed);
         }
         else
         {
            failed = (char) group_reduce_sum<nalu_hypre_int, NUM_GROUPS_PER_BLOCK, GROUP_SIZE>(item,
                                                                                          (nalu_hypre_int) failed,
                                                                                          s_HashKeys);
         }
      }

      if ((GROUP_SIZE >= NALU_HYPRE_WARP_SIZE || i < M) && lane_id == 0)
      {
#if defined(NALU_HYPRE_DEBUG)
         nalu_hypre_device_assert(ii >= 0);
#endif
         rc[ii] = jsum;

         if (CAN_FAIL)
         {
            rf[ii] = failed > 0;
         }
      }
   }
}

template <NALU_HYPRE_Int BIN, NALU_HYPRE_Int SHMEM_HASH_SIZE, NALU_HYPRE_Int GROUP_SIZE, bool HAS_RIND>
NALU_HYPRE_Int
nalu_hypre_spgemm_symbolic_rownnz( NALU_HYPRE_Int  m,
                              NALU_HYPRE_Int *row_ind, /* input: row indices (length of m) */
                              NALU_HYPRE_Int  k,
                              NALU_HYPRE_Int  n,
                              bool       need_ghash,
                              NALU_HYPRE_Int *d_ia,
                              NALU_HYPRE_Int *d_ja,
                              NALU_HYPRE_Int *d_ib,
                              NALU_HYPRE_Int *d_jb,
                              NALU_HYPRE_Int *d_rc,
                              bool       can_fail,
                              char      *d_rf  /* output: if symbolic mult. failed for each row */ )
{
   const NALU_HYPRE_Int num_groups_per_block = nalu_hypre_spgemm_get_num_groups_per_block<GROUP_SIZE>();
#if defined(NALU_HYPRE_USING_CUDA)
   const NALU_HYPRE_Int BDIMX                = nalu_hypre_min(4, GROUP_SIZE);
#elif defined(NALU_HYPRE_USING_HIP)
   const NALU_HYPRE_Int BDIMX                = nalu_hypre_min(2, GROUP_SIZE);
#endif
   const NALU_HYPRE_Int BDIMY                = GROUP_SIZE / BDIMX;

   /* CUDA kernel configurations: bDim.z is the number of groups in block */
   dim3 bDim(BDIMX, BDIMY, num_groups_per_block);
   nalu_hypre_assert(bDim.x * bDim.y == GROUP_SIZE);
   // grid dimension (number of blocks)
   const NALU_HYPRE_Int num_blocks = nalu_hypre_min( nalu_hypre_HandleSpgemmBlockNumDim(nalu_hypre_handle())[0][BIN],
                                           (NALU_HYPRE_Int) ((m + bDim.z - 1) / bDim.z) );
   dim3 gDim( num_blocks );
   // number of active groups
   NALU_HYPRE_Int num_act_groups = nalu_hypre_min((NALU_HYPRE_Int) (bDim.z * gDim.x), m);

   const char HASH_TYPE = NALU_HYPRE_SPGEMM_HASH_TYPE;
   if (HASH_TYPE != 'L' && HASH_TYPE != 'Q' && HASH_TYPE != 'D')
   {
      nalu_hypre_printf("Unrecognized hash type ... [L(inear), Q(uadratic), D(ouble)]\n");
   }

   /* ---------------------------------------------------------------------------
    * build hash table (no values)
    * ---------------------------------------------------------------------------*/
   NALU_HYPRE_Int *d_ghash_i = NULL;
   NALU_HYPRE_Int *d_ghash_j = NULL;
   NALU_HYPRE_Int  ghash_size = 0;

   if (need_ghash)
   {
      nalu_hypre_SpGemmCreateGlobalHashTable(m, row_ind, num_act_groups, d_rc, SHMEM_HASH_SIZE,
                                        &d_ghash_i, &d_ghash_j, NULL, &ghash_size);
   }

#ifdef NALU_HYPRE_SPGEMM_PRINTF
   NALU_HYPRE_SPGEMM_PRINT("%s[%d], BIN[%d]: m %d k %d n %d, HASH %c, SHMEM_HASH_SIZE %d, GROUP_SIZE %d, "
                      "can_fail %d, need_ghash %d, ghash %p size %d\n",
                      __FILE__, __LINE__, BIN, m, k, n,
                      HASH_TYPE, SHMEM_HASH_SIZE, GROUP_SIZE, can_fail, need_ghash, d_ghash_i, ghash_size);
   NALU_HYPRE_SPGEMM_PRINT("kernel spec [%d %d %d] x [%d %d %d]\n", gDim.x, gDim.y, gDim.z, bDim.x, bDim.y,
                      bDim.z);
#endif

#if defined(NALU_HYPRE_SPGEMM_DEVICE_USE_DSHMEM)
   const size_t shmem_bytes = num_groups_per_block * SHMEM_HASH_SIZE * sizeof(NALU_HYPRE_Int);
#else
   const size_t shmem_bytes = 0;
#endif

   /* ---------------------------------------------------------------------------
    * symbolic multiplication:
    * On output, it provides an upper bound of nnz in rows of C
    * ---------------------------------------------------------------------------*/
   nalu_hypre_assert(HAS_RIND == (row_ind != NULL) );

   /* <NUM_GROUPS_PER_BLOCK, GROUP_SIZE, SHMEM_HASH_SIZE, HAS_RIND, CAN_FAIL, HASHTYPE, HAS_GHASH> */

   if (can_fail)
   {
      if (ghash_size)
      {
         NALU_HYPRE_GPU_LAUNCH2(
            (nalu_hypre_spgemm_symbolic<num_groups_per_block, GROUP_SIZE, SHMEM_HASH_SIZE, HAS_RIND, true, HASH_TYPE, true>),
            gDim, bDim, shmem_bytes,
            m, row_ind, d_ia, d_ja, d_ib, d_jb, d_ghash_i, d_ghash_j, d_rc, d_rf );
      }
      else
      {
         NALU_HYPRE_GPU_LAUNCH2(
            (nalu_hypre_spgemm_symbolic<num_groups_per_block, GROUP_SIZE, SHMEM_HASH_SIZE, HAS_RIND, true, HASH_TYPE, false>),
            gDim, bDim, shmem_bytes,
            m, row_ind, d_ia, d_ja, d_ib, d_jb, d_ghash_i, d_ghash_j, d_rc, d_rf );
      }
   }
   else
   {
      if (ghash_size)
      {
         NALU_HYPRE_GPU_LAUNCH2(
            (nalu_hypre_spgemm_symbolic<num_groups_per_block, GROUP_SIZE, SHMEM_HASH_SIZE, HAS_RIND, false, HASH_TYPE, true>),
            gDim, bDim, shmem_bytes,
            m, row_ind, d_ia, d_ja, d_ib, d_jb, d_ghash_i, d_ghash_j, d_rc, d_rf );
      }
      else
      {
         NALU_HYPRE_GPU_LAUNCH2(
            (nalu_hypre_spgemm_symbolic<num_groups_per_block, GROUP_SIZE, SHMEM_HASH_SIZE, HAS_RIND, false, HASH_TYPE, false>),
            gDim, bDim, shmem_bytes,
            m, row_ind, d_ia, d_ja, d_ib, d_jb, d_ghash_i, d_ghash_j, d_rc, d_rf );
      }
   }

   nalu_hypre_TFree(d_ghash_i, NALU_HYPRE_MEMORY_DEVICE);
   nalu_hypre_TFree(d_ghash_j, NALU_HYPRE_MEMORY_DEVICE);

   return nalu_hypre_error_flag;
}

template <NALU_HYPRE_Int SHMEM_HASH_SIZE, NALU_HYPRE_Int GROUP_SIZE>
NALU_HYPRE_Int nalu_hypre_spgemm_symbolic_max_num_blocks( NALU_HYPRE_Int  multiProcessorCount,
                                                NALU_HYPRE_Int *num_blocks_ptr,
                                                NALU_HYPRE_Int *block_size_ptr )
{
   const char HASH_TYPE = NALU_HYPRE_SPGEMM_HASH_TYPE;
   const NALU_HYPRE_Int num_groups_per_block = nalu_hypre_spgemm_get_num_groups_per_block<GROUP_SIZE>();
   const NALU_HYPRE_Int block_size = num_groups_per_block * GROUP_SIZE;
   nalu_hypre_int numBlocksPerSm = 0;
#if defined(NALU_HYPRE_SPGEMM_DEVICE_USE_DSHMEM)
   const nalu_hypre_int shmem_bytes = num_groups_per_block * SHMEM_HASH_SIZE * sizeof(NALU_HYPRE_Int);
   nalu_hypre_int dynamic_shmem_size = shmem_bytes;
#else
   nalu_hypre_int dynamic_shmem_size = 0;
#endif

#if defined(NALU_HYPRE_SPGEMM_DEVICE_USE_DSHMEM)
#if defined(NALU_HYPRE_USING_CUDA)
   /* with CUDA, to use > 48K shared memory, must use dynamic and must opt-in. BIN = 10 requires 64K */
   const nalu_hypre_int max_shmem_optin = nalu_hypre_HandleDeviceMaxShmemPerBlock(nalu_hypre_handle())[1];
   if (dynamic_shmem_size <= max_shmem_optin)
   {
      NALU_HYPRE_CUDA_CALL( cudaFuncSetAttribute(
                          nalu_hypre_spgemm_symbolic<num_groups_per_block, GROUP_SIZE, SHMEM_HASH_SIZE, true, false, HASH_TYPE, true>,
                          cudaFuncAttributeMaxDynamicSharedMemorySize, dynamic_shmem_size) );

      NALU_HYPRE_CUDA_CALL( cudaFuncSetAttribute(
                          nalu_hypre_spgemm_symbolic<num_groups_per_block, GROUP_SIZE, SHMEM_HASH_SIZE, true, false, HASH_TYPE, false>,
                          cudaFuncAttributeMaxDynamicSharedMemorySize, dynamic_shmem_size) );

      NALU_HYPRE_CUDA_CALL( cudaFuncSetAttribute(
                          nalu_hypre_spgemm_symbolic<num_groups_per_block, GROUP_SIZE, SHMEM_HASH_SIZE, true, true,  HASH_TYPE, true>,
                          cudaFuncAttributeMaxDynamicSharedMemorySize, dynamic_shmem_size) );

      NALU_HYPRE_CUDA_CALL( cudaFuncSetAttribute(
                          nalu_hypre_spgemm_symbolic<num_groups_per_block, GROUP_SIZE, SHMEM_HASH_SIZE, true, true,  HASH_TYPE, false>,
                          cudaFuncAttributeMaxDynamicSharedMemorySize, dynamic_shmem_size) );

      /*
      NALU_HYPRE_CUDA_CALL( cudaFuncSetAttribute(
            nalu_hypre_spgemm_symbolic<num_groups_per_block, GROUP_SIZE, SHMEM_HASH_SIZE, false, true, HASH_TYPE, false>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, dynamic_shmem_size) );
      */
   }
#endif
#endif

#if defined(NALU_HYPRE_USING_CUDA)
   cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &numBlocksPerSm,
      nalu_hypre_spgemm_symbolic<num_groups_per_block, GROUP_SIZE, SHMEM_HASH_SIZE, true, false, HASH_TYPE, true>,
      block_size, dynamic_shmem_size);
#endif

#if defined(NALU_HYPRE_USING_HIP)
   hipOccupancyMaxActiveBlocksPerMultiprocessor(
      &numBlocksPerSm,
      nalu_hypre_spgemm_symbolic<num_groups_per_block, GROUP_SIZE, SHMEM_HASH_SIZE, true, false, HASH_TYPE, true>,
      block_size, dynamic_shmem_size);
#endif

   *num_blocks_ptr = multiProcessorCount * numBlocksPerSm;
   *block_size_ptr = block_size;

   return nalu_hypre_error_flag;
}

#endif /* NALU_HYPRE_USING_CUDA  || defined(NALU_HYPRE_USING_HIP) */
