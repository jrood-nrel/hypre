/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/
#include "seq_mv.h"

#if defined(NALU_HYPRE_USING_CUDA) || defined(NALU_HYPRE_USING_HIP)

#include "csr_spgemm_device.h"

NALU_HYPRE_Int hypreDevice_CSRSpGemmBinnedGetBlockNumDim()
{
   hypre_int multiProcessorCount = 0;
   /* bins 1, 2, ..., num_bins, are effective; 0 is reserved for empty rows */
   const NALU_HYPRE_Int num_bins = 10;

   hypre_HandleSpgemmNumBin(hypre_handle()) = num_bins;

#if defined(NALU_HYPRE_USING_CUDA)
   cudaDeviceGetAttribute(&multiProcessorCount, cudaDevAttrMultiProcessorCount,
                          hypre_HandleDevice(hypre_handle()));
#endif

#if defined(NALU_HYPRE_USING_HIP)
   hipDeviceGetAttribute(&multiProcessorCount, hipDeviceAttributeMultiprocessorCount,
                         hypre_HandleDevice(hypre_handle()));
#endif

   typedef NALU_HYPRE_Int arrType[4][NALU_HYPRE_SPGEMM_MAX_NBIN + 1];
   arrType &max_nblocks = hypre_HandleSpgemmBlockNumDim(hypre_handle());

   for (NALU_HYPRE_Int i = 0; i < num_bins + 1; i++)
   {
      max_nblocks[0][i] = max_nblocks[1][i] = max_nblocks[2][i] = max_nblocks[3][i] = 0;
   }

   /* symbolic */
   hypre_spgemm_symbolic_max_num_blocks< SYMBL_HASH_SIZE[1], T_GROUP_SIZE[1] >
   (multiProcessorCount, &max_nblocks[0][1], &max_nblocks[2][1]);

   hypre_spgemm_symbolic_max_num_blocks< SYMBL_HASH_SIZE[2], T_GROUP_SIZE[2] >
   (multiProcessorCount, &max_nblocks[0][2], &max_nblocks[2][2]);

   hypre_spgemm_symbolic_max_num_blocks< SYMBL_HASH_SIZE[3], T_GROUP_SIZE[3] >
   (multiProcessorCount, &max_nblocks[0][3], &max_nblocks[2][3]);

   hypre_spgemm_symbolic_max_num_blocks< SYMBL_HASH_SIZE[4], T_GROUP_SIZE[4] >
   (multiProcessorCount, &max_nblocks[0][4], &max_nblocks[2][4]);

   hypre_spgemm_symbolic_max_num_blocks< SYMBL_HASH_SIZE[5], T_GROUP_SIZE[5] >
   (multiProcessorCount, &max_nblocks[0][5], &max_nblocks[2][5]);

   hypre_spgemm_symbolic_max_num_blocks< SYMBL_HASH_SIZE[6], T_GROUP_SIZE[6] >
   (multiProcessorCount, &max_nblocks[0][6], &max_nblocks[2][6]);

   hypre_spgemm_symbolic_max_num_blocks< SYMBL_HASH_SIZE[7], T_GROUP_SIZE[7] >
   (multiProcessorCount, &max_nblocks[0][7], &max_nblocks[2][7]);

   hypre_spgemm_symbolic_max_num_blocks< SYMBL_HASH_SIZE[8], T_GROUP_SIZE[8] >
   (multiProcessorCount, &max_nblocks[0][8], &max_nblocks[2][8]);

   hypre_spgemm_symbolic_max_num_blocks< SYMBL_HASH_SIZE[9], T_GROUP_SIZE[9] >
   (multiProcessorCount, &max_nblocks[0][9], &max_nblocks[2][9]);

   hypre_spgemm_symbolic_max_num_blocks< SYMBL_HASH_SIZE[10], T_GROUP_SIZE[10] >
   (multiProcessorCount, &max_nblocks[0][10], &max_nblocks[2][10]);

   /* numeric */
   hypre_spgemm_numerical_max_num_blocks< NUMER_HASH_SIZE[1], T_GROUP_SIZE[1] >
   (multiProcessorCount, &max_nblocks[1][1], &max_nblocks[3][1]);

   hypre_spgemm_numerical_max_num_blocks< NUMER_HASH_SIZE[2], T_GROUP_SIZE[2] >
   (multiProcessorCount, &max_nblocks[1][2], &max_nblocks[3][2]);

   hypre_spgemm_numerical_max_num_blocks< NUMER_HASH_SIZE[3], T_GROUP_SIZE[3] >
   (multiProcessorCount, &max_nblocks[1][3], &max_nblocks[3][3]);

   hypre_spgemm_numerical_max_num_blocks< NUMER_HASH_SIZE[4], T_GROUP_SIZE[4] >
   (multiProcessorCount, &max_nblocks[1][4], &max_nblocks[3][4]);

   hypre_spgemm_numerical_max_num_blocks< NUMER_HASH_SIZE[5], T_GROUP_SIZE[5] >
   (multiProcessorCount, &max_nblocks[1][5], &max_nblocks[3][5]);

   hypre_spgemm_numerical_max_num_blocks< NUMER_HASH_SIZE[6], T_GROUP_SIZE[6] >
   (multiProcessorCount, &max_nblocks[1][6], &max_nblocks[3][6]);

   hypre_spgemm_numerical_max_num_blocks< NUMER_HASH_SIZE[7], T_GROUP_SIZE[7] >
   (multiProcessorCount, &max_nblocks[1][7], &max_nblocks[3][7]);

   hypre_spgemm_numerical_max_num_blocks< NUMER_HASH_SIZE[8], T_GROUP_SIZE[8] >
   (multiProcessorCount, &max_nblocks[1][8], &max_nblocks[3][8]);

   hypre_spgemm_numerical_max_num_blocks< NUMER_HASH_SIZE[9], T_GROUP_SIZE[9] >
   (multiProcessorCount, &max_nblocks[1][9], &max_nblocks[3][9]);

   hypre_spgemm_numerical_max_num_blocks< NUMER_HASH_SIZE[10], T_GROUP_SIZE[10] >
   (multiProcessorCount, &max_nblocks[1][10], &max_nblocks[3][10]);

   /* highest bin with nonzero num blocks */
   typedef NALU_HYPRE_Int arr2Type[2];
   arr2Type &high_bin = hypre_HandleSpgemmHighestBin(hypre_handle());

   for (NALU_HYPRE_Int i = num_bins; i >= 0; i--) { if (max_nblocks[0][i] > 0) { high_bin[0] = i; break; } }
   for (NALU_HYPRE_Int i = num_bins; i >= 0; i--) { if (max_nblocks[1][i] > 0) { high_bin[1] = i; break; } }

   /* this is just a heuristic; having more blocks (than max active) seems improving performance */
#if defined(NALU_HYPRE_USING_CUDA)
   for (NALU_HYPRE_Int i = 0; i < num_bins + 1; i++) { max_nblocks[0][i] *= 5; max_nblocks[1][i] *= 5; }
#endif

#if defined(NALU_HYPRE_SPGEMM_PRINTF)
   NALU_HYPRE_SPGEMM_PRINT("===========================================================================\n");
   NALU_HYPRE_SPGEMM_PRINT("SM count %d\n", multiProcessorCount);
   NALU_HYPRE_SPGEMM_PRINT("Highest Bin Symbl %d, Numer %d\n",
                      hypre_HandleSpgemmHighestBin(hypre_handle())[0],
                      hypre_HandleSpgemmHighestBin(hypre_handle())[1]);
   NALU_HYPRE_SPGEMM_PRINT("---------------------------------------------------------------------------\n");
   NALU_HYPRE_SPGEMM_PRINT("Bin:      ");
   for (NALU_HYPRE_Int i = 0; i < num_bins + 1; i++) { NALU_HYPRE_SPGEMM_PRINT("%5d ", i); } NALU_HYPRE_SPGEMM_PRINT("\n");
   NALU_HYPRE_SPGEMM_PRINT("---------------------------------------------------------------------------\n");
   NALU_HYPRE_SPGEMM_PRINT("Sym-Bdim: ");
   for (NALU_HYPRE_Int i = 0; i < num_bins + 1; i++) { NALU_HYPRE_SPGEMM_PRINT("%5d ", hypre_HandleSpgemmBlockNumDim(hypre_handle())[2][i]); }
   NALU_HYPRE_SPGEMM_PRINT("\n");
   NALU_HYPRE_SPGEMM_PRINT("Sym-Gdim: ");
   for (NALU_HYPRE_Int i = 0; i < num_bins + 1; i++) { NALU_HYPRE_SPGEMM_PRINT("%5d ", hypre_HandleSpgemmBlockNumDim(hypre_handle())[0][i]); }
   NALU_HYPRE_SPGEMM_PRINT("\n");
   NALU_HYPRE_SPGEMM_PRINT("Num-Bdim: ");
   for (NALU_HYPRE_Int i = 0; i < num_bins + 1; i++) { NALU_HYPRE_SPGEMM_PRINT("%5d ", hypre_HandleSpgemmBlockNumDim(hypre_handle())[3][i]); }
   NALU_HYPRE_SPGEMM_PRINT("\n");
   NALU_HYPRE_SPGEMM_PRINT("Num-Gdim: ");
   for (NALU_HYPRE_Int i = 0; i < num_bins + 1; i++) { NALU_HYPRE_SPGEMM_PRINT("%5d ", hypre_HandleSpgemmBlockNumDim(hypre_handle())[1][i]); }
   NALU_HYPRE_SPGEMM_PRINT("\n");
   NALU_HYPRE_SPGEMM_PRINT("===========================================================================\n");
#endif

   return hypre_error_flag;
}

#endif /* NALU_HYPRE_USING_CUDA  || defined(NALU_HYPRE_USING_HIP) */

