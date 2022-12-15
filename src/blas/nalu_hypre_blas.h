/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/***** DO NOT use this file outside of the BLAS directory *****/

/*--------------------------------------------------------------------------
 * This header renames the functions in BLAS to avoid conflicts
 *--------------------------------------------------------------------------*/

/* blas */
#define dasum_   nalu_hypre_dasum
#define daxpy_   nalu_hypre_daxpy
#define dcopy_   nalu_hypre_dcopy
#define ddot_    nalu_hypre_ddot
#define dgemm_   nalu_hypre_dgemm
#define dgemv_   nalu_hypre_dgemv
#define dger_    nalu_hypre_dger
#define dnrm2_   nalu_hypre_dnrm2
#define drot_    nalu_hypre_drot
#define dscal_   nalu_hypre_dscal
#define dswap_   nalu_hypre_dswap
#define dsymm_   nalu_hypre_dsymm
#define dsymv_   nalu_hypre_dsymv
#define dsyr2_   nalu_hypre_dsyr2
#define dsyr2k_  nalu_hypre_dsyr2k
#define dsyrk_   nalu_hypre_dsyrk
#define dtrmm_   nalu_hypre_dtrmm
#define dtrmv_   nalu_hypre_dtrmv
#define dtrsm_   nalu_hypre_dtrsm
#define dtrsv_   nalu_hypre_dtrsv
#define idamax_  nalu_hypre_idamax

/* f2c library routines */
#define s_cmp    nalu_hypre_s_cmp
#define s_copy   nalu_hypre_s_copy
#define s_cat    nalu_hypre_s_cat
#define d_lg10   nalu_hypre_d_lg10
#define d_sign   nalu_hypre_d_sign
#define pow_dd   nalu_hypre_pow_dd
#define pow_di   nalu_hypre_pow_di

/* these auxiliary routines have a different definition in LAPACK */
#define lsame_   nalu_hypre_blas_lsame
#define xerbla_  nalu_hypre_blas_xerbla

