/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/***** DO NOT use this file outside of the LAPACK directory *****/

/*--------------------------------------------------------------------------
 * This header renames the functions in LAPACK to avoid conflicts
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

/* lapack */
#define dbdsqr_  nalu_hypre_dbdsqr
#define dgebd2_  nalu_hypre_dgebd2
#define dgebrd_  nalu_hypre_dgebrd
#define dgelq2_  nalu_hypre_dgelq2
#define dgelqf_  nalu_hypre_dgelqf
#define dgels_   nalu_hypre_dgels
#define dgeqr2_  nalu_hypre_dgeqr2
#define dgeqrf_  nalu_hypre_dgeqrf
#define dgesvd_  nalu_hypre_dgesvd
#define dgetf2_  nalu_hypre_dgetf2
#define dgetrf_  nalu_hypre_dgetrf
#define dgetri_  nalu_hypre_dgetri
#define dgetrs_  nalu_hypre_dgetrs
#define dlasq1_  nalu_hypre_dlasq1
#define dlasq2_  nalu_hypre_dlasq2
#define dlasrt_  nalu_hypre_dlasrt
#define dorg2l_  nalu_hypre_dorg2l
#define dorg2r_  nalu_hypre_dorg2r
#define dorgbr_  nalu_hypre_dorgbr
#define dorgl2_  nalu_hypre_dorgl2
#define dorglq_  nalu_hypre_dorglq
#define dorgql_  nalu_hypre_dorgql
#define dorgqr_  nalu_hypre_dorgqr
#define dorgtr_  nalu_hypre_dorgtr
#define dorm2r_  nalu_hypre_dorm2r
#define dormbr_  nalu_hypre_dormbr
#define dorml2_  nalu_hypre_dorml2
#define dormlq_  nalu_hypre_dormlq
#define dormqr_  nalu_hypre_dormqr
#define dpotf2_  nalu_hypre_dpotf2
#define dpotrf_  nalu_hypre_dpotrf
#define dpotrs_  nalu_hypre_dpotrs
#define dsteqr_  nalu_hypre_dsteqr
#define dsterf_  nalu_hypre_dsterf
#define dsyev_   nalu_hypre_dsyev
#define dsygs2_  nalu_hypre_dsygs2
#define dsygst_  nalu_hypre_dsygst
#define dsygv_   nalu_hypre_dsygv
#define dsytd2_  nalu_hypre_dsytd2
#define dsytrd_  nalu_hypre_dsytrd
#define dtrti2_  nalu_hypre_dtrti2
#define dtrtri_  nalu_hypre_dtrtri

/* lapack auxiliary routines */
#define dlabad_  nalu_hypre_dlabad
#define dlabrd_  nalu_hypre_dlabrd
#define dlacpy_  nalu_hypre_dlacpy
#define dlae2_   nalu_hypre_dlae2
#define dlaev2_  nalu_hypre_dlaev2
#define dlamch_  nalu_hypre_dlamch
#define dlamc1_  nalu_hypre_dlamc1
#define dlamc2_  nalu_hypre_dlamc2
#define dlamc3_  nalu_hypre_dlamc3
#define dlamc4_  nalu_hypre_dlamc4
#define dlamc5_  nalu_hypre_dlamc5
#define dlange_  nalu_hypre_dlange
#define dlanst_  nalu_hypre_dlanst
#define dlansy_  nalu_hypre_dlansy
#define dlapy2_  nalu_hypre_dlapy2
#define dlarf_   nalu_hypre_dlarf
#define dlarfb_  nalu_hypre_dlarfb
#define dlarfg_  nalu_hypre_dlarfg
#define dlarft_  nalu_hypre_dlarft
#define dlartg_  nalu_hypre_dlartg
#define dlas2_   nalu_hypre_dlas2
#define dlascl_  nalu_hypre_dlascl
#define dlaset_  nalu_hypre_dlaset
#define dlasq3_  nalu_hypre_dlasq3
#define dlasq4_  nalu_hypre_dlasq4
#define dlasq5_  nalu_hypre_dlasq5
#define dlasq6_  nalu_hypre_dlasq6
#define dlasr_   nalu_hypre_dlasr
#define dlassq_  nalu_hypre_dlassq
#define dlasv2_  nalu_hypre_dlasv2
#define dlaswp_  nalu_hypre_dlaswp
#define dlatrd_  nalu_hypre_dlatrd
#define ieeeck_  nalu_hypre_ieeeck
#define ilaenv_  nalu_hypre_ilaenv

/* these auxiliary routines have a different definition in BLAS */
#define lsame_   nalu_hypre_lapack_lsame
#define xerbla_  nalu_hypre_lapack_xerbla

/* this is needed so that lapack can call external BLAS */
#include "_nalu_hypre_blas.h"
