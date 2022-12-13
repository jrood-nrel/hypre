/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Header file for HYPRE LAPACK
 *
 *****************************************************************************/

#ifndef NALU_HYPRE_LAPACK_H
#define NALU_HYPRE_LAPACK_H

#include "_hypre_utilities.h"
#include "fortran.h"

#ifdef __cplusplus
extern "C" {
#endif

/*--------------------------------------------------------------------------
 * Change all 'hypre_' names based on using HYPRE or external library
 *--------------------------------------------------------------------------*/

#ifndef NALU_HYPRE_USING_NALU_HYPRE_LAPACK

#define hypre_dbdsqr  hypre_F90_NAME_LAPACK(dbdsqr,DBDSQR)
#define hypre_dgebd2  hypre_F90_NAME_LAPACK(dgebd2,DGEBD2)
#define hypre_dgebrd  hypre_F90_NAME_LAPACK(dgebrd,DGEBRD)
#define hypre_dgelq2  hypre_F90_NAME_LAPACK(dgelq2,DGELQ2)
#define hypre_dgelqf  hypre_F90_NAME_LAPACK(dgelqf,DGELQF)
#define hypre_dgels   hypre_F90_NAME_LAPACK(dgels ,DGELS )
#define hypre_dgeqr2  hypre_F90_NAME_LAPACK(dgeqr2,DGEQR2)
#define hypre_dgeqrf  hypre_F90_NAME_LAPACK(dgeqrf,DGEQRF)
#define hypre_dgesvd  hypre_F90_NAME_LAPACK(dgesvd,DGESVD)
#define hypre_dgetf2  hypre_F90_NAME_LAPACK(dgetf2,DGETF2)
#define hypre_dgetrf  hypre_F90_NAME_LAPACK(dgetrf,DGETRF)
#define hypre_dgetri  hypre_F90_NAME_LAPACK(dgetri,DGETRI)
#define hypre_dgetrs  hypre_F90_NAME_LAPACK(dgetrs,DGETRS)
#define hypre_dlasq1  hypre_F90_NAME_LAPACK(dlasq1,DLASQ1)
#define hypre_dlasq2  hypre_F90_NAME_LAPACK(dlasq2,DLASQ2)
#define hypre_dlasrt  hypre_F90_NAME_LAPACK(dlasrt,DLASRT)
#define hypre_dorg2l  hypre_F90_NAME_LAPACK(dorg2l,DORG2L)
#define hypre_dorg2r  hypre_F90_NAME_LAPACK(dorg2r,DORG2R)
#define hypre_dorgbr  hypre_F90_NAME_LAPACK(dorgbr,DORGBR)
#define hypre_dorgl2  hypre_F90_NAME_LAPACK(dorgl2,DORGL2)
#define hypre_dorglq  hypre_F90_NAME_LAPACK(dorglq,DORGLQ)
#define hypre_dorgql  hypre_F90_NAME_LAPACK(dorgql,DORGQL)
#define hypre_dorgqr  hypre_F90_NAME_LAPACK(dorgqr,DORGQR)
#define hypre_dorgtr  hypre_F90_NAME_LAPACK(dorgtr,DORGTR)
#define hypre_dorm2r  hypre_F90_NAME_LAPACK(dorm2r,DORM2R)
#define hypre_dormbr  hypre_F90_NAME_LAPACK(dormbr,DORMBR)
#define hypre_dorml2  hypre_F90_NAME_LAPACK(dorml2,DORML2)
#define hypre_dormlq  hypre_F90_NAME_LAPACK(dormlq,DORMLQ)
#define hypre_dormqr  hypre_F90_NAME_LAPACK(dormqr,DORMQR)
#define hypre_dpotf2  hypre_F90_NAME_LAPACK(dpotf2,DPOTF2)
#define hypre_dpotrf  hypre_F90_NAME_LAPACK(dpotrf,DPOTRF)
#define hypre_dpotrs  hypre_F90_NAME_LAPACK(dpotrs,DPOTRS)
#define hypre_dsteqr  hypre_F90_NAME_LAPACK(dsteqr,DSTEQR)
#define hypre_dsterf  hypre_F90_NAME_LAPACK(dsterf,DSTERF)
#define hypre_dsyev   hypre_F90_NAME_LAPACK(dsyev ,DSYEV )
#define hypre_dsygs2  hypre_F90_NAME_LAPACK(dsygs2,DSYGS2)
#define hypre_dsygst  hypre_F90_NAME_LAPACK(dsygst,DSYGST)
#define hypre_dsygv   hypre_F90_NAME_LAPACK(dsygv ,DSYGV )
#define hypre_dsytd2  hypre_F90_NAME_LAPACK(dsytd2,DSYTD2)
#define hypre_dsytrd  hypre_F90_NAME_LAPACK(dsytrd,DSYTRD)
#define hypre_dtrti2  hypre_F90_NAME_LAPACK(dtrtri,DTRTI2)
#define hypre_dtrtri  hypre_F90_NAME_LAPACK(dtrtri,DTRTRI)

#endif

/*--------------------------------------------------------------------------
 * Prototypes
 *--------------------------------------------------------------------------*/

/* dbdsqr.c */
NALU_HYPRE_Int hypre_dbdsqr (const char *uplo , NALU_HYPRE_Int *n , NALU_HYPRE_Int *ncvt , NALU_HYPRE_Int *nru , NALU_HYPRE_Int *ncc , NALU_HYPRE_Real *d__ , NALU_HYPRE_Real *e , NALU_HYPRE_Real *vt , NALU_HYPRE_Int *ldvt , NALU_HYPRE_Real *u , NALU_HYPRE_Int *ldu , NALU_HYPRE_Real *c__ , NALU_HYPRE_Int *ldc , NALU_HYPRE_Real *work , NALU_HYPRE_Int *info );

/* dgebd2.c */
NALU_HYPRE_Int hypre_dgebd2 ( NALU_HYPRE_Int *m , NALU_HYPRE_Int *n , NALU_HYPRE_Real *a , NALU_HYPRE_Int *lda , NALU_HYPRE_Real *d__ , NALU_HYPRE_Real *e , NALU_HYPRE_Real *tauq , NALU_HYPRE_Real *taup , NALU_HYPRE_Real *work , NALU_HYPRE_Int *info );

/* dgebrd.c */
NALU_HYPRE_Int hypre_dgebrd ( NALU_HYPRE_Int *m , NALU_HYPRE_Int *n , NALU_HYPRE_Real *a , NALU_HYPRE_Int *lda , NALU_HYPRE_Real *d__ , NALU_HYPRE_Real *e , NALU_HYPRE_Real *tauq , NALU_HYPRE_Real *taup , NALU_HYPRE_Real *work , NALU_HYPRE_Int *lwork , NALU_HYPRE_Int *info );

/* dgelq2.c */
NALU_HYPRE_Int hypre_dgelq2 ( NALU_HYPRE_Int *m , NALU_HYPRE_Int *n , NALU_HYPRE_Real *a , NALU_HYPRE_Int *lda , NALU_HYPRE_Real *tau , NALU_HYPRE_Real *work , NALU_HYPRE_Int *info );

/* dgelqf.c */
NALU_HYPRE_Int hypre_dgelqf ( NALU_HYPRE_Int *m , NALU_HYPRE_Int *n , NALU_HYPRE_Real *a , NALU_HYPRE_Int *lda , NALU_HYPRE_Real *tau , NALU_HYPRE_Real *work , NALU_HYPRE_Int *lwork , NALU_HYPRE_Int *info );

/* dgels.c */
NALU_HYPRE_Int hypre_dgels ( char *trans , NALU_HYPRE_Int *m , NALU_HYPRE_Int *n , NALU_HYPRE_Int *nrhs , NALU_HYPRE_Real *a , NALU_HYPRE_Int *lda , NALU_HYPRE_Real *b , NALU_HYPRE_Int *ldb , NALU_HYPRE_Real *work , NALU_HYPRE_Int *lwork , NALU_HYPRE_Int *info );

/* dgeqr2.c */
NALU_HYPRE_Int hypre_dgeqr2 ( NALU_HYPRE_Int *m , NALU_HYPRE_Int *n , NALU_HYPRE_Real *a , NALU_HYPRE_Int *lda , NALU_HYPRE_Real *tau , NALU_HYPRE_Real *work , NALU_HYPRE_Int *info );

/* dgeqrf.c */
NALU_HYPRE_Int hypre_dgeqrf ( NALU_HYPRE_Int *m , NALU_HYPRE_Int *n , NALU_HYPRE_Real *a , NALU_HYPRE_Int *lda , NALU_HYPRE_Real *tau , NALU_HYPRE_Real *work , NALU_HYPRE_Int *lwork , NALU_HYPRE_Int *info );

/* dgesvd.c */
NALU_HYPRE_Int hypre_dgesvd ( char *jobu , char *jobvt , NALU_HYPRE_Int *m , NALU_HYPRE_Int *n , NALU_HYPRE_Real *a , NALU_HYPRE_Int *lda , NALU_HYPRE_Real *s , NALU_HYPRE_Real *u , NALU_HYPRE_Int *ldu , NALU_HYPRE_Real *vt , NALU_HYPRE_Int *ldvt , NALU_HYPRE_Real *work , NALU_HYPRE_Int *lwork , NALU_HYPRE_Int *info );

/* dgetf2.c */
NALU_HYPRE_Int hypre_dgetf2 ( NALU_HYPRE_Int *m , NALU_HYPRE_Int *n , NALU_HYPRE_Real *a , NALU_HYPRE_Int *lda , NALU_HYPRE_Int *ipiv , NALU_HYPRE_Int *info );

/* dgetrf.c */
NALU_HYPRE_Int hypre_dgetrf ( NALU_HYPRE_Int *m , NALU_HYPRE_Int *n , NALU_HYPRE_Real *a , NALU_HYPRE_Int *lda , NALU_HYPRE_Int *ipiv , NALU_HYPRE_Int *info );

/* dgetri.c */
NALU_HYPRE_Int hypre_dgetri ( NALU_HYPRE_Int *n, NALU_HYPRE_Real *a, NALU_HYPRE_Int *lda, NALU_HYPRE_Int *ipiv, NALU_HYPRE_Real *work, NALU_HYPRE_Int *lwork, NALU_HYPRE_Int *info);

/* dgetrs.c */
NALU_HYPRE_Int hypre_dgetrs ( const char *trans , NALU_HYPRE_Int *n , NALU_HYPRE_Int *nrhs , NALU_HYPRE_Real *a , NALU_HYPRE_Int *lda , NALU_HYPRE_Int *ipiv , NALU_HYPRE_Real *b , NALU_HYPRE_Int *ldb , NALU_HYPRE_Int *info );

/* dlasq1.c */
NALU_HYPRE_Int hypre_dlasq1 ( NALU_HYPRE_Int *n , NALU_HYPRE_Real *d__ , NALU_HYPRE_Real *e , NALU_HYPRE_Real *work , NALU_HYPRE_Int *info );

/* dlasq2.c */
NALU_HYPRE_Int hypre_dlasq2 ( NALU_HYPRE_Int *n , NALU_HYPRE_Real *z__ , NALU_HYPRE_Int *info );

/* dlasrt.c */
NALU_HYPRE_Int hypre_dlasrt (const char *id , NALU_HYPRE_Int *n , NALU_HYPRE_Real *d__ , NALU_HYPRE_Int *info );

/* dorg2l.c */
NALU_HYPRE_Int hypre_dorg2l ( NALU_HYPRE_Int *m , NALU_HYPRE_Int *n , NALU_HYPRE_Int *k , NALU_HYPRE_Real *a , NALU_HYPRE_Int *lda , NALU_HYPRE_Real *tau , NALU_HYPRE_Real *work , NALU_HYPRE_Int *info );

/* dorg2r.c */
NALU_HYPRE_Int hypre_dorg2r ( NALU_HYPRE_Int *m , NALU_HYPRE_Int *n , NALU_HYPRE_Int *k , NALU_HYPRE_Real *a , NALU_HYPRE_Int *lda , NALU_HYPRE_Real *tau , NALU_HYPRE_Real *work , NALU_HYPRE_Int *info );

/* dorgbr.c */
NALU_HYPRE_Int hypre_dorgbr (const char *vect , NALU_HYPRE_Int *m , NALU_HYPRE_Int *n , NALU_HYPRE_Int *k , NALU_HYPRE_Real *a , NALU_HYPRE_Int *lda , NALU_HYPRE_Real *tau , NALU_HYPRE_Real *work , NALU_HYPRE_Int *lwork , NALU_HYPRE_Int *info );

/* dorgl2.c */
NALU_HYPRE_Int hypre_dorgl2 ( NALU_HYPRE_Int *m , NALU_HYPRE_Int *n , NALU_HYPRE_Int *k , NALU_HYPRE_Real *a , NALU_HYPRE_Int *lda , NALU_HYPRE_Real *tau , NALU_HYPRE_Real *work , NALU_HYPRE_Int *info );

/* dorglq.c */
NALU_HYPRE_Int hypre_dorglq ( NALU_HYPRE_Int *m , NALU_HYPRE_Int *n , NALU_HYPRE_Int *k , NALU_HYPRE_Real *a , NALU_HYPRE_Int *lda , NALU_HYPRE_Real *tau , NALU_HYPRE_Real *work , NALU_HYPRE_Int *lwork , NALU_HYPRE_Int *info );

/* dorgql.c */
NALU_HYPRE_Int hypre_dorgql ( NALU_HYPRE_Int *m , NALU_HYPRE_Int *n , NALU_HYPRE_Int *k , NALU_HYPRE_Real *a , NALU_HYPRE_Int *lda , NALU_HYPRE_Real *tau , NALU_HYPRE_Real *work , NALU_HYPRE_Int *lwork , NALU_HYPRE_Int *info );

/* dorgqr.c */
NALU_HYPRE_Int hypre_dorgqr ( NALU_HYPRE_Int *m , NALU_HYPRE_Int *n , NALU_HYPRE_Int *k , NALU_HYPRE_Real *a , NALU_HYPRE_Int *lda , NALU_HYPRE_Real *tau , NALU_HYPRE_Real *work , NALU_HYPRE_Int *lwork , NALU_HYPRE_Int *info );

/* dorgtr.c */
NALU_HYPRE_Int hypre_dorgtr (const char *uplo , NALU_HYPRE_Int *n , NALU_HYPRE_Real *a , NALU_HYPRE_Int *lda , NALU_HYPRE_Real *tau , NALU_HYPRE_Real *work , NALU_HYPRE_Int *lwork , NALU_HYPRE_Int *info );

/* dorm2r.c */
NALU_HYPRE_Int hypre_dorm2r (const char *side ,const char *trans , NALU_HYPRE_Int *m , NALU_HYPRE_Int *n , NALU_HYPRE_Int *k , NALU_HYPRE_Real *a , NALU_HYPRE_Int *lda , NALU_HYPRE_Real *tau , NALU_HYPRE_Real *c__ , NALU_HYPRE_Int *ldc , NALU_HYPRE_Real *work , NALU_HYPRE_Int *info );

/* dormbr.c */
NALU_HYPRE_Int hypre_dormbr (const char *vect ,const char *side ,const char *trans , NALU_HYPRE_Int *m , NALU_HYPRE_Int *n , NALU_HYPRE_Int *k , NALU_HYPRE_Real *a , NALU_HYPRE_Int *lda , NALU_HYPRE_Real *tau , NALU_HYPRE_Real *c__ , NALU_HYPRE_Int *ldc , NALU_HYPRE_Real *work , NALU_HYPRE_Int *lwork , NALU_HYPRE_Int *info );

/* dorml2.c */
NALU_HYPRE_Int hypre_dorml2 (const char *side ,const char *trans , NALU_HYPRE_Int *m , NALU_HYPRE_Int *n , NALU_HYPRE_Int *k , NALU_HYPRE_Real *a , NALU_HYPRE_Int *lda , NALU_HYPRE_Real *tau , NALU_HYPRE_Real *c__ , NALU_HYPRE_Int *ldc , NALU_HYPRE_Real *work , NALU_HYPRE_Int *info );

/* dormlq.c */
NALU_HYPRE_Int hypre_dormlq (const char *side ,const char *trans , NALU_HYPRE_Int *m , NALU_HYPRE_Int *n , NALU_HYPRE_Int *k , NALU_HYPRE_Real *a , NALU_HYPRE_Int *lda , NALU_HYPRE_Real *tau , NALU_HYPRE_Real *c__ , NALU_HYPRE_Int *ldc , NALU_HYPRE_Real *work , NALU_HYPRE_Int *lwork , NALU_HYPRE_Int *info );

/* dormqr.c */
NALU_HYPRE_Int hypre_dormqr (const char *side ,const char *trans , NALU_HYPRE_Int *m , NALU_HYPRE_Int *n , NALU_HYPRE_Int *k , NALU_HYPRE_Real *a , NALU_HYPRE_Int *lda , NALU_HYPRE_Real *tau , NALU_HYPRE_Real *c__ , NALU_HYPRE_Int *ldc , NALU_HYPRE_Real *work , NALU_HYPRE_Int *lwork , NALU_HYPRE_Int *info );

/* dpotf2.c */
NALU_HYPRE_Int hypre_dpotf2 (const char *uplo , NALU_HYPRE_Int *n , NALU_HYPRE_Real *a , NALU_HYPRE_Int *lda , NALU_HYPRE_Int *info );

/* dpotrf.c */
NALU_HYPRE_Int hypre_dpotrf (const char *uplo , NALU_HYPRE_Int *n , NALU_HYPRE_Real *a , NALU_HYPRE_Int *lda , NALU_HYPRE_Int *info );

/* dpotrs.c */
NALU_HYPRE_Int hypre_dpotrs ( char *uplo , NALU_HYPRE_Int *n , NALU_HYPRE_Int *nrhs , NALU_HYPRE_Real *a , NALU_HYPRE_Int *lda , NALU_HYPRE_Real *b , NALU_HYPRE_Int *ldb , NALU_HYPRE_Int *info );

/* dsteqr.c */
NALU_HYPRE_Int hypre_dsteqr (const char *compz , NALU_HYPRE_Int *n , NALU_HYPRE_Real *d__ , NALU_HYPRE_Real *e , NALU_HYPRE_Real *z__ , NALU_HYPRE_Int *ldz , NALU_HYPRE_Real *work , NALU_HYPRE_Int *info );

/* dsterf.c */
NALU_HYPRE_Int hypre_dsterf ( NALU_HYPRE_Int *n , NALU_HYPRE_Real *d__ , NALU_HYPRE_Real *e , NALU_HYPRE_Int *info );

/* dsyev.c */
NALU_HYPRE_Int hypre_dsyev (const char *jobz ,const char *uplo , NALU_HYPRE_Int *n , NALU_HYPRE_Real *a , NALU_HYPRE_Int *lda , NALU_HYPRE_Real *w , NALU_HYPRE_Real *work , NALU_HYPRE_Int *lwork , NALU_HYPRE_Int *info );

/* dsygs2.c */
NALU_HYPRE_Int hypre_dsygs2 ( NALU_HYPRE_Int *itype ,const char *uplo , NALU_HYPRE_Int *n , NALU_HYPRE_Real *a , NALU_HYPRE_Int *lda , NALU_HYPRE_Real *b , NALU_HYPRE_Int *ldb , NALU_HYPRE_Int *info );

/* dsygst.c */
NALU_HYPRE_Int hypre_dsygst ( NALU_HYPRE_Int *itype ,const char *uplo , NALU_HYPRE_Int *n , NALU_HYPRE_Real *a , NALU_HYPRE_Int *lda , NALU_HYPRE_Real *b , NALU_HYPRE_Int *ldb , NALU_HYPRE_Int *info );

/* dsygv.c */
NALU_HYPRE_Int hypre_dsygv ( NALU_HYPRE_Int *itype , char *jobz , char *uplo , NALU_HYPRE_Int *n , NALU_HYPRE_Real *a , NALU_HYPRE_Int *lda , NALU_HYPRE_Real *b , NALU_HYPRE_Int *ldb , NALU_HYPRE_Real *w , NALU_HYPRE_Real *work , NALU_HYPRE_Int *lwork , NALU_HYPRE_Int *info );

/* dsytd2.c */
NALU_HYPRE_Int hypre_dsytd2 (const char *uplo , NALU_HYPRE_Int *n , NALU_HYPRE_Real *a , NALU_HYPRE_Int *lda , NALU_HYPRE_Real *d__ , NALU_HYPRE_Real *e , NALU_HYPRE_Real *tau , NALU_HYPRE_Int *info );

/* dsytrd.c */
NALU_HYPRE_Int hypre_dsytrd (const char *uplo , NALU_HYPRE_Int *n , NALU_HYPRE_Real *a , NALU_HYPRE_Int *lda , NALU_HYPRE_Real *d__ , NALU_HYPRE_Real *e , NALU_HYPRE_Real *tau , NALU_HYPRE_Real *work , NALU_HYPRE_Int *lwork , NALU_HYPRE_Int *info );

/* dtrti2.c */
NALU_HYPRE_Int hypre_dtrti2 (const char *uplo, const char *diag, NALU_HYPRE_Int *n, NALU_HYPRE_Real *a, NALU_HYPRE_Int *lda, NALU_HYPRE_Int *info);

/* dtrtri.c */
NALU_HYPRE_Int hypre_dtrtri (const char *uplo, const char *diag, NALU_HYPRE_Int *n, NALU_HYPRE_Real *a, NALU_HYPRE_Int *lda, NALU_HYPRE_Int *info);


#ifdef __cplusplus
}
#endif

#endif
