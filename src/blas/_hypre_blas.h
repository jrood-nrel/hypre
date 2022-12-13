/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Header file for HYPRE BLAS
 *
 *****************************************************************************/

#ifndef NALU_HYPRE_BLAS_H
#define NALU_HYPRE_BLAS_H

#include "_hypre_utilities.h"
#include "fortran.h"

#ifdef __cplusplus
extern "C" {
#endif

/*--------------------------------------------------------------------------
 * Change all 'hypre_' names based on using HYPRE or external library
 *--------------------------------------------------------------------------*/

#ifndef NALU_HYPRE_USING_NALU_HYPRE_BLAS

#define hypre_dasum   hypre_F90_NAME_BLAS(dasum ,DASUM )
#define hypre_daxpy   hypre_F90_NAME_BLAS(daxpy ,DAXPY )
#define hypre_dcopy   hypre_F90_NAME_BLAS(dcopy ,DCOPY )
#define hypre_ddot    hypre_F90_NAME_BLAS(ddot  ,DDOT  )
#define hypre_dgemm   hypre_F90_NAME_BLAS(dgemm ,DGEMM )
#define hypre_dgemv   hypre_F90_NAME_BLAS(dgemv ,DGEMV )
#define hypre_dger    hypre_F90_NAME_BLAS(dger  ,DGER  )
#define hypre_dnrm2   hypre_F90_NAME_BLAS(dnrm2 ,DNRM2 )
#define hypre_drot    hypre_F90_NAME_BLAS(drot  ,DROT  )
#define hypre_dscal   hypre_F90_NAME_BLAS(dscal ,DSCAL )
#define hypre_dswap   hypre_F90_NAME_BLAS(dswap ,DSWAP )
#define hypre_dsymm   hypre_F90_NAME_BLAS(dsymm ,DSYMM )
#define hypre_dsymv   hypre_F90_NAME_BLAS(dsymv ,DSYMV )
#define hypre_dsyr2   hypre_F90_NAME_BLAS(dsyr2 ,DSYR2 )
#define hypre_dsyr2k  hypre_F90_NAME_BLAS(dsyr2k,DSYR2K)
#define hypre_dsyrk   hypre_F90_NAME_BLAS(dsyrk ,DSYRK )
#define hypre_dtrmm   hypre_F90_NAME_BLAS(dtrmm ,DTRMM )
#define hypre_dtrmv   hypre_F90_NAME_BLAS(dtrmv ,DTRMV )
#define hypre_dtrsm   hypre_F90_NAME_BLAS(dtrsm ,DTRSM )
#define hypre_dtrsv   hypre_F90_NAME_BLAS(dtrsv ,DTRSV )
#define hypre_idamax  hypre_F90_NAME_BLAS(idamax,IDAMAX)

#endif

/*--------------------------------------------------------------------------
 * Prototypes
 *--------------------------------------------------------------------------*/

/* dasum.c */
NALU_HYPRE_Real hypre_dasum ( NALU_HYPRE_Int *n , NALU_HYPRE_Real *dx , NALU_HYPRE_Int *incx );

/* daxpy.c */
NALU_HYPRE_Int hypre_daxpy ( NALU_HYPRE_Int *n , NALU_HYPRE_Real *da , NALU_HYPRE_Real *dx , NALU_HYPRE_Int *incx , NALU_HYPRE_Real *dy , NALU_HYPRE_Int *incy );

/* dcopy.c */
NALU_HYPRE_Int hypre_dcopy ( NALU_HYPRE_Int *n , NALU_HYPRE_Real *dx , NALU_HYPRE_Int *incx , NALU_HYPRE_Real *dy , NALU_HYPRE_Int *incy );

/* ddot.c */
NALU_HYPRE_Real hypre_ddot ( NALU_HYPRE_Int *n , NALU_HYPRE_Real *dx , NALU_HYPRE_Int *incx , NALU_HYPRE_Real *dy , NALU_HYPRE_Int *incy );

/* dgemm.c */
NALU_HYPRE_Int hypre_dgemm ( const char *transa , const char *transb , NALU_HYPRE_Int *m , NALU_HYPRE_Int *n , NALU_HYPRE_Int *k , NALU_HYPRE_Real *alpha , NALU_HYPRE_Real *a , NALU_HYPRE_Int *lda , NALU_HYPRE_Real *b , NALU_HYPRE_Int *ldb , NALU_HYPRE_Real *beta , NALU_HYPRE_Real *c , NALU_HYPRE_Int *ldc );

/* dgemv.c */
NALU_HYPRE_Int hypre_dgemv ( const char *trans , NALU_HYPRE_Int *m , NALU_HYPRE_Int *n , NALU_HYPRE_Real *alpha , NALU_HYPRE_Real *a , NALU_HYPRE_Int *lda , NALU_HYPRE_Real *x , NALU_HYPRE_Int *incx , NALU_HYPRE_Real *beta , NALU_HYPRE_Real *y , NALU_HYPRE_Int *incy );

/* dger.c */
NALU_HYPRE_Int hypre_dger ( NALU_HYPRE_Int *m , NALU_HYPRE_Int *n , NALU_HYPRE_Real *alpha , NALU_HYPRE_Real *x , NALU_HYPRE_Int *incx , NALU_HYPRE_Real *y , NALU_HYPRE_Int *incy , NALU_HYPRE_Real *a , NALU_HYPRE_Int *lda );

/* dnrm2.c */
NALU_HYPRE_Real hypre_dnrm2 ( NALU_HYPRE_Int *n , NALU_HYPRE_Real *dx , NALU_HYPRE_Int *incx );

/* drot.c */
NALU_HYPRE_Int hypre_drot ( NALU_HYPRE_Int *n , NALU_HYPRE_Real *dx , NALU_HYPRE_Int *incx , NALU_HYPRE_Real *dy , NALU_HYPRE_Int *incy , NALU_HYPRE_Real *c , NALU_HYPRE_Real *s );

/* dscal.c */
NALU_HYPRE_Int hypre_dscal ( NALU_HYPRE_Int *n , NALU_HYPRE_Real *da , NALU_HYPRE_Real *dx , NALU_HYPRE_Int *incx );

/* dswap.c */
NALU_HYPRE_Int hypre_dswap ( NALU_HYPRE_Int *n , NALU_HYPRE_Real *dx , NALU_HYPRE_Int *incx , NALU_HYPRE_Real *dy , NALU_HYPRE_Int *incy );

/* dsymm.c */
NALU_HYPRE_Int hypre_dsymm ( const char *side , const char *uplo , NALU_HYPRE_Int *m , NALU_HYPRE_Int *n , NALU_HYPRE_Real *alpha , NALU_HYPRE_Real *a , NALU_HYPRE_Int *lda , NALU_HYPRE_Real *b , NALU_HYPRE_Int *ldb , NALU_HYPRE_Real *beta , NALU_HYPRE_Real *c__ , NALU_HYPRE_Int *ldc );

/* dsymv.c */
NALU_HYPRE_Int hypre_dsymv ( const char *uplo , NALU_HYPRE_Int *n , NALU_HYPRE_Real *alpha , NALU_HYPRE_Real *a , NALU_HYPRE_Int *lda , NALU_HYPRE_Real *x , NALU_HYPRE_Int *incx , NALU_HYPRE_Real *beta , NALU_HYPRE_Real *y , NALU_HYPRE_Int *incy );

/* dsyr2.c */
NALU_HYPRE_Int hypre_dsyr2 ( const char *uplo , NALU_HYPRE_Int *n , NALU_HYPRE_Real *alpha , NALU_HYPRE_Real *x , NALU_HYPRE_Int *incx , NALU_HYPRE_Real *y , NALU_HYPRE_Int *incy , NALU_HYPRE_Real *a , NALU_HYPRE_Int *lda );

/* dsyr2k.c */
NALU_HYPRE_Int hypre_dsyr2k ( const char *uplo , const char *trans , NALU_HYPRE_Int *n , NALU_HYPRE_Int *k , NALU_HYPRE_Real *alpha , NALU_HYPRE_Real *a , NALU_HYPRE_Int *lda , NALU_HYPRE_Real *b , NALU_HYPRE_Int *ldb , NALU_HYPRE_Real *beta , NALU_HYPRE_Real *c__ , NALU_HYPRE_Int *ldc );

/* dsyrk.c */
NALU_HYPRE_Int hypre_dsyrk ( const char *uplo , const char *trans , NALU_HYPRE_Int *n , NALU_HYPRE_Int *k , NALU_HYPRE_Real *alpha , NALU_HYPRE_Real *a , NALU_HYPRE_Int *lda , NALU_HYPRE_Real *beta , NALU_HYPRE_Real *c , NALU_HYPRE_Int *ldc );

/* dtrmm.c */
NALU_HYPRE_Int hypre_dtrmm ( const char *side , const char *uplo , const char *transa , const char *diag , NALU_HYPRE_Int *m , NALU_HYPRE_Int *n , NALU_HYPRE_Real *alpha , NALU_HYPRE_Real *a , NALU_HYPRE_Int *lda , NALU_HYPRE_Real *b , NALU_HYPRE_Int *ldb );

/* dtrmv.c */
NALU_HYPRE_Int hypre_dtrmv ( const char *uplo , const char *trans , const char *diag , NALU_HYPRE_Int *n , NALU_HYPRE_Real *a , NALU_HYPRE_Int *lda , NALU_HYPRE_Real *x , NALU_HYPRE_Int *incx );

/* dtrsm.c */
NALU_HYPRE_Int hypre_dtrsm ( const char *side , const char *uplo , const char *transa , const char *diag , NALU_HYPRE_Int *m , NALU_HYPRE_Int *n , NALU_HYPRE_Real *alpha , NALU_HYPRE_Real *a , NALU_HYPRE_Int *lda , NALU_HYPRE_Real *b , NALU_HYPRE_Int *ldb );

/* dtrsv.c */
NALU_HYPRE_Int hypre_dtrsv ( const char *uplo , const char *trans , const char *diag , NALU_HYPRE_Int *n , NALU_HYPRE_Real *a , NALU_HYPRE_Int *lda , NALU_HYPRE_Real *x , NALU_HYPRE_Int *incx );

/* idamax.c */
NALU_HYPRE_Int hypre_idamax ( NALU_HYPRE_Int *n , NALU_HYPRE_Real *dx , NALU_HYPRE_Int *incx );

#ifdef __cplusplus
}
#endif

#endif
