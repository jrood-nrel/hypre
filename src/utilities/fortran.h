/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Fortran <-> C interface macros
 *
 * Developers should only use the following in hypre:
 *   nalu_hypre_F90_NAME() or nalu_hypre_F90_IFACE()
 *   nalu_hypre_F90_NAME_BLAS()
 *   nalu_hypre_F90_NAME_LAPACK()
 *   any of the interface argument macros at the bottom of this file
 *
 *****************************************************************************/

#ifndef NALU_HYPRE_FORT_HEADER
#define NALU_HYPRE_FORT_HEADER

#include "_nalu_hypre_utilities.h"

#ifdef __cplusplus
extern "C" {
#endif

/*-------------------------------------------------------
 * Define specific name mangling macros to be used below
 *-------------------------------------------------------*/

#define nalu_hypre_F90_NAME_1(name,NAME) name
#define nalu_hypre_F90_NAME_2(name,NAME) name##_
#define nalu_hypre_F90_NAME_3(name,NAME) name##__
#define nalu_hypre_F90_NAME_4(name,NAME) NAME
#define nalu_hypre_F90_NAME_5(name,NAME) _##name##_

/*-------------------------------------------------------
 * Define nalu_hypre_F90_NAME and nalu_hypre_F90_IFACE
 *-------------------------------------------------------*/

#if   (NALU_HYPRE_FMANGLE == 1)
#define nalu_hypre_F90_NAME(name,NAME) nalu_hypre_F90_NAME_1(name,NAME)
#elif (NALU_HYPRE_FMANGLE == 2)
#define nalu_hypre_F90_NAME(name,NAME) nalu_hypre_F90_NAME_2(name,NAME)
#elif (NALU_HYPRE_FMANGLE == 3)
#define nalu_hypre_F90_NAME(name,NAME) nalu_hypre_F90_NAME_3(name,NAME)
#elif (NALU_HYPRE_FMANGLE == 4)
#define nalu_hypre_F90_NAME(name,NAME) nalu_hypre_F90_NAME_4(name,NAME)
#elif (NALU_HYPRE_FMANGLE == 5)
#define nalu_hypre_F90_NAME(name,NAME) nalu_hypre_F90_NAME_5(name,NAME)

#elif defined(NALU_HYPRE_FC_FUNC_)
/* NALU_HYPRE_FC_FUNC_ macro assumes underscores exist in name */
#define nalu_hypre_F90_NAME(name,NAME) NALU_HYPRE_FC_FUNC_(name,NAME)

#else
#define nalu_hypre_F90_NAME(name,NAME) nalu_hypre_F90_NAME_2(name,NAME)

#endif

#define nalu_hypre_F90_IFACE(name,NAME) nalu_hypre_F90_NAME(name,NAME)

/*-------------------------------------------------------
 * Define nalu_hypre_F90_NAME_BLAS
 *-------------------------------------------------------*/

#ifdef NALU_HYPRE_USING_NALU_HYPRE_BLAS
#define nalu_hypre_F90_NAME_BLAS(name,NAME)  nalu_hypre_##name

#elif (NALU_HYPRE_FMANGLE_BLAS == 1)
#define nalu_hypre_F90_NAME_BLAS(name,NAME) nalu_hypre_F90_NAME_1(name,NAME)
#elif (NALU_HYPRE_FMANGLE_BLAS == 2)
#define nalu_hypre_F90_NAME_BLAS(name,NAME) nalu_hypre_F90_NAME_2(name,NAME)
#elif (NALU_HYPRE_FMANGLE_BLAS == 3)
#define nalu_hypre_F90_NAME_BLAS(name,NAME) nalu_hypre_F90_NAME_3(name,NAME)
#elif (NALU_HYPRE_FMANGLE_BLAS == 4)
#define nalu_hypre_F90_NAME_BLAS(name,NAME) nalu_hypre_F90_NAME_4(name,NAME)
#elif (NALU_HYPRE_FMANGLE_BLAS == 5)
#define nalu_hypre_F90_NAME_BLAS(name,NAME) nalu_hypre_F90_NAME_5(name,NAME)

#elif defined(NALU_HYPRE_FC_FUNC)
/* NALU_HYPRE_FC_FUNC macro assumes NO underscores exist in name */
#define nalu_hypre_F90_NAME_BLAS(name,NAME) NALU_HYPRE_FC_FUNC(name,NAME)

#else
#define nalu_hypre_F90_NAME_BLAS(name,NAME) nalu_hypre_F90_NAME_2(name,NAME)

#endif

/*-------------------------------------------------------
 * Define nalu_hypre_F90_NAME_LAPACK
 *-------------------------------------------------------*/

#ifdef NALU_HYPRE_USING_NALU_HYPRE_LAPACK
#define nalu_hypre_F90_NAME_LAPACK(name,NAME)  nalu_hypre_##name

#elif (NALU_HYPRE_FMANGLE_LAPACK == 1)
#define nalu_hypre_F90_NAME_LAPACK(name,NAME) nalu_hypre_F90_NAME_1(name,NAME)
#elif (NALU_HYPRE_FMANGLE_LAPACK == 2)
#define nalu_hypre_F90_NAME_LAPACK(name,NAME) nalu_hypre_F90_NAME_2(name,NAME)
#elif (NALU_HYPRE_FMANGLE_LAPACK == 3)
#define nalu_hypre_F90_NAME_LAPACK(name,NAME) nalu_hypre_F90_NAME_3(name,NAME)
#elif (NALU_HYPRE_FMANGLE_LAPACK == 4)
#define nalu_hypre_F90_NAME_LAPACK(name,NAME) nalu_hypre_F90_NAME_4(name,NAME)
#elif (NALU_HYPRE_FMANGLE_LAPACK == 5)
#define nalu_hypre_F90_NAME_LAPACK(name,NAME) nalu_hypre_F90_NAME_5(name,NAME)

#elif defined(NALU_HYPRE_FC_FUNC)
/* NALU_HYPRE_FC_FUNC macro assumes NO underscores exist in name */
#define nalu_hypre_F90_NAME_LAPACK(name,NAME) NALU_HYPRE_FC_FUNC(name,NAME)

#else
#define nalu_hypre_F90_NAME_LAPACK(name,NAME) nalu_hypre_F90_NAME_2(name,NAME)

#endif

/*-------------------------------------------------------
 * Define interface argument types and macros
 *
 * A Fortran communicator is always the size of an integer
 * and hence usually the size of nalu_hypre_int.
 *-------------------------------------------------------*/

typedef nalu_hypre_int      nalu_hypre_F90_Comm;
typedef NALU_HYPRE_Int      nalu_hypre_F90_Int;
typedef NALU_HYPRE_BigInt   nalu_hypre_F90_BigInt;
typedef NALU_HYPRE_Int      nalu_hypre_F90_IntArray;
typedef NALU_HYPRE_Int      nalu_hypre_F90_IntArrayArray;
typedef NALU_HYPRE_BigInt   nalu_hypre_F90_BigIntArray;
typedef NALU_HYPRE_BigInt   nalu_hypre_F90_BigIntArrayArray;
typedef NALU_HYPRE_Real     nalu_hypre_F90_Real;
typedef NALU_HYPRE_Real     nalu_hypre_F90_RealArray;
typedef NALU_HYPRE_Complex  nalu_hypre_F90_Complex;
typedef NALU_HYPRE_Complex  nalu_hypre_F90_ComplexArray;
typedef NALU_HYPRE_Int     *nalu_hypre_F90_Obj;
typedef NALU_HYPRE_Int     *nalu_hypre_F90_ObjRef;

#define nalu_hypre_F90_PassComm(arg)             (nalu_hypre_MPI_Comm_f2c(*arg))
#define nalu_hypre_F90_PassInt(arg)              ((NALU_HYPRE_Int) *arg)
#define nalu_hypre_F90_PassIntRef(arg)           ((NALU_HYPRE_Int *) arg)
#define nalu_hypre_F90_PassIntArray(arg)         ((NALU_HYPRE_Int *) arg)
#define nalu_hypre_F90_PassIntArrayArray(arg)    ((NALU_HYPRE_Int **) arg)
#define nalu_hypre_F90_PassBigInt(arg)           ((NALU_HYPRE_BigInt) *arg)
#define nalu_hypre_F90_PassBigIntRef(arg)        ((NALU_HYPRE_BigInt *) arg)
#define nalu_hypre_F90_PassBigIntArray(arg)      ((NALU_HYPRE_BigInt *) arg)
#define nalu_hypre_F90_PassBigIntArrayArray(arg) ((NALU_HYPRE_BigInt **) arg)
#define nalu_hypre_F90_PassReal(arg)             ((NALU_HYPRE_Real) *arg)
#define nalu_hypre_F90_PassRealRef(arg)          ((NALU_HYPRE_Real *) arg)
#define nalu_hypre_F90_PassRealArray(arg)        ((NALU_HYPRE_Real *) arg)
#define nalu_hypre_F90_PassComplex(arg)          ((NALU_HYPRE_Complex) *arg)
#define nalu_hypre_F90_PassComplexRef(arg)       ((NALU_HYPRE_Complex *) arg)
#define nalu_hypre_F90_PassComplexArray(arg)     ((NALU_HYPRE_Complex *) arg)
#define nalu_hypre_F90_PassObj(obj,arg)          ((obj) *arg)
#define nalu_hypre_F90_PassObjRef(obj,arg)       ((obj *) arg)

#ifdef __cplusplus
}
#endif

#endif
