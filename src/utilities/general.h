/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * General structures and values
 *
 *****************************************************************************/

#ifndef nalu_hypre_GENERAL_HEADER
#define nalu_hypre_GENERAL_HEADER

#include <math.h>

/*--------------------------------------------------------------------------
 * typedefs
 *--------------------------------------------------------------------------*/

/* This allows us to consistently avoid 'int' throughout hypre */
typedef int                    nalu_hypre_int;
typedef long int               nalu_hypre_longint;
typedef unsigned int           nalu_hypre_uint;
typedef unsigned long int      nalu_hypre_ulongint;
typedef unsigned long long int nalu_hypre_ulonglongint;

/* This allows us to consistently avoid 'double' throughout hypre */
typedef double                 nalu_hypre_double;

/*--------------------------------------------------------------------------
 * Define various functions
 *--------------------------------------------------------------------------*/

#ifndef nalu_hypre_max
#define nalu_hypre_max(a,b)  (((a)<(b)) ? (b) : (a))
#endif
#ifndef nalu_hypre_min
#define nalu_hypre_min(a,b)  (((a)<(b)) ? (a) : (b))
#endif

#ifndef nalu_hypre_abs
#define nalu_hypre_abs(a)  (((a)>0) ? (a) : -(a))
#endif

#ifndef nalu_hypre_round
#define nalu_hypre_round(x)  ( ((x) < 0.0) ? ((NALU_HYPRE_Int)(x - 0.5)) : ((NALU_HYPRE_Int)(x + 0.5)) )
#endif

#ifndef nalu_hypre_pow2
#define nalu_hypre_pow2(i)  ( 1 << (i) )
#endif

#ifndef nalu_hypre_sqrt
#if defined(NALU_HYPRE_SINGLE)
#define nalu_hypre_sqrt sqrtf
#elif defined(NALU_HYPRE_LONG_DOUBLE)
#define nalu_hypre_sqrt sqrtl
#else
#define nalu_hypre_sqrt sqrt
#endif
#endif

#ifndef nalu_hypre_pow
#if defined(NALU_HYPRE_SINGLE)
#define nalu_hypre_pow powf
#elif defined(NALU_HYPRE_LONG_DOUBLE)
#define nalu_hypre_pow powl
#else
#define nalu_hypre_pow pow
#endif
#endif

#ifndef nalu_hypre_ceil
#if defined(NALU_HYPRE_SINGLE)
#define nalu_hypre_ceil ceilf
#elif defined(NALU_HYPRE_LONG_DOUBLE)
#define nalu_hypre_ceil ceill
#else
#define nalu_hypre_ceil ceil
#endif
#endif

#ifndef nalu_hypre_floor
#if defined(NALU_HYPRE_SINGLE)
#define nalu_hypre_floor floorf
#elif defined(NALU_HYPRE_LONG_DOUBLE)
#define nalu_hypre_floor floorl
#else
#define nalu_hypre_floor floor
#endif
#endif

#ifndef nalu_hypre_log
#if defined(NALU_HYPRE_SINGLE)
#define nalu_hypre_log logf
#elif defined(NALU_HYPRE_LONG_DOUBLE)
#define nalu_hypre_log logl
#else
#define nalu_hypre_log log
#endif
#endif

#ifndef nalu_hypre_exp
#if defined(NALU_HYPRE_SINGLE)
#define nalu_hypre_exp expf
#elif defined(NALU_HYPRE_LONG_DOUBLE)
#define nalu_hypre_exp expl
#else
#define nalu_hypre_exp exp
#endif
#endif

#ifndef nalu_hypre_sin
#if defined(NALU_HYPRE_SINGLE)
#define nalu_hypre_sin sinf
#elif defined(NALU_HYPRE_LONG_DOUBLE)
#define nalu_hypre_sin sinl
#else
#define nalu_hypre_sin sin
#endif
#endif

#ifndef nalu_hypre_cos
#if defined(NALU_HYPRE_SINGLE)
#define nalu_hypre_cos cosf
#elif defined(NALU_HYPRE_LONG_DOUBLE)
#define nalu_hypre_cos cosl
#else
#define nalu_hypre_cos cos
#endif
#endif

#ifndef nalu_hypre_atan
#if defined(NALU_HYPRE_SINGLE)
#define nalu_hypre_atan atanf
#elif defined(NALU_HYPRE_LONG_DOUBLE)
#define nalu_hypre_atan atanl
#else
#define nalu_hypre_atan atan
#endif
#endif

#ifndef nalu_hypre_fmod
#if defined(NALU_HYPRE_SINGLE)
#define nalu_hypre_fmod fmodf
#elif defined(NALU_HYPRE_LONG_DOUBLE)
#define nalu_hypre_fmod fmodl
#else
#define nalu_hypre_fmod fmod
#endif
#endif

#endif /* nalu_hypre_GENERAL_HEADER */
