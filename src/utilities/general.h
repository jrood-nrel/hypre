/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
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

#endif /* nalu_hypre_GENERAL_HEADER */

