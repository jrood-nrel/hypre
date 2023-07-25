/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_nalu_hypre_utilities.h"

#ifdef NALU_HYPRE_COMPLEX

#include <complex.h>

NALU_HYPRE_Complex
nalu_hypre_conj( NALU_HYPRE_Complex value )
{
#if defined(NALU_HYPRE_SINGLE)
   return conjf(value);
#elif defined(NALU_HYPRE_LONG_DOUBLE)
   return conjl(value);
#else
   return conj(value);
#endif
}

NALU_HYPRE_Real
nalu_hypre_cabs( NALU_HYPRE_Complex value )
{
#if defined(NALU_HYPRE_SINGLE)
   return cabsf(value);
#elif defined(NALU_HYPRE_LONG_DOUBLE)
   return cabsl(value);
#else
   return cabs(value);
#endif
}

NALU_HYPRE_Real
nalu_hypre_creal( NALU_HYPRE_Complex value )
{
#if defined(NALU_HYPRE_SINGLE)
   return crealf(value);
#elif defined(NALU_HYPRE_LONG_DOUBLE)
   return creall(value);
#else
   return creal(value);
#endif
}

NALU_HYPRE_Real
nalu_hypre_cimag( NALU_HYPRE_Complex value )
{
#if defined(NALU_HYPRE_SINGLE)
   return cimagf(value);
#elif defined(NALU_HYPRE_LONG_DOUBLE)
   return cimagl(value);
#else
   return cimag(value);
#endif
}

NALU_HYPRE_Complex
nalu_hypre_csqrt( NALU_HYPRE_Complex value )
{
#if defined(NALU_HYPRE_SINGLE)
   return csqrtf(value);
#elif defined(NALU_HYPRE_LONG_DOUBLE)
   return csqrtl(value);
#else
   return csqrt(value);
#endif
}

#endif
