/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_utilities.h"

#ifdef NALU_HYPRE_COMPLEX

#include <complex.h>

NALU_HYPRE_Complex
hypre_conj( NALU_HYPRE_Complex value )
{
   return conj(value);
}

NALU_HYPRE_Real
hypre_cabs( NALU_HYPRE_Complex value )
{
   return cabs(value);
}

NALU_HYPRE_Real
hypre_creal( NALU_HYPRE_Complex value )
{
   return creal(value);
}

NALU_HYPRE_Real
hypre_cimag( NALU_HYPRE_Complex value )
{
   return cimag(value);
}

NALU_HYPRE_Complex
hypre_csqrt( NALU_HYPRE_Complex value )
{
   return csqrt(value);
}

#endif
