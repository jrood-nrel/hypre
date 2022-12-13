/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * ConjGrad.h header file.
 *
 *****************************************************************************/

#ifndef _CONJGRAD_H
#define _CONJGRAD_H

void PCG_ParaSails(Matrix *mat, ParaSails *ps, NALU_HYPRE_Real *b, NALU_HYPRE_Real *x,
   NALU_HYPRE_Real tol, NALU_HYPRE_Int max_iter);
void FGMRES_ParaSails(Matrix *mat, ParaSails *ps, NALU_HYPRE_Real *b, NALU_HYPRE_Real *x,
   NALU_HYPRE_Int dim, NALU_HYPRE_Real tol, NALU_HYPRE_Int max_iter);

#endif /* _CONJGRAD_H */
