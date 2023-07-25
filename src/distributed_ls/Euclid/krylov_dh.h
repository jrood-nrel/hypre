/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#ifndef THREADED_KRYLOV_H
#define THREADED_KRYLOV_H

/* #include "blas_dh.h" */

extern void bicgstab_euclid(Mat_dh A, Euclid_dh ctx, NALU_HYPRE_Real *x, NALU_HYPRE_Real *b, 
                                                              NALU_HYPRE_Int *itsOUT);

extern void cg_euclid(Mat_dh A, Euclid_dh ctx, NALU_HYPRE_Real *x, NALU_HYPRE_Real *b, 
                                                              NALU_HYPRE_Int *itsOUT);

#endif
