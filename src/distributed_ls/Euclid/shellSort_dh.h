/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#ifndef SUPPORT_DH
#define SUPPORT_DH

/* #include "euclid_common.h" */

extern void shellSort_int(const NALU_HYPRE_Int n, NALU_HYPRE_Int *x);
extern void shellSort_float(NALU_HYPRE_Int n, NALU_HYPRE_Real *v);

/*
extern void shellSort_int_int(const NALU_HYPRE_Int n, NALU_HYPRE_Int *x, NALU_HYPRE_Int *y);
extern void shellSort_int_float(NALU_HYPRE_Int n, NALU_HYPRE_Int *x, NALU_HYPRE_Real *v);
extern void shellSort_int_int_float(NALU_HYPRE_Int n, NALU_HYPRE_Int *x, NALU_HYPRE_Int *y, NALU_HYPRE_Real *v);
*/

#endif
