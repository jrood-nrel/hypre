/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#ifndef GET_ROW_DH
#define GET_ROW_DH

/* #include "euclid_common.h" */

/* "row" refers to global row number */

extern void EuclidGetDimensions(void *A, NALU_HYPRE_Int *beg_row, NALU_HYPRE_Int *rowsLocal, NALU_HYPRE_Int *rowsGlobal);
extern void EuclidGetRow(void *A, NALU_HYPRE_Int row, NALU_HYPRE_Int *len, NALU_HYPRE_Int **ind, NALU_HYPRE_Real **val);
extern void EuclidRestoreRow(void *A, NALU_HYPRE_Int row, NALU_HYPRE_Int *len, NALU_HYPRE_Int **ind, NALU_HYPRE_Real **val);

extern NALU_HYPRE_Int EuclidReadLocalNz(void *A);

extern void PrintMatUsingGetRow(void* A, NALU_HYPRE_Int beg_row, NALU_HYPRE_Int m,
                          NALU_HYPRE_Int *n2o_row, NALU_HYPRE_Int *n2o_col, char *filename);


#endif

