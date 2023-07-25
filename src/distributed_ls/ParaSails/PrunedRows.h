/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * PrunedRows.h header file.
 *
 *****************************************************************************/

#include <stdio.h>
#include "Mem.h"
#include "DiagScale.h"

#ifndef _PRUNEDROWS_H
#define _PRUNEDROWS_H

typedef struct
{
    Mem      *mem;   /* storage for arrays, indices, and values */
    NALU_HYPRE_Int      size;

    NALU_HYPRE_Int     *len;
    NALU_HYPRE_Int    **ind;
}
PrunedRows;

PrunedRows *PrunedRowsCreate(Matrix *mat, NALU_HYPRE_Int size, DiagScale *diag_scale,
  NALU_HYPRE_Real thresh);
void PrunedRowsDestroy(PrunedRows *p);
NALU_HYPRE_Int *PrunedRowsAlloc(PrunedRows *p, NALU_HYPRE_Int len);
void PrunedRowsPut(PrunedRows *p, NALU_HYPRE_Int index, NALU_HYPRE_Int len, NALU_HYPRE_Int *ind);
void PrunedRowsGet(PrunedRows *p, NALU_HYPRE_Int index, NALU_HYPRE_Int *lenp, NALU_HYPRE_Int **indp);

#endif /* _PRUNEDROWS_H */
