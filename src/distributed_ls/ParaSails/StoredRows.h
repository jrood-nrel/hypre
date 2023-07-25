/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * StoredRows.h header file.
 *
 *****************************************************************************/

#include <stdio.h>
#include "Mem.h"
#include "Matrix.h"

#ifndef _STOREDROWS_H
#define _STOREDROWS_H

typedef struct
{
    Matrix   *mat;   /* the matrix corresponding to the rows stored here */
    Mem      *mem;   /* storage for arrays, indices, and values */

    NALU_HYPRE_Int      size;
    NALU_HYPRE_Int      num_loc;

    NALU_HYPRE_Int     *len;
    NALU_HYPRE_Int    **ind;
    NALU_HYPRE_Real **val;

    NALU_HYPRE_Int      count;
}
StoredRows;

StoredRows *StoredRowsCreate(Matrix *mat, NALU_HYPRE_Int size);
void    StoredRowsDestroy(StoredRows *p);
NALU_HYPRE_Int    *StoredRowsAllocInd(StoredRows *p, NALU_HYPRE_Int len);
NALU_HYPRE_Real *StoredRowsAllocVal(StoredRows *p, NALU_HYPRE_Int len);
void    StoredRowsPut(StoredRows *p, NALU_HYPRE_Int index, NALU_HYPRE_Int len, NALU_HYPRE_Int *ind, NALU_HYPRE_Real *val);
void    StoredRowsGet(StoredRows *p, NALU_HYPRE_Int index, NALU_HYPRE_Int *lenp, NALU_HYPRE_Int **indp, 
          NALU_HYPRE_Real **valp);

#endif /* _STOREDROWS_H */
