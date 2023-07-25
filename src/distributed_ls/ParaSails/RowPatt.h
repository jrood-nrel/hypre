/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * RowPatt.h header file.
 *
 *****************************************************************************/

#include <stdio.h>

#ifndef _ROWPATT_H
#define _ROWPATT_H

typedef struct
{
    NALU_HYPRE_Int  maxlen;
    NALU_HYPRE_Int  len;
    NALU_HYPRE_Int  prev_len;
    NALU_HYPRE_Int *ind;
    NALU_HYPRE_Int *mark;
    NALU_HYPRE_Int *buffer; /* buffer used for outputting indices */
    NALU_HYPRE_Int  buflen; /* length of this buffer */
}
RowPatt;

RowPatt *RowPattCreate(NALU_HYPRE_Int maxlen);
void RowPattDestroy(RowPatt *p);
void RowPattReset(RowPatt *p);
void RowPattMerge(RowPatt *p, NALU_HYPRE_Int len, NALU_HYPRE_Int *ind);
void RowPattMergeExt(RowPatt *p, NALU_HYPRE_Int len, NALU_HYPRE_Int *ind, NALU_HYPRE_Int num_loc);
void RowPattGet(RowPatt *p, NALU_HYPRE_Int *lenp, NALU_HYPRE_Int **indp);
void RowPattPrevLevel(RowPatt *p, NALU_HYPRE_Int *lenp, NALU_HYPRE_Int **indp);

#endif /* _ROWPATT_H */
