/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Numbering.h header file.
 *
 *****************************************************************************/

#include <stdio.h>
#include "Common.h"
#include "Matrix.h"
#include "Hash.h"

#ifndef _NUMBERING_H
#define _NUMBERING_H

struct numbering
{
    NALU_HYPRE_Int   size;    /* max number of indices that can be stored */
    NALU_HYPRE_Int   beg_row;
    NALU_HYPRE_Int   end_row;
    NALU_HYPRE_Int   num_loc; /* number of local indices */
    NALU_HYPRE_Int   num_ind; /* number of indices */

    NALU_HYPRE_Int  *local_to_global;
    Hash *hash;
};

typedef struct numbering Numbering;

Numbering *NumberingCreate(Matrix *m, NALU_HYPRE_Int size);
Numbering *NumberingCreateCopy(Numbering *orig);
void NumberingDestroy(Numbering *numb);
void NumberingLocalToGlobal(Numbering *numb, NALU_HYPRE_Int len, NALU_HYPRE_Int *local, NALU_HYPRE_Int *global);
void NumberingGlobalToLocal(Numbering *numb, NALU_HYPRE_Int len, NALU_HYPRE_Int *global, NALU_HYPRE_Int *local);

#endif /* _NUMBERING_H */
