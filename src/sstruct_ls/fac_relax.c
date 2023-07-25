/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *  FAC relaxation. Refinement patches are solved using system pfmg
 *  relaxation.
 ******************************************************************************/

#include "_nalu_hypre_sstruct_ls.h"
#include "fac.h"

#define DEBUG 0

NALU_HYPRE_Int
nalu_hypre_FacLocalRelax(void                 *relax_vdata,
                    nalu_hypre_SStructPMatrix *A,
                    nalu_hypre_SStructPVector *x,
                    nalu_hypre_SStructPVector *b,
                    NALU_HYPRE_Int             num_relax,
                    NALU_HYPRE_Int            *zero_guess)
{
   nalu_hypre_SysPFMGRelaxSetPreRelax(relax_vdata);
   nalu_hypre_SysPFMGRelaxSetMaxIter(relax_vdata, num_relax);
   nalu_hypre_SysPFMGRelaxSetZeroGuess(relax_vdata, *zero_guess);
   nalu_hypre_SysPFMGRelax(relax_vdata, A, b, x);
   zero_guess = 0;

   return 0;
}

