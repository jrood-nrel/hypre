/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#ifndef nalu_hypre_BLOCKTRIDIAG_HEADER
#define nalu_hypre_BLOCKTRIDIAG_HEADER

#include "parcsr_mv/_nalu_hypre_parcsr_mv.h"
#include "parcsr_ls/_nalu_hypre_parcsr_ls.h"

/*--------------------------------------------------------------------------
 * nalu_hypre_BlockTridiag
 *--------------------------------------------------------------------------*/

typedef struct
{
   NALU_HYPRE_Int    num_sweeps;
   NALU_HYPRE_Int    relax_type;
   NALU_HYPRE_Int    *index_set1, *index_set2;
   NALU_HYPRE_Int    print_level;
   NALU_HYPRE_Real threshold;
   nalu_hypre_ParCSRMatrix *A11, *A21, *A22;
   nalu_hypre_ParVector    *F1, *U1, *F2, *U2;
   NALU_HYPRE_Solver       precon1, precon2;

} nalu_hypre_BlockTridiagData;

/*--------------------------------------------------------------------------
 * functions for nalu_hypre_BlockTridiag
 *--------------------------------------------------------------------------*/

void *nalu_hypre_BlockTridiagCreate(void);
NALU_HYPRE_Int  nalu_hypre_BlockTridiagDestroy(void *);
NALU_HYPRE_Int  nalu_hypre_BlockTridiagSetup(void *, nalu_hypre_ParCSRMatrix *,
                                   nalu_hypre_ParVector *, nalu_hypre_ParVector *);
NALU_HYPRE_Int  nalu_hypre_BlockTridiagSolve(void *, nalu_hypre_ParCSRMatrix *,
                                   nalu_hypre_ParVector *, nalu_hypre_ParVector *);
NALU_HYPRE_Int  nalu_hypre_BlockTridiagSetIndexSet(void *, NALU_HYPRE_Int, NALU_HYPRE_Int *);
NALU_HYPRE_Int  nalu_hypre_BlockTridiagSetAMGStrengthThreshold(void *, NALU_HYPRE_Real);
NALU_HYPRE_Int  nalu_hypre_BlockTridiagSetAMGNumSweeps(void *, NALU_HYPRE_Int);
NALU_HYPRE_Int  nalu_hypre_BlockTridiagSetAMGRelaxType(void *, NALU_HYPRE_Int);
NALU_HYPRE_Int  nalu_hypre_BlockTridiagSetPrintLevel(void *, NALU_HYPRE_Int);

#endif

