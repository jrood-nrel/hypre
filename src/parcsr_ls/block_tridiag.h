/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#ifndef hypre_BLOCKTRIDIAG_HEADER
#define hypre_BLOCKTRIDIAG_HEADER

#include "parcsr_mv/_hypre_parcsr_mv.h"
#include "parcsr_ls/_hypre_parcsr_ls.h"

/*--------------------------------------------------------------------------
 * hypre_BlockTridiag
 *--------------------------------------------------------------------------*/

typedef struct
{
   NALU_HYPRE_Int    num_sweeps;
   NALU_HYPRE_Int    relax_type;
   NALU_HYPRE_Int    *index_set1, *index_set2;
   NALU_HYPRE_Int    print_level;
   NALU_HYPRE_Real threshold;
   hypre_ParCSRMatrix *A11, *A21, *A22;
   hypre_ParVector    *F1, *U1, *F2, *U2;
   NALU_HYPRE_Solver       precon1, precon2;

} hypre_BlockTridiagData;

/*--------------------------------------------------------------------------
 * functions for hypre_BlockTridiag
 *--------------------------------------------------------------------------*/

void *hypre_BlockTridiagCreate();
NALU_HYPRE_Int  hypre_BlockTridiagDestroy(void *);
NALU_HYPRE_Int  hypre_BlockTridiagSetup(void *, hypre_ParCSRMatrix *,
                                   hypre_ParVector *, hypre_ParVector *);
NALU_HYPRE_Int  hypre_BlockTridiagSolve(void *, hypre_ParCSRMatrix *,
                                   hypre_ParVector *, hypre_ParVector *);
NALU_HYPRE_Int  hypre_BlockTridiagSetIndexSet(void *, NALU_HYPRE_Int, NALU_HYPRE_Int *);
NALU_HYPRE_Int  hypre_BlockTridiagSetAMGStrengthThreshold(void *, NALU_HYPRE_Real);
NALU_HYPRE_Int  hypre_BlockTridiagSetAMGNumSweeps(void *, NALU_HYPRE_Int);
NALU_HYPRE_Int  hypre_BlockTridiagSetAMGRelaxType(void *, NALU_HYPRE_Int);
NALU_HYPRE_Int  hypre_BlockTridiagSetPrintLevel(void *, NALU_HYPRE_Int);

#endif

