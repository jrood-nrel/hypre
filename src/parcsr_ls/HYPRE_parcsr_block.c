/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * NALU_HYPRE_BlockTridiag interface
 *
 *****************************************************************************/

#include "block_tridiag.h"

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BlockTridiagCreate
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int NALU_HYPRE_BlockTridiagCreate(NALU_HYPRE_Solver *solver)
{
   *solver = (NALU_HYPRE_Solver) hypre_BlockTridiagCreate( ) ;
   return 0;
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_blockTridiagDestroy
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int NALU_HYPRE_BlockTridiagDestroy(NALU_HYPRE_Solver solver)
{
   return (hypre_BlockTridiagDestroy((void *) solver ));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BlockTridiagSetup
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int NALU_HYPRE_BlockTridiagSetup(NALU_HYPRE_Solver solver, NALU_HYPRE_ParCSRMatrix A,
                                  NALU_HYPRE_ParVector b, NALU_HYPRE_ParVector x)
{
   return (hypre_BlockTridiagSetup((void *) solver, (hypre_ParCSRMatrix *) A,
                                   (hypre_ParVector *) b, (hypre_ParVector *) x));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BlockTridiagSolve
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int NALU_HYPRE_BlockTridiagSolve(NALU_HYPRE_Solver solver, NALU_HYPRE_ParCSRMatrix A,
                                  NALU_HYPRE_ParVector b,   NALU_HYPRE_ParVector x)
{
   return (hypre_BlockTridiagSolve((void *) solver, (hypre_ParCSRMatrix *) A,
                                   (hypre_ParVector *) b, (hypre_ParVector *) x));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BlockTridiagSetIndexSet
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int NALU_HYPRE_BlockTridiagSetIndexSet(NALU_HYPRE_Solver solver, NALU_HYPRE_Int n, NALU_HYPRE_Int *inds)
{
   return (hypre_BlockTridiagSetIndexSet((void *) solver, n, inds));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BlockTridiagSetAMGStrengthThreshold
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int NALU_HYPRE_BlockTridiagSetAMGStrengthThreshold(NALU_HYPRE_Solver solver, NALU_HYPRE_Real thresh)
{
   return (hypre_BlockTridiagSetAMGStrengthThreshold((void *) solver, thresh));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BlockTridiagSetAMGNumSweeps
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int NALU_HYPRE_BlockTridiagSetAMGNumSweeps(NALU_HYPRE_Solver solver, NALU_HYPRE_Int num_sweeps)
{
   return (hypre_BlockTridiagSetAMGNumSweeps((void *) solver, num_sweeps));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BlockTridiagSetAMGRelaxType
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int NALU_HYPRE_BlockTridiagSetAMGRelaxType(NALU_HYPRE_Solver solver, NALU_HYPRE_Int relax_type)
{
   return (hypre_BlockTridiagSetAMGRelaxType( (void *) solver, relax_type));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BlockTridiagSetPrintLevel
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int NALU_HYPRE_BlockTridiagSetPrintLevel(NALU_HYPRE_Solver solver, NALU_HYPRE_Int print_level)
{
   return (hypre_BlockTridiagSetPrintLevel( (void *) solver, print_level));
}

