/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_parcsr_ls.h"

/*--------------------------------------------------------------------------
 * NALU_HYPRE_AMECreate
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int NALU_HYPRE_AMECreate(NALU_HYPRE_Solver *esolver)
{
   *esolver = (NALU_HYPRE_Solver) hypre_AMECreate();
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_AMEDestroy
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int NALU_HYPRE_AMEDestroy(NALU_HYPRE_Solver esolver)
{
   return hypre_AMEDestroy((void *) esolver);
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_AMESetup
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int NALU_HYPRE_AMESetup (NALU_HYPRE_Solver esolver)
{
   return hypre_AMESetup((void *) esolver);
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_AMESolve
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int NALU_HYPRE_AMESolve (NALU_HYPRE_Solver esolver)
{
   return hypre_AMESolve((void *) esolver);
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_AMESetAMSSolver
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int NALU_HYPRE_AMESetAMSSolver(NALU_HYPRE_Solver esolver,
                                NALU_HYPRE_Solver ams_solver)
{
   return hypre_AMESetAMSSolver((void *) esolver,
                                (void *) ams_solver);
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_AMESetMassMatrix
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int NALU_HYPRE_AMESetMassMatrix(NALU_HYPRE_Solver esolver,
                                 NALU_HYPRE_ParCSRMatrix M)
{
   return hypre_AMESetMassMatrix((void *) esolver,
                                 (hypre_ParCSRMatrix *) M);
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_AMESetBlockSize
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int NALU_HYPRE_AMESetBlockSize(NALU_HYPRE_Solver esolver,
                                NALU_HYPRE_Int block_size)
{
   return hypre_AMESetBlockSize((void *) esolver, block_size);
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_AMESetMaxIter
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int NALU_HYPRE_AMESetMaxIter(NALU_HYPRE_Solver esolver,
                              NALU_HYPRE_Int maxit)
{
   return hypre_AMESetMaxIter((void *) esolver, maxit);
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_AMESetTol
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int NALU_HYPRE_AMESetTol(NALU_HYPRE_Solver esolver,
                          NALU_HYPRE_Real tol)
{
   return hypre_AMESetTol((void *) esolver, tol);
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_AMESetRTol
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int NALU_HYPRE_AMESetRTol(NALU_HYPRE_Solver esolver,
                           NALU_HYPRE_Real tol)
{
   return hypre_AMESetRTol((void *) esolver, tol);
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_AMESetPrintLevel
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int NALU_HYPRE_AMESetPrintLevel(NALU_HYPRE_Solver esolver,
                                 NALU_HYPRE_Int print_level)
{
   return hypre_AMESetPrintLevel((void *) esolver, print_level);
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_AMEGetEigenvalues
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int NALU_HYPRE_AMEGetEigenvalues(NALU_HYPRE_Solver esolver,
                                  NALU_HYPRE_Real **eigenvalues)
{
   return hypre_AMEGetEigenvalues((void *) esolver, eigenvalues);
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_AMEGetEigenvectors
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int NALU_HYPRE_AMEGetEigenvectors(NALU_HYPRE_Solver esolver,
                                   NALU_HYPRE_ParVector **eigenvectors)
{
   return hypre_AMEGetEigenvectors((void *) esolver,
                                   eigenvectors);
}
