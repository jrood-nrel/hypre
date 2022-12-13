/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

#include "_hypre_parcsr_ls.h"
#include "_hypre_parcsr_mv.h"
#include "mli_solver_amg.h"

/******************************************************************************
 * constructor
 *---------------------------------------------------------------------------*/

MLI_Solver_AMG::MLI_Solver_AMG(char *name) : MLI_Solver(name)
{
   Amat_ = NULL;
   precond_ = NULL;
}

/******************************************************************************
 * destructor
 *---------------------------------------------------------------------------*/

MLI_Solver_AMG::~MLI_Solver_AMG()
{
   Amat_ = NULL;
   if (precond_ != NULL) NALU_HYPRE_BoomerAMGDestroy(precond_);
   precond_ = NULL;
}

/******************************************************************************
 * set up the solver
 *---------------------------------------------------------------------------*/

int MLI_Solver_AMG::setup(MLI_Matrix *mat)
{
   int                i, *nSweeps, *rTypes;
   double             *relaxWt, *relaxOmega;
   hypre_ParCSRMatrix *hypreA;

   Amat_  = mat;
   hypreA = (hypre_ParCSRMatrix *) Amat_->getMatrix();
   NALU_HYPRE_BoomerAMGCreate(&precond_);
   NALU_HYPRE_BoomerAMGSetMaxIter(precond_, 1);
   NALU_HYPRE_BoomerAMGSetCycleType(precond_, 1);
   NALU_HYPRE_BoomerAMGSetMaxLevels(precond_, 25);
   NALU_HYPRE_BoomerAMGSetMeasureType(precond_, 0);
   NALU_HYPRE_BoomerAMGSetDebugFlag(precond_, 0);
   NALU_HYPRE_BoomerAMGSetPrintLevel(precond_, 1);
   NALU_HYPRE_BoomerAMGSetCoarsenType(precond_, 0);
   NALU_HYPRE_BoomerAMGSetStrongThreshold(precond_, 0.8);
   nSweeps = hypre_TAlloc(int, 4 , NALU_HYPRE_MEMORY_HOST);
   for (i = 0; i < 4; i++) nSweeps[i] = 1;
   NALU_HYPRE_BoomerAMGSetNumGridSweeps(precond_, nSweeps);
   rTypes = hypre_TAlloc(int, 4 , NALU_HYPRE_MEMORY_HOST);
   for (i = 0; i < 4; i++) rTypes[i] = 6;
   relaxWt = hypre_TAlloc(double, 25 , NALU_HYPRE_MEMORY_HOST);
   for (i = 0; i < 25; i++) relaxWt[i] = 1.0;
   NALU_HYPRE_BoomerAMGSetRelaxWeight(precond_, relaxWt);
   relaxOmega = hypre_TAlloc(double, 25 , NALU_HYPRE_MEMORY_HOST);
   for (i = 0; i < 25; i++) relaxOmega[i] = 1.0;
   NALU_HYPRE_BoomerAMGSetOmega(precond_, relaxOmega);
   NALU_HYPRE_BoomerAMGSetup(precond_, (NALU_HYPRE_ParCSRMatrix) hypreA, 
         (NALU_HYPRE_ParVector) NULL, (NALU_HYPRE_ParVector) NULL);
   return 0;
}

/******************************************************************************
 * apply function
 *---------------------------------------------------------------------------*/

int MLI_Solver_AMG::solve(MLI_Vector *fIn, MLI_Vector *uIn)
{
   if (precond_ == NULL || Amat_ == NULL)
   {
      printf("MLI_Solver_AMG::solve ERROR - setup not called\n");
      exit(1);
   }
   NALU_HYPRE_ParCSRMatrix hypreA = (NALU_HYPRE_ParCSRMatrix) Amat_->getMatrix();
   NALU_HYPRE_ParVector f = (NALU_HYPRE_ParVector) fIn->getVector();
   NALU_HYPRE_ParVector u = (NALU_HYPRE_ParVector) uIn->getVector();
   NALU_HYPRE_BoomerAMGSolve(precond_, hypreA, f, u);
   return 0;
}

