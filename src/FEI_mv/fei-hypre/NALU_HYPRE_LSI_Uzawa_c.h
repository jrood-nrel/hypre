/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * NALU_HYPRE_LSI_Uzawa interface
 *
 *****************************************************************************/

#ifndef __NALU_HYPRE_UZAWA__
#define __NALU_HYPRE_UZAWA__

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <math.h>

#include "NALU_HYPRE.h"
#include "utilities/_hypre_utilities.h"
#include "parcsr_ls/_hypre_parcsr_ls.h"
#include "parcsr_mv/_hypre_parcsr_mv.h"
#include "NALU_HYPRE_LSI_UZAWA.h"

#ifdef __cplusplus
extern "C"
{
#endif

extern int NALU_HYPRE_LSI_UzawaCreate(MPI_Comm comm, NALU_HYPRE_Solver *solver);
extern int NALU_HYPRE_LSI_UzawaDestroy(NALU_HYPRE_Solver solver);
extern int NALU_HYPRE_LSI_UzawaSetMaxIterations(NALU_HYPRE_Solver solver, int iter);
extern int NALU_HYPRE_LSI_UzawaSetTolerance(NALU_HYPRE_Solver solver, double tol);
extern int NALU_HYPRE_LSI_UzawaSetParams(NALU_HYPRE_Solver solver, char *params);
extern int NALU_HYPRE_LSI_UzawaGetNumIterations(NALU_HYPRE_Solver solver, int *iter);
extern int NALU_HYPRE_LSI_UzawaSetup(NALU_HYPRE_Solver solver,NALU_HYPRE_ParCSRMatrix A,
                                NALU_HYPRE_ParVector b,NALU_HYPRE_ParVector x);
extern int NALU_HYPRE_LSI_UzawaSolve(NALU_HYPRE_Solver solver,NALU_HYPRE_ParCSRMatrix A,
                                NALU_HYPRE_ParVector b, NALU_HYPRE_ParVector x);

#ifdef __cplusplus
}
#endif

#endif

