/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * NALU_HYPRE_LSI_BlockP interface
 *
 *****************************************************************************/

#ifndef __NALU_HYPRE_BLOCKP__
#define __NALU_HYPRE_BLOCKP__

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <math.h>

#include "utilities/_nalu_hypre_utilities.h"
#include "parcsr_ls/NALU_HYPRE_parcsr_ls.h"
#include "parcsr_mv/_nalu_hypre_parcsr_mv.h"
#include "NALU_HYPRE_LSI_blkprec.h"

#ifdef __cplusplus
extern "C"
{
#endif

extern int NALU_HYPRE_LSI_BlockPrecondCreate(MPI_Comm comm, NALU_HYPRE_Solver *solver);
extern int NALU_HYPRE_LSI_BlockPrecondDestroy(NALU_HYPRE_Solver solver);
extern int NALU_HYPRE_LSI_BlockPrecondSetLumpedMasses(NALU_HYPRE_Solver solver,
                                                 int,double *);
extern int NALU_HYPRE_LSI_BlockPrecondSetParams(NALU_HYPRE_Solver solver, char *params);
extern int NALU_HYPRE_LSI_BlockPrecondSetLookup(NALU_HYPRE_Solver solver, NALU_HYPRE_Lookup *);
extern int NALU_HYPRE_LSI_BlockPrecondSetup(NALU_HYPRE_Solver solver,NALU_HYPRE_ParCSRMatrix A,
                                       NALU_HYPRE_ParVector b,NALU_HYPRE_ParVector x);
extern int NALU_HYPRE_LSI_BlockPrecondSolve(NALU_HYPRE_Solver solver,NALU_HYPRE_ParCSRMatrix A,
                                       NALU_HYPRE_ParVector b, NALU_HYPRE_ParVector x);
extern int NALU_HYPRE_LSI_BlockPrecondSetA11Tolerance(NALU_HYPRE_Solver solver, double);

#ifdef __cplusplus
}
#endif

#endif

