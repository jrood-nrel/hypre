/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * NALU_HYPRE_LSI_DSuperLU interface
 *
 *****************************************************************************/

#ifndef __NALU_HYPRE_DSUPERLU__
#define __NALU_HYPRE_DSUPERLU__

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <math.h>

#include "NALU_HYPRE.h"
#include "utilities/_nalu_hypre_utilities.h"
#include "parcsr_ls/NALU_HYPRE_parcsr_ls.h"
#include "parcsr_mv/_nalu_hypre_parcsr_mv.h"

#ifdef __cplusplus
extern "C"
{
#endif

extern int NALU_HYPRE_LSI_DSuperLUCreate(MPI_Comm comm, NALU_HYPRE_Solver *solver);
extern int NALU_HYPRE_LSI_DSuperLUDestroy(NALU_HYPRE_Solver solver);
extern int NALU_HYPRE_LSI_DSuperLUSetOutputLevel(NALU_HYPRE_Solver solver, int);
extern int NALU_HYPRE_LSI_DSuperLUSetup(NALU_HYPRE_Solver solver,NALU_HYPRE_ParCSRMatrix A,
                                   NALU_HYPRE_ParVector b,NALU_HYPRE_ParVector x);
extern int NALU_HYPRE_LSI_DSuperLUSolve(NALU_HYPRE_Solver solver,NALU_HYPRE_ParCSRMatrix A,
                                   NALU_HYPRE_ParVector b, NALU_HYPRE_ParVector x);

#ifdef __cplusplus
}
#endif

#endif

