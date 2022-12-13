/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * NALU_HYPRE_ParCSR_SuperLU interface
 *
 *****************************************************************************/

#ifndef __NALU_HYPRE_PARCSR_SUPERLU__
#define __NALU_HYPRE_PARCSR_SUPERLU__

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <math.h>

#include "HYPRE.h"
#include "utilities/_hypre_utilities.h"
#include "parcsr_mv/_hypre_parcsr_mv.h"

#ifdef __cplusplus
extern "C"
{
#endif

extern int NALU_HYPRE_ParCSR_SuperLUCreate(MPI_Comm comm, NALU_HYPRE_Solver *solver);
extern int NALU_HYPRE_ParCSR_SuperLUDestroy(NALU_HYPRE_Solver solver);
extern int NALU_HYPRE_ParCSR_SuperLUSetOutputLevel(NALU_HYPRE_Solver solver, int);
extern int NALU_HYPRE_ParCSR_SuperLUSetup(NALU_HYPRE_Solver solver,NALU_HYPRE_ParCSRMatrix A,
                        NALU_HYPRE_ParVector b,NALU_HYPRE_ParVector x);
extern int NALU_HYPRE_ParCSR_SuperLUSolve(NALU_HYPRE_Solver solver,NALU_HYPRE_ParCSRMatrix A,
                        NALU_HYPRE_ParVector b, NALU_HYPRE_ParVector x);

#ifdef __cplusplus
}
#endif

#endif

