/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * NALU_HYPRE_DDILUT interface
 *
 *****************************************************************************/

#ifndef __NALU_HYPRE_DDILUT__
#define __NALU_HYPRE_DDILUT__

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <math.h>

#include "utilities/_nalu_hypre_utilities.h"
#include "parcsr_ls/NALU_HYPRE_parcsr_ls.h"
#include "parcsr_mv/_nalu_hypre_parcsr_mv.h"

#ifdef __cplusplus
extern "C"
{
#endif

extern int NALU_HYPRE_LSI_DDIlutCreate( MPI_Comm comm, NALU_HYPRE_Solver *solver );
extern int NALU_HYPRE_LSI_DDIlutDestroy( NALU_HYPRE_Solver solver );
extern int NALU_HYPRE_LSI_DDIlutSetFillin( NALU_HYPRE_Solver solver, double fillin);
extern int NALU_HYPRE_LSI_DDIlutSetOutputLevel( NALU_HYPRE_Solver solver, int level);
extern int NALU_HYPRE_LSI_DDIlutSetDropTolerance( NALU_HYPRE_Solver solver, double thresh);
extern int NALU_HYPRE_LSI_DDIlutSetOverlap( NALU_HYPRE_Solver solver );
extern int NALU_HYPRE_LSI_DDIlutSetReorder( NALU_HYPRE_Solver solver );
extern int NALU_HYPRE_LSI_DDIlutSolve( NALU_HYPRE_Solver solver, NALU_HYPRE_ParCSRMatrix A,
                                  NALU_HYPRE_ParVector b,   NALU_HYPRE_ParVector x );
extern int NALU_HYPRE_LSI_DDIlutSetup( NALU_HYPRE_Solver solver, NALU_HYPRE_ParCSRMatrix A,
                                  NALU_HYPRE_ParVector b,   NALU_HYPRE_ParVector x );

#ifdef __cplusplus
}
#endif

#endif

