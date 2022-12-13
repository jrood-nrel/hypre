/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * NALU_HYPRE_LSI_MLI interface
 *
 *****************************************************************************/

#ifndef __NALU_HYPRE_LSI_MLI__
#define __NALU_HYPRE_LSI_MLI__

/******************************************************************************
 * system includes
 *---------------------------------------------------------------------------*/

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

/******************************************************************************
 * HYPRE internal libraries
 *---------------------------------------------------------------------------*/

#include "utilities/_hypre_utilities.h"
#include "HYPRE.h"
#include "parcsr_ls/NALU_HYPRE_parcsr_ls.h"
#include "parcsr_mv/_hypre_parcsr_mv.h"
#include "NALU_HYPRE_FEI_includes.h"

/******************************************************************************
 * Functions to access this data structure
 *---------------------------------------------------------------------------*/

#ifdef __cplusplus
extern "C"
{
#endif

extern int  NALU_HYPRE_LSI_MLICreate( MPI_Comm, NALU_HYPRE_Solver * );
extern int  NALU_HYPRE_LSI_MLIDestroy( NALU_HYPRE_Solver );
extern int  NALU_HYPRE_LSI_MLISetup( NALU_HYPRE_Solver, NALU_HYPRE_ParCSRMatrix,
                                NALU_HYPRE_ParVector,   NALU_HYPRE_ParVector );
extern int  NALU_HYPRE_LSI_MLISolve( NALU_HYPRE_Solver, NALU_HYPRE_ParCSRMatrix,
                                NALU_HYPRE_ParVector,   NALU_HYPRE_ParVector);
extern int  NALU_HYPRE_LSI_MLISetParams( NALU_HYPRE_Solver, char * );
extern int  NALU_HYPRE_LSI_MLICreateNodeEqnMap( NALU_HYPRE_Solver, int, int *, int *,
                                           int *procNRows );
extern int  NALU_HYPRE_LSI_MLIAdjustNodeEqnMap( NALU_HYPRE_Solver, int *, int * );
extern int  NALU_HYPRE_LSI_MLIGetNullSpace( NALU_HYPRE_Solver, int *, int *, double ** );
extern int  NALU_HYPRE_LSI_MLIAdjustNullSpace( NALU_HYPRE_Solver, int, int *,
                                          NALU_HYPRE_ParCSRMatrix );
extern int  NALU_HYPRE_LSI_MLISetFEData( NALU_HYPRE_Solver, void * );
extern int  NALU_HYPRE_LSI_MLISetSFEI( NALU_HYPRE_Solver, void * );

extern int  NALU_HYPRE_LSI_MLILoadNodalCoordinates( NALU_HYPRE_Solver, int, int, int *, 
                                   int, double *, int );
extern int  NALU_HYPRE_LSI_MLILoadMatrixScalings( NALU_HYPRE_Solver, int, double * );
extern int  NALU_HYPRE_LSI_MLILoadMaterialLabels( NALU_HYPRE_Solver, int, int * );

extern void *NALU_HYPRE_LSI_MLIFEDataCreate( MPI_Comm );
extern int  NALU_HYPRE_LSI_MLIFEDataDestroy( void * );
extern int  NALU_HYPRE_LSI_MLIFEDataInitFields( void *, int, int *, int * );
extern int  NALU_HYPRE_LSI_MLIFEDataInitElemBlock(void *, int, int, int, int *);
extern int  NALU_HYPRE_LSI_MLIFEDataInitElemNodeList(void *, int, int, int*);
extern int  NALU_HYPRE_LSI_MLIFEDataInitSharedNodes(void *, int, int *, int*, int **);
extern int  NALU_HYPRE_LSI_MLIFEDataInitComplete( void * );
extern int  NALU_HYPRE_LSI_MLIFEDataLoadElemMatrix(void *, int, int, int *, int,
                                              double **);
extern int  NALU_HYPRE_LSI_MLIFEDataWriteToFile( void *, char * );

extern void *NALU_HYPRE_LSI_MLISFEICreate( MPI_Comm );
extern int  NALU_HYPRE_LSI_MLISFEIDestroy( void * );
extern int  NALU_HYPRE_LSI_MLISFEILoadElemMatrices(void *, int, int, int *,
                                      double ***, int, int **);
extern int  NALU_HYPRE_LSI_MLISFEIAddNumElems(void *, int, int, int);

#ifdef __cplusplus
}
#endif

#endif

