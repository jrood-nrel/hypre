/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#ifndef nalu_hypre_FEI__
#define nalu_hypre_FEI__

#include "NALU_HYPRE.h"
#include "IJ_mv/NALU_HYPRE_IJ_mv.h"
#include "parcsr_mv/NALU_HYPRE_parcsr_mv.h"
#include "parcsr_mv/_nalu_hypre_parcsr_mv.h"
#include "parcsr_ls/_nalu_hypre_parcsr_ls.h"
#include "parcsr_ls/NALU_HYPRE_parcsr_ls.h"

#ifdef __cplusplus
extern "C"
{
#endif

/* bicgs.c */
void * nalu_hypre_BiCGSCreate( );
int nalu_hypre_BiCGSDestroy( void *bicgs_vdata );
int nalu_hypre_BiCGSSetup( void *bicgs_vdata, void *A, void *b, void *x         );
int nalu_hypre_BiCGSSolve(void  *bicgs_vdata, void  *A, void  *b, void  *x);
int nalu_hypre_BiCGSSetTol( void *bicgs_vdata, double tol );
int nalu_hypre_BiCGSSetMaxIter( void *bicgs_vdata, int max_iter );
int nalu_hypre_BiCGSSetStopCrit( void *bicgs_vdata, double stop_crit );
int nalu_hypre_BiCGSSetPrecond( void  *bicgs_vdata, int  (*precond)(void*,void*,void*,void*),
						   int  (*precond_setup)(void*,void*,void*,void*), void  *precond_data );
int nalu_hypre_BiCGSSetLogging( void *bicgs_vdata, int logging);
int nalu_hypre_BiCGSGetNumIterations(void *bicgs_vdata,int  *num_iterations);
int nalu_hypre_BiCGSGetFinalRelativeResidualNorm( void   *bicgs_vdata,
											 double *relative_residual_norm );

/* bicgstabl.c */
	void * nalu_hypre_BiCGSTABLCreate( );
	int nalu_hypre_BiCGSTABLDestroy( void *bicgstab_vdata );
	int nalu_hypre_BiCGSTABLSetup( void *bicgstab_vdata, void *A, void *b, void *x         );
	int nalu_hypre_BiCGSTABLSolve(void  *bicgstab_vdata, void  *A, void  *b, void  *x);
	int nalu_hypre_BiCGSTABLSetSize( void *bicgstab_vdata, int size );
	int nalu_hypre_BiCGSTABLSetMaxIter( void *bicgstab_vdata, int max_iter );
	int nalu_hypre_BiCGSTABLSetStopCrit( void *bicgstab_vdata, double stop_crit );
	int nalu_hypre_BiCGSTABLSetPrecond( void  *bicgstab_vdata, int  (*precond)(void*, void*, void*, void*),
								   int  (*precond_setup)(void*, void*, void*, void*), void  *precond_data );
	int nalu_hypre_BiCGSTABLSetLogging( void *bicgstab_vdata, int logging);
	int nalu_hypre_BiCGSTABLGetNumIterations(void *bicgstab_vdata,int  *num_iterations);
	int nalu_hypre_BiCGSTABLGetFinalRelativeResidualNorm( void   *bicgstab_vdata,
													 double *relative_residual_norm );
/* fgmres.c */
void *nalu_hypre_FGMRESCreate();
int nalu_hypre_FGMRESDestroy( void *fgmres_vdata );
int nalu_hypre_FGMRESSetup( void *fgmres_vdata, void *A, void *b, void *x );
int nalu_hypre_FGMRESSolve(void  *fgmres_vdata, void  *A, void  *b, void  *x);
int nalu_hypre_FGMRESSetKDim( void *fgmres_vdata, int k_dim );
int nalu_hypre_FGMRESSetTol( void *fgmres_vdata, double tol );
int nalu_hypre_FGMRESSetMaxIter( void *fgmres_vdata, int max_iter );
int nalu_hypre_FGMRESSetStopCrit( void *fgmres_vdata, double stop_crit );
int nalu_hypre_FGMRESSetPrecond( void *fgmres_vdata, int (*precond)(void*,void*,void*,void*),
								int  (*precond_setup)(void*,void*,void*,void*), void  *precond_data );
int nalu_hypre_FGMRESGetPrecond(void *fgmres_vdata, NALU_HYPRE_Solver *precond_data_ptr);
int nalu_hypre_FGMRESSetLogging( void *fgmres_vdata, int logging );	
int nalu_hypre_FGMRESGetNumIterations( void *fgmres_vdata, int *num_iterations );
int nalu_hypre_FGMRESGetFinalRelativeResidualNorm(void *fgmres_vdata,
												 double *relative_residual_norm );
int nalu_hypre_FGMRESUpdatePrecondTolerance(void *fgmres_vdata, int (*update_tol)(int*, double));

/* TFQmr.c */
void * nalu_hypre_TFQmrCreate();
int nalu_hypre_TFQmrDestroy( void *tfqmr_vdata );
int nalu_hypre_TFQmrSetup( void *tfqmr_vdata, void *A, void *b, void *x         );
int nalu_hypre_TFQmrSolve(void  *tfqmr_vdata, void  *A, void  *b, void  *x);
int nalu_hypre_TFQmrSetTol( void *tfqmr_vdata, double tol );
int nalu_hypre_TFQmrSetMaxIter( void *tfqmr_vdata, int max_iter );
int nalu_hypre_TFQmrSetStopCrit( void *tfqmr_vdata, double stop_crit );
int nalu_hypre_TFQmrSetPrecond( void  *tfqmr_vdata, int  (*precond)(void*,void*,void*,void*),
						   int  (*precond_setup)(void*,void*,void*,void*), void  *precond_data );
int nalu_hypre_TFQmrSetLogging( void *tfqmr_vdata, int logging);
int nalu_hypre_TFQmrGetNumIterations(void *tfqmr_vdata,int  *num_iterations);
int nalu_hypre_TFQmrGetFinalRelativeResidualNorm( void   *tfqmr_vdata,
											 double *relative_residual_norm );

#ifdef __cplusplus
}
#endif

#endif
