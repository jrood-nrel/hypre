/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#ifndef __NALU_HYPRE_FEI__
#define __NALU_HYPRE_FEI__

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

#include "NALU_HYPRE_MHMatrix.h"

typedef struct NALU_HYPRE_LSI_DDIlut_Struct
{
   MPI_Comm  comm;
   MH_Matrix *mh_mat;
   double    thresh;
   double    fillin;
   int       overlap;
   int       Nrows;
   int       extNrows;
   int       *mat_ia;
   int       *mat_ja;
   double    *mat_aa;
   int       outputLevel;
   int       reorder;
   int       *order_array;
   int       *reorder_array;
}
NALU_HYPRE_LSI_DDIlut;

/* NALU_HYPRE_LSI_ddict.c */
int NALU_HYPRE_LSI_DDICTCreate( MPI_Comm comm, NALU_HYPRE_Solver *solver );
int NALU_HYPRE_LSI_DDICTDestroy( NALU_HYPRE_Solver solver );
int NALU_HYPRE_LSI_DDICTSetFillin( NALU_HYPRE_Solver solver, double fillin);
int NALU_HYPRE_LSI_DDICTSetOutputLevel( NALU_HYPRE_Solver solver, int level);
int NALU_HYPRE_LSI_DDICTSetDropTolerance( NALU_HYPRE_Solver solver, double thresh);
int NALU_HYPRE_LSI_DDICTSolve( NALU_HYPRE_Solver solver, NALU_HYPRE_ParCSRMatrix A,
						  NALU_HYPRE_ParVector b,   NALU_HYPRE_ParVector x );
int NALU_HYPRE_LSI_DDICTSetup( NALU_HYPRE_Solver solver, NALU_HYPRE_ParCSRMatrix A,
						  NALU_HYPRE_ParVector b,   NALU_HYPRE_ParVector x );
/* NALU_HYPRE_LSI_ddilut.c */
int NALU_HYPRE_LSI_DDIlutCreate( MPI_Comm comm, NALU_HYPRE_Solver *solver );
int NALU_HYPRE_LSI_DDIlutDestroy( NALU_HYPRE_Solver solver );
int NALU_HYPRE_LSI_DDIlutSetFillin( NALU_HYPRE_Solver solver, double fillin);
int NALU_HYPRE_LSI_DDIlutSetDropTolerance( NALU_HYPRE_Solver solver, double thresh);
int NALU_HYPRE_LSI_DDIlutSetOverlap( NALU_HYPRE_Solver solver );
int NALU_HYPRE_LSI_DDIlutSetReorder( NALU_HYPRE_Solver solver );
int NALU_HYPRE_LSI_DDIlutSetOutputLevel( NALU_HYPRE_Solver solver, int level);
int NALU_HYPRE_LSI_DDIlutSolve( NALU_HYPRE_Solver solver, NALU_HYPRE_ParCSRMatrix A,
						   NALU_HYPRE_ParVector b,   NALU_HYPRE_ParVector x );
int NALU_HYPRE_LSI_DDIlutSetup( NALU_HYPRE_Solver solver, NALU_HYPRE_ParCSRMatrix A,
						   NALU_HYPRE_ParVector b,   NALU_HYPRE_ParVector x );
int NALU_HYPRE_LSI_DDIlutGetRowLengths(MH_Matrix *Amat, int *leng, int **recv_leng,
                                  MPI_Comm mpi_comm);
int NALU_HYPRE_LSI_DDIlutGetOffProcRows(MH_Matrix *Amat, int leng, int *recv_leng,
                           int Noffset, int *map, int *map2, int **int_buf,
								   double **dble_buf, MPI_Comm mpi_comm);
int NALU_HYPRE_LSI_DDIlutComposeOverlappedMatrix(MH_Matrix *mh_mat, 
											int *total_recv_leng, int **recv_lengths, int **int_buf, 
											double **dble_buf, int **sindex_array, int **sindex_array2, 
											int *offset, MPI_Comm mpi_comm);
int NALU_HYPRE_LSI_DDIlutDecompose(NALU_HYPRE_LSI_DDIlut *ilut_ptr,MH_Matrix *Amat,
							  int total_recv_leng, int *recv_lengths, int *ext_ja, double *ext_aa, 
							  int *map, int *map2, int Noffset);
int NALU_HYPRE_LSI_DDIlutDecompose2(NALU_HYPRE_LSI_DDIlut *ilut_ptr,MH_Matrix *Amat,
							   int total_recv_leng, int *recv_lengths, int *ext_ja, double *ext_aa, 
							   int *map, int *map2, int Noffset);
int NALU_HYPRE_LSI_DDIlutDecomposeNew(NALU_HYPRE_LSI_DDIlut *ilut_ptr,MH_Matrix *Amat,
								 int total_recv_leng, int *recv_lengths, int *ext_ja, double *ext_aa, 
								 int *map, int *map2, int Noffset);
	
/* nalu_hypre_lsi_misc.c */
void NALU_HYPRE_LSI_Get_IJAMatrixFromFile(double **val, int **ia, 
									 int **ja, int *N, double **rhs, char *matfile, char *rhsfile);
int NALU_HYPRE_LSI_Search(int *list,int value,int list_length);
int NALU_HYPRE_LSI_Search2(int key, int nlist, int *list);
int NALU_HYPRE_LSI_GetParCSRMatrix(NALU_HYPRE_IJMatrix Amat, int nrows, int nnz, 
                              int *ia_ptr, int *ja_ptr, double *a_ptr) ;
void NALU_HYPRE_LSI_qsort1a( int *ilist, int *ilist2, int left, int right);
int NALU_HYPRE_LSI_SplitDSort2(double *dlist, int nlist, int *ilist, int limit);
int NALU_HYPRE_LSI_SplitDSort(double *dlist, int nlist, int *ilist, int limit);
int NALU_HYPRE_LSI_SolveIdentity(NALU_HYPRE_Solver solver, NALU_HYPRE_ParCSRMatrix Amat,
                            NALU_HYPRE_ParVector b, NALU_HYPRE_ParVector x);
int NALU_HYPRE_LSI_Cuthill(int n, int *ia, int *ja, double *aa, int *order_array,
                      int *reorder_array);
int NALU_HYPRE_LSI_MatrixInverse( double **Amat, int ndim, double ***Cmat );
int NALU_HYPRE_LSI_PartitionMatrix( int nRows, int startRow, int *rowLengths,
                               int **colIndices, double **colValues,
                               int *nLabels, int **labels);


/* NALU_HYPRE_LSI_poly.c */
int NALU_HYPRE_LSI_PolyCreate( MPI_Comm comm, NALU_HYPRE_Solver *solver );
int NALU_HYPRE_LSI_PolyDestroy( NALU_HYPRE_Solver solver );
int NALU_HYPRE_LSI_PolySetOrder( NALU_HYPRE_Solver solver, int order);
int NALU_HYPRE_LSI_PolySetOutputLevel( NALU_HYPRE_Solver solver, int level);
int NALU_HYPRE_LSI_PolySolve( NALU_HYPRE_Solver solver, NALU_HYPRE_ParCSRMatrix A,
                                NALU_HYPRE_ParVector b,   NALU_HYPRE_ParVector x );
int NALU_HYPRE_LSI_PolySetup( NALU_HYPRE_Solver solver, NALU_HYPRE_ParCSRMatrix A,
                                NALU_HYPRE_ParVector b,   NALU_HYPRE_ParVector x );

/* NALU_HYPRE_LSI_schwarz.c */
int NALU_HYPRE_LSI_SchwarzCreate( MPI_Comm comm, NALU_HYPRE_Solver *solver );
int NALU_HYPRE_LSI_SchwarzDestroy( NALU_HYPRE_Solver solver );
int NALU_HYPRE_LSI_SchwarzSetBlockSize( NALU_HYPRE_Solver solver, int blksize);
int NALU_HYPRE_LSI_SchwarzSetNBlocks( NALU_HYPRE_Solver solver, int nblks);
int NALU_HYPRE_LSI_SchwarzSetILUTFillin( NALU_HYPRE_Solver solver, double fillin);
int NALU_HYPRE_LSI_SchwarzSetOutputLevel( NALU_HYPRE_Solver solver, int level);
int NALU_HYPRE_LSI_SchwarzSolve( NALU_HYPRE_Solver solver, NALU_HYPRE_ParCSRMatrix A,
							NALU_HYPRE_ParVector b,   NALU_HYPRE_ParVector x );
int NALU_HYPRE_LSI_SchwarzSetup( NALU_HYPRE_Solver solver, NALU_HYPRE_ParCSRMatrix A,
							NALU_HYPRE_ParVector b,   NALU_HYPRE_ParVector x );

/* NALU_HYPRE_parcsr_bicgs.c */
int NALU_HYPRE_ParCSRBiCGSCreate( MPI_Comm comm, NALU_HYPRE_Solver *solver );
int NALU_HYPRE_ParCSRBiCGSDestroy( NALU_HYPRE_Solver solver );
int NALU_HYPRE_ParCSRBiCGSSetup(NALU_HYPRE_Solver solver,NALU_HYPRE_ParCSRMatrix A,
						   NALU_HYPRE_ParVector b, NALU_HYPRE_ParVector x );
int NALU_HYPRE_ParCSRBiCGSSolve(NALU_HYPRE_Solver solver,NALU_HYPRE_ParCSRMatrix A,
						   NALU_HYPRE_ParVector b, NALU_HYPRE_ParVector x );
int NALU_HYPRE_ParCSRBiCGSSetTol( NALU_HYPRE_Solver solver, double tol );
int NALU_HYPRE_ParCSRBiCGSSetMaxIter( NALU_HYPRE_Solver solver, int max_iter );
int NALU_HYPRE_ParCSRBiCGSSetStopCrit( NALU_HYPRE_Solver solver, int stop_crit );
int NALU_HYPRE_ParCSRBiCGSSetPrecond( NALU_HYPRE_Solver  solver,
								 int (*precond)      (NALU_HYPRE_Solver sol, NALU_HYPRE_ParCSRMatrix matrix,
													  NALU_HYPRE_ParVector b, NALU_HYPRE_ParVector x),
								 int (*precond_setup)(NALU_HYPRE_Solver sol, NALU_HYPRE_ParCSRMatrix matrix,
													  NALU_HYPRE_ParVector b, NALU_HYPRE_ParVector x),
								 void                *precond_data );
int NALU_HYPRE_ParCSRBiCGSSetLogging( NALU_HYPRE_Solver solver, int logging);
int NALU_HYPRE_ParCSRBiCGSGetNumIterations(NALU_HYPRE_Solver solver,
									  int *num_iterations);
int NALU_HYPRE_ParCSRBiCGSGetFinalRelativeResidualNorm(NALU_HYPRE_Solver solver,
												  double *norm );

/* NALU_HYPRE_parcsr_bicgs.c */
int NALU_HYPRE_ParCSRBiCGSTABLCreate( MPI_Comm comm, NALU_HYPRE_Solver *solver );
int NALU_HYPRE_ParCSRBiCGSTABLDestroy( NALU_HYPRE_Solver solver );
int NALU_HYPRE_ParCSRBiCGSTABLSetup(NALU_HYPRE_Solver solver,NALU_HYPRE_ParCSRMatrix A,
                               NALU_HYPRE_ParVector b, NALU_HYPRE_ParVector x );
int NALU_HYPRE_ParCSRBiCGSTABLSolve(NALU_HYPRE_Solver solver,NALU_HYPRE_ParCSRMatrix A,
                                NALU_HYPRE_ParVector b, NALU_HYPRE_ParVector x );
int NALU_HYPRE_ParCSRBiCGSTABLSetTol( NALU_HYPRE_Solver solver, double tol );
int NALU_HYPRE_ParCSRBiCGSTABLSetSize( NALU_HYPRE_Solver solver, int size );
int NALU_HYPRE_ParCSRBiCGSTABLSetMaxIter( NALU_HYPRE_Solver solver, int max_iter );
int NALU_HYPRE_ParCSRBiCGSTABLSetStopCrit( NALU_HYPRE_Solver solver, int stop_crit );
int NALU_HYPRE_ParCSRBiCGSTABLSetPrecond( NALU_HYPRE_Solver  solver,
          int (*precond)      (NALU_HYPRE_Solver sol, NALU_HYPRE_ParCSRMatrix matrix,
			       NALU_HYPRE_ParVector b, NALU_HYPRE_ParVector x),
          int (*precond_setup)(NALU_HYPRE_Solver sol, NALU_HYPRE_ParCSRMatrix matrix,
			       NALU_HYPRE_ParVector b, NALU_HYPRE_ParVector x),
          void               *precond_data );
int NALU_HYPRE_ParCSRBiCGSTABLSetLogging( NALU_HYPRE_Solver solver, int logging);
int NALU_HYPRE_ParCSRBiCGSTABLGetNumIterations(NALU_HYPRE_Solver solver,
                                                 int *num_iterations);
int NALU_HYPRE_ParCSRBiCGSTABLGetFinalRelativeResidualNorm(NALU_HYPRE_Solver solver,
                                                       double *norm );

/* NALU_HYPRE_parcsr_fgmres.h */
int NALU_HYPRE_ParCSRFGMRESCreate( MPI_Comm comm, NALU_HYPRE_Solver *solver );
int NALU_HYPRE_ParCSRFGMRESDestroy( NALU_HYPRE_Solver solver );
int NALU_HYPRE_ParCSRFGMRESSetup(NALU_HYPRE_Solver solver,NALU_HYPRE_ParCSRMatrix A,
                                   NALU_HYPRE_ParVector b, NALU_HYPRE_ParVector x );
int NALU_HYPRE_ParCSRFGMRESSolve(NALU_HYPRE_Solver solver,NALU_HYPRE_ParCSRMatrix A,
                                   NALU_HYPRE_ParVector b, NALU_HYPRE_ParVector x );
int NALU_HYPRE_ParCSRFGMRESSetKDim(NALU_HYPRE_Solver solver, int kdim);
int NALU_HYPRE_ParCSRFGMRESSetTol(NALU_HYPRE_Solver solver, double tol);
int NALU_HYPRE_ParCSRFGMRESSetMaxIter(NALU_HYPRE_Solver solver, int max_iter);
int NALU_HYPRE_ParCSRFGMRESSetStopCrit(NALU_HYPRE_Solver solver, int stop_crit);
int NALU_HYPRE_ParCSRFGMRESSetPrecond(NALU_HYPRE_Solver  solver,
          int (*precond)(NALU_HYPRE_Solver sol, NALU_HYPRE_ParCSRMatrix matrix,
			NALU_HYPRE_ParVector b, NALU_HYPRE_ParVector x),
          int (*precond_setup)(NALU_HYPRE_Solver sol, NALU_HYPRE_ParCSRMatrix matrix,
			       NALU_HYPRE_ParVector b, NALU_HYPRE_ParVector x),
          void *precond_data);
int NALU_HYPRE_ParCSRFGMRESSetLogging(NALU_HYPRE_Solver solver, int logging);
int NALU_HYPRE_ParCSRFGMRESGetNumIterations(NALU_HYPRE_Solver solver,
                                              int *num_iterations);
int NALU_HYPRE_ParCSRFGMRESGetFinalRelativeResidualNorm(NALU_HYPRE_Solver solver,
                                                          double *norm );
int NALU_HYPRE_ParCSRFGMRESUpdatePrecondTolerance(NALU_HYPRE_Solver  solver,
                             int (*set_tolerance)(NALU_HYPRE_Solver sol, double));

/* NALU_HYPRE_parcsr_lsicg.c */
int NALU_HYPRE_ParCSRLSICGCreate(MPI_Comm comm, NALU_HYPRE_Solver *solver);
int NALU_HYPRE_ParCSRLSICGDestroy(NALU_HYPRE_Solver solver);
int NALU_HYPRE_ParCSRLSICGSetup(NALU_HYPRE_Solver solver,NALU_HYPRE_ParCSRMatrix A,
                                  NALU_HYPRE_ParVector b, NALU_HYPRE_ParVector x );
int NALU_HYPRE_ParCSRLSICGSolve(NALU_HYPRE_Solver solver,NALU_HYPRE_ParCSRMatrix A,
                                  NALU_HYPRE_ParVector b, NALU_HYPRE_ParVector x );
int NALU_HYPRE_ParCSRLSICGSetTol(NALU_HYPRE_Solver solver, double tol);
int NALU_HYPRE_ParCSRLSICGSetMaxIter(NALU_HYPRE_Solver solver, int max_iter);
int NALU_HYPRE_ParCSRLSICGSetStopCrit(NALU_HYPRE_Solver solver, int stop_crit);
int NALU_HYPRE_ParCSRLSICGSetPrecond(NALU_HYPRE_Solver  solver,
          int (*precond)      (NALU_HYPRE_Solver sol, NALU_HYPRE_ParCSRMatrix matrix,
			       NALU_HYPRE_ParVector b, NALU_HYPRE_ParVector x),
          int (*precond_setup)(NALU_HYPRE_Solver sol, NALU_HYPRE_ParCSRMatrix matrix,
			       NALU_HYPRE_ParVector b, NALU_HYPRE_ParVector x),
          void *precond_data );
int NALU_HYPRE_ParCSRLSICGSetLogging(NALU_HYPRE_Solver solver, int logging);
int NALU_HYPRE_ParCSRLSICGGetNumIterations(NALU_HYPRE_Solver solver,
                                             int *num_iterations);
int NALU_HYPRE_ParCSRLSICGGetFinalRelativeResidualNorm(NALU_HYPRE_Solver solver,
                                                         double *norm );

/* NALU_HYPRE_parcsr_maxwell.c */
int NALU_HYPRE_ParCSRCotreeCreate(MPI_Comm comm, NALU_HYPRE_Solver *solver);
int NALU_HYPRE_ParCSRCotreeDestroy(NALU_HYPRE_Solver solver);
int NALU_HYPRE_ParCSRCotreeSetup(NALU_HYPRE_Solver solver, NALU_HYPRE_ParCSRMatrix A,
                            NALU_HYPRE_ParVector b, NALU_HYPRE_ParVector x);
int NALU_HYPRE_ParCSRCotreeSolve(NALU_HYPRE_Solver solver, NALU_HYPRE_ParCSRMatrix A,
							NALU_HYPRE_ParVector b, NALU_HYPRE_ParVector x);
int NALU_HYPRE_ParCSRCotreeSetTol(NALU_HYPRE_Solver solver, double tol);
int NALU_HYPRE_ParCSRCotreeSetMaxIter(NALU_HYPRE_Solver solver, int max_iter);	

/* NALU_HYPRE_parcsr_superlu.c */
int NALU_HYPRE_ParCSR_SuperLUCreate(MPI_Comm comm, NALU_HYPRE_Solver *solver);
int NALU_HYPRE_ParCSR_SuperLUDestroy(NALU_HYPRE_Solver solver);
int NALU_HYPRE_ParCSR_SuperLUSetOutputLevel(NALU_HYPRE_Solver solver, int);
int NALU_HYPRE_ParCSR_SuperLUSetup(NALU_HYPRE_Solver solver,NALU_HYPRE_ParCSRMatrix A,
							  NALU_HYPRE_ParVector b,NALU_HYPRE_ParVector x);
int NALU_HYPRE_ParCSR_SuperLUSolve(NALU_HYPRE_Solver solver,NALU_HYPRE_ParCSRMatrix A,
							  NALU_HYPRE_ParVector b, NALU_HYPRE_ParVector x);
	
/* NALU_HYPRE_parcsr_symqmr.c */
int NALU_HYPRE_ParCSRSymQMRCreate( MPI_Comm comm, NALU_HYPRE_Solver *solver );
int NALU_HYPRE_ParCSRSymQMRDestroy( NALU_HYPRE_Solver solver );
int NALU_HYPRE_ParCSRSymQMRSetup(NALU_HYPRE_Solver solver,NALU_HYPRE_ParCSRMatrix A,
							NALU_HYPRE_ParVector b, NALU_HYPRE_ParVector x );
int NALU_HYPRE_ParCSRSymQMRSolve(NALU_HYPRE_Solver solver,NALU_HYPRE_ParCSRMatrix A,
							NALU_HYPRE_ParVector b, NALU_HYPRE_ParVector x );
int NALU_HYPRE_ParCSRSymQMRSetTol( NALU_HYPRE_Solver solver, double tol );
int NALU_HYPRE_ParCSRSymQMRSetMaxIter( NALU_HYPRE_Solver solver, int max_iter );
int NALU_HYPRE_ParCSRSymQMRSetStopCrit( NALU_HYPRE_Solver solver, int stop_crit );
int NALU_HYPRE_ParCSRSymQMRSetPrecond( NALU_HYPRE_Solver  solver,
								  int (*precond)      (NALU_HYPRE_Solver sol, NALU_HYPRE_ParCSRMatrix matrix,
													   NALU_HYPRE_ParVector b, NALU_HYPRE_ParVector x),
								  int (*precond_setup)(NALU_HYPRE_Solver sol, NALU_HYPRE_ParCSRMatrix matrix,
													   NALU_HYPRE_ParVector b, NALU_HYPRE_ParVector x),
								  void                *precond_data );
int NALU_HYPRE_ParCSRSymQMRSetLogging( NALU_HYPRE_Solver solver, int logging);
int NALU_HYPRE_ParCSRSymQMRGetNumIterations(NALU_HYPRE_Solver solver,
									   int *num_iterations);
int NALU_HYPRE_ParCSRSymQMRGetFinalRelativeResidualNorm(NALU_HYPRE_Solver solver,
												   double *norm );

/* NALU_HYPRE_parcsr_TFQmr.c */
int NALU_HYPRE_ParCSRTFQmrCreate( MPI_Comm comm, NALU_HYPRE_Solver *solver );
int NALU_HYPRE_ParCSRTFQmrDestroy( NALU_HYPRE_Solver solver );
int NALU_HYPRE_ParCSRTFQmrSetup(NALU_HYPRE_Solver solver,NALU_HYPRE_ParCSRMatrix A,
                               NALU_HYPRE_ParVector b, NALU_HYPRE_ParVector x );
int NALU_HYPRE_ParCSRTFQmrSolve(NALU_HYPRE_Solver solver,NALU_HYPRE_ParCSRMatrix A,
                                NALU_HYPRE_ParVector b, NALU_HYPRE_ParVector x );
int NALU_HYPRE_ParCSRTFQmrSetTol( NALU_HYPRE_Solver solver, double tol );
int NALU_HYPRE_ParCSRTFQmrSetMaxIter( NALU_HYPRE_Solver solver, int max_iter );
int NALU_HYPRE_ParCSRTFQmrSetStopCrit( NALU_HYPRE_Solver solver, int stop_crit );
int NALU_HYPRE_ParCSRTFQmrSetPrecond( NALU_HYPRE_Solver  solver,
          int (*precond)      (NALU_HYPRE_Solver sol, NALU_HYPRE_ParCSRMatrix matrix,
			       NALU_HYPRE_ParVector b, NALU_HYPRE_ParVector x),
          int (*precond_setup)(NALU_HYPRE_Solver sol, NALU_HYPRE_ParCSRMatrix matrix,
			       NALU_HYPRE_ParVector b, NALU_HYPRE_ParVector x),
          void               *precond_data );
int NALU_HYPRE_ParCSRTFQmrSetLogging( NALU_HYPRE_Solver solver, int logging);
int NALU_HYPRE_ParCSRTFQmrGetNumIterations(NALU_HYPRE_Solver solver,
                                                 int *num_iterations);
int NALU_HYPRE_ParCSRTFQmrGetFinalRelativeResidualNorm(NALU_HYPRE_Solver solver,
                                                       double *norm );

#ifdef __cplusplus
}
#endif

#endif
