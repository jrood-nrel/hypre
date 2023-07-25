/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Utility functions 
 *
 *****************************************************************************/

#ifndef __MLIUTILS__
#define __MLIUTILS__

#include <time.h>
#include "_nalu_hypre_utilities.h"
#include "_nalu_hypre_parcsr_mv.h"
#include "krylov.h"
#include "_nalu_hypre_parcsr_ls.h"
typedef struct MLI_Function_Struct MLI_Function;
#include "cmli.h"

/************************************************************************
 * structure for m-Jacobi preconditioner
 *----------------------------------------------------------------------*/

typedef struct 
{
   MPI_Comm comm_;
   int      degree_;
   double   *diagonal_;
   NALU_HYPRE_ParVector hypreRes_;
}
NALU_HYPRE_MLI_mJacobi;

/************************************************************************
 * place holder for function pointers
 *----------------------------------------------------------------------*/

struct MLI_Function_Struct
{
   int (*func_)(void*);
};

/************************************************************************
 * Utility function definitions
 *----------------------------------------------------------------------*/

#ifdef __cplusplus
extern "C"
{
#endif

int    MLI_Utils_HypreParCSRMatrixGetDestroyFunc(MLI_Function *funcPtr);
int    MLI_Utils_HypreCSRMatrixGetDestroyFunc(MLI_Function *funcPtr);
int    MLI_Utils_HypreParVectorGetDestroyFunc(MLI_Function *funcPtr);
int    MLI_Utils_HypreVectorGetDestroyFunc(MLI_Function *funcPtr);
int    MLI_Utils_HypreMatrixFormJacobi(void *A, double, void **J);
int    MLI_Utils_GenPartition(MPI_Comm comm, int n, int **part);
int    MLI_Utils_ScaleVec(nalu_hypre_ParCSRMatrix *Amat, nalu_hypre_ParVector *vec);
int    MLI_Utils_ComputeSpectralRadius(nalu_hypre_ParCSRMatrix *, double *);
int    MLI_Utils_ComputeExtremeRitzValues(nalu_hypre_ParCSRMatrix *, double *, int);
int    MLI_Utils_ComputeMatrixMaxNorm(nalu_hypre_ParCSRMatrix *, double *, int);
double MLI_Utils_WTime();
int    MLI_Utils_HypreMatrixPrint(void *, char *);
int    MLI_Utils_HypreMatrixGetInfo(void *, int *, double *);
int    MLI_Utils_HypreMatrixComputeRAP(void *P, void *A, void **RAP);
int    MLI_Utils_HypreMatrixCompress(void *A, int blksize, void **A2);
int    MLI_Utils_HypreBoolMatrixDecompress(void *S, int, void **S2, void *A);
int    MLI_Utils_QR(double *Q, double *R, int nrows, int ncols);
int    MLI_Utils_SVD(double *uArray, double *sArray, double *vtArray,
                 double *workArray, int m, int n, int workLen);
int    MLI_Utils_singular_vectors(int n, double *uArray);
int    MLI_Utils_ComputeLowEnergyLanczos(nalu_hypre_ParCSRMatrix *A, 
                 int maxIter, int num_vecs_to_return, double *le_vectors);
int    MLI_Utils_HypreMatrixReadTuminFormat(char *filename, MPI_Comm comm, 
                 int blksize, void **mat, int flag, double **scaleVec);
int    MLI_Utils_HypreMatrixReadIJAFormat(char *filename, MPI_Comm comm, 
                 int blksize, void **mat, int flag, double **scaleVec);
int    MLI_Utils_HypreParMatrixReadIJAFormat(char *filename, MPI_Comm comm, 
                 void **mat, int flag, double **scaleVec);
int    MLI_Utils_HypreMatrixReadHBFormat(char *filename, MPI_Comm comm, 
                 void **mat);
int    MLI_Utils_DoubleVectorRead(char *, MPI_Comm, int, int, double *vec);
int    MLI_Utils_DoubleParVectorRead(char *, MPI_Comm, int, int, double *vec);
int    MLI_Utils_ParCSRMLISetup(NALU_HYPRE_Solver, NALU_HYPRE_ParCSRMatrix, 
                                NALU_HYPRE_ParVector, NALU_HYPRE_ParVector);
int    MLI_Utils_ParCSRMLISolve(NALU_HYPRE_Solver, NALU_HYPRE_ParCSRMatrix, 
                                NALU_HYPRE_ParVector, NALU_HYPRE_ParVector);
int    MLI_Utils_mJacobiCreate(MPI_Comm, NALU_HYPRE_Solver *);
int    MLI_Utils_mJacobiDestroy(NALU_HYPRE_Solver);
int    MLI_Utils_mJacobiSetParams(NALU_HYPRE_Solver, int);
int    MLI_Utils_mJacobiSetup(NALU_HYPRE_Solver, NALU_HYPRE_ParCSRMatrix, 
                              NALU_HYPRE_ParVector, NALU_HYPRE_ParVector);
int    MLI_Utils_mJacobiSolve(NALU_HYPRE_Solver, NALU_HYPRE_ParCSRMatrix, 
                              NALU_HYPRE_ParVector, NALU_HYPRE_ParVector);
int    MLI_Utils_HyprePCGSolve(CMLI *, NALU_HYPRE_Matrix, NALU_HYPRE_Vector, 
                               NALU_HYPRE_Vector);
int    MLI_Utils_HypreGMRESSolve(void *, NALU_HYPRE_Matrix, NALU_HYPRE_Vector, 
                                 NALU_HYPRE_Vector, char *);
int    MLI_Utils_HypreBiCGSTABSolve(CMLI *, NALU_HYPRE_Matrix, NALU_HYPRE_Vector, 
                                    NALU_HYPRE_Vector);
int    MLI_Utils_BinarySearch(int, int *, int);
int    MLI_Utils_IntQSort2(int *, int *, int, int);
int    MLI_Utils_IntQSort2a(int *, double *, int, int);
int    MLI_Utils_DbleQSort2a(double *, int *, int, int);
int    MLI_Utils_IntMergeSort(int nlist, int *listLengs, int **lists,
                              int **list2, int *newNList, int **newList); 
int    MLI_Utils_DenseMatrixInverse(double **Amat, int ndim, double ***Bmat);
int    MLI_Utils_DenseMatvec(double **Amat, int ndim, double *x, double *Ax);

#ifdef __cplusplus
}
#endif

#endif

