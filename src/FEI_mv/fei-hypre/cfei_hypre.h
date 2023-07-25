/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#ifndef _cfei_nalu_hypre_h_
#define _cfei_nalu_hypre_h_

/*
   This header defines the prototype for the NALU_HYPRE-specific functions that
   uses the LinSysCore struct pointer, which is used by FEI_create.
*/

#ifndef CFEI_LinSysCore_DEFINED
#define CFEI_LinSysCore_DEFINED

/*
  First we define the LinSysCore struct which is kind of like an
  abstract type. ISIS_LinSysCore_create produces an instance of LinSysCore.
*/

struct LinSysCore_struct {
   void* lsc_;
};
typedef struct LinSysCore_struct LinSysCore;

#endif

#ifdef __cplusplus
LinearSystemCore *NALU_HYPRE_base_create( MPI_Comm comm );
#endif

#ifdef __cplusplus
extern "C" {
#endif

int NALU_HYPRE_LinSysCore_create(LinSysCore** lsc, MPI_Comm comm);

int NALU_HYPRE_LinSysCore_destroy(LinSysCore** lsc);

int NALU_HYPRE_LSC_BeginMappedMatrixLoad(LinSysCore* lsc);

int NALU_HYPRE_LSC_EndMappedMatrixLoad(LinSysCore* lsc);

int NALU_HYPRE_LSC_MappedMatrixLoad(LinSysCore* lsc, int row, int col, double val);

char *NALU_HYPRE_LSC_GetVersion(LinSysCore* lsc);

int NALU_HYPRE_LSC_GetFEDataObject(LinSysCore* lsc, void **object);

int NALU_HYPRE_LSC_parameters(LinSysCore* lsc, int numParams, char **params);

int NALU_HYPRE_LSC_setGlobalOffsets(LinSysCore* lsc, int leng, int* nodeOffsets,
                           int* eqnOffsets, int* blkEqnOffsets);

int NALU_HYPRE_LSC_setMatrixStructure(LinSysCore *lsc, int** ptColIndices,
                     int* ptRowLengths, int** blkColIndices, int* blkRowLengths,
                     int* ptRowsPerBlkRow);

int NALU_HYPRE_LSC_resetMatrixAndVector(LinSysCore *lsc, double val);

int NALU_HYPRE_LSC_resetMatrix(LinSysCore *lsc, double val);

int NALU_HYPRE_LSC_resetRHSVector(LinSysCore *lsc, double val);

int NALU_HYPRE_LSC_sumIntoSystemMatrix(LinSysCore *lsc, int numPtRows,
                     const int* ptRows, int numPtCols, const int* ptCols,
                     int numBlkRows, const int* blkRows, int numBlkCols,
                     const int* blkCols, const double* const* values);

int NALU_HYPRE_LSC_sumIntoRHSVector(LinSysCore *lsc, int num, const double* values, 
                             const int* indices);

int NALU_HYPRE_LSC_matrixLoadComplete(LinSysCore *lsc);

int NALU_HYPRE_LSC_enforceEssentialBC(LinSysCore *lsc, int* globalEqn, double* alpha, 
                             double* gamma, int leng);

int NALU_HYPRE_LSC_enforceRemoteEssBCs(LinSysCore *lsc,int numEqns,int* globalEqns,
                             int** colIndices, int* colIndLen, double** coefs);

int NALU_HYPRE_LSC_enforceOtherBC(LinSysCore *lsc, int* globalEqn, double* alpha, 
                             double* beta, double* gamma, int leng);

int NALU_HYPRE_LSC_putInitialGuess(LinSysCore *lsc, const int* eqnNumbers,
                             const double* values, int leng);

int NALU_HYPRE_LSC_getSolution(LinSysCore *lsc, double *answers, int leng);

int NALU_HYPRE_LSC_getSolnEntry(LinSysCore *lsc, int eqnNumber, double *answer);

int NALU_HYPRE_LSC_formResidual(LinSysCore *lsc, double *values, int leng);

int NALU_HYPRE_LSC_launchSolver(LinSysCore *lsc, int *solveStatus, int *iter);

int NALU_HYPRE_LSC_FEDataInitFields(LinSysCore* lsc, int nFields, int *fieldSizes,
                               int *fieldIDs);

int NALU_HYPRE_LSC_FEDataInitElemBlock(LinSysCore* lsc, int nElems, int nNodes,
                                  int nNodeFields, int *nodeFieldIDs);

int NALU_HYPRE_LSC_FEDataInitElemNodeList(LinSysCore* lsc, int elemID, int nNodes,
                                     int *nList);

int NALU_HYPRE_LSC_FEDataInitSharedNodes(LinSysCore* lsc, int nShared, int *sharedIDs,
                                    int *sharedPLengs, int **sharedProcs);

int NALU_HYPRE_LSC_FEDataInitComplete(LinSysCore* lsc);

int NALU_HYPRE_LSC_FEDataLoadElemMatrix(LinSysCore* lsc, int elemID, int nNodes,
                                   int *nList, int sDim, double **sMat);

#ifdef __cplusplus
}
#endif

#endif

