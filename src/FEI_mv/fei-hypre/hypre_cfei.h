/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#ifndef _nalu_hypre_cfei_h_
#define _nalu_hypre_cfei_h_

struct NALU_HYPRE_FEI_struct {
   void* fei_;
};
typedef struct NALU_HYPRE_FEI_struct NALU_HYPRE_FEI_Impl;

#ifdef __cplusplus
extern "C" {
#endif

NALU_HYPRE_FEI_Impl *NALU_HYPRE_FEI_create( MPI_Comm comm );
int NALU_HYPRE_FEI_destroy(NALU_HYPRE_FEI_Impl* fei);
int NALU_HYPRE_FEI_parameters(NALU_HYPRE_FEI_Impl *fei, int numParams, char **paramString);
int NALU_HYPRE_FEI_setSolveType(NALU_HYPRE_FEI_Impl *fei, int solveType);
int NALU_HYPRE_FEI_initFields(NALU_HYPRE_FEI_Impl *fei, int numFields, int *fieldSizes, 
                         int *fieldIDs);
int NALU_HYPRE_FEI_initElemBlock(NALU_HYPRE_FEI_Impl *fei, int elemBlockID, int numElements,
                            int numNodesPerElement, int *numFieldsPerNode,
                            int **nodalFieldIDs, int numElemDOFFieldsPerElement,
                            int *elemDOFFieldIDs, int interleaveStrategy);
int NALU_HYPRE_FEI_initElem(NALU_HYPRE_FEI_Impl *fei, int elemBlockID, int elemID, 
                            int *elemConn);
int NALU_HYPRE_FEI_initSharedNodes(NALU_HYPRE_FEI_Impl *fei, int nShared, int *sharedIDs, 
                              int *sharedLeng, int **sharedProcs);
int NALU_HYPRE_FEI_initComplete(NALU_HYPRE_FEI_Impl *fei);
int NALU_HYPRE_FEI_resetSystem(NALU_HYPRE_FEI_Impl *fei, double s);
int NALU_HYPRE_FEI_resetMatrix(NALU_HYPRE_FEI_Impl *fei, double s);
int NALU_HYPRE_FEI_resetRHSVector(NALU_HYPRE_FEI_Impl *fei, double s);
int NALU_HYPRE_FEI_resetInitialGuess(NALU_HYPRE_FEI_Impl *fei, double s);
int NALU_HYPRE_FEI_loadNodeBCs(NALU_HYPRE_FEI_Impl *fei, int nNodes, int *nodeIDs, 
                          int fieldID, double **alpha, double **beta, double **gamma);
int NALU_HYPRE_FEI_sumInElem(NALU_HYPRE_FEI_Impl *fei, int elemBlock, int elemID, int *elemConn,
                        double **elemStiff, double *elemLoad, int elemFormat);
int NALU_HYPRE_FEI_sumInElemMatrix(NALU_HYPRE_FEI_Impl *fei, int elemBlock, int elemID, 
                              int* elemConn, double **elemStiffness, int elemFormat);
int NALU_HYPRE_FEI_sumInElemRHS(NALU_HYPRE_FEI_Impl *fei, int elemBlock, int elemID, 
                           int *elemConn, double *elemLoad);
int NALU_HYPRE_FEI_loadComplete(NALU_HYPRE_FEI_Impl *fei);
int NALU_HYPRE_FEI_solve(NALU_HYPRE_FEI_Impl *fei, int *status);
int NALU_HYPRE_FEI_iterations(NALU_HYPRE_FEI_Impl *fei, int *iterTaken);
int NALU_HYPRE_FEI_residualNorm(NALU_HYPRE_FEI_Impl *fei, int whichNorm, int numFields, 
                           int* fieldIDs, double* norms);
int NALU_HYPRE_FEI_getNumBlockActNodes(NALU_HYPRE_FEI_Impl *fei, int blockID, int *nNodes);
int NALU_HYPRE_FEI_getNumBlockActEqns(NALU_HYPRE_FEI_Impl *fei, int blockID, int *nEqns);
int NALU_HYPRE_FEI_getBlockNodeIDList(NALU_HYPRE_FEI_Impl *fei, int blockID, int numNodes, 
                                 int *nodeIDList);
int NALU_HYPRE_FEI_getBlockNodeSolution(NALU_HYPRE_FEI_Impl *fei, int blockID, int numNodes, 
                                   int *nodeIDList, int *solnOffsets, double *solnValues);
int NALU_HYPRE_FEI_initCRMult(NALU_HYPRE_FEI_Impl *fei, int CRListLen, int *CRNodeList,
                         int *CRFieldList, int *CRID);
int NALU_HYPRE_FEI_loadCRMult(NALU_HYPRE_FEI_Impl *fei, int CRID, int CRListLen, int *CRNodeList,
                         int *CRFieldList, double *CRWeightList, double CRValue);

#ifdef __cplusplus
}
#endif

#endif

