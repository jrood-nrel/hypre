/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include <string.h>
#include <stdlib.h>

#include "NALU_HYPRE.h"
#include "LLNL_FEI_Impl.h"
#include "utilities/_nalu_hypre_utilities.h"
#include "nalu_hypre_cfei.h"

/******************************************************************************/
/* constructor                                                                */
/*----------------------------------------------------------------------------*/

extern "C" NALU_HYPRE_FEI_Impl *NALU_HYPRE_FEI_create( MPI_Comm comm ) 
{
   NALU_HYPRE_FEI_Impl *cfei;
   LLNL_FEI_Impl  *lfei;
   cfei = nalu_hypre_TAlloc(NALU_HYPRE_FEI_Impl, 1, NALU_HYPRE_MEMORY_HOST);
   lfei = new LLNL_FEI_Impl(comm);
   cfei->fei_ = (void *) lfei;
   return (cfei);
}

/******************************************************************************/
/* Destroy function                                                           */
/*----------------------------------------------------------------------------*/

extern "C" int NALU_HYPRE_FEI_destroy(NALU_HYPRE_FEI_Impl *fei) 
{
   LLNL_FEI_Impl *lfei;
   if (fei == NULL) return 1;
   lfei = (LLNL_FEI_Impl *) fei->fei_;
   if (lfei != NULL) delete lfei;
   return(0);
}

/******************************************************************************/
/* function for setting algorithmic parameters                                */
/*----------------------------------------------------------------------------*/

extern "C"
int NALU_HYPRE_FEI_parameters(NALU_HYPRE_FEI_Impl *fei, int numParams, char **paramString)
{
   LLNL_FEI_Impl *lfei;
   if (fei == NULL) return 1;
   if (fei->fei_ == NULL) return 1;
   lfei = (LLNL_FEI_Impl *) fei->fei_;
   lfei->parameters(numParams, paramString);
   return(0);
}

/******************************************************************************/
/* set solve type                                                             */
/*----------------------------------------------------------------------------*/

extern "C"
int NALU_HYPRE_FEI_setSolveType(NALU_HYPRE_FEI_Impl *fei, int solveType)
{
   LLNL_FEI_Impl *lfei;
   if (fei == NULL) return 1;
   if (fei->fei_ == NULL) return 1;
   lfei = (LLNL_FEI_Impl *) fei->fei_;
   lfei->setSolveType(solveType);
   return(0);
}

/******************************************************************************/
/* initialize different fields                                                */
/*----------------------------------------------------------------------------*/

extern "C"
int NALU_HYPRE_FEI_initFields(NALU_HYPRE_FEI_Impl *fei, int numFields, int *fieldSizes,
                         int *fieldIDs)
{
   LLNL_FEI_Impl *lfei;
   if (fei == NULL) return 1;
   if (fei->fei_ == NULL) return 1;
   lfei = (LLNL_FEI_Impl *) fei->fei_;
   lfei->initFields(numFields, fieldSizes, fieldIDs);
   return(0);
}

/******************************************************************************/
/* initialize element block                                                   */
/*----------------------------------------------------------------------------*/

extern "C"
int NALU_HYPRE_FEI_initElemBlock(NALU_HYPRE_FEI_Impl *fei, int elemBlockID, int numElements,
                            int numNodesPerElement, int *numFieldsPerNode,
                            int **nodalFieldIDs, int numElemDOFFieldsPerElement,
                            int *elemDOFFieldIDs, int interleaveStrategy)
{
   LLNL_FEI_Impl *lfei;
   if (fei == NULL) return 1;
   if (fei->fei_ == NULL) return 1;
   lfei = (LLNL_FEI_Impl *) fei->fei_;
   lfei->initElemBlock(elemBlockID, numElements, numNodesPerElement, 
                       numFieldsPerNode, nodalFieldIDs, 
                       numElemDOFFieldsPerElement, elemDOFFieldIDs, 
                       interleaveStrategy);
   return(0);
}

/******************************************************************************/
/* initialize element connectivity                                            */
/*----------------------------------------------------------------------------*/

extern "C"
int NALU_HYPRE_FEI_initElem(NALU_HYPRE_FEI_Impl *fei, int elemBlockID, int elemID,
                       int *elemConn)
{
   LLNL_FEI_Impl *lfei;
   if (fei == NULL) return 1;
   if (fei->fei_ == NULL) return 1;
   lfei = (LLNL_FEI_Impl *) fei->fei_;
   lfei->initElem(elemBlockID, elemID, elemConn);
   return(0);
}

/******************************************************************************/
/* initialize shared nodes                                                    */
/*----------------------------------------------------------------------------*/

extern "C"
int NALU_HYPRE_FEI_initSharedNodes(NALU_HYPRE_FEI_Impl *fei, int nShared, int *sharedIDs,
                              int *sharedLeng, int **sharedProcs)
{
   LLNL_FEI_Impl *lfei;
   if (fei == NULL) return 1;
   if (fei->fei_ == NULL) return 1;
   lfei = (LLNL_FEI_Impl *) fei->fei_;
   lfei->initSharedNodes(nShared, sharedIDs, sharedLeng, sharedProcs);
   return(0);
}

/******************************************************************************/
/* signal completion of initialization                                        */
/*----------------------------------------------------------------------------*/

extern "C"
int NALU_HYPRE_FEI_initComplete(NALU_HYPRE_FEI_Impl *fei)
{
   LLNL_FEI_Impl *lfei;
   if (fei == NULL) return 1;
   if (fei->fei_ == NULL) return 1;
   lfei = (LLNL_FEI_Impl *) fei->fei_;
   lfei->initComplete();
   return(0);
}

/******************************************************************************/
/* reset the whole system                                                     */
/*----------------------------------------------------------------------------*/

extern "C"
int NALU_HYPRE_FEI_resetSystem(NALU_HYPRE_FEI_Impl *fei, double s)
{
   LLNL_FEI_Impl *lfei;
   if (fei == NULL) return 1;
   if (fei->fei_ == NULL) return 1;
   lfei = (LLNL_FEI_Impl *) fei->fei_;
   lfei->resetSystem(s);
   return(0);
}

/******************************************************************************/
/* reset the matrix                                                           */
/*----------------------------------------------------------------------------*/

extern "C"
int NALU_HYPRE_FEI_resetMatrix(NALU_HYPRE_FEI_Impl *fei, double s)
{
   LLNL_FEI_Impl *lfei;
   if (fei == NULL) return 1;
   if (fei->fei_ == NULL) return 1;
   lfei = (LLNL_FEI_Impl *) fei->fei_;
   lfei->resetMatrix(s);
   return(0);
}

/******************************************************************************/
/* reset the right hand side                                                  */
/*----------------------------------------------------------------------------*/

extern "C"
int NALU_HYPRE_FEI_resetRHSVector(NALU_HYPRE_FEI_Impl *fei, double s)
{
   LLNL_FEI_Impl *lfei;
   if (fei == NULL) return 1;
   if (fei->fei_ == NULL) return 1;
   lfei = (LLNL_FEI_Impl *) fei->fei_;
   lfei->resetRHSVector(s);
   return(0);
}

/******************************************************************************/
/* reset the initial guess                                                    */
/*----------------------------------------------------------------------------*/

extern "C"
int NALU_HYPRE_FEI_resetInitialGuess(NALU_HYPRE_FEI_Impl *fei, double s)
{
   LLNL_FEI_Impl *lfei;
   if (fei == NULL) return 1;
   if (fei->fei_ == NULL) return 1;
   lfei = (LLNL_FEI_Impl *) fei->fei_;
   lfei->resetInitialGuess(s);
   return(0);
}

/******************************************************************************/
/* load boundary condition                                                    */
/*----------------------------------------------------------------------------*/

extern "C"
int NALU_HYPRE_FEI_loadNodeBCs(NALU_HYPRE_FEI_Impl *fei, int nNodes, int *nodeIDs,
                          int fieldID, double **alpha, double **beta, 
                          double **gamma)
{
   LLNL_FEI_Impl *lfei;
   if (fei == NULL) return 1;
   if (fei->fei_ == NULL) return 1;
   lfei = (LLNL_FEI_Impl *) fei->fei_;
   lfei->loadNodeBCs(nNodes, nodeIDs, fieldID, alpha, beta, gamma);
   return(0);
}

/******************************************************************************/
/* submit element stiffness matrix (with right hand side)                     */
/*----------------------------------------------------------------------------*/

extern "C"
int NALU_HYPRE_FEI_sumInElem(NALU_HYPRE_FEI_Impl *fei, int elemBlock, int elemID, 
                        int *elemConn, double **elemStiff, double *elemLoad, 
                        int elemFormat)
{
   LLNL_FEI_Impl *lfei;
   if (fei == NULL) return 1;
   if (fei->fei_ == NULL) return 1;
   lfei = (LLNL_FEI_Impl *) fei->fei_;
   lfei->sumInElem(elemBlock, elemID, elemConn, elemStiff, elemLoad, 
                   elemFormat);
   return(0);
}

/******************************************************************************/
/* submit element stiffness matrix                                            */
/*----------------------------------------------------------------------------*/

extern "C"
int NALU_HYPRE_FEI_sumInElemMatrix(NALU_HYPRE_FEI_Impl *fei, int elemBlock, int elemID, 
                              int *elemConn, double **elemStiff, int elemFormat)
{
   LLNL_FEI_Impl *lfei;
   if (fei == NULL) return 1;
   if (fei->fei_ == NULL) return 1;
   lfei = (LLNL_FEI_Impl *) fei->fei_;
   lfei->sumInElemMatrix(elemBlock,elemID,elemConn,elemStiff,elemFormat);
   return(0);
}

/******************************************************************************/
/* submit element right hand side                                             */
/*----------------------------------------------------------------------------*/

extern "C"
int NALU_HYPRE_FEI_sumInElemRHS(NALU_HYPRE_FEI_Impl *fei, int elemBlock, int elemID,
                           int *elemConn, double *elemLoad)
{
   LLNL_FEI_Impl *lfei;
   if (fei == NULL) return 1;
   if (fei->fei_ == NULL) return 1;
   lfei = (LLNL_FEI_Impl *) fei->fei_;
   lfei->sumInElemRHS(elemBlock, elemID, elemConn, elemLoad);
   return(0);
}

/******************************************************************************/
/* signal completion of loading                                               */
/*----------------------------------------------------------------------------*/

extern "C"
int NALU_HYPRE_FEI_loadComplete(NALU_HYPRE_FEI_Impl *fei)
{
   LLNL_FEI_Impl *lfei;
   if (fei == NULL) return 1;
   if (fei->fei_ == NULL) return 1;
   lfei = (LLNL_FEI_Impl *) fei->fei_;
   lfei->loadComplete();
   return(0);
}

/******************************************************************************/
/* solve the linear system                                                    */
/*----------------------------------------------------------------------------*/

extern "C"
int NALU_HYPRE_FEI_solve(NALU_HYPRE_FEI_Impl *fei, int *status)
{
   LLNL_FEI_Impl *lfei;
   if (fei == NULL) return 1;
   if (fei->fei_ == NULL) return 1;
   lfei = (LLNL_FEI_Impl *) fei->fei_;
   lfei->solve(status);
   return(0);
}

/******************************************************************************/
/* get the iteration count                                                    */
/*----------------------------------------------------------------------------*/

extern "C"
int NALU_HYPRE_FEI_iterations(NALU_HYPRE_FEI_Impl *fei, int *iterTaken)
{
   LLNL_FEI_Impl *lfei;
   if (fei == NULL) return 1;
   if (fei->fei_ == NULL) return 1;
   lfei = (LLNL_FEI_Impl *) fei->fei_;
   lfei->iterations(iterTaken);
   return(0);
}

/******************************************************************************/
/* compute residual norm of the solution                                      */
/*----------------------------------------------------------------------------*/

extern "C"
int NALU_HYPRE_FEI_residualNorm(NALU_HYPRE_FEI_Impl *fei, int whichNorm, int numFields,
                           int* fieldIDs, double* norms)
{
   LLNL_FEI_Impl *lfei;
   if (fei == NULL) return 1;
   if (fei->fei_ == NULL) return 1;
   lfei = (LLNL_FEI_Impl *) fei->fei_;
   lfei->residualNorm(whichNorm, numFields, fieldIDs, norms);
   return(0);
}

/******************************************************************************/
/* retrieve solution information (number of nodes)                            */
/*----------------------------------------------------------------------------*/

extern "C"
int NALU_HYPRE_FEI_getNumBlockActNodes(NALU_HYPRE_FEI_Impl *fei, int blockID, int *nNodes)
{
   LLNL_FEI_Impl *lfei;
   if (fei == NULL) return 1;
   if (fei->fei_ == NULL) return 1;
   lfei = (LLNL_FEI_Impl *) fei->fei_;
   lfei->getNumBlockActNodes(blockID, nNodes);
   return(0);
}

/******************************************************************************/
/* retrieve solution information (number of equations)                        */
/*----------------------------------------------------------------------------*/

extern "C"
int NALU_HYPRE_FEI_getNumBlockActEqns(NALU_HYPRE_FEI_Impl *fei, int blockID, int *nEqns)
{
   LLNL_FEI_Impl *lfei;
   if (fei == NULL) return 1;
   if (fei->fei_ == NULL) return 1;
   lfei = (LLNL_FEI_Impl *) fei->fei_;
   lfei->getNumBlockActEqns(blockID, nEqns);
   return(0);
}

/******************************************************************************/
/* retrieve solution information (node ID list)                               */
/*----------------------------------------------------------------------------*/

extern "C"
int NALU_HYPRE_FEI_getBlockNodeIDList(NALU_HYPRE_FEI_Impl *fei, int blockID, int numNodes,
                                 int *nodeIDList)
{
   LLNL_FEI_Impl *lfei;
   if (fei == NULL) return 1;
   if (fei->fei_ == NULL) return 1;
   lfei = (LLNL_FEI_Impl *) fei->fei_;
   lfei->getBlockNodeIDList(blockID, numNodes, nodeIDList);
   return(0);
}

/******************************************************************************/
/* retrieve solution information (actual solution)                            */
/*----------------------------------------------------------------------------*/

extern "C"
int NALU_HYPRE_FEI_getBlockNodeSolution(NALU_HYPRE_FEI_Impl *fei, int blockID, int numNodes,
                                   int *nodeIDList, int *solnOffsets, 
                                   double *solnValues)
{
   LLNL_FEI_Impl *lfei;
   if (fei == NULL) return 1;
   if (fei->fei_ == NULL) return 1;
   lfei = (LLNL_FEI_Impl *) fei->fei_;
   lfei->getBlockNodeSolution(blockID, numNodes, nodeIDList, solnOffsets, 
                              solnValues);
   return(0);
}

/******************************************************************************/
/* initialze constraints                                                      */
/*----------------------------------------------------------------------------*/

extern "C"
int NALU_HYPRE_FEI_initCRMult(NALU_HYPRE_FEI_Impl *fei, int CRListLen, int *CRNodeList,
                         int *CRFieldList, int *CRID)
{
   LLNL_FEI_Impl *lfei;
   if (fei == NULL) return 1;
   if (fei->fei_ == NULL) return 1;
   lfei = (LLNL_FEI_Impl *) fei->fei_;
   lfei->initCRMult(CRListLen, CRNodeList, CRFieldList, CRID);
   return(0);
}

/******************************************************************************/
/* load constraints                                                           */
/*----------------------------------------------------------------------------*/

extern "C"
int NALU_HYPRE_FEI_loadCRMult(NALU_HYPRE_FEI_Impl *fei, int CRID, int CRListLen, 
                         int *CRNodeList, int *CRFieldList, double *CRWeightList, 
                         double CRValue)
{
   LLNL_FEI_Impl *lfei;
   if (fei == NULL) return 1;
   if (fei->fei_ == NULL) return 1;
   lfei = (LLNL_FEI_Impl *) fei->fei_;
   lfei->loadCRMult(CRID, CRListLen, CRNodeList, CRFieldList, CRWeightList, 
                    CRValue);
   return(0);
}

