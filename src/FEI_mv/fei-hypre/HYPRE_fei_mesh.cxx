/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * NALU_HYPRE_fei_matrix functions
 *
 *****************************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#ifdef HAVE_FEI
#include "FEI_Implementation.h"
#endif
#include "LLNL_FEI_Impl.h"
#include "fei_mv.h"

/*****************************************************************************/
/* NALU_HYPRE_FEMeshCreate function                                               */
/*---------------------------------------------------------------------------*/

extern "C" int
NALU_HYPRE_FEMeshCreate(MPI_Comm comm, NALU_HYPRE_FEMesh *meshptr)
{
   NALU_HYPRE_FEMesh myMesh;
   myMesh = (NALU_HYPRE_FEMesh) hypre_TAlloc(NALU_HYPRE_FEMesh, 1, NALU_HYPRE_MEMORY_HOST);
   myMesh->comm_   = comm;
   myMesh->linSys_ = NULL;
   myMesh->feiPtr_ = NULL;
   myMesh->objectType_ = -1;
   (*meshptr) = myMesh;
   return 0;
}

/*****************************************************************************/
/* NALU_HYPRE_FEMatrixDestroy - Destroy a FEMatrix object.                        */
/*---------------------------------------------------------------------------*/

extern "C" int
NALU_HYPRE_FEMeshDestroy(NALU_HYPRE_FEMesh mesh)
{
   LLNL_FEI_Impl    *fei;
   LinearSystemCore *lsc;

   if (mesh)
   {
      if (mesh->feiPtr_ != NULL && mesh->objectType_ == 1)
      {
         fei = (LLNL_FEI_Impl *) mesh->feiPtr_;
         delete fei;
      }
      if (mesh->linSys_ != NULL && mesh->objectType_ == 1)
      {
         lsc = (LinearSystemCore *) mesh->linSys_;
         delete lsc;
      }
      free(mesh);
   }
   return 0;
}

/*****************************************************************************/
/* NALU_HYPRE_FEMatrixSetFEIObject                                                */
/*---------------------------------------------------------------------------*/

extern "C" int
NALU_HYPRE_FEMeshSetFEIObject(NALU_HYPRE_FEMesh mesh, void *feiObj, void *lscObj)
{
   int  numParams=1;
   char *paramString[1];
   LLNL_FEI_Impl *fei;

   if (mesh != NULL)
   {
#ifdef HAVE_FEI
      if (feiObj != NULL)
      {
         mesh->feiPtr_ = feiObj;
         mesh->linSys_ = lscObj;
         if (lscObj == NULL)
         {
            printf("NALU_HYPRE_FEMeshSetObject ERROR : lscObj not given.\n");
            mesh->feiPtr_ = NULL;
            return 1;
         }
         mesh->objectType_ = 2;
      }
      else
      {
         fei = (LLNL_FEI_Impl *) new LLNL_FEI_Impl(mesh->comm_);
         paramString[0] = new char[100];
         strcpy(paramString[0], "externalSolver HYPRE");
         fei->parameters(numParams, paramString);
         mesh->linSys_ = (void *) fei->lscPtr_->lsc_;
         mesh->feiPtr_ = (void *) fei;
         mesh->objectType_ = 1;
         delete [] paramString[0];
      }
#else
      fei = (LLNL_FEI_Impl *) new LLNL_FEI_Impl(mesh->comm_);
      paramString[0] = new char[100];
      strcpy(paramString[0], "externalSolver HYPRE");
      fei->parameters(numParams, paramString);
      mesh->linSys_ = (void *) fei->lscPtr_->lsc_;
      mesh->feiPtr_ = (void *) fei;
      mesh->objectType_ = 1;
      delete [] paramString[0];
#endif
   }
   return 0;
}

/*****************************************************************************/
/* NALU_HYPRE_FEMeshParameters                                                    */
/*---------------------------------------------------------------------------*/

extern "C" int
NALU_HYPRE_FEMeshParameters(NALU_HYPRE_FEMesh mesh, int numParams, char **paramStrings) 
{
   LLNL_FEI_Impl      *fei1;
#ifdef HAVE_FEI
   FEI_Implementation *fei2;
#endif
   int ierr=1;

   if ((mesh != NULL) && (mesh->feiPtr_ != NULL))
   {
#ifdef HAVE_FEI
      if (mesh->objectType_ == 1)
      {
         fei1 = (LLNL_FEI_Impl *) mesh->feiPtr_;
         ierr = fei1->parameters(numParams, paramStrings);
      }
      if (mesh->objectType_ == 2)
      {
         fei2 = (FEI_Implementation *) mesh->feiPtr_;
         ierr = fei2->parameters(numParams, paramStrings);
      }
#else
      fei1 = (LLNL_FEI_Impl *) mesh->feiPtr_;
      ierr = fei1->parameters(numParams, paramStrings);
#endif
   }
   return ierr;
}
/*****************************************************************************/
/* NALU_HYPRE_FEMeshInitFields                                                    */
/*---------------------------------------------------------------------------*/

extern "C" int
NALU_HYPRE_FEMeshInitFields(NALU_HYPRE_FEMesh mesh, int numFields, int *fieldSizes, 
                       int *fieldIDs)
{
   LLNL_FEI_Impl      *fei1;
#ifdef HAVE_FEI
   FEI_Implementation *fei2;
#endif
   int ierr=1;

   if ((mesh != NULL) && (mesh->feiPtr_ != NULL))
   {
#ifdef HAVE_FEI
      if (mesh->objectType_ == 1)
      {
         fei1 = (LLNL_FEI_Impl *) mesh->feiPtr_;
         ierr = fei1->initFields(numFields, fieldSizes, fieldIDs);
      }
      if (mesh->objectType_ == 2)
      {
         fei2 = (FEI_Implementation *) mesh->feiPtr_;
         ierr = fei2->initFields(numFields, fieldSizes, fieldIDs);
      }
#else
      fei1 = (LLNL_FEI_Impl *) mesh->feiPtr_;
      ierr = fei1->initFields(numFields, fieldSizes, fieldIDs);
#endif
   }
   return ierr;
}

/*****************************************************************************/
/* NALU_HYPRE_FEMeshInitElemBlock                                                 */
/*---------------------------------------------------------------------------*/

extern "C" int
NALU_HYPRE_FEMeshInitElemBlock(NALU_HYPRE_FEMesh mesh, int blockID, int nElements, 
                          int numNodesPerElement, int *numFieldsPerNode, 
                          int **nodalFieldIDs,
                          int numElemDOFFieldsPerElement,
                          int *elemDOFFieldIDs, int interleaveStrategy )
{
   int ierr=1;
   LLNL_FEI_Impl      *fei1;
#ifdef HAVE_FEI
   FEI_Implementation *fei2;
#endif

   if ((mesh != NULL) && (mesh->feiPtr_ != NULL))
   {
#ifdef HAVE_FEI
      if (mesh->objectType_ == 1)
      {
         fei1 = (LLNL_FEI_Impl *) mesh->feiPtr_;
         ierr = fei1->initElemBlock(blockID, nElements, numNodesPerElement,
                                numFieldsPerNode, nodalFieldIDs,
                                numElemDOFFieldsPerElement, elemDOFFieldIDs, 
                                interleaveStrategy);
      }
      if (mesh->objectType_ == 2)
      {
         fei2 = (FEI_Implementation *) mesh->feiPtr_;
         ierr = fei2->initElemBlock(blockID, nElements, numNodesPerElement,
                                numFieldsPerNode, nodalFieldIDs,
                                numElemDOFFieldsPerElement, elemDOFFieldIDs, 
                                interleaveStrategy);
      }
#else
     fei1 = (LLNL_FEI_Impl *) mesh->feiPtr_;
     ierr = fei1->initElemBlock(blockID, nElements, numNodesPerElement,
                                numFieldsPerNode, nodalFieldIDs,
                                numElemDOFFieldsPerElement, elemDOFFieldIDs, 
                                interleaveStrategy);
#endif
   }
   return ierr;
}

/*****************************************************************************/
/* NALU_HYPRE_FEMeshInitElem                                                      */
/*---------------------------------------------------------------------------*/

extern "C" int
NALU_HYPRE_FEMeshInitElem(NALU_HYPRE_FEMesh mesh, int blockID, int elemID, int *elemConn)
{
   int ierr=1;
   LLNL_FEI_Impl      *fei1;
#ifdef HAVE_FEI
   FEI_Implementation *fei2;
#endif

   if ((mesh != NULL) && (mesh->feiPtr_ != NULL))
   {
#ifdef HAVE_FEI
      if (mesh->objectType_ == 1)
      {
         fei1 = (LLNL_FEI_Impl *) mesh->feiPtr_;
         ierr = fei1->initElem(blockID, elemID, elemConn);
      }
      if (mesh->objectType_ == 2)
      {
         fei2 = (FEI_Implementation *) mesh->feiPtr_;
         ierr = fei2->initElem(blockID, elemID, elemConn);
      }
#else
      fei1 = (LLNL_FEI_Impl *) mesh->feiPtr_;
      ierr = fei1->initElem(blockID, elemID, elemConn);
#endif
   }
   return ierr;
}

/*****************************************************************************/
/* NALU_HYPRE_FEMeshInitSharedNodes                                               */
/*---------------------------------------------------------------------------*/

extern "C" int
NALU_HYPRE_FEMeshInitSharedNodes(NALU_HYPRE_FEMesh mesh, int nShared, int* sharedIDs, 
                            int* sharedLeng, int** sharedProcs)
{
   int ierr=1;
   LLNL_FEI_Impl      *fei1;
#ifdef HAVE_FEI
   FEI_Implementation *fei2;
#endif

   if ((mesh != NULL) && (mesh->feiPtr_ != NULL))
   {
#ifdef HAVE_FEI
      if (mesh->objectType_ == 1)
      {
         fei1 = (LLNL_FEI_Impl *) mesh->feiPtr_;
         ierr = fei1->initSharedNodes(nShared, sharedIDs, sharedLeng, 
                                      sharedProcs);
      }
      if (mesh->objectType_ == 2)
      {
         fei2 = (FEI_Implementation *) mesh->feiPtr_;
         ierr = fei2->initSharedNodes(nShared, sharedIDs, sharedLeng, 
                                      sharedProcs);
      }
#else
      fei1 = (LLNL_FEI_Impl *) mesh->feiPtr_;
      ierr = fei1->initSharedNodes(nShared, sharedIDs, sharedLeng, 
                                   sharedProcs);
#endif
   }
   return ierr;
}

/*****************************************************************************/
/* NALU_HYPRE_FEMeshInitComplete                                                  */
/*---------------------------------------------------------------------------*/

extern "C" int
NALU_HYPRE_FEMeshInitComplete(NALU_HYPRE_FEMesh mesh)
{
   int ierr=1;
   LLNL_FEI_Impl      *fei1;
#ifdef HAVE_FEI
   FEI_Implementation *fei2;
#endif

   if ((mesh != NULL) && (mesh->feiPtr_ != NULL))
   {
#ifdef HAVE_FEI
      if (mesh->objectType_ == 1)
      {
         fei1 = (LLNL_FEI_Impl *) mesh->feiPtr_;
         ierr = fei1->initComplete();
      }
      if (mesh->objectType_ == 2)
      {
         fei2 = (FEI_Implementation *) mesh->feiPtr_;
         ierr = fei2->initComplete();
      }
#else
      fei1 = (LLNL_FEI_Impl *) mesh->feiPtr_;
      ierr = fei1->initComplete();
#endif
   }
   return ierr;
}

/*****************************************************************************/
/* NALU_HYPRE_FEMeshLoadNodeBCs                                                   */
/*---------------------------------------------------------------------------*/

extern "C" int
NALU_HYPRE_FEMeshLoadNodeBCs(NALU_HYPRE_FEMesh mesh, int nNodes, int* nodeIDs, 
                        int fieldID, double** alpha, double** beta, 
                        double** gamma)

{
   int ierr=1;
   LLNL_FEI_Impl      *fei1;
#ifdef HAVE_FEI
   FEI_Implementation *fei2;
#endif

   if ((mesh != NULL) && (mesh->feiPtr_ != NULL))
   {
#ifdef HAVE_FEI
      if (mesh->objectType_ == 1)
      {
         fei1 = (LLNL_FEI_Impl *) mesh->feiPtr_;
         ierr = fei1->loadNodeBCs(nNodes,nodeIDs,fieldID,alpha,beta,gamma);
      }
      if (mesh->objectType_ == 2)
      {
         fei2 = (FEI_Implementation *) mesh->feiPtr_;
         ierr = fei2->loadNodeBCs(nNodes,nodeIDs,fieldID,alpha,beta,gamma);
      }
#else
      fei1 = (LLNL_FEI_Impl *) mesh->feiPtr_;
      ierr = fei1->loadNodeBCs(nNodes,nodeIDs,fieldID,alpha,beta,gamma);
#endif
   }
   return ierr;
}

/*****************************************************************************/
/* NALU_HYPRE_FEMeshSumInElem                                                     */
/*---------------------------------------------------------------------------*/

extern "C" int
NALU_HYPRE_FEMeshSumInElem(NALU_HYPRE_FEMesh mesh, int blockID, int elemID, int* elemConn,
                      double** elemStiffness, double *elemLoad, int elemFormat)
{
   int ierr=1;
   LLNL_FEI_Impl      *fei1;
#ifdef HAVE_FEI
   FEI_Implementation *fei2;
#endif

   if ((mesh != NULL) && (mesh->feiPtr_ != NULL))
   {
#ifdef HAVE_FEI
      if (mesh->objectType_ == 1)
      {
         fei1 = (LLNL_FEI_Impl *) mesh->feiPtr_;
         ierr = fei1->sumInElem(blockID, elemID, elemConn, elemStiffness,
                                elemLoad,  elemFormat);
      }
      if (mesh->objectType_ == 2)
      {
         fei2 = (FEI_Implementation *) mesh->feiPtr_;
         ierr = fei2->sumInElem(blockID, elemID, elemConn, elemStiffness,
                                elemLoad,  elemFormat);
      }
#else
      fei1 = (LLNL_FEI_Impl *) mesh->feiPtr_;
      ierr = fei1->sumInElem(blockID, elemID, elemConn, elemStiffness,
                             elemLoad,  elemFormat);
#endif
   }
   return ierr;
}

/*****************************************************************************/
/* NALU_HYPRE_FEMeshSumInElemMatrix                                               */
/*---------------------------------------------------------------------------*/

extern "C" int
NALU_HYPRE_FEMeshSumInElemMatrix(NALU_HYPRE_FEMesh mesh, int blockID, int elemID,
                            int* elemConn, double** elemStiffness, int elemFormat)
{
   int ierr=1;
   LLNL_FEI_Impl      *fei1;
#ifdef HAVE_FEI
   FEI_Implementation *fei2;
#endif

   if ((mesh != NULL) && (mesh->feiPtr_ != NULL))
   {
#ifdef HAVE_FEI
      if (mesh->objectType_ == 1)
      {
         fei1 = (LLNL_FEI_Impl *) mesh->feiPtr_;
         ierr = fei1->sumInElemMatrix(blockID, elemID, elemConn, elemStiffness, 
                                      elemFormat);
      }
      if (mesh->objectType_ == 2)
      {
         fei2 = (FEI_Implementation *) mesh->feiPtr_;
         ierr = fei2->sumInElemMatrix(blockID, elemID, elemConn, elemStiffness, 
                                      elemFormat);
      }
#else
      fei1 = (LLNL_FEI_Impl *) mesh->feiPtr_;
      ierr = fei1->sumInElemMatrix(blockID, elemID, elemConn, elemStiffness, 
                                   elemFormat);
#endif
   }
   return ierr;
}

/*****************************************************************************/
/* NALU_HYPRE_FEMeshSumInElemRHS                                                  */
/*---------------------------------------------------------------------------*/

extern "C" int
NALU_HYPRE_FEMeshSumInElemRHS(NALU_HYPRE_FEMesh mesh, int blockID, int elemID,
                         int* elemConn, double* elemLoad)
{
   int ierr=1;
   LLNL_FEI_Impl      *fei1;
#ifdef HAVE_FEI
   FEI_Implementation *fei2;
#endif

   if ((mesh != NULL) && (mesh->feiPtr_ != NULL))
   {
#ifdef HAVE_FEI
      if (mesh->objectType_ == 1)
      {
         fei1 = (LLNL_FEI_Impl *) mesh->feiPtr_;
         ierr = fei1->sumInElemRHS(blockID, elemID, elemConn, elemLoad);
      }
      if (mesh->objectType_ == 2)
      {
         fei2 = (FEI_Implementation *) mesh->feiPtr_;
         ierr = fei2->sumInElemRHS(blockID, elemID, elemConn, elemLoad);
      }
#else
      fei1 = (LLNL_FEI_Impl *) mesh->feiPtr_;
      ierr = fei1->sumInElemRHS(blockID, elemID, elemConn, elemLoad);
#endif
   }
   return ierr;
}

/*****************************************************************************/
/* NALU_HYPRE_FEMeshLoadComplete                                                  */
/*---------------------------------------------------------------------------*/

extern "C" int
NALU_HYPRE_FEMeshLoadComplete(NALU_HYPRE_FEMesh mesh)
{
   int ierr=1;
   LLNL_FEI_Impl      *fei1;
#ifdef HAVE_FEI
   FEI_Implementation *fei2;
#endif

   if ((mesh != NULL) && (mesh->feiPtr_ != NULL))
   {
#ifdef HAVE_FEI
      if (mesh->objectType_ == 1)
      {
         fei1 = (LLNL_FEI_Impl *) mesh->feiPtr_;
         ierr = fei1->loadComplete();
      }
      if (mesh->objectType_ == 2)
      {
         fei2 = (FEI_Implementation *) mesh->feiPtr_;
         ierr = fei2->loadComplete();
      }
#else
      fei1 = (LLNL_FEI_Impl *) mesh->feiPtr_;
      ierr = fei1->loadComplete();
#endif
   }
   return ierr;
}

/*****************************************************************************/
/* NALU_HYPRE_FEMeshSolve                                                         */
/*---------------------------------------------------------------------------*/

extern "C" int
NALU_HYPRE_FEMeshSolve(NALU_HYPRE_FEMesh mesh)
{
   int ierr=1;
   LLNL_FEI_Impl      *fei1;
#ifdef HAVE_FEI
   FEI_Implementation *fei2;
#endif

   if ((mesh != NULL) && (mesh->feiPtr_ != NULL))
   {
#ifdef HAVE_FEI
      if (mesh->objectType_ == 1)
      {
         fei1 = (LLNL_FEI_Impl *) mesh->feiPtr_;
         fei1->solve(&ierr);
      }
      if (mesh->objectType_ == 2)
      {
         fei2 = (FEI_Implementation *) mesh->feiPtr_;
         fei2->solve(&ierr);
      }
#else
      fei1 = (LLNL_FEI_Impl *) mesh->feiPtr_;
      fei1->solve(&ierr);
#endif
   }
   return ierr;
}

/*****************************************************************************/
/* NALU_HYPRE_FEMeshGetBlockNodeIDList                                            */
/*---------------------------------------------------------------------------*/

extern "C" int
NALU_HYPRE_FEMeshGetBlockNodeIDList(NALU_HYPRE_FEMesh mesh, int blockID, int numNodes, 
                               int *nodeIDList)
{
   int ierr=1;
   LLNL_FEI_Impl      *fei1;
#ifdef HAVE_FEI
   FEI_Implementation *fei2;
#endif

   if ((mesh != NULL) && (mesh->feiPtr_ != NULL))
   {
#ifdef HAVE_FEI
      if (mesh->objectType_ == 1)
      {
         fei1 = (LLNL_FEI_Impl *) mesh->feiPtr_;
         ierr = -9999;
         fei1->solve(&ierr);
         ierr = 1;
         ierr = fei1->getBlockNodeIDList(blockID, numNodes, nodeIDList);
      }
      if (mesh->objectType_ == 2)
      {
         fei2 = (FEI_Implementation *) mesh->feiPtr_;
         ierr = fei2->getBlockNodeIDList(blockID, numNodes, nodeIDList);
      }
#else
      fei1 = (LLNL_FEI_Impl *) mesh->feiPtr_;
      ierr = -9999;
      ierr = fei1->getBlockNodeIDList(blockID, numNodes, nodeIDList);
      ierr = 1;
#endif
   }
   return ierr;
}

/*****************************************************************************/
/* NALU_HYPRE_FEMeshGetBlockNodeSolution                                          */
/*---------------------------------------------------------------------------*/

extern "C" int
NALU_HYPRE_FEMeshGetBlockNodeSolution(NALU_HYPRE_FEMesh mesh, int blockID, int numNodes, 
                   int *nodeIDList, int *solnOffsets, double *solnValues)
{
   int ierr=1;
   LLNL_FEI_Impl      *fei1;
#ifdef HAVE_FEI
   FEI_Implementation *fei2;
#endif

   if ((mesh != NULL) && (mesh->feiPtr_ != NULL))
   {
#ifdef HAVE_FEI
      if (mesh->objectType_ == 1)
      {
         fei1 = (LLNL_FEI_Impl *) mesh->feiPtr_;
         ierr = fei1->getBlockNodeSolution(blockID, numNodes, nodeIDList,
                                       solnOffsets, solnValues);
      }
      if (mesh->objectType_ == 2)
      {
         fei2 = (FEI_Implementation *) mesh->feiPtr_;
         ierr = fei2->getBlockNodeSolution(blockID, numNodes, nodeIDList,
                                       solnOffsets, solnValues);
      }
#else
      fei1 = (LLNL_FEI_Impl *) mesh->feiPtr_;
      ierr = fei1->getBlockNodeSolution(blockID, numNodes, nodeIDList,
                                       solnOffsets, solnValues);
#endif
   }
   return ierr;
}

