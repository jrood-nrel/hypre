/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include <math.h>
#include <string.h>
#include "_nalu_hypre_parcsr_mv.h"
#include "_nalu_hypre_parcsr_ls.h"
#include "par_amg.h"
#include "mli_solver_hsgs.h"

/******************************************************************************
 * symmetric Gauss-Seidel relaxation scheme in BoomerAMG
 *****************************************************************************/

/******************************************************************************
 * constructor
 *---------------------------------------------------------------------------*/

MLI_Solver_HSGS::MLI_Solver_HSGS(char *name) : MLI_Solver(name)
{
   Amat_         = NULL;
   nSweeps_      = 1;
   relaxWeights_ = 1.0;
   relaxOmega_   = 1.0;
   mliVec_       = NULL;
   calcOmega_    = 1;
}

/******************************************************************************
 * destructor
 *---------------------------------------------------------------------------*/

MLI_Solver_HSGS::~MLI_Solver_HSGS()
{
   if (mliVec_ != NULL) delete mliVec_;
   mliVec_ = NULL;
}

/******************************************************************************
 * set up the smoother
 *---------------------------------------------------------------------------*/

int MLI_Solver_HSGS::setup(MLI_Matrix *mat)
{
   Amat_ = mat;
   if (mliVec_ != NULL) delete mliVec_;
   mliVec_ = Amat_->createVector();
   if (calcOmega_ == 1) calcOmega();
   return 0;
}

/******************************************************************************
 * apply function
 *---------------------------------------------------------------------------*/

int MLI_Solver_HSGS::solve(MLI_Vector *fIn, MLI_Vector *uIn)
{
   int                relaxType=6, relaxPts=0, iS;
   nalu_hypre_ParCSRMatrix *A;
   nalu_hypre_ParVector    *f, *u, *vTemp;
   nalu_hypre_ParVector    *zTemp = NULL;

   //int              mypid;
   //double           rnorm;
   //MPI_Comm         comm;

   /*-----------------------------------------------------------------
    * fetch machine and smoother parameters
    *-----------------------------------------------------------------*/

   A     = (nalu_hypre_ParCSRMatrix *) Amat_->getMatrix();
   u     = (nalu_hypre_ParVector *) uIn->getVector();
   f     = (nalu_hypre_ParVector *) fIn->getVector();
   vTemp = (nalu_hypre_ParVector *) mliVec_->getVector();

   /* AB: need an extra vector for threading */
   if (nalu_hypre_NumThreads() > 1)
   {
      zTemp = nalu_hypre_ParVectorCreate(nalu_hypre_ParCSRMatrixComm(A),
                                    nalu_hypre_ParCSRMatrixGlobalNumRows(A),
                                    nalu_hypre_ParCSRMatrixRowStarts(A));
      nalu_hypre_ParVectorInitialize(zTemp);
   }

   //comm  = nalu_hypre_ParCSRMatrixComm(A);
   //MPI_Comm_rank(comm, &mypid);
   for (iS = 0; iS < nSweeps_; iS++)
   {
      nalu_hypre_BoomerAMGRelax(A,f,NULL,relaxType,relaxPts,relaxWeights_,
                           relaxOmega_,NULL,u,vTemp, zTemp);
      //nalu_hypre_ParVectorCopy( f, vTemp );
      //nalu_hypre_ParCSRMatrixMatvec( -1.0, A, u, 1.0, vTemp );
      //rnorm = sqrt(nalu_hypre_ParVectorInnerProd( vTemp, vTemp ));
      //if ( mypid == 0 )
      //   printf("\tMLI_Solver_HSGS iter = %4d, rnorm = %e (omega=%e)\n",
      //             iS, rnorm, relaxWeights_);
   }

   if (zTemp)
      nalu_hypre_ParVectorDestroy(zTemp);

   return 0;
}

/******************************************************************************
 * set SGS parameters
 *---------------------------------------------------------------------------*/

int MLI_Solver_HSGS::setParams(char *paramString, int argc, char **argv)
{
   double *weights=NULL;
   char   param1[100];

   sscanf(paramString, "%s", param1);
   if (!strcmp(param1, "numSweeps"))
   {
      if ( argc != 1 )
      {
         printf("MLI_Solver_HSGS::setParams ERROR : needs 1 arg.\n");
         return 1;
      }
      nSweeps_ = *(int*) argv[0];
      if ( nSweeps_ < 1 ) nSweeps_ = 1;
      return 0;
   }
   else if ( !strcmp(param1, "relaxWeight") )
   {
      if ( argc != 2 && argc != 1 )
      {
         printf("MLI_Solver_HSGS::setParams ERROR : needs 1 or 2 args.\n");
         return 1;
      }
      if ( argc >= 1 ) nSweeps_ = *(int*)  argv[0];
      if ( argc == 2 ) weights = (double*) argv[1];
      if ( nSweeps_ < 1 ) nSweeps_ = 1;
      if ( weights != NULL ) relaxWeights_ = weights[0];
   }
   else if ( !strcmp(param1, "calcOmega") )
   {
      calcOmega_ = 1;
   }
   else return 1;
   return 0;
}

/******************************************************************************
 * calculate relax weight
 *---------------------------------------------------------------------------*/

int MLI_Solver_HSGS::calcOmega()
{
   int                relaxType=6, relaxTypes[2], level=0, numCGSweeps=10;
   nalu_hypre_ParCSRMatrix *A;
   nalu_hypre_ParVector    *vTemp;
   nalu_hypre_ParAMGData   *amgData;

   A = (nalu_hypre_ParCSRMatrix *) Amat_->getMatrix();
   amgData = (nalu_hypre_ParAMGData *) nalu_hypre_BoomerAMGCreate();
   amgData->CF_marker_array = new int*[1];
   amgData->CF_marker_array[0] = NULL;
   amgData->A_array = new nalu_hypre_ParCSRMatrix*[1];
   amgData->A_array[0] = A;
   vTemp = (nalu_hypre_ParVector *) mliVec_->getVector();
   amgData->Vtemp = vTemp;
   relaxTypes[0] = 0;
   relaxTypes[1] = relaxType;
   amgData->grid_relax_type = relaxTypes;
   amgData->smooth_num_levels = 0;
   amgData->smooth_type = 0;
   nalu_hypre_BoomerAMGCGRelaxWt((void *) amgData,level,numCGSweeps,&relaxOmega_);
   //printf("NALU_HYPRE/FEI/MLI HSGS : relaxOmega = %e\n", relaxOmega_);
   delete [] amgData->A_array;
   delete [] amgData->CF_marker_array;
   nalu_hypre_TFree(amgData, NALU_HYPRE_MEMORY_HOST);
   return 0;
}
