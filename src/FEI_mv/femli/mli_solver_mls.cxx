/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

#include "mli_solver_mls.h"
#include "_nalu_hypre_parcsr_mv.h"

#define hmin(x,y) (((x) < (y)) ? (x) : (y))

/******************************************************************************
 * constructor
 *---------------------------------------------------------------------------*/

MLI_Solver_MLS::MLI_Solver_MLS(char *name) : MLI_Solver(name)
{
   Amat_     = NULL;
   Vtemp_    = NULL;
   Wtemp_    = NULL;
   Ytemp_    = NULL;
   maxEigen_ = 0.0;
   mlsDeg_   = 1;
   mlsBoost_ = 1.1;
   mlsOver_  = 1.1;
   for ( int i = 0; i < 5; i++ ) mlsOm_[i] = 0.0;
   mlsOm2_   = 1.8;
   for ( int j = 0; j < 5; j++ ) mlsCf_[j] = 0.0;
   zeroInitialGuess_ = 0;
}

/******************************************************************************
 * destructor
 *---------------------------------------------------------------------------*/

MLI_Solver_MLS::~MLI_Solver_MLS()
{
   Amat_ = NULL;
   if ( Vtemp_ != NULL ) delete Vtemp_;
   if ( Wtemp_ != NULL ) delete Wtemp_;
   if ( Ytemp_ != NULL ) delete Ytemp_;
}

/******************************************************************************
 * set up the smoother
 * (This setup is modified from Marian Brezina's code in ML)
 *---------------------------------------------------------------------------*/

int MLI_Solver_MLS::setup(MLI_Matrix *mat)
{
   int    i, j, nGrid, MAX_DEG=5, nSamples=20000;
   double cosData0, cosData1, coord, *ritzValues;
   double sample, gridStep, rho, rho2;
   double pi=4.e0 * atan(1.e0); /* 3.141592653589793115998e0; */

   /*-----------------------------------------------------------------
    * check that proper spectral radius is passed in
    *-----------------------------------------------------------------*/

   Amat_ = mat;
   if ( maxEigen_ <= 0.0 )
   {
      ritzValues = new double[2];
      MLI_Utils_ComputeExtremeRitzValues( (nalu_hypre_ParCSRMatrix *)
                              Amat_->getMatrix(), ritzValues, 0 );
      maxEigen_ = ritzValues[0];
      delete [] ritzValues;
   }

   /*-----------------------------------------------------------------
    * compute the coefficients
    *-----------------------------------------------------------------*/

   for ( i = 0; i < MAX_DEG; i++ ) mlsOm_[i] = 0.e0;
   rho  = mlsOver_ * maxEigen_;
   cosData1 = 1.e0 / (2.e0 * (double) mlsDeg_ + 1.e0);
   for ( i = 0; i < mlsDeg_; i++ ) 
   {
      cosData0 = (2.0 * (double) i + 2.0) * pi;
      mlsOm_[i] = 2.e0 / (rho * (1.e0 - cos(cosData0 * cosData1)));
   }
   mlsCf_[0] = mlsOm_[0] + mlsOm_[1] + mlsOm_[2] + mlsOm_[3] + mlsOm_[4];
   mlsCf_[1] = -(mlsOm_[0]*mlsOm_[1] + mlsOm_[0]*mlsOm_[2]
               + mlsOm_[0]*mlsOm_[3] + mlsOm_[0]*mlsOm_[4]
               + mlsOm_[1]*mlsOm_[2] + mlsOm_[1]*mlsOm_[3]
               + mlsOm_[1]*mlsOm_[4] + mlsOm_[2]*mlsOm_[3]
               + mlsOm_[2]*mlsOm_[4] + mlsOm_[3]*mlsOm_[4]);
   mlsCf_[2] = +(mlsOm_[0]*mlsOm_[1]*mlsOm_[2] + mlsOm_[0]*mlsOm_[1]*mlsOm_[3]
               + mlsOm_[0]*mlsOm_[1]*mlsOm_[4] + mlsOm_[0]*mlsOm_[2]*mlsOm_[3]
               + mlsOm_[0]*mlsOm_[2]*mlsOm_[4] + mlsOm_[0]*mlsOm_[3]*mlsOm_[4]
               + mlsOm_[1]*mlsOm_[2]*mlsOm_[3] + mlsOm_[1]*mlsOm_[2]*mlsOm_[4]
               + mlsOm_[1]*mlsOm_[3]*mlsOm_[4] + mlsOm_[2]*mlsOm_[3]*mlsOm_[4]);
   mlsCf_[3] = -(mlsOm_[0]*mlsOm_[1]*mlsOm_[2]*mlsOm_[3]
               + mlsOm_[0]*mlsOm_[1]*mlsOm_[2]*mlsOm_[4]
               + mlsOm_[0]*mlsOm_[1]*mlsOm_[3]*mlsOm_[4]
               + mlsOm_[0]*mlsOm_[2]*mlsOm_[3]*mlsOm_[4]
               + mlsOm_[1]*mlsOm_[2]*mlsOm_[3]*mlsOm_[4]);
   mlsCf_[4] = mlsOm_[0] * mlsOm_[1] * mlsOm_[2] * mlsOm_[3] * mlsOm_[4];

   if ( mlsDeg_> 1 )
   {
      gridStep = rho / (double) nSamples;
      nGrid    = (int) hmin(((int)(rho/gridStep))+1, nSamples);

      rho2 = 0.e0;
      for ( i = 0; i < nGrid-1; i++ ) 
      {
         coord  = (double)(i+1) * gridStep;
         sample = 1.e0 - mlsOm_[0] * coord;
         for ( j = 1; j < mlsDeg_; j++) 
            sample *= (1.0 - mlsOm_[j] * coord);
         sample *= sample * coord;
         if (sample > rho2) rho2 = sample;
      }
   }
   else rho2 = 4.0 / ( 27.0 * mlsOm_[0] );

   if ( mlsDeg_ < 2) mlsBoost_ = 1.019e0;
   else              mlsBoost_ = 1.025e0;
   rho2 *= mlsBoost_;
   mlsOm2_ = 2.e0 / rho2;

   /*-----------------------------------------------------------------
    * allocate temporary vectors
    *-----------------------------------------------------------------*/

   if ( Vtemp_ != NULL ) delete Vtemp_;
   if ( Wtemp_ != NULL ) delete Wtemp_;
   if ( Ytemp_ != NULL ) delete Ytemp_;
   Vtemp_ = mat->createVector();
   Wtemp_ = mat->createVector();
   Ytemp_ = mat->createVector();
   return 0;
}

/******************************************************************************
 * apply function
 *---------------------------------------------------------------------------*/

int MLI_Solver_MLS::solve(MLI_Vector *fIn, MLI_Vector *uIn)
{
   int                 i, localNRows, deg;
   double              omega, coef, *uData;
   double              *VtempData, *WtempData, *YtempData;
   nalu_hypre_ParCSRMatrix  *A;
   nalu_hypre_CSRMatrix     *ADiag;
   nalu_hypre_ParVector     *Vtemp, *Wtemp, *Ytemp, *f, *u;

   /*-----------------------------------------------------------------
    * check that proper spectral radius is passed in
    *-----------------------------------------------------------------*/

   if ( maxEigen_ <= 0.0 )
   {
      printf("MLI_Solver_MLS::solver ERROR - maxEigen <= 0.\n"); 
      exit(1);
   }

   /*-----------------------------------------------------------------
    * fetch machine and smoother parameters
    *-----------------------------------------------------------------*/

   A          = (nalu_hypre_ParCSRMatrix *) Amat_->getMatrix();
   ADiag      = nalu_hypre_ParCSRMatrixDiag(A);
   localNRows = nalu_hypre_CSRMatrixNumRows(ADiag);
   f          = (nalu_hypre_ParVector *) fIn->getVector();
   u          = (nalu_hypre_ParVector *) uIn->getVector();
   uData      = nalu_hypre_VectorData(nalu_hypre_ParVectorLocalVector(u));

   /*-----------------------------------------------------------------
    * create temporary vector
    *-----------------------------------------------------------------*/

   Vtemp       = (nalu_hypre_ParVector *) Vtemp_->getVector();
   Wtemp       = (nalu_hypre_ParVector *) Wtemp_->getVector();
   Ytemp       = (nalu_hypre_ParVector *) Ytemp_->getVector();
   VtempData  = nalu_hypre_VectorData(nalu_hypre_ParVectorLocalVector(Vtemp));
   WtempData  = nalu_hypre_VectorData(nalu_hypre_ParVectorLocalVector(Wtemp));
   YtempData  = nalu_hypre_VectorData(nalu_hypre_ParVectorLocalVector(Ytemp));

   /*-----------------------------------------------------------------
    * Perform MLS iterations
    *-----------------------------------------------------------------*/
 
   /* compute  Vtemp = f - A u */

   nalu_hypre_ParVectorCopy(f,Vtemp); 
   if ( zeroInitialGuess_ != 0 )
   {
      nalu_hypre_ParCSRMatrixMatvec(-1.0, A, u, 1.0, Vtemp);
      zeroInitialGuess_ = 0;
   }

   if ( mlsDeg_ == 1 )
   {
      coef = mlsCf_[0] * mlsOver_;

      /* u = u + coef * Vtemp */

#ifdef NALU_HYPRE_USING_OPENMP
#pragma omp parallel for private(i) NALU_HYPRE_SMP_SCHEDULE
#endif
      for (i = 0; i < localNRows; i++) uData[i] += (coef * VtempData[i]);

      /* compute residual Vtemp = A u - f */

      nalu_hypre_ParVectorCopy(f,Vtemp); 
      nalu_hypre_ParCSRMatrixMatvec(1.0, A, u, -1.0, Vtemp);

      /* compute residual Wtemp = (I - omega * A)^deg Vtemp */

      nalu_hypre_ParVectorCopy(Vtemp,Wtemp); 
      for ( deg = 0; deg < mlsDeg_; deg++ ) 
      {
         omega = mlsOm_[deg];
         nalu_hypre_ParCSRMatrixMatvec(1.0, A, Wtemp, 0.0, Vtemp);
         for (i = 0; i < localNRows; i++) 
            WtempData[i] -= (omega * VtempData[i]);
      }

      /* compute residual Vtemp = (I - omega * A)^deg Wtemp */

      nalu_hypre_ParVectorCopy(Wtemp,Vtemp); 
      for ( deg = mlsDeg_-1; deg > -1; deg-- ) 
      {
         omega = mlsOm_[deg];
         nalu_hypre_ParCSRMatrixMatvec(1.0, A, Vtemp, 0.0, Wtemp);
         for (i = 0; i < localNRows; i++) 
            VtempData[i] -= (omega * WtempData[i]);
      }

      /* compute u = u - coef * Vtemp */

      coef = mlsOver_ * mlsOm2_;

#ifdef NALU_HYPRE_USING_OPENMP
#pragma omp parallel for private(i) NALU_HYPRE_SMP_SCHEDULE
#endif
      for (i = 0; i < localNRows; i++) uData[i] -= ( coef * VtempData[i] );

   }
   else
   {
      /* Ytemp = coef * Vtemp */

      coef = mlsCf_[0];

#ifdef NALU_HYPRE_USING_OPENMP
#pragma omp parallel for private(i) NALU_HYPRE_SMP_SCHEDULE
#endif
      for (i = 0; i < localNRows; i++) YtempData[i] = (coef * VtempData[i]);

      /* Wtemp = coef * Vtemp */

      for ( deg = 1; deg < mlsDeg_; deg++ ) 
      {
         nalu_hypre_ParCSRMatrixMatvec(1.0, A, Vtemp, 0.0, Wtemp);
         nalu_hypre_ParVectorCopy(Wtemp,Vtemp); 
         coef = mlsCf_[deg];

#ifdef NALU_HYPRE_USING_OPENMP
#pragma omp parallel for private(i) NALU_HYPRE_SMP_SCHEDULE
#endif
         for (i = 0; i < localNRows; i++) 
            YtempData[i] += ( coef * WtempData[i] );
      }

#ifdef NALU_HYPRE_USING_OPENMP
#pragma omp parallel for private(i) NALU_HYPRE_SMP_SCHEDULE
#endif
      for (i = 0; i < localNRows; i++) uData[i] += (mlsOver_ * YtempData[i]);

      /* compute residual Vtemp = A u - f */

      nalu_hypre_ParVectorCopy(f,Vtemp); 
      nalu_hypre_ParCSRMatrixMatvec(1.0, A, u, -1.0, Vtemp);

      /* compute residual Wtemp = (I - omega * A)^deg Vtemp */

      nalu_hypre_ParVectorCopy(Vtemp,Wtemp); 
      for ( deg = 0; deg < mlsDeg_; deg++ ) 
      {
         omega = mlsOm_[deg];
         nalu_hypre_ParCSRMatrixMatvec(1.0, A, Wtemp, 0.0, Vtemp);
         for (i = 0; i < localNRows; i++) 
            WtempData[i] -= (omega * VtempData[i]);
      }

      /* compute residual Vtemp = (I - omega * A)^deg Wtemp */

      nalu_hypre_ParVectorCopy(Wtemp,Vtemp); 
      for ( deg = mlsDeg_-1; deg > -1; deg-- ) 
      {
         omega = mlsOm_[deg];
         nalu_hypre_ParCSRMatrixMatvec(1.0, A, Vtemp, 0.0, Wtemp);
         for (i = 0; i < localNRows; i++) 
            VtempData[i] -= (omega * WtempData[i]);
      }

      /* compute u = u - coef * Vtemp */

      coef = mlsOver_ * mlsOm2_;

#ifdef NALU_HYPRE_USING_OPENMP
#pragma omp parallel for private(i) NALU_HYPRE_SMP_SCHEDULE
#endif
      for (i = 0; i < localNRows; i++) uData[i] -= ( coef * VtempData[i] );

   }
   return(0); 
}

/******************************************************************************
 * set MLS parameters
 *---------------------------------------------------------------------------*/

int MLI_Solver_MLS::setParams( char *paramString, int argc, char **argv )
{
   if ( !strcmp(paramString, "maxEigen") )
   {
      if ( argc != 1 ) 
      {
         printf("MLI_Solver_MLS::setParams ERROR : needs 1 or 2 args.\n");
         return 1;
      }
      maxEigen_ = *(double*) argv[0];
      if ( maxEigen_ < 0.0 ) 
      {
         printf("MLI_Solver_MLS::setParams ERROR - maxEigen <= 0 (%e)\n", 
                maxEigen_);
         maxEigen_ = 0.0;
         return 1;
      }
   }
   else if ( !strcmp(paramString, "zeroInitialGuess") )
   {
      zeroInitialGuess_ = 1;
   }
   return 0;
}

/******************************************************************************
 * set MLS parameters
 *---------------------------------------------------------------------------*/

int MLI_Solver_MLS::setParams( double eigen )
{
   if ( maxEigen_ <= 0.0 )
   {
      printf("MLI_Solver_MLS::setParams WARNING - maxEigen <= 0.\n");
      return 1; 
   }
   maxEigen_ = eigen;
   return 0;
}

