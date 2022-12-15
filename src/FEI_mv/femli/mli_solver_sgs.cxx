/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include <math.h>
#include <string.h>
#include "_nalu_hypre_parcsr_mv.h"
#include "mli_solver_sgs.h"

/******************************************************************************
 * symmetric Gauss-Seidel relaxation scheme
 *****************************************************************************/

/******************************************************************************
 * constructor
 *---------------------------------------------------------------------------*/

MLI_Solver_SGS::MLI_Solver_SGS(char *name) : MLI_Solver(name)
{
   Amat_             = NULL;
   zeroInitialGuess_ = 0;
   nSweeps_          = 1;
   relaxWeights_     = new double[1];
   relaxWeights_[0]  = 1.0;
   myColor_          = 0;
   numColors_        = 1;
   scheme_           = 1;
   printRNorm_       = 0;
   findOmega_        = 0;
   omegaIncrement_   = 0.05;
   omegaNumIncr_     = 20;
}

/******************************************************************************
 * destructor
 *---------------------------------------------------------------------------*/

MLI_Solver_SGS::~MLI_Solver_SGS()
{
   if ( relaxWeights_ != NULL ) delete [] relaxWeights_;
   relaxWeights_ = NULL;
}

/******************************************************************************
 * set up the smoother
 *---------------------------------------------------------------------------*/

int MLI_Solver_SGS::setup(MLI_Matrix *mat)
{
   Amat_ = mat;
   nalu_hypre_ParCSRMatrix *A;
   MPI_Comm           comm;

   if      ( scheme_ == 0 ) doProcColoring();
   else if ( scheme_ == 1 ) 
   {
      myColor_   = 0;
      numColors_ = 1;
      if ( findOmega_ == 1 ) findOmega();   
   }
   else
   {
      A    = (nalu_hypre_ParCSRMatrix *) Amat_->getMatrix();
      comm = nalu_hypre_ParCSRMatrixComm(A);
      MPI_Comm_size(comm, &numColors_);
      MPI_Comm_rank(comm, &myColor_);
   }
   return 0;
}

/******************************************************************************
 * apply function
 *---------------------------------------------------------------------------*/

int MLI_Solver_SGS::solve(MLI_Vector *fIn, MLI_Vector *uIn)
{
   int                 *ADiagI, *ADiagJ, *AOffdI, *AOffdJ;
   double              *ADiagA, *AOffdA, *uData, *fData;
   int                 iStart, iEnd, jj;
   int                 i, j, is, localNRows, extNRows, *tmpJ, relaxError=0;
   int                 iC, index, nprocs, mypid, nSends, start;
   double              res;
   double              zero = 0.0, relaxWeight, rnorm;
   double              *vBufData=NULL, *tmpData, *vExtData=NULL;
   MPI_Comm            comm;
   nalu_hypre_ParCSRMatrix     *A;
   nalu_hypre_CSRMatrix        *ADiag, *AOffd;
   nalu_hypre_ParVector        *f, *u;
   nalu_hypre_ParCSRCommPkg    *commPkg;
   nalu_hypre_ParCSRCommHandle *commHandle;
   MLI_Vector             *mliRvec;
   nalu_hypre_ParVector        *hypreR;

   /*-----------------------------------------------------------------
    * fetch machine and smoother parameters
    *-----------------------------------------------------------------*/

   A          = (nalu_hypre_ParCSRMatrix *) Amat_->getMatrix();
   comm       = nalu_hypre_ParCSRMatrixComm(A);
   commPkg    = nalu_hypre_ParCSRMatrixCommPkg(A);
   ADiag      = nalu_hypre_ParCSRMatrixDiag(A);
   localNRows = nalu_hypre_CSRMatrixNumRows(ADiag);
   ADiagI     = nalu_hypre_CSRMatrixI(ADiag);
   ADiagJ     = nalu_hypre_CSRMatrixJ(ADiag);
   ADiagA     = nalu_hypre_CSRMatrixData(ADiag);
   AOffd      = nalu_hypre_ParCSRMatrixOffd(A);
   extNRows   = nalu_hypre_CSRMatrixNumCols(AOffd);
   AOffdI     = nalu_hypre_CSRMatrixI(AOffd);
   AOffdJ     = nalu_hypre_CSRMatrixJ(AOffd);
   AOffdA     = nalu_hypre_CSRMatrixData(AOffd);
   u          = (nalu_hypre_ParVector *) uIn->getVector();
   uData      = nalu_hypre_VectorData(nalu_hypre_ParVectorLocalVector(u));
   f          = (nalu_hypre_ParVector *) fIn->getVector();
   fData      = nalu_hypre_VectorData(nalu_hypre_ParVectorLocalVector(f));
   MPI_Comm_size(comm,&nprocs);  
   MPI_Comm_rank(comm,&mypid);  

   if ( printRNorm_ == 1 )
   {
      mliRvec = Amat_->createVector();
      hypreR  = (nalu_hypre_ParVector *) mliRvec->getVector();
   }

   /*-----------------------------------------------------------------
    * setting up for interprocessor communication
    *-----------------------------------------------------------------*/

   if (nprocs > 1)
   {
      nSends = nalu_hypre_ParCSRCommPkgNumSends(commPkg);
      if ( nSends > 0 )
         vBufData = new double[nalu_hypre_ParCSRCommPkgSendMapStart(commPkg,nSends)];
      else vBufData = NULL;
      if ( extNRows > 0 ) vExtData = new double[extNRows];
      else                vExtData = NULL;
   }

   /*-----------------------------------------------------------------
    * perform SGS sweeps
    *-----------------------------------------------------------------*/
 
#if 0
   if ( relaxWeights_ != NULL && relaxWeights_[0] != 0.0 && 
        relaxWeights_[0] != 1.0 ) 
      printf("\t SGS smoother : relax weight != 1.0 (%e).\n",relaxWeights_[0]);
#endif

   relaxWeight = 1.0;
   for( is = 0; is < nSweeps_; is++ )
   {
      if ( relaxWeights_ != NULL ) relaxWeight = relaxWeights_[is];
      if ( relaxWeight <= 0.0 ) relaxWeight = 1.0;

      /*-----------------------------------------------------------------
       * forward sweep
       *-----------------------------------------------------------------*/

      for ( iC = 0; iC < numColors_; iC++ )
      {
         if (nprocs > 1)
         {
            if ( zeroInitialGuess_ == 0 )
            {
               index = 0;
               for (i = 0; i < nSends; i++)
               {
                  start = nalu_hypre_ParCSRCommPkgSendMapStart(commPkg, i);
                  for (j=start;j<nalu_hypre_ParCSRCommPkgSendMapStart(commPkg,i+1);
                       j++)
                     vBufData[index++]
                         = uData[nalu_hypre_ParCSRCommPkgSendMapElmt(commPkg,j)];
               }
               commHandle = nalu_hypre_ParCSRCommHandleCreate(1,commPkg,vBufData,
                                                         vExtData);
               nalu_hypre_ParCSRCommHandleDestroy(commHandle);
               commHandle = NULL;
            }
         }

         if ( iC == myColor_ )
         {
            for (i = 0; i < localNRows; i++)
            {
               if ( ADiagA[ADiagI[i]] != zero)
               {
                  res      = fData[i];
                  iStart   = ADiagI[i];
                  iEnd     = ADiagI[i+1];
                  tmpJ    = &(ADiagJ[iStart]);
                  tmpData = &(ADiagA[iStart]);
                  for (jj = iStart; jj < iEnd; jj++)
                     res -= (*tmpData++) * uData[*tmpJ++];
                  if ( (zeroInitialGuess_ == 0) && (nprocs > 1) )
                  {
                     iStart  = AOffdI[i];
                     iEnd    = AOffdI[i+1];
                     tmpJ    = &(AOffdJ[iStart]);
                     tmpData = &(AOffdA[iStart]);
                     for (jj = iStart; jj < iEnd; jj++)
                        res -= (*tmpData++) * vExtData[*tmpJ++];
                  }
                  uData[i] += relaxWeight * res / ADiagA[ADiagI[i]];
               }
               else printf("MLI_Solver_SGS error : diag = 0.\n");
            }
         }
         zeroInitialGuess_ = 0;
      }

      /*-----------------------------------------------------------------
       * backward sweep
       *-----------------------------------------------------------------*/

      for ( iC = numColors_-1; iC >= 0; iC-- )
      {
         if ( (numColors_ > 1) && (nprocs > 1) )
         {
            if ( zeroInitialGuess_ == 0 )
            {
               index = 0;
               for (i = 0; i < nSends; i++)
               {
                  start = nalu_hypre_ParCSRCommPkgSendMapStart(commPkg, i);
                  for (j=start;j<nalu_hypre_ParCSRCommPkgSendMapStart(commPkg,i+1);
                       j++)
                     vBufData[index++]
                         = uData[nalu_hypre_ParCSRCommPkgSendMapElmt(commPkg,j)];
               }
               commHandle = nalu_hypre_ParCSRCommHandleCreate(1,commPkg,vBufData,
                                                         vExtData);
               nalu_hypre_ParCSRCommHandleDestroy(commHandle);
               commHandle = NULL;
            }
         }

         if ( iC == myColor_ )
         {
            for (i = localNRows-1; i > -1; i--)
            {
               if ( ADiagA[ADiagI[i]] != zero)
               {
                  res     = fData[i];
                  iStart  = ADiagI[i];
                  iEnd    = ADiagI[i+1];
                  tmpJ    = &(ADiagJ[iStart]);
                  tmpData = &(ADiagA[iStart]);
                  for (jj = iStart; jj < iEnd; jj++)
                     res -= (*tmpData++) * uData[*tmpJ++];
                  if ( (zeroInitialGuess_ == 0 ) && (nprocs > 1) )
                  {
                     iStart  = AOffdI[i];
                     iEnd    = AOffdI[i+1];
                     tmpJ    = &(AOffdJ[iStart]);
                     tmpData = &(AOffdA[iStart]);
                     for (jj = iStart; jj < iEnd; jj++)
                        res -= (*tmpData++) * vExtData[*tmpJ++];
                  }
                  uData[i] += relaxWeight * res / ADiagA[ADiagI[i]];
               }
            }
         }
      }
      if ( printRNorm_ == 1 )
      {
         nalu_hypre_ParVectorCopy( f, hypreR );
         nalu_hypre_ParCSRMatrixMatvec( -1.0, A, u, 1.0, hypreR );
         rnorm = sqrt(nalu_hypre_ParVectorInnerProd( hypreR, hypreR ));
         if ( mypid == 0 )
            printf("\tMLI_Solver_SGS iter = %4d, rnorm = %e (omega=%e)\n", 
                   is, rnorm, relaxWeight);
      }
   }
   if ( printRNorm_ == 1 ) delete mliRvec;

   /*-----------------------------------------------------------------
    * clean up and return
    *-----------------------------------------------------------------*/

   if ( vExtData != NULL ) delete [] vExtData;
   if ( vBufData != NULL ) delete [] vBufData;
   return(relaxError); 
}

/******************************************************************************
 * set SGS parameters
 *---------------------------------------------------------------------------*/

int MLI_Solver_SGS::setParams( char *paramString, int argc, char **argv )
{
   int    i;
   double *weights=NULL;
   char   param1[100], param2[100];

   sscanf(paramString, "%s", param1);
   if ( !strcmp(param1, "numSweeps") )
   {
      if ( argc != 1 ) 
      {
         printf("MLI_Solver_SGS::setParams ERROR : needs 1 arg.\n");
         return 1;
      }
      nSweeps_ = *(int*) argv[0];
      if ( nSweeps_ < 1 ) nSweeps_ = 1;
      if ( relaxWeights_ != NULL ) delete [] relaxWeights_;
      relaxWeights_ = new double[nSweeps_];
      for ( i = 0; i < nSweeps_; i++ ) relaxWeights_[i] = 1.0;
      return 0;
   }
   else if ( !strcmp(param1, "relaxWeight") )
   {
      if ( argc != 2 && argc != 1 ) 
      {
         printf("MLI_Solver_SGS::setParams ERROR : needs 1 or 2 args.\n");
         return 1;
      }
      if ( argc >= 1 ) nSweeps_ = *(int*)  argv[0];
      if ( argc == 2 ) weights = (double*) argv[1];
      if ( nSweeps_ < 1 ) nSweeps_ = 1;
      if ( relaxWeights_ != NULL ) delete [] relaxWeights_;
      relaxWeights_ = NULL;
      if ( weights != NULL )
      {
         relaxWeights_ = new double[nSweeps_];
         for ( i = 0; i < nSweeps_; i++ ) relaxWeights_[i] = weights[i];
      }
   }
   else if ( !strcmp(param1, "zeroInitialGuess") )
   {
      zeroInitialGuess_ = 1;
      return 0;
   }
   else if ( !strcmp(param1, "setScheme") )
   {
      sscanf(paramString, "%s %s", param1, param2);
      if      ( !strcmp(param2, "multicolor") ) scheme_ = 0;
      else if ( !strcmp(param2, "parallel") )   scheme_ = 1;
      else if ( !strcmp(param2, "sequential") ) scheme_ = 2;
      return 0;
   }
   else if ( !strcmp(param1, "printRNorm") )
   {
      printRNorm_ = 1;
   }
   else if ( !strcmp(param1, "findOmega") )
   {
      findOmega_ = 1;
   }
   else
   {   
      printf("MLI_Solver_SGS::setParams - parameter not recognized.\n");
      printf("                 Params = %s\n", paramString);
      return 1;
   }
   return 0;
}

/******************************************************************************
 * set SGS parameters
 *---------------------------------------------------------------------------*/

int MLI_Solver_SGS::setParams( int nsweeps, double *weights )
{
   int i;

   if ( nsweeps <= 0 )
   {
      printf("MLI_Solver_SGS::setParams WARNING : nsweeps set to 1.\n");
      nsweeps = 1;
   }
   nSweeps_ = nsweeps;
   if ( relaxWeights_ != NULL ) delete [] relaxWeights_;
   relaxWeights_ = new double[nsweeps];
   if ( weights == NULL )
   {
      printf("MLI_Solver_SGS::setParams - relax_weights set to 1.0.\n");
      for ( i = 0; i < nsweeps; i++ ) relaxWeights_[i] = 1.0;
   }
   else
   {
      for ( i = 0; i < nsweeps; i++ ) 
      {
         if (weights[i] >= 0. && weights[i] <= 2.) 
            relaxWeights_[i] = weights[i];
         else 
         {
            printf("MLI_Solver_SGS::setParams - some weights set to 0.5.\n");
            relaxWeights_[i] = 1.0;
         }
      }
   }
   return 0;
}

/******************************************************************************
 * color processors
 *---------------------------------------------------------------------------*/

int MLI_Solver_SGS::doProcColoring()
{
   int                 nSends, *sendProcs, mypid, nprocs, *commGraphI;
   int                 *commGraphJ, *recvCounts, i, j, *colors, *colorsAux;
   int                 pIndex, pColor;
   MPI_Comm            comm;
   nalu_hypre_ParCSRMatrix  *A;
   nalu_hypre_ParCSRCommPkg *commPkg;

   A       = (nalu_hypre_ParCSRMatrix *) Amat_->getMatrix();
   comm    = nalu_hypre_ParCSRMatrixComm(A);
   commPkg = nalu_hypre_ParCSRMatrixCommPkg(A);
   if ( commPkg == NULL )
   {
      nalu_hypre_MatvecCommPkgCreate((nalu_hypre_ParCSRMatrix *) A);
      commPkg = nalu_hypre_ParCSRMatrixCommPkg(A);
   }
   nSends    = nalu_hypre_ParCSRCommPkgNumSends(commPkg);
   sendProcs = nalu_hypre_ParCSRCommPkgSendProcs(commPkg);

   MPI_Comm_rank(comm, &mypid);
   MPI_Comm_size(comm, &nprocs);

   commGraphI = new int[nprocs+1]; 
   recvCounts = new int[nprocs];
   MPI_Allgather(&nSends, 1, MPI_INT, recvCounts, 1, MPI_INT, comm);
   commGraphI[0] = 0;
   for ( i = 1; i <= nprocs; i++ )
      commGraphI[i] = commGraphI[i-1] + recvCounts[i-1];
   commGraphJ = new int[commGraphI[nprocs]]; 
   MPI_Allgatherv(sendProcs, nSends, MPI_INT, commGraphJ,
                  recvCounts, commGraphI, MPI_INT, comm);
   delete [] recvCounts;

#if 0
   if ( mypid == 0 )
   {
      for ( i = 0; i < nprocs; i++ )
         for ( j = commGraphI[i]; j < commGraphI[i+1]; j++ )
            printf("Graph(%d,%d)\n", i, commGraphJ[j]);
   }
#endif

   colors = new int[nprocs];
   colorsAux = new int[nprocs];
   for ( i = 0; i < nprocs; i++ ) colors[i] = colorsAux[i] = -1;
   for ( i = 0; i < nprocs; i++ )
   {
      for ( j = commGraphI[i]; j < commGraphI[i+1]; j++ )
      {
         pIndex = commGraphJ[j];
         pColor = colors[pIndex];
         if ( pColor >= 0 ) colorsAux[pColor] = 1;
      }
      for ( j = 0; j < nprocs; j++ ) 
         if ( colorsAux[j] < 0 ) break;
      colors[i] = j;
      for ( j = commGraphI[i]; j < commGraphI[i+1]; j++ )
      {
         pIndex = commGraphJ[j];
         pColor = colors[pIndex];
         if ( pColor >= 0 ) colorsAux[pColor] = -1;
      }
   }
   delete [] colorsAux;
   myColor_ = colors[mypid];
   numColors_ = 0;
   for ( j = 0; j < nprocs; j++ ) 
      if ( colors[j]+1 > numColors_ ) numColors_ = colors[j]+1;
   delete [] colors;
   if ( mypid == 0 )
      printf("\tMLI_Solver_SGS : number of colors = %d\n", numColors_);
   return 0;
}

/******************************************************************************
 * search for optimal omega
 *---------------------------------------------------------------------------*/

int MLI_Solver_SGS::findOmega()
{
   int                 *ADiagI, *ADiagJ, *AOffdI, *AOffdJ;
   double              *ADiagA, *AOffdA, *uData, *fData;
   int                 iStart, iEnd, jj;
   int                 i, j, is, iR, localNRows, extNRows, *tmpJ;
   int                 index, nprocs, mypid, nSends, start;
   double              res;
   double              zero = 0.0, relaxWeight, rnorm, *relNorms;
   double              *vBufData=NULL, *tmpData, *vExtData=NULL; 
   MPI_Comm            comm;
   nalu_hypre_ParCSRMatrix     *A;
   nalu_hypre_CSRMatrix        *ADiag, *AOffd;
   nalu_hypre_ParCSRCommPkg    *commPkg;
   nalu_hypre_ParCSRCommHandle *commHandle;
   MLI_Vector             *mliRvec, *mliFvec, *mliUvec;
   nalu_hypre_ParVector        *hypreR, *hypreF, *hypreU;

   /*-----------------------------------------------------------------
    * fetch machine and matrix parameters
    *-----------------------------------------------------------------*/

   A          = (nalu_hypre_ParCSRMatrix *) Amat_->getMatrix();
   comm       = nalu_hypre_ParCSRMatrixComm(A);
   commPkg    = nalu_hypre_ParCSRMatrixCommPkg(A);
   ADiag      = nalu_hypre_ParCSRMatrixDiag(A);
   localNRows = nalu_hypre_CSRMatrixNumRows(ADiag);
   ADiagI     = nalu_hypre_CSRMatrixI(ADiag);
   ADiagJ     = nalu_hypre_CSRMatrixJ(ADiag);
   ADiagA     = nalu_hypre_CSRMatrixData(ADiag);
   AOffd      = nalu_hypre_ParCSRMatrixOffd(A);
   extNRows   = nalu_hypre_CSRMatrixNumCols(AOffd);
   AOffdI     = nalu_hypre_CSRMatrixI(AOffd);
   AOffdJ     = nalu_hypre_CSRMatrixJ(AOffd);
   AOffdA     = nalu_hypre_CSRMatrixData(AOffd);
   MPI_Comm_size(comm,&nprocs);  
   MPI_Comm_rank(comm,&mypid);  

   /*-----------------------------------------------------------------
    * create temporary vectors
    *-----------------------------------------------------------------*/

   mliUvec = Amat_->createVector();
   hypreU  = (nalu_hypre_ParVector *) mliUvec->getVector();
   mliFvec = Amat_->createVector();
   hypreF  = (nalu_hypre_ParVector *) mliFvec->getVector();
   mliRvec = Amat_->createVector();
   hypreR  = (nalu_hypre_ParVector *) mliRvec->getVector();
   nalu_hypre_ParVectorSetRandomValues( hypreF, 23986131 ); 
   fData   = nalu_hypre_VectorData(nalu_hypre_ParVectorLocalVector(hypreF));
   uData   = nalu_hypre_VectorData(nalu_hypre_ParVectorLocalVector(hypreU));

   /*-----------------------------------------------------------------
    * setting up for interprocessor communication
    *-----------------------------------------------------------------*/

   if (nprocs > 1)
   {
      nSends = nalu_hypre_ParCSRCommPkgNumSends(commPkg);
      if ( nSends > 0 )
         vBufData = new double[nalu_hypre_ParCSRCommPkgSendMapStart(commPkg,nSends)];
      else vBufData = NULL;
      if ( extNRows > 0 ) vExtData = new double[extNRows];
      else                vExtData = NULL;
   }

   /*-----------------------------------------------------------------
    * perform SGS sweeps
    *-----------------------------------------------------------------*/
 
   relNorms = new double[omegaNumIncr_+1];
   relNorms[0] = sqrt(nalu_hypre_ParVectorInnerProd( hypreF, hypreF ));

   for( iR = 0; iR < omegaNumIncr_; iR++ )
   {
      relaxWeight = omegaIncrement_ * (iR + 1);
      nalu_hypre_ParVectorSetConstantValues(hypreU, zero);
      for( is = 0; is < nSweeps_+1; is++ )
      {
         /*--------------------------------------------------------------
          * forward sweep
          *--------------------------------------------------------------*/

         if (nprocs > 1)
         {
            if ( zeroInitialGuess_ == 0 )
            {
               index = 0;
               for (i = 0; i < nSends; i++)
               {
                  start = nalu_hypre_ParCSRCommPkgSendMapStart(commPkg, i);
                  for (j=start;j<nalu_hypre_ParCSRCommPkgSendMapStart(commPkg,i+1);
                       j++)
                     vBufData[index++]
                         = uData[nalu_hypre_ParCSRCommPkgSendMapElmt(commPkg,j)];
               }
               commHandle = nalu_hypre_ParCSRCommHandleCreate(1,commPkg,vBufData,
                                                         vExtData);
               nalu_hypre_ParCSRCommHandleDestroy(commHandle);
               commHandle = NULL;
            }
         }

         for (i = 0; i < localNRows; i++)
         {
            if ( ADiagA[ADiagI[i]] != zero)
            {
               res      = fData[i];
               iStart   = ADiagI[i];
               iEnd     = ADiagI[i+1];
               tmpJ    = &(ADiagJ[iStart]);
               tmpData = &(ADiagA[iStart]);
               for (jj = iStart; jj < iEnd; jj++)
                  res -= (*tmpData++) * uData[*tmpJ++];
               if ( (zeroInitialGuess_ == 0) && (nprocs > 1) )
               {
                  iStart  = AOffdI[i];
                  iEnd    = AOffdI[i+1];
                  tmpJ    = &(AOffdJ[iStart]);
                  tmpData = &(AOffdA[iStart]);
                  for (jj = iStart; jj < iEnd; jj++)
                     res -= (*tmpData++) * vExtData[*tmpJ++];
               }
               uData[i] += relaxWeight * res / ADiagA[ADiagI[i]];
            }
            else printf("MLI_Solver_SGS error : diag = 0.\n");
         }

         /*--------------------------------------------------------------
          * backward sweep
          *--------------------------------------------------------------*/

         for (i = localNRows-1; i > -1; i--)
         {
            if ( ADiagA[ADiagI[i]] != zero)
            {
               res     = fData[i];
               iStart  = ADiagI[i];
               iEnd    = ADiagI[i+1];
               tmpJ    = &(ADiagJ[iStart]);
               tmpData = &(ADiagA[iStart]);
               for (jj = iStart; jj < iEnd; jj++)
                  res -= (*tmpData++) * uData[*tmpJ++];
               if ( (zeroInitialGuess_ == 0 ) && (nprocs > 1) )
               {
                  iStart  = AOffdI[i];
                  iEnd    = AOffdI[i+1];
                  tmpJ    = &(AOffdJ[iStart]);
                  tmpData = &(AOffdA[iStart]);
                  for (jj = iStart; jj < iEnd; jj++)
                     res -= (*tmpData++) * vExtData[*tmpJ++];
               }
               uData[i] += relaxWeight * res / ADiagA[ADiagI[i]];
            }
         }
         zeroInitialGuess_ = 0;
         nalu_hypre_ParVectorCopy( hypreF, hypreR );
         nalu_hypre_ParCSRMatrixMatvec( -1.0, A, hypreU, 1.0, hypreR );
         rnorm = sqrt(nalu_hypre_ParVectorInnerProd( hypreR, hypreR ));
         if ( rnorm > 1.0e20 ) break;
      }
      relNorms[iR+1] = rnorm;
   }
   rnorm = relNorms[0];
   jj = 0;
   for ( iR = 1; iR <= omegaNumIncr_; iR++ )
   {
      if ( relNorms[iR] < rnorm ) 
      {
         rnorm = relNorms[iR];
         jj = iR;
      }
   }
   if ( mypid == 0 )
   {
      if ( jj == 0 )
         printf("MLI_Solver_SGS::findOmega ERROR - omega = 0.0.\n");
      else
         printf("MLI_Solver_SGS::findOmega - optimal omega = %e(%e)\n",
                omegaIncrement_*jj,rnorm/relNorms[0]); 
   }
   if ( relaxWeights_ != NULL ) delete [] relaxWeights_;
   relaxWeights_ = new double[nSweeps_+1];
   for ( i = 0; i < nSweeps_; i++ ) relaxWeights_[i] = omegaIncrement_ * jj;

   /*-----------------------------------------------------------------
    * clean up and return
    *-----------------------------------------------------------------*/

   delete mliRvec;
   delete mliUvec;
   delete mliFvec;
   if ( vExtData != NULL ) delete [] vExtData;
   if ( vBufData != NULL ) delete [] vBufData;
   delete [] relNorms;
   return 0;
}

