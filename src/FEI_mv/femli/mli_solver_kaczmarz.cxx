/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include <string.h>
#include "mli_solver_kaczmarz.h"
#include "_nalu_hypre_parcsr_mv.h"

/******************************************************************************
 * Kaczmarz relaxation scheme
 *****************************************************************************/

/******************************************************************************
 * constructor
 *---------------------------------------------------------------------------*/

MLI_Solver_Kaczmarz::MLI_Solver_Kaczmarz(char *name) : MLI_Solver(name)
{
   Amat_             = NULL;
   nSweeps_          = 1;
   AsqDiag_          = NULL;
   zeroInitialGuess_ = 0;
}

/******************************************************************************
 * destructor
 *---------------------------------------------------------------------------*/

MLI_Solver_Kaczmarz::~MLI_Solver_Kaczmarz()
{
   if ( AsqDiag_ != NULL ) delete [] AsqDiag_;
   AsqDiag_ = NULL;
}

/******************************************************************************
 * set up the smoother
 *---------------------------------------------------------------------------*/

int MLI_Solver_Kaczmarz::setup(MLI_Matrix *mat)
{
   int                irow, jcol, localNRows, *ADiagI, *AOffdI;
   double             *ADiagA, *AOffdA, rowNorm;
   nalu_hypre_ParCSRMatrix *A;
   nalu_hypre_CSRMatrix    *ADiag, *AOffd;

   Amat_ = mat;

   A          = (nalu_hypre_ParCSRMatrix *) Amat_->getMatrix();
   ADiag      = nalu_hypre_ParCSRMatrixDiag(A);
   AOffd      = nalu_hypre_ParCSRMatrixOffd(A);
   localNRows = nalu_hypre_CSRMatrixNumRows(ADiag);
   ADiagI     = nalu_hypre_CSRMatrixI(ADiag);
   ADiagA     = nalu_hypre_CSRMatrixData(ADiag);
   AOffdI     = nalu_hypre_CSRMatrixI(AOffd);
   AOffdA     = nalu_hypre_CSRMatrixData(AOffd);

   if ( AsqDiag_ != NULL ) delete [] AsqDiag_;
   AsqDiag_ = new double[localNRows];
   for ( irow = 0; irow < localNRows; irow++ )
   {
      rowNorm = 0.0;
      for ( jcol = ADiagI[irow]; jcol < ADiagI[irow+1]; jcol++ )
         rowNorm += (ADiagA[jcol] * ADiagA[jcol]);
      for ( jcol = AOffdI[irow]; jcol < AOffdI[irow+1]; jcol++ )
         rowNorm += (AOffdA[jcol] * AOffdA[jcol]);
      if ( rowNorm != 0.0 ) AsqDiag_[irow] = 1.0 / rowNorm;
      else                  AsqDiag_[irow] = 1.0;
   }
   return 0;
}

/******************************************************************************
 * apply function
 *---------------------------------------------------------------------------*/

int MLI_Solver_Kaczmarz::solve(MLI_Vector *fIn, MLI_Vector *uIn)
{
   nalu_hypre_ParCSRMatrix  *A;
   nalu_hypre_CSRMatrix     *ADiag, *AOffd;
   int                 *ADiagI, *ADiagJ, *AOffdI, *AOffdJ;
   double              *ADiagA, *AOffdA, *uData, *fData;
   int                 irow, jcol, is, localNRows, retFlag=0, nprocs, start;
   int                 nSends, extNRows, index, endp1;
   double              *vBufData, *vExtData, res;
   MPI_Comm            comm;
   nalu_hypre_ParCSRCommPkg    *commPkg;
   nalu_hypre_ParVector        *f, *u;
   nalu_hypre_ParCSRCommHandle *commHandle;

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
   f          = (nalu_hypre_ParVector *) fIn->getVector();
   uData      = nalu_hypre_VectorData(nalu_hypre_ParVectorLocalVector(u));
   fData      = nalu_hypre_VectorData(nalu_hypre_ParVectorLocalVector(f));
   MPI_Comm_size(comm,&nprocs);  

   /*-----------------------------------------------------------------
    * setting up for interprocessor communication
    *-----------------------------------------------------------------*/

   if (nprocs > 1)
   {
      nSends = nalu_hypre_ParCSRCommPkgNumSends(commPkg);
      vBufData = new double[nalu_hypre_ParCSRCommPkgSendMapStart(commPkg,nSends)];
      vExtData = new double[extNRows];
      for ( irow = 0; irow < extNRows; irow++ ) vExtData[irow] = 0.0;
   }

   /*-----------------------------------------------------------------
    * perform Kaczmarz sweeps
    *-----------------------------------------------------------------*/
 
   for( is = 0; is < nSweeps_; is++ )
   {
      if (nprocs > 1 && zeroInitialGuess_ != 1 )
      {
         index = 0;
         for (irow = 0; irow < nSends; irow++)
         {
            start = nalu_hypre_ParCSRCommPkgSendMapStart(commPkg, irow);
            endp1 = nalu_hypre_ParCSRCommPkgSendMapStart(commPkg,irow+1);
            for ( jcol = start; jcol < endp1; jcol++ )
               vBufData[index++]
                      = uData[nalu_hypre_ParCSRCommPkgSendMapElmt(commPkg,jcol)];
         }
         commHandle = nalu_hypre_ParCSRCommHandleCreate(1,commPkg,vBufData,
                                                   vExtData);
         nalu_hypre_ParCSRCommHandleDestroy(commHandle);
         commHandle = NULL;
      }

      for ( irow = 0; irow < localNRows; irow++ )
      {
         res = fData[irow];
         for ( jcol = ADiagI[irow];  jcol < ADiagI[irow+1]; jcol++ )
         {
            index = ADiagJ[jcol];
            res -= ADiagA[jcol] * uData[index];
         }
         if (nprocs > 1 && zeroInitialGuess_ != 1 )
         {
            for ( jcol = AOffdI[irow]; jcol < AOffdI[irow+1]; jcol++ )
            {
               index = AOffdJ[jcol];
               res -= AOffdA[jcol] * vExtData[index];
            }
         }
         res *= AsqDiag_[irow];
         for ( jcol = ADiagI[irow];  jcol < ADiagI[irow+1]; jcol++ )
         {
            index = ADiagJ[jcol];
            uData[index] += res * ADiagA[jcol];
         }
      }
      for ( irow = localNRows-1; irow >= 0; irow-- )
      {
         res = fData[irow];
         for ( jcol = ADiagI[irow];  jcol < ADiagI[irow+1]; jcol++ )
         {
            index = ADiagJ[jcol];
            res -= ADiagA[jcol] * uData[index];
         }
         if (nprocs > 1 && zeroInitialGuess_ != 1 )
         {
            for ( jcol = AOffdI[irow]; jcol < AOffdI[irow+1]; jcol++ )
            {
               index = AOffdJ[jcol];
               res -= AOffdA[jcol] * vExtData[index];
            }
         }
         res *= AsqDiag_[irow];
         for ( jcol = ADiagI[irow]; jcol < ADiagI[irow+1]; jcol++ )
         {
            index = ADiagJ[jcol];
            uData[index] += res * ADiagA[jcol];
         }
         for ( jcol = AOffdI[irow]; jcol < AOffdI[irow+1]; jcol++ )
         {
            index = AOffdJ[jcol];
            vExtData[index] += res * AOffdA[jcol];
         }
      }
      zeroInitialGuess_ = 0;
   }

   /*-----------------------------------------------------------------
    * clean up and return
    *-----------------------------------------------------------------*/

   if (nprocs > 1)
   {
      delete [] vExtData;
      delete [] vBufData;
   }
   return (retFlag); 
}

/******************************************************************************
 * set SGS parameters
 *---------------------------------------------------------------------------*/

int MLI_Solver_Kaczmarz::setParams(char *paramString, int argc, char **argv)
{
   if (!strcmp(paramString,"numSweeps") || !strcmp(paramString,"relaxWeight"))
   {
      if ( argc >= 1 ) nSweeps_ = *(int*)  argv[0];
      if ( nSweeps_ < 1 ) nSweeps_ = 1;
   }
   else if ( !strcmp(paramString, "zeroInitialGuess") )
   {
      zeroInitialGuess_ = 1;
   }
   return 0;
}

