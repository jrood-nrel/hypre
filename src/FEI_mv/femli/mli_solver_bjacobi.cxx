/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include <stdio.h>
#include <string.h>
#include "mli_solver_bjacobi.h"
#ifdef HAVE_ESSL
#include <essl.h>
#endif

#define switchSize 200

/******************************************************************************
 * BJacobi relaxation scheme 
 *****************************************************************************/

/******************************************************************************
 * constructor
 *--------------------------------------------------------------------------*/

MLI_Solver_BJacobi::MLI_Solver_BJacobi(char *name) : MLI_Solver(name)
{
   Amat_             = NULL;
   nSweeps_          = 1;
   relaxWeights_     = NULL;
   zeroInitialGuess_ = 0;
   useOverlap_       = 0;
   blockSize_        = 200;
   nBlocks_          = 0;
   blockLengths_     = NULL;
   blockSolvers_     = NULL;
   maxBlkLeng_       = 0;
   offNRows_         = 0;
   offRowIndices_    = NULL;
   offRowLengths_    = NULL;
   offCols_          = NULL;
   offVals_          = NULL;
   blkScheme_        = 0;      /* default = SuperLU */
   esslMatrices_     = NULL;
}

/******************************************************************************
 * destructor
 *--------------------------------------------------------------------------*/

MLI_Solver_BJacobi::~MLI_Solver_BJacobi()
{
   cleanBlocks();
   if (relaxWeights_ != NULL) delete [] relaxWeights_;
}

/******************************************************************************
 * setup 
 *--------------------------------------------------------------------------*/

int MLI_Solver_BJacobi::setup(MLI_Matrix *Amat_in)
{
   Amat_ = Amat_in;

   /*-----------------------------------------------------------------
    * clean up first
    *-----------------------------------------------------------------*/

   cleanBlocks();

   /*-----------------------------------------------------------------
    * fetch the extended (to other processors) portion of the matrix 
    *-----------------------------------------------------------------*/

   composeOverlappedMatrix();

   /*-----------------------------------------------------------------
    * construct the extended matrix
    *-----------------------------------------------------------------*/

#if HAVE_ESSL
   if ( blockSize_ <= switchSize && useOverlap_ == 0 ) blkScheme_ = 1;
#endif
   buildBlocks();

   return 0;
}

/******************************************************************************
 * solve function
 *---------------------------------------------------------------------------*/

int MLI_Solver_BJacobi::solve(MLI_Vector *f_in, MLI_Vector *u_in)
{
   int     iP, jP, nRecvs, *recvProcs, *recvStarts, nRecvBefore;
   int     blockStartRow, iB, iS, blockEndRow, blkLeng, length;
   int     localNRows, iStart, iEnd, irow, jcol, colIndex, index, mypid;
   int     nSends, numColsOffd, start, relaxError=0;
   int     nprocs, *partition, startRow, endRow, offOffset, *tmpJ;
   int     *ADiagI, *ADiagJ, *AOffdI, *AOffdJ, offIRow;
   double  *ADiagA, *AOffdA, *uData, *fData, *tmpA, *fExtData=NULL;
   double  relaxWeight, *vBufData=NULL, *vExtData=NULL, res;
   double  *dbleX=NULL, *dbleB=NULL;
   char    vecName[30];
   MPI_Comm               comm;
   nalu_hypre_ParCSRMatrix     *A;
   nalu_hypre_CSRMatrix        *ADiag, *AOffd;
   nalu_hypre_ParCSRCommPkg    *commPkg;
   nalu_hypre_ParCSRCommHandle *commHandle;
   nalu_hypre_ParVector        *f, *u;
   nalu_hypre_Vector           *sluB=NULL, *sluX=NULL;
   MLI_Vector             *mliX, *mliB;
#if HAVE_ESSL
   int                    info;
#endif

   /*-----------------------------------------------------------------
    * fetch machine and matrix parameters
    *-----------------------------------------------------------------*/

   A           = (nalu_hypre_ParCSRMatrix *) Amat_->getMatrix();
   comm        = nalu_hypre_ParCSRMatrixComm(A);
   commPkg     = nalu_hypre_ParCSRMatrixCommPkg(A);
   ADiag       = nalu_hypre_ParCSRMatrixDiag(A);
   localNRows  = nalu_hypre_CSRMatrixNumRows(ADiag);
   ADiagI      = nalu_hypre_CSRMatrixI(ADiag);
   ADiagJ      = nalu_hypre_CSRMatrixJ(ADiag);
   ADiagA      = nalu_hypre_CSRMatrixData(ADiag);
   AOffd       = nalu_hypre_ParCSRMatrixOffd(A);
   numColsOffd = nalu_hypre_CSRMatrixNumCols(AOffd);
   AOffdI      = nalu_hypre_CSRMatrixI(AOffd);
   AOffdJ      = nalu_hypre_CSRMatrixJ(AOffd);
   AOffdA      = nalu_hypre_CSRMatrixData(AOffd);
   u           = (nalu_hypre_ParVector *) u_in->getVector();
   uData       = nalu_hypre_VectorData(nalu_hypre_ParVectorLocalVector(u));
   f           = (nalu_hypre_ParVector *) f_in->getVector();
   fData       = nalu_hypre_VectorData(nalu_hypre_ParVectorLocalVector(f));
   partition   = nalu_hypre_ParVectorPartitioning(f);
   MPI_Comm_rank(comm,&mypid);  
   MPI_Comm_size(comm,&nprocs);  
   startRow    = partition[mypid];
   endRow      = partition[mypid+1] - 1;
   nRecvBefore = 0;
   if ( nprocs > 1 )
   {
      nRecvs      = nalu_hypre_ParCSRCommPkgNumRecvs(commPkg);
      recvProcs   = nalu_hypre_ParCSRCommPkgRecvProcs(commPkg);
      recvStarts  = nalu_hypre_ParCSRCommPkgRecvVecStarts(commPkg);
      if ( useOverlap_ )
      {
         for ( iP = 0; iP < nRecvs; iP++ )
            if ( recvProcs[iP] > mypid ) break;
         nRecvBefore = recvStarts[iP];
      } 
   }

   /*-----------------------------------------------------------------
    * setting up for interprocessor communication
    *-----------------------------------------------------------------*/

   if (nprocs > 1)
   {
      nSends = nalu_hypre_ParCSRCommPkgNumSends(commPkg);
      length = nalu_hypre_ParCSRCommPkgSendMapStart(commPkg,nSends);
      if ( length > 0 ) vBufData = new double[length];
      if ( numColsOffd > 0 )
      {
         vExtData = new double[numColsOffd];
         fExtData = new double[numColsOffd];
      }
      for ( irow = 0; irow < numColsOffd; irow++ ) vExtData[irow] = 0.0;
   }

   /*--------------------------------------------------------------------
    * communicate right hand side
    *--------------------------------------------------------------------*/

   if (nprocs > 1 && useOverlap_)
   {
      index = 0;
      for (iP = 0; iP < nSends; iP++)
      {
         start = nalu_hypre_ParCSRCommPkgSendMapStart(commPkg, iP);
         for (jP=start;jP<nalu_hypre_ParCSRCommPkgSendMapStart(commPkg,iP+1);jP++)
         {
            vBufData[index++]
                      = 0.5 * fData[nalu_hypre_ParCSRCommPkgSendMapElmt(commPkg,jP)];
            fData[nalu_hypre_ParCSRCommPkgSendMapElmt(commPkg,jP)] *= 0.5;
         }
      }
      commHandle = nalu_hypre_ParCSRCommHandleCreate(1,commPkg,vBufData,
                                                fExtData);
      nalu_hypre_ParCSRCommHandleDestroy(commHandle);
      commHandle = NULL;
   }

   if ( maxBlkLeng_ > 0 )
   {
      dbleB = new double[maxBlkLeng_];
      dbleX = new double[maxBlkLeng_];
   }
   if ( blkScheme_ == 0 )
   {
      if ( maxBlkLeng_ > 0 )
      {
         sluB  = nalu_hypre_SeqVectorCreate( maxBlkLeng_ );
         sluX  = nalu_hypre_SeqVectorCreate( maxBlkLeng_ );
      }
      nalu_hypre_VectorData(sluB) = dbleB;
      nalu_hypre_VectorData(sluX) = dbleX;
   }

   /*-----------------------------------------------------------------
    * perform block Jacobi sweeps
    *-----------------------------------------------------------------*/
 
   for ( iS = 0; iS < nSweeps_; iS++ )
   {
      if ( relaxWeights_ != NULL ) relaxWeight = relaxWeights_[iS];
      else                         relaxWeight = 1.0;
      if ( relaxWeight <= 0.0 ) relaxWeight = 1.0;

      if (nprocs > 1)
      {
         if ( zeroInitialGuess_ == 0 )
         {
            index = 0;
            for (iP = 0; iP < nSends; iP++)
            {
               start = nalu_hypre_ParCSRCommPkgSendMapStart(commPkg, iP);
               for (jP=start;
                    jP<nalu_hypre_ParCSRCommPkgSendMapStart(commPkg,iP+1); jP++)
                  vBufData[index++]
                      = uData[nalu_hypre_ParCSRCommPkgSendMapElmt(commPkg,jP)];
            }
            commHandle = nalu_hypre_ParCSRCommHandleCreate(1,commPkg,vBufData,
                                                      vExtData);
            nalu_hypre_ParCSRCommHandleDestroy(commHandle);
            commHandle = NULL;
         }
      }

      /*--------------------------------------------------------------
       * process each block 
       *--------------------------------------------------------------*/

      offOffset = offIRow = 0;
      if ( offRowLengths_ != NULL ) 
      {
         offOffset = 0;
         offIRow--;
      }
      for ( iB = 0; iB < nBlocks_; iB++ )
      {
         blkLeng = blockLengths_[iB];
         blockStartRow = iB * blockSize_ + startRow - nRecvBefore;
         blockEndRow   = blockStartRow + blkLeng - 1;

         for ( irow = blockStartRow; irow <= blockEndRow; irow++ )
         {
            index  = irow - startRow;
            if ( irow >= startRow && irow <= endRow )
            {
               res = fData[index];
               if ( zeroInitialGuess_ == 0 )
               {
                  iStart = ADiagI[index];
                  iEnd   = ADiagI[index+1];
                  tmpJ   = &(ADiagJ[iStart]);
                  tmpA   = &(ADiagA[iStart]);
                  for (jcol = iStart; jcol < iEnd; jcol++)
                     res -= *tmpA++ * uData[*tmpJ++];
                  if ( AOffdI != NULL )
                  {
                     iStart = AOffdI[index];
                     iEnd   = AOffdI[index+1];
                     tmpJ   = &(AOffdJ[iStart]);
                     tmpA   = &(AOffdA[iStart]);
                     for (jcol = iStart; jcol < iEnd; jcol++)
                        res -= *tmpA++ * vExtData[*tmpJ++];
                  }
               }
               dbleB[irow-blockStartRow] = res;
            }
            else
            {
               res = fExtData[offIRow];
               if ( zeroInitialGuess_ == 0 )
               {
                  iStart = 0;
                  iEnd   = offRowLengths_[offIRow];
                  tmpA   = &(offVals_[offOffset]);
                  tmpJ   = &(offCols_[offOffset]);
                  for (jcol = iStart; jcol < iEnd; jcol++)
                  {
                     colIndex = *tmpJ++;
                     if ( colIndex >= localNRows )   
                        res -= *tmpA++ * vExtData[colIndex-localNRows];
                     else if ( colIndex >= 0 )   
                        res -= *tmpA++ * uData[colIndex];
                     else tmpA++;
                  }
                  offOffset += iEnd;
               }
               dbleB[irow-blockStartRow] = res;
               offIRow++;
            }
         }
         if ( blkScheme_ == 0 )
         {
            nalu_hypre_VectorSize(sluB) = blkLeng;
            nalu_hypre_VectorSize(sluX) = blkLeng;
            strcpy( vecName, "NALU_HYPRE_Vector" );
            mliB = new MLI_Vector((void*) sluB, vecName, NULL);
            mliX = new MLI_Vector((void*) sluX, vecName, NULL);

            blockSolvers_[iB]->solve( mliB, mliX );

            delete mliB;
            delete mliX;
         }
         else
         {
            for ( irow = 0; irow < blkLeng; irow++ ) dbleX[irow] = dbleB[irow];
#if HAVE_ESSL
            dpps(esslMatrices_[iB], blkLeng, dbleX, 1);
#endif
         }
         for ( irow = blockStartRow; irow <= blockEndRow; irow++ )
         {
            if ( irow >= startRow && irow <= endRow )
               uData[irow-startRow] += relaxWeight * 
                                       dbleX[irow-blockStartRow];
            else
               vExtData[offIRow-blockSize_+irow-blockStartRow+1] += 
                         relaxWeight * dbleX[irow-blockStartRow];
         }
      }
      zeroInitialGuess_ = 0;
   }

   /*-----------------------------------------------------------------
    * symmetrize
    *-----------------------------------------------------------------*/

   if (nprocs > 1 && useOverlap_)
   {
      commHandle = nalu_hypre_ParCSRCommHandleCreate(2,commPkg,vExtData,
                                                vBufData);
      nalu_hypre_ParCSRCommHandleDestroy(commHandle);
      commHandle = NULL;
      index = 0;
      for (iP = 0; iP < nSends; iP++)
      {
         start = nalu_hypre_ParCSRCommPkgSendMapStart(commPkg, iP);
         for (jP=start;jP<nalu_hypre_ParCSRCommPkgSendMapStart(commPkg,iP+1);jP++)
         {
            iS = nalu_hypre_ParCSRCommPkgSendMapElmt(commPkg,jP); 
            uData[iS] = (uData[iS] + vBufData[index++]) * 0.5; 
            fData[iS] *= 2.0;
         }
      }
   }

   /*-----------------------------------------------------------------
    * clean up and return
    *-----------------------------------------------------------------*/

   if ( vExtData != NULL ) delete [] vExtData;
   if ( vBufData != NULL ) delete [] vBufData;
   if ( fExtData != NULL ) delete [] fExtData;
   if ( sluX != NULL ) nalu_hypre_SeqVectorDestroy( sluX );
   if ( sluB != NULL ) nalu_hypre_SeqVectorDestroy( sluB );

   return(relaxError); 
}

/******************************************************************************
 * set parameters
 *---------------------------------------------------------------------------*/

int MLI_Solver_BJacobi::setParams(char *paramString, int argc, char **argv)
{
   int    i;
   double *weights=NULL;
   char   param1[200];

   sscanf(paramString, "%s", param1);
   if ( !strcmp(param1, "blockSize") )
   {
      sscanf(paramString, "%s %d", param1, &blockSize_);
      if ( blockSize_ < 10 ) blockSize_ = 10;
      return 0;
   }
   else if ( !strcmp(param1, "numSweeps") )
   {
      sscanf(paramString, "%s %d", param1, &nSweeps_);
      if ( nSweeps_ < 1 ) nSweeps_ = 1;
      return 0;
   }
   else if ( !strcmp(param1, "relaxWeight") )
   {
      if ( argc != 2 && argc != 1 ) 
      {
         printf("Solver_BJacobi::setParams ERROR : needs 1 or 2 args.\n");
         return 1;
      }
      if ( argc >= 1 ) nSweeps_ = *(int*)   argv[0];
      if ( argc == 2 ) weights  = (double*) argv[1];
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
   return 1;
}

/******************************************************************************
 * compose overlapped matrix
 *--------------------------------------------------------------------------*/

int MLI_Solver_BJacobi::composeOverlappedMatrix()
{
   nalu_hypre_ParCSRMatrix *A;
   MPI_Comm    comm;
   MPI_Request *requests;
   MPI_Status  *status;
   int         i, j, k, mypid, nprocs, *partition, startRow, endRow;
   int         localNRows, extNRows, nSends, *sendProcs, nRecvs;
   int         *recvProcs, *recvStarts, proc, offset, length, reqNum; 
   int         totalSendNnz, totalRecvNnz, index, base, totalSends;
   int         totalRecvs, rowNum, rowSize, *colInd, *sendStarts;
   int         limit, *iSendBuf, curNnz, *recvIndices, colIndex; 
   double      *dSendBuf, *colVal;
   nalu_hypre_ParCSRCommPkg *commPkg;

   /*-----------------------------------------------------------------
    * fetch machine and matrix parameters 
    *-----------------------------------------------------------------*/

   A = (nalu_hypre_ParCSRMatrix *) Amat_->getMatrix();
   comm = nalu_hypre_ParCSRMatrixComm(A);
   MPI_Comm_rank(comm,&mypid);  
   MPI_Comm_size(comm,&nprocs);  
   if ( ! useOverlap_ || nprocs <= 1 ) return 0;
   NALU_HYPRE_ParCSRMatrixGetRowPartitioning((NALU_HYPRE_ParCSRMatrix) A, &partition);
   startRow   = partition[mypid];
   endRow     = partition[mypid+1] - 1;
   localNRows = endRow - startRow + 1;
   free( partition );

   /*-----------------------------------------------------------------
    * fetch matrix communication information (offNRows_)
    *-----------------------------------------------------------------*/

   extNRows = localNRows;
   if ( nprocs > 1 && useOverlap_ )
   {
      commPkg    = nalu_hypre_ParCSRMatrixCommPkg(A);
      nSends     = nalu_hypre_ParCSRCommPkgNumSends(commPkg);
      sendProcs  = nalu_hypre_ParCSRCommPkgSendProcs(commPkg);
      sendStarts = nalu_hypre_ParCSRCommPkgSendMapStarts(commPkg);
      nRecvs     = nalu_hypre_ParCSRCommPkgNumRecvs(commPkg);
      recvProcs  = nalu_hypre_ParCSRCommPkgRecvProcs(commPkg);
      recvStarts = nalu_hypre_ParCSRCommPkgRecvVecStarts(commPkg);
      for ( i = 0; i < nRecvs; i++ ) 
         extNRows += ( recvStarts[i+1] - recvStarts[i] );
      requests = new MPI_Request[nRecvs+nSends];
      totalSends  = sendStarts[nSends];
      totalRecvs  = recvStarts[nRecvs];
      if ( totalRecvs > 0 ) offRowLengths_ = new int[totalRecvs];
      else                  offRowLengths_ = NULL;
      recvIndices = nalu_hypre_ParCSRMatrixColMapOffd(A);
      if ( totalRecvs > 0 ) offRowIndices_ = new int[totalRecvs];
      else                  offRowIndices_ = NULL;
      for ( i = 0; i < totalRecvs; i++ ) 
         offRowIndices_[i] = recvIndices[i];
      offNRows_ = totalRecvs;
   }
   else nRecvs = nSends = offNRows_ = totalRecvs = totalSends = 0;

   /*-----------------------------------------------------------------
    * construct offRowLengths 
    *-----------------------------------------------------------------*/

   reqNum = 0;
   for (i = 0; i < nRecvs; i++)
   {
      proc   = recvProcs[i];
      offset = recvStarts[i];
      length = recvStarts[i+1] - offset;
      MPI_Irecv(&(offRowLengths_[offset]), length, MPI_INT, proc, 
                17304, comm, &(requests[reqNum++]));
   }
   if ( totalSends > 0 ) iSendBuf = new int[totalSends];

   index = totalSendNnz = 0;
   for (i = 0; i < nSends; i++)
   {
      proc   = sendProcs[i];
      offset = sendStarts[i];
      limit  = sendStarts[i+1];
      length = limit - offset;
      for (j = offset; j < limit; j++)
      {
         rowNum = nalu_hypre_ParCSRCommPkgSendMapElmt(commPkg,j) + startRow;
         nalu_hypre_ParCSRMatrixGetRow(A,rowNum,&rowSize,&colInd,NULL);
         iSendBuf[index++] = rowSize;
         totalSendNnz += rowSize;
         nalu_hypre_ParCSRMatrixRestoreRow(A,rowNum,&rowSize,&colInd,NULL);
      }
      MPI_Isend(&(iSendBuf[offset]), length, MPI_INT, proc, 17304, comm, 
                &(requests[reqNum++]));
   }
   status = new MPI_Status[reqNum];
   MPI_Waitall( reqNum, requests, status );
   delete [] status;
   if ( totalSends > 0 ) delete [] iSendBuf;

   /*-----------------------------------------------------------------
    * construct offCols 
    *-----------------------------------------------------------------*/

   totalRecvNnz = 0;
   for (i = 0; i < totalRecvs; i++) totalRecvNnz += offRowLengths_[i];
   if ( totalRecvNnz > 0 )
   {
      offCols_ = new int[totalRecvNnz];
      offVals_ = new double[totalRecvNnz];
   }
   reqNum = totalRecvNnz = 0;
   for (i = 0; i < nRecvs; i++)
   {
      proc   = recvProcs[i];
      offset = recvStarts[i];
      length = recvStarts[i+1] - offset;
      curNnz = 0;
      for (j = 0; j < length; j++) curNnz += offRowLengths_[offset+j];
      MPI_Irecv(&(offCols_[totalRecvNnz]), curNnz, MPI_INT, proc, 17305, 
                comm, &(requests[reqNum++]));
      totalRecvNnz += curNnz;
   }
   if ( totalSendNnz > 0 ) iSendBuf = new int[totalSendNnz];

   index = totalSendNnz = 0;
   for (i = 0; i < nSends; i++)
   {
      proc   = sendProcs[i];
      offset = sendStarts[i];
      limit  = sendStarts[i+1];
      length = limit - offset;
      base   = totalSendNnz;
      for (j = offset; j < limit; j++)
      {
         rowNum = nalu_hypre_ParCSRCommPkgSendMapElmt(commPkg,j) + startRow;
         nalu_hypre_ParCSRMatrixGetRow(A,rowNum,&rowSize,&colInd,NULL);
         for (k = 0; k < rowSize; k++) 
            iSendBuf[totalSendNnz++] = colInd[k];
         nalu_hypre_ParCSRMatrixRestoreRow(A,rowNum,&rowSize,&colInd,NULL);
      }
      length = totalSendNnz - base;
      MPI_Isend(&(iSendBuf[base]), length, MPI_INT, proc, 17305, comm, 
                &(requests[reqNum++]));
   }
   status = new MPI_Status[reqNum];
   if ( reqNum > 0 ) MPI_Waitall( reqNum, requests, status );
   delete [] status;
   if ( totalSendNnz > 0 ) delete [] iSendBuf;

   /*-----------------------------------------------------------------
    * construct offVals 
    *-----------------------------------------------------------------*/

   reqNum = totalRecvNnz = 0;
   for (i = 0; i < nRecvs; i++)
   {
      proc    = recvProcs[i];
      offset  = recvStarts[i];
      length  = recvStarts[i+1] - offset;
      curNnz = 0;
      for (j = 0; j < length; j++) curNnz += offRowLengths_[offset+j];
      MPI_Irecv(&(offVals_[totalRecvNnz]), curNnz, MPI_DOUBLE, proc, 
                17306, comm, &(requests[reqNum++]));
      totalRecvNnz += curNnz;
   }
   if ( totalSendNnz > 0 ) dSendBuf = new double[totalSendNnz];

   index = totalSendNnz = 0;
   for (i = 0; i < nSends; i++)
   {
      proc   = sendProcs[i];
      offset = sendStarts[i];
      limit  = sendStarts[i+1];
      length = limit - offset;
      base   = totalSendNnz;
      for (j = offset; j < limit; j++)
      {
         rowNum = nalu_hypre_ParCSRCommPkgSendMapElmt(commPkg,j) + startRow;
         nalu_hypre_ParCSRMatrixGetRow(A,rowNum,&rowSize,NULL,&colVal);
         for (k = 0; k < rowSize; k++) 
            dSendBuf[totalSendNnz++] = colVal[k];
         nalu_hypre_ParCSRMatrixRestoreRow(A,rowNum,&rowSize,NULL,&colVal);
      }
      length = totalSendNnz - base;
      MPI_Isend(&(dSendBuf[base]), length, MPI_DOUBLE, proc, 17306, comm, 
                &(requests[reqNum++]));
   }
   status = new MPI_Status[reqNum];
   if ( reqNum > 0 ) MPI_Waitall( reqNum, requests, status );
   delete [] status;
   if ( totalSendNnz > 0 ) delete [] dSendBuf;

   if ( nprocs > 1 && useOverlap_ ) delete [] requests;

   /*-----------------------------------------------------------------
    * convert column indices
    *-----------------------------------------------------------------*/

   offset = 0;
   for ( i = 0; i < offNRows_; i++ )
   {
      for ( j = 0; j < offRowLengths_[i]; j++ )
      {
         colIndex = offCols_[offset];
         if ( colIndex >= startRow && colIndex <= endRow )
            offCols_[offset] = colIndex - startRow;
         else
         {
            index = MLI_Utils_BinarySearch(colIndex,offRowIndices_,offNRows_);
            if ( index >= 0 ) offCols_[offset] = localNRows + index;
            else              offCols_[offset] = -1;
         }
         offset++;
      }
   }
   return 0;
}

/******************************************************************************
 * build the blocks 
 *--------------------------------------------------------------------------*/

int MLI_Solver_BJacobi::buildBlocks()
{
   int      iB, iP, mypid, nprocs, *partition, startRow, endRow;
   int      localNRows, nRecvs, *recvProcs, *recvStarts, nRecvBefore=0; 
   int      offRowOffset, offRowNnz, blockStartRow, blockEndRow, index;
   int      irow, jcol, colIndex, rowSize, *colInd, localRow, localNnz;
   int      blkLeng, *csrIA, *csrJA, offset, rowIndex;
   double   *colVal, *csrAA, *esslMatrix;
   char     sName[20];
   MPI_Comm comm;
   nalu_hypre_ParCSRCommPkg *commPkg;
   nalu_hypre_ParCSRMatrix  *A = (nalu_hypre_ParCSRMatrix *) Amat_->getMatrix();
   nalu_hypre_CSRMatrix     *seqA;
   MLI_Matrix          *mliMat;
   MLI_Function        *funcPtr=NULL;

   /*-----------------------------------------------------------------
    * fetch matrix information 
    *-----------------------------------------------------------------*/

   comm = nalu_hypre_ParCSRMatrixComm(A);
   MPI_Comm_rank(comm,&mypid);  
   MPI_Comm_size(comm,&nprocs);  
   NALU_HYPRE_ParCSRMatrixGetRowPartitioning((NALU_HYPRE_ParCSRMatrix) A, &partition);
   startRow   = partition[mypid];
   endRow     = partition[mypid+1] - 1;
   localNRows = endRow - startRow + 1;
   free( partition );
   if ( nprocs > 1 && useOverlap_ )
   {
      commPkg     = nalu_hypre_ParCSRMatrixCommPkg(A);
      nRecvs      = nalu_hypre_ParCSRCommPkgNumRecvs(commPkg);
      recvProcs   = nalu_hypre_ParCSRCommPkgRecvProcs(commPkg);
      recvStarts  = nalu_hypre_ParCSRCommPkgRecvVecStarts(commPkg);
      nRecvBefore = 0;
      for ( iP = 0; iP < nRecvs; iP++ )
         if ( recvProcs[iP] > mypid ) break;
      nRecvBefore = recvStarts[iP];
   }
   else nRecvs = 0;

   /*-----------------------------------------------------------------
    * compute block information (blockLengths_) 
    *-----------------------------------------------------------------*/

   nBlocks_ = ( localNRows + blockSize_ - 1 + offNRows_ ) / blockSize_;
   if ( nBlocks_ == 0 ) nBlocks_ = 1;
   blockLengths_ = new int[nBlocks_];
   for ( iB = 0; iB < nBlocks_; iB++ ) blockLengths_[iB] = blockSize_;
   blockLengths_[nBlocks_-1] = localNRows+offNRows_-blockSize_*(nBlocks_-1);
   maxBlkLeng_ = 0;
   for ( iB = 0; iB < nBlocks_; iB++ ) 
      maxBlkLeng_ = ( blockLengths_[iB] > maxBlkLeng_ ) ? 
                      blockLengths_[iB] : maxBlkLeng_;

   /*-----------------------------------------------------------------
    * construct block matrix inverses
    *-----------------------------------------------------------------*/

   if ( blkScheme_ == 0 )
   {
      strcpy( sName, "SeqSuperLU" );
      blockSolvers_ = new MLI_Solver_SeqSuperLU*[nBlocks_];
      for ( iB = 0; iB < nBlocks_; iB++ ) 
         blockSolvers_[iB] = new MLI_Solver_SeqSuperLU(sName);
      funcPtr = nalu_hypre_TAlloc(MLI_Function, 1, NALU_HYPRE_MEMORY_HOST);
   }
   else
   {
      esslMatrices_ = new double*[nBlocks_];
      for ( iB = 0; iB < nBlocks_; iB++ ) esslMatrices_[iB] = NULL;
   }

   offRowOffset = offRowNnz = 0;

   for ( iB = 0; iB < nBlocks_; iB++ )
   {
      blkLeng       = blockLengths_[iB];
      blockStartRow = iB * blockSize_ + startRow - nRecvBefore;
      blockEndRow   = blockStartRow + blkLeng - 1;
      localNnz      = 0;
      if ( blkScheme_ == 0 )
      {
         for ( irow = blockStartRow; irow <= blockEndRow; irow++ )
         {
            if ( irow >= startRow && irow <= endRow )
            {
               nalu_hypre_ParCSRMatrixGetRow(A, irow, &rowSize, &colInd, &colVal);
               localNnz += rowSize;
               nalu_hypre_ParCSRMatrixRestoreRow(A,irow,&rowSize,&colInd,&colVal);
            }
            else localNnz += offRowLengths_[offRowOffset+irow-blockStartRow];
         }
         seqA = nalu_hypre_CSRMatrixCreate( blkLeng, blkLeng, localNnz );
         csrIA = new int[blkLeng+1];
         csrJA = new int[localNnz];
         csrAA = new double[localNnz];
         localRow = 0;
         localNnz = 0;
         csrIA[0] = localNnz;

         for ( irow = blockStartRow; irow <= blockEndRow; irow++ )
         {
            if ( irow >= startRow && irow <= endRow )
            {
               nalu_hypre_ParCSRMatrixGetRow(A, irow, &rowSize, &colInd, &colVal);
               for ( jcol = 0; jcol < rowSize; jcol++ )
               {
                  colIndex = colInd[jcol];
                  if ((colIndex >= blockStartRow) && (colIndex <= blockEndRow))
                  {
                     csrJA[localNnz] = colIndex - blockStartRow; 
                     csrAA[localNnz++] = colVal[jcol];
                  }
               }
               nalu_hypre_ParCSRMatrixRestoreRow(A,irow,&rowSize,&colInd,&colVal);
            }
            else
            {
               rowSize = offRowLengths_[offRowOffset];
               colInd = &(offCols_[offRowNnz]);
               colVal = &(offVals_[offRowNnz]);
               for ( jcol = 0; jcol < rowSize; jcol++ )
               {
                  colIndex = colInd[jcol];
                  if ((colIndex >= blockStartRow) && (colIndex <= blockEndRow))
                  {
                     csrJA[localNnz] = colIndex - blockStartRow; 
                     csrAA[localNnz++] = colVal[jcol];
                  }
               }
               offRowOffset++;
               offRowNnz += rowSize;
            }
            localRow++;
            csrIA[localRow] = localNnz;
         }
         nalu_hypre_CSRMatrixI(seqA)    = csrIA;
         nalu_hypre_CSRMatrixJ(seqA)    = csrJA;
         nalu_hypre_CSRMatrixData(seqA) = csrAA;
         MLI_Utils_HypreCSRMatrixGetDestroyFunc(funcPtr);
         strcpy( sName, "NALU_HYPRE_CSR" );
         mliMat = new MLI_Matrix((void*) seqA, sName, funcPtr);
         blockSolvers_[iB]->setup( mliMat );
         delete mliMat;
      }
      else 
      {
         esslMatrices_[iB] = new double[blkLeng * (blkLeng+1)/2];
         esslMatrix = esslMatrices_[iB];
         for ( irow = 0; irow < blkLeng*(blkLeng+1)/2; irow++ )
            esslMatrix[irow] = 0.0;
         offset = 0;
         for ( irow = blockStartRow; irow <= blockEndRow; irow++ )
         {
            rowIndex = irow - blockStartRow;
            if ( irow >= startRow && irow <= endRow )
            {
               nalu_hypre_ParCSRMatrixGetRow(A, irow, &rowSize, &colInd, &colVal);
               for ( jcol = 0; jcol < rowSize; jcol++ )
               {
                  colIndex = colInd[jcol] - blockStartRow;
                  if ((colIndex >= rowIndex) && (colIndex < blkLeng))
                  {
                     index = colIndex - rowIndex;
                     esslMatrix[offset+index] = colVal[jcol];
                  }
               }
               nalu_hypre_ParCSRMatrixRestoreRow(A,irow,&rowSize,&colInd,&colVal);
            }
            else
            {
               rowSize = offRowLengths_[offRowOffset];
               colInd = &(offCols_[offRowNnz]);
               colVal = &(offVals_[offRowNnz]);
               for ( jcol = 0; jcol < rowSize; jcol++ )
               {
                  colIndex = colInd[jcol] - blockStartRow;
                  if ((colIndex >= rowIndex) && (colIndex < blkLeng))
                  {
                     index = colIndex - rowIndex;
                     esslMatrix[offset+index] = colVal[jcol];
                  }
               }
               offRowOffset++;
               offRowNnz += rowSize;
            }
            offset += blkLeng - irow + blockStartRow;
         }
#ifdef HAVE_ESSL
         dppf(esslMatrix, blkLeng, 1);
#endif
      }
   }

   if ( funcPtr != NULL ) free( funcPtr );
   return 0;
}

/******************************************************************************
 * adjust the off processor incoming matrix
 *--------------------------------------------------------------------------*/

int MLI_Solver_BJacobi::adjustOffColIndices()
{
   int                mypid, *partition, startRow, endRow, localNRows;
   int                offset, index, colIndex, irow, jcol;
   nalu_hypre_ParCSRMatrix *A;
   MPI_Comm           comm;

   /*-----------------------------------------------------------------
    * fetch machine and matrix parameters 
    *-----------------------------------------------------------------*/

   A = (nalu_hypre_ParCSRMatrix *) Amat_->getMatrix();
   comm = nalu_hypre_ParCSRMatrixComm(A);
   MPI_Comm_rank(comm,&mypid);  
   NALU_HYPRE_ParCSRMatrixGetRowPartitioning((NALU_HYPRE_ParCSRMatrix) A, &partition);
   startRow   = partition[mypid];
   endRow     = partition[mypid+1] - 1;
   localNRows = endRow - startRow + 1;
   free( partition );

   /*-----------------------------------------------------------------
    * convert column indices
    *-----------------------------------------------------------------*/

   offset = 0;
   for ( irow = 0; irow < offNRows_; irow++ )
   {
      for ( jcol = 0; jcol < offRowLengths_[irow]; jcol++ )
      {
         colIndex = offCols_[offset];
         if ( colIndex >= startRow && colIndex <= endRow )
            offCols_[offset] = colIndex - startRow;
         else
         {
            index = MLI_Utils_BinarySearch(colIndex,offRowIndices_,offNRows_);
            if ( index >= 0 ) offCols_[offset] = localNRows + index;
            else              offCols_[offset] = -1;
         }
         offset++;
      }
   }
   return 0;
}

/******************************************************************************
 * clean blocks
 *--------------------------------------------------------------------------*/

int MLI_Solver_BJacobi::cleanBlocks()
{
   int iB;

   if ( blockSolvers_ != NULL ) 
   {
      for ( iB = 0; iB < nBlocks_; iB++ ) delete blockSolvers_[iB];
      delete blockSolvers_;
   }
   if ( blockLengths_  != NULL ) delete [] blockLengths_;
   if ( offRowIndices_ != NULL ) delete [] offRowIndices_;
   if ( offRowLengths_ != NULL ) delete [] offRowLengths_;
   if ( offCols_       != NULL ) delete [] offCols_;
   if ( offVals_       != NULL ) delete [] offVals_;
   nBlocks_       = 0; 
   blockLengths_  = NULL;
   blockSolvers_  = NULL;
   offNRows_      = 0;
   offRowIndices_ = NULL;
   offRowLengths_ = NULL;
   offCols_       = NULL;
   offVals_       = NULL;
   if ( esslMatrices_ != NULL )
   {
      for ( iB = 0; iB < nBlocks_; iB++ )
      if ( esslMatrices_[iB] != NULL ) delete [] esslMatrices_[iB];
      delete [] esslMatrices_;
      esslMatrices_ = NULL;      
   }
   return 0;
}

