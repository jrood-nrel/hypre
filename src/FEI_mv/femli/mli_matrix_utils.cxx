/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include <string.h>
#include <stdio.h>
#include "NALU_HYPRE.h"
#include "_nalu_hypre_utilities.h"
#include "_nalu_hypre_parcsr_mv.h"
#include "NALU_HYPRE_IJ_mv.h"
#include "mli_matrix.h"
#include "mli_utils.h"

extern "C" {
   void nalu_hypre_qsort0(int *, int, int);
}

/***************************************************************************
 * compute triple matrix product function 
 *--------------------------------------------------------------------------*/

int MLI_Matrix_ComputePtAP(MLI_Matrix *Pmat, MLI_Matrix *Amat, 
                           MLI_Matrix **RAPmat_out)
{
   int          ierr;
   char         paramString[200];
   void         *Pmat2, *Amat2, *RAPmat2;
   MLI_Matrix   *RAPmat;
   MLI_Function *funcPtr;

   if ( strcmp(Pmat->getName(),"NALU_HYPRE_ParCSR") || 
        strcmp(Amat->getName(),"NALU_HYPRE_ParCSR") )
   {
      printf("MLI_Matrix_computePtAP ERROR - matrix has invalid type.\n");
      exit(1);
   }
   Pmat2 = (void *) Pmat->getMatrix();
   Amat2 = (void *) Amat->getMatrix();
   ierr = MLI_Utils_HypreMatrixComputeRAP(Pmat2,Amat2,&RAPmat2);
   if ( ierr ) printf("ERROR in MLI_Matrix_ComputePtAP\n");
   sprintf(paramString, "NALU_HYPRE_ParCSR");
   funcPtr = new MLI_Function();
   MLI_Utils_HypreParCSRMatrixGetDestroyFunc(funcPtr);
   RAPmat = new MLI_Matrix(RAPmat2,paramString,funcPtr);
   delete funcPtr;
   (*RAPmat_out) = RAPmat;
   return 0;
}

/***************************************************************************
 * compute triple matrix product function 
 *--------------------------------------------------------------------------*/

int MLI_Matrix_FormJacobi(MLI_Matrix *Amat, double alpha, MLI_Matrix **Jmat)
{
   int          ierr;
   char         paramString[200];
   void         *A, *J;
   MLI_Function *funcPtr;
   
   if ( strcmp(Amat->getName(),"NALU_HYPRE_ParCSR") ) 
   {
      printf("MLI_Matrix_FormJacobi ERROR - matrix has invalid type.\n");
      exit(1);
   }
   A = (void *) Amat->getMatrix();;
   ierr = MLI_Utils_HypreMatrixFormJacobi(A, alpha, &J);
   if ( ierr ) printf("ERROR in MLI_Matrix_FormJacobi\n");
   sprintf(paramString, "NALU_HYPRE_ParCSR");
   funcPtr = new MLI_Function();
   MLI_Utils_HypreParCSRMatrixGetDestroyFunc(funcPtr);
   (*Jmat) = new MLI_Matrix(J,paramString,funcPtr);
   delete funcPtr;
   return ierr;
}

/***************************************************************************
 * compress matrix by block size > 1
 *--------------------------------------------------------------------------*/

int MLI_Matrix_Compress(MLI_Matrix *Amat, int blksize, MLI_Matrix **Amat2)
{
   int          ierr;
   char         paramString[200];
   void         *A, *A2;
   MLI_Function *funcPtr;
   
   if ( strcmp(Amat->getName(),"NALU_HYPRE_ParCSR") ) 
   {
      printf("MLI_Matrix_Compress ERROR - matrix has invalid type.\n");
      exit(1);
   }
   if ( blksize <= 1 )
   {
      printf("MLI_Matrix_Compress WARNING - blksize <= 1.\n");
      (*Amat2) = NULL;
      return 1;
   }
   A = (void *) Amat->getMatrix();;
   ierr = MLI_Utils_HypreMatrixCompress(A, blksize, &A2);
   if ( ierr ) printf("ERROR in MLI_Matrix_Compress\n");
   sprintf(paramString, "NALU_HYPRE_ParCSR");
   funcPtr = new MLI_Function();
   MLI_Utils_HypreParCSRMatrixGetDestroyFunc(funcPtr);
   (*Amat2) = new MLI_Matrix(A2,paramString,funcPtr);
   delete funcPtr;
   return ierr;
}

/***************************************************************************
 * get submatrix given row indices
 *--------------------------------------------------------------------------*/

int MLI_Matrix_GetSubMatrix(MLI_Matrix *A_in, int nRows, int *rowIndices,
                            int *newNRows, double **newAA)
{
   int        mypid, nprocs, *partition, startRow, endRow;
   int        i, j, myNRows, irow, rowInd, rowLeng, *cols, *myRowIndices;
   double     *AA, *vals;
   nalu_hypre_ParCSRMatrix *A;
   MPI_Comm           comm;

   /*-----------------------------------------------------------------
    * fetch machine and matrix parameters (off_offset)
    *-----------------------------------------------------------------*/

   A = (nalu_hypre_ParCSRMatrix *) A_in;
   comm = nalu_hypre_ParCSRMatrixComm(A);
   MPI_Comm_rank(comm, &mypid);  
   MPI_Comm_size(comm, &nprocs);  
   NALU_HYPRE_ParCSRMatrixGetRowPartitioning((NALU_HYPRE_ParCSRMatrix) A, &partition);
   startRow = partition[mypid];
   endRow   = partition[mypid+1] - 1;
   free( partition );

   myNRows = 0;
   for ( irow = 0; irow < nRows; irow++ )
   {
      rowInd = rowIndices[irow];
      if ( rowInd >= startRow && rowInd < endRow )
      {
         nalu_hypre_ParCSRMatrixGetRow(A,rowInd,&rowLeng,&cols,NULL);
         myNRows += rowLeng;
         nalu_hypre_ParCSRMatrixRestoreRow(A,rowInd,&rowLeng,&cols,NULL);
      }
   }

   myRowIndices = new int[myNRows]; 
   myNRows = 0;
   for ( irow = 0; irow < nRows; irow++ )
   {
      rowInd = rowIndices[irow];
      if ( rowInd >= startRow && rowInd < endRow )
      {
         nalu_hypre_ParCSRMatrixGetRow(A,rowInd,&rowLeng,&cols,NULL);
         for ( i = 0; i < rowLeng; i++ )
            myRowIndices[myNRows++] = cols[i];
         nalu_hypre_ParCSRMatrixRestoreRow(A,rowInd,&rowLeng,&cols,NULL);
      }
   }

   nalu_hypre_qsort0(myRowIndices, 0, myNRows-1);
   j = 1;
   for ( i = 1; i < myNRows; i++ )
      if ( myRowIndices[i] != myRowIndices[j-1] ) 
         myRowIndices[j++] = myRowIndices[i]; 
   myNRows = j;

   AA = new double[myNRows*myNRows];
   for ( irow = 0; irow < myNRows*myNRows; irow++ ) AA[i] = 0.0;

   for ( irow = 0; irow < myNRows; irow++ )
   {
      rowInd = myRowIndices[irow];
      if ( rowInd >= startRow && rowInd < endRow )
      {
         nalu_hypre_ParCSRMatrixGetRow(A,rowInd,&rowLeng,&cols,&vals);
         for ( i = 0; i < rowLeng; i++ )
            AA[(cols[i]-startRow)*myNRows+irow] = vals[i]; 
         nalu_hypre_ParCSRMatrixRestoreRow(A,rowInd,&rowLeng,&cols,&vals);
      }
   }

   (*newAA) = AA;
   (*newNRows) = myNRows;
   return 0;
}

/***************************************************************************
 * get submatrix given row indices
 *--------------------------------------------------------------------------*/

int MLI_Matrix_GetOverlappedMatrix(MLI_Matrix *mli_mat, int *offNRows, 
                 int **offRowLengths, int **offCols, double **offVals)
{
   int         i, j, k, mypid, nprocs, *partition, startRow;
   int         nSends, *sendProcs, nRecvs;
   int         *recvProcs, *recvStarts, proc, offset, length, reqNum; 
   int         totalSendNnz, totalRecvNnz, index, base, totalSends;
   int         totalRecvs, rowNum, rowLength, *colInd, *sendStarts;
   int         limit, *isendBuf, *cols, curNnz, *rowIndices; 
   double      *dsendBuf, *vals, *colVal;
   nalu_hypre_ParCSRMatrix  *A;
   MPI_Comm            comm;
   MPI_Request         *requests;
   MPI_Status          *status;
   nalu_hypre_ParCSRCommPkg *commPkg;

   /*-----------------------------------------------------------------
    * fetch machine and matrix parameters (off_offset)
    *-----------------------------------------------------------------*/

   A    = (nalu_hypre_ParCSRMatrix *) mli_mat->getMatrix();
   comm = nalu_hypre_ParCSRMatrixComm(A);
   MPI_Comm_rank(comm,&mypid);  
   MPI_Comm_size(comm,&nprocs);  
   if ( nprocs == 1 )
   {
      (*offNRows) = 0;
      (*offRowLengths) = NULL;
      (*offCols) = NULL;
      (*offVals) = NULL;
      return 0;
   }
   NALU_HYPRE_ParCSRMatrixGetRowPartitioning((NALU_HYPRE_ParCSRMatrix) A, &partition);
   startRow   = partition[mypid];
   nalu_hypre_TFree( partition , NALU_HYPRE_MEMORY_HOST);

   /*-----------------------------------------------------------------
    * fetch matrix communication information (off_nrows)
    *-----------------------------------------------------------------*/

   nalu_hypre_MatvecCommPkgCreate((nalu_hypre_ParCSRMatrix *) A);
   commPkg    = nalu_hypre_ParCSRMatrixCommPkg(A);
   nSends     = nalu_hypre_ParCSRCommPkgNumSends(commPkg);
   sendProcs  = nalu_hypre_ParCSRCommPkgSendProcs(commPkg);
   sendStarts = nalu_hypre_ParCSRCommPkgSendMapStarts(commPkg);
   nRecvs     = nalu_hypre_ParCSRCommPkgNumRecvs(commPkg);
   recvProcs  = nalu_hypre_ParCSRCommPkgRecvProcs(commPkg);
   recvStarts = nalu_hypre_ParCSRCommPkgRecvVecStarts(commPkg);
   requests = nalu_hypre_CTAlloc( MPI_Request, nRecvs+nSends , NALU_HYPRE_MEMORY_HOST);
   totalSends  = sendStarts[nSends];
   totalRecvs  = recvStarts[nRecvs];
   (*offNRows) = totalRecvs;

   /*-----------------------------------------------------------------
    * construct offRowLengths 
    *-----------------------------------------------------------------*/

   if ( totalRecvs > 0 ) (*offRowLengths) = new int[totalRecvs];
   else                  (*offRowLengths) = NULL;
   reqNum = 0;
   for (i = 0; i < nRecvs; i++)
   {
      proc   = recvProcs[i];
      offset = recvStarts[i];
      length = recvStarts[i+1] - offset;
      MPI_Irecv(&((*offRowLengths)[offset]),length,MPI_INT,proc,13278,comm, 
                &requests[reqNum++]);
   }
   if ( totalSends > 0 ) isendBuf = nalu_hypre_CTAlloc( int, totalSends , NALU_HYPRE_MEMORY_HOST);
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
         nalu_hypre_ParCSRMatrixGetRow(A,rowNum,&rowLength,&colInd,NULL);
         isendBuf[index++] = rowLength;
         totalSendNnz += rowLength;
         nalu_hypre_ParCSRMatrixRestoreRow(A,rowNum,&rowLength,&colInd,NULL);
      }
      MPI_Isend(&isendBuf[offset], length, MPI_INT, proc, 13278, comm, 
                &requests[reqNum++]);
   }
   status = nalu_hypre_CTAlloc(MPI_Status, reqNum, NALU_HYPRE_MEMORY_HOST);
   MPI_Waitall( reqNum, requests, status );
   nalu_hypre_TFree( status , NALU_HYPRE_MEMORY_HOST);
   if ( totalSends > 0 ) nalu_hypre_TFree( isendBuf , NALU_HYPRE_MEMORY_HOST);

   /*-----------------------------------------------------------------
    * construct row indices 
    *-----------------------------------------------------------------*/

   if ( totalRecvs > 0 ) rowIndices = new int[totalRecvs];
   else                  rowIndices = NULL;
   reqNum = 0;
   for (i = 0; i < nRecvs; i++)
   {
      proc   = recvProcs[i];
      offset = recvStarts[i];
      length = recvStarts[i+1] - offset;
      MPI_Irecv(&(rowIndices[offset]), length, MPI_INT, proc, 13279, comm, 
                &requests[reqNum++]);
   }
   if ( totalSends > 0 ) isendBuf = nalu_hypre_CTAlloc( int, totalSends , NALU_HYPRE_MEMORY_HOST);
   index = 0;
   for (i = 0; i < nSends; i++)
   {
      proc   = sendProcs[i];
      offset = sendStarts[i];
      limit  = sendStarts[i+1];
      length = limit - offset;
      for (j = offset; j < limit; j++)
      {
         rowNum = nalu_hypre_ParCSRCommPkgSendMapElmt(commPkg,j) + startRow;
         isendBuf[index++] = rowNum;
      }
      MPI_Isend(&isendBuf[offset], length, MPI_INT, proc, 13279, comm, 
                &requests[reqNum++]);
   }
   status = nalu_hypre_CTAlloc(MPI_Status, reqNum, NALU_HYPRE_MEMORY_HOST);
   MPI_Waitall( reqNum, requests, status );
   nalu_hypre_TFree( status , NALU_HYPRE_MEMORY_HOST);
   if ( totalSends > 0 ) nalu_hypre_TFree( isendBuf , NALU_HYPRE_MEMORY_HOST);

   /*-----------------------------------------------------------------
    * construct offCols 
    *-----------------------------------------------------------------*/

   totalRecvNnz = 0;
   for (i = 0; i < totalRecvs; i++) totalRecvNnz += (*offRowLengths)[i];
   if ( totalRecvNnz > 0 )
   {
      cols = new int[totalRecvNnz];
      vals = new double[totalRecvNnz];
   }
   reqNum = totalRecvNnz = 0;
   for (i = 0; i < nRecvs; i++)
   {
      proc    = recvProcs[i];
      offset  = recvStarts[i];
      length  = recvStarts[i+1] - offset;
      curNnz = 0;
      for (j = 0; j < length; j++) curNnz += (*offRowLengths)[offset+j];
      MPI_Irecv(&cols[totalRecvNnz], curNnz, MPI_INT, proc, 13280, comm, 
                &requests[reqNum++]);
      totalRecvNnz += curNnz;
   }
   if ( totalSendNnz > 0 ) isendBuf = nalu_hypre_CTAlloc( int, totalSendNnz , NALU_HYPRE_MEMORY_HOST);
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
         nalu_hypre_ParCSRMatrixGetRow(A,rowNum,&rowLength,&colInd,NULL);
         for (k = 0; k < rowLength; k++) 
            isendBuf[totalSendNnz++] = colInd[k];
         nalu_hypre_ParCSRMatrixRestoreRow(A,rowNum,&rowLength,&colInd,NULL);
      }
      length = totalSendNnz - base;
      MPI_Isend(&isendBuf[base], length, MPI_INT, proc, 13280, comm, 
                &requests[reqNum++]);
   }
   status = nalu_hypre_CTAlloc(MPI_Status, reqNum, NALU_HYPRE_MEMORY_HOST);
   MPI_Waitall( reqNum, requests, status );
   nalu_hypre_TFree( status , NALU_HYPRE_MEMORY_HOST);
   if ( totalSendNnz > 0 ) nalu_hypre_TFree( isendBuf , NALU_HYPRE_MEMORY_HOST);

   /*-----------------------------------------------------------------
    * construct offVals 
    *-----------------------------------------------------------------*/

   reqNum = totalRecvNnz = 0;
   for (i = 0; i < nRecvs; i++)
   {
      proc   = recvProcs[i];
      offset = recvStarts[i];
      length = recvStarts[i+1] - offset;
      curNnz = 0;
      for (j = 0; j < length; j++) curNnz += (*offRowLengths)[offset+j];
      MPI_Irecv(&vals[totalRecvNnz], curNnz, MPI_DOUBLE, proc, 13281, comm, 
                &requests[reqNum++]);
      totalRecvNnz += curNnz;
   }
   if ( totalSendNnz > 0 ) dsendBuf = nalu_hypre_CTAlloc( double, totalSendNnz , NALU_HYPRE_MEMORY_HOST);
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
         nalu_hypre_ParCSRMatrixGetRow(A,rowNum,&rowLength,NULL,&colVal);
         for (k = 0; k < rowLength; k++) 
            dsendBuf[totalSendNnz++] = colVal[k];
         nalu_hypre_ParCSRMatrixRestoreRow(A,rowNum,&rowLength,NULL,&colVal);
      }
      length = totalSendNnz - base;
      MPI_Isend(&dsendBuf[base], length, MPI_DOUBLE, proc, 13281, comm, 
                &requests[reqNum++]);
   }
   status = nalu_hypre_CTAlloc(MPI_Status, reqNum, NALU_HYPRE_MEMORY_HOST);
   MPI_Waitall( reqNum, requests, status );
   nalu_hypre_TFree( status , NALU_HYPRE_MEMORY_HOST);
   if ( totalSendNnz > 0 ) nalu_hypre_TFree( dsendBuf , NALU_HYPRE_MEMORY_HOST);

   if ( nSends+nRecvs > 0 ) nalu_hypre_TFree( requests , NALU_HYPRE_MEMORY_HOST);

   (*offCols) = cols;
   (*offVals) = vals;
   return 0;
}

/***************************************************************************
 * perform matrix transpose (modified from parcsr_mv function by putting
 * diagonal entries at the beginning of the row)
 *--------------------------------------------------------------------------*/

void MLI_Matrix_Transpose(MLI_Matrix *Amat, MLI_Matrix **AmatT)
{
   int                one=1, ia, ia2, ib, iTemp, *ATDiagI, *ATDiagJ;
   int                localNRows;
   double             dTemp, *ATDiagA;
   char               paramString[30];
   nalu_hypre_CSRMatrix    *ATDiag;
   nalu_hypre_ParCSRMatrix *hypreA, *hypreAT;
   MLI_Matrix         *mli_AmatT;
   MLI_Function       *funcPtr;

   hypreA = (nalu_hypre_ParCSRMatrix *) Amat->getMatrix();
   nalu_hypre_ParCSRMatrixTranspose( hypreA, &hypreAT, one );
   ATDiag = nalu_hypre_ParCSRMatrixDiag(hypreAT);
   localNRows = nalu_hypre_CSRMatrixNumRows(ATDiag);
   ATDiagI = nalu_hypre_CSRMatrixI(ATDiag);
   ATDiagJ = nalu_hypre_CSRMatrixJ(ATDiag);
   ATDiagA = nalu_hypre_CSRMatrixData(ATDiag);

   /* -----------------------------------------------------------------------
    * move the diagonal entry to the beginning of the row
    * ----------------------------------------------------------------------*/

   for ( ia = 0; ia < localNRows; ia++ ) 
   {
      iTemp = -1;
      for ( ia2 = ATDiagI[ia]; ia2 < ATDiagI[ia+1]; ia2++ ) 
      {
         if ( ATDiagJ[ia2] == ia ) 
         {
            iTemp = ATDiagJ[ia2];
            dTemp = ATDiagA[ia2];
            break;
         }
      }
      if ( iTemp >= 0 )
      {
         for ( ib = ia2; ib > ATDiagI[ia]; ib-- ) 
         {
            ATDiagJ[ib] = ATDiagJ[ib-1];
            ATDiagA[ib] = ATDiagA[ib-1];
         }
         ATDiagJ[ATDiagI[ia]] = iTemp;
         ATDiagA[ATDiagI[ia]] = dTemp;
      }  
   }  

   /* -----------------------------------------------------------------------
    * construct MLI_Matrix
    * ----------------------------------------------------------------------*/

   sprintf( paramString, "NALU_HYPRE_ParCSRMatrix" );
   funcPtr = new MLI_Function();
   MLI_Utils_HypreParCSRMatrixGetDestroyFunc(funcPtr); 
   mli_AmatT = new MLI_Matrix((void*) hypreAT, paramString, funcPtr);
   delete funcPtr;

   *AmatT = mli_AmatT;
}
