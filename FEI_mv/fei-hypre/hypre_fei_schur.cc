/*BHEADER**********************************************************************
 * (c) 1999   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 *********************************************************************EHEADER*/

#include <stdlib.h>
#include <string.h>
#include <iostream.h>
#include <stdio.h>
#include <assert.h>

#include "utilities/utilities.h"
#include "src/Data.h"
#include "other/basicTypes.h"
#include "src/Utils.h"
#include "src/LinearSystemCore.h"
#include "HYPRE_LinSysCore.h"

#define abs(x) (((x) > 0.0) ? x : -(x))

//---------------------------------------------------------------------------
// parcsr_matrix_vector.h is put here instead of in HYPRE_LinSysCore.h 
// because it gives warning when compiling cfei.cc
//---------------------------------------------------------------------------

#include "parcsr_matrix_vector/parcsr_matrix_vector.h"

//---------------------------------------------------------------------------
// These are external functions needed internally here
//---------------------------------------------------------------------------

extern "C" {
   int hypre_ParAMGBuildCoarseOperator(hypre_ParCSRMatrix*,
                                       hypre_ParCSRMatrix*,
                                       hypre_ParCSRMatrix*,
                                       hypre_ParCSRMatrix**);
}

//******************************************************************************
// Given the matrix (A) within the object, compute the reduced system and put
// it in place.  Additional information given are :
//------------------------------------------------------------------------------

void HYPRE_LinSysCore::buildSchurSystem()
{
    int    i, j, k, ierr, ncnt, ncnt2;
    int    nRows, globalNRows, StartRow, EndRow, colIndex;
    int    nSchur, *schurList, globalNSchur, *globalSchurList;
    int    CStartRow, CNRows, CNCols, CGlobalNRows, CGlobalNCols;
    int    CTStartRow, CTNRows, CTNCols, CTGlobalNRows, CTGlobalNCols;
    int    MStartRow, MNRows, MNCols, MGlobalNRows, MGlobalNCols;
    int    rowSize, rowCount, rowIndex, maxRowSize, newRowSize;
    int    *CMatSize, *CTMatSize, *MMatSize, *colInd, *newColInd;
    int    *tempList, *recvCntArray, *displArray;
    int    procIndex, *ProcNRows, *ProcNSchur, searchIndex;
    double *colVal, *newColVal, *diagonal, ddata;

    HYPRE_IJMatrix     Cmat, CTmat, Mmat, Smat;
    HYPRE_ParCSRMatrix A_csr, C_csr, CT_csr, M_csr, S_csr;
    HYPRE_IJVector     f1, f2, f2hat;
    HYPRE_ParVector    f1_csr, f2_csr, f2hat_csr;

    //******************************************************************
    // initial set up 
    //------------------------------------------------------------------

    if ( mypid_ == 0 ) printf("%4d buildSchurSystem activated.\n",mypid_);
    StartRow = localStartRow_ - 1;
    EndRow   = localEndRow_ - 1;
    nRows    = localEndRow_ - localStartRow_ + 1;
    A_csr    = (HYPRE_ParCSRMatrix) HYPRE_IJMatrixGetLocalStorage(HYA_);
    if ( HYOutputLevel_ > 0 )
    {
       printf("%4d buildSchurSystem : StartRow/EndRow = %d %d\n",mypid_,
                                         StartRow,EndRow);
    }

    //******************************************************************
    // construct local and global information about where the constraints
    // are (this is given by user or searched within this code)
    //------------------------------------------------------------------

    //------------------------------------------------------------------
    // get information about processor offsets and globalNRows
    // (ProcNRows, globalNRows)
    //------------------------------------------------------------------
 
    ProcNRows   = new int[numProcs_];
    tempList    = new int[numProcs_];
    for ( i = 0; i < numProcs_; i++ ) tempList[i] = 0;
    tempList[mypid_] = StartRow;
    MPI_Allreduce(tempList, ProcNRows, numProcs_, MPI_INT, MPI_SUM, comm_);
    delete [] tempList;
    MPI_Allreduce(&nRows, &globalNRows,1,MPI_INT,MPI_SUM,comm_);

    //******************************************************************
    // compose the local and global Schur node lists
    //------------------------------------------------------------------

    //------------------------------------------------------------------
    // count the rows that have zero diagonals
    //------------------------------------------------------------------

    nSchur = 0;
    for ( i = StartRow; i <= EndRow; i++ ) 
    {
       HYPRE_ParCSRMatrixGetRow(A_csr,i,&rowSize,&colInd,&colVal);
       for (j = 0; j < rowSize; j++) 
          if ( colInd[j] == i && colVal[j] == 0.0 ) nSchur++;
       HYPRE_ParCSRMatrixRestoreRow(A_csr,i,&rowSize,&colInd,&colVal);
    }

    //------------------------------------------------------------------
    // allocate array for storing indices of selected nodes
    //------------------------------------------------------------------

    if ( nSchur > 0 ) schurList = new int[nSchur];
    else              schurList = NULL; 

    //------------------------------------------------------------------
    // compose the list of rows having zero diagonal
    // (nSchur, schurList)
    //------------------------------------------------------------------

    nSchur = 0;
    for ( i = StartRow; i <= EndRow; i++ ) 
    {
       HYPRE_ParCSRMatrixGetRow(A_csr,i,&rowSize,&colInd,&colVal);
       for (j = 0; j < rowSize; j++) 
          if ( colInd[j] == i && colVal[j] == 0.0 ) schurList[nSchur++] = i;
       HYPRE_ParCSRMatrixRestoreRow(A_csr,i,&rowSize,&colInd,&colVal);
    }

    //------------------------------------------------------------------
    // compose the global list of rows having zero diagonal
    // (globalNSchur, globalSchurList)
    //------------------------------------------------------------------

    MPI_Allreduce(&nSchur, &globalNSchur, 1, MPI_INT, MPI_SUM,comm_);
    if ( globalNSchur > 0 ) globalSchurList = new int[globalNSchur];

    recvCntArray = new int[numProcs_];
    displArray   = new int[numProcs_];
    MPI_Allgather(&nSchur, 1, MPI_INT, recvCntArray, 1, MPI_INT, comm_);
    displArray[0] = 0;
    for ( i = 1; i < numProcs_; i++ ) 
       displArray[i] = displArray[i-1] + recvCntArray[i-1];
    MPI_Allgatherv(schurList, nSchur, MPI_INT, globalSchurList,
                   recvCntArray, displArray, MPI_INT, comm_);
    delete [] recvCntArray;
    delete [] displArray;

    if ( HYOutputLevel_ > 1 )
    {
       for ( i = 0; i < nSchur; i++ )
          printf("%4d buildSchurSystem : schurList %d = %d\n",mypid_,
                 i,schurList[i]);
    }
 
    //------------------------------------------------------------------
    // get information about processor offsets for nSchur
    // (ProcNSchur)
    //------------------------------------------------------------------
 
    ProcNSchur = new int[numProcs_];
    tempList   = new int[numProcs_];
    for ( i = 0; i < numProcs_; i++ ) tempList[i] = 0;
    tempList[mypid_] = nSchur;
    MPI_Allreduce(tempList, ProcNSchur, numProcs_, MPI_INT, MPI_SUM, comm_);
    delete [] tempList;
    globalNSchur = 0;
    ncnt = 0;
    for ( i = 0; i < numProcs_; i++ ) 
    {
       globalNSchur  += ProcNSchur[i];
       ncnt2         = ProcNSchur[i];
       ProcNSchur[i] = ncnt;
       ncnt          += ncnt2;
    }

    //******************************************************************
    // construct Cmat
    //------------------------------------------------------------------

    //------------------------------------------------------------------
    // calculate the dimension of Cmat
    //------------------------------------------------------------------

    CNRows = nSchur;
    CNCols = nRows - nSchur;
    CGlobalNRows = globalNSchur;
    CGlobalNCols = globalNRows - globalNSchur;
    CStartRow    = ProcNSchur[mypid_];

    if ( HYOutputLevel_ > 0 )
    {
       printf("%4d buildSchurSystem : CStartRow  = %d\n",mypid_,CStartRow);
       printf("%4d buildSchurSystem : CGlobalDim = %d %d\n", mypid_, 
                                      CGlobalNRows, CGlobalNCols);
       printf("%4d buildSchurSystem : CLocalDim  = %d %d\n",mypid_,
                                         CNRows, CNCols);
    }

    //------------------------------------------------------------------
    // create a matrix context for Cmat
    //------------------------------------------------------------------

    ierr  = HYPRE_IJMatrixCreate(comm_,&Cmat,CGlobalNRows,CGlobalNCols);
    ierr += HYPRE_IJMatrixSetLocalStorageType(Cmat, HYPRE_PARCSR);
    ierr  = HYPRE_IJMatrixSetLocalSize(Cmat, CNRows, CNCols);
    assert(!ierr);

    //------------------------------------------------------------------
    // compute the number of nonzeros per row in Cmat and call set up
    //------------------------------------------------------------------

    maxRowSize = 0;
    CMatSize = new int[CNRows];

    for ( i = 0; i < nSchur; i++ ) 
    {
       rowIndex = schurList[i];
       HYPRE_ParCSRMatrixGetRow(A_csr,rowIndex,&rowSize,&colInd,&colVal);
       newRowSize = 0;
       for (j = 0; j < rowSize; j++) 
       {
          colIndex = colInd[j];
	  searchIndex = hypre_BinarySearch(globalSchurList,colIndex, 
                                           globalNSchur);
          if (searchIndex < 0) newRowSize++;
          else
             printf("buildSchurSystem WARNING : lower diag block != 0.\n");
       }
       CMatSize[i] = newRowSize;
       maxRowSize = ( newRowSize > maxRowSize ) ? newRowSize : maxRowSize;
       HYPRE_ParCSRMatrixRestoreRow(A_csr,rowIndex,&rowSize,&colInd,&colVal);
    }
    ierr  = HYPRE_IJMatrixSetRowSizes(Cmat, CMatSize);
    ierr += HYPRE_IJMatrixInitialize(Cmat);
    assert(!ierr);
    delete [] CMatSize;

    //------------------------------------------------------------------
    // load Cmat extracted from A
    //------------------------------------------------------------------

    newColInd = new int[maxRowSize+1];
    newColVal = new double[maxRowSize+1];
    rowCount  = CStartRow;
    for ( i = 0; i < nSchur; i++ )
    {
       rowIndex = schurList[i];
       HYPRE_ParCSRMatrixGetRow(A_csr,rowIndex,&rowSize,&colInd,&colVal);
       newRowSize = 0;
       for (j = 0; j < rowSize; j++) 
       {
          if ( colVal[j] != 0.0 )
          {
             colIndex = colInd[j];
	     searchIndex = HYFEI_BinarySearch(globalSchurList,colIndex, 
                                              globalNSchur); 
             if ( searchIndex < 0 ) 
             {
                searchIndex = - searchIndex - 1;
                for ( procIndex = 0; procIndex < numProcs_; procIndex++ )
                   if ( ProcNRows[procIndex] > colIndex ) break;
                procIndex--;
                colIndex = colInd[j]-ProcNSchur[procIndex]-searchIndex;
                newColInd[newRowSize]   = colIndex;
                newColVal[newRowSize++] = colVal[j];
                if ( colIndex < 0 || colIndex >= CGlobalNCols )
                {
                   printf("%4d buildSchurSystem WARNING : Cmat ", mypid_);
                   printf("out of range %d - %d (%d)\n", rowCount, colIndex, 
                           CGlobalNCols);
                } 
                if ( newRowSize > maxRowSize+1 ) 
                {
                   printf("%4d buildSchurSystem : WARNING - ",mypid_);
                   printf("passing array boundary(1).\n");
                }
             }
          } 
       }
       HYPRE_ParCSRMatrixRestoreRow(A_csr,rowIndex,&rowSize,&colInd,&colVal);
       HYPRE_IJMatrixInsertRow(Cmat,newRowSize,rowCount,newColInd,newColVal);
       rowCount++;
    }

    //------------------------------------------------------------------
    // finally assemble the matrix 
    //------------------------------------------------------------------

    HYPRE_IJMatrixAssemble(Cmat);
    C_csr = (HYPRE_ParCSRMatrix) HYPRE_IJMatrixGetLocalStorage(Cmat);
    hypre_MatvecCommPkgCreate((hypre_ParCSRMatrix *) C_csr);

    if ( HYOutputLevel_ > 3 )
    {
       ncnt = 0;
       MPI_Barrier(MPI_COMM_WORLD);
       while ( ncnt < numProcs_ ) 
       {
          if ( mypid_ == ncnt ) 
          {
             printf("====================================================\n");
             printf("%4d buildSchurSystem : matrix Cmat assembled %d.\n",
                                           mypid_,CStartRow);
             fflush(stdout);
             for ( i = CStartRow; i < CStartRow+nSchur; i++ ) 
             {
                HYPRE_ParCSRMatrixGetRow(C_csr,i,&rowSize,&colInd,&colVal);
                printf("Cmat ROW = %6d (%d)\n", i, rowSize);
                for ( j = 0; j < rowSize; j++ )
                   printf("   col = %6d, val = %e \n", colInd[j], colVal[j]);
                HYPRE_ParCSRMatrixRestoreRow(C_csr,i,&rowSize,&colInd,&colVal);
             }
             printf("====================================================\n");
          }
          ncnt++;
          MPI_Barrier(MPI_COMM_WORLD);
       }
    }

    //******************************************************************
    // construct the diagonal Mmat
    //------------------------------------------------------------------

    //------------------------------------------------------------------
    // calculate the dimension of Mmat
    //------------------------------------------------------------------

    MNRows = nRows - nSchur;
    MNCols = nRows - nSchur;
    MGlobalNRows = globalNRows - globalNSchur;
    MGlobalNCols = globalNRows - globalNSchur;
    MStartRow    = ProcNRows[mypid_] - ProcNSchur[mypid_];
    if ( HYOutputLevel_ > 0 )
    {
       printf("%4d buildSchurSystem : MStartRow  = %d\n",mypid_,MStartRow);
       printf("%4d buildSchurSystem : MGlobalDim = %d %d\n", mypid_, 
                                      MGlobalNRows, MGlobalNCols);
       printf("%4d buildSchurSystem : MLocalDim  = %d %d\n",mypid_,
                                      MNRows, MNCols);
    }

    //------------------------------------------------------------------
    // create a matrix context for Mmat
    //------------------------------------------------------------------

    ierr  = HYPRE_IJMatrixCreate(comm_,&Mmat,MGlobalNRows,MGlobalNCols);
    ierr += HYPRE_IJMatrixSetLocalStorageType(Mmat, HYPRE_PARCSR);
    ierr  = HYPRE_IJMatrixSetLocalSize(Mmat, MNRows, MNCols);
    MMatSize = new int[MNRows];
    for ( i = 0; i < MNRows; i++ ) MMatSize[i] = 1;
    ierr  = HYPRE_IJMatrixSetRowSizes(Mmat, MMatSize);
    ierr += HYPRE_IJMatrixInitialize(Mmat);
    assert(!ierr);
    delete [] MMatSize;

    //------------------------------------------------------------------
    // load Mmat
    //------------------------------------------------------------------

    diagonal = new double[MNRows];
    rowIndex = MStartRow;
    for ( i = StartRow; i <= EndRow; i++ ) 
    {
       searchIndex = hypre_BinarySearch(schurList, i, nSchur);
       if ( searchIndex < 0 )
       {
          HYPRE_ParCSRMatrixGetRow(A_csr,i,&rowSize,&colInd,&colVal);
          for (j = 0; j < rowSize; j++) 
          {
             colIndex = colInd[j];
             if ( colIndex == i ) { ddata = 1.0 / colVal[j]; break;}
          }
          diagonal[rowIndex-MStartRow] = ddata;
          HYPRE_ParCSRMatrixRestoreRow(A_csr,i,&rowSize,&colInd,&colVal);
          HYPRE_IJMatrixInsertRow(Mmat,1,rowIndex,&rowIndex,&ddata);
       }
    }

    //------------------------------------------------------------------
    // finally assemble Mmat
    //------------------------------------------------------------------

    HYPRE_IJMatrixAssemble(Mmat);
    M_csr = (HYPRE_ParCSRMatrix) HYPRE_IJMatrixGetLocalStorage(Mmat);

    //******************************************************************
    // construct CTmat (transpose of Cmat)
    //------------------------------------------------------------------

    //------------------------------------------------------------------
    // calculate the dimension of CTmat
    //------------------------------------------------------------------

    CTNRows = CNCols;
    CTNCols = CNRows;
    CTGlobalNRows = CGlobalNCols;
    CTGlobalNCols = CGlobalNRows;
    CTStartRow    = ProcNRows[mypid_] - ProcNSchur[mypid_];

    if ( HYOutputLevel_ > 0 )
    {
       printf("%4d buildSchurSystem : CTStartRow  = %d\n",mypid_,CTStartRow);
       printf("%4d buildSchurSystem : CTGlobalDim = %d %d\n", mypid_, 
                                      CTGlobalNRows, CTGlobalNCols);
       printf("%4d buildSchurSystem : CTLocalDim  = %d %d\n",mypid_,
                                      CTNRows, CTNCols);
    }

    //------------------------------------------------------------------
    // create a matrix context for CTmat
    //------------------------------------------------------------------

    ierr  = HYPRE_IJMatrixCreate(comm_,&CTmat,CTGlobalNRows,CTGlobalNCols);
    ierr += HYPRE_IJMatrixSetLocalStorageType(CTmat, HYPRE_PARCSR);
    ierr  = HYPRE_IJMatrixSetLocalSize(CTmat, CTNRows, CTNCols);
    assert(!ierr);

    //------------------------------------------------------------------
    // compute the number of nonzeros per row in CTmat and call set up
    //------------------------------------------------------------------

    maxRowSize = 0;
    CTMatSize = new int[CTNRows];

    rowCount = 0;
    for ( i = StartRow; i <= EndRow; i++ ) 
    {
       searchIndex = hypre_BinarySearch(schurList, i, nSchur);
       if ( searchIndex < 0 )
       {
          HYPRE_ParCSRMatrixGetRow(A_csr,i,&rowSize,&colInd,&colVal);
          newRowSize = 0;
          for (j = 0; j < rowSize; j++) 
          {
             colIndex = colInd[j];
             searchIndex = hypre_BinarySearch(globalSchurList,colIndex, 
                                              globalNSchur);
             if (searchIndex >= 0) newRowSize++;
          }
          CTMatSize[rowCount++] = newRowSize;
          maxRowSize = ( newRowSize > maxRowSize ) ? newRowSize : maxRowSize;
          HYPRE_ParCSRMatrixRestoreRow(A_csr,i,&rowSize,&colInd,&colVal);
       }
    }
    ierr  = HYPRE_IJMatrixSetRowSizes(CTmat, CTMatSize);
    ierr += HYPRE_IJMatrixInitialize(CTmat);
    assert(!ierr);
    delete [] CTMatSize;

    //------------------------------------------------------------------
    // load CTmat extracted from A
    //------------------------------------------------------------------

    newColInd = new int[maxRowSize+1];
    newColVal = new double[maxRowSize+1];
    rowCount  = CTStartRow;

    for ( i = StartRow; i <= EndRow; i++ ) 
    {
       searchIndex = hypre_BinarySearch(schurList, i, nSchur);
       if ( searchIndex < 0 )
       {
          HYPRE_ParCSRMatrixGetRow(A_csr,i,&rowSize,&colInd,&colVal);
          newRowSize = 0;
          for (j = 0; j < rowSize; j++) 
          {
             colIndex = colInd[j];
             searchIndex = hypre_BinarySearch(globalSchurList,colIndex, 
                                              globalNSchur);
             if (searchIndex >= 0) 
             {
                newColInd[newRowSize] = searchIndex;
                newColVal[newRowSize++] = colVal[j];
             }
          }
          HYPRE_ParCSRMatrixRestoreRow(A_csr,i,&rowSize,&colInd,&colVal);
          HYPRE_IJMatrixInsertRow(Mmat,1,rowCount,newColInd,newColVal);
          rowCount++;
       }
    }

    //------------------------------------------------------------------
    // finally assemble the matrix 
    //------------------------------------------------------------------

    HYPRE_IJMatrixAssemble(CTmat);
    CT_csr = (HYPRE_ParCSRMatrix) HYPRE_IJMatrixGetLocalStorage(CTmat);
    hypre_MatvecCommPkgCreate((hypre_ParCSRMatrix *) CT_csr);

    //******************************************************************
    // perform the triple matrix product
    //------------------------------------------------------------------

    if ( HYOutputLevel_ > 0 )
    {
       printf("%4d buildSchurSystem : Triple matrix product starts\n",mypid_);
    }
    hypre_ParAMGBuildCoarseOperator( (hypre_ParCSRMatrix *) CT_csr,
                                     (hypre_ParCSRMatrix *) M_csr,
                                     (hypre_ParCSRMatrix *) CT_csr,
                                     (hypre_ParCSRMatrix **) &S_csr);
    if ( HYOutputLevel_ > 0 )
    {
       printf("%4d buildSchurSystem : Triple matrix product ends\n",mypid_);
    }

    if ( HYOutputLevel_ > 3 )
    {
       MPI_Barrier(MPI_COMM_WORLD);
       ncnt = 0;
       while ( ncnt < numProcs_ )
       {
          if ( mypid_ == ncnt )
          {
             for ( i = CStartRow; i < CStartRow+CNCols; i++ ) {
                HYPRE_ParCSRMatrixGetRow(S_csr,i,&rowSize,&colInd, &colVal);
                printf("Schur ROW = %6d (%d)\n", i, rowSize);
                for ( j = 0; j < rowSize; j++ )
                   printf("   col = %6d, val = %e \n", colInd[j], colVal[j]);
                HYPRE_ParCSRMatrixRestoreRow(S_csr,i,&rowSize,&colInd,&colVal);
             }
          }
          MPI_Barrier(MPI_COMM_WORLD);
          ncnt++;
       }
    }

    // *****************************************************************
    // form modified right hand side  (f2 = f2 - C*M*f1)
    // *****************************************************************

    // *****************************************************************
    // form f2hat = C*M*f1
    //------------------------------------------------------------------

    HYPRE_IJVectorCreate(comm_, &f1, CTGlobalNRows);
    HYPRE_IJVectorSetLocalStorageType(f1, HYPRE_PARCSR);
    HYPRE_IJVectorSetLocalPartitioning(f1,CTStartRow,CTStartRow+CTNRows);
    ierr += HYPRE_IJVectorAssemble(f1);
    ierr += HYPRE_IJVectorInitialize(f1);
    assert(!ierr);

    HYPRE_IJVectorCreate(comm_, &f2hat, CTGlobalNRows);
    HYPRE_IJVectorSetLocalStorageType(f2hat, HYPRE_PARCSR);
    HYPRE_IJVectorSetLocalPartitioning(f2hat,CTStartRow,CTStartRow+CTNRows);
    ierr += HYPRE_IJVectorAssemble(f2hat);
    ierr += HYPRE_IJVectorInitialize(f2hat);
    ierr += HYPRE_IJVectorZeroLocalComponents(f2hat);
    assert(!ierr);

    rowCount = CTNRows;
    for ( i = StartRow; i <= EndRow; i++ ) 
    {
       searchIndex = hypre_BinarySearch(schurList, i, nSchur);
       if ( searchIndex < 0 )
       {
          HYPRE_IJVectorGetLocalComponents(HYb_, 1, &i, NULL, &ddata);
          ddata *= diagonal[rowCount-CTNRows];
          ierr = HYPRE_IJVectorSetLocalComponents(f1,1,&rowCount,NULL,&ddata);
          assert( !ierr );
          rowCount++;
       }
    } 
        
    f1_csr     = (HYPRE_ParVector) HYPRE_IJVectorGetLocalStorage(f1);
    f2hat_csr  = (HYPRE_ParVector) HYPRE_IJVectorGetLocalStorage(f2hat);
    HYPRE_ParCSRMatrixMatvec( 1.0, C_csr, f1_csr, 0.0, f2hat_csr );
    delete [] diagonal;
    HYPRE_IJVectorDestroy(f1); 

    //------------------------------------------------------------------
    // form f2 = f2 - f2hat 
    //------------------------------------------------------------------

    HYPRE_IJVectorCreate(comm_, &f2, CTGlobalNRows);
    HYPRE_IJVectorSetLocalStorageType(f2, HYPRE_PARCSR);
    HYPRE_IJVectorSetLocalPartitioning(f2,CTStartRow,CTStartRow+CTNRows);
    ierr += HYPRE_IJVectorAssemble(f2);
    ierr += HYPRE_IJVectorInitialize(f2);
    ierr += HYPRE_IJVectorZeroLocalComponents(f2);
    assert(!ierr);

    rowCount = CNRows;
    for ( i = StartRow; i <= EndRow; i++ ) 
    {
       rowIndex = schurList[i];
       HYPRE_IJVectorGetLocalComponents(HYb_, 1, &rowIndex, NULL, &ddata);
       ddata = - ddata;
       ierr = HYPRE_IJVectorSetLocalComponents(f2,1,&rowCount,NULL,&ddata);
       HYPRE_IJVectorGetLocalComponents(f2hat, 1, &rowCount, NULL, &ddata);
       HYPRE_IJVectorAddToLocalComponents(f2,1,&rowCount,NULL,&ddata);
       assert( !ierr );
       rowCount++;
    } 

    //******************************************************************
    // set up the system with the new matrix
    //------------------------------------------------------------------

    reducedA_ = Smat;
    ierr = HYPRE_IJVectorCreate(comm_, &reducedX_, globalNSchur);
    ierr = HYPRE_IJVectorSetLocalStorageType(reducedX_, HYPRE_PARCSR);
    ierr = HYPRE_IJVectorSetLocalPartitioning(reducedX_,CStartRow,
                                              CStartRow+CNRows);
    ierr = HYPRE_IJVectorAssemble(reducedX_);
    ierr = HYPRE_IJVectorInitialize(reducedX_);
    ierr = HYPRE_IJVectorZeroLocalComponents(reducedX_);
    assert(!ierr);

    ierr = HYPRE_IJVectorCreate(comm_, &reducedR_, globalNSchur);
    ierr = HYPRE_IJVectorSetLocalStorageType(reducedR_, HYPRE_PARCSR);
    ierr = HYPRE_IJVectorSetLocalPartitioning(reducedR_,CStartRow,
                                              CStartRow+CNRows);
    ierr = HYPRE_IJVectorAssemble(reducedR_);
    ierr = HYPRE_IJVectorInitialize(reducedR_);
    ierr = HYPRE_IJVectorZeroLocalComponents(reducedR_);
    assert(!ierr);

    currA_ = reducedA_;
    currB_ = reducedB_;
    currR_ = reducedR_;
    currX_ = reducedX_;

    //******************************************************************
    // save A21 and invA22 for solution recovery
    //------------------------------------------------------------------

    HYA21_    = CTmat; 
    HYinvA22_ = Mmat; 
    systemReduced_ = 1;

    //------------------------------------------------------------------
    // final clean up
    //------------------------------------------------------------------

    delete [] globalSchurList;
    delete [] schurList;
    delete [] ProcNRows;
    delete [] ProcNSchur;

    if ( colIndices_ != NULL )
    {
       for ( i = 0; i < localEndRow_-localStartRow_+1; i++ )
          if ( colIndices_[i] != NULL ) delete [] colIndices_[i];
       delete [] colIndices_;
       colIndices_ = NULL;
    }
    if ( colValues_ != NULL )
    {
       for ( j = 0; j < localEndRow_-localStartRow_+1; j++ )
          if ( colValues_[j] != NULL ) delete [] colValues_[j];
       delete [] colValues_;
       colValues_ = NULL;
       if ( rowLengths_ != NULL ) 
       {
          delete [] rowLengths_;
          rowLengths_ = NULL;
       }
    }
}

