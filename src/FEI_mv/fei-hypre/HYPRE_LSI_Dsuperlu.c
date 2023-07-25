/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * NALU_HYPRE_LSI_DSuperLU interface
 *
 *****************************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "utilities/_nalu_hypre_utilities.h"
#include "NALU_HYPRE.h"
#include "IJ_mv/NALU_HYPRE_IJ_mv.h"
#include "parcsr_mv/_nalu_hypre_parcsr_mv.h"
#include "parcsr_ls/NALU_HYPRE_parcsr_ls.h"

/*---------------------------------------------------------------------------
 * Distributed SUPERLU include files
 *-------------------------------------------------------------------------*/

#ifdef NALU_HYPRE_USING_DSUPERLU
#include "parcsr_ls/dsuperlu.h"
#include "superlu_ddefs.h"

typedef struct NALU_HYPRE_LSI_DSuperLU_Struct
{
   MPI_Comm           comm_;
   NALU_HYPRE_ParCSRMatrix Amat_;
   superlu_dist_options_t  options_;
   SuperMatrix        sluAmat_;
   dScalePermstruct_t ScalePermstruct_;
   SuperLUStat_t      stat_;
   dLUstruct_t        LUstruct_;
   dSOLVEstruct_t     SOLVEstruct_;
   int                globalNRows_;
   int                localNRows_;
   int                startRow_;
   int                outputLevel_;
   double             *berr_;
   gridinfo_t         sluGrid_;
   int                setupFlag_;
}
NALU_HYPRE_LSI_DSuperLU;

int NALU_HYPRE_LSI_DSuperLUGenMatrix(NALU_HYPRE_Solver solver);

/***************************************************************************
 * NALU_HYPRE_LSI_DSuperLUCreate - Return a DSuperLU object "solver".
 *--------------------------------------------------------------------------*/

int NALU_HYPRE_LSI_DSuperLUCreate( MPI_Comm comm, NALU_HYPRE_Solver *solver )
{
   nalu_hypre_DSLUData *dslu_data = NULL;
   dslu_data = nalu_hypre_CTAlloc(nalu_hypre_DSLUData, 1, NALU_HYPRE_MEMORY_HOST);
   *solver = (NALU_HYPRE_Solver) dslu_data;
   return 0;
}

/***************************************************************************
 * NALU_HYPRE_LSI_DSuperLUDestroy - Destroy a DSuperLU object.
 *--------------------------------------------------------------------------*/

int NALU_HYPRE_LSI_DSuperLUDestroy( NALU_HYPRE_Solver solver )
{
   nalu_hypre_DSLUData *dslu_data = (nalu_hypre_DSLUData *) solver;

   PStatFree(&(dslu_data->dslu_data_stat));
   Destroy_CompRowLoc_Matrix_dist(&(dslu_data->A_dslu));
   dScalePermstructFree(&(dslu_data->dslu_ScalePermstruct));
   dDestroy_LU(dslu_data->global_num_rows, &(dslu_data->dslu_data_grid), &(dslu_data->dslu_data_LU));
   dLUstructFree(&(dslu_data->dslu_data_LU));
   if (dslu_data->dslu_options.SolveInitialized)
   {
      dSolveFinalize(&(dslu_data->dslu_options), &(dslu_data->dslu_solve));
   }
   superlu_gridexit(&(dslu_data->dslu_data_grid));
   nalu_hypre_TFree(dslu_data->berr, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(dslu_data, NALU_HYPRE_MEMORY_HOST);
   return nalu_hypre_error_flag;
}

/***************************************************************************
 * NALU_HYPRE_LSI_DSuperLUSetOutputLevel - Set debug level
 *--------------------------------------------------------------------------*/

int NALU_HYPRE_LSI_DSuperLUSetOutputLevel(NALU_HYPRE_Solver solver, int level)
{
   NALU_HYPRE_LSI_DSuperLU *sluPtr = (NALU_HYPRE_LSI_DSuperLU *) solver;
   sluPtr->outputLevel_ = level;
   return 0;
}

/***************************************************************************
 * NALU_HYPRE_LSI_DSuperLUSetup - Set up function for LSI_DSuperLU.
 *--------------------------------------------------------------------------*/

int NALU_HYPRE_LSI_DSuperLUSetup(NALU_HYPRE_Solver solver, NALU_HYPRE_ParCSRMatrix A_csr,
                            NALU_HYPRE_ParVector b, NALU_HYPRE_ParVector x )
{
   /* Par Data Structure variables */
   NALU_HYPRE_BigInt global_num_rows = nalu_hypre_ParCSRMatrixGlobalNumRows(A_csr);
   MPI_Comm comm = nalu_hypre_ParCSRMatrixComm(A_csr);
   nalu_hypre_CSRMatrix *A_local;
   NALU_HYPRE_Int num_rows;
   NALU_HYPRE_Int num_procs, my_id;
   NALU_HYPRE_Int pcols = 1, prows = 1;
   NALU_HYPRE_BigInt *big_rowptr = NULL;
   nalu_hypre_DSLUData *dslu_data = (nalu_hypre_DSLUData *) solver;

   NALU_HYPRE_Int info = 0;
   NALU_HYPRE_Int nrhs = 0;

   nalu_hypre_MPI_Comm_size(comm, &num_procs);
   nalu_hypre_MPI_Comm_rank(comm, &my_id);

   /* Merge diag and offd into one matrix (global ids) */
   A_local = nalu_hypre_MergeDiagAndOffd(A_csr);

   num_rows = nalu_hypre_CSRMatrixNumRows(A_local);
   /* Now convert hypre matrix to a SuperMatrix */
#ifdef NALU_HYPRE_MIXEDINT
   {
      NALU_HYPRE_Int *rowptr = NULL;
      NALU_HYPRE_Int  i;
      rowptr = nalu_hypre_CSRMatrixI(A_local);
      big_rowptr = nalu_hypre_CTAlloc(NALU_HYPRE_BigInt, (num_rows + 1), NALU_HYPRE_MEMORY_HOST);
      for (i = 0; i < (num_rows + 1); i++)
      {
         big_rowptr[i] = (NALU_HYPRE_BigInt)rowptr[i];
      }
   }
#else
   big_rowptr = nalu_hypre_CSRMatrixI(A_local);
#endif
   dCreate_CompRowLoc_Matrix_dist(
      &(dslu_data->A_dslu), global_num_rows, global_num_rows,
      nalu_hypre_CSRMatrixNumNonzeros(A_local),
      num_rows,
      nalu_hypre_ParCSRMatrixFirstRowIndex(A_csr),
      nalu_hypre_CSRMatrixData(A_local),
      nalu_hypre_CSRMatrixBigJ(A_local), big_rowptr,
      SLU_NR_loc, SLU_D, SLU_GE);

   /* DOK: SuperLU frees assigned data, so set them to null before
    * calling nalu_hypre_CSRMatrixdestroy on A_local to avoid memory errors.
   */
#ifndef NALU_HYPRE_MIXEDINT
   nalu_hypre_CSRMatrixI(A_local) = NULL;
#endif
   nalu_hypre_CSRMatrixData(A_local) = NULL;
   nalu_hypre_CSRMatrixBigJ(A_local) = NULL;
   nalu_hypre_CSRMatrixDestroy(A_local);

   /*Create process grid */
   while (prows * pcols <= num_procs) { ++prows; }
   --prows;
   pcols = num_procs / prows;
   while (prows * pcols != num_procs)
   {
      prows -= 1;
      pcols = num_procs / prows;
   }
   //nalu_hypre_printf(" prows %d pcols %d\n", prows, pcols);

   superlu_gridinit(comm, prows, pcols, &(dslu_data->dslu_data_grid));

   set_default_options_dist(&(dslu_data->dslu_options));

   dslu_data->dslu_options.Fact = DOFACT;
   dslu_data->dslu_options.PrintStat = NO;
   /*dslu_data->dslu_options.IterRefine = SLU_DOUBLE;
   dslu_data->dslu_options.ColPerm = MMD_AT_PLUS_A;
   dslu_data->dslu_options.DiagPivotThresh = 1.0;
   dslu_data->dslu_options.ReplaceTinyPivot = NO; */

   dScalePermstructInit(global_num_rows, global_num_rows, &(dslu_data->dslu_ScalePermstruct));

   dLUstructInit(global_num_rows, &(dslu_data->dslu_data_LU));

   PStatInit(&(dslu_data->dslu_data_stat));

   dslu_data->global_num_rows = global_num_rows;

   dslu_data->berr = nalu_hypre_CTAlloc(NALU_HYPRE_Real, 1, NALU_HYPRE_MEMORY_HOST);
   dslu_data->berr[0] = 0.0;

   pdgssvx(&(dslu_data->dslu_options), &(dslu_data->A_dslu),
           &(dslu_data->dslu_ScalePermstruct), NULL, num_rows, nrhs,
           &(dslu_data->dslu_data_grid), &(dslu_data->dslu_data_LU),
           &(dslu_data->dslu_solve), dslu_data->berr, &(dslu_data->dslu_data_stat), &info);

   dslu_data->dslu_options.Fact = FACTORED;
   return nalu_hypre_error_flag;
}

/***************************************************************************
 * NALU_HYPRE_LSI_DSuperLUSolve - Solve function for DSuperLU.
 *--------------------------------------------------------------------------*/

int NALU_HYPRE_LSI_DSuperLUSolve( NALU_HYPRE_Solver solver, NALU_HYPRE_ParCSRMatrix A,
                             NALU_HYPRE_ParVector b, NALU_HYPRE_ParVector x )
{
   nalu_hypre_DSLUData *dslu_data = (nalu_hypre_DSLUData *) solver;
   NALU_HYPRE_Int info = 0;
   NALU_HYPRE_Real *B = nalu_hypre_VectorData(nalu_hypre_ParVectorLocalVector(x));
   NALU_HYPRE_Int size = nalu_hypre_VectorSize(nalu_hypre_ParVectorLocalVector(x));
   NALU_HYPRE_Int nrhs = 1;

   nalu_hypre_ParVectorCopy(b, x);

   pdgssvx(&(dslu_data->dslu_options), &(dslu_data->A_dslu),
           &(dslu_data->dslu_ScalePermstruct), B, size, nrhs,
           &(dslu_data->dslu_data_grid), &(dslu_data->dslu_data_LU),
           &(dslu_data->dslu_solve), dslu_data->berr, &(dslu_data->dslu_data_stat), &info);

   return nalu_hypre_error_flag;
}

/****************************************************************************
 * Create SuperLU matrix in CSR
 *--------------------------------------------------------------------------*/

int NALU_HYPRE_LSI_DSuperLUGenMatrix(NALU_HYPRE_Solver solver)
{
   int        nprocs, mypid, *csrIA, *csrJA, *procNRows, localNNZ;
   int        startRow, localNRows, rowSize, *colInd, irow, jcol;
   double     *csrAA, *colVal;
   NALU_HYPRE_LSI_DSuperLU *sluPtr = (NALU_HYPRE_LSI_DSuperLU *) solver;
   NALU_HYPRE_ParCSRMatrix Amat;
   MPI_Comm   mpiComm;

   /* ---------------------------------------------------------------- */
   /* fetch parallel machine parameters                                */
   /* ---------------------------------------------------------------- */

   mpiComm = sluPtr->comm_;
   MPI_Comm_rank(mpiComm, &mypid);
   MPI_Comm_size(mpiComm, &nprocs);

   /* ---------------------------------------------------------------- */
   /* fetch matrix information                                         */
   /* ---------------------------------------------------------------- */

   Amat = sluPtr->Amat_;
   NALU_HYPRE_ParCSRMatrixGetRowPartitioning(Amat, &procNRows);
   startRow = procNRows[mypid];
   sluPtr->startRow_ = startRow;
   localNNZ = 0;
   for (irow = startRow; irow < procNRows[mypid+1]; irow++)
   {
      NALU_HYPRE_ParCSRMatrixGetRow(Amat,irow,&rowSize,&colInd,&colVal);
      localNNZ += rowSize;
      NALU_HYPRE_ParCSRMatrixRestoreRow(Amat,irow,&rowSize,&colInd,&colVal);
   }
   localNRows = procNRows[mypid+1] - procNRows[mypid];
   sluPtr->localNRows_ = localNRows;
   sluPtr->globalNRows_ = procNRows[nprocs];
   csrIA = (int *) intMalloc_dist(localNRows+1);
   csrJA = (int *) intMalloc_dist(localNNZ);
   csrAA = (double *) doubleMalloc_dist(localNNZ);
   localNNZ = 0;

   csrIA[0] = localNNZ;
   for (irow = startRow; irow < procNRows[mypid+1]; irow++)
   {
      NALU_HYPRE_ParCSRMatrixGetRow(Amat,irow,&rowSize,&colInd,&colVal);
      for ( jcol = 0; jcol < rowSize; jcol++ )
      {
         csrJA[localNNZ] = colInd[jcol];
         csrAA[localNNZ++] = colVal[jcol];
      }
      csrIA[irow-startRow+1] = localNNZ;
      NALU_HYPRE_ParCSRMatrixRestoreRow(Amat,irow,&rowSize,&colInd,&colVal);
   }
   /*for (irow = startRow; irow < procNRows[mypid+1]; irow++)
    *   qsort1(csrJA, csrAA, csrIA[irow-startRow], csrIA[irow-startRow+1]-1);
    */

   /* ---------------------------------------------------------------- */
   /* create SuperLU matrix                                            */
   /* ---------------------------------------------------------------- */

   dCreate_CompRowLoc_Matrix_dist(&(sluPtr->sluAmat_), sluPtr->globalNRows_,
            sluPtr->globalNRows_, localNNZ, localNRows, startRow, csrAA,
            csrJA, csrIA, SLU_NR_loc, SLU_D, SLU_GE);
   nalu_hypre_TFree(procNRows, NALU_HYPRE_MEMORY_HOST);
   return 0;
}
#else
   int bogus;
#endif

