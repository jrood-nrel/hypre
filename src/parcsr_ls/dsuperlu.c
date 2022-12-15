/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_nalu_hypre_parcsr_ls.h"
#include <math.h>

#ifdef NALU_HYPRE_USING_DSUPERLU
#include "dsuperlu.h"

#include <math.h>
#include "superlu_ddefs.h"
/*
#ifndef nalu_hypre_DSLU_DATA_HEADER
#define nalu_hypre_DSLU_DATA_HEADER

typedef struct
{
   NALU_HYPRE_BigInt global_num_rows;
   SuperMatrix A_dslu;
   NALU_HYPRE_Real *berr;
   dLUstruct_t dslu_data_LU;
   SuperLUStat_t dslu_data_stat;
   superlu_dist_options_t dslu_options;
   gridinfo_t dslu_data_grid;
   dScalePermstruct_t dslu_ScalePermstruct;
   dSOLVEstruct_t dslu_solve;
}
nalu_hypre_DSLUData;

#endif
*/
NALU_HYPRE_Int nalu_hypre_SLUDistSetup( NALU_HYPRE_Solver *solver, nalu_hypre_ParCSRMatrix *A, NALU_HYPRE_Int print_level)
{
   /* Par Data Structure variables */
   NALU_HYPRE_BigInt global_num_rows = nalu_hypre_ParCSRMatrixGlobalNumRows(A);
   MPI_Comm           comm = nalu_hypre_ParCSRMatrixComm(A);
   nalu_hypre_CSRMatrix *A_local;
   NALU_HYPRE_Int num_rows;
   NALU_HYPRE_Int num_procs, my_id;
   NALU_HYPRE_Int pcols = 1, prows = 1;
   NALU_HYPRE_BigInt *big_rowptr = NULL;
   nalu_hypre_DSLUData *dslu_data = NULL;

   NALU_HYPRE_Int info = 0;
   NALU_HYPRE_Int nrhs = 0;

   nalu_hypre_MPI_Comm_size(comm, &num_procs);
   nalu_hypre_MPI_Comm_rank(comm, &my_id);

   /* destroy solver if already setup */
   //   if (solver != NULL) { nalu_hypre_SLUDistDestroy(solver); }
   /* allocate memory for new solver */
   dslu_data = nalu_hypre_CTAlloc(nalu_hypre_DSLUData, 1, NALU_HYPRE_MEMORY_HOST);

   /* Merge diag and offd into one matrix (global ids) */
   A_local = nalu_hypre_MergeDiagAndOffd(A);

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
      nalu_hypre_ParCSRMatrixFirstRowIndex(A),
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
   if (print_level == 0 || print_level == 2) { dslu_data->dslu_options.PrintStat = NO; }
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
   *solver = (NALU_HYPRE_Solver) dslu_data;
   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int nalu_hypre_SLUDistSolve( void* solver, nalu_hypre_ParVector *b, nalu_hypre_ParVector *x)
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

NALU_HYPRE_Int nalu_hypre_SLUDistDestroy( void* solver)
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
#endif
