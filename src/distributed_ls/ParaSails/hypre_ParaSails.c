/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * nalu_hypre_ParaSails
 *
 *****************************************************************************/

#include "Common.h"
#include "NALU_HYPRE_distributed_matrix_types.h"
#include "NALU_HYPRE_distributed_matrix_protos.h"
#include "nalu_hypre_ParaSails.h"
#include "Matrix.h"
#include "ParaSails.h"

/* these includes required for nalu_hypre_ParaSailsIJMatrix */
#include "../../IJ_mv/NALU_HYPRE_IJ_mv.h"
#include "../../NALU_HYPRE.h"
#include "../../utilities/_nalu_hypre_utilities.h"

typedef struct
{
   MPI_Comm   comm;
   ParaSails *ps;
}
   nalu_hypre_ParaSails_struct;

/*--------------------------------------------------------------------------
 * balance_info - Dump out information about the partitioning of the 
 * matrix, which affects load balance
 *--------------------------------------------------------------------------*/

#ifdef BALANCE_INFO
static void balance_info(MPI_Comm comm, Matrix *mat)
{
   NALU_HYPRE_Int mype, num_local, i, total;

   nalu_hypre_MPI_Comm_rank(comm, &mype);
   num_local = mat->end_row - mat->beg_row + 1;

   /* compute number of nonzeros on local matrix */
   total = 0;
   for (i=0; i<num_local; i++)
      total += mat->lens[i];

   /* each processor prints out its own info */
   nalu_hypre_printf("%4d: nrows %d, nnz %d, send %d (%d), recv %d (%d)\n",
                mype, num_local, total, mat->num_send, mat->sendlen,
                mat->num_recv, mat->recvlen);
}

static void matvec_timing(MPI_Comm comm, Matrix *mat)
{
   NALU_HYPRE_Real time0, time1;
   NALU_HYPRE_Real trial1, trial2, trial3, trial4, trial5, trial6;
   NALU_HYPRE_Real *temp1, *temp2;
   NALU_HYPRE_Int i, mype;
   NALU_HYPRE_Int n = mat->end_row - mat->beg_row + 1;

   temp1 = nalu_hypre_CTAlloc(NALU_HYPRE_Real, n, NALU_HYPRE_MEMORY_HOST);
   temp2 = nalu_hypre_CTAlloc(NALU_HYPRE_Real, n, NALU_HYPRE_MEMORY_HOST);

   /* warm-up */
   nalu_hypre_MPI_Barrier(comm);
   for (i=0; i<100; i++)
      MatrixMatvec(mat, temp1, temp2);

   nalu_hypre_MPI_Barrier(comm);
   time0 = nalu_hypre_MPI_Wtime();
   for (i=0; i<100; i++)
      MatrixMatvec(mat, temp1, temp2);
   nalu_hypre_MPI_Barrier(comm);
   time1 = nalu_hypre_MPI_Wtime();
   trial1 = time1-time0;

   nalu_hypre_MPI_Barrier(comm);
   time0 = nalu_hypre_MPI_Wtime();
   for (i=0; i<100; i++)
      MatrixMatvec(mat, temp1, temp2);
   nalu_hypre_MPI_Barrier(comm);
   time1 = nalu_hypre_MPI_Wtime();
   trial2 = time1-time0;

   nalu_hypre_MPI_Barrier(comm);
   time0 = nalu_hypre_MPI_Wtime();
   for (i=0; i<100; i++)
      MatrixMatvec(mat, temp1, temp2);
   nalu_hypre_MPI_Barrier(comm);
   time1 = nalu_hypre_MPI_Wtime();
   trial3 = time1-time0;

   nalu_hypre_MPI_Barrier(comm);
   time0 = nalu_hypre_MPI_Wtime();
   for (i=0; i<100; i++)
      MatrixMatvecSerial(mat, temp1, temp2);
   nalu_hypre_MPI_Barrier(comm);
   time1 = nalu_hypre_MPI_Wtime();
   trial4 = time1-time0;

   nalu_hypre_MPI_Barrier(comm);
   time0 = nalu_hypre_MPI_Wtime();
   for (i=0; i<100; i++)
      MatrixMatvecSerial(mat, temp1, temp2);
   nalu_hypre_MPI_Barrier(comm);
   time1 = nalu_hypre_MPI_Wtime();
   trial5 = time1-time0;

   nalu_hypre_MPI_Barrier(comm);
   time0 = nalu_hypre_MPI_Wtime();
   for (i=0; i<100; i++)
      MatrixMatvecSerial(mat, temp1, temp2);
   nalu_hypre_MPI_Barrier(comm);
   time1 = nalu_hypre_MPI_Wtime();
   trial6 = time1-time0;

   nalu_hypre_MPI_Comm_rank(comm, &mype);
   if (mype == 0)
      nalu_hypre_printf("Timings: %f %f %f Serial: %f %f %f\n", 
                   trial1, trial2, trial3, trial4, trial5, trial6);

   fflush(stdout);

   /* this is all we wanted, so don't waste any more cycles */
   exit(0);
}
#endif

/*--------------------------------------------------------------------------
 * convert_matrix - Create and convert distributed matrix to native 
 * data structure of ParaSails
 *--------------------------------------------------------------------------*/

static Matrix *convert_matrix(MPI_Comm comm, NALU_HYPRE_DistributedMatrix distmat)
{
   NALU_HYPRE_Int beg_row, end_row, row, dummy;
   NALU_HYPRE_Int len, *ind;
   NALU_HYPRE_Real *val;
   Matrix *mat;

   NALU_HYPRE_DistributedMatrixGetLocalRange(distmat, &beg_row, &end_row,
                                        &dummy, &dummy);

   mat = MatrixCreate(comm, beg_row, end_row);

   for (row=beg_row; row<=end_row; row++)
   {
      NALU_HYPRE_DistributedMatrixGetRow(distmat, row, &len, &ind, &val);
      MatrixSetRow(mat, row, len, ind, val);
      NALU_HYPRE_DistributedMatrixRestoreRow(distmat, row, &len, &ind, &val);
   }

   MatrixComplete(mat);

#ifdef BALANCE_INFO
   matvec_timing(comm, mat);
   balance_info(comm, mat);
#endif

   return mat;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ParaSailsCreate - Return a ParaSails preconditioner object "obj"
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int nalu_hypre_ParaSailsCreate(MPI_Comm comm, nalu_hypre_ParaSails *obj)
{
   nalu_hypre_ParaSails_struct *internal;

   internal = (nalu_hypre_ParaSails_struct *)
      nalu_hypre_CTAlloc(nalu_hypre_ParaSails_struct,  1, NALU_HYPRE_MEMORY_HOST);

   internal->comm = comm;
   internal->ps   = NULL;

   *obj = (nalu_hypre_ParaSails) internal;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ParaSailsDestroy - Destroy a ParaSails object "ps".
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int nalu_hypre_ParaSailsDestroy(nalu_hypre_ParaSails obj)
{
   nalu_hypre_ParaSails_struct *internal = (nalu_hypre_ParaSails_struct *) obj;

   ParaSailsDestroy(internal->ps);

   nalu_hypre_TFree(internal, NALU_HYPRE_MEMORY_HOST);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ParaSailsSetup - This function should be used if the preconditioner
 * pattern and values are set up with the same distributed matrix.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int nalu_hypre_ParaSailsSetup(nalu_hypre_ParaSails obj,
                               NALU_HYPRE_DistributedMatrix distmat, NALU_HYPRE_Int sym, NALU_HYPRE_Real thresh, NALU_HYPRE_Int nlevels,
                               NALU_HYPRE_Real filter, NALU_HYPRE_Real loadbal, NALU_HYPRE_Int logging)
{
   /* NALU_HYPRE_Real cost; */
   Matrix *mat;
   nalu_hypre_ParaSails_struct *internal = (nalu_hypre_ParaSails_struct *) obj;
   NALU_HYPRE_Int err;

   mat = convert_matrix(internal->comm, distmat);

   ParaSailsDestroy(internal->ps);

   internal->ps = ParaSailsCreate(internal->comm, 
                                  mat->beg_row, mat->end_row, sym);

   ParaSailsSetupPattern(internal->ps, mat, thresh, nlevels);

   if (logging)
      /* cost = */ ParaSailsStatsPattern(internal->ps, mat);

   internal->ps->loadbal_beta = loadbal;

   err = ParaSailsSetupValues(internal->ps, mat, filter);

   if (logging)
      ParaSailsStatsValues(internal->ps, mat);

   MatrixDestroy(mat);

   if (err)
   {
      nalu_hypre_error(NALU_HYPRE_ERROR_GENERIC);
   }
   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ParaSailsSetupPattern - Set up pattern using a distributed matrix.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int nalu_hypre_ParaSailsSetupPattern(nalu_hypre_ParaSails obj,
                                      NALU_HYPRE_DistributedMatrix distmat, NALU_HYPRE_Int sym, NALU_HYPRE_Real thresh, NALU_HYPRE_Int nlevels,
                                      NALU_HYPRE_Int logging)
{
   /* NALU_HYPRE_Real cost; */
   Matrix *mat;
   nalu_hypre_ParaSails_struct *internal = (nalu_hypre_ParaSails_struct *) obj;

   mat = convert_matrix(internal->comm, distmat);

   ParaSailsDestroy(internal->ps);

   internal->ps = ParaSailsCreate(internal->comm, 
                                  mat->beg_row, mat->end_row, sym);

   ParaSailsSetupPattern(internal->ps, mat, thresh, nlevels);

   if (logging)
      /* cost = */ ParaSailsStatsPattern(internal->ps, mat);

   MatrixDestroy(mat);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ParaSailsSetupValues - Set up values using a distributed matrix.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int nalu_hypre_ParaSailsSetupValues(nalu_hypre_ParaSails obj,
                                     NALU_HYPRE_DistributedMatrix distmat, NALU_HYPRE_Real filter, NALU_HYPRE_Real loadbal,
                                     NALU_HYPRE_Int logging)
{
   Matrix *mat;
   nalu_hypre_ParaSails_struct *internal = (nalu_hypre_ParaSails_struct *) obj;
   NALU_HYPRE_Int err;

   mat = convert_matrix(internal->comm, distmat);

   internal->ps->loadbal_beta = loadbal;
   internal->ps->setup_pattern_time = 0.0;

   err = ParaSailsSetupValues(internal->ps, mat, filter);

   if (logging)
      ParaSailsStatsValues(internal->ps, mat);

   MatrixDestroy(mat);

   if (err)
   {
      nalu_hypre_error(NALU_HYPRE_ERROR_GENERIC);
   }
   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ParaSailsApply - Apply the ParaSails preconditioner to an array 
 * "u", and return the result in the array "v".
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int nalu_hypre_ParaSailsApply(nalu_hypre_ParaSails obj, NALU_HYPRE_Real *u, NALU_HYPRE_Real *v)
{
   nalu_hypre_ParaSails_struct *internal = (nalu_hypre_ParaSails_struct *) obj;

   ParaSailsApply(internal->ps, u, v);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ParaSailsApplyTrans - Apply the ParaSails preconditioner, transposed
 * to an array "u", and return the result in the array "v".
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int nalu_hypre_ParaSailsApplyTrans(nalu_hypre_ParaSails obj, NALU_HYPRE_Real *u, NALU_HYPRE_Real *v)
{
   nalu_hypre_ParaSails_struct *internal = (nalu_hypre_ParaSails_struct *) obj;

   ParaSailsApplyTrans(internal->ps, u, v);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ParaSailsIJMatrix - Return the IJ matrix which is the sparse
 * approximate inverse (or its factor).  This matrix is a copy of the
 * matrix that is in ParaSails Matrix format.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ParaSailsBuildIJMatrix(nalu_hypre_ParaSails obj, NALU_HYPRE_IJMatrix *pij_A)
{
   nalu_hypre_ParaSails_struct *internal = (nalu_hypre_ParaSails_struct *) obj;
   ParaSails *ps = internal->ps;
   Matrix *mat = internal->ps->M;

   NALU_HYPRE_Int *diag_sizes, *offdiag_sizes, local_row, i, j;
   NALU_HYPRE_Int size;
   NALU_HYPRE_Int *col_inds;
   NALU_HYPRE_Real *values;

   NALU_HYPRE_IJMatrixCreate( ps->comm, ps->beg_row, ps->end_row,
                         ps->beg_row, ps->end_row,
                         pij_A );

   NALU_HYPRE_IJMatrixSetObjectType( *pij_A, NALU_HYPRE_PARCSR );

   diag_sizes = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  ps->end_row - ps->beg_row + 1, NALU_HYPRE_MEMORY_HOST);
   offdiag_sizes = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  ps->end_row - ps->beg_row + 1, NALU_HYPRE_MEMORY_HOST);
   local_row = 0;
   for (i=ps->beg_row; i<= ps->end_row; i++)
   {
      MatrixGetRow(mat, local_row, &size, &col_inds, &values);
      NumberingLocalToGlobal(ps->numb, size, col_inds, col_inds);

      for (j=0; j < size; j++)
      {
         if (col_inds[j] < ps->beg_row || col_inds[j] > ps->end_row)
            offdiag_sizes[local_row]++;
         else
            diag_sizes[local_row]++;
      }

      local_row++;
   }
   NALU_HYPRE_IJMatrixSetDiagOffdSizes( *pij_A, (const NALU_HYPRE_Int *) diag_sizes,
                                   (const NALU_HYPRE_Int *) offdiag_sizes );
   nalu_hypre_TFree(diag_sizes, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(offdiag_sizes, NALU_HYPRE_MEMORY_HOST);

   NALU_HYPRE_IJMatrixInitialize( *pij_A );

   local_row = 0;
   for (i=ps->beg_row; i<= ps->end_row; i++)
   {
      MatrixGetRow(mat, local_row, &size, &col_inds, &values);

      NALU_HYPRE_IJMatrixSetValues( *pij_A, 1, &size, &i, (const NALU_HYPRE_Int *) col_inds,
                               (const NALU_HYPRE_Real *) values );

      NumberingGlobalToLocal(ps->numb, size, col_inds, col_inds);

      local_row++;
   }

   NALU_HYPRE_IJMatrixAssemble( *pij_A );

   return nalu_hypre_error_flag;
}
