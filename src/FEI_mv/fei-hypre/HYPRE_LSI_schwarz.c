/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * NALU_HYPRE_Schwarz interface
 *
 *****************************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "utilities/_nalu_hypre_utilities.h"
#include "NALU_HYPRE.h"
#include "IJ_mv/NALU_HYPRE_IJ_mv.h"
#include "parcsr_mv/NALU_HYPRE_parcsr_mv.h"
#include "parcsr_mv/_nalu_hypre_parcsr_mv.h"
#include "parcsr_ls/NALU_HYPRE_parcsr_ls.h"
#include "NALU_HYPRE_MHMatrix.h"

#ifdef HAVE_ML

#include "ml_struct.h"
#include "ml_aggregate.h"

#endif

#include "NALU_HYPRE_MHMatrix.h"
#include "NALU_HYPRE_FEI.h"

typedef struct NALU_HYPRE_LSI_Schwarz_Struct
{
   MPI_Comm      comm;
   MH_Matrix     *mh_mat;
   int           Nrows;
   int           extNrows;
   int           ntimes;
   double        fillin;
   double        threshold;
   int           output_level;
   int           **bmat_ia;
   int           **bmat_ja;
   double        **bmat_aa;
   int           **aux_bmat_ia;
   int           **aux_bmat_ja;
   double        **aux_bmat_aa;
   int           nblocks;
   int           block_size;
   int           *blk_sizes;
   int           **blk_indices;
} NALU_HYPRE_LSI_Schwarz;

extern int  NALU_HYPRE_LSI_MLConstructMHMatrix(NALU_HYPRE_ParCSRMatrix,MH_Matrix *,
                                          MPI_Comm, int *, MH_Context *);
extern int  NALU_HYPRE_LSI_SchwarzDecompose(NALU_HYPRE_LSI_Schwarz *sch_ptr,
                 MH_Matrix *Amat, int total_recv_leng, int *recv_lengths,
                 int *ext_ja, double *ext_aa, int *map, int *map2,
                 int Noffset);
extern int  NALU_HYPRE_LSI_ILUTDecompose(NALU_HYPRE_LSI_Schwarz *sch_ptr);
extern void nalu_hypre_qsort0(int *, int, int);
extern int  NALU_HYPRE_LSI_SplitDSort(double*,int,int*,int);
extern int  MH_ExchBdry(double *, void *);
extern int  NALU_HYPRE_LSI_Search(int *, int, int);

#define habs(x) ((x) > 0 ? (x) : -(x))

/*--------------------------------------------------------------------------
 * NALU_HYPRE_LSI_SchwarzCreate - Return a Schwarz preconditioner object "solver"
 *-------------------------------------------------------------------------*/

int NALU_HYPRE_LSI_SchwarzCreate( MPI_Comm comm, NALU_HYPRE_Solver *solver )
{
   NALU_HYPRE_LSI_Schwarz *sch_ptr;

   sch_ptr = nalu_hypre_TAlloc(NALU_HYPRE_LSI_Schwarz, 1, NALU_HYPRE_MEMORY_HOST);

   if (sch_ptr == NULL) return 1;

   sch_ptr->comm        = comm;
   sch_ptr->mh_mat      = NULL;
   sch_ptr->bmat_ia     = NULL;
   sch_ptr->bmat_ja     = NULL;
   sch_ptr->bmat_aa     = NULL;
   sch_ptr->aux_bmat_ia = NULL;
   sch_ptr->aux_bmat_ja = NULL;
   sch_ptr->aux_bmat_aa = NULL;
   sch_ptr->fillin      = 0.0;
   sch_ptr->threshold   = 1.0e-16;
   sch_ptr->Nrows       = 0;
   sch_ptr->extNrows    = 0;
   sch_ptr->nblocks     = 1;
   sch_ptr->blk_sizes   = NULL;
   sch_ptr->block_size  = 1000;
   sch_ptr->blk_indices = NULL;
   sch_ptr->ntimes      = 1;
   sch_ptr->output_level = 0;
   *solver = (NALU_HYPRE_Solver) sch_ptr;
   return 0;
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_LSI_SchwarzDestroy - Destroy a Schwarz object.
 *-------------------------------------------------------------------------*/

int NALU_HYPRE_LSI_SchwarzDestroy( NALU_HYPRE_Solver solver )
{
   int               i;
   NALU_HYPRE_LSI_Schwarz *sch_ptr;

   sch_ptr = (NALU_HYPRE_LSI_Schwarz *) solver;
   if ( sch_ptr->bmat_ia  != NULL )
   {
      for ( i = 0; i < sch_ptr->nblocks; i++ )
         nalu_hypre_TFree(sch_ptr->bmat_ia[i], NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(sch_ptr->bmat_ia, NALU_HYPRE_MEMORY_HOST);
   }
   if ( sch_ptr->bmat_ja  != NULL )
   {
      for ( i = 0; i < sch_ptr->nblocks; i++ )
         nalu_hypre_TFree(sch_ptr->bmat_ja[i], NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(sch_ptr->bmat_ja, NALU_HYPRE_MEMORY_HOST);
   }
   if ( sch_ptr->bmat_aa  != NULL )
   {
      for ( i = 0; i < sch_ptr->nblocks; i++ )
         nalu_hypre_TFree(sch_ptr->bmat_aa[i], NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(sch_ptr->bmat_aa, NALU_HYPRE_MEMORY_HOST);
   }
   if ( sch_ptr->aux_bmat_ia  != NULL )
   {
      for ( i = 0; i < sch_ptr->nblocks; i++ )
         nalu_hypre_TFree(sch_ptr->aux_bmat_ia[i], NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(sch_ptr->aux_bmat_ia, NALU_HYPRE_MEMORY_HOST);
   }
   if ( sch_ptr->aux_bmat_ja  != NULL )
   {
      for ( i = 0; i < sch_ptr->nblocks; i++ )
         nalu_hypre_TFree(sch_ptr->aux_bmat_ja[i], NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(sch_ptr->aux_bmat_ja, NALU_HYPRE_MEMORY_HOST);
   }
   if ( sch_ptr->aux_bmat_aa  != NULL )
   {
      for ( i = 0; i < sch_ptr->nblocks; i++ )
         nalu_hypre_TFree(sch_ptr->aux_bmat_aa[i], NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(sch_ptr->aux_bmat_aa, NALU_HYPRE_MEMORY_HOST);
   }
   if ( sch_ptr->blk_sizes != NULL )
      nalu_hypre_TFree(sch_ptr->blk_sizes, NALU_HYPRE_MEMORY_HOST);
   if ( sch_ptr->blk_indices != NULL )
   {
      for ( i = 0; i < sch_ptr->nblocks; i++ )
         if ( sch_ptr->blk_indices[i] != NULL )
            nalu_hypre_TFree(sch_ptr->blk_indices[i], NALU_HYPRE_MEMORY_HOST);
   }
   if ( sch_ptr->mh_mat != NULL )
   {
      nalu_hypre_TFree(sch_ptr->mh_mat->sendProc, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(sch_ptr->mh_mat->sendLeng, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(sch_ptr->mh_mat->recvProc, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(sch_ptr->mh_mat->recvLeng, NALU_HYPRE_MEMORY_HOST);
      for ( i = 0; i < sch_ptr->mh_mat->sendProcCnt; i++ )
         nalu_hypre_TFree(sch_ptr->mh_mat->sendList[i], NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(sch_ptr->mh_mat->sendList, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(sch_ptr->mh_mat, NALU_HYPRE_MEMORY_HOST);
   }
   sch_ptr->mh_mat = NULL;
   nalu_hypre_TFree(sch_ptr, NALU_HYPRE_MEMORY_HOST);

   return 0;
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_LSI_SchwarzSetOutputLevel - Set debug level
 *-------------------------------------------------------------------------*/

int NALU_HYPRE_LSI_SchwarzSetOutputLevel(NALU_HYPRE_Solver solver, int level)
{
   NALU_HYPRE_LSI_Schwarz *sch_ptr = (NALU_HYPRE_LSI_Schwarz *) solver;

   sch_ptr->output_level = level;

   return 0;
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_LSI_SchwarzSetBlockSize - Set block size
 *-------------------------------------------------------------------------*/

int NALU_HYPRE_LSI_SchwarzSetNBlocks(NALU_HYPRE_Solver solver, int nblks)
{
   NALU_HYPRE_LSI_Schwarz *sch_ptr = (NALU_HYPRE_LSI_Schwarz *) solver;

   sch_ptr->nblocks = nblks;

   return 0;
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_LSI_SchwarzSetBlockSize - Set block size
 *-------------------------------------------------------------------------*/

int NALU_HYPRE_LSI_SchwarzSetBlockSize(NALU_HYPRE_Solver solver, int blksize)
{
   NALU_HYPRE_LSI_Schwarz *sch_ptr = (NALU_HYPRE_LSI_Schwarz *) solver;

   sch_ptr->block_size = blksize;

   return 0;
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_LSI_SchwarzSetILUTFillin - Set fillin for block solve
 *-------------------------------------------------------------------------*/

int NALU_HYPRE_LSI_SchwarzSetILUTFillin(NALU_HYPRE_Solver solver, double fillin)
{
   NALU_HYPRE_LSI_Schwarz *sch_ptr = (NALU_HYPRE_LSI_Schwarz *) solver;

   sch_ptr->fillin = fillin;

   return 0;
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_LSI_SchwarzSolve - Solve function for Schwarz.
 *-------------------------------------------------------------------------*/

int NALU_HYPRE_LSI_SchwarzSolve( NALU_HYPRE_Solver solver, NALU_HYPRE_ParCSRMatrix Amat,
                            NALU_HYPRE_ParVector b,   NALU_HYPRE_ParVector x )
{
   int               i, j, cnt, blk, index, max_blk_size, nrows;
   int               ntimes, Nrows, extNrows, nblocks, *indptr, column;
   int               *aux_mat_ia, *aux_mat_ja, *mat_ia, *mat_ja, *idiag;
   double            *dbuffer, *aux_mat_aa, *solbuf, *xbuffer;
   double            *rhs, *soln, *mat_aa, ddata;
   MH_Context        *context;
   NALU_HYPRE_LSI_Schwarz *sch_ptr = (NALU_HYPRE_LSI_Schwarz *) solver;

   /* ---------------------------------------------------------
    * fetch vectors
    * ---------------------------------------------------------*/

   rhs  = nalu_hypre_VectorData(nalu_hypre_ParVectorLocalVector((nalu_hypre_ParVector*) b));
   soln = nalu_hypre_VectorData(nalu_hypre_ParVectorLocalVector((nalu_hypre_ParVector*) x));

   /* ---------------------------------------------------------
    * fetch vectors
    * ---------------------------------------------------------*/

   ntimes      = sch_ptr->ntimes;
   Nrows       = sch_ptr->Nrows;
   extNrows    = sch_ptr->extNrows;
   nblocks     = sch_ptr->nblocks;
   max_blk_size = 0;
   for ( i = 0; i < nblocks; i++ )
      if (sch_ptr->blk_sizes[i] > max_blk_size)
         max_blk_size = sch_ptr->blk_sizes[i];

   /* ---------------------------------------------------------
    * initialize memory for interprocessor communication
    * ---------------------------------------------------------*/

   dbuffer = nalu_hypre_TAlloc(double, extNrows , NALU_HYPRE_MEMORY_HOST);
   for ( i = 0; i < Nrows; i++ ) dbuffer[i] = rhs[i];
   for ( i = 0; i < Nrows; i++ ) soln[i]    = 0.0;

   context = nalu_hypre_TAlloc(MH_Context, 1, NALU_HYPRE_MEMORY_HOST);
   context->comm = sch_ptr->comm;
   context->Amat = sch_ptr->mh_mat;

   /* ---------------------------------------------------------
    * communicate the rhs and put into dbuffer
    * ---------------------------------------------------------*/

   if ( extNrows > Nrows ) MH_ExchBdry(dbuffer, context);

   solbuf  = nalu_hypre_TAlloc(double, max_blk_size , NALU_HYPRE_MEMORY_HOST);
   idiag   = nalu_hypre_TAlloc(int, max_blk_size , NALU_HYPRE_MEMORY_HOST);
   xbuffer = nalu_hypre_TAlloc(double, extNrows , NALU_HYPRE_MEMORY_HOST);
   for ( i = Nrows; i < extNrows; i++ ) xbuffer[i] = 0.0;

   /* ---------------------------------------------------------
    * the first pass
    * ---------------------------------------------------------*/

   for ( blk = 0; blk < nblocks; blk++ )
   {
      nrows  = sch_ptr->blk_sizes[blk];
      if ( sch_ptr->blk_indices != NULL )
      {
         indptr = sch_ptr->blk_indices[blk];
         for ( i = 0; i < nrows; i++ ) solbuf[i] = dbuffer[indptr[i]];
      }
      else
      {
         for ( i = 0; i < nrows; i++ ) solbuf[i] = dbuffer[i];
      }
      mat_ia = sch_ptr->bmat_ia[blk];
      mat_ja = sch_ptr->bmat_ja[blk];
      mat_aa = sch_ptr->bmat_aa[blk];
      if ( nblocks > 1 )
      {
         aux_mat_ia  = sch_ptr->aux_bmat_ia[blk];
         aux_mat_ja  = sch_ptr->aux_bmat_ja[blk];
         aux_mat_aa  = sch_ptr->aux_bmat_aa[blk];
      }
      if ( nblocks > 1 )
      {
         for ( i = 0; i < nrows; i++ )
         {
            ddata = solbuf[i];
            for ( j = aux_mat_ia[i]; j < aux_mat_ia[i+1]; j++ )
            {
               index = aux_mat_ja[j];
               if (index<Nrows) ddata -= (aux_mat_aa[j]*soln[index]);
               else             ddata -= (aux_mat_aa[j]*xbuffer[index]);
            }
            solbuf[i] = ddata;
         }
      }
      for ( i = 0; i < nrows; i++ )
      {
         ddata = 0.0;
         for ( j = mat_ia[i]; j < mat_ia[i+1]; j++ )
         {
            column = mat_ja[j];
            if ( column == i ) { idiag[i] = j; break;}
            ddata += mat_aa[j] * solbuf[column];
         }
         solbuf[i] -= ddata;
      }
      for ( i = nrows-1; i >= 0; i-- )
      {
         ddata = 0.0;
         for ( j = idiag[i]+1; j < mat_ia[i+1]; j++ )
         {
            column = mat_ja[j];
            ddata += mat_aa[j] * solbuf[column];
         }
         solbuf[i] -= ddata;
         solbuf[i] /= mat_aa[idiag[i]];
      }
      if ( nblocks > 1 )
      {
         for ( i = 0; i < nrows; i++ )
         {
            if ( indptr[i] < Nrows ) soln[indptr[i]] = solbuf[i];
            else                     xbuffer[indptr[i]] = solbuf[i];
         }
      }
      else
      {
         for ( i = 0; i < nrows; i++ )
         {
            if ( i < Nrows ) soln[i] = solbuf[i];
            else             xbuffer[i] = solbuf[i];
         }
      }
   }

   for ( cnt = 1; cnt < ntimes; cnt++ )
   {
      for ( i = 0; i < Nrows; i++ ) xbuffer[i] = soln[i];
      if ( extNrows > Nrows ) MH_ExchBdry(xbuffer, context);

      for ( blk = 0; blk < nblocks; blk++ )
      {
         nrows   = sch_ptr->blk_sizes[blk];
         mat_ia  = sch_ptr->bmat_ia[blk];
         mat_ja  = sch_ptr->bmat_ja[blk];
         mat_aa  = sch_ptr->bmat_aa[blk];
         if ( nblocks > 1 )
         {
            indptr  = sch_ptr->blk_indices[blk];
            aux_mat_ia  = sch_ptr->aux_bmat_ia[blk];
            aux_mat_ja  = sch_ptr->aux_bmat_ja[blk];
            aux_mat_aa  = sch_ptr->aux_bmat_aa[blk];
            for ( i = 0; i < nrows; i++ )
            {
               ddata = dbuffer[indptr[i]];
               for ( j = aux_mat_ia[i]; j < aux_mat_ia[i+1]; j++ )
               {
                  index = aux_mat_ja[j];
                  if (index<Nrows) ddata -= (aux_mat_aa[j]*soln[index]);
                  else             ddata -= (aux_mat_aa[j]*xbuffer[index]);
               }
               solbuf[i] = ddata;
            }
         }
         else
            for ( i = 0; i < nrows; i++ ) solbuf[i] = dbuffer[i];

         for ( i = 0; i < nrows; i++ )
         {
            ddata = 0.0;
            for ( j = mat_ia[i]; j < mat_ia[i+1]; j++ )
            {
               column = mat_ja[j];
               if ( column == i ) { idiag[i] = j; break;}
               ddata += mat_aa[j] * solbuf[column];
            }
            solbuf[i] -= ddata;
         }
         for ( i = nrows-1; i >= 0; i-- )
         {
            ddata = 0.0;
            for ( j = idiag[i]+1; j < mat_ia[i+1]; j++ )
            {
               column = mat_ja[j];
               ddata += mat_aa[j] * solbuf[column];
            }
            solbuf[i] -= ddata;
            solbuf[i] /= mat_aa[idiag[i]];
         }
         if ( nblocks > 1 )
         {
            for ( i = 0; i < nrows; i++ )
               if ( indptr[i] < Nrows ) soln[indptr[i]] = solbuf[i];
               else                     xbuffer[indptr[i]] = solbuf[i];
         }
         else
         {
            for ( i = 0; i < nrows; i++ )
               if ( i < Nrows ) soln[i] = solbuf[i];
               else             xbuffer[i] = solbuf[i];
         }
      }
   }

   /* --------------------------------------------------------- */
   /* clean up                                                  */
   /* --------------------------------------------------------- */

   nalu_hypre_TFree(xbuffer, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(idiag, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(solbuf, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(dbuffer, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(context, NALU_HYPRE_MEMORY_HOST);

   return 0;
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_LSI_SchwarzSetup - Set up function for LSI_Schwarz.
 *-------------------------------------------------------------------------*/

int NALU_HYPRE_LSI_SchwarzSetup(NALU_HYPRE_Solver solver, NALU_HYPRE_ParCSRMatrix A_csr,
                           NALU_HYPRE_ParVector b,   NALU_HYPRE_ParVector x )
{
   int               i, offset, total_recv_leng, *recv_lengths=NULL;
   int               *int_buf=NULL, mypid, nprocs, overlap_flag=1,*parray;
   int               *map=NULL, *map2=NULL, *row_partition=NULL,*parray2;
   double            *dble_buf=NULL;
   MH_Context        *context=NULL;
   MH_Matrix         *mh_mat=NULL;
   MPI_Comm          comm;
   NALU_HYPRE_LSI_Schwarz *sch_ptr = (NALU_HYPRE_LSI_Schwarz *) solver;

   /* --------------------------------------------------------- */
   /* get the row information in my processors                  */
   /* --------------------------------------------------------- */

   comm = sch_ptr->comm;
   MPI_Comm_rank(comm, &mypid);
   MPI_Comm_size(comm, &nprocs);
   NALU_HYPRE_ParCSRMatrixGetRowPartitioning(A_csr, &row_partition);

   /* --------------------------------------------------------- */
   /* convert the incoming CSR matrix into a MH matrix          */
   /* --------------------------------------------------------- */

   context = nalu_hypre_TAlloc(MH_Context, 1, NALU_HYPRE_MEMORY_HOST);
   context->comm = comm;
   context->globalEqns = row_partition[nprocs];
   context->partition = nalu_hypre_TAlloc(int, (nprocs+1), NALU_HYPRE_MEMORY_HOST);
   for (i=0; i<=nprocs; i++) context->partition[i] = row_partition[i];
   nalu_hypre_TFree( row_partition , NALU_HYPRE_MEMORY_HOST);
   mh_mat = nalu_hypre_TAlloc( MH_Matrix, 1, NALU_HYPRE_MEMORY_HOST);
   context->Amat = mh_mat;
   NALU_HYPRE_LSI_MLConstructMHMatrix(A_csr, mh_mat, comm,
                                 context->partition,context);
   sch_ptr->Nrows = mh_mat->Nrows;
   sch_ptr->mh_mat = mh_mat;

   /* --------------------------------------------------------- */
   /* compose the enlarged overlapped local matrix              */
   /* --------------------------------------------------------- */

   if ( overlap_flag )
   {
      NALU_HYPRE_LSI_DDIlutComposeOverlappedMatrix(mh_mat, &total_recv_leng,
            &recv_lengths, &int_buf, &dble_buf, &map, &map2,&offset,comm);
   }
   else
   {
      total_recv_leng = 0;
      recv_lengths = NULL;
      int_buf = NULL;
      dble_buf = NULL;
      map = NULL;
      map2 = NULL;
      parray  = nalu_hypre_TAlloc(int, nprocs , NALU_HYPRE_MEMORY_HOST);
      parray2 = nalu_hypre_TAlloc(int, nprocs , NALU_HYPRE_MEMORY_HOST);
      for ( i = 0; i < nprocs; i++ ) parray2[i] = 0;
      parray2[mypid] = mh_mat->Nrows;
      MPI_Allreduce(parray2,parray,nprocs,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
      offset = 0;
      for (i = 0; i < mypid; i++) offset += parray[i];
      nalu_hypre_TFree(parray, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(parray2, NALU_HYPRE_MEMORY_HOST);
   }

   /* --------------------------------------------------------- */
   /* perform decomposition on local matrix                     */
   /* --------------------------------------------------------- */

   NALU_HYPRE_LSI_SchwarzDecompose(sch_ptr,mh_mat,total_recv_leng,recv_lengths,
                              int_buf, dble_buf, map, map2, offset);

   /* --------------------------------------------------------- */
   /* clean up                                                  */
   /* --------------------------------------------------------- */

   nalu_hypre_TFree(map, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(map2, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(int_buf, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(dble_buf, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(recv_lengths, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(context->partition, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(context, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(mh_mat->rowptr, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(mh_mat->colnum, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(mh_mat->values, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(mh_mat->map, NALU_HYPRE_MEMORY_HOST);

   return 0;
}

/**************************************************************************/
/* function for doing Schwarz decomposition                               */
/**************************************************************************/

int NALU_HYPRE_LSI_SchwarzDecompose(NALU_HYPRE_LSI_Schwarz *sch_ptr,MH_Matrix *Amat,
           int total_recv_leng, int *recv_lengths, int *ext_ja,
           double *ext_aa, int *map, int *map2, int Noffset)
{
   int               i, j, k, nnz, *mat_ia, *mat_ja;
   int               **bmat_ia, **bmat_ja;
   int               mypid, *blk_size, index, **blk_indices, **aux_bmat_ia;
   int               ncnt, rownum, offset, Nrows, extNrows, **aux_bmat_ja;
   int               *tmp_blk_leng, *cols, rowleng;
   int               nblocks, col_ind, init_size, aux_nnz, max_blk_size;
   int               *tmp_indices, cur_off_row, length;
   double            *mat_aa, *vals, **aux_bmat_aa, **bmat_aa;

   /* --------------------------------------------------------- */
   /* fetch Schwarz parameters                                  */
   /* --------------------------------------------------------- */

   MPI_Comm_rank(sch_ptr->comm, &mypid);
   Nrows          = sch_ptr->Nrows;
   extNrows       = Nrows + total_recv_leng;
   sch_ptr->Nrows = Nrows;
   sch_ptr->extNrows = extNrows;

   /* --------------------------------------------------------- */
   /* adjust the off-processor row data                         */
   /* --------------------------------------------------------- */

   offset = 0;
   for ( i = 0; i < total_recv_leng; i++ )
   {
      for ( j = offset; j < offset+recv_lengths[i]; j++ )
      {
         index = ext_ja[j];
         if ( index >= Noffset && index < Noffset+Nrows )
            ext_ja[j] = index - Noffset;
         else
         {
            col_ind = NALU_HYPRE_LSI_Search(map, index, extNrows-Nrows);
            if ( col_ind >= 0 ) ext_ja[j] = map2[col_ind] + Nrows;
            else                ext_ja[j] = -1;
         }
      }
      offset += recv_lengths[i];
   }

   /* --------------------------------------------------------- */
   /* compose the initial blk_size information                  */
   /* and extend the each block for the overlap                 */
   /* (at the end blk_indices and bli_size contains the info)   */
   /* --------------------------------------------------------- */

   if ( sch_ptr->nblocks == 1 )
   {
      nblocks = 1;
      max_blk_size = extNrows;
      sch_ptr->blk_sizes   = nalu_hypre_TAlloc(int, nblocks , NALU_HYPRE_MEMORY_HOST);
      blk_size = sch_ptr->blk_sizes;
      blk_size[0] = extNrows;
   }
   else
   {
      if ( sch_ptr->nblocks != 0 )
      {
         nblocks  = sch_ptr->nblocks;
         sch_ptr->block_size = (Nrows + nblocks / 2) / nblocks;
      }
      else
      {
         nblocks  = (Nrows - sch_ptr->block_size / 2) / sch_ptr->block_size + 1;
         sch_ptr->nblocks = nblocks;
      }
      sch_ptr->blk_indices = nalu_hypre_TAlloc(int*, nblocks , NALU_HYPRE_MEMORY_HOST);
      sch_ptr->blk_sizes   = nalu_hypre_TAlloc(int, nblocks , NALU_HYPRE_MEMORY_HOST);
      blk_indices  = sch_ptr->blk_indices;
      blk_size     = sch_ptr->blk_sizes;
      tmp_blk_leng = nalu_hypre_TAlloc(int, nblocks , NALU_HYPRE_MEMORY_HOST);
      for ( i = 0; i < nblocks-1; i++ ) blk_size[i] = sch_ptr->block_size;
      blk_size[nblocks-1] = Nrows - sch_ptr->block_size * (nblocks - 1 );
      for ( i = 0; i < nblocks; i++ )
      {
         tmp_blk_leng[i] = 5 * blk_size[i] + 5;
         blk_indices[i] = nalu_hypre_TAlloc(int, tmp_blk_leng[i] , NALU_HYPRE_MEMORY_HOST);
         for (j = 0; j < blk_size[i]; j++)
            blk_indices[i][j] = sch_ptr->block_size * i + j;
      }
      max_blk_size = 0;
      for ( i = 0; i < nblocks; i++ )
      {
         init_size = blk_size[i];
         for ( j = 0; j < init_size; j++ )
         {
            rownum = blk_indices[i][j];
            cols = &(Amat->colnum[Amat->rowptr[rownum]]);
            vals = &(Amat->values[Amat->rowptr[rownum]]);
            rowleng = Amat->rowptr[rownum+1] - Amat->rowptr[rownum];
            if ( blk_size[i] + rowleng > tmp_blk_leng[i] )
            {
               tmp_indices = blk_indices[i];
               tmp_blk_leng[i] = 2 * ( blk_size[i] + rowleng ) + 2;
               blk_indices[i] = nalu_hypre_TAlloc(int, tmp_blk_leng[i] , NALU_HYPRE_MEMORY_HOST);
               for (k = 0; k < blk_size[i]; k++)
                  blk_indices[i][k] = tmp_indices[k];
               nalu_hypre_TFree(tmp_indices, NALU_HYPRE_MEMORY_HOST);
            }
            for ( k = 0; k < rowleng; k++ )
            {
               col_ind = cols[k];
               blk_indices[i][blk_size[i]++] = col_ind;
            }
         }
         nalu_hypre_qsort0(blk_indices[i], 0, blk_size[i]-1);
         ncnt = 0;
         for ( j = 1; j < blk_size[i]; j++ )
            if ( blk_indices[i][j] != blk_indices[i][ncnt] )
              blk_indices[i][++ncnt] = blk_indices[i][j];
         blk_size[i] = ncnt + 1;
         if ( blk_size[i] > max_blk_size ) max_blk_size = blk_size[i];
      }
      nalu_hypre_TFree(tmp_blk_leng, NALU_HYPRE_MEMORY_HOST);
   }

   /* --------------------------------------------------------- */
   /* compute the memory requirements for each block            */
   /* --------------------------------------------------------- */

   sch_ptr->bmat_ia = nalu_hypre_TAlloc(int*, nblocks , NALU_HYPRE_MEMORY_HOST);
   sch_ptr->bmat_ja = nalu_hypre_TAlloc(int*, nblocks , NALU_HYPRE_MEMORY_HOST);
   sch_ptr->bmat_aa = nalu_hypre_TAlloc(double*, nblocks , NALU_HYPRE_MEMORY_HOST);
   bmat_ia = sch_ptr->bmat_ia;
   bmat_ja = sch_ptr->bmat_ja;
   bmat_aa = sch_ptr->bmat_aa;
   if ( nblocks != 1 )
   {
      sch_ptr->aux_bmat_ia = nalu_hypre_TAlloc(int*, nblocks , NALU_HYPRE_MEMORY_HOST);
      sch_ptr->aux_bmat_ja = nalu_hypre_TAlloc(int*, nblocks , NALU_HYPRE_MEMORY_HOST);
      sch_ptr->aux_bmat_aa = nalu_hypre_TAlloc(double*, nblocks , NALU_HYPRE_MEMORY_HOST);
      aux_bmat_ia = sch_ptr->aux_bmat_ia;
      aux_bmat_ja = sch_ptr->aux_bmat_ja;
      aux_bmat_aa = sch_ptr->aux_bmat_aa;
   }
   else
   {
      aux_bmat_ia = NULL;
      aux_bmat_ja = NULL;
      aux_bmat_aa = NULL;
   }

   /* --------------------------------------------------------- */
   /* compose each block into sch_ptr                           */
   /* --------------------------------------------------------- */

   cols = nalu_hypre_TAlloc(int,  max_blk_size , NALU_HYPRE_MEMORY_HOST);
   vals = nalu_hypre_TAlloc(double,  max_blk_size , NALU_HYPRE_MEMORY_HOST);

   for ( i = 0; i < nblocks; i++ )
   {
      nnz = aux_nnz = offset = cur_off_row = 0;
      if ( nblocks > 1 ) length = blk_size[i];
      else               length = extNrows;
      for ( j = 0; j < length; j++ )
      {
         if ( nblocks > 1 ) rownum = blk_indices[i][j];
         else               rownum = j;
         if ( rownum < Nrows )
         {
            rowleng = 0;
            for ( k = Amat->rowptr[rownum]; k < Amat->rowptr[rownum+1]; k++ )
               cols[rowleng++] = Amat->colnum[k];
         }
         else
         {
            for ( k = cur_off_row; k < rownum-Nrows; k++ )
               offset += recv_lengths[k];
            cur_off_row = rownum - Nrows;
            rowleng = 0;
            for ( k = offset; k < offset+recv_lengths[cur_off_row]; k++ )
               if ( ext_ja[k] != -1 ) cols[rowleng++] = ext_ja[k];
         }
         for ( k = 0; k < rowleng; k++ )
         {
            if ( nblocks > 1 )
               index = NALU_HYPRE_LSI_Search( blk_indices[i], cols[k], blk_size[i]);
            else
               index = cols[k];
            if ( index >= 0 ) nnz++;
            else              aux_nnz++;
         }
      }
      bmat_ia[i] = nalu_hypre_TAlloc(int,  (length + 1) , NALU_HYPRE_MEMORY_HOST);
      bmat_ja[i] = nalu_hypre_TAlloc(int,  nnz , NALU_HYPRE_MEMORY_HOST);
      bmat_aa[i] = nalu_hypre_TAlloc(double,  nnz , NALU_HYPRE_MEMORY_HOST);
      mat_ia = bmat_ia[i];
      mat_ja = bmat_ja[i];
      mat_aa = bmat_aa[i];
      if ( nblocks > 1 )
      {
         aux_bmat_ia[i] = nalu_hypre_TAlloc(int,  (blk_size[i] + 1) , NALU_HYPRE_MEMORY_HOST);
         aux_bmat_ja[i] = nalu_hypre_TAlloc(int,  aux_nnz , NALU_HYPRE_MEMORY_HOST);
         aux_bmat_aa[i] = nalu_hypre_TAlloc(double,  aux_nnz , NALU_HYPRE_MEMORY_HOST);
      }

      /* ------------------------------------------------------ */
      /* load the submatrices                                   */
      /* ------------------------------------------------------ */

      nnz = aux_nnz = offset = cur_off_row = 0;
      mat_ia[0] = 0;
      if ( nblocks > 1 ) aux_bmat_ia[i][0] = 0;

      for ( j = 0; j < blk_size[i]; j++ )
      {
         if ( nblocks > 1 ) rownum = blk_indices[i][j];
         else               rownum = j;
         if ( rownum < Nrows )
         {
            rowleng = 0;
            for ( k = Amat->rowptr[rownum]; k < Amat->rowptr[rownum+1]; k++ )
            {
               vals[rowleng]   = Amat->values[k];
               cols[rowleng++] = Amat->colnum[k];
            }
         }
         else
         {
            for ( k = cur_off_row; k < rownum-Nrows; k++ )
            {
               offset += recv_lengths[k];
            }
            cur_off_row = rownum - Nrows;
            rowleng = 0;
            for ( k = offset; k < offset+recv_lengths[cur_off_row]; k++ )
            {
               if ( ext_ja[k] != -1 )
               {
                  cols[rowleng] = ext_ja[k];
                  vals[rowleng++] = ext_aa[k];
               }
            }
         }
         for ( k = 0; k < rowleng; k++ )
         {
            if ( nblocks > 1 )
               index = NALU_HYPRE_LSI_Search( blk_indices[i], cols[k], blk_size[i]);
            else index = cols[k];
            if ( index >= 0 )
            {
               mat_ja[nnz] = index;
               mat_aa[nnz++] = vals[k];
            }
            else
            {
               aux_bmat_ja[i][aux_nnz] = cols[k];
               aux_bmat_aa[i][aux_nnz++] = vals[k];
            }
         }
         mat_ia[j+1] = nnz;
         if ( nblocks > 1 ) aux_bmat_ia[i][j+1] = aux_nnz;
      }
      for ( j = 0; j < mat_ia[blk_size[i]]; j++ )
         if ( mat_ja[j] < 0 || mat_ja[j] >= length )
            printf("block %d has index %d\n", i, mat_ja[j]);
   }

   nalu_hypre_TFree(cols, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(vals, NALU_HYPRE_MEMORY_HOST);

   /* --------------------------------------------------------- */
   /* decompose each block                                      */
   /* --------------------------------------------------------- */

   NALU_HYPRE_LSI_ILUTDecompose( sch_ptr );

   return 0;
}

/*************************************************************************/
/* function for doing ILUT decomposition                                 */
/*************************************************************************/

int NALU_HYPRE_LSI_ILUTDecompose( NALU_HYPRE_LSI_Schwarz *sch_ptr )
{

   int    i, j, k, blk, nrows, rleng, *cols, *track_array, track_leng;
   int    nblocks, max_blk_size, *mat_ia, *mat_ja, *new_ia, *new_ja;
   int    index, first, sortcnt, *sortcols, Lcount, Ucount, nnz, new_nnz;
   int    colIndex, mypid, output_level, printflag, printflag2;
   double fillin, *vals, *dble_buf, *rowNorms, *diagonal, *mat_aa, *new_aa;
   double *sortvals, ddata, tau, rel_tau, absval;

   /* --------------------------------------------------------- */
   /* preparation phase                                         */
   /* --------------------------------------------------------- */

   MPI_Comm_rank(sch_ptr->comm, &mypid);
   output_level = sch_ptr->output_level;
   nblocks = sch_ptr->nblocks;
   max_blk_size = 0;
   for ( blk = 0; blk < nblocks; blk++ )
      if ( sch_ptr->blk_sizes[blk] > max_blk_size )
         max_blk_size = sch_ptr->blk_sizes[blk];
   fillin = sch_ptr->fillin;
   tau    = sch_ptr->threshold;

   track_array = nalu_hypre_TAlloc(int,  max_blk_size , NALU_HYPRE_MEMORY_HOST);
   sortcols    = nalu_hypre_TAlloc(int,  max_blk_size , NALU_HYPRE_MEMORY_HOST);
   sortvals    = nalu_hypre_TAlloc(double,  max_blk_size , NALU_HYPRE_MEMORY_HOST);
   dble_buf    = nalu_hypre_TAlloc(double,  max_blk_size , NALU_HYPRE_MEMORY_HOST);
   diagonal    = nalu_hypre_TAlloc(double,  max_blk_size , NALU_HYPRE_MEMORY_HOST);
   rowNorms    = nalu_hypre_TAlloc(double,  max_blk_size , NALU_HYPRE_MEMORY_HOST);
   for ( i = 0; i < max_blk_size; i++ ) dble_buf[i] = 0.0;

   /* --------------------------------------------------------- */
   /* process the rows                                          */
   /* --------------------------------------------------------- */

   printflag = nblocks / 10 + 1;
   for ( blk = 0; blk < nblocks; blk++ )
   {
      if ( output_level > 0 && blk % printflag == 0 && blk != 0 )
         printf("%4d : Schwarz : processing block %6d (%6d)\n",mypid,blk,nblocks);
      mat_ia  = sch_ptr->bmat_ia[blk];
      mat_ja  = sch_ptr->bmat_ja[blk];
      mat_aa  = sch_ptr->bmat_aa[blk];
      nrows   = sch_ptr->blk_sizes[blk];
      nnz     = mat_ia[nrows];
      new_nnz = (int) (nnz * ( 1.0 + fillin ));
      new_ia  = nalu_hypre_TAlloc(int,  (nrows + 1 ) , NALU_HYPRE_MEMORY_HOST);
      new_ja  = nalu_hypre_TAlloc(int,  new_nnz , NALU_HYPRE_MEMORY_HOST);
      new_aa  = nalu_hypre_TAlloc(double,  new_nnz , NALU_HYPRE_MEMORY_HOST);
      nnz       = 0;
      new_ia[0] = nnz;
      for ( i = 0; i < nrows; i++ )
      {
         index = mat_ia[i];
         cols = &(mat_ja[index]);
         vals = &(mat_aa[index]);
         rleng = mat_ia[i+1] - index;
         ddata = 0.0;
         for ( j = 0; j < rleng; j++ ) ddata += habs( vals[j] );
         rowNorms[i] = ddata;
      }
      printflag2 = nrows / 10 + 1;
      for ( i = 0; i < nrows; i++ )
      {
         if ( output_level > 0 && i % printflag2 == 0 && i != 0 )
            printf("%4d : Schwarz : block %6d row %6d (%6d)\n",mypid,blk,
                   i, nrows);
         track_leng = 0;
         index = mat_ia[i];
         cols = &(mat_ja[index]);
         vals = &(mat_aa[index]);
         rleng = mat_ia[i+1] - index;
         for ( j = 0; j < rleng; j++ )
         {
            dble_buf[cols[j]] = vals[j];
            track_array[track_leng++] = cols[j];
         }
         Lcount = Ucount = first = 0;
         first  = nrows;
         for ( j = 0; j < track_leng; j++ )
         {
            index = track_array[j];
            if ( dble_buf[index] != 0 )
            {
               if ( index < i ) Lcount++;
               else if ( index > i ) Ucount++;
               else if ( index == i ) diagonal[i] = dble_buf[index];
               if ( index < first ) first = index;
            }
         }
         Lcount  = Lcount * fillin;
         Ucount  = Ucount * fillin;
         rel_tau = tau * rowNorms[i];
         for ( j = first; j < i; j++ )
         {
            if ( habs(dble_buf[j]) > rel_tau )
            {
               ddata = dble_buf[j] / diagonal[j];
               for ( k = new_ia[j]; k < new_ia[j+1]; k++ )
               {
                  colIndex = new_ja[k];
                  if ( colIndex > j )
                  {
                     if ( dble_buf[colIndex] != 0.0 )
                        dble_buf[colIndex] -= (ddata * new_aa[k]);
                     else
                     {
                        dble_buf[colIndex] = - (ddata * new_aa[k]);
                        if ( dble_buf[colIndex] != 0.0 )
                           track_array[track_leng++] = colIndex;
                     }
                  }
               }
               dble_buf[j] = ddata;
            }
            else dble_buf[j] = 0.0;
         }
         for ( j = 0; j < rleng; j++ )
         {
            vals[j] = dble_buf[cols[j]];
            if ( cols[j] != i ) dble_buf[cols[j]] = 0.0;
         }
         sortcnt = 0;
         for ( j = 0; j < track_leng; j++ )
         {
            index = track_array[j];
            if ( index < i )
            {
               absval = habs( dble_buf[index] );
               if ( absval > rel_tau )
               {
                  sortcols[sortcnt] = index;
                  sortvals[sortcnt++] = absval * rowNorms[index];
               }
               else dble_buf[index] = 0.0;
            }
         }
         if ( sortcnt > Lcount )
         {
            NALU_HYPRE_LSI_SplitDSort(sortvals,sortcnt,sortcols,Lcount);
            for ( j = Lcount; j < sortcnt; j++ ) dble_buf[sortcols[j]] = 0.0;
         }
         for ( j = 0; j < rleng; j++ )
         {
            if ( cols[j] < i && vals[j] != 0.0 )
            {
               new_aa[nnz] = vals[j];
               new_ja[nnz++] = cols[j];
            }
         }
         for ( j = 0; j < track_leng; j++ )
         {
            index = track_array[j];
            if ( index < i && dble_buf[index] != 0.0 )
            {
               new_aa[nnz] = dble_buf[index];
               new_ja[nnz++] = index;
               dble_buf[index] = 0.0;
            }
         }
         diagonal[i] = dble_buf[i];
         if ( habs(diagonal[i]) < 1.0e-12 ) diagonal[i] = 1.0E-12;
         new_aa[nnz] = diagonal[i];
         new_ja[nnz++] = i;
         sortcnt = 0;
         for ( j = 0; j < track_leng; j++ )
         {
            index = track_array[j];
            if ( index > i )
            {
               absval = habs( dble_buf[index] );
               if ( absval > rel_tau )
               {
                  sortcols[sortcnt] = index;
                  sortvals[sortcnt++] = absval * rowNorms[index];
               }
               else dble_buf[index] = 0.0;
            }
         }
         if ( sortcnt > Ucount )
         {
            NALU_HYPRE_LSI_SplitDSort(sortvals,sortcnt,sortcols,Ucount);
            for ( j = Ucount; j < sortcnt; j++ ) dble_buf[sortcols[j]] = 0.0;
         }
         for ( j = 0; j < rleng; j++ )
         {
            if ( cols[j] > i && vals[j] != 0.0 )
            {
               new_aa[nnz] = vals[j];
               new_ja[nnz++] = cols[j];
            }
         }
         for ( j = 0; j < track_leng; j++ )
         {
            index = track_array[j];
            if ( index > i && dble_buf[index] != 0.0 )
            {
               new_aa[nnz] = dble_buf[index];
               new_ja[nnz++] = index;
               dble_buf[index] = 0.0;
            }
         }
         dble_buf[i] = 0.0;
         new_ia[i+1] = nnz;
      }
      nalu_hypre_TFree(mat_ia, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(mat_ja, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(mat_aa, NALU_HYPRE_MEMORY_HOST);
      sch_ptr->bmat_ia[blk] = new_ia;
      sch_ptr->bmat_ja[blk] = new_ja;
      sch_ptr->bmat_aa[blk] = new_aa;
      if ( nnz > new_nnz )
      {
         printf("ERROR : nnz (%d) > new_nnz (%d) \n", nnz, new_nnz);
         exit(1);
      }
      for ( j = 0; j < new_ia[sch_ptr->blk_sizes[blk]]; j++ )
      {
         if ( new_ja[j] < 0 || new_ja[j] >= sch_ptr->blk_sizes[blk] )
         {
            printf("(2) block %d has index %d\n", blk, new_ja[j]);
            exit(1);
         }
      }
   }

   nalu_hypre_TFree(track_array, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(dble_buf, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(diagonal, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(rowNorms, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(sortcols, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(sortvals, NALU_HYPRE_MEMORY_HOST);

   return 0;
}

