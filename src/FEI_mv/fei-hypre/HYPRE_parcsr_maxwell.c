/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

#include "utilities/_nalu_hypre_utilities.h"
#include "NALU_HYPRE.h"
#include "IJ_mv/NALU_HYPRE_IJ_mv.h"
#include "parcsr_mv/NALU_HYPRE_parcsr_mv.h"
#include "parcsr_mv/_nalu_hypre_parcsr_mv.h"
#include "parcsr_ls/NALU_HYPRE_parcsr_ls.h"

#include "NALU_HYPRE_FEI.h"

/*--------------------------------------------------------------------------
 * nalu_hypre_CotreeData
 *--------------------------------------------------------------------------*/

typedef struct
{
   int      max_iter;
   double   tol;
   nalu_hypre_ParCSRMatrix *Aee;
   nalu_hypre_ParCSRMatrix *Att;
   nalu_hypre_ParCSRMatrix *Atc;
   nalu_hypre_ParCSRMatrix *Act;
   nalu_hypre_ParCSRMatrix *Acc;
   nalu_hypre_ParCSRMatrix *Gen;
   nalu_hypre_ParCSRMatrix *Gc;
   nalu_hypre_ParCSRMatrix *Gt;
   nalu_hypre_ParCSRMatrix *Gtinv;
   nalu_hypre_ParVector    *w;
} nalu_hypre_CotreeData;

/******************************************************************************
 *
 * NALU_HYPRE_ParCSRCotree interface
 *
 *****************************************************************************/

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRCotreeCreate
 *--------------------------------------------------------------------------*/

int NALU_HYPRE_ParCSRCotreeCreate(MPI_Comm comm, NALU_HYPRE_Solver *solver)
{
   nalu_hypre_CotreeData *cotree_data;
   void             *void_data;

   cotree_data = nalu_hypre_CTAlloc(nalu_hypre_CotreeData,  1, NALU_HYPRE_MEMORY_HOST);
   void_data = (void *) cotree_data;
   *solver = (NALU_HYPRE_Solver) void_data;

   (cotree_data -> Aee)                = NULL;
   (cotree_data -> Acc)                = NULL;
   (cotree_data -> Act)                = NULL;
   (cotree_data -> Atc)                = NULL;
   (cotree_data -> Att)                = NULL;
   (cotree_data -> Gen)                = NULL;
   (cotree_data -> Gc)                 = NULL;
   (cotree_data -> Gt)                 = NULL;
   (cotree_data -> Gtinv)              = NULL;
   (cotree_data -> tol)                = 1.0e-06;
   (cotree_data -> max_iter)           = 1000;
   (cotree_data -> w)                  = NULL;
   return 0;
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRCotreeDestroy
 *--------------------------------------------------------------------------*/

int NALU_HYPRE_ParCSRCotreeDestroy(NALU_HYPRE_Solver solver)
{
   void             *cotree_vdata = (void *) solver;
   nalu_hypre_CotreeData *cotree_data = (nalu_hypre_CotreeData *) cotree_vdata;

   if (cotree_data)
   {
      nalu_hypre_TFree(cotree_data, NALU_HYPRE_MEMORY_HOST);
      if ((cotree_data->w) != NULL)
      {
         nalu_hypre_ParVectorDestroy(cotree_data->w);
         cotree_data->w = NULL;
      }
      if ((cotree_data->Acc) != NULL)
      {
         nalu_hypre_ParCSRMatrixDestroy(cotree_data->Acc);
         cotree_data->Acc = NULL;
      }
      if ((cotree_data->Act) != NULL)
      {
         nalu_hypre_ParCSRMatrixDestroy(cotree_data->Act);
         cotree_data->Act = NULL;
      }
      if ((cotree_data->Atc) != NULL)
      {
         nalu_hypre_ParCSRMatrixDestroy(cotree_data->Atc);
         cotree_data->Atc = NULL;
      }
      if ((cotree_data->Att) != NULL)
      {
         nalu_hypre_ParCSRMatrixDestroy(cotree_data->Att);
         cotree_data->Att = NULL;
      }
      if ((cotree_data->Gc) != NULL)
      {
         nalu_hypre_ParCSRMatrixDestroy(cotree_data->Gc);
         cotree_data->Gc = NULL;
      }
      if ((cotree_data->Gt) != NULL)
      {
         nalu_hypre_ParCSRMatrixDestroy(cotree_data->Gt);
         cotree_data->Gt = NULL;
      }
      if ((cotree_data->Gtinv) != NULL)
      {
         nalu_hypre_ParCSRMatrixDestroy(cotree_data->Gtinv);
         cotree_data->Gtinv = NULL;
      }
   }
   return 0;
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRCotreeSetup
 *--------------------------------------------------------------------------*/

int NALU_HYPRE_ParCSRCotreeSetup(NALU_HYPRE_Solver solver, NALU_HYPRE_ParCSRMatrix A,
                            NALU_HYPRE_ParVector b, NALU_HYPRE_ParVector x)
{
   int           *partition, *new_partition, nprocs, *tindices, ii;
   void *vsolver = (void *) solver;
/*
   void *vA      = (void *) A;
   void *vb      = (void *) b;
   void *vx      = (void *) x;
*/
   nalu_hypre_CotreeData   *cotree_data = (nalu_hypre_CotreeData *) vsolver;
   nalu_hypre_ParCSRMatrix **submatrices;
   nalu_hypre_ParVector    *new_vector;
   MPI_Comm           comm;

   cotree_data->Aee = (nalu_hypre_ParCSRMatrix *) A;
   nalu_hypre_ParCSRMatrixGenSpanningTree(cotree_data->Gen, &tindices, 1);
   submatrices = nalu_hypre_TAlloc(nalu_hypre_ParCSRMatrix *, 1, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_ParCSRMatrixExtractSubmatrices(cotree_data->Aee, tindices,
                                        &submatrices);
   cotree_data->Att = submatrices[0];
   cotree_data->Atc = submatrices[1];
   cotree_data->Act = submatrices[2];
   cotree_data->Acc = submatrices[3];

   nalu_hypre_ParCSRMatrixExtractRowSubmatrices(cotree_data->Gen, tindices,
                                           &submatrices);
   cotree_data->Gt = submatrices[0];
   cotree_data->Gc = submatrices[1];
   nalu_hypre_TFree(submatrices, NALU_HYPRE_MEMORY_HOST);

   comm = nalu_hypre_ParCSRMatrixComm((nalu_hypre_ParCSRMatrix *) A);
   MPI_Comm_size(comm, &nprocs);
   partition = nalu_hypre_ParVectorPartitioning((nalu_hypre_ParVector *) b);
   new_partition = nalu_hypre_TAlloc(int, (nprocs+1) , NALU_HYPRE_MEMORY_HOST);
   for (ii = 0; ii <= nprocs; ii++) new_partition[ii] = partition[ii];
   /*   partition = nalu_hypre_ParVectorPartitioning((nalu_hypre_ParVector *) b);  */
   new_vector = nalu_hypre_ParVectorCreate(nalu_hypre_ParVectorComm((nalu_hypre_ParVector *)b),
         (int) nalu_hypre_ParVectorGlobalSize((nalu_hypre_ParVector *) b),
                   new_partition);
   nalu_hypre_ParVectorInitialize(new_vector);
   cotree_data->w = new_vector;
   return 0;
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRCotreeSolve
 * (1) Given initial E and f, compute residual R
 * (2) Use GMRES to solve for cotree system given Rc with preconditioner
 *     (a) (I + FF^t) solve
 *     (b) preconditioned \hat{Acc} solve
 *     (c) (I + FF^t) solve
 * (3) update E
 *--------------------------------------------------------------------------
 * (I + FF^t) x = y   where F = G_c G_t^{-1}
 * (1) w2 = G_c^t y
 * (2) Poisson solve A z1 = w2
 * (3) z2 = y - F G_t z1
 *--------------------------------------------------------------------------*/

int NALU_HYPRE_ParCSRCotreeSolve(NALU_HYPRE_Solver solver, NALU_HYPRE_ParCSRMatrix A,
                            NALU_HYPRE_ParVector b, NALU_HYPRE_ParVector x)
{
   void *cotree_vdata = (void *) solver;
   nalu_hypre_CotreeData *cotree_data  = (nalu_hypre_CotreeData *)cotree_vdata;
   cotree_data->w = NULL;
   return 0;
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRCotreeSetTol
 *--------------------------------------------------------------------------*/

int NALU_HYPRE_ParCSRCotreeSetTol(NALU_HYPRE_Solver solver, double tol)
{
   return 0;
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRCotreeSetMaxIter
 *--------------------------------------------------------------------------*/

int NALU_HYPRE_ParCSRCotreeSetMaxIter(NALU_HYPRE_Solver solver, int max_iter)
{
   return 0;
}

