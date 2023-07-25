/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 *****************************************************************************/

#include "_nalu_hypre_parcsr_ls.h"

/*==========================================================================*/
/*==========================================================================*/
/**
  Augments measures by some random value between 0 and 1.

  {\bf Input files:}
  _nalu_hypre_parcsr_ls.h

  @return Error code.

  @param S [IN]
  parent graph matrix in CSR format
  @param measure_array [IN/OUT]
  measures assigned to each node of the parent graph

  @see nalu_hypre_AMGIndepSet */
/*--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_BoomerAMGIndepSetInit( nalu_hypre_ParCSRMatrix *S,
                             NALU_HYPRE_Real         *measure_array,
                             NALU_HYPRE_Int           seq_rand)
{
   nalu_hypre_CSRMatrix *S_diag = nalu_hypre_ParCSRMatrixDiag(S);
   MPI_Comm         comm = nalu_hypre_ParCSRMatrixComm(S);
   NALU_HYPRE_Int        S_num_nodes = nalu_hypre_CSRMatrixNumRows(S_diag);
   NALU_HYPRE_BigInt     big_i;
   NALU_HYPRE_Int        i, my_id;
   NALU_HYPRE_Int        ierr = 0;

   nalu_hypre_MPI_Comm_rank(comm, &my_id);
   i = 2747 + my_id;
   if (seq_rand)
   {
      i = 2747;
   }
   nalu_hypre_SeedRand(i);
   if (seq_rand)
   {
      for (big_i = 0; big_i < nalu_hypre_ParCSRMatrixFirstRowIndex(S); big_i++)
      {
         nalu_hypre_Rand();
      }
   }
   for (i = 0; i < S_num_nodes; i++)
   {
      measure_array[i] += nalu_hypre_Rand();
   }

   return (ierr);
}

/*==========================================================================*/
/*==========================================================================*/
/**
  Select an independent set from a graph.  This graph is actually a
  subgraph of some parent graph.  The parent graph is described as a
  matrix in compressed sparse row format, where edges in the graph are
  represented by nonzero matrix coefficients (zero coefficients are
  ignored).  A positive measure is given for each node in the
  subgraph, and this is used to pick the independent set.  A measure
  of zero must be given for all other nodes in the parent graph.  The
  subgraph is a collection of nodes in the parent graph.

  Positive entries in the `IS\_marker' array indicate nodes in the
  independent set.  All other entries are zero.

  The algorithm proceeds by first setting all nodes in `graph\_array'
  to be in the independent set.  Nodes are then removed from the
  independent set by simply comparing the measures of adjacent nodes.

  {\bf Input files:}
  _nalu_hypre_parcsr_ls.h

  @return Error code.

  @param S [IN]
  parent graph matrix in CSR format
  @param measure_array [IN]
  measures assigned to each node of the parent graph
  @param graph_array [IN]
  node numbers in the subgraph to be partitioned
  @param graph_array_size [IN]
  number of nodes in the subgraph to be partitioned
  @param IS_marker [IN/OUT]
  marker array for independent set

  @see nalu_hypre_InitAMGIndepSet */
/*--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_BoomerAMGIndepSet( nalu_hypre_ParCSRMatrix *S,
                         NALU_HYPRE_Real         *measure_array,
                         NALU_HYPRE_Int          *graph_array,
                         NALU_HYPRE_Int           graph_array_size,
                         NALU_HYPRE_Int          *graph_array_offd,
                         NALU_HYPRE_Int           graph_array_offd_size,
                         NALU_HYPRE_Int          *IS_marker,
                         NALU_HYPRE_Int          *IS_marker_offd     )
{
   nalu_hypre_CSRMatrix *S_diag   = nalu_hypre_ParCSRMatrixDiag(S);
   NALU_HYPRE_Int       *S_diag_i = nalu_hypre_CSRMatrixI(S_diag);
   NALU_HYPRE_Int       *S_diag_j = nalu_hypre_CSRMatrixJ(S_diag);
   nalu_hypre_CSRMatrix *S_offd   = nalu_hypre_ParCSRMatrixOffd(S);
   NALU_HYPRE_Int       *S_offd_i = nalu_hypre_CSRMatrixI(S_offd);
   NALU_HYPRE_Int       *S_offd_j = NULL;

   NALU_HYPRE_Int        local_num_vars = nalu_hypre_CSRMatrixNumRows(S_diag);
   NALU_HYPRE_Int        i, j, ig, jS, jj;

   /*-------------------------------------------------------
    * Initialize IS_marker by putting all nodes in
    * the independent set.
    *-------------------------------------------------------*/

   if (nalu_hypre_CSRMatrixNumCols(S_offd))
   {
      S_offd_j = nalu_hypre_CSRMatrixJ(S_offd);
   }

   for (ig = 0; ig < graph_array_size; ig++)
   {
      i = graph_array[ig];
      if (measure_array[i] > 1)
      {
         IS_marker[i] = 1;
      }
   }
   for (ig = 0; ig < graph_array_offd_size; ig++)
   {
      i = graph_array_offd[ig];
      if (measure_array[i + local_num_vars] > 1)
      {
         IS_marker_offd[i] = 1;
      }
   }

   /*-------------------------------------------------------
    * Remove nodes from the initial independent set
    *-------------------------------------------------------*/

   for (ig = 0; ig < graph_array_size; ig++)
   {
      i = graph_array[ig];
      if (measure_array[i] > 1)
      {
         for (jS = S_diag_i[i]; jS < S_diag_i[i + 1]; jS++)
         {
            j = S_diag_j[jS];
            if (j < 0)
            {
               j = -j - 1;
            }

            /* only consider valid graph edges */
            /* if ( (measure_array[j] > 1) && (S_diag_data[jS]) ) */
            if (measure_array[j] > 1)
            {
               if (measure_array[i] > measure_array[j])
               {
                  IS_marker[j] = 0;
               }
               else if (measure_array[j] > measure_array[i])
               {
                  IS_marker[i] = 0;
               }
            }
         }
         for (jS = S_offd_i[i]; jS < S_offd_i[i + 1]; jS++)
         {
            jj = S_offd_j[jS];
            if (jj < 0)
            {
               jj = -jj - 1;
            }
            j = local_num_vars + jj;

            /* only consider valid graph edges */
            /* if ( (measure_array[j] > 1) && (S_offd_data[jS]) ) */
            if (measure_array[j] > 1)
            {
               if (measure_array[i] > measure_array[j])
               {
                  IS_marker_offd[jj] = 0;
               }
               else if (measure_array[j] > measure_array[i])
               {
                  IS_marker[i] = 0;
               }
            }
         }
      }
   }

   return nalu_hypre_error_flag;
}

