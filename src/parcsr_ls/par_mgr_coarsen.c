/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_nalu_hypre_parcsr_ls.h"
#include "par_mgr.h"

/*--------------------------------------------------------------------------
 * nalu_hypre_MGRCoarseParms
 *
 * Computes the fine and coarse partitioning arrays at once.
 *
 * TODO: Generate the dof_func array as in nalu_hypre_BoomerAMGCoarseParms
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_MGRCoarseParms(MPI_Comm          comm,
                     NALU_HYPRE_Int         num_rows,
                     nalu_hypre_IntArray   *CF_marker,
                     NALU_HYPRE_BigInt     *row_starts_cpts,
                     NALU_HYPRE_BigInt     *row_starts_fpts)
{
   NALU_HYPRE_Int     num_cpts;
   NALU_HYPRE_Int     num_fpts;

   NALU_HYPRE_BigInt  sbuffer_recv[2];
   NALU_HYPRE_BigInt  sbuffer_send[2];

   /* Count number of Coarse points */
   nalu_hypre_IntArrayCount(CF_marker, 1, &num_cpts);

   /* Count number of Fine points */
   nalu_hypre_IntArrayCount(CF_marker, -1, &num_fpts);

   /* Scan global starts */
   sbuffer_send[0] = (NALU_HYPRE_BigInt) num_cpts;
   sbuffer_send[1] = (NALU_HYPRE_BigInt) num_fpts;
   nalu_hypre_MPI_Scan(&sbuffer_send, &sbuffer_recv, 2, NALU_HYPRE_MPI_BIG_INT, nalu_hypre_MPI_SUM, comm);

   /* First points in next processor's range */
   row_starts_cpts[1] = sbuffer_recv[0];
   row_starts_fpts[1] = sbuffer_recv[1];

   /* First points in current processor's range */
   row_starts_cpts[0] = row_starts_cpts[1] - sbuffer_send[0];
   row_starts_fpts[0] = row_starts_fpts[1] - sbuffer_send[1];

   return nalu_hypre_error_flag;
}
