/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#ifndef nalu_hypre_PAR_AMGDD_COMP_GRID_HEADER
#define nalu_hypre_PAR_AMGDD_COMP_GRID_HEADER

/*--------------------------------------------------------------------------
 * nalu_hypre_AMGDDCommPkg
 *--------------------------------------------------------------------------*/

typedef struct
{
   // Info needed for subsequent psi_c residual communication
   NALU_HYPRE_Int           num_levels;     // levels in the amg hierarchy
   NALU_HYPRE_Int          *num_send_procs; // number of send procs to communicate with
   NALU_HYPRE_Int          *num_recv_procs; // number of recv procs to communicate with

   NALU_HYPRE_Int         **send_procs; // list of send procs
   NALU_HYPRE_Int         **recv_procs; // list of recv procs

   NALU_HYPRE_Int         **send_buffer_size; // size of send buffer on each level for each proc
   NALU_HYPRE_Int         **recv_buffer_size; // size of recv buffer on each level for each proc

   NALU_HYPRE_Int        ***num_send_nodes; // number of nodes to send on each composite level
   NALU_HYPRE_Int        ***num_recv_nodes; // number of nodes to recv on each composite level

   NALU_HYPRE_Int       ****send_flag; // flags which nodes to send after composite grid is built
   NALU_HYPRE_Int
   ****recv_map; // mapping from recv buffer to appropriate local indices on each comp grid
   NALU_HYPRE_Int       ****recv_red_marker; // marker indicating a redundant recv

} nalu_hypre_AMGDDCommPkg;

/*--------------------------------------------------------------------------
 * Accessor functions for the Comp Grid Comm Pkg structure
 *--------------------------------------------------------------------------*/

#define nalu_hypre_AMGDDCommPkgNumLevels(compGridCommPkg)      ((compGridCommPkg) -> num_levels)
#define nalu_hypre_AMGDDCommPkgNumSendProcs(compGridCommPkg)   ((compGridCommPkg) -> num_send_procs)
#define nalu_hypre_AMGDDCommPkgNumRecvProcs(compGridCommPkg)   ((compGridCommPkg) -> num_recv_procs)
#define nalu_hypre_AMGDDCommPkgSendProcs(compGridCommPkg)      ((compGridCommPkg) -> send_procs)
#define nalu_hypre_AMGDDCommPkgRecvProcs(compGridCommPkg)      ((compGridCommPkg) -> recv_procs)
#define nalu_hypre_AMGDDCommPkgSendBufferSize(compGridCommPkg) ((compGridCommPkg) -> send_buffer_size)
#define nalu_hypre_AMGDDCommPkgRecvBufferSize(compGridCommPkg) ((compGridCommPkg) -> recv_buffer_size)
#define nalu_hypre_AMGDDCommPkgNumSendNodes(compGridCommPkg)   ((compGridCommPkg) -> num_send_nodes)
#define nalu_hypre_AMGDDCommPkgNumRecvNodes(compGridCommPkg)   ((compGridCommPkg) -> num_recv_nodes)
#define nalu_hypre_AMGDDCommPkgSendFlag(compGridCommPkg)       ((compGridCommPkg) -> send_flag)
#define nalu_hypre_AMGDDCommPkgRecvMap(compGridCommPkg)        ((compGridCommPkg) -> recv_map)
#define nalu_hypre_AMGDDCommPkgRecvRedMarker(compGridCommPkg)  ((compGridCommPkg) -> recv_red_marker)

/*--------------------------------------------------------------------------
 * AMGDDCompGridMatrix (basically a coupled collection of CSR matrices)
 *--------------------------------------------------------------------------*/

typedef struct
{
   nalu_hypre_CSRMatrix      *owned_diag; // Domain: owned domain of mat. Range: owned range of mat.
   nalu_hypre_CSRMatrix      *owned_offd; // Domain: nonowned domain of mat. Range: owned range of mat.
   nalu_hypre_CSRMatrix
   *nonowned_diag; // Domain: nonowned domain of mat. Range: nonowned range of mat.
   nalu_hypre_CSRMatrix      *nonowned_offd; // Domain: owned domain of mat. Range: nonowned range of mat.

   nalu_hypre_CSRMatrix      *real_real;  // Domain: nonowned real. Range: nonowned real.
   nalu_hypre_CSRMatrix      *real_ghost; // Domain: nonowned ghost. Range: nonowned real.

   NALU_HYPRE_Int             owns_owned_matrices;
   NALU_HYPRE_Int             owns_offd_col_indices;

} nalu_hypre_AMGDDCompGridMatrix;

/*--------------------------------------------------------------------------
 * Accessor functions for the AMGDDCompGridMatrix structure
 *--------------------------------------------------------------------------*/

#define nalu_hypre_AMGDDCompGridMatrixOwnedDiag(matrix)          ((matrix) -> owned_diag)
#define nalu_hypre_AMGDDCompGridMatrixOwnedOffd(matrix)          ((matrix) -> owned_offd)
#define nalu_hypre_AMGDDCompGridMatrixNonOwnedDiag(matrix)       ((matrix) -> nonowned_diag)
#define nalu_hypre_AMGDDCompGridMatrixNonOwnedOffd(matrix)       ((matrix) -> nonowned_offd)
#define nalu_hypre_AMGDDCompGridMatrixRealReal(matrix)           ((matrix) -> real_real)
#define nalu_hypre_AMGDDCompGridMatrixRealGhost(matrix)          ((matrix) -> real_ghost)
#define nalu_hypre_AMGDDCompGridMatrixOwnsOwnedMatrices(matrix)  ((matrix) -> owns_owned_matrices)
#define nalu_hypre_AMGDDCompGridMatrixOwnsOffdColIndices(matrix) ((matrix) -> owns_offd_col_indices)

/*--------------------------------------------------------------------------
 * AMGDDCompGridVector
 *--------------------------------------------------------------------------*/

typedef struct
{
   nalu_hypre_Vector         *owned_vector;    // Original on-processor points (should be ordered)
   nalu_hypre_Vector         *nonowned_vector; // Off-processor points (not ordered)

   NALU_HYPRE_Int             num_real;
   NALU_HYPRE_Int             owns_owned_vector;

} nalu_hypre_AMGDDCompGridVector;

/*--------------------------------------------------------------------------
 * Accessor functions for the AMGDDCompGridVector structure
 *--------------------------------------------------------------------------*/

#define nalu_hypre_AMGDDCompGridVectorOwned(matrix)           ((matrix) -> owned_vector)
#define nalu_hypre_AMGDDCompGridVectorNonOwned(matrix)        ((matrix) -> nonowned_vector)
#define nalu_hypre_AMGDDCompGridVectorNumReal(vector)         ((vector) -> num_real)
#define nalu_hypre_AMGDDCompGridVectorOwnsOwnedVector(matrix) ((matrix) -> owns_owned_vector)

/*--------------------------------------------------------------------------
 * nalu_hypre_AMGDDCompGrid
 *--------------------------------------------------------------------------*/

typedef struct
{
   NALU_HYPRE_Int             level;
   NALU_HYPRE_MemoryLocation  memory_location;   /* memory location of matrices/vectors */

   NALU_HYPRE_Int             first_global_index;
   NALU_HYPRE_Int             last_global_index;
   NALU_HYPRE_Int             num_owned_nodes;
   NALU_HYPRE_Int             num_nonowned_nodes;
   NALU_HYPRE_Int             num_nonowned_real_nodes;
   NALU_HYPRE_Int             num_missing_col_indices;

   NALU_HYPRE_Int            *nonowned_global_indices;
   NALU_HYPRE_Int            *nonowned_coarse_indices;
   NALU_HYPRE_Int            *nonowned_real_marker;
   NALU_HYPRE_Int            *nonowned_sort;
   NALU_HYPRE_Int            *nonowned_invsort;
   NALU_HYPRE_Int            *nonowned_diag_missing_col_indices;

   NALU_HYPRE_Int            *owned_coarse_indices;

   nalu_hypre_AMGDDCompGridMatrix *A;
   nalu_hypre_AMGDDCompGridMatrix *P;
   nalu_hypre_AMGDDCompGridMatrix *R;

   nalu_hypre_AMGDDCompGridVector     *u;
   nalu_hypre_AMGDDCompGridVector     *f;
   nalu_hypre_AMGDDCompGridVector     *t;
   nalu_hypre_AMGDDCompGridVector     *s;
   nalu_hypre_AMGDDCompGridVector     *q;
   nalu_hypre_AMGDDCompGridVector     *temp;
   nalu_hypre_AMGDDCompGridVector     *temp2;
   nalu_hypre_AMGDDCompGridVector     *temp3;

   NALU_HYPRE_Real       *l1_norms;
   NALU_HYPRE_Int        *cf_marker_array;
   NALU_HYPRE_Int        *owned_relax_ordering;
   NALU_HYPRE_Int        *nonowned_relax_ordering;

} nalu_hypre_AMGDDCompGrid;

/*--------------------------------------------------------------------------
 * Accessor functions for the Comp Grid structure
 *--------------------------------------------------------------------------*/

#define nalu_hypre_AMGDDCompGridLevel(compGrid)                  ((compGrid) -> level)
#define nalu_hypre_AMGDDCompGridMemoryLocation(compGrid)         ((compGrid) -> memory_location)
#define nalu_hypre_AMGDDCompGridFirstGlobalIndex(compGrid)       ((compGrid) -> first_global_index)
#define nalu_hypre_AMGDDCompGridLastGlobalIndex(compGrid)        ((compGrid) -> last_global_index)
#define nalu_hypre_AMGDDCompGridNumOwnedNodes(compGrid)          ((compGrid) -> num_owned_nodes)
#define nalu_hypre_AMGDDCompGridNumNonOwnedNodes(compGrid)       ((compGrid) -> num_nonowned_nodes)
#define nalu_hypre_AMGDDCompGridNumNonOwnedRealNodes(compGrid)   ((compGrid) -> num_nonowned_real_nodes)
#define nalu_hypre_AMGDDCompGridNumMissingColIndices(compGrid)   ((compGrid) -> num_missing_col_indices)
#define nalu_hypre_AMGDDCompGridNonOwnedGlobalIndices(compGrid)  ((compGrid) -> nonowned_global_indices)
#define nalu_hypre_AMGDDCompGridNonOwnedCoarseIndices(compGrid)  ((compGrid) -> nonowned_coarse_indices)
#define nalu_hypre_AMGDDCompGridNonOwnedRealMarker(compGrid)     ((compGrid) -> nonowned_real_marker)
#define nalu_hypre_AMGDDCompGridNonOwnedSort(compGrid)           ((compGrid) -> nonowned_sort)
#define nalu_hypre_AMGDDCompGridNonOwnedInvSort(compGrid)        ((compGrid) -> nonowned_invsort)

#define nalu_hypre_AMGDDCompGridOwnedCoarseIndices(compGrid)            ((compGrid) -> owned_coarse_indices)
#define nalu_hypre_AMGDDCompGridNonOwnedDiagMissingColIndices(compGrid) ((compGrid) -> nonowned_diag_missing_col_indices)

#define nalu_hypre_AMGDDCompGridA(compGrid)     ((compGrid) -> A)
#define nalu_hypre_AMGDDCompGridP(compGrid)     ((compGrid) -> P)
#define nalu_hypre_AMGDDCompGridR(compGrid)     ((compGrid) -> R)
#define nalu_hypre_AMGDDCompGridU(compGrid)     ((compGrid) -> u)
#define nalu_hypre_AMGDDCompGridF(compGrid)     ((compGrid) -> f)
#define nalu_hypre_AMGDDCompGridT(compGrid)     ((compGrid) -> t)
#define nalu_hypre_AMGDDCompGridS(compGrid)     ((compGrid) -> s)
#define nalu_hypre_AMGDDCompGridQ(compGrid)     ((compGrid) -> q)
#define nalu_hypre_AMGDDCompGridTemp(compGrid)  ((compGrid) -> temp)
#define nalu_hypre_AMGDDCompGridTemp2(compGrid) ((compGrid) -> temp2)
#define nalu_hypre_AMGDDCompGridTemp3(compGrid) ((compGrid) -> temp3)

#define nalu_hypre_AMGDDCompGridL1Norms(compGrid)               ((compGrid) -> l1_norms)
#define nalu_hypre_AMGDDCompGridCFMarkerArray(compGrid)         ((compGrid) -> cf_marker_array)
#define nalu_hypre_AMGDDCompGridOwnedRelaxOrdering(compGrid)    ((compGrid) -> owned_relax_ordering)
#define nalu_hypre_AMGDDCompGridNonOwnedRelaxOrdering(compGrid) ((compGrid) -> nonowned_relax_ordering)

#endif
