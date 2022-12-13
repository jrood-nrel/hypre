/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#ifndef hypre_PAR_AMGDD_COMP_GRID_HEADER
#define hypre_PAR_AMGDD_COMP_GRID_HEADER

/*--------------------------------------------------------------------------
 * hypre_AMGDDCommPkg
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

} hypre_AMGDDCommPkg;

/*--------------------------------------------------------------------------
 * Accessor functions for the Comp Grid Comm Pkg structure
 *--------------------------------------------------------------------------*/

#define hypre_AMGDDCommPkgNumLevels(compGridCommPkg)      ((compGridCommPkg) -> num_levels)
#define hypre_AMGDDCommPkgNumSendProcs(compGridCommPkg)   ((compGridCommPkg) -> num_send_procs)
#define hypre_AMGDDCommPkgNumRecvProcs(compGridCommPkg)   ((compGridCommPkg) -> num_recv_procs)
#define hypre_AMGDDCommPkgSendProcs(compGridCommPkg)      ((compGridCommPkg) -> send_procs)
#define hypre_AMGDDCommPkgRecvProcs(compGridCommPkg)      ((compGridCommPkg) -> recv_procs)
#define hypre_AMGDDCommPkgSendBufferSize(compGridCommPkg) ((compGridCommPkg) -> send_buffer_size)
#define hypre_AMGDDCommPkgRecvBufferSize(compGridCommPkg) ((compGridCommPkg) -> recv_buffer_size)
#define hypre_AMGDDCommPkgNumSendNodes(compGridCommPkg)   ((compGridCommPkg) -> num_send_nodes)
#define hypre_AMGDDCommPkgNumRecvNodes(compGridCommPkg)   ((compGridCommPkg) -> num_recv_nodes)
#define hypre_AMGDDCommPkgSendFlag(compGridCommPkg)       ((compGridCommPkg) -> send_flag)
#define hypre_AMGDDCommPkgRecvMap(compGridCommPkg)        ((compGridCommPkg) -> recv_map)
#define hypre_AMGDDCommPkgRecvRedMarker(compGridCommPkg)  ((compGridCommPkg) -> recv_red_marker)

/*--------------------------------------------------------------------------
 * AMGDDCompGridMatrix (basically a coupled collection of CSR matrices)
 *--------------------------------------------------------------------------*/

typedef struct
{
   hypre_CSRMatrix      *owned_diag; // Domain: owned domain of mat. Range: owned range of mat.
   hypre_CSRMatrix      *owned_offd; // Domain: nonowned domain of mat. Range: owned range of mat.
   hypre_CSRMatrix
   *nonowned_diag; // Domain: nonowned domain of mat. Range: nonowned range of mat.
   hypre_CSRMatrix      *nonowned_offd; // Domain: owned domain of mat. Range: nonowned range of mat.

   hypre_CSRMatrix      *real_real;  // Domain: nonowned real. Range: nonowned real.
   hypre_CSRMatrix      *real_ghost; // Domain: nonowned ghost. Range: nonowned real.

   NALU_HYPRE_Int             owns_owned_matrices;
   NALU_HYPRE_Int             owns_offd_col_indices;

} hypre_AMGDDCompGridMatrix;

/*--------------------------------------------------------------------------
 * Accessor functions for the AMGDDCompGridMatrix structure
 *--------------------------------------------------------------------------*/

#define hypre_AMGDDCompGridMatrixOwnedDiag(matrix)          ((matrix) -> owned_diag)
#define hypre_AMGDDCompGridMatrixOwnedOffd(matrix)          ((matrix) -> owned_offd)
#define hypre_AMGDDCompGridMatrixNonOwnedDiag(matrix)       ((matrix) -> nonowned_diag)
#define hypre_AMGDDCompGridMatrixNonOwnedOffd(matrix)       ((matrix) -> nonowned_offd)
#define hypre_AMGDDCompGridMatrixRealReal(matrix)           ((matrix) -> real_real)
#define hypre_AMGDDCompGridMatrixRealGhost(matrix)          ((matrix) -> real_ghost)
#define hypre_AMGDDCompGridMatrixOwnsOwnedMatrices(matrix)  ((matrix) -> owns_owned_matrices)
#define hypre_AMGDDCompGridMatrixOwnsOffdColIndices(matrix) ((matrix) -> owns_offd_col_indices)

/*--------------------------------------------------------------------------
 * AMGDDCompGridVector
 *--------------------------------------------------------------------------*/

typedef struct
{
   hypre_Vector         *owned_vector;    // Original on-processor points (should be ordered)
   hypre_Vector         *nonowned_vector; // Off-processor points (not ordered)

   NALU_HYPRE_Int             num_real;
   NALU_HYPRE_Int             owns_owned_vector;

} hypre_AMGDDCompGridVector;

/*--------------------------------------------------------------------------
 * Accessor functions for the AMGDDCompGridVector structure
 *--------------------------------------------------------------------------*/

#define hypre_AMGDDCompGridVectorOwned(matrix)           ((matrix) -> owned_vector)
#define hypre_AMGDDCompGridVectorNonOwned(matrix)        ((matrix) -> nonowned_vector)
#define hypre_AMGDDCompGridVectorNumReal(vector)         ((vector) -> num_real)
#define hypre_AMGDDCompGridVectorOwnsOwnedVector(matrix) ((matrix) -> owns_owned_vector)

/*--------------------------------------------------------------------------
 * hypre_AMGDDCompGrid
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

   hypre_AMGDDCompGridMatrix *A;
   hypre_AMGDDCompGridMatrix *P;
   hypre_AMGDDCompGridMatrix *R;

   hypre_AMGDDCompGridVector     *u;
   hypre_AMGDDCompGridVector     *f;
   hypre_AMGDDCompGridVector     *t;
   hypre_AMGDDCompGridVector     *s;
   hypre_AMGDDCompGridVector     *q;
   hypre_AMGDDCompGridVector     *temp;
   hypre_AMGDDCompGridVector     *temp2;
   hypre_AMGDDCompGridVector     *temp3;

   NALU_HYPRE_Real       *l1_norms;
   NALU_HYPRE_Int        *cf_marker_array;
   NALU_HYPRE_Int        *owned_relax_ordering;
   NALU_HYPRE_Int        *nonowned_relax_ordering;

} hypre_AMGDDCompGrid;

/*--------------------------------------------------------------------------
 * Accessor functions for the Comp Grid structure
 *--------------------------------------------------------------------------*/

#define hypre_AMGDDCompGridLevel(compGrid)                  ((compGrid) -> level)
#define hypre_AMGDDCompGridMemoryLocation(compGrid)         ((compGrid) -> memory_location)
#define hypre_AMGDDCompGridFirstGlobalIndex(compGrid)       ((compGrid) -> first_global_index)
#define hypre_AMGDDCompGridLastGlobalIndex(compGrid)        ((compGrid) -> last_global_index)
#define hypre_AMGDDCompGridNumOwnedNodes(compGrid)          ((compGrid) -> num_owned_nodes)
#define hypre_AMGDDCompGridNumNonOwnedNodes(compGrid)       ((compGrid) -> num_nonowned_nodes)
#define hypre_AMGDDCompGridNumNonOwnedRealNodes(compGrid)   ((compGrid) -> num_nonowned_real_nodes)
#define hypre_AMGDDCompGridNumMissingColIndices(compGrid)   ((compGrid) -> num_missing_col_indices)
#define hypre_AMGDDCompGridNonOwnedGlobalIndices(compGrid)  ((compGrid) -> nonowned_global_indices)
#define hypre_AMGDDCompGridNonOwnedCoarseIndices(compGrid)  ((compGrid) -> nonowned_coarse_indices)
#define hypre_AMGDDCompGridNonOwnedRealMarker(compGrid)     ((compGrid) -> nonowned_real_marker)
#define hypre_AMGDDCompGridNonOwnedSort(compGrid)           ((compGrid) -> nonowned_sort)
#define hypre_AMGDDCompGridNonOwnedInvSort(compGrid)        ((compGrid) -> nonowned_invsort)

#define hypre_AMGDDCompGridOwnedCoarseIndices(compGrid)            ((compGrid) -> owned_coarse_indices)
#define hypre_AMGDDCompGridNonOwnedDiagMissingColIndices(compGrid) ((compGrid) -> nonowned_diag_missing_col_indices)

#define hypre_AMGDDCompGridA(compGrid)     ((compGrid) -> A)
#define hypre_AMGDDCompGridP(compGrid)     ((compGrid) -> P)
#define hypre_AMGDDCompGridR(compGrid)     ((compGrid) -> R)
#define hypre_AMGDDCompGridU(compGrid)     ((compGrid) -> u)
#define hypre_AMGDDCompGridF(compGrid)     ((compGrid) -> f)
#define hypre_AMGDDCompGridT(compGrid)     ((compGrid) -> t)
#define hypre_AMGDDCompGridS(compGrid)     ((compGrid) -> s)
#define hypre_AMGDDCompGridQ(compGrid)     ((compGrid) -> q)
#define hypre_AMGDDCompGridTemp(compGrid)  ((compGrid) -> temp)
#define hypre_AMGDDCompGridTemp2(compGrid) ((compGrid) -> temp2)
#define hypre_AMGDDCompGridTemp3(compGrid) ((compGrid) -> temp3)

#define hypre_AMGDDCompGridL1Norms(compGrid)               ((compGrid) -> l1_norms)
#define hypre_AMGDDCompGridCFMarkerArray(compGrid)         ((compGrid) -> cf_marker_array)
#define hypre_AMGDDCompGridOwnedRelaxOrdering(compGrid)    ((compGrid) -> owned_relax_ordering)
#define hypre_AMGDDCompGridNonOwnedRelaxOrdering(compGrid) ((compGrid) -> nonowned_relax_ordering)

#endif
