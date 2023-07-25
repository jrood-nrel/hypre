/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Header info for the SparseMSG solver
 *
 *****************************************************************************/

#ifndef nalu_hypre_SparseMSG_HEADER
#define nalu_hypre_SparseMSG_HEADER

/*--------------------------------------------------------------------------
 * nalu_hypre_SparseMSGData:
 *--------------------------------------------------------------------------*/

typedef struct
{
   MPI_Comm              comm;

   NALU_HYPRE_Real            tol;
   NALU_HYPRE_Int             max_iter;
   NALU_HYPRE_Int             rel_change;
   NALU_HYPRE_Int             zero_guess;
   NALU_HYPRE_Int             jump;

   NALU_HYPRE_Int             relax_type;     /* type of relaxation to use */
   NALU_HYPRE_Real            jacobi_weight;  /* weighted jacobi weight */
   NALU_HYPRE_Int             usr_jacobi_weight; /* indicator flag for user weight */

   NALU_HYPRE_Int             num_pre_relax;  /* number of pre relaxation sweeps */
   NALU_HYPRE_Int             num_post_relax; /* number of post relaxation sweeps */
   NALU_HYPRE_Int             num_fine_relax; /* number of fine relaxation sweeps */

   NALU_HYPRE_Int             num_grids[3];   /* number of grids in each dim */
   NALU_HYPRE_Int          num_all_grids;
   NALU_HYPRE_Int          num_levels;

   nalu_hypre_StructGrid    **grid_array;
   nalu_hypre_StructGrid    **Px_grid_array;
   nalu_hypre_StructGrid    **Py_grid_array;
   nalu_hypre_StructGrid    **Pz_grid_array;

   NALU_HYPRE_MemoryLocation  memory_location; /* memory location of data */
   NALU_HYPRE_Real           *data;
   nalu_hypre_StructMatrix  **A_array;
   nalu_hypre_StructMatrix  **Px_array;
   nalu_hypre_StructMatrix  **Py_array;
   nalu_hypre_StructMatrix  **Pz_array;
   nalu_hypre_StructMatrix  **RTx_array;
   nalu_hypre_StructMatrix  **RTy_array;
   nalu_hypre_StructMatrix  **RTz_array;
   nalu_hypre_StructVector  **b_array;
   nalu_hypre_StructVector  **x_array;

   /* temp vectors */
   nalu_hypre_StructVector  **t_array;
   nalu_hypre_StructVector  **r_array;
   nalu_hypre_StructVector  **e_array;

   nalu_hypre_StructVector  **visitx_array;
   nalu_hypre_StructVector  **visity_array;
   nalu_hypre_StructVector  **visitz_array;
   NALU_HYPRE_Int            *grid_on;

   void                **relax_array;
   void                **matvec_array;
   void                **restrictx_array;
   void                **restricty_array;
   void                **restrictz_array;
   void                **interpx_array;
   void                **interpy_array;
   void                **interpz_array;

   /* log info (always logged) */
   NALU_HYPRE_Int             num_iterations;
   NALU_HYPRE_Int             time_index;
   NALU_HYPRE_Int             print_level;

   /* additional log info (logged when `logging' > 0) */
   NALU_HYPRE_Int             logging;
   NALU_HYPRE_Real           *norms;
   NALU_HYPRE_Real           *rel_norms;

} nalu_hypre_SparseMSGData;

/*--------------------------------------------------------------------------
 * Utility routines:
 *--------------------------------------------------------------------------*/

#define nalu_hypre_SparseMSGMapIndex(lx, ly, lz, nl, index) \
index = (lx) + ((ly) * nl[0]) + ((lz) * nl[0] * nl[1])

#endif
