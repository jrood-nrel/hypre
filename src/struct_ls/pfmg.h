/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Header info for the PFMG solver
 *
 *****************************************************************************/

#ifndef nalu_hypre_PFMG_HEADER
#define nalu_hypre_PFMG_HEADER

/*--------------------------------------------------------------------------
 * nalu_hypre_PFMGData:
 *--------------------------------------------------------------------------*/

typedef struct
{
   MPI_Comm              comm;

   NALU_HYPRE_Real            tol;
   NALU_HYPRE_Int             max_iter;
   NALU_HYPRE_Int             rel_change;
   NALU_HYPRE_Int             zero_guess;
   NALU_HYPRE_Int             max_levels;  /* max_level <= 0 means no limit */

   NALU_HYPRE_Int             relax_type;     /* type of relaxation to use */
   NALU_HYPRE_Real            jacobi_weight;  /* weighted jacobi weight */
   NALU_HYPRE_Int             usr_jacobi_weight; /* indicator flag for user weight */

   NALU_HYPRE_Int             rap_type;       /* controls choice of RAP codes */
   NALU_HYPRE_Int             num_pre_relax;  /* number of pre relaxation sweeps */
   NALU_HYPRE_Int             num_post_relax; /* number of post relaxation sweeps */
   NALU_HYPRE_Int             skip_relax;     /* flag to allow skipping relaxation */
   NALU_HYPRE_Real            relax_weight;
   NALU_HYPRE_Real            dxyz[3];     /* parameters used to determine cdir */

   NALU_HYPRE_Int             num_levels;

   NALU_HYPRE_Int            *cdir_l;  /* coarsening directions */
   NALU_HYPRE_Int            *active_l;  /* flags to relax on level l*/

   nalu_hypre_StructGrid    **grid_l;
   nalu_hypre_StructGrid    **P_grid_l;

   NALU_HYPRE_MemoryLocation  memory_location; /* memory location of data */
   NALU_HYPRE_Real           *data;
   NALU_HYPRE_Real           *data_const;
   nalu_hypre_StructMatrix  **A_l;
   nalu_hypre_StructMatrix  **P_l;
   nalu_hypre_StructMatrix  **RT_l;
   nalu_hypre_StructVector  **b_l;
   nalu_hypre_StructVector  **x_l;

   /* temp vectors */
   nalu_hypre_StructVector  **tx_l;
   nalu_hypre_StructVector  **r_l;
   nalu_hypre_StructVector  **e_l;

   void                **relax_data_l;
   void                **matvec_data_l;
   void                **restrict_data_l;
   void                **interp_data_l;

   /* log info (always logged) */
   NALU_HYPRE_Int             num_iterations;
   NALU_HYPRE_Int             time_index;

   NALU_HYPRE_Int             print_level;
   /* additional log info (logged when `logging' > 0) */
   NALU_HYPRE_Int             logging;
   NALU_HYPRE_Real           *norms;
   NALU_HYPRE_Real           *rel_norms;
#if 0 //defined(NALU_HYPRE_USING_CUDA) || defined(NALU_HYPRE_USING_HIP)
   NALU_HYPRE_Int             devicelevel;
#endif

} nalu_hypre_PFMGData;

#endif
