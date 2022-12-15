/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Header info for the FAC solver
 *
 *****************************************************************************/
/*--------------------------------------------------------------------------
 * nalu_hypre_FACData:
 *--------------------------------------------------------------------------*/

#ifndef nalu_hypre_FAC_HEADER
#define nalu_hypre_FAC_HEADER

typedef struct
{
   MPI_Comm               comm;

   NALU_HYPRE_Int             *plevels;
   nalu_hypre_Index           *prefinements;

   NALU_HYPRE_Int              max_levels;
   NALU_HYPRE_Int             *level_to_part;
   NALU_HYPRE_Int             *part_to_level;
   nalu_hypre_Index           *refine_factors;       /* refine_factors[level] */

   nalu_hypre_SStructGrid    **grid_level;
   nalu_hypre_SStructGraph   **graph_level;

   nalu_hypre_SStructMatrix   *A_rap;
   nalu_hypre_SStructMatrix  **A_level;
   nalu_hypre_SStructVector  **b_level;
   nalu_hypre_SStructVector  **x_level;
   nalu_hypre_SStructVector  **r_level;
   nalu_hypre_SStructVector  **e_level;
   nalu_hypre_SStructPVector **tx_level;
   nalu_hypre_SStructVector   *tx;


   void                 **matvec_data_level;
   void                 **pmatvec_data_level;
   void                  *matvec_data;
   void                 **relax_data_level;
   void                 **restrict_data_level;
   void                 **interp_data_level;

   NALU_HYPRE_Int              csolver_type;
   NALU_HYPRE_SStructSolver    csolver;
   NALU_HYPRE_SStructSolver    cprecond;

   NALU_HYPRE_Real             tol;
   NALU_HYPRE_Int              max_cycles;
   NALU_HYPRE_Int              zero_guess;
   NALU_HYPRE_Int              relax_type;
   NALU_HYPRE_Real             jacobi_weight;  /* weighted jacobi weight */
   NALU_HYPRE_Int              usr_jacobi_weight; /* indicator flag for user weight */

   NALU_HYPRE_Int              num_pre_smooth;
   NALU_HYPRE_Int              num_post_smooth;

   /* log info (always logged) */
   NALU_HYPRE_Int              num_iterations;
   NALU_HYPRE_Int              time_index;
   NALU_HYPRE_Int              rel_change;
   NALU_HYPRE_Int              logging;
   NALU_HYPRE_Real            *norms;
   NALU_HYPRE_Real            *rel_norms;


} nalu_hypre_FACData;

/*--------------------------------------------------------------------------
 * Accessor macros: nalu_hypre_FACData
 *--------------------------------------------------------------------------*/

#define nalu_hypre_FACDataMaxLevels(fac_data)\
((fac_data) -> max_levels)
#define nalu_hypre_FACDataLevelToPart(fac_data)\
((fac_data) -> level_to_part)
#define nalu_hypre_FACDataPartToLevel(fac_data)\
((fac_data) -> part_to_level)
#define nalu_hypre_FACDataRefineFactors(fac_data)\
((fac_data) -> refine_factors)
#define nalu_hypre_FACDataRefineFactorsLevel(fac_data, level)\
((fac_data) -> refinements[level])


#endif
