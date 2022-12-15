/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 *
 *****************************************************************************/

#include "_nalu_hypre_parcsr_ls.h"

/*--------------------------------------------------------------------------
 * nalu_hypre_AMGHybridData:
 *--------------------------------------------------------------------------*/

typedef struct
{
   NALU_HYPRE_Real            tol;
   NALU_HYPRE_Real            a_tol;
   NALU_HYPRE_Real            cf_tol;
   NALU_HYPRE_Int             dscg_max_its;
   NALU_HYPRE_Int             pcg_max_its;
   NALU_HYPRE_Int             two_norm;
   NALU_HYPRE_Int             stop_crit;
   NALU_HYPRE_Int             rel_change;
   NALU_HYPRE_Int             recompute_residual;
   NALU_HYPRE_Int             recompute_residual_p;
   NALU_HYPRE_Int             solver_type;
   NALU_HYPRE_Int             k_dim;

   NALU_HYPRE_Int             pcg_default;              /* boolean */
   NALU_HYPRE_Int           (*pcg_precond_solve)(void*, void*, void*, void*);
   NALU_HYPRE_Int           (*pcg_precond_setup)(void*, void*, void*, void*);
   void                 *pcg_precond;
   void                 *pcg_solver;

   /* log info (always logged) */
   NALU_HYPRE_Int             dscg_num_its;
   NALU_HYPRE_Int             pcg_num_its;
   NALU_HYPRE_Real            final_rel_res_norm;
   NALU_HYPRE_Int             time_index;

   NALU_HYPRE_Real            setup_time1;
   NALU_HYPRE_Real            setup_time2;
   NALU_HYPRE_Real            solve_time1;
   NALU_HYPRE_Real            solve_time2;

   MPI_Comm              comm;

   /* additional information (place-holder currently used to print norms) */
   NALU_HYPRE_Int             logging;
   NALU_HYPRE_Int             print_level;

   /* info for BoomerAMG */
   NALU_HYPRE_Real            strong_threshold;
   NALU_HYPRE_Real            max_row_sum;
   NALU_HYPRE_Real            trunc_factor;
   NALU_HYPRE_Int             pmax;
   NALU_HYPRE_Int             setup_type;
   NALU_HYPRE_Int             max_levels;
   NALU_HYPRE_Int             measure_type;
   NALU_HYPRE_Int             coarsen_type;
   NALU_HYPRE_Int             interp_type;
   NALU_HYPRE_Int             cycle_type;
   NALU_HYPRE_Int             relax_order;
   NALU_HYPRE_Int             keepT;
   NALU_HYPRE_Int             max_coarse_size;
   NALU_HYPRE_Int             min_coarse_size;
   NALU_HYPRE_Int             seq_threshold;
   NALU_HYPRE_Int            *num_grid_sweeps;
   NALU_HYPRE_Int            *grid_relax_type;
   NALU_HYPRE_Int           **grid_relax_points;
   NALU_HYPRE_Real           *relax_weight;
   NALU_HYPRE_Real           *omega;
   NALU_HYPRE_Int             num_paths;
   NALU_HYPRE_Int             agg_num_levels;
   NALU_HYPRE_Int             agg_interp_type;
   NALU_HYPRE_Int             num_functions;
   NALU_HYPRE_Int             nodal;
   NALU_HYPRE_Int            *dof_func;

   /* data needed for non-Galerkin option */
   NALU_HYPRE_Int           nongalerk_num_tol;
   NALU_HYPRE_Real         *nongalerkin_tol;
} nalu_hypre_AMGHybridData;

/*--------------------------------------------------------------------------
 * nalu_hypre_AMGHybridCreate
 *--------------------------------------------------------------------------*/

void *
nalu_hypre_AMGHybridCreate( )
{
   nalu_hypre_AMGHybridData *AMGhybrid_data;

   AMGhybrid_data = nalu_hypre_CTAlloc(nalu_hypre_AMGHybridData,  1, NALU_HYPRE_MEMORY_HOST);

   (AMGhybrid_data -> time_index)  = nalu_hypre_InitializeTiming("AMGHybrid");

   /* set defaults */
   (AMGhybrid_data -> tol)               = 1.0e-06;
   (AMGhybrid_data -> a_tol)             = 0.0;
   (AMGhybrid_data -> cf_tol)            = 0.90;
   (AMGhybrid_data -> dscg_max_its)      = 1000;
   (AMGhybrid_data -> pcg_max_its)       = 200;
   (AMGhybrid_data -> two_norm)          = 0;
   (AMGhybrid_data -> stop_crit)         = 0;
   (AMGhybrid_data -> rel_change)        = 0;
   (AMGhybrid_data -> pcg_default)       = 1;
   (AMGhybrid_data -> solver_type)       = 1;
   (AMGhybrid_data -> pcg_precond_solve) = NULL;
   (AMGhybrid_data -> pcg_precond_setup) = NULL;
   (AMGhybrid_data -> pcg_precond)       = NULL;
   (AMGhybrid_data -> pcg_solver)        = NULL;
   (AMGhybrid_data -> setup_time1)       = 0.0;
   (AMGhybrid_data -> setup_time2)       = 0.0;
   (AMGhybrid_data -> solve_time1)       = 0.0;
   (AMGhybrid_data -> solve_time2)       = 0.0;

   /* initialize */
   (AMGhybrid_data -> dscg_num_its)      = 0;
   (AMGhybrid_data -> pcg_num_its)       = 0;
   (AMGhybrid_data -> logging)           = 0;
   (AMGhybrid_data -> print_level)       = 0;
   (AMGhybrid_data -> k_dim)             = 5;

   /* BoomerAMG info */
   (AMGhybrid_data -> setup_type)       = 1;
   (AMGhybrid_data -> strong_threshold)  = 0.25;
   (AMGhybrid_data -> max_row_sum)  = 0.9;
   (AMGhybrid_data -> trunc_factor)  = 0.0;
   (AMGhybrid_data -> pmax)  = 4;
   (AMGhybrid_data -> max_levels)  = 25;
   (AMGhybrid_data -> measure_type)  = 0;
   (AMGhybrid_data -> coarsen_type)  = 10;
   (AMGhybrid_data -> interp_type)  = 6;
   (AMGhybrid_data -> cycle_type)  = 1;
   (AMGhybrid_data -> relax_order)  = 0;
   (AMGhybrid_data -> keepT)  = 0;
   (AMGhybrid_data -> max_coarse_size)  = 9;
   (AMGhybrid_data -> min_coarse_size)  = 1;
   (AMGhybrid_data -> seq_threshold)  = 0;
   (AMGhybrid_data -> num_grid_sweeps)  = NULL;
   (AMGhybrid_data -> grid_relax_type)  = NULL;
   (AMGhybrid_data -> grid_relax_points)  = NULL;
   (AMGhybrid_data -> relax_weight)  = NULL;
   (AMGhybrid_data -> omega)  = NULL;
   (AMGhybrid_data -> agg_num_levels)  = 0;
   (AMGhybrid_data -> agg_interp_type)  = 4;
   (AMGhybrid_data -> num_paths)  = 1;
   (AMGhybrid_data -> num_functions)  = 1;
   (AMGhybrid_data -> nodal)  = 0;
   (AMGhybrid_data -> dof_func)  = NULL;
   (AMGhybrid_data -> nongalerk_num_tol)  = 0;
   (AMGhybrid_data -> nongalerkin_tol)  = NULL;

   return (void *) AMGhybrid_data;
}

/*-------------------------------------------------------------------------- *
  nalu_hypre_AMGHybridDestroy
  *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_AMGHybridDestroy( void  *AMGhybrid_vdata )
{
   nalu_hypre_AMGHybridData *AMGhybrid_data = (nalu_hypre_AMGHybridData *)AMGhybrid_vdata;
   NALU_HYPRE_Int i;

   if (AMGhybrid_data)
   {
      NALU_HYPRE_Int solver_type = (AMGhybrid_data -> solver_type);
      /*NALU_HYPRE_Int pcg_default = (AMGhybrid_data -> pcg_default);*/
      void *pcg_solver = (AMGhybrid_data -> pcg_solver);
      void *pcg_precond = (AMGhybrid_data -> pcg_precond);

      if (pcg_precond) { nalu_hypre_BoomerAMGDestroy(pcg_precond); }
      if (solver_type == 1) { nalu_hypre_PCGDestroy(pcg_solver); }
      if (solver_type == 2) { nalu_hypre_GMRESDestroy(pcg_solver); }
      if (solver_type == 3) { nalu_hypre_BiCGSTABDestroy(pcg_solver); }

      if (AMGhybrid_data -> num_grid_sweeps)
      {
         nalu_hypre_TFree( (AMGhybrid_data -> num_grid_sweeps), NALU_HYPRE_MEMORY_HOST);
         (AMGhybrid_data -> num_grid_sweeps) = NULL;
      }
      if (AMGhybrid_data -> grid_relax_type)
      {
         nalu_hypre_TFree( (AMGhybrid_data -> grid_relax_type), NALU_HYPRE_MEMORY_HOST);
         (AMGhybrid_data -> grid_relax_type) = NULL;
      }
      if (AMGhybrid_data -> grid_relax_points)
      {
         for (i = 0; i < 4; i++)
         {
            nalu_hypre_TFree( (AMGhybrid_data -> grid_relax_points)[i], NALU_HYPRE_MEMORY_HOST);
         }
         nalu_hypre_TFree( (AMGhybrid_data -> grid_relax_points), NALU_HYPRE_MEMORY_HOST);
         (AMGhybrid_data -> grid_relax_points) = NULL;
      }
      if (AMGhybrid_data -> relax_weight)
      {
         nalu_hypre_TFree( (AMGhybrid_data -> relax_weight), NALU_HYPRE_MEMORY_HOST);
         (AMGhybrid_data -> relax_weight) = NULL;
      }
      if (AMGhybrid_data -> omega)
      {
         nalu_hypre_TFree( (AMGhybrid_data -> omega), NALU_HYPRE_MEMORY_HOST);
         (AMGhybrid_data -> omega) = NULL;
      }
      if (AMGhybrid_data -> dof_func)
      {
         nalu_hypre_TFree( (AMGhybrid_data -> dof_func), NALU_HYPRE_MEMORY_HOST);
         (AMGhybrid_data -> dof_func) = NULL;
      }
      nalu_hypre_TFree(AMGhybrid_data, NALU_HYPRE_MEMORY_HOST);
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_AMGHybridSetTol
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_AMGHybridSetTol( void   *AMGhybrid_vdata,
                       NALU_HYPRE_Real  tol       )
{
   nalu_hypre_AMGHybridData *AMGhybrid_data = (nalu_hypre_AMGHybridData *) AMGhybrid_vdata;

   if (!AMGhybrid_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }
   if (tol < 0 || tol > 1)
   {
      nalu_hypre_error_in_arg(2);
      return nalu_hypre_error_flag;
   }
   (AMGhybrid_data -> tol) = tol;

   return nalu_hypre_error_flag;
}
/*--------------------------------------------------------------------------
 * nalu_hypre_AMGHybridSetAbsoluteTol
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_AMGHybridSetAbsoluteTol( void   *AMGhybrid_vdata,
                               NALU_HYPRE_Real  a_tol       )
{
   nalu_hypre_AMGHybridData *AMGhybrid_data = (nalu_hypre_AMGHybridData *) AMGhybrid_vdata;

   if (!AMGhybrid_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }
   if (a_tol < 0 || a_tol > 1)
   {
      nalu_hypre_error_in_arg(2);
      return nalu_hypre_error_flag;
   }
   (AMGhybrid_data -> a_tol) = a_tol;

   return nalu_hypre_error_flag;
}
/*--------------------------------------------------------------------------
 * nalu_hypre_AMGHybridSetConvergenceTol
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_AMGHybridSetConvergenceTol( void   *AMGhybrid_vdata,
                                  NALU_HYPRE_Real  cf_tol       )
{
   nalu_hypre_AMGHybridData *AMGhybrid_data = (nalu_hypre_AMGHybridData *) AMGhybrid_vdata;
   if (!AMGhybrid_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }
   if (cf_tol < 0 || cf_tol > 1)
   {
      nalu_hypre_error_in_arg(2);
      return nalu_hypre_error_flag;
   }

   (AMGhybrid_data -> cf_tol) = cf_tol;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_AMGHybridSetNonGalerkinTol
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_AMGHybridSetNonGalerkinTol( void   *AMGhybrid_vdata,
                                  NALU_HYPRE_Int  nongalerk_num_tol,
                                  NALU_HYPRE_Real *nongalerkin_tol       )
{
   nalu_hypre_AMGHybridData *AMGhybrid_data = (nalu_hypre_AMGHybridData *) AMGhybrid_vdata;
   if (!AMGhybrid_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }
   if (nongalerk_num_tol < 0)
   {
      nalu_hypre_error_in_arg(2);
      return nalu_hypre_error_flag;
   }

   (AMGhybrid_data -> nongalerk_num_tol) = nongalerk_num_tol;
   (AMGhybrid_data -> nongalerkin_tol) = nongalerkin_tol;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_AMGHybridSetDSCGMaxIter
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_AMGHybridSetDSCGMaxIter( void   *AMGhybrid_vdata,
                               NALU_HYPRE_Int     dscg_max_its )
{
   nalu_hypre_AMGHybridData *AMGhybrid_data = (nalu_hypre_AMGHybridData *) AMGhybrid_vdata;
   if (!AMGhybrid_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }
   if (dscg_max_its < 0)
   {
      nalu_hypre_error_in_arg(2);
      return nalu_hypre_error_flag;
   }

   (AMGhybrid_data -> dscg_max_its) = dscg_max_its;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_AMGHybridSetPCGMaxIter
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_AMGHybridSetPCGMaxIter( void   *AMGhybrid_vdata,
                              NALU_HYPRE_Int     pcg_max_its  )
{
   nalu_hypre_AMGHybridData *AMGhybrid_data = (nalu_hypre_AMGHybridData *) AMGhybrid_vdata;
   if (!AMGhybrid_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }
   if (pcg_max_its < 0)
   {
      nalu_hypre_error_in_arg(2);
      return nalu_hypre_error_flag;
   }

   (AMGhybrid_data -> pcg_max_its) = pcg_max_its;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_AMGHybridSetSetupType
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_AMGHybridSetSetupType( void   *AMGhybrid_vdata,
                             NALU_HYPRE_Int     setup_type  )
{
   nalu_hypre_AMGHybridData *AMGhybrid_data = (nalu_hypre_AMGHybridData *) AMGhybrid_vdata;
   if (!AMGhybrid_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   (AMGhybrid_data -> setup_type) = setup_type;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_AMGHybridSetSolverType
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_AMGHybridSetSolverType( void   *AMGhybrid_vdata,
                              NALU_HYPRE_Int     solver_type  )
{
   nalu_hypre_AMGHybridData *AMGhybrid_data = (nalu_hypre_AMGHybridData *) AMGhybrid_vdata;
   if (!AMGhybrid_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   (AMGhybrid_data -> solver_type) = solver_type;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_AMGHybridSetRecomputeResidual
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_AMGHybridSetRecomputeResidual( void      *AMGhybrid_vdata,
                                     NALU_HYPRE_Int  recompute_residual )
{
   nalu_hypre_AMGHybridData *AMGhybrid_data = (nalu_hypre_AMGHybridData *)AMGhybrid_vdata;
   if (!AMGhybrid_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   (AMGhybrid_data -> recompute_residual) = recompute_residual;

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_AMGHybridGetRecomputeResidual( void      *AMGhybrid_vdata,
                                     NALU_HYPRE_Int *recompute_residual )
{
   nalu_hypre_AMGHybridData *AMGhybrid_data = (nalu_hypre_AMGHybridData *)AMGhybrid_vdata;
   if (!AMGhybrid_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   *recompute_residual = (AMGhybrid_data -> recompute_residual);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_AMGHybridSetRecomputeResidualP
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_AMGHybridSetRecomputeResidualP( void      *AMGhybrid_vdata,
                                      NALU_HYPRE_Int  recompute_residual_p )
{
   nalu_hypre_AMGHybridData *AMGhybrid_data = (nalu_hypre_AMGHybridData *)AMGhybrid_vdata;
   if (!AMGhybrid_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   (AMGhybrid_data -> recompute_residual_p) = recompute_residual_p;

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_AMGHybridGetRecomputeResidualP( void      *AMGhybrid_vdata,
                                      NALU_HYPRE_Int *recompute_residual_p )
{
   nalu_hypre_AMGHybridData *AMGhybrid_data = (nalu_hypre_AMGHybridData *)AMGhybrid_vdata;
   if (!AMGhybrid_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   *recompute_residual_p = (AMGhybrid_data -> recompute_residual_p);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_AMGHybridSetKDim
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_AMGHybridSetKDim( void   *AMGhybrid_vdata,
                        NALU_HYPRE_Int     k_dim  )
{
   nalu_hypre_AMGHybridData *AMGhybrid_data = (nalu_hypre_AMGHybridData *) AMGhybrid_vdata;
   if (!AMGhybrid_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }
   if (k_dim < 1)
   {
      nalu_hypre_error_in_arg(2);
      return nalu_hypre_error_flag;
   }

   (AMGhybrid_data -> k_dim) = k_dim;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_AMGHybridSetStopCrit
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_AMGHybridSetStopCrit( void *AMGhybrid_vdata,
                            NALU_HYPRE_Int   stop_crit  )
{
   nalu_hypre_AMGHybridData *AMGhybrid_data = (nalu_hypre_AMGHybridData *) AMGhybrid_vdata;
   if (!AMGhybrid_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   (AMGhybrid_data -> stop_crit) = stop_crit;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_AMGHybridSetTwoNorm
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_AMGHybridSetTwoNorm( void *AMGhybrid_vdata,
                           NALU_HYPRE_Int   two_norm  )
{
   nalu_hypre_AMGHybridData *AMGhybrid_data = (nalu_hypre_AMGHybridData *) AMGhybrid_vdata;
   if (!AMGhybrid_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   (AMGhybrid_data -> two_norm) = two_norm;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_AMGHybridSetRelChange
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_AMGHybridSetRelChange( void *AMGhybrid_vdata,
                             NALU_HYPRE_Int   rel_change  )
{
   nalu_hypre_AMGHybridData *AMGhybrid_data = (nalu_hypre_AMGHybridData *) AMGhybrid_vdata;
   if (!AMGhybrid_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   (AMGhybrid_data -> rel_change) = rel_change;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_AMGHybridSetPrecond
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_AMGHybridSetPrecond( void  *pcg_vdata,
                           NALU_HYPRE_Int  (*pcg_precond_solve)(void*, void*, void*, void*),
                           NALU_HYPRE_Int  (*pcg_precond_setup)(void*, void*, void*, void*),
                           void  *pcg_precond          )
{
   nalu_hypre_AMGHybridData *pcg_data = (nalu_hypre_AMGHybridData *) pcg_vdata;
   if (!pcg_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   (pcg_data -> pcg_default)       = 0;
   (pcg_data -> pcg_precond_solve) = pcg_precond_solve;
   (pcg_data -> pcg_precond_setup) = pcg_precond_setup;
   (pcg_data -> pcg_precond)       = pcg_precond;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_AMGHybridSetLogging
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_AMGHybridSetLogging( void *AMGhybrid_vdata,
                           NALU_HYPRE_Int   logging  )
{
   nalu_hypre_AMGHybridData *AMGhybrid_data = (nalu_hypre_AMGHybridData *) AMGhybrid_vdata;
   if (!AMGhybrid_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   (AMGhybrid_data -> logging) = logging;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_AMGHybridSetPrintLevel
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_AMGHybridSetPrintLevel( void *AMGhybrid_vdata,
                              NALU_HYPRE_Int   print_level  )
{
   nalu_hypre_AMGHybridData *AMGhybrid_data = (nalu_hypre_AMGHybridData *) AMGhybrid_vdata;
   if (!AMGhybrid_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   (AMGhybrid_data -> print_level) = print_level;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_AMGHybridSetStrongThreshold
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_AMGHybridSetStrongThreshold( void *AMGhybrid_vdata,
                                   NALU_HYPRE_Real strong_threshold)
{
   nalu_hypre_AMGHybridData *AMGhybrid_data = (nalu_hypre_AMGHybridData *) AMGhybrid_vdata;
   if (!AMGhybrid_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }
   if (strong_threshold < 0 || strong_threshold > 1)
   {
      nalu_hypre_error_in_arg(2);
      return nalu_hypre_error_flag;
   }

   (AMGhybrid_data -> strong_threshold) = strong_threshold;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_AMGHybridSetMaxRowSum
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_AMGHybridSetMaxRowSum( void *AMGhybrid_vdata,
                             NALU_HYPRE_Real   max_row_sum  )
{
   nalu_hypre_AMGHybridData *AMGhybrid_data = (nalu_hypre_AMGHybridData *) AMGhybrid_vdata;
   if (!AMGhybrid_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }
   if (max_row_sum < 0 || max_row_sum > 1)
   {
      nalu_hypre_error_in_arg(2);
      return nalu_hypre_error_flag;
   }

   (AMGhybrid_data -> max_row_sum) = max_row_sum;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_AMGHybridSetTruncFactor
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_AMGHybridSetTruncFactor( void *AMGhybrid_vdata,
                               NALU_HYPRE_Real   trunc_factor  )
{
   nalu_hypre_AMGHybridData *AMGhybrid_data = (nalu_hypre_AMGHybridData *) AMGhybrid_vdata;
   if (!AMGhybrid_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }
   if (trunc_factor < 0 || trunc_factor > 1)
   {
      nalu_hypre_error_in_arg(2);
      return nalu_hypre_error_flag;
   }

   (AMGhybrid_data -> trunc_factor) = trunc_factor;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_AMGHybridSetPMaxElmts
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_AMGHybridSetPMaxElmts( void   *AMGhybrid_vdata,
                             NALU_HYPRE_Int    P_max_elmts )
{

   nalu_hypre_AMGHybridData *AMGhybrid_data = (nalu_hypre_AMGHybridData *) AMGhybrid_vdata;
   if (!AMGhybrid_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   if (P_max_elmts < 0)
   {
      nalu_hypre_error_in_arg(2);
      return nalu_hypre_error_flag;
   }

   (AMGhybrid_data -> pmax) = P_max_elmts;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_AMGHybridSetMaxLevels
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_AMGHybridSetMaxLevels( void *AMGhybrid_vdata,
                             NALU_HYPRE_Int   max_levels  )
{
   nalu_hypre_AMGHybridData *AMGhybrid_data = (nalu_hypre_AMGHybridData *) AMGhybrid_vdata;
   if (!AMGhybrid_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }
   if (max_levels < 1)
   {
      nalu_hypre_error_in_arg(2);
      return nalu_hypre_error_flag;
   }

   (AMGhybrid_data -> max_levels) = max_levels;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_AMGHybridSetMeasureType
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_AMGHybridSetMeasureType( void *AMGhybrid_vdata,
                               NALU_HYPRE_Int   measure_type  )
{
   nalu_hypre_AMGHybridData *AMGhybrid_data = (nalu_hypre_AMGHybridData *) AMGhybrid_vdata;
   if (!AMGhybrid_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   (AMGhybrid_data -> measure_type) = measure_type;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_AMGHybridSetCoarsenType
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_AMGHybridSetCoarsenType( void *AMGhybrid_vdata,
                               NALU_HYPRE_Int   coarsen_type  )
{
   nalu_hypre_AMGHybridData *AMGhybrid_data = (nalu_hypre_AMGHybridData *) AMGhybrid_vdata;
   if (!AMGhybrid_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   (AMGhybrid_data -> coarsen_type) = coarsen_type;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_AMGHybridSetInterpType
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_AMGHybridSetInterpType( void *AMGhybrid_vdata,
                              NALU_HYPRE_Int   interp_type  )
{
   nalu_hypre_AMGHybridData *AMGhybrid_data = (nalu_hypre_AMGHybridData *) AMGhybrid_vdata;
   if (!AMGhybrid_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }
   if (interp_type < 0)
   {
      nalu_hypre_error_in_arg(2);
      return nalu_hypre_error_flag;
   }

   (AMGhybrid_data -> interp_type) = interp_type;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_AMGHybridSetCycleType
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_AMGHybridSetCycleType( void *AMGhybrid_vdata,
                             NALU_HYPRE_Int   cycle_type  )
{
   nalu_hypre_AMGHybridData *AMGhybrid_data = (nalu_hypre_AMGHybridData *) AMGhybrid_vdata;
   if (!AMGhybrid_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }
   if (cycle_type < 1 || cycle_type > 2)
   {
      nalu_hypre_error_in_arg(2);
      return nalu_hypre_error_flag;
   }

   (AMGhybrid_data -> cycle_type) = cycle_type;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_AMGHybridSetNumSweeps
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_AMGHybridSetNumSweeps( void *AMGhybrid_vdata,
                             NALU_HYPRE_Int   num_sweeps  )
{
   nalu_hypre_AMGHybridData *AMGhybrid_data = (nalu_hypre_AMGHybridData *) AMGhybrid_vdata;
   NALU_HYPRE_Int                 *num_grid_sweeps;
   NALU_HYPRE_Int               i;
   if (!AMGhybrid_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }
   if (num_sweeps < 1)
   {
      nalu_hypre_error_in_arg(2);
      return nalu_hypre_error_flag;
   }

   if ((AMGhybrid_data -> num_grid_sweeps) == NULL)
   {
      (AMGhybrid_data -> num_grid_sweeps) = nalu_hypre_CTAlloc(NALU_HYPRE_Int, 4, NALU_HYPRE_MEMORY_HOST);
   }
   num_grid_sweeps = (AMGhybrid_data -> num_grid_sweeps);
   for (i = 0; i < 3; i++)
   {
      num_grid_sweeps[i] = num_sweeps;
   }
   num_grid_sweeps[3] = 1;
   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_AMGHybridSetCycleNumSweeps
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_AMGHybridSetCycleNumSweeps( void *AMGhybrid_vdata,
                                  NALU_HYPRE_Int   num_sweeps,
                                  NALU_HYPRE_Int   k)
{
   nalu_hypre_AMGHybridData *AMGhybrid_data = (nalu_hypre_AMGHybridData *) AMGhybrid_vdata;
   NALU_HYPRE_Int                 *num_grid_sweeps;
   NALU_HYPRE_Int               i;

   if (!AMGhybrid_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }
   if (num_sweeps < 1)
   {
      nalu_hypre_error_in_arg(2);
      return nalu_hypre_error_flag;
   }
   if (k < 1 || k > 3)
   {
      if (AMGhybrid_data -> print_level)
      {
         nalu_hypre_printf (" Warning! Invalid cycle! num_sweeps not set!\n");
      }
      nalu_hypre_error_in_arg(3);
      return nalu_hypre_error_flag;
   }

   num_grid_sweeps = (AMGhybrid_data -> num_grid_sweeps);
   if (num_grid_sweeps == NULL)
   {
      (AMGhybrid_data -> num_grid_sweeps) = nalu_hypre_CTAlloc(NALU_HYPRE_Int, 4, NALU_HYPRE_MEMORY_HOST);
      num_grid_sweeps = (AMGhybrid_data -> num_grid_sweeps);
      for (i = 0; i < 4; i++)
      {
         num_grid_sweeps[i] = 1;
      }
   }
   num_grid_sweeps[k] = num_sweeps;
   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_AMGHybridSetRelaxType
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_AMGHybridSetRelaxType( void *AMGhybrid_vdata,
                             NALU_HYPRE_Int  relax_type  )
{
   nalu_hypre_AMGHybridData *AMGhybrid_data = (nalu_hypre_AMGHybridData *) AMGhybrid_vdata;
   NALU_HYPRE_Int               *grid_relax_type;
   NALU_HYPRE_Int               i;
   if (!AMGhybrid_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   if ((AMGhybrid_data -> grid_relax_type) == NULL )
   {
      (AMGhybrid_data -> grid_relax_type) = nalu_hypre_CTAlloc(NALU_HYPRE_Int, 4, NALU_HYPRE_MEMORY_HOST);
   }
   grid_relax_type = (AMGhybrid_data -> grid_relax_type);
   for (i = 0; i < 3; i++)
   {
      grid_relax_type[i] = relax_type;
   }
   grid_relax_type[3] = 9;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_AMGHybridSetCycleRelaxType
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_AMGHybridSetCycleRelaxType( void *AMGhybrid_vdata,
                                  NALU_HYPRE_Int   relax_type,
                                  NALU_HYPRE_Int   k  )
{
   nalu_hypre_AMGHybridData *AMGhybrid_data = (nalu_hypre_AMGHybridData *) AMGhybrid_vdata;
   NALU_HYPRE_Int                 *grid_relax_type;
   if (!AMGhybrid_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   if (k < 1 || k > 3)
   {
      if (AMGhybrid_data -> print_level)
      {
         nalu_hypre_printf (" Warning! Invalid cycle! Relax type not set!\n");
      }
      nalu_hypre_error_in_arg(3);
      return nalu_hypre_error_flag;
   }

   grid_relax_type = (AMGhybrid_data -> grid_relax_type);
   if (grid_relax_type == NULL )
   {
      (AMGhybrid_data -> grid_relax_type) = nalu_hypre_CTAlloc(NALU_HYPRE_Int, 4, NALU_HYPRE_MEMORY_HOST);
      grid_relax_type = (AMGhybrid_data -> grid_relax_type);

      grid_relax_type[1] = 13;
      grid_relax_type[2] = 14;
      grid_relax_type[3] = 9;
   }
   grid_relax_type[k] = relax_type;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_AMGHybridSetRelaxOrder
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_AMGHybridSetRelaxOrder( void *AMGhybrid_vdata,
                              NALU_HYPRE_Int   relax_order  )
{
   nalu_hypre_AMGHybridData *AMGhybrid_data = (nalu_hypre_AMGHybridData *) AMGhybrid_vdata;
   if (!AMGhybrid_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   (AMGhybrid_data -> relax_order) = relax_order;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_AMGHybridSetKeepTranspose
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_AMGHybridSetKeepTranspose( void *AMGhybrid_vdata,
                                 NALU_HYPRE_Int   keepT  )
{
   nalu_hypre_AMGHybridData *AMGhybrid_data = (nalu_hypre_AMGHybridData *) AMGhybrid_vdata;
   if (!AMGhybrid_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   (AMGhybrid_data -> keepT) = keepT;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_AMGHybridSetMaxCoarseSize
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_AMGHybridSetMaxCoarseSize( void *AMGhybrid_vdata,
                                 NALU_HYPRE_Int   max_coarse_size  )
{
   nalu_hypre_AMGHybridData *AMGhybrid_data = (nalu_hypre_AMGHybridData *) AMGhybrid_vdata;
   if (!AMGhybrid_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }
   if (max_coarse_size < 1)
   {
      nalu_hypre_error_in_arg(2);
      return nalu_hypre_error_flag;
   }

   (AMGhybrid_data -> max_coarse_size) = max_coarse_size;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_AMGHybridSetMinCoarseSize
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_AMGHybridSetMinCoarseSize( void *AMGhybrid_vdata,
                                 NALU_HYPRE_Int   min_coarse_size  )
{
   nalu_hypre_AMGHybridData *AMGhybrid_data = (nalu_hypre_AMGHybridData *) AMGhybrid_vdata;
   if (!AMGhybrid_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }
   if (min_coarse_size < 0)
   {
      nalu_hypre_error_in_arg(2);
      return nalu_hypre_error_flag;
   }

   (AMGhybrid_data -> min_coarse_size) = min_coarse_size;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_AMGHybridSetSeqThreshold
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_AMGHybridSetSeqThreshold( void *AMGhybrid_vdata,
                                NALU_HYPRE_Int   seq_threshold  )
{
   nalu_hypre_AMGHybridData *AMGhybrid_data = (nalu_hypre_AMGHybridData *) AMGhybrid_vdata;
   if (!AMGhybrid_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }
   if (seq_threshold < 0)
   {
      nalu_hypre_error_in_arg(2);
      return nalu_hypre_error_flag;
   }

   (AMGhybrid_data -> seq_threshold) = seq_threshold;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_AMGHybridSetNumGridSweeps
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_AMGHybridSetNumGridSweeps( void *AMGhybrid_vdata,
                                 NALU_HYPRE_Int  *num_grid_sweeps  )
{
   nalu_hypre_AMGHybridData *AMGhybrid_data = (nalu_hypre_AMGHybridData *) AMGhybrid_vdata;
   if (!AMGhybrid_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }
   if (!num_grid_sweeps)
   {
      nalu_hypre_error_in_arg(2);
      return nalu_hypre_error_flag;
   }

   if ((AMGhybrid_data -> num_grid_sweeps) != NULL)
   {
      nalu_hypre_TFree((AMGhybrid_data -> num_grid_sweeps), NALU_HYPRE_MEMORY_HOST);
   }
   (AMGhybrid_data -> num_grid_sweeps) = num_grid_sweeps;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_AMGHybridSetGridRelaxType
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_AMGHybridSetGridRelaxType( void *AMGhybrid_vdata,
                                 NALU_HYPRE_Int  *grid_relax_type  )
{
   nalu_hypre_AMGHybridData *AMGhybrid_data = (nalu_hypre_AMGHybridData *) AMGhybrid_vdata;
   if (!AMGhybrid_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }
   if (!grid_relax_type)
   {
      nalu_hypre_error_in_arg(2);
      return nalu_hypre_error_flag;
   }

   if ((AMGhybrid_data -> grid_relax_type) != NULL )
   {
      nalu_hypre_TFree((AMGhybrid_data -> grid_relax_type), NALU_HYPRE_MEMORY_HOST);
   }
   (AMGhybrid_data -> grid_relax_type) = grid_relax_type;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_AMGHybridSetGridRelaxPoints
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_AMGHybridSetGridRelaxPoints( void *AMGhybrid_vdata,
                                   NALU_HYPRE_Int  **grid_relax_points  )
{
   nalu_hypre_AMGHybridData *AMGhybrid_data = (nalu_hypre_AMGHybridData *) AMGhybrid_vdata;
   if (!AMGhybrid_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }
   if (!grid_relax_points)
   {
      nalu_hypre_error_in_arg(2);
      return nalu_hypre_error_flag;
   }

   if ((AMGhybrid_data -> grid_relax_points) != NULL )
   {
      nalu_hypre_TFree((AMGhybrid_data -> grid_relax_points), NALU_HYPRE_MEMORY_HOST);
   }
   (AMGhybrid_data -> grid_relax_points) = grid_relax_points;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_AMGHybridSetRelaxWeight
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_AMGHybridSetRelaxWeight( void *AMGhybrid_vdata,
                               NALU_HYPRE_Real *relax_weight  )
{
   nalu_hypre_AMGHybridData *AMGhybrid_data = (nalu_hypre_AMGHybridData *) AMGhybrid_vdata;
   if (!AMGhybrid_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }
   if (!relax_weight)
   {
      nalu_hypre_error_in_arg(2);
      return nalu_hypre_error_flag;
   }

   if ((AMGhybrid_data -> relax_weight) != NULL )
   {
      nalu_hypre_TFree((AMGhybrid_data -> relax_weight), NALU_HYPRE_MEMORY_HOST);
   }
   (AMGhybrid_data -> relax_weight) = relax_weight;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_AMGHybridSetOmega
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_AMGHybridSetOmega( void *AMGhybrid_vdata,
                         NALU_HYPRE_Real *omega  )
{
   nalu_hypre_AMGHybridData *AMGhybrid_data = (nalu_hypre_AMGHybridData *) AMGhybrid_vdata;
   if (!AMGhybrid_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }
   if (!omega)
   {
      nalu_hypre_error_in_arg(2);
      return nalu_hypre_error_flag;
   }

   if ((AMGhybrid_data -> omega) != NULL )
   {
      nalu_hypre_TFree((AMGhybrid_data -> omega), NALU_HYPRE_MEMORY_HOST);
   }
   (AMGhybrid_data -> omega) = omega;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_AMGHybridSetRelaxWt
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_AMGHybridSetRelaxWt( void *AMGhybrid_vdata,
                           NALU_HYPRE_Real  relax_wt  )
{
   nalu_hypre_AMGHybridData *AMGhybrid_data = (nalu_hypre_AMGHybridData *) AMGhybrid_vdata;
   NALU_HYPRE_Int               i, num_levels;
   NALU_HYPRE_Real          *relax_wt_array;
   if (!AMGhybrid_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   num_levels = (AMGhybrid_data -> max_levels);
   relax_wt_array = (AMGhybrid_data -> relax_weight);
   if (relax_wt_array == NULL)
   {
      relax_wt_array = nalu_hypre_CTAlloc(NALU_HYPRE_Real, num_levels, NALU_HYPRE_MEMORY_HOST);
      (AMGhybrid_data -> relax_weight) = relax_wt_array;
   }
   for (i = 0; i < num_levels; i++)
   {
      relax_wt_array[i] = relax_wt;
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_AMGHybridSetLevelRelaxWt
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_AMGHybridSetLevelRelaxWt( void   *AMGhybrid_vdata,
                                NALU_HYPRE_Real  relax_wt,
                                NALU_HYPRE_Int     level  )
{
   nalu_hypre_AMGHybridData *AMGhybrid_data = (nalu_hypre_AMGHybridData *) AMGhybrid_vdata;
   NALU_HYPRE_Int                i, num_levels;
   NALU_HYPRE_Real          *relax_wt_array;
   if (!AMGhybrid_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   num_levels = (AMGhybrid_data -> max_levels);
   if (level > num_levels - 1)
   {
      if (AMGhybrid_data -> print_level)
      {
         nalu_hypre_printf (" Warning! Invalid level! Relax weight not set!\n");
      }
      nalu_hypre_error_in_arg(3);
      return nalu_hypre_error_flag;
   }
   relax_wt_array = (AMGhybrid_data -> relax_weight);
   if (relax_wt_array == NULL)
   {
      relax_wt_array = nalu_hypre_CTAlloc(NALU_HYPRE_Real, num_levels, NALU_HYPRE_MEMORY_HOST);
      for (i = 0; i < num_levels; i++)
      {
         relax_wt_array[i] = 1.0;
      }
      (AMGhybrid_data -> relax_weight) = relax_wt_array;
   }
   relax_wt_array[level] = relax_wt;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_AMGHybridSetOuterWt
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_AMGHybridSetOuterWt( void *AMGhybrid_vdata,
                           NALU_HYPRE_Real  outer_wt  )
{
   nalu_hypre_AMGHybridData *AMGhybrid_data = (nalu_hypre_AMGHybridData *) AMGhybrid_vdata;
   NALU_HYPRE_Int                i, num_levels;
   NALU_HYPRE_Real          *outer_wt_array;
   if (!AMGhybrid_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   num_levels = (AMGhybrid_data -> max_levels);
   outer_wt_array = (AMGhybrid_data -> omega);
   if (outer_wt_array == NULL)
   {
      outer_wt_array = nalu_hypre_CTAlloc(NALU_HYPRE_Real, num_levels, NALU_HYPRE_MEMORY_HOST);
      (AMGhybrid_data -> omega) = outer_wt_array;
   }
   for (i = 0; i < num_levels; i++)
   {
      outer_wt_array[i] = outer_wt;
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_AMGHybridSetLevelOuterWt
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_AMGHybridSetLevelOuterWt( void   *AMGhybrid_vdata,
                                NALU_HYPRE_Real  outer_wt,
                                NALU_HYPRE_Int     level  )
{
   nalu_hypre_AMGHybridData *AMGhybrid_data = (nalu_hypre_AMGHybridData *) AMGhybrid_vdata;
   NALU_HYPRE_Int                i, num_levels;
   NALU_HYPRE_Real          *outer_wt_array;
   if (!AMGhybrid_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   num_levels = (AMGhybrid_data -> max_levels);
   if (level > num_levels - 1)
   {
      if (AMGhybrid_data -> print_level)
      {
         nalu_hypre_printf (" Warning! Invalid level! Outer weight not set!\n");
      }
      nalu_hypre_error_in_arg(3);
      return nalu_hypre_error_flag;
   }
   outer_wt_array = (AMGhybrid_data -> omega);
   if (outer_wt_array == NULL)
   {
      outer_wt_array = nalu_hypre_CTAlloc(NALU_HYPRE_Real, num_levels, NALU_HYPRE_MEMORY_HOST);
      for (i = 0; i < num_levels; i++)
      {
         outer_wt_array[i] = 1.0;
      }
      (AMGhybrid_data -> omega) = outer_wt_array;
   }
   outer_wt_array[level] = outer_wt;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_AMGHybridSetNumPaths
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_AMGHybridSetNumPaths( void   *AMGhybrid_vdata,
                            NALU_HYPRE_Int    num_paths      )
{
   nalu_hypre_AMGHybridData *AMGhybrid_data = (nalu_hypre_AMGHybridData *) AMGhybrid_vdata;
   if (!AMGhybrid_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }
   if (num_paths < 1)
   {
      nalu_hypre_error_in_arg(2);
      return nalu_hypre_error_flag;
   }

   (AMGhybrid_data -> num_paths) = num_paths;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_AMGHybridSetDofFunc
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_AMGHybridSetDofFunc( void *AMGhybrid_vdata,
                           NALU_HYPRE_Int *dof_func  )
{
   nalu_hypre_AMGHybridData *AMGhybrid_data = (nalu_hypre_AMGHybridData *) AMGhybrid_vdata;
   if (!AMGhybrid_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }
   if (!dof_func)
   {
      nalu_hypre_error_in_arg(2);
      return nalu_hypre_error_flag;
   }

   if ((AMGhybrid_data -> dof_func) != NULL )
   {
      nalu_hypre_TFree((AMGhybrid_data -> dof_func), NALU_HYPRE_MEMORY_HOST);
   }
   (AMGhybrid_data -> dof_func) = dof_func;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_AMGHybridSetAggNumLevels
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_AMGHybridSetAggNumLevels( void   *AMGhybrid_vdata,
                                NALU_HYPRE_Int    agg_num_levels      )
{
   nalu_hypre_AMGHybridData *AMGhybrid_data = (nalu_hypre_AMGHybridData *) AMGhybrid_vdata;
   if (!AMGhybrid_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }
   if (agg_num_levels < 0)
   {
      nalu_hypre_error_in_arg(2);
      return nalu_hypre_error_flag;
   }

   (AMGhybrid_data -> agg_num_levels) = agg_num_levels;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_AMGHybridSetAggInterpType
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_AMGHybridSetAggInterpType( void     *AMGhybrid_vdata,
                                 NALU_HYPRE_Int agg_interp_type      )
{
   nalu_hypre_AMGHybridData *AMGhybrid_data = (nalu_hypre_AMGHybridData *) AMGhybrid_vdata;
   if (!AMGhybrid_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   (AMGhybrid_data -> agg_interp_type) = agg_interp_type;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_AMGHybridSetNumFunctions
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_AMGHybridSetNumFunctions( void   *AMGhybrid_vdata,
                                NALU_HYPRE_Int    num_functions      )
{
   nalu_hypre_AMGHybridData *AMGhybrid_data = (nalu_hypre_AMGHybridData *) AMGhybrid_vdata;
   if (!AMGhybrid_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }
   if (num_functions < 1)
   {
      nalu_hypre_error_in_arg(2);
      return nalu_hypre_error_flag;
   }

   (AMGhybrid_data -> num_functions) = num_functions;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_AMGHybridSetNodal
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_AMGHybridSetNodal( void   *AMGhybrid_vdata,
                         NALU_HYPRE_Int    nodal      )
{
   nalu_hypre_AMGHybridData *AMGhybrid_data = (nalu_hypre_AMGHybridData *) AMGhybrid_vdata;
   if (!AMGhybrid_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   (AMGhybrid_data -> nodal) = nodal;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_AMGHybridGetNumIterations
 *--------------------------------------------------------------------------*/
NALU_HYPRE_Int
nalu_hypre_AMGHybridGetSetupSolveTime( void          *AMGhybrid_vdata,
                                  NALU_HYPRE_Real    *time )
{
   nalu_hypre_AMGHybridData *AMGhybrid_data = (nalu_hypre_AMGHybridData *) AMGhybrid_vdata;
   if (!AMGhybrid_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   NALU_HYPRE_Real t[4];
   t[0] = AMGhybrid_data->setup_time1;
   t[1] = AMGhybrid_data->solve_time1;
   t[2] = AMGhybrid_data->setup_time2;
   t[3] = AMGhybrid_data->solve_time2;

   MPI_Comm comm = AMGhybrid_data->comm;

   nalu_hypre_MPI_Allreduce(t, time, 4, nalu_hypre_MPI_REAL, nalu_hypre_MPI_MAX, comm);

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_AMGHybridGetNumIterations( void   *AMGhybrid_vdata,
                                 NALU_HYPRE_Int    *num_its      )
{
   nalu_hypre_AMGHybridData *AMGhybrid_data = (nalu_hypre_AMGHybridData *) AMGhybrid_vdata;
   if (!AMGhybrid_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   *num_its = (AMGhybrid_data -> dscg_num_its) + (AMGhybrid_data -> pcg_num_its);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_AMGHybridGetDSCGNumIterations
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_AMGHybridGetDSCGNumIterations( void   *AMGhybrid_vdata,
                                     NALU_HYPRE_Int    *dscg_num_its )
{
   nalu_hypre_AMGHybridData *AMGhybrid_data = (nalu_hypre_AMGHybridData *) AMGhybrid_vdata;
   if (!AMGhybrid_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   *dscg_num_its = (AMGhybrid_data -> dscg_num_its);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_AMGHybridGetPCGNumIterations
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_AMGHybridGetPCGNumIterations( void   *AMGhybrid_vdata,
                                    NALU_HYPRE_Int    *pcg_num_its  )
{
   nalu_hypre_AMGHybridData *AMGhybrid_data = (nalu_hypre_AMGHybridData *) AMGhybrid_vdata;
   if (!AMGhybrid_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   *pcg_num_its = (AMGhybrid_data -> pcg_num_its);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_AMGHybridGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_AMGHybridGetFinalRelativeResidualNorm( void   *AMGhybrid_vdata,
                                             NALU_HYPRE_Real *final_rel_res_norm )
{
   nalu_hypre_AMGHybridData *AMGhybrid_data = (nalu_hypre_AMGHybridData *) AMGhybrid_vdata;
   if (!AMGhybrid_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   *final_rel_res_norm = (AMGhybrid_data -> final_rel_res_norm);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_AMGHybridSetup
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_AMGHybridSetup( void               *AMGhybrid_vdata,
                      nalu_hypre_ParCSRMatrix *A,
                      nalu_hypre_ParVector *b,
                      nalu_hypre_ParVector *x            )
{
   nalu_hypre_AMGHybridData *AMGhybrid_data = (nalu_hypre_AMGHybridData *) AMGhybrid_vdata;
   if (!AMGhybrid_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_AMGHybridSolve
 *--------------------------------------------------------------------------
 *
 * This solver is designed to solve Ax=b using a AMGhybrid algorithm. First
 * the solver uses diagonally scaled conjugate gradients. If sufficient
 * progress is not made, the algorithm switches to preconditioned
 * conjugate gradients with user-specified preconditioner.
 *
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_AMGHybridSolve( void               *AMGhybrid_vdata,
                      nalu_hypre_ParCSRMatrix *A,
                      nalu_hypre_ParVector    *b,
                      nalu_hypre_ParVector    *x )
{
   nalu_hypre_AMGHybridData  *AMGhybrid_data    = (nalu_hypre_AMGHybridData *) AMGhybrid_vdata;

   NALU_HYPRE_Real         tol;
   NALU_HYPRE_Real         a_tol;
   NALU_HYPRE_Real         cf_tol;
   NALU_HYPRE_Int          dscg_max_its;
   NALU_HYPRE_Int          pcg_max_its;
   NALU_HYPRE_Int          two_norm;
   NALU_HYPRE_Int          stop_crit;
   NALU_HYPRE_Int          rel_change;
   NALU_HYPRE_Int          recompute_residual;
   NALU_HYPRE_Int          recompute_residual_p;
   NALU_HYPRE_Int          logging;
   NALU_HYPRE_Int          print_level;
   NALU_HYPRE_Int          setup_type;
   NALU_HYPRE_Int          solver_type;
   NALU_HYPRE_Int          k_dim;
   /* BoomerAMG info */
   NALU_HYPRE_Real         strong_threshold;
   NALU_HYPRE_Real         max_row_sum;
   NALU_HYPRE_Real         trunc_factor;
   NALU_HYPRE_Int          pmax;
   NALU_HYPRE_Int          max_levels;
   NALU_HYPRE_Int          measure_type;
   NALU_HYPRE_Int          coarsen_type;
   NALU_HYPRE_Int          interp_type;
   NALU_HYPRE_Int          cycle_type;
   NALU_HYPRE_Int          num_paths;
   NALU_HYPRE_Int          agg_num_levels;
   NALU_HYPRE_Int          agg_interp_type;
   NALU_HYPRE_Int          num_functions;
   NALU_HYPRE_Int          nodal;
   NALU_HYPRE_Int          relax_order;
   NALU_HYPRE_Int          keepT;
   NALU_HYPRE_Int         *num_grid_sweeps;
   NALU_HYPRE_Int         *grid_relax_type;
   NALU_HYPRE_Int        **grid_relax_points;
   NALU_HYPRE_Real        *relax_weight;
   NALU_HYPRE_Real        *omega;
   NALU_HYPRE_Int         *dof_func;

   NALU_HYPRE_Int         *boom_ngs;
   NALU_HYPRE_Int         *boom_grt;
   NALU_HYPRE_Int         *boom_dof_func;
   NALU_HYPRE_Int        **boom_grp;
   NALU_HYPRE_Real        *boom_rlxw;
   NALU_HYPRE_Real        *boom_omega;

   NALU_HYPRE_Int          pcg_default;
   NALU_HYPRE_Int          (*pcg_precond_solve)(void*, void*, void*, void*);
   NALU_HYPRE_Int          (*pcg_precond_setup)(void*, void*, void*, void*);
   void              *pcg_precond;

   void              *pcg_solver;
   nalu_hypre_PCGFunctions *pcg_functions;
   nalu_hypre_GMRESFunctions *gmres_functions;
   nalu_hypre_BiCGSTABFunctions *bicgstab_functions;

   NALU_HYPRE_Int          dscg_num_its = 0;
   NALU_HYPRE_Int          pcg_num_its = 0;
   NALU_HYPRE_Int          converged = 0;
   NALU_HYPRE_Int          num_variables = nalu_hypre_VectorSize(nalu_hypre_ParVectorLocalVector(b));
   NALU_HYPRE_Real         res_norm;

   NALU_HYPRE_Int          i, j;
   NALU_HYPRE_Int          sol_print_level; /* print_level for solver */
   NALU_HYPRE_Int          pre_print_level; /* print_level for preconditioner */
   NALU_HYPRE_Int          max_coarse_size, seq_threshold;
   NALU_HYPRE_Int          min_coarse_size;
   NALU_HYPRE_Int          nongalerk_num_tol;
   NALU_HYPRE_Real        *nongalerkin_tol;

   NALU_HYPRE_Real         tt1, tt2;

   if (!AMGhybrid_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   AMGhybrid_data->setup_time1 = 0.0;
   AMGhybrid_data->setup_time2 = 0.0;
   AMGhybrid_data->solve_time1 = 0.0;
   AMGhybrid_data->solve_time2 = 0.0;
   MPI_Comm  comm = nalu_hypre_ParCSRMatrixComm(A);
   (AMGhybrid_data -> comm) = comm;
   /*-----------------------------------------------------------------------
    * Setup diagonal scaled solver
    *-----------------------------------------------------------------------*/
   tol            = (AMGhybrid_data -> tol);
   a_tol          = (AMGhybrid_data -> a_tol);
   cf_tol         = (AMGhybrid_data -> cf_tol);
   dscg_max_its   = (AMGhybrid_data -> dscg_max_its);
   pcg_max_its    = (AMGhybrid_data -> pcg_max_its);
   two_norm       = (AMGhybrid_data -> two_norm);
   stop_crit      = (AMGhybrid_data -> stop_crit);
   rel_change     = (AMGhybrid_data -> rel_change);
   recompute_residual   = (AMGhybrid_data -> recompute_residual);
   recompute_residual_p = (AMGhybrid_data -> recompute_residual_p);
   logging        = (AMGhybrid_data -> logging);
   print_level    = (AMGhybrid_data -> print_level);
   setup_type     = (AMGhybrid_data -> setup_type);
   solver_type    = (AMGhybrid_data -> solver_type);
   k_dim          = (AMGhybrid_data -> k_dim);
   strong_threshold = (AMGhybrid_data -> strong_threshold);
   max_row_sum = (AMGhybrid_data -> max_row_sum);
   trunc_factor = (AMGhybrid_data -> trunc_factor);
   pmax = (AMGhybrid_data -> pmax);
   max_levels = (AMGhybrid_data -> max_levels);
   measure_type = (AMGhybrid_data -> measure_type);
   coarsen_type = (AMGhybrid_data -> coarsen_type);
   interp_type = (AMGhybrid_data -> interp_type);
   cycle_type = (AMGhybrid_data -> cycle_type);
   num_paths = (AMGhybrid_data -> num_paths);
   agg_num_levels = (AMGhybrid_data -> agg_num_levels);
   agg_interp_type = (AMGhybrid_data -> agg_interp_type);
   num_functions = (AMGhybrid_data -> num_functions);
   nodal = (AMGhybrid_data -> nodal);
   num_grid_sweeps = (AMGhybrid_data -> num_grid_sweeps);
   grid_relax_type = (AMGhybrid_data -> grid_relax_type);
   grid_relax_points = (AMGhybrid_data -> grid_relax_points);
   relax_weight = (AMGhybrid_data -> relax_weight);
   relax_order = (AMGhybrid_data -> relax_order);
   keepT = (AMGhybrid_data -> keepT);
   omega = (AMGhybrid_data -> omega);
   max_coarse_size = (AMGhybrid_data -> max_coarse_size);
   min_coarse_size = (AMGhybrid_data -> min_coarse_size);
   seq_threshold = (AMGhybrid_data -> seq_threshold);
   dof_func = (AMGhybrid_data -> dof_func);
   pcg_default    = (AMGhybrid_data -> pcg_default);
   nongalerk_num_tol    = (AMGhybrid_data -> nongalerk_num_tol);
   nongalerkin_tol    = (AMGhybrid_data -> nongalerkin_tol);
   if (!b)
   {
      nalu_hypre_error_in_arg(3);
      return nalu_hypre_error_flag;
   }
   num_variables = nalu_hypre_VectorSize(nalu_hypre_ParVectorLocalVector(b));
   if (!A)
   {
      nalu_hypre_error_in_arg(2);
      return nalu_hypre_error_flag;
   }
   if (!x)
   {
      nalu_hypre_error_in_arg(4);
      return nalu_hypre_error_flag;
   }

   /* print_level definitions: xy,  sol_print_level = y, pre_print_level = x */
   pre_print_level = print_level / 10;
   sol_print_level = print_level - pre_print_level * 10;

   pcg_solver = (AMGhybrid_data -> pcg_solver);
   pcg_precond = (AMGhybrid_data -> pcg_precond);
   (AMGhybrid_data -> dscg_num_its) = 0;
   (AMGhybrid_data -> pcg_num_its) = 0;

   if (setup_type || pcg_precond == NULL)
   {
      if (pcg_precond)
      {
         nalu_hypre_BoomerAMGDestroy(pcg_precond);
         pcg_precond = NULL;
         (AMGhybrid_data -> pcg_precond) = NULL;
      }
      if (solver_type == 1)
      {
         tt1 = nalu_hypre_MPI_Wtime();

         if (pcg_solver == NULL)
         {
            pcg_functions =
               nalu_hypre_PCGFunctionsCreate(
                  nalu_hypre_ParKrylovCAlloc, nalu_hypre_ParKrylovFree,
                  nalu_hypre_ParKrylovCommInfo,
                  nalu_hypre_ParKrylovCreateVector,
                  nalu_hypre_ParKrylovDestroyVector, nalu_hypre_ParKrylovMatvecCreate,
                  nalu_hypre_ParKrylovMatvec,
                  nalu_hypre_ParKrylovMatvecDestroy,
                  nalu_hypre_ParKrylovInnerProd, nalu_hypre_ParKrylovCopyVector,
                  nalu_hypre_ParKrylovClearVector,
                  nalu_hypre_ParKrylovScaleVector, nalu_hypre_ParKrylovAxpy,
                  nalu_hypre_ParKrylovIdentitySetup, nalu_hypre_ParKrylovIdentity );
            pcg_solver = nalu_hypre_PCGCreate( pcg_functions );

            nalu_hypre_PCGSetTol(pcg_solver, tol);
            nalu_hypre_PCGSetAbsoluteTol(pcg_solver, a_tol);
            nalu_hypre_PCGSetTwoNorm(pcg_solver, two_norm);
            nalu_hypre_PCGSetStopCrit(pcg_solver, stop_crit);
            nalu_hypre_PCGSetRelChange(pcg_solver, rel_change);
            nalu_hypre_PCGSetRecomputeResidual(pcg_solver, recompute_residual);
            nalu_hypre_PCGSetRecomputeResidualP(pcg_solver, recompute_residual_p);
            nalu_hypre_PCGSetLogging(pcg_solver, logging);
            nalu_hypre_PCGSetPrintLevel(pcg_solver, sol_print_level);
            nalu_hypre_PCGSetHybrid(pcg_solver, -1);

            pcg_precond = NULL;
         }

         nalu_hypre_PCGSetMaxIter(pcg_solver, dscg_max_its);
         nalu_hypre_PCGSetConvergenceFactorTol(pcg_solver, cf_tol);
         nalu_hypre_PCGSetPrecond((void*) pcg_solver,
                             (NALU_HYPRE_Int (*)(void*, void*, void*, void*)) NALU_HYPRE_ParCSRDiagScale,
                             (NALU_HYPRE_Int (*)(void*, void*, void*, void*)) NALU_HYPRE_ParCSRDiagScaleSetup,
                             (void*) pcg_precond);

         nalu_hypre_PCGSetup(pcg_solver, (void*) A, (void*) b, (void*) x);
         (AMGhybrid_data -> pcg_solver) = pcg_solver;

         tt2 = nalu_hypre_MPI_Wtime();
         AMGhybrid_data->setup_time1 = tt2 - tt1;

         /*---------------------------------------------------------------------
          * Solve with DSCG.
          *---------------------------------------------------------------------*/
         tt1 = tt2;

         nalu_hypre_PCGSolve(pcg_solver, (void*) A, (void*) b, (void*) x);

         /*---------------------------------------------------------------------
          * Get information for DSCG.
          *---------------------------------------------------------------------*/
         nalu_hypre_PCGGetNumIterations(pcg_solver, &dscg_num_its);
         (AMGhybrid_data -> dscg_num_its) = dscg_num_its;
         nalu_hypre_PCGGetFinalRelativeResidualNorm(pcg_solver, &res_norm);

         nalu_hypre_PCGGetConverged(pcg_solver, &converged);

         tt2 = nalu_hypre_MPI_Wtime();
         AMGhybrid_data->solve_time1 = tt2 - tt1;
      }
      else if (solver_type == 2)
      {
         tt1 = nalu_hypre_MPI_Wtime();

         if (pcg_solver == NULL)
         {
            gmres_functions =
               nalu_hypre_GMRESFunctionsCreate(
                  nalu_hypre_ParKrylovCAlloc, nalu_hypre_ParKrylovFree,
                  nalu_hypre_ParKrylovCommInfo,
                  nalu_hypre_ParKrylovCreateVector,
                  nalu_hypre_ParKrylovCreateVectorArray,
                  nalu_hypre_ParKrylovDestroyVector, nalu_hypre_ParKrylovMatvecCreate,
                  nalu_hypre_ParKrylovMatvec,
                  nalu_hypre_ParKrylovMatvecDestroy,
                  nalu_hypre_ParKrylovInnerProd, nalu_hypre_ParKrylovCopyVector,
                  nalu_hypre_ParKrylovClearVector,
                  nalu_hypre_ParKrylovScaleVector, nalu_hypre_ParKrylovAxpy,
                  nalu_hypre_ParKrylovIdentitySetup, nalu_hypre_ParKrylovIdentity );
            pcg_solver = nalu_hypre_GMRESCreate( gmres_functions );

            nalu_hypre_GMRESSetTol(pcg_solver, tol);
            nalu_hypre_GMRESSetAbsoluteTol(pcg_solver, a_tol);
            nalu_hypre_GMRESSetKDim(pcg_solver, k_dim);
            nalu_hypre_GMRESSetStopCrit(pcg_solver, stop_crit);
            nalu_hypre_GMRESSetRelChange(pcg_solver, rel_change);
            nalu_hypre_GMRESSetLogging(pcg_solver, logging);
            nalu_hypre_GMRESSetPrintLevel(pcg_solver, sol_print_level);
            nalu_hypre_GMRESSetHybrid(pcg_solver, -1);

            pcg_precond = NULL;
         }

         nalu_hypre_GMRESSetMaxIter(pcg_solver, dscg_max_its);
         nalu_hypre_GMRESSetConvergenceFactorTol(pcg_solver, cf_tol);
         nalu_hypre_GMRESSetPrecond((void*) pcg_solver,
                               (NALU_HYPRE_Int (*)(void*, void*, void*, void*)) NALU_HYPRE_ParCSRDiagScale,
                               (NALU_HYPRE_Int (*)(void*, void*, void*, void*)) NALU_HYPRE_ParCSRDiagScaleSetup,
                               (void*) pcg_precond);

         nalu_hypre_GMRESSetup(pcg_solver, (void*) A, (void*) b, (void*) x);
         (AMGhybrid_data -> pcg_solver) = pcg_solver;

         tt2 = nalu_hypre_MPI_Wtime();
         AMGhybrid_data->setup_time1 = tt2 - tt1;

         /*---------------------------------------------------------------------
          * Solve with diagonal scaled GMRES
          *---------------------------------------------------------------------*/
         tt1 = tt2;

         nalu_hypre_GMRESSolve(pcg_solver, (void*) A, (void*) b, (void*) x);

         /*---------------------------------------------------------------------
          * Get information for GMRES
          *---------------------------------------------------------------------*/
         nalu_hypre_GMRESGetNumIterations(pcg_solver, &dscg_num_its);
         (AMGhybrid_data -> dscg_num_its) = dscg_num_its;
         nalu_hypre_GMRESGetFinalRelativeResidualNorm(pcg_solver, &res_norm);

         nalu_hypre_GMRESGetConverged(pcg_solver, &converged);

         tt2 = nalu_hypre_MPI_Wtime();
         AMGhybrid_data->solve_time1 = tt2 - tt1;
      }
      else if (solver_type == 3)
      {
         tt1 = nalu_hypre_MPI_Wtime();

         if (pcg_solver == NULL)
         {
            bicgstab_functions =
               nalu_hypre_BiCGSTABFunctionsCreate(
                  nalu_hypre_ParKrylovCreateVector,
                  nalu_hypre_ParKrylovDestroyVector, nalu_hypre_ParKrylovMatvecCreate,
                  nalu_hypre_ParKrylovMatvec,
                  nalu_hypre_ParKrylovMatvecDestroy,
                  nalu_hypre_ParKrylovInnerProd, nalu_hypre_ParKrylovCopyVector,
                  nalu_hypre_ParKrylovClearVector,
                  nalu_hypre_ParKrylovScaleVector, nalu_hypre_ParKrylovAxpy,
                  nalu_hypre_ParKrylovCommInfo,
                  nalu_hypre_ParKrylovIdentitySetup, nalu_hypre_ParKrylovIdentity );
            pcg_solver = nalu_hypre_BiCGSTABCreate( bicgstab_functions );

            nalu_hypre_BiCGSTABSetTol(pcg_solver, tol);
            nalu_hypre_BiCGSTABSetAbsoluteTol(pcg_solver, a_tol);
            nalu_hypre_BiCGSTABSetStopCrit(pcg_solver, stop_crit);
            nalu_hypre_BiCGSTABSetLogging(pcg_solver, logging);
            nalu_hypre_BiCGSTABSetPrintLevel(pcg_solver, sol_print_level);
            nalu_hypre_BiCGSTABSetHybrid(pcg_solver, -1);

            pcg_precond = NULL;
         }

         nalu_hypre_BiCGSTABSetMaxIter(pcg_solver, dscg_max_its);
         nalu_hypre_BiCGSTABSetConvergenceFactorTol(pcg_solver, cf_tol);
         nalu_hypre_BiCGSTABSetPrecond((void*) pcg_solver,
                                  (NALU_HYPRE_Int (*)(void*, void*, void*, void*)) NALU_HYPRE_ParCSRDiagScale,
                                  (NALU_HYPRE_Int (*)(void*, void*, void*, void*)) NALU_HYPRE_ParCSRDiagScaleSetup,
                                  (void*) pcg_precond);

         nalu_hypre_BiCGSTABSetup(pcg_solver, (void*) A, (void*) b, (void*) x);
         (AMGhybrid_data -> pcg_solver) = pcg_solver;

         tt2 = nalu_hypre_MPI_Wtime();
         AMGhybrid_data->setup_time1 = tt2 - tt1;

         /*---------------------------------------------------------------------
          * Solve with diagonal scaled BiCGSTAB
          *---------------------------------------------------------------------*/
         tt1 = tt2;

         nalu_hypre_BiCGSTABSolve(pcg_solver, (void*) A, (void*) b, (void*) x);

         /*---------------------------------------------------------------------
          * Get information for BiCGSTAB
          *---------------------------------------------------------------------*/
         nalu_hypre_BiCGSTABGetNumIterations(pcg_solver, &dscg_num_its);
         (AMGhybrid_data -> dscg_num_its) = dscg_num_its;
         nalu_hypre_BiCGSTABGetFinalRelativeResidualNorm(pcg_solver, &res_norm);

         nalu_hypre_BiCGSTABGetConverged(pcg_solver, &converged);

         tt2 = nalu_hypre_MPI_Wtime();
         AMGhybrid_data->solve_time1 = tt2 - tt1;
      }
   }

   /*---------------------------------------------------------------------
    * If converged, done...
    *---------------------------------------------------------------------*/
   if (converged)
   {
      if (logging)
      {
         (AMGhybrid_data -> final_rel_res_norm) = res_norm;
      }
   }
   /*-----------------------------------------------------------------------
    * ... otherwise, use AMG+solver
    *-----------------------------------------------------------------------*/
   else
   {
      tt1 = nalu_hypre_MPI_Wtime();

      /*--------------------------------------------------------------------
       * Free up previous PCG solver structure and set up a new one.
       *--------------------------------------------------------------------*/
      if (solver_type == 1)
      {
         nalu_hypre_PCGSetMaxIter(pcg_solver, pcg_max_its);
         nalu_hypre_PCGSetConvergenceFactorTol(pcg_solver, 0.0);
         nalu_hypre_PCGSetHybrid(pcg_solver, 0);
      }
      else if (solver_type == 2)
      {
         nalu_hypre_GMRESSetMaxIter(pcg_solver, pcg_max_its);
         nalu_hypre_GMRESSetConvergenceFactorTol(pcg_solver, 0.0);
         nalu_hypre_GMRESSetHybrid(pcg_solver, 0);
      }
      else if (solver_type == 3)
      {
         nalu_hypre_BiCGSTABSetMaxIter(pcg_solver, pcg_max_its);
         nalu_hypre_BiCGSTABSetConvergenceFactorTol(pcg_solver, 0.0);
         nalu_hypre_BiCGSTABSetHybrid(pcg_solver, 0);
      }

      /* Setup preconditioner */
      if (setup_type && pcg_default)
      {
         pcg_precond = nalu_hypre_BoomerAMGCreate();
         nalu_hypre_BoomerAMGSetMaxIter(pcg_precond, 1);
         nalu_hypre_BoomerAMGSetTol(pcg_precond, 0.0);
         nalu_hypre_BoomerAMGSetCoarsenType(pcg_precond, coarsen_type);
         nalu_hypre_BoomerAMGSetInterpType(pcg_precond, interp_type);
         nalu_hypre_BoomerAMGSetSetupType(pcg_precond, setup_type);
         nalu_hypre_BoomerAMGSetMeasureType(pcg_precond, measure_type);
         nalu_hypre_BoomerAMGSetStrongThreshold(pcg_precond, strong_threshold);
         nalu_hypre_BoomerAMGSetTruncFactor(pcg_precond, trunc_factor);
         nalu_hypre_BoomerAMGSetPMaxElmts(pcg_precond, pmax);
         nalu_hypre_BoomerAMGSetCycleType(pcg_precond, cycle_type);
         nalu_hypre_BoomerAMGSetPrintLevel(pcg_precond, pre_print_level);
         nalu_hypre_BoomerAMGSetMaxLevels(pcg_precond,  max_levels);
         nalu_hypre_BoomerAMGSetMaxRowSum(pcg_precond, max_row_sum);
         nalu_hypre_BoomerAMGSetMaxCoarseSize(pcg_precond, max_coarse_size);
         nalu_hypre_BoomerAMGSetMinCoarseSize(pcg_precond, min_coarse_size);
         nalu_hypre_BoomerAMGSetSeqThreshold(pcg_precond, seq_threshold);
         nalu_hypre_BoomerAMGSetAggNumLevels(pcg_precond, agg_num_levels);
         nalu_hypre_BoomerAMGSetAggInterpType(pcg_precond, agg_interp_type);
         nalu_hypre_BoomerAMGSetNumPaths(pcg_precond, num_paths);
         nalu_hypre_BoomerAMGSetNumFunctions(pcg_precond, num_functions);
         nalu_hypre_BoomerAMGSetNodal(pcg_precond, nodal);
         nalu_hypre_BoomerAMGSetRelaxOrder(pcg_precond, relax_order);
         nalu_hypre_BoomerAMGSetKeepTranspose(pcg_precond, keepT);
         nalu_hypre_BoomerAMGSetNonGalerkTol(pcg_precond, nongalerk_num_tol, nongalerkin_tol);
         if (grid_relax_type)
         {
            boom_grt = nalu_hypre_CTAlloc(NALU_HYPRE_Int, 4, NALU_HYPRE_MEMORY_HOST);
            for (i = 0; i < 4; i++)
            {
               boom_grt[i] = grid_relax_type[i];
            }
            nalu_hypre_BoomerAMGSetGridRelaxType(pcg_precond, boom_grt);
         }
         else
         {
            boom_grt = nalu_hypre_CTAlloc(NALU_HYPRE_Int, 4, NALU_HYPRE_MEMORY_HOST);
            boom_grt[0] = 3;
            boom_grt[1] = 13;
            boom_grt[2] = 14;
            boom_grt[3] = 9;
            nalu_hypre_BoomerAMGSetGridRelaxType(pcg_precond, boom_grt);
         }

         nalu_hypre_ParAMGDataUserCoarseRelaxType((nalu_hypre_ParAMGData *) pcg_precond) = boom_grt[3];
         nalu_hypre_ParAMGDataUserRelaxType((nalu_hypre_ParAMGData *) pcg_precond) = boom_grt[0];

         if (relax_weight)
         {
            boom_rlxw = nalu_hypre_CTAlloc(NALU_HYPRE_Real, max_levels, NALU_HYPRE_MEMORY_HOST);
            for (i = 0; i < max_levels; i++)
            {
               boom_rlxw[i] = relax_weight[i];
            }
            nalu_hypre_BoomerAMGSetRelaxWeight(pcg_precond, boom_rlxw);
         }
         if (omega)
         {
            boom_omega = nalu_hypre_CTAlloc(NALU_HYPRE_Real, max_levels, NALU_HYPRE_MEMORY_HOST);
            for (i = 0; i < max_levels; i++)
            {
               boom_omega[i] = omega[i];
            }
            nalu_hypre_BoomerAMGSetOmega(pcg_precond, boom_omega);
         }
         if (num_grid_sweeps)
         {
            boom_ngs = nalu_hypre_CTAlloc(NALU_HYPRE_Int, 4, NALU_HYPRE_MEMORY_HOST);
            for (i = 0; i < 4; i++)
            {
               boom_ngs[i] = num_grid_sweeps[i];
            }
            nalu_hypre_BoomerAMGSetNumGridSweeps(pcg_precond, boom_ngs);
            if (grid_relax_points)
            {
               boom_grp = nalu_hypre_CTAlloc(NALU_HYPRE_Int*, 4, NALU_HYPRE_MEMORY_HOST);
               for (i = 0; i < 4; i++)
               {
                  boom_grp[i] = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_grid_sweeps[i], NALU_HYPRE_MEMORY_HOST);
                  for (j = 0; j < num_grid_sweeps[i]; j++)
                  {
                     boom_grp[i][j] = grid_relax_points[i][j];
                  }
               }
               nalu_hypre_BoomerAMGSetGridRelaxPoints(pcg_precond, boom_grp);
            }
         }
         if (dof_func)
         {
            boom_dof_func = nalu_hypre_CTAlloc(NALU_HYPRE_Int, num_variables, NALU_HYPRE_MEMORY_HOST);
            for (i = 0; i < num_variables; i++)
            {
               boom_dof_func[i] = dof_func[i];
            }
            nalu_hypre_BoomerAMGSetDofFunc(pcg_precond, boom_dof_func);
         }
         pcg_precond_solve = (NALU_HYPRE_Int (*)(void*, void*, void*, void*)) nalu_hypre_BoomerAMGSolve;
         pcg_precond_setup = (NALU_HYPRE_Int (*)(void*, void*, void*, void*)) nalu_hypre_BoomerAMGSetup;
         (AMGhybrid_data -> pcg_precond_setup) = pcg_precond_setup;
         (AMGhybrid_data -> pcg_precond_solve) = pcg_precond_solve;
         (AMGhybrid_data -> pcg_precond) = pcg_precond;
         /*(AMGhybrid_data -> pcg_default) = 0;*/
         /*(AMGhybrid_data -> setup_type) = 0;*/
      }
      else
      {
         pcg_precond       = (AMGhybrid_data -> pcg_precond);
         pcg_precond_solve = (AMGhybrid_data -> pcg_precond_solve);
         pcg_precond_setup = (AMGhybrid_data -> pcg_precond_setup);
         nalu_hypre_BoomerAMGSetSetupType(pcg_precond, setup_type);
      }

      /* Complete setup of solver+AMG */
      if (solver_type == 1)
      {
         nalu_hypre_PCGSetPrecond((void*) pcg_solver,
                             (NALU_HYPRE_Int (*)(void*, void*, void*, void*)) pcg_precond_solve,
                             (NALU_HYPRE_Int (*)(void*, void*, void*, void*)) pcg_precond_setup,
                             (void*) pcg_precond);

         nalu_hypre_PCGSetup(pcg_solver, (void*) A, (void*) b, (void*) x);

         tt2 = nalu_hypre_MPI_Wtime();
         AMGhybrid_data->setup_time2 = tt2 - tt1;

         /* Solve */
         tt1 = tt2;

         nalu_hypre_PCGSolve(pcg_solver, (void*) A, (void*) b, (void*) x);

         /* Get information from PCG that is always logged in AMGhybrid solver*/
         nalu_hypre_PCGGetNumIterations(pcg_solver, &pcg_num_its);
         (AMGhybrid_data -> pcg_num_its)  = pcg_num_its;
         if (logging)
         {
            nalu_hypre_PCGGetFinalRelativeResidualNorm(pcg_solver, &res_norm);
            (AMGhybrid_data -> final_rel_res_norm) = res_norm;
         }

         tt2 = nalu_hypre_MPI_Wtime();
         AMGhybrid_data->solve_time2 = tt2 - tt1;
      }
      else if (solver_type == 2)
      {
         nalu_hypre_GMRESSetPrecond((void*) pcg_solver,
                               (NALU_HYPRE_Int (*)(void*, void*, void*, void*)) pcg_precond_solve,
                               (NALU_HYPRE_Int (*)(void*, void*, void*, void*)) pcg_precond_setup,
                               (void*) pcg_precond);

         nalu_hypre_GMRESSetup(pcg_solver, (void*) A, (void*) b, (void*) x);

         tt2 = nalu_hypre_MPI_Wtime();
         AMGhybrid_data->setup_time2 = tt2 - tt1;

         /* Solve */
         tt1 = tt2;

         nalu_hypre_GMRESSolve(pcg_solver, (void*) A, (void*) b, (void*) x);

         /* Get information from GMRES that is always logged in AMGhybrid solver*/
         nalu_hypre_GMRESGetNumIterations(pcg_solver, &pcg_num_its);
         (AMGhybrid_data -> pcg_num_its)  = pcg_num_its;
         if (logging)
         {
            nalu_hypre_GMRESGetFinalRelativeResidualNorm(pcg_solver, &res_norm);
            (AMGhybrid_data -> final_rel_res_norm) = res_norm;
         }

         tt2 = nalu_hypre_MPI_Wtime();
         AMGhybrid_data->solve_time2 = tt2 - tt1;
      }
      else if (solver_type == 3)
      {
         nalu_hypre_BiCGSTABSetPrecond((void*) pcg_solver,
                                  (NALU_HYPRE_Int (*)(void*, void*, void*, void*)) pcg_precond_solve,
                                  (NALU_HYPRE_Int (*)(void*, void*, void*, void*)) pcg_precond_setup,
                                  (void*) pcg_precond);

         nalu_hypre_BiCGSTABSetup(pcg_solver, (void*) A, (void*) b, (void*) x);

         tt2 = nalu_hypre_MPI_Wtime();
         AMGhybrid_data->setup_time2 = tt2 - tt1;

         /* Solve */
         tt1 = tt2;

         nalu_hypre_BiCGSTABSolve(pcg_solver, (void*) A, (void*) b, (void*) x);

         /* Get information from BiCGSTAB that is always logged in AMGhybrid solver*/
         nalu_hypre_BiCGSTABGetNumIterations(pcg_solver, &pcg_num_its);
         (AMGhybrid_data -> pcg_num_its)  = pcg_num_its;
         if (logging)
         {
            nalu_hypre_BiCGSTABGetFinalRelativeResidualNorm(pcg_solver, &res_norm);
            (AMGhybrid_data -> final_rel_res_norm) = res_norm;
         }

         tt2 = nalu_hypre_MPI_Wtime();
         AMGhybrid_data->solve_time2 = tt2 - tt1;
      }
   }

   return nalu_hypre_error_flag;
}

