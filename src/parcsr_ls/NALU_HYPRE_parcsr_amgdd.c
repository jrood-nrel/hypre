/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_parcsr_ls.h"

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGDDCreate
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGDDCreate( NALU_HYPRE_Solver *solver)
{
   if (!solver)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   *solver = (NALU_HYPRE_Solver) hypre_BoomerAMGDDCreate( ) ;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGDDDestroy
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGDDDestroy( NALU_HYPRE_Solver solver )
{
   return ( hypre_BoomerAMGDDDestroy( (void *) solver ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGDDSetup
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGDDSetup( NALU_HYPRE_Solver solver,
                        NALU_HYPRE_ParCSRMatrix A,
                        NALU_HYPRE_ParVector b,
                        NALU_HYPRE_ParVector x )
{
   return ( hypre_BoomerAMGDDSetup( (void *) solver,
                                    (hypre_ParCSRMatrix *) A,
                                    (hypre_ParVector *) b,
                                    (hypre_ParVector *) x ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSolve
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGDDSolve( NALU_HYPRE_Solver solver,
                        NALU_HYPRE_ParCSRMatrix A,
                        NALU_HYPRE_ParVector b,
                        NALU_HYPRE_ParVector x )
{
   return ( hypre_BoomerAMGDDSolve( (void *) solver,
                                    (hypre_ParCSRMatrix *) A,
                                    (hypre_ParVector *) b,
                                    (hypre_ParVector *) x ) );
}

/*-------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGDDSetStartLevel
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGDDSetStartLevel( NALU_HYPRE_Solver solver,
                                NALU_HYPRE_Int    start_level )
{
   return ( hypre_BoomerAMGDDSetStartLevel( (void *) solver, start_level ) );
}

NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGDDGetStartLevel( NALU_HYPRE_Solver  solver,
                                NALU_HYPRE_Int    *start_level )
{
   return ( hypre_BoomerAMGDDGetStartLevel( (void *) solver, start_level ) );
}

/*-------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGDDSetFACNumRelax
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGDDSetFACNumRelax( NALU_HYPRE_Solver solver,
                                 NALU_HYPRE_Int    amgdd_fac_num_relax )
{
   return ( hypre_BoomerAMGDDSetFACNumRelax( (void *) solver, amgdd_fac_num_relax ) );
}

NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGDDGetFACNumRelax( NALU_HYPRE_Solver  solver,
                                 NALU_HYPRE_Int    *amgdd_fac_num_relax )
{
   return ( hypre_BoomerAMGDDGetFACNumRelax( (void *) solver, amgdd_fac_num_relax ) );
}

/*-------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGDDSetFACNumCycles
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGDDSetFACNumCycles( NALU_HYPRE_Solver solver,
                                  NALU_HYPRE_Int    amgdd_fac_num_cycles )
{
   return ( hypre_BoomerAMGDDSetFACNumCycles( (void *) solver, amgdd_fac_num_cycles ) );
}

NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGDDGetFACNumCycles( NALU_HYPRE_Solver  solver,
                                  NALU_HYPRE_Int    *amgdd_fac_num_cycles  )
{
   return ( hypre_BoomerAMGDDGetFACNumCycles( (void *) solver, amgdd_fac_num_cycles ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGDDSetFACCycleType
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGDDSetFACCycleType( NALU_HYPRE_Solver solver,
                                  NALU_HYPRE_Int    amgdd_fac_cycle_type )
{
   return ( hypre_BoomerAMGDDSetFACCycleType( (void *) solver, amgdd_fac_cycle_type ) );
}

NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGDDGetFACCycleType( NALU_HYPRE_Solver  solver,
                                  NALU_HYPRE_Int    *amgdd_fac_cycle_type )
{
   return ( hypre_BoomerAMGDDGetFACCycleType( (void *) solver, amgdd_fac_cycle_type ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGDDSetFACRelaxType
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGDDSetFACRelaxType( NALU_HYPRE_Solver solver,
                                  NALU_HYPRE_Int    amgdd_fac_relax_type )
{
   return ( hypre_BoomerAMGDDSetFACRelaxType( (void *) solver, amgdd_fac_relax_type ) );
}

NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGDDGetFACRelaxType( NALU_HYPRE_Solver  solver,
                                  NALU_HYPRE_Int    *amgdd_fac_relax_type )
{
   return ( hypre_BoomerAMGDDGetFACRelaxType( (void *) solver, amgdd_fac_relax_type ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGDDSetFACRelaxWeight
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGDDSetFACRelaxWeight( NALU_HYPRE_Solver solver,
                                    NALU_HYPRE_Real   amgdd_fac_relax_weight )
{
   return ( hypre_BoomerAMGDDSetFACRelaxWeight( (void *) solver, amgdd_fac_relax_weight ) );
}

NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGDDGetFACRelaxWeight( NALU_HYPRE_Solver  solver,
                                    NALU_HYPRE_Real   *amgdd_fac_relax_weight )
{
   return ( hypre_BoomerAMGDDGetFACRelaxWeight( (void *) solver, amgdd_fac_relax_weight ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGDDSetPadding
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGDDSetPadding( NALU_HYPRE_Solver solver,
                             NALU_HYPRE_Int    padding )
{
   return ( hypre_BoomerAMGDDSetPadding( (void *) solver, padding ) );
}

NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGDDGetPadding( NALU_HYPRE_Solver  solver,
                             NALU_HYPRE_Int    *padding  )
{
   return ( hypre_BoomerAMGDDGetPadding( (void *) solver, padding ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGDDSetNumGhostLayers
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGDDSetNumGhostLayers( NALU_HYPRE_Solver solver,
                                    NALU_HYPRE_Int    num_ghost_layers )
{
   return ( hypre_BoomerAMGDDSetNumGhostLayers( (void *) solver, num_ghost_layers ) );
}

NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGDDGetNumGhostLayers( NALU_HYPRE_Solver  solver,
                                    NALU_HYPRE_Int    *num_ghost_layers )
{
   return ( hypre_BoomerAMGDDGetNumGhostLayers( (void *) solver, num_ghost_layers ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGDDSetUserFACRelaxation
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGDDSetUserFACRelaxation( NALU_HYPRE_Solver solver,
                                       NALU_HYPRE_Int (*userFACRelaxation)( void      *amgdd_vdata,
                                                                       NALU_HYPRE_Int  level,
                                                                       NALU_HYPRE_Int  cycle_param ) )
{
   return ( hypre_BoomerAMGDDSetUserFACRelaxation( (void *) solver, userFACRelaxation ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGDDGetAMG
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGDDGetAMG( NALU_HYPRE_Solver  solver,
                         NALU_HYPRE_Solver *amg_solver )
{
   return ( hypre_BoomerAMGDDGetAMG( (void *) solver, (void **) amg_solver ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGDDGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGDDGetFinalRelativeResidualNorm( NALU_HYPRE_Solver  solver,
                                               NALU_HYPRE_Real   *rel_resid_norm )
{
   NALU_HYPRE_Solver amg_solver;

   NALU_HYPRE_BoomerAMGDDGetAMG(solver, &amg_solver);
   return ( hypre_BoomerAMGGetRelResidualNorm( (void *) amg_solver, rel_resid_norm ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGDDGetNumIterations
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGDDGetNumIterations( NALU_HYPRE_Solver   solver,
                                   NALU_HYPRE_Int     *num_iterations )
{
   NALU_HYPRE_Solver amg_solver;

   NALU_HYPRE_BoomerAMGDDGetAMG(solver, &amg_solver);
   return ( hypre_BoomerAMGGetNumIterations( (void *) amg_solver, num_iterations ) );
}
