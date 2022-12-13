/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_parcsr_ls.h"

/*--------------------------------------------------------------------------
 * NALU_HYPRE_FSAICreate
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_FSAICreate( NALU_HYPRE_Solver *solver)
{
   if (!solver)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   *solver = (NALU_HYPRE_Solver) hypre_FSAICreate( ) ;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_FSAIDestroy
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_FSAIDestroy( NALU_HYPRE_Solver solver )
{
   return ( hypre_FSAIDestroy( (void *) solver ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_FSAISetup
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_FSAISetup( NALU_HYPRE_Solver       solver,
                 NALU_HYPRE_ParCSRMatrix A,
                 NALU_HYPRE_ParVector    b,
                 NALU_HYPRE_ParVector    x )
{
   return ( hypre_FSAISetup( (void *) solver,
                             (hypre_ParCSRMatrix *) A,
                             (hypre_ParVector *) b,
                             (hypre_ParVector *) x ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_FSAISolve
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_FSAISolve( NALU_HYPRE_Solver       solver,
                 NALU_HYPRE_ParCSRMatrix A,
                 NALU_HYPRE_ParVector    b,
                 NALU_HYPRE_ParVector    x )
{
   return ( hypre_FSAISolve( (void *) solver,
                             (hypre_ParCSRMatrix *) A,
                             (hypre_ParVector *) b,
                             (hypre_ParVector *) x ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_FSAISetAlgoType
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_FSAISetAlgoType( NALU_HYPRE_Solver solver,
                       NALU_HYPRE_Int    algo_type  )
{
   return ( hypre_FSAISetAlgoType( (void *) solver, algo_type ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_FSAIGetAlgoType
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_FSAIGetAlgoType( NALU_HYPRE_Solver  solver,
                       NALU_HYPRE_Int    *algo_type  )
{
   return ( hypre_FSAIGetAlgoType( (void *) solver, algo_type ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_FSAISetMaxSteps
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_FSAISetMaxSteps( NALU_HYPRE_Solver solver,
                       NALU_HYPRE_Int    max_steps  )
{
   return ( hypre_FSAISetMaxSteps( (void *) solver, max_steps ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_FSAIGetMaxSteps
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_FSAIGetMaxSteps( NALU_HYPRE_Solver  solver,
                       NALU_HYPRE_Int    *max_steps  )
{
   return ( hypre_FSAIGetMaxSteps( (void *) solver, max_steps ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_FSAISetMaxStepSize
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_FSAISetMaxStepSize( NALU_HYPRE_Solver solver,
                          NALU_HYPRE_Int    max_step_size )
{
   return ( hypre_FSAISetMaxStepSize( (void *) solver, max_step_size ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_FSAIGetMaxStepSize
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_FSAIGetMaxStepSize( NALU_HYPRE_Solver  solver,
                          NALU_HYPRE_Int    *max_step_size )
{
   return ( hypre_FSAIGetMaxStepSize( (void *) solver, max_step_size ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_FSAISetZeroGuess
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_FSAISetZeroGuess( NALU_HYPRE_Solver solver,
                        NALU_HYPRE_Int    zero_guess )
{
   return ( hypre_FSAISetZeroGuess( (void *) solver, zero_guess ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_FSAIGetZeroGuess
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_FSAIGetZeroGuess( NALU_HYPRE_Solver  solver,
                        NALU_HYPRE_Int    *zero_guess )
{
   return ( hypre_FSAIGetZeroGuess( (void *) solver, zero_guess ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_FSAISetKapTolerance
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_FSAISetKapTolerance( NALU_HYPRE_Solver solver,
                           NALU_HYPRE_Real   kap_tolerance )
{
   return ( hypre_FSAISetKapTolerance( (void *) solver, kap_tolerance ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_FSAIGetKapTolerance
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_FSAIGetKapTolerance( NALU_HYPRE_Solver  solver,
                           NALU_HYPRE_Real   *kap_tolerance )
{
   return ( hypre_FSAIGetKapTolerance( (void *) solver, kap_tolerance ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_FSAISetTolerance
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_FSAISetTolerance( NALU_HYPRE_Solver solver,
                        NALU_HYPRE_Real   tolerance )
{
   return ( hypre_FSAISetTolerance( (void *) solver, tolerance ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_FSAIGetTolerance
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_FSAIGetTolerance( NALU_HYPRE_Solver  solver,
                        NALU_HYPRE_Real   *tolerance )
{
   return ( hypre_FSAIGetTolerance( (void *) solver, tolerance ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_FSAISetOmega
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_FSAISetOmega( NALU_HYPRE_Solver solver,
                    NALU_HYPRE_Real   omega )
{
   return ( hypre_FSAISetOmega( (void *) solver, omega ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_FSAIGetOmega
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_FSAIGetOmega( NALU_HYPRE_Solver  solver,
                    NALU_HYPRE_Real   *omega )
{
   return ( hypre_FSAIGetOmega( (void *) solver, omega ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_FSAISetMaxIterations
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_FSAISetMaxIterations( NALU_HYPRE_Solver solver,
                            NALU_HYPRE_Int    max_iterations )
{
   return ( hypre_FSAISetMaxIterations( (void *) solver, max_iterations ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_FSAIGetMaxIterations
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_FSAIGetMaxIterations( NALU_HYPRE_Solver  solver,
                            NALU_HYPRE_Int    *max_iterations )
{
   return ( hypre_FSAIGetMaxIterations( (void *) solver, max_iterations ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_FSAISetEigMaxIters
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_FSAISetEigMaxIters( NALU_HYPRE_Solver solver,
                          NALU_HYPRE_Int    eig_max_iters )
{
   return ( hypre_FSAISetEigMaxIters( (void *) solver, eig_max_iters ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_FSAIGetEigMaxIters
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_FSAIGetEigMaxIters( NALU_HYPRE_Solver  solver,
                          NALU_HYPRE_Int    *eig_max_iters )
{
   return ( hypre_FSAIGetEigMaxIters( (void *) solver, eig_max_iters ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_FSAISetPrintLevel
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_FSAISetPrintLevel( NALU_HYPRE_Solver solver,
                         NALU_HYPRE_Int    print_level )
{
   return ( hypre_FSAISetPrintLevel( (void *) solver, print_level ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_FSAIGetPrintLevel
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_FSAIGetPrintLevel( NALU_HYPRE_Solver  solver,
                         NALU_HYPRE_Int    *print_level )
{
   return ( hypre_FSAIGetPrintLevel( (void *) solver, print_level ) );
}
