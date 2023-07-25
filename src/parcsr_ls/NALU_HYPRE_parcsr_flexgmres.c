/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_nalu_hypre_parcsr_ls.h"

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRFlexGMRESCreate
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParCSRFlexGMRESCreate( MPI_Comm comm, NALU_HYPRE_Solver *solver )
{
   nalu_hypre_FlexGMRESFunctions * fgmres_functions;

   if (!solver)
   {
      nalu_hypre_error_in_arg(2);
      return nalu_hypre_error_flag;
   }
   fgmres_functions =
      nalu_hypre_FlexGMRESFunctionsCreate(
         nalu_hypre_ParKrylovCAlloc, nalu_hypre_ParKrylovFree, nalu_hypre_ParKrylovCommInfo,
         nalu_hypre_ParKrylovCreateVector,
         nalu_hypre_ParKrylovCreateVectorArray,
         nalu_hypre_ParKrylovDestroyVector, nalu_hypre_ParKrylovMatvecCreate,
         nalu_hypre_ParKrylovMatvec, nalu_hypre_ParKrylovMatvecDestroy,
         nalu_hypre_ParKrylovInnerProd, nalu_hypre_ParKrylovCopyVector,
         nalu_hypre_ParKrylovClearVector,
         nalu_hypre_ParKrylovScaleVector, nalu_hypre_ParKrylovAxpy,
         nalu_hypre_ParKrylovIdentitySetup, nalu_hypre_ParKrylovIdentity );
   *solver = ( (NALU_HYPRE_Solver) nalu_hypre_FlexGMRESCreate( fgmres_functions ) );

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRFlexGMRESDestroy
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParCSRFlexGMRESDestroy( NALU_HYPRE_Solver solver )
{
   return ( nalu_hypre_FlexGMRESDestroy( (void *) solver ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRFlexGMRESSetup
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParCSRFlexGMRESSetup( NALU_HYPRE_Solver solver,
                            NALU_HYPRE_ParCSRMatrix A,
                            NALU_HYPRE_ParVector b,
                            NALU_HYPRE_ParVector x      )
{
   return ( NALU_HYPRE_FlexGMRESSetup( solver,
                                  (NALU_HYPRE_Matrix) A,
                                  (NALU_HYPRE_Vector) b,
                                  (NALU_HYPRE_Vector) x ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRFlexGMRESSolve
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParCSRFlexGMRESSolve( NALU_HYPRE_Solver solver,
                            NALU_HYPRE_ParCSRMatrix A,
                            NALU_HYPRE_ParVector b,
                            NALU_HYPRE_ParVector x      )
{
   return ( NALU_HYPRE_FlexGMRESSolve( solver,
                                  (NALU_HYPRE_Matrix) A,
                                  (NALU_HYPRE_Vector) b,
                                  (NALU_HYPRE_Vector) x ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRFlexGMRESSetKDim
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParCSRFlexGMRESSetKDim( NALU_HYPRE_Solver solver,
                              NALU_HYPRE_Int             k_dim    )
{
   return ( NALU_HYPRE_FlexGMRESSetKDim( solver, k_dim ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRFlexGMRESSetTol
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParCSRFlexGMRESSetTol( NALU_HYPRE_Solver solver,
                             NALU_HYPRE_Real         tol    )
{
   return ( NALU_HYPRE_FlexGMRESSetTol( solver, tol ) );
}
/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRFlexGMRESSetAbsoluteTol
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParCSRFlexGMRESSetAbsoluteTol( NALU_HYPRE_Solver solver,
                                     NALU_HYPRE_Real         a_tol    )
{
   return ( NALU_HYPRE_FlexGMRESSetAbsoluteTol( solver, a_tol ) );
}
/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRFlexGMRESSetMinIter
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParCSRFlexGMRESSetMinIter( NALU_HYPRE_Solver solver,
                                 NALU_HYPRE_Int          min_iter )
{
   return ( NALU_HYPRE_FlexGMRESSetMinIter( solver, min_iter ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRFlexGMRESSetMaxIter
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParCSRFlexGMRESSetMaxIter( NALU_HYPRE_Solver solver,
                                 NALU_HYPRE_Int          max_iter )
{
   return ( NALU_HYPRE_FlexGMRESSetMaxIter( solver, max_iter ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRFlexGMRESSetPrecond
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParCSRFlexGMRESSetPrecond( NALU_HYPRE_Solver          solver,
                                 NALU_HYPRE_PtrToParSolverFcn  precond,
                                 NALU_HYPRE_PtrToParSolverFcn  precond_setup,
                                 NALU_HYPRE_Solver          precond_solver )
{
   return ( NALU_HYPRE_FlexGMRESSetPrecond( solver,
                                       (NALU_HYPRE_PtrToSolverFcn) precond,
                                       (NALU_HYPRE_PtrToSolverFcn) precond_setup,
                                       precond_solver ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRFlexGMRESGetPrecond
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParCSRFlexGMRESGetPrecond( NALU_HYPRE_Solver  solver,
                                 NALU_HYPRE_Solver *precond_data_ptr )
{
   return ( NALU_HYPRE_FlexGMRESGetPrecond( solver, precond_data_ptr ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRFlexGMRESSetLogging
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParCSRFlexGMRESSetLogging( NALU_HYPRE_Solver solver,
                                 NALU_HYPRE_Int logging)
{
   return ( NALU_HYPRE_FlexGMRESSetLogging( solver, logging ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRFlexGMRESSetPrintLevel
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParCSRFlexGMRESSetPrintLevel( NALU_HYPRE_Solver solver,
                                    NALU_HYPRE_Int print_level)
{
   return ( NALU_HYPRE_FlexGMRESSetPrintLevel( solver, print_level ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRFlexGMRESGetNumIterations
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParCSRFlexGMRESGetNumIterations( NALU_HYPRE_Solver  solver,
                                       NALU_HYPRE_Int                *num_iterations )
{
   return ( NALU_HYPRE_FlexGMRESGetNumIterations( solver, num_iterations ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRFlexGMRESGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParCSRFlexGMRESGetFinalRelativeResidualNorm( NALU_HYPRE_Solver  solver,
                                                   NALU_HYPRE_Real         *norm   )
{
   return ( NALU_HYPRE_FlexGMRESGetFinalRelativeResidualNorm( solver, norm ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRFlexGMRESGetResidual
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParCSRFlexGMRESGetResidual( NALU_HYPRE_Solver  solver,
                                  NALU_HYPRE_ParVector *residual)
{
   return ( NALU_HYPRE_FlexGMRESGetResidual( solver, (void *) residual ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRFlexGMRESSetModifyPC
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int NALU_HYPRE_ParCSRFlexGMRESSetModifyPC( NALU_HYPRE_Solver  solver,
                                            NALU_HYPRE_PtrToModifyPCFcn modify_pc)

{
   return ( NALU_HYPRE_FlexGMRESSetModifyPC( solver,
                                        (NALU_HYPRE_PtrToModifyPCFcn) modify_pc));
}



