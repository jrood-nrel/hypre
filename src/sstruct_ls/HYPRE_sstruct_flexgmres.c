/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_nalu_hypre_sstruct_ls.h"

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructFlexGMRESCreate( MPI_Comm             comm,
                              NALU_HYPRE_SStructSolver *solver )
{
   nalu_hypre_FlexGMRESFunctions * fgmres_functions =
      nalu_hypre_FlexGMRESFunctionsCreate(
         nalu_hypre_SStructKrylovCAlloc, nalu_hypre_SStructKrylovFree, nalu_hypre_SStructKrylovCommInfo,
         nalu_hypre_SStructKrylovCreateVector,
         nalu_hypre_SStructKrylovCreateVectorArray,
         nalu_hypre_SStructKrylovDestroyVector, nalu_hypre_SStructKrylovMatvecCreate,
         nalu_hypre_SStructKrylovMatvec, nalu_hypre_SStructKrylovMatvecDestroy,
         nalu_hypre_SStructKrylovInnerProd, nalu_hypre_SStructKrylovCopyVector,
         nalu_hypre_SStructKrylovClearVector,
         nalu_hypre_SStructKrylovScaleVector, nalu_hypre_SStructKrylovAxpy,
         nalu_hypre_SStructKrylovIdentitySetup, nalu_hypre_SStructKrylovIdentity );

   *solver = ( (NALU_HYPRE_SStructSolver) nalu_hypre_FlexGMRESCreate( fgmres_functions ) );

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructFlexGMRESDestroy( NALU_HYPRE_SStructSolver solver )
{
   return ( nalu_hypre_FlexGMRESDestroy( (void *) solver ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructFlexGMRESSetup( NALU_HYPRE_SStructSolver solver,
                             NALU_HYPRE_SStructMatrix A,
                             NALU_HYPRE_SStructVector b,
                             NALU_HYPRE_SStructVector x )
{
   return ( NALU_HYPRE_FlexGMRESSetup( (NALU_HYPRE_Solver) solver,
                                  (NALU_HYPRE_Matrix) A,
                                  (NALU_HYPRE_Vector) b,
                                  (NALU_HYPRE_Vector) x ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructFlexGMRESSolve( NALU_HYPRE_SStructSolver solver,
                             NALU_HYPRE_SStructMatrix A,
                             NALU_HYPRE_SStructVector b,
                             NALU_HYPRE_SStructVector x )
{
   return ( NALU_HYPRE_FlexGMRESSolve( (NALU_HYPRE_Solver) solver,
                                  (NALU_HYPRE_Matrix) A,
                                  (NALU_HYPRE_Vector) b,
                                  (NALU_HYPRE_Vector) x ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructFlexGMRESSetKDim( NALU_HYPRE_SStructSolver solver,
                               NALU_HYPRE_Int           k_dim )
{
   return ( NALU_HYPRE_FlexGMRESSetKDim( (NALU_HYPRE_Solver) solver, k_dim ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructFlexGMRESSetTol( NALU_HYPRE_SStructSolver solver,
                              NALU_HYPRE_Real          tol )
{
   return ( NALU_HYPRE_FlexGMRESSetTol( (NALU_HYPRE_Solver) solver, tol ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructFlexGMRESSetAbsoluteTol( NALU_HYPRE_SStructSolver solver,
                                      NALU_HYPRE_Real          tol )
{
   return ( NALU_HYPRE_FlexGMRESSetAbsoluteTol( (NALU_HYPRE_Solver) solver, tol ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructFlexGMRESSetMinIter( NALU_HYPRE_SStructSolver solver,
                                  NALU_HYPRE_Int           min_iter )
{
   return ( NALU_HYPRE_FlexGMRESSetMinIter( (NALU_HYPRE_Solver) solver, min_iter ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructFlexGMRESSetMaxIter( NALU_HYPRE_SStructSolver solver,
                                  NALU_HYPRE_Int           max_iter )
{
   return ( NALU_HYPRE_FlexGMRESSetMaxIter( (NALU_HYPRE_Solver) solver, max_iter ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructFlexGMRESSetPrecond( NALU_HYPRE_SStructSolver          solver,
                                  NALU_HYPRE_PtrToSStructSolverFcn  precond,
                                  NALU_HYPRE_PtrToSStructSolverFcn  precond_setup,
                                  void *          precond_data )
{
   return ( NALU_HYPRE_FlexGMRESSetPrecond( (NALU_HYPRE_Solver) solver,
                                       (NALU_HYPRE_PtrToSolverFcn) precond,
                                       (NALU_HYPRE_PtrToSolverFcn) precond_setup,
                                       (NALU_HYPRE_Solver) precond_data ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructFlexGMRESSetLogging( NALU_HYPRE_SStructSolver solver,
                                  NALU_HYPRE_Int           logging )
{
   return ( NALU_HYPRE_FlexGMRESSetLogging( (NALU_HYPRE_Solver) solver, logging ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructFlexGMRESSetPrintLevel( NALU_HYPRE_SStructSolver solver,
                                     NALU_HYPRE_Int           level )
{
   return ( NALU_HYPRE_FlexGMRESSetPrintLevel( (NALU_HYPRE_Solver) solver, level ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructFlexGMRESGetNumIterations( NALU_HYPRE_SStructSolver  solver,
                                        NALU_HYPRE_Int           *num_iterations )
{
   return ( NALU_HYPRE_FlexGMRESGetNumIterations( (NALU_HYPRE_Solver) solver,
                                             num_iterations ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructFlexGMRESGetFinalRelativeResidualNorm( NALU_HYPRE_SStructSolver  solver,
                                                    NALU_HYPRE_Real          *norm )
{
   return ( NALU_HYPRE_FlexGMRESGetFinalRelativeResidualNorm( (NALU_HYPRE_Solver) solver,
                                                         norm ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructFlexGMRESGetResidual( NALU_HYPRE_SStructSolver  solver,
                                   void              **residual )
{
   return ( NALU_HYPRE_FlexGMRESGetResidual( (NALU_HYPRE_Solver) solver, residual ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/


NALU_HYPRE_Int NALU_HYPRE_SStructFlexGMRESSetModifyPC( NALU_HYPRE_SStructSolver  solver,
                                             NALU_HYPRE_PtrToModifyPCFcn modify_pc)

{
   return ( NALU_HYPRE_FlexGMRESSetModifyPC( (NALU_HYPRE_Solver) solver,
                                        (NALU_HYPRE_PtrToModifyPCFcn) modify_pc));

}

