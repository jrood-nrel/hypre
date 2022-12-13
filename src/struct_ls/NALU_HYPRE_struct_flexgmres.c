/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_struct_ls.h"

/*==========================================================================*/

NALU_HYPRE_Int
NALU_HYPRE_StructFlexGMRESCreate( MPI_Comm comm, NALU_HYPRE_StructSolver *solver )
{
   hypre_FlexGMRESFunctions * fgmres_functions =
      hypre_FlexGMRESFunctionsCreate(
         hypre_StructKrylovCAlloc, hypre_StructKrylovFree,
         hypre_StructKrylovCommInfo,
         hypre_StructKrylovCreateVector,
         hypre_StructKrylovCreateVectorArray,
         hypre_StructKrylovDestroyVector, hypre_StructKrylovMatvecCreate,
         hypre_StructKrylovMatvec, hypre_StructKrylovMatvecDestroy,
         hypre_StructKrylovInnerProd, hypre_StructKrylovCopyVector,
         hypre_StructKrylovClearVector,
         hypre_StructKrylovScaleVector, hypre_StructKrylovAxpy,
         hypre_StructKrylovIdentitySetup, hypre_StructKrylovIdentity );

   *solver = ( (NALU_HYPRE_StructSolver) hypre_FlexGMRESCreate( fgmres_functions ) );

   return hypre_error_flag;
}

/*==========================================================================*/

NALU_HYPRE_Int
NALU_HYPRE_StructFlexGMRESDestroy( NALU_HYPRE_StructSolver solver )
{
   return ( hypre_FlexGMRESDestroy( (void *) solver ) );
}

/*==========================================================================*/

NALU_HYPRE_Int
NALU_HYPRE_StructFlexGMRESSetup( NALU_HYPRE_StructSolver solver,
                            NALU_HYPRE_StructMatrix A,
                            NALU_HYPRE_StructVector b,
                            NALU_HYPRE_StructVector x      )
{
   return ( NALU_HYPRE_FlexGMRESSetup( (NALU_HYPRE_Solver) solver,
                                  (NALU_HYPRE_Matrix) A,
                                  (NALU_HYPRE_Vector) b,
                                  (NALU_HYPRE_Vector) x ) );
}

/*==========================================================================*/

NALU_HYPRE_Int
NALU_HYPRE_StructFlexGMRESSolve( NALU_HYPRE_StructSolver solver,
                            NALU_HYPRE_StructMatrix A,
                            NALU_HYPRE_StructVector b,
                            NALU_HYPRE_StructVector x      )
{
   return ( NALU_HYPRE_FlexGMRESSolve( (NALU_HYPRE_Solver) solver,
                                  (NALU_HYPRE_Matrix) A,
                                  (NALU_HYPRE_Vector) b,
                                  (NALU_HYPRE_Vector) x ) );
}

/*==========================================================================*/

NALU_HYPRE_Int
NALU_HYPRE_StructFlexGMRESSetTol( NALU_HYPRE_StructSolver solver,
                             NALU_HYPRE_Real         tol    )
{
   return ( NALU_HYPRE_FlexGMRESSetTol( (NALU_HYPRE_Solver) solver, tol ) );
}

/*==========================================================================*/

NALU_HYPRE_Int
NALU_HYPRE_StructFlexGMRESSetAbsoluteTol( NALU_HYPRE_StructSolver solver,
                                     NALU_HYPRE_Real         atol    )
{
   return ( NALU_HYPRE_FlexGMRESSetAbsoluteTol( (NALU_HYPRE_Solver) solver, atol ) );
}

/*==========================================================================*/

NALU_HYPRE_Int
NALU_HYPRE_StructFlexGMRESSetMaxIter( NALU_HYPRE_StructSolver solver,
                                 NALU_HYPRE_Int          max_iter )
{
   return ( NALU_HYPRE_FlexGMRESSetMaxIter( (NALU_HYPRE_Solver) solver, max_iter ) );
}

/*==========================================================================*/

NALU_HYPRE_Int
NALU_HYPRE_StructFlexGMRESSetKDim( NALU_HYPRE_StructSolver solver,
                              NALU_HYPRE_Int          k_dim )
{
   return ( NALU_HYPRE_FlexGMRESSetKDim( (NALU_HYPRE_Solver) solver, k_dim ) );
}

/*==========================================================================*/

NALU_HYPRE_Int
NALU_HYPRE_StructFlexGMRESSetPrecond( NALU_HYPRE_StructSolver         solver,
                                 NALU_HYPRE_PtrToStructSolverFcn precond,
                                 NALU_HYPRE_PtrToStructSolverFcn precond_setup,
                                 NALU_HYPRE_StructSolver         precond_solver )
{
   return ( NALU_HYPRE_FlexGMRESSetPrecond( (NALU_HYPRE_Solver) solver,
                                       (NALU_HYPRE_PtrToSolverFcn) precond,
                                       (NALU_HYPRE_PtrToSolverFcn) precond_setup,
                                       (NALU_HYPRE_Solver) precond_solver ) );
}

/*==========================================================================*/

NALU_HYPRE_Int
NALU_HYPRE_StructFlexGMRESSetLogging( NALU_HYPRE_StructSolver solver,
                                 NALU_HYPRE_Int          logging )
{
   return ( NALU_HYPRE_FlexGMRESSetLogging( (NALU_HYPRE_Solver) solver, logging ) );
}

/*==========================================================================*/

NALU_HYPRE_Int
NALU_HYPRE_StructFlexGMRESSetPrintLevel( NALU_HYPRE_StructSolver solver,
                                    NALU_HYPRE_Int          print_level )
{
   return ( NALU_HYPRE_FlexGMRESSetPrintLevel( (NALU_HYPRE_Solver) solver, print_level ) );
}

/*==========================================================================*/

NALU_HYPRE_Int
NALU_HYPRE_StructFlexGMRESGetNumIterations( NALU_HYPRE_StructSolver  solver,
                                       NALU_HYPRE_Int          *num_iterations )
{
   return ( NALU_HYPRE_FlexGMRESGetNumIterations( (NALU_HYPRE_Solver) solver,
                                             num_iterations ) );
}

/*==========================================================================*/

NALU_HYPRE_Int
NALU_HYPRE_StructFlexGMRESGetFinalRelativeResidualNorm( NALU_HYPRE_StructSolver  solver,
                                                   NALU_HYPRE_Real         *norm   )
{
   return ( NALU_HYPRE_FlexGMRESGetFinalRelativeResidualNorm( (NALU_HYPRE_Solver) solver,
                                                         norm ) );
}

/*==========================================================================*/

NALU_HYPRE_Int NALU_HYPRE_StructFlexGMRESSetModifyPC( NALU_HYPRE_StructSolver  solver,
                                            NALU_HYPRE_PtrToModifyPCFcn modify_pc)
{
   return ( NALU_HYPRE_FlexGMRESSetModifyPC( (NALU_HYPRE_Solver) solver,
                                        (NALU_HYPRE_PtrToModifyPCFcn) modify_pc));
}

