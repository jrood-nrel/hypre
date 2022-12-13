/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_parcsr_ls.h"

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SchwarzCreate
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SchwarzCreate( NALU_HYPRE_Solver *solver)
{
   if (!solver)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   *solver = (NALU_HYPRE_Solver) hypre_SchwarzCreate( ) ;
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SchwarzDestroy
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SchwarzDestroy( NALU_HYPRE_Solver solver )
{
   return ( hypre_SchwarzDestroy( (void *) solver ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SchwarzSetup
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SchwarzSetup(NALU_HYPRE_Solver solver,
                   NALU_HYPRE_ParCSRMatrix A,
                   NALU_HYPRE_ParVector b,
                   NALU_HYPRE_ParVector x      )
{
   return ( hypre_SchwarzSetup( (void *) solver,
                                (hypre_ParCSRMatrix *) A,
                                (hypre_ParVector *) b,
                                (hypre_ParVector *) x ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SchwarzSolve
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SchwarzSolve( NALU_HYPRE_Solver solver,
                    NALU_HYPRE_ParCSRMatrix A,
                    NALU_HYPRE_ParVector b,
                    NALU_HYPRE_ParVector x      )
{


   return ( hypre_SchwarzSolve( (void *) solver,
                                (hypre_ParCSRMatrix *) A,
                                (hypre_ParVector *) b,
                                (hypre_ParVector *) x ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SchwarzSetVariant
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SchwarzSetVariant( NALU_HYPRE_Solver solver,
                         NALU_HYPRE_Int    variant )
{
   return ( hypre_SchwarzSetVariant( (void *) solver, variant ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SchwarzSetOverlap
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SchwarzSetOverlap( NALU_HYPRE_Solver solver, NALU_HYPRE_Int overlap)
{
   return ( hypre_SchwarzSetOverlap( (void *) solver, overlap ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SchwarzSetDomainType
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SchwarzSetDomainType( NALU_HYPRE_Solver solver,
                            NALU_HYPRE_Int    domain_type  )
{
   return ( hypre_SchwarzSetDomainType( (void *) solver, domain_type ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SchwarzSetDomainStructure
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SchwarzSetDomainStructure( NALU_HYPRE_Solver solver,
                                 NALU_HYPRE_CSRMatrix domain_structure  )
{
   return ( hypre_SchwarzSetDomainStructure(
               (void *) solver, (hypre_CSRMatrix *) domain_structure ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SchwarzSetNumFunctions
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SchwarzSetNumFunctions( NALU_HYPRE_Solver  solver,
                              NALU_HYPRE_Int     num_functions  )
{
   return ( hypre_SchwarzSetNumFunctions( (void *) solver, num_functions ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SchwarzSetNonSymm
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SchwarzSetNonSymm( NALU_HYPRE_Solver  solver,
                         NALU_HYPRE_Int     use_nonsymm  )
{
   return ( hypre_SchwarzSetNonSymm( (void *) solver, use_nonsymm ));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SchwarzSetRelaxWeight
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SchwarzSetRelaxWeight( NALU_HYPRE_Solver  solver,
                             NALU_HYPRE_Real relax_weight)
{
   return ( hypre_SchwarzSetRelaxWeight((void *) solver, relax_weight));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SchwarzSetDofFunc
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SchwarzSetDofFunc( NALU_HYPRE_Solver  solver,
                         NALU_HYPRE_Int    *dof_func  )
{
   return ( hypre_SchwarzSetDofFunc( (void *) solver, dof_func ) );
}

