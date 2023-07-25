/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_nalu_hypre_parcsr_ls.h"

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SchwarzCreate
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SchwarzCreate( NALU_HYPRE_Solver *solver)
{
   if (!solver)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }
   *solver = (NALU_HYPRE_Solver) nalu_hypre_SchwarzCreate( ) ;
   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SchwarzDestroy
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SchwarzDestroy( NALU_HYPRE_Solver solver )
{
   return ( nalu_hypre_SchwarzDestroy( (void *) solver ) );
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
   return ( nalu_hypre_SchwarzSetup( (void *) solver,
                                (nalu_hypre_ParCSRMatrix *) A,
                                (nalu_hypre_ParVector *) b,
                                (nalu_hypre_ParVector *) x ) );
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


   return ( nalu_hypre_SchwarzSolve( (void *) solver,
                                (nalu_hypre_ParCSRMatrix *) A,
                                (nalu_hypre_ParVector *) b,
                                (nalu_hypre_ParVector *) x ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SchwarzSetVariant
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SchwarzSetVariant( NALU_HYPRE_Solver solver,
                         NALU_HYPRE_Int    variant )
{
   return ( nalu_hypre_SchwarzSetVariant( (void *) solver, variant ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SchwarzSetOverlap
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SchwarzSetOverlap( NALU_HYPRE_Solver solver, NALU_HYPRE_Int overlap)
{
   return ( nalu_hypre_SchwarzSetOverlap( (void *) solver, overlap ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SchwarzSetDomainType
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SchwarzSetDomainType( NALU_HYPRE_Solver solver,
                            NALU_HYPRE_Int    domain_type  )
{
   return ( nalu_hypre_SchwarzSetDomainType( (void *) solver, domain_type ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SchwarzSetDomainStructure
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SchwarzSetDomainStructure( NALU_HYPRE_Solver solver,
                                 NALU_HYPRE_CSRMatrix domain_structure  )
{
   return ( nalu_hypre_SchwarzSetDomainStructure(
               (void *) solver, (nalu_hypre_CSRMatrix *) domain_structure ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SchwarzSetNumFunctions
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SchwarzSetNumFunctions( NALU_HYPRE_Solver  solver,
                              NALU_HYPRE_Int     num_functions  )
{
   return ( nalu_hypre_SchwarzSetNumFunctions( (void *) solver, num_functions ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SchwarzSetNonSymm
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SchwarzSetNonSymm( NALU_HYPRE_Solver  solver,
                         NALU_HYPRE_Int     use_nonsymm  )
{
   return ( nalu_hypre_SchwarzSetNonSymm( (void *) solver, use_nonsymm ));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SchwarzSetRelaxWeight
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SchwarzSetRelaxWeight( NALU_HYPRE_Solver  solver,
                             NALU_HYPRE_Real relax_weight)
{
   return ( nalu_hypre_SchwarzSetRelaxWeight((void *) solver, relax_weight));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SchwarzSetDofFunc
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SchwarzSetDofFunc( NALU_HYPRE_Solver  solver,
                         NALU_HYPRE_Int    *dof_func  )
{
   return ( nalu_hypre_SchwarzSetDofFunc( (void *) solver, dof_func ) );
}

