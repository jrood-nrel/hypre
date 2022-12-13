/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_parcsr_ls.h"

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ADSCreate
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int NALU_HYPRE_ADSCreate(NALU_HYPRE_Solver *solver)
{
   *solver = (NALU_HYPRE_Solver) hypre_ADSCreate();
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ADSDestroy
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int NALU_HYPRE_ADSDestroy(NALU_HYPRE_Solver solver)
{
   return hypre_ADSDestroy((void *) solver);
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ADSSetup
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int NALU_HYPRE_ADSSetup (NALU_HYPRE_Solver solver,
                          NALU_HYPRE_ParCSRMatrix A,
                          NALU_HYPRE_ParVector b,
                          NALU_HYPRE_ParVector x)
{
   return hypre_ADSSetup((void *) solver,
                         (hypre_ParCSRMatrix *) A,
                         (hypre_ParVector *) b,
                         (hypre_ParVector *) x);
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ADSSolve
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int NALU_HYPRE_ADSSolve (NALU_HYPRE_Solver solver,
                          NALU_HYPRE_ParCSRMatrix A,
                          NALU_HYPRE_ParVector b,
                          NALU_HYPRE_ParVector x)
{
   return hypre_ADSSolve((void *) solver,
                         (hypre_ParCSRMatrix *) A,
                         (hypre_ParVector *) b,
                         (hypre_ParVector *) x);
}


/*--------------------------------------------------------------------------
 * NALU_HYPRE_ADSSetDiscreteCurl
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int NALU_HYPRE_ADSSetDiscreteCurl(NALU_HYPRE_Solver solver,
                                   NALU_HYPRE_ParCSRMatrix C)
{
   return hypre_ADSSetDiscreteCurl((void *) solver,
                                   (hypre_ParCSRMatrix *) C);
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ADSSetDiscreteGradient
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int NALU_HYPRE_ADSSetDiscreteGradient(NALU_HYPRE_Solver solver,
                                       NALU_HYPRE_ParCSRMatrix G)
{
   return hypre_ADSSetDiscreteGradient((void *) solver,
                                       (hypre_ParCSRMatrix *) G);
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ADSSetCoordinateVectors
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int NALU_HYPRE_ADSSetCoordinateVectors(NALU_HYPRE_Solver solver,
                                        NALU_HYPRE_ParVector x,
                                        NALU_HYPRE_ParVector y,
                                        NALU_HYPRE_ParVector z)
{
   return hypre_ADSSetCoordinateVectors((void *) solver,
                                        (hypre_ParVector *) x,
                                        (hypre_ParVector *) y,
                                        (hypre_ParVector *) z);
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ADSSetInterpolations
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int NALU_HYPRE_ADSSetInterpolations(NALU_HYPRE_Solver solver,
                                     NALU_HYPRE_ParCSRMatrix RT_Pi,
                                     NALU_HYPRE_ParCSRMatrix RT_Pix,
                                     NALU_HYPRE_ParCSRMatrix RT_Piy,
                                     NALU_HYPRE_ParCSRMatrix RT_Piz,
                                     NALU_HYPRE_ParCSRMatrix ND_Pi,
                                     NALU_HYPRE_ParCSRMatrix ND_Pix,
                                     NALU_HYPRE_ParCSRMatrix ND_Piy,
                                     NALU_HYPRE_ParCSRMatrix ND_Piz)
{
   return hypre_ADSSetInterpolations((void *) solver,
                                     (hypre_ParCSRMatrix *) RT_Pi,
                                     (hypre_ParCSRMatrix *) RT_Pix,
                                     (hypre_ParCSRMatrix *) RT_Piy,
                                     (hypre_ParCSRMatrix *) RT_Piz,
                                     (hypre_ParCSRMatrix *) ND_Pi,
                                     (hypre_ParCSRMatrix *) ND_Pix,
                                     (hypre_ParCSRMatrix *) ND_Piy,
                                     (hypre_ParCSRMatrix *) ND_Piz);

}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ADSSetMaxIter
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int NALU_HYPRE_ADSSetMaxIter(NALU_HYPRE_Solver solver,
                              NALU_HYPRE_Int maxit)
{
   return hypre_ADSSetMaxIter((void *) solver, maxit);
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ADSSetTol
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int NALU_HYPRE_ADSSetTol(NALU_HYPRE_Solver solver,
                          NALU_HYPRE_Real tol)
{
   return hypre_ADSSetTol((void *) solver, tol);
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ADSSetCycleType
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int NALU_HYPRE_ADSSetCycleType(NALU_HYPRE_Solver solver,
                                NALU_HYPRE_Int cycle_type)
{
   return hypre_ADSSetCycleType((void *) solver, cycle_type);
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ADSSetPrintLevel
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int NALU_HYPRE_ADSSetPrintLevel(NALU_HYPRE_Solver solver,
                                 NALU_HYPRE_Int print_level)
{
   return hypre_ADSSetPrintLevel((void *) solver, print_level);
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ADSSetSmoothingOptions
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int NALU_HYPRE_ADSSetSmoothingOptions(NALU_HYPRE_Solver solver,
                                       NALU_HYPRE_Int relax_type,
                                       NALU_HYPRE_Int relax_times,
                                       NALU_HYPRE_Real relax_weight,
                                       NALU_HYPRE_Real omega)
{
   return hypre_ADSSetSmoothingOptions((void *) solver,
                                       relax_type,
                                       relax_times,
                                       relax_weight,
                                       omega);
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ADSSetChebyOptions
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int NALU_HYPRE_ADSSetChebySmoothingOptions(NALU_HYPRE_Solver solver,
                                            NALU_HYPRE_Int cheby_order,
                                            NALU_HYPRE_Int cheby_fraction)
{
   return hypre_ADSSetChebySmoothingOptions((void *) solver,
                                            cheby_order,
                                            cheby_fraction);
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ADSSetAMSOptions
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int NALU_HYPRE_ADSSetAMSOptions(NALU_HYPRE_Solver solver,
                                 NALU_HYPRE_Int cycle_type,
                                 NALU_HYPRE_Int coarsen_type,
                                 NALU_HYPRE_Int agg_levels,
                                 NALU_HYPRE_Int relax_type,
                                 NALU_HYPRE_Real strength_threshold,
                                 NALU_HYPRE_Int interp_type,
                                 NALU_HYPRE_Int Pmax)
{
   return hypre_ADSSetAMSOptions((void *) solver,
                                 cycle_type,
                                 coarsen_type,
                                 agg_levels,
                                 relax_type,
                                 strength_threshold,
                                 interp_type,
                                 Pmax);
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ADSSetAlphaAMGOptions
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int NALU_HYPRE_ADSSetAMGOptions(NALU_HYPRE_Solver solver,
                                 NALU_HYPRE_Int coarsen_type,
                                 NALU_HYPRE_Int agg_levels,
                                 NALU_HYPRE_Int relax_type,
                                 NALU_HYPRE_Real strength_threshold,
                                 NALU_HYPRE_Int interp_type,
                                 NALU_HYPRE_Int Pmax)
{
   return hypre_ADSSetAMGOptions((void *) solver,
                                 coarsen_type,
                                 agg_levels,
                                 relax_type,
                                 strength_threshold,
                                 interp_type,
                                 Pmax);
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ADSGetNumIterations
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int NALU_HYPRE_ADSGetNumIterations(NALU_HYPRE_Solver solver,
                                    NALU_HYPRE_Int *num_iterations)
{
   return hypre_ADSGetNumIterations((void *) solver,
                                    num_iterations);
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ADSGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int NALU_HYPRE_ADSGetFinalRelativeResidualNorm(NALU_HYPRE_Solver solver,
                                                NALU_HYPRE_Real *rel_resid_norm)
{
   return hypre_ADSGetFinalRelativeResidualNorm((void *) solver,
                                                rel_resid_norm);
}
