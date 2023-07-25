/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_nalu_hypre_parcsr_ls.h"

/*--------------------------------------------------------------------------
 * NALU_HYPRE_AMSCreate
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int NALU_HYPRE_AMSCreate(NALU_HYPRE_Solver *solver)
{
   *solver = (NALU_HYPRE_Solver) nalu_hypre_AMSCreate();
   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_AMSDestroy
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int NALU_HYPRE_AMSDestroy(NALU_HYPRE_Solver solver)
{
   return nalu_hypre_AMSDestroy((void *) solver);
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_AMSSetup
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int NALU_HYPRE_AMSSetup (NALU_HYPRE_Solver solver,
                          NALU_HYPRE_ParCSRMatrix A,
                          NALU_HYPRE_ParVector b,
                          NALU_HYPRE_ParVector x)
{
   return nalu_hypre_AMSSetup((void *) solver,
                         (nalu_hypre_ParCSRMatrix *) A,
                         (nalu_hypre_ParVector *) b,
                         (nalu_hypre_ParVector *) x);
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_AMSSolve
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int NALU_HYPRE_AMSSolve (NALU_HYPRE_Solver solver,
                          NALU_HYPRE_ParCSRMatrix A,
                          NALU_HYPRE_ParVector b,
                          NALU_HYPRE_ParVector x)
{
   return nalu_hypre_AMSSolve((void *) solver,
                         (nalu_hypre_ParCSRMatrix *) A,
                         (nalu_hypre_ParVector *) b,
                         (nalu_hypre_ParVector *) x);
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_AMSSetDimension
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int NALU_HYPRE_AMSSetDimension(NALU_HYPRE_Solver solver,
                                NALU_HYPRE_Int dim)
{
   return nalu_hypre_AMSSetDimension((void *) solver, dim);
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_AMSSetDiscreteGradient
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int NALU_HYPRE_AMSSetDiscreteGradient(NALU_HYPRE_Solver solver,
                                       NALU_HYPRE_ParCSRMatrix G)
{
   return nalu_hypre_AMSSetDiscreteGradient((void *) solver,
                                       (nalu_hypre_ParCSRMatrix *) G);
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_AMSSetCoordinateVectors
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int NALU_HYPRE_AMSSetCoordinateVectors(NALU_HYPRE_Solver solver,
                                        NALU_HYPRE_ParVector x,
                                        NALU_HYPRE_ParVector y,
                                        NALU_HYPRE_ParVector z)
{
   return nalu_hypre_AMSSetCoordinateVectors((void *) solver,
                                        (nalu_hypre_ParVector *) x,
                                        (nalu_hypre_ParVector *) y,
                                        (nalu_hypre_ParVector *) z);
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_AMSSetEdgeConstantVectors
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int NALU_HYPRE_AMSSetEdgeConstantVectors(NALU_HYPRE_Solver solver,
                                          NALU_HYPRE_ParVector Gx,
                                          NALU_HYPRE_ParVector Gy,
                                          NALU_HYPRE_ParVector Gz)
{
   return nalu_hypre_AMSSetEdgeConstantVectors((void *) solver,
                                          (nalu_hypre_ParVector *) Gx,
                                          (nalu_hypre_ParVector *) Gy,
                                          (nalu_hypre_ParVector *) Gz);
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_AMSSetInterpolations
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int NALU_HYPRE_AMSSetInterpolations(NALU_HYPRE_Solver solver,
                                     NALU_HYPRE_ParCSRMatrix Pi,
                                     NALU_HYPRE_ParCSRMatrix Pix,
                                     NALU_HYPRE_ParCSRMatrix Piy,
                                     NALU_HYPRE_ParCSRMatrix Piz)
{
   return nalu_hypre_AMSSetInterpolations((void *) solver,
                                     (nalu_hypre_ParCSRMatrix *) Pi,
                                     (nalu_hypre_ParCSRMatrix *) Pix,
                                     (nalu_hypre_ParCSRMatrix *) Piy,
                                     (nalu_hypre_ParCSRMatrix *) Piz);
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_AMSSetAlphaPoissonMatrix
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int NALU_HYPRE_AMSSetAlphaPoissonMatrix(NALU_HYPRE_Solver solver,
                                         NALU_HYPRE_ParCSRMatrix A_alpha)
{
   return nalu_hypre_AMSSetAlphaPoissonMatrix((void *) solver,
                                         (nalu_hypre_ParCSRMatrix *) A_alpha);
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_AMSSetBetaPoissonMatrix
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int NALU_HYPRE_AMSSetBetaPoissonMatrix(NALU_HYPRE_Solver solver,
                                        NALU_HYPRE_ParCSRMatrix A_beta)
{
   return nalu_hypre_AMSSetBetaPoissonMatrix((void *) solver,
                                        (nalu_hypre_ParCSRMatrix *) A_beta);
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_AMSSetSetInteriorNodes
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int NALU_HYPRE_AMSSetInteriorNodes(NALU_HYPRE_Solver solver,
                                    NALU_HYPRE_ParVector interior_nodes)
{
   return nalu_hypre_AMSSetInteriorNodes((void *) solver,
                                    (nalu_hypre_ParVector *) interior_nodes);
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_AMSSetSetProjectionFrequency
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int NALU_HYPRE_AMSSetProjectionFrequency(NALU_HYPRE_Solver solver,
                                          NALU_HYPRE_Int projection_frequency)
{
   return nalu_hypre_AMSSetProjectionFrequency((void *) solver,
                                          projection_frequency);
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_AMSSetMaxIter
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int NALU_HYPRE_AMSSetMaxIter(NALU_HYPRE_Solver solver,
                              NALU_HYPRE_Int maxit)
{
   return nalu_hypre_AMSSetMaxIter((void *) solver, maxit);
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_AMSSetTol
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int NALU_HYPRE_AMSSetTol(NALU_HYPRE_Solver solver,
                          NALU_HYPRE_Real tol)
{
   return nalu_hypre_AMSSetTol((void *) solver, tol);
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_AMSSetCycleType
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int NALU_HYPRE_AMSSetCycleType(NALU_HYPRE_Solver solver,
                                NALU_HYPRE_Int cycle_type)
{
   return nalu_hypre_AMSSetCycleType((void *) solver, cycle_type);
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_AMSSetPrintLevel
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int NALU_HYPRE_AMSSetPrintLevel(NALU_HYPRE_Solver solver,
                                 NALU_HYPRE_Int print_level)
{
   return nalu_hypre_AMSSetPrintLevel((void *) solver, print_level);
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_AMSSetSmoothingOptions
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int NALU_HYPRE_AMSSetSmoothingOptions(NALU_HYPRE_Solver solver,
                                       NALU_HYPRE_Int relax_type,
                                       NALU_HYPRE_Int relax_times,
                                       NALU_HYPRE_Real relax_weight,
                                       NALU_HYPRE_Real omega)
{
   return nalu_hypre_AMSSetSmoothingOptions((void *) solver,
                                       relax_type,
                                       relax_times,
                                       relax_weight,
                                       omega);
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_AMSSetChebyOptions
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int NALU_HYPRE_AMSSetChebySmoothingOptions(NALU_HYPRE_Solver solver,
                                            NALU_HYPRE_Int cheby_order,
                                            NALU_HYPRE_Real cheby_fraction)
{
   return nalu_hypre_AMSSetChebySmoothingOptions((void *) solver,
                                            cheby_order,
                                            cheby_fraction);
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_AMSSetAlphaAMGOptions
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int NALU_HYPRE_AMSSetAlphaAMGOptions(NALU_HYPRE_Solver solver,
                                      NALU_HYPRE_Int alpha_coarsen_type,
                                      NALU_HYPRE_Int alpha_agg_levels,
                                      NALU_HYPRE_Int alpha_relax_type,
                                      NALU_HYPRE_Real alpha_strength_threshold,
                                      NALU_HYPRE_Int alpha_interp_type,
                                      NALU_HYPRE_Int alpha_Pmax)
{
   return nalu_hypre_AMSSetAlphaAMGOptions((void *) solver,
                                      alpha_coarsen_type,
                                      alpha_agg_levels,
                                      alpha_relax_type,
                                      alpha_strength_threshold,
                                      alpha_interp_type,
                                      alpha_Pmax);
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_AMSSetAlphaAMGCoarseRelaxType
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int NALU_HYPRE_AMSSetAlphaAMGCoarseRelaxType(NALU_HYPRE_Solver solver,
                                              NALU_HYPRE_Int alpha_coarse_relax_type)
{
   return nalu_hypre_AMSSetAlphaAMGCoarseRelaxType((void *) solver,
                                              alpha_coarse_relax_type);
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_AMSSetBetaAMGOptions
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int NALU_HYPRE_AMSSetBetaAMGOptions(NALU_HYPRE_Solver solver,
                                     NALU_HYPRE_Int beta_coarsen_type,
                                     NALU_HYPRE_Int beta_agg_levels,
                                     NALU_HYPRE_Int beta_relax_type,
                                     NALU_HYPRE_Real beta_strength_threshold,
                                     NALU_HYPRE_Int beta_interp_type,
                                     NALU_HYPRE_Int beta_Pmax)
{
   return nalu_hypre_AMSSetBetaAMGOptions((void *) solver,
                                     beta_coarsen_type,
                                     beta_agg_levels,
                                     beta_relax_type,
                                     beta_strength_threshold,
                                     beta_interp_type,
                                     beta_Pmax);
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_AMSSetBetaAMGCoarseRelaxType
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int NALU_HYPRE_AMSSetBetaAMGCoarseRelaxType(NALU_HYPRE_Solver solver,
                                             NALU_HYPRE_Int beta_coarse_relax_type)
{
   return nalu_hypre_AMSSetBetaAMGCoarseRelaxType((void *) solver,
                                             beta_coarse_relax_type);
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_AMSGetNumIterations
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int NALU_HYPRE_AMSGetNumIterations(NALU_HYPRE_Solver solver,
                                    NALU_HYPRE_Int *num_iterations)
{
   return nalu_hypre_AMSGetNumIterations((void *) solver,
                                    num_iterations);
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_AMSGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int NALU_HYPRE_AMSGetFinalRelativeResidualNorm(NALU_HYPRE_Solver solver,
                                                NALU_HYPRE_Real *rel_resid_norm)
{
   return nalu_hypre_AMSGetFinalRelativeResidualNorm((void *) solver,
                                                rel_resid_norm);
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_AMSProjectOutGradients
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int NALU_HYPRE_AMSProjectOutGradients(NALU_HYPRE_Solver solver,
                                       NALU_HYPRE_ParVector x)
{
   return nalu_hypre_AMSProjectOutGradients((void *) solver,
                                       (nalu_hypre_ParVector *) x);
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_AMSConstructDiscreteGradient
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int NALU_HYPRE_AMSConstructDiscreteGradient(NALU_HYPRE_ParCSRMatrix A,
                                             NALU_HYPRE_ParVector x_coord,
                                             NALU_HYPRE_BigInt *edge_vertex,
                                             NALU_HYPRE_Int edge_orientation,
                                             NALU_HYPRE_ParCSRMatrix *G)
{
   return nalu_hypre_AMSConstructDiscreteGradient((nalu_hypre_ParCSRMatrix *) A,
                                             (nalu_hypre_ParVector *) x_coord,
                                             edge_vertex,
                                             edge_orientation,
                                             (nalu_hypre_ParCSRMatrix **) G);
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_AMSFEISetup
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int NALU_HYPRE_AMSFEISetup(NALU_HYPRE_Solver solver,
                            NALU_HYPRE_ParCSRMatrix A,
                            NALU_HYPRE_ParVector b,
                            NALU_HYPRE_ParVector x,
                            NALU_HYPRE_BigInt *EdgeNodeList_,
                            NALU_HYPRE_BigInt *NodeNumbers_,
                            NALU_HYPRE_Int numEdges_,
                            NALU_HYPRE_Int numLocalNodes_,
                            NALU_HYPRE_Int numNodes_,
                            NALU_HYPRE_Real *NodalCoord_)
{
   return nalu_hypre_AMSFEISetup((void *) solver,
                            (nalu_hypre_ParCSRMatrix *) A,
                            (nalu_hypre_ParVector *) b,
                            (nalu_hypre_ParVector *) x,
                            numNodes_,
                            numLocalNodes_,
                            NodeNumbers_,
                            NodalCoord_,
                            numEdges_,
                            EdgeNodeList_);
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_AMSFEIDestroy
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int NALU_HYPRE_AMSFEIDestroy(NALU_HYPRE_Solver solver)
{
   return nalu_hypre_AMSFEIDestroy((void *) solver);
}
