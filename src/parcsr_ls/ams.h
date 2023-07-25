/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#ifndef nalu_hypre_AMS_DATA_HEADER
#define nalu_hypre_AMS_DATA_HEADER

/*--------------------------------------------------------------------------
 * Auxiliary space Maxwell Solver data
 *--------------------------------------------------------------------------*/
typedef struct
{
   /* Space dimension (2 or 3) */
   NALU_HYPRE_Int dim;

   /* Edge element (ND1) stiffness matrix */
   nalu_hypre_ParCSRMatrix *A;

   /* Discrete gradient matrix (vertex-to-edge) */
   nalu_hypre_ParCSRMatrix *G;
   /* Coarse grid matrix on the range of G^T */
   nalu_hypre_ParCSRMatrix *A_G;
   /* AMG solver for A_G */
   NALU_HYPRE_Solver B_G;
   /* Is the mass term coefficient zero? */
   NALU_HYPRE_Int beta_is_zero;

   /* Nedelec nodal interpolation matrix (vertex^dim-to-edge) */
   nalu_hypre_ParCSRMatrix *Pi;
   /* Coarse grid matrix on the range of Pi^T */
   nalu_hypre_ParCSRMatrix *A_Pi;
   /* AMG solver for A_Pi */
   NALU_HYPRE_Solver B_Pi;

   /* Components of the Nedelec interpolation matrix (vertex-to-edge each) */
   nalu_hypre_ParCSRMatrix *Pix, *Piy, *Piz;
   /* Coarse grid matrices on the ranges of Pi{x,y,z}^T */
   nalu_hypre_ParCSRMatrix *A_Pix, *A_Piy, *A_Piz;
   /* AMG solvers for A_Pi{x,y,z} */
   NALU_HYPRE_Solver B_Pix, B_Piy, B_Piz;

   /* Does the solver own the Nedelec interpolations? */
   NALU_HYPRE_Int owns_Pi;
   /* Does the solver own the coarse grid matrices? */
   NALU_HYPRE_Int owns_A_G, owns_A_Pi;

   /* Coordinates of the vertices (z = 0 if dim == 2) */
   nalu_hypre_ParVector *x, *y, *z;

   /* Representations of the constant vectors in the Nedelec basis */
   nalu_hypre_ParVector *Gx, *Gy, *Gz;

   /* Nodes in the interior of the zero-conductivity region */
   nalu_hypre_ParVector *interior_nodes;
   /* Discrete gradient matrix for the interior nodes only */
   nalu_hypre_ParCSRMatrix *G0;
   /* Coarse grid matrix on the interior nodes */
   nalu_hypre_ParCSRMatrix *A_G0;
   /* AMG solver for A_G0 */
   NALU_HYPRE_Solver B_G0;
   /* How frequently to project the r.h.s. onto Ker(G0^T)? */
   NALU_HYPRE_Int projection_frequency;
   /* Internal counter to use with projection_frequency in PCG */
   NALU_HYPRE_Int solve_counter;

   /* Solver options */
   NALU_HYPRE_Int maxit;
   NALU_HYPRE_Real tol;
   NALU_HYPRE_Int cycle_type;
   NALU_HYPRE_Int print_level;

   /* Smoothing options for A */
   NALU_HYPRE_Int A_relax_type;
   NALU_HYPRE_Int A_relax_times;
   nalu_hypre_Vector *A_l1_norms;
   NALU_HYPRE_Real A_relax_weight;
   NALU_HYPRE_Real A_omega;
   NALU_HYPRE_Real A_max_eig_est;
   NALU_HYPRE_Real A_min_eig_est;
   NALU_HYPRE_Int A_cheby_order;
   NALU_HYPRE_Real  A_cheby_fraction;

   /* AMG options for B_G */
   NALU_HYPRE_Int B_G_coarsen_type;
   NALU_HYPRE_Int B_G_agg_levels;
   NALU_HYPRE_Int B_G_relax_type;
   NALU_HYPRE_Int B_G_coarse_relax_type;
   NALU_HYPRE_Real B_G_theta;
   NALU_HYPRE_Int B_G_interp_type;
   NALU_HYPRE_Int B_G_Pmax;

   /* AMG options for B_Pi */
   NALU_HYPRE_Int B_Pi_coarsen_type;
   NALU_HYPRE_Int B_Pi_agg_levels;
   NALU_HYPRE_Int B_Pi_relax_type;
   NALU_HYPRE_Int B_Pi_coarse_relax_type;
   NALU_HYPRE_Real B_Pi_theta;
   NALU_HYPRE_Int B_Pi_interp_type;
   NALU_HYPRE_Int B_Pi_Pmax;

   /* Temporary vectors */
   nalu_hypre_ParVector *r0, *g0, *r1, *g1, *r2, *g2, *zz;

   /* Output log info */
   NALU_HYPRE_Int num_iterations;
   NALU_HYPRE_Real rel_resid_norm;

} nalu_hypre_AMSData;

/* Space dimension */
#define nalu_hypre_AMSDataDimension(ams_data) ((ams_data)->dim)

/* Edge stiffness matrix */
#define nalu_hypre_AMSDataA(ams_data) ((ams_data)->A)

/* Vertex space data */
#define nalu_hypre_AMSDataDiscreteGradient(ams_data) ((ams_data)->G)
#define nalu_hypre_AMSDataPoissonBeta(ams_data) ((ams_data)->A_G)
#define nalu_hypre_AMSDataPoissonBetaAMG(ams_data) ((ams_data)->B_G)
#define nalu_hypre_AMSDataOwnsPoissonBeta(ams_data) ((ams_data)->owns_A_G)
#define nalu_hypre_AMSDataBetaIsZero(ams_data) ((ams_data)->beta_is_zero)

/* Vector vertex space data */
#define nalu_hypre_AMSDataPiInterpolation(ams_data) ((ams_data)->Pi)
#define nalu_hypre_AMSDataOwnsPiInterpolation(ams_data) ((ams_data)->owns_Pi)
#define nalu_hypre_AMSDataPoissonAlpha(ams_data) ((ams_data)->A_Pi)
#define nalu_hypre_AMSDataPoissonAlphaAMG(ams_data) ((ams_data)->B_Pi)
#define nalu_hypre_AMSDataOwnsPoissonAlpha(ams_data) ((ams_data)->owns_A_Pi)

/* Vector vertex components data */
#define nalu_hypre_AMSDataPiXInterpolation(ams_data) ((ams_data)->Pix)
#define nalu_hypre_AMSDataPiYInterpolation(ams_data) ((ams_data)->Piy)
#define nalu_hypre_AMSDataPiZInterpolation(ams_data) ((ams_data)->Piz)
#define nalu_hypre_AMSDataPoissonAlphaX(ams_data) ((ams_data)->A_Pix)
#define nalu_hypre_AMSDataPoissonAlphaY(ams_data) ((ams_data)->A_Piy)
#define nalu_hypre_AMSDataPoissonAlphaZ(ams_data) ((ams_data)->A_Piz)
#define nalu_hypre_AMSDataPoissonAlphaXAMG(ams_data) ((ams_data)->B_Pix)
#define nalu_hypre_AMSDataPoissonAlphaYAMG(ams_data) ((ams_data)->B_Piy)
#define nalu_hypre_AMSDataPoissonAlphaZAMG(ams_data) ((ams_data)->B_Piz)

/* Coordinates of the vertices */
#define nalu_hypre_AMSDataVertexCoordinateX(ams_data) ((ams_data)->x)
#define nalu_hypre_AMSDataVertexCoordinateY(ams_data) ((ams_data)->y)
#define nalu_hypre_AMSDataVertexCoordinateZ(ams_data) ((ams_data)->z)

/* Representations of the constant vectors in the Nedelec basis */
#define nalu_hypre_AMSDataEdgeConstantX(ams_data) ((ams_data)->Gx)
#define nalu_hypre_AMSDataEdgeConstantY(ams_data) ((ams_data)->Gy)
#define nalu_hypre_AMSDataEdgeConstantZ(ams_data) ((ams_data)->Gz)

/* Interior zero conductivity region */
#define nalu_hypre_AMSDataInteriorNodes(ams_data) ((ams_data)->interior_nodes)
#define nalu_hypre_AMSDataInteriorDiscreteGradient(ams_data) ((ams_data)->G0)
#define nalu_hypre_AMSDataInteriorPoissonBeta(ams_data) ((ams_data)->A_G0)
#define nalu_hypre_AMSDataInteriorPoissonBetaAMG(ams_data) ((ams_data)->B_G0)
#define nalu_hypre_AMSDataInteriorProjectionFrequency(ams_data) ((ams_data)->projection_frequency)
#define nalu_hypre_AMSDataInteriorSolveCounter(ams_data) ((ams_data)->solve_counter)

/* Solver options */
#define nalu_hypre_AMSDataMaxIter(ams_data) ((ams_data)->maxit)
#define nalu_hypre_AMSDataTol(ams_data) ((ams_data)->tol)
#define nalu_hypre_AMSDataCycleType(ams_data) ((ams_data)->cycle_type)
#define nalu_hypre_AMSDataPrintLevel(ams_data) ((ams_data)->print_level)

/* Smoothing and AMG options */
#define nalu_hypre_AMSDataARelaxType(ams_data) ((ams_data)->A_relax_type)
#define nalu_hypre_AMSDataARelaxTimes(ams_data) ((ams_data)->A_relax_times)
#define nalu_hypre_AMSDataAL1Norms(ams_data) ((ams_data)->A_l1_norms)
#define nalu_hypre_AMSDataARelaxWeight(ams_data) ((ams_data)->A_relax_weight)
#define nalu_hypre_AMSDataAOmega(ams_data) ((ams_data)->A_omega)
#define nalu_hypre_AMSDataAMaxEigEst(ams_data) ((ams_data)->A_max_eig_est)
#define nalu_hypre_AMSDataAMinEigEst(ams_data) ((ams_data)->A_min_eig_est)
#define nalu_hypre_AMSDataAChebyOrder(ams_data) ((ams_data)->A_cheby_order)
#define nalu_hypre_AMSDataAChebyFraction(ams_data) ((ams_data)->A_cheby_fraction)

#define nalu_hypre_AMSDataPoissonBetaAMGCoarsenType(ams_data) ((ams_data)->B_G_coarsen_type)
#define nalu_hypre_AMSDataPoissonBetaAMGAggLevels(ams_data) ((ams_data)->B_G_agg_levels)
#define nalu_hypre_AMSDataPoissonBetaAMGRelaxType(ams_data) ((ams_data)->B_G_relax_type)
#define nalu_hypre_AMSDataPoissonBetaAMGCoarseRelaxType(ams_data) ((ams_data)->B_G_coarse_relax_type)
#define nalu_hypre_AMSDataPoissonBetaAMGStrengthThreshold(ams_data) ((ams_data)->B_G_theta)
#define nalu_hypre_AMSDataPoissonBetaAMGInterpType(ams_data) ((ams_data)->B_G_interp_type)
#define nalu_hypre_AMSDataPoissonBetaAMGPMax(ams_data) ((ams_data)->B_G_Pmax)

#define nalu_hypre_AMSDataPoissonAlphaAMGCoarsenType(ams_data) ((ams_data)->B_Pi_coarsen_type)
#define nalu_hypre_AMSDataPoissonAlphaAMGAggLevels(ams_data) ((ams_data)->B_Pi_agg_levels)
#define nalu_hypre_AMSDataPoissonAlphaAMGRelaxType(ams_data) ((ams_data)->B_Pi_relax_type)
#define nalu_hypre_AMSDataPoissonAlphaAMGCoarseRelaxType(ams_data) ((ams_data)->B_Pi_coarse_relax_type)
#define nalu_hypre_AMSDataPoissonAlphaAMGStrengthThreshold(ams_data) ((ams_data)->B_Pi_theta)
#define nalu_hypre_AMSDataPoissonAlphaAMGInterpType(ams_data) ((ams_data)->B_Pi_interp_type)
#define nalu_hypre_AMSDataPoissonAlphaAMGPMax(ams_data) ((ams_data)->B_Pi_Pmax)

/* Temporary vectors */
#define nalu_hypre_AMSDataTempEdgeVectorR(ams_data) ((ams_data)->r0)
#define nalu_hypre_AMSDataTempEdgeVectorG(ams_data) ((ams_data)->g0)
#define nalu_hypre_AMSDataTempVertexVectorR(ams_data) ((ams_data)->r1)
#define nalu_hypre_AMSDataTempVertexVectorG(ams_data) ((ams_data)->g1)
#define nalu_hypre_AMSDataTempVecVertexVectorR(ams_data) ((ams_data)->r2)
#define nalu_hypre_AMSDataTempVecVertexVectorG(ams_data) ((ams_data)->g2)
#define nalu_hypre_AMSDataTempVecVertexVectorZZ(ams_data) ((ams_data)->zz)

/* Output log info */
#define nalu_hypre_AMSDataNumIterations(ams_data) ((ams_data)->num_iterations)
#define nalu_hypre_AMSDataResidualNorm(ams_data) ((ams_data)->rel_resid_norm)

#endif
