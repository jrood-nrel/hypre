/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#ifndef nalu_hypre_ADS_DATA_HEADER
#define nalu_hypre_ADS_DATA_HEADER

/*--------------------------------------------------------------------------
 * Auxiliary space Divergence Solver data
 *--------------------------------------------------------------------------*/
typedef struct
{
   /* Face element (RT0) stiffness matrix */
   nalu_hypre_ParCSRMatrix *A;

   /* Discrete curl matrix (edge-to-face) */
   nalu_hypre_ParCSRMatrix *C;
   /* Coarse grid matrix on the range of C^T */
   nalu_hypre_ParCSRMatrix *A_C;
   /* AMS solver for A_C */
   NALU_HYPRE_Solver B_C;

   /* Raviart-Thomas nodal interpolation matrix (vertex^3-to-face) */
   nalu_hypre_ParCSRMatrix *Pi;
   /* Coarse grid matrix on the range of Pi^T */
   nalu_hypre_ParCSRMatrix *A_Pi;
   /* AMG solver for A_Pi */
   NALU_HYPRE_Solver B_Pi;

   /* Components of the face interpolation matrix (vertex-to-face each) */
   nalu_hypre_ParCSRMatrix *Pix, *Piy, *Piz;
   /* Coarse grid matrices on the ranges of Pi{x,y,z}^T */
   nalu_hypre_ParCSRMatrix *A_Pix, *A_Piy, *A_Piz;
   /* AMG solvers for A_Pi{x,y,z} */
   NALU_HYPRE_Solver B_Pix, B_Piy, B_Piz;

   /* Does the solver own the RT/ND interpolations matrices? */
   NALU_HYPRE_Int owns_Pi;
   /* The (high-order) edge interpolation matrix and its components */
   nalu_hypre_ParCSRMatrix *ND_Pi, *ND_Pix, *ND_Piy, *ND_Piz;

   /* Discrete gradient matrix (vertex-to-edge) */
   nalu_hypre_ParCSRMatrix *G;
   /* Coordinates of the vertices */
   nalu_hypre_ParVector *x, *y, *z;

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

   /* AMS options for B_C */
   NALU_HYPRE_Int B_C_cycle_type;
   NALU_HYPRE_Int B_C_coarsen_type;
   NALU_HYPRE_Int B_C_agg_levels;
   NALU_HYPRE_Int B_C_relax_type;
   NALU_HYPRE_Real B_C_theta;
   NALU_HYPRE_Int B_C_interp_type;
   NALU_HYPRE_Int B_C_Pmax;

   /* AMG options for B_Pi */
   NALU_HYPRE_Int B_Pi_coarsen_type;
   NALU_HYPRE_Int B_Pi_agg_levels;
   NALU_HYPRE_Int B_Pi_relax_type;
   NALU_HYPRE_Real B_Pi_theta;
   NALU_HYPRE_Int B_Pi_interp_type;
   NALU_HYPRE_Int B_Pi_Pmax;

   /* Temporary vectors */
   nalu_hypre_ParVector *r0, *g0, *r1, *g1, *r2, *g2, *zz;

   /* Output log info */
   NALU_HYPRE_Int num_iterations;
   NALU_HYPRE_Real rel_resid_norm;

} nalu_hypre_ADSData;

/* Face stiffness matrix */
#define nalu_hypre_ADSDataA(ads_data) ((ads_data)->A)

/* Face space data */
#define nalu_hypre_ADSDataDiscreteCurl(ads_data) ((ads_data)->C)
#define nalu_hypre_ADSDataCurlCurlA(ads_data) ((ads_data)->A_C)
#define nalu_hypre_ADSDataCurlCurlAMS(ads_data) ((ads_data)->B_C)

/* Vector vertex space data */
#define nalu_hypre_ADSDataPiInterpolation(ads_data) ((ads_data)->Pi)
#define nalu_hypre_ADSDataOwnsPiInterpolation(ads_data) ((ads_data)->owns_Pi)
#define nalu_hypre_ADSDataPoissonA(ads_data) ((ads_data)->A_Pi)
#define nalu_hypre_ADSDataPoissonAMG(ads_data) ((ads_data)->B_Pi)

/* Discrete gradient and coordinates of the vertices */
#define nalu_hypre_ADSDataDiscreteGradient(ads_data) ((ads_data)->G)
#define nalu_hypre_ADSDataVertexCoordinateX(ads_data) ((ads_data)->x)
#define nalu_hypre_ADSDataVertexCoordinateY(ads_data) ((ads_data)->y)
#define nalu_hypre_ADSDataVertexCoordinateZ(ads_data) ((ads_data)->z)

/* Solver options */
#define nalu_hypre_ADSDataMaxIter(ads_data) ((ads_data)->maxit)
#define nalu_hypre_ADSDataTol(ads_data) ((ads_data)->tol)
#define nalu_hypre_ADSDataCycleType(ads_data) ((ads_data)->cycle_type)
#define nalu_hypre_ADSDataPrintLevel(ads_data) ((ads_data)->print_level)

/* Smoothing options */
#define nalu_hypre_ADSDataARelaxType(ads_data) ((ads_data)->A_relax_type)
#define nalu_hypre_ADSDataARelaxTimes(ads_data) ((ads_data)->A_relax_times)
#define nalu_hypre_ADSDataAL1Norms(ads_data) ((ads_data)->A_l1_norms)
#define nalu_hypre_ADSDataARelaxWeight(ads_data) ((ads_data)->A_relax_weight)
#define nalu_hypre_ADSDataAOmega(ads_data) ((ads_data)->A_omega)
#define nalu_hypre_ADSDataAMaxEigEst(ads_data) ((ads_data)->A_max_eig_est)
#define nalu_hypre_ADSDataAMinEigEst(ads_data) ((ads_data)->A_min_eig_est)
#define nalu_hypre_ADSDataAChebyOrder(ads_data) ((ads_data)->A_cheby_order)
#define nalu_hypre_ADSDataAChebyFraction(ads_data) ((ads_data)->A_cheby_fraction)

/* AMS options */
#define nalu_hypre_ADSDataAMSCycleType(ads_data) ((ads_data)->B_C_cycle_type)
#define nalu_hypre_ADSDataAMSCoarsenType(ads_data) ((ads_data)->B_C_coarsen_type)
#define nalu_hypre_ADSDataAMSAggLevels(ads_data) ((ads_data)->B_C_agg_levels)
#define nalu_hypre_ADSDataAMSRelaxType(ads_data) ((ads_data)->B_C_relax_type)
#define nalu_hypre_ADSDataAMSStrengthThreshold(ads_data) ((ads_data)->B_C_theta)
#define nalu_hypre_ADSDataAMSInterpType(ads_data) ((ads_data)->B_C_interp_type)
#define nalu_hypre_ADSDataAMSPmax(ads_data) ((ads_data)->B_C_Pmax)

/* AMG options */
#define nalu_hypre_ADSDataAMGCoarsenType(ads_data) ((ads_data)->B_Pi_coarsen_type)
#define nalu_hypre_ADSDataAMGAggLevels(ads_data) ((ads_data)->B_Pi_agg_levels)
#define nalu_hypre_ADSDataAMGRelaxType(ads_data) ((ads_data)->B_Pi_relax_type)
#define nalu_hypre_ADSDataAMGStrengthThreshold(ads_data) ((ads_data)->B_Pi_theta)
#define nalu_hypre_ADSDataAMGInterpType(ads_data) ((ads_data)->B_Pi_interp_type)
#define nalu_hypre_ADSDataAMGPmax(ads_data) ((ads_data)->B_Pi_Pmax)

/* Temporary vectors */
#define nalu_hypre_ADSDataTempFaceVectorR(ads_data) ((ads_data)->r0)
#define nalu_hypre_ADSDataTempFaceVectorG(ads_data) ((ads_data)->g0)
#define nalu_hypre_ADSDataTempEdgeVectorR(ads_data) ((ads_data)->r1)
#define nalu_hypre_ADSDataTempEdgeVectorG(ads_data) ((ads_data)->g1)
#define nalu_hypre_ADSDataTempVertexVectorR(ads_data) ((ads_data)->r2)
#define nalu_hypre_ADSDataTempVertexVectorG(ads_data) ((ads_data)->g2)

#endif
