/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/* ads.c */
void *hypre_ADSCreate ( void );
NALU_HYPRE_Int hypre_ADSDestroy ( void *solver );
NALU_HYPRE_Int hypre_ADSSetDiscreteCurl ( void *solver, hypre_ParCSRMatrix *C );
NALU_HYPRE_Int hypre_ADSSetDiscreteGradient ( void *solver, hypre_ParCSRMatrix *G );
NALU_HYPRE_Int hypre_ADSSetCoordinateVectors ( void *solver, hypre_ParVector *x, hypre_ParVector *y,
                                          hypre_ParVector *z );
NALU_HYPRE_Int hypre_ADSSetInterpolations ( void *solver, hypre_ParCSRMatrix *RT_Pi,
                                       hypre_ParCSRMatrix *RT_Pix, hypre_ParCSRMatrix *RT_Piy, hypre_ParCSRMatrix *RT_Piz,
                                       hypre_ParCSRMatrix *ND_Pi, hypre_ParCSRMatrix *ND_Pix, hypre_ParCSRMatrix *ND_Piy,
                                       hypre_ParCSRMatrix *ND_Piz );
NALU_HYPRE_Int hypre_ADSSetMaxIter ( void *solver, NALU_HYPRE_Int maxit );
NALU_HYPRE_Int hypre_ADSSetTol ( void *solver, NALU_HYPRE_Real tol );
NALU_HYPRE_Int hypre_ADSSetCycleType ( void *solver, NALU_HYPRE_Int cycle_type );
NALU_HYPRE_Int hypre_ADSSetPrintLevel ( void *solver, NALU_HYPRE_Int print_level );
NALU_HYPRE_Int hypre_ADSSetSmoothingOptions ( void *solver, NALU_HYPRE_Int A_relax_type,
                                         NALU_HYPRE_Int A_relax_times, NALU_HYPRE_Real A_relax_weight, NALU_HYPRE_Real A_omega );
NALU_HYPRE_Int hypre_ADSSetChebySmoothingOptions ( void *solver, NALU_HYPRE_Int A_cheby_order,
                                              NALU_HYPRE_Int A_cheby_fraction );
NALU_HYPRE_Int hypre_ADSSetAMSOptions ( void *solver, NALU_HYPRE_Int B_C_cycle_type,
                                   NALU_HYPRE_Int B_C_coarsen_type, NALU_HYPRE_Int B_C_agg_levels, NALU_HYPRE_Int B_C_relax_type,
                                   NALU_HYPRE_Real B_C_theta, NALU_HYPRE_Int B_C_interp_type, NALU_HYPRE_Int B_C_Pmax );
NALU_HYPRE_Int hypre_ADSSetAMGOptions ( void *solver, NALU_HYPRE_Int B_Pi_coarsen_type,
                                   NALU_HYPRE_Int B_Pi_agg_levels, NALU_HYPRE_Int B_Pi_relax_type, NALU_HYPRE_Real B_Pi_theta,
                                   NALU_HYPRE_Int B_Pi_interp_type, NALU_HYPRE_Int B_Pi_Pmax );
NALU_HYPRE_Int hypre_ADSComputePi ( hypre_ParCSRMatrix *A, hypre_ParCSRMatrix *C, hypre_ParCSRMatrix *G,
                               hypre_ParVector *x, hypre_ParVector *y, hypre_ParVector *z, hypre_ParCSRMatrix *PiNDx,
                               hypre_ParCSRMatrix *PiNDy, hypre_ParCSRMatrix *PiNDz, hypre_ParCSRMatrix **Pi_ptr );
NALU_HYPRE_Int hypre_ADSComputePixyz ( hypre_ParCSRMatrix *A, hypre_ParCSRMatrix *C,
                                  hypre_ParCSRMatrix *G, hypre_ParVector *x, hypre_ParVector *y, hypre_ParVector *z,
                                  hypre_ParCSRMatrix *PiNDx, hypre_ParCSRMatrix *PiNDy, hypre_ParCSRMatrix *PiNDz,
                                  hypre_ParCSRMatrix **Pix_ptr, hypre_ParCSRMatrix **Piy_ptr, hypre_ParCSRMatrix **Piz_ptr );
NALU_HYPRE_Int hypre_ADSSetup ( void *solver, hypre_ParCSRMatrix *A, hypre_ParVector *b,
                           hypre_ParVector *x );
NALU_HYPRE_Int hypre_ADSSolve ( void *solver, hypre_ParCSRMatrix *A, hypre_ParVector *b,
                           hypre_ParVector *x );
NALU_HYPRE_Int hypre_ADSGetNumIterations ( void *solver, NALU_HYPRE_Int *num_iterations );
NALU_HYPRE_Int hypre_ADSGetFinalRelativeResidualNorm ( void *solver, NALU_HYPRE_Real *rel_resid_norm );

/* ame.c */
void *hypre_AMECreate ( void );
NALU_HYPRE_Int hypre_AMEDestroy ( void *esolver );
NALU_HYPRE_Int hypre_AMESetAMSSolver ( void *esolver, void *ams_solver );
NALU_HYPRE_Int hypre_AMESetMassMatrix ( void *esolver, hypre_ParCSRMatrix *M );
NALU_HYPRE_Int hypre_AMESetBlockSize ( void *esolver, NALU_HYPRE_Int block_size );
NALU_HYPRE_Int hypre_AMESetMaxIter ( void *esolver, NALU_HYPRE_Int maxit );
NALU_HYPRE_Int hypre_AMESetTol ( void *esolver, NALU_HYPRE_Real tol );
NALU_HYPRE_Int hypre_AMESetRTol ( void *esolver, NALU_HYPRE_Real tol );
NALU_HYPRE_Int hypre_AMESetPrintLevel ( void *esolver, NALU_HYPRE_Int print_level );
NALU_HYPRE_Int hypre_AMESetup ( void *esolver );
NALU_HYPRE_Int hypre_AMEDiscrDivFreeComponent ( void *esolver, hypre_ParVector *b );
void hypre_AMEOperatorA ( void *data, void *x, void *y );
void hypre_AMEMultiOperatorA ( void *data, void *x, void *y );
void hypre_AMEOperatorM ( void *data, void *x, void *y );
void hypre_AMEMultiOperatorM ( void *data, void *x, void *y );
void hypre_AMEOperatorB ( void *data, void *x, void *y );
void hypre_AMEMultiOperatorB ( void *data, void *x, void *y );
NALU_HYPRE_Int hypre_AMESolve ( void *esolver );
NALU_HYPRE_Int hypre_AMEGetEigenvectors ( void *esolver, NALU_HYPRE_ParVector **eigenvectors_ptr );
NALU_HYPRE_Int hypre_AMEGetEigenvalues ( void *esolver, NALU_HYPRE_Real **eigenvalues_ptr );

/* amg_hybrid.c */
void *hypre_AMGHybridCreate ( void );
NALU_HYPRE_Int hypre_AMGHybridDestroy ( void *AMGhybrid_vdata );
NALU_HYPRE_Int hypre_AMGHybridSetTol ( void *AMGhybrid_vdata, NALU_HYPRE_Real tol );
NALU_HYPRE_Int hypre_AMGHybridSetAbsoluteTol ( void *AMGhybrid_vdata, NALU_HYPRE_Real a_tol );
NALU_HYPRE_Int hypre_AMGHybridSetConvergenceTol ( void *AMGhybrid_vdata, NALU_HYPRE_Real cf_tol );
NALU_HYPRE_Int hypre_AMGHybridSetNonGalerkinTol ( void *AMGhybrid_vdata, NALU_HYPRE_Int nongalerk_num_tol,
                                             NALU_HYPRE_Real *nongalerkin_tol );
NALU_HYPRE_Int hypre_AMGHybridSetDSCGMaxIter ( void *AMGhybrid_vdata, NALU_HYPRE_Int dscg_max_its );
NALU_HYPRE_Int hypre_AMGHybridSetPCGMaxIter ( void *AMGhybrid_vdata, NALU_HYPRE_Int pcg_max_its );
NALU_HYPRE_Int hypre_AMGHybridSetSetupType ( void *AMGhybrid_vdata, NALU_HYPRE_Int setup_type );
NALU_HYPRE_Int hypre_AMGHybridSetSolverType ( void *AMGhybrid_vdata, NALU_HYPRE_Int solver_type );
NALU_HYPRE_Int hypre_AMGHybridSetRecomputeResidual ( void *AMGhybrid_vdata,
                                                NALU_HYPRE_Int recompute_residual );
NALU_HYPRE_Int hypre_AMGHybridGetRecomputeResidual ( void *AMGhybrid_vdata,
                                                NALU_HYPRE_Int *recompute_residual );
NALU_HYPRE_Int hypre_AMGHybridSetRecomputeResidualP ( void *AMGhybrid_vdata,
                                                 NALU_HYPRE_Int recompute_residual_p );
NALU_HYPRE_Int hypre_AMGHybridGetRecomputeResidualP ( void *AMGhybrid_vdata,
                                                 NALU_HYPRE_Int *recompute_residual_p );
NALU_HYPRE_Int hypre_AMGHybridSetKDim ( void *AMGhybrid_vdata, NALU_HYPRE_Int k_dim );
NALU_HYPRE_Int hypre_AMGHybridSetStopCrit ( void *AMGhybrid_vdata, NALU_HYPRE_Int stop_crit );
NALU_HYPRE_Int hypre_AMGHybridSetTwoNorm ( void *AMGhybrid_vdata, NALU_HYPRE_Int two_norm );
NALU_HYPRE_Int hypre_AMGHybridSetRelChange ( void *AMGhybrid_vdata, NALU_HYPRE_Int rel_change );
NALU_HYPRE_Int hypre_AMGHybridSetPrecond ( void *pcg_vdata, NALU_HYPRE_Int (*pcg_precond_solve )(void*, void*,
                                                                                       void*, void*), NALU_HYPRE_Int (*pcg_precond_setup )(void*, void*, void*, void*), void *pcg_precond );
NALU_HYPRE_Int hypre_AMGHybridSetLogging ( void *AMGhybrid_vdata, NALU_HYPRE_Int logging );
NALU_HYPRE_Int hypre_AMGHybridSetPrintLevel ( void *AMGhybrid_vdata, NALU_HYPRE_Int print_level );
NALU_HYPRE_Int hypre_AMGHybridSetStrongThreshold ( void *AMGhybrid_vdata, NALU_HYPRE_Real strong_threshold );
NALU_HYPRE_Int hypre_AMGHybridSetMaxRowSum ( void *AMGhybrid_vdata, NALU_HYPRE_Real max_row_sum );
NALU_HYPRE_Int hypre_AMGHybridSetTruncFactor ( void *AMGhybrid_vdata, NALU_HYPRE_Real trunc_factor );
NALU_HYPRE_Int hypre_AMGHybridSetPMaxElmts ( void *AMGhybrid_vdata, NALU_HYPRE_Int P_max_elmts );
NALU_HYPRE_Int hypre_AMGHybridSetMaxLevels ( void *AMGhybrid_vdata, NALU_HYPRE_Int max_levels );
NALU_HYPRE_Int hypre_AMGHybridSetMeasureType ( void *AMGhybrid_vdata, NALU_HYPRE_Int measure_type );
NALU_HYPRE_Int hypre_AMGHybridSetCoarsenType ( void *AMGhybrid_vdata, NALU_HYPRE_Int coarsen_type );
NALU_HYPRE_Int hypre_AMGHybridSetInterpType ( void *AMGhybrid_vdata, NALU_HYPRE_Int interp_type );
NALU_HYPRE_Int hypre_AMGHybridSetCycleType ( void *AMGhybrid_vdata, NALU_HYPRE_Int cycle_type );
NALU_HYPRE_Int hypre_AMGHybridSetNumSweeps ( void *AMGhybrid_vdata, NALU_HYPRE_Int num_sweeps );
NALU_HYPRE_Int hypre_AMGHybridSetCycleNumSweeps ( void *AMGhybrid_vdata, NALU_HYPRE_Int num_sweeps,
                                             NALU_HYPRE_Int k );
NALU_HYPRE_Int hypre_AMGHybridSetRelaxType ( void *AMGhybrid_vdata, NALU_HYPRE_Int relax_type );
NALU_HYPRE_Int hypre_AMGHybridSetKeepTranspose ( void *AMGhybrid_vdata, NALU_HYPRE_Int keepT );
NALU_HYPRE_Int hypre_AMGHybridSetSplittingStrategy( void *AMGhybrid_vdata,
                                               NALU_HYPRE_Int splitting_strategy );
NALU_HYPRE_Int hypre_AMGHybridSetCycleRelaxType ( void *AMGhybrid_vdata, NALU_HYPRE_Int relax_type,
                                             NALU_HYPRE_Int k );
NALU_HYPRE_Int hypre_AMGHybridSetRelaxOrder ( void *AMGhybrid_vdata, NALU_HYPRE_Int relax_order );
NALU_HYPRE_Int hypre_AMGHybridSetMaxCoarseSize ( void *AMGhybrid_vdata, NALU_HYPRE_Int max_coarse_size );
NALU_HYPRE_Int hypre_AMGHybridSetMinCoarseSize ( void *AMGhybrid_vdata, NALU_HYPRE_Int min_coarse_size );
NALU_HYPRE_Int hypre_AMGHybridSetSeqThreshold ( void *AMGhybrid_vdata, NALU_HYPRE_Int seq_threshold );
NALU_HYPRE_Int hypre_AMGHybridSetNumGridSweeps ( void *AMGhybrid_vdata, NALU_HYPRE_Int *num_grid_sweeps );
NALU_HYPRE_Int hypre_AMGHybridSetGridRelaxType ( void *AMGhybrid_vdata, NALU_HYPRE_Int *grid_relax_type );
NALU_HYPRE_Int hypre_AMGHybridSetGridRelaxPoints ( void *AMGhybrid_vdata,
                                              NALU_HYPRE_Int **grid_relax_points );
NALU_HYPRE_Int hypre_AMGHybridSetRelaxWeight ( void *AMGhybrid_vdata, NALU_HYPRE_Real *relax_weight );
NALU_HYPRE_Int hypre_AMGHybridSetOmega ( void *AMGhybrid_vdata, NALU_HYPRE_Real *omega );
NALU_HYPRE_Int hypre_AMGHybridSetRelaxWt ( void *AMGhybrid_vdata, NALU_HYPRE_Real relax_wt );
NALU_HYPRE_Int hypre_AMGHybridSetLevelRelaxWt ( void *AMGhybrid_vdata, NALU_HYPRE_Real relax_wt,
                                           NALU_HYPRE_Int level );
NALU_HYPRE_Int hypre_AMGHybridSetOuterWt ( void *AMGhybrid_vdata, NALU_HYPRE_Real outer_wt );
NALU_HYPRE_Int hypre_AMGHybridSetLevelOuterWt ( void *AMGhybrid_vdata, NALU_HYPRE_Real outer_wt,
                                           NALU_HYPRE_Int level );
NALU_HYPRE_Int hypre_AMGHybridSetNumPaths ( void *AMGhybrid_vdata, NALU_HYPRE_Int num_paths );
NALU_HYPRE_Int hypre_AMGHybridSetDofFunc ( void *AMGhybrid_vdata, NALU_HYPRE_Int *dof_func );
NALU_HYPRE_Int hypre_AMGHybridSetAggNumLevels ( void *AMGhybrid_vdata, NALU_HYPRE_Int agg_num_levels );
NALU_HYPRE_Int hypre_AMGHybridSetAggInterpType ( void *AMGhybrid_vdata, NALU_HYPRE_Int agg_interp_type );
NALU_HYPRE_Int hypre_AMGHybridSetNumFunctions ( void *AMGhybrid_vdata, NALU_HYPRE_Int num_functions );
NALU_HYPRE_Int hypre_AMGHybridSetNodal ( void *AMGhybrid_vdata, NALU_HYPRE_Int nodal );
NALU_HYPRE_Int hypre_AMGHybridGetSetupSolveTime( void *AMGhybrid_vdata, NALU_HYPRE_Real *time );
NALU_HYPRE_Int hypre_AMGHybridGetNumIterations ( void *AMGhybrid_vdata, NALU_HYPRE_Int *num_its );
NALU_HYPRE_Int hypre_AMGHybridGetDSCGNumIterations ( void *AMGhybrid_vdata, NALU_HYPRE_Int *dscg_num_its );
NALU_HYPRE_Int hypre_AMGHybridGetPCGNumIterations ( void *AMGhybrid_vdata, NALU_HYPRE_Int *pcg_num_its );
NALU_HYPRE_Int hypre_AMGHybridGetFinalRelativeResidualNorm ( void *AMGhybrid_vdata,
                                                        NALU_HYPRE_Real *final_rel_res_norm );
NALU_HYPRE_Int hypre_AMGHybridSetup ( void *AMGhybrid_vdata, hypre_ParCSRMatrix *A, hypre_ParVector *b,
                                 hypre_ParVector *x );
NALU_HYPRE_Int hypre_AMGHybridSolve ( void *AMGhybrid_vdata, hypre_ParCSRMatrix *A, hypre_ParVector *b,
                                 hypre_ParVector *x );

/* ams.c */
NALU_HYPRE_Int hypre_ParCSRRelax ( hypre_ParCSRMatrix *A, hypre_ParVector *f, NALU_HYPRE_Int relax_type,
                              NALU_HYPRE_Int relax_times, NALU_HYPRE_Real *l1_norms, NALU_HYPRE_Real relax_weight, NALU_HYPRE_Real omega,
                              NALU_HYPRE_Real max_eig_est, NALU_HYPRE_Real min_eig_est, NALU_HYPRE_Int cheby_order, NALU_HYPRE_Real cheby_fraction,
                              hypre_ParVector *u, hypre_ParVector *v, hypre_ParVector *z );
hypre_ParVector *hypre_ParVectorInRangeOf ( hypre_ParCSRMatrix *A );
hypre_ParVector *hypre_ParVectorInDomainOf ( hypre_ParCSRMatrix *A );
NALU_HYPRE_Int hypre_ParVectorBlockSplit ( hypre_ParVector *x, hypre_ParVector *x_ [3 ], NALU_HYPRE_Int dim );
NALU_HYPRE_Int hypre_ParVectorBlockGather ( hypre_ParVector *x, hypre_ParVector *x_ [3 ],
                                       NALU_HYPRE_Int dim );
NALU_HYPRE_Int hypre_BoomerAMGBlockSolve ( void *B, hypre_ParCSRMatrix *A, hypre_ParVector *b,
                                      hypre_ParVector *x );
NALU_HYPRE_Int hypre_ParCSRMatrixFixZeroRows ( hypre_ParCSRMatrix *A );
NALU_HYPRE_Int hypre_ParCSRComputeL1Norms ( hypre_ParCSRMatrix *A, NALU_HYPRE_Int option,
                                       NALU_HYPRE_Int *cf_marker, NALU_HYPRE_Real **l1_norm_ptr );
NALU_HYPRE_Int hypre_ParCSRMatrixSetDiagRows ( hypre_ParCSRMatrix *A, NALU_HYPRE_Real d );
void *hypre_AMSCreate ( void );
NALU_HYPRE_Int hypre_AMSDestroy ( void *solver );
NALU_HYPRE_Int hypre_AMSSetDimension ( void *solver, NALU_HYPRE_Int dim );
NALU_HYPRE_Int hypre_AMSSetDiscreteGradient ( void *solver, hypre_ParCSRMatrix *G );
NALU_HYPRE_Int hypre_AMSSetCoordinateVectors ( void *solver, hypre_ParVector *x, hypre_ParVector *y,
                                          hypre_ParVector *z );
NALU_HYPRE_Int hypre_AMSSetEdgeConstantVectors ( void *solver, hypre_ParVector *Gx, hypre_ParVector *Gy,
                                            hypre_ParVector *Gz );
NALU_HYPRE_Int hypre_AMSSetInterpolations ( void *solver, hypre_ParCSRMatrix *Pi,
                                       hypre_ParCSRMatrix *Pix, hypre_ParCSRMatrix *Piy, hypre_ParCSRMatrix *Piz );
NALU_HYPRE_Int hypre_AMSSetAlphaPoissonMatrix ( void *solver, hypre_ParCSRMatrix *A_Pi );
NALU_HYPRE_Int hypre_AMSSetBetaPoissonMatrix ( void *solver, hypre_ParCSRMatrix *A_G );
NALU_HYPRE_Int hypre_AMSSetInteriorNodes ( void *solver, hypre_ParVector *interior_nodes );
NALU_HYPRE_Int hypre_AMSSetProjectionFrequency ( void *solver, NALU_HYPRE_Int projection_frequency );
NALU_HYPRE_Int hypre_AMSSetMaxIter ( void *solver, NALU_HYPRE_Int maxit );
NALU_HYPRE_Int hypre_AMSSetTol ( void *solver, NALU_HYPRE_Real tol );
NALU_HYPRE_Int hypre_AMSSetCycleType ( void *solver, NALU_HYPRE_Int cycle_type );
NALU_HYPRE_Int hypre_AMSSetPrintLevel ( void *solver, NALU_HYPRE_Int print_level );
NALU_HYPRE_Int hypre_AMSSetSmoothingOptions ( void *solver, NALU_HYPRE_Int A_relax_type,
                                         NALU_HYPRE_Int A_relax_times, NALU_HYPRE_Real A_relax_weight, NALU_HYPRE_Real A_omega );
NALU_HYPRE_Int hypre_AMSSetChebySmoothingOptions ( void *solver, NALU_HYPRE_Int A_cheby_order,
                                              NALU_HYPRE_Int A_cheby_fraction );
NALU_HYPRE_Int hypre_AMSSetAlphaAMGOptions ( void *solver, NALU_HYPRE_Int B_Pi_coarsen_type,
                                        NALU_HYPRE_Int B_Pi_agg_levels, NALU_HYPRE_Int B_Pi_relax_type, NALU_HYPRE_Real B_Pi_theta,
                                        NALU_HYPRE_Int B_Pi_interp_type, NALU_HYPRE_Int B_Pi_Pmax );
NALU_HYPRE_Int hypre_AMSSetAlphaAMGCoarseRelaxType ( void *solver, NALU_HYPRE_Int B_Pi_coarse_relax_type );
NALU_HYPRE_Int hypre_AMSSetBetaAMGOptions ( void *solver, NALU_HYPRE_Int B_G_coarsen_type,
                                       NALU_HYPRE_Int B_G_agg_levels, NALU_HYPRE_Int B_G_relax_type, NALU_HYPRE_Real B_G_theta, NALU_HYPRE_Int B_G_interp_type,
                                       NALU_HYPRE_Int B_G_Pmax );
NALU_HYPRE_Int hypre_AMSSetBetaAMGCoarseRelaxType ( void *solver, NALU_HYPRE_Int B_G_coarse_relax_type );
NALU_HYPRE_Int hypre_AMSComputePi ( hypre_ParCSRMatrix *A, hypre_ParCSRMatrix *G, hypre_ParVector *Gx,
                               hypre_ParVector *Gy, hypre_ParVector *Gz, NALU_HYPRE_Int dim, hypre_ParCSRMatrix **Pi_ptr );
NALU_HYPRE_Int hypre_AMSComputePixyz ( hypre_ParCSRMatrix *A, hypre_ParCSRMatrix *G, hypre_ParVector *Gx,
                                  hypre_ParVector *Gy, hypre_ParVector *Gz, NALU_HYPRE_Int dim, hypre_ParCSRMatrix **Pix_ptr,
                                  hypre_ParCSRMatrix **Piy_ptr, hypre_ParCSRMatrix **Piz_ptr );
NALU_HYPRE_Int hypre_AMSComputeGPi ( hypre_ParCSRMatrix *A, hypre_ParCSRMatrix *G, hypre_ParVector *Gx,
                                hypre_ParVector *Gy, hypre_ParVector *Gz, NALU_HYPRE_Int dim, hypre_ParCSRMatrix **GPi_ptr );
NALU_HYPRE_Int hypre_AMSSetup ( void *solver, hypre_ParCSRMatrix *A, hypre_ParVector *b,
                           hypre_ParVector *x );
NALU_HYPRE_Int hypre_AMSSolve ( void *solver, hypre_ParCSRMatrix *A, hypre_ParVector *b,
                           hypre_ParVector *x );
NALU_HYPRE_Int hypre_ParCSRSubspacePrec ( hypre_ParCSRMatrix *A0, NALU_HYPRE_Int A0_relax_type,
                                     NALU_HYPRE_Int A0_relax_times, NALU_HYPRE_Real *A0_l1_norms, NALU_HYPRE_Real A0_relax_weight, NALU_HYPRE_Real A0_omega,
                                     NALU_HYPRE_Real A0_max_eig_est, NALU_HYPRE_Real A0_min_eig_est, NALU_HYPRE_Int A0_cheby_order,
                                     NALU_HYPRE_Real A0_cheby_fraction, hypre_ParCSRMatrix **A, NALU_HYPRE_Solver *B, NALU_HYPRE_PtrToSolverFcn *HB,
                                     hypre_ParCSRMatrix **P, hypre_ParVector **r, hypre_ParVector **g, hypre_ParVector *x,
                                     hypre_ParVector *y, hypre_ParVector *r0, hypre_ParVector *g0, char *cycle, hypre_ParVector *z );
NALU_HYPRE_Int hypre_AMSGetNumIterations ( void *solver, NALU_HYPRE_Int *num_iterations );
NALU_HYPRE_Int hypre_AMSGetFinalRelativeResidualNorm ( void *solver, NALU_HYPRE_Real *rel_resid_norm );
NALU_HYPRE_Int hypre_AMSProjectOutGradients ( void *solver, hypre_ParVector *x );
NALU_HYPRE_Int hypre_AMSConstructDiscreteGradient ( hypre_ParCSRMatrix *A, hypre_ParVector *x_coord,
                                               NALU_HYPRE_BigInt *edge_vertex, NALU_HYPRE_Int edge_orientation, hypre_ParCSRMatrix **G_ptr );
NALU_HYPRE_Int hypre_AMSFEISetup ( void *solver, hypre_ParCSRMatrix *A, hypre_ParVector *b,
                              hypre_ParVector *x, NALU_HYPRE_Int num_vert, NALU_HYPRE_Int num_local_vert, NALU_HYPRE_BigInt *vert_number,
                              NALU_HYPRE_Real *vert_coord, NALU_HYPRE_Int num_edges, NALU_HYPRE_BigInt *edge_vertex );
NALU_HYPRE_Int hypre_AMSFEIDestroy ( void *solver );
NALU_HYPRE_Int hypre_ParCSRComputeL1NormsThreads ( hypre_ParCSRMatrix *A, NALU_HYPRE_Int option,
                                              NALU_HYPRE_Int num_threads, NALU_HYPRE_Int *cf_marker, NALU_HYPRE_Real **l1_norm_ptr );

/* aux_interp.c */
NALU_HYPRE_Int hypre_alt_insert_new_nodes ( hypre_ParCSRCommPkg *comm_pkg,
                                       hypre_ParCSRCommPkg *extend_comm_pkg, NALU_HYPRE_Int *IN_marker, NALU_HYPRE_Int full_off_procNodes,
                                       NALU_HYPRE_Int *OUT_marker );
NALU_HYPRE_Int hypre_big_insert_new_nodes ( hypre_ParCSRCommPkg *comm_pkg,
                                       hypre_ParCSRCommPkg *extend_comm_pkg, NALU_HYPRE_Int *IN_marker, NALU_HYPRE_Int full_off_procNodes,
                                       NALU_HYPRE_BigInt offset, NALU_HYPRE_BigInt *OUT_marker );
NALU_HYPRE_Int hypre_ssort ( NALU_HYPRE_BigInt *data, NALU_HYPRE_Int n );
NALU_HYPRE_Int hypre_index_of_minimum ( NALU_HYPRE_BigInt *data, NALU_HYPRE_Int n );
void hypre_swap_int ( NALU_HYPRE_BigInt *data, NALU_HYPRE_Int a, NALU_HYPRE_Int b );
void hypre_initialize_vecs ( NALU_HYPRE_Int diag_n, NALU_HYPRE_Int offd_n, NALU_HYPRE_Int *diag_ftc,
                             NALU_HYPRE_BigInt *offd_ftc, NALU_HYPRE_Int *diag_pm, NALU_HYPRE_Int *offd_pm, NALU_HYPRE_Int *tmp_CF );
/*NALU_HYPRE_Int hypre_new_offd_nodes(NALU_HYPRE_Int **found , NALU_HYPRE_Int num_cols_A_offd , NALU_HYPRE_Int *A_ext_i , NALU_HYPRE_Int *A_ext_j, NALU_HYPRE_Int num_cols_S_offd, NALU_HYPRE_Int *col_map_offd, NALU_HYPRE_Int col_1, NALU_HYPRE_Int col_n, NALU_HYPRE_Int *Sop_i, NALU_HYPRE_Int *Sop_j, NALU_HYPRE_Int *CF_marker_offd );*/
NALU_HYPRE_Int hypre_exchange_marker(hypre_ParCSRCommPkg *comm_pkg, NALU_HYPRE_Int *IN_marker,
                                NALU_HYPRE_Int *OUT_marker);
NALU_HYPRE_Int hypre_exchange_interp_data( NALU_HYPRE_Int **CF_marker_offd, NALU_HYPRE_Int **dof_func_offd,
                                      hypre_CSRMatrix **A_ext, NALU_HYPRE_Int *full_off_procNodes, hypre_CSRMatrix **Sop,
                                      hypre_ParCSRCommPkg **extend_comm_pkg, hypre_ParCSRMatrix *A, NALU_HYPRE_Int *CF_marker,
                                      hypre_ParCSRMatrix *S, NALU_HYPRE_Int num_functions, NALU_HYPRE_Int *dof_func,
                                      NALU_HYPRE_Int skip_fine_or_same_sign);
void hypre_build_interp_colmap(hypre_ParCSRMatrix *P, NALU_HYPRE_Int full_off_procNodes,
                               NALU_HYPRE_Int *tmp_CF_marker_offd, NALU_HYPRE_BigInt *fine_to_coarse_offd);

/* block_tridiag.c */
void *hypre_BlockTridiagCreate ( void );
NALU_HYPRE_Int hypre_BlockTridiagDestroy ( void *data );
NALU_HYPRE_Int hypre_BlockTridiagSetup ( void *data, hypre_ParCSRMatrix *A, hypre_ParVector *b,
                                    hypre_ParVector *x );
NALU_HYPRE_Int hypre_BlockTridiagSolve ( void *data, hypre_ParCSRMatrix *A, hypre_ParVector *b,
                                    hypre_ParVector *x );
NALU_HYPRE_Int hypre_BlockTridiagSetIndexSet ( void *data, NALU_HYPRE_Int n, NALU_HYPRE_Int *inds );
NALU_HYPRE_Int hypre_BlockTridiagSetAMGStrengthThreshold ( void *data, NALU_HYPRE_Real thresh );
NALU_HYPRE_Int hypre_BlockTridiagSetAMGNumSweeps ( void *data, NALU_HYPRE_Int nsweeps );
NALU_HYPRE_Int hypre_BlockTridiagSetAMGRelaxType ( void *data, NALU_HYPRE_Int relax_type );
NALU_HYPRE_Int hypre_BlockTridiagSetPrintLevel ( void *data, NALU_HYPRE_Int print_level );

/* driver.c */
NALU_HYPRE_Int BuildParFromFile ( NALU_HYPRE_Int argc, char *argv [], NALU_HYPRE_Int arg_index,
                             NALU_HYPRE_ParCSRMatrix *A_ptr );
NALU_HYPRE_Int BuildParLaplacian ( NALU_HYPRE_Int argc, char *argv [], NALU_HYPRE_Int arg_index,
                              NALU_HYPRE_ParCSRMatrix *A_ptr );
NALU_HYPRE_Int BuildParDifConv ( NALU_HYPRE_Int argc, char *argv [], NALU_HYPRE_Int arg_index,
                            NALU_HYPRE_ParCSRMatrix *A_ptr );
NALU_HYPRE_Int BuildParFromOneFile ( NALU_HYPRE_Int argc, char *argv [], NALU_HYPRE_Int arg_index,
                                NALU_HYPRE_ParCSRMatrix *A_ptr );
NALU_HYPRE_Int BuildRhsParFromOneFile ( NALU_HYPRE_Int argc, char *argv [], NALU_HYPRE_Int arg_index,
                                   NALU_HYPRE_ParCSRMatrix A, NALU_HYPRE_ParVector *b_ptr );
NALU_HYPRE_Int BuildParLaplacian9pt ( NALU_HYPRE_Int argc, char *argv [], NALU_HYPRE_Int arg_index,
                                 NALU_HYPRE_ParCSRMatrix *A_ptr );
NALU_HYPRE_Int BuildParLaplacian27pt ( NALU_HYPRE_Int argc, char *argv [], NALU_HYPRE_Int arg_index,
                                  NALU_HYPRE_ParCSRMatrix *A_ptr );

/* gen_redcs_mat.c */
NALU_HYPRE_Int hypre_seqAMGSetup ( hypre_ParAMGData *amg_data, NALU_HYPRE_Int p_level,
                              NALU_HYPRE_Int coarse_threshold );
NALU_HYPRE_Int hypre_seqAMGCycle ( hypre_ParAMGData *amg_data, NALU_HYPRE_Int p_level,
                              hypre_ParVector **Par_F_array, hypre_ParVector **Par_U_array );
NALU_HYPRE_Int hypre_GenerateSubComm ( MPI_Comm comm, NALU_HYPRE_Int participate, MPI_Comm *new_comm_ptr );
void hypre_merge_lists ( NALU_HYPRE_Int *list1, NALU_HYPRE_Int *list2, hypre_int *np1,
                         hypre_MPI_Datatype *dptr );

/* NALU_HYPRE_ads.c */
NALU_HYPRE_Int NALU_HYPRE_ADSCreate ( NALU_HYPRE_Solver *solver );
NALU_HYPRE_Int NALU_HYPRE_ADSDestroy ( NALU_HYPRE_Solver solver );
NALU_HYPRE_Int NALU_HYPRE_ADSSetup ( NALU_HYPRE_Solver solver, NALU_HYPRE_ParCSRMatrix A, NALU_HYPRE_ParVector b,
                           NALU_HYPRE_ParVector x );
NALU_HYPRE_Int NALU_HYPRE_ADSSolve ( NALU_HYPRE_Solver solver, NALU_HYPRE_ParCSRMatrix A, NALU_HYPRE_ParVector b,
                           NALU_HYPRE_ParVector x );
NALU_HYPRE_Int NALU_HYPRE_ADSSetDiscreteCurl ( NALU_HYPRE_Solver solver, NALU_HYPRE_ParCSRMatrix C );
NALU_HYPRE_Int NALU_HYPRE_ADSSetDiscreteGradient ( NALU_HYPRE_Solver solver, NALU_HYPRE_ParCSRMatrix G );
NALU_HYPRE_Int NALU_HYPRE_ADSSetCoordinateVectors ( NALU_HYPRE_Solver solver, NALU_HYPRE_ParVector x, NALU_HYPRE_ParVector y,
                                          NALU_HYPRE_ParVector z );
NALU_HYPRE_Int NALU_HYPRE_ADSSetInterpolations ( NALU_HYPRE_Solver solver, NALU_HYPRE_ParCSRMatrix RT_Pi,
                                       NALU_HYPRE_ParCSRMatrix RT_Pix, NALU_HYPRE_ParCSRMatrix RT_Piy, NALU_HYPRE_ParCSRMatrix RT_Piz,
                                       NALU_HYPRE_ParCSRMatrix ND_Pi, NALU_HYPRE_ParCSRMatrix ND_Pix, NALU_HYPRE_ParCSRMatrix ND_Piy,
                                       NALU_HYPRE_ParCSRMatrix ND_Piz );
NALU_HYPRE_Int NALU_HYPRE_ADSSetMaxIter ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int maxit );
NALU_HYPRE_Int NALU_HYPRE_ADSSetTol ( NALU_HYPRE_Solver solver, NALU_HYPRE_Real tol );
NALU_HYPRE_Int NALU_HYPRE_ADSSetCycleType ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int cycle_type );
NALU_HYPRE_Int NALU_HYPRE_ADSSetPrintLevel ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int print_level );
NALU_HYPRE_Int NALU_HYPRE_ADSSetSmoothingOptions ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int relax_type,
                                         NALU_HYPRE_Int relax_times, NALU_HYPRE_Real relax_weight, NALU_HYPRE_Real omega );
NALU_HYPRE_Int NALU_HYPRE_ADSSetChebySmoothingOptions ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int cheby_order,
                                              NALU_HYPRE_Int cheby_fraction );
NALU_HYPRE_Int NALU_HYPRE_ADSSetAMSOptions ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int cycle_type,
                                   NALU_HYPRE_Int coarsen_type, NALU_HYPRE_Int agg_levels, NALU_HYPRE_Int relax_type, NALU_HYPRE_Real strength_threshold,
                                   NALU_HYPRE_Int interp_type, NALU_HYPRE_Int Pmax );
NALU_HYPRE_Int NALU_HYPRE_ADSSetAMGOptions ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int coarsen_type,
                                   NALU_HYPRE_Int agg_levels, NALU_HYPRE_Int relax_type, NALU_HYPRE_Real strength_threshold, NALU_HYPRE_Int interp_type,
                                   NALU_HYPRE_Int Pmax );
NALU_HYPRE_Int NALU_HYPRE_ADSGetNumIterations ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int *num_iterations );
NALU_HYPRE_Int NALU_HYPRE_ADSGetFinalRelativeResidualNorm ( NALU_HYPRE_Solver solver, NALU_HYPRE_Real *rel_resid_norm );

/* NALU_HYPRE_ame.c */
NALU_HYPRE_Int NALU_HYPRE_AMECreate ( NALU_HYPRE_Solver *esolver );
NALU_HYPRE_Int NALU_HYPRE_AMEDestroy ( NALU_HYPRE_Solver esolver );
NALU_HYPRE_Int NALU_HYPRE_AMESetup ( NALU_HYPRE_Solver esolver );
NALU_HYPRE_Int NALU_HYPRE_AMESolve ( NALU_HYPRE_Solver esolver );
NALU_HYPRE_Int NALU_HYPRE_AMESetAMSSolver ( NALU_HYPRE_Solver esolver, NALU_HYPRE_Solver ams_solver );
NALU_HYPRE_Int NALU_HYPRE_AMESetMassMatrix ( NALU_HYPRE_Solver esolver, NALU_HYPRE_ParCSRMatrix M );
NALU_HYPRE_Int NALU_HYPRE_AMESetBlockSize ( NALU_HYPRE_Solver esolver, NALU_HYPRE_Int block_size );
NALU_HYPRE_Int NALU_HYPRE_AMESetMaxIter ( NALU_HYPRE_Solver esolver, NALU_HYPRE_Int maxit );
NALU_HYPRE_Int NALU_HYPRE_AMESetTol ( NALU_HYPRE_Solver esolver, NALU_HYPRE_Real tol );
NALU_HYPRE_Int NALU_HYPRE_AMESetRTol ( NALU_HYPRE_Solver esolver, NALU_HYPRE_Real tol );
NALU_HYPRE_Int NALU_HYPRE_AMESetPrintLevel ( NALU_HYPRE_Solver esolver, NALU_HYPRE_Int print_level );
NALU_HYPRE_Int NALU_HYPRE_AMEGetEigenvalues ( NALU_HYPRE_Solver esolver, NALU_HYPRE_Real **eigenvalues );
NALU_HYPRE_Int NALU_HYPRE_AMEGetEigenvectors ( NALU_HYPRE_Solver esolver, NALU_HYPRE_ParVector **eigenvectors );

/* NALU_HYPRE_ams.c */
NALU_HYPRE_Int NALU_HYPRE_AMSCreate ( NALU_HYPRE_Solver *solver );
NALU_HYPRE_Int NALU_HYPRE_AMSDestroy ( NALU_HYPRE_Solver solver );
NALU_HYPRE_Int NALU_HYPRE_AMSSetup ( NALU_HYPRE_Solver solver, NALU_HYPRE_ParCSRMatrix A, NALU_HYPRE_ParVector b,
                           NALU_HYPRE_ParVector x );
NALU_HYPRE_Int NALU_HYPRE_AMSSolve ( NALU_HYPRE_Solver solver, NALU_HYPRE_ParCSRMatrix A, NALU_HYPRE_ParVector b,
                           NALU_HYPRE_ParVector x );
NALU_HYPRE_Int NALU_HYPRE_AMSSetDimension ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int dim );
NALU_HYPRE_Int NALU_HYPRE_AMSSetDiscreteGradient ( NALU_HYPRE_Solver solver, NALU_HYPRE_ParCSRMatrix G );
NALU_HYPRE_Int NALU_HYPRE_AMSSetCoordinateVectors ( NALU_HYPRE_Solver solver, NALU_HYPRE_ParVector x, NALU_HYPRE_ParVector y,
                                          NALU_HYPRE_ParVector z );
NALU_HYPRE_Int NALU_HYPRE_AMSSetEdgeConstantVectors ( NALU_HYPRE_Solver solver, NALU_HYPRE_ParVector Gx,
                                            NALU_HYPRE_ParVector Gy, NALU_HYPRE_ParVector Gz );
NALU_HYPRE_Int NALU_HYPRE_AMSSetInterpolations ( NALU_HYPRE_Solver solver, NALU_HYPRE_ParCSRMatrix Pi,
                                       NALU_HYPRE_ParCSRMatrix Pix, NALU_HYPRE_ParCSRMatrix Piy, NALU_HYPRE_ParCSRMatrix Piz );
NALU_HYPRE_Int NALU_HYPRE_AMSSetAlphaPoissonMatrix ( NALU_HYPRE_Solver solver, NALU_HYPRE_ParCSRMatrix A_alpha );
NALU_HYPRE_Int NALU_HYPRE_AMSSetBetaPoissonMatrix ( NALU_HYPRE_Solver solver, NALU_HYPRE_ParCSRMatrix A_beta );
NALU_HYPRE_Int NALU_HYPRE_AMSSetInteriorNodes ( NALU_HYPRE_Solver solver, NALU_HYPRE_ParVector interior_nodes );
NALU_HYPRE_Int NALU_HYPRE_AMSSetProjectionFrequency ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int projection_frequency );
NALU_HYPRE_Int NALU_HYPRE_AMSSetMaxIter ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int maxit );
NALU_HYPRE_Int NALU_HYPRE_AMSSetTol ( NALU_HYPRE_Solver solver, NALU_HYPRE_Real tol );
NALU_HYPRE_Int NALU_HYPRE_AMSSetCycleType ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int cycle_type );
NALU_HYPRE_Int NALU_HYPRE_AMSSetPrintLevel ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int print_level );
NALU_HYPRE_Int NALU_HYPRE_AMSSetSmoothingOptions ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int relax_type,
                                         NALU_HYPRE_Int relax_times, NALU_HYPRE_Real relax_weight, NALU_HYPRE_Real omega );
NALU_HYPRE_Int NALU_HYPRE_AMSSetChebySmoothingOptions ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int cheby_order,
                                              NALU_HYPRE_Int cheby_fraction );
NALU_HYPRE_Int NALU_HYPRE_AMSSetAlphaAMGOptions ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int alpha_coarsen_type,
                                        NALU_HYPRE_Int alpha_agg_levels, NALU_HYPRE_Int alpha_relax_type, NALU_HYPRE_Real alpha_strength_threshold,
                                        NALU_HYPRE_Int alpha_interp_type, NALU_HYPRE_Int alpha_Pmax );
NALU_HYPRE_Int NALU_HYPRE_AMSSetAlphaAMGCoarseRelaxType ( NALU_HYPRE_Solver solver,
                                                NALU_HYPRE_Int alpha_coarse_relax_type );
NALU_HYPRE_Int NALU_HYPRE_AMSSetBetaAMGOptions ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int beta_coarsen_type,
                                       NALU_HYPRE_Int beta_agg_levels, NALU_HYPRE_Int beta_relax_type, NALU_HYPRE_Real beta_strength_threshold,
                                       NALU_HYPRE_Int beta_interp_type, NALU_HYPRE_Int beta_Pmax );
NALU_HYPRE_Int NALU_HYPRE_AMSSetBetaAMGCoarseRelaxType ( NALU_HYPRE_Solver solver,
                                               NALU_HYPRE_Int beta_coarse_relax_type );
NALU_HYPRE_Int NALU_HYPRE_AMSGetNumIterations ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int *num_iterations );
NALU_HYPRE_Int NALU_HYPRE_AMSGetFinalRelativeResidualNorm ( NALU_HYPRE_Solver solver, NALU_HYPRE_Real *rel_resid_norm );
NALU_HYPRE_Int NALU_HYPRE_AMSProjectOutGradients ( NALU_HYPRE_Solver solver, NALU_HYPRE_ParVector x );
NALU_HYPRE_Int NALU_HYPRE_AMSConstructDiscreteGradient ( NALU_HYPRE_ParCSRMatrix A, NALU_HYPRE_ParVector x_coord,
                                               NALU_HYPRE_BigInt *edge_vertex, NALU_HYPRE_Int edge_orientation, NALU_HYPRE_ParCSRMatrix *G );
NALU_HYPRE_Int NALU_HYPRE_AMSFEISetup ( NALU_HYPRE_Solver solver, NALU_HYPRE_ParCSRMatrix A, NALU_HYPRE_ParVector b,
                              NALU_HYPRE_ParVector x, NALU_HYPRE_BigInt *EdgeNodeList_, NALU_HYPRE_BigInt *NodeNumbers_, NALU_HYPRE_Int numEdges_,
                              NALU_HYPRE_Int numLocalNodes_, NALU_HYPRE_Int numNodes_, NALU_HYPRE_Real *NodalCoord_ );
NALU_HYPRE_Int NALU_HYPRE_AMSFEIDestroy ( NALU_HYPRE_Solver solver );

/* NALU_HYPRE_parcsr_amg.c */
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGCreate ( NALU_HYPRE_Solver *solver );
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGDestroy ( NALU_HYPRE_Solver solver );
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetup ( NALU_HYPRE_Solver solver, NALU_HYPRE_ParCSRMatrix A, NALU_HYPRE_ParVector b,
                                 NALU_HYPRE_ParVector x );
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSolve ( NALU_HYPRE_Solver solver, NALU_HYPRE_ParCSRMatrix A, NALU_HYPRE_ParVector b,
                                 NALU_HYPRE_ParVector x );
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSolveT ( NALU_HYPRE_Solver solver, NALU_HYPRE_ParCSRMatrix A, NALU_HYPRE_ParVector b,
                                  NALU_HYPRE_ParVector x );
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetRestriction ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int restr_par );
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetIsTriangular ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int is_triangular );
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetGMRESSwitchR ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int gmres_switch );
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetMaxLevels ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int max_levels );
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGGetMaxLevels ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int *max_levels );
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetMaxCoarseSize ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int max_coarse_size );
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGGetMaxCoarseSize ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int *max_coarse_size );
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetMinCoarseSize ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int min_coarse_size );
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGGetMinCoarseSize ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int *min_coarse_size );
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetSeqThreshold ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int seq_threshold );
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGGetSeqThreshold ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int *seq_threshold );
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetRedundant ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int redundant );
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGGetRedundant ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int *redundant );
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetCoarsenCutFactor( NALU_HYPRE_Solver solver, NALU_HYPRE_Int coarsen_cut_factor );
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGGetCoarsenCutFactor( NALU_HYPRE_Solver solver, NALU_HYPRE_Int *coarsen_cut_factor );
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetStrongThreshold ( NALU_HYPRE_Solver solver, NALU_HYPRE_Real strong_threshold );
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGGetStrongThreshold ( NALU_HYPRE_Solver solver, NALU_HYPRE_Real *strong_threshold );
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetStrongThresholdR ( NALU_HYPRE_Solver solver, NALU_HYPRE_Real strong_threshold );
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGGetStrongThresholdR ( NALU_HYPRE_Solver solver, NALU_HYPRE_Real *strong_threshold );
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetFilterThresholdR ( NALU_HYPRE_Solver solver, NALU_HYPRE_Real filter_threshold );
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGGetFilterThresholdR ( NALU_HYPRE_Solver solver, NALU_HYPRE_Real *filter_threshold );
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetGMRESSwitchR ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int gmres_switch );
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetSabs ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int Sabs );
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetMaxRowSum ( NALU_HYPRE_Solver solver, NALU_HYPRE_Real max_row_sum );
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGGetMaxRowSum ( NALU_HYPRE_Solver solver, NALU_HYPRE_Real *max_row_sum );
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetTruncFactor ( NALU_HYPRE_Solver solver, NALU_HYPRE_Real trunc_factor );
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGGetTruncFactor ( NALU_HYPRE_Solver solver, NALU_HYPRE_Real *trunc_factor );
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetPMaxElmts ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int P_max_elmts );
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGGetPMaxElmts ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int *P_max_elmts );
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetJacobiTruncThreshold ( NALU_HYPRE_Solver solver,
                                                   NALU_HYPRE_Real jacobi_trunc_threshold );
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGGetJacobiTruncThreshold ( NALU_HYPRE_Solver solver,
                                                   NALU_HYPRE_Real *jacobi_trunc_threshold );
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetPostInterpType ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int post_interp_type );
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGGetPostInterpType ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int *post_interp_type );
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetSCommPkgSwitch ( NALU_HYPRE_Solver solver, NALU_HYPRE_Real S_commpkg_switch );
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetInterpType ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int interp_type );
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetSepWeight ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int sep_weight );
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetMinIter ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int min_iter );
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetMaxIter ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int max_iter );
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGGetMaxIter ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int *max_iter );
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetCoarsenType ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int coarsen_type );
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGGetCoarsenType ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int *coarsen_type );
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetMeasureType ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int measure_type );
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGGetMeasureType ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int *measure_type );
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetSetupType ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int setup_type );
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetOldDefault ( NALU_HYPRE_Solver solver );
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetFCycle ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int fcycle );
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGGetFCycle ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int *fcycle );
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetCycleType ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int cycle_type );
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGGetCycleType ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int *cycle_type );
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetConvergeType ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int type );
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGGetConvergeType ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int *type );
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetTol ( NALU_HYPRE_Solver solver, NALU_HYPRE_Real tol );
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGGetTol ( NALU_HYPRE_Solver solver, NALU_HYPRE_Real *tol );
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetNumGridSweeps ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int *num_grid_sweeps );
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetNumSweeps ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int num_sweeps );
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetCycleNumSweeps ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int num_sweeps,
                                             NALU_HYPRE_Int k );
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGGetCycleNumSweeps ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int *num_sweeps,
                                             NALU_HYPRE_Int k );
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGInitGridRelaxation ( NALU_HYPRE_Int **num_grid_sweeps_ptr,
                                              NALU_HYPRE_Int **grid_relax_type_ptr, NALU_HYPRE_Int ***grid_relax_points_ptr, NALU_HYPRE_Int coarsen_type,
                                              NALU_HYPRE_Real **relax_weights_ptr, NALU_HYPRE_Int max_levels );
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetGridRelaxType ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int *grid_relax_type );
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetRelaxType ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int relax_type );
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetCycleRelaxType ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int relax_type,
                                             NALU_HYPRE_Int k );
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGGetCycleRelaxType ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int *relax_type,
                                             NALU_HYPRE_Int k );
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetRelaxOrder ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int relax_order );
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetGridRelaxPoints ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int **grid_relax_points );
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetRelaxWeight ( NALU_HYPRE_Solver solver, NALU_HYPRE_Real *relax_weight );
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetRelaxWt ( NALU_HYPRE_Solver solver, NALU_HYPRE_Real relax_wt );
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetLevelRelaxWt ( NALU_HYPRE_Solver solver, NALU_HYPRE_Real relax_wt,
                                           NALU_HYPRE_Int level );
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetOmega ( NALU_HYPRE_Solver solver, NALU_HYPRE_Real *omega );
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetOuterWt ( NALU_HYPRE_Solver solver, NALU_HYPRE_Real outer_wt );
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetLevelOuterWt ( NALU_HYPRE_Solver solver, NALU_HYPRE_Real outer_wt,
                                           NALU_HYPRE_Int level );
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetSmoothType ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int smooth_type );
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGGetSmoothType ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int *smooth_type );
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetSmoothNumLevels ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int smooth_num_levels );
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGGetSmoothNumLevels ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int *smooth_num_levels );
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetSmoothNumSweeps ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int smooth_num_sweeps );
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGGetSmoothNumSweeps ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int *smooth_num_sweeps );
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetLogging ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int logging );
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGGetLogging ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int *logging );
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetPrintLevel ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int print_level );
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGGetPrintLevel ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int *print_level );
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetPrintFileName ( NALU_HYPRE_Solver solver, const char *print_file_name );
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetDebugFlag ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int debug_flag );
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGGetDebugFlag ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int *debug_flag );
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGGetNumIterations ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int *num_iterations );
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGGetCumNumIterations ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int *cum_num_iterations );
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGGetResidual ( NALU_HYPRE_Solver solver, NALU_HYPRE_ParVector *residual );
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGGetFinalRelativeResidualNorm ( NALU_HYPRE_Solver solver,
                                                        NALU_HYPRE_Real *rel_resid_norm );
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetVariant ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int variant );
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGGetVariant ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int *variant );
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetOverlap ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int overlap );
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGGetOverlap ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int *overlap );
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetDomainType ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int domain_type );
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGGetDomainType ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int *domain_type );
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetSchwarzRlxWeight ( NALU_HYPRE_Solver solver, NALU_HYPRE_Real schwarz_rlx_weight );
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGGetSchwarzRlxWeight ( NALU_HYPRE_Solver solver,
                                               NALU_HYPRE_Real *schwarz_rlx_weight );
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetSchwarzUseNonSymm ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int use_nonsymm );
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetSym ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int sym );
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetLevel ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int level );
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetThreshold ( NALU_HYPRE_Solver solver, NALU_HYPRE_Real threshold );
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetFilter ( NALU_HYPRE_Solver solver, NALU_HYPRE_Real filter );
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetDropTol ( NALU_HYPRE_Solver solver, NALU_HYPRE_Real drop_tol );
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetMaxNzPerRow ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int max_nz_per_row );
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetEuclidFile ( NALU_HYPRE_Solver solver, char *euclidfile );
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetEuLevel ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int eu_level );
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetEuSparseA ( NALU_HYPRE_Solver solver, NALU_HYPRE_Real eu_sparse_A );
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetEuBJ ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int eu_bj );
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetILUType( NALU_HYPRE_Solver solver, NALU_HYPRE_Int ilu_type);
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetILULevel( NALU_HYPRE_Solver solver, NALU_HYPRE_Int ilu_lfil);
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetILUMaxRowNnz( NALU_HYPRE_Solver  solver, NALU_HYPRE_Int ilu_max_row_nnz);
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetILUMaxIter( NALU_HYPRE_Solver solver, NALU_HYPRE_Int ilu_max_iter);
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetILUDroptol( NALU_HYPRE_Solver solver, NALU_HYPRE_Real ilu_droptol);
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetILUTriSolve( NALU_HYPRE_Solver solver, NALU_HYPRE_Int ilu_tri_solve);
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetILULowerJacobiIters( NALU_HYPRE_Solver solver,
                                                 NALU_HYPRE_Int ilu_lower_jacobi_iters);
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetILUUpperJacobiIters( NALU_HYPRE_Solver solver,
                                                 NALU_HYPRE_Int ilu_upper_jacobi_iters);
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetILULocalReordering( NALU_HYPRE_Solver solver, NALU_HYPRE_Int ilu_reordering_type);
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetFSAIMaxSteps ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int max_steps );
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetFSAIMaxStepSize ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int max_step_size );
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetFSAIEigMaxIters ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int eig_max_iters );
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetFSAIKapTolerance ( NALU_HYPRE_Solver solver, NALU_HYPRE_Real kap_tolerance );
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetNumFunctions ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int num_functions );
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGGetNumFunctions ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int *num_functions );
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetNodal ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int nodal );
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetNodalLevels ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int nodal_levels );
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetNodalDiag ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int nodal );
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetKeepSameSign ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int keep_same_sign );
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetDofFunc ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int *dof_func );
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetNumPaths ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int num_paths );
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetAggNumLevels ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int agg_num_levels );
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetAggInterpType ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int agg_interp_type );
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetAggTruncFactor ( NALU_HYPRE_Solver solver, NALU_HYPRE_Real agg_trunc_factor );
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetAddTruncFactor ( NALU_HYPRE_Solver solver, NALU_HYPRE_Real add_trunc_factor );
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetMultAddTruncFactor ( NALU_HYPRE_Solver solver, NALU_HYPRE_Real add_trunc_factor );
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetAggP12TruncFactor ( NALU_HYPRE_Solver solver,
                                                NALU_HYPRE_Real agg_P12_trunc_factor );
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetAggPMaxElmts ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int agg_P_max_elmts );
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetAddPMaxElmts ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int add_P_max_elmts );
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetMultAddPMaxElmts ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int add_P_max_elmts );
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetAddRelaxType ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int add_rlx_type );
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetAddRelaxWt ( NALU_HYPRE_Solver solver, NALU_HYPRE_Real add_rlx_wt );
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetAggP12MaxElmts ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int agg_P12_max_elmts );
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetNumCRRelaxSteps ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int num_CR_relax_steps );
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetCRRate ( NALU_HYPRE_Solver solver, NALU_HYPRE_Real CR_rate );
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetCRStrongTh ( NALU_HYPRE_Solver solver, NALU_HYPRE_Real CR_strong_th );
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetADropTol( NALU_HYPRE_Solver solver, NALU_HYPRE_Real A_drop_tol  );
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetADropType( NALU_HYPRE_Solver solver, NALU_HYPRE_Int A_drop_type  );
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetISType ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int IS_type );
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetCRUseCG ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int CR_use_CG );
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetGSMG ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int gsmg );
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetNumSamples ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int gsmg );
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetCGCIts ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int its );
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetPlotGrids ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int plotgrids );
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetPlotFileName ( NALU_HYPRE_Solver solver, const char *plotfilename );
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetCoordDim ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int coorddim );
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetCoordinates ( NALU_HYPRE_Solver solver, float *coordinates );
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGGetGridHierarchy(NALU_HYPRE_Solver solver, NALU_HYPRE_Int *cgrid );
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetChebyOrder ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int order );
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetChebyFraction ( NALU_HYPRE_Solver solver, NALU_HYPRE_Real ratio );
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetChebyEigEst ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int eig_est );
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetChebyVariant ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int variant );
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetChebyScale ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int scale );
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetInterpVectors ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int num_vectors,
                                            NALU_HYPRE_ParVector *vectors );
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetInterpVecVariant ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int num );
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetInterpVecQMax ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int q_max );
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetInterpVecAbsQTrunc ( NALU_HYPRE_Solver solver, NALU_HYPRE_Real q_trunc );
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetSmoothInterpVectors ( NALU_HYPRE_Solver solver,
                                                  NALU_HYPRE_Int smooth_interp_vectors );
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetInterpRefine ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int num_refine );
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetInterpVecFirstLevel ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int level );
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetAdditive ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int additive );
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGGetAdditive ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int *additive );
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetMultAdditive ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int mult_additive );
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGGetMultAdditive ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int *mult_additive );
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetSimple ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int simple );
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGGetSimple ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int *simple );
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetAddLastLvl ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int add_last_lvl );
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetNonGalerkinTol ( NALU_HYPRE_Solver solver, NALU_HYPRE_Real nongalerkin_tol );
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetLevelNonGalerkinTol ( NALU_HYPRE_Solver solver, NALU_HYPRE_Real nongalerkin_tol,
                                                  NALU_HYPRE_Int level );
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetNonGalerkTol ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int nongalerk_num_tol,
                                           NALU_HYPRE_Real *nongalerk_tol );
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetRAP2 ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int rap2 );
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetModuleRAP2 ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int mod_rap2 );
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetKeepTranspose ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int keepTranspose );
#ifdef NALU_HYPRE_USING_DSUPERLU
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetDSLUThreshold ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int slu_threshold );
#endif
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetCpointsToKeep( NALU_HYPRE_Solver solver, NALU_HYPRE_Int cpt_coarse_level,
                                           NALU_HYPRE_Int num_cpt_coarse, NALU_HYPRE_BigInt *cpt_coarse_index);
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetCPoints( NALU_HYPRE_Solver solver, NALU_HYPRE_Int cpt_coarse_level,
                                     NALU_HYPRE_Int num_cpt_coarse, NALU_HYPRE_BigInt *cpt_coarse_index);
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetIsolatedFPoints( NALU_HYPRE_Solver solver, NALU_HYPRE_Int num_isolated_fpt,
                                             NALU_HYPRE_BigInt *isolated_fpt_index );
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetFPoints( NALU_HYPRE_Solver solver, NALU_HYPRE_Int num_fpt,
                                     NALU_HYPRE_BigInt *fpt_index );
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetCumNnzAP ( NALU_HYPRE_Solver solver , NALU_HYPRE_Real cum_nnz_AP );
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGGetCumNnzAP ( NALU_HYPRE_Solver solver , NALU_HYPRE_Real *cum_nnz_AP );

/* NALU_HYPRE_parcsr_amgdd.c */
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGDDSetup ( NALU_HYPRE_Solver solver, NALU_HYPRE_ParCSRMatrix A, NALU_HYPRE_ParVector b,
                                   NALU_HYPRE_ParVector x );
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGDDSolve ( NALU_HYPRE_Solver solver, NALU_HYPRE_ParCSRMatrix A, NALU_HYPRE_ParVector b,
                                   NALU_HYPRE_ParVector x );
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGDDSetStartLevel ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int start_level );
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGDDGetStartLevel ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int *start_level );
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGDDSetFACNumCycles ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int fac_num_cycles );
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGDDGetFACNumCycles ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int *fac_num_cycles );
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGDDSetFACCycleType ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int fac_cycle_type );
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGDDGetFACCycleType ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int *fac_cycle_type );
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGDDSetFACNumRelax ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int fac_num_relax );
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGDDGetFACNumRelax ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int *fac_num_relax );
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGDDSetFACRelaxType ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int fac_relax_type );
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGDDGetFACRelaxType ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int *fac_relax_type );
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGDDSetFACRelaxWeight ( NALU_HYPRE_Solver solver, NALU_HYPRE_Real fac_relax_weight );
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGDDGetFACRelaxWeight ( NALU_HYPRE_Solver solver, NALU_HYPRE_Real *fac_relax_weight );
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGDDSetPadding ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int padding );
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGDDGetPadding ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int *padding );
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGDDSetNumGhostLayers ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int num_ghost_layers );
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGDDGetNumGhostLayers ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int *num_ghost_layers );
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGDDSetUserFACRelaxation( NALU_HYPRE_Solver solver,
                                                 NALU_HYPRE_Int (*userFACRelaxation)( void *amgdd_vdata, NALU_HYPRE_Int level, NALU_HYPRE_Int cycle_param ) );
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGDDGetAMG ( NALU_HYPRE_Solver solver, NALU_HYPRE_Solver *amg_solver );

/* NALU_HYPRE_parcsr_bicgstab.c */
NALU_HYPRE_Int NALU_HYPRE_ParCSRBiCGSTABCreate ( MPI_Comm comm, NALU_HYPRE_Solver *solver );
NALU_HYPRE_Int NALU_HYPRE_ParCSRBiCGSTABDestroy ( NALU_HYPRE_Solver solver );
NALU_HYPRE_Int NALU_HYPRE_ParCSRBiCGSTABSetup ( NALU_HYPRE_Solver solver, NALU_HYPRE_ParCSRMatrix A, NALU_HYPRE_ParVector b,
                                      NALU_HYPRE_ParVector x );
NALU_HYPRE_Int NALU_HYPRE_ParCSRBiCGSTABSolve ( NALU_HYPRE_Solver solver, NALU_HYPRE_ParCSRMatrix A, NALU_HYPRE_ParVector b,
                                      NALU_HYPRE_ParVector x );
NALU_HYPRE_Int NALU_HYPRE_ParCSRBiCGSTABSetTol ( NALU_HYPRE_Solver solver, NALU_HYPRE_Real tol );
NALU_HYPRE_Int NALU_HYPRE_ParCSRBiCGSTABSetAbsoluteTol ( NALU_HYPRE_Solver solver, NALU_HYPRE_Real a_tol );
NALU_HYPRE_Int NALU_HYPRE_ParCSRBiCGSTABSetMinIter ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int min_iter );
NALU_HYPRE_Int NALU_HYPRE_ParCSRBiCGSTABSetMaxIter ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int max_iter );
NALU_HYPRE_Int NALU_HYPRE_ParCSRBiCGSTABSetStopCrit ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int stop_crit );
NALU_HYPRE_Int NALU_HYPRE_ParCSRBiCGSTABSetPrecond ( NALU_HYPRE_Solver solver, NALU_HYPRE_PtrToParSolverFcn precond,
                                           NALU_HYPRE_PtrToParSolverFcn precond_setup, NALU_HYPRE_Solver precond_solver );
NALU_HYPRE_Int NALU_HYPRE_ParCSRBiCGSTABGetPrecond ( NALU_HYPRE_Solver solver, NALU_HYPRE_Solver *precond_data_ptr );
NALU_HYPRE_Int NALU_HYPRE_ParCSRBiCGSTABSetLogging ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int logging );
NALU_HYPRE_Int NALU_HYPRE_ParCSRBiCGSTABSetPrintLevel ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int print_level );
NALU_HYPRE_Int NALU_HYPRE_ParCSRBiCGSTABGetNumIterations ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int *num_iterations );
NALU_HYPRE_Int NALU_HYPRE_ParCSRBiCGSTABGetFinalRelativeResidualNorm ( NALU_HYPRE_Solver solver,
                                                             NALU_HYPRE_Real *norm );
NALU_HYPRE_Int NALU_HYPRE_ParCSRBiCGSTABGetResidual ( NALU_HYPRE_Solver solver, NALU_HYPRE_ParVector *residual );

/* NALU_HYPRE_parcsr_block.c */
NALU_HYPRE_Int NALU_HYPRE_BlockTridiagCreate ( NALU_HYPRE_Solver *solver );
NALU_HYPRE_Int NALU_HYPRE_BlockTridiagDestroy ( NALU_HYPRE_Solver solver );
NALU_HYPRE_Int NALU_HYPRE_BlockTridiagSetup ( NALU_HYPRE_Solver solver, NALU_HYPRE_ParCSRMatrix A, NALU_HYPRE_ParVector b,
                                    NALU_HYPRE_ParVector x );
NALU_HYPRE_Int NALU_HYPRE_BlockTridiagSolve ( NALU_HYPRE_Solver solver, NALU_HYPRE_ParCSRMatrix A, NALU_HYPRE_ParVector b,
                                    NALU_HYPRE_ParVector x );
NALU_HYPRE_Int NALU_HYPRE_BlockTridiagSetIndexSet ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int n, NALU_HYPRE_Int *inds );
NALU_HYPRE_Int NALU_HYPRE_BlockTridiagSetAMGStrengthThreshold ( NALU_HYPRE_Solver solver, NALU_HYPRE_Real thresh );
NALU_HYPRE_Int NALU_HYPRE_BlockTridiagSetAMGNumSweeps ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int num_sweeps );
NALU_HYPRE_Int NALU_HYPRE_BlockTridiagSetAMGRelaxType ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int relax_type );
NALU_HYPRE_Int NALU_HYPRE_BlockTridiagSetPrintLevel ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int print_level );

/* NALU_HYPRE_parcsr_cgnr.c */
NALU_HYPRE_Int NALU_HYPRE_ParCSRCGNRCreate ( MPI_Comm comm, NALU_HYPRE_Solver *solver );
NALU_HYPRE_Int NALU_HYPRE_ParCSRCGNRDestroy ( NALU_HYPRE_Solver solver );
NALU_HYPRE_Int NALU_HYPRE_ParCSRCGNRSetup ( NALU_HYPRE_Solver solver, NALU_HYPRE_ParCSRMatrix A, NALU_HYPRE_ParVector b,
                                  NALU_HYPRE_ParVector x );
NALU_HYPRE_Int NALU_HYPRE_ParCSRCGNRSolve ( NALU_HYPRE_Solver solver, NALU_HYPRE_ParCSRMatrix A, NALU_HYPRE_ParVector b,
                                  NALU_HYPRE_ParVector x );
NALU_HYPRE_Int NALU_HYPRE_ParCSRCGNRSetTol ( NALU_HYPRE_Solver solver, NALU_HYPRE_Real tol );
NALU_HYPRE_Int NALU_HYPRE_ParCSRCGNRSetMinIter ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int min_iter );
NALU_HYPRE_Int NALU_HYPRE_ParCSRCGNRSetMaxIter ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int max_iter );
NALU_HYPRE_Int NALU_HYPRE_ParCSRCGNRSetStopCrit ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int stop_crit );
NALU_HYPRE_Int NALU_HYPRE_ParCSRCGNRSetPrecond ( NALU_HYPRE_Solver solver, NALU_HYPRE_PtrToParSolverFcn precond,
                                       NALU_HYPRE_PtrToParSolverFcn precondT, NALU_HYPRE_PtrToParSolverFcn precond_setup,
                                       NALU_HYPRE_Solver precond_solver );
NALU_HYPRE_Int NALU_HYPRE_ParCSRCGNRGetPrecond ( NALU_HYPRE_Solver solver, NALU_HYPRE_Solver *precond_data_ptr );
NALU_HYPRE_Int NALU_HYPRE_ParCSRCGNRSetLogging ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int logging );
NALU_HYPRE_Int NALU_HYPRE_ParCSRCGNRGetNumIterations ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int *num_iterations );
NALU_HYPRE_Int NALU_HYPRE_ParCSRCGNRGetFinalRelativeResidualNorm ( NALU_HYPRE_Solver solver, NALU_HYPRE_Real *norm );

/* NALU_HYPRE_parcsr_Euclid.c */
NALU_HYPRE_Int NALU_HYPRE_EuclidCreate ( MPI_Comm comm, NALU_HYPRE_Solver *solver );
NALU_HYPRE_Int NALU_HYPRE_EuclidDestroy ( NALU_HYPRE_Solver solver );
NALU_HYPRE_Int NALU_HYPRE_EuclidSetup ( NALU_HYPRE_Solver solver, NALU_HYPRE_ParCSRMatrix A, NALU_HYPRE_ParVector b,
                              NALU_HYPRE_ParVector x );
NALU_HYPRE_Int NALU_HYPRE_EuclidSolve ( NALU_HYPRE_Solver solver, NALU_HYPRE_ParCSRMatrix A, NALU_HYPRE_ParVector bb,
                              NALU_HYPRE_ParVector xx );
NALU_HYPRE_Int NALU_HYPRE_EuclidSetParams ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int argc, char *argv []);
NALU_HYPRE_Int NALU_HYPRE_EuclidSetParamsFromFile ( NALU_HYPRE_Solver solver, char *filename );
NALU_HYPRE_Int NALU_HYPRE_EuclidSetLevel ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int level );
NALU_HYPRE_Int NALU_HYPRE_EuclidSetBJ ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int bj );
NALU_HYPRE_Int NALU_HYPRE_EuclidSetStats ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int eu_stats );
NALU_HYPRE_Int NALU_HYPRE_EuclidSetMem ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int eu_mem );
NALU_HYPRE_Int NALU_HYPRE_EuclidSetSparseA ( NALU_HYPRE_Solver solver, NALU_HYPRE_Real sparse_A );
NALU_HYPRE_Int NALU_HYPRE_EuclidSetRowScale ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int row_scale );
NALU_HYPRE_Int NALU_HYPRE_EuclidSetILUT ( NALU_HYPRE_Solver solver, NALU_HYPRE_Real ilut );

/* NALU_HYPRE_parcsr_flexgmres.c */
NALU_HYPRE_Int NALU_HYPRE_ParCSRFlexGMRESCreate ( MPI_Comm comm, NALU_HYPRE_Solver *solver );
NALU_HYPRE_Int NALU_HYPRE_ParCSRFlexGMRESDestroy ( NALU_HYPRE_Solver solver );
NALU_HYPRE_Int NALU_HYPRE_ParCSRFlexGMRESSetup ( NALU_HYPRE_Solver solver, NALU_HYPRE_ParCSRMatrix A, NALU_HYPRE_ParVector b,
                                       NALU_HYPRE_ParVector x );
NALU_HYPRE_Int NALU_HYPRE_ParCSRFlexGMRESSolve ( NALU_HYPRE_Solver solver, NALU_HYPRE_ParCSRMatrix A, NALU_HYPRE_ParVector b,
                                       NALU_HYPRE_ParVector x );
NALU_HYPRE_Int NALU_HYPRE_ParCSRFlexGMRESSetKDim ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int k_dim );
NALU_HYPRE_Int NALU_HYPRE_ParCSRFlexGMRESSetTol ( NALU_HYPRE_Solver solver, NALU_HYPRE_Real tol );
NALU_HYPRE_Int NALU_HYPRE_ParCSRFlexGMRESSetAbsoluteTol ( NALU_HYPRE_Solver solver, NALU_HYPRE_Real a_tol );
NALU_HYPRE_Int NALU_HYPRE_ParCSRFlexGMRESSetMinIter ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int min_iter );
NALU_HYPRE_Int NALU_HYPRE_ParCSRFlexGMRESSetMaxIter ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int max_iter );
NALU_HYPRE_Int NALU_HYPRE_ParCSRFlexGMRESSetPrecond ( NALU_HYPRE_Solver solver, NALU_HYPRE_PtrToParSolverFcn precond,
                                            NALU_HYPRE_PtrToParSolverFcn precond_setup, NALU_HYPRE_Solver precond_solver );
NALU_HYPRE_Int NALU_HYPRE_ParCSRFlexGMRESGetPrecond ( NALU_HYPRE_Solver solver, NALU_HYPRE_Solver *precond_data_ptr );
NALU_HYPRE_Int NALU_HYPRE_ParCSRFlexGMRESSetLogging ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int logging );
NALU_HYPRE_Int NALU_HYPRE_ParCSRFlexGMRESSetPrintLevel ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int print_level );
NALU_HYPRE_Int NALU_HYPRE_ParCSRFlexGMRESGetNumIterations ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int *num_iterations );
NALU_HYPRE_Int NALU_HYPRE_ParCSRFlexGMRESGetFinalRelativeResidualNorm ( NALU_HYPRE_Solver solver,
                                                              NALU_HYPRE_Real *norm );
NALU_HYPRE_Int NALU_HYPRE_ParCSRFlexGMRESGetResidual ( NALU_HYPRE_Solver solver, NALU_HYPRE_ParVector *residual );
NALU_HYPRE_Int NALU_HYPRE_ParCSRFlexGMRESSetModifyPC ( NALU_HYPRE_Solver solver,
                                             NALU_HYPRE_PtrToModifyPCFcn modify_pc );

/* NALU_HYPRE_parcsr_gmres.c */
NALU_HYPRE_Int NALU_HYPRE_ParCSRGMRESCreate ( MPI_Comm comm, NALU_HYPRE_Solver *solver );
NALU_HYPRE_Int NALU_HYPRE_ParCSRGMRESDestroy ( NALU_HYPRE_Solver solver );
NALU_HYPRE_Int NALU_HYPRE_ParCSRGMRESSetup ( NALU_HYPRE_Solver solver, NALU_HYPRE_ParCSRMatrix A, NALU_HYPRE_ParVector b,
                                   NALU_HYPRE_ParVector x );
NALU_HYPRE_Int NALU_HYPRE_ParCSRGMRESSolve ( NALU_HYPRE_Solver solver, NALU_HYPRE_ParCSRMatrix A, NALU_HYPRE_ParVector b,
                                   NALU_HYPRE_ParVector x );
NALU_HYPRE_Int NALU_HYPRE_ParCSRGMRESSetKDim ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int k_dim );
NALU_HYPRE_Int NALU_HYPRE_ParCSRGMRESSetTol ( NALU_HYPRE_Solver solver, NALU_HYPRE_Real tol );
NALU_HYPRE_Int NALU_HYPRE_ParCSRGMRESSetAbsoluteTol ( NALU_HYPRE_Solver solver, NALU_HYPRE_Real a_tol );
NALU_HYPRE_Int NALU_HYPRE_ParCSRGMRESSetMinIter ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int min_iter );
NALU_HYPRE_Int NALU_HYPRE_ParCSRGMRESSetMaxIter ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int max_iter );
NALU_HYPRE_Int NALU_HYPRE_ParCSRGMRESSetStopCrit ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int stop_crit );
NALU_HYPRE_Int NALU_HYPRE_ParCSRGMRESSetPrecond ( NALU_HYPRE_Solver solver, NALU_HYPRE_PtrToParSolverFcn precond,
                                        NALU_HYPRE_PtrToParSolverFcn precond_setup, NALU_HYPRE_Solver precond_solver );
NALU_HYPRE_Int NALU_HYPRE_ParCSRGMRESGetPrecond ( NALU_HYPRE_Solver solver, NALU_HYPRE_Solver *precond_data_ptr );
NALU_HYPRE_Int NALU_HYPRE_ParCSRGMRESSetLogging ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int logging );
NALU_HYPRE_Int NALU_HYPRE_ParCSRGMRESSetPrintLevel ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int print_level );
NALU_HYPRE_Int NALU_HYPRE_ParCSRGMRESGetNumIterations ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int *num_iterations );
NALU_HYPRE_Int NALU_HYPRE_ParCSRGMRESGetFinalRelativeResidualNorm ( NALU_HYPRE_Solver solver, NALU_HYPRE_Real *norm );
NALU_HYPRE_Int NALU_HYPRE_ParCSRGMRESGetResidual ( NALU_HYPRE_Solver solver, NALU_HYPRE_ParVector *residual );


/*NALU_HYPRE_parcsr_cogmres.c*/
NALU_HYPRE_Int NALU_HYPRE_ParCSRCOGMRESCreate ( MPI_Comm comm, NALU_HYPRE_Solver *solver );
NALU_HYPRE_Int NALU_HYPRE_ParCSRCOGMRESDestroy ( NALU_HYPRE_Solver solver );
NALU_HYPRE_Int NALU_HYPRE_ParCSRCOGMRESSetup ( NALU_HYPRE_Solver solver, NALU_HYPRE_ParCSRMatrix A, NALU_HYPRE_ParVector b,
                                     NALU_HYPRE_ParVector x );
NALU_HYPRE_Int NALU_HYPRE_ParCSRCOGMRESSolve ( NALU_HYPRE_Solver solver, NALU_HYPRE_ParCSRMatrix A, NALU_HYPRE_ParVector b,
                                     NALU_HYPRE_ParVector x );
NALU_HYPRE_Int NALU_HYPRE_ParCSRCOGMRESSetKDim ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int k_dim );
NALU_HYPRE_Int NALU_HYPRE_ParCSRCOGMRESSetCGS2 ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int cgs2 );
NALU_HYPRE_Int NALU_HYPRE_ParCSRCOGMRESSetTol ( NALU_HYPRE_Solver solver, NALU_HYPRE_Real tol );
NALU_HYPRE_Int NALU_HYPRE_ParCSRCOGMRESSetAbsoluteTol ( NALU_HYPRE_Solver solver, NALU_HYPRE_Real a_tol );
NALU_HYPRE_Int NALU_HYPRE_ParCSRCOGMRESSetMinIter ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int min_iter );
NALU_HYPRE_Int NALU_HYPRE_ParCSRCOGMRESSetMaxIter ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int max_iter );
NALU_HYPRE_Int NALU_HYPRE_ParCSRCOGMRESSetPrecond ( NALU_HYPRE_Solver solver, NALU_HYPRE_PtrToParSolverFcn precond,
                                          NALU_HYPRE_PtrToParSolverFcn precond_setup, NALU_HYPRE_Solver precond_solver );
NALU_HYPRE_Int NALU_HYPRE_ParCSRCOGMRESGetPrecond ( NALU_HYPRE_Solver solver, NALU_HYPRE_Solver *precond_data_ptr );
NALU_HYPRE_Int NALU_HYPRE_ParCSRCOGMRESSetLogging ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int logging );
NALU_HYPRE_Int NALU_HYPRE_ParCSRCOGMRESSetPrintLevel ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int print_level );
NALU_HYPRE_Int NALU_HYPRE_ParCSRCOGMRESGetNumIterations ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int *num_iterations );
NALU_HYPRE_Int NALU_HYPRE_ParCSRCOGMRESGetFinalRelativeResidualNorm ( NALU_HYPRE_Solver solver, NALU_HYPRE_Real *norm );
NALU_HYPRE_Int NALU_HYPRE_ParCSRCOGMRESGetResidual ( NALU_HYPRE_Solver solver, NALU_HYPRE_ParVector *residual );



/* NALU_HYPRE_parcsr_hybrid.c */
NALU_HYPRE_Int NALU_HYPRE_ParCSRHybridCreate ( NALU_HYPRE_Solver *solver );
NALU_HYPRE_Int NALU_HYPRE_ParCSRHybridDestroy ( NALU_HYPRE_Solver solver );
NALU_HYPRE_Int NALU_HYPRE_ParCSRHybridSetup ( NALU_HYPRE_Solver solver, NALU_HYPRE_ParCSRMatrix A, NALU_HYPRE_ParVector b,
                                    NALU_HYPRE_ParVector x );
NALU_HYPRE_Int NALU_HYPRE_ParCSRHybridSolve ( NALU_HYPRE_Solver solver, NALU_HYPRE_ParCSRMatrix A, NALU_HYPRE_ParVector b,
                                    NALU_HYPRE_ParVector x );
NALU_HYPRE_Int NALU_HYPRE_ParCSRHybridSetTol ( NALU_HYPRE_Solver solver, NALU_HYPRE_Real tol );
NALU_HYPRE_Int NALU_HYPRE_ParCSRHybridSetAbsoluteTol ( NALU_HYPRE_Solver solver, NALU_HYPRE_Real tol );
NALU_HYPRE_Int NALU_HYPRE_ParCSRHybridSetConvergenceTol ( NALU_HYPRE_Solver solver, NALU_HYPRE_Real cf_tol );
NALU_HYPRE_Int NALU_HYPRE_ParCSRHybridSetDSCGMaxIter ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int dscg_max_its );
NALU_HYPRE_Int NALU_HYPRE_ParCSRHybridSetPCGMaxIter ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int pcg_max_its );
NALU_HYPRE_Int NALU_HYPRE_ParCSRHybridSetSetupType ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int setup_type );
NALU_HYPRE_Int NALU_HYPRE_ParCSRHybridSetSolverType ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int solver_type );
NALU_HYPRE_Int NALU_HYPRE_ParCSRHybridSetKDim ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int k_dim );
NALU_HYPRE_Int NALU_HYPRE_ParCSRHybridSetTwoNorm ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int two_norm );
NALU_HYPRE_Int NALU_HYPRE_ParCSRHybridSetStopCrit ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int stop_crit );
NALU_HYPRE_Int NALU_HYPRE_ParCSRHybridSetRelChange ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int rel_change );
NALU_HYPRE_Int NALU_HYPRE_ParCSRHybridSetPrecond ( NALU_HYPRE_Solver solver, NALU_HYPRE_PtrToParSolverFcn precond,
                                         NALU_HYPRE_PtrToParSolverFcn precond_setup, NALU_HYPRE_Solver precond_solver );
NALU_HYPRE_Int NALU_HYPRE_ParCSRHybridSetLogging ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int logging );
NALU_HYPRE_Int NALU_HYPRE_ParCSRHybridSetPrintLevel ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int print_level );
NALU_HYPRE_Int NALU_HYPRE_ParCSRHybridSetStrongThreshold ( NALU_HYPRE_Solver solver, NALU_HYPRE_Real strong_threshold );
NALU_HYPRE_Int NALU_HYPRE_ParCSRHybridSetMaxRowSum ( NALU_HYPRE_Solver solver, NALU_HYPRE_Real max_row_sum );
NALU_HYPRE_Int NALU_HYPRE_ParCSRHybridSetTruncFactor ( NALU_HYPRE_Solver solver, NALU_HYPRE_Real trunc_factor );
NALU_HYPRE_Int NALU_HYPRE_ParCSRHybridSetPMaxElmts ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int p_max );
NALU_HYPRE_Int NALU_HYPRE_ParCSRHybridSetMaxLevels ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int max_levels );
NALU_HYPRE_Int NALU_HYPRE_ParCSRHybridSetMeasureType ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int measure_type );
NALU_HYPRE_Int NALU_HYPRE_ParCSRHybridSetCoarsenType ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int coarsen_type );
NALU_HYPRE_Int NALU_HYPRE_ParCSRHybridSetInterpType ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int interp_type );
NALU_HYPRE_Int NALU_HYPRE_ParCSRHybridSetCycleType ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int cycle_type );
NALU_HYPRE_Int NALU_HYPRE_ParCSRHybridSetNumGridSweeps ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int *num_grid_sweeps );
NALU_HYPRE_Int NALU_HYPRE_ParCSRHybridSetGridRelaxType ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int *grid_relax_type );
NALU_HYPRE_Int NALU_HYPRE_ParCSRHybridSetGridRelaxPoints ( NALU_HYPRE_Solver solver,
                                                 NALU_HYPRE_Int **grid_relax_points );
NALU_HYPRE_Int NALU_HYPRE_ParCSRHybridSetNumSweeps ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int num_sweeps );
NALU_HYPRE_Int NALU_HYPRE_ParCSRHybridSetCycleNumSweeps ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int num_sweeps,
                                                NALU_HYPRE_Int k );
NALU_HYPRE_Int NALU_HYPRE_ParCSRHybridSetRelaxType ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int relax_type );
NALU_HYPRE_Int NALU_HYPRE_ParCSRHybridSetKeepTranspose ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int keepT );
NALU_HYPRE_Int NALU_HYPRE_ParCSRHybridSetCycleRelaxType ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int relax_type,
                                                NALU_HYPRE_Int k );
NALU_HYPRE_Int NALU_HYPRE_ParCSRHybridSetRelaxOrder ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int relax_order );
NALU_HYPRE_Int NALU_HYPRE_ParCSRHybridSetMaxCoarseSize ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int max_coarse_size );
NALU_HYPRE_Int NALU_HYPRE_ParCSRHybridSetMinCoarseSize ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int min_coarse_size );
NALU_HYPRE_Int NALU_HYPRE_ParCSRHybridSetSeqThreshold ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int seq_threshold );
NALU_HYPRE_Int NALU_HYPRE_ParCSRHybridSetRelaxWt ( NALU_HYPRE_Solver solver, NALU_HYPRE_Real relax_wt );
NALU_HYPRE_Int NALU_HYPRE_ParCSRHybridSetLevelRelaxWt ( NALU_HYPRE_Solver solver, NALU_HYPRE_Real relax_wt,
                                              NALU_HYPRE_Int level );
NALU_HYPRE_Int NALU_HYPRE_ParCSRHybridSetOuterWt ( NALU_HYPRE_Solver solver, NALU_HYPRE_Real outer_wt );
NALU_HYPRE_Int NALU_HYPRE_ParCSRHybridSetLevelOuterWt ( NALU_HYPRE_Solver solver, NALU_HYPRE_Real outer_wt,
                                              NALU_HYPRE_Int level );
NALU_HYPRE_Int NALU_HYPRE_ParCSRHybridSetRelaxWeight ( NALU_HYPRE_Solver solver, NALU_HYPRE_Real *relax_weight );
NALU_HYPRE_Int NALU_HYPRE_ParCSRHybridSetOmega ( NALU_HYPRE_Solver solver, NALU_HYPRE_Real *omega );
NALU_HYPRE_Int NALU_HYPRE_ParCSRHybridSetAggNumLevels ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int agg_num_levels );
NALU_HYPRE_Int NALU_HYPRE_ParCSRHybridSetNumPaths ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int num_paths );
NALU_HYPRE_Int NALU_HYPRE_ParCSRHybridSetNumFunctions ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int num_functions );
NALU_HYPRE_Int NALU_HYPRE_ParCSRHybridSetNodal ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int nodal );
NALU_HYPRE_Int NALU_HYPRE_ParCSRHybridSetDofFunc ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int *dof_func );
NALU_HYPRE_Int NALU_HYPRE_ParCSRHybridSetNonGalerkinTol ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int nongalerk_num_tol,
                                                NALU_HYPRE_Real *nongalerkin_tol );
NALU_HYPRE_Int NALU_HYPRE_ParCSRHybridGetNumIterations ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int *num_its );
NALU_HYPRE_Int NALU_HYPRE_ParCSRHybridGetDSCGNumIterations ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int *dscg_num_its );
NALU_HYPRE_Int NALU_HYPRE_ParCSRHybridGetPCGNumIterations ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int *pcg_num_its );
NALU_HYPRE_Int NALU_HYPRE_ParCSRHybridGetFinalRelativeResidualNorm ( NALU_HYPRE_Solver solver, NALU_HYPRE_Real *norm );
NALU_HYPRE_Int NALU_HYPRE_ParCSRHybridGetSetupSolveTime( NALU_HYPRE_Solver solver, NALU_HYPRE_Real *time );

/* NALU_HYPRE_parcsr_int.c */
NALU_HYPRE_Int hypre_ParSetRandomValues ( void *v, NALU_HYPRE_Int seed );
NALU_HYPRE_Int hypre_ParPrintVector ( void *v, const char *file );
void *hypre_ParReadVector ( MPI_Comm comm, const char *file );
NALU_HYPRE_Int hypre_ParVectorSize ( void *x );
NALU_HYPRE_Int NALU_HYPRE_ParCSRMultiVectorPrint ( void *x_, const char *fileName );
void *NALU_HYPRE_ParCSRMultiVectorRead ( MPI_Comm comm, void *ii_, const char *fileName );
NALU_HYPRE_Int aux_maskCount ( NALU_HYPRE_Int n, NALU_HYPRE_Int *mask );
void aux_indexFromMask ( NALU_HYPRE_Int n, NALU_HYPRE_Int *mask, NALU_HYPRE_Int *index );
NALU_HYPRE_Int NALU_HYPRE_TempParCSRSetupInterpreter ( mv_InterfaceInterpreter *i );
NALU_HYPRE_Int NALU_HYPRE_ParCSRSetupInterpreter ( mv_InterfaceInterpreter *i );
NALU_HYPRE_Int NALU_HYPRE_ParCSRSetupMatvec ( NALU_HYPRE_MatvecFunctions *mv );

/* NALU_HYPRE_parcsr_lgmres.c */
NALU_HYPRE_Int NALU_HYPRE_ParCSRLGMRESCreate ( MPI_Comm comm, NALU_HYPRE_Solver *solver );
NALU_HYPRE_Int NALU_HYPRE_ParCSRLGMRESDestroy ( NALU_HYPRE_Solver solver );
NALU_HYPRE_Int NALU_HYPRE_ParCSRLGMRESSetup ( NALU_HYPRE_Solver solver, NALU_HYPRE_ParCSRMatrix A, NALU_HYPRE_ParVector b,
                                    NALU_HYPRE_ParVector x );
NALU_HYPRE_Int NALU_HYPRE_ParCSRLGMRESSolve ( NALU_HYPRE_Solver solver, NALU_HYPRE_ParCSRMatrix A, NALU_HYPRE_ParVector b,
                                    NALU_HYPRE_ParVector x );
NALU_HYPRE_Int NALU_HYPRE_ParCSRLGMRESSetKDim ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int k_dim );
NALU_HYPRE_Int NALU_HYPRE_ParCSRLGMRESSetAugDim ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int aug_dim );
NALU_HYPRE_Int NALU_HYPRE_ParCSRLGMRESSetTol ( NALU_HYPRE_Solver solver, NALU_HYPRE_Real tol );
NALU_HYPRE_Int NALU_HYPRE_ParCSRLGMRESSetAbsoluteTol ( NALU_HYPRE_Solver solver, NALU_HYPRE_Real a_tol );
NALU_HYPRE_Int NALU_HYPRE_ParCSRLGMRESSetMinIter ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int min_iter );
NALU_HYPRE_Int NALU_HYPRE_ParCSRLGMRESSetMaxIter ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int max_iter );
NALU_HYPRE_Int NALU_HYPRE_ParCSRLGMRESSetPrecond ( NALU_HYPRE_Solver solver, NALU_HYPRE_PtrToParSolverFcn precond,
                                         NALU_HYPRE_PtrToParSolverFcn precond_setup, NALU_HYPRE_Solver precond_solver );
NALU_HYPRE_Int NALU_HYPRE_ParCSRLGMRESGetPrecond ( NALU_HYPRE_Solver solver, NALU_HYPRE_Solver *precond_data_ptr );
NALU_HYPRE_Int NALU_HYPRE_ParCSRLGMRESSetLogging ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int logging );
NALU_HYPRE_Int NALU_HYPRE_ParCSRLGMRESSetPrintLevel ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int print_level );
NALU_HYPRE_Int NALU_HYPRE_ParCSRLGMRESGetNumIterations ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int *num_iterations );
NALU_HYPRE_Int NALU_HYPRE_ParCSRLGMRESGetFinalRelativeResidualNorm ( NALU_HYPRE_Solver solver, NALU_HYPRE_Real *norm );
NALU_HYPRE_Int NALU_HYPRE_ParCSRLGMRESGetResidual ( NALU_HYPRE_Solver solver, NALU_HYPRE_ParVector *residual );

/* NALU_HYPRE_parcsr_ParaSails.c */
NALU_HYPRE_Int NALU_HYPRE_ParCSRParaSailsCreate ( MPI_Comm comm, NALU_HYPRE_Solver *solver );
NALU_HYPRE_Int NALU_HYPRE_ParCSRParaSailsDestroy ( NALU_HYPRE_Solver solver );
NALU_HYPRE_Int NALU_HYPRE_ParCSRParaSailsSetup ( NALU_HYPRE_Solver solver, NALU_HYPRE_ParCSRMatrix A, NALU_HYPRE_ParVector b,
                                       NALU_HYPRE_ParVector x );
NALU_HYPRE_Int NALU_HYPRE_ParCSRParaSailsSolve ( NALU_HYPRE_Solver solver, NALU_HYPRE_ParCSRMatrix A, NALU_HYPRE_ParVector b,
                                       NALU_HYPRE_ParVector x );
NALU_HYPRE_Int NALU_HYPRE_ParCSRParaSailsSetParams ( NALU_HYPRE_Solver solver, NALU_HYPRE_Real thresh,
                                           NALU_HYPRE_Int nlevels );
NALU_HYPRE_Int NALU_HYPRE_ParCSRParaSailsSetFilter ( NALU_HYPRE_Solver solver, NALU_HYPRE_Real filter );
NALU_HYPRE_Int NALU_HYPRE_ParCSRParaSailsGetFilter ( NALU_HYPRE_Solver solver, NALU_HYPRE_Real *filter );
NALU_HYPRE_Int NALU_HYPRE_ParCSRParaSailsSetSym ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int sym );
NALU_HYPRE_Int NALU_HYPRE_ParCSRParaSailsSetLoadbal ( NALU_HYPRE_Solver solver, NALU_HYPRE_Real loadbal );
NALU_HYPRE_Int NALU_HYPRE_ParCSRParaSailsGetLoadbal ( NALU_HYPRE_Solver solver, NALU_HYPRE_Real *loadbal );
NALU_HYPRE_Int NALU_HYPRE_ParCSRParaSailsSetReuse ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int reuse );
NALU_HYPRE_Int NALU_HYPRE_ParCSRParaSailsSetLogging ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int logging );
NALU_HYPRE_Int NALU_HYPRE_ParaSailsCreate ( MPI_Comm comm, NALU_HYPRE_Solver *solver );
NALU_HYPRE_Int NALU_HYPRE_ParaSailsDestroy ( NALU_HYPRE_Solver solver );
NALU_HYPRE_Int NALU_HYPRE_ParaSailsSetup ( NALU_HYPRE_Solver solver, NALU_HYPRE_ParCSRMatrix A, NALU_HYPRE_ParVector b,
                                 NALU_HYPRE_ParVector x );
NALU_HYPRE_Int NALU_HYPRE_ParaSailsSolve ( NALU_HYPRE_Solver solver, NALU_HYPRE_ParCSRMatrix A, NALU_HYPRE_ParVector b,
                                 NALU_HYPRE_ParVector x );
NALU_HYPRE_Int NALU_HYPRE_ParaSailsSetParams ( NALU_HYPRE_Solver solver, NALU_HYPRE_Real thresh, NALU_HYPRE_Int nlevels );
NALU_HYPRE_Int NALU_HYPRE_ParaSailsSetThresh ( NALU_HYPRE_Solver solver, NALU_HYPRE_Real thresh );
NALU_HYPRE_Int NALU_HYPRE_ParaSailsGetThresh ( NALU_HYPRE_Solver solver, NALU_HYPRE_Real *thresh );
NALU_HYPRE_Int NALU_HYPRE_ParaSailsSetNlevels ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int nlevels );
NALU_HYPRE_Int NALU_HYPRE_ParaSailsGetNlevels ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int *nlevels );
NALU_HYPRE_Int NALU_HYPRE_ParaSailsSetFilter ( NALU_HYPRE_Solver solver, NALU_HYPRE_Real filter );
NALU_HYPRE_Int NALU_HYPRE_ParaSailsGetFilter ( NALU_HYPRE_Solver solver, NALU_HYPRE_Real *filter );
NALU_HYPRE_Int NALU_HYPRE_ParaSailsSetSym ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int sym );
NALU_HYPRE_Int NALU_HYPRE_ParaSailsGetSym ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int *sym );
NALU_HYPRE_Int NALU_HYPRE_ParaSailsSetLoadbal ( NALU_HYPRE_Solver solver, NALU_HYPRE_Real loadbal );
NALU_HYPRE_Int NALU_HYPRE_ParaSailsGetLoadbal ( NALU_HYPRE_Solver solver, NALU_HYPRE_Real *loadbal );
NALU_HYPRE_Int NALU_HYPRE_ParaSailsSetReuse ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int reuse );
NALU_HYPRE_Int NALU_HYPRE_ParaSailsGetReuse ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int *reuse );
NALU_HYPRE_Int NALU_HYPRE_ParaSailsSetLogging ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int logging );
NALU_HYPRE_Int NALU_HYPRE_ParaSailsGetLogging ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int *logging );
NALU_HYPRE_Int NALU_HYPRE_ParaSailsBuildIJMatrix ( NALU_HYPRE_Solver solver, NALU_HYPRE_IJMatrix *pij_A );

/* NALU_HYPRE_parcsr_fsai.c */
NALU_HYPRE_Int NALU_HYPRE_FSAICreate ( NALU_HYPRE_Solver *solver);
NALU_HYPRE_Int NALU_HYPRE_FSAIDestroy ( NALU_HYPRE_Solver solver );
NALU_HYPRE_Int NALU_HYPRE_FSAISetup ( NALU_HYPRE_Solver solver, NALU_HYPRE_ParCSRMatrix A, NALU_HYPRE_ParVector b,
                            NALU_HYPRE_ParVector x );
NALU_HYPRE_Int NALU_HYPRE_FSAISolve ( NALU_HYPRE_Solver solver, NALU_HYPRE_ParCSRMatrix A, NALU_HYPRE_ParVector b,
                            NALU_HYPRE_ParVector x );
NALU_HYPRE_Int NALU_HYPRE_FSAISetAlgoType ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int algo_type );
NALU_HYPRE_Int NALU_HYPRE_FSAIGetAlgoType ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int *algo_type );
NALU_HYPRE_Int NALU_HYPRE_FSAISetMaxSteps ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int max_steps );
NALU_HYPRE_Int NALU_HYPRE_FSAIGetMaxSteps ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int *max_steps );
NALU_HYPRE_Int NALU_HYPRE_FSAISetMaxStepSize ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int max_step_size );
NALU_HYPRE_Int NALU_HYPRE_FSAIGetMaxStepSize ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int *max_step_size );
NALU_HYPRE_Int NALU_HYPRE_FSAISetKapTolerance ( NALU_HYPRE_Solver solver, NALU_HYPRE_Real  kap_tolerance );
NALU_HYPRE_Int NALU_HYPRE_FSAIGetKapTolerance ( NALU_HYPRE_Solver solver, NALU_HYPRE_Real *kap_tolerance );
NALU_HYPRE_Int NALU_HYPRE_FSAISetTolerance ( NALU_HYPRE_Solver solver, NALU_HYPRE_Real tolerance );
NALU_HYPRE_Int NALU_HYPRE_FSAIGetTolerance ( NALU_HYPRE_Solver solver, NALU_HYPRE_Real *tolerance );
NALU_HYPRE_Int NALU_HYPRE_FSAISetOmega ( NALU_HYPRE_Solver solver, NALU_HYPRE_Real omega );
NALU_HYPRE_Int NALU_HYPRE_FSAIGetOmega ( NALU_HYPRE_Solver solver, NALU_HYPRE_Real *omega );
NALU_HYPRE_Int NALU_HYPRE_FSAISetMaxIterations ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int max_iterations );
NALU_HYPRE_Int NALU_HYPRE_FSAIGetMaxIterations ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int *max_iterations );
NALU_HYPRE_Int NALU_HYPRE_FSAISetEigMaxIters ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int eig_max_iters );
NALU_HYPRE_Int NALU_HYPRE_FSAIGetEigMaxIters ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int *eig_max_iters );
NALU_HYPRE_Int NALU_HYPRE_FSAISetZeroGuess ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int zero_guess );
NALU_HYPRE_Int NALU_HYPRE_FSAIGetZeroGuess ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int *zero_guess );
NALU_HYPRE_Int NALU_HYPRE_FSAISetPrintLevel ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int print_level );
NALU_HYPRE_Int NALU_HYPRE_FSAIGetPrintLevel ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int *print_level );

/* NALU_HYPRE_parcsr_pcg.c */
NALU_HYPRE_Int NALU_HYPRE_ParCSRPCGCreate ( MPI_Comm comm, NALU_HYPRE_Solver *solver );
NALU_HYPRE_Int NALU_HYPRE_ParCSRPCGDestroy ( NALU_HYPRE_Solver solver );
NALU_HYPRE_Int NALU_HYPRE_ParCSRPCGSetup ( NALU_HYPRE_Solver solver, NALU_HYPRE_ParCSRMatrix A, NALU_HYPRE_ParVector b,
                                 NALU_HYPRE_ParVector x );
NALU_HYPRE_Int NALU_HYPRE_ParCSRPCGSolve ( NALU_HYPRE_Solver solver, NALU_HYPRE_ParCSRMatrix A, NALU_HYPRE_ParVector b,
                                 NALU_HYPRE_ParVector x );
NALU_HYPRE_Int NALU_HYPRE_ParCSRPCGSetTol ( NALU_HYPRE_Solver solver, NALU_HYPRE_Real tol );
NALU_HYPRE_Int NALU_HYPRE_ParCSRPCGSetAbsoluteTol ( NALU_HYPRE_Solver solver, NALU_HYPRE_Real a_tol );
NALU_HYPRE_Int NALU_HYPRE_ParCSRPCGSetMaxIter ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int max_iter );
NALU_HYPRE_Int NALU_HYPRE_ParCSRPCGSetStopCrit ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int stop_crit );
NALU_HYPRE_Int NALU_HYPRE_ParCSRPCGSetTwoNorm ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int two_norm );
NALU_HYPRE_Int NALU_HYPRE_ParCSRPCGSetRelChange ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int rel_change );
NALU_HYPRE_Int NALU_HYPRE_ParCSRPCGSetPrecond ( NALU_HYPRE_Solver solver, NALU_HYPRE_PtrToParSolverFcn precond,
                                      NALU_HYPRE_PtrToParSolverFcn precond_setup, NALU_HYPRE_Solver precond_solver );
NALU_HYPRE_Int NALU_HYPRE_ParCSRPCGGetPrecond ( NALU_HYPRE_Solver solver, NALU_HYPRE_Solver *precond_data_ptr );
NALU_HYPRE_Int NALU_HYPRE_ParCSRPCGSetPrintLevel ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int level );
NALU_HYPRE_Int NALU_HYPRE_ParCSRPCGSetLogging ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int level );
NALU_HYPRE_Int NALU_HYPRE_ParCSRPCGGetNumIterations ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int *num_iterations );
NALU_HYPRE_Int NALU_HYPRE_ParCSRPCGGetFinalRelativeResidualNorm ( NALU_HYPRE_Solver solver, NALU_HYPRE_Real *norm );
NALU_HYPRE_Int NALU_HYPRE_ParCSRPCGGetResidual ( NALU_HYPRE_Solver solver, NALU_HYPRE_ParVector *residual );
NALU_HYPRE_Int NALU_HYPRE_ParCSRDiagScaleSetup ( NALU_HYPRE_Solver solver, NALU_HYPRE_ParCSRMatrix A, NALU_HYPRE_ParVector y,
                                       NALU_HYPRE_ParVector x );
NALU_HYPRE_Int NALU_HYPRE_ParCSRDiagScale ( NALU_HYPRE_Solver solver, NALU_HYPRE_ParCSRMatrix HA, NALU_HYPRE_ParVector Hy,
                                  NALU_HYPRE_ParVector Hx );
NALU_HYPRE_Int NALU_HYPRE_ParCSROnProcTriSetup ( NALU_HYPRE_Solver solver, NALU_HYPRE_ParCSRMatrix HA,
                                       NALU_HYPRE_ParVector Hy, NALU_HYPRE_ParVector Hx );
NALU_HYPRE_Int NALU_HYPRE_ParCSROnProcTriSolve ( NALU_HYPRE_Solver solver, NALU_HYPRE_ParCSRMatrix HA,
                                       NALU_HYPRE_ParVector Hy, NALU_HYPRE_ParVector Hx );

/* NALU_HYPRE_parcsr_pilut.c */
NALU_HYPRE_Int NALU_HYPRE_ParCSRPilutCreate ( MPI_Comm comm, NALU_HYPRE_Solver *solver );
NALU_HYPRE_Int NALU_HYPRE_ParCSRPilutDestroy ( NALU_HYPRE_Solver solver );
NALU_HYPRE_Int NALU_HYPRE_ParCSRPilutSetup ( NALU_HYPRE_Solver solver, NALU_HYPRE_ParCSRMatrix A, NALU_HYPRE_ParVector b,
                                   NALU_HYPRE_ParVector x );
NALU_HYPRE_Int NALU_HYPRE_ParCSRPilutSolve ( NALU_HYPRE_Solver solver, NALU_HYPRE_ParCSRMatrix A, NALU_HYPRE_ParVector b,
                                   NALU_HYPRE_ParVector x );
NALU_HYPRE_Int NALU_HYPRE_ParCSRPilutSetMaxIter ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int max_iter );
NALU_HYPRE_Int NALU_HYPRE_ParCSRPilutSetDropTolerance ( NALU_HYPRE_Solver solver, NALU_HYPRE_Real tol );
NALU_HYPRE_Int NALU_HYPRE_ParCSRPilutSetFactorRowSize ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int size );

/* NALU_HYPRE_parcsr_schwarz.c */
NALU_HYPRE_Int NALU_HYPRE_SchwarzCreate ( NALU_HYPRE_Solver *solver );
NALU_HYPRE_Int NALU_HYPRE_SchwarzDestroy ( NALU_HYPRE_Solver solver );
NALU_HYPRE_Int NALU_HYPRE_SchwarzSetup ( NALU_HYPRE_Solver solver, NALU_HYPRE_ParCSRMatrix A, NALU_HYPRE_ParVector b,
                               NALU_HYPRE_ParVector x );
NALU_HYPRE_Int NALU_HYPRE_SchwarzSolve ( NALU_HYPRE_Solver solver, NALU_HYPRE_ParCSRMatrix A, NALU_HYPRE_ParVector b,
                               NALU_HYPRE_ParVector x );
NALU_HYPRE_Int NALU_HYPRE_SchwarzSetVariant ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int variant );
NALU_HYPRE_Int NALU_HYPRE_SchwarzSetOverlap ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int overlap );
NALU_HYPRE_Int NALU_HYPRE_SchwarzSetDomainType ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int domain_type );
NALU_HYPRE_Int NALU_HYPRE_SchwarzSetDomainStructure ( NALU_HYPRE_Solver solver, NALU_HYPRE_CSRMatrix domain_structure );
NALU_HYPRE_Int NALU_HYPRE_SchwarzSetNumFunctions ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int num_functions );
NALU_HYPRE_Int NALU_HYPRE_SchwarzSetNonSymm ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int use_nonsymm );
NALU_HYPRE_Int NALU_HYPRE_SchwarzSetRelaxWeight ( NALU_HYPRE_Solver solver, NALU_HYPRE_Real relax_weight );
NALU_HYPRE_Int NALU_HYPRE_SchwarzSetDofFunc ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int *dof_func );

/* par_add_cycle.c */
NALU_HYPRE_Int hypre_BoomerAMGAdditiveCycle ( void *amg_vdata );
NALU_HYPRE_Int hypre_CreateLambda ( void *amg_vdata );
NALU_HYPRE_Int hypre_CreateDinv ( void *amg_vdata );

/* par_amg.c */
void *hypre_BoomerAMGCreate ( void );
NALU_HYPRE_Int hypre_BoomerAMGDestroy ( void *data );
NALU_HYPRE_Int hypre_BoomerAMGSetRestriction ( void *data, NALU_HYPRE_Int restr_par );
NALU_HYPRE_Int hypre_BoomerAMGSetIsTriangular ( void *data, NALU_HYPRE_Int is_triangular );
NALU_HYPRE_Int hypre_BoomerAMGSetGMRESSwitchR ( void *data, NALU_HYPRE_Int gmres_switch );
NALU_HYPRE_Int hypre_BoomerAMGSetMaxLevels ( void *data, NALU_HYPRE_Int max_levels );
NALU_HYPRE_Int hypre_BoomerAMGGetMaxLevels ( void *data, NALU_HYPRE_Int *max_levels );
NALU_HYPRE_Int hypre_BoomerAMGSetMaxCoarseSize ( void *data, NALU_HYPRE_Int max_coarse_size );
NALU_HYPRE_Int hypre_BoomerAMGGetMaxCoarseSize ( void *data, NALU_HYPRE_Int *max_coarse_size );
NALU_HYPRE_Int hypre_BoomerAMGSetMinCoarseSize ( void *data, NALU_HYPRE_Int min_coarse_size );
NALU_HYPRE_Int hypre_BoomerAMGGetMinCoarseSize ( void *data, NALU_HYPRE_Int *min_coarse_size );
NALU_HYPRE_Int hypre_BoomerAMGSetSeqThreshold ( void *data, NALU_HYPRE_Int seq_threshold );
NALU_HYPRE_Int hypre_BoomerAMGGetSeqThreshold ( void *data, NALU_HYPRE_Int *seq_threshold );
NALU_HYPRE_Int hypre_BoomerAMGSetCoarsenCutFactor( void *data, NALU_HYPRE_Int coarsen_cut_factor );
NALU_HYPRE_Int hypre_BoomerAMGGetCoarsenCutFactor( void *data, NALU_HYPRE_Int *coarsen_cut_factor );
NALU_HYPRE_Int hypre_BoomerAMGSetRedundant ( void *data, NALU_HYPRE_Int redundant );
NALU_HYPRE_Int hypre_BoomerAMGGetRedundant ( void *data, NALU_HYPRE_Int *redundant );
NALU_HYPRE_Int hypre_BoomerAMGSetStrongThreshold ( void *data, NALU_HYPRE_Real strong_threshold );
NALU_HYPRE_Int hypre_BoomerAMGGetStrongThreshold ( void *data, NALU_HYPRE_Real *strong_threshold );
NALU_HYPRE_Int hypre_BoomerAMGSetStrongThresholdR ( void *data, NALU_HYPRE_Real strong_threshold );
NALU_HYPRE_Int hypre_BoomerAMGGetStrongThresholdR ( void *data, NALU_HYPRE_Real *strong_threshold );
NALU_HYPRE_Int hypre_BoomerAMGSetFilterThresholdR ( void *data, NALU_HYPRE_Real filter_threshold );
NALU_HYPRE_Int hypre_BoomerAMGGetFilterThresholdR ( void *data, NALU_HYPRE_Real *filter_threshold );
NALU_HYPRE_Int hypre_BoomerAMGSetSabs ( void *data, NALU_HYPRE_Int Sabs );
NALU_HYPRE_Int hypre_BoomerAMGSetMaxRowSum ( void *data, NALU_HYPRE_Real max_row_sum );
NALU_HYPRE_Int hypre_BoomerAMGGetMaxRowSum ( void *data, NALU_HYPRE_Real *max_row_sum );
NALU_HYPRE_Int hypre_BoomerAMGSetTruncFactor ( void *data, NALU_HYPRE_Real trunc_factor );
NALU_HYPRE_Int hypre_BoomerAMGGetTruncFactor ( void *data, NALU_HYPRE_Real *trunc_factor );
NALU_HYPRE_Int hypre_BoomerAMGSetPMaxElmts ( void *data, NALU_HYPRE_Int P_max_elmts );
NALU_HYPRE_Int hypre_BoomerAMGGetPMaxElmts ( void *data, NALU_HYPRE_Int *P_max_elmts );
NALU_HYPRE_Int hypre_BoomerAMGSetJacobiTruncThreshold ( void *data, NALU_HYPRE_Real jacobi_trunc_threshold );
NALU_HYPRE_Int hypre_BoomerAMGGetJacobiTruncThreshold ( void *data, NALU_HYPRE_Real *jacobi_trunc_threshold );
NALU_HYPRE_Int hypre_BoomerAMGSetPostInterpType ( void *data, NALU_HYPRE_Int post_interp_type );
NALU_HYPRE_Int hypre_BoomerAMGGetPostInterpType ( void *data, NALU_HYPRE_Int *post_interp_type );
NALU_HYPRE_Int hypre_BoomerAMGSetInterpType ( void *data, NALU_HYPRE_Int interp_type );
NALU_HYPRE_Int hypre_BoomerAMGGetInterpType ( void *data, NALU_HYPRE_Int *interp_type );
NALU_HYPRE_Int hypre_BoomerAMGSetSepWeight ( void *data, NALU_HYPRE_Int sep_weight );
NALU_HYPRE_Int hypre_BoomerAMGSetMinIter ( void *data, NALU_HYPRE_Int min_iter );
NALU_HYPRE_Int hypre_BoomerAMGGetMinIter ( void *data, NALU_HYPRE_Int *min_iter );
NALU_HYPRE_Int hypre_BoomerAMGSetMaxIter ( void *data, NALU_HYPRE_Int max_iter );
NALU_HYPRE_Int hypre_BoomerAMGGetMaxIter ( void *data, NALU_HYPRE_Int *max_iter );
NALU_HYPRE_Int hypre_BoomerAMGSetCoarsenType ( void *data, NALU_HYPRE_Int coarsen_type );
NALU_HYPRE_Int hypre_BoomerAMGGetCoarsenType ( void *data, NALU_HYPRE_Int *coarsen_type );
NALU_HYPRE_Int hypre_BoomerAMGSetMeasureType ( void *data, NALU_HYPRE_Int measure_type );
NALU_HYPRE_Int hypre_BoomerAMGGetMeasureType ( void *data, NALU_HYPRE_Int *measure_type );
NALU_HYPRE_Int hypre_BoomerAMGSetSetupType ( void *data, NALU_HYPRE_Int setup_type );
NALU_HYPRE_Int hypre_BoomerAMGGetSetupType ( void *data, NALU_HYPRE_Int *setup_type );
NALU_HYPRE_Int hypre_BoomerAMGSetFCycle ( void *data, NALU_HYPRE_Int fcycle );
NALU_HYPRE_Int hypre_BoomerAMGGetFCycle ( void *data, NALU_HYPRE_Int *fcycle );
NALU_HYPRE_Int hypre_BoomerAMGSetCycleType ( void *data, NALU_HYPRE_Int cycle_type );
NALU_HYPRE_Int hypre_BoomerAMGGetCycleType ( void *data, NALU_HYPRE_Int *cycle_type );
NALU_HYPRE_Int hypre_BoomerAMGSetConvergeType ( void *data, NALU_HYPRE_Int type );
NALU_HYPRE_Int hypre_BoomerAMGGetConvergeType ( void *data, NALU_HYPRE_Int *type );
NALU_HYPRE_Int hypre_BoomerAMGSetTol ( void *data, NALU_HYPRE_Real tol );
NALU_HYPRE_Int hypre_BoomerAMGGetTol ( void *data, NALU_HYPRE_Real *tol );
NALU_HYPRE_Int hypre_BoomerAMGSetNumSweeps ( void *data, NALU_HYPRE_Int num_sweeps );
NALU_HYPRE_Int hypre_BoomerAMGSetCycleNumSweeps ( void *data, NALU_HYPRE_Int num_sweeps, NALU_HYPRE_Int k );
NALU_HYPRE_Int hypre_BoomerAMGGetCycleNumSweeps ( void *data, NALU_HYPRE_Int *num_sweeps, NALU_HYPRE_Int k );
NALU_HYPRE_Int hypre_BoomerAMGSetNumGridSweeps ( void *data, NALU_HYPRE_Int *num_grid_sweeps );
NALU_HYPRE_Int hypre_BoomerAMGGetNumGridSweeps ( void *data, NALU_HYPRE_Int **num_grid_sweeps );
NALU_HYPRE_Int hypre_BoomerAMGSetRelaxType ( void *data, NALU_HYPRE_Int relax_type );
NALU_HYPRE_Int hypre_BoomerAMGSetCycleRelaxType ( void *data, NALU_HYPRE_Int relax_type, NALU_HYPRE_Int k );
NALU_HYPRE_Int hypre_BoomerAMGGetCycleRelaxType ( void *data, NALU_HYPRE_Int *relax_type, NALU_HYPRE_Int k );
NALU_HYPRE_Int hypre_BoomerAMGSetRelaxOrder ( void *data, NALU_HYPRE_Int relax_order );
NALU_HYPRE_Int hypre_BoomerAMGGetRelaxOrder ( void *data, NALU_HYPRE_Int *relax_order );
NALU_HYPRE_Int hypre_BoomerAMGSetGridRelaxType ( void *data, NALU_HYPRE_Int *grid_relax_type );
NALU_HYPRE_Int hypre_BoomerAMGGetGridRelaxType ( void *data, NALU_HYPRE_Int **grid_relax_type );
NALU_HYPRE_Int hypre_BoomerAMGSetGridRelaxPoints ( void *data, NALU_HYPRE_Int **grid_relax_points );
NALU_HYPRE_Int hypre_BoomerAMGGetGridRelaxPoints ( void *data, NALU_HYPRE_Int ***grid_relax_points );
NALU_HYPRE_Int hypre_BoomerAMGSetRelaxWeight ( void *data, NALU_HYPRE_Real *relax_weight );
NALU_HYPRE_Int hypre_BoomerAMGGetRelaxWeight ( void *data, NALU_HYPRE_Real **relax_weight );
NALU_HYPRE_Int hypre_BoomerAMGSetRelaxWt ( void *data, NALU_HYPRE_Real relax_weight );
NALU_HYPRE_Int hypre_BoomerAMGSetLevelRelaxWt ( void *data, NALU_HYPRE_Real relax_weight, NALU_HYPRE_Int level );
NALU_HYPRE_Int hypre_BoomerAMGGetLevelRelaxWt ( void *data, NALU_HYPRE_Real *relax_weight, NALU_HYPRE_Int level );
NALU_HYPRE_Int hypre_BoomerAMGSetOmega ( void *data, NALU_HYPRE_Real *omega );
NALU_HYPRE_Int hypre_BoomerAMGGetOmega ( void *data, NALU_HYPRE_Real **omega );
NALU_HYPRE_Int hypre_BoomerAMGSetOuterWt ( void *data, NALU_HYPRE_Real omega );
NALU_HYPRE_Int hypre_BoomerAMGSetLevelOuterWt ( void *data, NALU_HYPRE_Real omega, NALU_HYPRE_Int level );
NALU_HYPRE_Int hypre_BoomerAMGGetLevelOuterWt ( void *data, NALU_HYPRE_Real *omega, NALU_HYPRE_Int level );
NALU_HYPRE_Int hypre_BoomerAMGSetSmoothType ( void *data, NALU_HYPRE_Int smooth_type );
NALU_HYPRE_Int hypre_BoomerAMGGetSmoothType ( void *data, NALU_HYPRE_Int *smooth_type );
NALU_HYPRE_Int hypre_BoomerAMGSetSmoothNumLevels ( void *data, NALU_HYPRE_Int smooth_num_levels );
NALU_HYPRE_Int hypre_BoomerAMGGetSmoothNumLevels ( void *data, NALU_HYPRE_Int *smooth_num_levels );
NALU_HYPRE_Int hypre_BoomerAMGSetSmoothNumSweeps ( void *data, NALU_HYPRE_Int smooth_num_sweeps );
NALU_HYPRE_Int hypre_BoomerAMGGetSmoothNumSweeps ( void *data, NALU_HYPRE_Int *smooth_num_sweeps );
NALU_HYPRE_Int hypre_BoomerAMGSetLogging ( void *data, NALU_HYPRE_Int logging );
NALU_HYPRE_Int hypre_BoomerAMGGetLogging ( void *data, NALU_HYPRE_Int *logging );
NALU_HYPRE_Int hypre_BoomerAMGSetPrintLevel ( void *data, NALU_HYPRE_Int print_level );
NALU_HYPRE_Int hypre_BoomerAMGGetPrintLevel ( void *data, NALU_HYPRE_Int *print_level );
NALU_HYPRE_Int hypre_BoomerAMGSetPrintFileName ( void *data, const char *print_file_name );
NALU_HYPRE_Int hypre_BoomerAMGGetPrintFileName ( void *data, char **print_file_name );
NALU_HYPRE_Int hypre_BoomerAMGSetNumIterations ( void *data, NALU_HYPRE_Int num_iterations );
NALU_HYPRE_Int hypre_BoomerAMGSetDebugFlag ( void *data, NALU_HYPRE_Int debug_flag );
NALU_HYPRE_Int hypre_BoomerAMGGetDebugFlag ( void *data, NALU_HYPRE_Int *debug_flag );
NALU_HYPRE_Int hypre_BoomerAMGSetGSMG ( void *data, NALU_HYPRE_Int par );
NALU_HYPRE_Int hypre_BoomerAMGSetNumSamples ( void *data, NALU_HYPRE_Int par );
NALU_HYPRE_Int hypre_BoomerAMGSetCGCIts ( void *data, NALU_HYPRE_Int its );
NALU_HYPRE_Int hypre_BoomerAMGSetPlotGrids ( void *data, NALU_HYPRE_Int plotgrids );
NALU_HYPRE_Int hypre_BoomerAMGSetPlotFileName ( void *data, const char *plot_file_name );
NALU_HYPRE_Int hypre_BoomerAMGSetCoordDim ( void *data, NALU_HYPRE_Int coorddim );
NALU_HYPRE_Int hypre_BoomerAMGSetCoordinates ( void *data, float *coordinates );
NALU_HYPRE_Int hypre_BoomerAMGGetGridHierarchy(void *data, NALU_HYPRE_Int *cgrid );
NALU_HYPRE_Int hypre_BoomerAMGSetNumFunctions ( void *data, NALU_HYPRE_Int num_functions );
NALU_HYPRE_Int hypre_BoomerAMGGetNumFunctions ( void *data, NALU_HYPRE_Int *num_functions );
NALU_HYPRE_Int hypre_BoomerAMGSetNodal ( void *data, NALU_HYPRE_Int nodal );
NALU_HYPRE_Int hypre_BoomerAMGSetNodalLevels ( void *data, NALU_HYPRE_Int nodal_levels );
NALU_HYPRE_Int hypre_BoomerAMGSetNodalDiag ( void *data, NALU_HYPRE_Int nodal );
NALU_HYPRE_Int hypre_BoomerAMGSetKeepSameSign ( void *data, NALU_HYPRE_Int keep_same_sign );
NALU_HYPRE_Int hypre_BoomerAMGSetNumPaths ( void *data, NALU_HYPRE_Int num_paths );
NALU_HYPRE_Int hypre_BoomerAMGSetAggNumLevels ( void *data, NALU_HYPRE_Int agg_num_levels );
NALU_HYPRE_Int hypre_BoomerAMGSetAggInterpType ( void *data, NALU_HYPRE_Int agg_interp_type );
NALU_HYPRE_Int hypre_BoomerAMGSetAggPMaxElmts ( void *data, NALU_HYPRE_Int agg_P_max_elmts );
NALU_HYPRE_Int hypre_BoomerAMGSetMultAddPMaxElmts ( void *data, NALU_HYPRE_Int add_P_max_elmts );
NALU_HYPRE_Int hypre_BoomerAMGSetAddRelaxType ( void *data, NALU_HYPRE_Int add_rlx_type );
NALU_HYPRE_Int hypre_BoomerAMGSetAddRelaxWt ( void *data, NALU_HYPRE_Real add_rlx_wt );
NALU_HYPRE_Int hypre_BoomerAMGSetAggP12MaxElmts ( void *data, NALU_HYPRE_Int agg_P12_max_elmts );
NALU_HYPRE_Int hypre_BoomerAMGSetAggTruncFactor ( void *data, NALU_HYPRE_Real agg_trunc_factor );
NALU_HYPRE_Int hypre_BoomerAMGSetMultAddTruncFactor ( void *data, NALU_HYPRE_Real add_trunc_factor );
NALU_HYPRE_Int hypre_BoomerAMGSetAggP12TruncFactor ( void *data, NALU_HYPRE_Real agg_P12_trunc_factor );
NALU_HYPRE_Int hypre_BoomerAMGSetNumCRRelaxSteps ( void *data, NALU_HYPRE_Int num_CR_relax_steps );
NALU_HYPRE_Int hypre_BoomerAMGSetCRRate ( void *data, NALU_HYPRE_Real CR_rate );
NALU_HYPRE_Int hypre_BoomerAMGSetCRStrongTh ( void *data, NALU_HYPRE_Real CR_strong_th );
NALU_HYPRE_Int hypre_BoomerAMGSetADropTol( void     *data, NALU_HYPRE_Real  A_drop_tol );
NALU_HYPRE_Int hypre_BoomerAMGSetADropType( void     *data, NALU_HYPRE_Int  A_drop_type );
NALU_HYPRE_Int hypre_BoomerAMGSetISType ( void *data, NALU_HYPRE_Int IS_type );
NALU_HYPRE_Int hypre_BoomerAMGSetCRUseCG ( void *data, NALU_HYPRE_Int CR_use_CG );
NALU_HYPRE_Int hypre_BoomerAMGSetNumPoints ( void *data, NALU_HYPRE_Int num_points );
NALU_HYPRE_Int hypre_BoomerAMGSetDofFunc ( void *data, NALU_HYPRE_Int *dof_func );
NALU_HYPRE_Int hypre_BoomerAMGSetPointDofMap ( void *data, NALU_HYPRE_Int *point_dof_map );
NALU_HYPRE_Int hypre_BoomerAMGSetDofPoint ( void *data, NALU_HYPRE_Int *dof_point );
NALU_HYPRE_Int hypre_BoomerAMGGetNumIterations ( void *data, NALU_HYPRE_Int *num_iterations );
NALU_HYPRE_Int hypre_BoomerAMGGetCumNumIterations ( void *data, NALU_HYPRE_Int *cum_num_iterations );
NALU_HYPRE_Int hypre_BoomerAMGGetResidual ( void *data, hypre_ParVector **resid );
NALU_HYPRE_Int hypre_BoomerAMGGetRelResidualNorm ( void *data, NALU_HYPRE_Real *rel_resid_norm );
NALU_HYPRE_Int hypre_BoomerAMGSetVariant ( void *data, NALU_HYPRE_Int variant );
NALU_HYPRE_Int hypre_BoomerAMGGetVariant ( void *data, NALU_HYPRE_Int *variant );
NALU_HYPRE_Int hypre_BoomerAMGSetOverlap ( void *data, NALU_HYPRE_Int overlap );
NALU_HYPRE_Int hypre_BoomerAMGGetOverlap ( void *data, NALU_HYPRE_Int *overlap );
NALU_HYPRE_Int hypre_BoomerAMGSetDomainType ( void *data, NALU_HYPRE_Int domain_type );
NALU_HYPRE_Int hypre_BoomerAMGGetDomainType ( void *data, NALU_HYPRE_Int *domain_type );
NALU_HYPRE_Int hypre_BoomerAMGSetSchwarzRlxWeight ( void *data, NALU_HYPRE_Real schwarz_rlx_weight );
NALU_HYPRE_Int hypre_BoomerAMGGetSchwarzRlxWeight ( void *data, NALU_HYPRE_Real *schwarz_rlx_weight );
NALU_HYPRE_Int hypre_BoomerAMGSetSchwarzUseNonSymm ( void *data, NALU_HYPRE_Int use_nonsymm );
NALU_HYPRE_Int hypre_BoomerAMGSetSym ( void *data, NALU_HYPRE_Int sym );
NALU_HYPRE_Int hypre_BoomerAMGSetLevel ( void *data, NALU_HYPRE_Int level );
NALU_HYPRE_Int hypre_BoomerAMGSetThreshold ( void *data, NALU_HYPRE_Real thresh );
NALU_HYPRE_Int hypre_BoomerAMGSetFilter ( void *data, NALU_HYPRE_Real filter );
NALU_HYPRE_Int hypre_BoomerAMGSetDropTol ( void *data, NALU_HYPRE_Real drop_tol );
NALU_HYPRE_Int hypre_BoomerAMGSetMaxNzPerRow ( void *data, NALU_HYPRE_Int max_nz_per_row );
NALU_HYPRE_Int hypre_BoomerAMGSetEuclidFile ( void *data, char *euclidfile );
NALU_HYPRE_Int hypre_BoomerAMGSetEuLevel ( void *data, NALU_HYPRE_Int eu_level );
NALU_HYPRE_Int hypre_BoomerAMGSetEuSparseA ( void *data, NALU_HYPRE_Real eu_sparse_A );
NALU_HYPRE_Int hypre_BoomerAMGSetEuBJ ( void *data, NALU_HYPRE_Int eu_bj );
NALU_HYPRE_Int hypre_BoomerAMGSetILUType( void *data, NALU_HYPRE_Int ilu_type);
NALU_HYPRE_Int hypre_BoomerAMGSetILULevel( void *data, NALU_HYPRE_Int ilu_lfil);
NALU_HYPRE_Int hypre_BoomerAMGSetILUDroptol( void *data, NALU_HYPRE_Real ilu_droptol);
NALU_HYPRE_Int hypre_BoomerAMGSetILUTriSolve( void *data, NALU_HYPRE_Int ilu_tri_solve);
NALU_HYPRE_Int hypre_BoomerAMGSetILULowerJacobiIters( void *data, NALU_HYPRE_Int ilu_lower_jacobi_iters);
NALU_HYPRE_Int hypre_BoomerAMGSetILUUpperJacobiIters( void *data, NALU_HYPRE_Int ilu_upper_jacobi_iters);
NALU_HYPRE_Int hypre_BoomerAMGSetILUMaxIter( void *data, NALU_HYPRE_Int ilu_max_iter);
NALU_HYPRE_Int hypre_BoomerAMGSetILUMaxRowNnz( void *data, NALU_HYPRE_Int ilu_max_row_nnz);
NALU_HYPRE_Int hypre_BoomerAMGSetILULocalReordering( void *data, NALU_HYPRE_Int ilu_reordering_type);
NALU_HYPRE_Int hypre_BoomerAMGSetFSAIMaxSteps ( void *data, NALU_HYPRE_Int fsai_max_steps);
NALU_HYPRE_Int hypre_BoomerAMGSetFSAIMaxStepSize ( void *data, NALU_HYPRE_Int fsai_max_step_size);
NALU_HYPRE_Int hypre_BoomerAMGSetFSAIEigMaxIters ( void *data, NALU_HYPRE_Int fsai_eig_max_iters);
NALU_HYPRE_Int hypre_BoomerAMGSetFSAIKapTolerance ( void *data, NALU_HYPRE_Real fsai_kap_tolerance);
NALU_HYPRE_Int hypre_BoomerAMGSetChebyOrder ( void *data, NALU_HYPRE_Int order );
NALU_HYPRE_Int hypre_BoomerAMGSetChebyFraction ( void *data, NALU_HYPRE_Real ratio );
NALU_HYPRE_Int hypre_BoomerAMGSetChebyEigEst ( void *data, NALU_HYPRE_Int eig_est );
NALU_HYPRE_Int hypre_BoomerAMGSetChebyVariant ( void *data, NALU_HYPRE_Int variant );
NALU_HYPRE_Int hypre_BoomerAMGSetChebyScale ( void *data, NALU_HYPRE_Int scale );
NALU_HYPRE_Int hypre_BoomerAMGSetInterpVectors ( void *solver, NALU_HYPRE_Int num_vectors,
                                            hypre_ParVector **interp_vectors );
NALU_HYPRE_Int hypre_BoomerAMGSetInterpVecVariant ( void *solver, NALU_HYPRE_Int var );
NALU_HYPRE_Int hypre_BoomerAMGSetInterpVecQMax ( void *data, NALU_HYPRE_Int q_max );
NALU_HYPRE_Int hypre_BoomerAMGSetInterpVecAbsQTrunc ( void *data, NALU_HYPRE_Real q_trunc );
NALU_HYPRE_Int hypre_BoomerAMGSetSmoothInterpVectors ( void *solver, NALU_HYPRE_Int smooth_interp_vectors );
NALU_HYPRE_Int hypre_BoomerAMGSetInterpRefine ( void *data, NALU_HYPRE_Int num_refine );
NALU_HYPRE_Int hypre_BoomerAMGSetInterpVecFirstLevel ( void *data, NALU_HYPRE_Int level );
NALU_HYPRE_Int hypre_BoomerAMGSetAdditive ( void *data, NALU_HYPRE_Int additive );
NALU_HYPRE_Int hypre_BoomerAMGGetAdditive ( void *data, NALU_HYPRE_Int *additive );
NALU_HYPRE_Int hypre_BoomerAMGSetMultAdditive ( void *data, NALU_HYPRE_Int mult_additive );
NALU_HYPRE_Int hypre_BoomerAMGGetMultAdditive ( void *data, NALU_HYPRE_Int *mult_additive );
NALU_HYPRE_Int hypre_BoomerAMGSetSimple ( void *data, NALU_HYPRE_Int simple );
NALU_HYPRE_Int hypre_BoomerAMGGetSimple ( void *data, NALU_HYPRE_Int *simple );
NALU_HYPRE_Int hypre_BoomerAMGSetAddLastLvl ( void *data, NALU_HYPRE_Int add_last_lvl );
NALU_HYPRE_Int hypre_BoomerAMGSetNonGalerkinTol ( void *data, NALU_HYPRE_Real nongalerkin_tol );
NALU_HYPRE_Int hypre_BoomerAMGSetLevelNonGalerkinTol ( void *data, NALU_HYPRE_Real nongalerkin_tol,
                                                  NALU_HYPRE_Int level );
NALU_HYPRE_Int hypre_BoomerAMGSetNonGalerkTol ( void *data, NALU_HYPRE_Int nongalerk_num_tol,
                                           NALU_HYPRE_Real *nongalerk_tol );
NALU_HYPRE_Int hypre_BoomerAMGSetRAP2 ( void *data, NALU_HYPRE_Int rap2 );
NALU_HYPRE_Int hypre_BoomerAMGSetModuleRAP2 ( void *data, NALU_HYPRE_Int mod_rap2 );
NALU_HYPRE_Int hypre_BoomerAMGSetKeepTranspose ( void *data, NALU_HYPRE_Int keepTranspose );
#ifdef NALU_HYPRE_USING_DSUPERLU
NALU_HYPRE_Int hypre_BoomerAMGSetDSLUThreshold ( void *data, NALU_HYPRE_Int slu_threshold );
#endif
NALU_HYPRE_Int hypre_BoomerAMGSetCPoints( void *data, NALU_HYPRE_Int cpt_coarse_level,
                                     NALU_HYPRE_Int  num_cpt_coarse, NALU_HYPRE_BigInt *cpt_coarse_index );
NALU_HYPRE_Int hypre_BoomerAMGSetFPoints( void *data, NALU_HYPRE_Int isolated, NALU_HYPRE_Int num_points,
                                     NALU_HYPRE_BigInt *indices );
NALU_HYPRE_Int hypre_BoomerAMGSetCumNnzAP ( void *data , NALU_HYPRE_Real cum_nnz_AP );
NALU_HYPRE_Int hypre_BoomerAMGGetCumNnzAP ( void *data , NALU_HYPRE_Real *cum_nnz_AP );

/* par_amg_setup.c */
NALU_HYPRE_Int hypre_BoomerAMGSetup ( void *amg_vdata, hypre_ParCSRMatrix *A, hypre_ParVector *f,
                                 hypre_ParVector *u );

/* par_amg_solve.c */
NALU_HYPRE_Int hypre_BoomerAMGSolve ( void *amg_vdata, hypre_ParCSRMatrix *A, hypre_ParVector *f,
                                 hypre_ParVector *u );

/* par_amg_solveT.c */
NALU_HYPRE_Int hypre_BoomerAMGSolveT ( void *amg_vdata, hypre_ParCSRMatrix *A, hypre_ParVector *f,
                                  hypre_ParVector *u );
NALU_HYPRE_Int hypre_BoomerAMGCycleT ( void *amg_vdata, hypre_ParVector **F_array,
                                  hypre_ParVector **U_array );
NALU_HYPRE_Int hypre_BoomerAMGRelaxT ( hypre_ParCSRMatrix *A, hypre_ParVector *f, NALU_HYPRE_Int *cf_marker,
                                  NALU_HYPRE_Int relax_type, NALU_HYPRE_Int relax_points, NALU_HYPRE_Real relax_weight, hypre_ParVector *u,
                                  hypre_ParVector *Vtemp );

/* par_cgc_coarsen.c */
NALU_HYPRE_Int hypre_BoomerAMGCoarsenCGCb ( hypre_ParCSRMatrix *S, hypre_ParCSRMatrix *A,
                                       NALU_HYPRE_Int measure_type, NALU_HYPRE_Int coarsen_type, NALU_HYPRE_Int cgc_its, NALU_HYPRE_Int debug_flag,
                                       hypre_IntArray **CF_marker_ptr );
NALU_HYPRE_Int hypre_BoomerAMGCoarsenCGC ( hypre_ParCSRMatrix *S, NALU_HYPRE_Int numberofgrids,
                                      NALU_HYPRE_Int coarsen_type, NALU_HYPRE_Int *CF_marker );
NALU_HYPRE_Int hypre_AmgCGCPrepare ( hypre_ParCSRMatrix *S, NALU_HYPRE_Int nlocal, NALU_HYPRE_Int *CF_marker,
                                NALU_HYPRE_Int **CF_marker_offd, NALU_HYPRE_Int coarsen_type, NALU_HYPRE_Int **vrange );
//NALU_HYPRE_Int hypre_AmgCGCPrepare ( hypre_ParCSRMatrix *S , NALU_HYPRE_Int nlocal , NALU_HYPRE_Int *CF_marker , NALU_HYPRE_BigInt **CF_marker_offd , NALU_HYPRE_Int coarsen_type , NALU_HYPRE_BigInt **vrange );
NALU_HYPRE_Int hypre_AmgCGCGraphAssemble ( hypre_ParCSRMatrix *S, NALU_HYPRE_Int *vertexrange,
                                      NALU_HYPRE_Int *CF_marker, NALU_HYPRE_Int *CF_marker_offd, NALU_HYPRE_Int coarsen_type, NALU_HYPRE_IJMatrix *ijG );
NALU_HYPRE_Int hypre_AmgCGCChoose ( hypre_CSRMatrix *G, NALU_HYPRE_Int *vertexrange, NALU_HYPRE_Int mpisize,
                               NALU_HYPRE_Int **coarse );
NALU_HYPRE_Int hypre_AmgCGCBoundaryFix ( hypre_ParCSRMatrix *S, NALU_HYPRE_Int *CF_marker,
                                    NALU_HYPRE_Int *CF_marker_offd );

/* par_cg_relax_wt.c */
NALU_HYPRE_Int hypre_BoomerAMGCGRelaxWt ( void *amg_vdata, NALU_HYPRE_Int level, NALU_HYPRE_Int num_cg_sweeps,
                                     NALU_HYPRE_Real *rlx_wt_ptr );
NALU_HYPRE_Int hypre_Bisection ( NALU_HYPRE_Int n, NALU_HYPRE_Real *diag, NALU_HYPRE_Real *offd, NALU_HYPRE_Real y,
                            NALU_HYPRE_Real z, NALU_HYPRE_Real tol, NALU_HYPRE_Int k, NALU_HYPRE_Real *ev_ptr );

/* par_cheby.c */
NALU_HYPRE_Int hypre_ParCSRRelax_Cheby_Setup ( hypre_ParCSRMatrix *A, NALU_HYPRE_Real max_eig,
                                          NALU_HYPRE_Real min_eig, NALU_HYPRE_Real fraction, NALU_HYPRE_Int order, NALU_HYPRE_Int scale, NALU_HYPRE_Int variant,
                                          NALU_HYPRE_Real **coefs_ptr, NALU_HYPRE_Real **ds_ptr );
NALU_HYPRE_Int hypre_ParCSRRelax_Cheby_Solve ( hypre_ParCSRMatrix *A, hypre_ParVector *f,
                                          NALU_HYPRE_Real *ds_data, NALU_HYPRE_Real *coefs, NALU_HYPRE_Int order, NALU_HYPRE_Int scale, NALU_HYPRE_Int variant,
                                          hypre_ParVector *u, hypre_ParVector *v, hypre_ParVector *r, hypre_ParVector *orig_u_vec,
                                          hypre_ParVector *tmp_vec);

NALU_HYPRE_Int hypre_ParCSRRelax_Cheby_SolveHost ( hypre_ParCSRMatrix *A, hypre_ParVector *f,
                                              NALU_HYPRE_Real *ds_data, NALU_HYPRE_Real *coefs, NALU_HYPRE_Int order, NALU_HYPRE_Int scale, NALU_HYPRE_Int variant,
                                              hypre_ParVector *u, hypre_ParVector *v, hypre_ParVector *r, hypre_ParVector *orig_u_vec,
                                              hypre_ParVector *tmp_vec);

/* par_cheby_device.c */
NALU_HYPRE_Int hypre_ParCSRRelax_Cheby_SolveDevice ( hypre_ParCSRMatrix *A, hypre_ParVector *f,
                                                NALU_HYPRE_Real *ds_data, NALU_HYPRE_Real *coefs, NALU_HYPRE_Int order, NALU_HYPRE_Int scale, NALU_HYPRE_Int variant,
                                                hypre_ParVector *u, hypre_ParVector *v, hypre_ParVector *r, hypre_ParVector *orig_u_vec,
                                                hypre_ParVector *tmp_vec);

/* par_coarsen.c */
NALU_HYPRE_Int hypre_BoomerAMGCoarsen ( hypre_ParCSRMatrix *S, hypre_ParCSRMatrix *A, NALU_HYPRE_Int CF_init,
                                   NALU_HYPRE_Int debug_flag, hypre_IntArray **CF_marker_ptr );
NALU_HYPRE_Int hypre_BoomerAMGCoarsenRuge ( hypre_ParCSRMatrix *S, hypre_ParCSRMatrix *A,
                                       NALU_HYPRE_Int measure_type, NALU_HYPRE_Int coarsen_type, NALU_HYPRE_Int cut_factor, NALU_HYPRE_Int debug_flag,
                                       hypre_IntArray **CF_marker_ptr );
NALU_HYPRE_Int hypre_BoomerAMGCoarsenFalgout ( hypre_ParCSRMatrix *S, hypre_ParCSRMatrix *A,
                                          NALU_HYPRE_Int measure_type, NALU_HYPRE_Int cut_factor, NALU_HYPRE_Int debug_flag,
                                          hypre_IntArray **CF_marker_ptr );
NALU_HYPRE_Int hypre_BoomerAMGCoarsenHMIS ( hypre_ParCSRMatrix *S, hypre_ParCSRMatrix *A,
                                       NALU_HYPRE_Int measure_type, NALU_HYPRE_Int cut_factor, NALU_HYPRE_Int debug_flag,
                                       hypre_IntArray **CF_marker_ptr );
NALU_HYPRE_Int hypre_BoomerAMGCoarsenPMIS ( hypre_ParCSRMatrix *S, hypre_ParCSRMatrix *A,
                                       NALU_HYPRE_Int CF_init, NALU_HYPRE_Int debug_flag, hypre_IntArray **CF_marker_ptr );
NALU_HYPRE_Int hypre_BoomerAMGCoarsenPMISHost ( hypre_ParCSRMatrix *S, hypre_ParCSRMatrix *A,
                                           NALU_HYPRE_Int CF_init, NALU_HYPRE_Int debug_flag, hypre_IntArray **CF_marker_ptr );

NALU_HYPRE_Int hypre_BoomerAMGCoarsenPMISDevice( hypre_ParCSRMatrix *S, hypre_ParCSRMatrix *A,
                                            NALU_HYPRE_Int CF_init, NALU_HYPRE_Int debug_flag, hypre_IntArray **CF_marker_ptr );

/* par_coarsen_device.c */
NALU_HYPRE_Int hypre_GetGlobalMeasureDevice( hypre_ParCSRMatrix *S, hypre_ParCSRCommPkg *comm_pkg,
                                        NALU_HYPRE_Int CF_init, NALU_HYPRE_Int aug_rand, NALU_HYPRE_Real *measure_diag, NALU_HYPRE_Real *measure_offd,
                                        NALU_HYPRE_Real *real_send_buf );

/* par_coarse_parms.c */
NALU_HYPRE_Int hypre_BoomerAMGCoarseParms ( MPI_Comm comm, NALU_HYPRE_Int local_num_variables,
                                       NALU_HYPRE_Int num_functions, hypre_IntArray *dof_func, hypre_IntArray *CF_marker,
                                       hypre_IntArray **coarse_dof_func_ptr, NALU_HYPRE_BigInt *coarse_pnts_global );
NALU_HYPRE_Int hypre_BoomerAMGCoarseParmsHost ( MPI_Comm comm, NALU_HYPRE_Int local_num_variables,
                                           NALU_HYPRE_Int num_functions, hypre_IntArray *dof_func, hypre_IntArray *CF_marker,
                                           hypre_IntArray **coarse_dof_func_ptr, NALU_HYPRE_BigInt *coarse_pnts_global );
NALU_HYPRE_Int hypre_BoomerAMGCoarseParmsDevice ( MPI_Comm comm, NALU_HYPRE_Int local_num_variables,
                                             NALU_HYPRE_Int num_functions, hypre_IntArray *dof_func, hypre_IntArray *CF_marker,
                                             hypre_IntArray **coarse_dof_func_ptr, NALU_HYPRE_BigInt *coarse_pnts_global );
NALU_HYPRE_Int hypre_BoomerAMGInitDofFuncDevice( NALU_HYPRE_Int *dof_func, NALU_HYPRE_Int local_size,
                                            NALU_HYPRE_Int offset, NALU_HYPRE_Int num_functions );

/* par_coordinates.c */
float *GenerateCoordinates ( MPI_Comm comm, NALU_HYPRE_BigInt nx, NALU_HYPRE_BigInt ny, NALU_HYPRE_BigInt nz,
                             NALU_HYPRE_Int P, NALU_HYPRE_Int Q, NALU_HYPRE_Int R, NALU_HYPRE_Int p, NALU_HYPRE_Int q, NALU_HYPRE_Int r, NALU_HYPRE_Int coorddim );

/* par_cr.c */
NALU_HYPRE_Int hypre_BoomerAMGCoarsenCR1 ( hypre_ParCSRMatrix *A, hypre_IntArray **CF_marker_ptr,
                                      NALU_HYPRE_BigInt *coarse_size_ptr, NALU_HYPRE_Int num_CR_relax_steps, NALU_HYPRE_Int IS_type,
                                      NALU_HYPRE_Int CRaddCpoints );
NALU_HYPRE_Int hypre_cr ( NALU_HYPRE_Int *A_i, NALU_HYPRE_Int *A_j, NALU_HYPRE_Real *A_data, NALU_HYPRE_Int n, NALU_HYPRE_Int *cf,
                     NALU_HYPRE_Int rlx, NALU_HYPRE_Real omega, NALU_HYPRE_Real tg, NALU_HYPRE_Int mu );
NALU_HYPRE_Int hypre_GraphAdd ( Link *list, NALU_HYPRE_Int *head, NALU_HYPRE_Int *tail, NALU_HYPRE_Int index,
                           NALU_HYPRE_Int istack );
NALU_HYPRE_Int hypre_GraphRemove ( Link *list, NALU_HYPRE_Int *head, NALU_HYPRE_Int *tail, NALU_HYPRE_Int index );
NALU_HYPRE_Int hypre_IndepSetGreedy ( NALU_HYPRE_Int *A_i, NALU_HYPRE_Int *A_j, NALU_HYPRE_Int n, NALU_HYPRE_Int *cf );
NALU_HYPRE_Int hypre_IndepSetGreedyS ( NALU_HYPRE_Int *A_i, NALU_HYPRE_Int *A_j, NALU_HYPRE_Int n, NALU_HYPRE_Int *cf );
NALU_HYPRE_Int hypre_fptjaccr ( NALU_HYPRE_Int *cf, NALU_HYPRE_Int *A_i, NALU_HYPRE_Int *A_j, NALU_HYPRE_Real *A_data,
                           NALU_HYPRE_Int n, NALU_HYPRE_Real *e0, NALU_HYPRE_Real omega, NALU_HYPRE_Real *e1 );
NALU_HYPRE_Int hypre_fptgscr ( NALU_HYPRE_Int *cf, NALU_HYPRE_Int *A_i, NALU_HYPRE_Int *A_j, NALU_HYPRE_Real *A_data,
                          NALU_HYPRE_Int n, NALU_HYPRE_Real *e0, NALU_HYPRE_Real *e1 );
NALU_HYPRE_Int hypre_formu ( NALU_HYPRE_Int *cf, NALU_HYPRE_Int n, NALU_HYPRE_Real *e1, NALU_HYPRE_Int *A_i,
                        NALU_HYPRE_Real rho );
NALU_HYPRE_Int hypre_BoomerAMGIndepRS ( hypre_ParCSRMatrix *S, NALU_HYPRE_Int measure_type,
                                   NALU_HYPRE_Int debug_flag, NALU_HYPRE_Int *CF_marker );
NALU_HYPRE_Int hypre_BoomerAMGIndepRSa ( hypre_ParCSRMatrix *S, NALU_HYPRE_Int measure_type,
                                    NALU_HYPRE_Int debug_flag, NALU_HYPRE_Int *CF_marker );
NALU_HYPRE_Int hypre_BoomerAMGIndepHMIS ( hypre_ParCSRMatrix *S, NALU_HYPRE_Int measure_type,
                                     NALU_HYPRE_Int debug_flag, NALU_HYPRE_Int *CF_marker );
NALU_HYPRE_Int hypre_BoomerAMGIndepHMISa ( hypre_ParCSRMatrix *S, NALU_HYPRE_Int measure_type,
                                      NALU_HYPRE_Int debug_flag, NALU_HYPRE_Int *CF_marker );
NALU_HYPRE_Int hypre_BoomerAMGIndepPMIS ( hypre_ParCSRMatrix *S, NALU_HYPRE_Int CF_init, NALU_HYPRE_Int debug_flag,
                                     NALU_HYPRE_Int *CF_marker );
NALU_HYPRE_Int hypre_BoomerAMGIndepPMISa ( hypre_ParCSRMatrix *S, NALU_HYPRE_Int CF_init,
                                      NALU_HYPRE_Int debug_flag, NALU_HYPRE_Int *CF_marker );
NALU_HYPRE_Int hypre_BoomerAMGCoarsenCR ( hypre_ParCSRMatrix *A, hypre_IntArray **CF_marker_ptr,
                                     NALU_HYPRE_BigInt *coarse_size_ptr, NALU_HYPRE_Int num_CR_relax_steps, NALU_HYPRE_Int IS_type,
                                     NALU_HYPRE_Int num_functions, NALU_HYPRE_Int rlx_type, NALU_HYPRE_Real relax_weight, NALU_HYPRE_Real omega,
                                     NALU_HYPRE_Real theta, NALU_HYPRE_Solver smoother, hypre_ParCSRMatrix *AN, NALU_HYPRE_Int useCG,
                                     hypre_ParCSRMatrix *S );

/* par_cycle.c */
NALU_HYPRE_Int hypre_BoomerAMGCycle ( void *amg_vdata, hypre_ParVector **F_array,
                                 hypre_ParVector **U_array );

/* par_difconv.c */
NALU_HYPRE_ParCSRMatrix GenerateDifConv ( MPI_Comm comm, NALU_HYPRE_BigInt nx, NALU_HYPRE_BigInt ny,
                                     NALU_HYPRE_BigInt nz, NALU_HYPRE_Int P, NALU_HYPRE_Int Q, NALU_HYPRE_Int R, NALU_HYPRE_Int p, NALU_HYPRE_Int q, NALU_HYPRE_Int r,
                                     NALU_HYPRE_Real *value );

/* par_gsmg.c */
NALU_HYPRE_Int hypre_ParCSRMatrixFillSmooth ( NALU_HYPRE_Int nsamples, NALU_HYPRE_Real *samples,
                                         hypre_ParCSRMatrix *S, hypre_ParCSRMatrix *A, NALU_HYPRE_Int num_functions, NALU_HYPRE_Int *dof_func );
NALU_HYPRE_Real hypre_ParCSRMatrixChooseThresh ( hypre_ParCSRMatrix *S );
NALU_HYPRE_Int hypre_ParCSRMatrixThreshold ( hypre_ParCSRMatrix *A, NALU_HYPRE_Real thresh );
NALU_HYPRE_Int hypre_BoomerAMGCreateSmoothVecs ( void *data, hypre_ParCSRMatrix *A, NALU_HYPRE_Int num_sweeps,
                                            NALU_HYPRE_Int level, NALU_HYPRE_Real **SmoothVecs_p );
NALU_HYPRE_Int hypre_BoomerAMGCreateSmoothDirs ( void *data, hypre_ParCSRMatrix *A,
                                            NALU_HYPRE_Real *SmoothVecs, NALU_HYPRE_Real thresh, NALU_HYPRE_Int num_functions, NALU_HYPRE_Int *dof_func,
                                            hypre_ParCSRMatrix **S_ptr );
NALU_HYPRE_Int hypre_BoomerAMGNormalizeVecs ( NALU_HYPRE_Int n, NALU_HYPRE_Int num, NALU_HYPRE_Real *V );
NALU_HYPRE_Int hypre_BoomerAMGFitVectors ( NALU_HYPRE_Int ip, NALU_HYPRE_Int n, NALU_HYPRE_Int num, const NALU_HYPRE_Real *V,
                                      NALU_HYPRE_Int nc, const NALU_HYPRE_Int *ind, NALU_HYPRE_Real *val );
NALU_HYPRE_Int hypre_BoomerAMGBuildInterpLS ( hypre_ParCSRMatrix *A, NALU_HYPRE_Int *CF_marker,
                                         hypre_ParCSRMatrix *S, NALU_HYPRE_BigInt *num_cpts_global, NALU_HYPRE_Int num_functions, NALU_HYPRE_Int *dof_func,
                                         NALU_HYPRE_Int debug_flag, NALU_HYPRE_Real trunc_factor, NALU_HYPRE_Int num_smooth, NALU_HYPRE_Real *SmoothVecs,
                                         hypre_ParCSRMatrix **P_ptr );
NALU_HYPRE_Int hypre_BoomerAMGBuildInterpGSMG ( hypre_ParCSRMatrix *A, NALU_HYPRE_Int *CF_marker,
                                           hypre_ParCSRMatrix *S, NALU_HYPRE_BigInt *num_cpts_global, NALU_HYPRE_Int num_functions, NALU_HYPRE_Int *dof_func,
                                           NALU_HYPRE_Int debug_flag, NALU_HYPRE_Real trunc_factor, hypre_ParCSRMatrix **P_ptr );

/* par_indepset.c */
NALU_HYPRE_Int hypre_BoomerAMGIndepSetInit ( hypre_ParCSRMatrix *S, NALU_HYPRE_Real *measure_array,
                                        NALU_HYPRE_Int seq_rand );
NALU_HYPRE_Int hypre_BoomerAMGIndepSet ( hypre_ParCSRMatrix *S, NALU_HYPRE_Real *measure_array,
                                    NALU_HYPRE_Int *graph_array, NALU_HYPRE_Int graph_array_size, NALU_HYPRE_Int *graph_array_offd,
                                    NALU_HYPRE_Int graph_array_offd_size, NALU_HYPRE_Int *IS_marker, NALU_HYPRE_Int *IS_marker_offd );

NALU_HYPRE_Int hypre_BoomerAMGIndepSetInitDevice( hypre_ParCSRMatrix *S, NALU_HYPRE_Real *measure_array,
                                             NALU_HYPRE_Int aug_rand);

NALU_HYPRE_Int hypre_BoomerAMGIndepSetDevice( hypre_ParCSRMatrix *S, NALU_HYPRE_Real *measure_diag,
                                         NALU_HYPRE_Real *measure_offd, NALU_HYPRE_Int graph_diag_size, NALU_HYPRE_Int *graph_diag,
                                         NALU_HYPRE_Int *IS_marker_diag, NALU_HYPRE_Int *IS_marker_offd, hypre_ParCSRCommPkg *comm_pkg,
                                         NALU_HYPRE_Int *int_send_buf );

/* par_interp.c */
NALU_HYPRE_Int hypre_BoomerAMGBuildInterp ( hypre_ParCSRMatrix *A, NALU_HYPRE_Int *CF_marker,
                                       hypre_ParCSRMatrix *S, NALU_HYPRE_BigInt *num_cpts_global, NALU_HYPRE_Int num_functions, NALU_HYPRE_Int *dof_func,
                                       NALU_HYPRE_Int debug_flag, NALU_HYPRE_Real trunc_factor, NALU_HYPRE_Int max_elmts, hypre_ParCSRMatrix **P_ptr );
NALU_HYPRE_Int hypre_BoomerAMGBuildInterpHE ( hypre_ParCSRMatrix *A, NALU_HYPRE_Int *CF_marker,
                                         hypre_ParCSRMatrix *S, NALU_HYPRE_BigInt *num_cpts_global, NALU_HYPRE_Int num_functions, NALU_HYPRE_Int *dof_func,
                                         NALU_HYPRE_Int debug_flag, NALU_HYPRE_Real trunc_factor, NALU_HYPRE_Int max_elmts, hypre_ParCSRMatrix **P_ptr );
NALU_HYPRE_Int hypre_BoomerAMGBuildDirInterp ( hypre_ParCSRMatrix *A, NALU_HYPRE_Int *CF_marker,
                                          hypre_ParCSRMatrix *S, NALU_HYPRE_BigInt *num_cpts_global, NALU_HYPRE_Int num_functions, NALU_HYPRE_Int *dof_func,
                                          NALU_HYPRE_Int debug_flag, NALU_HYPRE_Real trunc_factor, NALU_HYPRE_Int max_elmts, NALU_HYPRE_Int interp_type,
                                          hypre_ParCSRMatrix **P_ptr );
NALU_HYPRE_Int hypre_BoomerAMGBuildDirInterpDevice( hypre_ParCSRMatrix *A, NALU_HYPRE_Int *CF_marker,
                                               hypre_ParCSRMatrix *S, NALU_HYPRE_BigInt *num_cpts_global, NALU_HYPRE_Int num_functions, NALU_HYPRE_Int *dof_func,
                                               NALU_HYPRE_Int debug_flag, NALU_HYPRE_Real trunc_factor, NALU_HYPRE_Int max_elmts, NALU_HYPRE_Int interp_type,
                                               hypre_ParCSRMatrix **P_ptr );

NALU_HYPRE_Int hypre_BoomerAMGInterpTruncation ( hypre_ParCSRMatrix *P, NALU_HYPRE_Real trunc_factor,
                                            NALU_HYPRE_Int max_elmts );
NALU_HYPRE_Int hypre_BoomerAMGInterpTruncationDevice( hypre_ParCSRMatrix *P, NALU_HYPRE_Real trunc_factor,
                                                 NALU_HYPRE_Int max_elmts );

NALU_HYPRE_Int hypre_BoomerAMGBuildInterpModUnk ( hypre_ParCSRMatrix *A, NALU_HYPRE_Int *CF_marker,
                                             hypre_ParCSRMatrix *S, NALU_HYPRE_BigInt *num_cpts_global, NALU_HYPRE_Int num_functions, NALU_HYPRE_Int *dof_func,
                                             NALU_HYPRE_Int debug_flag, NALU_HYPRE_Real trunc_factor, NALU_HYPRE_Int max_elmts, hypre_ParCSRMatrix **P_ptr );
NALU_HYPRE_Int hypre_BoomerAMGTruncandBuild ( hypre_ParCSRMatrix *P, NALU_HYPRE_Real trunc_factor,
                                         NALU_HYPRE_Int max_elmts );
hypre_ParCSRMatrix *hypre_CreateC ( hypre_ParCSRMatrix *A, NALU_HYPRE_Real w );

NALU_HYPRE_Int hypre_BoomerAMGBuildInterpOnePntHost( hypre_ParCSRMatrix  *A, NALU_HYPRE_Int *CF_marker,
                                                hypre_ParCSRMatrix *S, NALU_HYPRE_BigInt *num_cpts_global, NALU_HYPRE_Int num_functions, NALU_HYPRE_Int *dof_func,
                                                NALU_HYPRE_Int debug_flag, hypre_ParCSRMatrix **P_ptr);
NALU_HYPRE_Int hypre_BoomerAMGBuildInterpOnePntDevice( hypre_ParCSRMatrix  *A, NALU_HYPRE_Int *CF_marker,
                                                  hypre_ParCSRMatrix *S, NALU_HYPRE_BigInt *num_cpts_global, NALU_HYPRE_Int num_functions, NALU_HYPRE_Int *dof_func,
                                                  NALU_HYPRE_Int debug_flag, hypre_ParCSRMatrix **P_ptr);
NALU_HYPRE_Int hypre_BoomerAMGBuildInterpOnePnt( hypre_ParCSRMatrix  *A, NALU_HYPRE_Int *CF_marker,
                                            hypre_ParCSRMatrix *S, NALU_HYPRE_BigInt *num_cpts_global, NALU_HYPRE_Int num_functions, NALU_HYPRE_Int *dof_func,
                                            NALU_HYPRE_Int debug_flag, hypre_ParCSRMatrix **P_ptr);

/* par_jacobi_interp.c */
void hypre_BoomerAMGJacobiInterp ( hypre_ParCSRMatrix *A, hypre_ParCSRMatrix **P,
                                   hypre_ParCSRMatrix *S, NALU_HYPRE_Int num_functions, NALU_HYPRE_Int *dof_func, NALU_HYPRE_Int *CF_marker,
                                   NALU_HYPRE_Int level, NALU_HYPRE_Real truncation_threshold, NALU_HYPRE_Real truncation_threshold_minus );
void hypre_BoomerAMGJacobiInterp_1 ( hypre_ParCSRMatrix *A, hypre_ParCSRMatrix **P,
                                     hypre_ParCSRMatrix *S, NALU_HYPRE_Int *CF_marker, NALU_HYPRE_Int level, NALU_HYPRE_Real truncation_threshold,
                                     NALU_HYPRE_Real truncation_threshold_minus, NALU_HYPRE_Int *dof_func, NALU_HYPRE_Int *dof_func_offd,
                                     NALU_HYPRE_Real weight_AF );
void hypre_BoomerAMGTruncateInterp ( hypre_ParCSRMatrix *P, NALU_HYPRE_Real eps, NALU_HYPRE_Real dlt,
                                     NALU_HYPRE_Int *CF_marker );
NALU_HYPRE_Int hypre_ParCSRMatrix_dof_func_offd ( hypre_ParCSRMatrix *A, NALU_HYPRE_Int num_functions,
                                             NALU_HYPRE_Int *dof_func, NALU_HYPRE_Int **dof_func_offd );

/* par_laplace_27pt.c */
NALU_HYPRE_ParCSRMatrix GenerateLaplacian27pt ( MPI_Comm comm, NALU_HYPRE_BigInt nx, NALU_HYPRE_BigInt ny,
                                           NALU_HYPRE_BigInt nz, NALU_HYPRE_Int P, NALU_HYPRE_Int Q, NALU_HYPRE_Int R, NALU_HYPRE_Int p, NALU_HYPRE_Int q, NALU_HYPRE_Int r,
                                           NALU_HYPRE_Real *value );
NALU_HYPRE_Int hypre_map3 ( NALU_HYPRE_BigInt ix, NALU_HYPRE_BigInt iy, NALU_HYPRE_BigInt iz, NALU_HYPRE_Int p, NALU_HYPRE_Int q,
                       NALU_HYPRE_Int r, NALU_HYPRE_Int P, NALU_HYPRE_Int Q, NALU_HYPRE_Int R, NALU_HYPRE_BigInt *nx_part, NALU_HYPRE_BigInt *ny_part,
                       NALU_HYPRE_BigInt *nz_part );

/* par_laplace_9pt.c */
NALU_HYPRE_ParCSRMatrix GenerateLaplacian9pt ( MPI_Comm comm, NALU_HYPRE_BigInt nx, NALU_HYPRE_BigInt ny,
                                          NALU_HYPRE_Int P, NALU_HYPRE_Int Q, NALU_HYPRE_Int p, NALU_HYPRE_Int q, NALU_HYPRE_Real *value );
NALU_HYPRE_BigInt hypre_map2 ( NALU_HYPRE_BigInt ix, NALU_HYPRE_BigInt iy, NALU_HYPRE_Int p, NALU_HYPRE_Int q,
                          NALU_HYPRE_BigInt nx, NALU_HYPRE_BigInt *nx_part, NALU_HYPRE_BigInt *ny_part );

/* par_laplace.c */
NALU_HYPRE_ParCSRMatrix GenerateLaplacian ( MPI_Comm comm, NALU_HYPRE_BigInt ix, NALU_HYPRE_BigInt ny,
                                       NALU_HYPRE_BigInt nz, NALU_HYPRE_Int P, NALU_HYPRE_Int Q, NALU_HYPRE_Int R, NALU_HYPRE_Int p, NALU_HYPRE_Int q, NALU_HYPRE_Int r,
                                       NALU_HYPRE_Real *value );
NALU_HYPRE_BigInt hypre_map ( NALU_HYPRE_BigInt ix, NALU_HYPRE_BigInt iy, NALU_HYPRE_BigInt iz, NALU_HYPRE_Int p,
                         NALU_HYPRE_Int q, NALU_HYPRE_Int r, NALU_HYPRE_BigInt nx, NALU_HYPRE_BigInt ny, NALU_HYPRE_BigInt *nx_part,
                         NALU_HYPRE_BigInt *ny_part, NALU_HYPRE_BigInt *nz_part );
NALU_HYPRE_ParCSRMatrix GenerateSysLaplacian ( MPI_Comm comm, NALU_HYPRE_BigInt nx, NALU_HYPRE_BigInt ny,
                                          NALU_HYPRE_BigInt nz, NALU_HYPRE_Int P, NALU_HYPRE_Int Q, NALU_HYPRE_Int R, NALU_HYPRE_Int p, NALU_HYPRE_Int q, NALU_HYPRE_Int r,
                                          NALU_HYPRE_Int num_fun, NALU_HYPRE_Real *mtrx, NALU_HYPRE_Real *value );
NALU_HYPRE_ParCSRMatrix GenerateSysLaplacianVCoef ( MPI_Comm comm, NALU_HYPRE_BigInt nx, NALU_HYPRE_BigInt ny,
                                               NALU_HYPRE_BigInt nz, NALU_HYPRE_Int P, NALU_HYPRE_Int Q, NALU_HYPRE_Int R, NALU_HYPRE_Int p, NALU_HYPRE_Int q, NALU_HYPRE_Int r,
                                               NALU_HYPRE_Int num_fun, NALU_HYPRE_Real *mtrx, NALU_HYPRE_Real *value );

/* par_lr_interp.c */
NALU_HYPRE_Int hypre_BoomerAMGBuildStdInterp ( hypre_ParCSRMatrix *A, NALU_HYPRE_Int *CF_marker,
                                          hypre_ParCSRMatrix *S, NALU_HYPRE_BigInt *num_cpts_global, NALU_HYPRE_Int num_functions, NALU_HYPRE_Int *dof_func,
                                          NALU_HYPRE_Int debug_flag, NALU_HYPRE_Real trunc_factor, NALU_HYPRE_Int max_elmts, NALU_HYPRE_Int sep_weight,
                                          hypre_ParCSRMatrix **P_ptr );
NALU_HYPRE_Int hypre_BoomerAMGBuildExtPIInterp ( hypre_ParCSRMatrix *A, NALU_HYPRE_Int *CF_marker,
                                            hypre_ParCSRMatrix *S, NALU_HYPRE_BigInt *num_cpts_global, NALU_HYPRE_Int num_functions, NALU_HYPRE_Int *dof_func,
                                            NALU_HYPRE_Int debug_flag, NALU_HYPRE_Real trunc_factor, NALU_HYPRE_Int max_elmts, hypre_ParCSRMatrix **P_ptr );
NALU_HYPRE_Int hypre_BoomerAMGBuildExtPIInterpHost ( hypre_ParCSRMatrix *A, NALU_HYPRE_Int *CF_marker,
                                                hypre_ParCSRMatrix *S, NALU_HYPRE_BigInt *num_cpts_global, NALU_HYPRE_Int num_functions, NALU_HYPRE_Int *dof_func,
                                                NALU_HYPRE_Int debug_flag, NALU_HYPRE_Real trunc_factor, NALU_HYPRE_Int max_elmts, hypre_ParCSRMatrix **P_ptr );
NALU_HYPRE_Int hypre_BoomerAMGBuildExtPICCInterp ( hypre_ParCSRMatrix *A, NALU_HYPRE_Int *CF_marker,
                                              hypre_ParCSRMatrix *S, NALU_HYPRE_BigInt *num_cpts_global, NALU_HYPRE_Int num_functions, NALU_HYPRE_Int *dof_func,
                                              NALU_HYPRE_Int debug_flag, NALU_HYPRE_Real trunc_factor, NALU_HYPRE_Int max_elmts, hypre_ParCSRMatrix **P_ptr );
NALU_HYPRE_Int hypre_BoomerAMGBuildFFInterp ( hypre_ParCSRMatrix *A, NALU_HYPRE_Int *CF_marker,
                                         hypre_ParCSRMatrix *S, NALU_HYPRE_BigInt *num_cpts_global, NALU_HYPRE_Int num_functions, NALU_HYPRE_Int *dof_func,
                                         NALU_HYPRE_Int debug_flag, NALU_HYPRE_Real trunc_factor, NALU_HYPRE_Int max_elmts, hypre_ParCSRMatrix **P_ptr );
NALU_HYPRE_Int hypre_BoomerAMGBuildFF1Interp ( hypre_ParCSRMatrix *A, NALU_HYPRE_Int *CF_marker,
                                          hypre_ParCSRMatrix *S, NALU_HYPRE_BigInt *num_cpts_global, NALU_HYPRE_Int num_functions, NALU_HYPRE_Int *dof_func,
                                          NALU_HYPRE_Int debug_flag, NALU_HYPRE_Real trunc_factor, NALU_HYPRE_Int max_elmts, hypre_ParCSRMatrix **P_ptr );
NALU_HYPRE_Int hypre_BoomerAMGBuildExtInterp ( hypre_ParCSRMatrix *A, NALU_HYPRE_Int *CF_marker,
                                          hypre_ParCSRMatrix *S, NALU_HYPRE_BigInt *num_cpts_global, NALU_HYPRE_Int num_functions, NALU_HYPRE_Int *dof_func,
                                          NALU_HYPRE_Int debug_flag, NALU_HYPRE_Real trunc_factor, NALU_HYPRE_Int max_elmts, hypre_ParCSRMatrix **P_ptr );

/* par_lr_interp_device.c */
NALU_HYPRE_Int hypre_BoomerAMGBuildExtInterpDevice(hypre_ParCSRMatrix *A, NALU_HYPRE_Int *CF_marker,
                                              hypre_ParCSRMatrix   *S, NALU_HYPRE_BigInt *num_cpts_global, NALU_HYPRE_Int num_functions,
                                              NALU_HYPRE_Int *dof_func, NALU_HYPRE_Int debug_flag, NALU_HYPRE_Real trunc_factor, NALU_HYPRE_Int max_elmts,
                                              hypre_ParCSRMatrix  **P_ptr);
NALU_HYPRE_Int hypre_BoomerAMGBuildExtPIInterpDevice( hypre_ParCSRMatrix *A, NALU_HYPRE_Int *CF_marker,
                                                 hypre_ParCSRMatrix *S, NALU_HYPRE_BigInt *num_cpts_global, NALU_HYPRE_Int num_functions, NALU_HYPRE_Int *dof_func,
                                                 NALU_HYPRE_Int debug_flag, NALU_HYPRE_Real trunc_factor, NALU_HYPRE_Int max_elmts, hypre_ParCSRMatrix **P_ptr);
NALU_HYPRE_Int hypre_BoomerAMGBuildExtPEInterpDevice(hypre_ParCSRMatrix *A, NALU_HYPRE_Int *CF_marker,
                                                hypre_ParCSRMatrix   *S, NALU_HYPRE_BigInt *num_cpts_global, NALU_HYPRE_Int num_functions,
                                                NALU_HYPRE_Int *dof_func, NALU_HYPRE_Int debug_flag, NALU_HYPRE_Real trunc_factor, NALU_HYPRE_Int max_elmts,
                                                hypre_ParCSRMatrix  **P_ptr);

/* par_mod_lr_interp.c */
NALU_HYPRE_Int hypre_BoomerAMGBuildModExtInterp(hypre_ParCSRMatrix *A, NALU_HYPRE_Int *CF_marker,
                                           hypre_ParCSRMatrix *S, NALU_HYPRE_BigInt *num_cpts_global, NALU_HYPRE_Int num_functions, NALU_HYPRE_Int *dof_func,
                                           NALU_HYPRE_Int debug_flag, NALU_HYPRE_Real trunc_factor, NALU_HYPRE_Int max_elmts, hypre_ParCSRMatrix  **P_ptr);
NALU_HYPRE_Int hypre_BoomerAMGBuildModExtPIInterp(hypre_ParCSRMatrix *A, NALU_HYPRE_Int *CF_marker,
                                             hypre_ParCSRMatrix *S, NALU_HYPRE_BigInt *num_cpts_global, NALU_HYPRE_Int num_functions, NALU_HYPRE_Int *dof_func,
                                             NALU_HYPRE_Int debug_flag, NALU_HYPRE_Real trunc_factor, NALU_HYPRE_Int max_elmts, hypre_ParCSRMatrix  **P_ptr);
NALU_HYPRE_Int hypre_BoomerAMGBuildModExtPEInterp(hypre_ParCSRMatrix *A, NALU_HYPRE_Int *CF_marker,
                                             hypre_ParCSRMatrix *S, NALU_HYPRE_BigInt *num_cpts_global, NALU_HYPRE_Int num_functions, NALU_HYPRE_Int *dof_func,
                                             NALU_HYPRE_Int debug_flag, NALU_HYPRE_Real trunc_factor, NALU_HYPRE_Int max_elmts, hypre_ParCSRMatrix **P_ptr);

/* par_2s_interp.c */
NALU_HYPRE_Int hypre_BoomerAMGBuildModPartialExtInterpHost ( hypre_ParCSRMatrix *A, NALU_HYPRE_Int *CF_marker,
                                                        hypre_ParCSRMatrix *S, NALU_HYPRE_BigInt *num_cpts_global, NALU_HYPRE_BigInt *num_old_cpts_global,
                                                        NALU_HYPRE_Int num_functions, NALU_HYPRE_Int *dof_func, NALU_HYPRE_Int debug_flag, NALU_HYPRE_Real trunc_factor,
                                                        NALU_HYPRE_Int max_elmts, hypre_ParCSRMatrix **P_ptr );
NALU_HYPRE_Int hypre_BoomerAMGBuildModPartialExtInterpDevice ( hypre_ParCSRMatrix *A,
                                                          NALU_HYPRE_Int *CF_marker, hypre_ParCSRMatrix *S, NALU_HYPRE_BigInt *num_cpts_global,
                                                          NALU_HYPRE_BigInt *num_old_cpts_global, NALU_HYPRE_Int debug_flag, NALU_HYPRE_Real trunc_factor,
                                                          NALU_HYPRE_Int max_elmts, hypre_ParCSRMatrix **P_ptr );
NALU_HYPRE_Int hypre_BoomerAMGBuildModPartialExtInterp ( hypre_ParCSRMatrix *A, NALU_HYPRE_Int *CF_marker,
                                                    hypre_ParCSRMatrix *S, NALU_HYPRE_BigInt *num_cpts_global, NALU_HYPRE_BigInt *num_old_cpts_global,
                                                    NALU_HYPRE_Int num_functions, NALU_HYPRE_Int *dof_func, NALU_HYPRE_Int debug_flag, NALU_HYPRE_Real trunc_factor,
                                                    NALU_HYPRE_Int max_elmts, hypre_ParCSRMatrix **P_ptr );
NALU_HYPRE_Int hypre_BoomerAMGBuildModPartialExtPEInterpHost ( hypre_ParCSRMatrix *A,
                                                          NALU_HYPRE_Int *CF_marker, hypre_ParCSRMatrix *S, NALU_HYPRE_BigInt *num_cpts_global,
                                                          NALU_HYPRE_BigInt *num_old_cpts_global, NALU_HYPRE_Int num_functions, NALU_HYPRE_Int *dof_func,
                                                          NALU_HYPRE_Int debug_flag, NALU_HYPRE_Real trunc_factor, NALU_HYPRE_Int max_elmts, hypre_ParCSRMatrix **P_ptr );
NALU_HYPRE_Int hypre_BoomerAMGBuildModPartialExtPEInterpDevice ( hypre_ParCSRMatrix *A,
                                                            NALU_HYPRE_Int *CF_marker, hypre_ParCSRMatrix *S, NALU_HYPRE_BigInt *num_cpts_global,
                                                            NALU_HYPRE_BigInt *num_old_cpts_global, NALU_HYPRE_Int debug_flag, NALU_HYPRE_Real trunc_factor,
                                                            NALU_HYPRE_Int max_elmts, hypre_ParCSRMatrix **P_ptr );
NALU_HYPRE_Int hypre_BoomerAMGBuildModPartialExtPEInterp ( hypre_ParCSRMatrix *A, NALU_HYPRE_Int *CF_marker,
                                                      hypre_ParCSRMatrix *S, NALU_HYPRE_BigInt *num_cpts_global, NALU_HYPRE_BigInt *num_old_cpts_global,
                                                      NALU_HYPRE_Int num_functions, NALU_HYPRE_Int *dof_func, NALU_HYPRE_Int debug_flag, NALU_HYPRE_Real trunc_factor,
                                                      NALU_HYPRE_Int max_elmts, hypre_ParCSRMatrix **P_ptr );

/* par_mod_multi_interp.c */
NALU_HYPRE_Int hypre_BoomerAMGBuildModMultipass ( hypre_ParCSRMatrix *A, NALU_HYPRE_Int *CF_marker,
                                             hypre_ParCSRMatrix *S, NALU_HYPRE_BigInt *num_cpts_global, NALU_HYPRE_Real trunc_factor,
                                             NALU_HYPRE_Int P_max_elmts, NALU_HYPRE_Int interp_type, NALU_HYPRE_Int num_functions, NALU_HYPRE_Int *dof_func,
                                             hypre_ParCSRMatrix **P_ptr );
NALU_HYPRE_Int hypre_BoomerAMGBuildModMultipassHost ( hypre_ParCSRMatrix *A, NALU_HYPRE_Int *CF_marker,
                                                 hypre_ParCSRMatrix *S, NALU_HYPRE_BigInt *num_cpts_global, NALU_HYPRE_Real trunc_factor,
                                                 NALU_HYPRE_Int P_max_elmts, NALU_HYPRE_Int interp_type, NALU_HYPRE_Int num_functions, NALU_HYPRE_Int *dof_func,
                                                 hypre_ParCSRMatrix **P_ptr );
NALU_HYPRE_Int hypre_GenerateMultipassPi ( hypre_ParCSRMatrix *A, hypre_ParCSRMatrix *S,
                                      NALU_HYPRE_BigInt *c_pts_starts, NALU_HYPRE_Int *pass_order, NALU_HYPRE_Int *pass_marker,
                                      NALU_HYPRE_Int *pass_marker_offd, NALU_HYPRE_Int num_points, NALU_HYPRE_Int color, NALU_HYPRE_Real *row_sums,
                                      hypre_ParCSRMatrix **P_ptr );
NALU_HYPRE_Int hypre_GenerateMultiPi ( hypre_ParCSRMatrix *A, hypre_ParCSRMatrix *S,
                                  hypre_ParCSRMatrix *P, NALU_HYPRE_BigInt *c_pts_starts, NALU_HYPRE_Int *pass_order, NALU_HYPRE_Int *pass_marker,
                                  NALU_HYPRE_Int *pass_marker_offd, NALU_HYPRE_Int num_points, NALU_HYPRE_Int color, NALU_HYPRE_Int num_functions,
                                  NALU_HYPRE_Int *dof_func, NALU_HYPRE_Int *dof_func_offd, hypre_ParCSRMatrix **Pi_ptr );
NALU_HYPRE_Int hypre_BoomerAMGBuildModMultipassDevice ( hypre_ParCSRMatrix *A, NALU_HYPRE_Int *CF_marker,
                                                   hypre_ParCSRMatrix *S, NALU_HYPRE_BigInt *num_cpts_global, NALU_HYPRE_Real trunc_factor,
                                                   NALU_HYPRE_Int P_max_elmts, NALU_HYPRE_Int interp_type, NALU_HYPRE_Int num_functions, NALU_HYPRE_Int *dof_func,
                                                   hypre_ParCSRMatrix **P_ptr );
NALU_HYPRE_Int hypre_GenerateMultipassPiDevice ( hypre_ParCSRMatrix *A, hypre_ParCSRMatrix *S,
                                            NALU_HYPRE_BigInt *c_pts_starts, NALU_HYPRE_Int *pass_order, NALU_HYPRE_Int *pass_marker,
                                            NALU_HYPRE_Int *pass_marker_offd, NALU_HYPRE_Int num_points, NALU_HYPRE_Int color, NALU_HYPRE_Real *row_sums,
                                            hypre_ParCSRMatrix **P_ptr );
NALU_HYPRE_Int hypre_GenerateMultiPiDevice ( hypre_ParCSRMatrix *A, hypre_ParCSRMatrix *S,
                                        hypre_ParCSRMatrix *P, NALU_HYPRE_BigInt *c_pts_starts, NALU_HYPRE_Int *pass_order, NALU_HYPRE_Int *pass_marker,
                                        NALU_HYPRE_Int *pass_marker_offd, NALU_HYPRE_Int num_points, NALU_HYPRE_Int color, NALU_HYPRE_Int num_functions,
                                        NALU_HYPRE_Int *dof_func, NALU_HYPRE_Int *dof_func_offd, hypre_ParCSRMatrix **Pi_ptr );

/* par_multi_interp.c */
NALU_HYPRE_Int hypre_BoomerAMGBuildMultipass ( hypre_ParCSRMatrix *A, NALU_HYPRE_Int *CF_marker,
                                          hypre_ParCSRMatrix *S, NALU_HYPRE_BigInt *num_cpts_global, NALU_HYPRE_Int num_functions, NALU_HYPRE_Int *dof_func,
                                          NALU_HYPRE_Int debug_flag, NALU_HYPRE_Real trunc_factor, NALU_HYPRE_Int P_max_elmts, NALU_HYPRE_Int weight_option,
                                          hypre_ParCSRMatrix **P_ptr );
NALU_HYPRE_Int hypre_BoomerAMGBuildMultipassHost ( hypre_ParCSRMatrix *A, NALU_HYPRE_Int *CF_marker,
                                              hypre_ParCSRMatrix *S, NALU_HYPRE_BigInt *num_cpts_global, NALU_HYPRE_Int num_functions, NALU_HYPRE_Int *dof_func,
                                              NALU_HYPRE_Int debug_flag, NALU_HYPRE_Real trunc_factor, NALU_HYPRE_Int P_max_elmts, NALU_HYPRE_Int weight_option,
                                              hypre_ParCSRMatrix **P_ptr );

/* par_nodal_systems.c */
NALU_HYPRE_Int hypre_BoomerAMGCreateNodalA ( hypre_ParCSRMatrix *A, NALU_HYPRE_Int num_functions,
                                        NALU_HYPRE_Int *dof_func, NALU_HYPRE_Int option, NALU_HYPRE_Int diag_option, hypre_ParCSRMatrix **AN_ptr );
NALU_HYPRE_Int hypre_BoomerAMGCreateScalarCFS ( hypre_ParCSRMatrix *SN, hypre_ParCSRMatrix *A,
                                           NALU_HYPRE_Int *CFN_marker, NALU_HYPRE_Int num_functions, NALU_HYPRE_Int nodal, NALU_HYPRE_Int keep_same_sign,
                                           hypre_IntArray **dof_func_ptr, hypre_IntArray **CF_marker_ptr, hypre_ParCSRMatrix **S_ptr );
NALU_HYPRE_Int hypre_BoomerAMGCreateScalarCF ( NALU_HYPRE_Int *CFN_marker, NALU_HYPRE_Int num_functions,
                                          NALU_HYPRE_Int num_nodes, hypre_IntArray **dof_func_ptr, hypre_IntArray **CF_marker_ptr );

/* par_nongalerkin.c */
NALU_HYPRE_Int hypre_GrabSubArray ( NALU_HYPRE_Int *indices, NALU_HYPRE_Int start, NALU_HYPRE_Int end,
                               NALU_HYPRE_BigInt *array, NALU_HYPRE_BigInt *output );
NALU_HYPRE_Int hypre_IntersectTwoArrays ( NALU_HYPRE_Int *x, NALU_HYPRE_Real *x_data, NALU_HYPRE_Int x_length,
                                     NALU_HYPRE_Int *y, NALU_HYPRE_Int y_length, NALU_HYPRE_Int *z, NALU_HYPRE_Real *output_x_data,
                                     NALU_HYPRE_Int *intersect_length );
NALU_HYPRE_Int hypre_IntersectTwoBigArrays ( NALU_HYPRE_BigInt *x, NALU_HYPRE_Real *x_data, NALU_HYPRE_Int x_length,
                                        NALU_HYPRE_BigInt *y, NALU_HYPRE_Int y_length, NALU_HYPRE_BigInt *z, NALU_HYPRE_Real *output_x_data,
                                        NALU_HYPRE_Int *intersect_length );
NALU_HYPRE_Int hypre_SortedCopyParCSRData ( hypre_ParCSRMatrix *A, hypre_ParCSRMatrix *B );
NALU_HYPRE_Int hypre_BoomerAMG_MyCreateS ( hypre_ParCSRMatrix *A, NALU_HYPRE_Real strength_threshold,
                                      NALU_HYPRE_Real max_row_sum, NALU_HYPRE_Int num_functions, NALU_HYPRE_Int *dof_func, hypre_ParCSRMatrix **S_ptr );
NALU_HYPRE_Int hypre_BoomerAMGCreateSFromCFMarker(hypre_ParCSRMatrix    *A,
                                             NALU_HYPRE_Real strength_threshold, NALU_HYPRE_Real max_row_sum, NALU_HYPRE_Int *CF_marker,
                                             NALU_HYPRE_Int num_functions, NALU_HYPRE_Int *dof_func, NALU_HYPRE_Int SMRK, hypre_ParCSRMatrix    **S_ptr);
NALU_HYPRE_Int hypre_NonGalerkinIJBufferInit ( NALU_HYPRE_Int *ijbuf_cnt, NALU_HYPRE_Int *ijbuf_rowcounter,
                                          NALU_HYPRE_Int *ijbuf_numcols );
NALU_HYPRE_Int hypre_NonGalerkinIJBigBufferInit ( NALU_HYPRE_Int *ijbuf_cnt, NALU_HYPRE_Int *ijbuf_rowcounter,
                                             NALU_HYPRE_BigInt *ijbuf_numcols );
NALU_HYPRE_Int hypre_NonGalerkinIJBufferNewRow ( NALU_HYPRE_BigInt *ijbuf_rownums, NALU_HYPRE_Int *ijbuf_numcols,
                                            NALU_HYPRE_Int *ijbuf_rowcounter, NALU_HYPRE_BigInt new_row );
NALU_HYPRE_Int hypre_NonGalerkinIJBufferCompressRow ( NALU_HYPRE_Int *ijbuf_cnt, NALU_HYPRE_Int ijbuf_rowcounter,
                                                 NALU_HYPRE_Real *ijbuf_data, NALU_HYPRE_BigInt *ijbuf_cols, NALU_HYPRE_BigInt *ijbuf_rownums,
                                                 NALU_HYPRE_Int *ijbuf_numcols );
NALU_HYPRE_Int hypre_NonGalerkinIJBufferCompress ( NALU_HYPRE_MemoryLocation memory_location,
                                              NALU_HYPRE_Int ijbuf_size, NALU_HYPRE_Int *ijbuf_cnt,
                                              NALU_HYPRE_Int *ijbuf_rowcounter, NALU_HYPRE_Real **ijbuf_data, NALU_HYPRE_BigInt **ijbuf_cols,
                                              NALU_HYPRE_BigInt **ijbuf_rownums, NALU_HYPRE_Int **ijbuf_numcols );
NALU_HYPRE_Int hypre_NonGalerkinIJBufferWrite ( NALU_HYPRE_IJMatrix B, NALU_HYPRE_Int *ijbuf_cnt,
                                           NALU_HYPRE_Int ijbuf_size, NALU_HYPRE_Int *ijbuf_rowcounter, NALU_HYPRE_Real **ijbuf_data,
                                           NALU_HYPRE_BigInt **ijbuf_cols, NALU_HYPRE_BigInt **ijbuf_rownums, NALU_HYPRE_Int **ijbuf_numcols,
                                           NALU_HYPRE_BigInt row_to_write, NALU_HYPRE_BigInt col_to_write, NALU_HYPRE_Real val_to_write );
NALU_HYPRE_Int hypre_NonGalerkinIJBufferEmpty ( NALU_HYPRE_IJMatrix B, NALU_HYPRE_Int ijbuf_size,
                                           NALU_HYPRE_Int *ijbuf_cnt, NALU_HYPRE_Int ijbuf_rowcounter, NALU_HYPRE_Real **ijbuf_data,
                                           NALU_HYPRE_BigInt **ijbuf_cols, NALU_HYPRE_BigInt **ijbuf_rownums, NALU_HYPRE_Int **ijbuf_numcols );
hypre_ParCSRMatrix * hypre_NonGalerkinSparsityPattern(hypre_ParCSRMatrix *R_IAP,
                                                      hypre_ParCSRMatrix *RAP, NALU_HYPRE_Int * CF_marker, NALU_HYPRE_Real droptol, NALU_HYPRE_Int sym_collapse,
                                                      NALU_HYPRE_Int collapse_beta );
NALU_HYPRE_Int hypre_BoomerAMGBuildNonGalerkinCoarseOperator( hypre_ParCSRMatrix **RAP_ptr,
                                                         hypre_ParCSRMatrix *AP, NALU_HYPRE_Real strong_threshold, NALU_HYPRE_Real max_row_sum,
                                                         NALU_HYPRE_Int num_functions, NALU_HYPRE_Int * dof_func_value, NALU_HYPRE_Int * CF_marker, NALU_HYPRE_Real droptol,
                                                         NALU_HYPRE_Int sym_collapse, NALU_HYPRE_Real lump_percent, NALU_HYPRE_Int collapse_beta );

/* par_rap.c */
NALU_HYPRE_Int hypre_BoomerAMGBuildCoarseOperator ( hypre_ParCSRMatrix *RT, hypre_ParCSRMatrix *A,
                                               hypre_ParCSRMatrix *P, hypre_ParCSRMatrix **RAP_ptr );
NALU_HYPRE_Int hypre_BoomerAMGBuildCoarseOperatorKT ( hypre_ParCSRMatrix *RT, hypre_ParCSRMatrix *A,
                                                 hypre_ParCSRMatrix *P, NALU_HYPRE_Int keepTranspose, hypre_ParCSRMatrix **RAP_ptr );

/* par_rap_communication.c */
NALU_HYPRE_Int hypre_GetCommPkgRTFromCommPkgA ( hypre_ParCSRMatrix *RT, hypre_ParCSRMatrix *A,
                                           NALU_HYPRE_Int *fine_to_coarse, NALU_HYPRE_Int *tmp_map_offd );
NALU_HYPRE_Int hypre_GenerateSendMapAndCommPkg ( MPI_Comm comm, NALU_HYPRE_Int num_sends, NALU_HYPRE_Int num_recvs,
                                            NALU_HYPRE_Int *recv_procs, NALU_HYPRE_Int *send_procs, NALU_HYPRE_Int *recv_vec_starts, hypre_ParCSRMatrix *A );

/* par_relax.c */
NALU_HYPRE_Int hypre_BoomerAMGRelax ( hypre_ParCSRMatrix *A, hypre_ParVector *f, NALU_HYPRE_Int *cf_marker,
                                 NALU_HYPRE_Int relax_type, NALU_HYPRE_Int relax_points, NALU_HYPRE_Real relax_weight, NALU_HYPRE_Real omega,
                                 NALU_HYPRE_Real *l1_norms, hypre_ParVector *u, hypre_ParVector *Vtemp, hypre_ParVector *Ztemp );
NALU_HYPRE_Int hypre_GaussElimSetup ( hypre_ParAMGData *amg_data, NALU_HYPRE_Int level,
                                 NALU_HYPRE_Int relax_type );
NALU_HYPRE_Int hypre_GaussElimSolve ( hypre_ParAMGData *amg_data, NALU_HYPRE_Int level,
                                 NALU_HYPRE_Int relax_type );
NALU_HYPRE_Int hypre_BoomerAMGRelaxHybridGaussSeidel_core( hypre_ParCSRMatrix *A, hypre_ParVector *f,
                                                      NALU_HYPRE_Int *cf_marker, NALU_HYPRE_Int relax_points, NALU_HYPRE_Real relax_weight, NALU_HYPRE_Real omega,
                                                      NALU_HYPRE_Real *l1_norms, hypre_ParVector *u, hypre_ParVector *Vtemp, hypre_ParVector *Ztemp,
                                                      NALU_HYPRE_Int GS_order, NALU_HYPRE_Int Symm, NALU_HYPRE_Int Skip_diag, NALU_HYPRE_Int forced_seq,
                                                      NALU_HYPRE_Int Topo_order );
NALU_HYPRE_Int hypre_BoomerAMGRelax0WeightedJacobi( hypre_ParCSRMatrix *A, hypre_ParVector *f,
                                               NALU_HYPRE_Int *cf_marker, NALU_HYPRE_Int relax_points, NALU_HYPRE_Real relax_weight, hypre_ParVector *u,
                                               hypre_ParVector *Vtemp );

NALU_HYPRE_Int hypre_BoomerAMGRelaxHybridSOR( hypre_ParCSRMatrix *A, hypre_ParVector *f, NALU_HYPRE_Int *cf_marker, NALU_HYPRE_Int relax_points, NALU_HYPRE_Real relax_weight, NALU_HYPRE_Real omega, NALU_HYPRE_Real *l1_norms, hypre_ParVector *u, hypre_ParVector *Vtemp, hypre_ParVector *Ztemp, NALU_HYPRE_Int direction, NALU_HYPRE_Int symm, NALU_HYPRE_Int skip_diag, NALU_HYPRE_Int force_seq );

NALU_HYPRE_Int hypre_BoomerAMGRelax1GaussSeidel( hypre_ParCSRMatrix *A, hypre_ParVector *f,
                                            NALU_HYPRE_Int *cf_marker, NALU_HYPRE_Int relax_points, hypre_ParVector *u );

NALU_HYPRE_Int hypre_BoomerAMGRelax2GaussSeidel( hypre_ParCSRMatrix *A, hypre_ParVector *f,
                                            NALU_HYPRE_Int *cf_marker, NALU_HYPRE_Int relax_points, hypre_ParVector *u );

NALU_HYPRE_Int hypre_BoomerAMGRelax5ChaoticHybridGaussSeidel( hypre_ParCSRMatrix *A, hypre_ParVector *f,
                                                         NALU_HYPRE_Int *cf_marker, NALU_HYPRE_Int relax_points, hypre_ParVector *u );

NALU_HYPRE_Int hypre_BoomerAMGRelax3HybridGaussSeidel( hypre_ParCSRMatrix *A, hypre_ParVector *f,
                                                  NALU_HYPRE_Int *cf_marker, NALU_HYPRE_Int relax_points, NALU_HYPRE_Real relax_weight, NALU_HYPRE_Real omega,
                                                  hypre_ParVector *u, hypre_ParVector *Vtemp, hypre_ParVector *Ztemp );

NALU_HYPRE_Int hypre_BoomerAMGRelax4HybridGaussSeidel( hypre_ParCSRMatrix *A, hypre_ParVector *f,
                                                  NALU_HYPRE_Int *cf_marker, NALU_HYPRE_Int relax_points, NALU_HYPRE_Real relax_weight, NALU_HYPRE_Real omega,
                                                  hypre_ParVector *u, hypre_ParVector *Vtemp, hypre_ParVector *Ztemp );

NALU_HYPRE_Int hypre_BoomerAMGRelax6HybridSSOR( hypre_ParCSRMatrix *A, hypre_ParVector *f,
                                           NALU_HYPRE_Int *cf_marker, NALU_HYPRE_Int relax_points, NALU_HYPRE_Real relax_weight, NALU_HYPRE_Real omega,
                                           hypre_ParVector *u, hypre_ParVector *Vtemp, hypre_ParVector *Ztemp );

NALU_HYPRE_Int hypre_BoomerAMGRelax7Jacobi( hypre_ParCSRMatrix *A, hypre_ParVector *f,
                                       NALU_HYPRE_Int *cf_marker, NALU_HYPRE_Int relax_points, NALU_HYPRE_Real relax_weight, NALU_HYPRE_Real *l1_norms,
                                       hypre_ParVector *u, hypre_ParVector *Vtemp );

NALU_HYPRE_Int hypre_BoomerAMGRelax8HybridL1SSOR( hypre_ParCSRMatrix *A, hypre_ParVector *f,
                                             NALU_HYPRE_Int *cf_marker, NALU_HYPRE_Int relax_points, NALU_HYPRE_Real relax_weight, NALU_HYPRE_Real omega,
                                             NALU_HYPRE_Real *l1_norms, hypre_ParVector *u, hypre_ParVector *Vtemp, hypre_ParVector *Ztemp );

NALU_HYPRE_Int hypre_BoomerAMGRelax10TopoOrderedGaussSeidel( hypre_ParCSRMatrix *A, hypre_ParVector *f,
                                                        NALU_HYPRE_Int *cf_marker, NALU_HYPRE_Int relax_points, NALU_HYPRE_Real relax_weight, NALU_HYPRE_Real omega,
                                                        hypre_ParVector *u, hypre_ParVector *Vtemp, hypre_ParVector *Ztemp );

NALU_HYPRE_Int hypre_BoomerAMGRelax13HybridL1GaussSeidel( hypre_ParCSRMatrix *A, hypre_ParVector *f,
                                                     NALU_HYPRE_Int *cf_marker, NALU_HYPRE_Int relax_points, NALU_HYPRE_Real relax_weight, NALU_HYPRE_Real omega,
                                                     NALU_HYPRE_Real *l1_norms, hypre_ParVector *u, hypre_ParVector *Vtemp, hypre_ParVector *Ztemp );

NALU_HYPRE_Int hypre_BoomerAMGRelax14HybridL1GaussSeidel( hypre_ParCSRMatrix *A, hypre_ParVector *f,
                                                     NALU_HYPRE_Int *cf_marker, NALU_HYPRE_Int relax_points, NALU_HYPRE_Real relax_weight, NALU_HYPRE_Real omega,
                                                     NALU_HYPRE_Real *l1_norms, hypre_ParVector *u, hypre_ParVector *Vtemp, hypre_ParVector *Ztemp );

NALU_HYPRE_Int hypre_BoomerAMGRelax18WeightedL1Jacobi( hypre_ParCSRMatrix *A, hypre_ParVector *f,
                                                  NALU_HYPRE_Int *cf_marker, NALU_HYPRE_Int relax_points, NALU_HYPRE_Real relax_weight, NALU_HYPRE_Real *l1_norms,
                                                  hypre_ParVector *u, hypre_ParVector *Vtemp );

NALU_HYPRE_Int hypre_BoomerAMGRelax19GaussElim( hypre_ParCSRMatrix *A, hypre_ParVector *f,
                                           hypre_ParVector *u );

NALU_HYPRE_Int hypre_BoomerAMGRelax98GaussElimPivot( hypre_ParCSRMatrix *A, hypre_ParVector *f,
                                                hypre_ParVector *u );

NALU_HYPRE_Int hypre_BoomerAMGRelaxKaczmarz( hypre_ParCSRMatrix *A, hypre_ParVector *f, NALU_HYPRE_Real omega,
                                        NALU_HYPRE_Real *l1_norms, hypre_ParVector *u );

NALU_HYPRE_Int hypre_BoomerAMGRelaxTwoStageGaussSeidelDevice ( hypre_ParCSRMatrix *A, hypre_ParVector *f,
                                                          NALU_HYPRE_Real relax_weight, NALU_HYPRE_Real omega,
                                                          NALU_HYPRE_Real *A_diag_diag, hypre_ParVector *u,
                                                          hypre_ParVector *r, hypre_ParVector *z,
                                                          NALU_HYPRE_Int choice );

NALU_HYPRE_Int hypre_BoomerAMGRelax11TwoStageGaussSeidel( hypre_ParCSRMatrix *A, hypre_ParVector *f,
                                                     NALU_HYPRE_Int *cf_marker, NALU_HYPRE_Int relax_points,
                                                     NALU_HYPRE_Real relax_weight, NALU_HYPRE_Real omega,
                                                     NALU_HYPRE_Real *A_diag_diag, hypre_ParVector *u,
                                                     hypre_ParVector *Vtemp, hypre_ParVector *Ztemp );

NALU_HYPRE_Int hypre_BoomerAMGRelax12TwoStageGaussSeidel( hypre_ParCSRMatrix *A, hypre_ParVector *f,
                                                     NALU_HYPRE_Int *cf_marker, NALU_HYPRE_Int relax_points,
                                                     NALU_HYPRE_Real relax_weight, NALU_HYPRE_Real omega,
                                                     NALU_HYPRE_Real *A_diag_diag, hypre_ParVector *u,
                                                     hypre_ParVector *Vtemp, hypre_ParVector *Ztemp );

/* par_realx_device.c */
NALU_HYPRE_Int hypre_BoomerAMGRelaxHybridGaussSeidelDevice( hypre_ParCSRMatrix *A, hypre_ParVector *f,
                                                       NALU_HYPRE_Int *cf_marker, NALU_HYPRE_Int relax_points,
                                                       NALU_HYPRE_Real relax_weight, NALU_HYPRE_Real omega,
                                                       NALU_HYPRE_Real *l1_norms, hypre_ParVector *u,
                                                       hypre_ParVector *Vtemp, hypre_ParVector *Ztemp,
                                                       NALU_HYPRE_Int GS_order, NALU_HYPRE_Int Symm );

/* par_relax_interface.c */
NALU_HYPRE_Int hypre_BoomerAMGRelaxIF ( hypre_ParCSRMatrix *A, hypre_ParVector *f, NALU_HYPRE_Int *cf_marker,
                                   NALU_HYPRE_Int relax_type, NALU_HYPRE_Int relax_order, NALU_HYPRE_Int cycle_type, NALU_HYPRE_Real relax_weight,
                                   NALU_HYPRE_Real omega, NALU_HYPRE_Real *l1_norms, hypre_ParVector *u, hypre_ParVector *Vtemp,
                                   hypre_ParVector *Ztemp );

/* par_relax_more.c */
NALU_HYPRE_Int hypre_ParCSRMaxEigEstimate ( hypre_ParCSRMatrix *A, NALU_HYPRE_Int scale, NALU_HYPRE_Real *max_eig,
                                       NALU_HYPRE_Real *min_eig );
NALU_HYPRE_Int hypre_ParCSRMaxEigEstimateHost ( hypre_ParCSRMatrix *A, NALU_HYPRE_Int scale,
                                           NALU_HYPRE_Real *max_eig, NALU_HYPRE_Real *min_eig );
NALU_HYPRE_Int hypre_ParCSRMaxEigEstimateCG ( hypre_ParCSRMatrix *A, NALU_HYPRE_Int scale, NALU_HYPRE_Int max_iter,
                                         NALU_HYPRE_Real *max_eig, NALU_HYPRE_Real *min_eig );
NALU_HYPRE_Int hypre_ParCSRMaxEigEstimateCGHost ( hypre_ParCSRMatrix *A, NALU_HYPRE_Int scale,
                                             NALU_HYPRE_Int max_iter, NALU_HYPRE_Real *max_eig, NALU_HYPRE_Real *min_eig );
NALU_HYPRE_Int hypre_ParCSRRelax_Cheby ( hypre_ParCSRMatrix *A, hypre_ParVector *f, NALU_HYPRE_Real max_eig,
                                    NALU_HYPRE_Real min_eig, NALU_HYPRE_Real fraction, NALU_HYPRE_Int order, NALU_HYPRE_Int scale, NALU_HYPRE_Int variant,
                                    hypre_ParVector *u, hypre_ParVector *v, hypre_ParVector *r );
NALU_HYPRE_Int hypre_BoomerAMGRelax_FCFJacobi ( hypre_ParCSRMatrix *A, hypre_ParVector *f,
                                           NALU_HYPRE_Int *cf_marker, NALU_HYPRE_Real relax_weight, 
                                           hypre_ParVector *u, hypre_ParVector *Vtemp );
NALU_HYPRE_Int hypre_ParCSRRelax_CG ( NALU_HYPRE_Solver solver, hypre_ParCSRMatrix *A, hypre_ParVector *f,
                                 hypre_ParVector *u, NALU_HYPRE_Int num_its );
NALU_HYPRE_Int hypre_LINPACKcgtql1 ( NALU_HYPRE_Int *n, NALU_HYPRE_Real *d, NALU_HYPRE_Real *e, NALU_HYPRE_Int *ierr );
NALU_HYPRE_Real hypre_LINPACKcgpthy ( NALU_HYPRE_Real *a, NALU_HYPRE_Real *b );
NALU_HYPRE_Int hypre_ParCSRRelax_L1_Jacobi ( hypre_ParCSRMatrix *A, hypre_ParVector *f,
                                        NALU_HYPRE_Int *cf_marker, NALU_HYPRE_Int relax_points, NALU_HYPRE_Real relax_weight, NALU_HYPRE_Real *l1_norms,
                                        hypre_ParVector *u, hypre_ParVector *Vtemp );
NALU_HYPRE_Int hypre_LINPACKcgtql1(NALU_HYPRE_Int*, NALU_HYPRE_Real *, NALU_HYPRE_Real *, NALU_HYPRE_Int *);
NALU_HYPRE_Real hypre_LINPACKcgpthy(NALU_HYPRE_Real*, NALU_HYPRE_Real*);

/* par_relax_more_device.c */
NALU_HYPRE_Int hypre_ParCSRMaxEigEstimateDevice ( hypre_ParCSRMatrix *A, NALU_HYPRE_Int scale,
                                             NALU_HYPRE_Real *max_eig, NALU_HYPRE_Real *min_eig );
NALU_HYPRE_Int hypre_ParCSRMaxEigEstimateCGDevice ( hypre_ParCSRMatrix *A, NALU_HYPRE_Int scale,
                                               NALU_HYPRE_Int max_iter, NALU_HYPRE_Real *max_eig, NALU_HYPRE_Real *min_eig );

/* par_rotate_7pt.c */
NALU_HYPRE_ParCSRMatrix GenerateRotate7pt ( MPI_Comm comm, NALU_HYPRE_BigInt nx, NALU_HYPRE_BigInt ny, NALU_HYPRE_Int P,
                                       NALU_HYPRE_Int Q, NALU_HYPRE_Int p, NALU_HYPRE_Int q, NALU_HYPRE_Real alpha, NALU_HYPRE_Real eps );

/* par_scaled_matnorm.c */
NALU_HYPRE_Int hypre_ParCSRMatrixScaledNorm ( hypre_ParCSRMatrix *A, NALU_HYPRE_Real *scnorm );

/* par_schwarz.c */
void *hypre_SchwarzCreate ( void );
NALU_HYPRE_Int hypre_SchwarzDestroy ( void *data );
NALU_HYPRE_Int hypre_SchwarzSetup ( void *schwarz_vdata, hypre_ParCSRMatrix *A, hypre_ParVector *f,
                               hypre_ParVector *u );
NALU_HYPRE_Int hypre_SchwarzSolve ( void *schwarz_vdata, hypre_ParCSRMatrix *A, hypre_ParVector *f,
                               hypre_ParVector *u );
NALU_HYPRE_Int hypre_SchwarzCFSolve ( void *schwarz_vdata, hypre_ParCSRMatrix *A, hypre_ParVector *f,
                                 hypre_ParVector *u, NALU_HYPRE_Int *CF_marker, NALU_HYPRE_Int rlx_pt );
NALU_HYPRE_Int hypre_SchwarzSetVariant ( void *data, NALU_HYPRE_Int variant );
NALU_HYPRE_Int hypre_SchwarzSetDomainType ( void *data, NALU_HYPRE_Int domain_type );
NALU_HYPRE_Int hypre_SchwarzSetOverlap ( void *data, NALU_HYPRE_Int overlap );
NALU_HYPRE_Int hypre_SchwarzSetNumFunctions ( void *data, NALU_HYPRE_Int num_functions );
NALU_HYPRE_Int hypre_SchwarzSetNonSymm ( void *data, NALU_HYPRE_Int value );
NALU_HYPRE_Int hypre_SchwarzSetRelaxWeight ( void *data, NALU_HYPRE_Real relax_weight );
NALU_HYPRE_Int hypre_SchwarzSetDomainStructure ( void *data, hypre_CSRMatrix *domain_structure );
NALU_HYPRE_Int hypre_SchwarzSetScale ( void *data, NALU_HYPRE_Real *scale );
NALU_HYPRE_Int hypre_SchwarzReScale ( void *data, NALU_HYPRE_Int size, NALU_HYPRE_Real value );
NALU_HYPRE_Int hypre_SchwarzSetDofFunc ( void *data, NALU_HYPRE_Int *dof_func );

/* par_stats.c */
NALU_HYPRE_Int hypre_BoomerAMGSetupStats ( void *amg_vdata, hypre_ParCSRMatrix *A );
NALU_HYPRE_Int hypre_BoomerAMGWriteSolverParams ( void *data );

/* par_strength.c */
NALU_HYPRE_Int hypre_BoomerAMGCreateS ( hypre_ParCSRMatrix *A, NALU_HYPRE_Real strength_threshold,
                                   NALU_HYPRE_Real max_row_sum, NALU_HYPRE_Int num_functions, NALU_HYPRE_Int *dof_func, hypre_ParCSRMatrix **S_ptr );
NALU_HYPRE_Int hypre_BoomerAMGCreateSabs ( hypre_ParCSRMatrix *A, NALU_HYPRE_Real strength_threshold,
                                      NALU_HYPRE_Real max_row_sum, NALU_HYPRE_Int num_functions, NALU_HYPRE_Int *dof_func, hypre_ParCSRMatrix **S_ptr );
NALU_HYPRE_Int hypre_BoomerAMGCreateSCommPkg ( hypre_ParCSRMatrix *A, hypre_ParCSRMatrix *S,
                                          NALU_HYPRE_Int **col_offd_S_to_A_ptr );
NALU_HYPRE_Int hypre_BoomerAMGCreate2ndS ( hypre_ParCSRMatrix *S, NALU_HYPRE_Int *CF_marker,
                                      NALU_HYPRE_Int num_paths, NALU_HYPRE_BigInt *coarse_row_starts, hypre_ParCSRMatrix **C_ptr );
NALU_HYPRE_Int hypre_BoomerAMGCorrectCFMarker ( hypre_IntArray *CF_marker,
                                           hypre_IntArray *new_CF_marker );
NALU_HYPRE_Int hypre_BoomerAMGCorrectCFMarkerHost ( hypre_IntArray *CF_marker,
                                               hypre_IntArray *new_CF_marker );
NALU_HYPRE_Int hypre_BoomerAMGCorrectCFMarkerDevice ( hypre_IntArray *CF_marker,
                                                 hypre_IntArray *new_CF_marker );
NALU_HYPRE_Int hypre_BoomerAMGCorrectCFMarker2 ( hypre_IntArray *CF_marker,
                                            hypre_IntArray *new_CF_marker );
NALU_HYPRE_Int hypre_BoomerAMGCorrectCFMarker2Host ( hypre_IntArray *CF_marker,
                                                hypre_IntArray *new_CF_marker );
NALU_HYPRE_Int hypre_BoomerAMGCorrectCFMarker2Device ( hypre_IntArray *CF_marker,
                                                  hypre_IntArray *new_CF_marker );
NALU_HYPRE_Int hypre_BoomerAMGCreateSHost(hypre_ParCSRMatrix *A, NALU_HYPRE_Real strength_threshold,
                                     NALU_HYPRE_Real max_row_sum, NALU_HYPRE_Int num_functions, NALU_HYPRE_Int *dof_func, hypre_ParCSRMatrix **S_ptr);
NALU_HYPRE_Int hypre_BoomerAMGCreateSDevice(hypre_ParCSRMatrix *A, NALU_HYPRE_Int abs_soc,
                                       NALU_HYPRE_Real strength_threshold, NALU_HYPRE_Real max_row_sum, NALU_HYPRE_Int num_functions, NALU_HYPRE_Int *dof_func,
                                       hypre_ParCSRMatrix **S_ptr);
NALU_HYPRE_Int hypre_BoomerAMGCreateSabsHost ( hypre_ParCSRMatrix *A, NALU_HYPRE_Real strength_threshold,
                                          NALU_HYPRE_Real max_row_sum, NALU_HYPRE_Int num_functions, NALU_HYPRE_Int *dof_func, hypre_ParCSRMatrix **S_ptr );
NALU_HYPRE_Int hypre_BoomerAMGCreate2ndSDevice( hypre_ParCSRMatrix *S, NALU_HYPRE_Int *CF_marker,
                                           NALU_HYPRE_Int num_paths, NALU_HYPRE_BigInt *coarse_row_starts, hypre_ParCSRMatrix **C_ptr);
NALU_HYPRE_Int hypre_BoomerAMGMakeSocFromSDevice( hypre_ParCSRMatrix *A, hypre_ParCSRMatrix *S);


/* par_sv_interp.c */
NALU_HYPRE_Int hypre_BoomerAMGSmoothInterpVectors ( hypre_ParCSRMatrix *A, NALU_HYPRE_Int num_smooth_vecs,
                                               hypre_ParVector **smooth_vecs, NALU_HYPRE_Int smooth_steps );
NALU_HYPRE_Int hypre_BoomerAMGCoarsenInterpVectors ( hypre_ParCSRMatrix *P, NALU_HYPRE_Int num_smooth_vecs,
                                                hypre_ParVector **smooth_vecs, NALU_HYPRE_Int *CF_marker, hypre_ParVector ***new_smooth_vecs,
                                                NALU_HYPRE_Int expand_level, NALU_HYPRE_Int num_functions );
NALU_HYPRE_Int hypre_BoomerAMG_GMExpandInterp ( hypre_ParCSRMatrix *A, hypre_ParCSRMatrix **P,
                                           NALU_HYPRE_Int num_smooth_vecs, hypre_ParVector **smooth_vecs, NALU_HYPRE_Int *nf, NALU_HYPRE_Int *dof_func,
                                           hypre_IntArray **coarse_dof_func, NALU_HYPRE_Int variant, NALU_HYPRE_Int level, NALU_HYPRE_Real abs_trunc,
                                           NALU_HYPRE_Real *weights, NALU_HYPRE_Int q_max, NALU_HYPRE_Int *CF_marker, NALU_HYPRE_Int interp_vec_first_level );
NALU_HYPRE_Int hypre_BoomerAMGRefineInterp ( hypre_ParCSRMatrix *A, hypre_ParCSRMatrix *P,
                                        NALU_HYPRE_BigInt *num_cpts_global, NALU_HYPRE_Int *nf, NALU_HYPRE_Int *dof_func, NALU_HYPRE_Int *CF_marker,
                                        NALU_HYPRE_Int level );

/* par_sv_interp_ln.c */
NALU_HYPRE_Int hypre_BoomerAMG_LNExpandInterp ( hypre_ParCSRMatrix *A, hypre_ParCSRMatrix **P,
                                           NALU_HYPRE_BigInt *num_cpts_global, NALU_HYPRE_Int *nf, NALU_HYPRE_Int *dof_func, hypre_IntArray **coarse_dof_func,
                                           NALU_HYPRE_Int *CF_marker, NALU_HYPRE_Int level, NALU_HYPRE_Real *weights, NALU_HYPRE_Int num_smooth_vecs,
                                           hypre_ParVector **smooth_vecs, NALU_HYPRE_Real abs_trunc, NALU_HYPRE_Int q_max,
                                           NALU_HYPRE_Int interp_vec_first_level );

/* par_sv_interp_lsfit.c */
NALU_HYPRE_Int hypre_BoomerAMGFitInterpVectors ( hypre_ParCSRMatrix *A, hypre_ParCSRMatrix **P,
                                            NALU_HYPRE_Int num_smooth_vecs, hypre_ParVector **smooth_vecs, hypre_ParVector **coarse_smooth_vecs,
                                            NALU_HYPRE_Real delta, NALU_HYPRE_Int num_functions, NALU_HYPRE_Int *dof_func, NALU_HYPRE_Int *CF_marker,
                                            NALU_HYPRE_Int max_elmts, NALU_HYPRE_Real trunc_factor, NALU_HYPRE_Int variant, NALU_HYPRE_Int level );

/* partial.c */
NALU_HYPRE_Int hypre_BoomerAMGBuildPartialExtPIInterp ( hypre_ParCSRMatrix *A, NALU_HYPRE_Int *CF_marker,
                                                   hypre_ParCSRMatrix *S, NALU_HYPRE_BigInt *num_cpts_global, NALU_HYPRE_BigInt *num_old_cpts_global,
                                                   NALU_HYPRE_Int num_functions, NALU_HYPRE_Int *dof_func, NALU_HYPRE_Int debug_flag, NALU_HYPRE_Real trunc_factor,
                                                   NALU_HYPRE_Int max_elmts, hypre_ParCSRMatrix **P_ptr );
NALU_HYPRE_Int hypre_BoomerAMGBuildPartialStdInterp ( hypre_ParCSRMatrix *A, NALU_HYPRE_Int *CF_marker,
                                                 hypre_ParCSRMatrix *S, NALU_HYPRE_BigInt *num_cpts_global, NALU_HYPRE_BigInt *num_old_cpts_global,
                                                 NALU_HYPRE_Int num_functions, NALU_HYPRE_Int *dof_func, NALU_HYPRE_Int debug_flag, NALU_HYPRE_Real trunc_factor,
                                                 NALU_HYPRE_Int max_elmts, NALU_HYPRE_Int sep_weight, hypre_ParCSRMatrix **P_ptr );
NALU_HYPRE_Int hypre_BoomerAMGBuildPartialExtInterp ( hypre_ParCSRMatrix *A, NALU_HYPRE_Int *CF_marker,
                                                 hypre_ParCSRMatrix *S, NALU_HYPRE_BigInt *num_cpts_global, NALU_HYPRE_BigInt *num_old_cpts_global,
                                                 NALU_HYPRE_Int num_functions, NALU_HYPRE_Int *dof_func, NALU_HYPRE_Int debug_flag, NALU_HYPRE_Real trunc_factor,
                                                 NALU_HYPRE_Int max_elmts, hypre_ParCSRMatrix **P_ptr );

/* par_vardifconv.c */
NALU_HYPRE_ParCSRMatrix GenerateVarDifConv ( MPI_Comm comm, NALU_HYPRE_BigInt nx, NALU_HYPRE_BigInt ny,
                                        NALU_HYPRE_BigInt nz, NALU_HYPRE_Int P, NALU_HYPRE_Int Q, NALU_HYPRE_Int R, NALU_HYPRE_Int p, NALU_HYPRE_Int q, NALU_HYPRE_Int r,
                                        NALU_HYPRE_Real eps, NALU_HYPRE_ParVector *rhs_ptr );
NALU_HYPRE_Real afun ( NALU_HYPRE_Real xx, NALU_HYPRE_Real yy, NALU_HYPRE_Real zz );
NALU_HYPRE_Real bfun ( NALU_HYPRE_Real xx, NALU_HYPRE_Real yy, NALU_HYPRE_Real zz );
NALU_HYPRE_Real cfun ( NALU_HYPRE_Real xx, NALU_HYPRE_Real yy, NALU_HYPRE_Real zz );
NALU_HYPRE_Real dfun ( NALU_HYPRE_Real xx, NALU_HYPRE_Real yy, NALU_HYPRE_Real zz );
NALU_HYPRE_Real efun ( NALU_HYPRE_Real xx, NALU_HYPRE_Real yy, NALU_HYPRE_Real zz );
NALU_HYPRE_Real ffun ( NALU_HYPRE_Real xx, NALU_HYPRE_Real yy, NALU_HYPRE_Real zz );
NALU_HYPRE_Real gfun ( NALU_HYPRE_Real xx, NALU_HYPRE_Real yy, NALU_HYPRE_Real zz );
NALU_HYPRE_Real rfun ( NALU_HYPRE_Real xx, NALU_HYPRE_Real yy, NALU_HYPRE_Real zz );
NALU_HYPRE_Real bndfun ( NALU_HYPRE_Real xx, NALU_HYPRE_Real yy, NALU_HYPRE_Real zz );

/* par_vardifconv_rs.c */
NALU_HYPRE_ParCSRMatrix GenerateRSVarDifConv ( MPI_Comm comm, NALU_HYPRE_BigInt nx, NALU_HYPRE_BigInt ny,
                                          NALU_HYPRE_BigInt nz, NALU_HYPRE_Int P, NALU_HYPRE_Int Q, NALU_HYPRE_Int R, NALU_HYPRE_Int p, NALU_HYPRE_Int q, NALU_HYPRE_Int r,
                                          NALU_HYPRE_Real eps, NALU_HYPRE_ParVector *rhs_ptr, NALU_HYPRE_Int type );
NALU_HYPRE_Real afun_rs ( NALU_HYPRE_Real xx, NALU_HYPRE_Real yy, NALU_HYPRE_Real zz );
NALU_HYPRE_Real bfun_rs ( NALU_HYPRE_Real xx, NALU_HYPRE_Real yy, NALU_HYPRE_Real zz );
NALU_HYPRE_Real cfun_rs ( NALU_HYPRE_Real xx, NALU_HYPRE_Real yy, NALU_HYPRE_Real zz );
NALU_HYPRE_Real dfun_rs ( NALU_HYPRE_Real xx, NALU_HYPRE_Real yy, NALU_HYPRE_Real zz );
NALU_HYPRE_Real efun_rs ( NALU_HYPRE_Real xx, NALU_HYPRE_Real yy, NALU_HYPRE_Real zz );
NALU_HYPRE_Real ffun_rs ( NALU_HYPRE_Real xx, NALU_HYPRE_Real yy, NALU_HYPRE_Real zz );
NALU_HYPRE_Real gfun_rs ( NALU_HYPRE_Real xx, NALU_HYPRE_Real yy, NALU_HYPRE_Real zz );
NALU_HYPRE_Real rfun_rs ( NALU_HYPRE_Real xx, NALU_HYPRE_Real yy, NALU_HYPRE_Real zz );
NALU_HYPRE_Real bndfun_rs ( NALU_HYPRE_Real xx, NALU_HYPRE_Real yy, NALU_HYPRE_Real zz );


/* pcg_par.c */
void *hypre_ParKrylovCAlloc ( size_t count, size_t elt_size, NALU_HYPRE_MemoryLocation location );
NALU_HYPRE_Int hypre_ParKrylovFree ( void *ptr );
void *hypre_ParKrylovCreateVector ( void *vvector );
void *hypre_ParKrylovCreateVectorArray ( NALU_HYPRE_Int n, void *vvector );
NALU_HYPRE_Int hypre_ParKrylovDestroyVector ( void *vvector );
void *hypre_ParKrylovMatvecCreate ( void *A, void *x );
NALU_HYPRE_Int hypre_ParKrylovMatvec ( void *matvec_data, NALU_HYPRE_Complex alpha, void *A, void *x,
                                  NALU_HYPRE_Complex beta, void *y );
NALU_HYPRE_Int hypre_ParKrylovMatvecT ( void *matvec_data, NALU_HYPRE_Complex alpha, void *A, void *x,
                                   NALU_HYPRE_Complex beta, void *y );
NALU_HYPRE_Int hypre_ParKrylovMatvecDestroy ( void *matvec_data );
NALU_HYPRE_Real hypre_ParKrylovInnerProd ( void *x, void *y );
NALU_HYPRE_Int hypre_ParKrylovMassInnerProd ( void *x, void **y, NALU_HYPRE_Int k, NALU_HYPRE_Int unroll,
                                         void *result );
NALU_HYPRE_Int hypre_ParKrylovMassDotpTwo ( void *x, void *y, void **z, NALU_HYPRE_Int k, NALU_HYPRE_Int unroll,
                                       void *result_x, void *result_y );
NALU_HYPRE_Int hypre_ParKrylovMassAxpy( NALU_HYPRE_Complex *alpha, void **x, void *y, NALU_HYPRE_Int k,
                                   NALU_HYPRE_Int unroll);
NALU_HYPRE_Int hypre_ParKrylovCopyVector ( void *x, void *y );
NALU_HYPRE_Int hypre_ParKrylovClearVector ( void *x );
NALU_HYPRE_Int hypre_ParKrylovScaleVector ( NALU_HYPRE_Complex alpha, void *x );
NALU_HYPRE_Int hypre_ParKrylovAxpy ( NALU_HYPRE_Complex alpha, void *x, void *y );
NALU_HYPRE_Int hypre_ParKrylovCommInfo ( void *A, NALU_HYPRE_Int *my_id, NALU_HYPRE_Int *num_procs );
NALU_HYPRE_Int hypre_ParKrylovIdentitySetup ( void *vdata, void *A, void *b, void *x );
NALU_HYPRE_Int hypre_ParKrylovIdentity ( void *vdata, void *A, void *b, void *x );

/* schwarz.c */
NALU_HYPRE_Int hypre_AMGNodalSchwarzSmoother ( hypre_CSRMatrix *A, NALU_HYPRE_Int num_functions,
                                          NALU_HYPRE_Int option, hypre_CSRMatrix **domain_structure_pointer );
NALU_HYPRE_Int hypre_ParMPSchwarzSolve ( hypre_ParCSRMatrix *par_A, hypre_CSRMatrix *A_boundary,
                                    hypre_ParVector *rhs_vector, hypre_CSRMatrix *domain_structure, hypre_ParVector *par_x,
                                    NALU_HYPRE_Real relax_wt, NALU_HYPRE_Real *scale, hypre_ParVector *Vtemp, NALU_HYPRE_Int *pivots,
                                    NALU_HYPRE_Int use_nonsymm );
NALU_HYPRE_Int hypre_MPSchwarzSolve ( hypre_ParCSRMatrix *par_A, hypre_Vector *rhs_vector,
                                 hypre_CSRMatrix *domain_structure, hypre_ParVector *par_x, NALU_HYPRE_Real relax_wt,
                                 hypre_Vector *aux_vector, NALU_HYPRE_Int *pivots, NALU_HYPRE_Int use_nonsymm );
NALU_HYPRE_Int hypre_MPSchwarzCFSolve ( hypre_ParCSRMatrix *par_A, hypre_Vector *rhs_vector,
                                   hypre_CSRMatrix *domain_structure, hypre_ParVector *par_x, NALU_HYPRE_Real relax_wt,
                                   hypre_Vector *aux_vector, NALU_HYPRE_Int *CF_marker, NALU_HYPRE_Int rlx_pt, NALU_HYPRE_Int *pivots,
                                   NALU_HYPRE_Int use_nonsymm );
NALU_HYPRE_Int hypre_MPSchwarzFWSolve ( hypre_ParCSRMatrix *par_A, hypre_Vector *rhs_vector,
                                   hypre_CSRMatrix *domain_structure, hypre_ParVector *par_x, NALU_HYPRE_Real relax_wt,
                                   hypre_Vector *aux_vector, NALU_HYPRE_Int *pivots, NALU_HYPRE_Int use_nonsymm );
NALU_HYPRE_Int hypre_MPSchwarzCFFWSolve ( hypre_ParCSRMatrix *par_A, hypre_Vector *rhs_vector,
                                     hypre_CSRMatrix *domain_structure, hypre_ParVector *par_x, NALU_HYPRE_Real relax_wt,
                                     hypre_Vector *aux_vector, NALU_HYPRE_Int *CF_marker, NALU_HYPRE_Int rlx_pt, NALU_HYPRE_Int *pivots,
                                     NALU_HYPRE_Int use_nonsymm );
NALU_HYPRE_Int transpose_matrix_create ( NALU_HYPRE_Int **i_face_element_pointer,
                                    NALU_HYPRE_Int **j_face_element_pointer, NALU_HYPRE_Int *i_element_face, NALU_HYPRE_Int *j_element_face,
                                    NALU_HYPRE_Int num_elements, NALU_HYPRE_Int num_faces );
NALU_HYPRE_Int matrix_matrix_product ( NALU_HYPRE_Int **i_element_edge_pointer,
                                  NALU_HYPRE_Int **j_element_edge_pointer, NALU_HYPRE_Int *i_element_face, NALU_HYPRE_Int *j_element_face,
                                  NALU_HYPRE_Int *i_face_edge, NALU_HYPRE_Int *j_face_edge, NALU_HYPRE_Int num_elements, NALU_HYPRE_Int num_faces,
                                  NALU_HYPRE_Int num_edges );
NALU_HYPRE_Int hypre_AMGCreateDomainDof ( hypre_CSRMatrix *A, NALU_HYPRE_Int domain_type, NALU_HYPRE_Int overlap,
                                     NALU_HYPRE_Int num_functions, NALU_HYPRE_Int *dof_func, hypre_CSRMatrix **domain_structure_pointer,
                                     NALU_HYPRE_Int **piv_pointer, NALU_HYPRE_Int use_nonsymm );
NALU_HYPRE_Int hypre_AMGeAgglomerate ( NALU_HYPRE_Int *i_AE_element, NALU_HYPRE_Int *j_AE_element,
                                  NALU_HYPRE_Int *i_face_face, NALU_HYPRE_Int *j_face_face, NALU_HYPRE_Int *w_face_face, NALU_HYPRE_Int *i_face_element,
                                  NALU_HYPRE_Int *j_face_element, NALU_HYPRE_Int *i_element_face, NALU_HYPRE_Int *j_element_face,
                                  NALU_HYPRE_Int *i_face_to_prefer_weight, NALU_HYPRE_Int *i_face_weight, NALU_HYPRE_Int num_faces,
                                  NALU_HYPRE_Int num_elements, NALU_HYPRE_Int *num_AEs_pointer );
NALU_HYPRE_Int hypre_update_entry ( NALU_HYPRE_Int weight, NALU_HYPRE_Int *weight_max, NALU_HYPRE_Int *previous,
                               NALU_HYPRE_Int *next, NALU_HYPRE_Int *first, NALU_HYPRE_Int *last, NALU_HYPRE_Int head, NALU_HYPRE_Int tail, NALU_HYPRE_Int i );
NALU_HYPRE_Int hypre_remove_entry ( NALU_HYPRE_Int weight, NALU_HYPRE_Int *weight_max, NALU_HYPRE_Int *previous,
                               NALU_HYPRE_Int *next, NALU_HYPRE_Int *first, NALU_HYPRE_Int *last, NALU_HYPRE_Int head, NALU_HYPRE_Int tail, NALU_HYPRE_Int i );
NALU_HYPRE_Int hypre_move_entry ( NALU_HYPRE_Int weight, NALU_HYPRE_Int *weight_max, NALU_HYPRE_Int *previous,
                             NALU_HYPRE_Int *next, NALU_HYPRE_Int *first, NALU_HYPRE_Int *last, NALU_HYPRE_Int head, NALU_HYPRE_Int tail, NALU_HYPRE_Int i );
NALU_HYPRE_Int hypre_matinv ( NALU_HYPRE_Real *x, NALU_HYPRE_Real *a, NALU_HYPRE_Int k );
NALU_HYPRE_Int hypre_parCorrRes ( hypre_ParCSRMatrix *A, hypre_ParVector *x, hypre_Vector *rhs,
                             hypre_Vector **tmp_ptr );
NALU_HYPRE_Int hypre_AdSchwarzSolve ( hypre_ParCSRMatrix *par_A, hypre_ParVector *par_rhs,
                                 hypre_CSRMatrix *domain_structure, NALU_HYPRE_Real *scale, hypre_ParVector *par_x,
                                 hypre_ParVector *par_aux, NALU_HYPRE_Int *pivots, NALU_HYPRE_Int use_nonsymm );
NALU_HYPRE_Int hypre_AdSchwarzCFSolve ( hypre_ParCSRMatrix *par_A, hypre_ParVector *par_rhs,
                                   hypre_CSRMatrix *domain_structure, NALU_HYPRE_Real *scale, hypre_ParVector *par_x,
                                   hypre_ParVector *par_aux, NALU_HYPRE_Int *CF_marker, NALU_HYPRE_Int rlx_pt, NALU_HYPRE_Int *pivots,
                                   NALU_HYPRE_Int use_nonsymm );
NALU_HYPRE_Int hypre_GenerateScale ( hypre_CSRMatrix *domain_structure, NALU_HYPRE_Int num_variables,
                                NALU_HYPRE_Real relaxation_weight, NALU_HYPRE_Real **scale_pointer );
NALU_HYPRE_Int hypre_ParAdSchwarzSolve ( hypre_ParCSRMatrix *A, hypre_ParVector *F,
                                    hypre_CSRMatrix *domain_structure, NALU_HYPRE_Real *scale, hypre_ParVector *X, hypre_ParVector *Vtemp,
                                    NALU_HYPRE_Int *pivots, NALU_HYPRE_Int use_nonsymm );
NALU_HYPRE_Int hypre_ParAMGCreateDomainDof ( hypre_ParCSRMatrix *A, NALU_HYPRE_Int domain_type,
                                        NALU_HYPRE_Int overlap, NALU_HYPRE_Int num_functions, NALU_HYPRE_Int *dof_func,
                                        hypre_CSRMatrix **domain_structure_pointer, NALU_HYPRE_Int **piv_pointer, NALU_HYPRE_Int use_nonsymm );
NALU_HYPRE_Int hypre_ParGenerateScale ( hypre_ParCSRMatrix *A, hypre_CSRMatrix *domain_structure,
                                   NALU_HYPRE_Real relaxation_weight, NALU_HYPRE_Real **scale_pointer );
NALU_HYPRE_Int hypre_ParGenerateHybridScale ( hypre_ParCSRMatrix *A, hypre_CSRMatrix *domain_structure,
                                         hypre_CSRMatrix **A_boundary_pointer, NALU_HYPRE_Real **scale_pointer );

/* par_restr.c,  par_lr_restr.c */
NALU_HYPRE_Int hypre_BoomerAMGBuildRestrAIR( hypre_ParCSRMatrix *A, NALU_HYPRE_Int *CF_marker,
                                        hypre_ParCSRMatrix *S, NALU_HYPRE_BigInt *num_cpts_global, NALU_HYPRE_Int num_functions, NALU_HYPRE_Int *dof_func,
                                        NALU_HYPRE_Real filter_thresholdR, NALU_HYPRE_Int debug_flag, hypre_ParCSRMatrix **R_ptr,
                                        NALU_HYPRE_Int is_triangular, NALU_HYPRE_Int gmres_switch);
NALU_HYPRE_Int hypre_BoomerAMGBuildRestrDist2AIR( hypre_ParCSRMatrix *A, NALU_HYPRE_Int *CF_marker,
                                             hypre_ParCSRMatrix *S, NALU_HYPRE_BigInt *num_cpts_global, NALU_HYPRE_Int num_functions, NALU_HYPRE_Int *dof_func,
                                             NALU_HYPRE_Real filter_thresholdR, NALU_HYPRE_Int debug_flag, hypre_ParCSRMatrix **R_ptr, NALU_HYPRE_Int AIR1_5,
                                             NALU_HYPRE_Int is_triangular, NALU_HYPRE_Int gmres_switch);
NALU_HYPRE_Int hypre_BoomerAMGBuildRestrNeumannAIR( hypre_ParCSRMatrix *A, NALU_HYPRE_Int *CF_marker,
                                               NALU_HYPRE_BigInt *num_cpts_global, NALU_HYPRE_Int num_functions, NALU_HYPRE_Int *dof_func, NALU_HYPRE_Int NeumannDeg,
                                               NALU_HYPRE_Real strong_thresholdR, NALU_HYPRE_Real filter_thresholdR, NALU_HYPRE_Int debug_flag,
                                               hypre_ParCSRMatrix **R_ptr);
NALU_HYPRE_Int hypre_BoomerAMGBuildRestrNeumannAIRDevice( hypre_ParCSRMatrix *A, NALU_HYPRE_Int *CF_marker,
                                                     NALU_HYPRE_BigInt *num_cpts_global, NALU_HYPRE_Int num_functions, NALU_HYPRE_Int *dof_func, NALU_HYPRE_Int NeumannDeg,
                                                     NALU_HYPRE_Real strong_thresholdR, NALU_HYPRE_Real filter_thresholdR, NALU_HYPRE_Int debug_flag,
                                                     hypre_ParCSRMatrix **R_ptr);
NALU_HYPRE_Int hypre_BoomerAMGCFMarkerTo1minus1Device( NALU_HYPRE_Int *CF_marker, NALU_HYPRE_Int size );

#ifdef NALU_HYPRE_USING_DSUPERLU
/* superlu.c */
NALU_HYPRE_Int hypre_SLUDistSetup( NALU_HYPRE_Solver *solver, hypre_ParCSRMatrix *A, NALU_HYPRE_Int print_level);
NALU_HYPRE_Int hypre_SLUDistSolve( void* solver, hypre_ParVector *b, hypre_ParVector *x);
NALU_HYPRE_Int hypre_SLUDistDestroy( void* solver);
#endif

/* par_mgr.c */
void *hypre_MGRCreate ( void );
NALU_HYPRE_Int hypre_MGRDestroy ( void *mgr_vdata );
NALU_HYPRE_Int hypre_MGRCycle( void *mgr_vdata, hypre_ParVector **F_array, hypre_ParVector **U_array );
void *hypre_MGRCreateFrelaxVcycleData();
NALU_HYPRE_Int hypre_MGRDestroyFrelaxVcycleData( void *mgr_vdata );
void *hypre_MGRCreateGSElimData();
NALU_HYPRE_Int hypre_MGRDestroyGSElimData( void *mgr_vdata );
NALU_HYPRE_Int hypre_MGRSetupFrelaxVcycleData( void *mgr_vdata, hypre_ParCSRMatrix *A,
                                          hypre_ParVector *f, hypre_ParVector *u, NALU_HYPRE_Int level);
NALU_HYPRE_Int hypre_MGRFrelaxVcycle ( void *mgr_vdata, hypre_ParVector *f, hypre_ParVector *u );
NALU_HYPRE_Int hypre_MGRSetCpointsByBlock( void *mgr_vdata, NALU_HYPRE_Int  block_size,
                                      NALU_HYPRE_Int  max_num_levels, NALU_HYPRE_Int *block_num_coarse_points, NALU_HYPRE_Int  **block_coarse_indexes);
NALU_HYPRE_Int hypre_MGRSetCpointsByContiguousBlock( void *mgr_vdata, NALU_HYPRE_Int  block_size,
                                                NALU_HYPRE_Int  max_num_levels, NALU_HYPRE_BigInt *begin_idx_array, NALU_HYPRE_Int *block_num_coarse_points,
                                                NALU_HYPRE_Int  **block_coarse_indexes);
NALU_HYPRE_Int hypre_MGRSetCpointsByPointMarkerArray( void *mgr_vdata, NALU_HYPRE_Int  block_size,
                                                 NALU_HYPRE_Int  max_num_levels, NALU_HYPRE_Int *block_num_coarse_points, NALU_HYPRE_Int  **block_coarse_indexes,
                                                 NALU_HYPRE_Int *point_marker_array);
NALU_HYPRE_Int hypre_MGRCoarsen(hypre_ParCSRMatrix *S,  hypre_ParCSRMatrix *A,
                           NALU_HYPRE_Int final_coarse_size, NALU_HYPRE_Int *final_coarse_indexes, NALU_HYPRE_Int debug_flag,
                           hypre_IntArray **CF_marker, NALU_HYPRE_Int last_level);
NALU_HYPRE_Int hypre_MGRSetReservedCoarseNodes(void      *mgr_vdata, NALU_HYPRE_Int reserved_coarse_size,
                                          NALU_HYPRE_BigInt *reserved_coarse_nodes);
NALU_HYPRE_Int hypre_MGRSetReservedCpointsLevelToKeep( void      *mgr_vdata, NALU_HYPRE_Int level);
NALU_HYPRE_Int hypre_MGRSetMaxGlobalSmoothIters( void *mgr_vdata, NALU_HYPRE_Int max_iter );
NALU_HYPRE_Int hypre_MGRSetGlobalSmoothType( void *mgr_vdata, NALU_HYPRE_Int iter_type );
NALU_HYPRE_Int hypre_MGRSetNonCpointsToFpoints( void      *mgr_vdata, NALU_HYPRE_Int nonCptToFptFlag);

//NALU_HYPRE_Int hypre_MGRInitCFMarker(NALU_HYPRE_Int num_variables, NALU_HYPRE_Int *CF_marker, NALU_HYPRE_Int initial_coarse_size,NALU_HYPRE_Int *initial_coarse_indexes);
//NALU_HYPRE_Int hypre_MGRUpdateCoarseIndexes(NALU_HYPRE_Int num_variables, NALU_HYPRE_Int *CF_marker, NALU_HYPRE_Int initial_coarse_size,NALU_HYPRE_Int *initial_coarse_indexes);
NALU_HYPRE_Int hypre_MGRRelaxL1JacobiDevice( hypre_ParCSRMatrix *A, hypre_ParVector *f,
                                        NALU_HYPRE_Int *CF_marker_host, NALU_HYPRE_Int relax_points, NALU_HYPRE_Real relax_weight, NALU_HYPRE_Real *l1_norms,
                                        hypre_ParVector *u, hypre_ParVector *Vtemp );
NALU_HYPRE_Int hypre_MGRBuildPDevice(hypre_ParCSRMatrix *A, NALU_HYPRE_Int *CF_marker_host,
                                NALU_HYPRE_BigInt *num_cpts_global, NALU_HYPRE_Int method, hypre_ParCSRMatrix **P_ptr);
NALU_HYPRE_Int hypre_MGRBuildP(hypre_ParCSRMatrix *A, NALU_HYPRE_Int *CF_marker,
                          NALU_HYPRE_BigInt *num_cpts_global, NALU_HYPRE_Int method, NALU_HYPRE_Int debug_flag, hypre_ParCSRMatrix **P_ptr);
NALU_HYPRE_Int hypre_MGRBuildInterp(hypre_ParCSRMatrix *A, NALU_HYPRE_Int *CF_marker, hypre_ParCSRMatrix *S,
                               NALU_HYPRE_BigInt *num_cpts_global, NALU_HYPRE_Int num_functions, NALU_HYPRE_Int *dof_func, NALU_HYPRE_Int debug_flag,
                               NALU_HYPRE_Real trunc_factor, NALU_HYPRE_Int max_elmts, NALU_HYPRE_Int block_jacobi_bsize, hypre_ParCSRMatrix  **P,
                               NALU_HYPRE_Int method, NALU_HYPRE_Int numsweeps);
NALU_HYPRE_Int hypre_MGRBuildRestrict(hypre_ParCSRMatrix *A, NALU_HYPRE_Int *CF_marker,
                                 NALU_HYPRE_BigInt *num_cpts_global, NALU_HYPRE_Int num_functions, NALU_HYPRE_Int *dof_func, NALU_HYPRE_Int debug_flag,
                                 NALU_HYPRE_Real trunc_factor, NALU_HYPRE_Int max_elmts, NALU_HYPRE_Real strong_threshold, NALU_HYPRE_Real max_row_sum,
                                 NALU_HYPRE_Int blk_size, hypre_ParCSRMatrix  **RT, NALU_HYPRE_Int method, NALU_HYPRE_Int numsweeps);
//NALU_HYPRE_Int hypre_MGRBuildRestrictionToper(hypre_ParCSRMatrix *AT, NALU_HYPRE_Int *CF_marker, hypre_ParCSRMatrix *ST, NALU_HYPRE_Int *num_cpts_global,NALU_HYPRE_Int num_functions,NALU_HYPRE_Int *dof_func,NALU_HYPRE_Int debug_flag,NALU_HYPRE_Real trunc_factor, NALU_HYPRE_Int max_elmts, hypre_ParCSRMatrix  **RT,NALU_HYPRE_Int last_level,NALU_HYPRE_Int level, NALU_HYPRE_Int numsweeps);
//NALU_HYPRE_Int hypre_BoomerAMGBuildInjectionInterp( hypre_ParCSRMatrix   *A, NALU_HYPRE_Int *CF_marker, NALU_HYPRE_Int *num_cpts_global, NALU_HYPRE_Int num_functions, NALU_HYPRE_Int debug_flag,NALU_HYPRE_Int init_data,hypre_ParCSRMatrix  **P_ptr);
NALU_HYPRE_Int hypre_MGRBuildBlockJacobiWp( hypre_ParCSRMatrix *A, NALU_HYPRE_Int blk_size,
                                       NALU_HYPRE_Int *CF_marker, NALU_HYPRE_BigInt *cpts_starts_in,
                                       hypre_ParCSRMatrix **Wp_ptr);
NALU_HYPRE_Int hypre_ParCSRMatrixExtractBlockDiag(hypre_ParCSRMatrix   *A, NALU_HYPRE_Int blk_size,
                                             NALU_HYPRE_Int point_type, NALU_HYPRE_Int *CF_marker,
                                             NALU_HYPRE_Int *inv_size_ptr, NALU_HYPRE_Real **diaginv_ptr, NALU_HYPRE_Int diag_type);
NALU_HYPRE_Int hypre_ParCSRMatrixBlockDiagMatrix(  hypre_ParCSRMatrix *A, NALU_HYPRE_Int blk_size,
                                              NALU_HYPRE_Int point_type, NALU_HYPRE_Int *CF_marker,
                                              hypre_ParCSRMatrix **B_ptr, NALU_HYPRE_Int diag_type);
NALU_HYPRE_Int hypre_MGRSetCoarseSolver( void  *mgr_vdata, NALU_HYPRE_Int  (*coarse_grid_solver_solve)(void*,
                                                                                             void*, void*, void*), NALU_HYPRE_Int  (*coarse_grid_solver_setup)(void*, void*, void*, void*),
                                    void  *coarse_grid_solver );
NALU_HYPRE_Int hypre_MGRSetFSolver( void  *mgr_vdata, NALU_HYPRE_Int  (*fine_grid_solver_solve)(void*, void*,
                                                                                      void*, void*), NALU_HYPRE_Int  (*fine_grid_solver_setup)(void*, void*, void*, void*), void  *fsolver );
NALU_HYPRE_Int hypre_MGRSetup( void *mgr_vdata, hypre_ParCSRMatrix *A, hypre_ParVector    *f,
                          hypre_ParVector    *u );
NALU_HYPRE_Int hypre_MGRSolve( void *mgr_vdata, hypre_ParCSRMatrix *A, hypre_ParVector *f,
                          hypre_ParVector  *u );
NALU_HYPRE_Int hypre_block_jacobi_scaling(hypre_ParCSRMatrix *A, hypre_ParCSRMatrix **B_ptr,
                                     void               *mgr_vdata, NALU_HYPRE_Int             debug_flag);
NALU_HYPRE_Int hypre_MGRBlockRelaxSolve(hypre_ParCSRMatrix *A, hypre_ParVector *f, hypre_ParVector *u,
                                   NALU_HYPRE_Real blk_size, NALU_HYPRE_Int n_block, NALU_HYPRE_Int left_size, NALU_HYPRE_Int method, NALU_HYPRE_Real *diaginv,
                                   hypre_ParVector *Vtemp);
NALU_HYPRE_Int hypre_MGRBlockRelaxSetup(hypre_ParCSRMatrix *A, NALU_HYPRE_Int blk_size,
                                   NALU_HYPRE_Real **diaginvptr);
//NALU_HYPRE_Int hypre_blockRelax(hypre_ParCSRMatrix *A, hypre_ParVector *f, hypre_ParVector *u,
//                           NALU_HYPRE_Int blk_size, NALU_HYPRE_Int reserved_coarse_size, NALU_HYPRE_Int method, hypre_ParVector *Vtemp,
//                           hypre_ParVector *Ztemp);
NALU_HYPRE_Int hypre_block_jacobi_solve(hypre_ParCSRMatrix *A, hypre_ParVector *f, hypre_ParVector *u,
                                   NALU_HYPRE_Int blk_size, NALU_HYPRE_Int method,
                                   NALU_HYPRE_Real *diaginv, hypre_ParVector    *Vtemp);
//NALU_HYPRE_Int hypre_MGRBuildAffRAP( MPI_Comm comm, NALU_HYPRE_Int local_num_variables, NALU_HYPRE_Int num_functions,
//NALU_HYPRE_Int *dof_func, NALU_HYPRE_Int *CF_marker, NALU_HYPRE_Int **coarse_dof_func_ptr, NALU_HYPRE_BigInt **coarse_pnts_global_ptr,
//hypre_ParCSRMatrix *A, NALU_HYPRE_Int debug_flag, hypre_ParCSRMatrix **P_f_ptr, hypre_ParCSRMatrix **A_ff_ptr );
NALU_HYPRE_Int hypre_MGRGetSubBlock( hypre_ParCSRMatrix *A, NALU_HYPRE_Int *row_cf_marker,
                                NALU_HYPRE_Int *col_cf_marker, NALU_HYPRE_Int debug_flag, hypre_ParCSRMatrix **A_ff_ptr );
NALU_HYPRE_Int hypre_MGRBuildAff( hypre_ParCSRMatrix *A, NALU_HYPRE_Int *CF_marker, NALU_HYPRE_Int debug_flag,
                             hypre_ParCSRMatrix **A_ff_ptr );
NALU_HYPRE_Int hypre_MGRApproximateInverse(hypre_ParCSRMatrix *A, hypre_ParCSRMatrix **A_inv);
NALU_HYPRE_Int hypre_MGRAddVectorP ( hypre_IntArray *CF_marker, NALU_HYPRE_Int point_type, NALU_HYPRE_Real a,
                                hypre_ParVector *fromVector, NALU_HYPRE_Real b, hypre_ParVector **toVector );
NALU_HYPRE_Int hypre_MGRAddVectorR ( hypre_IntArray *CF_marker, NALU_HYPRE_Int point_type, NALU_HYPRE_Real a,
                                hypre_ParVector *fromVector, NALU_HYPRE_Real b, hypre_ParVector **toVector );
NALU_HYPRE_Int hypre_MGRComputeNonGalerkinCoarseGrid(hypre_ParCSRMatrix *A, hypre_ParCSRMatrix *P,
                                                hypre_ParCSRMatrix *RT, NALU_HYPRE_Int bsize, NALU_HYPRE_Int ordering, NALU_HYPRE_Int method, NALU_HYPRE_Int Pmax,
                                                NALU_HYPRE_Int *CF_marker, hypre_ParCSRMatrix **A_h_ptr);

NALU_HYPRE_Int hypre_MGRWriteSolverParams(void *mgr_vdata);
NALU_HYPRE_Int hypre_MGRSetAffSolverType( void *systg_vdata, NALU_HYPRE_Int *aff_solver_type );
NALU_HYPRE_Int hypre_MGRSetCoarseSolverType( void *systg_vdata, NALU_HYPRE_Int coarse_solver_type );
NALU_HYPRE_Int hypre_MGRSetCoarseSolverIter( void *systg_vdata, NALU_HYPRE_Int coarse_solver_iter );
NALU_HYPRE_Int hypre_MGRSetFineSolverIter( void *systg_vdata, NALU_HYPRE_Int fine_solver_iter );
NALU_HYPRE_Int hypre_MGRSetFineSolverMaxLevels( void *systg_vdata, NALU_HYPRE_Int fine_solver_max_levels );
NALU_HYPRE_Int hypre_MGRSetMaxCoarseLevels( void *mgr_vdata, NALU_HYPRE_Int maxlev );
NALU_HYPRE_Int hypre_MGRSetBlockSize( void *mgr_vdata, NALU_HYPRE_Int bsize );
NALU_HYPRE_Int hypre_MGRSetRelaxType( void *mgr_vdata, NALU_HYPRE_Int relax_type );
NALU_HYPRE_Int hypre_MGRSetFRelaxMethod( void *mgr_vdata, NALU_HYPRE_Int relax_method);
NALU_HYPRE_Int hypre_MGRSetLevelFRelaxMethod( void *mgr_vdata, NALU_HYPRE_Int *relax_method);
NALU_HYPRE_Int hypre_MGRSetLevelFRelaxType( void *mgr_vdata, NALU_HYPRE_Int *relax_type);
NALU_HYPRE_Int hypre_MGRSetLevelFRelaxNumFunctions( void *mgr_vdata, NALU_HYPRE_Int *num_functions);
NALU_HYPRE_Int hypre_MGRSetCoarseGridMethod( void *mgr_vdata, NALU_HYPRE_Int *cg_method);
NALU_HYPRE_Int hypre_MGRSetRestrictType( void *mgr_vdata, NALU_HYPRE_Int restrictType);
NALU_HYPRE_Int hypre_MGRSetLevelRestrictType( void *mgr_vdata, NALU_HYPRE_Int *restrictType);
NALU_HYPRE_Int hypre_MGRSetInterpType( void *mgr_vdata, NALU_HYPRE_Int interpType);
NALU_HYPRE_Int hypre_MGRSetLevelInterpType( void *mgr_vdata, NALU_HYPRE_Int *interpType);
NALU_HYPRE_Int hypre_MGRSetNumRelaxSweeps( void *mgr_vdata, NALU_HYPRE_Int nsweeps );
NALU_HYPRE_Int hypre_MGRSetLevelNumRelaxSweeps( void *mgr_vdata, NALU_HYPRE_Int *nsweeps );
NALU_HYPRE_Int hypre_MGRSetNumInterpSweeps( void *mgr_vdata, NALU_HYPRE_Int nsweeps );
NALU_HYPRE_Int hypre_MGRSetNumRestrictSweeps( void *mgr_vdata, NALU_HYPRE_Int nsweeps );
NALU_HYPRE_Int hypre_MGRSetLevelSmoothType( void *mgr_vdata, NALU_HYPRE_Int *level_smooth_type);
NALU_HYPRE_Int hypre_MGRSetLevelSmoothIters( void *mgr_vdata, NALU_HYPRE_Int *level_smooth_iters);
NALU_HYPRE_Int hypre_MGRSetGlobalSmoothCycle( void *mgr_vdata, NALU_HYPRE_Int global_smooth_cycle );
NALU_HYPRE_Int hypre_MGRSetPrintLevel( void *mgr_vdata, NALU_HYPRE_Int print_level );
NALU_HYPRE_Int hypre_MGRSetFrelaxPrintLevel( void *mgr_vdata, NALU_HYPRE_Int print_level );
NALU_HYPRE_Int hypre_MGRSetCoarseGridPrintLevel( void *mgr_vdata, NALU_HYPRE_Int print_level );
NALU_HYPRE_Int hypre_MGRSetTruncateCoarseGridThreshold( void *mgr_vdata, NALU_HYPRE_Real threshold);
NALU_HYPRE_Int hypre_MGRSetBlockJacobiBlockSize( void *mgr_vdata, NALU_HYPRE_Int blk_size);
NALU_HYPRE_Int hypre_MGRSetLogging( void *mgr_vdata, NALU_HYPRE_Int logging );
NALU_HYPRE_Int hypre_MGRSetMaxIter( void *mgr_vdata, NALU_HYPRE_Int max_iter );
NALU_HYPRE_Int hypre_MGRSetPMaxElmts( void *mgr_vdata, NALU_HYPRE_Int P_max_elmts);
NALU_HYPRE_Int hypre_MGRSetTol( void *mgr_vdata, NALU_HYPRE_Real tol );
#ifdef NALU_HYPRE_USING_DSUPERLU
void *hypre_MGRDirectSolverCreate( void );
NALU_HYPRE_Int hypre_MGRDirectSolverSetup( void *solver, hypre_ParCSRMatrix *A, hypre_ParVector *f,
                                      hypre_ParVector *u);
NALU_HYPRE_Int hypre_MGRDirectSolverSolve( void *solver, hypre_ParCSRMatrix *A, hypre_ParVector *f,
                                      hypre_ParVector *u);
NALU_HYPRE_Int hypre_MGRDirectSolverDestroy( void *solver );
#endif
// Accessor functions
NALU_HYPRE_Int hypre_MGRGetNumIterations( void *mgr_vdata, NALU_HYPRE_Int *num_iterations );
NALU_HYPRE_Int hypre_MGRGetFinalRelativeResidualNorm( void *mgr_vdata, NALU_HYPRE_Real *res_norm );
NALU_HYPRE_Int hypre_MGRGetCoarseGridConvergenceFactor( void *mgr_data, NALU_HYPRE_Real *conv_factor );

/* par_ilu.c */
void *hypre_ILUCreate ( void );
NALU_HYPRE_Int hypre_ILUDestroy ( void *ilu_vdata );
NALU_HYPRE_Int hypre_ILUSetLevelOfFill( void *ilu_vdata, NALU_HYPRE_Int lfil );
NALU_HYPRE_Int hypre_ILUSetMaxNnzPerRow( void *ilu_vdata, NALU_HYPRE_Int nzmax );
NALU_HYPRE_Int hypre_ILUSetDropThreshold( void *ilu_vdata, NALU_HYPRE_Real threshold );
NALU_HYPRE_Int hypre_ILUSetDropThresholdArray( void *ilu_vdata, NALU_HYPRE_Real *threshold );
NALU_HYPRE_Int hypre_ILUSetType( void *ilu_vdata, NALU_HYPRE_Int ilu_type );
NALU_HYPRE_Int hypre_ILUSetMaxIter( void *ilu_vdata, NALU_HYPRE_Int max_iter );
NALU_HYPRE_Int hypre_ILUSetTol( void *ilu_vdata, NALU_HYPRE_Real tol );
NALU_HYPRE_Int hypre_ILUSetTriSolve( void *ilu_vdata, NALU_HYPRE_Int tri_solve );
NALU_HYPRE_Int hypre_ILUSetLowerJacobiIters( void *ilu_vdata, NALU_HYPRE_Int lower_jacobi_iters );
NALU_HYPRE_Int hypre_ILUSetUpperJacobiIters( void *ilu_vdata, NALU_HYPRE_Int upper_jacobi_iters );
NALU_HYPRE_Int hypre_ILUSetup( void *ilu_vdata, hypre_ParCSRMatrix *A, hypre_ParVector    *f,
                          hypre_ParVector    *u );
NALU_HYPRE_Int hypre_ILUSolve( void *ilu_vdata, hypre_ParCSRMatrix *A, hypre_ParVector *f,
                          hypre_ParVector  *u );
NALU_HYPRE_Int hypre_ILUSetPrintLevel( void *ilu_vdata, NALU_HYPRE_Int print_level );
NALU_HYPRE_Int hypre_ILUSetLogging( void *ilu_vdata, NALU_HYPRE_Int logging );
NALU_HYPRE_Int hypre_ILUSetLocalReordering( void *ilu_vdata, NALU_HYPRE_Int ordering_type );
NALU_HYPRE_Int hypre_ILUSetSchurSolverKDIM( void *ilu_vdata, NALU_HYPRE_Int ss_kDim );
NALU_HYPRE_Int hypre_ILUSetSchurSolverMaxIter( void *ilu_vdata, NALU_HYPRE_Int ss_max_iter );
NALU_HYPRE_Int hypre_ILUSetSchurSolverTol( void *ilu_vdata, NALU_HYPRE_Real ss_tol );
NALU_HYPRE_Int hypre_ILUSetSchurSolverAbsoluteTol( void *ilu_vdata, NALU_HYPRE_Real ss_absolute_tol );
NALU_HYPRE_Int hypre_ILUSetSchurSolverLogging( void *ilu_vdata, NALU_HYPRE_Int ss_logging );
NALU_HYPRE_Int hypre_ILUSetSchurSolverPrintLevel( void *ilu_vdata, NALU_HYPRE_Int ss_print_level );
NALU_HYPRE_Int hypre_ILUSetSchurSolverRelChange( void *ilu_vdata, NALU_HYPRE_Int ss_rel_change );
NALU_HYPRE_Int hypre_ILUSetSchurPrecondILUType( void *ilu_vdata, NALU_HYPRE_Int sp_ilu_type );
NALU_HYPRE_Int hypre_ILUSetSchurPrecondILULevelOfFill( void *ilu_vdata, NALU_HYPRE_Int sp_ilu_lfil );
NALU_HYPRE_Int hypre_ILUSetSchurPrecondILUMaxNnzPerRow( void *ilu_vdata, NALU_HYPRE_Int sp_ilu_max_row_nnz );
NALU_HYPRE_Int hypre_ILUSetSchurPrecondILUDropThreshold( void *ilu_vdata, NALU_HYPRE_Real sp_ilu_droptol );
NALU_HYPRE_Int hypre_ILUSetSchurPrecondILUDropThresholdArray( void *ilu_vdata,
                                                         NALU_HYPRE_Real *sp_ilu_droptol );
NALU_HYPRE_Int hypre_ILUSetSchurPrecondPrintLevel( void *ilu_vdata, NALU_HYPRE_Int sp_print_level );
NALU_HYPRE_Int hypre_ILUSetSchurPrecondMaxIter( void *ilu_vdata, NALU_HYPRE_Int sp_max_iter );
NALU_HYPRE_Int hypre_ILUSetSchurPrecondTol( void *ilu_vdata, NALU_HYPRE_Int sp_tol );
NALU_HYPRE_Int hypre_ILUMinHeapAddI(NALU_HYPRE_Int *heap, NALU_HYPRE_Int len);
NALU_HYPRE_Int hypre_ILUMinHeapAddIIIi(NALU_HYPRE_Int *heap, NALU_HYPRE_Int *I1, NALU_HYPRE_Int *Ii1, NALU_HYPRE_Int len);
NALU_HYPRE_Int hypre_ILUMinHeapAddIRIi(NALU_HYPRE_Int *heap, NALU_HYPRE_Real *I1, NALU_HYPRE_Int *Ii1, NALU_HYPRE_Int len);
NALU_HYPRE_Int hypre_ILUMaxHeapAddRabsIIi(NALU_HYPRE_Real *heap, NALU_HYPRE_Int *I1, NALU_HYPRE_Int *Ii1,
                                     NALU_HYPRE_Int len);
NALU_HYPRE_Int hypre_ILUMaxrHeapAddRabsI(NALU_HYPRE_Real *heap, NALU_HYPRE_Int *I1, NALU_HYPRE_Int len);
NALU_HYPRE_Int hypre_ILUMinHeapRemoveI(NALU_HYPRE_Int *heap, NALU_HYPRE_Int len);
NALU_HYPRE_Int hypre_ILUMinHeapRemoveIIIi(NALU_HYPRE_Int *heap, NALU_HYPRE_Int *I1, NALU_HYPRE_Int *Ii1, NALU_HYPRE_Int len);
NALU_HYPRE_Int hypre_ILUMinHeapRemoveIRIi(NALU_HYPRE_Int *heap, NALU_HYPRE_Real *I1, NALU_HYPRE_Int *Ii1,
                                     NALU_HYPRE_Int len);
NALU_HYPRE_Int hypre_ILUMaxHeapRemoveRabsIIi(NALU_HYPRE_Real *heap, NALU_HYPRE_Int *I1, NALU_HYPRE_Int *Ii1,
                                        NALU_HYPRE_Int len);
NALU_HYPRE_Int hypre_ILUMaxrHeapRemoveRabsI(NALU_HYPRE_Real *heap, NALU_HYPRE_Int *I1, NALU_HYPRE_Int len);
NALU_HYPRE_Int hypre_ILUMaxQSplitRI(NALU_HYPRE_Real *array, NALU_HYPRE_Int *I1, NALU_HYPRE_Int left, NALU_HYPRE_Int bound,
                               NALU_HYPRE_Int right);
NALU_HYPRE_Int hypre_ILUMaxQSplitRabsI(NALU_HYPRE_Real *array, NALU_HYPRE_Int *I1, NALU_HYPRE_Int left, NALU_HYPRE_Int bound,
                                  NALU_HYPRE_Int right);
//NALU_HYPRE_Int hypre_quickSortIR (NALU_HYPRE_Int *a, NALU_HYPRE_Real *b, NALU_HYPRE_Int *iw, const NALU_HYPRE_Int lo, const NALU_HYPRE_Int hi);
NALU_HYPRE_Int hypre_ILUSortOffdColmap(hypre_ParCSRMatrix *A);
NALU_HYPRE_Int hypre_ILUMaxRabs(NALU_HYPRE_Real *array_data, NALU_HYPRE_Int *array_j, NALU_HYPRE_Int start,
                           NALU_HYPRE_Int end, NALU_HYPRE_Int nLU, NALU_HYPRE_Int *rperm, NALU_HYPRE_Real *value, NALU_HYPRE_Int *index,
                           NALU_HYPRE_Real *l1_norm, NALU_HYPRE_Int *nnz);
NALU_HYPRE_Int hypre_ILUGetPermddPQPre(NALU_HYPRE_Int n, NALU_HYPRE_Int nLU, NALU_HYPRE_Int *A_diag_i,
                                  NALU_HYPRE_Int *A_diag_j, NALU_HYPRE_Real *A_diag_data, NALU_HYPRE_Real tol, NALU_HYPRE_Int *perm, NALU_HYPRE_Int *rperm,
                                  NALU_HYPRE_Int *pperm_pre, NALU_HYPRE_Int *qperm_pre, NALU_HYPRE_Int *nB);
NALU_HYPRE_Int hypre_ILUGetPermddPQ(hypre_ParCSRMatrix *A, NALU_HYPRE_Int **pperm, NALU_HYPRE_Int **qperm,
                               NALU_HYPRE_Real tol, NALU_HYPRE_Int *nB, NALU_HYPRE_Int *nI, NALU_HYPRE_Int reordering_type);
NALU_HYPRE_Int hypre_ILUGetInteriorExteriorPerm(hypre_ParCSRMatrix *A, NALU_HYPRE_Int **perm, NALU_HYPRE_Int *nLU,
                                           NALU_HYPRE_Int reordering_type);
NALU_HYPRE_Int hypre_ILUGetLocalPerm(hypre_ParCSRMatrix *A, NALU_HYPRE_Int **perm, NALU_HYPRE_Int *nLU,
                                NALU_HYPRE_Int reordering_type);
NALU_HYPRE_Int hypre_ILUWriteSolverParams(void *ilu_vdata);
NALU_HYPRE_Int hypre_ILUBuildRASExternalMatrix(hypre_ParCSRMatrix *A, NALU_HYPRE_Int *rperm, NALU_HYPRE_Int **E_i,
                                          NALU_HYPRE_Int **E_j, NALU_HYPRE_Real **E_data);
NALU_HYPRE_Int hypre_ILUSetupILU0(hypre_ParCSRMatrix *A, NALU_HYPRE_Int *perm, NALU_HYPRE_Int *qperm,
                             NALU_HYPRE_Int nLU, NALU_HYPRE_Int nI, hypre_ParCSRMatrix **Lptr, NALU_HYPRE_Real** Dptr,
                             hypre_ParCSRMatrix **Uptr, hypre_ParCSRMatrix **Sptr, NALU_HYPRE_Int **u_end);
NALU_HYPRE_Int hypre_ILUSetupMILU0(hypre_ParCSRMatrix *A, NALU_HYPRE_Int *permp, NALU_HYPRE_Int *qpermp,
                              NALU_HYPRE_Int nLU, NALU_HYPRE_Int nI, hypre_ParCSRMatrix **Lptr, NALU_HYPRE_Real** Dptr,
                              hypre_ParCSRMatrix **Uptr, hypre_ParCSRMatrix **Sptr, NALU_HYPRE_Int **u_end, NALU_HYPRE_Int modified);
NALU_HYPRE_Int hypre_ILUSetupILUK(hypre_ParCSRMatrix *A, NALU_HYPRE_Int lfil, NALU_HYPRE_Int *permp,
                             NALU_HYPRE_Int *qpermp, NALU_HYPRE_Int nLU, NALU_HYPRE_Int nI, hypre_ParCSRMatrix **Lptr, NALU_HYPRE_Real** Dptr,
                             hypre_ParCSRMatrix **Uptr, hypre_ParCSRMatrix **Sptr, NALU_HYPRE_Int **u_end);
NALU_HYPRE_Int hypre_ILUSetupILUKSymbolic(NALU_HYPRE_Int n, NALU_HYPRE_Int *A_diag_i, NALU_HYPRE_Int *A_diag_j,
                                     NALU_HYPRE_Int lfil, NALU_HYPRE_Int *perm, NALU_HYPRE_Int *rperm, NALU_HYPRE_Int *iw, NALU_HYPRE_Int nLU,
                                     NALU_HYPRE_Int *L_diag_i, NALU_HYPRE_Int *U_diag_i, NALU_HYPRE_Int *S_diag_i, NALU_HYPRE_Int **L_diag_j,
                                     NALU_HYPRE_Int **U_diag_j, NALU_HYPRE_Int **S_diag_j, NALU_HYPRE_Int **u_end);
NALU_HYPRE_Int hypre_ILUSetupILUT(hypre_ParCSRMatrix *A, NALU_HYPRE_Int lfil, NALU_HYPRE_Real *tol,
                             NALU_HYPRE_Int *permp, NALU_HYPRE_Int *qpermp, NALU_HYPRE_Int nLU, NALU_HYPRE_Int nI, hypre_ParCSRMatrix **Lptr,
                             NALU_HYPRE_Real** Dptr, hypre_ParCSRMatrix **Uptr, hypre_ParCSRMatrix **Sptr, NALU_HYPRE_Int **u_end);
NALU_HYPRE_Int hypre_ILUSetupILU0RAS(hypre_ParCSRMatrix *A, NALU_HYPRE_Int *perm, NALU_HYPRE_Int nLU,
                                hypre_ParCSRMatrix **Lptr, NALU_HYPRE_Real** Dptr, hypre_ParCSRMatrix **Uptr);
NALU_HYPRE_Int hypre_ILUSetupILUKRASSymbolic(NALU_HYPRE_Int n, NALU_HYPRE_Int *A_diag_i, NALU_HYPRE_Int *A_diag_j,
                                        NALU_HYPRE_Int *A_offd_i, NALU_HYPRE_Int *A_offd_j, NALU_HYPRE_Int *E_i, NALU_HYPRE_Int *E_j, NALU_HYPRE_Int ext,
                                        NALU_HYPRE_Int lfil, NALU_HYPRE_Int *perm, NALU_HYPRE_Int *rperm, NALU_HYPRE_Int *iw, NALU_HYPRE_Int nLU,
                                        NALU_HYPRE_Int *L_diag_i, NALU_HYPRE_Int *U_diag_i, NALU_HYPRE_Int **L_diag_j, NALU_HYPRE_Int **U_diag_j);
NALU_HYPRE_Int hypre_ILUSetupILUKRAS(hypre_ParCSRMatrix *A, NALU_HYPRE_Int lfil, NALU_HYPRE_Int *perm,
                                NALU_HYPRE_Int nLU, hypre_ParCSRMatrix **Lptr, NALU_HYPRE_Real** Dptr, hypre_ParCSRMatrix **Uptr);
NALU_HYPRE_Int hypre_ILUSetupILUTRAS(hypre_ParCSRMatrix *A, NALU_HYPRE_Int lfil, NALU_HYPRE_Real *tol,
                                NALU_HYPRE_Int *perm, NALU_HYPRE_Int nLU, hypre_ParCSRMatrix **Lptr, NALU_HYPRE_Real** Dptr,
                                hypre_ParCSRMatrix **Uptr);
NALU_HYPRE_Int hypre_ILUSolveLU(hypre_ParCSRMatrix *A, hypre_ParVector *f, hypre_ParVector *u,
                           NALU_HYPRE_Int *perm, NALU_HYPRE_Int nLU, hypre_ParCSRMatrix *L, NALU_HYPRE_Real* D, hypre_ParCSRMatrix *U,
                           hypre_ParVector *utemp, hypre_ParVector *ftemp);
NALU_HYPRE_Int hypre_ILUSolveLUIter(hypre_ParCSRMatrix *A, hypre_ParVector *f, hypre_ParVector *u,
                               NALU_HYPRE_Int *perm, NALU_HYPRE_Int nLU, hypre_ParCSRMatrix *L, NALU_HYPRE_Real* D, hypre_ParCSRMatrix *U,
                               hypre_ParVector *utemp, hypre_ParVector *ftemp, hypre_Vector *xtemp, NALU_HYPRE_Int lower_jacobi_iters,
                               NALU_HYPRE_Int upper_jacobi_iters);
NALU_HYPRE_Int hypre_ILUSolveSchurGMRES(hypre_ParCSRMatrix *A, hypre_ParVector *f, hypre_ParVector *u,
                                   NALU_HYPRE_Int *perm, NALU_HYPRE_Int *qperm, NALU_HYPRE_Int nLU, hypre_ParCSRMatrix *L, NALU_HYPRE_Real* D,
                                   hypre_ParCSRMatrix *U, hypre_ParCSRMatrix *S, hypre_ParVector *ftemp, hypre_ParVector *utemp,
                                   NALU_HYPRE_Solver schur_solver, NALU_HYPRE_Solver schur_precond, hypre_ParVector *rhs, hypre_ParVector *x,
                                   NALU_HYPRE_Int *u_end);
NALU_HYPRE_Int hypre_ILUSolveSchurNSH(hypre_ParCSRMatrix *A, hypre_ParVector *f, hypre_ParVector *u,
                                 NALU_HYPRE_Int *perm, NALU_HYPRE_Int nLU, hypre_ParCSRMatrix *L, NALU_HYPRE_Real* D, hypre_ParCSRMatrix *U,
                                 hypre_ParCSRMatrix *S, hypre_ParVector *ftemp, hypre_ParVector *utemp, NALU_HYPRE_Solver schur_solver,
                                 hypre_ParVector *rhs, hypre_ParVector *x, NALU_HYPRE_Int *u_end);
NALU_HYPRE_Int hypre_ILUSolveLURAS(hypre_ParCSRMatrix *A, hypre_ParVector *f, hypre_ParVector *u,
                              NALU_HYPRE_Int *perm, hypre_ParCSRMatrix *L, NALU_HYPRE_Real* D, hypre_ParCSRMatrix *U,
                              hypre_ParVector *ftemp, hypre_ParVector *utemp, NALU_HYPRE_Real *fext, NALU_HYPRE_Real *uext);
NALU_HYPRE_Int hypre_ILUSetSchurNSHDropThreshold( void *ilu_vdata, NALU_HYPRE_Real threshold);
NALU_HYPRE_Int hypre_ILUSetSchurNSHDropThresholdArray( void *ilu_vdata, NALU_HYPRE_Real *threshold);
NALU_HYPRE_Int hypre_ILUSetupRAPILU0(hypre_ParCSRMatrix *A, NALU_HYPRE_Int *perm, NALU_HYPRE_Int n, NALU_HYPRE_Int nLU,
                                hypre_ParCSRMatrix **Lptr, NALU_HYPRE_Real **Dptr, hypre_ParCSRMatrix **Uptr, hypre_ParCSRMatrix **mLptr,
                                NALU_HYPRE_Real **mDptr, hypre_ParCSRMatrix **mUptr, NALU_HYPRE_Int **u_end);
NALU_HYPRE_Int hypre_ILUSolveRAPGMRESHOST(hypre_ParCSRMatrix *A, hypre_ParVector *f, hypre_ParVector *u,
                                     NALU_HYPRE_Int *perm, NALU_HYPRE_Int nLU, hypre_ParCSRMatrix *L, NALU_HYPRE_Real *D, hypre_ParCSRMatrix *U,
                                     hypre_ParCSRMatrix *mL, NALU_HYPRE_Real *mD, hypre_ParCSRMatrix *mU, hypre_ParVector *ftemp,
                                     hypre_ParVector *utemp, hypre_ParVector *xtemp, hypre_ParVector *ytemp, NALU_HYPRE_Solver schur_solver,
                                     NALU_HYPRE_Solver schur_precond, hypre_ParVector *rhs, hypre_ParVector *x, NALU_HYPRE_Int *u_end);
NALU_HYPRE_Int hypre_ParILURAPSchurGMRESSolveH( void *ilu_vdata, void *ilu_vdata2, hypre_ParVector *f,
                                           hypre_ParVector *u );
NALU_HYPRE_Int hypre_ParILURAPSchurGMRESDummySetupH(void *a, void *b, void *c, void *d);
NALU_HYPRE_Int hypre_ParILURAPSchurGMRESCommInfoH( void *ilu_vdata, NALU_HYPRE_Int *my_id,
                                              NALU_HYPRE_Int *num_procs);
void *hypre_ParILURAPSchurGMRESMatvecCreateH(void *ilu_vdata, void *x);
NALU_HYPRE_Int hypre_ParILURAPSchurGMRESMatvecH(void *matvec_data, NALU_HYPRE_Complex alpha, void *ilu_vdata,
                                           void *x, NALU_HYPRE_Complex beta, void *y);
NALU_HYPRE_Int hypre_ParILURAPSchurGMRESMatvecDestroyH(void *matvec_data );
NALU_HYPRE_Int hypre_ILULocalRCM( hypre_CSRMatrix *A, NALU_HYPRE_Int start, NALU_HYPRE_Int end, NALU_HYPRE_Int **permp,
                             NALU_HYPRE_Int **qpermp, NALU_HYPRE_Int sym);
NALU_HYPRE_Int hypre_ILULocalRCMNumbering(hypre_CSRMatrix *A, NALU_HYPRE_Int root, NALU_HYPRE_Int *marker,
                                     NALU_HYPRE_Int *perm, NALU_HYPRE_Int *current_nump);
NALU_HYPRE_Int hypre_ILULocalRCMFindPPNode( hypre_CSRMatrix *A, NALU_HYPRE_Int *rootp, NALU_HYPRE_Int *marker);
NALU_HYPRE_Int hypre_ILULocalRCMMindegree(NALU_HYPRE_Int n, NALU_HYPRE_Int *degree, NALU_HYPRE_Int *marker,
                                     NALU_HYPRE_Int *rootp);
NALU_HYPRE_Int hypre_ILULocalRCMOrder( hypre_CSRMatrix *A, NALU_HYPRE_Int *perm);
NALU_HYPRE_Int hypre_ILULocalRCMBuildLevel(hypre_CSRMatrix *A, NALU_HYPRE_Int root, NALU_HYPRE_Int *marker,
                                      NALU_HYPRE_Int *level_i, NALU_HYPRE_Int *level_j, NALU_HYPRE_Int *nlevp);
NALU_HYPRE_Int hypre_ILULocalRCMQsort(NALU_HYPRE_Int *perm, NALU_HYPRE_Int start, NALU_HYPRE_Int end,
                                 NALU_HYPRE_Int *degree);
NALU_HYPRE_Int hypre_ILULocalRCMReverse(NALU_HYPRE_Int *perm, NALU_HYPRE_Int start, NALU_HYPRE_Int end);
// Newton-Schultz-Hotelling (NSH) functions
void * hypre_NSHCreate();
NALU_HYPRE_Int hypre_NSHDestroy( void *data );
NALU_HYPRE_Int hypre_NSHSetup( void *nsh_vdata, hypre_ParCSRMatrix *A, hypre_ParVector *f,
                          hypre_ParVector *u );
NALU_HYPRE_Int hypre_NSHSolve( void *nsh_vdata, hypre_ParCSRMatrix *A, hypre_ParVector *f,
                          hypre_ParVector *u );
NALU_HYPRE_Int hypre_NSHWriteSolverParams(void *nsh_vdata);
NALU_HYPRE_Int hypre_NSHSetPrintLevel( void *nsh_vdata, NALU_HYPRE_Int print_level );
NALU_HYPRE_Int hypre_NSHSetLogging( void *nsh_vdata, NALU_HYPRE_Int logging );
NALU_HYPRE_Int hypre_NSHSetMaxIter( void *nsh_vdata, NALU_HYPRE_Int max_iter );
NALU_HYPRE_Int hypre_NSHSetTol( void *nsh_vdata, NALU_HYPRE_Real tol );
NALU_HYPRE_Int hypre_NSHSetGlobalSolver( void *nsh_vdata, NALU_HYPRE_Int global_solver );
NALU_HYPRE_Int hypre_NSHSetDropThreshold( void *nsh_vdata, NALU_HYPRE_Real droptol );
NALU_HYPRE_Int hypre_NSHSetDropThresholdArray( void *nsh_vdata, NALU_HYPRE_Real *droptol );
NALU_HYPRE_Int hypre_NSHSetMRMaxIter( void *nsh_vdata, NALU_HYPRE_Int mr_max_iter );
NALU_HYPRE_Int hypre_NSHSetMRTol( void *nsh_vdata, NALU_HYPRE_Real mr_tol );
NALU_HYPRE_Int hypre_NSHSetMRMaxRowNnz( void *nsh_vdata, NALU_HYPRE_Int mr_max_row_nnz );
NALU_HYPRE_Int hypre_NSHSetColVersion( void *nsh_vdata, NALU_HYPRE_Int mr_col_version );
NALU_HYPRE_Int hypre_NSHSetNSHMaxIter( void *nsh_vdata, NALU_HYPRE_Int nsh_max_iter );
NALU_HYPRE_Int hypre_NSHSetNSHTol( void *nsh_vdata, NALU_HYPRE_Real nsh_tol );
NALU_HYPRE_Int hypre_NSHSetNSHMaxRowNnz( void *nsh_vdata, NALU_HYPRE_Int nsh_max_row_nnz );
NALU_HYPRE_Int hypre_CSRMatrixNormFro(hypre_CSRMatrix *A, NALU_HYPRE_Real *norm_io);
NALU_HYPRE_Int hypre_CSRMatrixResNormFro(hypre_CSRMatrix *A, NALU_HYPRE_Real *norm_io);
NALU_HYPRE_Int hypre_ParCSRMatrixNormFro(hypre_ParCSRMatrix *A, NALU_HYPRE_Real *norm_io);
NALU_HYPRE_Int hypre_ParCSRMatrixResNormFro(hypre_ParCSRMatrix *A, NALU_HYPRE_Real *norm_io);
NALU_HYPRE_Int hypre_CSRMatrixTrace(hypre_CSRMatrix *A, NALU_HYPRE_Real *trace_io);
NALU_HYPRE_Int hypre_CSRMatrixDropInplace(hypre_CSRMatrix *A, NALU_HYPRE_Real droptol, NALU_HYPRE_Int max_row_nnz);
NALU_HYPRE_Int hypre_ILUCSRMatrixInverseSelfPrecondMRGlobal(hypre_CSRMatrix *matA, hypre_CSRMatrix **M,
                                                       NALU_HYPRE_Real droptol, NALU_HYPRE_Real tol, NALU_HYPRE_Real eps_tol, NALU_HYPRE_Int max_row_nnz, NALU_HYPRE_Int max_iter,
                                                       NALU_HYPRE_Int print_level );
NALU_HYPRE_Int hypre_ILUParCSRInverseNSH(hypre_ParCSRMatrix *A, hypre_ParCSRMatrix **M,
                                    NALU_HYPRE_Real *droptol, NALU_HYPRE_Real mr_tol, NALU_HYPRE_Real nsh_tol, NALU_HYPRE_Real eps_tol,
                                    NALU_HYPRE_Int mr_max_row_nnz, NALU_HYPRE_Int nsh_max_row_nnz, NALU_HYPRE_Int mr_max_iter, NALU_HYPRE_Int nsh_max_iter,
                                    NALU_HYPRE_Int mr_col_version, NALU_HYPRE_Int print_level);
NALU_HYPRE_Int hypre_NSHSolveInverse(hypre_ParCSRMatrix *A, hypre_ParVector *f, hypre_ParVector *u,
                                hypre_ParCSRMatrix *M, hypre_ParVector *ftemp, hypre_ParVector *utemp);
// Accessor functions
NALU_HYPRE_Int hypre_ILUGetNumIterations( void *ilu_vdata, NALU_HYPRE_Int *num_iterations );
NALU_HYPRE_Int hypre_ILUGetFinalRelativeResidualNorm( void *ilu_vdata, NALU_HYPRE_Real *res_norm );

/* par_amgdd.c */
void *hypre_BoomerAMGDDCreate ( void );
NALU_HYPRE_Int hypre_BoomerAMGDDDestroy ( void *data );
NALU_HYPRE_Int hypre_BoomerAMGDDSetStartLevel ( void *data, NALU_HYPRE_Int start_level );
NALU_HYPRE_Int hypre_BoomerAMGDDGetStartLevel ( void *data, NALU_HYPRE_Int *start_level );
NALU_HYPRE_Int hypre_BoomerAMGDDSetFACNumCycles ( void *data, NALU_HYPRE_Int fac_num_cycles );
NALU_HYPRE_Int hypre_BoomerAMGDDGetFACNumCycles ( void *data, NALU_HYPRE_Int *fac_num_cycles );
NALU_HYPRE_Int hypre_BoomerAMGDDSetFACCycleType ( void *data, NALU_HYPRE_Int fac_cycle_type );
NALU_HYPRE_Int hypre_BoomerAMGDDGetFACCycleType ( void *data, NALU_HYPRE_Int *fac_cycle_type );
NALU_HYPRE_Int hypre_BoomerAMGDDSetFACNumRelax ( void *data, NALU_HYPRE_Int fac_num_relax );
NALU_HYPRE_Int hypre_BoomerAMGDDGetFACNumRelax ( void *data, NALU_HYPRE_Int *fac_num_relax );
NALU_HYPRE_Int hypre_BoomerAMGDDSetFACRelaxType ( void *data, NALU_HYPRE_Int fac_relax_type );
NALU_HYPRE_Int hypre_BoomerAMGDDGetFACRelaxType ( void *data, NALU_HYPRE_Int *fac_relax_type );
NALU_HYPRE_Int hypre_BoomerAMGDDSetFACRelaxWeight ( void *data, NALU_HYPRE_Real fac_relax_weight );
NALU_HYPRE_Int hypre_BoomerAMGDDGetFACRelaxWeight ( void *data, NALU_HYPRE_Real *fac_relax_weight );
NALU_HYPRE_Int hypre_BoomerAMGDDSetPadding ( void *data, NALU_HYPRE_Int padding );
NALU_HYPRE_Int hypre_BoomerAMGDDGetPadding ( void *data, NALU_HYPRE_Int *padding );
NALU_HYPRE_Int hypre_BoomerAMGDDSetNumGhostLayers ( void *data, NALU_HYPRE_Int num_ghost_layers );
NALU_HYPRE_Int hypre_BoomerAMGDDGetNumGhostLayers ( void *data, NALU_HYPRE_Int *num_ghost_layers );
NALU_HYPRE_Int hypre_BoomerAMGDDSetUserFACRelaxation( void *data,
                                                 NALU_HYPRE_Int (*userFACRelaxation)( void *amgdd_vdata, NALU_HYPRE_Int level, NALU_HYPRE_Int cycle_param ) );
NALU_HYPRE_Int hypre_BoomerAMGDDGetAMG ( void *data, void **amg_solver );

/* par_amgdd_solve.c */
NALU_HYPRE_Int hypre_BoomerAMGDDSolve ( void *solver, hypre_ParCSRMatrix *A, hypre_ParVector *b,
                                   hypre_ParVector *x );
NALU_HYPRE_Int hypre_BoomerAMGDD_Cycle ( hypre_ParAMGDDData *amgdd_data );
NALU_HYPRE_Int hypre_BoomerAMGDD_ResidualCommunication ( hypre_ParAMGDDData *amgdd_data );
NALU_HYPRE_Complex* hypre_BoomerAMGDD_PackResidualBuffer ( hypre_AMGDDCompGrid **compGrid,
                                                      hypre_AMGDDCommPkg *compGridCommPkg, NALU_HYPRE_Int current_level, NALU_HYPRE_Int proc );
NALU_HYPRE_Int hypre_BoomerAMGDD_UnpackResidualBuffer ( NALU_HYPRE_Complex *buffer,
                                                   hypre_AMGDDCompGrid **compGrid, hypre_AMGDDCommPkg *compGridCommPkg, NALU_HYPRE_Int current_level,
                                                   NALU_HYPRE_Int proc );

/* par_amgdd_setup.c */
NALU_HYPRE_Int hypre_BoomerAMGDDSetup ( void *amgdd_vdata, hypre_ParCSRMatrix *A, hypre_ParVector *b,
                                   hypre_ParVector *x );

/* par_amgdd_fac_cycle.c */
NALU_HYPRE_Int hypre_BoomerAMGDD_FAC ( void *amgdd_vdata, NALU_HYPRE_Int first_iteration );
NALU_HYPRE_Int hypre_BoomerAMGDD_FAC_Cycle ( void *amgdd_vdata, NALU_HYPRE_Int level, NALU_HYPRE_Int cycle_type,
                                        NALU_HYPRE_Int first_iteration );
NALU_HYPRE_Int hypre_BoomerAMGDD_FAC_FCycle ( void *amgdd_vdata, NALU_HYPRE_Int first_iteration );
NALU_HYPRE_Int hypre_BoomerAMGDD_FAC_Interpolate ( hypre_AMGDDCompGrid *compGrid_f,
                                              hypre_AMGDDCompGrid *compGrid_c );
NALU_HYPRE_Int hypre_BoomerAMGDD_FAC_Restrict ( hypre_AMGDDCompGrid *compGrid_f,
                                           hypre_AMGDDCompGrid *compGrid_c, NALU_HYPRE_Int first_iteration );
NALU_HYPRE_Int hypre_BoomerAMGDD_FAC_Relax ( void *amgdd_vdata, NALU_HYPRE_Int level, NALU_HYPRE_Int cycle_param );
NALU_HYPRE_Int hypre_BoomerAMGDD_FAC_CFL1JacobiHost ( void *amgdd_vdata, NALU_HYPRE_Int level,
                                                 NALU_HYPRE_Int relax_set );
NALU_HYPRE_Int hypre_BoomerAMGDD_FAC_Jacobi ( void *amgdd_vdata, NALU_HYPRE_Int level,
                                         NALU_HYPRE_Int cycle_param );
NALU_HYPRE_Int hypre_BoomerAMGDD_FAC_JacobiHost ( void *amgdd_vdata, NALU_HYPRE_Int level );
NALU_HYPRE_Int hypre_BoomerAMGDD_FAC_GaussSeidel ( void *amgdd_vdata, NALU_HYPRE_Int level,
                                              NALU_HYPRE_Int cycle_param );
NALU_HYPRE_Int hypre_BoomerAMGDD_FAC_CFL1Jacobi ( void *amgdd_vdata, NALU_HYPRE_Int level,
                                             NALU_HYPRE_Int cycle_param );
NALU_HYPRE_Int hypre_BoomerAMGDD_FAC_OrderedGaussSeidel ( void *amgdd_vdata, NALU_HYPRE_Int level,
                                                     NALU_HYPRE_Int cycle_param );

/* par_amgdd_fac_cycles_device.c */
NALU_HYPRE_Int hypre_BoomerAMGDD_FAC_JacobiDevice ( void *amgdd_vdata, NALU_HYPRE_Int level );
NALU_HYPRE_Int hypre_BoomerAMGDD_FAC_CFL1JacobiDevice ( void *amgdd_vdata, NALU_HYPRE_Int level,
                                                   NALU_HYPRE_Int relax_set );

/* par_amgdd_comp_grid.c */
hypre_AMGDDCompGridMatrix* hypre_AMGDDCompGridMatrixCreate();
NALU_HYPRE_Int hypre_AMGDDCompGridMatrixDestroy ( hypre_AMGDDCompGridMatrix *matrix );
NALU_HYPRE_Int hypre_AMGDDCompGridMatvec ( NALU_HYPRE_Complex alpha, hypre_AMGDDCompGridMatrix *A,
                                      hypre_AMGDDCompGridVector *x, NALU_HYPRE_Complex beta, hypre_AMGDDCompGridVector *y );
NALU_HYPRE_Int hypre_AMGDDCompGridRealMatvec ( NALU_HYPRE_Complex alpha, hypre_AMGDDCompGridMatrix *A,
                                          hypre_AMGDDCompGridVector *x, NALU_HYPRE_Complex beta, hypre_AMGDDCompGridVector *y );
hypre_AMGDDCompGridVector* hypre_AMGDDCompGridVectorCreate();
NALU_HYPRE_Int hypre_AMGDDCompGridVectorInitialize ( hypre_AMGDDCompGridVector *vector,
                                                NALU_HYPRE_Int num_owned, NALU_HYPRE_Int num_nonowned, NALU_HYPRE_Int num_real );
NALU_HYPRE_Int hypre_AMGDDCompGridVectorDestroy ( hypre_AMGDDCompGridVector *vector );
NALU_HYPRE_Real hypre_AMGDDCompGridVectorInnerProd ( hypre_AMGDDCompGridVector *x,
                                                hypre_AMGDDCompGridVector *y );
NALU_HYPRE_Real hypre_AMGDDCompGridVectorRealInnerProd ( hypre_AMGDDCompGridVector *x,
                                                    hypre_AMGDDCompGridVector *y );
NALU_HYPRE_Int hypre_AMGDDCompGridVectorScale ( NALU_HYPRE_Complex alpha, hypre_AMGDDCompGridVector *x );
NALU_HYPRE_Int hypre_AMGDDCompGridVectorRealScale ( NALU_HYPRE_Complex alpha, hypre_AMGDDCompGridVector *x );
NALU_HYPRE_Int hypre_AMGDDCompGridVectorAxpy ( NALU_HYPRE_Complex alpha, hypre_AMGDDCompGridVector *x,
                                          hypre_AMGDDCompGridVector *y );
NALU_HYPRE_Int hypre_AMGDDCompGridVectorRealAxpy ( NALU_HYPRE_Complex alpha, hypre_AMGDDCompGridVector *x,
                                              hypre_AMGDDCompGridVector *y );
NALU_HYPRE_Int hypre_AMGDDCompGridVectorSetConstantValues ( hypre_AMGDDCompGridVector *vector,
                                                       NALU_HYPRE_Complex value );
NALU_HYPRE_Int hypre_AMGDDCompGridVectorRealSetConstantValues ( hypre_AMGDDCompGridVector *vector,
                                                           NALU_HYPRE_Complex value );
NALU_HYPRE_Int hypre_AMGDDCompGridVectorCopy ( hypre_AMGDDCompGridVector *x,
                                          hypre_AMGDDCompGridVector *y );
NALU_HYPRE_Int hypre_AMGDDCompGridVectorRealCopy ( hypre_AMGDDCompGridVector *x,
                                              hypre_AMGDDCompGridVector *y );
hypre_AMGDDCompGrid *hypre_AMGDDCompGridCreate();
NALU_HYPRE_Int hypre_AMGDDCompGridDestroy ( hypre_AMGDDCompGrid *compGrid );
NALU_HYPRE_Int hypre_AMGDDCompGridInitialize ( hypre_ParAMGDDData *amgdd_data, NALU_HYPRE_Int padding,
                                          NALU_HYPRE_Int level );
NALU_HYPRE_Int hypre_AMGDDCompGridSetupRelax ( hypre_ParAMGDDData *amgdd_data );
NALU_HYPRE_Int hypre_AMGDDCompGridFinalize ( hypre_ParAMGDDData *amgdd_data );
NALU_HYPRE_Int hypre_AMGDDCompGridSetupRealDofMarker ( hypre_AMGDDCompGrid **compGrid,
                                                  NALU_HYPRE_Int num_levels, NALU_HYPRE_Int num_ghost_layers );
NALU_HYPRE_Int hypre_AMGDDCompGridResize ( hypre_AMGDDCompGrid *compGrid, NALU_HYPRE_Int new_size,
                                      NALU_HYPRE_Int need_coarse_info );
NALU_HYPRE_Int hypre_AMGDDCompGridSetupLocalIndices ( hypre_AMGDDCompGrid **compGrid,
                                                 NALU_HYPRE_Int *num_added_nodes, NALU_HYPRE_Int ****recv_map, NALU_HYPRE_Int num_recv_procs,
                                                 NALU_HYPRE_Int **A_tmp_info, NALU_HYPRE_Int start_level, NALU_HYPRE_Int num_levels );
NALU_HYPRE_Int hypre_AMGDDCompGridSetupLocalIndicesP ( hypre_ParAMGDDData *amgdd_data );
hypre_AMGDDCommPkg *hypre_AMGDDCommPkgCreate ( NALU_HYPRE_Int num_levels );
NALU_HYPRE_Int hypre_AMGDDCommPkgSendLevelDestroy ( hypre_AMGDDCommPkg *amgddCommPkg, NALU_HYPRE_Int level,
                                               NALU_HYPRE_Int proc );
NALU_HYPRE_Int hypre_AMGDDCommPkgRecvLevelDestroy ( hypre_AMGDDCommPkg *amgddCommPkg, NALU_HYPRE_Int level,
                                               NALU_HYPRE_Int proc );
NALU_HYPRE_Int hypre_AMGDDCommPkgDestroy ( hypre_AMGDDCommPkg *compGridCommPkg );
NALU_HYPRE_Int hypre_AMGDDCommPkgFinalize ( hypre_ParAMGData* amg_data,
                                       hypre_AMGDDCommPkg *compGridCommPkg, hypre_AMGDDCompGrid **compGrid );

/* par_amgdd_helpers.c */
NALU_HYPRE_Int hypre_BoomerAMGDD_SetupNearestProcessorNeighbors ( hypre_ParCSRMatrix *A,
                                                             hypre_AMGDDCommPkg *compGridCommPkg, NALU_HYPRE_Int level, NALU_HYPRE_Int *padding,
                                                             NALU_HYPRE_Int num_ghost_layers );
NALU_HYPRE_Int hypre_BoomerAMGDD_RecursivelyBuildPsiComposite ( NALU_HYPRE_Int node, NALU_HYPRE_Int m,
                                                           hypre_AMGDDCompGrid *compGrid, NALU_HYPRE_Int *add_flag, NALU_HYPRE_Int use_sort );
NALU_HYPRE_Int hypre_BoomerAMGDD_MarkCoarse ( NALU_HYPRE_Int *list, NALU_HYPRE_Int *marker,
                                         NALU_HYPRE_Int *owned_coarse_indices, NALU_HYPRE_Int *nonowned_coarse_indices, NALU_HYPRE_Int *sort_map,
                                         NALU_HYPRE_Int num_owned, NALU_HYPRE_Int total_num_nodes, NALU_HYPRE_Int num_owned_coarse, NALU_HYPRE_Int list_size,
                                         NALU_HYPRE_Int dist, NALU_HYPRE_Int use_sort, NALU_HYPRE_Int *nodes_to_add );
NALU_HYPRE_Int hypre_BoomerAMGDD_UnpackRecvBuffer ( hypre_ParAMGDDData *amgdd_data,
                                               NALU_HYPRE_Int *recv_buffer, NALU_HYPRE_Int **A_tmp_info, NALU_HYPRE_Int *recv_map_send_buffer_size,
                                               NALU_HYPRE_Int *nodes_added_on_level, NALU_HYPRE_Int current_level, NALU_HYPRE_Int buffer_number );
NALU_HYPRE_Int* hypre_BoomerAMGDD_PackSendBuffer ( hypre_ParAMGDDData *amgdd_data, NALU_HYPRE_Int proc,
                                              NALU_HYPRE_Int current_level, NALU_HYPRE_Int *padding, NALU_HYPRE_Int *send_flag_buffer_size );
NALU_HYPRE_Int hypre_BoomerAMGDD_PackRecvMapSendBuffer ( NALU_HYPRE_Int *recv_map_send_buffer,
                                                    NALU_HYPRE_Int **recv_red_marker, NALU_HYPRE_Int *num_recv_nodes, NALU_HYPRE_Int *recv_buffer_size,
                                                    NALU_HYPRE_Int current_level, NALU_HYPRE_Int num_levels );
NALU_HYPRE_Int hypre_BoomerAMGDD_UnpackSendFlagBuffer ( hypre_AMGDDCompGrid **compGrid,
                                                   NALU_HYPRE_Int *send_flag_buffer, NALU_HYPRE_Int **send_flag, NALU_HYPRE_Int *num_send_nodes,
                                                   NALU_HYPRE_Int *send_buffer_size, NALU_HYPRE_Int current_level, NALU_HYPRE_Int num_levels );
NALU_HYPRE_Int hypre_BoomerAMGDD_CommunicateRemainingMatrixInfo ( hypre_ParAMGDDData* amgdd_data );
NALU_HYPRE_Int hypre_BoomerAMGDD_FixUpRecvMaps ( hypre_AMGDDCompGrid **compGrid,
                                            hypre_AMGDDCommPkg *compGridCommPkg, NALU_HYPRE_Int start_level, NALU_HYPRE_Int num_levels );

/* par_fsai.c */
void* hypre_FSAICreate();
NALU_HYPRE_Int hypre_FSAIDestroy ( void *data );
NALU_HYPRE_Int hypre_FSAISetAlgoType ( void *data, NALU_HYPRE_Int algo_type );
NALU_HYPRE_Int hypre_FSAISetMaxSteps ( void *data, NALU_HYPRE_Int max_steps );
NALU_HYPRE_Int hypre_FSAISetMaxStepSize ( void *data, NALU_HYPRE_Int max_step_size );
NALU_HYPRE_Int hypre_FSAISetKapTolerance ( void *data, NALU_HYPRE_Real kap_tolerance );
NALU_HYPRE_Int hypre_FSAISetMaxIterations ( void *data, NALU_HYPRE_Int max_iterations );
NALU_HYPRE_Int hypre_FSAISetEigMaxIters ( void *data, NALU_HYPRE_Int eig_max_iters );
NALU_HYPRE_Int hypre_FSAISetZeroGuess ( void *data, NALU_HYPRE_Int zero_guess );
NALU_HYPRE_Int hypre_FSAISetTolerance ( void *data, NALU_HYPRE_Real tolerance );
NALU_HYPRE_Int hypre_FSAISetOmega ( void *data, NALU_HYPRE_Real omega );
NALU_HYPRE_Int hypre_FSAISetLogging ( void *data, NALU_HYPRE_Int logging );
NALU_HYPRE_Int hypre_FSAISetNumIterations ( void *data, NALU_HYPRE_Int num_iterations );
NALU_HYPRE_Int hypre_FSAISetPrintLevel ( void *data, NALU_HYPRE_Int print_level );
NALU_HYPRE_Int hypre_FSAIGetAlgoType ( void *data, NALU_HYPRE_Int *algo_type );
NALU_HYPRE_Int hypre_FSAIGetMaxSteps ( void *data, NALU_HYPRE_Int *max_steps );
NALU_HYPRE_Int hypre_FSAIGetMaxStepSize ( void *data, NALU_HYPRE_Int *max_step_size );
NALU_HYPRE_Int hypre_FSAIGetKapTolerance ( void *data, NALU_HYPRE_Real *kap_tolerance );
NALU_HYPRE_Int hypre_FSAIGetMaxIterations ( void *data, NALU_HYPRE_Int *max_iterations );
NALU_HYPRE_Int hypre_FSAIGetEigMaxIters ( void *data, NALU_HYPRE_Int *eig_max_iters );
NALU_HYPRE_Int hypre_FSAIGetZeroGuess ( void *data, NALU_HYPRE_Int *zero_guess );
NALU_HYPRE_Int hypre_FSAIGetTolerance ( void *data, NALU_HYPRE_Real *tolerance );
NALU_HYPRE_Int hypre_FSAIGetOmega ( void *data, NALU_HYPRE_Real *omega );
NALU_HYPRE_Int hypre_FSAIGetLogging ( void *data, NALU_HYPRE_Int *logging );
NALU_HYPRE_Int hypre_FSAIGetNumIterations ( void *data, NALU_HYPRE_Int *num_iterations );
NALU_HYPRE_Int hypre_FSAIGetPrintLevel ( void *data, NALU_HYPRE_Int *print_level );

/* par_fsai_setup.c */
NALU_HYPRE_Int hypre_CSRMatrixExtractDenseMat ( hypre_CSRMatrix *A, hypre_Vector *A_sub,
                                           NALU_HYPRE_Int *S_Pattern, NALU_HYPRE_Int S_nnz, NALU_HYPRE_Int *marker );
NALU_HYPRE_Int hypre_CSRMatrixExtractDenseRow ( hypre_CSRMatrix *A, hypre_Vector *A_subrow,
                                           NALU_HYPRE_Int *marker, NALU_HYPRE_Int row_num );
NALU_HYPRE_Int hypre_FindKapGrad ( hypre_CSRMatrix *A_diag, hypre_Vector *kaporin_gradient,
                              NALU_HYPRE_Int *kap_grad_nonzeros, hypre_Vector *G_temp, NALU_HYPRE_Int *S_Pattern, NALU_HYPRE_Int S_nnz,
                              NALU_HYPRE_Int max_row_size, NALU_HYPRE_Int row_num, NALU_HYPRE_Int *kg_marker );
NALU_HYPRE_Int hypre_AddToPattern ( hypre_Vector *kaporin_gradient, NALU_HYPRE_Int *kap_grad_nonzeros,
                               NALU_HYPRE_Int *S_Pattern, NALU_HYPRE_Int *S_nnz, NALU_HYPRE_Int *kg_marker, NALU_HYPRE_Int max_step_size );
NALU_HYPRE_Int hypre_FSAISetup ( void *fsai_vdata, hypre_ParCSRMatrix *A, hypre_ParVector *f,
                            hypre_ParVector *u );
NALU_HYPRE_Int hypre_FSAIPrintStats ( void *fsai_vdata, hypre_ParCSRMatrix *A );
NALU_HYPRE_Int hypre_FSAIComputeOmega ( void *fsai_vdata, hypre_ParCSRMatrix *A );
void hypre_swap2_ci ( NALU_HYPRE_Complex *v, NALU_HYPRE_Int *w, NALU_HYPRE_Int i, NALU_HYPRE_Int j );
void hypre_qsort2_ci ( NALU_HYPRE_Complex *v, NALU_HYPRE_Int *w, NALU_HYPRE_Int left, NALU_HYPRE_Int right );

/* par_fsai_solve.c */
NALU_HYPRE_Int hypre_FSAISolve ( void *fsai_vdata, hypre_ParCSRMatrix *A, hypre_ParVector *b,
                            hypre_ParVector *x );
