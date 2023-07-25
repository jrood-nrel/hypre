/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/* coarsen.c */
NALU_HYPRE_Int nalu_hypre_StructMapFineToCoarse ( nalu_hypre_Index findex, nalu_hypre_Index index, nalu_hypre_Index stride,
                                        nalu_hypre_Index cindex );
NALU_HYPRE_Int nalu_hypre_StructMapCoarseToFine ( nalu_hypre_Index cindex, nalu_hypre_Index index, nalu_hypre_Index stride,
                                        nalu_hypre_Index findex );
NALU_HYPRE_Int nalu_hypre_StructCoarsen ( nalu_hypre_StructGrid *fgrid, nalu_hypre_Index index, nalu_hypre_Index stride,
                                NALU_HYPRE_Int prune, nalu_hypre_StructGrid **cgrid_ptr );

/* cyclic_reduction.c */
void *nalu_hypre_CyclicReductionCreate ( MPI_Comm comm );
nalu_hypre_StructMatrix *nalu_hypre_CycRedCreateCoarseOp ( nalu_hypre_StructMatrix *A,
                                                 nalu_hypre_StructGrid *coarse_grid, NALU_HYPRE_Int cdir );
NALU_HYPRE_Int nalu_hypre_CycRedSetupCoarseOp ( nalu_hypre_StructMatrix *A, nalu_hypre_StructMatrix *Ac,
                                      nalu_hypre_Index cindex, nalu_hypre_Index cstride, NALU_HYPRE_Int cdir );
NALU_HYPRE_Int nalu_hypre_CyclicReductionSetup ( void *cyc_red_vdata, nalu_hypre_StructMatrix *A,
                                       nalu_hypre_StructVector *b, nalu_hypre_StructVector *x );
NALU_HYPRE_Int nalu_hypre_CyclicReduction ( void *cyc_red_vdata, nalu_hypre_StructMatrix *A, nalu_hypre_StructVector *b,
                                  nalu_hypre_StructVector *x );
NALU_HYPRE_Int nalu_hypre_CyclicReductionSetCDir ( void *cyc_red_vdata, NALU_HYPRE_Int cdir );
NALU_HYPRE_Int nalu_hypre_CyclicReductionSetBase ( void *cyc_red_vdata, nalu_hypre_Index base_index,
                                         nalu_hypre_Index base_stride );
NALU_HYPRE_Int nalu_hypre_CyclicReductionDestroy ( void *cyc_red_vdata );
NALU_HYPRE_Int nalu_hypre_CyclicReductionSetMaxLevel( void   *cyc_red_vdata, NALU_HYPRE_Int   max_level  );

/* general.c */
NALU_HYPRE_Int nalu_hypre_Log2 ( NALU_HYPRE_Int p );

/* hybrid.c */
void *nalu_hypre_HybridCreate ( MPI_Comm comm );
NALU_HYPRE_Int nalu_hypre_HybridDestroy ( void *hybrid_vdata );
NALU_HYPRE_Int nalu_hypre_HybridSetTol ( void *hybrid_vdata, NALU_HYPRE_Real tol );
NALU_HYPRE_Int nalu_hypre_HybridSetConvergenceTol ( void *hybrid_vdata, NALU_HYPRE_Real cf_tol );
NALU_HYPRE_Int nalu_hypre_HybridSetDSCGMaxIter ( void *hybrid_vdata, NALU_HYPRE_Int dscg_max_its );
NALU_HYPRE_Int nalu_hypre_HybridSetPCGMaxIter ( void *hybrid_vdata, NALU_HYPRE_Int pcg_max_its );
NALU_HYPRE_Int nalu_hypre_HybridSetPCGAbsoluteTolFactor ( void *hybrid_vdata, NALU_HYPRE_Real pcg_atolf );
NALU_HYPRE_Int nalu_hypre_HybridSetTwoNorm ( void *hybrid_vdata, NALU_HYPRE_Int two_norm );
NALU_HYPRE_Int nalu_hypre_HybridSetStopCrit ( void *hybrid_vdata, NALU_HYPRE_Int stop_crit );
NALU_HYPRE_Int nalu_hypre_HybridSetRelChange ( void *hybrid_vdata, NALU_HYPRE_Int rel_change );
NALU_HYPRE_Int nalu_hypre_HybridSetSolverType ( void *hybrid_vdata, NALU_HYPRE_Int solver_type );
NALU_HYPRE_Int nalu_hypre_HybridSetRecomputeResidual( void *hybrid_vdata, NALU_HYPRE_Int recompute_residual );
NALU_HYPRE_Int nalu_hypre_HybridGetRecomputeResidual( void *hybrid_vdata, NALU_HYPRE_Int *recompute_residual );
NALU_HYPRE_Int nalu_hypre_HybridSetRecomputeResidualP( void *hybrid_vdata, NALU_HYPRE_Int recompute_residual_p );
NALU_HYPRE_Int nalu_hypre_HybridGetRecomputeResidualP( void *hybrid_vdata, NALU_HYPRE_Int *recompute_residual_p );
NALU_HYPRE_Int nalu_hypre_HybridSetKDim ( void *hybrid_vdata, NALU_HYPRE_Int k_dim );
NALU_HYPRE_Int nalu_hypre_HybridSetPrecond ( void *pcg_vdata, NALU_HYPRE_Int (*pcg_precond_solve )(void*, void*,
                                                                                    void*, void*), NALU_HYPRE_Int (*pcg_precond_setup )(void*, void*, void*, void*), void *pcg_precond );
NALU_HYPRE_Int nalu_hypre_HybridSetLogging ( void *hybrid_vdata, NALU_HYPRE_Int logging );
NALU_HYPRE_Int nalu_hypre_HybridSetPrintLevel ( void *hybrid_vdata, NALU_HYPRE_Int print_level );
NALU_HYPRE_Int nalu_hypre_HybridGetNumIterations ( void *hybrid_vdata, NALU_HYPRE_Int *num_its );
NALU_HYPRE_Int nalu_hypre_HybridGetDSCGNumIterations ( void *hybrid_vdata, NALU_HYPRE_Int *dscg_num_its );
NALU_HYPRE_Int nalu_hypre_HybridGetPCGNumIterations ( void *hybrid_vdata, NALU_HYPRE_Int *pcg_num_its );
NALU_HYPRE_Int nalu_hypre_HybridGetFinalRelativeResidualNorm ( void *hybrid_vdata,
                                                     NALU_HYPRE_Real *final_rel_res_norm );
NALU_HYPRE_Int nalu_hypre_HybridSetup ( void *hybrid_vdata, nalu_hypre_StructMatrix *A, nalu_hypre_StructVector *b,
                              nalu_hypre_StructVector *x );
NALU_HYPRE_Int nalu_hypre_HybridSolve ( void *hybrid_vdata, nalu_hypre_StructMatrix *A, nalu_hypre_StructVector *b,
                              nalu_hypre_StructVector *x );

/* NALU_HYPRE_struct_int.c */
NALU_HYPRE_Int nalu_hypre_StructVectorSetRandomValues ( nalu_hypre_StructVector *vector, NALU_HYPRE_Int seed );
NALU_HYPRE_Int nalu_hypre_StructSetRandomValues ( void *v, NALU_HYPRE_Int seed );

/* NALU_HYPRE_struct_pfmg.c */
NALU_HYPRE_Int nalu_hypre_PFMGSetDeviceLevel( void *pfmg_vdata, NALU_HYPRE_Int   device_level  );

/* jacobi.c */
void *nalu_hypre_JacobiCreate ( MPI_Comm comm );
NALU_HYPRE_Int nalu_hypre_JacobiDestroy ( void *jacobi_vdata );
NALU_HYPRE_Int nalu_hypre_JacobiSetup ( void *jacobi_vdata, nalu_hypre_StructMatrix *A, nalu_hypre_StructVector *b,
                              nalu_hypre_StructVector *x );
NALU_HYPRE_Int nalu_hypre_JacobiSolve ( void *jacobi_vdata, nalu_hypre_StructMatrix *A, nalu_hypre_StructVector *b,
                              nalu_hypre_StructVector *x );
NALU_HYPRE_Int nalu_hypre_JacobiSetTol ( void *jacobi_vdata, NALU_HYPRE_Real tol );
NALU_HYPRE_Int nalu_hypre_JacobiGetTol ( void *jacobi_vdata, NALU_HYPRE_Real *tol );
NALU_HYPRE_Int nalu_hypre_JacobiSetMaxIter ( void *jacobi_vdata, NALU_HYPRE_Int max_iter );
NALU_HYPRE_Int nalu_hypre_JacobiGetMaxIter ( void *jacobi_vdata, NALU_HYPRE_Int *max_iter );
NALU_HYPRE_Int nalu_hypre_JacobiSetZeroGuess ( void *jacobi_vdata, NALU_HYPRE_Int zero_guess );
NALU_HYPRE_Int nalu_hypre_JacobiGetZeroGuess ( void *jacobi_vdata, NALU_HYPRE_Int *zero_guess );
NALU_HYPRE_Int nalu_hypre_JacobiGetNumIterations ( void *jacobi_vdata, NALU_HYPRE_Int *num_iterations );
NALU_HYPRE_Int nalu_hypre_JacobiSetTempVec ( void *jacobi_vdata, nalu_hypre_StructVector *t );
NALU_HYPRE_Int nalu_hypre_JacobiGetFinalRelativeResidualNorm ( void *jacobi_vdata, NALU_HYPRE_Real *norm );

/* pcg_struct.c */
void *nalu_hypre_StructKrylovCAlloc ( size_t count, size_t elt_size, NALU_HYPRE_MemoryLocation location );
NALU_HYPRE_Int nalu_hypre_StructKrylovFree ( void *ptr );
void *nalu_hypre_StructKrylovCreateVector ( void *vvector );
void *nalu_hypre_StructKrylovCreateVectorArray ( NALU_HYPRE_Int n, void *vvector );
NALU_HYPRE_Int nalu_hypre_StructKrylovDestroyVector ( void *vvector );
void *nalu_hypre_StructKrylovMatvecCreate ( void *A, void *x );
NALU_HYPRE_Int nalu_hypre_StructKrylovMatvec ( void *matvec_data, NALU_HYPRE_Complex alpha, void *A, void *x,
                                     NALU_HYPRE_Complex beta, void *y );
NALU_HYPRE_Int nalu_hypre_StructKrylovMatvecDestroy ( void *matvec_data );
NALU_HYPRE_Real nalu_hypre_StructKrylovInnerProd ( void *x, void *y );
NALU_HYPRE_Int nalu_hypre_StructKrylovCopyVector ( void *x, void *y );
NALU_HYPRE_Int nalu_hypre_StructKrylovClearVector ( void *x );
NALU_HYPRE_Int nalu_hypre_StructKrylovScaleVector ( NALU_HYPRE_Complex alpha, void *x );
NALU_HYPRE_Int nalu_hypre_StructKrylovAxpy ( NALU_HYPRE_Complex alpha, void *x, void *y );
NALU_HYPRE_Int nalu_hypre_StructKrylovIdentitySetup ( void *vdata, void *A, void *b, void *x );
NALU_HYPRE_Int nalu_hypre_StructKrylovIdentity ( void *vdata, void *A, void *b, void *x );
NALU_HYPRE_Int nalu_hypre_StructKrylovCommInfo ( void *A, NALU_HYPRE_Int *my_id, NALU_HYPRE_Int *num_procs );

/* pfmg2_setup_rap.c */
nalu_hypre_StructMatrix *nalu_hypre_PFMG2CreateRAPOp ( nalu_hypre_StructMatrix *R, nalu_hypre_StructMatrix *A,
                                             nalu_hypre_StructMatrix *P, nalu_hypre_StructGrid *coarse_grid, NALU_HYPRE_Int cdir );
NALU_HYPRE_Int nalu_hypre_PFMG2BuildRAPSym ( nalu_hypre_StructMatrix *A, nalu_hypre_StructMatrix *P,
                                   nalu_hypre_StructMatrix *R, NALU_HYPRE_Int cdir, nalu_hypre_Index cindex, nalu_hypre_Index cstride,
                                   nalu_hypre_StructMatrix *RAP );
NALU_HYPRE_Int nalu_hypre_PFMG2BuildRAPSym_onebox_FSS5_CC0 ( NALU_HYPRE_Int ci, NALU_HYPRE_Int fi,
                                                   nalu_hypre_StructMatrix *A, nalu_hypre_StructMatrix *P, nalu_hypre_StructMatrix *R, NALU_HYPRE_Int cdir,
                                                   nalu_hypre_Index cindex, nalu_hypre_Index cstride, nalu_hypre_StructMatrix *RAP );
NALU_HYPRE_Int nalu_hypre_PFMG2BuildRAPSym_onebox_FSS5_CC1 ( NALU_HYPRE_Int ci, NALU_HYPRE_Int fi,
                                                   nalu_hypre_StructMatrix *A, nalu_hypre_StructMatrix *P, nalu_hypre_StructMatrix *R, NALU_HYPRE_Int cdir,
                                                   nalu_hypre_Index cindex, nalu_hypre_Index cstride, nalu_hypre_StructMatrix *RAP );
NALU_HYPRE_Int nalu_hypre_PFMG2BuildRAPSym_onebox_FSS9_CC0 ( NALU_HYPRE_Int ci, NALU_HYPRE_Int fi,
                                                   nalu_hypre_StructMatrix *A, nalu_hypre_StructMatrix *P, nalu_hypre_StructMatrix *R, NALU_HYPRE_Int cdir,
                                                   nalu_hypre_Index cindex, nalu_hypre_Index cstride, nalu_hypre_StructMatrix *RAP );
NALU_HYPRE_Int nalu_hypre_PFMG2BuildRAPSym_onebox_FSS9_CC1 ( NALU_HYPRE_Int ci, NALU_HYPRE_Int fi,
                                                   nalu_hypre_StructMatrix *A, nalu_hypre_StructMatrix *P, nalu_hypre_StructMatrix *R, NALU_HYPRE_Int cdir,
                                                   nalu_hypre_Index cindex, nalu_hypre_Index cstride, nalu_hypre_StructMatrix *RAP );
NALU_HYPRE_Int nalu_hypre_PFMG2BuildRAPNoSym ( nalu_hypre_StructMatrix *A, nalu_hypre_StructMatrix *P,
                                     nalu_hypre_StructMatrix *R, NALU_HYPRE_Int cdir, nalu_hypre_Index cindex, nalu_hypre_Index cstride,
                                     nalu_hypre_StructMatrix *RAP );
NALU_HYPRE_Int nalu_hypre_PFMG2BuildRAPNoSym_onebox_FSS5_CC0 ( NALU_HYPRE_Int ci, NALU_HYPRE_Int fi,
                                                     nalu_hypre_StructMatrix *A, nalu_hypre_StructMatrix *P, nalu_hypre_StructMatrix *R, NALU_HYPRE_Int cdir,
                                                     nalu_hypre_Index cindex, nalu_hypre_Index cstride, nalu_hypre_StructMatrix *RAP );
NALU_HYPRE_Int nalu_hypre_PFMG2BuildRAPNoSym_onebox_FSS5_CC1 ( NALU_HYPRE_Int ci, NALU_HYPRE_Int fi,
                                                     nalu_hypre_StructMatrix *A, nalu_hypre_StructMatrix *P, nalu_hypre_StructMatrix *R, NALU_HYPRE_Int cdir,
                                                     nalu_hypre_Index cindex, nalu_hypre_Index cstride, nalu_hypre_StructMatrix *RAP );
NALU_HYPRE_Int nalu_hypre_PFMG2BuildRAPNoSym_onebox_FSS9_CC0 ( NALU_HYPRE_Int ci, NALU_HYPRE_Int fi,
                                                     nalu_hypre_StructMatrix *A, nalu_hypre_StructMatrix *P, nalu_hypre_StructMatrix *R, NALU_HYPRE_Int cdir,
                                                     nalu_hypre_Index cindex, nalu_hypre_Index cstride, nalu_hypre_StructMatrix *RAP );
NALU_HYPRE_Int nalu_hypre_PFMG2BuildRAPNoSym_onebox_FSS9_CC1 ( NALU_HYPRE_Int ci, NALU_HYPRE_Int fi,
                                                     nalu_hypre_StructMatrix *A, nalu_hypre_StructMatrix *P, nalu_hypre_StructMatrix *R, NALU_HYPRE_Int cdir,
                                                     nalu_hypre_Index cindex, nalu_hypre_Index cstride, nalu_hypre_StructMatrix *RAP );

/* pfmg3_setup_rap.c */
nalu_hypre_StructMatrix *nalu_hypre_PFMG3CreateRAPOp ( nalu_hypre_StructMatrix *R, nalu_hypre_StructMatrix *A,
                                             nalu_hypre_StructMatrix *P, nalu_hypre_StructGrid *coarse_grid, NALU_HYPRE_Int cdir );
NALU_HYPRE_Int nalu_hypre_PFMG3BuildRAPSym ( nalu_hypre_StructMatrix *A, nalu_hypre_StructMatrix *P,
                                   nalu_hypre_StructMatrix *R, NALU_HYPRE_Int cdir, nalu_hypre_Index cindex, nalu_hypre_Index cstride,
                                   nalu_hypre_StructMatrix *RAP );
NALU_HYPRE_Int nalu_hypre_PFMG3BuildRAPSym_onebox_FSS07_CC0 ( NALU_HYPRE_Int ci, NALU_HYPRE_Int fi,
                                                    nalu_hypre_StructMatrix *A, nalu_hypre_StructMatrix *P, nalu_hypre_StructMatrix *R, NALU_HYPRE_Int cdir,
                                                    nalu_hypre_Index cindex, nalu_hypre_Index cstride, nalu_hypre_StructMatrix *RAP );
NALU_HYPRE_Int nalu_hypre_PFMG3BuildRAPSym_onebox_FSS07_CC1 ( NALU_HYPRE_Int ci, NALU_HYPRE_Int fi,
                                                    nalu_hypre_StructMatrix *A, nalu_hypre_StructMatrix *P, nalu_hypre_StructMatrix *R, NALU_HYPRE_Int cdir,
                                                    nalu_hypre_Index cindex, nalu_hypre_Index cstride, nalu_hypre_StructMatrix *RAP );
NALU_HYPRE_Int nalu_hypre_PFMG3BuildRAPSym_onebox_FSS19_CC0 ( NALU_HYPRE_Int ci, NALU_HYPRE_Int fi,
                                                    nalu_hypre_StructMatrix *A, nalu_hypre_StructMatrix *P, nalu_hypre_StructMatrix *R, NALU_HYPRE_Int cdir,
                                                    nalu_hypre_Index cindex, nalu_hypre_Index cstride, nalu_hypre_StructMatrix *RAP );
NALU_HYPRE_Int nalu_hypre_PFMG3BuildRAPSym_onebox_FSS19_CC1 ( NALU_HYPRE_Int ci, NALU_HYPRE_Int fi,
                                                    nalu_hypre_StructMatrix *A, nalu_hypre_StructMatrix *P, nalu_hypre_StructMatrix *R, NALU_HYPRE_Int cdir,
                                                    nalu_hypre_Index cindex, nalu_hypre_Index cstride, nalu_hypre_StructMatrix *RAP );
NALU_HYPRE_Int nalu_hypre_PFMG3BuildRAPSym_onebox_FSS27_CC0 ( NALU_HYPRE_Int ci, NALU_HYPRE_Int fi,
                                                    nalu_hypre_StructMatrix *A, nalu_hypre_StructMatrix *P, nalu_hypre_StructMatrix *R, NALU_HYPRE_Int cdir,
                                                    nalu_hypre_Index cindex, nalu_hypre_Index cstride, nalu_hypre_StructMatrix *RAP );
NALU_HYPRE_Int nalu_hypre_PFMG3BuildRAPSym_onebox_FSS27_CC1 ( NALU_HYPRE_Int ci, NALU_HYPRE_Int fi,
                                                    nalu_hypre_StructMatrix *A, nalu_hypre_StructMatrix *P, nalu_hypre_StructMatrix *R, NALU_HYPRE_Int cdir,
                                                    nalu_hypre_Index cindex, nalu_hypre_Index cstride, nalu_hypre_StructMatrix *RAP );
NALU_HYPRE_Int nalu_hypre_PFMG3BuildRAPNoSym ( nalu_hypre_StructMatrix *A, nalu_hypre_StructMatrix *P,
                                     nalu_hypre_StructMatrix *R, NALU_HYPRE_Int cdir, nalu_hypre_Index cindex, nalu_hypre_Index cstride,
                                     nalu_hypre_StructMatrix *RAP );
NALU_HYPRE_Int nalu_hypre_PFMG3BuildRAPNoSym_onebox_FSS07_CC0 ( NALU_HYPRE_Int ci, NALU_HYPRE_Int fi,
                                                      nalu_hypre_StructMatrix *A, nalu_hypre_StructMatrix *P, nalu_hypre_StructMatrix *R, NALU_HYPRE_Int cdir,
                                                      nalu_hypre_Index cindex, nalu_hypre_Index cstride, nalu_hypre_StructMatrix *RAP );
NALU_HYPRE_Int nalu_hypre_PFMG3BuildRAPNoSym_onebox_FSS07_CC1 ( NALU_HYPRE_Int ci, NALU_HYPRE_Int fi,
                                                      nalu_hypre_StructMatrix *A, nalu_hypre_StructMatrix *P, nalu_hypre_StructMatrix *R, NALU_HYPRE_Int cdir,
                                                      nalu_hypre_Index cindex, nalu_hypre_Index cstride, nalu_hypre_StructMatrix *RAP );
NALU_HYPRE_Int nalu_hypre_PFMG3BuildRAPNoSym_onebox_FSS19_CC0 ( NALU_HYPRE_Int ci, NALU_HYPRE_Int fi,
                                                      nalu_hypre_StructMatrix *A, nalu_hypre_StructMatrix *P, nalu_hypre_StructMatrix *R, NALU_HYPRE_Int cdir,
                                                      nalu_hypre_Index cindex, nalu_hypre_Index cstride, nalu_hypre_StructMatrix *RAP );
NALU_HYPRE_Int nalu_hypre_PFMG3BuildRAPNoSym_onebox_FSS19_CC1 ( NALU_HYPRE_Int ci, NALU_HYPRE_Int fi,
                                                      nalu_hypre_StructMatrix *A, nalu_hypre_StructMatrix *P, nalu_hypre_StructMatrix *R, NALU_HYPRE_Int cdir,
                                                      nalu_hypre_Index cindex, nalu_hypre_Index cstride, nalu_hypre_StructMatrix *RAP );
NALU_HYPRE_Int nalu_hypre_PFMG3BuildRAPNoSym_onebox_FSS27_CC0 ( NALU_HYPRE_Int ci, NALU_HYPRE_Int fi,
                                                      nalu_hypre_StructMatrix *A, nalu_hypre_StructMatrix *P, nalu_hypre_StructMatrix *R, NALU_HYPRE_Int cdir,
                                                      nalu_hypre_Index cindex, nalu_hypre_Index cstride, nalu_hypre_StructMatrix *RAP );
NALU_HYPRE_Int nalu_hypre_PFMG3BuildRAPNoSym_onebox_FSS27_CC1 ( NALU_HYPRE_Int ci, NALU_HYPRE_Int fi,
                                                      nalu_hypre_StructMatrix *A, nalu_hypre_StructMatrix *P, nalu_hypre_StructMatrix *R, NALU_HYPRE_Int cdir,
                                                      nalu_hypre_Index cindex, nalu_hypre_Index cstride, nalu_hypre_StructMatrix *RAP );

/* pfmg.c */
void *nalu_hypre_PFMGCreate ( MPI_Comm comm );
NALU_HYPRE_Int nalu_hypre_PFMGDestroy ( void *pfmg_vdata );
NALU_HYPRE_Int nalu_hypre_PFMGSetTol ( void *pfmg_vdata, NALU_HYPRE_Real tol );
NALU_HYPRE_Int nalu_hypre_PFMGGetTol ( void *pfmg_vdata, NALU_HYPRE_Real *tol );
NALU_HYPRE_Int nalu_hypre_PFMGSetMaxIter ( void *pfmg_vdata, NALU_HYPRE_Int max_iter );
NALU_HYPRE_Int nalu_hypre_PFMGGetMaxIter ( void *pfmg_vdata, NALU_HYPRE_Int *max_iter );
NALU_HYPRE_Int nalu_hypre_PFMGSetMaxLevels ( void *pfmg_vdata, NALU_HYPRE_Int max_levels );
NALU_HYPRE_Int nalu_hypre_PFMGGetMaxLevels ( void *pfmg_vdata, NALU_HYPRE_Int *max_levels );
NALU_HYPRE_Int nalu_hypre_PFMGSetRelChange ( void *pfmg_vdata, NALU_HYPRE_Int rel_change );
NALU_HYPRE_Int nalu_hypre_PFMGGetRelChange ( void *pfmg_vdata, NALU_HYPRE_Int *rel_change );
NALU_HYPRE_Int nalu_hypre_PFMGSetZeroGuess ( void *pfmg_vdata, NALU_HYPRE_Int zero_guess );
NALU_HYPRE_Int nalu_hypre_PFMGGetZeroGuess ( void *pfmg_vdata, NALU_HYPRE_Int *zero_guess );
NALU_HYPRE_Int nalu_hypre_PFMGSetRelaxType ( void *pfmg_vdata, NALU_HYPRE_Int relax_type );
NALU_HYPRE_Int nalu_hypre_PFMGGetRelaxType ( void *pfmg_vdata, NALU_HYPRE_Int *relax_type );
NALU_HYPRE_Int nalu_hypre_PFMGSetJacobiWeight ( void *pfmg_vdata, NALU_HYPRE_Real weight );
NALU_HYPRE_Int nalu_hypre_PFMGGetJacobiWeight ( void *pfmg_vdata, NALU_HYPRE_Real *weight );
NALU_HYPRE_Int nalu_hypre_PFMGSetRAPType ( void *pfmg_vdata, NALU_HYPRE_Int rap_type );
NALU_HYPRE_Int nalu_hypre_PFMGGetRAPType ( void *pfmg_vdata, NALU_HYPRE_Int *rap_type );
NALU_HYPRE_Int nalu_hypre_PFMGSetNumPreRelax ( void *pfmg_vdata, NALU_HYPRE_Int num_pre_relax );
NALU_HYPRE_Int nalu_hypre_PFMGGetNumPreRelax ( void *pfmg_vdata, NALU_HYPRE_Int *num_pre_relax );
NALU_HYPRE_Int nalu_hypre_PFMGSetNumPostRelax ( void *pfmg_vdata, NALU_HYPRE_Int num_post_relax );
NALU_HYPRE_Int nalu_hypre_PFMGGetNumPostRelax ( void *pfmg_vdata, NALU_HYPRE_Int *num_post_relax );
NALU_HYPRE_Int nalu_hypre_PFMGSetSkipRelax ( void *pfmg_vdata, NALU_HYPRE_Int skip_relax );
NALU_HYPRE_Int nalu_hypre_PFMGGetSkipRelax ( void *pfmg_vdata, NALU_HYPRE_Int *skip_relax );
NALU_HYPRE_Int nalu_hypre_PFMGSetDxyz ( void *pfmg_vdata, NALU_HYPRE_Real *dxyz );
NALU_HYPRE_Int nalu_hypre_PFMGSetLogging ( void *pfmg_vdata, NALU_HYPRE_Int logging );
NALU_HYPRE_Int nalu_hypre_PFMGGetLogging ( void *pfmg_vdata, NALU_HYPRE_Int *logging );
NALU_HYPRE_Int nalu_hypre_PFMGSetPrintLevel ( void *pfmg_vdata, NALU_HYPRE_Int print_level );
NALU_HYPRE_Int nalu_hypre_PFMGGetPrintLevel ( void *pfmg_vdata, NALU_HYPRE_Int *print_level );
NALU_HYPRE_Int nalu_hypre_PFMGGetNumIterations ( void *pfmg_vdata, NALU_HYPRE_Int *num_iterations );
NALU_HYPRE_Int nalu_hypre_PFMGPrintLogging ( void *pfmg_vdata, NALU_HYPRE_Int myid );
NALU_HYPRE_Int nalu_hypre_PFMGGetFinalRelativeResidualNorm ( void *pfmg_vdata,
                                                   NALU_HYPRE_Real *relative_residual_norm );

/* pfmg_relax.c */
void *nalu_hypre_PFMGRelaxCreate ( MPI_Comm comm );
NALU_HYPRE_Int nalu_hypre_PFMGRelaxDestroy ( void *pfmg_relax_vdata );
NALU_HYPRE_Int nalu_hypre_PFMGRelax ( void *pfmg_relax_vdata, nalu_hypre_StructMatrix *A, nalu_hypre_StructVector *b,
                            nalu_hypre_StructVector *x );
NALU_HYPRE_Int nalu_hypre_PFMGRelaxSetup ( void *pfmg_relax_vdata, nalu_hypre_StructMatrix *A,
                                 nalu_hypre_StructVector *b, nalu_hypre_StructVector *x );
NALU_HYPRE_Int nalu_hypre_PFMGRelaxSetType ( void *pfmg_relax_vdata, NALU_HYPRE_Int relax_type );
NALU_HYPRE_Int nalu_hypre_PFMGRelaxSetJacobiWeight ( void *pfmg_relax_vdata, NALU_HYPRE_Real weight );
NALU_HYPRE_Int nalu_hypre_PFMGRelaxSetPreRelax ( void *pfmg_relax_vdata );
NALU_HYPRE_Int nalu_hypre_PFMGRelaxSetPostRelax ( void *pfmg_relax_vdata );
NALU_HYPRE_Int nalu_hypre_PFMGRelaxSetTol ( void *pfmg_relax_vdata, NALU_HYPRE_Real tol );
NALU_HYPRE_Int nalu_hypre_PFMGRelaxSetMaxIter ( void *pfmg_relax_vdata, NALU_HYPRE_Int max_iter );
NALU_HYPRE_Int nalu_hypre_PFMGRelaxSetZeroGuess ( void *pfmg_relax_vdata, NALU_HYPRE_Int zero_guess );
NALU_HYPRE_Int nalu_hypre_PFMGRelaxSetTempVec ( void *pfmg_relax_vdata, nalu_hypre_StructVector *t );

/* pfmg_setup.c */
NALU_HYPRE_Int nalu_hypre_PFMGSetup ( void *pfmg_vdata, nalu_hypre_StructMatrix *A, nalu_hypre_StructVector *b,
                            nalu_hypre_StructVector *x );
NALU_HYPRE_Int nalu_hypre_PFMGComputeDxyz ( nalu_hypre_StructMatrix *A, NALU_HYPRE_Real *dxyz, NALU_HYPRE_Real *mean,
                                  NALU_HYPRE_Real *deviation);
NALU_HYPRE_Int nalu_hypre_PFMGComputeDxyz_CS  ( NALU_HYPRE_Int bi, nalu_hypre_StructMatrix *A, NALU_HYPRE_Real *cxyz,
                                      NALU_HYPRE_Real *sqcxyz);
NALU_HYPRE_Int nalu_hypre_PFMGComputeDxyz_SS5 ( NALU_HYPRE_Int bi, nalu_hypre_StructMatrix *A, NALU_HYPRE_Real *cxyz,
                                      NALU_HYPRE_Real *sqcxyz);
NALU_HYPRE_Int nalu_hypre_PFMGComputeDxyz_SS9 ( NALU_HYPRE_Int bi, nalu_hypre_StructMatrix *A, NALU_HYPRE_Real *cxyz,
                                      NALU_HYPRE_Real *sqcxyz);
NALU_HYPRE_Int nalu_hypre_PFMGComputeDxyz_SS7 ( NALU_HYPRE_Int bi, nalu_hypre_StructMatrix *A, NALU_HYPRE_Real *cxyz,
                                      NALU_HYPRE_Real *sqcxyz);
NALU_HYPRE_Int nalu_hypre_PFMGComputeDxyz_SS19( NALU_HYPRE_Int bi, nalu_hypre_StructMatrix *A, NALU_HYPRE_Real *cxyz,
                                      NALU_HYPRE_Real *sqcxyz);
NALU_HYPRE_Int nalu_hypre_PFMGComputeDxyz_SS27( NALU_HYPRE_Int bi, nalu_hypre_StructMatrix *A, NALU_HYPRE_Real *cxyz,
                                      NALU_HYPRE_Real *sqcxyz);
NALU_HYPRE_Int nalu_hypre_ZeroDiagonal ( nalu_hypre_StructMatrix *A );

/* pfmg_setup_interp.c */
nalu_hypre_StructMatrix *nalu_hypre_PFMGCreateInterpOp ( nalu_hypre_StructMatrix *A, nalu_hypre_StructGrid *cgrid,
                                               NALU_HYPRE_Int cdir, NALU_HYPRE_Int rap_type );
NALU_HYPRE_Int nalu_hypre_PFMGSetupInterpOp ( nalu_hypre_StructMatrix *A, NALU_HYPRE_Int cdir, nalu_hypre_Index findex,
                                    nalu_hypre_Index stride, nalu_hypre_StructMatrix *P, NALU_HYPRE_Int rap_type );
NALU_HYPRE_Int nalu_hypre_PFMGSetupInterpOp_CC0 ( NALU_HYPRE_Int i, nalu_hypre_StructMatrix *A, nalu_hypre_Box *A_dbox,
                                        NALU_HYPRE_Int cdir, nalu_hypre_Index stride, nalu_hypre_Index stridec, nalu_hypre_Index start, nalu_hypre_IndexRef startc,
                                        nalu_hypre_Index loop_size, nalu_hypre_Box *P_dbox, NALU_HYPRE_Int Pstenc0, NALU_HYPRE_Int Pstenc1, NALU_HYPRE_Real *Pp0,
                                        NALU_HYPRE_Real *Pp1, NALU_HYPRE_Int rap_type, NALU_HYPRE_Int si0, NALU_HYPRE_Int si1 );
NALU_HYPRE_Int nalu_hypre_PFMGSetupInterpOp_CC1 ( NALU_HYPRE_Int i, nalu_hypre_StructMatrix *A, nalu_hypre_Box *A_dbox,
                                        NALU_HYPRE_Int cdir, nalu_hypre_Index stride, nalu_hypre_Index stridec, nalu_hypre_Index start, nalu_hypre_IndexRef startc,
                                        nalu_hypre_Index loop_size, nalu_hypre_Box *P_dbox, NALU_HYPRE_Int Pstenc0, NALU_HYPRE_Int Pstenc1, NALU_HYPRE_Real *Pp0,
                                        NALU_HYPRE_Real *Pp1, NALU_HYPRE_Int rap_type, NALU_HYPRE_Int si0, NALU_HYPRE_Int si1 );
NALU_HYPRE_Int nalu_hypre_PFMGSetupInterpOp_CC2 ( NALU_HYPRE_Int i, nalu_hypre_StructMatrix *A, nalu_hypre_Box *A_dbox,
                                        NALU_HYPRE_Int cdir, nalu_hypre_Index stride, nalu_hypre_Index stridec, nalu_hypre_Index start, nalu_hypre_IndexRef startc,
                                        nalu_hypre_Index loop_size, nalu_hypre_Box *P_dbox, NALU_HYPRE_Int Pstenc0, NALU_HYPRE_Int Pstenc1, NALU_HYPRE_Real *Pp0,
                                        NALU_HYPRE_Real *Pp1, NALU_HYPRE_Int rap_type, NALU_HYPRE_Int si0, NALU_HYPRE_Int si1 );
NALU_HYPRE_Int nalu_hypre_PFMGSetupInterpOp_CC0_SS5 ( NALU_HYPRE_Int i, nalu_hypre_StructMatrix *A, nalu_hypre_Box *A_dbox,
                                            NALU_HYPRE_Int cdir, nalu_hypre_Index stride, nalu_hypre_Index stridec, nalu_hypre_Index start, nalu_hypre_IndexRef startc,
                                            nalu_hypre_Index loop_size, nalu_hypre_Box *P_dbox, NALU_HYPRE_Int Pstenc0, NALU_HYPRE_Int Pstenc1, NALU_HYPRE_Real *Pp0,
                                            NALU_HYPRE_Real *Pp1, NALU_HYPRE_Int rap_type, nalu_hypre_Index *P_stencil_shape );
NALU_HYPRE_Int nalu_hypre_PFMGSetupInterpOp_CC0_SS9 ( NALU_HYPRE_Int i, nalu_hypre_StructMatrix *A, nalu_hypre_Box *A_dbox,
                                            NALU_HYPRE_Int cdir, nalu_hypre_Index stride, nalu_hypre_Index stridec, nalu_hypre_Index start, nalu_hypre_IndexRef startc,
                                            nalu_hypre_Index loop_size, nalu_hypre_Box *P_dbox, NALU_HYPRE_Int Pstenc0, NALU_HYPRE_Int Pstenc1, NALU_HYPRE_Real *Pp0,
                                            NALU_HYPRE_Real *Pp1, NALU_HYPRE_Int rap_type, nalu_hypre_Index *P_stencil_shape );
NALU_HYPRE_Int nalu_hypre_PFMGSetupInterpOp_CC0_SS7 ( NALU_HYPRE_Int i, nalu_hypre_StructMatrix *A, nalu_hypre_Box *A_dbox,
                                            NALU_HYPRE_Int cdir, nalu_hypre_Index stride, nalu_hypre_Index stridec, nalu_hypre_Index start, nalu_hypre_IndexRef startc,
                                            nalu_hypre_Index loop_size, nalu_hypre_Box *P_dbox, NALU_HYPRE_Int Pstenc0, NALU_HYPRE_Int Pstenc1, NALU_HYPRE_Real *Pp0,
                                            NALU_HYPRE_Real *Pp1, NALU_HYPRE_Int rap_type, nalu_hypre_Index *P_stencil_shape );

NALU_HYPRE_Int nalu_hypre_PFMGSetupInterpOp_CC0_SS15 ( NALU_HYPRE_Int i, nalu_hypre_StructMatrix *A, nalu_hypre_Box *A_dbox,
                                             NALU_HYPRE_Int cdir, nalu_hypre_Index stride, nalu_hypre_Index stridec, nalu_hypre_Index start, nalu_hypre_IndexRef startc,
                                             nalu_hypre_Index loop_size, nalu_hypre_Box *P_dbox, NALU_HYPRE_Int Pstenc0, NALU_HYPRE_Int Pstenc1, NALU_HYPRE_Real *Pp0,
                                             NALU_HYPRE_Real *Pp1, NALU_HYPRE_Int rap_type, nalu_hypre_Index *P_stencil_shape );

NALU_HYPRE_Int nalu_hypre_PFMGSetupInterpOp_CC0_SS19 ( NALU_HYPRE_Int i, nalu_hypre_StructMatrix *A, nalu_hypre_Box *A_dbox,
                                             NALU_HYPRE_Int cdir, nalu_hypre_Index stride, nalu_hypre_Index stridec, nalu_hypre_Index start, nalu_hypre_IndexRef startc,
                                             nalu_hypre_Index loop_size, nalu_hypre_Box *P_dbox, NALU_HYPRE_Int Pstenc0, NALU_HYPRE_Int Pstenc1, NALU_HYPRE_Real *Pp0,
                                             NALU_HYPRE_Real *Pp1, NALU_HYPRE_Int rap_type, nalu_hypre_Index *P_stencil_shape );

NALU_HYPRE_Int nalu_hypre_PFMGSetupInterpOp_CC0_SS27 ( NALU_HYPRE_Int i, nalu_hypre_StructMatrix *A, nalu_hypre_Box *A_dbox,
                                             NALU_HYPRE_Int cdir, nalu_hypre_Index stride, nalu_hypre_Index stridec, nalu_hypre_Index start, nalu_hypre_IndexRef startc,
                                             nalu_hypre_Index loop_size, nalu_hypre_Box *P_dbox, NALU_HYPRE_Int Pstenc0, NALU_HYPRE_Int Pstenc1, NALU_HYPRE_Real *Pp0,
                                             NALU_HYPRE_Real *Pp1, NALU_HYPRE_Int rap_type, nalu_hypre_Index *P_stencil_shape );
/* pfmg_setup_rap5.c */
nalu_hypre_StructMatrix *nalu_hypre_PFMGCreateCoarseOp5 ( nalu_hypre_StructMatrix *R, nalu_hypre_StructMatrix *A,
                                                nalu_hypre_StructMatrix *P, nalu_hypre_StructGrid *coarse_grid, NALU_HYPRE_Int cdir );
NALU_HYPRE_Int nalu_hypre_PFMGBuildCoarseOp5 ( nalu_hypre_StructMatrix *A, nalu_hypre_StructMatrix *P,
                                     nalu_hypre_StructMatrix *R, NALU_HYPRE_Int cdir, nalu_hypre_Index cindex, nalu_hypre_Index cstride,
                                     nalu_hypre_StructMatrix *RAP );
NALU_HYPRE_Int nalu_hypre_PFMGBuildCoarseOp5_onebox_CC0 ( NALU_HYPRE_Int fi, NALU_HYPRE_Int ci, nalu_hypre_StructMatrix *A,
                                                nalu_hypre_StructMatrix *P, nalu_hypre_StructMatrix *R, NALU_HYPRE_Int cdir, nalu_hypre_Index cindex,
                                                nalu_hypre_Index cstride, nalu_hypre_StructMatrix *RAP );
NALU_HYPRE_Int nalu_hypre_PFMGBuildCoarseOp5_onebox_CC1 ( NALU_HYPRE_Int fi, NALU_HYPRE_Int ci, nalu_hypre_StructMatrix *A,
                                                nalu_hypre_StructMatrix *P, nalu_hypre_StructMatrix *R, NALU_HYPRE_Int cdir, nalu_hypre_Index cindex,
                                                nalu_hypre_Index cstride, nalu_hypre_StructMatrix *RAP );
NALU_HYPRE_Int nalu_hypre_PFMGBuildCoarseOp5_onebox_CC2 ( NALU_HYPRE_Int fi, NALU_HYPRE_Int ci, nalu_hypre_StructMatrix *A,
                                                nalu_hypre_StructMatrix *P, nalu_hypre_StructMatrix *R, NALU_HYPRE_Int cdir, nalu_hypre_Index cindex,
                                                nalu_hypre_Index cstride, nalu_hypre_StructMatrix *RAP );

/* pfmg_setup_rap7.c */
nalu_hypre_StructMatrix *nalu_hypre_PFMGCreateCoarseOp7 ( nalu_hypre_StructMatrix *R, nalu_hypre_StructMatrix *A,
                                                nalu_hypre_StructMatrix *P, nalu_hypre_StructGrid *coarse_grid, NALU_HYPRE_Int cdir );
NALU_HYPRE_Int nalu_hypre_PFMGBuildCoarseOp7 ( nalu_hypre_StructMatrix *A, nalu_hypre_StructMatrix *P,
                                     nalu_hypre_StructMatrix *R, NALU_HYPRE_Int cdir, nalu_hypre_Index cindex, nalu_hypre_Index cstride,
                                     nalu_hypre_StructMatrix *RAP );

/* pfmg_setup_rap.c */
nalu_hypre_StructMatrix *nalu_hypre_PFMGCreateRAPOp ( nalu_hypre_StructMatrix *R, nalu_hypre_StructMatrix *A,
                                            nalu_hypre_StructMatrix *P, nalu_hypre_StructGrid *coarse_grid, NALU_HYPRE_Int cdir, NALU_HYPRE_Int rap_type );
NALU_HYPRE_Int nalu_hypre_PFMGSetupRAPOp ( nalu_hypre_StructMatrix *R, nalu_hypre_StructMatrix *A,
                                 nalu_hypre_StructMatrix *P, NALU_HYPRE_Int cdir, nalu_hypre_Index cindex, nalu_hypre_Index cstride, NALU_HYPRE_Int rap_type,
                                 nalu_hypre_StructMatrix *Ac );

/* pfmg_solve.c */
NALU_HYPRE_Int nalu_hypre_PFMGSolve ( void *pfmg_vdata, nalu_hypre_StructMatrix *A, nalu_hypre_StructVector *b,
                            nalu_hypre_StructVector *x );

/* point_relax.c */
void *nalu_hypre_PointRelaxCreate ( MPI_Comm comm );
NALU_HYPRE_Int nalu_hypre_PointRelaxDestroy ( void *relax_vdata );
NALU_HYPRE_Int nalu_hypre_PointRelaxSetup ( void *relax_vdata, nalu_hypre_StructMatrix *A, nalu_hypre_StructVector *b,
                                  nalu_hypre_StructVector *x );
NALU_HYPRE_Int nalu_hypre_PointRelax ( void *relax_vdata, nalu_hypre_StructMatrix *A, nalu_hypre_StructVector *b,
                             nalu_hypre_StructVector *x );
NALU_HYPRE_Int nalu_hypre_PointRelax_core0 ( void *relax_vdata, nalu_hypre_StructMatrix *A,
                                   NALU_HYPRE_Int constant_coefficient, nalu_hypre_Box *compute_box, NALU_HYPRE_Real *bp, NALU_HYPRE_Real *xp,
                                   NALU_HYPRE_Real *tp, NALU_HYPRE_Int boxarray_id, nalu_hypre_Box *A_data_box, nalu_hypre_Box *b_data_box,
                                   nalu_hypre_Box *x_data_box, nalu_hypre_Box *t_data_box, nalu_hypre_IndexRef stride );
NALU_HYPRE_Int nalu_hypre_PointRelax_core12 ( void *relax_vdata, nalu_hypre_StructMatrix *A,
                                    NALU_HYPRE_Int constant_coefficient, nalu_hypre_Box *compute_box, NALU_HYPRE_Real *bp, NALU_HYPRE_Real *xp,
                                    NALU_HYPRE_Real *tp, NALU_HYPRE_Int boxarray_id, nalu_hypre_Box *A_data_box, nalu_hypre_Box *b_data_box,
                                    nalu_hypre_Box *x_data_box, nalu_hypre_Box *t_data_box, nalu_hypre_IndexRef stride );
NALU_HYPRE_Int nalu_hypre_PointRelaxSetTol ( void *relax_vdata, NALU_HYPRE_Real tol );
NALU_HYPRE_Int nalu_hypre_PointRelaxGetTol ( void *relax_vdata, NALU_HYPRE_Real *tol );
NALU_HYPRE_Int nalu_hypre_PointRelaxSetMaxIter ( void *relax_vdata, NALU_HYPRE_Int max_iter );
NALU_HYPRE_Int nalu_hypre_PointRelaxGetMaxIter ( void *relax_vdata, NALU_HYPRE_Int *max_iter );
NALU_HYPRE_Int nalu_hypre_PointRelaxSetZeroGuess ( void *relax_vdata, NALU_HYPRE_Int zero_guess );
NALU_HYPRE_Int nalu_hypre_PointRelaxGetZeroGuess ( void *relax_vdata, NALU_HYPRE_Int *zero_guess );
NALU_HYPRE_Int nalu_hypre_PointRelaxGetNumIterations ( void *relax_vdata, NALU_HYPRE_Int *num_iterations );
NALU_HYPRE_Int nalu_hypre_PointRelaxSetWeight ( void *relax_vdata, NALU_HYPRE_Real weight );
NALU_HYPRE_Int nalu_hypre_PointRelaxSetNumPointsets ( void *relax_vdata, NALU_HYPRE_Int num_pointsets );
NALU_HYPRE_Int nalu_hypre_PointRelaxSetPointset ( void *relax_vdata, NALU_HYPRE_Int pointset,
                                        NALU_HYPRE_Int pointset_size, nalu_hypre_Index pointset_stride, nalu_hypre_Index *pointset_indices );
NALU_HYPRE_Int nalu_hypre_PointRelaxSetPointsetRank ( void *relax_vdata, NALU_HYPRE_Int pointset,
                                            NALU_HYPRE_Int pointset_rank );
NALU_HYPRE_Int nalu_hypre_PointRelaxSetTempVec ( void *relax_vdata, nalu_hypre_StructVector *t );
NALU_HYPRE_Int nalu_hypre_PointRelaxGetFinalRelativeResidualNorm ( void *relax_vdata, NALU_HYPRE_Real *norm );
NALU_HYPRE_Int nalu_hypre_relax_wtx ( void *relax_vdata, NALU_HYPRE_Int pointset, nalu_hypre_StructVector *t,
                            nalu_hypre_StructVector *x );
NALU_HYPRE_Int nalu_hypre_relax_copy ( void *relax_vdata, NALU_HYPRE_Int pointset, nalu_hypre_StructVector *t,
                             nalu_hypre_StructVector *x );

/* red_black_constantcoef_gs.c */
NALU_HYPRE_Int nalu_hypre_RedBlackConstantCoefGS ( void *relax_vdata, nalu_hypre_StructMatrix *A,
                                         nalu_hypre_StructVector *b, nalu_hypre_StructVector *x );

/* red_black_gs.c */
void *nalu_hypre_RedBlackGSCreate ( MPI_Comm comm );
NALU_HYPRE_Int nalu_hypre_RedBlackGSDestroy ( void *relax_vdata );
NALU_HYPRE_Int nalu_hypre_RedBlackGSSetup ( void *relax_vdata, nalu_hypre_StructMatrix *A, nalu_hypre_StructVector *b,
                                  nalu_hypre_StructVector *x );
NALU_HYPRE_Int nalu_hypre_RedBlackGS ( void *relax_vdata, nalu_hypre_StructMatrix *A, nalu_hypre_StructVector *b,
                             nalu_hypre_StructVector *x );
NALU_HYPRE_Int nalu_hypre_RedBlackGSSetTol ( void *relax_vdata, NALU_HYPRE_Real tol );
NALU_HYPRE_Int nalu_hypre_RedBlackGSSetMaxIter ( void *relax_vdata, NALU_HYPRE_Int max_iter );
NALU_HYPRE_Int nalu_hypre_RedBlackGSSetZeroGuess ( void *relax_vdata, NALU_HYPRE_Int zero_guess );
NALU_HYPRE_Int nalu_hypre_RedBlackGSSetStartRed ( void *relax_vdata );
NALU_HYPRE_Int nalu_hypre_RedBlackGSSetStartBlack ( void *relax_vdata );

/* semi.c */
NALU_HYPRE_Int nalu_hypre_StructInterpAssemble ( nalu_hypre_StructMatrix *A, nalu_hypre_StructMatrix *P,
                                       NALU_HYPRE_Int P_stored_as_transpose, NALU_HYPRE_Int cdir, nalu_hypre_Index index, nalu_hypre_Index stride );

/* semi_interp.c */
void *nalu_hypre_SemiInterpCreate ( void );
NALU_HYPRE_Int nalu_hypre_SemiInterpSetup ( void *interp_vdata, nalu_hypre_StructMatrix *P,
                                  NALU_HYPRE_Int P_stored_as_transpose, nalu_hypre_StructVector *xc, nalu_hypre_StructVector *e, nalu_hypre_Index cindex,
                                  nalu_hypre_Index findex, nalu_hypre_Index stride );
NALU_HYPRE_Int nalu_hypre_SemiInterp ( void *interp_vdata, nalu_hypre_StructMatrix *P, nalu_hypre_StructVector *xc,
                             nalu_hypre_StructVector *e );
NALU_HYPRE_Int nalu_hypre_SemiInterpDestroy ( void *interp_vdata );

/* semi_restrict.c */
void *nalu_hypre_SemiRestrictCreate ( void );
NALU_HYPRE_Int nalu_hypre_SemiRestrictSetup ( void *restrict_vdata, nalu_hypre_StructMatrix *R,
                                    NALU_HYPRE_Int R_stored_as_transpose, nalu_hypre_StructVector *r, nalu_hypre_StructVector *rc, nalu_hypre_Index cindex,
                                    nalu_hypre_Index findex, nalu_hypre_Index stride );
NALU_HYPRE_Int nalu_hypre_SemiRestrict ( void *restrict_vdata, nalu_hypre_StructMatrix *R, nalu_hypre_StructVector *r,
                               nalu_hypre_StructVector *rc );
NALU_HYPRE_Int nalu_hypre_SemiRestrictDestroy ( void *restrict_vdata );

/* semi_setup_rap.c */
nalu_hypre_StructMatrix *nalu_hypre_SemiCreateRAPOp ( nalu_hypre_StructMatrix *R, nalu_hypre_StructMatrix *A,
                                            nalu_hypre_StructMatrix *P, nalu_hypre_StructGrid *coarse_grid, NALU_HYPRE_Int cdir,
                                            NALU_HYPRE_Int P_stored_as_transpose );
NALU_HYPRE_Int nalu_hypre_SemiBuildRAP ( nalu_hypre_StructMatrix *A, nalu_hypre_StructMatrix *P, nalu_hypre_StructMatrix *R,
                               NALU_HYPRE_Int cdir, nalu_hypre_Index cindex, nalu_hypre_Index cstride, NALU_HYPRE_Int P_stored_as_transpose,
                               nalu_hypre_StructMatrix *RAP );

/* smg2_setup_rap.c */
nalu_hypre_StructMatrix *nalu_hypre_SMG2CreateRAPOp ( nalu_hypre_StructMatrix *R, nalu_hypre_StructMatrix *A,
                                            nalu_hypre_StructMatrix *PT, nalu_hypre_StructGrid *coarse_grid );
NALU_HYPRE_Int nalu_hypre_SMG2BuildRAPSym ( nalu_hypre_StructMatrix *A, nalu_hypre_StructMatrix *PT,
                                  nalu_hypre_StructMatrix *R, nalu_hypre_StructMatrix *RAP, nalu_hypre_Index cindex, nalu_hypre_Index cstride );
NALU_HYPRE_Int nalu_hypre_SMG2BuildRAPNoSym ( nalu_hypre_StructMatrix *A, nalu_hypre_StructMatrix *PT,
                                    nalu_hypre_StructMatrix *R, nalu_hypre_StructMatrix *RAP, nalu_hypre_Index cindex, nalu_hypre_Index cstride );
NALU_HYPRE_Int nalu_hypre_SMG2RAPPeriodicSym ( nalu_hypre_StructMatrix *RAP, nalu_hypre_Index cindex,
                                     nalu_hypre_Index cstride );
NALU_HYPRE_Int nalu_hypre_SMG2RAPPeriodicNoSym ( nalu_hypre_StructMatrix *RAP, nalu_hypre_Index cindex,
                                       nalu_hypre_Index cstride );

/* smg3_setup_rap.c */
nalu_hypre_StructMatrix *nalu_hypre_SMG3CreateRAPOp ( nalu_hypre_StructMatrix *R, nalu_hypre_StructMatrix *A,
                                            nalu_hypre_StructMatrix *PT, nalu_hypre_StructGrid *coarse_grid );
NALU_HYPRE_Int nalu_hypre_SMG3BuildRAPSym ( nalu_hypre_StructMatrix *A, nalu_hypre_StructMatrix *PT,
                                  nalu_hypre_StructMatrix *R, nalu_hypre_StructMatrix *RAP, nalu_hypre_Index cindex, nalu_hypre_Index cstride );
NALU_HYPRE_Int nalu_hypre_SMG3BuildRAPNoSym ( nalu_hypre_StructMatrix *A, nalu_hypre_StructMatrix *PT,
                                    nalu_hypre_StructMatrix *R, nalu_hypre_StructMatrix *RAP, nalu_hypre_Index cindex, nalu_hypre_Index cstride );
NALU_HYPRE_Int nalu_hypre_SMG3RAPPeriodicSym ( nalu_hypre_StructMatrix *RAP, nalu_hypre_Index cindex,
                                     nalu_hypre_Index cstride );
NALU_HYPRE_Int nalu_hypre_SMG3RAPPeriodicNoSym ( nalu_hypre_StructMatrix *RAP, nalu_hypre_Index cindex,
                                       nalu_hypre_Index cstride );

/* smg_axpy.c */
NALU_HYPRE_Int nalu_hypre_SMGAxpy ( NALU_HYPRE_Real alpha, nalu_hypre_StructVector *x, nalu_hypre_StructVector *y,
                          nalu_hypre_Index base_index, nalu_hypre_Index base_stride );

/* smg.c */
void *nalu_hypre_SMGCreate ( MPI_Comm comm );
NALU_HYPRE_Int nalu_hypre_SMGDestroy ( void *smg_vdata );
NALU_HYPRE_Int nalu_hypre_SMGSetMemoryUse ( void *smg_vdata, NALU_HYPRE_Int memory_use );
NALU_HYPRE_Int nalu_hypre_SMGGetMemoryUse ( void *smg_vdata, NALU_HYPRE_Int *memory_use );
NALU_HYPRE_Int nalu_hypre_SMGSetTol ( void *smg_vdata, NALU_HYPRE_Real tol );
NALU_HYPRE_Int nalu_hypre_SMGGetTol ( void *smg_vdata, NALU_HYPRE_Real *tol );
NALU_HYPRE_Int nalu_hypre_SMGSetMaxIter ( void *smg_vdata, NALU_HYPRE_Int max_iter );
NALU_HYPRE_Int nalu_hypre_SMGGetMaxIter ( void *smg_vdata, NALU_HYPRE_Int *max_iter );
NALU_HYPRE_Int nalu_hypre_SMGSetRelChange ( void *smg_vdata, NALU_HYPRE_Int rel_change );
NALU_HYPRE_Int nalu_hypre_SMGGetRelChange ( void *smg_vdata, NALU_HYPRE_Int *rel_change );
NALU_HYPRE_Int nalu_hypre_SMGSetZeroGuess ( void *smg_vdata, NALU_HYPRE_Int zero_guess );
NALU_HYPRE_Int nalu_hypre_SMGGetZeroGuess ( void *smg_vdata, NALU_HYPRE_Int *zero_guess );
NALU_HYPRE_Int nalu_hypre_SMGSetNumPreRelax ( void *smg_vdata, NALU_HYPRE_Int num_pre_relax );
NALU_HYPRE_Int nalu_hypre_SMGGetNumPreRelax ( void *smg_vdata, NALU_HYPRE_Int *num_pre_relax );
NALU_HYPRE_Int nalu_hypre_SMGSetNumPostRelax ( void *smg_vdata, NALU_HYPRE_Int num_post_relax );
NALU_HYPRE_Int nalu_hypre_SMGGetNumPostRelax ( void *smg_vdata, NALU_HYPRE_Int *num_post_relax );
NALU_HYPRE_Int nalu_hypre_SMGSetBase ( void *smg_vdata, nalu_hypre_Index base_index, nalu_hypre_Index base_stride );
NALU_HYPRE_Int nalu_hypre_SMGSetLogging ( void *smg_vdata, NALU_HYPRE_Int logging );
NALU_HYPRE_Int nalu_hypre_SMGGetLogging ( void *smg_vdata, NALU_HYPRE_Int *logging );
NALU_HYPRE_Int nalu_hypre_SMGSetPrintLevel ( void *smg_vdata, NALU_HYPRE_Int print_level );
NALU_HYPRE_Int nalu_hypre_SMGGetPrintLevel ( void *smg_vdata, NALU_HYPRE_Int *print_level );
NALU_HYPRE_Int nalu_hypre_SMGGetNumIterations ( void *smg_vdata, NALU_HYPRE_Int *num_iterations );
NALU_HYPRE_Int nalu_hypre_SMGPrintLogging ( void *smg_vdata, NALU_HYPRE_Int myid );
NALU_HYPRE_Int nalu_hypre_SMGGetFinalRelativeResidualNorm ( void *smg_vdata,
                                                  NALU_HYPRE_Real *relative_residual_norm );
NALU_HYPRE_Int nalu_hypre_SMGSetStructVectorConstantValues ( nalu_hypre_StructVector *vector, NALU_HYPRE_Real values,
                                                   nalu_hypre_BoxArray *box_array, nalu_hypre_Index stride );
NALU_HYPRE_Int nalu_hypre_StructSMGSetMaxLevel( void   *smg_vdata, NALU_HYPRE_Int   max_level  );

/* smg_relax.c */
void *nalu_hypre_SMGRelaxCreate ( MPI_Comm comm );
NALU_HYPRE_Int nalu_hypre_SMGRelaxDestroyTempVec ( void *relax_vdata );
NALU_HYPRE_Int nalu_hypre_SMGRelaxDestroyARem ( void *relax_vdata );
NALU_HYPRE_Int nalu_hypre_SMGRelaxDestroyASol ( void *relax_vdata );
NALU_HYPRE_Int nalu_hypre_SMGRelaxDestroy ( void *relax_vdata );
NALU_HYPRE_Int nalu_hypre_SMGRelax ( void *relax_vdata, nalu_hypre_StructMatrix *A, nalu_hypre_StructVector *b,
                           nalu_hypre_StructVector *x );
NALU_HYPRE_Int nalu_hypre_SMGRelaxSetup ( void *relax_vdata, nalu_hypre_StructMatrix *A, nalu_hypre_StructVector *b,
                                nalu_hypre_StructVector *x );
NALU_HYPRE_Int nalu_hypre_SMGRelaxSetupTempVec ( void *relax_vdata, nalu_hypre_StructMatrix *A,
                                       nalu_hypre_StructVector *b, nalu_hypre_StructVector *x );
NALU_HYPRE_Int nalu_hypre_SMGRelaxSetupARem ( void *relax_vdata, nalu_hypre_StructMatrix *A, nalu_hypre_StructVector *b,
                                    nalu_hypre_StructVector *x );
NALU_HYPRE_Int nalu_hypre_SMGRelaxSetupASol ( void *relax_vdata, nalu_hypre_StructMatrix *A, nalu_hypre_StructVector *b,
                                    nalu_hypre_StructVector *x );
NALU_HYPRE_Int nalu_hypre_SMGRelaxSetTempVec ( void *relax_vdata, nalu_hypre_StructVector *temp_vec );
NALU_HYPRE_Int nalu_hypre_SMGRelaxSetMemoryUse ( void *relax_vdata, NALU_HYPRE_Int memory_use );
NALU_HYPRE_Int nalu_hypre_SMGRelaxSetTol ( void *relax_vdata, NALU_HYPRE_Real tol );
NALU_HYPRE_Int nalu_hypre_SMGRelaxSetMaxIter ( void *relax_vdata, NALU_HYPRE_Int max_iter );
NALU_HYPRE_Int nalu_hypre_SMGRelaxSetZeroGuess ( void *relax_vdata, NALU_HYPRE_Int zero_guess );
NALU_HYPRE_Int nalu_hypre_SMGRelaxSetNumSpaces ( void *relax_vdata, NALU_HYPRE_Int num_spaces );
NALU_HYPRE_Int nalu_hypre_SMGRelaxSetNumPreSpaces ( void *relax_vdata, NALU_HYPRE_Int num_pre_spaces );
NALU_HYPRE_Int nalu_hypre_SMGRelaxSetNumRegSpaces ( void *relax_vdata, NALU_HYPRE_Int num_reg_spaces );
NALU_HYPRE_Int nalu_hypre_SMGRelaxSetSpace ( void *relax_vdata, NALU_HYPRE_Int i, NALU_HYPRE_Int space_index,
                                   NALU_HYPRE_Int space_stride );
NALU_HYPRE_Int nalu_hypre_SMGRelaxSetRegSpaceRank ( void *relax_vdata, NALU_HYPRE_Int i,
                                          NALU_HYPRE_Int reg_space_rank );
NALU_HYPRE_Int nalu_hypre_SMGRelaxSetPreSpaceRank ( void *relax_vdata, NALU_HYPRE_Int i,
                                          NALU_HYPRE_Int pre_space_rank );
NALU_HYPRE_Int nalu_hypre_SMGRelaxSetBase ( void *relax_vdata, nalu_hypre_Index base_index,
                                  nalu_hypre_Index base_stride );
NALU_HYPRE_Int nalu_hypre_SMGRelaxSetNumPreRelax ( void *relax_vdata, NALU_HYPRE_Int num_pre_relax );
NALU_HYPRE_Int nalu_hypre_SMGRelaxSetNumPostRelax ( void *relax_vdata, NALU_HYPRE_Int num_post_relax );
NALU_HYPRE_Int nalu_hypre_SMGRelaxSetNewMatrixStencil ( void *relax_vdata,
                                              nalu_hypre_StructStencil *diff_stencil );
NALU_HYPRE_Int nalu_hypre_SMGRelaxSetupBaseBoxArray ( void *relax_vdata, nalu_hypre_StructMatrix *A,
                                            nalu_hypre_StructVector *b, nalu_hypre_StructVector *x );
NALU_HYPRE_Int nalu_hypre_SMGRelaxSetMaxLevel( void *relax_vdata, NALU_HYPRE_Int   num_max_level );

/* smg_residual.c */
void *nalu_hypre_SMGResidualCreate ( void );
NALU_HYPRE_Int nalu_hypre_SMGResidualSetup ( void *residual_vdata, nalu_hypre_StructMatrix *A,
                                   nalu_hypre_StructVector *x, nalu_hypre_StructVector *b, nalu_hypre_StructVector *r );
NALU_HYPRE_Int nalu_hypre_SMGResidual ( void *residual_vdata, nalu_hypre_StructMatrix *A, nalu_hypre_StructVector *x,
                              nalu_hypre_StructVector *b, nalu_hypre_StructVector *r );
NALU_HYPRE_Int nalu_hypre_SMGResidualSetBase ( void *residual_vdata, nalu_hypre_Index base_index,
                                     nalu_hypre_Index base_stride );
NALU_HYPRE_Int nalu_hypre_SMGResidualDestroy ( void *residual_vdata );

/* smg_residual_unrolled.c */
void *nalu_hypre_SMGResidualCreate ( void );
NALU_HYPRE_Int nalu_hypre_SMGResidualSetup ( void *residual_vdata, nalu_hypre_StructMatrix *A,
                                   nalu_hypre_StructVector *x, nalu_hypre_StructVector *b, nalu_hypre_StructVector *r );
NALU_HYPRE_Int nalu_hypre_SMGResidual ( void *residual_vdata, nalu_hypre_StructMatrix *A, nalu_hypre_StructVector *x,
                              nalu_hypre_StructVector *b, nalu_hypre_StructVector *r );
NALU_HYPRE_Int nalu_hypre_SMGResidualSetBase ( void *residual_vdata, nalu_hypre_Index base_index,
                                     nalu_hypre_Index base_stride );
NALU_HYPRE_Int nalu_hypre_SMGResidualDestroy ( void *residual_vdata );

/* smg_setup.c */
NALU_HYPRE_Int nalu_hypre_SMGSetup ( void *smg_vdata, nalu_hypre_StructMatrix *A, nalu_hypre_StructVector *b,
                           nalu_hypre_StructVector *x );

/* smg_setup_interp.c */
nalu_hypre_StructMatrix *nalu_hypre_SMGCreateInterpOp ( nalu_hypre_StructMatrix *A, nalu_hypre_StructGrid *cgrid,
                                              NALU_HYPRE_Int cdir );
NALU_HYPRE_Int nalu_hypre_SMGSetupInterpOp ( void *relax_data, nalu_hypre_StructMatrix *A, nalu_hypre_StructVector *b,
                                   nalu_hypre_StructVector *x, nalu_hypre_StructMatrix *PT, NALU_HYPRE_Int cdir, nalu_hypre_Index cindex,
                                   nalu_hypre_Index findex, nalu_hypre_Index stride );

/* smg_setup_rap.c */
nalu_hypre_StructMatrix *nalu_hypre_SMGCreateRAPOp ( nalu_hypre_StructMatrix *R, nalu_hypre_StructMatrix *A,
                                           nalu_hypre_StructMatrix *PT, nalu_hypre_StructGrid *coarse_grid );
NALU_HYPRE_Int nalu_hypre_SMGSetupRAPOp ( nalu_hypre_StructMatrix *R, nalu_hypre_StructMatrix *A,
                                nalu_hypre_StructMatrix *PT, nalu_hypre_StructMatrix *Ac, nalu_hypre_Index cindex, nalu_hypre_Index cstride );

/* smg_setup_restrict.c */
nalu_hypre_StructMatrix *nalu_hypre_SMGCreateRestrictOp ( nalu_hypre_StructMatrix *A, nalu_hypre_StructGrid *cgrid,
                                                NALU_HYPRE_Int cdir );
NALU_HYPRE_Int nalu_hypre_SMGSetupRestrictOp ( nalu_hypre_StructMatrix *A, nalu_hypre_StructMatrix *R,
                                     nalu_hypre_StructVector *temp_vec, NALU_HYPRE_Int cdir, nalu_hypre_Index cindex, nalu_hypre_Index cstride );

/* smg_solve.c */
NALU_HYPRE_Int nalu_hypre_SMGSolve ( void *smg_vdata, nalu_hypre_StructMatrix *A, nalu_hypre_StructVector *b,
                           nalu_hypre_StructVector *x );

/* sparse_msg2_setup_rap.c */
nalu_hypre_StructMatrix *nalu_hypre_SparseMSG2CreateRAPOp ( nalu_hypre_StructMatrix *R, nalu_hypre_StructMatrix *A,
                                                  nalu_hypre_StructMatrix *P, nalu_hypre_StructGrid *coarse_grid, NALU_HYPRE_Int cdir );
NALU_HYPRE_Int nalu_hypre_SparseMSG2BuildRAPSym ( nalu_hypre_StructMatrix *A, nalu_hypre_StructMatrix *P,
                                        nalu_hypre_StructMatrix *R, NALU_HYPRE_Int cdir, nalu_hypre_Index cindex, nalu_hypre_Index cstride,
                                        nalu_hypre_Index stridePR, nalu_hypre_StructMatrix *RAP );
NALU_HYPRE_Int nalu_hypre_SparseMSG2BuildRAPNoSym ( nalu_hypre_StructMatrix *A, nalu_hypre_StructMatrix *P,
                                          nalu_hypre_StructMatrix *R, NALU_HYPRE_Int cdir, nalu_hypre_Index cindex, nalu_hypre_Index cstride,
                                          nalu_hypre_Index stridePR, nalu_hypre_StructMatrix *RAP );

/* sparse_msg3_setup_rap.c */
nalu_hypre_StructMatrix *nalu_hypre_SparseMSG3CreateRAPOp ( nalu_hypre_StructMatrix *R, nalu_hypre_StructMatrix *A,
                                                  nalu_hypre_StructMatrix *P, nalu_hypre_StructGrid *coarse_grid, NALU_HYPRE_Int cdir );
NALU_HYPRE_Int nalu_hypre_SparseMSG3BuildRAPSym ( nalu_hypre_StructMatrix *A, nalu_hypre_StructMatrix *P,
                                        nalu_hypre_StructMatrix *R, NALU_HYPRE_Int cdir, nalu_hypre_Index cindex, nalu_hypre_Index cstride,
                                        nalu_hypre_Index stridePR, nalu_hypre_StructMatrix *RAP );
NALU_HYPRE_Int nalu_hypre_SparseMSG3BuildRAPNoSym ( nalu_hypre_StructMatrix *A, nalu_hypre_StructMatrix *P,
                                          nalu_hypre_StructMatrix *R, NALU_HYPRE_Int cdir, nalu_hypre_Index cindex, nalu_hypre_Index cstride,
                                          nalu_hypre_Index stridePR, nalu_hypre_StructMatrix *RAP );

/* sparse_msg.c */
void *nalu_hypre_SparseMSGCreate ( MPI_Comm comm );
NALU_HYPRE_Int nalu_hypre_SparseMSGDestroy ( void *smsg_vdata );
NALU_HYPRE_Int nalu_hypre_SparseMSGSetTol ( void *smsg_vdata, NALU_HYPRE_Real tol );
NALU_HYPRE_Int nalu_hypre_SparseMSGSetMaxIter ( void *smsg_vdata, NALU_HYPRE_Int max_iter );
NALU_HYPRE_Int nalu_hypre_SparseMSGSetJump ( void *smsg_vdata, NALU_HYPRE_Int jump );
NALU_HYPRE_Int nalu_hypre_SparseMSGSetRelChange ( void *smsg_vdata, NALU_HYPRE_Int rel_change );
NALU_HYPRE_Int nalu_hypre_SparseMSGSetZeroGuess ( void *smsg_vdata, NALU_HYPRE_Int zero_guess );
NALU_HYPRE_Int nalu_hypre_SparseMSGSetRelaxType ( void *smsg_vdata, NALU_HYPRE_Int relax_type );
NALU_HYPRE_Int nalu_hypre_SparseMSGSetJacobiWeight ( void *smsg_vdata, NALU_HYPRE_Real weight );
NALU_HYPRE_Int nalu_hypre_SparseMSGSetNumPreRelax ( void *smsg_vdata, NALU_HYPRE_Int num_pre_relax );
NALU_HYPRE_Int nalu_hypre_SparseMSGSetNumPostRelax ( void *smsg_vdata, NALU_HYPRE_Int num_post_relax );
NALU_HYPRE_Int nalu_hypre_SparseMSGSetNumFineRelax ( void *smsg_vdata, NALU_HYPRE_Int num_fine_relax );
NALU_HYPRE_Int nalu_hypre_SparseMSGSetLogging ( void *smsg_vdata, NALU_HYPRE_Int logging );
NALU_HYPRE_Int nalu_hypre_SparseMSGSetPrintLevel ( void *smsg_vdata, NALU_HYPRE_Int print_level );
NALU_HYPRE_Int nalu_hypre_SparseMSGGetNumIterations ( void *smsg_vdata, NALU_HYPRE_Int *num_iterations );
NALU_HYPRE_Int nalu_hypre_SparseMSGPrintLogging ( void *smsg_vdata, NALU_HYPRE_Int myid );
NALU_HYPRE_Int nalu_hypre_SparseMSGGetFinalRelativeResidualNorm ( void *smsg_vdata,
                                                        NALU_HYPRE_Real *relative_residual_norm );

/* sparse_msg_filter.c */
NALU_HYPRE_Int nalu_hypre_SparseMSGFilterSetup ( nalu_hypre_StructMatrix *A, NALU_HYPRE_Int *num_grids, NALU_HYPRE_Int lx,
                                       NALU_HYPRE_Int ly, NALU_HYPRE_Int lz, NALU_HYPRE_Int jump, nalu_hypre_StructVector *visitx, nalu_hypre_StructVector *visity,
                                       nalu_hypre_StructVector *visitz );
NALU_HYPRE_Int nalu_hypre_SparseMSGFilter ( nalu_hypre_StructVector *visit, nalu_hypre_StructVector *e, NALU_HYPRE_Int lx,
                                  NALU_HYPRE_Int ly, NALU_HYPRE_Int lz, NALU_HYPRE_Int jump );

/* sparse_msg_interp.c */
void *nalu_hypre_SparseMSGInterpCreate ( void );
NALU_HYPRE_Int nalu_hypre_SparseMSGInterpSetup ( void *interp_vdata, nalu_hypre_StructMatrix *P,
                                       nalu_hypre_StructVector *xc, nalu_hypre_StructVector *e, nalu_hypre_Index cindex, nalu_hypre_Index findex,
                                       nalu_hypre_Index stride, nalu_hypre_Index strideP );
NALU_HYPRE_Int nalu_hypre_SparseMSGInterp ( void *interp_vdata, nalu_hypre_StructMatrix *P, nalu_hypre_StructVector *xc,
                                  nalu_hypre_StructVector *e );
NALU_HYPRE_Int nalu_hypre_SparseMSGInterpDestroy ( void *interp_vdata );

/* sparse_msg_restrict.c */
void *nalu_hypre_SparseMSGRestrictCreate ( void );
NALU_HYPRE_Int nalu_hypre_SparseMSGRestrictSetup ( void *restrict_vdata, nalu_hypre_StructMatrix *R,
                                         nalu_hypre_StructVector *r, nalu_hypre_StructVector *rc, nalu_hypre_Index cindex, nalu_hypre_Index findex,
                                         nalu_hypre_Index stride, nalu_hypre_Index strideR );
NALU_HYPRE_Int nalu_hypre_SparseMSGRestrict ( void *restrict_vdata, nalu_hypre_StructMatrix *R,
                                    nalu_hypre_StructVector *r, nalu_hypre_StructVector *rc );
NALU_HYPRE_Int nalu_hypre_SparseMSGRestrictDestroy ( void *restrict_vdata );

/* sparse_msg_setup.c */
NALU_HYPRE_Int nalu_hypre_SparseMSGSetup ( void *smsg_vdata, nalu_hypre_StructMatrix *A, nalu_hypre_StructVector *b,
                                 nalu_hypre_StructVector *x );

/* sparse_msg_setup_rap.c */
nalu_hypre_StructMatrix *nalu_hypre_SparseMSGCreateRAPOp ( nalu_hypre_StructMatrix *R, nalu_hypre_StructMatrix *A,
                                                 nalu_hypre_StructMatrix *P, nalu_hypre_StructGrid *coarse_grid, NALU_HYPRE_Int cdir );
NALU_HYPRE_Int nalu_hypre_SparseMSGSetupRAPOp ( nalu_hypre_StructMatrix *R, nalu_hypre_StructMatrix *A,
                                      nalu_hypre_StructMatrix *P, NALU_HYPRE_Int cdir, nalu_hypre_Index cindex, nalu_hypre_Index cstride,
                                      nalu_hypre_Index stridePR, nalu_hypre_StructMatrix *Ac );

/* sparse_msg_solve.c */
NALU_HYPRE_Int nalu_hypre_SparseMSGSolve ( void *smsg_vdata, nalu_hypre_StructMatrix *A, nalu_hypre_StructVector *b,
                                 nalu_hypre_StructVector *x );
