/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#ifndef hypre_ParILU_DATA_HEADER
#define hypre_ParILU_DATA_HEADER

/*--------------------------------------------------------------------------
 * hypre_ParILUData
 *--------------------------------------------------------------------------*/
typedef struct hypre_ParILUData_struct
{
#ifdef NALU_HYPRE_USING_CUDA
   /* Data slots for cusparse-based ilu0 */
   cusparseMatDescr_t      matL_des;//lower tri with ones on diagonal
   cusparseMatDescr_t      matU_des;//upper tri
   csrsv2Info_t            matAL_info;//ILU info for L of A block (used in A-smoothing)
   csrsv2Info_t            matAU_info;//ILU info for U of A block
   csrsv2Info_t            matBL_info;//ILU info for L of B block
   csrsv2Info_t            matBU_info;//ILU info for U of B block
   csrsv2Info_t            matSL_info;//ILU info for L of S block
   csrsv2Info_t            matSU_info;//ILU info for U of S block
   cusparseSolvePolicy_t   ilu_solve_policy;//Use/Don't use level
   void                    *ilu_solve_buffer;//working array on device memory

   /* on GPU, we have to form E and F explicitly, since we don't have much control to it
    *
    */
   hypre_CSRMatrix         *matALU_d;//the matrix holding ILU of A (for A-smoothing)
   hypre_CSRMatrix         *matBLU_d;//the matrix holding ILU of B
   hypre_CSRMatrix         *matSLU_d;//the matrix holding ILU of S
   hypre_CSRMatrix         *matE_d;
   hypre_CSRMatrix         *matF_d;
   hypre_ParCSRMatrix      *Aperm;
   hypre_ParCSRMatrix      *R;
   hypre_ParCSRMatrix      *P;
   hypre_Vector            *Ftemp_upper;
   hypre_Vector            *Utemp_lower;
   NALU_HYPRE_Int               *A_diag_fake;//fake diagonal, pretend the diagonal matrix is empty
   hypre_Vector            *Adiag_diag;
#endif
   //general data
   NALU_HYPRE_Int            global_solver;
   hypre_ParCSRMatrix   *matA;
   hypre_ParCSRMatrix   *matL;
   NALU_HYPRE_Real           *matD;
   hypre_ParCSRMatrix   *matU;
   hypre_ParCSRMatrix   *matmL;
   NALU_HYPRE_Real           *matmD;
   hypre_ParCSRMatrix   *matmU;
   hypre_ParCSRMatrix   *matS;
   NALU_HYPRE_Real
   *droptol;/* should be an array of 3 element, for B, (E and F), S respectively */
   NALU_HYPRE_Int            lfil;
   NALU_HYPRE_Int            maxRowNnz;
   NALU_HYPRE_Int            *CF_marker_array;
   NALU_HYPRE_Int            *perm;
   NALU_HYPRE_Int            *qperm;
   NALU_HYPRE_Real           tol_ddPQ;
   hypre_ParVector      *F;
   hypre_ParVector      *U;
   hypre_ParVector      *residual;
   NALU_HYPRE_Real           *rel_res_norms;
   NALU_HYPRE_Int            num_iterations;
   NALU_HYPRE_Real           *l1_norms;
   NALU_HYPRE_Real           final_rel_residual_norm;
   NALU_HYPRE_Real           tol;
   NALU_HYPRE_Real           operator_complexity;

   NALU_HYPRE_Int            logging;
   NALU_HYPRE_Int            print_level;
   NALU_HYPRE_Int            max_iter;

   NALU_HYPRE_Int            tri_solve;
   NALU_HYPRE_Int            lower_jacobi_iters;
   NALU_HYPRE_Int            upper_jacobi_iters;

   NALU_HYPRE_Int            ilu_type;
   NALU_HYPRE_Int            nLU;
   NALU_HYPRE_Int            nI;

   /* used when schur block is formed */
   NALU_HYPRE_Int            *u_end;

   /* temp vectors for solve phase */
   hypre_ParVector      *Utemp;
   hypre_ParVector      *Ftemp;
   hypre_ParVector      *Xtemp;
   hypre_ParVector      *Ytemp;
   hypre_Vector         *Ztemp;
   NALU_HYPRE_Real           *uext;
   NALU_HYPRE_Real           *fext;

   /* data structure sor solving Schur System */
   NALU_HYPRE_Solver         schur_solver;
   NALU_HYPRE_Solver         schur_precond;
   hypre_ParVector      *rhs;
   hypre_ParVector      *x;

   /* schur solver data */

   /* -> GENERAL-SLOTS */
   NALU_HYPRE_Int            ss_logging;
   NALU_HYPRE_Int            ss_print_level;

   /* -> SCHUR-GMRES */
   NALU_HYPRE_Int            ss_kDim;/* max number of iterations for GMRES */
   NALU_HYPRE_Int            ss_max_iter;/* max number of iterations for GMRES solve */
   NALU_HYPRE_Real           ss_tol;/* stop iteration tol for GMRES */
   NALU_HYPRE_Real           ss_absolute_tol;/* absolute tol for GMRES or tol for NSH solve */
   NALU_HYPRE_Int            ss_rel_change;

   /* -> SCHUR-NSH */
   NALU_HYPRE_Int            ss_nsh_setup_max_iter;/* number of iterations for NSH inverse */
   NALU_HYPRE_Int            ss_nsh_solve_max_iter;/* max number of iterations for NSH solve */
   NALU_HYPRE_Real           ss_nsh_setup_tol;/* stop iteration tol for NSH inverse */
   NALU_HYPRE_Real           ss_nsh_solve_tol;/* absolute tol for NSH solve */
   NALU_HYPRE_Int            ss_nsh_max_row_nnz;/* max rows of nonzeros for NSH */
   NALU_HYPRE_Int            ss_nsh_mr_col_version;/* MR column version setting in NSH */
   NALU_HYPRE_Int            ss_nsh_mr_max_row_nnz;/* max rows for MR  */
   NALU_HYPRE_Real           *ss_nsh_droptol;/* droptol array for NSH */
   NALU_HYPRE_Int            ss_nsh_mr_max_iter;/* max MR iteration */
   NALU_HYPRE_Real           ss_nsh_mr_tol;

   /* schur precond data */
   NALU_HYPRE_Int            sp_ilu_type;/* ilu type is use ILU */
   NALU_HYPRE_Int            sp_ilu_lfil;/* level of fill in for ILUK */
   NALU_HYPRE_Int            sp_ilu_max_row_nnz;/* max rows for ILUT  */
   /* droptol for ILUT or MR
    * ILUT: [0], [1], [2] B, E&F, S respectively
    * NSH: [0] for MR, [1] for NSH
    */
   NALU_HYPRE_Real           *sp_ilu_droptol;/* droptol array for ILUT */
   NALU_HYPRE_Int            sp_print_level;
   NALU_HYPRE_Int            sp_max_iter;/* max precond iter or max MR iteration */
   NALU_HYPRE_Int            sp_tri_solve;
   NALU_HYPRE_Int            sp_lower_jacobi_iters;
   NALU_HYPRE_Int            sp_upper_jacobi_iters;
   NALU_HYPRE_Real           sp_tol;

   NALU_HYPRE_Int            test_opt;
   /* local reordering */
   NALU_HYPRE_Int            reordering_type;

} hypre_ParILUData;

#define hypre_ParILUDataTestOption(ilu_data)                   ((ilu_data) -> test_opt)

#ifdef NALU_HYPRE_USING_CUDA
#define hypre_ParILUDataMatLMatrixDescription(ilu_data)        ((ilu_data) -> matL_des)
#define hypre_ParILUDataMatUMatrixDescription(ilu_data)        ((ilu_data) -> matU_des)
#define hypre_ParILUDataMatALILUSolveInfo(ilu_data)            ((ilu_data) -> matAL_info)
#define hypre_ParILUDataMatAUILUSolveInfo(ilu_data)            ((ilu_data) -> matAU_info)
#define hypre_ParILUDataMatBLILUSolveInfo(ilu_data)            ((ilu_data) -> matBL_info)
#define hypre_ParILUDataMatBUILUSolveInfo(ilu_data)            ((ilu_data) -> matBU_info)
#define hypre_ParILUDataMatSLILUSolveInfo(ilu_data)            ((ilu_data) -> matSL_info)
#define hypre_ParILUDataMatSUILUSolveInfo(ilu_data)            ((ilu_data) -> matSU_info)
#define hypre_ParILUDataILUSolveBuffer(ilu_data)               ((ilu_data) -> ilu_solve_buffer)
#define hypre_ParILUDataILUSolvePolicy(ilu_data)               ((ilu_data) -> ilu_solve_policy)
#define hypre_ParILUDataMatAILUDevice(ilu_data)                ((ilu_data) -> matALU_d)
#define hypre_ParILUDataMatBILUDevice(ilu_data)                ((ilu_data) -> matBLU_d)
#define hypre_ParILUDataMatSILUDevice(ilu_data)                ((ilu_data) -> matSLU_d)
#define hypre_ParILUDataMatEDevice(ilu_data)                   ((ilu_data) -> matE_d)
#define hypre_ParILUDataMatFDevice(ilu_data)                   ((ilu_data) -> matF_d)
#define hypre_ParILUDataAperm(ilu_data)                        ((ilu_data) -> Aperm)
#define hypre_ParILUDataR(ilu_data)                            ((ilu_data) -> R)
#define hypre_ParILUDataP(ilu_data)                            ((ilu_data) -> P)
#define hypre_ParILUDataFTempUpper(ilu_data)                   ((ilu_data) -> Ftemp_upper)
#define hypre_ParILUDataUTempLower(ilu_data)                   ((ilu_data) -> Utemp_lower)
#define hypre_ParILUDataMatAFakeDiagonal(ilu_data)             ((ilu_data) -> A_diag_fake)
#define hypre_ParILUDataADiagDiag(ilu_data)                    ((ilu_data) -> Adiag_diag)
#endif

#define hypre_ParILUDataGlobalSolver(ilu_data)                 ((ilu_data) -> global_solver)
#define hypre_ParILUDataMatA(ilu_data)                         ((ilu_data) -> matA)
#define hypre_ParILUDataMatL(ilu_data)                         ((ilu_data) -> matL)
#define hypre_ParILUDataMatD(ilu_data)                         ((ilu_data) -> matD)
#define hypre_ParILUDataMatU(ilu_data)                         ((ilu_data) -> matU)
#define hypre_ParILUDataMatLModified(ilu_data)                 ((ilu_data) -> matmL)
#define hypre_ParILUDataMatDModified(ilu_data)                 ((ilu_data) -> matmD)
#define hypre_ParILUDataMatUModified(ilu_data)                 ((ilu_data) -> matmU)
#define hypre_ParILUDataMatS(ilu_data)                         ((ilu_data) -> matS)
#define hypre_ParILUDataDroptol(ilu_data)                      ((ilu_data) -> droptol)
#define hypre_ParILUDataLfil(ilu_data)                         ((ilu_data) -> lfil)
#define hypre_ParILUDataMaxRowNnz(ilu_data)                    ((ilu_data) -> maxRowNnz)
#define hypre_ParILUDataCFMarkerArray(ilu_data)                ((ilu_data) -> CF_marker_array)
#define hypre_ParILUDataPerm(ilu_data)                         ((ilu_data) -> perm)
#define hypre_ParILUDataPPerm(ilu_data)                        ((ilu_data) -> perm)
#define hypre_ParILUDataQPerm(ilu_data)                        ((ilu_data) -> qperm)
#define hypre_ParILUDataTolDDPQ(ilu_data)                      ((ilu_data) -> tol_ddPQ)
#define hypre_ParILUDataF(ilu_data)                            ((ilu_data) -> F)
#define hypre_ParILUDataU(ilu_data)                            ((ilu_data) -> U)
#define hypre_ParILUDataResidual(ilu_data)                     ((ilu_data) -> residual)
#define hypre_ParILUDataRelResNorms(ilu_data)                  ((ilu_data) -> rel_res_norms)
#define hypre_ParILUDataNumIterations(ilu_data)                ((ilu_data) -> num_iterations)
#define hypre_ParILUDataL1Norms(ilu_data)                      ((ilu_data) -> l1_norms)
#define hypre_ParILUDataFinalRelResidualNorm(ilu_data)         ((ilu_data) -> final_rel_residual_norm)
#define hypre_ParILUDataTol(ilu_data)                          ((ilu_data) -> tol)
#define hypre_ParILUDataOperatorComplexity(ilu_data)           ((ilu_data) -> operator_complexity)
#define hypre_ParILUDataLogging(ilu_data)                      ((ilu_data) -> logging)
#define hypre_ParILUDataPrintLevel(ilu_data)                   ((ilu_data) -> print_level)
#define hypre_ParILUDataMaxIter(ilu_data)                      ((ilu_data) -> max_iter)
#define hypre_ParILUDataTriSolve(ilu_data)                     ((ilu_data) -> tri_solve)
#define hypre_ParILUDataLowerJacobiIters(ilu_data)             ((ilu_data) -> lower_jacobi_iters)
#define hypre_ParILUDataUpperJacobiIters(ilu_data)             ((ilu_data) -> upper_jacobi_iters)
#define hypre_ParILUDataIluType(ilu_data)                      ((ilu_data) -> ilu_type)
#define hypre_ParILUDataNLU(ilu_data)                          ((ilu_data) -> nLU)
#define hypre_ParILUDataNI(ilu_data)                           ((ilu_data) -> nI)
#define hypre_ParILUDataUEnd(ilu_data)                         ((ilu_data) -> u_end)
#define hypre_ParILUDataXTemp(ilu_data)                        ((ilu_data) -> Xtemp)
#define hypre_ParILUDataYTemp(ilu_data)                        ((ilu_data) -> Ytemp)
#define hypre_ParILUDataZTemp(ilu_data)                        ((ilu_data) -> Ztemp)
#define hypre_ParILUDataUTemp(ilu_data)                        ((ilu_data) -> Utemp)
#define hypre_ParILUDataFTemp(ilu_data)                        ((ilu_data) -> Ftemp)
#define hypre_ParILUDataUExt(ilu_data)                         ((ilu_data) -> uext)
#define hypre_ParILUDataFExt(ilu_data)                         ((ilu_data) -> fext)
#define hypre_ParILUDataSchurSolver(ilu_data)                  ((ilu_data) -> schur_solver)
#define hypre_ParILUDataSchurPrecond(ilu_data)                 ((ilu_data) -> schur_precond)
#define hypre_ParILUDataRhs(ilu_data)                          ((ilu_data) -> rhs)
#define hypre_ParILUDataX(ilu_data)                            ((ilu_data) -> x)
#define hypre_ParILUDataReorderingType(ilu_data)               ((ilu_data) -> reordering_type)
/* Schur System */
#define hypre_ParILUDataSchurGMRESKDim(ilu_data)               ((ilu_data) -> ss_kDim)
#define hypre_ParILUDataSchurGMRESMaxIter(ilu_data)            ((ilu_data) -> ss_max_iter)
#define hypre_ParILUDataSchurGMRESTol(ilu_data)                ((ilu_data) -> ss_tol)
#define hypre_ParILUDataSchurGMRESAbsoluteTol(ilu_data)        ((ilu_data) -> ss_absolute_tol)
#define hypre_ParILUDataSchurGMRESRelChange(ilu_data)          ((ilu_data) -> ss_rel_change)
#define hypre_ParILUDataSchurPrecondIluType(ilu_data)          ((ilu_data) -> sp_ilu_type)
#define hypre_ParILUDataSchurPrecondIluLfil(ilu_data)          ((ilu_data) -> sp_ilu_lfil)
#define hypre_ParILUDataSchurPrecondIluMaxRowNnz(ilu_data)     ((ilu_data) -> sp_ilu_max_row_nnz)
#define hypre_ParILUDataSchurPrecondIluDroptol(ilu_data)       ((ilu_data) -> sp_ilu_droptol)
#define hypre_ParILUDataSchurPrecondPrintLevel(ilu_data)       ((ilu_data) -> sp_print_level)
#define hypre_ParILUDataSchurPrecondMaxIter(ilu_data)          ((ilu_data) -> sp_max_iter)
#define hypre_ParILUDataSchurPrecondTriSolve(ilu_data)         ((ilu_data) -> sp_tri_solve)
#define hypre_ParILUDataSchurPrecondLowerJacobiIters(ilu_data) ((ilu_data) -> sp_lower_jacobi_iters)
#define hypre_ParILUDataSchurPrecondUpperJacobiIters(ilu_data) ((ilu_data) -> sp_upper_jacobi_iters)
#define hypre_ParILUDataSchurPrecondTol(ilu_data)              ((ilu_data) -> sp_tol)

#define hypre_ParILUDataSchurNSHMaxNumIter(ilu_data)           ((ilu_data) -> ss_nsh_setup_max_iter)
#define hypre_ParILUDataSchurNSHSolveMaxIter(ilu_data)         ((ilu_data) -> ss_nsh_solve_max_iter)
#define hypre_ParILUDataSchurNSHTol(ilu_data)                  ((ilu_data) -> ss_nsh_setup_tol)
#define hypre_ParILUDataSchurNSHSolveTol(ilu_data)             ((ilu_data) -> ss_nsh_solve_tol)
#define hypre_ParILUDataSchurNSHMaxRowNnz(ilu_data)            ((ilu_data) -> ss_nsh_max_row_nnz)
#define hypre_ParILUDataSchurMRColVersion(ilu_data)            ((ilu_data) -> ss_nsh_mr_col_version)
#define hypre_ParILUDataSchurMRMaxRowNnz(ilu_data)             ((ilu_data) -> ss_nsh_mr_max_row_nnz)
#define hypre_ParILUDataSchurNSHDroptol(ilu_data)              ((ilu_data) -> ss_nsh_droptol)
#define hypre_ParILUDataSchurMRMaxIter(ilu_data)               ((ilu_data) -> ss_nsh_mr_max_iter)
#define hypre_ParILUDataSchurMRTol(ilu_data)                   ((ilu_data) -> ss_nsh_mr_tol)

#define hypre_ParILUDataSchurSolverLogging(ilu_data)           ((ilu_data) -> ss_logging)
#define hypre_ParILUDataSchurSolverPrintLevel(ilu_data)        ((ilu_data) -> ss_print_level)

#define FMRK  -1
#define CMRK  1
#define UMRK  0
#define S_CMRK  2

#define FPT(i, bsize) (((i) % (bsize)) == FMRK)
#define CPT(i, bsize) (((i) % (bsize)) == CMRK)

#define MAT_TOL 1e-14
#define EXPAND_FACT 1.3





/* NSH data structure */

typedef struct hypre_ParNSHData_struct
{
   /* solver information */
   NALU_HYPRE_Int             global_solver;
   hypre_ParCSRMatrix    *matA;
   hypre_ParCSRMatrix    *matM;
   hypre_ParVector       *F;
   hypre_ParVector       *U;
   hypre_ParVector       *residual;
   NALU_HYPRE_Real            *rel_res_norms;
   NALU_HYPRE_Int             num_iterations;
   NALU_HYPRE_Real            *l1_norms;
   NALU_HYPRE_Real            final_rel_residual_norm;
   NALU_HYPRE_Real            tol;
   NALU_HYPRE_Real            operator_complexity;

   NALU_HYPRE_Int             logging;
   NALU_HYPRE_Int             print_level;
   NALU_HYPRE_Int             max_iter;

   /* common data slots */
   /* droptol[0]: droptol for MR
    * droptol[1]: droptol for NSH
    */
   NALU_HYPRE_Real            *droptol;
   NALU_HYPRE_Int             own_droptol_data;

   /* temp vectors for solve phase */
   hypre_ParVector       *Utemp;
   hypre_ParVector       *Ftemp;

   /* data slots for local MR */
   NALU_HYPRE_Int             mr_max_iter;
   NALU_HYPRE_Real            mr_tol;
   NALU_HYPRE_Int             mr_max_row_nnz;
   NALU_HYPRE_Int             mr_col_version;/* global version or column version MR */

   /* data slots for global NSH */
   NALU_HYPRE_Int             nsh_max_iter;
   NALU_HYPRE_Real            nsh_tol;
   NALU_HYPRE_Int             nsh_max_row_nnz;
} hypre_ParNSHData;

#define hypre_ParNSHDataGlobalSolver(nsh_data)           ((nsh_data) -> global_solver)
#define hypre_ParNSHDataMatA(nsh_data)                   ((nsh_data) -> matA)
#define hypre_ParNSHDataMatM(nsh_data)                   ((nsh_data) -> matM)
#define hypre_ParNSHDataF(nsh_data)                      ((nsh_data) -> F)
#define hypre_ParNSHDataU(nsh_data)                      ((nsh_data) -> U)
#define hypre_ParNSHDataResidual(nsh_data)               ((nsh_data) -> residual)
#define hypre_ParNSHDataRelResNorms(nsh_data)            ((nsh_data) -> rel_res_norms)
#define hypre_ParNSHDataNumIterations(nsh_data)          ((nsh_data) -> num_iterations)
#define hypre_ParNSHDataL1Norms(nsh_data)                ((nsh_data) -> l1_norms)
#define hypre_ParNSHDataFinalRelResidualNorm(nsh_data)   ((nsh_data) -> final_rel_residual_norm)
#define hypre_ParNSHDataTol(nsh_data)                    ((nsh_data) -> tol)
#define hypre_ParNSHDataOperatorComplexity(nsh_data)     ((nsh_data) -> operator_complexity)
#define hypre_ParNSHDataLogging(nsh_data)                ((nsh_data) -> logging)
#define hypre_ParNSHDataPrintLevel(nsh_data)             ((nsh_data) -> print_level)
#define hypre_ParNSHDataMaxIter(nsh_data)                ((nsh_data) -> max_iter)
#define hypre_ParNSHDataDroptol(nsh_data)                ((nsh_data) -> droptol)
#define hypre_ParNSHDataOwnDroptolData(nsh_data)         ((nsh_data) -> own_droptol_data)
#define hypre_ParNSHDataUTemp(nsh_data)                  ((nsh_data) -> Utemp)
#define hypre_ParNSHDataFTemp(nsh_data)                  ((nsh_data) -> Ftemp)
#define hypre_ParNSHDataMRMaxIter(nsh_data)              ((nsh_data) -> mr_max_iter)
#define hypre_ParNSHDataMRTol(nsh_data)                  ((nsh_data) -> mr_tol)
#define hypre_ParNSHDataMRMaxRowNnz(nsh_data)            ((nsh_data) -> mr_max_row_nnz)
#define hypre_ParNSHDataMRColVersion(nsh_data)           ((nsh_data) -> mr_col_version)
#define hypre_ParNSHDataNSHMaxIter(nsh_data)             ((nsh_data) -> nsh_max_iter)
#define hypre_ParNSHDataNSHTol(nsh_data)                 ((nsh_data) -> nsh_tol)
#define hypre_ParNSHDataNSHMaxRowNnz(nsh_data)           ((nsh_data) -> nsh_max_row_nnz)

//#define DIVIDE_TOL 1e-32

#ifdef NALU_HYPRE_USING_GPU
NALU_HYPRE_Int hypre_ILUSolveDeviceLUIter(hypre_ParCSRMatrix *A, hypre_CSRMatrix *matLU_d,
                                     hypre_ParVector *f,  hypre_ParVector *u, NALU_HYPRE_Int *perm, NALU_HYPRE_Int n, hypre_ParVector *ftemp,
                                     hypre_ParVector *utemp, hypre_Vector *xtemp_local,
                                     hypre_Vector **Adiag_diag, NALU_HYPRE_Int lower_jacobi_iters, NALU_HYPRE_Int upper_jacobi_iters);
NALU_HYPRE_Int hypre_ILUSolveLUJacobiIter(hypre_CSRMatrix *A, hypre_Vector *work1_local,
                                     hypre_Vector *work2_local,
                                     hypre_Vector *inout_local, hypre_Vector *diag_diag, NALU_HYPRE_Int lower_jacobi_iters,
                                     NALU_HYPRE_Int upper_jacobi_iters, NALU_HYPRE_Int my_id);
NALU_HYPRE_Int hypre_ILUSolveLJacobiIter(hypre_CSRMatrix *A, hypre_Vector *input_local,
                                    hypre_Vector *work_local,
                                    hypre_Vector *output_local, NALU_HYPRE_Int lower_jacobi_iters);
NALU_HYPRE_Int hypre_ILUSolveUJacobiIter(hypre_CSRMatrix *A, hypre_Vector *input_local,
                                    hypre_Vector *work_local,
                                    hypre_Vector *output_local, hypre_Vector *diag_diag, NALU_HYPRE_Int upper_jacobi_iters);
#endif

#ifdef NALU_HYPRE_USING_CUDA
NALU_HYPRE_Int hypre_ILUSolveCusparseLU(hypre_ParCSRMatrix *A, cusparseMatDescr_t matL_des,
                                   cusparseMatDescr_t matU_des, csrsv2Info_t matL_info, csrsv2Info_t matU_info,
                                   hypre_CSRMatrix *matLU_d, cusparseSolvePolicy_t ilu_solve_policy, void *ilu_solve_buffer,
                                   hypre_ParVector *f,  hypre_ParVector *u, NALU_HYPRE_Int *perm, NALU_HYPRE_Int n, hypre_ParVector *ftemp,
                                   hypre_ParVector *utemp);
NALU_HYPRE_Int hypre_ILUSolveCusparseSchurGMRES(hypre_ParCSRMatrix *A, hypre_ParVector *f,
                                           hypre_ParVector *u, NALU_HYPRE_Int *perm, NALU_HYPRE_Int nLU, hypre_ParCSRMatrix *S, hypre_ParVector *ftemp,
                                           hypre_ParVector *utemp, NALU_HYPRE_Solver schur_solver, NALU_HYPRE_Solver schur_precond, hypre_ParVector *rhs,
                                           hypre_ParVector *x, NALU_HYPRE_Int *u_end, cusparseMatDescr_t matL_des, cusparseMatDescr_t matU_des,
                                           csrsv2Info_t matBL_info, csrsv2Info_t matBU_info, csrsv2Info_t matSL_info, csrsv2Info_t matSU_info,
                                           hypre_CSRMatrix *matBLU_d, hypre_CSRMatrix *matE_d, hypre_CSRMatrix *matF_d,
                                           cusparseSolvePolicy_t ilu_solve_policy, void *ilu_solve_buffer);
NALU_HYPRE_Int hypre_ILUSolveRAPGMRES(hypre_ParCSRMatrix *A, hypre_ParVector *f, hypre_ParVector *u,
                                 NALU_HYPRE_Int *perm, NALU_HYPRE_Int nLU, hypre_ParCSRMatrix *S, hypre_ParVector *ftemp,
                                 hypre_ParVector *utemp, hypre_ParVector *xtemp, hypre_ParVector *ytemp, NALU_HYPRE_Solver schur_solver,
                                 NALU_HYPRE_Solver schur_precond, hypre_ParVector *rhs, hypre_ParVector *x, NALU_HYPRE_Int *u_end,
                                 cusparseMatDescr_t matL_des, cusparseMatDescr_t matU_des, csrsv2Info_t matAL_info,
                                 csrsv2Info_t matAU_info, csrsv2Info_t matBL_info, csrsv2Info_t matBU_info, csrsv2Info_t matSL_info,
                                 csrsv2Info_t matSU_info, hypre_ParCSRMatrix *Aperm, hypre_CSRMatrix *matALU_d,
                                 hypre_CSRMatrix *matBLU_d, hypre_CSRMatrix *matE_d, hypre_CSRMatrix *matF_d,
                                 cusparseSolvePolicy_t ilu_solve_policy, void *ilu_solve_buffer, NALU_HYPRE_Int test_opt);
NALU_HYPRE_Int hypre_ParILUCusparseExtractDiagonalCSR(hypre_ParCSRMatrix *A, NALU_HYPRE_Int *perm,
                                                 NALU_HYPRE_Int *rqperm, hypre_CSRMatrix **A_diagp);
NALU_HYPRE_Int hypre_ParILUCusparseILUExtractEBFC(hypre_CSRMatrix *A_diag, NALU_HYPRE_Int nLU,
                                             hypre_CSRMatrix **Bp, hypre_CSRMatrix **Cp, hypre_CSRMatrix **Ep, hypre_CSRMatrix **Fp);
NALU_HYPRE_Int NALU_HYPRE_ILUSetupCusparseCSRILU0(hypre_CSRMatrix *A, cusparseSolvePolicy_t ilu_solve_policy);
NALU_HYPRE_Int NALU_HYPRE_ILUSetupCusparseCSRILU0SetupSolve(hypre_CSRMatrix *A, cusparseMatDescr_t matL_des,
                                                  cusparseMatDescr_t matU_des, cusparseSolvePolicy_t ilu_solve_policy, csrsv2Info_t *matL_infop,
                                                  csrsv2Info_t *matU_infop, NALU_HYPRE_Int *buffer_sizep, void **bufferp);
NALU_HYPRE_Int hypre_ILUSetupILU0Device(hypre_ParCSRMatrix *A, NALU_HYPRE_Int *perm, NALU_HYPRE_Int *qperm,
                                   NALU_HYPRE_Int n, NALU_HYPRE_Int nLU, cusparseMatDescr_t matL_des, cusparseMatDescr_t matU_des,
                                   cusparseSolvePolicy_t ilu_solve_policy, void **bufferp, csrsv2Info_t *matBL_infop,
                                   csrsv2Info_t *matBU_infop, csrsv2Info_t *matSL_infop, csrsv2Info_t *matSU_infop,
                                   hypre_CSRMatrix **BLUptr, hypre_ParCSRMatrix **matSptr, hypre_CSRMatrix **Eptr,
                                   hypre_CSRMatrix **Fptr, NALU_HYPRE_Int **A_fake_diag_ip, NALU_HYPRE_Int tri_solve);
NALU_HYPRE_Int hypre_ILUSetupILUKDevice(hypre_ParCSRMatrix *A, NALU_HYPRE_Int lfil, NALU_HYPRE_Int *perm,
                                   NALU_HYPRE_Int *qperm, NALU_HYPRE_Int n, NALU_HYPRE_Int nLU, cusparseMatDescr_t matL_des,
                                   cusparseMatDescr_t matU_des, cusparseSolvePolicy_t ilu_solve_policy, void **bufferp,
                                   csrsv2Info_t *matBL_infop, csrsv2Info_t *matBU_infop, csrsv2Info_t *matSL_infop,
                                   csrsv2Info_t *matSU_infop, hypre_CSRMatrix **BLUptr, hypre_ParCSRMatrix **matSptr,
                                   hypre_CSRMatrix **Eptr, hypre_CSRMatrix **Fptr, NALU_HYPRE_Int **A_fake_diag_ip, NALU_HYPRE_Int tri_solve);
NALU_HYPRE_Int hypre_ILUSetupILUTDevice(hypre_ParCSRMatrix *A, NALU_HYPRE_Int lfil, NALU_HYPRE_Real *tol,
                                   NALU_HYPRE_Int *perm, NALU_HYPRE_Int *qperm, NALU_HYPRE_Int n, NALU_HYPRE_Int nLU, cusparseMatDescr_t matL_des,
                                   cusparseMatDescr_t matU_des, cusparseSolvePolicy_t ilu_solve_policy, void **bufferp,
                                   csrsv2Info_t *matBL_infop, csrsv2Info_t *matBU_infop, csrsv2Info_t *matSL_infop,
                                   csrsv2Info_t *matSU_infop, hypre_CSRMatrix **BLUptr, hypre_ParCSRMatrix **matSptr,
                                   hypre_CSRMatrix **Eptr, hypre_CSRMatrix **Fptr, NALU_HYPRE_Int **A_fake_diag_ip, NALU_HYPRE_Int tri_solve);
NALU_HYPRE_Int hypre_ParILURAPReorder(hypre_ParCSRMatrix *A, NALU_HYPRE_Int *perm, NALU_HYPRE_Int *rqperm,
                                 hypre_ParCSRMatrix **A_pq);
NALU_HYPRE_Int hypre_ILUSetupLDUtoCusparse(hypre_ParCSRMatrix *L, NALU_HYPRE_Real *D, hypre_ParCSRMatrix *U,
                                      hypre_ParCSRMatrix **LDUp);
NALU_HYPRE_Int hypre_ParILURAPBuildRP(hypre_ParCSRMatrix *A, hypre_ParCSRMatrix *BLUm,
                                 hypre_ParCSRMatrix* E, hypre_ParCSRMatrix *F, cusparseMatDescr_t matL_des,
                                 cusparseMatDescr_t matU_des, hypre_ParCSRMatrix **Rp, hypre_ParCSRMatrix **Pp);
NALU_HYPRE_Int hypre_ILUSetupRAPMILU0(hypre_ParCSRMatrix *A, hypre_ParCSRMatrix **ALUp,
                                 NALU_HYPRE_Int modified);
NALU_HYPRE_Int hypre_ILUSetupRAPILU0Device(hypre_ParCSRMatrix *A, NALU_HYPRE_Int *perm, NALU_HYPRE_Int n,
                                      NALU_HYPRE_Int nLU, cusparseMatDescr_t matL_des, cusparseMatDescr_t matU_des,
                                      cusparseSolvePolicy_t ilu_solve_policy, void **bufferp, csrsv2Info_t *matAL_infop,
                                      csrsv2Info_t *matAU_infop, csrsv2Info_t *matBL_infop, csrsv2Info_t *matBU_infop,
                                      csrsv2Info_t *matSL_infop, csrsv2Info_t *matSU_infop, hypre_ParCSRMatrix **Apermptr,
                                      hypre_ParCSRMatrix **matSptr, hypre_CSRMatrix **ALUptr, hypre_CSRMatrix **BLUptr,
                                      hypre_CSRMatrix **CLUptr, hypre_CSRMatrix **Eptr, hypre_CSRMatrix **Fptr, NALU_HYPRE_Int test_opt);
NALU_HYPRE_Int hypre_ParILUCusparseSchurGMRESDummySetup(void *a, void *b, void *c, void *d);
NALU_HYPRE_Int hypre_ParILUCusparseSchurGMRESDummySolve(void *ilu_vdata, void *ilu_vdata2,
                                                   hypre_ParVector *f, hypre_ParVector *u);
NALU_HYPRE_Int hypre_ParILUCusparseSchurGMRESCommInfo(void *ilu_vdata, NALU_HYPRE_Int *my_id,
                                                 NALU_HYPRE_Int *num_procs);
void *hypre_ParILUCusparseSchurGMRESMatvecCreate(void *ilu_vdata, void *x);
NALU_HYPRE_Int hypre_ParILUCusparseSchurGMRESMatvec(void *matvec_data, NALU_HYPRE_Complex alpha,
                                               void *ilu_vdata, void *x, NALU_HYPRE_Complex beta, void *y);
NALU_HYPRE_Int hypre_ParILUCusparseSchurGMRESMatvecDestroy(void *matvec_data );
NALU_HYPRE_Int hypre_ParILURAPSchurGMRESDummySetup(void *a, void *b, void *c, void *d);
NALU_HYPRE_Int hypre_ParILURAPSchurGMRESSolve(void *ilu_vdata, void *ilu_vdata2, hypre_ParVector *f,
                                         hypre_ParVector *u);
void *hypre_ParILURAPSchurGMRESMatvecCreate(void *ilu_vdata, void *x);
NALU_HYPRE_Int hypre_ParILURAPSchurGMRESMatvec(void *matvec_data, NALU_HYPRE_Complex alpha, void *ilu_vdata,
                                          void *x, NALU_HYPRE_Complex beta, void *y);
NALU_HYPRE_Int hypre_ParILURAPSchurGMRESMatvecDestroy(void *matvec_data );
#endif

#endif /* #ifndef hypre_ParILU_DATA_HEADER */

