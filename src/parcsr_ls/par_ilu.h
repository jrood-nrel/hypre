/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#ifndef nalu_hypre_ParILU_DATA_HEADER
#define nalu_hypre_ParILU_DATA_HEADER

/*--------------------------------------------------------------------------
 * nalu_hypre_ParILUData
 *--------------------------------------------------------------------------*/
typedef struct nalu_hypre_ParILUData_struct
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
   nalu_hypre_CSRMatrix         *matALU_d;//the matrix holding ILU of A (for A-smoothing)
   nalu_hypre_CSRMatrix         *matBLU_d;//the matrix holding ILU of B
   nalu_hypre_CSRMatrix         *matSLU_d;//the matrix holding ILU of S
   nalu_hypre_CSRMatrix         *matE_d;
   nalu_hypre_CSRMatrix         *matF_d;
   nalu_hypre_ParCSRMatrix      *Aperm;
   nalu_hypre_ParCSRMatrix      *R;
   nalu_hypre_ParCSRMatrix      *P;
   nalu_hypre_Vector            *Ftemp_upper;
   nalu_hypre_Vector            *Utemp_lower;
   NALU_HYPRE_Int               *A_diag_fake;//fake diagonal, pretend the diagonal matrix is empty
   nalu_hypre_Vector            *Adiag_diag;
#endif
   //general data
   NALU_HYPRE_Int            global_solver;
   nalu_hypre_ParCSRMatrix   *matA;
   nalu_hypre_ParCSRMatrix   *matL;
   NALU_HYPRE_Real           *matD;
   nalu_hypre_ParCSRMatrix   *matU;
   nalu_hypre_ParCSRMatrix   *matmL;
   NALU_HYPRE_Real           *matmD;
   nalu_hypre_ParCSRMatrix   *matmU;
   nalu_hypre_ParCSRMatrix   *matS;
   NALU_HYPRE_Real
   *droptol;/* should be an array of 3 element, for B, (E and F), S respectively */
   NALU_HYPRE_Int            lfil;
   NALU_HYPRE_Int            maxRowNnz;
   NALU_HYPRE_Int            *CF_marker_array;
   NALU_HYPRE_Int            *perm;
   NALU_HYPRE_Int            *qperm;
   NALU_HYPRE_Real           tol_ddPQ;
   nalu_hypre_ParVector      *F;
   nalu_hypre_ParVector      *U;
   nalu_hypre_ParVector      *residual;
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
   nalu_hypre_ParVector      *Utemp;
   nalu_hypre_ParVector      *Ftemp;
   nalu_hypre_ParVector      *Xtemp;
   nalu_hypre_ParVector      *Ytemp;
   nalu_hypre_Vector         *Ztemp;
   NALU_HYPRE_Real           *uext;
   NALU_HYPRE_Real           *fext;

   /* data structure sor solving Schur System */
   NALU_HYPRE_Solver         schur_solver;
   NALU_HYPRE_Solver         schur_precond;
   nalu_hypre_ParVector      *rhs;
   nalu_hypre_ParVector      *x;

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

} nalu_hypre_ParILUData;

#define nalu_hypre_ParILUDataTestOption(ilu_data)                   ((ilu_data) -> test_opt)

#ifdef NALU_HYPRE_USING_CUDA
#define nalu_hypre_ParILUDataMatLMatrixDescription(ilu_data)        ((ilu_data) -> matL_des)
#define nalu_hypre_ParILUDataMatUMatrixDescription(ilu_data)        ((ilu_data) -> matU_des)
#define nalu_hypre_ParILUDataMatALILUSolveInfo(ilu_data)            ((ilu_data) -> matAL_info)
#define nalu_hypre_ParILUDataMatAUILUSolveInfo(ilu_data)            ((ilu_data) -> matAU_info)
#define nalu_hypre_ParILUDataMatBLILUSolveInfo(ilu_data)            ((ilu_data) -> matBL_info)
#define nalu_hypre_ParILUDataMatBUILUSolveInfo(ilu_data)            ((ilu_data) -> matBU_info)
#define nalu_hypre_ParILUDataMatSLILUSolveInfo(ilu_data)            ((ilu_data) -> matSL_info)
#define nalu_hypre_ParILUDataMatSUILUSolveInfo(ilu_data)            ((ilu_data) -> matSU_info)
#define nalu_hypre_ParILUDataILUSolveBuffer(ilu_data)               ((ilu_data) -> ilu_solve_buffer)
#define nalu_hypre_ParILUDataILUSolvePolicy(ilu_data)               ((ilu_data) -> ilu_solve_policy)
#define nalu_hypre_ParILUDataMatAILUDevice(ilu_data)                ((ilu_data) -> matALU_d)
#define nalu_hypre_ParILUDataMatBILUDevice(ilu_data)                ((ilu_data) -> matBLU_d)
#define nalu_hypre_ParILUDataMatSILUDevice(ilu_data)                ((ilu_data) -> matSLU_d)
#define nalu_hypre_ParILUDataMatEDevice(ilu_data)                   ((ilu_data) -> matE_d)
#define nalu_hypre_ParILUDataMatFDevice(ilu_data)                   ((ilu_data) -> matF_d)
#define nalu_hypre_ParILUDataAperm(ilu_data)                        ((ilu_data) -> Aperm)
#define nalu_hypre_ParILUDataR(ilu_data)                            ((ilu_data) -> R)
#define nalu_hypre_ParILUDataP(ilu_data)                            ((ilu_data) -> P)
#define nalu_hypre_ParILUDataFTempUpper(ilu_data)                   ((ilu_data) -> Ftemp_upper)
#define nalu_hypre_ParILUDataUTempLower(ilu_data)                   ((ilu_data) -> Utemp_lower)
#define nalu_hypre_ParILUDataMatAFakeDiagonal(ilu_data)             ((ilu_data) -> A_diag_fake)
#define nalu_hypre_ParILUDataADiagDiag(ilu_data)                    ((ilu_data) -> Adiag_diag)
#endif

#define nalu_hypre_ParILUDataGlobalSolver(ilu_data)                 ((ilu_data) -> global_solver)
#define nalu_hypre_ParILUDataMatA(ilu_data)                         ((ilu_data) -> matA)
#define nalu_hypre_ParILUDataMatL(ilu_data)                         ((ilu_data) -> matL)
#define nalu_hypre_ParILUDataMatD(ilu_data)                         ((ilu_data) -> matD)
#define nalu_hypre_ParILUDataMatU(ilu_data)                         ((ilu_data) -> matU)
#define nalu_hypre_ParILUDataMatLModified(ilu_data)                 ((ilu_data) -> matmL)
#define nalu_hypre_ParILUDataMatDModified(ilu_data)                 ((ilu_data) -> matmD)
#define nalu_hypre_ParILUDataMatUModified(ilu_data)                 ((ilu_data) -> matmU)
#define nalu_hypre_ParILUDataMatS(ilu_data)                         ((ilu_data) -> matS)
#define nalu_hypre_ParILUDataDroptol(ilu_data)                      ((ilu_data) -> droptol)
#define nalu_hypre_ParILUDataLfil(ilu_data)                         ((ilu_data) -> lfil)
#define nalu_hypre_ParILUDataMaxRowNnz(ilu_data)                    ((ilu_data) -> maxRowNnz)
#define nalu_hypre_ParILUDataCFMarkerArray(ilu_data)                ((ilu_data) -> CF_marker_array)
#define nalu_hypre_ParILUDataPerm(ilu_data)                         ((ilu_data) -> perm)
#define nalu_hypre_ParILUDataPPerm(ilu_data)                        ((ilu_data) -> perm)
#define nalu_hypre_ParILUDataQPerm(ilu_data)                        ((ilu_data) -> qperm)
#define nalu_hypre_ParILUDataTolDDPQ(ilu_data)                      ((ilu_data) -> tol_ddPQ)
#define nalu_hypre_ParILUDataF(ilu_data)                            ((ilu_data) -> F)
#define nalu_hypre_ParILUDataU(ilu_data)                            ((ilu_data) -> U)
#define nalu_hypre_ParILUDataResidual(ilu_data)                     ((ilu_data) -> residual)
#define nalu_hypre_ParILUDataRelResNorms(ilu_data)                  ((ilu_data) -> rel_res_norms)
#define nalu_hypre_ParILUDataNumIterations(ilu_data)                ((ilu_data) -> num_iterations)
#define nalu_hypre_ParILUDataL1Norms(ilu_data)                      ((ilu_data) -> l1_norms)
#define nalu_hypre_ParILUDataFinalRelResidualNorm(ilu_data)         ((ilu_data) -> final_rel_residual_norm)
#define nalu_hypre_ParILUDataTol(ilu_data)                          ((ilu_data) -> tol)
#define nalu_hypre_ParILUDataOperatorComplexity(ilu_data)           ((ilu_data) -> operator_complexity)
#define nalu_hypre_ParILUDataLogging(ilu_data)                      ((ilu_data) -> logging)
#define nalu_hypre_ParILUDataPrintLevel(ilu_data)                   ((ilu_data) -> print_level)
#define nalu_hypre_ParILUDataMaxIter(ilu_data)                      ((ilu_data) -> max_iter)
#define nalu_hypre_ParILUDataTriSolve(ilu_data)                     ((ilu_data) -> tri_solve)
#define nalu_hypre_ParILUDataLowerJacobiIters(ilu_data)             ((ilu_data) -> lower_jacobi_iters)
#define nalu_hypre_ParILUDataUpperJacobiIters(ilu_data)             ((ilu_data) -> upper_jacobi_iters)
#define nalu_hypre_ParILUDataIluType(ilu_data)                      ((ilu_data) -> ilu_type)
#define nalu_hypre_ParILUDataNLU(ilu_data)                          ((ilu_data) -> nLU)
#define nalu_hypre_ParILUDataNI(ilu_data)                           ((ilu_data) -> nI)
#define nalu_hypre_ParILUDataUEnd(ilu_data)                         ((ilu_data) -> u_end)
#define nalu_hypre_ParILUDataXTemp(ilu_data)                        ((ilu_data) -> Xtemp)
#define nalu_hypre_ParILUDataYTemp(ilu_data)                        ((ilu_data) -> Ytemp)
#define nalu_hypre_ParILUDataZTemp(ilu_data)                        ((ilu_data) -> Ztemp)
#define nalu_hypre_ParILUDataUTemp(ilu_data)                        ((ilu_data) -> Utemp)
#define nalu_hypre_ParILUDataFTemp(ilu_data)                        ((ilu_data) -> Ftemp)
#define nalu_hypre_ParILUDataUExt(ilu_data)                         ((ilu_data) -> uext)
#define nalu_hypre_ParILUDataFExt(ilu_data)                         ((ilu_data) -> fext)
#define nalu_hypre_ParILUDataSchurSolver(ilu_data)                  ((ilu_data) -> schur_solver)
#define nalu_hypre_ParILUDataSchurPrecond(ilu_data)                 ((ilu_data) -> schur_precond)
#define nalu_hypre_ParILUDataRhs(ilu_data)                          ((ilu_data) -> rhs)
#define nalu_hypre_ParILUDataX(ilu_data)                            ((ilu_data) -> x)
#define nalu_hypre_ParILUDataReorderingType(ilu_data)               ((ilu_data) -> reordering_type)
/* Schur System */
#define nalu_hypre_ParILUDataSchurGMRESKDim(ilu_data)               ((ilu_data) -> ss_kDim)
#define nalu_hypre_ParILUDataSchurGMRESMaxIter(ilu_data)            ((ilu_data) -> ss_max_iter)
#define nalu_hypre_ParILUDataSchurGMRESTol(ilu_data)                ((ilu_data) -> ss_tol)
#define nalu_hypre_ParILUDataSchurGMRESAbsoluteTol(ilu_data)        ((ilu_data) -> ss_absolute_tol)
#define nalu_hypre_ParILUDataSchurGMRESRelChange(ilu_data)          ((ilu_data) -> ss_rel_change)
#define nalu_hypre_ParILUDataSchurPrecondIluType(ilu_data)          ((ilu_data) -> sp_ilu_type)
#define nalu_hypre_ParILUDataSchurPrecondIluLfil(ilu_data)          ((ilu_data) -> sp_ilu_lfil)
#define nalu_hypre_ParILUDataSchurPrecondIluMaxRowNnz(ilu_data)     ((ilu_data) -> sp_ilu_max_row_nnz)
#define nalu_hypre_ParILUDataSchurPrecondIluDroptol(ilu_data)       ((ilu_data) -> sp_ilu_droptol)
#define nalu_hypre_ParILUDataSchurPrecondPrintLevel(ilu_data)       ((ilu_data) -> sp_print_level)
#define nalu_hypre_ParILUDataSchurPrecondMaxIter(ilu_data)          ((ilu_data) -> sp_max_iter)
#define nalu_hypre_ParILUDataSchurPrecondTriSolve(ilu_data)         ((ilu_data) -> sp_tri_solve)
#define nalu_hypre_ParILUDataSchurPrecondLowerJacobiIters(ilu_data) ((ilu_data) -> sp_lower_jacobi_iters)
#define nalu_hypre_ParILUDataSchurPrecondUpperJacobiIters(ilu_data) ((ilu_data) -> sp_upper_jacobi_iters)
#define nalu_hypre_ParILUDataSchurPrecondTol(ilu_data)              ((ilu_data) -> sp_tol)

#define nalu_hypre_ParILUDataSchurNSHMaxNumIter(ilu_data)           ((ilu_data) -> ss_nsh_setup_max_iter)
#define nalu_hypre_ParILUDataSchurNSHSolveMaxIter(ilu_data)         ((ilu_data) -> ss_nsh_solve_max_iter)
#define nalu_hypre_ParILUDataSchurNSHTol(ilu_data)                  ((ilu_data) -> ss_nsh_setup_tol)
#define nalu_hypre_ParILUDataSchurNSHSolveTol(ilu_data)             ((ilu_data) -> ss_nsh_solve_tol)
#define nalu_hypre_ParILUDataSchurNSHMaxRowNnz(ilu_data)            ((ilu_data) -> ss_nsh_max_row_nnz)
#define nalu_hypre_ParILUDataSchurMRColVersion(ilu_data)            ((ilu_data) -> ss_nsh_mr_col_version)
#define nalu_hypre_ParILUDataSchurMRMaxRowNnz(ilu_data)             ((ilu_data) -> ss_nsh_mr_max_row_nnz)
#define nalu_hypre_ParILUDataSchurNSHDroptol(ilu_data)              ((ilu_data) -> ss_nsh_droptol)
#define nalu_hypre_ParILUDataSchurMRMaxIter(ilu_data)               ((ilu_data) -> ss_nsh_mr_max_iter)
#define nalu_hypre_ParILUDataSchurMRTol(ilu_data)                   ((ilu_data) -> ss_nsh_mr_tol)

#define nalu_hypre_ParILUDataSchurSolverLogging(ilu_data)           ((ilu_data) -> ss_logging)
#define nalu_hypre_ParILUDataSchurSolverPrintLevel(ilu_data)        ((ilu_data) -> ss_print_level)

#define FMRK  -1
#define CMRK  1
#define UMRK  0
#define S_CMRK  2

#define FPT(i, bsize) (((i) % (bsize)) == FMRK)
#define CPT(i, bsize) (((i) % (bsize)) == CMRK)

#define MAT_TOL 1e-14
#define EXPAND_FACT 1.3





/* NSH data structure */

typedef struct nalu_hypre_ParNSHData_struct
{
   /* solver information */
   NALU_HYPRE_Int             global_solver;
   nalu_hypre_ParCSRMatrix    *matA;
   nalu_hypre_ParCSRMatrix    *matM;
   nalu_hypre_ParVector       *F;
   nalu_hypre_ParVector       *U;
   nalu_hypre_ParVector       *residual;
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
   nalu_hypre_ParVector       *Utemp;
   nalu_hypre_ParVector       *Ftemp;

   /* data slots for local MR */
   NALU_HYPRE_Int             mr_max_iter;
   NALU_HYPRE_Real            mr_tol;
   NALU_HYPRE_Int             mr_max_row_nnz;
   NALU_HYPRE_Int             mr_col_version;/* global version or column version MR */

   /* data slots for global NSH */
   NALU_HYPRE_Int             nsh_max_iter;
   NALU_HYPRE_Real            nsh_tol;
   NALU_HYPRE_Int             nsh_max_row_nnz;
} nalu_hypre_ParNSHData;

#define nalu_hypre_ParNSHDataGlobalSolver(nsh_data)           ((nsh_data) -> global_solver)
#define nalu_hypre_ParNSHDataMatA(nsh_data)                   ((nsh_data) -> matA)
#define nalu_hypre_ParNSHDataMatM(nsh_data)                   ((nsh_data) -> matM)
#define nalu_hypre_ParNSHDataF(nsh_data)                      ((nsh_data) -> F)
#define nalu_hypre_ParNSHDataU(nsh_data)                      ((nsh_data) -> U)
#define nalu_hypre_ParNSHDataResidual(nsh_data)               ((nsh_data) -> residual)
#define nalu_hypre_ParNSHDataRelResNorms(nsh_data)            ((nsh_data) -> rel_res_norms)
#define nalu_hypre_ParNSHDataNumIterations(nsh_data)          ((nsh_data) -> num_iterations)
#define nalu_hypre_ParNSHDataL1Norms(nsh_data)                ((nsh_data) -> l1_norms)
#define nalu_hypre_ParNSHDataFinalRelResidualNorm(nsh_data)   ((nsh_data) -> final_rel_residual_norm)
#define nalu_hypre_ParNSHDataTol(nsh_data)                    ((nsh_data) -> tol)
#define nalu_hypre_ParNSHDataOperatorComplexity(nsh_data)     ((nsh_data) -> operator_complexity)
#define nalu_hypre_ParNSHDataLogging(nsh_data)                ((nsh_data) -> logging)
#define nalu_hypre_ParNSHDataPrintLevel(nsh_data)             ((nsh_data) -> print_level)
#define nalu_hypre_ParNSHDataMaxIter(nsh_data)                ((nsh_data) -> max_iter)
#define nalu_hypre_ParNSHDataDroptol(nsh_data)                ((nsh_data) -> droptol)
#define nalu_hypre_ParNSHDataOwnDroptolData(nsh_data)         ((nsh_data) -> own_droptol_data)
#define nalu_hypre_ParNSHDataUTemp(nsh_data)                  ((nsh_data) -> Utemp)
#define nalu_hypre_ParNSHDataFTemp(nsh_data)                  ((nsh_data) -> Ftemp)
#define nalu_hypre_ParNSHDataMRMaxIter(nsh_data)              ((nsh_data) -> mr_max_iter)
#define nalu_hypre_ParNSHDataMRTol(nsh_data)                  ((nsh_data) -> mr_tol)
#define nalu_hypre_ParNSHDataMRMaxRowNnz(nsh_data)            ((nsh_data) -> mr_max_row_nnz)
#define nalu_hypre_ParNSHDataMRColVersion(nsh_data)           ((nsh_data) -> mr_col_version)
#define nalu_hypre_ParNSHDataNSHMaxIter(nsh_data)             ((nsh_data) -> nsh_max_iter)
#define nalu_hypre_ParNSHDataNSHTol(nsh_data)                 ((nsh_data) -> nsh_tol)
#define nalu_hypre_ParNSHDataNSHMaxRowNnz(nsh_data)           ((nsh_data) -> nsh_max_row_nnz)

//#define DIVIDE_TOL 1e-32

#ifdef NALU_HYPRE_USING_GPU
NALU_HYPRE_Int nalu_hypre_ILUSolveDeviceLUIter(nalu_hypre_ParCSRMatrix *A, nalu_hypre_CSRMatrix *matLU_d,
                                     nalu_hypre_ParVector *f,  nalu_hypre_ParVector *u, NALU_HYPRE_Int *perm, NALU_HYPRE_Int n, nalu_hypre_ParVector *ftemp,
                                     nalu_hypre_ParVector *utemp, nalu_hypre_Vector *xtemp_local,
                                     nalu_hypre_Vector **Adiag_diag, NALU_HYPRE_Int lower_jacobi_iters, NALU_HYPRE_Int upper_jacobi_iters);
NALU_HYPRE_Int nalu_hypre_ILUSolveLUJacobiIter(nalu_hypre_CSRMatrix *A, nalu_hypre_Vector *work1_local,
                                     nalu_hypre_Vector *work2_local,
                                     nalu_hypre_Vector *inout_local, nalu_hypre_Vector *diag_diag, NALU_HYPRE_Int lower_jacobi_iters,
                                     NALU_HYPRE_Int upper_jacobi_iters, NALU_HYPRE_Int my_id);
NALU_HYPRE_Int nalu_hypre_ILUSolveLJacobiIter(nalu_hypre_CSRMatrix *A, nalu_hypre_Vector *input_local,
                                    nalu_hypre_Vector *work_local,
                                    nalu_hypre_Vector *output_local, NALU_HYPRE_Int lower_jacobi_iters);
NALU_HYPRE_Int nalu_hypre_ILUSolveUJacobiIter(nalu_hypre_CSRMatrix *A, nalu_hypre_Vector *input_local,
                                    nalu_hypre_Vector *work_local,
                                    nalu_hypre_Vector *output_local, nalu_hypre_Vector *diag_diag, NALU_HYPRE_Int upper_jacobi_iters);
#endif

#ifdef NALU_HYPRE_USING_CUDA
NALU_HYPRE_Int nalu_hypre_ILUSolveCusparseLU(nalu_hypre_ParCSRMatrix *A, cusparseMatDescr_t matL_des,
                                   cusparseMatDescr_t matU_des, csrsv2Info_t matL_info, csrsv2Info_t matU_info,
                                   nalu_hypre_CSRMatrix *matLU_d, cusparseSolvePolicy_t ilu_solve_policy, void *ilu_solve_buffer,
                                   nalu_hypre_ParVector *f,  nalu_hypre_ParVector *u, NALU_HYPRE_Int *perm, NALU_HYPRE_Int n, nalu_hypre_ParVector *ftemp,
                                   nalu_hypre_ParVector *utemp);
NALU_HYPRE_Int nalu_hypre_ILUSolveCusparseSchurGMRES(nalu_hypre_ParCSRMatrix *A, nalu_hypre_ParVector *f,
                                           nalu_hypre_ParVector *u, NALU_HYPRE_Int *perm, NALU_HYPRE_Int nLU, nalu_hypre_ParCSRMatrix *S, nalu_hypre_ParVector *ftemp,
                                           nalu_hypre_ParVector *utemp, NALU_HYPRE_Solver schur_solver, NALU_HYPRE_Solver schur_precond, nalu_hypre_ParVector *rhs,
                                           nalu_hypre_ParVector *x, NALU_HYPRE_Int *u_end, cusparseMatDescr_t matL_des, cusparseMatDescr_t matU_des,
                                           csrsv2Info_t matBL_info, csrsv2Info_t matBU_info, csrsv2Info_t matSL_info, csrsv2Info_t matSU_info,
                                           nalu_hypre_CSRMatrix *matBLU_d, nalu_hypre_CSRMatrix *matE_d, nalu_hypre_CSRMatrix *matF_d,
                                           cusparseSolvePolicy_t ilu_solve_policy, void *ilu_solve_buffer);
NALU_HYPRE_Int nalu_hypre_ILUSolveRAPGMRES(nalu_hypre_ParCSRMatrix *A, nalu_hypre_ParVector *f, nalu_hypre_ParVector *u,
                                 NALU_HYPRE_Int *perm, NALU_HYPRE_Int nLU, nalu_hypre_ParCSRMatrix *S, nalu_hypre_ParVector *ftemp,
                                 nalu_hypre_ParVector *utemp, nalu_hypre_ParVector *xtemp, nalu_hypre_ParVector *ytemp, NALU_HYPRE_Solver schur_solver,
                                 NALU_HYPRE_Solver schur_precond, nalu_hypre_ParVector *rhs, nalu_hypre_ParVector *x, NALU_HYPRE_Int *u_end,
                                 cusparseMatDescr_t matL_des, cusparseMatDescr_t matU_des, csrsv2Info_t matAL_info,
                                 csrsv2Info_t matAU_info, csrsv2Info_t matBL_info, csrsv2Info_t matBU_info, csrsv2Info_t matSL_info,
                                 csrsv2Info_t matSU_info, nalu_hypre_ParCSRMatrix *Aperm, nalu_hypre_CSRMatrix *matALU_d,
                                 nalu_hypre_CSRMatrix *matBLU_d, nalu_hypre_CSRMatrix *matE_d, nalu_hypre_CSRMatrix *matF_d,
                                 cusparseSolvePolicy_t ilu_solve_policy, void *ilu_solve_buffer, NALU_HYPRE_Int test_opt);
NALU_HYPRE_Int nalu_hypre_ParILUCusparseExtractDiagonalCSR(nalu_hypre_ParCSRMatrix *A, NALU_HYPRE_Int *perm,
                                                 NALU_HYPRE_Int *rqperm, nalu_hypre_CSRMatrix **A_diagp);
NALU_HYPRE_Int nalu_hypre_ParILUCusparseILUExtractEBFC(nalu_hypre_CSRMatrix *A_diag, NALU_HYPRE_Int nLU,
                                             nalu_hypre_CSRMatrix **Bp, nalu_hypre_CSRMatrix **Cp, nalu_hypre_CSRMatrix **Ep, nalu_hypre_CSRMatrix **Fp);
NALU_HYPRE_Int NALU_HYPRE_ILUSetupCusparseCSRILU0(nalu_hypre_CSRMatrix *A, cusparseSolvePolicy_t ilu_solve_policy);
NALU_HYPRE_Int NALU_HYPRE_ILUSetupCusparseCSRILU0SetupSolve(nalu_hypre_CSRMatrix *A, cusparseMatDescr_t matL_des,
                                                  cusparseMatDescr_t matU_des, cusparseSolvePolicy_t ilu_solve_policy, csrsv2Info_t *matL_infop,
                                                  csrsv2Info_t *matU_infop, NALU_HYPRE_Int *buffer_sizep, void **bufferp);
NALU_HYPRE_Int nalu_hypre_ILUSetupILU0Device(nalu_hypre_ParCSRMatrix *A, NALU_HYPRE_Int *perm, NALU_HYPRE_Int *qperm,
                                   NALU_HYPRE_Int n, NALU_HYPRE_Int nLU, cusparseMatDescr_t matL_des, cusparseMatDescr_t matU_des,
                                   cusparseSolvePolicy_t ilu_solve_policy, void **bufferp, csrsv2Info_t *matBL_infop,
                                   csrsv2Info_t *matBU_infop, csrsv2Info_t *matSL_infop, csrsv2Info_t *matSU_infop,
                                   nalu_hypre_CSRMatrix **BLUptr, nalu_hypre_ParCSRMatrix **matSptr, nalu_hypre_CSRMatrix **Eptr,
                                   nalu_hypre_CSRMatrix **Fptr, NALU_HYPRE_Int **A_fake_diag_ip, NALU_HYPRE_Int tri_solve);
NALU_HYPRE_Int nalu_hypre_ILUSetupILUKDevice(nalu_hypre_ParCSRMatrix *A, NALU_HYPRE_Int lfil, NALU_HYPRE_Int *perm,
                                   NALU_HYPRE_Int *qperm, NALU_HYPRE_Int n, NALU_HYPRE_Int nLU, cusparseMatDescr_t matL_des,
                                   cusparseMatDescr_t matU_des, cusparseSolvePolicy_t ilu_solve_policy, void **bufferp,
                                   csrsv2Info_t *matBL_infop, csrsv2Info_t *matBU_infop, csrsv2Info_t *matSL_infop,
                                   csrsv2Info_t *matSU_infop, nalu_hypre_CSRMatrix **BLUptr, nalu_hypre_ParCSRMatrix **matSptr,
                                   nalu_hypre_CSRMatrix **Eptr, nalu_hypre_CSRMatrix **Fptr, NALU_HYPRE_Int **A_fake_diag_ip, NALU_HYPRE_Int tri_solve);
NALU_HYPRE_Int nalu_hypre_ILUSetupILUTDevice(nalu_hypre_ParCSRMatrix *A, NALU_HYPRE_Int lfil, NALU_HYPRE_Real *tol,
                                   NALU_HYPRE_Int *perm, NALU_HYPRE_Int *qperm, NALU_HYPRE_Int n, NALU_HYPRE_Int nLU, cusparseMatDescr_t matL_des,
                                   cusparseMatDescr_t matU_des, cusparseSolvePolicy_t ilu_solve_policy, void **bufferp,
                                   csrsv2Info_t *matBL_infop, csrsv2Info_t *matBU_infop, csrsv2Info_t *matSL_infop,
                                   csrsv2Info_t *matSU_infop, nalu_hypre_CSRMatrix **BLUptr, nalu_hypre_ParCSRMatrix **matSptr,
                                   nalu_hypre_CSRMatrix **Eptr, nalu_hypre_CSRMatrix **Fptr, NALU_HYPRE_Int **A_fake_diag_ip, NALU_HYPRE_Int tri_solve);
NALU_HYPRE_Int nalu_hypre_ParILURAPReorder(nalu_hypre_ParCSRMatrix *A, NALU_HYPRE_Int *perm, NALU_HYPRE_Int *rqperm,
                                 nalu_hypre_ParCSRMatrix **A_pq);
NALU_HYPRE_Int nalu_hypre_ILUSetupLDUtoCusparse(nalu_hypre_ParCSRMatrix *L, NALU_HYPRE_Real *D, nalu_hypre_ParCSRMatrix *U,
                                      nalu_hypre_ParCSRMatrix **LDUp);
NALU_HYPRE_Int nalu_hypre_ParILURAPBuildRP(nalu_hypre_ParCSRMatrix *A, nalu_hypre_ParCSRMatrix *BLUm,
                                 nalu_hypre_ParCSRMatrix* E, nalu_hypre_ParCSRMatrix *F, cusparseMatDescr_t matL_des,
                                 cusparseMatDescr_t matU_des, nalu_hypre_ParCSRMatrix **Rp, nalu_hypre_ParCSRMatrix **Pp);
NALU_HYPRE_Int nalu_hypre_ILUSetupRAPMILU0(nalu_hypre_ParCSRMatrix *A, nalu_hypre_ParCSRMatrix **ALUp,
                                 NALU_HYPRE_Int modified);
NALU_HYPRE_Int nalu_hypre_ILUSetupRAPILU0Device(nalu_hypre_ParCSRMatrix *A, NALU_HYPRE_Int *perm, NALU_HYPRE_Int n,
                                      NALU_HYPRE_Int nLU, cusparseMatDescr_t matL_des, cusparseMatDescr_t matU_des,
                                      cusparseSolvePolicy_t ilu_solve_policy, void **bufferp, csrsv2Info_t *matAL_infop,
                                      csrsv2Info_t *matAU_infop, csrsv2Info_t *matBL_infop, csrsv2Info_t *matBU_infop,
                                      csrsv2Info_t *matSL_infop, csrsv2Info_t *matSU_infop, nalu_hypre_ParCSRMatrix **Apermptr,
                                      nalu_hypre_ParCSRMatrix **matSptr, nalu_hypre_CSRMatrix **ALUptr, nalu_hypre_CSRMatrix **BLUptr,
                                      nalu_hypre_CSRMatrix **CLUptr, nalu_hypre_CSRMatrix **Eptr, nalu_hypre_CSRMatrix **Fptr, NALU_HYPRE_Int test_opt);
NALU_HYPRE_Int nalu_hypre_ParILUCusparseSchurGMRESDummySetup(void *a, void *b, void *c, void *d);
NALU_HYPRE_Int nalu_hypre_ParILUCusparseSchurGMRESDummySolve(void *ilu_vdata, void *ilu_vdata2,
                                                   nalu_hypre_ParVector *f, nalu_hypre_ParVector *u);
NALU_HYPRE_Int nalu_hypre_ParILUCusparseSchurGMRESCommInfo(void *ilu_vdata, NALU_HYPRE_Int *my_id,
                                                 NALU_HYPRE_Int *num_procs);
void *nalu_hypre_ParILUCusparseSchurGMRESMatvecCreate(void *ilu_vdata, void *x);
NALU_HYPRE_Int nalu_hypre_ParILUCusparseSchurGMRESMatvec(void *matvec_data, NALU_HYPRE_Complex alpha,
                                               void *ilu_vdata, void *x, NALU_HYPRE_Complex beta, void *y);
NALU_HYPRE_Int nalu_hypre_ParILUCusparseSchurGMRESMatvecDestroy(void *matvec_data );
NALU_HYPRE_Int nalu_hypre_ParILURAPSchurGMRESDummySetup(void *a, void *b, void *c, void *d);
NALU_HYPRE_Int nalu_hypre_ParILURAPSchurGMRESSolve(void *ilu_vdata, void *ilu_vdata2, nalu_hypre_ParVector *f,
                                         nalu_hypre_ParVector *u);
void *nalu_hypre_ParILURAPSchurGMRESMatvecCreate(void *ilu_vdata, void *x);
NALU_HYPRE_Int nalu_hypre_ParILURAPSchurGMRESMatvec(void *matvec_data, NALU_HYPRE_Complex alpha, void *ilu_vdata,
                                          void *x, NALU_HYPRE_Complex beta, void *y);
NALU_HYPRE_Int nalu_hypre_ParILURAPSchurGMRESMatvecDestroy(void *matvec_data );
#endif

#endif /* #ifndef nalu_hypre_ParILU_DATA_HEADER */

