/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Incomplete LU factorization smoother
 *
 *****************************************************************************/

#include "_nalu_hypre_parcsr_ls.h"
#include "_nalu_hypre_utilities.hpp"
#include "par_ilu.h"

/* Create */
void *
nalu_hypre_ILUCreate()
{
   nalu_hypre_ParILUData                       *ilu_data;

   ilu_data                               = nalu_hypre_CTAlloc(nalu_hypre_ParILUData,  1, NALU_HYPRE_MEMORY_HOST);

#if defined(NALU_HYPRE_USING_CUDA) && defined(NALU_HYPRE_USING_CUSPARSE)
   nalu_hypre_ParILUDataMatLMatrixDescription(ilu_data) = NULL;
   nalu_hypre_ParILUDataMatUMatrixDescription(ilu_data) = NULL;
   nalu_hypre_ParILUDataMatBLILUSolveInfo(ilu_data) = NULL;
   nalu_hypre_ParILUDataMatBUILUSolveInfo(ilu_data) = NULL;
   nalu_hypre_ParILUDataMatSLILUSolveInfo(ilu_data) = NULL;
   nalu_hypre_ParILUDataMatSUILUSolveInfo(ilu_data) = NULL;
   nalu_hypre_ParILUDataILUSolveBuffer(ilu_data) = NULL;
   nalu_hypre_ParILUDataILUSolvePolicy(ilu_data) = CUSPARSE_SOLVE_POLICY_USE_LEVEL;
   nalu_hypre_ParILUDataAperm(ilu_data) = NULL;
   nalu_hypre_ParILUDataMatBILUDevice(ilu_data) = NULL;
   nalu_hypre_ParILUDataMatSILUDevice(ilu_data) = NULL;
   nalu_hypre_ParILUDataMatEDevice(ilu_data) = NULL;
   nalu_hypre_ParILUDataMatFDevice(ilu_data) = NULL;
   nalu_hypre_ParILUDataR(ilu_data) = NULL;
   nalu_hypre_ParILUDataP(ilu_data) = NULL;
   nalu_hypre_ParILUDataFTempUpper(ilu_data) = NULL;
   nalu_hypre_ParILUDataUTempLower(ilu_data) = NULL;
   nalu_hypre_ParILUDataMatAFakeDiagonal(ilu_data) = NULL;
   nalu_hypre_ParILUDataADiagDiag(ilu_data) = NULL;
#endif

   /* general data */
   nalu_hypre_ParILUDataGlobalSolver(ilu_data) = 0;
   nalu_hypre_ParILUDataMatA(ilu_data) = NULL;
   nalu_hypre_ParILUDataMatL(ilu_data) = NULL;
   nalu_hypre_ParILUDataMatD(ilu_data) = NULL;
   nalu_hypre_ParILUDataMatU(ilu_data) = NULL;
   nalu_hypre_ParILUDataMatS(ilu_data) = NULL;
   nalu_hypre_ParILUDataSchurSolver(ilu_data) = NULL;
   nalu_hypre_ParILUDataSchurPrecond(ilu_data) = NULL;
   nalu_hypre_ParILUDataRhs(ilu_data) = NULL;
   nalu_hypre_ParILUDataX(ilu_data) = NULL;

   nalu_hypre_ParILUDataDroptol(ilu_data) = nalu_hypre_TAlloc(NALU_HYPRE_Real, 3, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_ParILUDataDroptol(ilu_data)[0] = 1.0e-02;/* droptol for B */
   nalu_hypre_ParILUDataDroptol(ilu_data)[1] = 1.0e-02;/* droptol for E and F */
   nalu_hypre_ParILUDataDroptol(ilu_data)[2] = 1.0e-02;/* droptol for S */
   nalu_hypre_ParILUDataLfil(ilu_data) = 0;
   nalu_hypre_ParILUDataMaxRowNnz(ilu_data) = 1000;
   nalu_hypre_ParILUDataCFMarkerArray(ilu_data) = NULL;
   nalu_hypre_ParILUDataPerm(ilu_data) = NULL;
   nalu_hypre_ParILUDataQPerm(ilu_data) = NULL;
   nalu_hypre_ParILUDataTolDDPQ(ilu_data) = 1.0e-01;

   nalu_hypre_ParILUDataF(ilu_data) = NULL;
   nalu_hypre_ParILUDataU(ilu_data) = NULL;
   nalu_hypre_ParILUDataFTemp(ilu_data) = NULL;
   nalu_hypre_ParILUDataUTemp(ilu_data) = NULL;
   nalu_hypre_ParILUDataXTemp(ilu_data) = NULL;
   nalu_hypre_ParILUDataYTemp(ilu_data) = NULL;
   nalu_hypre_ParILUDataZTemp(ilu_data) = NULL;
   nalu_hypre_ParILUDataUExt(ilu_data) = NULL;
   nalu_hypre_ParILUDataFExt(ilu_data) = NULL;
   nalu_hypre_ParILUDataResidual(ilu_data) = NULL;
   nalu_hypre_ParILUDataRelResNorms(ilu_data) = NULL;

   nalu_hypre_ParILUDataNumIterations(ilu_data) = 0;

   nalu_hypre_ParILUDataMaxIter(ilu_data) = 20;
   nalu_hypre_ParILUDataTriSolve(ilu_data) = 1;
   nalu_hypre_ParILUDataLowerJacobiIters(ilu_data) = 5;
   nalu_hypre_ParILUDataUpperJacobiIters(ilu_data) = 5;
   nalu_hypre_ParILUDataTol(ilu_data) = 1.0e-7;

   nalu_hypre_ParILUDataLogging(ilu_data) = 0;
   nalu_hypre_ParILUDataPrintLevel(ilu_data) = 0;

   nalu_hypre_ParILUDataL1Norms(ilu_data) = NULL;

   nalu_hypre_ParILUDataOperatorComplexity(ilu_data) = 0.;

   nalu_hypre_ParILUDataIluType(ilu_data) = 0;
   nalu_hypre_ParILUDataNLU(ilu_data) = 0;
   nalu_hypre_ParILUDataNI(ilu_data) = 0;
   nalu_hypre_ParILUDataUEnd(ilu_data) = NULL;

   /* reordering_type default to use local RCM */
   nalu_hypre_ParILUDataReorderingType(ilu_data) = 1;

   /* see nalu_hypre_ILUSetType for more default values */
   nalu_hypre_ParILUDataTestOption(ilu_data) = 0;

   /* -> GENERAL-SLOTS */
   nalu_hypre_ParILUDataSchurSolverLogging(ilu_data) = 0;
   nalu_hypre_ParILUDataSchurSolverPrintLevel(ilu_data) = 0;

   /* -> SCHUR-GMRES */
   nalu_hypre_ParILUDataSchurGMRESKDim(ilu_data) = 5;
   nalu_hypre_ParILUDataSchurGMRESMaxIter(ilu_data) = 5;
   nalu_hypre_ParILUDataSchurGMRESTol(ilu_data) = 0.0;
   nalu_hypre_ParILUDataSchurGMRESAbsoluteTol(ilu_data) = 0.0;
   nalu_hypre_ParILUDataSchurGMRESRelChange(ilu_data) = 0;

   /* schur precond data */
   nalu_hypre_ParILUDataSchurPrecondIluType(ilu_data) = 0;
   nalu_hypre_ParILUDataSchurPrecondIluLfil(ilu_data) = 0;
   nalu_hypre_ParILUDataSchurPrecondIluMaxRowNnz(ilu_data) = 100;
   nalu_hypre_ParILUDataSchurPrecondIluDroptol(ilu_data) =
      NULL;/* this is not the default option, set it only when switched to */
   nalu_hypre_ParILUDataSchurPrecondPrintLevel(ilu_data) = 0;
   nalu_hypre_ParILUDataSchurPrecondMaxIter(ilu_data) = 1;
   nalu_hypre_ParILUDataSchurPrecondTriSolve(ilu_data) = 1;
   nalu_hypre_ParILUDataSchurPrecondLowerJacobiIters(ilu_data) = 5;
   nalu_hypre_ParILUDataSchurPrecondUpperJacobiIters(ilu_data) = 5;
   nalu_hypre_ParILUDataSchurPrecondTol(ilu_data) = 0.0;

   /* -> SCHUR-NSH */
   nalu_hypre_ParILUDataSchurNSHSolveMaxIter(ilu_data) = 5;
   nalu_hypre_ParILUDataSchurNSHSolveTol(ilu_data) = 0.0;
   nalu_hypre_ParILUDataSchurNSHDroptol(ilu_data) =
      NULL;/* this is not the default option, set it only when switched to */

   nalu_hypre_ParILUDataSchurNSHMaxNumIter(ilu_data) = 2;
   nalu_hypre_ParILUDataSchurNSHMaxRowNnz(ilu_data) = 1000;
   nalu_hypre_ParILUDataSchurNSHTol(ilu_data) = 1e-09;

   nalu_hypre_ParILUDataSchurMRMaxIter(ilu_data) = 2;
   nalu_hypre_ParILUDataSchurMRColVersion(ilu_data) = 0;
   nalu_hypre_ParILUDataSchurMRMaxRowNnz(ilu_data) = 200;
   nalu_hypre_ParILUDataSchurMRTol(ilu_data) = 1e-09;

   return (void *)                        ilu_data;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/
/* Destroy */
NALU_HYPRE_Int
nalu_hypre_ILUDestroy( void *data )
{
   nalu_hypre_ParILUData * ilu_data = (nalu_hypre_ParILUData*) data;

#if defined(NALU_HYPRE_USING_CUDA) && defined(NALU_HYPRE_USING_CUSPARSE)
   if (nalu_hypre_ParILUDataILUSolveBuffer(ilu_data))
   {
      nalu_hypre_TFree(nalu_hypre_ParILUDataILUSolveBuffer(ilu_data), NALU_HYPRE_MEMORY_DEVICE);
      nalu_hypre_ParILUDataILUSolveBuffer(ilu_data) = NULL;
   }
   if (nalu_hypre_ParILUDataMatLMatrixDescription(ilu_data))
   {
      NALU_HYPRE_CUSPARSE_CALL( (cusparseDestroyMatDescr(nalu_hypre_ParILUDataMatLMatrixDescription(ilu_data))) );
      nalu_hypre_ParILUDataMatLMatrixDescription(ilu_data) = NULL;
   }
   if (nalu_hypre_ParILUDataMatUMatrixDescription(ilu_data))
   {
      NALU_HYPRE_CUSPARSE_CALL( (cusparseDestroyMatDescr(nalu_hypre_ParILUDataMatUMatrixDescription(ilu_data))) );
      nalu_hypre_ParILUDataMatUMatrixDescription(ilu_data) = NULL;
   }
   if (nalu_hypre_ParILUDataMatALILUSolveInfo(ilu_data))
   {
      NALU_HYPRE_CUSPARSE_CALL( (cusparseDestroyCsrsv2Info(nalu_hypre_ParILUDataMatALILUSolveInfo(ilu_data))) );
      nalu_hypre_ParILUDataMatALILUSolveInfo(ilu_data) = NULL;
   }
   if (nalu_hypre_ParILUDataMatAUILUSolveInfo(ilu_data))
   {
      NALU_HYPRE_CUSPARSE_CALL( (cusparseDestroyCsrsv2Info(nalu_hypre_ParILUDataMatAUILUSolveInfo(ilu_data))) );
      nalu_hypre_ParILUDataMatAUILUSolveInfo(ilu_data) = NULL;
   }
   if (nalu_hypre_ParILUDataMatBLILUSolveInfo(ilu_data))
   {
      NALU_HYPRE_CUSPARSE_CALL( (cusparseDestroyCsrsv2Info(nalu_hypre_ParILUDataMatBLILUSolveInfo(ilu_data))) );
      nalu_hypre_ParILUDataMatBLILUSolveInfo(ilu_data) = NULL;
   }
   if (nalu_hypre_ParILUDataMatBUILUSolveInfo(ilu_data))
   {
      NALU_HYPRE_CUSPARSE_CALL( (cusparseDestroyCsrsv2Info(nalu_hypre_ParILUDataMatBUILUSolveInfo(ilu_data))) );
      nalu_hypre_ParILUDataMatBUILUSolveInfo(ilu_data) = NULL;
   }
   if (nalu_hypre_ParILUDataMatSLILUSolveInfo(ilu_data))
   {
      NALU_HYPRE_CUSPARSE_CALL( (cusparseDestroyCsrsv2Info(nalu_hypre_ParILUDataMatSLILUSolveInfo(ilu_data))) );
      nalu_hypre_ParILUDataMatSLILUSolveInfo(ilu_data) = NULL;
   }
   if (nalu_hypre_ParILUDataMatSUILUSolveInfo(ilu_data))
   {
      NALU_HYPRE_CUSPARSE_CALL( (cusparseDestroyCsrsv2Info(nalu_hypre_ParILUDataMatSUILUSolveInfo(ilu_data))) );
      nalu_hypre_ParILUDataMatSUILUSolveInfo(ilu_data) = NULL;
   }
   if (nalu_hypre_ParILUDataMatAILUDevice(ilu_data))
   {
      nalu_hypre_CSRMatrixDestroy( nalu_hypre_ParILUDataMatAILUDevice(ilu_data) );
      nalu_hypre_ParILUDataMatAILUDevice(ilu_data) = NULL;
   }
   if (nalu_hypre_ParILUDataMatBILUDevice(ilu_data))
   {
      nalu_hypre_CSRMatrixDestroy( nalu_hypre_ParILUDataMatBILUDevice(ilu_data) );
      nalu_hypre_ParILUDataMatBILUDevice(ilu_data) = NULL;
   }
   if (nalu_hypre_ParILUDataMatSILUDevice(ilu_data))
   {
      nalu_hypre_CSRMatrixDestroy( nalu_hypre_ParILUDataMatSILUDevice(ilu_data) );
      nalu_hypre_ParILUDataMatSILUDevice(ilu_data) = NULL;
   }
   if (nalu_hypre_ParILUDataMatEDevice(ilu_data))
   {
      nalu_hypre_CSRMatrixDestroy( nalu_hypre_ParILUDataMatEDevice(ilu_data) );
      nalu_hypre_ParILUDataMatEDevice(ilu_data) = NULL;
   }
   if (nalu_hypre_ParILUDataMatFDevice(ilu_data))
   {
      nalu_hypre_CSRMatrixDestroy( nalu_hypre_ParILUDataMatFDevice(ilu_data) );
      nalu_hypre_ParILUDataMatFDevice(ilu_data) = NULL;
   }
   if (nalu_hypre_ParILUDataAperm(ilu_data))
   {
      nalu_hypre_ParCSRMatrixDestroy( nalu_hypre_ParILUDataAperm(ilu_data) );
      nalu_hypre_ParILUDataAperm(ilu_data) = NULL;
   }
   if (nalu_hypre_ParILUDataR(ilu_data))
   {
      nalu_hypre_ParCSRMatrixDestroy( nalu_hypre_ParILUDataR(ilu_data) );
      nalu_hypre_ParILUDataR(ilu_data) = NULL;
   }
   if (nalu_hypre_ParILUDataP(ilu_data))
   {
      nalu_hypre_ParCSRMatrixDestroy( nalu_hypre_ParILUDataP(ilu_data) );
      nalu_hypre_ParILUDataP(ilu_data) = NULL;
   }
   if (nalu_hypre_ParILUDataFTempUpper(ilu_data))
   {
      nalu_hypre_SeqVectorDestroy( nalu_hypre_ParILUDataFTempUpper(ilu_data) );
      nalu_hypre_ParILUDataFTempUpper(ilu_data) = NULL;
   }
   if (nalu_hypre_ParILUDataUTempLower(ilu_data))
   {
      nalu_hypre_SeqVectorDestroy( nalu_hypre_ParILUDataUTempLower(ilu_data) );
      nalu_hypre_ParILUDataUTempLower(ilu_data) = NULL;
   }
   if (nalu_hypre_ParILUDataMatAFakeDiagonal(ilu_data))
   {
      nalu_hypre_TFree( nalu_hypre_ParILUDataMatAFakeDiagonal(ilu_data), NALU_HYPRE_MEMORY_DEVICE);
      nalu_hypre_ParILUDataMatAFakeDiagonal(ilu_data) = NULL;
   }
   if (nalu_hypre_ParILUDataADiagDiag(ilu_data))
   {
      nalu_hypre_SeqVectorDestroy(nalu_hypre_ParILUDataADiagDiag(ilu_data));
      nalu_hypre_ParILUDataADiagDiag(ilu_data) = NULL;
   }
#endif

   /* final residual vector */
   if (nalu_hypre_ParILUDataResidual(ilu_data))
   {
      nalu_hypre_ParVectorDestroy( nalu_hypre_ParILUDataResidual(ilu_data) );
      nalu_hypre_ParILUDataResidual(ilu_data) = NULL;
   }
   if (nalu_hypre_ParILUDataRelResNorms(ilu_data))
   {
      nalu_hypre_TFree( nalu_hypre_ParILUDataRelResNorms(ilu_data), NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_ParILUDataRelResNorms(ilu_data) = NULL;
   }
   /* temp vectors for solve phase */
   if (nalu_hypre_ParILUDataUTemp(ilu_data))
   {
      nalu_hypre_ParVectorDestroy( nalu_hypre_ParILUDataUTemp(ilu_data) );
      nalu_hypre_ParILUDataUTemp(ilu_data) = NULL;
   }
   if (nalu_hypre_ParILUDataFTemp(ilu_data))
   {
      nalu_hypre_ParVectorDestroy( nalu_hypre_ParILUDataFTemp(ilu_data) );
      nalu_hypre_ParILUDataFTemp(ilu_data) = NULL;
   }
   if (nalu_hypre_ParILUDataXTemp(ilu_data))
   {
      nalu_hypre_ParVectorDestroy( nalu_hypre_ParILUDataXTemp(ilu_data) );
      nalu_hypre_ParILUDataXTemp(ilu_data) = NULL;
   }
   if (nalu_hypre_ParILUDataYTemp(ilu_data))
   {
      nalu_hypre_ParVectorDestroy( nalu_hypre_ParILUDataYTemp(ilu_data) );
      nalu_hypre_ParILUDataYTemp(ilu_data) = NULL;
   }
   if (nalu_hypre_ParILUDataZTemp(ilu_data))
   {
      nalu_hypre_SeqVectorDestroy(nalu_hypre_ParILUDataZTemp(ilu_data));
      nalu_hypre_ParILUDataZTemp(ilu_data) = NULL;
   }
   if (nalu_hypre_ParILUDataUExt(ilu_data))
   {
      nalu_hypre_TFree(nalu_hypre_ParILUDataUExt(ilu_data), NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_ParILUDataUExt(ilu_data) = NULL;
   }
   if (nalu_hypre_ParILUDataFExt(ilu_data))
   {
      nalu_hypre_TFree(nalu_hypre_ParILUDataFExt(ilu_data), NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_ParILUDataFExt(ilu_data) = NULL;
   }
   if (nalu_hypre_ParILUDataRhs(ilu_data))
   {
      nalu_hypre_ParVectorDestroy( nalu_hypre_ParILUDataRhs(ilu_data) );
      nalu_hypre_ParILUDataRhs(ilu_data) = NULL;
   }
   if (nalu_hypre_ParILUDataX(ilu_data))
   {
      nalu_hypre_ParVectorDestroy( nalu_hypre_ParILUDataX(ilu_data) );
      nalu_hypre_ParILUDataX(ilu_data) = NULL;
   }
   /* l1_norms */
   if (nalu_hypre_ParILUDataL1Norms(ilu_data))
   {
      nalu_hypre_TFree(nalu_hypre_ParILUDataL1Norms(ilu_data), NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_ParILUDataL1Norms(ilu_data) = NULL;
   }

   /* u_end */
   if (nalu_hypre_ParILUDataUEnd(ilu_data))
   {
      nalu_hypre_TFree(nalu_hypre_ParILUDataUEnd(ilu_data), NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_ParILUDataUEnd(ilu_data) = NULL;
   }

   /* Factors */
   if (nalu_hypre_ParILUDataMatL(ilu_data))
   {
      nalu_hypre_ParCSRMatrixDestroy(nalu_hypre_ParILUDataMatL(ilu_data));
      nalu_hypre_ParILUDataMatL(ilu_data) = NULL;
   }
   if (nalu_hypre_ParILUDataMatU(ilu_data))
   {
      nalu_hypre_ParCSRMatrixDestroy(nalu_hypre_ParILUDataMatU(ilu_data));
      nalu_hypre_ParILUDataMatU(ilu_data) = NULL;
   }
   if (nalu_hypre_ParILUDataMatD(ilu_data))
   {
      nalu_hypre_TFree(nalu_hypre_ParILUDataMatD(ilu_data), NALU_HYPRE_MEMORY_DEVICE);
      nalu_hypre_ParILUDataMatD(ilu_data) = NULL;
   }
   if (nalu_hypre_ParILUDataMatLModified(ilu_data))
   {
      nalu_hypre_ParCSRMatrixDestroy(nalu_hypre_ParILUDataMatLModified(ilu_data));
      nalu_hypre_ParILUDataMatLModified(ilu_data) = NULL;
   }
   if (nalu_hypre_ParILUDataMatUModified(ilu_data))
   {
      nalu_hypre_ParCSRMatrixDestroy(nalu_hypre_ParILUDataMatUModified(ilu_data));
      nalu_hypre_ParILUDataMatUModified(ilu_data) = NULL;
   }
   if (nalu_hypre_ParILUDataMatDModified(ilu_data))
   {
      nalu_hypre_TFree(nalu_hypre_ParILUDataMatDModified(ilu_data), NALU_HYPRE_MEMORY_DEVICE);
      nalu_hypre_ParILUDataMatDModified(ilu_data) = NULL;
   }
   if (nalu_hypre_ParILUDataMatS(ilu_data))
   {
      nalu_hypre_ParCSRMatrixDestroy(nalu_hypre_ParILUDataMatS(ilu_data));
      nalu_hypre_ParILUDataMatS(ilu_data) = NULL;
   }
   if (nalu_hypre_ParILUDataSchurSolver(ilu_data))
   {
      switch (nalu_hypre_ParILUDataIluType(ilu_data))
      {
         case 10: case 11: case 40: case 41: case 50:
            NALU_HYPRE_ParCSRGMRESDestroy(nalu_hypre_ParILUDataSchurSolver(ilu_data)); //GMRES for Schur
            break;
         case 20: case 21:
            nalu_hypre_NSHDestroy(nalu_hypre_ParILUDataSchurSolver(ilu_data));//NSH for Schur
            break;
         default:
            break;
      }
      nalu_hypre_ParILUDataSchurSolver(ilu_data) = NULL;
   }
   if (nalu_hypre_ParILUDataSchurPrecond(ilu_data))
   {
      switch (nalu_hypre_ParILUDataIluType(ilu_data))
      {
         case 10: case 11: case 40: case 41:
#if defined(NALU_HYPRE_USING_CUDA) && defined(NALU_HYPRE_USING_CUSPARSE)
            if (nalu_hypre_ParILUDataIluType(ilu_data) != 10 && nalu_hypre_ParILUDataIluType(ilu_data) != 11)
            {
#endif
               NALU_HYPRE_ILUDestroy(nalu_hypre_ParILUDataSchurPrecond(ilu_data)); //ILU as precond for Schur
#if defined(NALU_HYPRE_USING_CUDA) && defined(NALU_HYPRE_USING_CUSPARSE)
            }
#endif
            break;
         default:
            break;
      }
      nalu_hypre_ParILUDataSchurPrecond(ilu_data) = NULL;
   }
   /* CF marker array */
   if (nalu_hypre_ParILUDataCFMarkerArray(ilu_data))
   {
      nalu_hypre_TFree(nalu_hypre_ParILUDataCFMarkerArray(ilu_data), NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_ParILUDataCFMarkerArray(ilu_data) = NULL;
   }
   /* permutation array */
   if (nalu_hypre_ParILUDataPerm(ilu_data))
   {
      nalu_hypre_TFree(nalu_hypre_ParILUDataPerm(ilu_data), NALU_HYPRE_MEMORY_DEVICE);
      nalu_hypre_ParILUDataPerm(ilu_data) = NULL;
   }
   if (nalu_hypre_ParILUDataQPerm(ilu_data))
   {
      nalu_hypre_TFree(nalu_hypre_ParILUDataQPerm(ilu_data), NALU_HYPRE_MEMORY_DEVICE);
      nalu_hypre_ParILUDataQPerm(ilu_data) = NULL;
   }
   /* droptol array */
   if (nalu_hypre_ParILUDataDroptol(ilu_data))
   {
      nalu_hypre_TFree(nalu_hypre_ParILUDataDroptol(ilu_data), NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_ParILUDataDroptol(ilu_data) = NULL;
   }
   if (nalu_hypre_ParILUDataSchurPrecondIluDroptol(ilu_data))
   {
      nalu_hypre_TFree(nalu_hypre_ParILUDataSchurPrecondIluDroptol(ilu_data), NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_ParILUDataSchurPrecondIluDroptol(ilu_data) = NULL;
   }
   if (nalu_hypre_ParILUDataSchurNSHDroptol(ilu_data))
   {
      nalu_hypre_TFree(nalu_hypre_ParILUDataSchurNSHDroptol(ilu_data), NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_ParILUDataSchurNSHDroptol(ilu_data) = NULL;
   }
   /* ilu data */
   nalu_hypre_TFree(ilu_data, NALU_HYPRE_MEMORY_HOST);

   return nalu_hypre_error_flag;
}
/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/
/* set fill level (for ilu(k)) */
NALU_HYPRE_Int
nalu_hypre_ILUSetLevelOfFill( void *ilu_vdata, NALU_HYPRE_Int lfil )
{
   nalu_hypre_ParILUData *ilu_data = (nalu_hypre_ParILUData*) ilu_vdata;
   nalu_hypre_ParILUDataLfil(ilu_data) = lfil;
   return nalu_hypre_error_flag;
}
/* set max non-zeros per row in factors (for ilut) */
NALU_HYPRE_Int
nalu_hypre_ILUSetMaxNnzPerRow( void *ilu_vdata, NALU_HYPRE_Int nzmax )
{
   nalu_hypre_ParILUData *ilu_data = (nalu_hypre_ParILUData*) ilu_vdata;
   nalu_hypre_ParILUDataMaxRowNnz(ilu_data) = nzmax;
   return nalu_hypre_error_flag;
}
/* set threshold for dropping in LU factors (for ilut) */
NALU_HYPRE_Int
nalu_hypre_ILUSetDropThreshold( void *ilu_vdata, NALU_HYPRE_Real threshold )
{
   nalu_hypre_ParILUData   *ilu_data = (nalu_hypre_ParILUData*) ilu_vdata;
   if (!(nalu_hypre_ParILUDataDroptol(ilu_data)))
   {
      nalu_hypre_ParILUDataDroptol(ilu_data) = nalu_hypre_TAlloc(NALU_HYPRE_Real, 3, NALU_HYPRE_MEMORY_HOST);
   }
   nalu_hypre_ParILUDataDroptol(ilu_data)[0] = threshold;
   nalu_hypre_ParILUDataDroptol(ilu_data)[1] = threshold;
   nalu_hypre_ParILUDataDroptol(ilu_data)[2] = threshold;
   return nalu_hypre_error_flag;
}
/* set array of threshold for dropping in LU factors (for ilut) */
NALU_HYPRE_Int
nalu_hypre_ILUSetDropThresholdArray( void *ilu_vdata, NALU_HYPRE_Real *threshold )
{
   nalu_hypre_ParILUData   *ilu_data = (nalu_hypre_ParILUData*) ilu_vdata;
   if (!(nalu_hypre_ParILUDataDroptol(ilu_data)))
   {
      nalu_hypre_ParILUDataDroptol(ilu_data) = nalu_hypre_TAlloc(NALU_HYPRE_Real, 3, NALU_HYPRE_MEMORY_HOST);
   }
   nalu_hypre_TMemcpy( nalu_hypre_ParILUDataDroptol(ilu_data), threshold, NALU_HYPRE_Real, 3, NALU_HYPRE_MEMORY_HOST,
                  NALU_HYPRE_MEMORY_HOST);
   return nalu_hypre_error_flag;
}
/* set ILU factorization type */
NALU_HYPRE_Int
nalu_hypre_ILUSetType( void *ilu_vdata, NALU_HYPRE_Int ilu_type )
{
   nalu_hypre_ParILUData *ilu_data = (nalu_hypre_ParILUData*) ilu_vdata;

   /* destroy schur solver and/or preconditioner if already have one */
   if (nalu_hypre_ParILUDataSchurSolver(ilu_data))
   {
      switch (nalu_hypre_ParILUDataIluType(ilu_data))
      {
         case 10: case 11: case 40: case 41: case 50:
            NALU_HYPRE_ParCSRGMRESDestroy(nalu_hypre_ParILUDataSchurSolver(ilu_data)); //GMRES for Schur
            break;
         case 20: case 21:
            nalu_hypre_NSHDestroy(nalu_hypre_ParILUDataSchurSolver(ilu_data));//NSH for Schur
            break;
         default:
            break;
      }
      nalu_hypre_ParILUDataSchurSolver(ilu_data) = NULL;
   }
   if (nalu_hypre_ParILUDataSchurPrecond(ilu_data))
   {
      switch (nalu_hypre_ParILUDataIluType(ilu_data))
      {
         case 10: case 11: case 40: case 41:
#if defined(NALU_HYPRE_USING_CUDA) && defined(NALU_HYPRE_USING_CUSPARSE)
            if (nalu_hypre_ParILUDataIluType(ilu_data) != 10 && nalu_hypre_ParILUDataIluType(ilu_data) != 11)
            {
#endif
               NALU_HYPRE_ILUDestroy(nalu_hypre_ParILUDataSchurPrecond(ilu_data)); //ILU as precond for Schur
#if defined(NALU_HYPRE_USING_CUDA) && defined(NALU_HYPRE_USING_CUSPARSE)
            }
#endif
            break;
         default:
            break;
      }
      nalu_hypre_ParILUDataSchurPrecond(ilu_data) = NULL;
   }

   nalu_hypre_ParILUDataIluType(ilu_data) = ilu_type;
   /* reset default value, not a large cost
    * assume we won't change back from
    */
   switch (ilu_type)
   {
      /* NSH type */
      case 20: case 21:
      {
         /* only set value when user has not assiged value before */
         if (!(nalu_hypre_ParILUDataSchurNSHDroptol(ilu_data)))
         {
            nalu_hypre_ParILUDataSchurNSHDroptol(ilu_data) = nalu_hypre_TAlloc(NALU_HYPRE_Real, 2, NALU_HYPRE_MEMORY_HOST);
            nalu_hypre_ParILUDataSchurNSHDroptol(ilu_data)[0] = 1e-02;
            nalu_hypre_ParILUDataSchurNSHDroptol(ilu_data)[1] = 1e-02;
         }
         break;
      }
      case 10: case 11: case 40: case 41: case 50:
      {
         /* Set value of droptol for solving Schur system (if not set by user) */
         /* NOTE: This is currently not exposed to users */
         if (!(nalu_hypre_ParILUDataSchurPrecondIluDroptol(ilu_data)))
         {
            nalu_hypre_ParILUDataSchurPrecondIluDroptol(ilu_data) = nalu_hypre_TAlloc(NALU_HYPRE_Real, 3, NALU_HYPRE_MEMORY_HOST);
            nalu_hypre_ParILUDataSchurPrecondIluDroptol(ilu_data)[0] = 1e-02;
            nalu_hypre_ParILUDataSchurPrecondIluDroptol(ilu_data)[1] = 1e-02;
            nalu_hypre_ParILUDataSchurPrecondIluDroptol(ilu_data)[2] = 1e-02;
         }
         break;
      }
      default:
         break;
   }

   return nalu_hypre_error_flag;
}
/* Set max number of iterations for ILU solver */
NALU_HYPRE_Int
nalu_hypre_ILUSetMaxIter( void *ilu_vdata, NALU_HYPRE_Int max_iter )
{
   nalu_hypre_ParILUData   *ilu_data = (nalu_hypre_ParILUData*) ilu_vdata;
   nalu_hypre_ParILUDataMaxIter(ilu_data) = max_iter;
   return nalu_hypre_error_flag;
}
/* Set ILU triangular solver type */
NALU_HYPRE_Int
nalu_hypre_ILUSetTriSolve( void *ilu_vdata, NALU_HYPRE_Int tri_solve )
{
   nalu_hypre_ParILUData   *ilu_data = (nalu_hypre_ParILUData*) ilu_vdata;
   nalu_hypre_ParILUDataTriSolve(ilu_data) = tri_solve;
   return nalu_hypre_error_flag;
}
/* Set Lower Jacobi iterations for iterative triangular solver */
NALU_HYPRE_Int
nalu_hypre_ILUSetLowerJacobiIters( void *ilu_vdata, NALU_HYPRE_Int lower_jacobi_iters )
{
   nalu_hypre_ParILUData   *ilu_data = (nalu_hypre_ParILUData*) ilu_vdata;
   nalu_hypre_ParILUDataLowerJacobiIters(ilu_data) = lower_jacobi_iters;
   return nalu_hypre_error_flag;
}
/* Set Upper Jacobi iterations for iterative triangular solver */
NALU_HYPRE_Int
nalu_hypre_ILUSetUpperJacobiIters( void *ilu_vdata, NALU_HYPRE_Int upper_jacobi_iters )
{
   nalu_hypre_ParILUData   *ilu_data = (nalu_hypre_ParILUData*) ilu_vdata;
   nalu_hypre_ParILUDataUpperJacobiIters(ilu_data) = upper_jacobi_iters;
   return nalu_hypre_error_flag;
}
/* Set convergence tolerance for ILU solver */
NALU_HYPRE_Int
nalu_hypre_ILUSetTol( void *ilu_vdata, NALU_HYPRE_Real tol )
{
   nalu_hypre_ParILUData   *ilu_data = (nalu_hypre_ParILUData*) ilu_vdata;
   nalu_hypre_ParILUDataTol(ilu_data) = tol;
   return nalu_hypre_error_flag;
}
/* Set print level for ilu solver */
NALU_HYPRE_Int
nalu_hypre_ILUSetPrintLevel( void *ilu_vdata, NALU_HYPRE_Int print_level )
{
   nalu_hypre_ParILUData   *ilu_data = (nalu_hypre_ParILUData*) ilu_vdata;
   nalu_hypre_ParILUDataPrintLevel(ilu_data) = print_level;
   return nalu_hypre_error_flag;
}
/* Set print level for ilu solver */
NALU_HYPRE_Int
nalu_hypre_ILUSetLogging( void *ilu_vdata, NALU_HYPRE_Int logging )
{
   nalu_hypre_ParILUData   *ilu_data = (nalu_hypre_ParILUData*) ilu_vdata;
   nalu_hypre_ParILUDataLogging(ilu_data) = logging;
   return nalu_hypre_error_flag;
}
/* Set type of reordering for local matrix */
NALU_HYPRE_Int
nalu_hypre_ILUSetLocalReordering( void *ilu_vdata, NALU_HYPRE_Int ordering_type )
{
   nalu_hypre_ParILUData   *ilu_data = (nalu_hypre_ParILUData*) ilu_vdata;
   nalu_hypre_ParILUDataReorderingType(ilu_data) = ordering_type;
   return nalu_hypre_error_flag;
}

/* Set KDim (for GMRES) for Solver of Schur System */
NALU_HYPRE_Int
nalu_hypre_ILUSetSchurSolverKDIM( void *ilu_vdata, NALU_HYPRE_Int ss_kDim )
{
   nalu_hypre_ParILUData   *ilu_data = (nalu_hypre_ParILUData*) ilu_vdata;
   nalu_hypre_ParILUDataSchurGMRESKDim(ilu_data) = ss_kDim;
   return nalu_hypre_error_flag;
}
/* Set max iteration for Solver of Schur System */
NALU_HYPRE_Int
nalu_hypre_ILUSetSchurSolverMaxIter( void *ilu_vdata, NALU_HYPRE_Int ss_max_iter )
{
   nalu_hypre_ParILUData   *ilu_data = (nalu_hypre_ParILUData*) ilu_vdata;
   /* for the GMRES solve, the max iter is same as kdim by default */
   nalu_hypre_ParILUDataSchurGMRESKDim(ilu_data) = ss_max_iter;
   nalu_hypre_ParILUDataSchurGMRESMaxIter(ilu_data) = ss_max_iter;

   /* also set this value for NSH solve */
   nalu_hypre_ParILUDataSchurNSHSolveMaxIter(ilu_data) = ss_max_iter;

   return nalu_hypre_error_flag;
}
/* Set convergence tolerance for Solver of Schur System */
NALU_HYPRE_Int
nalu_hypre_ILUSetSchurSolverTol( void *ilu_vdata, NALU_HYPRE_Real ss_tol )
{
   nalu_hypre_ParILUData   *ilu_data = (nalu_hypre_ParILUData*) ilu_vdata;
   nalu_hypre_ParILUDataSchurGMRESTol(ilu_data) = ss_tol;
   return nalu_hypre_error_flag;
}
/* Set absolute tolerance for Solver of Schur System */
NALU_HYPRE_Int
nalu_hypre_ILUSetSchurSolverAbsoluteTol( void *ilu_vdata, NALU_HYPRE_Real ss_absolute_tol )
{
   nalu_hypre_ParILUData   *ilu_data = (nalu_hypre_ParILUData*) ilu_vdata;
   nalu_hypre_ParILUDataSchurGMRESAbsoluteTol(ilu_data) = ss_absolute_tol;
   return nalu_hypre_error_flag;
}
/* Set logging for Solver of Schur System */
NALU_HYPRE_Int
nalu_hypre_ILUSetSchurSolverLogging( void *ilu_vdata, NALU_HYPRE_Int ss_logging )
{
   nalu_hypre_ParILUData   *ilu_data = (nalu_hypre_ParILUData*) ilu_vdata;
   nalu_hypre_ParILUDataSchurSolverLogging(ilu_data) = ss_logging;
   return nalu_hypre_error_flag;
}
/* Set print level for Solver of Schur System */
NALU_HYPRE_Int
nalu_hypre_ILUSetSchurSolverPrintLevel( void *ilu_vdata, NALU_HYPRE_Int ss_print_level )
{
   nalu_hypre_ParILUData   *ilu_data = (nalu_hypre_ParILUData*) ilu_vdata;
   nalu_hypre_ParILUDataSchurSolverPrintLevel(ilu_data) = ss_print_level;
   return nalu_hypre_error_flag;
}
/* Set rel change (for GMRES) for Solver of Schur System */
NALU_HYPRE_Int
nalu_hypre_ILUSetSchurSolverRelChange( void *ilu_vdata, NALU_HYPRE_Int ss_rel_change )
{
   nalu_hypre_ParILUData   *ilu_data = (nalu_hypre_ParILUData*) ilu_vdata;
   nalu_hypre_ParILUDataSchurGMRESRelChange(ilu_data) = ss_rel_change;
   return nalu_hypre_error_flag;
}
/* Set ILU type for Precond of Schur System */
NALU_HYPRE_Int
nalu_hypre_ILUSetSchurPrecondILUType( void *ilu_vdata, NALU_HYPRE_Int sp_ilu_type )
{
   nalu_hypre_ParILUData   *ilu_data = (nalu_hypre_ParILUData*) ilu_vdata;
   nalu_hypre_ParILUDataSchurPrecondIluType(ilu_data) = sp_ilu_type;
   return nalu_hypre_error_flag;
}
/* Set ILU level of fill for Precond of Schur System */
NALU_HYPRE_Int
nalu_hypre_ILUSetSchurPrecondILULevelOfFill( void *ilu_vdata, NALU_HYPRE_Int sp_ilu_lfil )
{
   nalu_hypre_ParILUData   *ilu_data = (nalu_hypre_ParILUData*) ilu_vdata;
   nalu_hypre_ParILUDataSchurPrecondIluLfil(ilu_data) = sp_ilu_lfil;
   return nalu_hypre_error_flag;
}
/* Set ILU max nonzeros per row for Precond of Schur System */
NALU_HYPRE_Int
nalu_hypre_ILUSetSchurPrecondILUMaxNnzPerRow( void *ilu_vdata, NALU_HYPRE_Int sp_ilu_max_row_nnz )
{
   nalu_hypre_ParILUData   *ilu_data = (nalu_hypre_ParILUData*) ilu_vdata;
   nalu_hypre_ParILUDataSchurPrecondIluMaxRowNnz(ilu_data) = sp_ilu_max_row_nnz;
   return nalu_hypre_error_flag;
}
/* Set ILU drop threshold for ILUT for Precond of Schur System
 * We don't want to influence the original ILU, so create new array if not own data
 */
NALU_HYPRE_Int
nalu_hypre_ILUSetSchurPrecondILUDropThreshold( void *ilu_vdata, NALU_HYPRE_Real sp_ilu_droptol )
{
   nalu_hypre_ParILUData   *ilu_data = (nalu_hypre_ParILUData*) ilu_vdata;
   if (!(nalu_hypre_ParILUDataSchurPrecondIluDroptol(ilu_data)))
   {
      nalu_hypre_ParILUDataSchurPrecondIluDroptol(ilu_data) = nalu_hypre_TAlloc(NALU_HYPRE_Real, 3, NALU_HYPRE_MEMORY_HOST);
   }
   nalu_hypre_ParILUDataSchurPrecondIluDroptol(ilu_data)[0]   = sp_ilu_droptol;
   nalu_hypre_ParILUDataSchurPrecondIluDroptol(ilu_data)[1]   = sp_ilu_droptol;
   nalu_hypre_ParILUDataSchurPrecondIluDroptol(ilu_data)[2]   = sp_ilu_droptol;
   return nalu_hypre_error_flag;
}
/* Set array of ILU drop threshold for ILUT for Precond of Schur System */
NALU_HYPRE_Int
nalu_hypre_ILUSetSchurPrecondILUDropThresholdArray( void *ilu_vdata, NALU_HYPRE_Real *sp_ilu_droptol )
{
   nalu_hypre_ParILUData   *ilu_data = (nalu_hypre_ParILUData*) ilu_vdata;
   if (!(nalu_hypre_ParILUDataSchurPrecondIluDroptol(ilu_data)))
   {
      nalu_hypre_ParILUDataSchurPrecondIluDroptol(ilu_data) = nalu_hypre_TAlloc(NALU_HYPRE_Real, 3, NALU_HYPRE_MEMORY_HOST);
   }
   nalu_hypre_TMemcpy( nalu_hypre_ParILUDataSchurPrecondIluDroptol(ilu_data), sp_ilu_droptol, NALU_HYPRE_Real, 3,
                  NALU_HYPRE_MEMORY_HOST, NALU_HYPRE_MEMORY_HOST);
   return nalu_hypre_error_flag;
}
/* Set print level for Precond of Schur System */
NALU_HYPRE_Int
nalu_hypre_ILUSetSchurPrecondPrintLevel( void *ilu_vdata, NALU_HYPRE_Int sp_print_level )
{
   nalu_hypre_ParILUData   *ilu_data = (nalu_hypre_ParILUData*) ilu_vdata;
   nalu_hypre_ParILUDataSchurPrecondPrintLevel(ilu_data) = sp_print_level;
   return nalu_hypre_error_flag;
}
/* Set max number of iterations for Precond of Schur System */
NALU_HYPRE_Int
nalu_hypre_ILUSetSchurPrecondMaxIter( void *ilu_vdata, NALU_HYPRE_Int sp_max_iter )
{
   nalu_hypre_ParILUData   *ilu_data = (nalu_hypre_ParILUData*) ilu_vdata;
   nalu_hypre_ParILUDataSchurPrecondMaxIter(ilu_data) = sp_max_iter;
   return nalu_hypre_error_flag;
}
/* Set triangular solver type for Precond of Schur System */
NALU_HYPRE_Int
nalu_hypre_ILUSetSchurPrecondTriSolve( void *ilu_vdata, NALU_HYPRE_Int sp_tri_solve )
{
   nalu_hypre_ParILUData   *ilu_data = (nalu_hypre_ParILUData*) ilu_vdata;
   nalu_hypre_ParILUDataSchurPrecondTriSolve(ilu_data) = sp_tri_solve;
   return nalu_hypre_error_flag;
}
/* Set Lower Jacobi iterations for Precond of Schur System */
NALU_HYPRE_Int
nalu_hypre_ILUSetSchurPrecondLowerJacobiIters( void *ilu_vdata, NALU_HYPRE_Int sp_lower_jacobi_iters )
{
   nalu_hypre_ParILUData   *ilu_data = (nalu_hypre_ParILUData*) ilu_vdata;
   nalu_hypre_ParILUDataSchurPrecondLowerJacobiIters(ilu_data) = sp_lower_jacobi_iters;
   return nalu_hypre_error_flag;
}
/* Set Upper Jacobi iterations for Precond of Schur System */
NALU_HYPRE_Int
nalu_hypre_ILUSetSchurPrecondUpperJacobiIters( void *ilu_vdata, NALU_HYPRE_Int sp_upper_jacobi_iters )
{
   nalu_hypre_ParILUData   *ilu_data = (nalu_hypre_ParILUData*) ilu_vdata;
   nalu_hypre_ParILUDataSchurPrecondUpperJacobiIters(ilu_data) = sp_upper_jacobi_iters;
   return nalu_hypre_error_flag;
}
/* Set onvergence tolerance for Precond of Schur System */
NALU_HYPRE_Int
nalu_hypre_ILUSetSchurPrecondTol( void *ilu_vdata, NALU_HYPRE_Int sp_tol )
{
   nalu_hypre_ParILUData   *ilu_data = (nalu_hypre_ParILUData*) ilu_vdata;
   nalu_hypre_ParILUDataSchurPrecondTol(ilu_data) = sp_tol;
   return nalu_hypre_error_flag;
}
/* Set tolorance for dropping in NSH for Schur System
 * We don't want to influence the original ILU, so create new array if not own data
 */
NALU_HYPRE_Int
nalu_hypre_ILUSetSchurNSHDropThreshold( void *ilu_vdata, NALU_HYPRE_Real threshold)
{
   nalu_hypre_ParILUData   *ilu_data = (nalu_hypre_ParILUData*) ilu_vdata;
   if (!(nalu_hypre_ParILUDataSchurNSHDroptol(ilu_data)))
   {
      nalu_hypre_ParILUDataSchurNSHDroptol(ilu_data) = nalu_hypre_TAlloc(NALU_HYPRE_Real, 2, NALU_HYPRE_MEMORY_HOST);
   }
   nalu_hypre_ParILUDataSchurNSHDroptol(ilu_data)[0]           = threshold;
   nalu_hypre_ParILUDataSchurNSHDroptol(ilu_data)[1]           = threshold;
   return nalu_hypre_error_flag;
}
/* Set tolorance array for NSH for Schur System
 *    - threshold[0] : threshold for Minimal Residual iteration (initial guess for NSH).
 *    - threshold[1] : threshold for Newton–Schulz–Hotelling iteration.
*/
NALU_HYPRE_Int
nalu_hypre_ILUSetSchurNSHDropThresholdArray( void *ilu_vdata, NALU_HYPRE_Real *threshold)
{
   nalu_hypre_ParILUData   *ilu_data = (nalu_hypre_ParILUData*) ilu_vdata;
   if (!(nalu_hypre_ParILUDataSchurNSHDroptol(ilu_data)))
   {
      nalu_hypre_ParILUDataSchurNSHDroptol(ilu_data) = nalu_hypre_TAlloc(NALU_HYPRE_Real, 2, NALU_HYPRE_MEMORY_HOST);
   }
   nalu_hypre_TMemcpy( nalu_hypre_ParILUDataSchurNSHDroptol(ilu_data), threshold, NALU_HYPRE_Real, 2,
                  NALU_HYPRE_MEMORY_HOST, NALU_HYPRE_MEMORY_HOST);
   return nalu_hypre_error_flag;
}

/* Get number of iterations for ILU solver */
NALU_HYPRE_Int
nalu_hypre_ILUGetNumIterations( void *ilu_vdata, NALU_HYPRE_Int *num_iterations )
{
   nalu_hypre_ParILUData  *ilu_data = (nalu_hypre_ParILUData*) ilu_vdata;

   if (!ilu_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }
   *num_iterations = nalu_hypre_ParILUDataNumIterations(ilu_data);

   return nalu_hypre_error_flag;
}
/* Get residual norms for ILU solver */
NALU_HYPRE_Int
nalu_hypre_ILUGetFinalRelativeResidualNorm( void *ilu_vdata, NALU_HYPRE_Real *res_norm )
{
   nalu_hypre_ParILUData  *ilu_data = (nalu_hypre_ParILUData*) ilu_vdata;

   if (!ilu_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }
   *res_norm = nalu_hypre_ParILUDataFinalRelResidualNorm(ilu_data);

   return nalu_hypre_error_flag;
}
/*
 * Quicksort of the elements in a from low to high.
 * The elements in b are permuted according to the sorted a.
 * The elements in iw are permuted reverse according to the sorted a as it's index
 *   ie, iw[a1] and iw[a2] will be switched if a1 and a2 are switched
 * lo and hi are the extents of the region of the array a, that is to be sorted.
*/
/*
NALU_HYPRE_Int
nalu_hypre_quickSortIR (NALU_HYPRE_Int *a, NALU_HYPRE_Real *b, NALU_HYPRE_Int *iw, const NALU_HYPRE_Int lo, const NALU_HYPRE_Int hi)
{
   NALU_HYPRE_Int i=lo, j=hi;
   NALU_HYPRE_Int v;
   NALU_HYPRE_Int mid = (lo+hi)>>1;
   NALU_HYPRE_Int x=ceil(a[mid]);
   NALU_HYPRE_Real q;
   //  partition
   do
   {
      while (a[i]<x) i++;
      while (a[j]>x) j--;
      if (i<=j)
      {
          v=a[i]; a[i]=a[j]; a[j]=v;
          q=b[i]; b[i]=b[j]; b[j]=q;
          v=iw[a[i]];iw[a[i]]=iw[a[j]];iw[a[j]]=v;
          i++; j--;
      }
   } while (i<=j);
   //  recursion
   if (lo<j) nalu_hypre_quickSortIR(a, b, iw, lo, j);
   if (i<hi) nalu_hypre_quickSortIR(a, b, iw, i, hi);

   return nalu_hypre_error_flag;
}
*/
/* Print solver params */
NALU_HYPRE_Int
nalu_hypre_ILUWriteSolverParams(void *ilu_vdata)
{
   nalu_hypre_ParILUData  *ilu_data = (nalu_hypre_ParILUData*) ilu_vdata;
   nalu_hypre_printf("ILU Setup parameters: \n");
   nalu_hypre_printf("ILU factorization type: %d : ", nalu_hypre_ParILUDataIluType(ilu_data));
   switch (nalu_hypre_ParILUDataIluType(ilu_data))
   {
      case 0:
#ifdef NALU_HYPRE_USING_CUDA
         if ( nalu_hypre_ParILUDataLfil(ilu_data) == 0 )
         {
            nalu_hypre_printf("Block Jacobi with GPU-accelerated ILU0 \n");
            nalu_hypre_printf("Operator Complexity (Fill factor) = %f \n",
                         nalu_hypre_ParILUDataOperatorComplexity(ilu_data));
         }
         else
         {
#endif
            nalu_hypre_printf("Block Jacobi with ILU(%d) \n", nalu_hypre_ParILUDataLfil(ilu_data));
            nalu_hypre_printf("Operator Complexity (Fill factor) = %f \n",
                         nalu_hypre_ParILUDataOperatorComplexity(ilu_data));
#ifdef NALU_HYPRE_USING_CUDA
         }
#endif
         break;
      case 1:
         nalu_hypre_printf("Block Jacobi with ILUT \n");
         nalu_hypre_printf("drop tolerance for B = %e, E&F = %e, S = %e \n", nalu_hypre_ParILUDataDroptol(ilu_data)[0],
                      nalu_hypre_ParILUDataDroptol(ilu_data)[1], nalu_hypre_ParILUDataDroptol(ilu_data)[2]);
         nalu_hypre_printf("Max nnz per row = %d \n", nalu_hypre_ParILUDataMaxRowNnz(ilu_data));
         nalu_hypre_printf("Operator Complexity (Fill factor) = %f \n",
                      nalu_hypre_ParILUDataOperatorComplexity(ilu_data));
         break;
      case 10:
#ifdef NALU_HYPRE_USING_CUDA
         if ( nalu_hypre_ParILUDataLfil(ilu_data) == 0 )
         {
            nalu_hypre_printf("ILU-GMRES with GPU-accelerated ILU0 \n");
            nalu_hypre_printf("Operator Complexity (Fill factor) = %f \n",
                         nalu_hypre_ParILUDataOperatorComplexity(ilu_data));
         }
         else
         {
#endif
            nalu_hypre_printf("ILU-GMRES with ILU(%d) \n", nalu_hypre_ParILUDataLfil(ilu_data));
            nalu_hypre_printf("Operator Complexity (Fill factor) = %f \n",
                         nalu_hypre_ParILUDataOperatorComplexity(ilu_data));
#ifdef NALU_HYPRE_USING_CUDA
         }
#endif
         break;
      case 11:
         nalu_hypre_printf("ILU-GMRES with ILUT \n");
         nalu_hypre_printf("drop tolerance for B = %e, E&F = %e, S = %e \n", nalu_hypre_ParILUDataDroptol(ilu_data)[0],
                      nalu_hypre_ParILUDataDroptol(ilu_data)[1], nalu_hypre_ParILUDataDroptol(ilu_data)[2]);
         nalu_hypre_printf("Max nnz per row = %d \n", nalu_hypre_ParILUDataMaxRowNnz(ilu_data));
         nalu_hypre_printf("Operator Complexity (Fill factor) = %f \n",
                      nalu_hypre_ParILUDataOperatorComplexity(ilu_data));
         break;
      case 20:
         nalu_hypre_printf("Newton-Schulz-Hotelling with ILU(%d) \n", nalu_hypre_ParILUDataLfil(ilu_data));
         nalu_hypre_printf("Operator Complexity (Fill factor) = %f \n",
                      nalu_hypre_ParILUDataOperatorComplexity(ilu_data));
         break;
      case 21:
         nalu_hypre_printf("Newton-Schulz-Hotelling with ILUT \n");
         nalu_hypre_printf("drop tolerance for B = %e, E&F = %e, S = %e \n", nalu_hypre_ParILUDataDroptol(ilu_data)[0],
                      nalu_hypre_ParILUDataDroptol(ilu_data)[1], nalu_hypre_ParILUDataDroptol(ilu_data)[2]);
         nalu_hypre_printf("Max nnz per row = %d \n", nalu_hypre_ParILUDataMaxRowNnz(ilu_data));
         nalu_hypre_printf("Operator Complexity (Fill factor) = %f \n",
                      nalu_hypre_ParILUDataOperatorComplexity(ilu_data));
         break;
      case 30:
         nalu_hypre_printf("RAS with ILU(%d) \n", nalu_hypre_ParILUDataLfil(ilu_data));
         nalu_hypre_printf("Operator Complexity (Fill factor) = %f \n",
                      nalu_hypre_ParILUDataOperatorComplexity(ilu_data));
         break;
      case 31:
         nalu_hypre_printf("RAS with ILUT \n");
         nalu_hypre_printf("drop tolerance for B = %e, E&F = %e, S = %e \n", nalu_hypre_ParILUDataDroptol(ilu_data)[0],
                      nalu_hypre_ParILUDataDroptol(ilu_data)[1], nalu_hypre_ParILUDataDroptol(ilu_data)[2]);
         nalu_hypre_printf("Max nnz per row = %d \n", nalu_hypre_ParILUDataMaxRowNnz(ilu_data));
         nalu_hypre_printf("Operator Complexity (Fill factor) = %f \n",
                      nalu_hypre_ParILUDataOperatorComplexity(ilu_data));
         break;
      case 40:
         nalu_hypre_printf("ddPQ-ILU-GMRES with ILU(%d) \n", nalu_hypre_ParILUDataLfil(ilu_data));
         nalu_hypre_printf("Operator Complexity (Fill factor) = %f \n",
                      nalu_hypre_ParILUDataOperatorComplexity(ilu_data));
         break;
      case 41:
         nalu_hypre_printf("ddPQ-ILU-GMRES with ILUT \n");
         nalu_hypre_printf("drop tolerance for B = %e, E&F = %e, S = %e \n", nalu_hypre_ParILUDataDroptol(ilu_data)[0],
                      nalu_hypre_ParILUDataDroptol(ilu_data)[1], nalu_hypre_ParILUDataDroptol(ilu_data)[2]);
         nalu_hypre_printf("Max nnz per row = %d \n", nalu_hypre_ParILUDataMaxRowNnz(ilu_data));
         nalu_hypre_printf("Operator Complexity (Fill factor) = %f \n",
                      nalu_hypre_ParILUDataOperatorComplexity(ilu_data));
         break;
      case 50:
         nalu_hypre_printf("RAP-Modified-ILU with ILU(%d) \n", nalu_hypre_ParILUDataLfil(ilu_data));
         nalu_hypre_printf("Operator Complexity (Fill factor) = %f \n",
                      nalu_hypre_ParILUDataOperatorComplexity(ilu_data));
         break;
      default: nalu_hypre_printf("Unknown type \n");
         break;
   }

   nalu_hypre_printf("\n ILU Solver Parameters: \n");
   nalu_hypre_printf("Max number of iterations: %d\n", nalu_hypre_ParILUDataMaxIter(ilu_data));
   nalu_hypre_printf("Triangular solver type: %d\n", nalu_hypre_ParILUDataTriSolve(ilu_data));
   nalu_hypre_printf("Lower Jacobi Iterations: %d\n", nalu_hypre_ParILUDataLowerJacobiIters(ilu_data));
   nalu_hypre_printf("Upper Jacobi Iterations: %d\n", nalu_hypre_ParILUDataUpperJacobiIters(ilu_data));
   nalu_hypre_printf("Stopping tolerance: %e\n", nalu_hypre_ParILUDataTol(ilu_data));

   return nalu_hypre_error_flag;
}

/* helper functions */
/*
 * Add an element to the heap
 * I means NALU_HYPRE_Int
 * R means NALU_HYPRE_Real
 * max/min heap
 * r means heap goes from 0 to -1, -2 instead of 0 1 2
 * Ii and Ri means orderd by value of heap, like iw for ILU
 * heap: array of that heap
 * len: the current length of the heap
 * WARNING: You should first put that element to the end of the heap
 *    and add the length of heap by one before call this function.
 * the reason is that we don't want to change something outside the
 *    heap, so left it to the user
 */
NALU_HYPRE_Int
nalu_hypre_ILUMinHeapAddI(NALU_HYPRE_Int *heap, NALU_HYPRE_Int len)
{
   /* parent, left, right */
   NALU_HYPRE_Int p;
   len--;/* now len is the current index */
   while (len > 0)
   {
      /* get the parent index */
      p = (len - 1) / 2;
      if (heap[p] > heap[len])
      {
         /* this is smaller */
         nalu_hypre_swap(heap, p, len);
         len = p;
      }
      else
      {
         break;
      }
   }
   return nalu_hypre_error_flag;
}

/* see nalu_hypre_ILUMinHeapAddI for detail instructions */
NALU_HYPRE_Int
nalu_hypre_ILUMinHeapAddIIIi(NALU_HYPRE_Int *heap, NALU_HYPRE_Int *I1, NALU_HYPRE_Int *Ii1, NALU_HYPRE_Int len)
{
   /* parent, left, right */
   NALU_HYPRE_Int p;
   len--;/* now len is the current index */
   while (len > 0)
   {
      /* get the parent index */
      p = (len - 1) / 2;
      if (heap[p] > heap[len])
      {
         /* this is smaller */
         nalu_hypre_swap(Ii1, heap[p], heap[len]);
         nalu_hypre_swap2i(heap, I1, p, len);
         len = p;
      }
      else
      {
         break;
      }
   }
   return nalu_hypre_error_flag;
}

/* see nalu_hypre_ILUMinHeapAddI for detail instructions */
NALU_HYPRE_Int
nalu_hypre_ILUMinHeapAddIRIi(NALU_HYPRE_Int *heap, NALU_HYPRE_Real *I1, NALU_HYPRE_Int *Ii1, NALU_HYPRE_Int len)
{
   /* parent, left, right */
   NALU_HYPRE_Int p;
   len--;/* now len is the current index */
   while (len > 0)
   {
      /* get the parent index */
      p = (len - 1) / 2;
      if (heap[p] > heap[len])
      {
         /* this is smaller */
         nalu_hypre_swap(Ii1, heap[p], heap[len]);
         nalu_hypre_swap2(heap, I1, p, len);
         len = p;
      }
      else
      {
         break;
      }
   }
   return nalu_hypre_error_flag;
}

/* see nalu_hypre_ILUMinHeapAddI for detail instructions */
NALU_HYPRE_Int
nalu_hypre_ILUMaxHeapAddRabsIIi(NALU_HYPRE_Real *heap, NALU_HYPRE_Int *I1, NALU_HYPRE_Int *Ii1, NALU_HYPRE_Int len)
{
   /* parent, left, right */
   NALU_HYPRE_Int p;
   len--;/* now len is the current index */
   while (len > 0)
   {
      /* get the parent index */
      p = (len - 1) / 2;
      if (nalu_hypre_abs(heap[p]) < nalu_hypre_abs(heap[len]))
      {
         /* this is smaller */
         nalu_hypre_swap(Ii1, heap[p], heap[len]);
         nalu_hypre_swap2(I1, heap, p, len);
         len = p;
      }
      else
      {
         break;
      }
   }
   return nalu_hypre_error_flag;
}

/* see nalu_hypre_ILUMinHeapAddI for detail instructions */
NALU_HYPRE_Int
nalu_hypre_ILUMaxrHeapAddRabsI(NALU_HYPRE_Real *heap, NALU_HYPRE_Int *I1, NALU_HYPRE_Int len)
{
   /* parent, left, right */
   NALU_HYPRE_Int p;
   len--;/* now len is the current index */
   while (len > 0)
   {
      /* get the parent index */
      p = (len - 1) / 2;
      if (nalu_hypre_abs(heap[-p]) < nalu_hypre_abs(heap[-len]))
      {
         /* this is smaller */
         nalu_hypre_swap2(I1, heap, -p, -len);
         len = p;
      }
      else
      {
         break;
      }
   }
   return nalu_hypre_error_flag;
}

/*
 * Swap the first element with the last element of the heap,
 *    reduce size by one, and maintain the heap structure
 * I means NALU_HYPRE_Int
 * R means NALU_HYPRE_Real
 * max/min heap
 * r means heap goes from 0 to -1, -2 instead of 0 1 2
 * Ii and Ri means orderd by value of heap, like iw for ILU
 * heap: aray of that heap
 * len: current length of the heap
 * WARNING: Remember to change the len youself
 */
NALU_HYPRE_Int
nalu_hypre_ILUMinHeapRemoveI(NALU_HYPRE_Int *heap, NALU_HYPRE_Int len)
{
   /* parent, left, right */
   NALU_HYPRE_Int p, l, r;
   len--;/* now len is the max index */
   /* swap the first element to last */
   nalu_hypre_swap(heap, 0, len);
   p = 0;
   l = 1;
   /* while I'm still in the heap */
   while (l < len)
   {
      r = 2 * p + 2;
      /* two childs, pick the smaller one */
      l = r >= len || heap[l] < heap[r] ? l : r;
      if (heap[l] < heap[p])
      {
         nalu_hypre_swap(heap, l, p);
         p = l;
         l = 2 * p + 1;
      }
      else
      {
         break;
      }
   }
   return nalu_hypre_error_flag;
}

/* see nalu_hypre_ILUMinHeapRemoveI for detail instructions */
NALU_HYPRE_Int
nalu_hypre_ILUMinHeapRemoveIIIi(NALU_HYPRE_Int *heap, NALU_HYPRE_Int *I1, NALU_HYPRE_Int *Ii1, NALU_HYPRE_Int len)
{
   /* parent, left, right */
   NALU_HYPRE_Int p, l, r;
   len--;/* now len is the max index */
   /* swap the first element to last */
   nalu_hypre_swap(Ii1, heap[0], heap[len]);
   nalu_hypre_swap2i(heap, I1, 0, len);
   p = 0;
   l = 1;
   /* while I'm still in the heap */
   while (l < len)
   {
      r = 2 * p + 2;
      /* two childs, pick the smaller one */
      l = r >= len || heap[l] < heap[r] ? l : r;
      if (heap[l] < heap[p])
      {
         nalu_hypre_swap(Ii1, heap[p], heap[l]);
         nalu_hypre_swap2i(heap, I1, l, p);
         p = l;
         l = 2 * p + 1;
      }
      else
      {
         break;
      }
   }
   return nalu_hypre_error_flag;
}

/* see nalu_hypre_ILUMinHeapRemoveI for detail instructions */
NALU_HYPRE_Int
nalu_hypre_ILUMinHeapRemoveIRIi(NALU_HYPRE_Int *heap, NALU_HYPRE_Real *I1, NALU_HYPRE_Int *Ii1, NALU_HYPRE_Int len)
{
   /* parent, left, right */
   NALU_HYPRE_Int p, l, r;
   len--;/* now len is the max index */
   /* swap the first element to last */
   nalu_hypre_swap(Ii1, heap[0], heap[len]);
   nalu_hypre_swap2(heap, I1, 0, len);
   p = 0;
   l = 1;
   /* while I'm still in the heap */
   while (l < len)
   {
      r = 2 * p + 2;
      /* two childs, pick the smaller one */
      l = r >= len || heap[l] < heap[r] ? l : r;
      if (heap[l] < heap[p])
      {
         nalu_hypre_swap(Ii1, heap[p], heap[l]);
         nalu_hypre_swap2(heap, I1, l, p);
         p = l;
         l = 2 * p + 1;
      }
      else
      {
         break;
      }
   }
   return nalu_hypre_error_flag;
}

/* see nalu_hypre_ILUMinHeapRemoveI for detail instructions */
NALU_HYPRE_Int
nalu_hypre_ILUMaxHeapRemoveRabsIIi(NALU_HYPRE_Real *heap, NALU_HYPRE_Int *I1, NALU_HYPRE_Int *Ii1, NALU_HYPRE_Int len)
{
   /* parent, left, right */
   NALU_HYPRE_Int p, l, r;
   len--;/* now len is the max index */
   /* swap the first element to last */
   nalu_hypre_swap(Ii1, heap[0], heap[len]);
   nalu_hypre_swap2(I1, heap, 0, len);
   p = 0;
   l = 1;
   /* while I'm still in the heap */
   while (l < len)
   {
      r = 2 * p + 2;
      /* two childs, pick the smaller one */
      l = r >= len || nalu_hypre_abs(heap[l]) > nalu_hypre_abs(heap[r]) ? l : r;
      if (nalu_hypre_abs(heap[l]) > nalu_hypre_abs(heap[p]))
      {
         nalu_hypre_swap(Ii1, heap[p], heap[l]);
         nalu_hypre_swap2(I1, heap, l, p);
         p = l;
         l = 2 * p + 1;
      }
      else
      {
         break;
      }
   }
   return nalu_hypre_error_flag;
}

/* see nalu_hypre_ILUMinHeapRemoveI for detail instructions */
NALU_HYPRE_Int
nalu_hypre_ILUMaxrHeapRemoveRabsI(NALU_HYPRE_Real *heap, NALU_HYPRE_Int *I1, NALU_HYPRE_Int len)
{
   /* parent, left, right */
   NALU_HYPRE_Int p, l, r;
   len--;/* now len is the max index */
   /* swap the first element to last */
   nalu_hypre_swap2(I1, heap, 0, -len);
   p = 0;
   l = 1;
   /* while I'm still in the heap */
   while (l < len)
   {
      r = 2 * p + 2;
      /* two childs, pick the smaller one */
      l = r >= len || nalu_hypre_abs(heap[-l]) > nalu_hypre_abs(heap[-r]) ? l : r;
      if (nalu_hypre_abs(heap[-l]) > nalu_hypre_abs(heap[-p]))
      {
         nalu_hypre_swap2(I1, heap, -l, -p);
         p = l;
         l = 2 * p + 1;
      }
      else
      {
         break;
      }
   }
   return nalu_hypre_error_flag;
}

/* Split based on quick sort algorithm (avoid sorting the entire array)
 * find the largest k elements out of original array
 * array: input array for compare
 * I: integer array bind with array
 * k: largest k elements
 * len: length of the array
 */
NALU_HYPRE_Int
nalu_hypre_ILUMaxQSplitRabsI(NALU_HYPRE_Real *array, NALU_HYPRE_Int *I, NALU_HYPRE_Int left, NALU_HYPRE_Int bound,
                        NALU_HYPRE_Int right)
{
   NALU_HYPRE_Int i, last;
   if (left >= right)
   {
      return nalu_hypre_error_flag;
   }
   nalu_hypre_swap2(I, array, left, (left + right) / 2);
   last = left;
   for (i = left + 1 ; i <= right ; i ++)
   {
      if (nalu_hypre_abs(array[i]) > nalu_hypre_abs(array[left]))
      {
         nalu_hypre_swap2(I, array, ++last, i);
      }
   }
   nalu_hypre_swap2(I, array, left, last);
   nalu_hypre_ILUMaxQSplitRabsI(array, I, left, bound, last - 1);
   if (bound > last)
   {
      nalu_hypre_ILUMaxQSplitRabsI(array, I, last + 1, bound, right);
   }

   return nalu_hypre_error_flag;
}

/* Helper function to search max value from a row
 * array: the array we work on
 * start: the start of the search range
 * end: the end of the search range
 * nLU: ignore rows (new row index) after nLU
 * rperm: reverse permutation array rperm[old] = new.
 *        if rperm set to NULL, ingore nLU and rperm
 * value: return the value ge get (absolute value)
 * index: return the index of that value, could be NULL which means not return
 * l1_norm: return the l1_norm of the array, could be NULL which means no return
 * nnz: return the number of nonzeros inside this array, could be NULL which means no return
 */
NALU_HYPRE_Int
nalu_hypre_ILUMaxRabs(NALU_HYPRE_Real *array_data, NALU_HYPRE_Int *array_j, NALU_HYPRE_Int start, NALU_HYPRE_Int end,
                 NALU_HYPRE_Int nLU, NALU_HYPRE_Int *rperm, NALU_HYPRE_Real *value, NALU_HYPRE_Int *index, NALU_HYPRE_Real *l1_norm,
                 NALU_HYPRE_Int *nnz)
{
   NALU_HYPRE_Int i, idx, col;
   NALU_HYPRE_Real val, max_value, norm, nz;

   nz = 0;
   norm = 0.0;
   max_value = -1.0;
   idx = -1;
   if (rperm)
   {
      /* apply rperm and nLU */
      for (i = start ; i < end ; i ++)
      {
         col = rperm[array_j[i]];
         if (col > nLU)
         {
            /* this old column is in new external part */
            continue;
         }
         nz ++;
         val = nalu_hypre_abs(array_data[i]);
         norm += val;
         if (max_value < val)
         {
            max_value = val;
            idx = i;
         }
      }
   }
   else
   {
      /* basic search */
      for (i = start ; i < end ; i ++)
      {
         val = nalu_hypre_abs(array_data[i]);
         norm += val;
         if (max_value < val)
         {
            max_value = val;
            idx = i;
         }
      }
      nz = end - start;
   }

   *value = max_value;
   if (index)
   {
      *index = idx;
   }
   if (l1_norm)
   {
      *l1_norm = norm;
   }
   if (nnz)
   {
      *nnz = nz;
   }

   return nalu_hypre_error_flag;
}

/* Pre selection for ddPQ, this is the basic version considering row sparsity
 * n: size of matrix
 * nLU: size we consider ddPQ reorder, only first nLU*nLU block is considered
 * A_diag_i/j/data: information of A
 * tol: tol for ddPQ, normally between 0.1-0.3
 * *perm: current row order
 * *rperm: current column order
 * *pperm_pre: output ddPQ pre row roder
 * *qperm_pre: output ddPQ pre column order
 */
NALU_HYPRE_Int
nalu_hypre_ILUGetPermddPQPre(NALU_HYPRE_Int n, NALU_HYPRE_Int nLU, NALU_HYPRE_Int *A_diag_i, NALU_HYPRE_Int *A_diag_j,
                        NALU_HYPRE_Real *A_diag_data, NALU_HYPRE_Real tol, NALU_HYPRE_Int *perm, NALU_HYPRE_Int *rperm,
                        NALU_HYPRE_Int *pperm_pre, NALU_HYPRE_Int *qperm_pre, NALU_HYPRE_Int *nB)
{
   NALU_HYPRE_Int   i, ii, nB_pre, k1, k2;
   NALU_HYPRE_Real  gtol, max_value, norm;

   NALU_HYPRE_Int   *jcol, *jnnz;
   NALU_HYPRE_Real  *weight;

   weight      = nalu_hypre_TAlloc(NALU_HYPRE_Real, nLU + 1, NALU_HYPRE_MEMORY_HOST);
   jcol        = nalu_hypre_TAlloc(NALU_HYPRE_Int, nLU + 1, NALU_HYPRE_MEMORY_HOST);
   jnnz        = nalu_hypre_TAlloc(NALU_HYPRE_Int, nLU + 1, NALU_HYPRE_MEMORY_HOST);

   max_value   = -1.0;
   /* first need to build gtol */
   for ( ii = 0 ; ii < nLU ; ii ++)
   {
      /* find real row */
      i = perm[ii];
      k1 = A_diag_i[i];
      k2 = A_diag_i[i + 1];
      /* find max|a| of that row and its index */
      nalu_hypre_ILUMaxRabs(A_diag_data, A_diag_j, k1, k2, nLU, rperm, weight + ii, jcol + ii, &norm,
                       jnnz + ii);
      weight[ii] /= norm;
      if (weight[ii] > max_value)
      {
         max_value = weight[ii];
      }
   }

   gtol = tol * max_value;

   /* second loop to pre select B */
   nB_pre = 0;
   for ( ii = 0 ; ii < nLU ; ii ++)
   {
      /* keep this row */
      if (weight[ii] > gtol)
      {
         weight[nB_pre] /= (NALU_HYPRE_Real)(jnnz[ii]);
         pperm_pre[nB_pre] = perm[ii];
         qperm_pre[nB_pre++] = A_diag_j[jcol[ii]];
      }
   }

   *nB = nB_pre;

   /* sort from small to large */
   nalu_hypre_qsort3(weight, pperm_pre, qperm_pre, 0, nB_pre - 1);

   nalu_hypre_TFree(weight, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(jcol, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(jnnz, NALU_HYPRE_MEMORY_HOST);

   return nalu_hypre_error_flag;
}

/* Get ddPQ version perm array for ParCSR
 * Greedy matching selection
 * ddPQ is a two-side permutation for diagonal dominance
 * A: the input matrix
 * pperm: row permutation
 * qperm: col permutation
 * nB: the size of B block
 * nI: number of interial nodes
 * tol: the dropping tolorance for ddPQ
 * reordering_type: Type of reordering for the interior nodes.
 * Currently only supports RCM reordering. Set to 0 for no reordering.
 */

NALU_HYPRE_Int
nalu_hypre_ILUGetPermddPQ(nalu_hypre_ParCSRMatrix *A, NALU_HYPRE_Int **io_pperm, NALU_HYPRE_Int **io_qperm,
                     NALU_HYPRE_Real tol, NALU_HYPRE_Int *nB, NALU_HYPRE_Int *nI, NALU_HYPRE_Int reordering_type)
{
   NALU_HYPRE_Int         i, nB_pre, irow, jcol, nLU;
   NALU_HYPRE_Int         *pperm, *qperm;
   NALU_HYPRE_Int         *rpperm, *rqperm, *pperm_pre, *qperm_pre;

   /* data objects for A */
   nalu_hypre_CSRMatrix   *A_diag = nalu_hypre_ParCSRMatrixDiag(A);
   NALU_HYPRE_Int         *A_diag_i = nalu_hypre_CSRMatrixI(A_diag);
   NALU_HYPRE_Int         *A_diag_j = nalu_hypre_CSRMatrixJ(A_diag);
   NALU_HYPRE_Real        *A_diag_data = nalu_hypre_CSRMatrixData(A_diag);

   /* problem size */
   NALU_HYPRE_Int         n = nalu_hypre_CSRMatrixNumRows(A_diag);

   /* 1: Setup and create memory
    */

   pperm             = NULL;
   qperm             = nalu_hypre_TAlloc(NALU_HYPRE_Int, n, NALU_HYPRE_MEMORY_DEVICE);
   rpperm            = nalu_hypre_TAlloc(NALU_HYPRE_Int, n, NALU_HYPRE_MEMORY_HOST);
   rqperm            = nalu_hypre_TAlloc(NALU_HYPRE_Int, n, NALU_HYPRE_MEMORY_HOST);

   /* 2: Find interior nodes first
    */
   nalu_hypre_ILUGetInteriorExteriorPerm( A, &pperm, &nLU, 0);
   *nI = nLU;

   /* 3: Pre selection on interial nodes
    * this pre selection puts external nodes to the last
    * also provide candidate rows for B block
    */

   /* build reverse permutation array
    * rperm[old] = new
    */
   for (i = 0 ; i < n ; i ++)
   {
      rpperm[pperm[i]] = i;
   }

   /* build place holder for pre selection pairs */
   pperm_pre = nalu_hypre_TAlloc(NALU_HYPRE_Int, nLU, NALU_HYPRE_MEMORY_HOST);
   qperm_pre = nalu_hypre_TAlloc(NALU_HYPRE_Int, nLU, NALU_HYPRE_MEMORY_HOST);

   /* pre selection */
   nalu_hypre_ILUGetPermddPQPre(n, nLU, A_diag_i, A_diag_j, A_diag_data, tol, pperm, rpperm, pperm_pre,
                           qperm_pre, &nB_pre);

   /* 4: Build B block
    * Greedy selection
    */

   /* rperm[old] = new */
   for (i = 0 ; i < nLU ; i ++)
   {
      rpperm[pperm[i]] = -1;
   }

   nalu_hypre_TMemcpy( rqperm, rpperm, NALU_HYPRE_Int, n, NALU_HYPRE_MEMORY_HOST, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TMemcpy( qperm, pperm, NALU_HYPRE_Int, n, NALU_HYPRE_MEMORY_DEVICE, NALU_HYPRE_MEMORY_DEVICE);

   /* we sort from small to large, so we need to go from back to start
    * we only need nB_pre to start the loop, after that we could use it for size of B
    */
   for (i = nB_pre - 1, nB_pre = 0 ; i >= 0 ; i --)
   {
      irow = pperm_pre[i];
      jcol = qperm_pre[i];

      /* this col is not yet taken */
      if (rqperm[jcol] < 0)
      {
         rpperm[irow] = nB_pre;
         rqperm[jcol] = nB_pre;
         pperm[nB_pre] = irow;
         qperm[nB_pre++] = jcol;
      }
   }

   /* 5: Complete the permutation
    * rperm[old] = new
    * those still mapped to a new index means not yet covered
    */
   nLU = nB_pre;
   for (i = 0 ; i < n ; i ++)
   {
      if (rpperm[i] < 0)
      {
         pperm[nB_pre++] = i;
      }
   }
   nB_pre = nLU;
   for (i = 0 ; i < n ; i ++)
   {
      if (rqperm[i] < 0)
      {
         qperm[nB_pre++] = i;
      }
   }

   /* Finishing up and free
    */

   switch (reordering_type)
   {
      case 0:
         /* no RCM in this case */
         break;
      case 1:
         /* RCM */
         nalu_hypre_ILULocalRCM( nalu_hypre_ParCSRMatrixDiag(A), 0, nLU, &pperm, &qperm, 0);
         break;
      default:
         /* RCM */
         nalu_hypre_ILULocalRCM( nalu_hypre_ParCSRMatrixDiag(A), 0, nLU, &pperm, &qperm, 0);
         break;
   }

   *nB = nLU;
   *io_pperm = pperm;
   *io_qperm = qperm;

   nalu_hypre_TFree( rpperm, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree( rqperm, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree( pperm_pre, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree( qperm_pre, NALU_HYPRE_MEMORY_HOST);
   return nalu_hypre_error_flag;
}

/*
 * Get perm array from parcsr matrix based on diag and offdiag matrix
 * Just simply loop through the rows of offd of A, check for nonzero rows
 * Put interior nodes at the beginning
 * A: parcsr matrix
 * perm: permutation array
 * nLU: number of interial nodes
 * reordering_type: Type of (additional) reordering for the interior nodes.
 * Currently only supports RCM reordering. Set to 0 for no reordering.
 */
NALU_HYPRE_Int
nalu_hypre_ILUGetInteriorExteriorPerm(nalu_hypre_ParCSRMatrix *A, NALU_HYPRE_Int **perm, NALU_HYPRE_Int *nLU,
                                 NALU_HYPRE_Int reordering_type)
{
   /* get basic information of A */
   NALU_HYPRE_Int            n = nalu_hypre_ParCSRMatrixNumRows(A);
   NALU_HYPRE_Int            i, j, first, last, start, end;
   NALU_HYPRE_Int            num_sends, send_map_start, send_map_end, col;
   nalu_hypre_CSRMatrix      *A_offd;
   NALU_HYPRE_Int            *A_offd_i;
   A_offd               = nalu_hypre_ParCSRMatrixOffd(A);
   A_offd_i             = nalu_hypre_CSRMatrixI(A_offd);
   first                = 0;
   last                 = n - 1;
   NALU_HYPRE_Int            *temp_perm = nalu_hypre_TAlloc(NALU_HYPRE_Int, n, NALU_HYPRE_MEMORY_DEVICE);
   NALU_HYPRE_Int            *marker = nalu_hypre_CTAlloc(NALU_HYPRE_Int, n, NALU_HYPRE_MEMORY_HOST);

   /* first get col nonzero from com_pkg */
   /* get comm_pkg, craete one if we not yet have one */
   nalu_hypre_ParCSRCommPkg  *comm_pkg = nalu_hypre_ParCSRMatrixCommPkg(A);
   if (!comm_pkg)
   {
      nalu_hypre_MatvecCommPkgCreate(A);
      comm_pkg = nalu_hypre_ParCSRMatrixCommPkg(A);
   }

   /* now directly take adavantage of comm_pkg */
   num_sends = nalu_hypre_ParCSRCommPkgNumSends(comm_pkg);
   for ( i = 0 ; i < num_sends ; i ++ )
   {
      send_map_start = nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
      send_map_end = nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, i + 1);
      for ( j = send_map_start ; j < send_map_end ; j ++)
      {
         col = nalu_hypre_ParCSRCommPkgSendMapElmt(comm_pkg, j);
         if (marker[col] == 0)
         {
            temp_perm[last--] = col;
            marker[col] = -1;
         }
      }
   }

   /* now deal with the row */
   for ( i = 0 ; i < n ; i ++)
   {
      if (marker[i] == 0)
      {
         start = A_offd_i[i];
         end = A_offd_i[i + 1];
         if (start == end)
         {
            temp_perm[first++] = i;
         }
         else
         {
            temp_perm[last--] = i;
         }
      }
   }
   switch (reordering_type)
   {
      case 0:
         /* no RCM in this case */
         break;
      case 1:
         /* RCM */
         nalu_hypre_ILULocalRCM( nalu_hypre_ParCSRMatrixDiag(A), 0, first, &temp_perm, &temp_perm, 1);
         break;
      default:
         /* RCM */
         nalu_hypre_ILULocalRCM( nalu_hypre_ParCSRMatrixDiag(A), 0, first, &temp_perm, &temp_perm, 1);
         break;
   }

   /* set out values */
   *nLU = first;
   if ((*perm) != NULL) { nalu_hypre_TFree(*perm, NALU_HYPRE_MEMORY_DEVICE); }
   *perm = temp_perm;

   nalu_hypre_TFree(marker, NALU_HYPRE_MEMORY_HOST);
   return nalu_hypre_error_flag;
}

/*
 * Get the (local) ordering of the diag (local) matrix (no permutation). This is the permutation used for the block-jacobi case
 * A: parcsr matrix
 * perm: permutation array
 * nLU: number of interior nodes
 * reordering_type: Type of (additional) reordering for the nodes.
 * Currently only supports RCM reordering. Set to 0 for no reordering.
 */
NALU_HYPRE_Int
nalu_hypre_ILUGetLocalPerm(nalu_hypre_ParCSRMatrix *A, NALU_HYPRE_Int **perm, NALU_HYPRE_Int *nLU,
                      NALU_HYPRE_Int reordering_type)
{
   /* get basic information of A */
   NALU_HYPRE_Int            n = nalu_hypre_ParCSRMatrixNumRows(A);
   NALU_HYPRE_Int            i;
   NALU_HYPRE_Int            *temp_perm = nalu_hypre_TAlloc(NALU_HYPRE_Int, n, NALU_HYPRE_MEMORY_DEVICE);

   /* set perm array */
   for ( i = 0 ; i < n ; i ++ )
   {
      temp_perm[i] = i;
   }
   switch (reordering_type)
   {
      case 0:
         /* no RCM in this case */
         break;
      case 1:
         /* RCM */
         nalu_hypre_ILULocalRCM( nalu_hypre_ParCSRMatrixDiag(A), 0, n, &temp_perm, &temp_perm, 1);
         break;
      default:
         /* RCM */
         nalu_hypre_ILULocalRCM( nalu_hypre_ParCSRMatrixDiag(A), 0, n, &temp_perm, &temp_perm, 1);
         break;
   }
   *nLU = n;
   if ((*perm) != NULL) { nalu_hypre_TFree(*perm, NALU_HYPRE_MEMORY_DEVICE); }
   *perm = temp_perm;

   return nalu_hypre_error_flag;
}

#if 0
/* Build the expanded matrix for RAS-1
 * A: input ParCSR matrix
 * E_i, E_j, E_data: information for external matrix
 * rperm: reverse permutation to build real index, rperm[old] = new
 *
 * NOTE: Modified to avoid communicating BigInt arrays - DOK
 */
NALU_HYPRE_Int
nalu_hypre_ILUBuildRASExternalMatrix(nalu_hypre_ParCSRMatrix *A, NALU_HYPRE_Int *rperm, NALU_HYPRE_Int **E_i,
                                NALU_HYPRE_Int **E_j, NALU_HYPRE_Real **E_data)
{
   NALU_HYPRE_Int                i, i1, i2, j, jj, k, row, k1, k2, k3, lend, leno, col, l1, l2;
   NALU_HYPRE_BigInt    big_col;

   /* data objects for communication */
   MPI_Comm                 comm = nalu_hypre_ParCSRMatrixComm(A);
   nalu_hypre_ParCSRCommPkg      *comm_pkg;
   nalu_hypre_ParCSRCommPkg      *comm_pkg_tmp = NULL;
   nalu_hypre_ParCSRCommHandle   *comm_handle_count;
   nalu_hypre_ParCSRCommHandle   *comm_handle_marker;
   nalu_hypre_ParCSRCommHandle   *comm_handle_j;
   nalu_hypre_ParCSRCommHandle   *comm_handle_data;
   NALU_HYPRE_BigInt                *col_starts;
   NALU_HYPRE_Int                total_rows;
   NALU_HYPRE_Int                num_sends;
   NALU_HYPRE_Int                num_recvs;
   NALU_HYPRE_Int                begin, end;
   NALU_HYPRE_Int                my_id, num_procs, proc_id;

   /* data objects for buffers in communication */
   NALU_HYPRE_Int                *send_map;
   NALU_HYPRE_Int                *send_count = NULL, *send_disp = NULL;
   NALU_HYPRE_Int                *send_count_offd = NULL;
   NALU_HYPRE_Int                *recv_count = NULL, *recv_disp = NULL, *recv_marker = NULL;
   NALU_HYPRE_Int                *send_buf_int = NULL;
   NALU_HYPRE_Int                *recv_buf_int = NULL;
   NALU_HYPRE_Real               *send_buf_real = NULL, *recv_buf_real = NULL;
   NALU_HYPRE_Int                *send_disp_comm = NULL, *recv_disp_comm = NULL;

   /* data objects for A */
   nalu_hypre_CSRMatrix          *A_diag = nalu_hypre_ParCSRMatrixDiag(A);
   nalu_hypre_CSRMatrix          *A_offd = nalu_hypre_ParCSRMatrixOffd(A);
   NALU_HYPRE_BigInt             *A_col_starts = nalu_hypre_ParCSRMatrixColStarts(A);
   NALU_HYPRE_BigInt             *A_offd_colmap = nalu_hypre_ParCSRMatrixColMapOffd(A);
   NALU_HYPRE_Real               *A_diag_data = nalu_hypre_CSRMatrixData(A_diag);
   NALU_HYPRE_Int                *A_diag_i = nalu_hypre_CSRMatrixI(A_diag);
   NALU_HYPRE_Int                *A_diag_j = nalu_hypre_CSRMatrixJ(A_diag);
   NALU_HYPRE_Int                *A_offd_i = nalu_hypre_CSRMatrixI(A_offd);
   NALU_HYPRE_Int                *A_offd_j = nalu_hypre_CSRMatrixJ(A_offd);
   NALU_HYPRE_Real               *A_offd_data = nalu_hypre_CSRMatrixData(A_offd);

   /* size */
   NALU_HYPRE_Int                n = nalu_hypre_CSRMatrixNumCols(A_diag);
   NALU_HYPRE_Int                m = nalu_hypre_CSRMatrixNumCols(A_offd);

   /* 1: setup part
    * allocate memory and setup working array
    */

   /* MPI stuff */
   nalu_hypre_MPI_Comm_size(comm, &num_procs);
   nalu_hypre_MPI_Comm_rank(comm, &my_id);

   /* now check communication package */
   comm_pkg = nalu_hypre_ParCSRMatrixCommPkg(A);
   /* create if not yet built */
   if (!comm_pkg)
   {
      nalu_hypre_MatvecCommPkgCreate(A);
      comm_pkg = nalu_hypre_ParCSRMatrixCommPkg(A);
   }

   /* get communication information */
   send_map          = nalu_hypre_ParCSRCommPkgSendMapElmts(comm_pkg);
   num_sends         = nalu_hypre_ParCSRCommPkgNumSends(comm_pkg);
   send_disp_comm    = nalu_hypre_TAlloc(NALU_HYPRE_Int, num_sends + 1, NALU_HYPRE_MEMORY_HOST);
   begin             = nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, 0);
   end               = nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends);
   total_rows        = end - begin;
   num_recvs         = nalu_hypre_ParCSRCommPkgNumRecvs(comm_pkg);
   recv_disp_comm    = nalu_hypre_TAlloc(NALU_HYPRE_Int, num_recvs + 1, NALU_HYPRE_MEMORY_HOST);

   /* create buffers */
   send_count        = nalu_hypre_TAlloc(NALU_HYPRE_Int, total_rows, NALU_HYPRE_MEMORY_HOST);
   send_disp         = nalu_hypre_TAlloc(NALU_HYPRE_Int, total_rows + 1, NALU_HYPRE_MEMORY_HOST);
   send_count_offd   = nalu_hypre_CTAlloc(NALU_HYPRE_Int, total_rows, NALU_HYPRE_MEMORY_HOST);
   recv_count        = nalu_hypre_TAlloc(NALU_HYPRE_Int, m, NALU_HYPRE_MEMORY_HOST);
   recv_marker       = nalu_hypre_TAlloc(NALU_HYPRE_Int, m, NALU_HYPRE_MEMORY_HOST);
   recv_disp         = nalu_hypre_TAlloc(NALU_HYPRE_Int, m + 1, NALU_HYPRE_MEMORY_HOST);

   /* 2: communication part 1 to get amount of send and recv */

   /* first we need to know the global start */
   col_starts        = nalu_hypre_TAlloc(NALU_HYPRE_BigInt, num_procs + 1, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_MPI_Allgather(A_col_starts + 1, 1, NALU_HYPRE_MPI_BIG_INT, col_starts + 1, 1, NALU_HYPRE_MPI_BIG_INT,
                       comm);
   col_starts[0]     = 0;

   send_disp[0]      = 0;
   send_disp_comm[0] = 0;
   /* now loop to know how many to send per row */
   for ( i = 0 ; i < num_sends ; i ++ )
   {
      /* update disp for comm package */
      send_disp_comm[i + 1] = send_disp_comm[i];
      /* get the proc we are sending to */
      proc_id = nalu_hypre_ParCSRCommPkgSendProc(comm_pkg, i);
      /* set start end of this proc */
      l1 = nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
      l2 = nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, i + 1);
      /* loop through rows we need to send */
      for ( j = l1 ; j < l2 ; j ++ )
      {
         /* reset length */
         leno = lend = 0;
         /* we need to send out this row */
         row = nalu_hypre_ParCSRCommPkgSendMapElmt(comm_pkg, j);

         /* check how many we need to send from diagonal first */
         k1 = A_diag_i[row], k2 = A_diag_i[row + 1];
         for ( k = k1 ; k < k2 ; k ++ )
         {
            col = A_diag_j[k];
            if (nalu_hypre_BinarySearch(send_map + l1, col, l2 - l1) >= 0 )
            {
               lend++;
            }
         }

         /* check how many we need to send from offdiagonal */
         k1 = A_offd_i[row], k2 = A_offd_i[row + 1];
         for ( k = k1 ; k < k2 ; k ++ )
         {
            /* get real column number of this offdiagonal column */
            big_col = A_offd_colmap[A_offd_j[k]];
            if (big_col >= col_starts[proc_id] && big_col < col_starts[proc_id + 1])
            {
               /* this column is in diagonal range of proc_id
                * everything in diagonal range need to be in the factorization
                */
               leno++;
            }
         }
         send_count_offd[j]   = leno;
         send_count[j]        = leno + lend;
         send_disp[j + 1]       = send_disp[j] + send_count[j];
         send_disp_comm[i + 1] += send_count[j];
      }
   }

   /* 3: new communication to know how many we need to receive for each external row
    * main communication, 11 is integer
    */
   comm_handle_count    = nalu_hypre_ParCSRCommHandleCreate(11, comm_pkg, send_count, recv_count);
   comm_handle_marker   = nalu_hypre_ParCSRCommHandleCreate(11, comm_pkg, send_count_offd, recv_marker);
   nalu_hypre_ParCSRCommHandleDestroy(comm_handle_count);
   nalu_hypre_ParCSRCommHandleDestroy(comm_handle_marker);

   recv_disp[0] = 0;
   recv_disp_comm[0] = 0;
   /* now build the recv disp array */
   for (i = 0 ; i < num_recvs ; i ++)
   {
      recv_disp_comm[i + 1] = recv_disp_comm[i];
      k1 = nalu_hypre_ParCSRCommPkgRecvVecStart( comm_pkg, i );
      k2 = nalu_hypre_ParCSRCommPkgRecvVecStart( comm_pkg, i + 1 );
      for (j = k1 ; j < k2 ; j ++)
      {
         recv_disp[j + 1] = recv_disp[j] + recv_count[j];
         recv_disp_comm[i + 1] += recv_count[j];
      }
   }

   /* 4: ready to start real communication
    * now we know how many we need to send out, create send/recv buffers
    */
   send_buf_int   = nalu_hypre_TAlloc(NALU_HYPRE_Int, send_disp[total_rows], NALU_HYPRE_MEMORY_HOST);
   send_buf_real  = nalu_hypre_TAlloc(NALU_HYPRE_Real, send_disp[total_rows], NALU_HYPRE_MEMORY_HOST);
   recv_buf_int   = nalu_hypre_TAlloc(NALU_HYPRE_Int, recv_disp[m], NALU_HYPRE_MEMORY_HOST);
   recv_buf_real  = nalu_hypre_TAlloc(NALU_HYPRE_Real, recv_disp[m], NALU_HYPRE_MEMORY_HOST);

   /* fill send buffer */
   for ( i = 0 ; i < num_sends ; i ++ )
   {
      /* get the proc we are sending to */
      proc_id = nalu_hypre_ParCSRCommPkgSendProc(comm_pkg, i);
      /* set start end of this proc */
      l1 = nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
      l2 = nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, i + 1);
      /* loop through rows we need to apply communication */
      for ( j = l1 ; j < l2 ; j ++ )
      {
         /* reset length
          * one remark here, the diagonal we send becomes
          *    off diagonal part for reciver
          */
         leno = send_disp[j];
         lend = leno + send_count_offd[j];
         /* we need to send out this row */
         row = nalu_hypre_ParCSRCommPkgSendMapElmt(comm_pkg, j);

         /* fill diagonal first */
         k1 = A_diag_i[row], k2 = A_diag_i[row + 1];
         for ( k = k1 ; k < k2 ; k ++ )
         {
            col = A_diag_j[k];
            if (nalu_hypre_BinarySearch(send_map + l1, col, l2 - l1) >= 0)
            {
               send_buf_real[lend] = A_diag_data[k];
               /* the diag part becomes offd for recv part, so update index
                * set up to global index
                * set it to be negative
                */
               send_buf_int[lend++] = col;// + col_starts[my_id];
            }
         }

         /* fill offdiagonal */
         k1 = A_offd_i[row], k2 = A_offd_i[row + 1];
         for ( k = k1 ; k < k2 ; k ++ )
         {
            /* get real column number of this offdiagonal column */
            big_col = A_offd_colmap[A_offd_j[k]];
            if (big_col >= col_starts[proc_id] && big_col < col_starts[proc_id + 1])
            {
               /* this column is in diagonal range of proc_id
                * everything in diagonal range need to be in the factorization
                */
               send_buf_real[leno] = A_offd_data[k];
               /* the offd part becomes diagonal for recv part, so update index */
               send_buf_int[leno++] = (NALU_HYPRE_Int)(big_col - col_starts[proc_id]);
            }
         }
      }
   }

   /* now build new comm_pkg for this communication */
   nalu_hypre_ParCSRCommPkgCreateAndFill(comm,
                                    num_recvs,
                                    nalu_hypre_ParCSRCommPkgRecvProcs(comm_pkg),
                                    recv_disp_comm,
                                    num_sends,
                                    nalu_hypre_ParCSRCommPkgSendProcs(comm_pkg),
                                    send_disp_comm,
                                    NULL,
                                    &comm_pkg_tmp);

   /* communication */
   comm_handle_j = nalu_hypre_ParCSRCommHandleCreate(11, comm_pkg_tmp, send_buf_int, recv_buf_int);
   comm_handle_data = nalu_hypre_ParCSRCommHandleCreate(1, comm_pkg_tmp, send_buf_real, recv_buf_real);
   nalu_hypre_ParCSRCommHandleDestroy(comm_handle_j);
   nalu_hypre_ParCSRCommHandleDestroy(comm_handle_data);

   /* Update the index to be real index */
   /* Dealing with diagonal part */
   for (i = 0 ; i < m ; i++ )
   {
      k1 = recv_disp[i];
      k2 = recv_disp[i] + recv_marker[i];
      k3 = recv_disp[i + 1];
      for (j = k1 ; j < k2 ; j ++ )
      {
         recv_buf_int[j] = rperm[recv_buf_int[j]];
      }
   }

   /* Dealing with off-diagonal part */
   for (i = 0 ; i < num_recvs ; i ++)
   {
      proc_id = nalu_hypre_ParCSRCommPkgRecvProc( comm_pkg_tmp, i);
      i1 = nalu_hypre_ParCSRCommPkgRecvVecStart( comm_pkg_tmp, i );
      i2 = nalu_hypre_ParCSRCommPkgRecvVecStart( comm_pkg_tmp, i + 1 );
      for (j = i1 ; j < i2 ; j++)
      {
         k1 = recv_disp[j] + recv_marker[j];
         k2 = recv_disp[j + 1];

         for (jj = k1 ; jj < k2 ; jj++)
         {
            /* Correct index to get actual global index */
            big_col = recv_buf_int[jj] + col_starts[proc_id];
            recv_buf_int[jj] = nalu_hypre_BigBinarySearch( A_offd_colmap, big_col, m) + n;
         }
      }
   }

   /* Assign data */
   *E_i     = recv_disp;
   *E_j     = recv_buf_int;
   *E_data  = recv_buf_real;

   /* 5: finish and free
    */

   nalu_hypre_TFree(send_disp_comm, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(recv_disp_comm, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(comm_pkg_tmp, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(col_starts, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(send_count, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(send_disp, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(send_count_offd, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(recv_count, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(send_buf_int, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(send_buf_real, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(recv_marker, NALU_HYPRE_MEMORY_HOST);

   return nalu_hypre_error_flag;
}
#else
/* Build the expanded matrix for RAS-1
 * A: input ParCSR matrix
 * E_i, E_j, E_data: information for external matrix
 * rperm: reverse permutation to build real index, rperm[old] = new
 */
NALU_HYPRE_Int
nalu_hypre_ILUBuildRASExternalMatrix(nalu_hypre_ParCSRMatrix *A, NALU_HYPRE_Int *rperm, NALU_HYPRE_Int **E_i,
                                NALU_HYPRE_Int **E_j, NALU_HYPRE_Real **E_data)
{
   NALU_HYPRE_Int                i, j, idx;
   NALU_HYPRE_BigInt   big_col;

   /* data objects for communication */
   MPI_Comm                 comm = nalu_hypre_ParCSRMatrixComm(A);
   NALU_HYPRE_Int                my_id;

   /* data objects for A */
   nalu_hypre_CSRMatrix          *A_diag = nalu_hypre_ParCSRMatrixDiag(A);
   nalu_hypre_CSRMatrix          *A_offd = nalu_hypre_ParCSRMatrixOffd(A);
   NALU_HYPRE_BigInt   *A_col_starts = nalu_hypre_ParCSRMatrixColStarts(A);
   NALU_HYPRE_BigInt   *A_offd_colmap = nalu_hypre_ParCSRMatrixColMapOffd(A);
   NALU_HYPRE_Int                *A_diag_i = nalu_hypre_CSRMatrixI(A_diag);
   NALU_HYPRE_Int                *A_offd_i = nalu_hypre_CSRMatrixI(A_offd);

   /* data objects for external A matrix */
   // Need to check the new version of nalu_hypre_ParcsrGetExternalRows
   nalu_hypre_CSRMatrix          *A_ext = NULL;
   // # up to local offd cols, no need to be NALU_HYPRE_BigInt
   NALU_HYPRE_Int                *A_ext_i = NULL;
   // Return global index, NALU_HYPRE_BigInt required
   NALU_HYPRE_BigInt   *A_ext_j = NULL;
   NALU_HYPRE_Real               *A_ext_data = NULL;

   /* data objects for output */
   NALU_HYPRE_Int                E_nnz;
   NALU_HYPRE_Int                *E_ext_i = NULL;
   // Local index, no need to use NALU_HYPRE_BigInt
   NALU_HYPRE_Int                *E_ext_j = NULL;
   NALU_HYPRE_Real               *E_ext_data = NULL;

   //guess non-zeros for E before start
   NALU_HYPRE_Int                E_init_alloc;

   /* size */
   NALU_HYPRE_Int                n = nalu_hypre_CSRMatrixNumCols(A_diag);
   NALU_HYPRE_Int                m = nalu_hypre_CSRMatrixNumCols(A_offd);
   NALU_HYPRE_Int                A_diag_nnz = A_diag_i[n];
   NALU_HYPRE_Int                A_offd_nnz = A_offd_i[n];

   /* 1: Set up phase and get external rows
    * Use the HYPRE build-in function
    */

   /* MPI stuff */
   //nalu_hypre_MPI_Comm_size(comm, &num_procs);
   nalu_hypre_MPI_Comm_rank(comm, &my_id);

   /* Param of nalu_hypre_ParcsrGetExternalRows:
    * nalu_hypre_ParCSRMatrix   *A          [in]  -> Input parcsr matrix.
    * NALU_HYPRE_Int            indies_len  [in]  -> Input length of indices_len array
    * NALU_HYPRE_Int            *indices    [in]  -> Input global indices of rows we want to get
    * nalu_hypre_CSRMatrix      **A_ext     [out] -> Return the external CSR matrix.
    * nalu_hypre_ParCSRCommPkg  commpkg_out [out] -> Return commpkg if set to a point. Use NULL here since we don't want it.
    */
   //   nalu_hypre_ParcsrGetExternalRows( A, m, A_offd_colmap, &A_ext, NULL );
   A_ext = nalu_hypre_ParCSRMatrixExtractBExt(A, A, 1);

   A_ext_i              = nalu_hypre_CSRMatrixI(A_ext);
   //This should be NALU_HYPRE_BigInt since this is global index, use big_j in csr */
   A_ext_j = nalu_hypre_CSRMatrixBigJ(A_ext);
   A_ext_data           = nalu_hypre_CSRMatrixData(A_ext);

   /* guess memory we need to allocate to E_j */
   E_init_alloc =  nalu_hypre_max( (NALU_HYPRE_Int) ( A_diag_nnz / (NALU_HYPRE_Real) n / (NALU_HYPRE_Real) n *
                                            (NALU_HYPRE_Real) m * (NALU_HYPRE_Real) m + A_offd_nnz), 1);

   /* Initial guess */
   E_ext_i     = nalu_hypre_TAlloc(NALU_HYPRE_Int, m + 1, NALU_HYPRE_MEMORY_HOST);
   E_ext_j     = nalu_hypre_TAlloc(NALU_HYPRE_Int, E_init_alloc, NALU_HYPRE_MEMORY_HOST);
   E_ext_data  = nalu_hypre_TAlloc(NALU_HYPRE_Real, E_init_alloc, NALU_HYPRE_MEMORY_HOST);

   /* 2: Discard unecessary cols
    * Search A_ext_j, discard those cols not belong to current proc
    * First check diag, and search in offd_col_map
    */

   E_nnz       = 0;
   E_ext_i[0]  = 0;

   for ( i = 0 ;  i < m ; i ++)
   {
      E_ext_i[i] = E_nnz;
      for ( j = A_ext_i[i] ; j < A_ext_i[i + 1] ; j ++)
      {
         big_col = A_ext_j[j];
         /* First check if that belongs to the diagonal part */
         if ( big_col >= A_col_starts[0] && big_col < A_col_starts[1] )
         {
            /* this is a diagonal entry, rperm (map old to new) and shift it */

            /* Note here, the result of big_col - A_col_starts[0] in no longer a NALU_HYPRE_BigInt */
            idx = (NALU_HYPRE_Int)(big_col - A_col_starts[0]);
            E_ext_j[E_nnz]       = rperm[idx];
            E_ext_data[E_nnz++]  = A_ext_data[j];
         }

         /* If not, apply binary search to check if is offdiagonal */
         else
         {
            /* Search, result is not NALU_HYPRE_BigInt */
            E_ext_j[E_nnz] = nalu_hypre_BigBinarySearch( A_offd_colmap, big_col, m);
            if ( E_ext_j[E_nnz] >= 0)
            {
               /* this is an offdiagonal entry */
               E_ext_j[E_nnz]      = E_ext_j[E_nnz] + n;
               E_ext_data[E_nnz++] = A_ext_data[j];
            }
            else
            {
               /* skip capacity check */
               continue;
            }
         }
         /* capacity check, allocate new memory when full */
         if (E_nnz >= E_init_alloc)
         {
            NALU_HYPRE_Int tmp;
            tmp = E_init_alloc;
            E_init_alloc   = E_init_alloc * EXPAND_FACT + 1;
            E_ext_j        = nalu_hypre_TReAlloc_v2(E_ext_j, NALU_HYPRE_Int, tmp, NALU_HYPRE_Int, E_init_alloc,
                                               NALU_HYPRE_MEMORY_HOST);
            E_ext_data     = nalu_hypre_TReAlloc_v2(E_ext_data, NALU_HYPRE_Real, tmp, NALU_HYPRE_Real, E_init_alloc,
                                               NALU_HYPRE_MEMORY_HOST);
         }
      }
   }
   E_ext_i[m] = E_nnz;

   /* 3: Free and finish up
    * Free memory, set E_i, E_j and E_data
    */

   *E_i     = E_ext_i;
   *E_j     = E_ext_j;
   *E_data  = E_ext_data;

   nalu_hypre_CSRMatrixDestroy(A_ext);

   return nalu_hypre_error_flag;

}
#endif

/* This function sort offdiagonal map as well as J array for offdiagonal part
 * A: The input CSR matrix
 */
NALU_HYPRE_Int
nalu_hypre_ILUSortOffdColmap(nalu_hypre_ParCSRMatrix *A)
{
   NALU_HYPRE_Int i;
   nalu_hypre_CSRMatrix *A_offd    = nalu_hypre_ParCSRMatrixOffd(A);
   NALU_HYPRE_Int *A_offd_j        = nalu_hypre_CSRMatrixJ(A_offd);
   NALU_HYPRE_BigInt *A_offd_colmap   = nalu_hypre_ParCSRMatrixColMapOffd(A);
   NALU_HYPRE_Int len              = nalu_hypre_CSRMatrixNumCols(A_offd);
   NALU_HYPRE_Int nnz              = nalu_hypre_CSRMatrixNumNonzeros(A_offd);
   NALU_HYPRE_Int *perm            = nalu_hypre_TAlloc(NALU_HYPRE_Int, len, NALU_HYPRE_MEMORY_HOST);
   NALU_HYPRE_Int *rperm           = nalu_hypre_TAlloc(NALU_HYPRE_Int, len, NALU_HYPRE_MEMORY_HOST);

   for (i = 0 ; i < len ; i ++)
   {
      perm[i] = i;
   }

   nalu_hypre_BigQsort2i(A_offd_colmap, perm, 0, len - 1);

   for (i = 0 ; i < len ; i ++)
   {
      rperm[perm[i]] = i;
   }

   for (i = 0 ; i < nnz ; i ++)
   {
      A_offd_j[i] = rperm[A_offd_j[i]];
   }

   nalu_hypre_TFree(perm, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(rperm, NALU_HYPRE_MEMORY_HOST);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ILULocalRCM
 *--------------------------------------------------------------------------*/

/* This function computes the RCM ordering of a sub matrix of
 * sparse matrix B = A(perm,perm)
 * For nonsymmetrix problem, is the RCM ordering of B + B'
 * A: The input CSR matrix
 * start:      the start position of the submatrix in B
 * end:        the end position of the submatrix in B ( exclude end, [start,end) )
 * permp:      pointer to the row permutation array such that B = A(perm, perm)
 *             point to NULL if you want to work directly on A
 *             on return, permp will point to the new permutation where
 *             in [start, end) the matrix will reordered
 * qpermp:     pointer to the col permutation array such that B = A(perm, perm)
 *             point to NULL or equal to permp if you want symmetric order
 *             on return, qpermp will point to the new permutation where
 *             in [start, end) the matrix will reordered
 * sym:        set to nonzero to work on A only(symmetric), otherwise A + A'.
 *             WARNING: if you use non-symmetric reordering, that is,
 *             different row and col reordering, the resulting A might be non-symmetric.
 *             Be careful if you are using non-symmetric reordering
 */
NALU_HYPRE_Int
nalu_hypre_ILULocalRCM( nalu_hypre_CSRMatrix *A, NALU_HYPRE_Int start, NALU_HYPRE_Int end,
                   NALU_HYPRE_Int **permp, NALU_HYPRE_Int **qpermp, NALU_HYPRE_Int sym)
{
   NALU_HYPRE_Int               i, j, row, col, r1, r2;

   NALU_HYPRE_Int               num_nodes      = end - start;
   NALU_HYPRE_Int               n              = nalu_hypre_CSRMatrixNumRows(A);
   NALU_HYPRE_Int               ncol           = nalu_hypre_CSRMatrixNumCols(A);
   NALU_HYPRE_Int               *A_i           = nalu_hypre_CSRMatrixI(A);
   NALU_HYPRE_Int               *A_j           = nalu_hypre_CSRMatrixJ(A);
   nalu_hypre_CSRMatrix         *GT            = NULL;
   nalu_hypre_CSRMatrix         *GGT           = NULL;
   //    NALU_HYPRE_Int               *AAT_i         = NULL;
   //    NALU_HYPRE_Int               *AAT_j         = NULL;
   NALU_HYPRE_Int               A_nnz          = nalu_hypre_CSRMatrixNumNonzeros(A);
   nalu_hypre_CSRMatrix         *G             = NULL;
   NALU_HYPRE_Int               *G_i           = NULL;
   NALU_HYPRE_Int               *G_j           = NULL;
   NALU_HYPRE_Real              *G_data           = NULL;
   NALU_HYPRE_Int               *G_perm        = NULL;
   NALU_HYPRE_Int               G_nnz;
   NALU_HYPRE_Int               G_capacity;
   NALU_HYPRE_Int               *perm_temp     = NULL;
   NALU_HYPRE_Int               *perm          = *permp;
   NALU_HYPRE_Int               *qperm         = *qpermp;
   NALU_HYPRE_Int               *rqperm        = NULL;

   /* 1: Preprosessing
    * Check error in input, set some parameters
    */
   if (num_nodes <= 0)
   {
      /* don't do this if we are too small */
      return nalu_hypre_error_flag;
   }
   if (n != ncol || end > n || start < 0)
   {
      /* don't do this if the input has error */
      nalu_hypre_printf("Error input, abort RCM\n");
      return nalu_hypre_error_flag;
   }
   if (!perm)
   {
      /* create permutation array if we don't have one yet */
      perm = nalu_hypre_TAlloc( NALU_HYPRE_Int, n, NALU_HYPRE_MEMORY_DEVICE);
      for (i = 0 ; i < n ; i ++)
      {
         perm[i] = i;
      }
   }
   if (!qperm)
   {
      /* symmetric reordering, just point it to row reordering */
      qperm = perm;
   }
   rqperm = nalu_hypre_TAlloc(NALU_HYPRE_Int, n, NALU_HYPRE_MEMORY_HOST);
   for (i = 0 ; i < n ; i ++)
   {
      rqperm[qperm[i]] = i;
   }
   /* 2: Build Graph
    * Build Graph for RCM ordering
    */
   G = nalu_hypre_CSRMatrixCreate(num_nodes, num_nodes, 0);
   nalu_hypre_CSRMatrixInitialize(G);
   nalu_hypre_CSRMatrixSetDataOwner(G, 1);
   G_i = nalu_hypre_CSRMatrixI(G);
   if (sym)
   {
      /* Directly use A */
      G_nnz = 0;
      G_capacity = nalu_hypre_max(A_nnz * n * n / num_nodes / num_nodes - num_nodes, 1);
      G_j = nalu_hypre_TAlloc(NALU_HYPRE_Int, G_capacity, NALU_HYPRE_MEMORY_DEVICE);
      for (i = 0 ; i < num_nodes ; i ++)
      {
         G_i[i] = G_nnz;
         row = perm[i + start];
         r1 = A_i[row];
         r2 = A_i[row + 1];
         for (j = r1 ; j < r2 ; j ++)
         {
            col = rqperm[A_j[j]];
            if (col != row && col >= start && col < end)
            {
               /* this is an entry in G */
               G_j[G_nnz++] = col - start;
               if (G_nnz >= G_capacity)
               {
                  NALU_HYPRE_Int tmp = G_capacity;
                  G_capacity = G_capacity * EXPAND_FACT + 1;
                  G_j = nalu_hypre_TReAlloc_v2(G_j, NALU_HYPRE_Int, tmp, NALU_HYPRE_Int, G_capacity, NALU_HYPRE_MEMORY_DEVICE);
               }
            }
         }
      }
      G_i[num_nodes] = G_nnz;
      if (G_nnz == 0)
      {
         //G has only diagonal, no need to do any kind of RCM
         nalu_hypre_TFree(G_j, NALU_HYPRE_MEMORY_DEVICE);
         nalu_hypre_TFree(rqperm, NALU_HYPRE_MEMORY_HOST);
         *permp   = perm;
         *qpermp  = qperm;
         nalu_hypre_CSRMatrixDestroy(G);
         return nalu_hypre_error_flag;
      }
      nalu_hypre_CSRMatrixJ(G) = G_j;
      nalu_hypre_CSRMatrixNumNonzeros(G) = G_nnz;
   }
   else
   {
      /* Use A + A' */
      G_nnz = 0;
      G_capacity = nalu_hypre_max(A_nnz * n * n / num_nodes / num_nodes - num_nodes, 1);
      G_j = nalu_hypre_TAlloc(NALU_HYPRE_Int, G_capacity, NALU_HYPRE_MEMORY_DEVICE);
      for (i = 0 ; i < num_nodes ; i ++)
      {
         G_i[i] = G_nnz;
         row = perm[i + start];
         r1 = A_i[row];
         r2 = A_i[row + 1];
         for (j = r1 ; j < r2 ; j ++)
         {
            col = rqperm[A_j[j]];
            if (col != row && col >= start && col < end)
            {
               /* this is an entry in G */
               G_j[G_nnz++] = col - start;
               if (G_nnz >= G_capacity)
               {
                  NALU_HYPRE_Int tmp = G_capacity;
                  G_capacity = G_capacity * EXPAND_FACT + 1;
                  G_j = nalu_hypre_TReAlloc_v2(G_j, NALU_HYPRE_Int, tmp, NALU_HYPRE_Int, G_capacity, NALU_HYPRE_MEMORY_DEVICE);
               }
            }
         }
      }
      G_i[num_nodes] = G_nnz;
      if (G_nnz == 0)
      {
         //G has only diagonal, no need to do any kind of RCM
         nalu_hypre_TFree(G_j, NALU_HYPRE_MEMORY_DEVICE);
         nalu_hypre_TFree(rqperm, NALU_HYPRE_MEMORY_HOST);
         *permp   = perm;
         *qpermp  = qperm;
         nalu_hypre_CSRMatrixDestroy(G);
         return nalu_hypre_error_flag;
      }
      nalu_hypre_CSRMatrixJ(G) = G_j;
      G_data = nalu_hypre_CTAlloc(NALU_HYPRE_Real, G_nnz, NALU_HYPRE_MEMORY_DEVICE);
      nalu_hypre_CSRMatrixData(G) = G_data;
      nalu_hypre_CSRMatrixNumNonzeros(G) = G_nnz;

      /* now sum G with G' */
      nalu_hypre_CSRMatrixTranspose(G, &GT, 1);
      GGT = nalu_hypre_CSRMatrixAdd(1.0, G, 1.0, GT);
      nalu_hypre_CSRMatrixDestroy(G);
      nalu_hypre_CSRMatrixDestroy(GT);
      G = GGT;
      GGT = NULL;
   }

   /* 3: Build Graph
    * Build RCM
    */
   /* no need to be shared, but perm should be shared */
   G_perm = nalu_hypre_TAlloc(NALU_HYPRE_Int, num_nodes, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_ILULocalRCMOrder( G, G_perm);

   /* 4: Post processing
    * Free, set value, return
    */

   /* update to new index */
   perm_temp = nalu_hypre_TAlloc(NALU_HYPRE_Int, num_nodes, NALU_HYPRE_MEMORY_HOST);
   for ( i = 0 ; i < num_nodes ; i ++)
   {
      perm_temp[i] = perm[i + start];
   }
   for ( i = 0 ; i < num_nodes ; i ++)
   {
      perm[i + start] = perm_temp[G_perm[i]];
   }
   if (perm != qperm)
   {
      for ( i = 0 ; i < num_nodes ; i ++)
      {
         perm_temp[i] = qperm[i + start];
      }
      for ( i = 0 ; i < num_nodes ; i ++)
      {
         qperm[i + start] = perm_temp[G_perm[i]];
      }
   }
   *permp   = perm;
   *qpermp  = qperm;
   nalu_hypre_CSRMatrixDestroy(G);

   nalu_hypre_TFree(G_perm, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(perm_temp, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(rqperm, NALU_HYPRE_MEMORY_HOST);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ILULocalRCMMindegree
 *--------------------------------------------------------------------------*/

/* This function finds the unvisited node with the minimum degree
 */
NALU_HYPRE_Int
nalu_hypre_ILULocalRCMMindegree(NALU_HYPRE_Int n, NALU_HYPRE_Int *degree, NALU_HYPRE_Int *marker, NALU_HYPRE_Int *rootp)
{
   NALU_HYPRE_Int i;
   NALU_HYPRE_Int min_degree = n + 1;
   NALU_HYPRE_Int root = 0;
   for (i = 0 ; i < n ; i ++)
   {
      if (marker[i] < 0)
      {
         if (degree[i] < min_degree)
         {
            root = i;
            min_degree = degree[i];
         }
      }
   }
   *rootp = root;
   return 0;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ILULocalRCMOrder
 *--------------------------------------------------------------------------*/

/* This function actually does the RCM ordering of a symmetric csr matrix (entire)
 * A: the csr matrix, A_data is not needed
 * perm: the permutation array, space should be allocated outside
 */
NALU_HYPRE_Int
nalu_hypre_ILULocalRCMOrder( nalu_hypre_CSRMatrix *A, NALU_HYPRE_Int *perm)
{
   NALU_HYPRE_Int      i, root;
   NALU_HYPRE_Int      *degree     = NULL;
   NALU_HYPRE_Int      *marker     = NULL;
   NALU_HYPRE_Int      *A_i        = nalu_hypre_CSRMatrixI(A);
   NALU_HYPRE_Int      n           = nalu_hypre_CSRMatrixNumRows(A);
   NALU_HYPRE_Int      current_num;
   /* get the degree for each node */
   degree = nalu_hypre_TAlloc(NALU_HYPRE_Int, n, NALU_HYPRE_MEMORY_HOST);
   marker = nalu_hypre_TAlloc(NALU_HYPRE_Int, n, NALU_HYPRE_MEMORY_HOST);
   for (i = 0 ; i < n ; i ++)
   {
      degree[i] = A_i[i + 1] - A_i[i];
      marker[i] = -1;
   }

   /* start RCM loop */
   current_num = 0;
   while (current_num < n)
   {
      nalu_hypre_ILULocalRCMMindegree( n, degree, marker, &root);
      /* This is a new connect component */
      nalu_hypre_ILULocalRCMFindPPNode(A, &root, marker);

      /* Numbering of this component */
      nalu_hypre_ILULocalRCMNumbering(A, root, marker, perm, &current_num);
   }

   /* free */
   nalu_hypre_TFree(degree, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(marker, NALU_HYPRE_MEMORY_HOST);
   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ILULocalRCMFindPPNode
 *--------------------------------------------------------------------------*/

/* This function find a pseudo-peripheral node start from root
 * A: the csr matrix, A_data is not needed
 * rootp: pointer to the root, on return will be a end of the pseudo-peripheral
 * marker: the marker array for unvisited node
 */
NALU_HYPRE_Int
nalu_hypre_ILULocalRCMFindPPNode( nalu_hypre_CSRMatrix *A, NALU_HYPRE_Int *rootp, NALU_HYPRE_Int *marker)
{
   NALU_HYPRE_Int      i, r1, r2, row, min_degree, lev_degree, nlev, newnlev;

   NALU_HYPRE_Int      root           = *rootp;
   NALU_HYPRE_Int      n              = nalu_hypre_CSRMatrixNumRows(A);
   NALU_HYPRE_Int      *A_i           = nalu_hypre_CSRMatrixI(A);
   /* at most n levels */
   NALU_HYPRE_Int      *level_i       = nalu_hypre_TAlloc(NALU_HYPRE_Int, n + 1, NALU_HYPRE_MEMORY_HOST);
   NALU_HYPRE_Int      *level_j       = nalu_hypre_TAlloc(NALU_HYPRE_Int, n, NALU_HYPRE_MEMORY_HOST);

   /* build initial level structure from root */
   nalu_hypre_ILULocalRCMBuildLevel( A, root, marker, level_i, level_j, &newnlev);

   nlev  = newnlev - 1;
   while (nlev < newnlev)
   {
      nlev = newnlev;
      r1 =  level_i[nlev - 1];
      r2 =  level_i[nlev];
      min_degree = n;
      for (i = r1 ; i < r2 ; i ++)
      {
         /* select the last level, pick min-degree node */
         row = level_j[i];
         lev_degree = A_i[row + 1] - A_i[row];
         if (min_degree > lev_degree)
         {
            min_degree = lev_degree;
            root = row;
         }
      }
      nalu_hypre_ILULocalRCMBuildLevel( A, root, marker, level_i, level_j, &newnlev);
   }

   *rootp = root;
   /* free */
   nalu_hypre_TFree(level_i, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(level_j, NALU_HYPRE_MEMORY_HOST);
   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ILULocalRCMBuildLevel
 *--------------------------------------------------------------------------*/

/* This function build level structure start from root
 * A: the csr matrix, A_data is not needed
 * root: pointer to the root
 * marker: the marker array for unvisited node
 * level_i: points to the start/end of position on level_j, similar to CSR Matrix
 * level_j: store node number on each level
 * nlevp: return the number of level on this level structure
 */
NALU_HYPRE_Int
nalu_hypre_ILULocalRCMBuildLevel(nalu_hypre_CSRMatrix *A, NALU_HYPRE_Int root, NALU_HYPRE_Int *marker,
                            NALU_HYPRE_Int *level_i, NALU_HYPRE_Int *level_j, NALU_HYPRE_Int *nlevp)
{
   NALU_HYPRE_Int      i, j, l1, l2, l_current, r1, r2, rowi, rowj, nlev;
   NALU_HYPRE_Int      *A_i = nalu_hypre_CSRMatrixI(A);
   NALU_HYPRE_Int      *A_j = nalu_hypre_CSRMatrixJ(A);

   /* set first level first */
   level_i[0] = 0;
   level_j[0] = root;
   marker[root] = 0;
   nlev = 1;
   l1 = 0;
   l2 = 1;
   l_current = l2;

   //explore nbhds of all nodes in current level
   while (l2 > l1)
   {
      level_i[nlev++] = l2;
      /* loop through last level */
      for (i = l1 ; i < l2 ; i ++)
      {
         /* the node to explore */
         rowi = level_j[i];
         r1 = A_i[rowi];
         r2 = A_i[rowi + 1];
         for (j = r1 ; j < r2 ; j ++)
         {
            rowj = A_j[j];
            if ( marker[rowj] < 0 )
            {
               /* Aha, an unmarked row */
               marker[rowj] = 0;
               level_j[l_current++] = rowj;
            }
         }
      }
      l1 = l2;
      l2 = l_current;
   }
   /* after this we always have a "ghost" last level */
   nlev --;

   /* reset marker */
   for (i = 0 ; i < l2 ; i ++)
   {
      marker[level_j[i]] = -1;
   }

   *nlevp = nlev;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ILULocalRCMNumbering
 *--------------------------------------------------------------------------*/

/* This function generate numbering for a connect component
 * A: the csr matrix, A_data is not needed
 * root: pointer to the root
 * marker: the marker array for unvisited node
 * perm: permutation array
 * current_nump: number of nodes already have a perm value
 */

NALU_HYPRE_Int
nalu_hypre_ILULocalRCMNumbering(nalu_hypre_CSRMatrix *A, NALU_HYPRE_Int root, NALU_HYPRE_Int *marker, NALU_HYPRE_Int *perm,
                           NALU_HYPRE_Int *current_nump)
{
   NALU_HYPRE_Int        i, j, l1, l2, r1, r2, rowi, rowj, row_start, row_end;
   NALU_HYPRE_Int        *A_i        = nalu_hypre_CSRMatrixI(A);
   NALU_HYPRE_Int        *A_j        = nalu_hypre_CSRMatrixJ(A);
   NALU_HYPRE_Int        current_num = *current_nump;


   marker[root]        = 0;
   l1                  = current_num;
   perm[current_num++] = root;
   l2                  = current_num;

   while (l2 > l1)
   {
      /* loop through all nodes is current level */
      for (i = l1 ; i < l2 ; i ++)
      {
         rowi = perm[i];
         r1 = A_i[rowi];
         r2 = A_i[rowi + 1];
         row_start = current_num;
         for (j = r1 ; j < r2 ; j ++)
         {
            rowj = A_j[j];
            if (marker[rowj] < 0)
            {
               /* save the degree in marker and add it to perm */
               marker[rowj] = A_i[rowj + 1] - A_i[rowj];
               perm[current_num++] = rowj;
            }
         }
         row_end = current_num;
         nalu_hypre_ILULocalRCMQsort(perm, row_start, row_end - 1, marker);
      }
      l1 = l2;
      l2 = current_num;
   }

   //reverse
   nalu_hypre_ILULocalRCMReverse(perm, *current_nump, current_num - 1);
   *current_nump = current_num;
   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ILULocalRCMQsort
 *--------------------------------------------------------------------------*/

/* This qsort is very specialized, not worth to put into utilities
 * Sort a part of array perm based on degree value (ascend)
 * That is, if degree[perm[i]] < degree[perm[j]], we should have i < j
 * perm: the perm array
 * start: start in perm
 * end: end in perm
 * degree: degree array
 */

NALU_HYPRE_Int
nalu_hypre_ILULocalRCMQsort(NALU_HYPRE_Int *perm, NALU_HYPRE_Int start, NALU_HYPRE_Int end, NALU_HYPRE_Int *degree)
{

   NALU_HYPRE_Int i, mid;
   if (start >= end)
   {
      return nalu_hypre_error_flag;
   }

   nalu_hypre_swap(perm, start, (start + end) / 2);
   mid = start;
   //loop to split
   for (i = start + 1 ; i <= end ; i ++)
   {
      if (degree[perm[i]] < degree[perm[start]])
      {
         nalu_hypre_swap(perm, ++mid, i);
      }
   }
   nalu_hypre_swap(perm, start, mid);
   nalu_hypre_ILULocalRCMQsort(perm, mid + 1, end, degree);
   nalu_hypre_ILULocalRCMQsort(perm, start, mid - 1, degree);
   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ILULocalRCMReverse
 *--------------------------------------------------------------------------*/

/* Last step in RCM, reverse it
 * perm: perm array
 * srart: start position
 * end: end position
 */

NALU_HYPRE_Int
nalu_hypre_ILULocalRCMReverse(NALU_HYPRE_Int *perm, NALU_HYPRE_Int start, NALU_HYPRE_Int end)
{
   NALU_HYPRE_Int     i, j;
   NALU_HYPRE_Int     mid = (start + end + 1) / 2;

   for (i = start, j = end ; i < mid ; i ++, j--)
   {
      nalu_hypre_swap(perm, i, j);
   }
   return nalu_hypre_error_flag;
}

#if defined(NALU_HYPRE_USING_CUDA) && defined(NALU_HYPRE_USING_CUSPARSE)

/*--------------------------------------------------------------------------
 * nalu_hypre_ParILUCusparseSchurGMRESDummySetup
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ParILUCusparseSchurGMRESDummySetup(void *a, void *b, void *c, void *d)
{
   /* Null GMRES setup, does nothing */
   return 0;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ParILUCusparseSchurGMRESDummySolve
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ParILUCusparseSchurGMRESDummySolve( void               *ilu_vdata,
                                          void               *ilu_vdata2,
                                          nalu_hypre_ParVector    *f,
                                          nalu_hypre_ParVector    *u )
{
   /* Unit GMRES preconditioner, just copy data from one slot to another */
   nalu_hypre_ParILUData *ilu_data                = (nalu_hypre_ParILUData*) ilu_vdata;
   nalu_hypre_ParCSRMatrix *A                     = nalu_hypre_ParILUDataMatS(ilu_data);
   nalu_hypre_CSRMatrix         *A_diag           = nalu_hypre_ParCSRMatrixDiag(A);
   NALU_HYPRE_Int               n_local           = nalu_hypre_CSRMatrixNumRows(A_diag);

   nalu_hypre_Vector            *u_local          = nalu_hypre_ParVectorLocalVector(u);
   NALU_HYPRE_Real              *u_data           = nalu_hypre_VectorData(u_local);

   nalu_hypre_Vector            *f_local          = nalu_hypre_ParVectorLocalVector(f);
   NALU_HYPRE_Real              *f_data           = nalu_hypre_VectorData(f_local);

   cudaDeviceSynchronize();
   nalu_hypre_TMemcpy(u_data, f_data, NALU_HYPRE_Real, n_local, NALU_HYPRE_MEMORY_DEVICE, NALU_HYPRE_MEMORY_DEVICE);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ParILUCusparseSchurGMRESCommInfo
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ParILUCusparseSchurGMRESCommInfo( void *ilu_vdata, NALU_HYPRE_Int *my_id, NALU_HYPRE_Int *num_procs)
{
   /* get comm info from ilu_data */
   nalu_hypre_ParILUData *ilu_data = (nalu_hypre_ParILUData*) ilu_vdata;
   nalu_hypre_ParCSRMatrix *A = nalu_hypre_ParILUDataMatS(ilu_data);
   MPI_Comm comm = nalu_hypre_ParCSRMatrixComm ( A );
   nalu_hypre_MPI_Comm_size(comm, num_procs);
   nalu_hypre_MPI_Comm_rank(comm, my_id);
   return 0;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ParILUCusparseSchurGMRESMatvecCreate
 *--------------------------------------------------------------------------*/

void *
nalu_hypre_ParILUCusparseSchurGMRESMatvecCreate( void   *ilu_vdata,
                                            void   *x )
{
   /* Null matvec create */
   void *matvec_data;
   matvec_data = NULL;
   return ( matvec_data );
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ParILUCusparseSchurGMRESMatvec
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ParILUCusparseSchurGMRESMatvec( void   *matvec_data,
                                      NALU_HYPRE_Complex  alpha,
                                      void   *ilu_vdata,
                                      void   *x,
                                      NALU_HYPRE_Complex  beta,
                                      void   *y           )
{
   /* Slightly different, for this new matvec, the diagonal of the original matrix
    * is the LU factorization. Thus, the matvec is done in an different way
    * |IS_1 E_12 E_13|
    * |E_21 IS_2 E_23| = S
    * |E_31 E_32 IS_3|
    *
    * |IS_1          |
    * |     IS_2     | = M
    * |          IS_3|
    *
    * Solve Sy = g is just M^{-1}S = M^{-1}g
    *
    * |      I       IS_1^{-1}E_12 IS_1^{-1}E_13|
    * |IS_2^{-1}E_21       I       IS_2^{-1}E_23| = M^{-1}S
    * |IS_3^{-1}E_31 IS_3^{-1}E_32       I      |
    *
    * */

   /* get matrix information first */
   nalu_hypre_ParILUData *ilu_data                   = (nalu_hypre_ParILUData*) ilu_vdata;
   nalu_hypre_ParCSRMatrix *A                        = nalu_hypre_ParILUDataMatS(ilu_data);

   /* fist step, apply matvec on empty diagonal slot */
   nalu_hypre_CSRMatrix   *A_diag                    = nalu_hypre_ParCSRMatrixDiag(A);
   NALU_HYPRE_Int         *A_diag_i                  = nalu_hypre_CSRMatrixI(A_diag);
   NALU_HYPRE_Int         *A_diag_j                  = nalu_hypre_CSRMatrixJ(A_diag);
   NALU_HYPRE_Real        *A_diag_data               = nalu_hypre_CSRMatrixData(A_diag);
   NALU_HYPRE_Int         A_diag_n                   = nalu_hypre_CSRMatrixNumRows(A_diag);
   NALU_HYPRE_Int         A_diag_nnz                 = A_diag_i[A_diag_n];
   NALU_HYPRE_Int         *A_diag_fake_i             = nalu_hypre_ParILUDataMatAFakeDiagonal(ilu_data);

   cusparseMatDescr_t      matL_des             = nalu_hypre_ParILUDataMatLMatrixDescription(ilu_data);
   cusparseMatDescr_t      matU_des             = nalu_hypre_ParILUDataMatUMatrixDescription(ilu_data);
   void                    *ilu_solve_buffer    = nalu_hypre_ParILUDataILUSolveBuffer(
                                                     ilu_data);//device memory
   cusparseSolvePolicy_t   ilu_solve_policy     = nalu_hypre_ParILUDataILUSolvePolicy(ilu_data);
   csrsv2Info_t            matSL_info           = nalu_hypre_ParILUDataMatSLILUSolveInfo(ilu_data);
   csrsv2Info_t            matSU_info           = nalu_hypre_ParILUDataMatSUILUSolveInfo(ilu_data);

   nalu_hypre_ParVector         *xtemp               = nalu_hypre_ParILUDataXTemp(ilu_data);
   nalu_hypre_Vector            *xtemp_local         = nalu_hypre_ParVectorLocalVector(xtemp);
   NALU_HYPRE_Real              *xtemp_data          = nalu_hypre_VectorData(xtemp_local);
   nalu_hypre_ParVector         *ytemp               = nalu_hypre_ParILUDataYTemp(ilu_data);
   nalu_hypre_Vector            *ytemp_local         = nalu_hypre_ParVectorLocalVector(ytemp);
   NALU_HYPRE_Real              *ytemp_data          = nalu_hypre_VectorData(ytemp_local);
   NALU_HYPRE_Real              zero                 = 0.0;
   NALU_HYPRE_Real              one                  = 1.0;

   cusparseHandle_t handle = nalu_hypre_HandleCusparseHandle(nalu_hypre_handle());

   /* Matvec with
    *         |  O  E_12 E_13|
    * alpha * |E_21   O  E_23|
    *         |E_31 E_32   O |
    * store in xtemp
    */
   nalu_hypre_CSRMatrixI(A_diag)                     = A_diag_fake_i;
   nalu_hypre_ParCSRMatrixMatvec( alpha, (nalu_hypre_ParCSRMatrix *) A, (nalu_hypre_ParVector *) x, zero, xtemp );
   nalu_hypre_CSRMatrixI(A_diag)                     = A_diag_i;

   /* Compute U^{-1}*L^{-1}*(A_offd * x)
    * Or in another word, matvec with
    *         |      O       IS_1^{-1}E_12 IS_1^{-1}E_13|
    * alpha * |IS_2^{-1}E_21       O       IS_2^{-1}E_23|
    *         |IS_3^{-1}E_31 IS_3^{-1}E_32       O      |
    * store in xtemp
    */
   if ( A_diag_n > 0 )
   {
      /* L solve - Forward solve */
      NALU_HYPRE_CUSPARSE_CALL(nalu_hypre_cusparse_csrsv2_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                      A_diag_n, A_diag_nnz, &one, matL_des,
                                                      A_diag_data, A_diag_i, A_diag_j, matSL_info,
                                                      xtemp_data, ytemp_data, ilu_solve_policy, ilu_solve_buffer));

      /* U solve - Backward substitution */
      NALU_HYPRE_CUSPARSE_CALL(nalu_hypre_cusparse_csrsv2_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                      A_diag_n, A_diag_nnz, &one, matU_des,
                                                      A_diag_data, A_diag_i, A_diag_j, matSU_info,
                                                      ytemp_data, xtemp_data, ilu_solve_policy, ilu_solve_buffer));
   }

   /* now add the original x onto it */
   nalu_hypre_ParVectorAxpy( alpha, (nalu_hypre_ParVector *) x, (nalu_hypre_ParVector *) xtemp);

   /* finall, add that into y and get final result */
   nalu_hypre_ParVectorScale( beta, (nalu_hypre_ParVector *) y );
   nalu_hypre_ParVectorAxpy( one, xtemp, (nalu_hypre_ParVector *) y);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ParILUCusparseSchurGMRESMatvecDestroy
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ParILUCusparseSchurGMRESMatvecDestroy( void *matvec_data )
{
   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ParILUCusparseSchurGMRESDummySetup
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ParILURAPSchurGMRESDummySetup(void *a, void *b, void *c, void *d)
{
   /* Null GMRES setup, does nothing */
   return 0;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ParILUCusparseSchurGMRESDummySolve
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ParILURAPSchurGMRESSolve( void               *ilu_vdata,
                                void               *ilu_vdata2,
                                nalu_hypre_ParVector    *f,
                                nalu_hypre_ParVector    *u )
{
   /* Unit GMRES preconditioner, just copy data from one slot to another */
   nalu_hypre_ParILUData        *ilu_data            = (nalu_hypre_ParILUData*) ilu_vdata;
   //nalu_hypre_ParCSRMatrix      *Aperm               = nalu_hypre_ParILUDataAperm(ilu_data);

   cusparseHandle_t        handle               = nalu_hypre_HandleCusparseHandle(nalu_hypre_handle());
   cusparseMatDescr_t      matL_des             = nalu_hypre_ParILUDataMatLMatrixDescription(ilu_data);
   cusparseMatDescr_t      matU_des             = nalu_hypre_ParILUDataMatUMatrixDescription(ilu_data);
   void                    *ilu_solve_buffer    = nalu_hypre_ParILUDataILUSolveBuffer(
                                                     ilu_data);//device memory
   cusparseSolvePolicy_t   ilu_solve_policy     = nalu_hypre_ParILUDataILUSolvePolicy(ilu_data);

   nalu_hypre_ParCSRMatrix      *S                   = nalu_hypre_ParILUDataMatS(ilu_data);
   nalu_hypre_CSRMatrix         *SLU                 = nalu_hypre_ParCSRMatrixDiag(S);
   NALU_HYPRE_Int               *SLU_i               = nalu_hypre_CSRMatrixI(SLU);
   NALU_HYPRE_Int               *SLU_j               = nalu_hypre_CSRMatrixJ(SLU);
   NALU_HYPRE_Real              *SLU_data            = nalu_hypre_CSRMatrixData(SLU);
   NALU_HYPRE_Int               SLU_nnz              = nalu_hypre_CSRMatrixNumNonzeros(SLU);

   csrsv2Info_t            matSL_info           = nalu_hypre_ParILUDataMatSLILUSolveInfo(ilu_data);
   csrsv2Info_t            matSU_info           = nalu_hypre_ParILUDataMatSUILUSolveInfo(ilu_data);

   //NALU_HYPRE_Int               n                    = nalu_hypre_CSRMatrixNumRows(nalu_hypre_ParCSRMatrixDiag(Aperm));
   NALU_HYPRE_Int               m                    = nalu_hypre_CSRMatrixNumRows(SLU);
   //NALU_HYPRE_Int               nLU                  = n - m;

   nalu_hypre_ParVector         *f_vec               = (nalu_hypre_ParVector *) f;
   nalu_hypre_Vector            *f_local             = nalu_hypre_ParVectorLocalVector(f_vec);
   NALU_HYPRE_Real              *f_data              = nalu_hypre_VectorData(f_local);
   nalu_hypre_ParVector         *u_vec               = (nalu_hypre_ParVector *) u;
   nalu_hypre_Vector            *u_local             = nalu_hypre_ParVectorLocalVector(u_vec);
   NALU_HYPRE_Real              *u_data              = nalu_hypre_VectorData(u_local);
   nalu_hypre_ParVector         *rhs                 = nalu_hypre_ParILUDataRhs(ilu_data);
   nalu_hypre_Vector            *rhs_local           = nalu_hypre_ParVectorLocalVector(rhs);
   NALU_HYPRE_Real              *rhs_data            = nalu_hypre_VectorData(rhs_local);

   //NALU_HYPRE_Real zero = 0.0;
   NALU_HYPRE_Real one = 1.0;
   //NALU_HYPRE_Real mone = -1.0;

   if (m > 0)
   {
      /* L solve */
      NALU_HYPRE_CUSPARSE_CALL(nalu_hypre_cusparse_csrsv2_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                      m, SLU_nnz, &one, matL_des,
                                                      SLU_data, SLU_i, SLU_j, matSL_info,
                                                      f_data, rhs_data, ilu_solve_policy, ilu_solve_buffer));
      /* U solve */
      NALU_HYPRE_CUSPARSE_CALL(nalu_hypre_cusparse_csrsv2_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                      m, SLU_nnz, &one, matU_des,
                                                      SLU_data, SLU_i, SLU_j, matSU_info,
                                                      rhs_data, u_data, ilu_solve_policy, ilu_solve_buffer));
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ParILUCusparseSchurGMRESMatvecCreate
 *--------------------------------------------------------------------------*/

void *
nalu_hypre_ParILURAPSchurGMRESMatvecCreate( void   *ilu_vdata,
                                       void   *x )
{
   /* Null matvec create */
   void *matvec_data;
   matvec_data = NULL;
   return ( matvec_data );
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ParILUCusparseSchurGMRESMatvec
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ParILURAPSchurGMRESMatvec( void   *matvec_data,
                                 NALU_HYPRE_Complex  alpha,
                                 void   *ilu_vdata,
                                 void   *x,
                                 NALU_HYPRE_Complex  beta,
                                 void   *y           )
{
   /* Compute y = alpha * S * x + beta * y
    * */

   /* get matrix information first */
   nalu_hypre_ParILUData        *ilu_data            = (nalu_hypre_ParILUData*) ilu_vdata;

   NALU_HYPRE_Int               test_opt             = nalu_hypre_ParILUDataTestOption(ilu_data);

   switch (test_opt)
   {
      case 1:
      {
         /* S = R * A * P */
         nalu_hypre_ParCSRMatrix      *Aperm               = nalu_hypre_ParILUDataAperm(ilu_data);

         cusparseHandle_t        handle               = nalu_hypre_HandleCusparseHandle(nalu_hypre_handle());
         cusparseMatDescr_t      matL_des             = nalu_hypre_ParILUDataMatLMatrixDescription(ilu_data);
         cusparseMatDescr_t      matU_des             = nalu_hypre_ParILUDataMatUMatrixDescription(ilu_data);
         void                    *ilu_solve_buffer    = nalu_hypre_ParILUDataILUSolveBuffer(
                                                           ilu_data);//device memory
         cusparseSolvePolicy_t   ilu_solve_policy     = nalu_hypre_ParILUDataILUSolvePolicy(ilu_data);

         nalu_hypre_CSRMatrix         *EiU                 = nalu_hypre_ParILUDataMatEDevice(ilu_data);
         nalu_hypre_CSRMatrix         *iLF                 = nalu_hypre_ParILUDataMatFDevice(ilu_data);
         nalu_hypre_CSRMatrix         *BLU                 = nalu_hypre_ParILUDataMatBILUDevice(ilu_data);
         NALU_HYPRE_Int               *BLU_i               = nalu_hypre_CSRMatrixI(BLU);
         NALU_HYPRE_Int               *BLU_j               = nalu_hypre_CSRMatrixJ(BLU);
         NALU_HYPRE_Real              *BLU_data            = nalu_hypre_CSRMatrixData(BLU);
         NALU_HYPRE_Int               BLU_nnz              = nalu_hypre_CSRMatrixNumNonzeros(BLU);

         csrsv2Info_t            matBL_info           = nalu_hypre_ParILUDataMatBLILUSolveInfo(ilu_data);
         csrsv2Info_t            matBU_info           = nalu_hypre_ParILUDataMatBUILUSolveInfo(ilu_data);

         NALU_HYPRE_Int               n                    = nalu_hypre_CSRMatrixNumRows(nalu_hypre_ParCSRMatrixDiag(
                                                                                  Aperm));
         NALU_HYPRE_Int               nLU                  = nalu_hypre_CSRMatrixNumRows(BLU);
         NALU_HYPRE_Int               m                    = n - nLU;

         nalu_hypre_ParVector         *x_vec               = (nalu_hypre_ParVector *) x;
         nalu_hypre_Vector            *x_local             = nalu_hypre_ParVectorLocalVector(x_vec);
         NALU_HYPRE_Real              *x_data              = nalu_hypre_VectorData(x_local);
         nalu_hypre_ParVector         *y_vec               = (nalu_hypre_ParVector *) y;
         nalu_hypre_Vector            *y_local             = nalu_hypre_ParVectorLocalVector(y_vec);
         //NALU_HYPRE_Real              *y_data              = nalu_hypre_VectorData(y_local);
         nalu_hypre_ParVector         *xtemp               = nalu_hypre_ParILUDataUTemp(ilu_data);
         nalu_hypre_Vector            *xtemp_local         = nalu_hypre_ParVectorLocalVector(xtemp);
         NALU_HYPRE_Real              *xtemp_data          = nalu_hypre_VectorData(xtemp_local);
         nalu_hypre_ParVector         *ytemp               = nalu_hypre_ParILUDataYTemp(ilu_data);
         nalu_hypre_Vector            *ytemp_local         = nalu_hypre_ParVectorLocalVector(ytemp);
         NALU_HYPRE_Real              *ytemp_data          = nalu_hypre_VectorData(ytemp_local);

         nalu_hypre_Vector *xtemp_upper           = nalu_hypre_SeqVectorCreate(nLU);
         nalu_hypre_Vector *ytemp_upper           = nalu_hypre_SeqVectorCreate(nLU);
         nalu_hypre_Vector *xtemp_lower           = nalu_hypre_SeqVectorCreate(m);
         nalu_hypre_VectorOwnsData(xtemp_upper)   = 0;
         nalu_hypre_VectorOwnsData(ytemp_upper)   = 0;
         nalu_hypre_VectorOwnsData(xtemp_lower)   = 0;
         nalu_hypre_VectorData(xtemp_upper)       = xtemp_data;
         nalu_hypre_VectorData(ytemp_upper)       = ytemp_data;
         nalu_hypre_VectorData(xtemp_lower)       = xtemp_data + nLU;
         nalu_hypre_SeqVectorInitialize(xtemp_upper);
         nalu_hypre_SeqVectorInitialize(ytemp_upper);
         nalu_hypre_SeqVectorInitialize(xtemp_lower);

         NALU_HYPRE_Real zero = 0.0;
         NALU_HYPRE_Real one = 1.0;
         NALU_HYPRE_Real mone = -1.0;

         /* first step, compute P*x put in y */
         /* -Fx */
         nalu_hypre_CSRMatrixMatvec(mone, iLF, x_local, zero, ytemp_upper);
         /* -L^{-1}Fx */
         /* L solve */
         NALU_HYPRE_CUSPARSE_CALL(nalu_hypre_cusparse_csrsv2_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                         nLU, BLU_nnz, &one, matL_des,
                                                         BLU_data, BLU_i, BLU_j, matBL_info,
                                                         ytemp_data, xtemp_data, ilu_solve_policy, ilu_solve_buffer));

         /* -U{-1}L^{-1}Fx */
         /* U solve */
         NALU_HYPRE_CUSPARSE_CALL(nalu_hypre_cusparse_csrsv2_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                         nLU, BLU_nnz, &one, matU_des,
                                                         BLU_data, BLU_i, BLU_j, matBU_info,
                                                         xtemp_data, ytemp_data, ilu_solve_policy, ilu_solve_buffer));

         /* now copy data to y_lower */
         nalu_hypre_TMemcpy( ytemp_data + nLU, x_data, NALU_HYPRE_Real, m, NALU_HYPRE_MEMORY_DEVICE, NALU_HYPRE_MEMORY_DEVICE);

         /* second step, compute A*P*x store in xtemp */
         nalu_hypre_ParCSRMatrixMatvec( one, Aperm, ytemp, zero, xtemp);

         /* third step, compute R*A*P*x */
         /* solve L^{-1} */
         /* L solve */
         NALU_HYPRE_CUSPARSE_CALL(nalu_hypre_cusparse_csrsv2_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                         nLU, BLU_nnz, &one, matL_des,
                                                         BLU_data, BLU_i, BLU_j, matBL_info,
                                                         xtemp_data, ytemp_data, ilu_solve_policy, ilu_solve_buffer));

         /* U^{-1}L^{-1} */
         /* U solve */
         NALU_HYPRE_CUSPARSE_CALL(nalu_hypre_cusparse_csrsv2_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                         nLU, BLU_nnz, &one, matU_des,
                                                         BLU_data, BLU_i, BLU_j, matBU_info,
                                                         ytemp_data, xtemp_data, ilu_solve_policy, ilu_solve_buffer));

         /* -EU^{-1}L^{-1} */
         nalu_hypre_CSRMatrixMatvec(mone * alpha, EiU, xtemp_upper, beta, y_local);
         /* I*lower-EU^{-1}L^{-1}*upper */
         nalu_hypre_SeqVectorAxpy(alpha, xtemp_lower, y_local);

         /* over */
         nalu_hypre_SeqVectorDestroy(xtemp_upper);
         nalu_hypre_SeqVectorDestroy(ytemp_upper);
         nalu_hypre_SeqVectorDestroy(xtemp_lower);
      }
      break;
      case 2:
      {
         /* S = C - EU^{-1} * L^{-1}F */

         //nalu_hypre_ParCSRMatrix      *Aperm               = nalu_hypre_ParILUDataAperm(ilu_data);

         //cusparseHandle_t        handle               = nalu_hypre_HandleCusparseHandle(nalu_hypre_handle());
         //cusparseMatDescr_t      matL_des             = nalu_hypre_ParILUDataMatLMatrixDescription(ilu_data);
         //cusparseMatDescr_t      matU_des             = nalu_hypre_ParILUDataMatUMatrixDescription(ilu_data);
         //void                    *ilu_solve_buffer    = nalu_hypre_ParILUDataILUSolveBuffer(ilu_data);//device memory
         //cusparseSolvePolicy_t   ilu_solve_policy     = nalu_hypre_ParILUDataILUSolvePolicy(ilu_data);

         nalu_hypre_CSRMatrix         *EiU                 = nalu_hypre_ParILUDataMatEDevice(ilu_data);
         nalu_hypre_CSRMatrix         *iLF                 = nalu_hypre_ParILUDataMatFDevice(ilu_data);
         nalu_hypre_CSRMatrix         *C                   = nalu_hypre_ParILUDataMatSILUDevice(ilu_data);

         //NALU_HYPRE_Int               n                    = nalu_hypre_CSRMatrixNumRows(nalu_hypre_ParCSRMatrixDiag(Aperm));
         NALU_HYPRE_Int               nLU                  = nalu_hypre_CSRMatrixNumRows(C);
         //NALU_HYPRE_Int               m                    = n - nLU;

         nalu_hypre_ParVector         *x_vec               = (nalu_hypre_ParVector *) x;
         nalu_hypre_Vector            *x_local             = nalu_hypre_ParVectorLocalVector(x_vec);
         //NALU_HYPRE_Real              *x_data              = nalu_hypre_VectorData(x_local);
         nalu_hypre_ParVector         *y_vec               = (nalu_hypre_ParVector *) y;
         nalu_hypre_Vector            *y_local             = nalu_hypre_ParVectorLocalVector(y_vec);
         //NALU_HYPRE_Real              *y_data              = nalu_hypre_VectorData(y_local);
         nalu_hypre_ParVector         *xtemp               = nalu_hypre_ParILUDataUTemp(ilu_data);
         nalu_hypre_Vector            *xtemp_local         = nalu_hypre_ParVectorLocalVector(xtemp);
         NALU_HYPRE_Real              *xtemp_data          = nalu_hypre_VectorData(xtemp_local);

         nalu_hypre_Vector *xtemp_upper           = nalu_hypre_SeqVectorCreate(nLU);
         nalu_hypre_VectorOwnsData(xtemp_upper)   = 0;
         nalu_hypre_VectorData(xtemp_upper)       = xtemp_data;
         nalu_hypre_SeqVectorInitialize(xtemp_upper);

         NALU_HYPRE_Real zero = 0.0;
         NALU_HYPRE_Real one = 1.0;
         NALU_HYPRE_Real mone = -1.0;

         /* first step, compute EB^{-1}F*x put in y */
         /* -L^{-1}Fx */
         nalu_hypre_CSRMatrixMatvec(mone, iLF, x_local, zero, xtemp_upper);
         /* - alpha EU^{-1}L^{-1}Fx + beta * y */
         nalu_hypre_CSRMatrixMatvec( alpha, EiU, xtemp_upper, beta, y_local);
         /* alpha * C - alpha EU^{-1}L^{-1}Fx + beta y */
         nalu_hypre_CSRMatrixMatvec( alpha, C, x_local, one, y_local);

         /* over */
         nalu_hypre_SeqVectorDestroy(xtemp_upper);
      }
      break;
      case 3:
      {
         /* S = C - EU^{-1} * L^{-1}F */

         //nalu_hypre_ParCSRMatrix      *Aperm               = nalu_hypre_ParILUDataAperm(ilu_data);

         cusparseHandle_t        handle               = nalu_hypre_HandleCusparseHandle(nalu_hypre_handle());
         cusparseMatDescr_t      matL_des             = nalu_hypre_ParILUDataMatLMatrixDescription(ilu_data);
         cusparseMatDescr_t      matU_des             = nalu_hypre_ParILUDataMatUMatrixDescription(ilu_data);
         void                    *ilu_solve_buffer    = nalu_hypre_ParILUDataILUSolveBuffer(
                                                           ilu_data);//device memory
         cusparseSolvePolicy_t   ilu_solve_policy     = nalu_hypre_ParILUDataILUSolvePolicy(ilu_data);

         nalu_hypre_CSRMatrix         *EiU                 = nalu_hypre_ParILUDataMatEDevice(ilu_data);
         nalu_hypre_CSRMatrix         *iLF                 = nalu_hypre_ParILUDataMatFDevice(ilu_data);
         nalu_hypre_CSRMatrix         *C                   = nalu_hypre_ParILUDataMatSILUDevice(ilu_data);
         nalu_hypre_CSRMatrix         *BLU                 = nalu_hypre_ParILUDataMatBILUDevice(ilu_data);
         NALU_HYPRE_Int               *BLU_i               = nalu_hypre_CSRMatrixI(BLU);
         NALU_HYPRE_Int               *BLU_j               = nalu_hypre_CSRMatrixJ(BLU);
         NALU_HYPRE_Real              *BLU_data            = nalu_hypre_CSRMatrixData(BLU);
         NALU_HYPRE_Int               BLU_nnz              = nalu_hypre_CSRMatrixNumNonzeros(BLU);

         csrsv2Info_t            matBL_info           = nalu_hypre_ParILUDataMatBLILUSolveInfo(ilu_data);
         csrsv2Info_t            matBU_info           = nalu_hypre_ParILUDataMatBUILUSolveInfo(ilu_data);

         //NALU_HYPRE_Int               n                    = nalu_hypre_CSRMatrixNumRows(nalu_hypre_ParCSRMatrixDiag(Aperm));
         NALU_HYPRE_Int               nLU                  = nalu_hypre_CSRMatrixNumRows(C);
         //NALU_HYPRE_Int               m                    = n - nLU;

         nalu_hypre_ParVector         *x_vec               = (nalu_hypre_ParVector *) x;
         nalu_hypre_Vector            *x_local             = nalu_hypre_ParVectorLocalVector(x_vec);
         //NALU_HYPRE_Real              *x_data              = nalu_hypre_VectorData(x_local);
         nalu_hypre_ParVector         *y_vec               = (nalu_hypre_ParVector *) y;
         nalu_hypre_Vector            *y_local             = nalu_hypre_ParVectorLocalVector(y_vec);
         //NALU_HYPRE_Real              *y_data              = nalu_hypre_VectorData(y_local);
         nalu_hypre_ParVector         *xtemp               = nalu_hypre_ParILUDataUTemp(ilu_data);
         nalu_hypre_Vector            *xtemp_local         = nalu_hypre_ParVectorLocalVector(xtemp);
         NALU_HYPRE_Real              *xtemp_data          = nalu_hypre_VectorData(xtemp_local);
         nalu_hypre_ParVector         *ytemp               = nalu_hypre_ParILUDataYTemp(ilu_data);
         nalu_hypre_Vector            *ytemp_local         = nalu_hypre_ParVectorLocalVector(ytemp);
         NALU_HYPRE_Real              *ytemp_data          = nalu_hypre_VectorData(ytemp_local);

         nalu_hypre_Vector *xtemp_upper           = nalu_hypre_SeqVectorCreate(nLU);
         nalu_hypre_VectorOwnsData(xtemp_upper)   = 0;
         nalu_hypre_VectorData(xtemp_upper)       = xtemp_data;
         nalu_hypre_SeqVectorInitialize(xtemp_upper);

         NALU_HYPRE_Real zero = 0.0;
         NALU_HYPRE_Real one = 1.0;
         NALU_HYPRE_Real mone = -1.0;

         /* first step, compute EB^{-1}F*x put in y */
         /* -Fx */
         nalu_hypre_CSRMatrixMatvec(mone, iLF, x_local, zero, xtemp_upper);
         /* -L^{-1}Fx */
         /* L solve */
         NALU_HYPRE_CUSPARSE_CALL(nalu_hypre_cusparse_csrsv2_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                         nLU, BLU_nnz, &one, matL_des,
                                                         BLU_data, BLU_i, BLU_j, matBL_info,
                                                         xtemp_data, ytemp_data, ilu_solve_policy, ilu_solve_buffer));
         /* -U^{-1}L^{-1}Fx */
         /* U solve */
         NALU_HYPRE_CUSPARSE_CALL(nalu_hypre_cusparse_csrsv2_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                         nLU, BLU_nnz, &one, matU_des,
                                                         BLU_data, BLU_i, BLU_j, matBU_info,
                                                         ytemp_data, xtemp_data, ilu_solve_policy, ilu_solve_buffer));

         /* - alpha EU^{-1}L^{-1}Fx + beta * y */
         nalu_hypre_CSRMatrixMatvec( alpha, EiU, xtemp_upper, beta, y_local);
         /* alpha * C - alpha EU^{-1}L^{-1}Fx + beta y */
         nalu_hypre_CSRMatrixMatvec( alpha, C, x_local, one, y_local);

         /* over */
         nalu_hypre_SeqVectorDestroy(xtemp_upper);
      }
      break;
   case 0: default:
      {
         /* S = R * A * P */

         nalu_hypre_ParCSRMatrix      *Aperm               = nalu_hypre_ParILUDataAperm(ilu_data);

         cusparseHandle_t        handle               = nalu_hypre_HandleCusparseHandle(nalu_hypre_handle());
         cusparseMatDescr_t      matL_des             = nalu_hypre_ParILUDataMatLMatrixDescription(ilu_data);
         cusparseMatDescr_t      matU_des             = nalu_hypre_ParILUDataMatUMatrixDescription(ilu_data);
         void                    *ilu_solve_buffer    = nalu_hypre_ParILUDataILUSolveBuffer(
                                                           ilu_data);//device memory
         cusparseSolvePolicy_t   ilu_solve_policy     = nalu_hypre_ParILUDataILUSolvePolicy(ilu_data);

         nalu_hypre_CSRMatrix         *EiU                 = nalu_hypre_ParILUDataMatEDevice(ilu_data);
         nalu_hypre_CSRMatrix         *iLF                 = nalu_hypre_ParILUDataMatFDevice(ilu_data);
         nalu_hypre_CSRMatrix         *BLU                 = nalu_hypre_ParILUDataMatBILUDevice(ilu_data);
         NALU_HYPRE_Int               *BLU_i               = nalu_hypre_CSRMatrixI(BLU);
         NALU_HYPRE_Int               *BLU_j               = nalu_hypre_CSRMatrixJ(BLU);
         NALU_HYPRE_Real              *BLU_data            = nalu_hypre_CSRMatrixData(BLU);
         NALU_HYPRE_Int               BLU_nnz              = nalu_hypre_CSRMatrixNumNonzeros(BLU);

         csrsv2Info_t            matBL_info           = nalu_hypre_ParILUDataMatBLILUSolveInfo(ilu_data);
         csrsv2Info_t            matBU_info           = nalu_hypre_ParILUDataMatBUILUSolveInfo(ilu_data);

         NALU_HYPRE_Int               n                    = nalu_hypre_CSRMatrixNumRows(nalu_hypre_ParCSRMatrixDiag(
                                                                                  Aperm));
         NALU_HYPRE_Int               nLU                  = nalu_hypre_CSRMatrixNumRows(BLU);
         NALU_HYPRE_Int               m                    = n - nLU;

         nalu_hypre_ParVector         *x_vec               = (nalu_hypre_ParVector *) x;
         nalu_hypre_Vector            *x_local             = nalu_hypre_ParVectorLocalVector(x_vec);
         NALU_HYPRE_Real              *x_data              = nalu_hypre_VectorData(x_local);
         nalu_hypre_ParVector         *y_vec               = (nalu_hypre_ParVector *) y;
         nalu_hypre_Vector            *y_local             = nalu_hypre_ParVectorLocalVector(y_vec);
         //NALU_HYPRE_Real              *y_data              = nalu_hypre_VectorData(y_local);
         nalu_hypre_ParVector         *xtemp               = nalu_hypre_ParILUDataUTemp(ilu_data);
         nalu_hypre_Vector            *xtemp_local         = nalu_hypre_ParVectorLocalVector(xtemp);
         NALU_HYPRE_Real              *xtemp_data          = nalu_hypre_VectorData(xtemp_local);
         nalu_hypre_ParVector         *ytemp               = nalu_hypre_ParILUDataYTemp(ilu_data);
         nalu_hypre_Vector            *ytemp_local         = nalu_hypre_ParVectorLocalVector(ytemp);
         NALU_HYPRE_Real              *ytemp_data          = nalu_hypre_VectorData(ytemp_local);

         nalu_hypre_Vector *xtemp_upper           = nalu_hypre_SeqVectorCreate(nLU);
         nalu_hypre_Vector *ytemp_upper           = nalu_hypre_SeqVectorCreate(nLU);
         nalu_hypre_Vector *ytemp_lower           = nalu_hypre_SeqVectorCreate(m);
         nalu_hypre_VectorOwnsData(xtemp_upper)   = 0;
         nalu_hypre_VectorOwnsData(ytemp_upper)   = 0;
         nalu_hypre_VectorOwnsData(ytemp_lower)   = 0;
         nalu_hypre_VectorData(xtemp_upper)       = xtemp_data;
         nalu_hypre_VectorData(ytemp_upper)       = ytemp_data;
         nalu_hypre_VectorData(ytemp_lower)       = ytemp_data + nLU;
         nalu_hypre_SeqVectorInitialize(xtemp_upper);
         nalu_hypre_SeqVectorInitialize(ytemp_upper);
         nalu_hypre_SeqVectorInitialize(ytemp_lower);

         NALU_HYPRE_Real zero = 0.0;
         NALU_HYPRE_Real one = 1.0;
         NALU_HYPRE_Real mone = -1.0;

         /* first step, compute P*x put in y */
         /* -L^{-1}Fx */
         nalu_hypre_CSRMatrixMatvec(mone, iLF, x_local, zero, xtemp_upper);

         /* -U{-1}L^{-1}Fx */
         if (nLU > 0)
         {
            /* U solve */
            NALU_HYPRE_CUSPARSE_CALL(nalu_hypre_cusparse_csrsv2_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                            nLU, BLU_nnz, &one, matU_des,
                                                            BLU_data, BLU_i, BLU_j, matBU_info,
                                                            xtemp_data, ytemp_data, ilu_solve_policy, ilu_solve_buffer));
         }
         /* now copy data to y_lower */
         nalu_hypre_TMemcpy( ytemp_data + nLU, x_data, NALU_HYPRE_Real, m, NALU_HYPRE_MEMORY_DEVICE, NALU_HYPRE_MEMORY_DEVICE);

         /* second step, compute A*P*x store in xtemp */
         nalu_hypre_ParCSRMatrixMatvec( one, Aperm, ytemp, zero, xtemp);

         /* third step, compute R*A*P*x */
         /* copy partial data in */
         nalu_hypre_TMemcpy( ytemp_data + nLU, xtemp_data + nLU, NALU_HYPRE_Real, m, NALU_HYPRE_MEMORY_DEVICE,
                        NALU_HYPRE_MEMORY_DEVICE);

         /* solve L^{-1} */
         if (nLU > 0)
         {
            /* L solve */
            NALU_HYPRE_CUSPARSE_CALL(nalu_hypre_cusparse_csrsv2_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                            nLU, BLU_nnz, &one, matL_des,
                                                            BLU_data, BLU_i, BLU_j, matBL_info,
                                                            xtemp_data, ytemp_data, ilu_solve_policy, ilu_solve_buffer));
         }
         /* -EU^{-1}L^{-1} */
         nalu_hypre_CSRMatrixMatvec(mone * alpha, EiU, ytemp_upper, beta, y_local);
         nalu_hypre_SeqVectorAxpy(alpha, ytemp_lower, y_local);

         /* over */
         nalu_hypre_SeqVectorDestroy(xtemp_upper);
         nalu_hypre_SeqVectorDestroy(ytemp_upper);
         nalu_hypre_SeqVectorDestroy(ytemp_lower);
      }
      break;
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ParILUCusparseSchurGMRESMatvecDestroy
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ParILURAPSchurGMRESMatvecDestroy( void *matvec_data )
{
   return 0;
}

#else

NALU_HYPRE_Int
nalu_hypre_ParILURAPSchurGMRESDummySetupH(void *a, void *b, void *c, void *d)
{
   /* Null GMRES setup, does nothing */
   return 0;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ParILUCusparseSchurGMRESDummySolve
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ParILURAPSchurGMRESSolveH( void               *ilu_vdata,
                                 void               *ilu_vdata2,
                                 nalu_hypre_ParVector    *f,
                                 nalu_hypre_ParVector    *u )
{
   /* Unit GMRES preconditioner, just copy data from one slot to another */
   nalu_hypre_ParILUData        *ilu_data            = (nalu_hypre_ParILUData*) ilu_vdata;

   NALU_HYPRE_Int               i, j, k1, k2, col;

   nalu_hypre_ParCSRMatrix      *L                   = nalu_hypre_ParILUDataMatLModified(ilu_data);
   nalu_hypre_CSRMatrix         *L_diag              = nalu_hypre_ParCSRMatrixDiag(L);
   NALU_HYPRE_Int               *L_diag_i            = nalu_hypre_CSRMatrixI(L_diag);
   NALU_HYPRE_Int               *L_diag_j            = nalu_hypre_CSRMatrixJ(L_diag);
   NALU_HYPRE_Real              *L_diag_data         = nalu_hypre_CSRMatrixData(L_diag);
   NALU_HYPRE_Real              *D                   = nalu_hypre_ParILUDataMatDModified(ilu_data);
   nalu_hypre_ParCSRMatrix      *U                   = nalu_hypre_ParILUDataMatUModified(ilu_data);
   nalu_hypre_CSRMatrix         *U_diag              = nalu_hypre_ParCSRMatrixDiag(U);
   NALU_HYPRE_Int               *U_diag_i            = nalu_hypre_CSRMatrixI(U_diag);
   NALU_HYPRE_Int               *U_diag_j            = nalu_hypre_CSRMatrixJ(U_diag);
   NALU_HYPRE_Real              *U_diag_data         = nalu_hypre_CSRMatrixData(U_diag);

   NALU_HYPRE_Int               n                    = nalu_hypre_CSRMatrixNumRows(nalu_hypre_ParCSRMatrixDiag(L));
   NALU_HYPRE_Int               nLU                  = nalu_hypre_ParILUDataNLU(ilu_data);
   NALU_HYPRE_Int               m                    = n - nLU;

   nalu_hypre_Vector            *f_local             = nalu_hypre_ParVectorLocalVector(f);
   NALU_HYPRE_Real              *f_data              = nalu_hypre_VectorData(f_local);
   nalu_hypre_Vector            *u_local             = nalu_hypre_ParVectorLocalVector(u);
   NALU_HYPRE_Real              *u_data              = nalu_hypre_VectorData(u_local);

   nalu_hypre_ParVector         *utemp               = nalu_hypre_ParILUDataUTemp(ilu_data);
   nalu_hypre_Vector            *utemp_local         = nalu_hypre_ParVectorLocalVector(utemp);
   NALU_HYPRE_Real              *utemp_data          = nalu_hypre_VectorData(utemp_local);

   NALU_HYPRE_Int               *u_end               = nalu_hypre_ParILUDataUEnd(ilu_data);

   /* permuted L solve */
   for (i = 0 ; i < m ; i ++)
   {
      utemp_data[i] = f_data[i];
      k1 = u_end[i + nLU] ; k2 = L_diag_i[i + nLU + 1];
      for (j = k1 ; j < k2 ; j ++)
      {
         col = L_diag_j[j];
         utemp_data[i] -= L_diag_data[j] * utemp_data[col - nLU];
      }
   }

   /* U solve */
   for (i = m - 1 ; i >= 0 ; i --)
   {
      u_data[i] = utemp_data[i];
      k1 = U_diag_i[i + nLU] ; k2 = U_diag_i[i + 1 + nLU];
      for (j = k1 ; j < k2 ; j ++)
      {
         col = U_diag_j[j];
         u_data[i] -= U_diag_data[j] * u_data[col - nLU];
      }
      u_data[i] *= D[i];
   }


   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ParILUCusparseSchurGMRESCommInfo
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ParILURAPSchurGMRESCommInfoH( void *ilu_vdata, NALU_HYPRE_Int *my_id, NALU_HYPRE_Int *num_procs)
{
   /* get comm info from ilu_data */
   nalu_hypre_ParILUData *ilu_data = (nalu_hypre_ParILUData*) ilu_vdata;
   nalu_hypre_ParCSRMatrix *A = nalu_hypre_ParILUDataMatA(ilu_data);
   MPI_Comm comm = nalu_hypre_ParCSRMatrixComm ( A );
   nalu_hypre_MPI_Comm_size(comm, num_procs);
   nalu_hypre_MPI_Comm_rank(comm, my_id);
   return 0;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ParILUCusparseSchurGMRESMatvecCreate
 *--------------------------------------------------------------------------*/

void *
nalu_hypre_ParILURAPSchurGMRESMatvecCreateH( void   *ilu_vdata,
                                        void   *x )
{
   /* Null matvec create */
   void *matvec_data;
   matvec_data = NULL;
   return ( matvec_data );
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ParILUCusparseSchurGMRESMatvec
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ParILURAPSchurGMRESMatvecH( void   *matvec_data,
                                  NALU_HYPRE_Complex  alpha,
                                  void   *ilu_vdata,
                                  void   *x,
                                  NALU_HYPRE_Complex  beta,
                                  void   *y           )
{
   /* Compute y = alpha * S * x + beta * y
    * */
   NALU_HYPRE_Int               i, j, k1, k2, col;
   NALU_HYPRE_Real              one = 1.0;
   NALU_HYPRE_Real              zero = 0.0;

   /* get matrix information first */
   nalu_hypre_ParILUData        *ilu_data            = (nalu_hypre_ParILUData*) ilu_vdata;

   /* only option 1, use W and Z */
   NALU_HYPRE_Int               *u_end               = nalu_hypre_ParILUDataUEnd(ilu_data);
   nalu_hypre_ParCSRMatrix      *A                   = nalu_hypre_ParILUDataMatA(ilu_data);
   nalu_hypre_ParCSRMatrix      *mL                  = nalu_hypre_ParILUDataMatLModified(ilu_data);
   NALU_HYPRE_Real              *mD                  = nalu_hypre_ParILUDataMatDModified(ilu_data);
   nalu_hypre_ParCSRMatrix      *mU                  = nalu_hypre_ParILUDataMatUModified(ilu_data);

   nalu_hypre_CSRMatrix         *mL_diag             = nalu_hypre_ParCSRMatrixDiag(mL);
   NALU_HYPRE_Int               *mL_diag_i           = nalu_hypre_CSRMatrixI(mL_diag);
   NALU_HYPRE_Int               *mL_diag_j           = nalu_hypre_CSRMatrixJ(mL_diag);
   NALU_HYPRE_Real              *mL_diag_data        = nalu_hypre_CSRMatrixData(mL_diag);

   nalu_hypre_CSRMatrix         *mU_diag             = nalu_hypre_ParCSRMatrixDiag(mU);
   NALU_HYPRE_Int               *mU_diag_i           = nalu_hypre_CSRMatrixI(mU_diag);
   NALU_HYPRE_Int               *mU_diag_j           = nalu_hypre_CSRMatrixJ(mU_diag);
   NALU_HYPRE_Real              *mU_diag_data        = nalu_hypre_CSRMatrixData(mU_diag);

   NALU_HYPRE_Int               *perm                = nalu_hypre_ParILUDataPerm(ilu_data);
   NALU_HYPRE_Int               n                    = nalu_hypre_ParCSRMatrixNumRows(A);
   NALU_HYPRE_Int               nLU                  = nalu_hypre_ParILUDataNLU(ilu_data);

   nalu_hypre_ParVector         *x_vec               = (nalu_hypre_ParVector *) x;
   nalu_hypre_Vector            *x_local             = nalu_hypre_ParVectorLocalVector(x_vec);
   NALU_HYPRE_Real              *x_data              = nalu_hypre_VectorData(x_local);
   nalu_hypre_ParVector         *y_vec               = (nalu_hypre_ParVector *) y;
   nalu_hypre_Vector            *y_local             = nalu_hypre_ParVectorLocalVector(y_vec);
   NALU_HYPRE_Real              *y_data              = nalu_hypre_VectorData(y_local);

   nalu_hypre_ParVector   *utemp       = nalu_hypre_ParILUDataUTemp(ilu_data);
   nalu_hypre_Vector      *utemp_local = nalu_hypre_ParVectorLocalVector(utemp);
   NALU_HYPRE_Real        *utemp_data  = nalu_hypre_VectorData(utemp_local);

   nalu_hypre_ParVector   *ftemp       = nalu_hypre_ParILUDataFTemp(ilu_data);
   nalu_hypre_Vector      *ftemp_local = nalu_hypre_ParVectorLocalVector(ftemp);
   NALU_HYPRE_Real        *ftemp_data  = nalu_hypre_VectorData(ftemp_local);

   nalu_hypre_ParVector   *ytemp       = nalu_hypre_ParILUDataYTemp(ilu_data);
   nalu_hypre_Vector      *ytemp_local = nalu_hypre_ParVectorLocalVector(ytemp);
   NALU_HYPRE_Real        *ytemp_data  = nalu_hypre_VectorData(ytemp_local);

   /* S = R * A * P */
   /* matvec */
   /* first compute alpha * P * x
    * P = [ -U\inv U_12 ]
    *     [  I          ]
    */
   /* matvec */
   for (i = 0 ; i < nLU ; i ++)
   {
      ytemp_data[i] = 0.0;
      k1 = u_end[i] ; k2 = mU_diag_i[i + 1];
      for (j = k1 ; j < k2 ; j ++)
      {
         col = mU_diag_j[j];
         ytemp_data[i] -= alpha * mU_diag_data[j] * x_data[col - nLU];
      }
   }
   /* U solve */
   for (i = nLU - 1 ; i >= 0 ; i --)
   {
      ftemp_data[perm[i]] = ytemp_data[i];
      k1 = mU_diag_i[i] ; k2 = u_end[i];
      for (j = k1 ; j < k2 ; j ++)
      {
         col = mU_diag_j[j];
         ftemp_data[perm[i]] -= mU_diag_data[j] * ftemp_data[perm[col]];
      }
      ftemp_data[perm[i]] *= mD[i];
   }

   /* update with I */
   for (i = nLU ; i < n ; i ++)
   {
      ftemp_data[perm[i]] = alpha * x_data[i - nLU];
   }

   /* apply alpha*A*P*x */
   nalu_hypre_ParCSRMatrixMatvec(one, A, ftemp, zero, utemp);

   // R = [-L21 L\inv, I]

   /* first is L solve */
   for (i = 0 ; i < nLU ; i ++)
   {
      ytemp_data[i] = utemp_data[perm[i]];
      k1 = mL_diag_i[i] ; k2 = mL_diag_i[i + 1];
      for (j = k1 ; j < k2 ; j ++)
      {
         col = mL_diag_j[j];
         ytemp_data[i] -= mL_diag_data[j] * ytemp_data[col];
      }
   }

   /* apply -W * utemp on this, and take care of the I part */
   for (i = nLU ; i < n ; i ++)
   {
      y_data[i - nLU] = beta * y_data[i - nLU] + utemp_data[perm[i]];
      k1 = mL_diag_i[i] ; k2 = u_end[i];
      for (j = k1 ; j < k2 ; j ++)
      {
         col = mL_diag_j[j];
         y_data[i - nLU] -= mL_diag_data[j] * ytemp_data[col];
      }
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ParILUCusparseSchurGMRESMatvecDestroy
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ParILURAPSchurGMRESMatvecDestroyH( void *matvec_data )
{
   return 0;
}

#endif /* if defined(NALU_HYPRE_USING_CUDA) && defined(NALU_HYPRE_USING_CUSPARSE) */

/* NSH create and solve and help functions */

/* Create */
void *
nalu_hypre_NSHCreate()
{
   nalu_hypre_ParNSHData  *nsh_data;

   nsh_data = nalu_hypre_CTAlloc(nalu_hypre_ParNSHData,  1, NALU_HYPRE_MEMORY_HOST);

   /* general data */
   nalu_hypre_ParNSHDataMatA(nsh_data)                  = NULL;
   nalu_hypre_ParNSHDataMatM(nsh_data)                  = NULL;
   nalu_hypre_ParNSHDataF(nsh_data)                     = NULL;
   nalu_hypre_ParNSHDataU(nsh_data)                     = NULL;
   nalu_hypre_ParNSHDataResidual(nsh_data)              = NULL;
   nalu_hypre_ParNSHDataRelResNorms(nsh_data)           = NULL;
   nalu_hypre_ParNSHDataNumIterations(nsh_data)         = 0;
   nalu_hypre_ParNSHDataL1Norms(nsh_data)               = NULL;
   nalu_hypre_ParNSHDataFinalRelResidualNorm(nsh_data)  = 0.0;
   nalu_hypre_ParNSHDataTol(nsh_data)                   = 1e-09;
   nalu_hypre_ParNSHDataLogging(nsh_data)               = 2;
   nalu_hypre_ParNSHDataPrintLevel(nsh_data)            = 2;
   nalu_hypre_ParNSHDataMaxIter(nsh_data)               = 5;

   nalu_hypre_ParNSHDataOperatorComplexity(nsh_data)    = 0.0;
   nalu_hypre_ParNSHDataDroptol(nsh_data)               = nalu_hypre_TAlloc(NALU_HYPRE_Real, 2, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_ParNSHDataOwnDroptolData(nsh_data)        = 1;
   nalu_hypre_ParNSHDataDroptol(nsh_data)[0]            = 1.0e-02;/* droptol for MR */
   nalu_hypre_ParNSHDataDroptol(nsh_data)[1]            = 1.0e-02;/* droptol for NSH */
   nalu_hypre_ParNSHDataUTemp(nsh_data)                 = NULL;
   nalu_hypre_ParNSHDataFTemp(nsh_data)                 = NULL;

   /* MR data */
   nalu_hypre_ParNSHDataMRMaxIter(nsh_data)             = 2;
   nalu_hypre_ParNSHDataMRTol(nsh_data)                 = 1e-09;
   nalu_hypre_ParNSHDataMRMaxRowNnz(nsh_data)           = 800;
   nalu_hypre_ParNSHDataMRColVersion(nsh_data)          = 0;

   /* NSH data */
   nalu_hypre_ParNSHDataNSHMaxIter(nsh_data)            = 2;
   nalu_hypre_ParNSHDataNSHTol(nsh_data)                = 1e-09;
   nalu_hypre_ParNSHDataNSHMaxRowNnz(nsh_data)          = 1000;

   return (void *) nsh_data;
}

/* Destroy */
NALU_HYPRE_Int
nalu_hypre_NSHDestroy( void *data )
{
   nalu_hypre_ParNSHData * nsh_data = (nalu_hypre_ParNSHData*) data;

   /* residual */
   if (nalu_hypre_ParNSHDataResidual(nsh_data))
   {
      nalu_hypre_ParVectorDestroy( nalu_hypre_ParNSHDataResidual(nsh_data) );
      nalu_hypre_ParNSHDataResidual(nsh_data) = NULL;
   }

   /* residual norms */
   if (nalu_hypre_ParNSHDataRelResNorms(nsh_data))
   {
      nalu_hypre_TFree( nalu_hypre_ParNSHDataRelResNorms(nsh_data), NALU_HYPRE_MEMORY_HOST );
      nalu_hypre_ParNSHDataRelResNorms(nsh_data) = NULL;
   }

   /* l1 norms */
   if (nalu_hypre_ParNSHDataL1Norms(nsh_data))
   {
      nalu_hypre_TFree( nalu_hypre_ParNSHDataL1Norms(nsh_data), NALU_HYPRE_MEMORY_HOST );
      nalu_hypre_ParNSHDataL1Norms(nsh_data) = NULL;
   }

   /* temp arrays */
   if (nalu_hypre_ParNSHDataUTemp(nsh_data))
   {
      nalu_hypre_ParVectorDestroy( nalu_hypre_ParNSHDataUTemp(nsh_data) );
      nalu_hypre_ParNSHDataUTemp(nsh_data) = NULL;
   }
   if (nalu_hypre_ParNSHDataFTemp(nsh_data))
   {
      nalu_hypre_ParVectorDestroy( nalu_hypre_ParNSHDataFTemp(nsh_data) );
      nalu_hypre_ParNSHDataFTemp(nsh_data) = NULL;
   }

   /* approx inverse matrix */
   if (nalu_hypre_ParNSHDataMatM(nsh_data))
   {
      nalu_hypre_ParCSRMatrixDestroy( nalu_hypre_ParNSHDataMatM(nsh_data) );
      nalu_hypre_ParNSHDataMatM(nsh_data) = NULL;
   }

   /* droptol array */
   if (nalu_hypre_ParNSHDataOwnDroptolData(nsh_data))
   {
      nalu_hypre_TFree(nalu_hypre_ParNSHDataDroptol(nsh_data), NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_ParNSHDataOwnDroptolData(nsh_data) = 0;
      nalu_hypre_ParNSHDataDroptol(nsh_data) = NULL;
   }

   /* nsh data */
   nalu_hypre_TFree(nsh_data, NALU_HYPRE_MEMORY_HOST);

   return nalu_hypre_error_flag;
}

/* Print solver params */
NALU_HYPRE_Int
nalu_hypre_NSHWriteSolverParams(void *nsh_vdata)
{
   nalu_hypre_ParNSHData  *nsh_data = (nalu_hypre_ParNSHData*) nsh_vdata;
   nalu_hypre_printf("Newton–Schulz–Hotelling Setup parameters: \n");
   nalu_hypre_printf("NSH max iterations = %d \n", nalu_hypre_ParNSHDataNSHMaxIter(nsh_data));
   nalu_hypre_printf("NSH drop tolerance = %e \n", nalu_hypre_ParNSHDataDroptol(nsh_data)[1]);
   nalu_hypre_printf("NSH max nnz per row = %d \n", nalu_hypre_ParNSHDataNSHMaxRowNnz(nsh_data));
   nalu_hypre_printf("MR max iterations = %d \n", nalu_hypre_ParNSHDataMRMaxIter(nsh_data));
   nalu_hypre_printf("MR drop tolerance = %e \n", nalu_hypre_ParNSHDataDroptol(nsh_data)[0]);
   nalu_hypre_printf("MR max nnz per row = %d \n", nalu_hypre_ParNSHDataMRMaxRowNnz(nsh_data));
   nalu_hypre_printf("Operator Complexity (Fill factor) = %f \n",
                nalu_hypre_ParNSHDataOperatorComplexity(nsh_data));
   nalu_hypre_printf("\n Newton–Schulz–Hotelling Solver Parameters: \n");
   nalu_hypre_printf("Max number of iterations: %d\n", nalu_hypre_ParNSHDataMaxIter(nsh_data));
   nalu_hypre_printf("Stopping tolerance: %e\n", nalu_hypre_ParNSHDataTol(nsh_data));

   return nalu_hypre_error_flag;
}

/* set print level */
NALU_HYPRE_Int
nalu_hypre_NSHSetPrintLevel( void *nsh_vdata, NALU_HYPRE_Int print_level )
{
   nalu_hypre_ParNSHData   *nsh_data = (nalu_hypre_ParNSHData*) nsh_vdata;
   nalu_hypre_ParNSHDataPrintLevel(nsh_data) = print_level;
   return nalu_hypre_error_flag;
}
/* set logging level */
NALU_HYPRE_Int
nalu_hypre_NSHSetLogging( void *nsh_vdata, NALU_HYPRE_Int logging )
{
   nalu_hypre_ParNSHData   *nsh_data = (nalu_hypre_ParNSHData*) nsh_vdata;
   nalu_hypre_ParNSHDataLogging(nsh_data) = logging;
   return nalu_hypre_error_flag;
}
/* set max iteration */
NALU_HYPRE_Int
nalu_hypre_NSHSetMaxIter( void *nsh_vdata, NALU_HYPRE_Int max_iter )
{
   nalu_hypre_ParNSHData   *nsh_data = (nalu_hypre_ParNSHData*) nsh_vdata;
   nalu_hypre_ParNSHDataMaxIter(nsh_data) = max_iter;
   return nalu_hypre_error_flag;
}
/* set solver iteration tol */
NALU_HYPRE_Int
nalu_hypre_NSHSetTol( void *nsh_vdata, NALU_HYPRE_Real tol )
{
   nalu_hypre_ParNSHData   *nsh_data = (nalu_hypre_ParNSHData*) nsh_vdata;
   nalu_hypre_ParNSHDataTol(nsh_data) = tol;
   return nalu_hypre_error_flag;
}
/* set global solver */
NALU_HYPRE_Int
nalu_hypre_NSHSetGlobalSolver( void *nsh_vdata, NALU_HYPRE_Int global_solver )
{
   nalu_hypre_ParNSHData   *nsh_data = (nalu_hypre_ParNSHData*) nsh_vdata;
   nalu_hypre_ParNSHDataGlobalSolver(nsh_data) = global_solver;
   return nalu_hypre_error_flag;
}
/* set all droptols */
NALU_HYPRE_Int
nalu_hypre_NSHSetDropThreshold( void *nsh_vdata, NALU_HYPRE_Real droptol )
{
   nalu_hypre_ParNSHData   *nsh_data = (nalu_hypre_ParNSHData*) nsh_vdata;
   nalu_hypre_ParNSHDataDroptol(nsh_data)[0] = droptol;
   nalu_hypre_ParNSHDataDroptol(nsh_data)[1] = droptol;
   return nalu_hypre_error_flag;
}
/* set array of droptols */
NALU_HYPRE_Int
nalu_hypre_NSHSetDropThresholdArray( void *nsh_vdata, NALU_HYPRE_Real *droptol )
{
   nalu_hypre_ParNSHData   *nsh_data = (nalu_hypre_ParNSHData*) nsh_vdata;
   if (nalu_hypre_ParNSHDataOwnDroptolData(nsh_data))
   {
      nalu_hypre_TFree(nalu_hypre_ParNSHDataDroptol(nsh_data), NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_ParNSHDataOwnDroptolData(nsh_data) = 0;
   }
   nalu_hypre_ParNSHDataDroptol(nsh_data) = droptol;
   return nalu_hypre_error_flag;
}
/* set MR max iter */
NALU_HYPRE_Int
nalu_hypre_NSHSetMRMaxIter( void *nsh_vdata, NALU_HYPRE_Int mr_max_iter )
{
   nalu_hypre_ParNSHData   *nsh_data = (nalu_hypre_ParNSHData*) nsh_vdata;
   nalu_hypre_ParNSHDataMRMaxIter(nsh_data) = mr_max_iter;
   return nalu_hypre_error_flag;
}
/* set MR tol */
NALU_HYPRE_Int
nalu_hypre_NSHSetMRTol( void *nsh_vdata, NALU_HYPRE_Real mr_tol )
{
   nalu_hypre_ParNSHData   *nsh_data = (nalu_hypre_ParNSHData*) nsh_vdata;
   nalu_hypre_ParNSHDataMRTol(nsh_data) = mr_tol;
   return nalu_hypre_error_flag;
}
/* set MR max nonzeros of a row */
NALU_HYPRE_Int
nalu_hypre_NSHSetMRMaxRowNnz( void *nsh_vdata, NALU_HYPRE_Int mr_max_row_nnz )
{
   nalu_hypre_ParNSHData   *nsh_data = (nalu_hypre_ParNSHData*) nsh_vdata;
   nalu_hypre_ParNSHDataMRMaxRowNnz(nsh_data) = mr_max_row_nnz;
   return nalu_hypre_error_flag;
}
/* set MR version, column version or global version */
NALU_HYPRE_Int
nalu_hypre_NSHSetColVersion( void *nsh_vdata, NALU_HYPRE_Int mr_col_version )
{
   nalu_hypre_ParNSHData   *nsh_data = (nalu_hypre_ParNSHData*) nsh_vdata;
   nalu_hypre_ParNSHDataMRColVersion(nsh_data) = mr_col_version;
   return nalu_hypre_error_flag;
}
/* set NSH max iter */
NALU_HYPRE_Int
nalu_hypre_NSHSetNSHMaxIter( void *nsh_vdata, NALU_HYPRE_Int nsh_max_iter )
{
   nalu_hypre_ParNSHData   *nsh_data = (nalu_hypre_ParNSHData*) nsh_vdata;
   nalu_hypre_ParNSHDataNSHMaxIter(nsh_data) = nsh_max_iter;
   return nalu_hypre_error_flag;
}
/* set NSH tol */
NALU_HYPRE_Int
nalu_hypre_NSHSetNSHTol( void *nsh_vdata, NALU_HYPRE_Real nsh_tol )
{
   nalu_hypre_ParNSHData   *nsh_data = (nalu_hypre_ParNSHData*) nsh_vdata;
   nalu_hypre_ParNSHDataNSHTol(nsh_data) = nsh_tol;
   return nalu_hypre_error_flag;
}
/* set NSH max nonzeros of a row */
NALU_HYPRE_Int
nalu_hypre_NSHSetNSHMaxRowNnz( void *nsh_vdata, NALU_HYPRE_Int nsh_max_row_nnz )
{
   nalu_hypre_ParNSHData   *nsh_data = (nalu_hypre_ParNSHData*) nsh_vdata;
   nalu_hypre_ParNSHDataNSHMaxRowNnz(nsh_data) = nsh_max_row_nnz;
   return nalu_hypre_error_flag;
}


/* Compute the F norm of CSR matrix
 * A: the target CSR matrix
 * norm_io: output
 */
NALU_HYPRE_Int
nalu_hypre_CSRMatrixNormFro(nalu_hypre_CSRMatrix *A, NALU_HYPRE_Real *norm_io)
{
   NALU_HYPRE_Real norm = 0.0;
   NALU_HYPRE_Real *data = nalu_hypre_CSRMatrixData(A);
   NALU_HYPRE_Int i, k;
   k = nalu_hypre_CSRMatrixNumNonzeros(A);
   /* main loop */
   for (i = 0 ; i < k ; i ++)
   {
      norm += data[i] * data[i];
   }
   *norm_io = sqrt(norm);
   return nalu_hypre_error_flag;

}

/* Compute the norm of I-A where I is identity matrix and A is a CSR matrix
 * A: the target CSR matrix
 * norm_io: the output
 */
NALU_HYPRE_Int
nalu_hypre_CSRMatrixResNormFro(nalu_hypre_CSRMatrix *A, NALU_HYPRE_Real *norm_io)
{
   NALU_HYPRE_Real        norm = 0.0, value;
   NALU_HYPRE_Int         i, j, k1, k2, n;
   NALU_HYPRE_Int         *idx  = nalu_hypre_CSRMatrixI(A);
   NALU_HYPRE_Int         *cols = nalu_hypre_CSRMatrixJ(A);
   NALU_HYPRE_Real        *data = nalu_hypre_CSRMatrixData(A);

   n = nalu_hypre_CSRMatrixNumRows(A);
   /* main loop to sum up data */
   for (i = 0 ; i < n ; i ++)
   {
      k1 = idx[i];
      k2 = idx[i + 1];
      /* check if we have diagonal in A */
      if (k2 > k1)
      {
         if (cols[k1] == i)
         {
            /* reduce 1 on diagonal */
            value = data[k1] - 1.0;
            norm += value * value;
         }
         else
         {
            /* we don't have diagonal in A, so we need to add 1 to norm */
            norm += 1.0;
            norm += data[k1] * data[k1];
         }
      }
      else
      {
         /* we don't have diagonal in A, so we need to add 1 to norm */
         norm += 1.0;
      }
      /* and the rest of the code */
      for (j = k1 + 1 ; j < k2 ; j ++)
      {
         norm += data[j] * data[j];
      }
   }
   *norm_io = sqrt(norm);
   return nalu_hypre_error_flag;
}

/* Compute the F norm of ParCSR matrix
 * A: the target CSR matrix
 */
NALU_HYPRE_Int
nalu_hypre_ParCSRMatrixNormFro(nalu_hypre_ParCSRMatrix *A, NALU_HYPRE_Real *norm_io)
{
   NALU_HYPRE_Real        local_norm = 0.0;
   NALU_HYPRE_Real        global_norm;
   MPI_Comm          comm = nalu_hypre_ParCSRMatrixComm(A);

   nalu_hypre_CSRMatrix   *A_diag = nalu_hypre_ParCSRMatrixDiag(A);
   nalu_hypre_CSRMatrix   *A_offd = nalu_hypre_ParCSRMatrixOffd(A);

   nalu_hypre_CSRMatrixNormFro(A_diag, &local_norm);
   /* use global_norm to store offd for now */
   nalu_hypre_CSRMatrixNormFro(A_offd, &global_norm);

   /* square and sum them */
   local_norm *= local_norm;
   local_norm += global_norm * global_norm;

   /* do communication to get global total sum */
   nalu_hypre_MPI_Allreduce(&local_norm, &global_norm, 1, NALU_HYPRE_MPI_REAL, nalu_hypre_MPI_SUM, comm);

   *norm_io = sqrt(global_norm);
   return nalu_hypre_error_flag;

}

/* Compute the F norm of ParCSR matrix
 * Norm of I-A
 * A: the target CSR matrix
 */
NALU_HYPRE_Int
nalu_hypre_ParCSRMatrixResNormFro(nalu_hypre_ParCSRMatrix *A, NALU_HYPRE_Real *norm_io)
{
   NALU_HYPRE_Real        local_norm = 0.0;
   NALU_HYPRE_Real        global_norm;
   MPI_Comm          comm = nalu_hypre_ParCSRMatrixComm(A);

   nalu_hypre_CSRMatrix   *A_diag = nalu_hypre_ParCSRMatrixDiag(A);
   nalu_hypre_CSRMatrix   *A_offd = nalu_hypre_ParCSRMatrixOffd(A);

   /* compute I-A for diagonal */
   nalu_hypre_CSRMatrixResNormFro(A_diag, &local_norm);
   /* use global_norm to store offd for now */
   nalu_hypre_CSRMatrixNormFro(A_offd, &global_norm);

   /* square and sum them */
   local_norm *= local_norm;
   local_norm += global_norm * global_norm;

   /* do communication to get global total sum */
   nalu_hypre_MPI_Allreduce(&local_norm, &global_norm, 1, NALU_HYPRE_MPI_REAL, nalu_hypre_MPI_SUM, comm);

   *norm_io = sqrt(global_norm);
   return nalu_hypre_error_flag;

}

/* Compute the trace of CSR matrix
 * A: the target CSR matrix
 * trace_io: the output trace
 */
NALU_HYPRE_Int
nalu_hypre_CSRMatrixTrace(nalu_hypre_CSRMatrix *A, NALU_HYPRE_Real *trace_io)
{
   NALU_HYPRE_Real  trace = 0.0;
   NALU_HYPRE_Int   *idx = nalu_hypre_CSRMatrixI(A);
   NALU_HYPRE_Int   *cols = nalu_hypre_CSRMatrixJ(A);
   NALU_HYPRE_Real  *data = nalu_hypre_CSRMatrixData(A);
   NALU_HYPRE_Int i, k1, k2, n;
   n = nalu_hypre_CSRMatrixNumRows(A);
   for (i = 0 ; i < n ; i ++)
   {
      k1 = idx[i];
      k2 = idx[i + 1];
      if (cols[k1] == i && k2 > k1)
      {
         /* only add when diagonal is nonzero */
         trace += data[k1];
      }
   }

   *trace_io = trace;
   return nalu_hypre_error_flag;

}

/* Apply dropping to CSR matrix
 * A: the target CSR matrix
 * droptol: all entries have smaller absolute value than this will be dropped
 * max_row_nnz: max nonzeros allowed for each row, only largest max_row_nnz kept
 * we NEVER drop diagonal entry if exists
 */
NALU_HYPRE_Int
nalu_hypre_CSRMatrixDropInplace(nalu_hypre_CSRMatrix *A, NALU_HYPRE_Real droptol, NALU_HYPRE_Int max_row_nnz)
{
   NALU_HYPRE_Int      i, j, k1, k2;
   NALU_HYPRE_Int      *idx, len, drop_len;
   NALU_HYPRE_Real     *data, value, itol, norm;

   /* info of matrix A */
   NALU_HYPRE_Int      n = nalu_hypre_CSRMatrixNumRows(A);
   NALU_HYPRE_Int      m = nalu_hypre_CSRMatrixNumCols(A);
   NALU_HYPRE_Int      *A_i = nalu_hypre_CSRMatrixI(A);
   NALU_HYPRE_Int      *A_j = nalu_hypre_CSRMatrixJ(A);
   NALU_HYPRE_Real     *A_data = nalu_hypre_CSRMatrixData(A);
   NALU_HYPRE_Real     nnzA = nalu_hypre_CSRMatrixNumNonzeros(A);

   /* new data */
   NALU_HYPRE_Int      *new_i;
   NALU_HYPRE_Int      *new_j;
   NALU_HYPRE_Real     *new_data;

   /* memory */
   NALU_HYPRE_Int      capacity;
   NALU_HYPRE_Int      ctrA;

   /* setup */
   capacity = nnzA * 0.3 + 1;
   ctrA = 0;
   new_i = nalu_hypre_TAlloc(NALU_HYPRE_Int, n + 1, NALU_HYPRE_MEMORY_DEVICE);
   new_j = nalu_hypre_TAlloc(NALU_HYPRE_Int, capacity, NALU_HYPRE_MEMORY_DEVICE);
   new_data = nalu_hypre_TAlloc(NALU_HYPRE_Real, capacity, NALU_HYPRE_MEMORY_DEVICE);

   idx = nalu_hypre_TAlloc(NALU_HYPRE_Int, m, NALU_HYPRE_MEMORY_DEVICE);
   data = nalu_hypre_TAlloc(NALU_HYPRE_Real, m, NALU_HYPRE_MEMORY_DEVICE);

   /* start of main loop */
   new_i[0] = 0;
   for (i = 0 ; i < n ; i ++)
   {
      len = 0;
      k1 = A_i[i];
      k2 = A_i[i + 1];
      /* compute droptol for current row */
      norm = 0.0;
      for (j = k1 ; j < k2 ; j ++)
      {
         norm += nalu_hypre_abs(A_data[j]);
      }
      if (k2 > k1)
      {
         norm /= (NALU_HYPRE_Real)(k2 - k1);
      }
      itol = droptol * norm;
      /* we don't want to drop the diagonal entry, so use an if statement here */
      if (A_j[k1] == i)
      {
         /* we have diagonal entry, skip it */
         idx[len] = A_j[k1];
         data[len++] = A_data[k1];
         for (j = k1 + 1 ; j < k2 ; j ++)
         {
            value = A_data[j];
            if (nalu_hypre_abs(value) < itol)
            {
               /* skip small element */
               continue;
            }
            idx[len] = A_j[j];
            data[len++] = A_data[j];
         }

         /* now apply drop on length */
         if (len > max_row_nnz)
         {
            drop_len = max_row_nnz;
            nalu_hypre_ILUMaxQSplitRabsI( data + 1, idx + 1, 0, drop_len - 1, len - 2);
         }
         else
         {
            /* don't need to sort, we keep all of them */
            drop_len = len;
         }
         /* copy data */
         while (ctrA + drop_len > capacity)
         {
            NALU_HYPRE_Int tmp = capacity;
            capacity = capacity * EXPAND_FACT + 1;
            new_j = nalu_hypre_TReAlloc_v2(new_j, NALU_HYPRE_Int, tmp, NALU_HYPRE_Int, capacity, NALU_HYPRE_MEMORY_DEVICE);
            new_data = nalu_hypre_TReAlloc_v2(new_data, NALU_HYPRE_Real, tmp, NALU_HYPRE_Real, capacity, NALU_HYPRE_MEMORY_DEVICE);
         }
         nalu_hypre_TMemcpy( new_j + ctrA, idx, NALU_HYPRE_Int, drop_len, NALU_HYPRE_MEMORY_DEVICE, NALU_HYPRE_MEMORY_DEVICE);
         nalu_hypre_TMemcpy( new_data + ctrA, data, NALU_HYPRE_Real, drop_len, NALU_HYPRE_MEMORY_DEVICE,
                        NALU_HYPRE_MEMORY_DEVICE);
         ctrA += drop_len;
         new_i[i + 1] = ctrA;
      }
      else
      {
         /* we don't have diagonal entry */
         for (j = k1 ; j < k2 ; j ++)
         {
            value = A_data[j];
            if (nalu_hypre_abs(value) < itol)
            {
               /* skip small element */
               continue;
            }
            idx[len] = A_j[j];
            data[len++] = A_data[j];
         }

         /* now apply drop on length */
         if (len > max_row_nnz)
         {
            drop_len = max_row_nnz;
            nalu_hypre_ILUMaxQSplitRabsI( data, idx, 0, drop_len, len - 1);
         }
         else
         {
            /* don't need to sort, we keep all of them */
            drop_len = len;
         }

         /* copy data */
         while (ctrA + drop_len > capacity)
         {
            NALU_HYPRE_Int tmp = capacity;
            capacity = capacity * EXPAND_FACT + 1;
            new_j = nalu_hypre_TReAlloc_v2(new_j, NALU_HYPRE_Int, tmp, NALU_HYPRE_Int, capacity, NALU_HYPRE_MEMORY_DEVICE);
            new_data = nalu_hypre_TReAlloc_v2(new_data, NALU_HYPRE_Real, tmp, NALU_HYPRE_Real, capacity, NALU_HYPRE_MEMORY_DEVICE);
         }
         nalu_hypre_TMemcpy( new_j + ctrA, idx, NALU_HYPRE_Int, drop_len, NALU_HYPRE_MEMORY_DEVICE, NALU_HYPRE_MEMORY_DEVICE);
         nalu_hypre_TMemcpy( new_data + ctrA, data, NALU_HYPRE_Real, drop_len, NALU_HYPRE_MEMORY_DEVICE,
                        NALU_HYPRE_MEMORY_DEVICE);
         ctrA += drop_len;
         new_i[i + 1] = ctrA;
      }
   }/* end of main loop */
   /* destory data if A own them */
   if (nalu_hypre_CSRMatrixOwnsData(A))
   {
      nalu_hypre_TFree(A_i, NALU_HYPRE_MEMORY_DEVICE);
      nalu_hypre_TFree(A_j, NALU_HYPRE_MEMORY_DEVICE);
      nalu_hypre_TFree(A_data, NALU_HYPRE_MEMORY_DEVICE);
   }

   nalu_hypre_CSRMatrixI(A) = new_i;
   nalu_hypre_CSRMatrixJ(A) = new_j;
   nalu_hypre_CSRMatrixData(A) = new_data;
   nalu_hypre_CSRMatrixNumNonzeros(A) = ctrA;
   nalu_hypre_CSRMatrixOwnsData(A) = 1;

   nalu_hypre_TFree(idx, NALU_HYPRE_MEMORY_DEVICE);
   nalu_hypre_TFree(data, NALU_HYPRE_MEMORY_DEVICE);

   return nalu_hypre_error_flag;
}

/* Compute the inverse with MR of original CSR matrix
 * Global(not by each column) and out place version
 * A: the input matrix
 * M: the output matrix
 * droptol: the dropping tolorance
 * tol: when to stop the iteration
 * eps_tol: to avoid divide by 0
 * max_row_nnz: max number of nonzeros per row
 * max_iter: max number of iterations
 * print_level: the print level of this algorithm
 */
NALU_HYPRE_Int
nalu_hypre_ILUCSRMatrixInverseSelfPrecondMRGlobal(nalu_hypre_CSRMatrix *matA, nalu_hypre_CSRMatrix **M,
                                             NALU_HYPRE_Real droptol,
                                             NALU_HYPRE_Real tol, NALU_HYPRE_Real eps_tol, NALU_HYPRE_Int max_row_nnz, NALU_HYPRE_Int max_iter,
                                             NALU_HYPRE_Int print_level )
{
   NALU_HYPRE_Int         i, k1, k2;
   NALU_HYPRE_Real        value, trace1, trace2, alpha, r_norm;

   /* martix A */
   NALU_HYPRE_Int         *A_i = nalu_hypre_CSRMatrixI(matA);
   NALU_HYPRE_Int         *A_j = nalu_hypre_CSRMatrixJ(matA);
   NALU_HYPRE_Real        *A_data = nalu_hypre_CSRMatrixData(matA);

   /* complexity */
   NALU_HYPRE_Real        nnzA = nalu_hypre_CSRMatrixNumNonzeros(matA);
   NALU_HYPRE_Real        nnzM;

   /* inverse matrix */
   nalu_hypre_CSRMatrix   *inM = *M;
   nalu_hypre_CSRMatrix   *matM;
   NALU_HYPRE_Int         *M_i;
   NALU_HYPRE_Int         *M_j;
   NALU_HYPRE_Real        *M_data;

   /* idendity matrix */
   nalu_hypre_CSRMatrix   *matI;
   NALU_HYPRE_Int         *I_i;
   NALU_HYPRE_Int         *I_j;
   NALU_HYPRE_Real        *I_data;

   /* helper matrices */
   nalu_hypre_CSRMatrix   *matR;
   nalu_hypre_CSRMatrix   *matR_temp;
   nalu_hypre_CSRMatrix   *matZ;
   nalu_hypre_CSRMatrix   *matC;
   nalu_hypre_CSRMatrix   *matW;

   NALU_HYPRE_Real        time_s, time_e;

   NALU_HYPRE_Int         n = nalu_hypre_CSRMatrixNumRows(matA);

   /* create initial guess and matrix I */
   matM = nalu_hypre_CSRMatrixCreate(n, n, n);
   M_i = nalu_hypre_TAlloc(NALU_HYPRE_Int, n + 1, NALU_HYPRE_MEMORY_DEVICE);
   M_j = nalu_hypre_TAlloc(NALU_HYPRE_Int, n, NALU_HYPRE_MEMORY_DEVICE);
   M_data = nalu_hypre_TAlloc(NALU_HYPRE_Real, n, NALU_HYPRE_MEMORY_DEVICE);

   matI = nalu_hypre_CSRMatrixCreate(n, n, n);
   I_i = nalu_hypre_TAlloc(NALU_HYPRE_Int, n + 1, NALU_HYPRE_MEMORY_DEVICE);
   I_j = nalu_hypre_TAlloc(NALU_HYPRE_Int, n, NALU_HYPRE_MEMORY_DEVICE);
   I_data = nalu_hypre_TAlloc(NALU_HYPRE_Real, n, NALU_HYPRE_MEMORY_DEVICE);

   /* now loop to create initial guess */
   M_i[0] = 0;
   I_i[0] = 0;
   for (i = 0 ; i < n ; i ++)
   {
      M_i[i + 1] = i + 1;
      M_j[i] = i;
      k1 = A_i[i];
      k2 = A_i[i + 1];
      if (k2 > k1)
      {
         if (A_j[k1] == i)
         {
            value = A_data[k1];
            if (nalu_hypre_abs(value) < MAT_TOL)
            {
               value = 1.0;
            }
            M_data[i] = 1.0 / value;
         }
         else
         {
            M_data[i] = 1.0;
         }
      }
      else
      {
         M_data[i] = 1.0;
      }
      I_i[i + 1] = i + 1;
      I_j[i] = i;
      I_data[i] = 1.0;
   }

   nalu_hypre_CSRMatrixI(matM) = M_i;
   nalu_hypre_CSRMatrixJ(matM) = M_j;
   nalu_hypre_CSRMatrixData(matM) = M_data;
   nalu_hypre_CSRMatrixOwnsData(matM) = 1;

   nalu_hypre_CSRMatrixI(matI) = I_i;
   nalu_hypre_CSRMatrixJ(matI) = I_j;
   nalu_hypre_CSRMatrixData(matI) = I_data;
   nalu_hypre_CSRMatrixOwnsData(matI) = 1;

   /* now start the main loop */
   if (print_level > 1)
   {
      /* time the iteration */
      time_s = nalu_hypre_MPI_Wtime();
   }

   /* main loop */
   for (i = 0 ; i < max_iter ; i ++)
   {
      nnzM = nalu_hypre_CSRMatrixNumNonzeros(matM);
      /* R = I - AM */
      matR_temp = nalu_hypre_CSRMatrixMultiply(matA, matM);

      nalu_hypre_CSRMatrixScale(matR_temp, -1.0);

      matR = nalu_hypre_CSRMatrixAdd(1.0, matI, 1.0, matR_temp);
      nalu_hypre_CSRMatrixDestroy(matR_temp);

      /* r_norm */
      nalu_hypre_CSRMatrixNormFro(matR, &r_norm);
      if (r_norm < tol)
      {
         break;
      }

      /* Z = MR and dropping */
      matZ = nalu_hypre_CSRMatrixMultiply(matM, matR);
      //nalu_hypre_CSRMatrixNormFro(matZ, &z_norm);
      nalu_hypre_CSRMatrixDropInplace(matZ, droptol, max_row_nnz);

      /* C = A*Z */
      matC = nalu_hypre_CSRMatrixMultiply(matA, matZ);

      /* W = R' * C */
      nalu_hypre_CSRMatrixTranspose(matR, &matR_temp, 1);
      matW = nalu_hypre_CSRMatrixMultiply(matR_temp, matC);

      /* trace and alpha */
      nalu_hypre_CSRMatrixTrace(matW, &trace1);
      nalu_hypre_CSRMatrixNormFro(matC, &trace2);
      trace2 *= trace2;

      if (nalu_hypre_abs(trace2) < eps_tol)
      {
         break;
      }

      alpha = trace1 / trace2;

      /* M - M + alpha * Z */
      nalu_hypre_CSRMatrixScale(matZ, alpha);

      nalu_hypre_CSRMatrixDestroy(matR);
      matR = nalu_hypre_CSRMatrixAdd(1.0, matM, 1.0, matZ);
      nalu_hypre_CSRMatrixDestroy(matM);
      matM = matR;

      nalu_hypre_CSRMatrixDestroy(matZ);
      nalu_hypre_CSRMatrixDestroy(matW);
      nalu_hypre_CSRMatrixDestroy(matC);
      nalu_hypre_CSRMatrixDestroy(matR_temp);

   }/* end of main loop i for compute inverse matrix */

   /* time if we need to print */
   if (print_level > 1)
   {
      time_e = nalu_hypre_MPI_Wtime();
      if (i == 0)
      {
         i = 1;
      }
      nalu_hypre_printf("matrix size %5d\nfinal norm at loop %5d is %16.12f, time per iteration is %16.12f, complexity is %16.12f out of maximum %16.12f\n",
                   n, i, r_norm, (time_e - time_s) / i, nnzM / nnzA, n / nnzA * n);
   }

   nalu_hypre_CSRMatrixDestroy(matI);
   if (inM)
   {
      nalu_hypre_CSRMatrixDestroy(inM);
   }
   *M = matM;

   return nalu_hypre_error_flag;

}

/* Compute inverse with NSH method
 * Use MR to get local initial guess
 * A: input matrix
 * M: output matrix
 * droptol: droptol array. droptol[0] for MR and droptol[1] for NSH.
 * mr_tol: tol for stop iteration for MR
 * nsh_tol: tol for stop iteration for NSH
 * esp_tol: tol for avoid divide by 0
 * mr_max_row_nnz: max number of nonzeros for MR
 * nsh_max_row_nnz: max number of nonzeros for NSH
 * mr_max_iter: max number of iterations for MR
 * nsh_max_iter: max number of iterations for NSH
 * mr_col_version: column version of global version
 */
NALU_HYPRE_Int
nalu_hypre_ILUParCSRInverseNSH(nalu_hypre_ParCSRMatrix *A, nalu_hypre_ParCSRMatrix **M, NALU_HYPRE_Real *droptol,
                          NALU_HYPRE_Real mr_tol,
                          NALU_HYPRE_Real nsh_tol, NALU_HYPRE_Real eps_tol, NALU_HYPRE_Int mr_max_row_nnz, NALU_HYPRE_Int nsh_max_row_nnz,
                          NALU_HYPRE_Int mr_max_iter, NALU_HYPRE_Int nsh_max_iter, NALU_HYPRE_Int mr_col_version,
                          NALU_HYPRE_Int print_level)
{
   NALU_HYPRE_Int               i;

   /* data slots for matrices */
   nalu_hypre_ParCSRMatrix      *matM = NULL;
   nalu_hypre_ParCSRMatrix      *inM = *M;
   nalu_hypre_ParCSRMatrix      *AM, *MAM;
   NALU_HYPRE_Real              norm, s_norm;
   MPI_Comm                comm = nalu_hypre_ParCSRMatrixComm(A);
   NALU_HYPRE_Int               myid;


   nalu_hypre_CSRMatrix         *A_diag = nalu_hypre_ParCSRMatrixDiag(A);
   nalu_hypre_CSRMatrix         *M_diag = NULL;
   nalu_hypre_CSRMatrix         *M_offd;
   NALU_HYPRE_Int               *M_offd_i;

   NALU_HYPRE_Real              time_s, time_e;

   NALU_HYPRE_Int               n = nalu_hypre_CSRMatrixNumRows(A_diag);

   /* setup */
   nalu_hypre_MPI_Comm_rank(comm, &myid);

   M_offd_i = nalu_hypre_CTAlloc(NALU_HYPRE_Int, n + 1, NALU_HYPRE_MEMORY_DEVICE);

   if (mr_col_version)
   {
      nalu_hypre_printf("Column version is not yet support, switch to global version\n");
   }

   /* call MR to build loacl initial matrix
    * droptol here should be larger
    * we want same number for MR and NSH to let user set them eaiser
    * but we don't want a too dense MR initial guess
    */
   nalu_hypre_ILUCSRMatrixInverseSelfPrecondMRGlobal(A_diag, &M_diag, droptol[0] * 10.0, mr_tol, eps_tol,
                                                mr_max_row_nnz, mr_max_iter, print_level );

   /* create parCSR matM */
   matM = nalu_hypre_ParCSRMatrixCreate( comm,
                                    nalu_hypre_ParCSRMatrixGlobalNumRows(A),
                                    nalu_hypre_ParCSRMatrixGlobalNumRows(A),
                                    nalu_hypre_ParCSRMatrixRowStarts(A),
                                    nalu_hypre_ParCSRMatrixColStarts(A),
                                    0,
                                    nalu_hypre_CSRMatrixNumNonzeros(M_diag),
                                    0 );

   nalu_hypre_CSRMatrixDestroy(nalu_hypre_ParCSRMatrixDiag(matM));
   nalu_hypre_ParCSRMatrixDiag(matM) = M_diag;

   M_offd = nalu_hypre_ParCSRMatrixOffd(matM);
   nalu_hypre_CSRMatrixI(M_offd) = M_offd_i;
   nalu_hypre_CSRMatrixNumRownnz(M_offd) = 0;
   nalu_hypre_CSRMatrixOwnsData(M_offd)  = 1;

   /* now start NSH
    * Mj+1 = 2Mj - MjAMj
    */

   AM = nalu_hypre_ParMatmul(A, matM);
   nalu_hypre_ParCSRMatrixResNormFro(AM, &norm);
   s_norm = norm;
   nalu_hypre_ParCSRMatrixDestroy(AM);
   if (print_level > 1)
   {
      if (myid == 0)
      {
         nalu_hypre_printf("before NSH the norm is %16.12f\n", norm);
      }
      time_s = nalu_hypre_MPI_Wtime();
   }

   for (i = 0 ; i < nsh_max_iter ; i ++)
   {
      /* compute XjAXj */
      AM = nalu_hypre_ParMatmul(A, matM);
      nalu_hypre_ParCSRMatrixResNormFro(AM, &norm);
      if (norm < nsh_tol)
      {
         break;
      }
      MAM = nalu_hypre_ParMatmul(matM, AM);
      nalu_hypre_ParCSRMatrixDestroy(AM);

      /* apply dropping */
      //nalu_hypre_ParCSRMatrixNormFro(MAM, &norm);
      /* drop small entries based on 2-norm */
      nalu_hypre_ParCSRMatrixDropSmallEntries(MAM, droptol[1], 2);

      /* update Mj+1 = 2Mj - MjAMj
       * the result holds it own start/end data!
       */
      nalu_hypre_ParCSRMatrixAdd(2.0, matM, -1.0, MAM, &AM);
      nalu_hypre_ParCSRMatrixDestroy(matM);
      matM = AM;

      /* destroy */
      nalu_hypre_ParCSRMatrixDestroy(MAM);
   }

   if (print_level > 1)
   {
      time_e = nalu_hypre_MPI_Wtime();
      /* at this point of time, norm has to be already computed */
      if (i == 0)
      {
         i = 1;
      }
      if (myid == 0)
      {
         nalu_hypre_printf("after %5d NSH iterations the norm is %16.12f, time per iteration is %16.12f\n", i,
                      norm, (time_e - time_s) / i);
      }
   }

   if (s_norm < norm)
   {
      /* the residual norm increase after NSH iteration, need to let user know */
      if (myid == 0)
      {
         nalu_hypre_printf("Warning: NSH divergence, probably bad approximate invese matrix.\n");
      }
   }

   if (inM)
   {
      nalu_hypre_ParCSRMatrixDestroy(inM);
   }
   *M = matM;

   return nalu_hypre_error_flag;
}
