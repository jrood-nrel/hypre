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

/*--------------------------------------------------------------------------
 * nalu_hypre_ILUCreate
 *--------------------------------------------------------------------------*/

void *
nalu_hypre_ILUCreate( void )
{
   nalu_hypre_ParILUData *ilu_data;

   ilu_data = nalu_hypre_CTAlloc(nalu_hypre_ParILUData, 1, NALU_HYPRE_MEMORY_HOST);

#if defined(NALU_HYPRE_USING_CUDA) || defined(NALU_HYPRE_USING_HIP)
   nalu_hypre_ParILUDataAperm(ilu_data)                        = NULL;
   nalu_hypre_ParILUDataMatBILUDevice(ilu_data)                = NULL;
   nalu_hypre_ParILUDataMatSILUDevice(ilu_data)                = NULL;
   nalu_hypre_ParILUDataMatEDevice(ilu_data)                   = NULL;
   nalu_hypre_ParILUDataMatFDevice(ilu_data)                   = NULL;
   nalu_hypre_ParILUDataR(ilu_data)                            = NULL;
   nalu_hypre_ParILUDataP(ilu_data)                            = NULL;
   nalu_hypre_ParILUDataFTempUpper(ilu_data)                   = NULL;
   nalu_hypre_ParILUDataUTempLower(ilu_data)                   = NULL;
   nalu_hypre_ParILUDataADiagDiag(ilu_data)                    = NULL;
   nalu_hypre_ParILUDataSDiagDiag(ilu_data)                    = NULL;
#endif

   /* general data */
   nalu_hypre_ParILUDataGlobalSolver(ilu_data)                 = 0;
   nalu_hypre_ParILUDataMatA(ilu_data)                         = NULL;
   nalu_hypre_ParILUDataMatL(ilu_data)                         = NULL;
   nalu_hypre_ParILUDataMatD(ilu_data)                         = NULL;
   nalu_hypre_ParILUDataMatU(ilu_data)                         = NULL;
   nalu_hypre_ParILUDataMatS(ilu_data)                         = NULL;
   nalu_hypre_ParILUDataSchurSolver(ilu_data)                  = NULL;
   nalu_hypre_ParILUDataSchurPrecond(ilu_data)                 = NULL;
   nalu_hypre_ParILUDataRhs(ilu_data)                          = NULL;
   nalu_hypre_ParILUDataX(ilu_data)                            = NULL;

   /* TODO (VPM): Transform this into a stack array */
   nalu_hypre_ParILUDataDroptol(ilu_data) = nalu_hypre_TAlloc(NALU_HYPRE_Real, 3, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_ParILUDataDroptol(ilu_data)[0]                   = 1.0e-02; /* droptol for B */
   nalu_hypre_ParILUDataDroptol(ilu_data)[1]                   = 1.0e-02; /* droptol for E and F */
   nalu_hypre_ParILUDataDroptol(ilu_data)[2]                   = 1.0e-02; /* droptol for S */
   nalu_hypre_ParILUDataLfil(ilu_data)                         = 0;
   nalu_hypre_ParILUDataMaxRowNnz(ilu_data)                    = 1000;
   nalu_hypre_ParILUDataCFMarkerArray(ilu_data)                = NULL;
   nalu_hypre_ParILUDataPerm(ilu_data)                         = NULL;
   nalu_hypre_ParILUDataQPerm(ilu_data)                        = NULL;
   nalu_hypre_ParILUDataTolDDPQ(ilu_data)                      = 1.0e-01;
   nalu_hypre_ParILUDataF(ilu_data)                            = NULL;
   nalu_hypre_ParILUDataU(ilu_data)                            = NULL;
   nalu_hypre_ParILUDataFTemp(ilu_data)                        = NULL;
   nalu_hypre_ParILUDataUTemp(ilu_data)                        = NULL;
   nalu_hypre_ParILUDataXTemp(ilu_data)                        = NULL;
   nalu_hypre_ParILUDataYTemp(ilu_data)                        = NULL;
   nalu_hypre_ParILUDataZTemp(ilu_data)                        = NULL;
   nalu_hypre_ParILUDataUExt(ilu_data)                         = NULL;
   nalu_hypre_ParILUDataFExt(ilu_data)                         = NULL;
   nalu_hypre_ParILUDataResidual(ilu_data)                     = NULL;
   nalu_hypre_ParILUDataRelResNorms(ilu_data)                  = NULL;
   nalu_hypre_ParILUDataNumIterations(ilu_data)                = 0;
   nalu_hypre_ParILUDataMaxIter(ilu_data)                      = 20;
   nalu_hypre_ParILUDataTriSolve(ilu_data)                     = 1;
   nalu_hypre_ParILUDataLowerJacobiIters(ilu_data)             = 5;
   nalu_hypre_ParILUDataUpperJacobiIters(ilu_data)             = 5;
   nalu_hypre_ParILUDataTol(ilu_data)                          = 1.0e-7;
   nalu_hypre_ParILUDataLogging(ilu_data)                      = 0;
   nalu_hypre_ParILUDataPrintLevel(ilu_data)                   = 0;
   nalu_hypre_ParILUDataL1Norms(ilu_data)                      = NULL;
   nalu_hypre_ParILUDataOperatorComplexity(ilu_data)           = 0.;
   nalu_hypre_ParILUDataIluType(ilu_data)                      = 0;
   nalu_hypre_ParILUDataNLU(ilu_data)                          = 0;
   nalu_hypre_ParILUDataNI(ilu_data)                           = 0;
   nalu_hypre_ParILUDataUEnd(ilu_data)                         = NULL;

   /* reordering_type default to use local RCM */
   nalu_hypre_ParILUDataReorderingType(ilu_data)               = 1;

   /* see nalu_hypre_ILUSetType for more default values */
   nalu_hypre_ParILUDataTestOption(ilu_data)                   = 0;

   /* -> General slots */
   nalu_hypre_ParILUDataSchurSolverLogging(ilu_data)           = 0;
   nalu_hypre_ParILUDataSchurSolverPrintLevel(ilu_data)        = 0;

   /* -> Schur-GMRES */
   nalu_hypre_ParILUDataSchurGMRESKDim(ilu_data)               = 5;
   nalu_hypre_ParILUDataSchurGMRESMaxIter(ilu_data)            = 5;
   nalu_hypre_ParILUDataSchurGMRESTol(ilu_data)                = 0.0;
   nalu_hypre_ParILUDataSchurGMRESAbsoluteTol(ilu_data)        = 0.0;
   nalu_hypre_ParILUDataSchurGMRESRelChange(ilu_data)          = 0;

   /* -> Schur precond data */
   nalu_hypre_ParILUDataSchurPrecondIluType(ilu_data)          = 0;
   nalu_hypre_ParILUDataSchurPrecondIluLfil(ilu_data)          = 0;
   nalu_hypre_ParILUDataSchurPrecondIluMaxRowNnz(ilu_data)     = 100;
   nalu_hypre_ParILUDataSchurPrecondIluDroptol(ilu_data)       = NULL;
   nalu_hypre_ParILUDataSchurPrecondPrintLevel(ilu_data)       = 0;
   nalu_hypre_ParILUDataSchurPrecondMaxIter(ilu_data)          = 1;
   nalu_hypre_ParILUDataSchurPrecondTriSolve(ilu_data)         = 1;
   nalu_hypre_ParILUDataSchurPrecondLowerJacobiIters(ilu_data) = 5;
   nalu_hypre_ParILUDataSchurPrecondUpperJacobiIters(ilu_data) = 5;
   nalu_hypre_ParILUDataSchurPrecondTol(ilu_data)              = 0.0;

   /* -> Schur-NSH */
   nalu_hypre_ParILUDataSchurNSHSolveMaxIter(ilu_data)         = 5;
   nalu_hypre_ParILUDataSchurNSHSolveTol(ilu_data)             = 0.0;
   nalu_hypre_ParILUDataSchurNSHDroptol(ilu_data)              = NULL;
   nalu_hypre_ParILUDataSchurNSHMaxNumIter(ilu_data)           = 2;
   nalu_hypre_ParILUDataSchurNSHMaxRowNnz(ilu_data)            = 1000;
   nalu_hypre_ParILUDataSchurNSHTol(ilu_data)                  = 1e-09;
   nalu_hypre_ParILUDataSchurMRMaxIter(ilu_data)               = 2;
   nalu_hypre_ParILUDataSchurMRColVersion(ilu_data)            = 0;
   nalu_hypre_ParILUDataSchurMRMaxRowNnz(ilu_data)             = 200;
   nalu_hypre_ParILUDataSchurMRTol(ilu_data)                   = 1e-09;

   return ilu_data;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ILUDestroy
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ILUDestroy( void *data )
{
   nalu_hypre_ParILUData      *ilu_data = (nalu_hypre_ParILUData*) data;
   NALU_HYPRE_MemoryLocation   memory_location;

   if (ilu_data)
   {
      /* Get memory location from L factor */
      if (nalu_hypre_ParILUDataMatL(ilu_data))
      {
         memory_location = nalu_hypre_ParCSRMatrixMemoryLocation(nalu_hypre_ParILUDataMatL(ilu_data));
      }
      else
      {
         /* Use default memory location */
         NALU_HYPRE_GetMemoryLocation(&memory_location);
      }

      /* GPU additional data */
#if defined(NALU_HYPRE_USING_CUDA) || defined(NALU_HYPRE_USING_HIP)
      nalu_hypre_ParCSRMatrixDestroy( nalu_hypre_ParILUDataAperm(ilu_data) );
      nalu_hypre_ParCSRMatrixDestroy( nalu_hypre_ParILUDataR(ilu_data) );
      nalu_hypre_ParCSRMatrixDestroy( nalu_hypre_ParILUDataP(ilu_data) );

      nalu_hypre_CSRMatrixDestroy( nalu_hypre_ParILUDataMatAILUDevice(ilu_data) );
      nalu_hypre_CSRMatrixDestroy( nalu_hypre_ParILUDataMatBILUDevice(ilu_data) );
      nalu_hypre_CSRMatrixDestroy( nalu_hypre_ParILUDataMatSILUDevice(ilu_data) );
      nalu_hypre_CSRMatrixDestroy( nalu_hypre_ParILUDataMatEDevice(ilu_data) );
      nalu_hypre_CSRMatrixDestroy( nalu_hypre_ParILUDataMatFDevice(ilu_data) );
      nalu_hypre_SeqVectorDestroy( nalu_hypre_ParILUDataFTempUpper(ilu_data) );
      nalu_hypre_SeqVectorDestroy( nalu_hypre_ParILUDataUTempLower(ilu_data) );
      nalu_hypre_SeqVectorDestroy( nalu_hypre_ParILUDataADiagDiag(ilu_data) );
      nalu_hypre_SeqVectorDestroy( nalu_hypre_ParILUDataSDiagDiag(ilu_data) );
#endif

      /* final residual vector */
      nalu_hypre_ParVectorDestroy( nalu_hypre_ParILUDataResidual(ilu_data) );
      nalu_hypre_TFree( nalu_hypre_ParILUDataRelResNorms(ilu_data), NALU_HYPRE_MEMORY_HOST );

      /* temp vectors for solve phase */
      nalu_hypre_ParVectorDestroy( nalu_hypre_ParILUDataUTemp(ilu_data) );
      nalu_hypre_ParVectorDestroy( nalu_hypre_ParILUDataFTemp(ilu_data) );
      nalu_hypre_ParVectorDestroy( nalu_hypre_ParILUDataXTemp(ilu_data) );
      nalu_hypre_ParVectorDestroy( nalu_hypre_ParILUDataYTemp(ilu_data) );
      nalu_hypre_ParVectorDestroy( nalu_hypre_ParILUDataZTemp(ilu_data) );
      nalu_hypre_ParVectorDestroy( nalu_hypre_ParILUDataRhs(ilu_data) );
      nalu_hypre_ParVectorDestroy( nalu_hypre_ParILUDataX(ilu_data) );
      nalu_hypre_TFree( nalu_hypre_ParILUDataUExt(ilu_data), NALU_HYPRE_MEMORY_HOST );
      nalu_hypre_TFree( nalu_hypre_ParILUDataFExt(ilu_data), NALU_HYPRE_MEMORY_HOST );

      /* l1_norms */
      nalu_hypre_TFree( nalu_hypre_ParILUDataL1Norms(ilu_data), NALU_HYPRE_MEMORY_HOST );

      /* u_end */
      nalu_hypre_TFree( nalu_hypre_ParILUDataUEnd(ilu_data), NALU_HYPRE_MEMORY_HOST );

      /* Factors */
      nalu_hypre_ParCSRMatrixDestroy( nalu_hypre_ParILUDataMatS(ilu_data) );
      nalu_hypre_ParCSRMatrixDestroy( nalu_hypre_ParILUDataMatL(ilu_data) );
      nalu_hypre_ParCSRMatrixDestroy( nalu_hypre_ParILUDataMatU(ilu_data) );
      nalu_hypre_ParCSRMatrixDestroy( nalu_hypre_ParILUDataMatLModified(ilu_data) );
      nalu_hypre_ParCSRMatrixDestroy( nalu_hypre_ParILUDataMatUModified(ilu_data) );
      nalu_hypre_TFree( nalu_hypre_ParILUDataMatD(ilu_data), memory_location );
      nalu_hypre_TFree( nalu_hypre_ParILUDataMatDModified(ilu_data), memory_location );

      if (nalu_hypre_ParILUDataSchurSolver(ilu_data))
      {
         switch (nalu_hypre_ParILUDataIluType(ilu_data))
         {
            case 10: case 11: case 40: case 41: case 50:
               /* GMRES for Schur */
               NALU_HYPRE_ParCSRGMRESDestroy(nalu_hypre_ParILUDataSchurSolver(ilu_data));
               break;

            case 20: case 21:
               /* NSH for Schur */
               nalu_hypre_NSHDestroy(nalu_hypre_ParILUDataSchurSolver(ilu_data));
               break;

            default:
               break;
         }
      }

      /* ILU as precond for Schur */
      if ( nalu_hypre_ParILUDataSchurPrecond(ilu_data)  &&
#if defined(NALU_HYPRE_USING_CUDA) || defined(NALU_HYPRE_USING_HIP)
           nalu_hypre_ParILUDataIluType(ilu_data) != 10 &&
           nalu_hypre_ParILUDataIluType(ilu_data) != 11 &&
#endif
           (nalu_hypre_ParILUDataIluType(ilu_data) == 10 ||
            nalu_hypre_ParILUDataIluType(ilu_data) == 11 ||
            nalu_hypre_ParILUDataIluType(ilu_data) == 40 ||
            nalu_hypre_ParILUDataIluType(ilu_data) == 41) )
      {
         NALU_HYPRE_ILUDestroy( nalu_hypre_ParILUDataSchurPrecond(ilu_data) );
      }

      /* CF marker array */
      nalu_hypre_TFree( nalu_hypre_ParILUDataCFMarkerArray(ilu_data), NALU_HYPRE_MEMORY_HOST );

      /* permutation array */
      nalu_hypre_TFree( nalu_hypre_ParILUDataPerm(ilu_data), memory_location );
      nalu_hypre_TFree( nalu_hypre_ParILUDataQPerm(ilu_data), memory_location );

      /* droptol array - TODO (VPM): remove this after changing to static array */
      nalu_hypre_TFree( nalu_hypre_ParILUDataDroptol(ilu_data), NALU_HYPRE_MEMORY_HOST );
      nalu_hypre_TFree( nalu_hypre_ParILUDataSchurPrecondIluDroptol(ilu_data), NALU_HYPRE_MEMORY_HOST );
      nalu_hypre_TFree( nalu_hypre_ParILUDataSchurNSHDroptol(ilu_data), NALU_HYPRE_MEMORY_HOST );
   }

   /* ILU data */
   nalu_hypre_TFree(ilu_data, NALU_HYPRE_MEMORY_HOST);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ILUSetLevelOfFill
 *
 * Set fill level for ILUK
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ILUSetLevelOfFill( void      *ilu_vdata,
                         NALU_HYPRE_Int  lfil )
{
   nalu_hypre_ParILUData *ilu_data = (nalu_hypre_ParILUData*) ilu_vdata;

   nalu_hypre_ParILUDataLfil(ilu_data) = lfil;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ILUSetMaxNnzPerRow
 *
 * Set max non-zeros per row in factors for ILUT
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ILUSetMaxNnzPerRow( void      *ilu_vdata,
                          NALU_HYPRE_Int  nzmax )
{
   nalu_hypre_ParILUData *ilu_data = (nalu_hypre_ParILUData*) ilu_vdata;

   nalu_hypre_ParILUDataMaxRowNnz(ilu_data) = nzmax;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ILUSetDropThreshold
 *
 * Set threshold for dropping in LU factors for ILUT
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ILUSetDropThreshold( void       *ilu_vdata,
                           NALU_HYPRE_Real  threshold )
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

/*--------------------------------------------------------------------------
 * nalu_hypre_ILUSetDropThresholdArray
 *
 * Set array of threshold for dropping in LU factors for ILUT
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ILUSetDropThresholdArray( void       *ilu_vdata,
                                NALU_HYPRE_Real *threshold )
{
   nalu_hypre_ParILUData   *ilu_data = (nalu_hypre_ParILUData*) ilu_vdata;

   if (!(nalu_hypre_ParILUDataDroptol(ilu_data)))
   {
      nalu_hypre_ParILUDataDroptol(ilu_data) = nalu_hypre_TAlloc(NALU_HYPRE_Real, 3, NALU_HYPRE_MEMORY_HOST);
   }

   nalu_hypre_ParILUDataDroptol(ilu_data)[0] = threshold[0];
   nalu_hypre_ParILUDataDroptol(ilu_data)[1] = threshold[1];
   nalu_hypre_ParILUDataDroptol(ilu_data)[2] = threshold[2];

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ILUSetType
 *
 * Set ILU factorization type
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ILUSetType( void      *ilu_vdata,
                  NALU_HYPRE_Int  ilu_type )
{
   nalu_hypre_ParILUData *ilu_data = (nalu_hypre_ParILUData*) ilu_vdata;

   /* Destroy schur solver and/or preconditioner if already have one */
   if (nalu_hypre_ParILUDataSchurSolver(ilu_data))
   {
      switch (nalu_hypre_ParILUDataIluType(ilu_data))
      {
         case 10: case 11: case 40: case 41: case 50:
            //GMRES for Schur
            NALU_HYPRE_ParCSRGMRESDestroy(nalu_hypre_ParILUDataSchurSolver(ilu_data));
            break;

         case 20: case 21:
            //  NSH for Schur
            nalu_hypre_NSHDestroy(nalu_hypre_ParILUDataSchurSolver(ilu_data));
            break;

         default:
            break;
      }
      nalu_hypre_ParILUDataSchurSolver(ilu_data) = NULL;
   }

   /* ILU as precond for Schur */
   if ( nalu_hypre_ParILUDataSchurPrecond(ilu_data)    &&
#if defined(NALU_HYPRE_USING_CUDA) || defined(NALU_HYPRE_USING_HIP)
        (nalu_hypre_ParILUDataIluType(ilu_data) != 10  &&
         nalu_hypre_ParILUDataIluType(ilu_data) != 11) &&
#endif
        (nalu_hypre_ParILUDataIluType(ilu_data) == 10  ||
         nalu_hypre_ParILUDataIluType(ilu_data) == 11  ||
         nalu_hypre_ParILUDataIluType(ilu_data) == 40  ||
         nalu_hypre_ParILUDataIluType(ilu_data) == 41) )
   {
      NALU_HYPRE_ILUDestroy(nalu_hypre_ParILUDataSchurPrecond(ilu_data));
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

/*--------------------------------------------------------------------------
 * nalu_hypre_ILUSetMaxIter
 *
 * Set max number of iterations for ILU solver
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ILUSetMaxIter( void     *ilu_vdata,
                     NALU_HYPRE_Int  max_iter )
{
   nalu_hypre_ParILUData   *ilu_data = (nalu_hypre_ParILUData*) ilu_vdata;
   nalu_hypre_ParILUDataMaxIter(ilu_data) = max_iter;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ILUSetTriSolve
 *
 * Set ILU triangular solver type
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ILUSetTriSolve( void      *ilu_vdata,
                      NALU_HYPRE_Int  tri_solve )
{
   nalu_hypre_ParILUData   *ilu_data = (nalu_hypre_ParILUData*) ilu_vdata;

   nalu_hypre_ParILUDataTriSolve(ilu_data) = tri_solve;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ILUSetLowerJacobiIters
 *
 * Set Lower Jacobi iterations for iterative triangular solver
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ILUSetLowerJacobiIters( void     *ilu_vdata,
                              NALU_HYPRE_Int lower_jacobi_iters )
{
   nalu_hypre_ParILUData   *ilu_data = (nalu_hypre_ParILUData*) ilu_vdata;

   nalu_hypre_ParILUDataLowerJacobiIters(ilu_data) = lower_jacobi_iters;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ILUSetUpperJacobiIters
 *
 * Set Upper Jacobi iterations for iterative triangular solver
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ILUSetUpperJacobiIters( void      *ilu_vdata,
                              NALU_HYPRE_Int  upper_jacobi_iters )
{
   nalu_hypre_ParILUData   *ilu_data = (nalu_hypre_ParILUData*) ilu_vdata;

   nalu_hypre_ParILUDataUpperJacobiIters(ilu_data) = upper_jacobi_iters;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ILUSetTol
 *
 * Set convergence tolerance for ILU solver
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ILUSetTol( void       *ilu_vdata,
                 NALU_HYPRE_Real  tol )
{
   nalu_hypre_ParILUData   *ilu_data = (nalu_hypre_ParILUData*) ilu_vdata;

   nalu_hypre_ParILUDataTol(ilu_data) = tol;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ILUSetPrintLevel
 *
 * Set print level for ILU solver
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ILUSetPrintLevel( void      *ilu_vdata,
                        NALU_HYPRE_Int  print_level )
{
   nalu_hypre_ParILUData   *ilu_data = (nalu_hypre_ParILUData*) ilu_vdata;

   nalu_hypre_ParILUDataPrintLevel(ilu_data) = print_level;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ILUSetLogging
 *
 * Set print level for ilu solver
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ILUSetLogging( void      *ilu_vdata,
                     NALU_HYPRE_Int  logging )
{
   nalu_hypre_ParILUData   *ilu_data = (nalu_hypre_ParILUData*) ilu_vdata;

   nalu_hypre_ParILUDataLogging(ilu_data) = logging;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ILUSetLocalReordering
 *
 * Set type of reordering for local matrix
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ILUSetLocalReordering( void      *ilu_vdata,
                             NALU_HYPRE_Int  ordering_type )
{
   nalu_hypre_ParILUData   *ilu_data = (nalu_hypre_ParILUData*) ilu_vdata;

   nalu_hypre_ParILUDataReorderingType(ilu_data) = ordering_type;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ILUSetSchurSolverKDIM
 *
 * Set KDim (for GMRES) for Solver of Schur System
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ILUSetSchurSolverKDIM( void      *ilu_vdata,
                             NALU_HYPRE_Int  ss_kDim )
{
   nalu_hypre_ParILUData   *ilu_data = (nalu_hypre_ParILUData*) ilu_vdata;

   nalu_hypre_ParILUDataSchurGMRESKDim(ilu_data) = ss_kDim;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ILUSetSchurSolverMaxIter
 *
 * Set max iteration for Solver of Schur System
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ILUSetSchurSolverMaxIter( void      *ilu_vdata,
                                NALU_HYPRE_Int  ss_max_iter )
{
   nalu_hypre_ParILUData   *ilu_data = (nalu_hypre_ParILUData*) ilu_vdata;

   /* for the GMRES solve, the max iter is same as kdim by default */
   nalu_hypre_ParILUDataSchurGMRESKDim(ilu_data) = ss_max_iter;
   nalu_hypre_ParILUDataSchurGMRESMaxIter(ilu_data) = ss_max_iter;

   /* also set this value for NSH solve */
   nalu_hypre_ParILUDataSchurNSHSolveMaxIter(ilu_data) = ss_max_iter;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ILUSetSchurSolverTol
 *
 * Set convergence tolerance for Solver of Schur System
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ILUSetSchurSolverTol( void       *ilu_vdata,
                            NALU_HYPRE_Real  ss_tol )
{
   nalu_hypre_ParILUData   *ilu_data = (nalu_hypre_ParILUData*) ilu_vdata;

   nalu_hypre_ParILUDataSchurGMRESTol(ilu_data) = ss_tol;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ILUSetSchurSolverAbsoluteTol
 *
 * Set absolute tolerance for Solver of Schur System
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ILUSetSchurSolverAbsoluteTol( void       *ilu_vdata,
                                    NALU_HYPRE_Real  ss_absolute_tol )
{
   nalu_hypre_ParILUData   *ilu_data = (nalu_hypre_ParILUData*) ilu_vdata;

   nalu_hypre_ParILUDataSchurGMRESAbsoluteTol(ilu_data) = ss_absolute_tol;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ILUSetSchurSolverLogging
 *
 * Set logging for Solver of Schur System
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ILUSetSchurSolverLogging( void      *ilu_vdata,
                                NALU_HYPRE_Int  ss_logging )
{
   nalu_hypre_ParILUData   *ilu_data = (nalu_hypre_ParILUData*) ilu_vdata;

   nalu_hypre_ParILUDataSchurSolverLogging(ilu_data) = ss_logging;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ILUSetSchurSolverPrintLevel
 *
 * Set print level for Solver of Schur System
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ILUSetSchurSolverPrintLevel( void      *ilu_vdata,
                                   NALU_HYPRE_Int  ss_print_level )
{
   nalu_hypre_ParILUData   *ilu_data = (nalu_hypre_ParILUData*) ilu_vdata;

   nalu_hypre_ParILUDataSchurSolverPrintLevel(ilu_data) = ss_print_level;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ILUSetSchurSolverRelChange
 *
 * Set rel change (for GMRES) for Solver of Schur System
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ILUSetSchurSolverRelChange( void *ilu_vdata, NALU_HYPRE_Int ss_rel_change )
{
   nalu_hypre_ParILUData   *ilu_data = (nalu_hypre_ParILUData*) ilu_vdata;

   nalu_hypre_ParILUDataSchurGMRESRelChange(ilu_data) = ss_rel_change;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ILUSetSchurPrecondILUType
 *
 * Set ILU type for Precond of Schur System
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ILUSetSchurPrecondILUType( void *ilu_vdata, NALU_HYPRE_Int sp_ilu_type )
{
   nalu_hypre_ParILUData   *ilu_data = (nalu_hypre_ParILUData*) ilu_vdata;

   nalu_hypre_ParILUDataSchurPrecondIluType(ilu_data) = sp_ilu_type;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ILUSetSchurPrecondILULevelOfFill
 *
 * Set ILU level of fill for Precond of Schur System
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ILUSetSchurPrecondILULevelOfFill( void *ilu_vdata, NALU_HYPRE_Int sp_ilu_lfil )
{
   nalu_hypre_ParILUData   *ilu_data = (nalu_hypre_ParILUData*) ilu_vdata;

   nalu_hypre_ParILUDataSchurPrecondIluLfil(ilu_data) = sp_ilu_lfil;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ILUSetSchurPrecondILUMaxNnzPerRow
 *
 * Set ILU max nonzeros per row for Precond of Schur System
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ILUSetSchurPrecondILUMaxNnzPerRow( void      *ilu_vdata,
                                         NALU_HYPRE_Int  sp_ilu_max_row_nnz )
{
   nalu_hypre_ParILUData   *ilu_data = (nalu_hypre_ParILUData*) ilu_vdata;

   nalu_hypre_ParILUDataSchurPrecondIluMaxRowNnz(ilu_data) = sp_ilu_max_row_nnz;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ILUSetSchurPrecondILUDropThreshold
 *
 * Set ILU drop threshold for ILUT for Precond of Schur System
 * We don't want to influence the original ILU, so create new array if
 * not own data
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ILUSetSchurPrecondILUDropThreshold( void       *ilu_vdata,
                                          NALU_HYPRE_Real  sp_ilu_droptol )
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

/*--------------------------------------------------------------------------
 * nalu_hypre_ILUSetSchurPrecondILUDropThresholdArray
 *
 * Set array of ILU drop threshold for ILUT for Precond of Schur System
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ILUSetSchurPrecondILUDropThresholdArray( void       *ilu_vdata,
                                               NALU_HYPRE_Real *sp_ilu_droptol )
{
   nalu_hypre_ParILUData   *ilu_data = (nalu_hypre_ParILUData*) ilu_vdata;
   if (!(nalu_hypre_ParILUDataSchurPrecondIluDroptol(ilu_data)))
   {
      nalu_hypre_ParILUDataSchurPrecondIluDroptol(ilu_data) = nalu_hypre_TAlloc(NALU_HYPRE_Real, 3, NALU_HYPRE_MEMORY_HOST);
   }

   nalu_hypre_ParILUDataSchurPrecondIluDroptol(ilu_data)[0] = sp_ilu_droptol[0];
   nalu_hypre_ParILUDataSchurPrecondIluDroptol(ilu_data)[1] = sp_ilu_droptol[1];
   nalu_hypre_ParILUDataSchurPrecondIluDroptol(ilu_data)[2] = sp_ilu_droptol[2];

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ILUSetSchurPrecondPrintLevel
 *
 * Set print level for Precond of Schur System
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ILUSetSchurPrecondPrintLevel( void      *ilu_vdata,
                                    NALU_HYPRE_Int  sp_print_level )
{
   nalu_hypre_ParILUData   *ilu_data = (nalu_hypre_ParILUData*) ilu_vdata;

   nalu_hypre_ParILUDataSchurPrecondPrintLevel(ilu_data) = sp_print_level;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ILUSetSchurPrecondMaxIter
 *
 * Set max number of iterations for Precond of Schur System
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ILUSetSchurPrecondMaxIter( void      *ilu_vdata,
                                 NALU_HYPRE_Int  sp_max_iter )
{
   nalu_hypre_ParILUData   *ilu_data = (nalu_hypre_ParILUData*) ilu_vdata;

   nalu_hypre_ParILUDataSchurPrecondMaxIter(ilu_data) = sp_max_iter;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ILUSetSchurPrecondTriSolve
 *
 * Set triangular solver type for Precond of Schur System
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ILUSetSchurPrecondTriSolve( void      *ilu_vdata,
                                  NALU_HYPRE_Int  sp_tri_solve )
{
   nalu_hypre_ParILUData   *ilu_data = (nalu_hypre_ParILUData*) ilu_vdata;

   nalu_hypre_ParILUDataSchurPrecondTriSolve(ilu_data) = sp_tri_solve;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ILUSetSchurPrecondLowerJacobiIters
 *
 * Set Lower Jacobi iterations for Precond of Schur System
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ILUSetSchurPrecondLowerJacobiIters( void      *ilu_vdata,
                                          NALU_HYPRE_Int  sp_lower_jacobi_iters )
{
   nalu_hypre_ParILUData   *ilu_data = (nalu_hypre_ParILUData*) ilu_vdata;

   nalu_hypre_ParILUDataSchurPrecondLowerJacobiIters(ilu_data) = sp_lower_jacobi_iters;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ILUSetSchurPrecondUpperJacobiIters
 *
 * Set Upper Jacobi iterations for Precond of Schur System
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ILUSetSchurPrecondUpperJacobiIters( void      *ilu_vdata,
                                          NALU_HYPRE_Int  sp_upper_jacobi_iters )
{
   nalu_hypre_ParILUData   *ilu_data = (nalu_hypre_ParILUData*) ilu_vdata;

   nalu_hypre_ParILUDataSchurPrecondUpperJacobiIters(ilu_data) = sp_upper_jacobi_iters;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ILUSetSchurPrecondTol
 *
 * Set onvergence tolerance for Precond of Schur System
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ILUSetSchurPrecondTol( void      *ilu_vdata,
                             NALU_HYPRE_Int  sp_tol )
{
   nalu_hypre_ParILUData   *ilu_data = (nalu_hypre_ParILUData*) ilu_vdata;

   nalu_hypre_ParILUDataSchurPrecondTol(ilu_data) = sp_tol;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ILUSetSchurNSHDropThreshold
 *
 * Set tolorance for dropping in NSH for Schur System
 * We don't want to influence the original ILU, so create new array if
 * not own data
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ILUSetSchurNSHDropThreshold( void       *ilu_vdata,
                                   NALU_HYPRE_Real  threshold )
{
   nalu_hypre_ParILUData   *ilu_data = (nalu_hypre_ParILUData*) ilu_vdata;

   if (!(nalu_hypre_ParILUDataSchurNSHDroptol(ilu_data)))
   {
      nalu_hypre_ParILUDataSchurNSHDroptol(ilu_data) = nalu_hypre_TAlloc(NALU_HYPRE_Real, 2, NALU_HYPRE_MEMORY_HOST);
   }

   nalu_hypre_ParILUDataSchurNSHDroptol(ilu_data)[0] = threshold;
   nalu_hypre_ParILUDataSchurNSHDroptol(ilu_data)[1] = threshold;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ILUSetSchurNSHDropThresholdArray
 *
 * Set tolorance array for NSH for Schur System
 *    - threshold[0] : threshold for Minimal Residual iteration (initial guess for NSH).
 *    - threshold[1] : threshold for Newton-Schulz-Hotelling iteration.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ILUSetSchurNSHDropThresholdArray( void       *ilu_vdata,
                                        NALU_HYPRE_Real *threshold )
{
   nalu_hypre_ParILUData   *ilu_data = (nalu_hypre_ParILUData*) ilu_vdata;

   if (!(nalu_hypre_ParILUDataSchurNSHDroptol(ilu_data)))
   {
      nalu_hypre_ParILUDataSchurNSHDroptol(ilu_data) = nalu_hypre_TAlloc(NALU_HYPRE_Real, 2, NALU_HYPRE_MEMORY_HOST);
   }

   nalu_hypre_ParILUDataSchurNSHDroptol(ilu_data)[0] = threshold[0];
   nalu_hypre_ParILUDataSchurNSHDroptol(ilu_data)[1] = threshold[1];

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ILUGetNumIterations
 *
 * Get number of iterations for ILU solver
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ILUGetNumIterations( void      *ilu_vdata,
                           NALU_HYPRE_Int *num_iterations )
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

/*--------------------------------------------------------------------------
 * nalu_hypre_ILUGetFinalRelativeResidualNorm
 *
 * Get residual norms for ILU solver
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ILUGetFinalRelativeResidualNorm( void       *ilu_vdata,
                                       NALU_HYPRE_Real *res_norm )
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

/*--------------------------------------------------------------------------
 * nalu_hypre_ILUWriteSolverParams
 *
 * Print solver params
 *
 * TODO (VPM): check runtime switch to decide whether running on host or device
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ILUWriteSolverParams(void *ilu_vdata)
{
   nalu_hypre_ParILUData  *ilu_data = (nalu_hypre_ParILUData*) ilu_vdata;

   nalu_hypre_printf("ILU Setup parameters: \n");
   nalu_hypre_printf("ILU factorization type: %d : ", nalu_hypre_ParILUDataIluType(ilu_data));
   switch (nalu_hypre_ParILUDataIluType(ilu_data))
   {
      case 0:
#if defined(NALU_HYPRE_USING_CUDA) || defined(NALU_HYPRE_USING_HIP)
         if ( nalu_hypre_ParILUDataLfil(ilu_data) == 0 )
         {
            nalu_hypre_printf("Block Jacobi with GPU-accelerated ILU0 \n");
            nalu_hypre_printf("Operator Complexity (Fill factor) = %f \n",
                         nalu_hypre_ParILUDataOperatorComplexity(ilu_data));
         }
         else
#endif
         {
            nalu_hypre_printf("Block Jacobi with ILU(%d) \n", nalu_hypre_ParILUDataLfil(ilu_data));
            nalu_hypre_printf("Operator Complexity (Fill factor) = %f \n",
                         nalu_hypre_ParILUDataOperatorComplexity(ilu_data));
         }
         break;

      case 1:
         nalu_hypre_printf("Block Jacobi with ILUT \n");
         nalu_hypre_printf("drop tolerance for B = %e, E&F = %e, S = %e \n",
                      nalu_hypre_ParILUDataDroptol(ilu_data)[0],
                      nalu_hypre_ParILUDataDroptol(ilu_data)[1],
                      nalu_hypre_ParILUDataDroptol(ilu_data)[2]);
         nalu_hypre_printf("Max nnz per row = %d \n", nalu_hypre_ParILUDataMaxRowNnz(ilu_data));
         nalu_hypre_printf("Operator Complexity (Fill factor) = %f \n",
                      nalu_hypre_ParILUDataOperatorComplexity(ilu_data));
         break;

      case 10:
#if defined(NALU_HYPRE_USING_CUDA) || defined(NALU_HYPRE_USING_HIP)
         if ( nalu_hypre_ParILUDataLfil(ilu_data) == 0 )
         {
            nalu_hypre_printf("ILU-GMRES with GPU-accelerated ILU0 \n");
            nalu_hypre_printf("Operator Complexity (Fill factor) = %f \n",
                         nalu_hypre_ParILUDataOperatorComplexity(ilu_data));
         }
         else
#endif
         {
            nalu_hypre_printf("ILU-GMRES with ILU(%d) \n", nalu_hypre_ParILUDataLfil(ilu_data));
            nalu_hypre_printf("Operator Complexity (Fill factor) = %f \n",
                         nalu_hypre_ParILUDataOperatorComplexity(ilu_data));
         }
         break;

      case 11:
         nalu_hypre_printf("ILU-GMRES with ILUT \n");
         nalu_hypre_printf("drop tolerance for B = %e, E&F = %e, S = %e \n",
                      nalu_hypre_ParILUDataDroptol(ilu_data)[0],
                      nalu_hypre_ParILUDataDroptol(ilu_data)[1],
                      nalu_hypre_ParILUDataDroptol(ilu_data)[2]);
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
         nalu_hypre_printf("drop tolerance for B = %e, E&F = %e, S = %e \n",
                      nalu_hypre_ParILUDataDroptol(ilu_data)[0],
                      nalu_hypre_ParILUDataDroptol(ilu_data)[1],
                      nalu_hypre_ParILUDataDroptol(ilu_data)[2]);
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
         nalu_hypre_printf("drop tolerance for B = %e, E&F = %e, S = %e \n",
                      nalu_hypre_ParILUDataDroptol(ilu_data)[0],
                      nalu_hypre_ParILUDataDroptol(ilu_data)[1],
                      nalu_hypre_ParILUDataDroptol(ilu_data)[2]);
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
         nalu_hypre_printf("drop tolerance for B = %e, E&F = %e, S = %e \n",
                      nalu_hypre_ParILUDataDroptol(ilu_data)[0],
                      nalu_hypre_ParILUDataDroptol(ilu_data)[1],
                      nalu_hypre_ParILUDataDroptol(ilu_data)[2]);
         nalu_hypre_printf("Max nnz per row = %d \n", nalu_hypre_ParILUDataMaxRowNnz(ilu_data));
         nalu_hypre_printf("Operator Complexity (Fill factor) = %f \n",
                      nalu_hypre_ParILUDataOperatorComplexity(ilu_data));
         break;

      case 50:
         nalu_hypre_printf("RAP-Modified-ILU with ILU(%d) \n", nalu_hypre_ParILUDataLfil(ilu_data));
         nalu_hypre_printf("Operator Complexity (Fill factor) = %f \n",
                      nalu_hypre_ParILUDataOperatorComplexity(ilu_data));
         break;

      default:
         nalu_hypre_printf("Unknown type \n");
         break;
   }

   nalu_hypre_printf("\n ILU Solver Parameters: \n");
   nalu_hypre_printf("Max number of iterations: %d\n", nalu_hypre_ParILUDataMaxIter(ilu_data));
   if (nalu_hypre_ParILUDataTriSolve(ilu_data))
   {
      nalu_hypre_printf("  Triangular solver type: exact (1)\n");
   }
   else
   {
      nalu_hypre_printf("  Triangular solver type: iterative (0)\n");
      nalu_hypre_printf(" Lower Jacobi Iterations: %d\n", nalu_hypre_ParILUDataLowerJacobiIters(ilu_data));
      nalu_hypre_printf(" Upper Jacobi Iterations: %d\n", nalu_hypre_ParILUDataUpperJacobiIters(ilu_data));
   }
   nalu_hypre_printf("      Stopping tolerance: %e\n", nalu_hypre_ParILUDataTol(ilu_data));

   return nalu_hypre_error_flag;
}

/******************************************************************************
 *
 * ILU helper functions
 *
 * TODO (VPM): move these to a new "par_ilu_utils.c" file
 *
 *****************************************************************************/

/*--------------------------------------------------------------------------
 * nalu_hypre_ILUMinHeapAddI
 *
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
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ILUMinHeapAddI(NALU_HYPRE_Int *heap, NALU_HYPRE_Int len)
{
   /* parent, left, right */
   NALU_HYPRE_Int p;

   len--; /* now len is the current index */
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

/*--------------------------------------------------------------------------
 * nalu_hypre_ILUMinHeapAddIIIi
 *
 * See nalu_hypre_ILUMinHeapAddI for detail instructions
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ILUMinHeapAddIIIi(NALU_HYPRE_Int *heap, NALU_HYPRE_Int *I1, NALU_HYPRE_Int *Ii1, NALU_HYPRE_Int len)
{
   /* parent, left, right */
   NALU_HYPRE_Int p;

   len--; /* now len is the current index */
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

/*--------------------------------------------------------------------------
 * nalu_hypre_ILUMinHeapAddIRIi
 *
 * see nalu_hypre_ILUMinHeapAddI for detail instructions
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ILUMinHeapAddIRIi(NALU_HYPRE_Int *heap, NALU_HYPRE_Real *I1, NALU_HYPRE_Int *Ii1, NALU_HYPRE_Int len)
{
   /* parent, left, right */
   NALU_HYPRE_Int p;

   len--; /* now len is the current index */
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

/*--------------------------------------------------------------------------
 * nalu_hypre_ILUMaxrHeapAddRabsI
 *
 * See nalu_hypre_ILUMinHeapAddI for detail instructions
 *--------------------------------------------------------------------------*/

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

/*--------------------------------------------------------------------------
 * nalu_hypre_ILUMinHeapRemoveI
 *
 * Swap the first element with the last element of the heap,
 *    reduce size by one, and maintain the heap structure
 * I means NALU_HYPRE_Int
 * R means NALU_HYPRE_Real
 * max/min heap
 * r means heap goes from 0 to -1, -2 instead of 0 1 2
 * Ii and Ri means orderd by value of heap, like iw for ILU
 * heap: aray of that heap
 * len: current length of the heap
 * WARNING: Remember to change the len yourself
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ILUMinHeapRemoveI(NALU_HYPRE_Int *heap, NALU_HYPRE_Int len)
{
   /* parent, left, right */
   NALU_HYPRE_Int p, l, r;

   len--; /* now len is the max index */

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

/*--------------------------------------------------------------------------
 * nalu_hypre_ILUMinHeapRemoveIIIi
 *
 * See nalu_hypre_ILUMinHeapRemoveI for detail instructions
 *--------------------------------------------------------------------------*/

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

/*--------------------------------------------------------------------------
 * nalu_hypre_ILUMinHeapRemoveIRIi
 *
 * See nalu_hypre_ILUMinHeapRemoveI for detail instructions
 *--------------------------------------------------------------------------*/

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

/*--------------------------------------------------------------------------
 * nalu_hypre_ILUMaxrHeapRemoveRabsI
 *
 * See nalu_hypre_ILUMinHeapRemoveI for detail instructions
 *--------------------------------------------------------------------------*/

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

/*--------------------------------------------------------------------------
 * nalu_hypre_ILUMaxQSplitRabsI
 *
 * Split based on quick sort algorithm (avoid sorting the entire array)
 * find the largest k elements out of original array
 *
 * arrayR: input array for compare
 * arrayI: integer array bind with array
 * k: largest k elements
 * len: length of the array
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ILUMaxQSplitRabsI(NALU_HYPRE_Real *arrayR,
                        NALU_HYPRE_Int  *arrayI,
                        NALU_HYPRE_Int   left,
                        NALU_HYPRE_Int   bound,
                        NALU_HYPRE_Int   right)
{
   NALU_HYPRE_Int i, last;

   if (left >= right)
   {
      return nalu_hypre_error_flag;
   }

   nalu_hypre_swap2(arrayI, arrayR, left, (left + right) / 2);
   last = left;
   for (i = left + 1 ; i <= right ; i ++)
   {
      if (nalu_hypre_abs(arrayR[i]) > nalu_hypre_abs(arrayR[left]))
      {
         nalu_hypre_swap2(arrayI, arrayR, ++last, i);
      }
   }

   nalu_hypre_swap2(arrayI, arrayR, left, last);
   nalu_hypre_ILUMaxQSplitRabsI(arrayR, arrayI, left, bound, last - 1);
   if (bound > last)
   {
      nalu_hypre_ILUMaxQSplitRabsI(arrayR, arrayI, last + 1, bound, right);
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ILUMaxRabs
 *
 * Helper function to search max value from a row
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
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ILUMaxRabs(NALU_HYPRE_Real  *array_data,
                 NALU_HYPRE_Int   *array_j,
                 NALU_HYPRE_Int    start,
                 NALU_HYPRE_Int    end,
                 NALU_HYPRE_Int    nLU,
                 NALU_HYPRE_Int   *rperm,
                 NALU_HYPRE_Real  *value,
                 NALU_HYPRE_Int   *index,
                 NALU_HYPRE_Real  *l1_norm,
                 NALU_HYPRE_Int   *nnz)
{
   NALU_HYPRE_Int i, idx, col, nz;
   NALU_HYPRE_Real val, max_value, norm;

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

/*--------------------------------------------------------------------------
 * nalu_hypre_ILUGetPermddPQPre
 *
 * Pre selection for ddPQ, this is the basic version considering row sparsity
 * n: size of matrix
 * nLU: size we consider ddPQ reorder, only first nLU*nLU block is considered
 * A_diag_i/j/data: information of A
 * tol: tol for ddPQ, normally between 0.1-0.3
 * *perm: current row order
 * *rperm: current column order
 * *pperm_pre: output ddPQ pre row roder
 * *qperm_pre: output ddPQ pre column order
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ILUGetPermddPQPre(NALU_HYPRE_Int   n,
                        NALU_HYPRE_Int   nLU,
                        NALU_HYPRE_Int  *A_diag_i,
                        NALU_HYPRE_Int  *A_diag_j,
                        NALU_HYPRE_Real *A_diag_data,
                        NALU_HYPRE_Real  tol,
                        NALU_HYPRE_Int  *perm,
                        NALU_HYPRE_Int  *rperm,
                        NALU_HYPRE_Int  *pperm_pre,
                        NALU_HYPRE_Int  *qperm_pre,
                        NALU_HYPRE_Int  *nB)
{
   NALU_HYPRE_Int   i, ii, nB_pre, k1, k2;
   NALU_HYPRE_Real  gtol, max_value, norm;

   NALU_HYPRE_Int   *jcol, *jnnz;
   NALU_HYPRE_Real  *weight;

   weight = nalu_hypre_TAlloc(NALU_HYPRE_Real, nLU + 1, NALU_HYPRE_MEMORY_HOST);
   jcol   = nalu_hypre_TAlloc(NALU_HYPRE_Int, nLU + 1, NALU_HYPRE_MEMORY_HOST);
   jnnz   = nalu_hypre_TAlloc(NALU_HYPRE_Int, nLU + 1, NALU_HYPRE_MEMORY_HOST);

   max_value = -1.0;

   /* first need to build gtol */
   for (ii = 0; ii < nLU; ii++)
   {
      /* find real row */
      i = perm[ii];
      k1 = A_diag_i[i];
      k2 = A_diag_i[i + 1];

      /* find max|a| of that row and its index */
      nalu_hypre_ILUMaxRabs(A_diag_data, A_diag_j, k1, k2, nLU, rperm,
                       weight + ii, jcol + ii, &norm, jnnz + ii);
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

/*--------------------------------------------------------------------------
 * nalu_hypre_ILUGetPermddPQ
 *
 * Get ddPQ version perm array for ParCSR matrices. ddPQ is a two-side
 * permutation for diagonal dominance. Greedy matching selection
 *
 * Parameters:
 *   A: the input matrix
 *   pperm: row permutation (lives at memory_location_A)
 *   qperm: col permutation (lives at memory_location_A)
 *   nB: the size of B block
 *   nI: number of interial nodes
 *   tol: the dropping tolorance for ddPQ
 *   reordering_type: Type of reordering for the interior nodes.
 *
 * Currently only supports RCM reordering. Set to 0 for no reordering.
 *
 * TODO (VPM): Change permutation arrays types to nalu_hypre_IntArray
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ILUGetPermddPQ(nalu_hypre_ParCSRMatrix   *A,
                     NALU_HYPRE_Int           **io_pperm,
                     NALU_HYPRE_Int           **io_qperm,
                     NALU_HYPRE_Real            tol,
                     NALU_HYPRE_Int            *nB,
                     NALU_HYPRE_Int            *nI,
                     NALU_HYPRE_Int             reordering_type)
{
   /* data objects for A */
   nalu_hypre_CSRMatrix       *A_diag          = nalu_hypre_ParCSRMatrixDiag(A);
   NALU_HYPRE_Int              n               = nalu_hypre_CSRMatrixNumRows(A_diag);
   NALU_HYPRE_MemoryLocation   memory_location = nalu_hypre_CSRMatrixMemoryLocation(A_diag);

   nalu_hypre_CSRMatrix       *h_A_diag;
   NALU_HYPRE_Int             *A_diag_i;
   NALU_HYPRE_Int             *A_diag_j;
   NALU_HYPRE_Complex         *A_diag_data;

   /* Local variables */
   NALU_HYPRE_Int              i, nB_pre, irow, jcol, nLU;
   NALU_HYPRE_Int             *pperm, *qperm;
   NALU_HYPRE_Int             *new_pperm, *new_qperm;
   NALU_HYPRE_Int             *rpperm, *rqperm, *pperm_pre, *qperm_pre;
   NALU_HYPRE_MemoryLocation   memory_location_perm;

   /* 1: Setup and create memory */
   pperm  = NULL;
   qperm  = nalu_hypre_TAlloc(NALU_HYPRE_Int, n, NALU_HYPRE_MEMORY_HOST);
   rpperm = nalu_hypre_TAlloc(NALU_HYPRE_Int, n, NALU_HYPRE_MEMORY_HOST);
   rqperm = nalu_hypre_TAlloc(NALU_HYPRE_Int, n, NALU_HYPRE_MEMORY_HOST);

   /* 2: Find interior nodes first */
   nalu_hypre_ILUGetInteriorExteriorPerm(A, NALU_HYPRE_MEMORY_HOST, &pperm, &nLU, 0);

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

   /* Set/Move A_diag to host memory */
   h_A_diag = (nalu_hypre_GetActualMemLocation(memory_location) == nalu_hypre_MEMORY_DEVICE) ?
              nalu_hypre_CSRMatrixClone_v2(A_diag, 1, NALU_HYPRE_MEMORY_HOST) : A_diag;
   A_diag_i = nalu_hypre_CSRMatrixI(h_A_diag);
   A_diag_j = nalu_hypre_CSRMatrixJ(h_A_diag);
   A_diag_data = nalu_hypre_CSRMatrixData(h_A_diag);

   /* pre selection */
   nalu_hypre_ILUGetPermddPQPre(n, nLU, A_diag_i, A_diag_j, A_diag_data, tol,
                           pperm, rpperm, pperm_pre, qperm_pre, &nB_pre);

   /* 4: Build B block
    * Greedy selection
    */

   /* rperm[old] = new */
   for (i = 0 ; i < nLU ; i ++)
   {
      rpperm[pperm[i]] = -1;
   }

   nalu_hypre_TMemcpy(rqperm, rpperm, NALU_HYPRE_Int, n, NALU_HYPRE_MEMORY_HOST, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TMemcpy(qperm, pperm, NALU_HYPRE_Int, n, NALU_HYPRE_MEMORY_HOST, NALU_HYPRE_MEMORY_HOST);

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

   /* Apply RCM reordering */
   if (reordering_type != 0)
   {
      nalu_hypre_ILULocalRCM(h_A_diag, 0, nLU, &pperm, &qperm, 0);
      memory_location_perm = memory_location;
   }
   else
   {
      memory_location_perm = NALU_HYPRE_MEMORY_HOST;
   }

   /* Move to device memory if needed */
   if (memory_location_perm != memory_location)
   {
      new_pperm = nalu_hypre_TAlloc(NALU_HYPRE_Int, n, memory_location);
      new_qperm = nalu_hypre_TAlloc(NALU_HYPRE_Int, n, memory_location);

      nalu_hypre_TMemcpy(new_pperm, pperm, NALU_HYPRE_Int, n,
                    memory_location, memory_location_perm);
      nalu_hypre_TMemcpy(new_qperm, qperm, NALU_HYPRE_Int, n,
                    memory_location, memory_location_perm);

      nalu_hypre_TFree(pperm, memory_location_perm);
      nalu_hypre_TFree(qperm, memory_location_perm);

      pperm = new_pperm;
      qperm = new_qperm;
   }

   /* Output pointers */
   *nI = nLU;
   *nB = nLU;
   *io_pperm = pperm;
   *io_qperm = qperm;

   /* Free memory */
   if (h_A_diag != A_diag)
   {
      nalu_hypre_CSRMatrixDestroy(h_A_diag);
   }
   nalu_hypre_TFree(rpperm, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(rqperm, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(pperm_pre, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(qperm_pre, NALU_HYPRE_MEMORY_HOST);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ILUGetInteriorExteriorPerm
 *
 * Get perm array from parcsr matrix based on diag and offdiag matrix
 * Just simply loop through the rows of offd of A, check for nonzero rows
 * Put interior nodes at the beginning
 *
 * Parameters:
 *   A: parcsr matrix
 *   perm: permutation array
 *   nLU: number of interial nodes
 *   reordering_type: Type of (additional) reordering for the interior nodes.
 *
 * Currently only supports RCM reordering. Set to 0 for no reordering.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ILUGetInteriorExteriorPerm(nalu_hypre_ParCSRMatrix   *A,
                                 NALU_HYPRE_MemoryLocation  memory_location,
                                 NALU_HYPRE_Int           **perm,
                                 NALU_HYPRE_Int            *nLU,
                                 NALU_HYPRE_Int             reordering_type)
{
   /* get basic information of A */
   NALU_HYPRE_Int              n        = nalu_hypre_ParCSRMatrixNumRows(A);
   nalu_hypre_CSRMatrix       *A_diag   = nalu_hypre_ParCSRMatrixDiag(A);
   nalu_hypre_CSRMatrix       *A_offd   = nalu_hypre_ParCSRMatrixOffd(A);
   nalu_hypre_ParCSRCommPkg   *comm_pkg = nalu_hypre_ParCSRMatrixCommPkg(A);
   NALU_HYPRE_MemoryLocation   A_memory_location = nalu_hypre_ParCSRMatrixMemoryLocation(A);

   NALU_HYPRE_Int             *A_offd_i;
   NALU_HYPRE_Int              i, j, first, last, start, end;
   NALU_HYPRE_Int              num_sends, send_map_start, send_map_end, col;

   /* Local arrays */
   NALU_HYPRE_Int             *tperm   = nalu_hypre_TAlloc(NALU_HYPRE_Int, n, memory_location);
   NALU_HYPRE_Int             *h_tperm = nalu_hypre_CTAlloc(NALU_HYPRE_Int, n, NALU_HYPRE_MEMORY_HOST);
   NALU_HYPRE_Int             *marker  = nalu_hypre_CTAlloc(NALU_HYPRE_Int, n, NALU_HYPRE_MEMORY_HOST);

   /* Get comm_pkg, create one if not present */
   if (!comm_pkg)
   {
      nalu_hypre_MatvecCommPkgCreate(A);
      comm_pkg = nalu_hypre_ParCSRMatrixCommPkg(A);
   }

   /* Set A_offd_i on the host */
   if (nalu_hypre_GetActualMemLocation(A_memory_location) == nalu_hypre_MEMORY_DEVICE)
   {
      /* Move A_offd_i to host */
      A_offd_i = nalu_hypre_CTAlloc(NALU_HYPRE_Int, n + 1, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TMemcpy(A_offd_i,  nalu_hypre_CSRMatrixI(A_offd), NALU_HYPRE_Int, n + 1,
                    NALU_HYPRE_MEMORY_HOST, NALU_HYPRE_MEMORY_DEVICE);
   }
   else
   {
      A_offd_i = nalu_hypre_CSRMatrixI(A_offd);
   }

   /* Set initial interior/exterior pointers */
   first = 0;
   last  = n - 1;

   /* now directly take advantage of comm_pkg */
   num_sends = nalu_hypre_ParCSRCommPkgNumSends(comm_pkg);
   for (i = 0; i < num_sends; i++)
   {
      send_map_start = nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
      send_map_end   = nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, i + 1);
      for (j = send_map_start; j < send_map_end; j++)
      {
         col = nalu_hypre_ParCSRCommPkgSendMapElmt(comm_pkg, j);
         if (marker[col] == 0)
         {
            h_tperm[last--] = col;
            marker[col] = -1;
         }
      }
   }

   /* now deal with the row */
   for (i = 0; i < n; i++)
   {
      if (marker[i] == 0)
      {
         start = A_offd_i[i];
         end = A_offd_i[i + 1];
         if (start == end)
         {
            h_tperm[first++] = i;
         }
         else
         {
            h_tperm[last--] = i;
         }
      }
   }

   if (reordering_type != 0)
   {
      /* Apply RCM. Note: h_tperm lives at A_memory_location at output */
      nalu_hypre_ILULocalRCM(A_diag, 0, first, &h_tperm, &h_tperm, 1);

      /* Move permutation vector to final memory location */
      nalu_hypre_TMemcpy(tperm, h_tperm, NALU_HYPRE_Int, n, memory_location, A_memory_location);

      /* Free memory */
      nalu_hypre_TFree(h_tperm, A_memory_location);
   }
   else
   {
      /* Move permutation vector to final memory location */
      nalu_hypre_TMemcpy(tperm, h_tperm, NALU_HYPRE_Int, n, memory_location, NALU_HYPRE_MEMORY_HOST);

      /* Free memory */
      nalu_hypre_TFree(h_tperm, NALU_HYPRE_MEMORY_HOST);
   }

   /* Free memory */
   nalu_hypre_TFree(marker, NALU_HYPRE_MEMORY_HOST);
   if (A_offd_i != nalu_hypre_CSRMatrixI(A_offd))
   {
      nalu_hypre_TFree(A_offd_i, NALU_HYPRE_MEMORY_HOST);
   }

   /* Set output values */
   if ((*perm) != NULL)
   {
      nalu_hypre_TFree(*perm, memory_location);
   }
   *perm = tperm;
   *nLU = first;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ILUGetLocalPerm
 *
 * Get the (local) ordering of the diag (local) matrix (no permutation).
 * This is the permutation used for the block-jacobi case.
 *
 * Parameters:
 *   A: parcsr matrix
 *   perm: permutation array
 *   nLU: number of interior nodes
 *   reordering_type: Type of (additional) reordering for the nodes.
 *
 * Currently only supports RCM reordering. Set to 0 for no reordering.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ILUGetLocalPerm(nalu_hypre_ParCSRMatrix  *A,
                      NALU_HYPRE_Int          **perm_ptr,
                      NALU_HYPRE_Int           *nLU,
                      NALU_HYPRE_Int            reordering_type)
{
   /* get basic information of A */
   NALU_HYPRE_Int             num_rows = nalu_hypre_ParCSRMatrixNumRows(A);
   nalu_hypre_CSRMatrix      *A_diag = nalu_hypre_ParCSRMatrixDiag(A);

   /* Local variables */
   NALU_HYPRE_Int            *perm = NULL;

   /* Compute local RCM ordering on the host */
   if (reordering_type != 0)
   {
      nalu_hypre_ILULocalRCM(A_diag, 0, num_rows, &perm, &perm, 1);
   }

   /* Set output pointers */
   *nLU = num_rows;
   *perm_ptr = perm;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ILUBuildRASExternalMatrix
 *
 * Build the expanded matrix for RAS-1
 * A: input ParCSR matrix
 * E_i, E_j, E_data: information for external matrix
 * rperm: reverse permutation to build real index, rperm[old] = new
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ILUBuildRASExternalMatrix(nalu_hypre_ParCSRMatrix  *A,
                                NALU_HYPRE_Int           *rperm,
                                NALU_HYPRE_Int          **E_i,
                                NALU_HYPRE_Int          **E_j,
                                NALU_HYPRE_Real         **E_data)
{
   /* data objects for communication */
   MPI_Comm                 comm = nalu_hypre_ParCSRMatrixComm(A);
   NALU_HYPRE_Int                my_id;

   /* data objects for A */
   nalu_hypre_CSRMatrix          *A_diag = nalu_hypre_ParCSRMatrixDiag(A);
   nalu_hypre_CSRMatrix          *A_offd = nalu_hypre_ParCSRMatrixOffd(A);
   NALU_HYPRE_BigInt             *A_col_starts = nalu_hypre_ParCSRMatrixColStarts(A);
   NALU_HYPRE_BigInt             *A_offd_colmap = nalu_hypre_ParCSRMatrixColMapOffd(A);
   NALU_HYPRE_Int                *A_diag_i = nalu_hypre_CSRMatrixI(A_diag);
   NALU_HYPRE_Int                *A_offd_i = nalu_hypre_CSRMatrixI(A_offd);

   /* data objects for external A matrix */
   // Need to check the new version of nalu_hypre_ParcsrGetExternalRows
   nalu_hypre_CSRMatrix          *A_ext = NULL;
   // # up to local offd cols, no need to be NALU_HYPRE_BigInt
   NALU_HYPRE_Int                *A_ext_i = NULL;
   // Return global index, NALU_HYPRE_BigInt required
   NALU_HYPRE_BigInt             *A_ext_j = NULL;
   NALU_HYPRE_Real               *A_ext_data = NULL;

   /* data objects for output */
   NALU_HYPRE_Int                 E_nnz;
   NALU_HYPRE_Int                *E_ext_i = NULL;
   // Local index, no need to use NALU_HYPRE_BigInt
   NALU_HYPRE_Int                *E_ext_j = NULL;
   NALU_HYPRE_Real               *E_ext_data = NULL;

   //guess non-zeros for E before start
   NALU_HYPRE_Int                 E_init_alloc;

   /* size */
   NALU_HYPRE_Int                 n = nalu_hypre_CSRMatrixNumCols(A_diag);
   NALU_HYPRE_Int                 m = nalu_hypre_CSRMatrixNumCols(A_offd);
   NALU_HYPRE_Int                 A_diag_nnz = A_diag_i[n];
   NALU_HYPRE_Int                 A_offd_nnz = A_offd_i[n];

   NALU_HYPRE_Int                 i, j, idx;
   NALU_HYPRE_BigInt              big_col;

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
            E_init_alloc   = (NALU_HYPRE_Int)(E_init_alloc * EXPAND_FACT + 1);
            E_ext_j        = nalu_hypre_TReAlloc_v2(E_ext_j, NALU_HYPRE_Int, tmp, NALU_HYPRE_Int,
                                               E_init_alloc, NALU_HYPRE_MEMORY_HOST);
            E_ext_data     = nalu_hypre_TReAlloc_v2(E_ext_data, NALU_HYPRE_Real, tmp, NALU_HYPRE_Real,
                                               E_init_alloc, NALU_HYPRE_MEMORY_HOST);
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

/*--------------------------------------------------------------------------
 * nalu_hypre_ILUSortOffdColmap
 *
 * This function sort offdiagonal map as well as J array for offdiagonal part
 * A: The input CSR matrix.
 *
 * TODO (VPM): This work should be done via nalu_hypre_ParCSRMatrixPermute. This
 * function needs to be implemented.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ILUSortOffdColmap(nalu_hypre_ParCSRMatrix *A)
{
   nalu_hypre_CSRMatrix      *A_offd          = nalu_hypre_ParCSRMatrixOffd(A);
   NALU_HYPRE_Int            *A_offd_j        = nalu_hypre_CSRMatrixJ(A_offd);
   NALU_HYPRE_Int             A_offd_nnz      = nalu_hypre_CSRMatrixNumNonzeros(A_offd);
   NALU_HYPRE_Int             A_offd_num_cols = nalu_hypre_CSRMatrixNumCols(A_offd);
   NALU_HYPRE_MemoryLocation  memory_location = nalu_hypre_CSRMatrixMemoryLocation(A_offd);
   NALU_HYPRE_BigInt         *col_map_offd    = nalu_hypre_ParCSRMatrixColMapOffd(A);

   NALU_HYPRE_Int            *h_A_offd_j;

   NALU_HYPRE_Int            *perm  = nalu_hypre_TAlloc(NALU_HYPRE_Int, A_offd_num_cols, NALU_HYPRE_MEMORY_HOST);
   NALU_HYPRE_Int            *rperm = nalu_hypre_TAlloc(NALU_HYPRE_Int, A_offd_num_cols, NALU_HYPRE_MEMORY_HOST);
   NALU_HYPRE_Int             i;

   /* Set/Move A_offd_j on the host */
   if (nalu_hypre_GetActualMemLocation(memory_location) == nalu_hypre_MEMORY_DEVICE)
   {
      h_A_offd_j = nalu_hypre_TAlloc(NALU_HYPRE_Int, A_offd_nnz, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TMemcpy(h_A_offd_j, A_offd_j, NALU_HYPRE_Int, A_offd_nnz,
                    NALU_HYPRE_MEMORY_HOST, NALU_HYPRE_MEMORY_DEVICE);
   }
   else
   {
      h_A_offd_j = A_offd_j;
   }

   for (i = 0; i < A_offd_num_cols; i++)
   {
      perm[i] = i;
   }

   nalu_hypre_BigQsort2i(col_map_offd, perm, 0, A_offd_num_cols - 1);

   for (i = 0; i < A_offd_num_cols; i++)
   {
      rperm[perm[i]] = i;
   }

   for (i = 0; i < A_offd_nnz; i++)
   {
      h_A_offd_j[i] = rperm[h_A_offd_j[i]];
   }

   if (h_A_offd_j != A_offd_j)
   {
      nalu_hypre_TMemcpy(A_offd_j, h_A_offd_j, NALU_HYPRE_Int, A_offd_nnz,
                    NALU_HYPRE_MEMORY_DEVICE, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(h_A_offd_j, NALU_HYPRE_MEMORY_HOST);
   }

   /* Free memory */
   nalu_hypre_TFree(perm, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(rperm, NALU_HYPRE_MEMORY_HOST);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ILULocalRCMBuildFinalPerm
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ILULocalRCMBuildFinalPerm(NALU_HYPRE_Int   start,
                                NALU_HYPRE_Int   end,
                                NALU_HYPRE_Int  *G_perm,
                                NALU_HYPRE_Int  *perm,
                                NALU_HYPRE_Int  *qperm,
                                NALU_HYPRE_Int **permp,
                                NALU_HYPRE_Int **qpermp)
{
   /* update to new index */
   NALU_HYPRE_Int i = 0;
   NALU_HYPRE_Int num_nodes = end - start;
   NALU_HYPRE_Int *perm_temp = nalu_hypre_TAlloc(NALU_HYPRE_Int, num_nodes, NALU_HYPRE_MEMORY_HOST);

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

   nalu_hypre_TFree(perm_temp, NALU_HYPRE_MEMORY_HOST);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ILULocalRCM
 *
 * This function computes the RCM ordering of a sub matrix of
 * sparse matrix B = A(perm,perm)
 * For nonsymmetrix problem, is the RCM ordering of B + B'
 * A: The input CSR matrix
 * start:      the start position of the submatrix in B
 * end:        the end position of the submatrix in B ( exclude end, [start,end) )
 * permp:      pointer to the row permutation array such that B = A(perm, perm)
 *             point to NULL if you want to work directly on A
 *             on return, permp will point to the new permutation where
 *             in [start, end) the matrix will reordered. if *permp is not NULL,
 *             we assume that it lives on the host memory at input. At output,
 *             it lives in the same memory location as A.
 * qpermp:     pointer to the col permutation array such that B = A(perm, perm)
 *             point to NULL or equal to permp if you want symmetric order
 *             on return, qpermp will point to the new permutation where
 *             in [start, end) the matrix will reordered. if *qpermp is not NULL,
 *             we assume that it lives on the host memory at input. At output,
 *             it lives in the same memory location as A.
 * sym:        set to nonzero to work on A only(symmetric), otherwise A + A'.
 *             WARNING: if you use non-symmetric reordering, that is,
 *             different row and col reordering, the resulting A might be non-symmetric.
 *             Be careful if you are using non-symmetric reordering
 *
 * TODO (VPM): Implement RCM computation on the device.
 *             Use IntArray for perm.
 *             Move this function and internal RCM calls to parcsr_mv.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ILULocalRCM(nalu_hypre_CSRMatrix *A,
                  NALU_HYPRE_Int        start,
                  NALU_HYPRE_Int        end,
                  NALU_HYPRE_Int      **permp,
                  NALU_HYPRE_Int      **qpermp,
                  NALU_HYPRE_Int        sym)
{
   /* Input variables */
   NALU_HYPRE_Int               num_nodes       = end - start;
   NALU_HYPRE_Int               n               = nalu_hypre_CSRMatrixNumRows(A);
   NALU_HYPRE_Int               ncol            = nalu_hypre_CSRMatrixNumCols(A);
   NALU_HYPRE_MemoryLocation    memory_location = nalu_hypre_CSRMatrixMemoryLocation(A);
   NALU_HYPRE_Int               A_nnz           = nalu_hypre_CSRMatrixNumNonzeros(A);
   NALU_HYPRE_Int              *A_i;
   NALU_HYPRE_Int              *A_j;

   /* Local variables */
   nalu_hypre_CSRMatrix         *GT        = NULL;
   nalu_hypre_CSRMatrix         *GGT       = NULL;
   nalu_hypre_CSRMatrix         *G         = NULL;
   NALU_HYPRE_Int               *G_i       = NULL;
   NALU_HYPRE_Int               *G_j       = NULL;
   NALU_HYPRE_Int               *G_perm    = NULL;
   NALU_HYPRE_Int               *perm_temp = NULL;
   NALU_HYPRE_Int               *rqperm    = NULL;
   NALU_HYPRE_Int               *d_perm    = NULL;
   NALU_HYPRE_Int               *d_qperm   = NULL;
   NALU_HYPRE_Int               *perm      = *permp;
   NALU_HYPRE_Int               *qperm     = *qpermp;

   NALU_HYPRE_Int                perm_is_qperm;
   NALU_HYPRE_Int                i, j, row, col, r1, r2;
   NALU_HYPRE_Int                G_nnz, G_capacity;

   /* Set flag for computing row and column permutations (true) or only row permutation (false) */
   perm_is_qperm = (perm == qperm) ? 1 : 0;

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

   NALU_HYPRE_ANNOTATE_FUNC_BEGIN;
   nalu_hypre_GpuProfilingPushRange("ILULocalRCM");

   /* create permutation array if we don't have one yet */
   if (!perm)
   {
      perm = nalu_hypre_TAlloc(NALU_HYPRE_Int, n, NALU_HYPRE_MEMORY_HOST);
      for (i = 0; i < n; i++)
      {
         perm[i] = i;
      }
   }

   /* Check for symmetric reordering, then point qperm to row reordering */
   if (!qperm)
   {
      qperm = perm;
   }

   /* Compute reverse qperm ordering */
   rqperm = nalu_hypre_TAlloc(NALU_HYPRE_Int, n, NALU_HYPRE_MEMORY_HOST);
   for (i = 0; i < n; i++)
   {
      rqperm[qperm[i]] = i;
   }

   /* Set/Move A_i and A_j to host */
   if (nalu_hypre_GetActualMemLocation(memory_location) == nalu_hypre_MEMORY_DEVICE)
   {
      A_i = nalu_hypre_TAlloc(NALU_HYPRE_Int, n + 1, NALU_HYPRE_MEMORY_HOST);
      A_j = nalu_hypre_TAlloc(NALU_HYPRE_Int, A_nnz, NALU_HYPRE_MEMORY_HOST);

      nalu_hypre_TMemcpy(A_i, nalu_hypre_CSRMatrixI(A), NALU_HYPRE_Int, n + 1,
                    NALU_HYPRE_MEMORY_HOST, NALU_HYPRE_MEMORY_DEVICE);
      nalu_hypre_TMemcpy(A_j, nalu_hypre_CSRMatrixJ(A), NALU_HYPRE_Int, A_nnz,
                    NALU_HYPRE_MEMORY_HOST, NALU_HYPRE_MEMORY_DEVICE);
   }
   else
   {
      A_i = nalu_hypre_CSRMatrixI(A);
      A_j = nalu_hypre_CSRMatrixJ(A);
   }

   /* 2: Build Graph
    * Build Graph for RCM ordering
    */
   G_nnz = 0;
   G_capacity = nalu_hypre_max((A_nnz * n * n / num_nodes / num_nodes) - num_nodes, 1);
   G_i = nalu_hypre_TAlloc(NALU_HYPRE_Int, num_nodes + 1, NALU_HYPRE_MEMORY_HOST);
   G_j = nalu_hypre_TAlloc(NALU_HYPRE_Int, G_capacity, NALU_HYPRE_MEMORY_HOST);

   /* TODO (VPM): Extend nalu_hypre_CSRMatrixPermute to replace the block below */
   for (i = 0; i < num_nodes; i++)
   {
      G_i[i] = G_nnz;
      row = perm[i + start];
      r1 = A_i[row];
      r2 = A_i[row + 1];
      for (j = r1; j < r2; j ++)
      {
         col = rqperm[A_j[j]];
         if (col != row && col >= start && col < end)
         {
            /* this is an entry in G */
            G_j[G_nnz++] = col - start;
            if (G_nnz >= G_capacity)
            {
               NALU_HYPRE_Int tmp = G_capacity;
               G_capacity = (NALU_HYPRE_Int) (G_capacity * EXPAND_FACT + 1);
               G_j = nalu_hypre_TReAlloc_v2(G_j, NALU_HYPRE_Int, tmp, NALU_HYPRE_Int,
                                       G_capacity, NALU_HYPRE_MEMORY_HOST);
            }
         }
      }
   }
   G_i[num_nodes] = G_nnz;

   /* Free memory */
   if (A_i != nalu_hypre_CSRMatrixI(A))
   {
      nalu_hypre_TFree(A_i, NALU_HYPRE_MEMORY_HOST);
   }
   if (A_j != nalu_hypre_CSRMatrixJ(A))
   {
      nalu_hypre_TFree(A_j, NALU_HYPRE_MEMORY_HOST);
   }

   /* Create matrix G on the host */
   G = nalu_hypre_CSRMatrixCreate(num_nodes, num_nodes, G_nnz);
   nalu_hypre_CSRMatrixMemoryLocation(G) = NALU_HYPRE_MEMORY_HOST;
   nalu_hypre_CSRMatrixI(G) = G_i;
   nalu_hypre_CSRMatrixJ(G) = G_j;

   /* Check if G is not empty (no need to do any kind of RCM) */
   if (G_nnz > 0)
   {
      /* Sum G with G' if G is nonsymmetric */
      if (!sym)
      {
         nalu_hypre_CSRMatrixData(G) = nalu_hypre_CTAlloc(NALU_HYPRE_Complex, G_nnz, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_CSRMatrixTranspose(G, &GT, 1);
         GGT = nalu_hypre_CSRMatrixAdd(1.0, G, 1.0, GT);
         nalu_hypre_CSRMatrixDestroy(G);
         nalu_hypre_CSRMatrixDestroy(GT);
         G = GGT;
         GGT = NULL;
      }

      /* 3: Build RCM on the host */
      G_perm = nalu_hypre_TAlloc(NALU_HYPRE_Int, num_nodes, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_ILULocalRCMOrder(G, G_perm);

      /* 4: Post processing
       * Free, set value, return
       */

      /* update to new index */
      perm_temp = nalu_hypre_TAlloc(NALU_HYPRE_Int, num_nodes, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TMemcpy(perm_temp, &perm[start], NALU_HYPRE_Int, num_nodes,
                    NALU_HYPRE_MEMORY_HOST, NALU_HYPRE_MEMORY_HOST);
      for (i = 0; i < num_nodes; i++)
      {
         perm[i + start] = perm_temp[G_perm[i]];
      }

      if (!perm_is_qperm)
      {
         nalu_hypre_TMemcpy(perm_temp, &qperm[start], NALU_HYPRE_Int, num_nodes,
                       NALU_HYPRE_MEMORY_HOST, NALU_HYPRE_MEMORY_HOST);
         for (i = 0; i < num_nodes; i++)
         {
            qperm[i + start] = perm_temp[G_perm[i]];
         }
      }
   }

   /* Move to device memory if needed */
   if (memory_location == NALU_HYPRE_MEMORY_DEVICE)
   {
      d_perm = nalu_hypre_TAlloc(NALU_HYPRE_Int, n, NALU_HYPRE_MEMORY_DEVICE);
      nalu_hypre_TMemcpy(d_perm, perm, NALU_HYPRE_Int, n,
                    NALU_HYPRE_MEMORY_DEVICE, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(perm, NALU_HYPRE_MEMORY_HOST);

      perm = d_perm;
      if (perm_is_qperm)
      {
         qperm = d_perm;
      }
      else
      {
         d_qperm = nalu_hypre_TAlloc(NALU_HYPRE_Int, n, NALU_HYPRE_MEMORY_DEVICE);
         nalu_hypre_TMemcpy(d_qperm, qperm, NALU_HYPRE_Int, n,
                       NALU_HYPRE_MEMORY_DEVICE, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(qperm, NALU_HYPRE_MEMORY_HOST);

         qperm = d_qperm;
      }
   }

   /* Set output pointers */
   *permp  = perm;
   *qpermp = qperm;

   /* Free memory */
   nalu_hypre_CSRMatrixDestroy(G);
   nalu_hypre_TFree(G_perm, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(perm_temp, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(rqperm, NALU_HYPRE_MEMORY_HOST);

   nalu_hypre_GpuProfilingPopRange();
   NALU_HYPRE_ANNOTATE_FUNC_END;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ILULocalRCMMindegree
 *
 * This function finds the unvisited node with the minimum degree
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ILULocalRCMMindegree(NALU_HYPRE_Int  n,
                           NALU_HYPRE_Int *degree,
                           NALU_HYPRE_Int *marker,
                           NALU_HYPRE_Int *rootp)
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
 *
 * This function actually does the RCM ordering of a symmetric CSR matrix (entire)
 * A: the csr matrix, A_data is not needed
 * perm: the permutation array, space should be allocated outside
 * This is pure host code.
 *--------------------------------------------------------------------------*/

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

   /* Free */
   nalu_hypre_TFree(degree, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(marker, NALU_HYPRE_MEMORY_HOST);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ILULocalRCMFindPPNode
 *
 * This function find a pseudo-peripheral node start from root
 *   A: the csr matrix, A_data is not needed
 *   rootp: pointer to the root, on return will be a end of the pseudo-peripheral
 *   marker: the marker array for unvisited node
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ILULocalRCMFindPPNode( nalu_hypre_CSRMatrix *A, NALU_HYPRE_Int *rootp, NALU_HYPRE_Int *marker)
{
   NALU_HYPRE_Int      i, r1, r2, row, min_degree, lev_degree, nlev, newnlev;

   NALU_HYPRE_Int      root           = *rootp;
   NALU_HYPRE_Int      n              = nalu_hypre_CSRMatrixNumRows(A);
   NALU_HYPRE_Int     *A_i            = nalu_hypre_CSRMatrixI(A);

   /* at most n levels */
   NALU_HYPRE_Int     *level_i        = nalu_hypre_TAlloc(NALU_HYPRE_Int, n + 1, NALU_HYPRE_MEMORY_HOST);
   NALU_HYPRE_Int     *level_j        = nalu_hypre_TAlloc(NALU_HYPRE_Int, n, NALU_HYPRE_MEMORY_HOST);

   /* build initial level structure from root */
   nalu_hypre_ILULocalRCMBuildLevel(A, root, marker, level_i, level_j, &newnlev);

   nlev = newnlev - 1;
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

   /* Set output pointers */
   *rootp = root;

   /* Free */
   nalu_hypre_TFree(level_i, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(level_j, NALU_HYPRE_MEMORY_HOST);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ILULocalRCMBuildLevel
 *
 * This function build level structure start from root
 *   A: the csr matrix, A_data is not needed
 *   root: pointer to the root
 *   marker: the marker array for unvisited node
 *   level_i: points to the start/end of position on level_j, similar to CSR Matrix
 *   level_j: store node number on each level
 *   nlevp: return the number of level on this level structure
 *--------------------------------------------------------------------------*/

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

   /* Explore nbhds of all nodes in current level */
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
 *
 * This function generate numbering for a connect component
 *   A: the csr matrix, A_data is not needed
 *   root: pointer to the root
 *   marker: the marker array for unvisited node
 *   perm: permutation array
 *   current_nump: number of nodes already have a perm value
 *--------------------------------------------------------------------------*/

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
 *
 * This qsort is very specialized, not worth to put into utilities
 * Sort a part of array perm based on degree value (ascend)
 * That is, if degree[perm[i]] < degree[perm[j]], we should have i < j
 *   perm: the perm array
 *   start: start in perm
 *   end: end in perm
 *   degree: degree array
 *--------------------------------------------------------------------------*/

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

   /* Loop to split */
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
 *
 * Last step in RCM, reverse it
 * perm: perm array
 * srart: start position
 * end: end position
 *--------------------------------------------------------------------------*/

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

/* TODO (VPM): Change this block to another file? */
#if defined(NALU_HYPRE_USING_CUDA) || defined(NALU_HYPRE_USING_HIP)

/*--------------------------------------------------------------------------
 * nalu_hypre_ParILUSchurGMRESDummySolveDevice
 *
 * Unit GMRES preconditioner, just copy data from one slot to another
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ParILUSchurGMRESDummySolveDevice( void             *ilu_vdata,
                                        void             *ilu_vdata2,
                                        nalu_hypre_ParVector  *f,
                                        nalu_hypre_ParVector  *u )
{
   nalu_hypre_ParILUData    *ilu_data = (nalu_hypre_ParILUData*) ilu_vdata;
   nalu_hypre_ParCSRMatrix  *S        = nalu_hypre_ParILUDataMatS(ilu_data);
   NALU_HYPRE_Int            n_local  = nalu_hypre_ParCSRMatrixNumRows(S);

   nalu_hypre_Vector        *u_local = nalu_hypre_ParVectorLocalVector(u);
   NALU_HYPRE_Complex       *u_data  = nalu_hypre_VectorData(u_local);

   nalu_hypre_Vector        *f_local = nalu_hypre_ParVectorLocalVector(f);
   NALU_HYPRE_Complex       *f_data  = nalu_hypre_VectorData(f_local);

   nalu_hypre_TMemcpy(u_data, f_data, NALU_HYPRE_Real, n_local, NALU_HYPRE_MEMORY_DEVICE, NALU_HYPRE_MEMORY_DEVICE);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ParILUSchurGMRESCommInfoDevice
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ParILUSchurGMRESCommInfoDevice(void       *ilu_vdata,
                                     NALU_HYPRE_Int  *my_id,
                                     NALU_HYPRE_Int  *num_procs)
{
   /* get comm info from ilu_data */
   nalu_hypre_ParILUData     *ilu_data = (nalu_hypre_ParILUData*) ilu_vdata;
   nalu_hypre_ParCSRMatrix   *S        = nalu_hypre_ParILUDataMatS(ilu_data);
   MPI_Comm              comm     = nalu_hypre_ParCSRMatrixComm(S);

   nalu_hypre_MPI_Comm_size(comm, num_procs);
   nalu_hypre_MPI_Comm_rank(comm, my_id);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ParILURAPSchurGMRESSolveDevice
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ParILURAPSchurGMRESSolveDevice( void               *ilu_vdata,
                                      void               *ilu_vdata2,
                                      nalu_hypre_ParVector    *par_f,
                                      nalu_hypre_ParVector    *par_u )
{
   nalu_hypre_ParILUData        *ilu_data  = (nalu_hypre_ParILUData*) ilu_vdata;
   nalu_hypre_ParCSRMatrix      *S         = nalu_hypre_ParILUDataMatS(ilu_data);
   nalu_hypre_CSRMatrix         *SLU       = nalu_hypre_ParCSRMatrixDiag(S);

   nalu_hypre_ParVector         *par_rhs   = nalu_hypre_ParILUDataRhs(ilu_data);
   nalu_hypre_Vector            *rhs       = nalu_hypre_ParVectorLocalVector(par_rhs);
   nalu_hypre_Vector            *f         = nalu_hypre_ParVectorLocalVector(par_f);
   nalu_hypre_Vector            *u         = nalu_hypre_ParVectorLocalVector(par_u);

   /* L solve */
   nalu_hypre_CSRMatrixTriLowerUpperSolveDevice('L', 1, SLU, NULL, f, rhs);

   /* U solve */
   nalu_hypre_CSRMatrixTriLowerUpperSolveDevice('U', 0, SLU, NULL, rhs, u);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ParILURAPSchurGMRESMatvecDevice
 *
 * Compute y = alpha * S * x + beta * y
 *
 * TODO (VPM): Unify this function with nalu_hypre_ParILURAPSchurGMRESMatvecHost
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ParILURAPSchurGMRESMatvecDevice( void           *matvec_data,
                                       NALU_HYPRE_Complex   alpha,
                                       void           *ilu_vdata,
                                       void           *x,
                                       NALU_HYPRE_Complex   beta,
                                       void           *y )
{
   /* Get matrix information first */
   nalu_hypre_ParILUData       *ilu_data    = (nalu_hypre_ParILUData*) ilu_vdata;
   NALU_HYPRE_Int               test_opt    = nalu_hypre_ParILUDataTestOption(ilu_data);
   nalu_hypre_ParCSRMatrix     *Aperm       = nalu_hypre_ParILUDataAperm(ilu_data);
   NALU_HYPRE_Int               n           = nalu_hypre_CSRMatrixNumRows(nalu_hypre_ParCSRMatrixDiag(Aperm));
   nalu_hypre_CSRMatrix        *EiU         = nalu_hypre_ParILUDataMatEDevice(ilu_data);
   nalu_hypre_CSRMatrix        *iLF         = nalu_hypre_ParILUDataMatFDevice(ilu_data);
   nalu_hypre_CSRMatrix        *BLU         = nalu_hypre_ParILUDataMatBILUDevice(ilu_data);
   nalu_hypre_CSRMatrix        *C           = nalu_hypre_ParILUDataMatSILUDevice(ilu_data);

   nalu_hypre_ParVector        *x_vec       = (nalu_hypre_ParVector *) x;
   nalu_hypre_Vector           *x_local     = nalu_hypre_ParVectorLocalVector(x_vec);
   NALU_HYPRE_Real             *x_data      = nalu_hypre_VectorData(x_local);
   nalu_hypre_ParVector        *xtemp       = nalu_hypre_ParILUDataUTemp(ilu_data);
   nalu_hypre_Vector           *xtemp_local = nalu_hypre_ParVectorLocalVector(xtemp);
   NALU_HYPRE_Real             *xtemp_data  = nalu_hypre_VectorData(xtemp_local);

   nalu_hypre_ParVector        *y_vec       = (nalu_hypre_ParVector *) y;
   nalu_hypre_Vector           *y_local     = nalu_hypre_ParVectorLocalVector(y_vec);
   nalu_hypre_ParVector        *ytemp       = nalu_hypre_ParILUDataYTemp(ilu_data);
   nalu_hypre_Vector           *ytemp_local = nalu_hypre_ParVectorLocalVector(ytemp);
   NALU_HYPRE_Real             *ytemp_data  = nalu_hypre_VectorData(ytemp_local);

   NALU_HYPRE_Int               nLU;
   NALU_HYPRE_Int               m;
   nalu_hypre_Vector           *xtemp_upper;
   nalu_hypre_Vector           *xtemp_lower;
   nalu_hypre_Vector           *ytemp_upper;
   nalu_hypre_Vector           *ytemp_lower;

   switch (test_opt)
   {
      case 1:
         /* S = R * A * P */
         nLU                               = nalu_hypre_CSRMatrixNumRows(BLU);
         m                                 = n - nLU;
         xtemp_upper                       = nalu_hypre_SeqVectorCreate(nLU);
         ytemp_upper                       = nalu_hypre_SeqVectorCreate(nLU);
         xtemp_lower                       = nalu_hypre_SeqVectorCreate(m);
         nalu_hypre_VectorOwnsData(xtemp_upper) = 0;
         nalu_hypre_VectorOwnsData(ytemp_upper) = 0;
         nalu_hypre_VectorOwnsData(xtemp_lower) = 0;
         nalu_hypre_VectorData(xtemp_upper)     = xtemp_data;
         nalu_hypre_VectorData(ytemp_upper)     = ytemp_data;
         nalu_hypre_VectorData(xtemp_lower)     = xtemp_data + nLU;

         nalu_hypre_SeqVectorInitialize(xtemp_upper);
         nalu_hypre_SeqVectorInitialize(ytemp_upper);
         nalu_hypre_SeqVectorInitialize(xtemp_lower);

         /* first step, compute P*x put in y */
         /* -Fx */
         nalu_hypre_CSRMatrixMatvec(-1.0, iLF, x_local, 0.0, ytemp_upper);

         /* -L^{-1}Fx */
         /* L solve */
         nalu_hypre_CSRMatrixTriLowerUpperSolveDevice('L', 1, BLU, NULL, ytemp_local, xtemp_local);

         /* -U{-1}L^{-1}Fx */
         /* U solve */
         nalu_hypre_CSRMatrixTriLowerUpperSolveDevice('U', 0, BLU, NULL, xtemp_local, ytemp_local);

         /* now copy data to y_lower */
         nalu_hypre_TMemcpy(ytemp_data + nLU, x_data, NALU_HYPRE_Real, m,
                       NALU_HYPRE_MEMORY_DEVICE, NALU_HYPRE_MEMORY_DEVICE);

         /* second step, compute A*P*x store in xtemp */
         nalu_hypre_ParCSRMatrixMatvec(1.0, Aperm, ytemp, 0.0, xtemp);

         /* third step, compute R*A*P*x */
         /* solve L^{-1} */
         /* L solve */
         nalu_hypre_CSRMatrixTriLowerUpperSolveDevice('L', 1, BLU, NULL, xtemp_local, ytemp_local);

         /* U^{-1}L^{-1} */
         /* U solve */
         nalu_hypre_CSRMatrixTriLowerUpperSolveDevice('U', 0, BLU, NULL, ytemp_local, xtemp_local);

         /* -EU^{-1}L^{-1} */
         nalu_hypre_CSRMatrixMatvec(-alpha, EiU, xtemp_upper, beta, y_local);

         /* I*lower-EU^{-1}L^{-1}*upper */
         nalu_hypre_SeqVectorAxpy(alpha, xtemp_lower, y_local);

         nalu_hypre_SeqVectorDestroy(xtemp_upper);
         nalu_hypre_SeqVectorDestroy(ytemp_upper);
         nalu_hypre_SeqVectorDestroy(xtemp_lower);
         break;

      case 2:
         /* S = C - EU^{-1} * L^{-1}F */
         nLU                               = nalu_hypre_CSRMatrixNumRows(C);
         xtemp_upper                       = nalu_hypre_SeqVectorCreate(nLU);
         nalu_hypre_VectorOwnsData(xtemp_upper) = 0;
         nalu_hypre_VectorData(xtemp_upper)     = xtemp_data;

         nalu_hypre_SeqVectorInitialize(xtemp_upper);

         /* first step, compute EB^{-1}F*x put in y */
         /* -L^{-1}Fx */
         nalu_hypre_CSRMatrixMatvec(-1.0, iLF, x_local, 0.0, xtemp_upper);

         /* - alpha EU^{-1}L^{-1}Fx + beta * y */
         nalu_hypre_CSRMatrixMatvec(alpha, EiU, xtemp_upper, beta, y_local);

         /* alpha * C - alpha EU^{-1}L^{-1}Fx + beta y */
         nalu_hypre_CSRMatrixMatvec(alpha, C, x_local, 1.0, y_local);
         nalu_hypre_SeqVectorDestroy(xtemp_upper);
         break;

      case 3:
         /* S = C - EU^{-1} * L^{-1}F */
         nLU                               = nalu_hypre_CSRMatrixNumRows(C);
         xtemp_upper                       = nalu_hypre_SeqVectorCreate(nLU);
         nalu_hypre_VectorOwnsData(xtemp_upper) = 0;
         nalu_hypre_VectorData(xtemp_upper)     = xtemp_data;
         nalu_hypre_SeqVectorInitialize(xtemp_upper);

         /* first step, compute EB^{-1}F*x put in y */
         /* -Fx */
         nalu_hypre_CSRMatrixMatvec(-1.0, iLF, x_local, 0.0, xtemp_upper);

         /* -L^{-1}Fx */
         /* L solve */
         nalu_hypre_CSRMatrixTriLowerUpperSolveDevice('L', 1, BLU, NULL, xtemp_local, ytemp_local);

         /* -U^{-1}L^{-1}Fx */
         /* U solve */
         nalu_hypre_CSRMatrixTriLowerUpperSolveDevice('U', 0, BLU, NULL, ytemp_local, xtemp_local);

         /* - alpha EU^{-1}L^{-1}Fx + beta * y */
         nalu_hypre_CSRMatrixMatvec(alpha, EiU, xtemp_upper, beta, y_local);

         /* alpha * C - alpha EU^{-1}L^{-1}Fx + beta y */
         nalu_hypre_CSRMatrixMatvec(alpha, C, x_local, 1.0, y_local);
         nalu_hypre_SeqVectorDestroy(xtemp_upper);
         break;

   case 0: default:
         /* S = R * A * P */
         nLU                               = nalu_hypre_CSRMatrixNumRows(BLU);
         m                                 = n - nLU;
         xtemp_upper                       = nalu_hypre_SeqVectorCreate(nLU);
         ytemp_upper                       = nalu_hypre_SeqVectorCreate(nLU);
         ytemp_lower                       = nalu_hypre_SeqVectorCreate(m);
         nalu_hypre_VectorOwnsData(xtemp_upper) = 0;
         nalu_hypre_VectorOwnsData(ytemp_upper) = 0;
         nalu_hypre_VectorOwnsData(ytemp_lower) = 0;
         nalu_hypre_VectorData(xtemp_upper)     = xtemp_data;
         nalu_hypre_VectorData(ytemp_upper)     = ytemp_data;
         nalu_hypre_VectorData(ytemp_lower)     = ytemp_data + nLU;

         nalu_hypre_SeqVectorInitialize(xtemp_upper);
         nalu_hypre_SeqVectorInitialize(ytemp_upper);
         nalu_hypre_SeqVectorInitialize(ytemp_lower);

         /* first step, compute P*x put in y */
         /* -L^{-1}Fx */
         nalu_hypre_CSRMatrixMatvec(-1.0, iLF, x_local, 0.0, xtemp_upper);

         /* -U{-1}L^{-1}Fx */
         /* U solve */
         nalu_hypre_CSRMatrixTriLowerUpperSolveDevice('U', 0, BLU, NULL, xtemp_local, ytemp_local);

         /* now copy data to y_lower */
         nalu_hypre_TMemcpy(ytemp_data + nLU, x_data, NALU_HYPRE_Real, m,
                       NALU_HYPRE_MEMORY_DEVICE, NALU_HYPRE_MEMORY_DEVICE);

         /* second step, compute A*P*x store in xtemp */
         nalu_hypre_ParCSRMatrixMatvec(1.0, Aperm, ytemp, 0.0, xtemp);

         /* third step, compute R*A*P*x */
         /* copy partial data in */
         nalu_hypre_TMemcpy(ytemp_data + nLU, xtemp_data + nLU, NALU_HYPRE_Real, m,
                       NALU_HYPRE_MEMORY_DEVICE, NALU_HYPRE_MEMORY_DEVICE);

         /* solve L^{-1} */
         /* L solve */
         nalu_hypre_CSRMatrixTriLowerUpperSolveDevice('L', 1, BLU, NULL, xtemp_local, ytemp_local);

         /* -EU^{-1}L^{-1} */
         nalu_hypre_CSRMatrixMatvec(-alpha, EiU, ytemp_upper, beta, y_local);
         nalu_hypre_SeqVectorAxpy(alpha, ytemp_lower, y_local);

         /* over */
         nalu_hypre_SeqVectorDestroy(xtemp_upper);
         nalu_hypre_SeqVectorDestroy(ytemp_upper);
         nalu_hypre_SeqVectorDestroy(ytemp_lower);
         break;
   } /* switch (test_opt) */

   return nalu_hypre_error_flag;
}

#endif /* if defined(NALU_HYPRE_USING_CUDA) || defined(NALU_HYPRE_USING_HIP) */

/*--------------------------------------------------------------------------
 * nalu_hypre_ParILURAPSchurGMRESSolveHost
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ParILURAPSchurGMRESSolveHost( void               *ilu_vdata,
                                    void               *ilu_vdata2,
                                    nalu_hypre_ParVector    *f,
                                    nalu_hypre_ParVector    *u )
{
   nalu_hypre_ParILUData        *ilu_data     = (nalu_hypre_ParILUData*) ilu_vdata;
   nalu_hypre_ParCSRMatrix      *L            = nalu_hypre_ParILUDataMatLModified(ilu_data);
   nalu_hypre_CSRMatrix         *L_diag       = nalu_hypre_ParCSRMatrixDiag(L);
   NALU_HYPRE_Int               *L_diag_i     = nalu_hypre_CSRMatrixI(L_diag);
   NALU_HYPRE_Int               *L_diag_j     = nalu_hypre_CSRMatrixJ(L_diag);
   NALU_HYPRE_Real              *L_diag_data  = nalu_hypre_CSRMatrixData(L_diag);

   NALU_HYPRE_Real              *D            = nalu_hypre_ParILUDataMatDModified(ilu_data);

   nalu_hypre_ParCSRMatrix      *U            = nalu_hypre_ParILUDataMatUModified(ilu_data);
   nalu_hypre_CSRMatrix         *U_diag       = nalu_hypre_ParCSRMatrixDiag(U);
   NALU_HYPRE_Int               *U_diag_i     = nalu_hypre_CSRMatrixI(U_diag);
   NALU_HYPRE_Int               *U_diag_j     = nalu_hypre_CSRMatrixJ(U_diag);
   NALU_HYPRE_Real              *U_diag_data  = nalu_hypre_CSRMatrixData(U_diag);

   NALU_HYPRE_Int               n             = nalu_hypre_CSRMatrixNumRows(L_diag);
   NALU_HYPRE_Int               nLU           = nalu_hypre_ParILUDataNLU(ilu_data);
   NALU_HYPRE_Int               m             = n - nLU;

   nalu_hypre_Vector            *f_local      = nalu_hypre_ParVectorLocalVector(f);
   NALU_HYPRE_Real              *f_data       = nalu_hypre_VectorData(f_local);
   nalu_hypre_Vector            *u_local      = nalu_hypre_ParVectorLocalVector(u);
   NALU_HYPRE_Real              *u_data       = nalu_hypre_VectorData(u_local);
   nalu_hypre_ParVector         *utemp        = nalu_hypre_ParILUDataUTemp(ilu_data);
   nalu_hypre_Vector            *utemp_local  = nalu_hypre_ParVectorLocalVector(utemp);
   NALU_HYPRE_Real              *utemp_data   = nalu_hypre_VectorData(utemp_local);
   NALU_HYPRE_Int               *u_end        = nalu_hypre_ParILUDataUEnd(ilu_data);

   NALU_HYPRE_Int                i, j, k1, k2, col;

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
 * nalu_hypre_ParILURAPSchurGMRESCommInfoHost
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ParILURAPSchurGMRESCommInfoHost(void      *ilu_vdata,
                                      NALU_HYPRE_Int *my_id,
                                      NALU_HYPRE_Int *num_procs)
{
   /* get comm info from ilu_data */
   nalu_hypre_ParILUData    *ilu_data = (nalu_hypre_ParILUData*) ilu_vdata;
   nalu_hypre_ParCSRMatrix  *A        = nalu_hypre_ParILUDataMatA(ilu_data);
   MPI_Comm             comm     = nalu_hypre_ParCSRMatrixComm(A);

   nalu_hypre_MPI_Comm_size(comm, num_procs);
   nalu_hypre_MPI_Comm_rank(comm, my_id);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ParILURAPSchurGMRESMatvecHost
 *
 * Compute y = alpha * S * x + beta * y
 *
 * TODO (VPM): Unify this function with nalu_hypre_ParILURAPSchurGMRESMatvecDevice
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ParILURAPSchurGMRESMatvecHost( void          *matvec_data,
                                     NALU_HYPRE_Complex  alpha,
                                     void          *ilu_vdata,
                                     void          *x,
                                     NALU_HYPRE_Complex  beta,
                                     void          *y )
{
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

   nalu_hypre_ParVector         *utemp               = nalu_hypre_ParILUDataUTemp(ilu_data);
   nalu_hypre_Vector            *utemp_local         = nalu_hypre_ParVectorLocalVector(utemp);
   NALU_HYPRE_Real              *utemp_data          = nalu_hypre_VectorData(utemp_local);

   nalu_hypre_ParVector         *ftemp               = nalu_hypre_ParILUDataFTemp(ilu_data);
   nalu_hypre_Vector            *ftemp_local         = nalu_hypre_ParVectorLocalVector(ftemp);
   NALU_HYPRE_Real              *ftemp_data          = nalu_hypre_VectorData(ftemp_local);

   nalu_hypre_ParVector         *ytemp               = nalu_hypre_ParILUDataYTemp(ilu_data);
   nalu_hypre_Vector            *ytemp_local         = nalu_hypre_ParVectorLocalVector(ytemp);
   NALU_HYPRE_Real              *ytemp_data          = nalu_hypre_VectorData(ytemp_local);

   NALU_HYPRE_Int               i, j, k1, k2, col;
   NALU_HYPRE_Real              one  = 1.0;
   NALU_HYPRE_Real              zero = 0.0;

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

/******************************************************************************
 *
 * NSH create and solve and help functions.
 *
 * TODO (VPM): Move NSH code to separate files?
 *****************************************************************************/

/*--------------------------------------------------------------------------
 * nalu_hypre_NSHCreate
 *--------------------------------------------------------------------------*/

void *
nalu_hypre_NSHCreate( void )
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

/*--------------------------------------------------------------------------
 * nalu_hypre_NSHDestroy
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_NSHDestroy( void *data )
{
   nalu_hypre_ParNSHData * nsh_data = (nalu_hypre_ParNSHData*) data;

   /* residual */
   nalu_hypre_ParVectorDestroy( nalu_hypre_ParNSHDataResidual(nsh_data) );
   nalu_hypre_ParNSHDataResidual(nsh_data) = NULL;

   /* residual norms */
   nalu_hypre_TFree( nalu_hypre_ParNSHDataRelResNorms(nsh_data), NALU_HYPRE_MEMORY_HOST );
   nalu_hypre_ParNSHDataRelResNorms(nsh_data) = NULL;

   /* l1 norms */
   nalu_hypre_TFree( nalu_hypre_ParNSHDataL1Norms(nsh_data), NALU_HYPRE_MEMORY_HOST );
   nalu_hypre_ParNSHDataL1Norms(nsh_data) = NULL;

   /* temp arrays */
   nalu_hypre_ParVectorDestroy( nalu_hypre_ParNSHDataUTemp(nsh_data) );
   nalu_hypre_ParVectorDestroy( nalu_hypre_ParNSHDataFTemp(nsh_data) );
   nalu_hypre_ParNSHDataUTemp(nsh_data) = NULL;
   nalu_hypre_ParNSHDataFTemp(nsh_data) = NULL;

   /* approx inverse matrix */
   nalu_hypre_ParCSRMatrixDestroy( nalu_hypre_ParNSHDataMatM(nsh_data) );
   nalu_hypre_ParNSHDataMatM(nsh_data) = NULL;

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

/*--------------------------------------------------------------------------
 * nalu_hypre_NSHWriteSolverParams
 *
 * Print solver params
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_NSHWriteSolverParams( void *nsh_vdata )
{
   nalu_hypre_ParNSHData  *nsh_data = (nalu_hypre_ParNSHData*) nsh_vdata;
   nalu_hypre_printf("Newton-Schulz-Hotelling Setup parameters: \n");
   nalu_hypre_printf("NSH max iterations = %d \n", nalu_hypre_ParNSHDataNSHMaxIter(nsh_data));
   nalu_hypre_printf("NSH drop tolerance = %e \n", nalu_hypre_ParNSHDataDroptol(nsh_data)[1]);
   nalu_hypre_printf("NSH max nnz per row = %d \n", nalu_hypre_ParNSHDataNSHMaxRowNnz(nsh_data));
   nalu_hypre_printf("MR max iterations = %d \n", nalu_hypre_ParNSHDataMRMaxIter(nsh_data));
   nalu_hypre_printf("MR drop tolerance = %e \n", nalu_hypre_ParNSHDataDroptol(nsh_data)[0]);
   nalu_hypre_printf("MR max nnz per row = %d \n", nalu_hypre_ParNSHDataMRMaxRowNnz(nsh_data));
   nalu_hypre_printf("Operator Complexity (Fill factor) = %f \n",
                nalu_hypre_ParNSHDataOperatorComplexity(nsh_data));
   nalu_hypre_printf("\n Newton-Schulz-Hotelling Solver Parameters: \n");
   nalu_hypre_printf("Max number of iterations: %d\n", nalu_hypre_ParNSHDataMaxIter(nsh_data));
   nalu_hypre_printf("Stopping tolerance: %e\n", nalu_hypre_ParNSHDataTol(nsh_data));

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_NSHSetPrintLevel
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_NSHSetPrintLevel( void *nsh_vdata, NALU_HYPRE_Int print_level )
{
   nalu_hypre_ParNSHData   *nsh_data = (nalu_hypre_ParNSHData*) nsh_vdata;
   nalu_hypre_ParNSHDataPrintLevel(nsh_data) = print_level;
   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_NSHSetLogging
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_NSHSetLogging( void *nsh_vdata, NALU_HYPRE_Int logging )
{
   nalu_hypre_ParNSHData   *nsh_data = (nalu_hypre_ParNSHData*) nsh_vdata;
   nalu_hypre_ParNSHDataLogging(nsh_data) = logging;
   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_NSHSetMaxIter
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_NSHSetMaxIter( void *nsh_vdata, NALU_HYPRE_Int max_iter )
{
   nalu_hypre_ParNSHData   *nsh_data = (nalu_hypre_ParNSHData*) nsh_vdata;
   nalu_hypre_ParNSHDataMaxIter(nsh_data) = max_iter;
   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_NSHSetTol
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_NSHSetTol( void *nsh_vdata, NALU_HYPRE_Real tol )
{
   nalu_hypre_ParNSHData   *nsh_data = (nalu_hypre_ParNSHData*) nsh_vdata;
   nalu_hypre_ParNSHDataTol(nsh_data) = tol;
   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_NSHSetGlobalSolver
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_NSHSetGlobalSolver( void *nsh_vdata, NALU_HYPRE_Int global_solver )
{
   nalu_hypre_ParNSHData   *nsh_data = (nalu_hypre_ParNSHData*) nsh_vdata;
   nalu_hypre_ParNSHDataGlobalSolver(nsh_data) = global_solver;
   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_NSHSetDropThreshold
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_NSHSetDropThreshold( void *nsh_vdata, NALU_HYPRE_Real droptol )
{
   nalu_hypre_ParNSHData   *nsh_data = (nalu_hypre_ParNSHData*) nsh_vdata;
   nalu_hypre_ParNSHDataDroptol(nsh_data)[0] = droptol;
   nalu_hypre_ParNSHDataDroptol(nsh_data)[1] = droptol;
   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_NSHSetDropThresholdArray
 *--------------------------------------------------------------------------*/

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

/*--------------------------------------------------------------------------
 * nalu_hypre_NSHSetMRMaxIter
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_NSHSetMRMaxIter( void *nsh_vdata, NALU_HYPRE_Int mr_max_iter )
{
   nalu_hypre_ParNSHData   *nsh_data = (nalu_hypre_ParNSHData*) nsh_vdata;
   nalu_hypre_ParNSHDataMRMaxIter(nsh_data) = mr_max_iter;
   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_NSHSetMRTol
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_NSHSetMRTol( void *nsh_vdata, NALU_HYPRE_Real mr_tol )
{
   nalu_hypre_ParNSHData   *nsh_data = (nalu_hypre_ParNSHData*) nsh_vdata;
   nalu_hypre_ParNSHDataMRTol(nsh_data) = mr_tol;
   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_NSHSetMRMaxRowNnz
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_NSHSetMRMaxRowNnz( void *nsh_vdata, NALU_HYPRE_Int mr_max_row_nnz )
{
   nalu_hypre_ParNSHData   *nsh_data = (nalu_hypre_ParNSHData*) nsh_vdata;
   nalu_hypre_ParNSHDataMRMaxRowNnz(nsh_data) = mr_max_row_nnz;
   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_NSHSetColVersion
 *
 * set MR version, column version or global version
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_NSHSetColVersion( void *nsh_vdata, NALU_HYPRE_Int mr_col_version )
{
   nalu_hypre_ParNSHData   *nsh_data = (nalu_hypre_ParNSHData*) nsh_vdata;
   nalu_hypre_ParNSHDataMRColVersion(nsh_data) = mr_col_version;
   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_NSHSetNSHMaxIter
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_NSHSetNSHMaxIter( void *nsh_vdata, NALU_HYPRE_Int nsh_max_iter )
{
   nalu_hypre_ParNSHData   *nsh_data = (nalu_hypre_ParNSHData*) nsh_vdata;
   nalu_hypre_ParNSHDataNSHMaxIter(nsh_data) = nsh_max_iter;
   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_NSHSetNSHTol
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_NSHSetNSHTol( void *nsh_vdata, NALU_HYPRE_Real nsh_tol )
{
   nalu_hypre_ParNSHData   *nsh_data = (nalu_hypre_ParNSHData*) nsh_vdata;
   nalu_hypre_ParNSHDataNSHTol(nsh_data) = nsh_tol;
   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_NSHSetNSHMaxRowNnz
 *
 * Set NSH max nonzeros of a row
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_NSHSetNSHMaxRowNnz( void *nsh_vdata, NALU_HYPRE_Int nsh_max_row_nnz )
{
   nalu_hypre_ParNSHData   *nsh_data = (nalu_hypre_ParNSHData*) nsh_vdata;
   nalu_hypre_ParNSHDataNSHMaxRowNnz(nsh_data) = nsh_max_row_nnz;
   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_CSRMatrixNormFro
 *
 * Compute the F norm of CSR matrix
 * A: the target CSR matrix
 * norm_io: output
 *
 * TODO (VPM): Move this function to seq_mv
 *--------------------------------------------------------------------------*/

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
   *norm_io = nalu_hypre_sqrt(norm);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_CSRMatrixResNormFro
 *
 * Compute the norm of I-A where I is identity matrix and A is a CSR matrix
 * A: the target CSR matrix
 * norm_io: the output
 *
 * TODO (VPM): Move this function to seq_mv
 *--------------------------------------------------------------------------*/

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
   *norm_io = nalu_hypre_sqrt(norm);
   return nalu_hypre_error_flag;
}


/*--------------------------------------------------------------------------
 * nalu_hypre_ParCSRMatrixNormFro
 *
 * Compute the F norm of ParCSR matrix
 * A: the target CSR matrix
 *
 * TODO (VPM): Move this function to parcsr_mv
 *--------------------------------------------------------------------------*/

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

   *norm_io = nalu_hypre_sqrt(global_norm);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ParCSRMatrixResNormFro
 *
 * Compute the F norm of ParCSR matrix
 * Norm of I-A
 * A: the target CSR matrix
 *
 * TODO (VPM): Move this function to parcsr_mv
 *--------------------------------------------------------------------------*/

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

   *norm_io = nalu_hypre_sqrt(global_norm);
   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_CSRMatrixTrace
 *
 * Compute the trace of CSR matrix
 * A: the target CSR matrix
 * trace_io: the output trace
 *
 * TODO (VPM): Move this function to seq_mv
 *--------------------------------------------------------------------------*/

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

/*--------------------------------------------------------------------------
 * nalu_hypre_CSRMatrixDropInplace
 *
 * Apply dropping to CSR matrix
 * A: the target CSR matrix
 * droptol: all entries have smaller absolute value than this will be dropped
 * max_row_nnz: max nonzeros allowed for each row, only largest max_row_nnz kept
 * we NEVER drop diagonal entry if exists
 *
 * TODO (VPM): Move this function to seq_mv
 *--------------------------------------------------------------------------*/

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
   NALU_HYPRE_MemoryLocation memory_location = nalu_hypre_CSRMatrixMemoryLocation(A);

   /* new data */
   NALU_HYPRE_Int      *new_i;
   NALU_HYPRE_Int      *new_j;
   NALU_HYPRE_Real     *new_data;

   /* memory */
   NALU_HYPRE_Int      capacity;
   NALU_HYPRE_Int      ctrA;

   /* setup */
   capacity = (NALU_HYPRE_Int)(nnzA * 0.3 + 1);
   ctrA = 0;
   new_i = nalu_hypre_TAlloc(NALU_HYPRE_Int, n + 1, memory_location);
   new_j = nalu_hypre_TAlloc(NALU_HYPRE_Int, capacity, memory_location);
   new_data = nalu_hypre_TAlloc(NALU_HYPRE_Real, capacity, memory_location);

   idx = nalu_hypre_TAlloc(NALU_HYPRE_Int, m, memory_location);
   data = nalu_hypre_TAlloc(NALU_HYPRE_Real, m, memory_location);

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
            capacity = (NALU_HYPRE_Int)(capacity * EXPAND_FACT + 1);
            new_j = nalu_hypre_TReAlloc_v2(new_j, NALU_HYPRE_Int, tmp,
                                      NALU_HYPRE_Int, capacity, memory_location);
            new_data = nalu_hypre_TReAlloc_v2(new_data, NALU_HYPRE_Real, tmp,
                                         NALU_HYPRE_Real, capacity, memory_location);
         }
         nalu_hypre_TMemcpy(new_j + ctrA, idx, NALU_HYPRE_Int, drop_len, memory_location, memory_location);
         nalu_hypre_TMemcpy(new_data + ctrA, data, NALU_HYPRE_Real, drop_len, memory_location,
                       memory_location);
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
            capacity = (NALU_HYPRE_Int)(capacity * EXPAND_FACT + 1);
            new_j = nalu_hypre_TReAlloc_v2(new_j, NALU_HYPRE_Int, tmp,
                                      NALU_HYPRE_Int, capacity, memory_location);
            new_data = nalu_hypre_TReAlloc_v2(new_data, NALU_HYPRE_Real, tmp,
                                         NALU_HYPRE_Real, capacity, memory_location);
         }
         nalu_hypre_TMemcpy(new_j + ctrA, idx, NALU_HYPRE_Int, drop_len, memory_location, memory_location);
         nalu_hypre_TMemcpy(new_data + ctrA, data, NALU_HYPRE_Real, drop_len, memory_location,
                       memory_location);
         ctrA += drop_len;
         new_i[i + 1] = ctrA;
      }
   }/* end of main loop */
   /* destory data if A own them */
   if (nalu_hypre_CSRMatrixOwnsData(A))
   {
      nalu_hypre_TFree(A_i, memory_location);
      nalu_hypre_TFree(A_j, memory_location);
      nalu_hypre_TFree(A_data, memory_location);
   }

   nalu_hypre_CSRMatrixI(A) = new_i;
   nalu_hypre_CSRMatrixJ(A) = new_j;
   nalu_hypre_CSRMatrixData(A) = new_data;
   nalu_hypre_CSRMatrixNumNonzeros(A) = ctrA;
   nalu_hypre_CSRMatrixOwnsData(A) = 1;

   nalu_hypre_TFree(idx, memory_location);
   nalu_hypre_TFree(data, memory_location);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ILUCSRMatrixInverseSelfPrecondMRGlobal
 *
 * Compute the inverse with MR of original CSR matrix
 * Global(not by each column) and out place version
 * A: the input matrix
 * M: the output matrix
 * droptol: the dropping tolorance
 * tol: when to stop the iteration
 * eps_tol: to avoid divide by 0
 * max_row_nnz: max number of nonzeros per row
 * max_iter: max number of iterations
 * print_level: the print level of this algorithm
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ILUCSRMatrixInverseSelfPrecondMRGlobal(nalu_hypre_CSRMatrix  *matA,
                                             nalu_hypre_CSRMatrix **M,
                                             NALU_HYPRE_Real        droptol,
                                             NALU_HYPRE_Real        tol,
                                             NALU_HYPRE_Real        eps_tol,
                                             NALU_HYPRE_Int         max_row_nnz,
                                             NALU_HYPRE_Int         max_iter,
                                             NALU_HYPRE_Int         print_level)
{
   /* matrix A */
   NALU_HYPRE_Int         *A_i = nalu_hypre_CSRMatrixI(matA);
   NALU_HYPRE_Int         *A_j = nalu_hypre_CSRMatrixJ(matA);
   NALU_HYPRE_Real        *A_data = nalu_hypre_CSRMatrixData(matA);
   NALU_HYPRE_MemoryLocation memory_location = nalu_hypre_CSRMatrixMemoryLocation(matA);

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
   NALU_HYPRE_Int         i, k1, k2;
   NALU_HYPRE_Real        value, trace1, trace2, alpha, r_norm;

   NALU_HYPRE_Int         n = nalu_hypre_CSRMatrixNumRows(matA);

   /* create initial guess and matrix I */
   matM = nalu_hypre_CSRMatrixCreate(n, n, n);
   M_i = nalu_hypre_TAlloc(NALU_HYPRE_Int, n + 1, memory_location);
   M_j = nalu_hypre_TAlloc(NALU_HYPRE_Int, n, memory_location);
   M_data = nalu_hypre_TAlloc(NALU_HYPRE_Real, n, memory_location);

   matI = nalu_hypre_CSRMatrixCreate(n, n, n);
   I_i = nalu_hypre_TAlloc(NALU_HYPRE_Int, n + 1, memory_location);
   I_j = nalu_hypre_TAlloc(NALU_HYPRE_Int, n, memory_location);
   I_data = nalu_hypre_TAlloc(NALU_HYPRE_Real, n, memory_location);

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

/*--------------------------------------------------------------------------
 * nalu_hypre_ILUParCSRInverseNSH
 *
 * Compute inverse with NSH method
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
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ILUParCSRInverseNSH(nalu_hypre_ParCSRMatrix  *A,
                          nalu_hypre_ParCSRMatrix **M,
                          NALU_HYPRE_Real          *droptol,
                          NALU_HYPRE_Real           mr_tol,
                          NALU_HYPRE_Real           nsh_tol,
                          NALU_HYPRE_Real           eps_tol,
                          NALU_HYPRE_Int            mr_max_row_nnz,
                          NALU_HYPRE_Int            nsh_max_row_nnz,
                          NALU_HYPRE_Int            mr_max_iter,
                          NALU_HYPRE_Int            nsh_max_iter,
                          NALU_HYPRE_Int            mr_col_version,
                          NALU_HYPRE_Int            print_level)
{
   /* data slots for matrices */
   nalu_hypre_ParCSRMatrix      *matM = NULL;
   nalu_hypre_ParCSRMatrix      *inM = *M;
   nalu_hypre_ParCSRMatrix      *AM, *MAM;
   NALU_HYPRE_Real              norm, s_norm;
   MPI_Comm                comm = nalu_hypre_ParCSRMatrixComm(A);
   NALU_HYPRE_Int               myid;
   NALU_HYPRE_MemoryLocation    memory_location = nalu_hypre_ParCSRMatrixMemoryLocation(A);

   nalu_hypre_CSRMatrix         *A_diag = nalu_hypre_ParCSRMatrixDiag(A);
   nalu_hypre_CSRMatrix         *M_diag = NULL;
   nalu_hypre_CSRMatrix         *M_offd;
   NALU_HYPRE_Int               *M_offd_i;

   NALU_HYPRE_Real              time_s, time_e;

   NALU_HYPRE_Int               n = nalu_hypre_CSRMatrixNumRows(A_diag);
   NALU_HYPRE_Int               i;

   /* setup */
   nalu_hypre_MPI_Comm_rank(comm, &myid);

   M_offd_i = nalu_hypre_CTAlloc(NALU_HYPRE_Int, n + 1, memory_location);

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
