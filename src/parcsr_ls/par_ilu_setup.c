/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_nalu_hypre_parcsr_ls.h"

/*--------------------------------------------------------------------------
 * nalu_hypre_ILUSetup
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ILUSetup( void               *ilu_vdata,
                nalu_hypre_ParCSRMatrix *A,
                nalu_hypre_ParVector    *f,
                nalu_hypre_ParVector    *u )
{
   MPI_Comm              comm                = nalu_hypre_ParCSRMatrixComm(A);
   NALU_HYPRE_MemoryLocation  memory_location     = nalu_hypre_ParCSRMatrixMemoryLocation(A);
   nalu_hypre_ParILUData     *ilu_data            = (nalu_hypre_ParILUData*) ilu_vdata;
   nalu_hypre_ParILUData     *schur_precond_ilu;
   nalu_hypre_ParNSHData     *schur_solver_nsh;

   /* Pointers to ilu data */
   NALU_HYPRE_Int             logging             = nalu_hypre_ParILUDataLogging(ilu_data);
   NALU_HYPRE_Int             print_level         = nalu_hypre_ParILUDataPrintLevel(ilu_data);
   NALU_HYPRE_Int             ilu_type            = nalu_hypre_ParILUDataIluType(ilu_data);
   NALU_HYPRE_Int             nLU                 = nalu_hypre_ParILUDataNLU(ilu_data);
   NALU_HYPRE_Int             nI                  = nalu_hypre_ParILUDataNI(ilu_data);
   NALU_HYPRE_Int             fill_level          = nalu_hypre_ParILUDataLfil(ilu_data);
   NALU_HYPRE_Int             max_row_elmts       = nalu_hypre_ParILUDataMaxRowNnz(ilu_data);
   NALU_HYPRE_Real           *droptol             = nalu_hypre_ParILUDataDroptol(ilu_data);
   NALU_HYPRE_Int            *CF_marker_array     = nalu_hypre_ParILUDataCFMarkerArray(ilu_data);
   NALU_HYPRE_Int            *perm                = nalu_hypre_ParILUDataPerm(ilu_data);
   NALU_HYPRE_Int            *qperm               = nalu_hypre_ParILUDataQPerm(ilu_data);
   NALU_HYPRE_Real            tol_ddPQ            = nalu_hypre_ParILUDataTolDDPQ(ilu_data);

   /* Pointers to device data, note that they are not NULL only when needed */
#if defined(NALU_HYPRE_USING_CUDA) || defined(NALU_HYPRE_USING_HIP)
   NALU_HYPRE_Int             test_opt            = nalu_hypre_ParILUDataTestOption(ilu_data);
   nalu_hypre_ParCSRMatrix   *Aperm               = nalu_hypre_ParILUDataAperm(ilu_data);
   nalu_hypre_ParCSRMatrix   *R                   = nalu_hypre_ParILUDataR(ilu_data);
   nalu_hypre_ParCSRMatrix   *P                   = nalu_hypre_ParILUDataP(ilu_data);
   nalu_hypre_CSRMatrix      *matALU_d            = nalu_hypre_ParILUDataMatAILUDevice(ilu_data);
   nalu_hypre_CSRMatrix      *matBLU_d            = nalu_hypre_ParILUDataMatBILUDevice(ilu_data);
   nalu_hypre_CSRMatrix      *matSLU_d            = nalu_hypre_ParILUDataMatSILUDevice(ilu_data);
   nalu_hypre_CSRMatrix      *matE_d              = nalu_hypre_ParILUDataMatEDevice(ilu_data);
   nalu_hypre_CSRMatrix      *matF_d              = nalu_hypre_ParILUDataMatFDevice(ilu_data);
   nalu_hypre_Vector         *Ftemp_upper         = NULL;
   nalu_hypre_Vector         *Utemp_lower         = NULL;
   nalu_hypre_Vector         *Adiag_diag          = NULL;
   nalu_hypre_Vector         *Sdiag_diag          = NULL;
#endif

   nalu_hypre_ParCSRMatrix   *matA                = nalu_hypre_ParILUDataMatA(ilu_data);
   nalu_hypre_ParCSRMatrix   *matL                = nalu_hypre_ParILUDataMatL(ilu_data);
   NALU_HYPRE_Real           *matD                = nalu_hypre_ParILUDataMatD(ilu_data);
   nalu_hypre_ParCSRMatrix   *matU                = nalu_hypre_ParILUDataMatU(ilu_data);
   nalu_hypre_ParCSRMatrix   *matmL               = nalu_hypre_ParILUDataMatLModified(ilu_data);
   NALU_HYPRE_Real           *matmD               = nalu_hypre_ParILUDataMatDModified(ilu_data);
   nalu_hypre_ParCSRMatrix   *matmU               = nalu_hypre_ParILUDataMatUModified(ilu_data);
   nalu_hypre_ParCSRMatrix   *matS                = nalu_hypre_ParILUDataMatS(ilu_data);
   NALU_HYPRE_Int             n                   = nalu_hypre_CSRMatrixNumRows(nalu_hypre_ParCSRMatrixDiag(A));
   NALU_HYPRE_Int             reordering_type     = nalu_hypre_ParILUDataReorderingType(ilu_data);
   NALU_HYPRE_Real            nnzS;  /* Total nnz in S */
   NALU_HYPRE_Real            nnzS_offd_local;
   NALU_HYPRE_Real            nnzS_offd;
   NALU_HYPRE_Int             size_C /* Total size of coarse grid */;

   nalu_hypre_ParVector      *Utemp               = NULL;
   nalu_hypre_ParVector      *Ftemp               = NULL;
   nalu_hypre_ParVector      *Xtemp               = NULL;
   nalu_hypre_ParVector      *Ytemp               = NULL;
   nalu_hypre_ParVector      *Ztemp               = NULL;
   NALU_HYPRE_Real           *uext                = NULL;
   NALU_HYPRE_Real           *fext                = NULL;
   nalu_hypre_ParVector      *rhs                 = NULL;
   nalu_hypre_ParVector      *x                   = NULL;

   /* TODO (VPM): Change F_array and U_array variable names */
   nalu_hypre_ParVector      *F_array             = nalu_hypre_ParILUDataF(ilu_data);
   nalu_hypre_ParVector      *U_array             = nalu_hypre_ParILUDataU(ilu_data);
   nalu_hypre_ParVector      *residual            = nalu_hypre_ParILUDataResidual(ilu_data);
   NALU_HYPRE_Real           *rel_res_norms       = nalu_hypre_ParILUDataRelResNorms(ilu_data);

   /* might need for Schur Complement */
   NALU_HYPRE_Int            *u_end                = NULL;
   NALU_HYPRE_Solver          schur_solver         = NULL;
   NALU_HYPRE_Solver          schur_precond        = NULL;
   NALU_HYPRE_Solver          schur_precond_gotten = NULL;

   /* Whether or not to use exact (direct) triangular solves */
   NALU_HYPRE_Int             tri_solve            = nalu_hypre_ParILUDataTriSolve(ilu_data);

   /* help to build external */
   nalu_hypre_ParCSRCommPkg  *comm_pkg;
   NALU_HYPRE_Int             buffer_size;
   NALU_HYPRE_Int             num_sends;
   NALU_HYPRE_Int             send_size;
   NALU_HYPRE_Int             recv_size;
   NALU_HYPRE_Int             num_procs, my_id;

#if defined (NALU_HYPRE_USING_GPU)
   NALU_HYPRE_ExecutionPolicy exec = nalu_hypre_GetExecPolicy1(nalu_hypre_ParCSRMatrixMemoryLocation(A));

   /* TODO (VPM): Placeholder check to avoid -Wunused-variable warning. Remove this! */
   if (exec != NALU_HYPRE_EXEC_DEVICE && exec != NALU_HYPRE_EXEC_HOST)
   {
      nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC, "Need to run either on host or device!");
      return nalu_hypre_error_flag;
   }
#endif

   /* Sanity checks */
#if defined(NALU_HYPRE_USING_CUDA) && !defined(NALU_HYPRE_USING_CUSPARSE)
   {
      nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC, "ILU CUDA build requires cuSPARSE!");
      return nalu_hypre_error_flag;
   }
#elif defined(NALU_HYPRE_USING_HIP) && !defined(NALU_HYPRE_USING_ROCSPARSE)
   {
      nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC, "ILU HIP build requires rocSPARSE!");
      return nalu_hypre_error_flag;
   }
#endif

   /* ----- begin -----*/
   NALU_HYPRE_ANNOTATE_FUNC_BEGIN;
   nalu_hypre_GpuProfilingPushRange("nalu_hypre_ILUSetup");

   nalu_hypre_MPI_Comm_size(comm, &num_procs);
   nalu_hypre_MPI_Comm_rank(comm, &my_id);

#if defined(NALU_HYPRE_USING_CUDA) || defined(NALU_HYPRE_USING_HIP)
   nalu_hypre_CSRMatrixDestroy(matALU_d); matALU_d = NULL;
   nalu_hypre_CSRMatrixDestroy(matSLU_d); matSLU_d = NULL;
   nalu_hypre_CSRMatrixDestroy(matBLU_d); matBLU_d = NULL;
   nalu_hypre_CSRMatrixDestroy(matE_d);   matE_d   = NULL;
   nalu_hypre_CSRMatrixDestroy(matF_d);   matF_d   = NULL;
   nalu_hypre_ParCSRMatrixDestroy(Aperm); Aperm    = NULL;
   nalu_hypre_ParCSRMatrixDestroy(R);     R        = NULL;
   nalu_hypre_ParCSRMatrixDestroy(P);     P        = NULL;

   nalu_hypre_SeqVectorDestroy(nalu_hypre_ParILUDataFTempUpper(ilu_data));
   nalu_hypre_SeqVectorDestroy(nalu_hypre_ParILUDataUTempLower(ilu_data));
   nalu_hypre_ParVectorDestroy(nalu_hypre_ParILUDataXTemp(ilu_data));
   nalu_hypre_ParVectorDestroy(nalu_hypre_ParILUDataYTemp(ilu_data));
   nalu_hypre_ParVectorDestroy(nalu_hypre_ParILUDataZTemp(ilu_data));
   nalu_hypre_SeqVectorDestroy(nalu_hypre_ParILUDataADiagDiag(ilu_data));
   nalu_hypre_SeqVectorDestroy(nalu_hypre_ParILUDataSDiagDiag(ilu_data));

   nalu_hypre_ParILUDataFTempUpper(ilu_data) = NULL;
   nalu_hypre_ParILUDataUTempLower(ilu_data) = NULL;
   nalu_hypre_ParILUDataXTemp(ilu_data)      = NULL;
   nalu_hypre_ParILUDataYTemp(ilu_data)      = NULL;
   nalu_hypre_ParILUDataZTemp(ilu_data)      = NULL;
   nalu_hypre_ParILUDataADiagDiag(ilu_data)  = NULL;
   nalu_hypre_ParILUDataSDiagDiag(ilu_data)  = NULL;
#endif

   /* Free previously allocated data, if any not destroyed */
   nalu_hypre_ParCSRMatrixDestroy(matL);  matL  = NULL;
   nalu_hypre_ParCSRMatrixDestroy(matU);  matU  = NULL;
   nalu_hypre_ParCSRMatrixDestroy(matmL); matmL = NULL;
   nalu_hypre_ParCSRMatrixDestroy(matmU); matmU = NULL;
   nalu_hypre_ParCSRMatrixDestroy(matS);  matS  = NULL;

   nalu_hypre_TFree(matD, NALU_HYPRE_MEMORY_DEVICE);
   nalu_hypre_TFree(matmD, NALU_HYPRE_MEMORY_DEVICE);
   nalu_hypre_TFree(CF_marker_array, NALU_HYPRE_MEMORY_HOST);

   /* clear old l1_norm data, if created */
   nalu_hypre_TFree(nalu_hypre_ParILUDataL1Norms(ilu_data), NALU_HYPRE_MEMORY_HOST);

   /* setup temporary storage
    * first check is they've already here
    */
   nalu_hypre_ParVectorDestroy(nalu_hypre_ParILUDataUTemp(ilu_data));
   nalu_hypre_ParVectorDestroy(nalu_hypre_ParILUDataFTemp(ilu_data));
   nalu_hypre_ParVectorDestroy(nalu_hypre_ParILUDataRhs(ilu_data));
   nalu_hypre_ParVectorDestroy(nalu_hypre_ParILUDataX(ilu_data));
   nalu_hypre_ParVectorDestroy(nalu_hypre_ParILUDataResidual(ilu_data));
   nalu_hypre_TFree(nalu_hypre_ParILUDataUExt(ilu_data), NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(nalu_hypre_ParILUDataFExt(ilu_data), NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(nalu_hypre_ParILUDataUEnd(ilu_data), NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(nalu_hypre_ParILUDataRelResNorms(ilu_data), NALU_HYPRE_MEMORY_HOST);

   nalu_hypre_ParILUDataUTemp(ilu_data) = NULL;
   nalu_hypre_ParILUDataFTemp(ilu_data) = NULL;
   nalu_hypre_ParILUDataRhs(ilu_data) = NULL;
   nalu_hypre_ParILUDataX(ilu_data) = NULL;
   nalu_hypre_ParILUDataResidual(ilu_data) = NULL;

   if (nalu_hypre_ParILUDataSchurSolver(ilu_data))
   {
      switch (ilu_type)
      {
         case 10: case 11: case 40: case 41: case 50:
            NALU_HYPRE_ParCSRGMRESDestroy(nalu_hypre_ParILUDataSchurSolver(ilu_data)); //GMRES for Schur
            break;

         case 20: case 21:
            nalu_hypre_NSHDestroy(nalu_hypre_ParILUDataSchurSolver(ilu_data)); //NSH for Schur
            break;

         default:
            break;
      }
      (nalu_hypre_ParILUDataSchurSolver(ilu_data)) = NULL;
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
      NALU_HYPRE_ILUDestroy(nalu_hypre_ParILUDataSchurPrecond(ilu_data));
      nalu_hypre_ParILUDataSchurPrecond(ilu_data) = NULL;
   }

   /* Create work vectors */
   Utemp = nalu_hypre_ParVectorCreate(nalu_hypre_ParCSRMatrixComm(A),
                                 nalu_hypre_ParCSRMatrixGlobalNumRows(A),
                                 nalu_hypre_ParCSRMatrixRowStarts(A));
   nalu_hypre_ParVectorInitialize(Utemp);
   nalu_hypre_ParILUDataUTemp(ilu_data) = Utemp;

   Ftemp = nalu_hypre_ParVectorCreate(nalu_hypre_ParCSRMatrixComm(A),
                                 nalu_hypre_ParCSRMatrixGlobalNumRows(A),
                                 nalu_hypre_ParCSRMatrixRowStarts(A));
   nalu_hypre_ParVectorInitialize(Ftemp);
   nalu_hypre_ParILUDataFTemp(ilu_data) = Ftemp;

   /* set matrix, solution and rhs pointers */
   matA    = A;
   F_array = f;
   U_array = u;

   /* Create perm array if necessary */
   if (!perm)
   {
      switch (ilu_type)
      {
         case 10: case 11: case 20: case 21: case 30: case 31: case 50:
            /* symmetric */
            nalu_hypre_ILUGetInteriorExteriorPerm(matA, memory_location, &perm, &nLU, reordering_type);
            break;

         case 40: case 41:
            /* ddPQ */
            nalu_hypre_ILUGetPermddPQ(matA, &perm, &qperm, tol_ddPQ, &nLU, &nI, reordering_type);
            break;

         case 0: case 1:
         default:
            /* RCM or none */
            nalu_hypre_ILUGetLocalPerm(matA, &perm, &nLU, reordering_type);
            break;
      }
   }

   /* Factorization */
   switch (ilu_type)
   {
      case 0:
#if defined(NALU_HYPRE_USING_CUDA) || defined(NALU_HYPRE_USING_HIP)
         if (exec == NALU_HYPRE_EXEC_DEVICE)
         {
            if (fill_level == 0)
            {
               /* BJ + device_ilu0() */
               nalu_hypre_ILUSetupILUDevice(0, matA, 0, NULL, perm, perm, n, n, &matBLU_d,
                                       &matS, &matE_d, &matF_d, tri_solve);
            }
            else
            {
#if !defined(NALU_HYPRE_USING_UNIFIED_MEMORY)
               nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC,
                                 "ILUK setup on device runs requires unified memory!");
               return nalu_hypre_error_flag;
#endif

               /* BJ + nalu_hypre_iluk(), setup the device solve */
               nalu_hypre_ILUSetupILUDevice(1, matA, fill_level, NULL, perm, perm, n, n,
                                       &matBLU_d, &matS, &matE_d, &matF_d, tri_solve);
            }
         }
         else
#endif
         {
            /* BJ + nalu_hypre_iluk() */
            nalu_hypre_ILUSetupILUK(matA, fill_level, perm, perm, n, n,
                               &matL, &matD, &matU, &matS, &u_end);
         }
         break;

      case 1:
#if defined(NALU_HYPRE_USING_CUDA) || defined(NALU_HYPRE_USING_HIP)
         if (exec == NALU_HYPRE_EXEC_DEVICE)
         {
#if !defined(NALU_HYPRE_USING_UNIFIED_MEMORY)
            nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC,
                              "ILUT setup on device runs requires unified memory!");
            return nalu_hypre_error_flag;
#endif
            /* BJ + nalu_hypre_ilut(), setup the device solve */
            nalu_hypre_ILUSetupILUDevice(2, matA, max_row_elmts, droptol, perm, perm, n, n,
                                    &matBLU_d, &matS, &matE_d, &matF_d, tri_solve);
         }
         else
#endif
         {
            /* BJ + nalu_hypre_ilut() */
            nalu_hypre_ILUSetupILUT(matA, max_row_elmts, droptol, perm, perm, n, n,
                               &matL, &matD, &matU, &matS, &u_end);
         }
         break;

      case 10:
#if defined(NALU_HYPRE_USING_CUDA) || defined(NALU_HYPRE_USING_HIP)
         if (exec == NALU_HYPRE_EXEC_DEVICE)
         {
            if (fill_level == 0)
            {
               /* GMRES + device_ilu0() - Only support ILU0 */
               nalu_hypre_ILUSetupILUDevice(0, matA, 0, NULL, perm, perm, n, nLU,
                                       &matBLU_d, &matS, &matE_d, &matF_d, 1);
            }
            else
            {
#if !defined(NALU_HYPRE_USING_UNIFIED_MEMORY)
               nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC,
                                 "GMRES+ILUK setup on device runs requires unified memory!");
               return nalu_hypre_error_flag;
#endif

               /* GMRES + nalu_hypre_iluk() */
               nalu_hypre_ILUSetupILUDevice(1, matA, fill_level, NULL, perm, perm,
                                       n, nLU, &matBLU_d, &matS, &matE_d, &matF_d, 1);
            }
         }
         else
#endif
         {
            /* GMRES + nalu_hypre_iluk() */
            nalu_hypre_ILUSetupILUK(matA, fill_level, perm, perm, nLU, nLU,
                               &matL, &matD, &matU, &matS, &u_end);
         }
         break;

      case 11:
#if defined(NALU_HYPRE_USING_CUDA) || defined(NALU_HYPRE_USING_HIP)
         if (exec == NALU_HYPRE_EXEC_DEVICE)
         {
#if !defined(NALU_HYPRE_USING_UNIFIED_MEMORY)
            nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC,
                              "GMRES+ILUT setup on device runs requires unified memory!");
            return nalu_hypre_error_flag;
#endif

            /* GMRES + nalu_hypre_ilut() */
            nalu_hypre_ILUSetupILUDevice(2, matA, max_row_elmts, droptol, perm, perm, n, nLU,
                                    &matBLU_d, &matS, &matE_d, &matF_d, 1);
         }
         else
#endif
         {
            /* GMRES + nalu_hypre_ilut() */
            nalu_hypre_ILUSetupILUT(matA, max_row_elmts, droptol, perm, perm, nLU, nLU,
                               &matL, &matD, &matU, &matS, &u_end);
         }
         break;

      case 20:
#if defined(NALU_HYPRE_USING_GPU) && !defined(NALU_HYPRE_USING_UNIFIED_MEMORY)
         if (exec == NALU_HYPRE_EXEC_DEVICE)
         {
            nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC,
                              "NSH+ILUK setup on device runs requires unified memory!");
            return nalu_hypre_error_flag;
         }
#endif

         /* Newton Schulz Hotelling + nalu_hypre_iluk() */
         nalu_hypre_ILUSetupILUK(matA, fill_level, perm, perm, nLU, nLU,
                            &matL, &matD, &matU, &matS, &u_end);
         break;

      case 21:
#if defined(NALU_HYPRE_USING_GPU) && !defined(NALU_HYPRE_USING_UNIFIED_MEMORY)
         if (exec == NALU_HYPRE_EXEC_DEVICE)
         {
            nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC,
                              "NSH+ILUT setup on device runs requires unified memory!");
            return nalu_hypre_error_flag;
         }
#endif

         /* Newton Schulz Hotelling + nalu_hypre_ilut() */
         nalu_hypre_ILUSetupILUT(matA, max_row_elmts, droptol, perm, perm, nLU, nLU,
                            &matL, &matD, &matU, &matS, &u_end);
         break;

      case 30:
#if defined(NALU_HYPRE_USING_GPU) && !defined(NALU_HYPRE_USING_UNIFIED_MEMORY)
         if (exec == NALU_HYPRE_EXEC_DEVICE)
         {
            nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC,
                              "RAS+ILUK setup on device runs requires unified memory!");
            return nalu_hypre_error_flag;
         }
#endif

         /* RAS + nalu_hypre_iluk() */
         nalu_hypre_ILUSetupILUKRAS(matA, fill_level, perm, nLU,
                               &matL, &matD, &matU);
         break;

      case 31:
#if defined(NALU_HYPRE_USING_GPU) && !defined(NALU_HYPRE_USING_UNIFIED_MEMORY)
         if (exec == NALU_HYPRE_EXEC_DEVICE)
         {
            nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC,
                              "RAS+ILUT setup on device runs requires unified memory!");
            return nalu_hypre_error_flag;
         }
#endif

         /* RAS + nalu_hypre_ilut() */
         nalu_hypre_ILUSetupILUTRAS(matA, max_row_elmts, droptol,
                               perm, nLU, &matL, &matD, &matU);
         break;

      case 40:
#if defined(NALU_HYPRE_USING_GPU) && !defined(NALU_HYPRE_USING_UNIFIED_MEMORY)
         if (exec == NALU_HYPRE_EXEC_DEVICE)
         {
            nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC,
                              "ddPQ+GMRES+ILUK setup on device runs requires unified memory!");
            return nalu_hypre_error_flag;
         }
#endif

         /* ddPQ + GMRES + nalu_hypre_iluk() */
         nalu_hypre_ILUSetupILUK(matA, fill_level, perm, qperm, nLU, nI,
                            &matL, &matD, &matU, &matS, &u_end);
         break;

      case 41:
#if defined(NALU_HYPRE_USING_GPU) && !defined(NALU_HYPRE_USING_UNIFIED_MEMORY)
         if (exec == NALU_HYPRE_EXEC_DEVICE)
         {
            nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC,
                              "ddPQ+GMRES+ILUT setup on device runs requires unified memory!");
            return nalu_hypre_error_flag;
         }
#endif

         /* ddPQ + GMRES + nalu_hypre_ilut() */
         nalu_hypre_ILUSetupILUT(matA, max_row_elmts, droptol, perm, qperm, nLU, nI,
                            &matL, &matD, &matU, &matS, &u_end);
         break;

      case 50:
#if defined(NALU_HYPRE_USING_CUDA) || defined(NALU_HYPRE_USING_HIP)
         if (exec == NALU_HYPRE_EXEC_DEVICE)
         {
#if !defined(NALU_HYPRE_USING_UNIFIED_MEMORY)
            nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC,
                              "GMRES+ILU0-RAP setup on device runs requires unified memory!");
            return nalu_hypre_error_flag;
#endif

            /* RAP + nalu_hypre_modified_ilu0 */
            nalu_hypre_ILUSetupRAPILU0Device(matA, perm, n, nLU,
                                        &Aperm, &matS, &matALU_d, &matBLU_d,
                                        &matSLU_d, &matE_d, &matF_d, test_opt);
         }
         else
#endif
         {
            /* RAP + nalu_hypre_modified_ilu0 */
            nalu_hypre_ILUSetupRAPILU0(matA, perm, n, nLU, &matL, &matD, &matU,
                                  &matmL, &matmD, &matmU, &u_end);
         }
         break;

      default:
#if defined(NALU_HYPRE_USING_CUDA) || defined(NALU_HYPRE_USING_HIP)
         if (exec == NALU_HYPRE_EXEC_DEVICE)
         {
            /* BJ + device_ilu0() */
            nalu_hypre_ILUSetupILUDevice(0, matA, 0, NULL, perm, perm, n, n,
                                    &matBLU_d, &matS, &matE_d, &matF_d, tri_solve);
         }
         else
#endif
         {
            /* BJ + nalu_hypre_ilu0() */
            nalu_hypre_ILUSetupILU0(matA, perm, perm, n, n, &matL,
                               &matD, &matU, &matS, &u_end);
         }
         break;
   }

   /* Create additional temporary vector for iterative triangular solve */
   if (!tri_solve)
   {
      Ztemp = nalu_hypre_ParVectorCreate(nalu_hypre_ParCSRMatrixComm(A),
                                    nalu_hypre_ParCSRMatrixGlobalNumRows(A),
                                    nalu_hypre_ParCSRMatrixRowStarts(A));
      nalu_hypre_ParVectorInitialize(Ztemp);
   }

   /* setup Schur solver - TODO (VPM): merge host and device paths below */
   switch (ilu_type)
   {
      case 0: case 1:
      default:
         break;

      case 10: case 11:
         if (matS)
         {
            /* Create work vectors */
#if defined(NALU_HYPRE_USING_CUDA) || defined(NALU_HYPRE_USING_HIP)
            if (exec == NALU_HYPRE_EXEC_DEVICE)
            {
               Xtemp = nalu_hypre_ParVectorCreate(nalu_hypre_ParCSRMatrixComm(matS),
                                             nalu_hypre_ParCSRMatrixGlobalNumRows(matS),
                                             nalu_hypre_ParCSRMatrixRowStarts(matS));
               nalu_hypre_ParVectorInitialize(Xtemp);

               Ytemp = nalu_hypre_ParVectorCreate(nalu_hypre_ParCSRMatrixComm(matS),
                                             nalu_hypre_ParCSRMatrixGlobalNumRows(matS),
                                             nalu_hypre_ParCSRMatrixRowStarts(matS));
               nalu_hypre_ParVectorInitialize(Ytemp);

               Ftemp_upper = nalu_hypre_SeqVectorCreate(nLU);
               nalu_hypre_VectorOwnsData(Ftemp_upper)   = 0;
               nalu_hypre_VectorData(Ftemp_upper)       = nalu_hypre_VectorData(
                                                        nalu_hypre_ParVectorLocalVector(Ftemp));
               nalu_hypre_SeqVectorInitialize(Ftemp_upper);

               Utemp_lower = nalu_hypre_SeqVectorCreate(n - nLU);
               nalu_hypre_VectorOwnsData(Utemp_lower)   = 0;
               nalu_hypre_VectorData(Utemp_lower)       = nalu_hypre_VectorData(
                                                        nalu_hypre_ParVectorLocalVector(Utemp)) + nLU;
               nalu_hypre_SeqVectorInitialize(Utemp_lower);

               /* create GMRES */
               //            NALU_HYPRE_ParCSRGMRESCreate(comm, &schur_solver);

               nalu_hypre_GMRESFunctions * gmres_functions;

               gmres_functions =
                  nalu_hypre_GMRESFunctionsCreate(
                     nalu_hypre_ParKrylovCAlloc,
                     nalu_hypre_ParKrylovFree,
                     nalu_hypre_ParILUSchurGMRESCommInfoDevice, //parCSR A -> ilu_data
                     nalu_hypre_ParKrylovCreateVector,
                     nalu_hypre_ParKrylovCreateVectorArray,
                     nalu_hypre_ParKrylovDestroyVector,
                     nalu_hypre_ParKrylovMatvecCreate, //parCSR A -- inactive
                     ((tri_solve == 1) ?
                      nalu_hypre_ParILUSchurGMRESMatvecDevice :
                      nalu_hypre_ParILUSchurGMRESMatvecJacIterDevice), //parCSR A -> ilu_data
                     nalu_hypre_ParKrylovMatvecDestroy, //parCSR A -- inactive
                     nalu_hypre_ParKrylovInnerProd,
                     nalu_hypre_ParKrylovCopyVector,
                     nalu_hypre_ParKrylovClearVector,
                     nalu_hypre_ParKrylovScaleVector,
                     nalu_hypre_ParKrylovAxpy,
                     nalu_hypre_ParKrylovIdentitySetup, //parCSR A -- inactive
                     nalu_hypre_ParKrylovIdentity ); //parCSR A -- inactive
               schur_solver = ( (NALU_HYPRE_Solver) nalu_hypre_GMRESCreate( gmres_functions ) );

               /* setup GMRES parameters */
               NALU_HYPRE_GMRESSetKDim            (schur_solver, nalu_hypre_ParILUDataSchurGMRESKDim(ilu_data));
               NALU_HYPRE_GMRESSetMaxIter         (schur_solver,
                                              nalu_hypre_ParILUDataSchurGMRESMaxIter(ilu_data));/* we don't need that many solves */
               NALU_HYPRE_GMRESSetTol             (schur_solver, nalu_hypre_ParILUDataSchurGMRESTol(ilu_data));
               NALU_HYPRE_GMRESSetAbsoluteTol     (schur_solver, nalu_hypre_ParILUDataSchurGMRESAbsoluteTol(ilu_data));
               NALU_HYPRE_GMRESSetLogging         (schur_solver, nalu_hypre_ParILUDataSchurSolverLogging(ilu_data));
               NALU_HYPRE_GMRESSetPrintLevel      (schur_solver,
                                              nalu_hypre_ParILUDataSchurSolverPrintLevel(ilu_data));/* set to zero now, don't print */
               NALU_HYPRE_GMRESSetRelChange       (schur_solver, nalu_hypre_ParILUDataSchurGMRESRelChange(ilu_data));

               /* setup preconditioner parameters */
               /* create Unit precond */
               schur_precond = (NALU_HYPRE_Solver) ilu_vdata;

               /* add preconditioner to solver */
               NALU_HYPRE_GMRESSetPrecond(schur_solver,
                                     (NALU_HYPRE_PtrToSolverFcn) nalu_hypre_ParILUSchurGMRESDummySolveDevice,
                                     (NALU_HYPRE_PtrToSolverFcn) nalu_hypre_ParKrylovIdentitySetup,
                                     schur_precond);

               NALU_HYPRE_GMRESGetPrecond(schur_solver, &schur_precond_gotten);
               if (schur_precond_gotten != (schur_precond))
               {
                  nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC, "Schur complement got bad precond!");
                  nalu_hypre_GpuProfilingPopRange();
                  NALU_HYPRE_ANNOTATE_FUNC_END;

                  return nalu_hypre_error_flag;
               }

               /* need to create working vector rhs and x for Schur System */
               rhs = nalu_hypre_ParVectorCreate(comm,
                                           nalu_hypre_ParCSRMatrixGlobalNumRows(matS),
                                           nalu_hypre_ParCSRMatrixRowStarts(matS));
               nalu_hypre_ParVectorInitialize(rhs);
               x = nalu_hypre_ParVectorCreate(comm,
                                         nalu_hypre_ParCSRMatrixGlobalNumRows(matS),
                                         nalu_hypre_ParCSRMatrixRowStarts(matS));
               nalu_hypre_ParVectorInitialize(x);

               /* setup solver */
               NALU_HYPRE_GMRESSetup(schur_solver,
                                (NALU_HYPRE_Matrix) ilu_vdata,
                                (NALU_HYPRE_Vector) rhs,
                                (NALU_HYPRE_Vector) x);

               /* solve for right-hand-side consists of only 1 */
               nalu_hypre_Vector      *rhs_local = nalu_hypre_ParVectorLocalVector(rhs);
               //NALU_HYPRE_Real        *Xtemp_data  = nalu_hypre_VectorData(Xtemp_local);
               nalu_hypre_SeqVectorSetConstantValues(rhs_local, 1.0);

               /* update ilu_data */
               nalu_hypre_ParILUDataSchurSolver   (ilu_data) = schur_solver;
               nalu_hypre_ParILUDataSchurPrecond  (ilu_data) = schur_precond;
               nalu_hypre_ParILUDataRhs           (ilu_data) = rhs;
               nalu_hypre_ParILUDataX             (ilu_data) = x;
            }
            else
#endif
            {
               /* setup GMRES parameters */
               NALU_HYPRE_ParCSRGMRESCreate(comm, &schur_solver);

               NALU_HYPRE_GMRESSetKDim            (schur_solver, nalu_hypre_ParILUDataSchurGMRESKDim(ilu_data));
               NALU_HYPRE_GMRESSetMaxIter         (schur_solver,
                                              nalu_hypre_ParILUDataSchurGMRESMaxIter(ilu_data));/* we don't need that many solves */
               NALU_HYPRE_GMRESSetTol             (schur_solver, nalu_hypre_ParILUDataSchurGMRESTol(ilu_data));
               NALU_HYPRE_GMRESSetAbsoluteTol     (schur_solver, nalu_hypre_ParILUDataSchurGMRESAbsoluteTol(ilu_data));
               NALU_HYPRE_GMRESSetLogging         (schur_solver, nalu_hypre_ParILUDataSchurSolverLogging(ilu_data));
               NALU_HYPRE_GMRESSetPrintLevel      (schur_solver,
                                              nalu_hypre_ParILUDataSchurSolverPrintLevel(ilu_data));/* set to zero now, don't print */
               NALU_HYPRE_GMRESSetRelChange       (schur_solver, nalu_hypre_ParILUDataSchurGMRESRelChange(ilu_data));

               /* setup preconditioner parameters */
               /* create precond, the default is ILU0 */
               NALU_HYPRE_ILUCreate               (&schur_precond);
               NALU_HYPRE_ILUSetType              (schur_precond, nalu_hypre_ParILUDataSchurPrecondIluType(ilu_data));
               NALU_HYPRE_ILUSetLevelOfFill       (schur_precond, nalu_hypre_ParILUDataSchurPrecondIluLfil(ilu_data));
               NALU_HYPRE_ILUSetMaxNnzPerRow      (schur_precond, nalu_hypre_ParILUDataSchurPrecondIluMaxRowNnz(ilu_data));
               NALU_HYPRE_ILUSetDropThresholdArray(schur_precond, nalu_hypre_ParILUDataSchurPrecondIluDroptol(ilu_data));
               NALU_HYPRE_ILUSetPrintLevel        (schur_precond, nalu_hypre_ParILUDataSchurPrecondPrintLevel(ilu_data));
               NALU_HYPRE_ILUSetTriSolve          (schur_precond, nalu_hypre_ParILUDataSchurPrecondTriSolve(ilu_data));
               NALU_HYPRE_ILUSetMaxIter           (schur_precond, nalu_hypre_ParILUDataSchurPrecondMaxIter(ilu_data));
               NALU_HYPRE_ILUSetLowerJacobiIters  (schur_precond,
                                              nalu_hypre_ParILUDataSchurPrecondLowerJacobiIters(ilu_data));
               NALU_HYPRE_ILUSetUpperJacobiIters  (schur_precond,
                                              nalu_hypre_ParILUDataSchurPrecondUpperJacobiIters(ilu_data));
               NALU_HYPRE_ILUSetTol               (schur_precond, nalu_hypre_ParILUDataSchurPrecondTol(ilu_data));

               /* add preconditioner to solver */
               NALU_HYPRE_GMRESSetPrecond(schur_solver,
                                     (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_ILUSolve,
                                     (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_ILUSetup,
                                     schur_precond);

               NALU_HYPRE_GMRESGetPrecond(schur_solver, &schur_precond_gotten);
               if (schur_precond_gotten != (schur_precond))
               {
                  nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC, "Schur complement got bad precond!");
                  nalu_hypre_GpuProfilingPopRange();
                  NALU_HYPRE_ANNOTATE_FUNC_END;

                  return nalu_hypre_error_flag;
               }

               /* need to create working vector rhs and x for Schur System */
               rhs = nalu_hypre_ParVectorCreate(comm,
                                           nalu_hypre_ParCSRMatrixGlobalNumRows(matS),
                                           nalu_hypre_ParCSRMatrixRowStarts(matS));
               nalu_hypre_ParVectorInitialize(rhs);
               x = nalu_hypre_ParVectorCreate(comm,
                                         nalu_hypre_ParCSRMatrixGlobalNumRows(matS),
                                         nalu_hypre_ParCSRMatrixRowStarts(matS));
               nalu_hypre_ParVectorInitialize(x);

               /* setup solver */
               NALU_HYPRE_GMRESSetup(schur_solver,
                                (NALU_HYPRE_Matrix) matS,
                                (NALU_HYPRE_Vector) rhs,
                                (NALU_HYPRE_Vector) x);

               /* update ilu_data */
               nalu_hypre_ParILUDataSchurSolver   (ilu_data) = schur_solver;
               nalu_hypre_ParILUDataSchurPrecond  (ilu_data) = schur_precond;
               nalu_hypre_ParILUDataRhs           (ilu_data) = rhs;
               nalu_hypre_ParILUDataX             (ilu_data) = x;
            }
         }
         break;

      case 20: case 21:
         if (matS)
         {
            /* approximate inverse preconditioner */
            schur_solver = (NALU_HYPRE_Solver)nalu_hypre_NSHCreate();

            /* set NSH parameters */
            nalu_hypre_NSHSetMaxIter           (schur_solver, nalu_hypre_ParILUDataSchurNSHSolveMaxIter(ilu_data));
            nalu_hypre_NSHSetTol               (schur_solver, nalu_hypre_ParILUDataSchurNSHSolveTol(ilu_data));
            nalu_hypre_NSHSetLogging           (schur_solver, nalu_hypre_ParILUDataSchurSolverLogging(ilu_data));
            nalu_hypre_NSHSetPrintLevel        (schur_solver, nalu_hypre_ParILUDataSchurSolverPrintLevel(ilu_data));
            nalu_hypre_NSHSetDropThresholdArray(schur_solver, nalu_hypre_ParILUDataSchurNSHDroptol(ilu_data));

            nalu_hypre_NSHSetNSHMaxIter        (schur_solver, nalu_hypre_ParILUDataSchurNSHMaxNumIter(ilu_data));
            nalu_hypre_NSHSetNSHMaxRowNnz      (schur_solver, nalu_hypre_ParILUDataSchurNSHMaxRowNnz(ilu_data));
            nalu_hypre_NSHSetNSHTol            (schur_solver, nalu_hypre_ParILUDataSchurNSHTol(ilu_data));

            nalu_hypre_NSHSetMRMaxIter         (schur_solver, nalu_hypre_ParILUDataSchurMRMaxIter(ilu_data));
            nalu_hypre_NSHSetMRMaxRowNnz       (schur_solver, nalu_hypre_ParILUDataSchurMRMaxRowNnz(ilu_data));
            nalu_hypre_NSHSetMRTol             (schur_solver, nalu_hypre_ParILUDataSchurMRTol(ilu_data));
            nalu_hypre_NSHSetColVersion        (schur_solver, nalu_hypre_ParILUDataSchurMRColVersion(ilu_data));

            /* need to create working vector rhs and x for Schur System */
            rhs = nalu_hypre_ParVectorCreate(comm,
                                        nalu_hypre_ParCSRMatrixGlobalNumRows(matS),
                                        nalu_hypre_ParCSRMatrixRowStarts(matS));
            nalu_hypre_ParVectorInitialize(rhs);
            x = nalu_hypre_ParVectorCreate(comm,
                                      nalu_hypre_ParCSRMatrixGlobalNumRows(matS),
                                      nalu_hypre_ParCSRMatrixRowStarts(matS));
            nalu_hypre_ParVectorInitialize(x);

            /* setup solver */
            nalu_hypre_NSHSetup(schur_solver, matS, rhs, x);

            nalu_hypre_ParILUDataSchurSolver(ilu_data) = schur_solver;
            nalu_hypre_ParILUDataRhs        (ilu_data) = rhs;
            nalu_hypre_ParILUDataX          (ilu_data) = x;
         }
         break;

      case 30 : case 31:
         /* now check communication package */
         comm_pkg = nalu_hypre_ParCSRMatrixCommPkg(matA);

         /* create if not yet built */
         if (!comm_pkg)
         {
            nalu_hypre_MatvecCommPkgCreate(matA);
            comm_pkg = nalu_hypre_ParCSRMatrixCommPkg(matA);
         }

         /* create uext and fext */
         num_sends = nalu_hypre_ParCSRCommPkgNumSends(comm_pkg);
         send_size = nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends) -
                     nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, 0);
         recv_size = nalu_hypre_CSRMatrixNumCols(nalu_hypre_ParCSRMatrixOffd(matA));
         buffer_size = send_size > recv_size ? send_size : recv_size;

         /* TODO (VPM): Check these memory locations */
         fext = nalu_hypre_TAlloc(NALU_HYPRE_Real, buffer_size, NALU_HYPRE_MEMORY_HOST);
         uext = nalu_hypre_TAlloc(NALU_HYPRE_Real, buffer_size, NALU_HYPRE_MEMORY_HOST);
         break;

      case 40: case 41:
         if (matS)
         {
            /* setup GMRES parameters */
            NALU_HYPRE_ParCSRGMRESCreate(comm, &schur_solver);

            NALU_HYPRE_GMRESSetKDim            (schur_solver, nalu_hypre_ParILUDataSchurGMRESKDim(ilu_data));
            NALU_HYPRE_GMRESSetMaxIter         (schur_solver,
                                           nalu_hypre_ParILUDataSchurGMRESMaxIter(ilu_data));/* we don't need that many solves */
            NALU_HYPRE_GMRESSetTol             (schur_solver, nalu_hypre_ParILUDataSchurGMRESTol(ilu_data));
            NALU_HYPRE_GMRESSetAbsoluteTol     (schur_solver, nalu_hypre_ParILUDataSchurGMRESAbsoluteTol(ilu_data));
            NALU_HYPRE_GMRESSetLogging         (schur_solver, nalu_hypre_ParILUDataSchurSolverLogging(ilu_data));
            NALU_HYPRE_GMRESSetPrintLevel      (schur_solver,
                                           nalu_hypre_ParILUDataSchurSolverPrintLevel(ilu_data));/* set to zero now, don't print */
            NALU_HYPRE_GMRESSetRelChange       (schur_solver, nalu_hypre_ParILUDataSchurGMRESRelChange(ilu_data));

            /* setup preconditioner parameters */
            /* create precond, the default is ILU0 */
            NALU_HYPRE_ILUCreate               (&schur_precond);
            NALU_HYPRE_ILUSetType              (schur_precond, nalu_hypre_ParILUDataSchurPrecondIluType(ilu_data));
            NALU_HYPRE_ILUSetLevelOfFill       (schur_precond, nalu_hypre_ParILUDataSchurPrecondIluLfil(ilu_data));
            NALU_HYPRE_ILUSetMaxNnzPerRow      (schur_precond, nalu_hypre_ParILUDataSchurPrecondIluMaxRowNnz(ilu_data));
            NALU_HYPRE_ILUSetDropThresholdArray(schur_precond, nalu_hypre_ParILUDataSchurPrecondIluDroptol(ilu_data));
            NALU_HYPRE_ILUSetPrintLevel        (schur_precond, nalu_hypre_ParILUDataSchurPrecondPrintLevel(ilu_data));
            NALU_HYPRE_ILUSetMaxIter           (schur_precond, nalu_hypre_ParILUDataSchurPrecondMaxIter(ilu_data));
            NALU_HYPRE_ILUSetTriSolve          (schur_precond, nalu_hypre_ParILUDataSchurPrecondTriSolve(ilu_data));
            NALU_HYPRE_ILUSetLowerJacobiIters  (schur_precond,
                                           nalu_hypre_ParILUDataSchurPrecondLowerJacobiIters(ilu_data));
            NALU_HYPRE_ILUSetUpperJacobiIters  (schur_precond,
                                           nalu_hypre_ParILUDataSchurPrecondUpperJacobiIters(ilu_data));
            NALU_HYPRE_ILUSetTol               (schur_precond, nalu_hypre_ParILUDataSchurPrecondTol(ilu_data));

            /* add preconditioner to solver */
            NALU_HYPRE_GMRESSetPrecond(schur_solver,
                                  (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_ILUSolve,
                                  (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_ILUSetup,
                                  schur_precond);

            NALU_HYPRE_GMRESGetPrecond(schur_solver, &schur_precond_gotten);
            if (schur_precond_gotten != (schur_precond))
            {
               nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC, "Schur complement got bad precond!");
               nalu_hypre_GpuProfilingPopRange();
               NALU_HYPRE_ANNOTATE_FUNC_END;

               return nalu_hypre_error_flag;
            }

            /* need to create working vector rhs and x for Schur System */
            rhs = nalu_hypre_ParVectorCreate(comm,
                                        nalu_hypre_ParCSRMatrixGlobalNumRows(matS),
                                        nalu_hypre_ParCSRMatrixRowStarts(matS));
            nalu_hypre_ParVectorInitialize(rhs);
            x = nalu_hypre_ParVectorCreate(comm,
                                      nalu_hypre_ParCSRMatrixGlobalNumRows(matS),
                                      nalu_hypre_ParCSRMatrixRowStarts(matS));
            nalu_hypre_ParVectorInitialize(x);

            /* setup solver */
            NALU_HYPRE_GMRESSetup(schur_solver,
                             (NALU_HYPRE_Matrix) matS,
                             (NALU_HYPRE_Vector) rhs,
                             (NALU_HYPRE_Vector) x);

            /* update ilu_data */
            nalu_hypre_ParILUDataSchurSolver   (ilu_data) = schur_solver;
            nalu_hypre_ParILUDataSchurPrecond  (ilu_data) = schur_precond;
            nalu_hypre_ParILUDataRhs           (ilu_data) = rhs;
            nalu_hypre_ParILUDataX             (ilu_data) = x;
         }
         break;

      case 50:
#if defined(NALU_HYPRE_USING_CUDA) || defined(NALU_HYPRE_USING_HIP)
         if (matS && exec == NALU_HYPRE_EXEC_DEVICE)
         {
            Xtemp = nalu_hypre_ParVectorCreate(nalu_hypre_ParCSRMatrixComm(matA),
                                          nalu_hypre_ParCSRMatrixGlobalNumRows(matA),
                                          nalu_hypre_ParCSRMatrixRowStarts(matA));
            nalu_hypre_ParVectorInitialize(Xtemp);

            Ytemp = nalu_hypre_ParVectorCreate(nalu_hypre_ParCSRMatrixComm(matA),
                                          nalu_hypre_ParCSRMatrixGlobalNumRows(matA),
                                          nalu_hypre_ParCSRMatrixRowStarts(matA));
            nalu_hypre_ParVectorInitialize(Ytemp);

            Ftemp_upper = nalu_hypre_SeqVectorCreate(nLU);
            nalu_hypre_VectorOwnsData(Ftemp_upper) = 0;
            nalu_hypre_VectorData(Ftemp_upper) = nalu_hypre_VectorData(nalu_hypre_ParVectorLocalVector(Ftemp));
            nalu_hypre_SeqVectorInitialize(Ftemp_upper);

            Utemp_lower = nalu_hypre_SeqVectorCreate(n - nLU);
            nalu_hypre_VectorOwnsData(Utemp_lower) = 0;
            nalu_hypre_VectorData(Utemp_lower) = nLU +
                                            nalu_hypre_VectorData(nalu_hypre_ParVectorLocalVector(Utemp));
            nalu_hypre_SeqVectorInitialize(Utemp_lower);

            /* create GMRES */
            //            NALU_HYPRE_ParCSRGMRESCreate(comm, &schur_solver);

            nalu_hypre_GMRESFunctions * gmres_functions;

            gmres_functions =
               nalu_hypre_GMRESFunctionsCreate(
                  nalu_hypre_ParKrylovCAlloc,
                  nalu_hypre_ParKrylovFree,
                  nalu_hypre_ParILUSchurGMRESCommInfoDevice, //parCSR A -> ilu_data
                  nalu_hypre_ParKrylovCreateVector,
                  nalu_hypre_ParKrylovCreateVectorArray,
                  nalu_hypre_ParKrylovDestroyVector,
                  nalu_hypre_ParKrylovMatvecCreate, //parCSR A -- inactive
                  nalu_hypre_ParILURAPSchurGMRESMatvecDevice, //parCSR A -> ilu_data
                  nalu_hypre_ParKrylovMatvecDestroy, //parCSR A -- inactive
                  nalu_hypre_ParKrylovInnerProd,
                  nalu_hypre_ParKrylovCopyVector,
                  nalu_hypre_ParKrylovClearVector,
                  nalu_hypre_ParKrylovScaleVector,
                  nalu_hypre_ParKrylovAxpy,
                  nalu_hypre_ParKrylovIdentitySetup, //parCSR A -- inactive
                  nalu_hypre_ParKrylovIdentity ); //parCSR A -- inactive
            schur_solver = (NALU_HYPRE_Solver) nalu_hypre_GMRESCreate(gmres_functions);

            /* setup GMRES parameters */
            /* at least should apply 1 solve */
            if (nalu_hypre_ParILUDataSchurGMRESKDim(ilu_data) == 0)
            {
               nalu_hypre_ParILUDataSchurGMRESKDim(ilu_data)++;
            }
            NALU_HYPRE_GMRESSetKDim            (schur_solver, nalu_hypre_ParILUDataSchurGMRESKDim(ilu_data));
            NALU_HYPRE_GMRESSetMaxIter         (schur_solver,
                                           nalu_hypre_ParILUDataSchurGMRESMaxIter(ilu_data));/* we don't need that many solves */
            NALU_HYPRE_GMRESSetTol             (schur_solver, nalu_hypre_ParILUDataSchurGMRESTol(ilu_data));
            NALU_HYPRE_GMRESSetAbsoluteTol     (schur_solver, nalu_hypre_ParILUDataSchurGMRESAbsoluteTol(ilu_data));
            NALU_HYPRE_GMRESSetLogging         (schur_solver, nalu_hypre_ParILUDataSchurSolverLogging(ilu_data));
            NALU_HYPRE_GMRESSetPrintLevel      (schur_solver,
                                           nalu_hypre_ParILUDataSchurSolverPrintLevel(ilu_data));/* set to zero now, don't print */
            NALU_HYPRE_GMRESSetRelChange       (schur_solver, nalu_hypre_ParILUDataSchurGMRESRelChange(ilu_data));

            /* setup preconditioner parameters */
            /* create Schur precond */
            schur_precond = (NALU_HYPRE_Solver) ilu_vdata;

            /* add preconditioner to solver */
            NALU_HYPRE_GMRESSetPrecond(schur_solver,
                                  (NALU_HYPRE_PtrToSolverFcn) nalu_hypre_ParILURAPSchurGMRESSolveDevice,
                                  (NALU_HYPRE_PtrToSolverFcn) nalu_hypre_ParKrylovIdentitySetup,
                                  schur_precond);
            NALU_HYPRE_GMRESGetPrecond(schur_solver, &schur_precond_gotten);

            if (schur_precond_gotten != (schur_precond))
            {
               nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC, "Schur complement got bad precond!");
               nalu_hypre_GpuProfilingPopRange();
               NALU_HYPRE_ANNOTATE_FUNC_END;

               return nalu_hypre_error_flag;
            }

            /* need to create working vector rhs and x for Schur System */
            rhs = nalu_hypre_ParVectorCreate(comm,
                                        nalu_hypre_ParCSRMatrixGlobalNumRows(matS),
                                        nalu_hypre_ParCSRMatrixRowStarts(matS));
            nalu_hypre_ParVectorInitialize(rhs);
            x = nalu_hypre_ParVectorCreate(comm,
                                      nalu_hypre_ParCSRMatrixGlobalNumRows(matS),
                                      nalu_hypre_ParCSRMatrixRowStarts(matS));
            nalu_hypre_ParVectorInitialize(x);

            /* setup solver */
            NALU_HYPRE_GMRESSetup(schur_solver,
                             (NALU_HYPRE_Matrix) ilu_vdata,
                             (NALU_HYPRE_Vector) rhs,
                             (NALU_HYPRE_Vector) x);

            /* solve for right-hand-side consists of only 1 */
            //nalu_hypre_Vector      *rhs_local = nalu_hypre_ParVectorLocalVector(rhs);
            //NALU_HYPRE_Real        *Xtemp_data  = nalu_hypre_VectorData(Xtemp_local);
            //nalu_hypre_SeqVectorSetConstantValues(rhs_local, 1.0);

            /* Update ilu_data */
            nalu_hypre_ParILUDataSchurSolver(ilu_data)  = schur_solver;
            nalu_hypre_ParILUDataSchurPrecond(ilu_data) = schur_precond;
            nalu_hypre_ParILUDataRhs(ilu_data)          = rhs;
            nalu_hypre_ParILUDataX(ilu_data)            = x;
         }
         else
#endif
         {
            /* Need to create working vector rhs and x for Schur System */
            NALU_HYPRE_Int      m = n - nLU;
            NALU_HYPRE_BigInt   global_start, S_total_rows, S_row_starts[2];
            NALU_HYPRE_BigInt   big_m = (NALU_HYPRE_BigInt) m;

            nalu_hypre_MPI_Allreduce(&big_m, &S_total_rows, 1, NALU_HYPRE_MPI_BIG_INT, nalu_hypre_MPI_SUM, comm);

            if (S_total_rows > 0)
            {
               Xtemp = nalu_hypre_ParVectorCreate(nalu_hypre_ParCSRMatrixComm(matA),
                                             nalu_hypre_ParCSRMatrixGlobalNumRows(matA),
                                             nalu_hypre_ParCSRMatrixRowStarts(matA));
               nalu_hypre_ParVectorInitialize(Xtemp);

               Ytemp = nalu_hypre_ParVectorCreate(nalu_hypre_ParCSRMatrixComm(matA),
                                             nalu_hypre_ParCSRMatrixGlobalNumRows(matA),
                                             nalu_hypre_ParCSRMatrixRowStarts(matA));
               nalu_hypre_ParVectorInitialize(Ytemp);

               nalu_hypre_MPI_Scan(&big_m, &global_start, 1, NALU_HYPRE_MPI_BIG_INT, nalu_hypre_MPI_SUM, comm);
               S_row_starts[0] = global_start - big_m;
               S_row_starts[1] = global_start;

               rhs = nalu_hypre_ParVectorCreate(comm,
                                           S_total_rows,
                                           S_row_starts);
               nalu_hypre_ParVectorInitialize(rhs);

               x = nalu_hypre_ParVectorCreate(comm,
                                         S_total_rows,
                                         S_row_starts);
               nalu_hypre_ParVectorInitialize(x);

               /* create GMRES */
               //            NALU_HYPRE_ParCSRGMRESCreate(comm, &schur_solver);

               nalu_hypre_GMRESFunctions * gmres_functions;

               gmres_functions =
                  nalu_hypre_GMRESFunctionsCreate(
                     nalu_hypre_ParKrylovCAlloc,
                     nalu_hypre_ParKrylovFree,
                     nalu_hypre_ParILURAPSchurGMRESCommInfoHost, //parCSR A -> ilu_data
                     nalu_hypre_ParKrylovCreateVector,
                     nalu_hypre_ParKrylovCreateVectorArray,
                     nalu_hypre_ParKrylovDestroyVector,
                     nalu_hypre_ParKrylovMatvecCreate, //parCSR A -- inactive
                     nalu_hypre_ParILURAPSchurGMRESMatvecHost, //parCSR A -> ilu_data
                     nalu_hypre_ParKrylovMatvecDestroy, //parCSR A -- inactive
                     nalu_hypre_ParKrylovInnerProd,
                     nalu_hypre_ParKrylovCopyVector,
                     nalu_hypre_ParKrylovClearVector,
                     nalu_hypre_ParKrylovScaleVector,
                     nalu_hypre_ParKrylovAxpy,
                     nalu_hypre_ParKrylovIdentitySetup, //parCSR A -- inactive
                     nalu_hypre_ParKrylovIdentity ); //parCSR A -- inactive
               schur_solver = (NALU_HYPRE_Solver) nalu_hypre_GMRESCreate(gmres_functions);

               /* setup GMRES parameters */
               /* at least should apply 1 solve */
               if (nalu_hypre_ParILUDataSchurGMRESKDim(ilu_data) == 0)
               {
                  nalu_hypre_ParILUDataSchurGMRESKDim(ilu_data)++;
               }
               NALU_HYPRE_GMRESSetKDim            (schur_solver, nalu_hypre_ParILUDataSchurGMRESKDim(ilu_data));
               NALU_HYPRE_GMRESSetMaxIter         (schur_solver,
                                              nalu_hypre_ParILUDataSchurGMRESMaxIter(ilu_data));/* we don't need that many solves */
               NALU_HYPRE_GMRESSetTol             (schur_solver, nalu_hypre_ParILUDataSchurGMRESTol(ilu_data));
               NALU_HYPRE_GMRESSetAbsoluteTol     (schur_solver, nalu_hypre_ParILUDataSchurGMRESAbsoluteTol(ilu_data));
               NALU_HYPRE_GMRESSetLogging         (schur_solver, nalu_hypre_ParILUDataSchurSolverLogging(ilu_data));
               NALU_HYPRE_GMRESSetPrintLevel      (schur_solver,
                                              nalu_hypre_ParILUDataSchurSolverPrintLevel(ilu_data));/* set to zero now, don't print */
               NALU_HYPRE_GMRESSetRelChange       (schur_solver, nalu_hypre_ParILUDataSchurGMRESRelChange(ilu_data));

               /* setup preconditioner parameters */
               /* create Schur precond */
               schur_precond = (NALU_HYPRE_Solver) ilu_vdata;

               /* add preconditioner to solver */
               NALU_HYPRE_GMRESSetPrecond(schur_solver,
                                     (NALU_HYPRE_PtrToSolverFcn) nalu_hypre_ParILURAPSchurGMRESSolveHost,
                                     (NALU_HYPRE_PtrToSolverFcn) nalu_hypre_ParKrylovIdentitySetup,
                                     schur_precond);
               NALU_HYPRE_GMRESGetPrecond(schur_solver, &schur_precond_gotten);
               if (schur_precond_gotten != (schur_precond))
               {
                  nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC, "Schur complement got bad precond!");
                  nalu_hypre_GpuProfilingPopRange();
                  NALU_HYPRE_ANNOTATE_FUNC_END;

                  return nalu_hypre_error_flag;
               }

               /* setup solver */
               NALU_HYPRE_GMRESSetup(schur_solver,
                                (NALU_HYPRE_Matrix) ilu_vdata,
                                (NALU_HYPRE_Vector) rhs,
                                (NALU_HYPRE_Vector) x);

               /* solve for right-hand-side consists of only 1 */
               //nalu_hypre_Vector      *rhs_local = nalu_hypre_ParVectorLocalVector(rhs);
               //NALU_HYPRE_Real        *Xtemp_data  = nalu_hypre_VectorData(Xtemp_local);
               //nalu_hypre_SeqVectorSetConstantValues(rhs_local, 1.0);
            } /* if (S_total_rows > 0) */

            /* Update ilu_data */
            nalu_hypre_ParILUDataSchurSolver(ilu_data)  = schur_solver;
            nalu_hypre_ParILUDataSchurPrecond(ilu_data) = schur_precond;
            nalu_hypre_ParILUDataRhs(ilu_data)          = rhs;
            nalu_hypre_ParILUDataX(ilu_data)            = x;
         }
         break;
   }

   /* set pointers to ilu data */
   /* set device data pointers */
#if defined(NALU_HYPRE_USING_CUDA) || defined(NALU_HYPRE_USING_HIP)
   nalu_hypre_ParILUDataMatAILUDevice(ilu_data) = matALU_d;
   nalu_hypre_ParILUDataMatBILUDevice(ilu_data) = matBLU_d;
   nalu_hypre_ParILUDataMatSILUDevice(ilu_data) = matSLU_d;
   nalu_hypre_ParILUDataMatEDevice(ilu_data)    = matE_d;
   nalu_hypre_ParILUDataMatFDevice(ilu_data)    = matF_d;
   nalu_hypre_ParILUDataAperm(ilu_data)         = Aperm;
   nalu_hypre_ParILUDataR(ilu_data)             = R;
   nalu_hypre_ParILUDataP(ilu_data)             = P;
   nalu_hypre_ParILUDataFTempUpper(ilu_data)    = Ftemp_upper;
   nalu_hypre_ParILUDataUTempLower(ilu_data)    = Utemp_lower;
   nalu_hypre_ParILUDataADiagDiag(ilu_data)     = Adiag_diag;
   nalu_hypre_ParILUDataSDiagDiag(ilu_data)     = Sdiag_diag;
#endif

   /* Set pointers to ilu data */
   nalu_hypre_ParILUDataMatA(ilu_data)          = matA;
   nalu_hypre_ParILUDataXTemp(ilu_data)         = Xtemp;
   nalu_hypre_ParILUDataYTemp(ilu_data)         = Ytemp;
   nalu_hypre_ParILUDataZTemp(ilu_data)         = Ztemp;
   nalu_hypre_ParILUDataF(ilu_data)             = F_array;
   nalu_hypre_ParILUDataU(ilu_data)             = U_array;
   nalu_hypre_ParILUDataMatL(ilu_data)          = matL;
   nalu_hypre_ParILUDataMatD(ilu_data)          = matD;
   nalu_hypre_ParILUDataMatU(ilu_data)          = matU;
   nalu_hypre_ParILUDataMatLModified(ilu_data)  = matmL;
   nalu_hypre_ParILUDataMatDModified(ilu_data)  = matmD;
   nalu_hypre_ParILUDataMatUModified(ilu_data)  = matmU;
   nalu_hypre_ParILUDataMatS(ilu_data)          = matS;
   nalu_hypre_ParILUDataCFMarkerArray(ilu_data) = CF_marker_array;
   nalu_hypre_ParILUDataPerm(ilu_data)          = perm;
   nalu_hypre_ParILUDataQPerm(ilu_data)         = qperm;
   nalu_hypre_ParILUDataNLU(ilu_data)           = nLU;
   nalu_hypre_ParILUDataNI(ilu_data)            = nI;
   nalu_hypre_ParILUDataUEnd(ilu_data)          = u_end;
   nalu_hypre_ParILUDataUExt(ilu_data)          = uext;
   nalu_hypre_ParILUDataFExt(ilu_data)          = fext;

   /* compute operator complexity */
   nalu_hypre_ParCSRMatrixSetDNumNonzeros(matA);
   nnzS = 0.0;

   /* size_C is the size of global coarse grid, upper left part */
   size_C = nalu_hypre_ParCSRMatrixGlobalNumRows(matA);

   /* switch to compute complexity */
#if defined(NALU_HYPRE_USING_CUDA) || defined(NALU_HYPRE_USING_HIP)
   NALU_HYPRE_Int nnzBEF = 0;
   NALU_HYPRE_Int nnzG; /* Global nnz */

   if (ilu_type == 0 && fill_level == 0)
   {
      /* The nnz is for sure 1.0 in this case */
      nalu_hypre_ParILUDataOperatorComplexity(ilu_data) =  1.0;
   }
   else if (ilu_type == 10 && fill_level == 0)
   {
      /* The nnz is the sum of different parts */
      if (matBLU_d)
      {
         nnzBEF  += nalu_hypre_CSRMatrixNumNonzeros(matBLU_d);
      }
      if (matE_d)
      {
         nnzBEF  += nalu_hypre_CSRMatrixNumNonzeros(matE_d);
      }
      if (matF_d)
      {
         nnzBEF  += nalu_hypre_CSRMatrixNumNonzeros(matF_d);
      }
      nalu_hypre_MPI_Allreduce(&nnzBEF, &nnzG, 1, NALU_HYPRE_MPI_INT, nalu_hypre_MPI_SUM, comm);
      if (matS)
      {
         nalu_hypre_ParCSRMatrixSetDNumNonzeros(matS);
         nnzS = nalu_hypre_ParCSRMatrixDNumNonzeros(matS);
         /* if we have Schur system need to reduce it from size_C */
      }
      nalu_hypre_ParILUDataOperatorComplexity(ilu_data) =  ((NALU_HYPRE_Real)nnzG + nnzS) /
                                                      nalu_hypre_ParCSRMatrixDNumNonzeros(matA);
   }
   else if (ilu_type == 50)
   {
      nalu_hypre_ParILUDataOperatorComplexity(ilu_data) =  1.0;
   }
   else if (ilu_type == 0 || ilu_type == 1 || ilu_type == 10 || ilu_type == 11)
   {
      if (matBLU_d)
      {
         nnzBEF  += nalu_hypre_CSRMatrixNumNonzeros(matBLU_d);
      }
      if (matE_d)
      {
         nnzBEF  += nalu_hypre_CSRMatrixNumNonzeros(matE_d);
      }
      if (matF_d)
      {
         nnzBEF  += nalu_hypre_CSRMatrixNumNonzeros(matF_d);
      }
      nalu_hypre_MPI_Allreduce(&nnzBEF, &nnzG, 1, NALU_HYPRE_MPI_INT, nalu_hypre_MPI_SUM, comm);
      if (matS)
      {
         nalu_hypre_ParCSRMatrixSetDNumNonzeros(matS);
         nnzS = nalu_hypre_ParCSRMatrixDNumNonzeros(matS);
         /* if we have Schur system need to reduce it from size_C */
      }
      nalu_hypre_ParILUDataOperatorComplexity(ilu_data) =  ((NALU_HYPRE_Real)nnzG + nnzS) /
                                                      nalu_hypre_ParCSRMatrixDNumNonzeros(matA);
   }
   else
   {
#endif
      if (matS)
      {
         nalu_hypre_ParCSRMatrixSetDNumNonzeros(matS);
         nnzS = nalu_hypre_ParCSRMatrixDNumNonzeros(matS);

         /* If we have Schur system need to reduce it from size_C */
         size_C -= nalu_hypre_ParCSRMatrixGlobalNumRows(matS);
         switch (ilu_type)
         {
            case 10: case 11: case 40: case 41: case 50:
               /* Now we need to compute the preconditioner */
               schur_precond_ilu = (nalu_hypre_ParILUData*) (nalu_hypre_ParILUDataSchurPrecond(ilu_data));

               /* borrow i for local nnz of S */
               nnzS_offd_local = nalu_hypre_CSRMatrixNumNonzeros(nalu_hypre_ParCSRMatrixOffd(matS));
               nalu_hypre_MPI_Allreduce(&nnzS_offd_local, &nnzS_offd, 1, NALU_HYPRE_MPI_REAL,
                                   nalu_hypre_MPI_SUM, comm);
               nnzS = nnzS * nalu_hypre_ParILUDataOperatorComplexity(schur_precond_ilu) + nnzS_offd;
               break;

            case 20: case 21:
               schur_solver_nsh = (nalu_hypre_ParNSHData*) nalu_hypre_ParILUDataSchurSolver(ilu_data);
               nnzS *= nalu_hypre_ParNSHDataOperatorComplexity(schur_solver_nsh);
               break;

            default:
               break;
         }
      }

      nalu_hypre_ParILUDataOperatorComplexity(ilu_data) = ((NALU_HYPRE_Real)size_C + nnzS +
                                                      nalu_hypre_ParCSRMatrixDNumNonzeros(matL) +
                                                      nalu_hypre_ParCSRMatrixDNumNonzeros(matU)) /
                                                     nalu_hypre_ParCSRMatrixDNumNonzeros(matA);
#if defined(NALU_HYPRE_USING_CUDA) || defined(NALU_HYPRE_USING_HIP)
   }
#endif
   if ((my_id == 0) && (print_level > 0))
   {
      nalu_hypre_printf("ILU SETUP: operator complexity = %f  \n",
                   nalu_hypre_ParILUDataOperatorComplexity(ilu_data));
   }

   if (logging > 1)
   {
      residual =
         nalu_hypre_ParVectorCreate(nalu_hypre_ParCSRMatrixComm(matA),
                               nalu_hypre_ParCSRMatrixGlobalNumRows(matA),
                               nalu_hypre_ParCSRMatrixRowStarts(matA) );
      nalu_hypre_ParVectorInitialize(residual);
      nalu_hypre_ParILUDataResidual(ilu_data) = residual;
   }
   else
   {
      nalu_hypre_ParILUDataResidual(ilu_data) = NULL;
   }
   rel_res_norms = nalu_hypre_CTAlloc(NALU_HYPRE_Real,
                                 nalu_hypre_ParILUDataMaxIter(ilu_data),
                                 NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_ParILUDataRelResNorms(ilu_data) = rel_res_norms;

   nalu_hypre_GpuProfilingPopRange();
   NALU_HYPRE_ANNOTATE_FUNC_END;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ParILUExtractEBFC
 *
 * Extract submatrix from diagonal part of A into
 *    | B F |
 *    | E C |
 *
 * A = input matrix
 * perm = permutation array indicating ordering of rows. Perm could come from a
 *    CF_marker array or a reordering routine.
 * qperm = permutation array indicating ordering of columns
 * Bp = pointer to the output B matrix.
 * Cp = pointer to the output C matrix.
 * Ep = pointer to the output E matrix.
 * Fp = pointer to the output F matrix.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ParILUExtractEBFC(nalu_hypre_CSRMatrix   *A_diag,
                        NALU_HYPRE_Int          nLU,
                        nalu_hypre_CSRMatrix  **Bp,
                        nalu_hypre_CSRMatrix  **Cp,
                        nalu_hypre_CSRMatrix  **Ep,
                        nalu_hypre_CSRMatrix  **Fp)
{
   /* Get necessary slots */
   NALU_HYPRE_Int            n                = nalu_hypre_CSRMatrixNumRows(A_diag);
   NALU_HYPRE_Int            nnz_A_diag       = nalu_hypre_CSRMatrixNumNonzeros(A_diag);
   NALU_HYPRE_MemoryLocation memory_location  = nalu_hypre_CSRMatrixMemoryLocation(A_diag);

   nalu_hypre_CSRMatrix     *B = NULL;
   nalu_hypre_CSRMatrix     *C = NULL;
   nalu_hypre_CSRMatrix     *E = NULL;
   nalu_hypre_CSRMatrix     *F = NULL;
   NALU_HYPRE_Int            i, j, row, col;

   nalu_hypre_assert(nLU >= 0 && nLU <= n);

   if (nLU == n)
   {
      /* No Schur complement */
      B = nalu_hypre_CSRMatrixCreate(n, n, nnz_A_diag);
      C = nalu_hypre_CSRMatrixCreate(0, 0, 0);
      E = nalu_hypre_CSRMatrixCreate(0, 0, 0);
      F = nalu_hypre_CSRMatrixCreate(0, 0, 0);

      nalu_hypre_CSRMatrixInitialize_v2(B, 0, memory_location);
      nalu_hypre_CSRMatrixInitialize_v2(C, 0, memory_location);
      nalu_hypre_CSRMatrixInitialize_v2(E, 0, memory_location);
      nalu_hypre_CSRMatrixInitialize_v2(F, 0, memory_location);

      nalu_hypre_CSRMatrixCopy(A_diag, B, 1);
   }
   else if (nLU == 0)
   {
      /* All Schur complement */
      C = nalu_hypre_CSRMatrixCreate(n, n, nnz_A_diag);
      B = nalu_hypre_CSRMatrixCreate(0, 0, 0);
      E = nalu_hypre_CSRMatrixCreate(0, 0, 0);
      F = nalu_hypre_CSRMatrixCreate(0, 0, 0);

      nalu_hypre_CSRMatrixInitialize_v2(C, 0, memory_location);
      nalu_hypre_CSRMatrixInitialize_v2(B, 0, memory_location);
      nalu_hypre_CSRMatrixInitialize_v2(E, 0, memory_location);
      nalu_hypre_CSRMatrixInitialize_v2(F, 0, memory_location);

      nalu_hypre_CSRMatrixCopy(A_diag, C, 1);
   }
   else
   {
      /* Has schur complement */
      NALU_HYPRE_Int         m = n - nLU;
      NALU_HYPRE_Int         capacity_B;
      NALU_HYPRE_Int         capacity_E;
      NALU_HYPRE_Int         capacity_F;
      NALU_HYPRE_Int         capacity_C;
      NALU_HYPRE_Int         ctrB;
      NALU_HYPRE_Int         ctrC;
      NALU_HYPRE_Int         ctrE;
      NALU_HYPRE_Int         ctrF;

      NALU_HYPRE_Int        *B_i    = NULL;
      NALU_HYPRE_Int        *C_i    = NULL;
      NALU_HYPRE_Int        *E_i    = NULL;
      NALU_HYPRE_Int        *F_i    = NULL;
      NALU_HYPRE_Int        *B_j    = NULL;
      NALU_HYPRE_Int        *C_j    = NULL;
      NALU_HYPRE_Int        *E_j    = NULL;
      NALU_HYPRE_Int        *F_j    = NULL;
      NALU_HYPRE_Complex    *B_data = NULL;
      NALU_HYPRE_Complex    *C_data = NULL;
      NALU_HYPRE_Complex    *E_data = NULL;
      NALU_HYPRE_Complex    *F_data = NULL;

      nalu_hypre_CSRMatrix  *h_A_diag;
      NALU_HYPRE_Int        *A_diag_i;
      NALU_HYPRE_Int        *A_diag_j;
      NALU_HYPRE_Complex    *A_diag_data;

      /* Create/Get host pointer for A_diag */
      h_A_diag = (nalu_hypre_GetActualMemLocation(memory_location) == nalu_hypre_MEMORY_DEVICE) ?
                 nalu_hypre_CSRMatrixClone_v2(A_diag, 1, NALU_HYPRE_MEMORY_HOST) : A_diag;
      A_diag_i = nalu_hypre_CSRMatrixI(h_A_diag);
      A_diag_j = nalu_hypre_CSRMatrixJ(h_A_diag);
      A_diag_data = nalu_hypre_CSRMatrixData(h_A_diag);

      /* Estimate # of nonzeros */
      capacity_B = (NALU_HYPRE_Int) (nLU + nalu_hypre_ceil(nnz_A_diag * 1.0 * nLU / n * nLU / n));
      capacity_C = (NALU_HYPRE_Int) (m + nalu_hypre_ceil(nnz_A_diag * 1.0 * m / n * m / n));
      capacity_E = (NALU_HYPRE_Int) (nalu_hypre_min(m, nLU) + nalu_hypre_ceil(nnz_A_diag * 1.0 * nLU / n * m / n));
      capacity_F = capacity_E;

      /* Create CSRMatrices */
      B = nalu_hypre_CSRMatrixCreate(nLU, nLU, capacity_B);
      C = nalu_hypre_CSRMatrixCreate(m, m, capacity_C);
      E = nalu_hypre_CSRMatrixCreate(m, nLU, capacity_E);
      F = nalu_hypre_CSRMatrixCreate(nLU, m, capacity_F);

      /* Initialize matrices on the host */
      nalu_hypre_CSRMatrixInitialize_v2(B, 0, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_CSRMatrixInitialize_v2(C, 0, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_CSRMatrixInitialize_v2(E, 0, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_CSRMatrixInitialize_v2(F, 0, NALU_HYPRE_MEMORY_HOST);

      /* Access pointers */
      B_i    = nalu_hypre_CSRMatrixI(B);
      B_j    = nalu_hypre_CSRMatrixJ(B);
      B_data = nalu_hypre_CSRMatrixData(B);

      C_i    = nalu_hypre_CSRMatrixI(C);
      C_j    = nalu_hypre_CSRMatrixJ(C);
      C_data = nalu_hypre_CSRMatrixData(C);

      E_i    = nalu_hypre_CSRMatrixI(E);
      E_j    = nalu_hypre_CSRMatrixJ(E);
      E_data = nalu_hypre_CSRMatrixData(E);

      F_i    = nalu_hypre_CSRMatrixI(F);
      F_j    = nalu_hypre_CSRMatrixJ(F);
      F_data = nalu_hypre_CSRMatrixData(F);

      ctrB = ctrC = ctrE = ctrF = 0;

      /* Loop to copy data */
      /* B and F first */
      for (i = 0; i < nLU; i++)
      {
         B_i[i] = ctrB;
         F_i[i] = ctrF;
         for (j = A_diag_i[i]; j < A_diag_i[i + 1]; j++)
         {
            col = A_diag_j[j];
            if (col >= nLU)
            {
               break;
            }
            B_j[ctrB] = col;
            B_data[ctrB++] = A_diag_data[j];
            /* check capacity */
            if (ctrB >= capacity_B)
            {
               NALU_HYPRE_Int tmp;
               tmp = capacity_B;
               capacity_B = (NALU_HYPRE_Int)(capacity_B * EXPAND_FACT + 1);
               B_j = nalu_hypre_TReAlloc_v2(B_j, NALU_HYPRE_Int, tmp, NALU_HYPRE_Int,
                                       capacity_B, NALU_HYPRE_MEMORY_HOST);
               B_data = nalu_hypre_TReAlloc_v2(B_data, NALU_HYPRE_Complex, tmp, NALU_HYPRE_Complex,
                                          capacity_B, NALU_HYPRE_MEMORY_HOST);
            }
         }
         for (; j < A_diag_i[i + 1]; j++)
         {
            col = A_diag_j[j];
            col = col - nLU;
            F_j[ctrF] = col;
            F_data[ctrF++] = A_diag_data[j];
            if (ctrF >= capacity_F)
            {
               NALU_HYPRE_Int tmp;
               tmp = capacity_F;
               capacity_F = (NALU_HYPRE_Int)(capacity_F * EXPAND_FACT + 1);
               F_j = nalu_hypre_TReAlloc_v2(F_j, NALU_HYPRE_Int, tmp, NALU_HYPRE_Int,
                                       capacity_F, NALU_HYPRE_MEMORY_HOST);
               F_data = nalu_hypre_TReAlloc_v2(F_data, NALU_HYPRE_Complex, tmp, NALU_HYPRE_Complex,
                                          capacity_F, NALU_HYPRE_MEMORY_HOST);
            }
         }
      }
      B_i[nLU] = ctrB;
      F_i[nLU] = ctrF;

      /* E and C afterward */
      for (i = nLU; i < n; i++)
      {
         row = i - nLU;
         E_i[row] = ctrE;
         C_i[row] = ctrC;
         for (j = A_diag_i[i]; j < A_diag_i[i + 1]; j++)
         {
            col = A_diag_j[j];
            if (col >= nLU)
            {
               break;
            }
            E_j[ctrE] = col;
            E_data[ctrE++] = A_diag_data[j];
            /* check capacity */
            if (ctrE >= capacity_E)
            {
               NALU_HYPRE_Int tmp;
               tmp = capacity_E;
               capacity_E = (NALU_HYPRE_Int)(capacity_E * EXPAND_FACT + 1);
               E_j = nalu_hypre_TReAlloc_v2(E_j, NALU_HYPRE_Int, tmp, NALU_HYPRE_Int,
                                       capacity_E, NALU_HYPRE_MEMORY_HOST);
               E_data = nalu_hypre_TReAlloc_v2(E_data, NALU_HYPRE_Complex, tmp, NALU_HYPRE_Complex,
                                          capacity_E, NALU_HYPRE_MEMORY_HOST);
            }
         }
         for (; j < A_diag_i[i + 1]; j++)
         {
            col = A_diag_j[j];
            col = col - nLU;
            C_j[ctrC] = col;
            C_data[ctrC++] = A_diag_data[j];
            if (ctrC >= capacity_C)
            {
               NALU_HYPRE_Int tmp;
               tmp = capacity_C;
               capacity_C = (NALU_HYPRE_Int)(capacity_C * EXPAND_FACT + 1);
               C_j = nalu_hypre_TReAlloc_v2(C_j, NALU_HYPRE_Int, tmp, NALU_HYPRE_Int,
                                       capacity_C, NALU_HYPRE_MEMORY_HOST);
               C_data = nalu_hypre_TReAlloc_v2(C_data, NALU_HYPRE_Complex, tmp, NALU_HYPRE_Complex,
                                          capacity_C, NALU_HYPRE_MEMORY_HOST);
            }
         }
      }
      E_i[m] = ctrE;
      C_i[m] = ctrC;

      nalu_hypre_assert((ctrB + ctrC + ctrE + ctrF) == nnz_A_diag);

      /* Update pointers */
      nalu_hypre_CSRMatrixJ(B)           = B_j;
      nalu_hypre_CSRMatrixData(B)        = B_data;
      nalu_hypre_CSRMatrixNumNonzeros(B) = ctrB;

      nalu_hypre_CSRMatrixJ(C)           = C_j;
      nalu_hypre_CSRMatrixData(C)        = C_data;
      nalu_hypre_CSRMatrixNumNonzeros(C) = ctrC;

      nalu_hypre_CSRMatrixJ(E)           = E_j;
      nalu_hypre_CSRMatrixData(E)        = E_data;
      nalu_hypre_CSRMatrixNumNonzeros(E) = ctrE;

      nalu_hypre_CSRMatrixJ(F)           = F_j;
      nalu_hypre_CSRMatrixData(F)        = F_data;
      nalu_hypre_CSRMatrixNumNonzeros(F) = ctrF;

      /* Migrate to final memory location */
      nalu_hypre_CSRMatrixMigrate(B, memory_location);
      nalu_hypre_CSRMatrixMigrate(C, memory_location);
      nalu_hypre_CSRMatrixMigrate(E, memory_location);
      nalu_hypre_CSRMatrixMigrate(F, memory_location);

      /* Free memory */
      if (h_A_diag != A_diag)
      {
         nalu_hypre_CSRMatrixDestroy(h_A_diag);
      }
   }

   /* Set output pointers */
   *Bp = B;
   *Cp = C;
   *Ep = E;
   *Fp = F;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ParILURAPReorder
 *
 * Reorder matrix A based on local permutation, i.e., combine local
 * permutation into global permutation)
 *
 * WARNING: We don't put diagonal to the first entry of each row
 *
 * A = input matrix
 * perm = permutation array indicating ordering of rows.
 *        Perm could come from a CF_marker array or a reordering routine.
 * rqperm = reverse permutation array indicating ordering of columns
 * A_pq = pointer to the output par CSR matrix.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ParILURAPReorder(nalu_hypre_ParCSRMatrix  *A,
                       NALU_HYPRE_Int           *perm,
                       NALU_HYPRE_Int           *rqperm,
                       nalu_hypre_ParCSRMatrix **A_pq)
{
   /* Get necessary slots */
   MPI_Comm             comm            = nalu_hypre_ParCSRMatrixComm(A);
   nalu_hypre_CSRMatrix     *A_diag          = nalu_hypre_ParCSRMatrixDiag(A);
   NALU_HYPRE_Int            n               = nalu_hypre_CSRMatrixNumRows(A_diag);
   NALU_HYPRE_MemoryLocation memory_location = nalu_hypre_ParCSRMatrixMemoryLocation(A);

   /* Permutation matrices */
   nalu_hypre_ParCSRMatrix  *P, *Q, *PAQ, *PA;
   nalu_hypre_CSRMatrix     *P_diag, *Q_diag;
   NALU_HYPRE_Int           *P_diag_i, *P_diag_j, *Q_diag_i, *Q_diag_j;
   NALU_HYPRE_Complex       *P_diag_data, *Q_diag_data;
   NALU_HYPRE_Int           *h_perm, *h_rqperm;

   /* Local variables */
   NALU_HYPRE_Int            i;

   /* Trivial case */
   if (!perm && !rqperm)
   {
      *A_pq = nalu_hypre_ParCSRMatrixClone(A, 1);

      return nalu_hypre_error_flag;
   }
   else if (!perm && rqperm)
   {
      nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_ARG, "(!perm && rqperm) should not be possible!");
   }
   else if (perm && !rqperm)
   {
      nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_ARG, "(perm && !rqperm) should not be possible!");
   }

   /* Create permutation matrices P = I(perm,:) and Q(rqperm,:), such that Apq = PAQ */
   P = nalu_hypre_ParCSRMatrixCreate(comm,
                                nalu_hypre_ParCSRMatrixGlobalNumRows(A),
                                nalu_hypre_ParCSRMatrixGlobalNumRows(A),
                                nalu_hypre_ParCSRMatrixRowStarts(A),
                                nalu_hypre_ParCSRMatrixColStarts(A),
                                0,
                                n,
                                0);

   Q = nalu_hypre_ParCSRMatrixCreate(comm,
                                nalu_hypre_ParCSRMatrixGlobalNumRows(A),
                                nalu_hypre_ParCSRMatrixGlobalNumRows(A),
                                nalu_hypre_ParCSRMatrixRowStarts(A),
                                nalu_hypre_ParCSRMatrixColStarts(A),
                                0,
                                n,
                                0);

   nalu_hypre_ParCSRMatrixInitialize_v2(P, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_ParCSRMatrixInitialize_v2(Q, NALU_HYPRE_MEMORY_HOST);

   P_diag      = nalu_hypre_ParCSRMatrixDiag(P);
   Q_diag      = nalu_hypre_ParCSRMatrixDiag(Q);

   P_diag_i    = nalu_hypre_CSRMatrixI(P_diag);
   P_diag_j    = nalu_hypre_CSRMatrixJ(P_diag);
   P_diag_data = nalu_hypre_CSRMatrixData(P_diag);

   Q_diag_i    = nalu_hypre_CSRMatrixI(Q_diag);
   Q_diag_j    = nalu_hypre_CSRMatrixJ(Q_diag);
   Q_diag_data = nalu_hypre_CSRMatrixData(Q_diag);

   /* Set/Move permutation vectors on host */
   if (nalu_hypre_GetActualMemLocation(memory_location) == nalu_hypre_MEMORY_DEVICE)
   {
      h_perm   = nalu_hypre_TAlloc(NALU_HYPRE_Int, n, NALU_HYPRE_MEMORY_HOST);
      h_rqperm = nalu_hypre_TAlloc(NALU_HYPRE_Int, n, NALU_HYPRE_MEMORY_HOST);

      nalu_hypre_TMemcpy(h_perm,   perm,   NALU_HYPRE_Int, n, NALU_HYPRE_MEMORY_HOST, memory_location);
      nalu_hypre_TMemcpy(h_rqperm, rqperm, NALU_HYPRE_Int, n, NALU_HYPRE_MEMORY_HOST, memory_location);
   }
   else
   {
      h_perm   = perm;
      h_rqperm = rqperm;
   }

   /* Fill data */
#if defined(NALU_HYPRE_USING_OPENMP)
   #pragma omp parallel for private(i) NALU_HYPRE_SMP_SCHEDULE
#endif
   for (i = 0; i < n; i++)
   {
      P_diag_i[i] = i;
      P_diag_j[i] = h_perm[i];
      P_diag_data[i] = 1.0;

      Q_diag_i[i] = i;
      Q_diag_j[i] = h_rqperm[i];
      Q_diag_data[i] = 1.0;
   }
   P_diag_i[n] = n;
   Q_diag_i[n] = n;

   /* Move to final memory location */
   nalu_hypre_ParCSRMatrixMigrate(P, memory_location);
   nalu_hypre_ParCSRMatrixMigrate(Q, memory_location);

   /* Update A */
   PA  = nalu_hypre_ParCSRMatMat(P, A);
   PAQ = nalu_hypre_ParCSRMatMat(PA, Q);
   //PAQ = nalu_hypre_ParCSRMatrixRAPKT(P, A, Q, 0);

   /* free and return */
   nalu_hypre_ParCSRMatrixDestroy(P);
   nalu_hypre_ParCSRMatrixDestroy(Q);
   nalu_hypre_ParCSRMatrixDestroy(PA);
   if (h_perm != perm)
   {
      nalu_hypre_TFree(h_perm, NALU_HYPRE_MEMORY_HOST);
   }
   if (h_rqperm != rqperm)
   {
      nalu_hypre_TFree(h_rqperm, NALU_HYPRE_MEMORY_HOST);
   }

   *A_pq = PAQ;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ILUSetupLDUtoCusparse
 *
 * Convert the L, D, U style to the cusparse style
 * Assume the diagonal of L and U are the ilu factorization, directly combine them
 *
 * TODO (VPM): Check this function's name
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ILUSetupLDUtoCusparse(nalu_hypre_ParCSRMatrix  *L,
                            NALU_HYPRE_Real          *D,
                            nalu_hypre_ParCSRMatrix  *U,
                            nalu_hypre_ParCSRMatrix **LDUp)
{
   /* data slots */
   NALU_HYPRE_Int            i, j, pos;

   nalu_hypre_CSRMatrix      *L_diag        = nalu_hypre_ParCSRMatrixDiag(L);
   nalu_hypre_CSRMatrix      *U_diag        = nalu_hypre_ParCSRMatrixDiag(U);
   NALU_HYPRE_Int            *L_diag_i      = nalu_hypre_CSRMatrixI(L_diag);
   NALU_HYPRE_Int            *L_diag_j      = nalu_hypre_CSRMatrixJ(L_diag);
   NALU_HYPRE_Real           *L_diag_data   = nalu_hypre_CSRMatrixData(L_diag);
   NALU_HYPRE_Int            *U_diag_i      = nalu_hypre_CSRMatrixI(U_diag);
   NALU_HYPRE_Int            *U_diag_j      = nalu_hypre_CSRMatrixJ(U_diag);
   NALU_HYPRE_Real           *U_diag_data   = nalu_hypre_CSRMatrixData(U_diag);
   NALU_HYPRE_Int            n              = nalu_hypre_ParCSRMatrixNumRows(L);
   NALU_HYPRE_Int            nnz_L          = L_diag_i[n];
   NALU_HYPRE_Int            nnz_U          = U_diag_i[n];
   NALU_HYPRE_Int            nnz_LDU        = n + nnz_L + nnz_U;

   nalu_hypre_ParCSRMatrix   *LDU;
   nalu_hypre_CSRMatrix      *LDU_diag;
   NALU_HYPRE_Int            *LDU_diag_i;
   NALU_HYPRE_Int            *LDU_diag_j;
   NALU_HYPRE_Real           *LDU_diag_data;

   /* MPI */
   MPI_Comm             comm                 = nalu_hypre_ParCSRMatrixComm(L);
   NALU_HYPRE_Int            num_procs,  my_id;

   nalu_hypre_MPI_Comm_size(comm, &num_procs);
   nalu_hypre_MPI_Comm_rank(comm, &my_id);


   /* cuda data slot */

   /* create matrix */

   LDU = nalu_hypre_ParCSRMatrixCreate(comm,
                                  nalu_hypre_ParCSRMatrixGlobalNumRows(L),
                                  nalu_hypre_ParCSRMatrixGlobalNumRows(L),
                                  nalu_hypre_ParCSRMatrixRowStarts(L),
                                  nalu_hypre_ParCSRMatrixColStarts(L),
                                  0,
                                  nnz_LDU,
                                  0);

   LDU_diag = nalu_hypre_ParCSRMatrixDiag(LDU);
   LDU_diag_i = nalu_hypre_TAlloc(NALU_HYPRE_Int, n + 1, NALU_HYPRE_MEMORY_DEVICE);
   LDU_diag_j = nalu_hypre_TAlloc(NALU_HYPRE_Int, nnz_LDU, NALU_HYPRE_MEMORY_DEVICE);
   LDU_diag_data = nalu_hypre_TAlloc(NALU_HYPRE_Real, nnz_LDU, NALU_HYPRE_MEMORY_DEVICE);

   pos = 0;

   for (i = 1; i <= n; i++)
   {
      LDU_diag_i[i - 1] = pos;
      for (j = L_diag_i[i - 1]; j < L_diag_i[i]; j++)
      {
         LDU_diag_j[pos] = L_diag_j[j];
         LDU_diag_data[pos++] = L_diag_data[j];
      }
      LDU_diag_j[pos] = i - 1;
      LDU_diag_data[pos++] = 1.0 / D[i - 1];
      for (j = U_diag_i[i - 1]; j < U_diag_i[i]; j++)
      {
         LDU_diag_j[pos] = U_diag_j[j];
         LDU_diag_data[pos++] = U_diag_data[j];
      }
   }
   LDU_diag_i[n] = pos;

   nalu_hypre_CSRMatrixI(LDU_diag)    = LDU_diag_i;
   nalu_hypre_CSRMatrixJ(LDU_diag)    = LDU_diag_j;
   nalu_hypre_CSRMatrixData(LDU_diag) = LDU_diag_data;

   /* now sort */
   nalu_hypre_CSRMatrixSortRow(LDU_diag);
   nalu_hypre_ParCSRMatrixDiag(LDU) = LDU_diag;

   *LDUp = LDU;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ILUSetupRAPMILU0
 *
 * Apply the (modified) ILU factorization to the diagonal block of A only.
 *
 * A: matrix
 * ALUp: pointer to the result, factorization stroed on the diagonal
 * modified: set to 0 to use classical ILU0
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ILUSetupRAPMILU0(nalu_hypre_ParCSRMatrix  *A,
                       nalu_hypre_ParCSRMatrix **ALUp,
                       NALU_HYPRE_Int            modified)
{
   NALU_HYPRE_Int             n = nalu_hypre_CSRMatrixNumRows(nalu_hypre_ParCSRMatrixDiag(A));

   /* Get necessary slots */
   nalu_hypre_ParCSRMatrix   *L, *U, *S, *ALU;
   NALU_HYPRE_Real           *D;
   NALU_HYPRE_Int            *u_end;

   /* u_end is the end position of the upper triangular part
     (if we need E and F implicitly), not used here */
   nalu_hypre_ILUSetupMILU0(A, NULL, NULL, n, n, &L, &D, &U, &S, &u_end, modified);
   nalu_hypre_TFree(u_end, NALU_HYPRE_MEMORY_HOST);

   /* TODO (VPM): Change this function's name */
   nalu_hypre_ILUSetupLDUtoCusparse(L, D, U, &ALU);

   /* Free memory */
   nalu_hypre_ParCSRMatrixDestroy(L);
   nalu_hypre_TFree(D, NALU_HYPRE_MEMORY_DEVICE);
   nalu_hypre_ParCSRMatrixDestroy(U);

   *ALUp = ALU;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ILUSetupRAPILU0Device
 *
 * Modified ILU(0) with RAP like solve
 * A = input matrix
 *
 * TODO (VPM): Move this function to par_setup_device.c?
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ILUSetupRAPILU0Device(nalu_hypre_ParCSRMatrix  *A,
                            NALU_HYPRE_Int           *perm,
                            NALU_HYPRE_Int            n,
                            NALU_HYPRE_Int            nLU,
                            nalu_hypre_ParCSRMatrix **Apermptr,
                            nalu_hypre_ParCSRMatrix **matSptr,
                            nalu_hypre_CSRMatrix    **ALUptr,
                            nalu_hypre_CSRMatrix    **BLUptr,
                            nalu_hypre_CSRMatrix    **CLUptr,
                            nalu_hypre_CSRMatrix    **Eptr,
                            nalu_hypre_CSRMatrix    **Fptr,
                            NALU_HYPRE_Int            test_opt)
{
   MPI_Comm             comm          = nalu_hypre_ParCSRMatrixComm(A);
   NALU_HYPRE_Int           *rperm         = NULL;
   NALU_HYPRE_Int            m             = n - nLU;
   NALU_HYPRE_Int            i;
   NALU_HYPRE_Int            num_procs,  my_id;

   /* Matrix Structure */
   nalu_hypre_ParCSRMatrix   *Apq, *ALU, *ALUm, *S;
   nalu_hypre_CSRMatrix      *Amd, *Ad, *SLU, *Apq_diag;

   nalu_hypre_MPI_Comm_size(comm, &num_procs);
   nalu_hypre_MPI_Comm_rank(comm, &my_id);

   rperm = nalu_hypre_CTAlloc(NALU_HYPRE_Int, n, NALU_HYPRE_MEMORY_HOST);

   for (i = 0; i < n; i++)
   {
      rperm[perm[i]] = i;
   }

   /* first we need to compute the ILU0 factorization of B */

   /* Copy diagonal matrix into a new place with permutation
    * That is, Apq = A(perm,qperm);
    */
   nalu_hypre_ParILURAPReorder(A, perm, rperm, &Apq);

   /* do the full ILU0 and modified ILU0 */
   nalu_hypre_ILUSetupRAPMILU0(Apq, &ALU, 0);
   nalu_hypre_ILUSetupRAPMILU0(Apq, &ALUm, 1);

   nalu_hypre_CSRMatrix *dB, *dS, *dE, *dF;

   /* get modified and extract LU factorization */
   Amd = nalu_hypre_ParCSRMatrixDiag(ALUm);
   Ad  = nalu_hypre_ParCSRMatrixDiag(ALU);
   switch (test_opt)
   {
      case 1:
      {
         /* RAP where we save E and F */
         Apq_diag = nalu_hypre_ParCSRMatrixDiag(Apq);
         nalu_hypre_CSRMatrixSortRow(Apq_diag);
         nalu_hypre_ParILUExtractEBFC(Apq_diag, nLU, &dB, &dS, Eptr, Fptr);

         /* get modified ILU of B */
         nalu_hypre_ParILUExtractEBFC(Amd, nLU, BLUptr, &SLU, &dE, &dF);
         nalu_hypre_CSRMatrixDestroy(dB);
         nalu_hypre_CSRMatrixDestroy(dS);
         nalu_hypre_CSRMatrixDestroy(dE);
         nalu_hypre_CSRMatrixDestroy(dF);

         break;
      }

      case 2:
      {
         /* C-EB^{-1}F where we save EU^{-1}, L^{-1}F as sparse matrices */
         Apq_diag = nalu_hypre_ParCSRMatrixDiag(Apq);
         nalu_hypre_CSRMatrixSortRow(Apq_diag);
         nalu_hypre_ParILUExtractEBFC(Apq_diag, nLU, &dB, CLUptr, &dE, &dF);

         /* get modified ILU of B */
         nalu_hypre_ParILUExtractEBFC(Amd, nLU, BLUptr, &SLU, Eptr, Fptr);
         nalu_hypre_CSRMatrixDestroy(dB);
         nalu_hypre_CSRMatrixDestroy(dE);
         nalu_hypre_CSRMatrixDestroy(dF);

         break;
      }

      case 3:
      {
         /* C-EB^{-1}F where we save E and F */
         Apq_diag = nalu_hypre_ParCSRMatrixDiag(Apq);
         nalu_hypre_CSRMatrixSortRow(Apq_diag);
         nalu_hypre_ParILUExtractEBFC(Apq_diag, nLU, &dB, CLUptr, Eptr, Fptr);

         /* get modified ILU of B */
         nalu_hypre_ParILUExtractEBFC(Amd, nLU, BLUptr, &SLU, &dE, &dF);
         nalu_hypre_CSRMatrixDestroy(dB);
         nalu_hypre_CSRMatrixDestroy(dE);
         nalu_hypre_CSRMatrixDestroy(dF);

         break;
      }

      case 4:
      {
         /* RAP where we save EU^{-1}, L^{-1}F as sparse matrices */
         nalu_hypre_ParILUExtractEBFC(Ad, nLU, BLUptr, &SLU, Eptr, Fptr);

         break;
      }

      case 0:
      default:
      {
         /* RAP where we save EU^{-1}, L^{-1}F as sparse matrices */
         nalu_hypre_ParILUExtractEBFC(Amd, nLU, BLUptr, &SLU, Eptr, Fptr);

         break;
      }
   }

   *ALUptr = nalu_hypre_ParCSRMatrixDiag(ALU);

   nalu_hypre_ParCSRMatrixDiag(ALU) = NULL; /* not a good practice to manipulate parcsr's csr */
   nalu_hypre_ParCSRMatrixDestroy(ALU);
   nalu_hypre_ParCSRMatrixDestroy(ALUm);

   /* start forming parCSR matrix S */

   NALU_HYPRE_BigInt   S_total_rows, S_row_starts[2];
   NALU_HYPRE_BigInt   big_m = (NALU_HYPRE_BigInt)m;
   nalu_hypre_MPI_Allreduce(&big_m, &S_total_rows, 1, NALU_HYPRE_MPI_BIG_INT, nalu_hypre_MPI_SUM, comm);

   if (S_total_rows > 0)
   {
      {
         NALU_HYPRE_BigInt global_start;
         nalu_hypre_MPI_Scan(&big_m, &global_start, 1, NALU_HYPRE_MPI_BIG_INT, nalu_hypre_MPI_SUM, comm);
         S_row_starts[0] = global_start - big_m;
         S_row_starts[1] = global_start;
      }

      S = nalu_hypre_ParCSRMatrixCreate( nalu_hypre_ParCSRMatrixComm(A),
                                    S_total_rows,
                                    S_total_rows,
                                    S_row_starts,
                                    S_row_starts,
                                    0,
                                    0,
                                    0);

      nalu_hypre_CSRMatrixDestroy(nalu_hypre_ParCSRMatrixDiag(S));
      nalu_hypre_ParCSRMatrixDiag(S) = SLU;
   }
   else
   {
      S = NULL;
      nalu_hypre_CSRMatrixDestroy(SLU);
   }

   *matSptr  = S;
   *Apermptr = Apq;

   nalu_hypre_TFree(rperm, NALU_HYPRE_MEMORY_HOST);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ILUSetupRAPILU0
 *
 * Modified ILU(0) with RAP like solve
 *
 * A = input matrix
 * Not explicitly forming the matrix
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ILUSetupRAPILU0(nalu_hypre_ParCSRMatrix  *A,
                      NALU_HYPRE_Int           *perm,
                      NALU_HYPRE_Int            n,
                      NALU_HYPRE_Int            nLU,
                      nalu_hypre_ParCSRMatrix **Lptr,
                      NALU_HYPRE_Real         **Dptr,
                      nalu_hypre_ParCSRMatrix **Uptr,
                      nalu_hypre_ParCSRMatrix **mLptr,
                      NALU_HYPRE_Real         **mDptr,
                      nalu_hypre_ParCSRMatrix **mUptr,
                      NALU_HYPRE_Int          **u_end)
{
   nalu_hypre_ParCSRMatrix   *S_temp = NULL;
   NALU_HYPRE_Int            *u_temp = NULL;

   NALU_HYPRE_Int            *u_end_array;

   nalu_hypre_CSRMatrix      *L_diag, *U_diag;
   NALU_HYPRE_Int            *L_diag_i, *U_diag_i;
   NALU_HYPRE_Int            *L_diag_j, *U_diag_j;
   NALU_HYPRE_Complex        *L_diag_data, *U_diag_data;

   nalu_hypre_CSRMatrix      *mL_diag, *mU_diag;
   NALU_HYPRE_Int            *mL_diag_i, *mU_diag_i;
   NALU_HYPRE_Int            *mL_diag_j, *mU_diag_j;
   NALU_HYPRE_Complex        *mL_diag_data, *mU_diag_data;

   NALU_HYPRE_Int            i;

   /* Standard ILU0 factorization */
   nalu_hypre_ILUSetupMILU0(A, perm, perm, n, n, Lptr, Dptr, Uptr, &S_temp, &u_temp, 0);

   /* Free memory */
   nalu_hypre_ParCSRMatrixDestroy(S_temp);
   nalu_hypre_TFree(u_temp, NALU_HYPRE_MEMORY_HOST);

   /* Modified ILU0 factorization */
   nalu_hypre_ILUSetupMILU0(A, perm, perm, n, n, mLptr, mDptr, mUptr, &S_temp, &u_temp, 1);

   /* Free memory */
   nalu_hypre_ParCSRMatrixDestroy(S_temp);
   nalu_hypre_TFree(u_temp, NALU_HYPRE_MEMORY_HOST);

   /* Pointer to the start location */
   u_end_array  = nalu_hypre_TAlloc(NALU_HYPRE_Int, n, NALU_HYPRE_MEMORY_HOST);
   U_diag       = nalu_hypre_ParCSRMatrixDiag(*Uptr);
   U_diag_i     = nalu_hypre_CSRMatrixI(U_diag);
   U_diag_j     = nalu_hypre_CSRMatrixJ(U_diag);
   U_diag_data  = nalu_hypre_CSRMatrixData(U_diag);
   mU_diag      = nalu_hypre_ParCSRMatrixDiag(*mUptr);
   mU_diag_i    = nalu_hypre_CSRMatrixI(mU_diag);
   mU_diag_j    = nalu_hypre_CSRMatrixJ(mU_diag);
   mU_diag_data = nalu_hypre_CSRMatrixData(mU_diag);

   /* first sort the Upper part U */
   for (i = 0; i < nLU; i++)
   {
      nalu_hypre_qsort1(U_diag_j, U_diag_data, U_diag_i[i], U_diag_i[i + 1] - 1);
      nalu_hypre_qsort1(mU_diag_j, mU_diag_data, mU_diag_i[i], mU_diag_i[i + 1] - 1);
      nalu_hypre_BinarySearch2(U_diag_j, nLU, U_diag_i[i], U_diag_i[i + 1] - 1, u_end_array + i);
   }

   L_diag       = nalu_hypre_ParCSRMatrixDiag(*Lptr);
   L_diag_i     = nalu_hypre_CSRMatrixI(L_diag);
   L_diag_j     = nalu_hypre_CSRMatrixJ(L_diag);
   L_diag_data  = nalu_hypre_CSRMatrixData(L_diag);
   mL_diag      = nalu_hypre_ParCSRMatrixDiag(*mLptr);
   mL_diag_i    = nalu_hypre_CSRMatrixI(mL_diag);
   mL_diag_j    = nalu_hypre_CSRMatrixJ(mL_diag);
   mL_diag_data = nalu_hypre_CSRMatrixData(mL_diag);

   /* now sort the Lower part L */
   for (i = nLU; i < n; i++)
   {
      nalu_hypre_qsort1(L_diag_j, L_diag_data, L_diag_i[i], L_diag_i[i + 1] - 1);
      nalu_hypre_qsort1(mL_diag_j, mL_diag_data, mL_diag_i[i], mL_diag_i[i + 1] - 1);
      nalu_hypre_BinarySearch2(L_diag_j, nLU, L_diag_i[i], L_diag_i[i + 1] - 1, u_end_array + i);
   }

   *u_end = u_end_array;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ILUSetupILU0
 *
 * Setup ILU(0)
 *
 * A = input matrix
 * perm = permutation array indicating ordering of rows.
 *        Perm could come from a CF_marker array or a reordering routine.
 *         When set to NULL, identity permutation is used.
 * qperm = permutation array indicating ordering of columns.
 *         When set to NULL, identity permutation is used.
 * nI = number of interial unknowns
 * nLU = size of incomplete factorization, nLU should obey nLU <= nI.
 *       Schur complement is formed if nLU < n
 * Lptr, Dptr, Uptr, Sptr = L, D, U, S factors.
 * will form global Schur Matrix if nLU < n
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ILUSetupILU0(nalu_hypre_ParCSRMatrix  *A,
                   NALU_HYPRE_Int           *perm,
                   NALU_HYPRE_Int           *qperm,
                   NALU_HYPRE_Int            nLU,
                   NALU_HYPRE_Int            nI,
                   nalu_hypre_ParCSRMatrix **Lptr,
                   NALU_HYPRE_Real         **Dptr,
                   nalu_hypre_ParCSRMatrix **Uptr,
                   nalu_hypre_ParCSRMatrix **Sptr,
                   NALU_HYPRE_Int          **u_end)
{
   return nalu_hypre_ILUSetupMILU0(A, perm, qperm, nLU, nI, Lptr, Dptr, Uptr, Sptr, u_end, 0);
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ILUSetupILU0
 *
 * Setup modified ILU(0)
 *
 * A = input matrix
 * perm = permutation array indicating ordering of rows.
 *        Perm could come from a CF_marker array or a reordering routine.
 *        When set to NULL, indentity permutation is used.
 * qperm = permutation array indicating ordering of columns.
 *         When set to NULL, identity permutation is used.
 * nI = number of interior unknowns
 * nLU = size of incomplete factorization, nLU should obey nLU <= nI.
 *       Schur complement is formed if nLU < n
 * Lptr, Dptr, Uptr, Sptr = L, D, U, S factors.
 * modified set to 0 to use classical ILU
 * will form global Schur Matrix if nLU < n
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ILUSetupMILU0(nalu_hypre_ParCSRMatrix  *A,
                    NALU_HYPRE_Int           *permp,
                    NALU_HYPRE_Int           *qpermp,
                    NALU_HYPRE_Int            nLU,
                    NALU_HYPRE_Int            nI,
                    nalu_hypre_ParCSRMatrix **Lptr,
                    NALU_HYPRE_Real         **Dptr,
                    nalu_hypre_ParCSRMatrix **Uptr,
                    nalu_hypre_ParCSRMatrix **Sptr,
                    NALU_HYPRE_Int          **u_end,
                    NALU_HYPRE_Int            modified)
{
   NALU_HYPRE_Int                i, ii, j, k, k1, k2, k3, ctrU, ctrL, ctrS;
   NALU_HYPRE_Int                lenl, lenu, jpiv, col, jpos;
   NALU_HYPRE_Int                *iw, *iL, *iU;
   NALU_HYPRE_Real               dd, t, dpiv, lxu, *wU, *wL;
   NALU_HYPRE_Real               drop;

   /* communication stuffs for S */
   MPI_Comm                  comm = nalu_hypre_ParCSRMatrixComm(A);
   NALU_HYPRE_Int                 S_offd_nnz, S_offd_ncols;
   nalu_hypre_ParCSRCommPkg      *comm_pkg;
   nalu_hypre_ParCSRCommHandle   *comm_handle;
   NALU_HYPRE_Int                 num_sends, begin, end;
   NALU_HYPRE_BigInt             *send_buf        = NULL;
   NALU_HYPRE_Int                 num_procs, my_id;

   /* data objects for A */
   nalu_hypre_CSRMatrix          *A_diag          = nalu_hypre_ParCSRMatrixDiag(A);
   nalu_hypre_CSRMatrix          *A_offd          = nalu_hypre_ParCSRMatrixOffd(A);
   NALU_HYPRE_Real               *A_diag_data     = nalu_hypre_CSRMatrixData(A_diag);
   NALU_HYPRE_Int                *A_diag_i        = nalu_hypre_CSRMatrixI(A_diag);
   NALU_HYPRE_Int                *A_diag_j        = nalu_hypre_CSRMatrixJ(A_diag);
   NALU_HYPRE_Real               *A_offd_data     = nalu_hypre_CSRMatrixData(A_offd);
   NALU_HYPRE_Int                *A_offd_i        = nalu_hypre_CSRMatrixI(A_offd);
   NALU_HYPRE_Int                *A_offd_j        = nalu_hypre_CSRMatrixJ(A_offd);
   NALU_HYPRE_MemoryLocation      memory_location = nalu_hypre_ParCSRMatrixMemoryLocation(A);

   /* size of problem and schur system */
   NALU_HYPRE_Int                n                = nalu_hypre_CSRMatrixNumRows(A_diag);
   NALU_HYPRE_Int                m                = n - nLU;
   NALU_HYPRE_Int                e                = nI - nLU;
   NALU_HYPRE_Int                m_e              = n - nI;
   NALU_HYPRE_Real               local_nnz, total_nnz;
   NALU_HYPRE_Int                *u_end_array;

   /* data objects for L, D, U */
   nalu_hypre_ParCSRMatrix       *matL;
   nalu_hypre_ParCSRMatrix       *matU;
   nalu_hypre_CSRMatrix          *L_diag;
   nalu_hypre_CSRMatrix          *U_diag;
   NALU_HYPRE_Real               *D_data;
   NALU_HYPRE_Real               *L_diag_data;
   NALU_HYPRE_Int                *L_diag_i;
   NALU_HYPRE_Int                *L_diag_j;
   NALU_HYPRE_Real               *U_diag_data;
   NALU_HYPRE_Int                *U_diag_i;
   NALU_HYPRE_Int                *U_diag_j;

   /* data objects for S */
   nalu_hypre_ParCSRMatrix       *matS = NULL;
   nalu_hypre_CSRMatrix          *S_diag;
   nalu_hypre_CSRMatrix          *S_offd;
   NALU_HYPRE_Real               *S_diag_data     = NULL;
   NALU_HYPRE_Int                *S_diag_i        = NULL;
   NALU_HYPRE_Int                *S_diag_j        = NULL;
   NALU_HYPRE_Int                *S_offd_i        = NULL;
   NALU_HYPRE_Int                *S_offd_j        = NULL;
   NALU_HYPRE_BigInt             *S_offd_colmap   = NULL;
   NALU_HYPRE_Real               *S_offd_data;
   NALU_HYPRE_BigInt             col_starts[2];
   NALU_HYPRE_BigInt             total_rows;

   /* memory management */
   NALU_HYPRE_Int                initial_alloc    = 0;
   NALU_HYPRE_Int                capacity_L;
   NALU_HYPRE_Int                capacity_U;
   NALU_HYPRE_Int                capacity_S       = 0;
   NALU_HYPRE_Int                nnz_A            = A_diag_i[n];

   /* reverse permutation array */
   NALU_HYPRE_Int                *rperm;
   NALU_HYPRE_Int                *perm, *qperm;

   /* start setup
    * get communication stuffs first
    */
   nalu_hypre_MPI_Comm_size(comm, &num_procs);
   nalu_hypre_MPI_Comm_rank(comm, &my_id);
   comm_pkg = nalu_hypre_ParCSRMatrixCommPkg(A);

   /* setup if not yet built */
   if (!comm_pkg)
   {
      nalu_hypre_MatvecCommPkgCreate(A);
      comm_pkg = nalu_hypre_ParCSRMatrixCommPkg(A);
   }

   /* check for correctness */
   if (nLU < 0 || nLU > n)
   {
      nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_ARG, "WARNING: nLU out of range.\n");
   }
   if (e < 0)
   {
      nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_ARG, "WARNING: nLU should not exceed nI.\n");
   }

   /* Allocate memory for u_end array */
   u_end_array    = nalu_hypre_TAlloc(NALU_HYPRE_Int, nLU, NALU_HYPRE_MEMORY_HOST);

   /* Allocate memory for L,D,U,S factors */
   if (n > 0)
   {
      initial_alloc  = (NALU_HYPRE_Int)(nLU + nalu_hypre_ceil((nnz_A / 2.0) * nLU / n));
      capacity_S     = (NALU_HYPRE_Int)(m + nalu_hypre_ceil((nnz_A / 2.0) * m / n));
   }
   capacity_L     = initial_alloc;
   capacity_U     = initial_alloc;

   D_data         = nalu_hypre_TAlloc(NALU_HYPRE_Real, n, memory_location);
   L_diag_i       = nalu_hypre_TAlloc(NALU_HYPRE_Int, n + 1, memory_location);
   L_diag_j       = nalu_hypre_TAlloc(NALU_HYPRE_Int, capacity_L, memory_location);
   L_diag_data    = nalu_hypre_TAlloc(NALU_HYPRE_Real, capacity_L, memory_location);
   U_diag_i       = nalu_hypre_TAlloc(NALU_HYPRE_Int, n + 1, memory_location);
   U_diag_j       = nalu_hypre_TAlloc(NALU_HYPRE_Int, capacity_U, memory_location);
   U_diag_data    = nalu_hypre_TAlloc(NALU_HYPRE_Real, capacity_U, memory_location);
   S_diag_i       = nalu_hypre_TAlloc(NALU_HYPRE_Int, m + 1, memory_location);
   S_diag_j       = nalu_hypre_TAlloc(NALU_HYPRE_Int, capacity_S, memory_location);
   S_diag_data    = nalu_hypre_TAlloc(NALU_HYPRE_Real, capacity_S, memory_location);

   /* allocate working arrays */
   iw             = nalu_hypre_TAlloc(NALU_HYPRE_Int, 3 * n, NALU_HYPRE_MEMORY_HOST);
   iL             = iw + n;
   rperm          = iw + 2 * n;
   wL             = nalu_hypre_TAlloc(NALU_HYPRE_Real, n, NALU_HYPRE_MEMORY_HOST);

   ctrU        = ctrL        = ctrS        = 0;
   L_diag_i[0] = U_diag_i[0] = S_diag_i[0] = 0;
   /* set marker array iw to -1 */
   for (i = 0; i < n; i++)
   {
      iw[i] = -1;
   }

   /* get reverse permutation (rperm).
    * create permutation if they are null
    * rperm holds the reordered indexes.
    * rperm only used for column
    */

   if (!permp)
   {
      perm = nalu_hypre_TAlloc(NALU_HYPRE_Int, n, memory_location);
      for (i = 0; i < n; i++)
      {
         perm[i] = i;
      }
   }
   else
   {
      perm = permp;
   }

   if (!qpermp)
   {
      qperm = nalu_hypre_TAlloc(NALU_HYPRE_Int, n, memory_location);
      for (i = 0; i < n; i++)
      {
         qperm[i] = i;
      }
   }
   else
   {
      qperm = qpermp;
   }

   for (i = 0; i < n; i++)
   {
      rperm[qperm[i]] = i;
   }

   /*---------  Begin Factorization. Work in permuted space  ----*/
   for (ii = 0; ii < nLU; ii++)
   {
      // get row i
      i = perm[ii];
      // get extents of row i
      k1 = A_diag_i[i];
      k2 = A_diag_i[i + 1];
      // track the drop
      drop = 0.0;

      /*-------------------- unpack L & U-parts of row of A in arrays w */
      iU = iL + ii;
      wU = wL + ii;
      /*--------------------  diagonal entry */
      dd = 0.0;
      lenl  = lenu = 0;
      iw[ii] = ii;
      /*-------------------- scan & unwrap column */
      for (j = k1; j < k2; j++)
      {
         col = rperm[A_diag_j[j]];
         t = A_diag_data[j];
         if ( col < ii )
         {
            iw[col] = lenl;
            iL[lenl] = col;
            wL[lenl++] = t;
         }
         else if (col > ii)
         {
            iw[col] = lenu;
            iU[lenu] = col;
            wU[lenu++] = t;
         }
         else
         {
            dd = t;
         }
      }

      /* eliminate row */
      /*-------------------------------------------------------------------------
       *  In order to do the elimination in the correct order we must select the
       *  smallest column index among iL[k], k = j, j+1, ..., lenl-1. For ILU(0),
       *  no new fill-ins are expect, so we can pre-sort iL and wL prior to the
       *  entering the elimination loop.
       *-----------------------------------------------------------------------*/
      //      nalu_hypre_quickSortIR(iL, wL, iw, 0, (lenl-1));
      nalu_hypre_qsort3ir(iL, wL, iw, 0, (lenl - 1));
      for (j = 0; j < lenl; j++)
      {
         jpiv = iL[j];
         /* get factor/ pivot element */
         dpiv = wL[j] * D_data[jpiv];
         /* store entry in L */
         wL[j] = dpiv;

         /* zero out element - reset pivot */
         iw[jpiv] = -1;
         /* combine current row and pivot row */
         for (k = U_diag_i[jpiv]; k < U_diag_i[jpiv + 1]; k++)
         {
            col = U_diag_j[k];
            jpos = iw[col];

            /* Only fill-in nonzero pattern (jpos != 0) */
            if (jpos < 0)
            {
               drop = drop - U_diag_data[k] * dpiv;
               continue;
            }

            lxu = - U_diag_data[k] * dpiv;
            if (col < ii)
            {
               /* dealing with L part */
               wL[jpos] += lxu;
            }
            else if (col > ii)
            {
               /* dealing with U part */
               wU[jpos] += lxu;
            }
            else
            {
               /* diagonal update */
               dd += lxu;
            }
         }
      }
      /* modify when necessary */
      if (modified)
      {
         dd = dd + drop;
      }

      /* restore iw (only need to restore diagonal and U part */
      iw[ii] = -1;
      for (j = 0; j < lenu; j++)
      {
         iw[iU[j]] = -1;
      }

      /* Update LDU factors */
      /* L part */
      /* Check that memory is sufficient */
      if (lenl > 0)
      {
         while ((ctrL + lenl) > capacity_L)
         {
            NALU_HYPRE_Int tmp = capacity_L;
            capacity_L = (NALU_HYPRE_Int)(capacity_L * EXPAND_FACT + 1);
            L_diag_j = nalu_hypre_TReAlloc_v2(L_diag_j, NALU_HYPRE_Int, tmp, NALU_HYPRE_Int,
                                         capacity_L, memory_location);
            L_diag_data = nalu_hypre_TReAlloc_v2(L_diag_data, NALU_HYPRE_Real, tmp, NALU_HYPRE_Real,
                                            capacity_L, memory_location);
         }
         nalu_hypre_TMemcpy(&L_diag_j[ctrL], iL, NALU_HYPRE_Int, lenl,
                       memory_location, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TMemcpy(&L_diag_data[ctrL], wL, NALU_HYPRE_Real, lenl,
                       memory_location, NALU_HYPRE_MEMORY_HOST);
      }
      L_diag_i[ii + 1] = (ctrL += lenl);

      /* diagonal part (we store the inverse) */
      if (nalu_hypre_abs(dd) < MAT_TOL)
      {
         dd = 1.0e-6;
      }
      D_data[ii] = 1. / dd;

      /* U part */
      /* Check that memory is sufficient */
      if (lenu > 0)
      {
         while ((ctrU + lenu) > capacity_U)
         {
            NALU_HYPRE_Int tmp = capacity_U;
            capacity_U = (NALU_HYPRE_Int)(capacity_U * EXPAND_FACT + 1);
            U_diag_j = nalu_hypre_TReAlloc_v2(U_diag_j, NALU_HYPRE_Int, tmp, NALU_HYPRE_Int,
                                         capacity_U, memory_location);
            U_diag_data = nalu_hypre_TReAlloc_v2(U_diag_data, NALU_HYPRE_Real, tmp, NALU_HYPRE_Real,
                                            capacity_U, memory_location);
         }
         nalu_hypre_TMemcpy(&U_diag_j[ctrU], iU, NALU_HYPRE_Int, lenu,
                       memory_location, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TMemcpy(&U_diag_data[ctrU], wU, NALU_HYPRE_Real, lenu,
                       memory_location, NALU_HYPRE_MEMORY_HOST);
      }
      U_diag_i[ii + 1] = (ctrU += lenu);

      /* check and build u_end array */
      if (m > 0)
      {
         nalu_hypre_qsort1(U_diag_j, U_diag_data, U_diag_i[ii], U_diag_i[ii + 1] - 1);
         nalu_hypre_BinarySearch2(U_diag_j, nLU, U_diag_i[ii], U_diag_i[ii + 1] - 1, u_end_array + ii);
      }
      else
      {
         /* Everything is in U */
         u_end_array[ii] = ctrU;
      }

   }

   /*---------  Begin Factorization in Schur Complement part  ----*/
   for (ii = nLU; ii < n; ii++)
   {
      // get row i
      i = perm[ii];
      // get extents of row i
      k1 = A_diag_i[i];
      k2 = A_diag_i[i + 1];
      drop = 0.0;

      /*-------------------- unpack L & U-parts of row of A in arrays w */
      iU = iL + nLU + 1;
      wU = wL + nLU + 1;
      /*--------------------  diagonal entry */
      dd = 0.0;
      lenl  = lenu = 0;
      iw[ii] = nLU;
      /*-------------------- scan & unwrap column */
      for (j = k1; j < k2; j++)
      {
         col = rperm[A_diag_j[j]];
         t = A_diag_data[j];
         if ( col < nLU )
         {
            iw[col] = lenl;
            iL[lenl] = col;
            wL[lenl++] = t;
         }
         else if (col != ii)
         {
            iw[col] = lenu;
            iU[lenu] = col;
            wU[lenu++] = t;
         }
         else
         {
            dd = t;
         }
      }

      /* eliminate row */
      /*-------------------------------------------------------------------------
       *  In order to do the elimination in the correct order we must select the
       *  smallest column index among iL[k], k = j, j+1, ..., lenl-1. For ILU(0),
       *  no new fill-ins are expect, so we can pre-sort iL and wL prior to the
       *  entering the elimination loop.
       *-----------------------------------------------------------------------*/
      //      nalu_hypre_quickSortIR(iL, wL, iw, 0, (lenl-1));
      nalu_hypre_qsort3ir(iL, wL, iw, 0, (lenl - 1));
      for (j = 0; j < lenl; j++)
      {
         jpiv = iL[j];
         /* get factor/ pivot element */
         dpiv = wL[j] * D_data[jpiv];
         /* store entry in L */
         wL[j] = dpiv;

         /* zero out element - reset pivot */
         iw[jpiv] = -1;
         /* combine current row and pivot row */
         for (k = U_diag_i[jpiv]; k < U_diag_i[jpiv + 1]; k++)
         {
            col = U_diag_j[k];
            jpos = iw[col];

            /* Only fill-in nonzero pattern (jpos != 0) */
            if (jpos < 0)
            {
               drop = drop - U_diag_data[k] * dpiv;
               continue;
            }

            lxu = - U_diag_data[k] * dpiv;
            if (col < nLU)
            {
               /* dealing with L part */
               wL[jpos] += lxu;
            }
            else if (col != ii)
            {
               /* dealing with U part */
               wU[jpos] += lxu;
            }
            else
            {
               /* diagonal update */
               dd += lxu;
            }
         }
      }
      if (modified)
      {
         dd = dd + drop;
      }
      /* restore iw (only need to restore diagonal and U part */
      iw[ii] = -1;
      for (j = 0; j < lenu; j++)
      {
         iw[iU[j]] = -1;
      }

      /* Update LDU factors */
      /* L part */
      /* Check that memory is sufficient */
      if (lenl > 0)
      {
         while ((ctrL + lenl) > capacity_L)
         {
            NALU_HYPRE_Int tmp = capacity_L;
            capacity_L = (NALU_HYPRE_Int)(capacity_L * EXPAND_FACT + 1);
            L_diag_j = nalu_hypre_TReAlloc_v2(L_diag_j, NALU_HYPRE_Int, tmp, NALU_HYPRE_Int,
                                         capacity_L, memory_location);
            L_diag_data = nalu_hypre_TReAlloc_v2(L_diag_data, NALU_HYPRE_Real, tmp, NALU_HYPRE_Real,
                                            capacity_L, memory_location);
         }
         nalu_hypre_TMemcpy(&L_diag_j[ctrL], iL, NALU_HYPRE_Int, lenl,
                       memory_location, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TMemcpy(&L_diag_data[ctrL], wL, NALU_HYPRE_Real, lenl,
                       memory_location, NALU_HYPRE_MEMORY_HOST);
      }
      L_diag_i[ii + 1] = (ctrL += lenl);

      /* S part */
      /* Check that memory is sufficient */
      while ((ctrS + lenu + 1) > capacity_S)
      {
         NALU_HYPRE_Int tmp = capacity_S;
         capacity_S = (NALU_HYPRE_Int)(capacity_S * EXPAND_FACT + 1);
         S_diag_j = nalu_hypre_TReAlloc_v2(S_diag_j, NALU_HYPRE_Int, tmp, NALU_HYPRE_Int,
                                      capacity_S, memory_location);
         S_diag_data = nalu_hypre_TReAlloc_v2(S_diag_data, NALU_HYPRE_Real, tmp, NALU_HYPRE_Real,
                                         capacity_S, memory_location);
      }
      /* remember S in under a new index system! */
      S_diag_j[ctrS] = ii - nLU;
      S_diag_data[ctrS] = dd;
      for (j = 0; j < lenu; j++)
      {
         S_diag_j[ctrS + 1 + j] = iU[j] - nLU;
      }
      //nalu_hypre_TMemcpy(S_diag_data+ctrS+1, wU, NALU_HYPRE_Real, lenu, memory_location, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TMemcpy(S_diag_data + ctrS + 1, wU, NALU_HYPRE_Real, lenu,
                    memory_location, NALU_HYPRE_MEMORY_HOST);
      S_diag_i[ii - nLU + 1] = ctrS += (lenu + 1);
   }
   /* Assemble LDUS matrices */
   /* zero out unfactored rows for U and D */
   for (k = nLU; k < n; k++)
   {
      U_diag_i[k + 1] = ctrU;
      D_data[k] = 1.;
   }

   /* First create Schur complement if necessary
    * Check if we need to create Schur complement
    */
   NALU_HYPRE_BigInt big_m = (NALU_HYPRE_BigInt)m;
   nalu_hypre_MPI_Allreduce(&big_m, &total_rows, 1, NALU_HYPRE_MPI_BIG_INT, nalu_hypre_MPI_SUM, comm);

   /* only form when total_rows > 0 */
   if (total_rows > 0)
   {
      /* now create S */
      /* need to get new column start */
      {
         NALU_HYPRE_BigInt global_start;
         nalu_hypre_MPI_Scan(&big_m, &global_start, 1, NALU_HYPRE_MPI_BIG_INT, nalu_hypre_MPI_SUM, comm);
         col_starts[0] = global_start - m;
         col_starts[1] = global_start;
      }

      /* We did nothing to A_offd, so all the data kept, just reorder them
       * The create function takes comm, global num rows/cols,
       *    row/col start, num cols offd, nnz diag, nnz offd
       */
      S_offd_nnz = nalu_hypre_CSRMatrixNumNonzeros(A_offd);
      S_offd_ncols = nalu_hypre_CSRMatrixNumCols(A_offd);

      matS = nalu_hypre_ParCSRMatrixCreate( comm,
                                       total_rows,
                                       total_rows,
                                       col_starts,
                                       col_starts,
                                       S_offd_ncols,
                                       ctrS,
                                       S_offd_nnz);

      /* first put diagonal data in */
      S_diag = nalu_hypre_ParCSRMatrixDiag(matS);

      nalu_hypre_CSRMatrixI(S_diag) = S_diag_i;
      nalu_hypre_CSRMatrixData(S_diag) = S_diag_data;
      nalu_hypre_CSRMatrixJ(S_diag) = S_diag_j;

      /* now start to construct offdiag of S */
      S_offd = nalu_hypre_ParCSRMatrixOffd(matS);
      S_offd_i = nalu_hypre_TAlloc(NALU_HYPRE_Int, m + 1, memory_location);
      S_offd_j = nalu_hypre_TAlloc(NALU_HYPRE_Int, S_offd_nnz, memory_location);
      S_offd_data = nalu_hypre_TAlloc(NALU_HYPRE_Real, S_offd_nnz, memory_location);
      S_offd_colmap = nalu_hypre_CTAlloc(NALU_HYPRE_BigInt, S_offd_ncols, NALU_HYPRE_MEMORY_HOST);

      /* simply use a loop to copy data from A_offd */
      S_offd_i[0] = 0;
      k3 = 0;
      for (i = 1; i <= e; i++)
      {
         S_offd_i[i] = k3;
      }
      for (i = 0; i < m_e; i++)
      {
         col = perm[i + nI];
         k1 = A_offd_i[col];
         k2 = A_offd_i[col + 1];
         for (j = k1; j < k2; j++)
         {
            S_offd_j[k3] = A_offd_j[j];
            S_offd_data[k3++] = A_offd_data[j];
         }
         S_offd_i[i + 1 + e] = k3;
      }

      /* give I, J, DATA to S_offd */
      nalu_hypre_CSRMatrixI(S_offd) = S_offd_i;
      nalu_hypre_CSRMatrixJ(S_offd) = S_offd_j;
      nalu_hypre_CSRMatrixData(S_offd) = S_offd_data;

      /* now we need to update S_offd_colmap */

      /* get total num of send */
      num_sends = nalu_hypre_ParCSRCommPkgNumSends(comm_pkg);
      begin = nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, 0);
      end = nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends);
      send_buf = nalu_hypre_TAlloc(NALU_HYPRE_BigInt, end - begin, NALU_HYPRE_MEMORY_HOST);
      /* copy new index into send_buf */
      for (i = begin; i < end; i++)
      {
         send_buf[i - begin] = rperm[nalu_hypre_ParCSRCommPkgSendMapElmt(comm_pkg, i)] -
                               nLU + col_starts[0];
      }
      /* main communication */
      comm_handle = nalu_hypre_ParCSRCommHandleCreate(21, comm_pkg, send_buf, S_offd_colmap);
      nalu_hypre_ParCSRCommHandleDestroy(comm_handle);

      /* setup index */
      nalu_hypre_ParCSRMatrixColMapOffd(matS) = S_offd_colmap;

      nalu_hypre_ILUSortOffdColmap(matS);

      /* free */
      nalu_hypre_TFree(send_buf, NALU_HYPRE_MEMORY_HOST);
   } /* end of forming S */

   /* create S finished */

   matL = nalu_hypre_ParCSRMatrixCreate( comm,
                                    nalu_hypre_ParCSRMatrixGlobalNumRows(A),
                                    nalu_hypre_ParCSRMatrixGlobalNumRows(A),
                                    nalu_hypre_ParCSRMatrixRowStarts(A),
                                    nalu_hypre_ParCSRMatrixColStarts(A),
                                    0,
                                    ctrL,
                                    0 );

   L_diag = nalu_hypre_ParCSRMatrixDiag(matL);
   nalu_hypre_CSRMatrixI(L_diag) = L_diag_i;
   if (ctrL)
   {
      nalu_hypre_CSRMatrixData(L_diag) = L_diag_data;
      nalu_hypre_CSRMatrixJ(L_diag) = L_diag_j;
   }
   else
   {
      /* we've allocated some memory, so free if not used */
      nalu_hypre_TFree(L_diag_j, memory_location);
      nalu_hypre_TFree(L_diag_data, memory_location);
   }
   /* store (global) total number of nonzeros */
   local_nnz = (NALU_HYPRE_Real) ctrL;
   nalu_hypre_MPI_Allreduce(&local_nnz, &total_nnz, 1, NALU_HYPRE_MPI_REAL, nalu_hypre_MPI_SUM, comm);
   nalu_hypre_ParCSRMatrixDNumNonzeros(matL) = total_nnz;

   matU = nalu_hypre_ParCSRMatrixCreate( comm,
                                    nalu_hypre_ParCSRMatrixGlobalNumRows(A),
                                    nalu_hypre_ParCSRMatrixGlobalNumRows(A),
                                    nalu_hypre_ParCSRMatrixRowStarts(A),
                                    nalu_hypre_ParCSRMatrixColStarts(A),
                                    0,
                                    ctrU,
                                    0 );

   U_diag = nalu_hypre_ParCSRMatrixDiag(matU);
   nalu_hypre_CSRMatrixI(U_diag) = U_diag_i;
   if (ctrU)
   {
      nalu_hypre_CSRMatrixData(U_diag) = U_diag_data;
      nalu_hypre_CSRMatrixJ(U_diag) = U_diag_j;
   }
   else
   {
      /* we've allocated some memory, so free if not used */
      nalu_hypre_TFree(U_diag_j, memory_location);
      nalu_hypre_TFree(U_diag_data, memory_location);
   }
   /* store (global) total number of nonzeros */
   local_nnz = (NALU_HYPRE_Real) ctrU;
   nalu_hypre_MPI_Allreduce(&local_nnz, &total_nnz, 1, NALU_HYPRE_MPI_REAL, nalu_hypre_MPI_SUM, comm);
   nalu_hypre_ParCSRMatrixDNumNonzeros(matU) = total_nnz;
   /* free memory */
   nalu_hypre_TFree(wL, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(iw, NALU_HYPRE_MEMORY_HOST);
   if (!matS)
   {
      /* we allocate some memory for S, need to free if unused */
      nalu_hypre_TFree(S_diag_i, memory_location);
   }

   if (!permp)
   {
      nalu_hypre_TFree(perm, memory_location);
   }
   if (!qpermp)
   {
      nalu_hypre_TFree(qperm, memory_location);
   }

   /* set matrix pointers */
   *Lptr = matL;
   *Dptr = D_data;
   *Uptr = matU;
   *Sptr = matS;
   *u_end = u_end_array;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ILUSetupILUKSymbolic
 *
 * Setup ILU(k) symbolic factorization
 *
 * n = total rows of input
 * lfil = level of fill-in, the k in ILU(k)
 * perm = permutation array indicating ordering of factorization.
 * rperm = reverse permutation array, used here to avoid duplicate memory allocation
 * iw = working array, used here to avoid duplicate memory allocation
 * nLU = size of computed LDU factorization.
 * A/L/U/S_diag_i = the I slot of A, L, U and S
 * A/L/U/S_diag_j = the J slot of A, L, U and S
 *
 * Will form global Schur Matrix if nLU < n
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ILUSetupILUKSymbolic(NALU_HYPRE_Int   n,
                           NALU_HYPRE_Int  *A_diag_i,
                           NALU_HYPRE_Int  *A_diag_j,
                           NALU_HYPRE_Int   lfil,
                           NALU_HYPRE_Int  *perm,
                           NALU_HYPRE_Int  *rperm,
                           NALU_HYPRE_Int  *iw,
                           NALU_HYPRE_Int   nLU,
                           NALU_HYPRE_Int  *L_diag_i,
                           NALU_HYPRE_Int  *U_diag_i,
                           NALU_HYPRE_Int  *S_diag_i,
                           NALU_HYPRE_Int **L_diag_j,
                           NALU_HYPRE_Int **U_diag_j,
                           NALU_HYPRE_Int **S_diag_j,
                           NALU_HYPRE_Int **u_end)
{
   /*
    * 1: Setup and create buffers
    * A_diag_*: tempory pointer for the diagonal matrix of A and its '*' slot
    * ii: outer loop from 0 to nLU - 1
    * i: the real col number in diag inside the outer loop
    * iw:  working array store the reverse of active col number
    * iL: working array store the active col number
    * iLev: working array store the active level of current row
    * lenl/u: current position in iw and so
    * ctrL/U/S: global position in J
    */

   NALU_HYPRE_Int         *temp_L_diag_j, *temp_U_diag_j, *temp_S_diag_j = NULL, *u_levels;
   NALU_HYPRE_Int         *iL, *iLev;
   NALU_HYPRE_Int         ii, i, j, k, ku, lena, lenl, lenu, lenh, ilev, lev, col, icol;
   NALU_HYPRE_Int         m = n - nLU;
   NALU_HYPRE_Int         *u_end_array;

   /* memory management */
   NALU_HYPRE_Int         ctrL;
   NALU_HYPRE_Int         ctrU;
   NALU_HYPRE_Int         ctrS;
   NALU_HYPRE_Int         capacity_L;
   NALU_HYPRE_Int         capacity_U;
   NALU_HYPRE_Int         capacity_S;
   NALU_HYPRE_Int         initial_alloc = 0;
   NALU_HYPRE_Int         nnz_A;
   NALU_HYPRE_MemoryLocation memory_location;

   /* Get default memory location */
   NALU_HYPRE_GetMemoryLocation(&memory_location);

   /* set iL and iLev to right place in iw array */
   iL                = iw + n;
   iLev              = iw + 2 * n;

   /* setup initial memory used */
   nnz_A             = A_diag_i[n];
   if (n > 0)
   {
      initial_alloc     = (NALU_HYPRE_Int)(nLU + nalu_hypre_ceil((nnz_A / 2.0) * nLU / n));
   }
   capacity_L        = initial_alloc;
   capacity_U        = initial_alloc;

   /* allocate other memory for L and U struct */
   temp_L_diag_j     = nalu_hypre_CTAlloc(NALU_HYPRE_Int, capacity_L, memory_location);
   temp_U_diag_j     = nalu_hypre_CTAlloc(NALU_HYPRE_Int, capacity_U, memory_location);

   if (m > 0)
   {
      capacity_S     = (NALU_HYPRE_Int)(m + nalu_hypre_ceil(nnz_A / 2.0 * m / n));
      temp_S_diag_j  = nalu_hypre_CTAlloc(NALU_HYPRE_Int, capacity_S, memory_location);
   }

   u_end_array       = nalu_hypre_TAlloc(NALU_HYPRE_Int, nLU, NALU_HYPRE_MEMORY_HOST);
   u_levels          = nalu_hypre_CTAlloc(NALU_HYPRE_Int, capacity_U, NALU_HYPRE_MEMORY_HOST);
   ctrL = ctrU = ctrS = 0;

   /* set initial value for working array */
   for (ii = 0 ; ii < n; ii++)
   {
      iw[ii] = -1;
   }

   /*
    * 2: Start of main loop
    * those in iL are NEW col index (after permutation)
    */
   for (ii = 0; ii < nLU; ii++)
   {
      i = perm[ii];
      lenl = 0;
      lenh = 0;/* this is the current length of heap */
      lenu = ii;
      lena = A_diag_i[i + 1];
      /* put those already inside original pattern, and set their level to 0 */
      for (j = A_diag_i[i]; j < lena; j++)
      {
         /* get the neworder of that col */
         col = rperm[A_diag_j[j]];
         if (col < ii)
         {
            /*
             * this is an entry in L
             * we maintain a heap structure for L part
             */
            iL[lenh] = col;
            iLev[lenh] = 0;
            iw[col] = lenh++;
            /*now miantian a heap structure*/
            nalu_hypre_ILUMinHeapAddIIIi(iL, iLev, iw, lenh);
         }
         else if (col > ii)
         {
            /* this is an entry in U */
            iL[lenu] = col;
            iLev[lenu] = 0;
            iw[col] = lenu++;
         }
      }/* end of j loop for adding pattern in original matrix */

      /*
       * search lower part of current row and update pattern based on level
       */
      while (lenh > 0)
      {
         /*
          * k is now the new col index after permutation
          * the first element of the heap is the smallest
          */
         k = iL[0];
         ilev = iLev[0];
         /*
          * we now need to maintain the heap structure
          */
         nalu_hypre_ILUMinHeapRemoveIIIi(iL, iLev, iw, lenh);
         lenh--;
         /* copy to the end of array */
         lenl++;
         /* reset iw for that, not using anymore */
         iw[k] = -1;
         nalu_hypre_swap2i(iL, iLev, ii - lenl, lenh);
         /*
          * now the elimination on current row could start.
          * eliminate row k (new index) from current row
          */
         ku = U_diag_i[k + 1];
         for (j = U_diag_i[k]; j < ku; j++)
         {
            col = temp_U_diag_j[j];
            lev = u_levels[j] + ilev + 1;
            /* ignore large level */
            icol = iw[col];
            /* skill large level */
            if (lev > lfil)
            {
               continue;
            }
            if (icol < 0)
            {
               /* not yet in */
               if (col < ii)
               {
                  /*
                   * if we add to the left L, we need to maintian the
                   *    heap structure
                   */
                  iL[lenh] = col;
                  iLev[lenh] = lev;
                  iw[col] = lenh++;
                  /*swap it with the element right after the heap*/

                  /* maintain the heap */
                  nalu_hypre_ILUMinHeapAddIIIi(iL, iLev, iw, lenh);
               }
               else if (col > ii)
               {
                  iL[lenu] = col;
                  iLev[lenu] = lev;
                  iw[col] = lenu++;
               }
            }
            else
            {
               iLev[icol] = nalu_hypre_min(lev, iLev[icol]);
            }
         }/* end of loop j for level update */
      }/* end of while loop for iith row */

      /* now update everything, indices, levels and so */
      L_diag_i[ii + 1] = L_diag_i[ii] + lenl;
      if (lenl > 0)
      {
         /* check if memory is enough */
         while (ctrL + lenl > capacity_L)
         {
            NALU_HYPRE_Int tmp = capacity_L;
            capacity_L = (NALU_HYPRE_Int)(capacity_L * EXPAND_FACT + 1);
            temp_L_diag_j = nalu_hypre_TReAlloc_v2(temp_L_diag_j, NALU_HYPRE_Int, tmp, NALU_HYPRE_Int, capacity_L,
                                              memory_location);
         }
         /* now copy L data, reverse order */
         for (j = 0; j < lenl; j++)
         {
            temp_L_diag_j[ctrL + j] = iL[ii - j - 1];
         }
         ctrL += lenl;
      }
      k = lenu - ii;
      U_diag_i[ii + 1] = U_diag_i[ii] + k;
      if (k > 0)
      {
         /* check if memory is enough */
         while (ctrU + k > capacity_U)
         {
            NALU_HYPRE_Int tmp = capacity_U;
            capacity_U = (NALU_HYPRE_Int)(capacity_U * EXPAND_FACT + 1);
            temp_U_diag_j = nalu_hypre_TReAlloc_v2(temp_U_diag_j, NALU_HYPRE_Int, tmp, NALU_HYPRE_Int, capacity_U,
                                              memory_location);
            u_levels = nalu_hypre_TReAlloc_v2(u_levels, NALU_HYPRE_Int, tmp, NALU_HYPRE_Int, capacity_U, NALU_HYPRE_MEMORY_HOST);
         }
         //nalu_hypre_TMemcpy(temp_U_diag_j+ctrU,iL+ii,NALU_HYPRE_Int,k,memory_location,NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TMemcpy(temp_U_diag_j + ctrU, iL + ii, NALU_HYPRE_Int, k,
                       memory_location, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TMemcpy(u_levels + ctrU, iLev + ii, NALU_HYPRE_Int, k,
                       NALU_HYPRE_MEMORY_HOST, NALU_HYPRE_MEMORY_HOST);
         ctrU += k;
      }
      if (m > 0)
      {
         nalu_hypre_qsort2i(temp_U_diag_j, u_levels, U_diag_i[ii], U_diag_i[ii + 1] - 1);
         nalu_hypre_BinarySearch2(temp_U_diag_j, nLU, U_diag_i[ii], U_diag_i[ii + 1] - 1, u_end_array + ii);
      }
      else
      {
         /* Everything is in U */
         u_end_array[ii] = ctrU;
      }

      /* reset iw */
      for (j = ii; j < lenu; j++)
      {
         iw[iL[j]] = -1;
      }

   }/* end of main loop ii from 0 to nLU-1 */

   /* another loop to set EU^-1 and Schur complement */
   for (ii = nLU; ii < n; ii++)
   {
      i = perm[ii];
      lenl = 0;
      lenh = 0;/* this is the current length of heap */
      lenu = nLU;/* now this stores S, start from nLU */
      lena = A_diag_i[i + 1];
      /* put those already inside original pattern, and set their level to 0 */
      for (j = A_diag_i[i]; j < lena; j++)
      {
         /* get the neworder of that col */
         col = rperm[A_diag_j[j]];
         if (col < nLU)
         {
            /*
             * this is an entry in L
             * we maintain a heap structure for L part
             */
            iL[lenh] = col;
            iLev[lenh] = 0;
            iw[col] = lenh++;
            /*now miantian a heap structure*/
            nalu_hypre_ILUMinHeapAddIIIi(iL, iLev, iw, lenh);
         }
         else if (col != ii) /* we for sure to add ii, avoid duplicate */
         {
            /* this is an entry in S */
            iL[lenu] = col;
            iLev[lenu] = 0;
            iw[col] = lenu++;
         }
      }/* end of j loop for adding pattern in original matrix */

      /*
       * search lower part of current row and update pattern based on level
       */
      while (lenh > 0)
      {
         /*
          * k is now the new col index after permutation
          * the first element of the heap is the smallest
          */
         k = iL[0];
         ilev = iLev[0];
         /*
          * we now need to maintain the heap structure
          */
         nalu_hypre_ILUMinHeapRemoveIIIi(iL, iLev, iw, lenh);
         lenh--;
         /* copy to the end of array */
         lenl++;
         /* reset iw for that, not using anymore */
         iw[k] = -1;
         nalu_hypre_swap2i(iL, iLev, nLU - lenl, lenh);
         /*
          * now the elimination on current row could start.
          * eliminate row k (new index) from current row
          */
         ku = U_diag_i[k + 1];
         for (j = U_diag_i[k]; j < ku; j++)
         {
            col = temp_U_diag_j[j];
            lev = u_levels[j] + ilev + 1;
            /* ignore large level */
            icol = iw[col];
            /* skill large level */
            if (lev > lfil)
            {
               continue;
            }
            if (icol < 0)
            {
               /* not yet in */
               if (col < nLU)
               {
                  /*
                   * if we add to the left L, we need to maintian the
                   *    heap structure
                   */
                  iL[lenh] = col;
                  iLev[lenh] = lev;
                  iw[col] = lenh++;
                  /*swap it with the element right after the heap*/

                  /* maintain the heap */
                  nalu_hypre_ILUMinHeapAddIIIi(iL, iLev, iw, lenh);
               }
               else if (col != ii)
               {
                  /* S part */
                  iL[lenu] = col;
                  iLev[lenu] = lev;
                  iw[col] = lenu++;
               }
            }
            else
            {
               iLev[icol] = nalu_hypre_min(lev, iLev[icol]);
            }
         }/* end of loop j for level update */
      }/* end of while loop for iith row */

      /* now update everything, indices, levels and so */
      L_diag_i[ii + 1] = L_diag_i[ii] + lenl;
      if (lenl > 0)
      {
         /* check if memory is enough */
         while (ctrL + lenl > capacity_L)
         {
            NALU_HYPRE_Int tmp = capacity_L;
            capacity_L = (NALU_HYPRE_Int)(capacity_L * EXPAND_FACT + 1);
            temp_L_diag_j = nalu_hypre_TReAlloc_v2(temp_L_diag_j, NALU_HYPRE_Int, tmp, NALU_HYPRE_Int,
                                              capacity_L, memory_location);
         }
         /* now copy L data, reverse order */
         for (j = 0; j < lenl; j++)
         {
            temp_L_diag_j[ctrL + j] = iL[nLU - j - 1];
         }
         ctrL += lenl;
      }
      k = lenu - nLU + 1;
      /* check if memory is enough */
      while (ctrS + k > capacity_S)
      {
         NALU_HYPRE_Int tmp = capacity_S;
         capacity_S = (NALU_HYPRE_Int)(capacity_S * EXPAND_FACT + 1);
         temp_S_diag_j = nalu_hypre_TReAlloc_v2(temp_S_diag_j, NALU_HYPRE_Int, tmp, NALU_HYPRE_Int, capacity_S,
                                           memory_location);
      }
      temp_S_diag_j[ctrS] = ii;/* must have diagonal */
      //nalu_hypre_TMemcpy(temp_S_diag_j+ctrS+1,iL+nLU,NALU_HYPRE_Int,k-1,memory_location,NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TMemcpy(temp_S_diag_j + ctrS + 1, iL + nLU, NALU_HYPRE_Int, k - 1,
                    memory_location, NALU_HYPRE_MEMORY_HOST);
      ctrS += k;
      S_diag_i[ii - nLU + 1] = ctrS;

      /* reset iw */
      for (j = nLU; j < lenu; j++)
      {
         iw[iL[j]] = -1;
      }

   }/* end of main loop ii from nLU to n-1 */

   /*
    * 3: Update the struct for L, U and S
    */
   for (k = nLU; k < n; k++)
   {
      U_diag_i[k + 1] = U_diag_i[nLU];
   }
   /*
    * 4: Finishing up and free memory
    */
   nalu_hypre_TFree(u_levels, NALU_HYPRE_MEMORY_HOST);

   *L_diag_j = temp_L_diag_j;
   *U_diag_j = temp_U_diag_j;
   *S_diag_j = temp_S_diag_j;
   *u_end = u_end_array;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ILUSetupILUK
 *
 * Setup ILU(k) numeric factorization
 *
 * A: input matrix
 * lfil: level of fill-in, the k in ILU(k)
 * permp: permutation array indicating ordering of factorization.
 *        Perm could come from a CF_marker array or a reordering routine.
 * qpermp: column permutation array.
 * nLU: size of computed LDU factorization.
 * nI: number of interial unknowns, nI should obey nI >= nLU
 * Lptr, Dptr, Uptr: L, D, U factors.
 * Sprt: Schur Complement, if no Schur Complement, it will be set to NULL
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ILUSetupILUK(nalu_hypre_ParCSRMatrix  *A,
                   NALU_HYPRE_Int            lfil,
                   NALU_HYPRE_Int           *permp,
                   NALU_HYPRE_Int           *qpermp,
                   NALU_HYPRE_Int            nLU,
                   NALU_HYPRE_Int            nI,
                   nalu_hypre_ParCSRMatrix **Lptr,
                   NALU_HYPRE_Real         **Dptr,
                   nalu_hypre_ParCSRMatrix **Uptr,
                   nalu_hypre_ParCSRMatrix **Sptr,
                   NALU_HYPRE_Int          **u_end)
{
   /*
    * 1: Setup and create buffers
    * matL/U: the ParCSR matrix for L and U
    * L/U_diag: the diagonal csr matrix of matL/U
    * A_diag_*: tempory pointer for the diagonal matrix of A and its '*' slot
    * ii = outer loop from 0 to nLU - 1
    * i = the real col number in diag inside the outer loop
    * iw =  working array store the reverse of active col number
    * iL = working array store the active col number
    */

   /* call ILU0 if lfil is 0 */
   if (lfil == 0)
   {
      return nalu_hypre_ILUSetupILU0( A, permp, qpermp, nLU, nI, Lptr, Dptr, Uptr, Sptr, u_end);
   }

   NALU_HYPRE_Real              local_nnz, total_nnz;
   NALU_HYPRE_Int               i, ii, j, k, k1, k2, k3, kl, ku, jpiv, col, icol;
   NALU_HYPRE_Int               *iw;
   MPI_Comm                comm = nalu_hypre_ParCSRMatrixComm(A);
   NALU_HYPRE_Int               num_procs,  my_id;

   /* data objects for A */
   nalu_hypre_CSRMatrix         *A_diag        = nalu_hypre_ParCSRMatrixDiag(A);
   nalu_hypre_CSRMatrix         *A_offd        = nalu_hypre_ParCSRMatrixOffd(A);
   NALU_HYPRE_Real              *A_diag_data   = nalu_hypre_CSRMatrixData(A_diag);
   NALU_HYPRE_Int               *A_diag_i      = nalu_hypre_CSRMatrixI(A_diag);
   NALU_HYPRE_Int               *A_diag_j      = nalu_hypre_CSRMatrixJ(A_diag);
   NALU_HYPRE_Real              *A_offd_data   = nalu_hypre_CSRMatrixData(A_offd);
   NALU_HYPRE_Int               *A_offd_i      = nalu_hypre_CSRMatrixI(A_offd);
   NALU_HYPRE_Int               *A_offd_j      = nalu_hypre_CSRMatrixJ(A_offd);
   NALU_HYPRE_MemoryLocation     memory_location = nalu_hypre_ParCSRMatrixMemoryLocation(A);

   /* data objects for L, D, U */
   nalu_hypre_ParCSRMatrix      *matL;
   nalu_hypre_ParCSRMatrix      *matU;
   nalu_hypre_CSRMatrix         *L_diag;
   nalu_hypre_CSRMatrix         *U_diag;
   NALU_HYPRE_Real              *D_data;
   NALU_HYPRE_Real              *L_diag_data   = NULL;
   NALU_HYPRE_Int               *L_diag_i;
   NALU_HYPRE_Int               *L_diag_j      = NULL;
   NALU_HYPRE_Real              *U_diag_data   = NULL;
   NALU_HYPRE_Int               *U_diag_i;
   NALU_HYPRE_Int               *U_diag_j      = NULL;

   /* data objects for S */
   nalu_hypre_ParCSRMatrix      *matS          = NULL;
   nalu_hypre_CSRMatrix         *S_diag;
   nalu_hypre_CSRMatrix         *S_offd;
   NALU_HYPRE_Real              *S_diag_data   = NULL;
   NALU_HYPRE_Int               *S_diag_i      = NULL;
   NALU_HYPRE_Int               *S_diag_j      = NULL;
   NALU_HYPRE_Int               *S_offd_i      = NULL;
   NALU_HYPRE_Int               *S_offd_j      = NULL;
   NALU_HYPRE_BigInt            *S_offd_colmap = NULL;
   NALU_HYPRE_Real              *S_offd_data;
   NALU_HYPRE_Int               S_offd_nnz, S_offd_ncols;
   NALU_HYPRE_BigInt            col_starts[2];
   NALU_HYPRE_BigInt            total_rows;

   /* communication */
   nalu_hypre_ParCSRCommPkg     *comm_pkg;
   nalu_hypre_ParCSRCommHandle  *comm_handle;
   NALU_HYPRE_BigInt            *send_buf      = NULL;

   /* problem size */
   NALU_HYPRE_Int               n;
   NALU_HYPRE_Int               m;
   NALU_HYPRE_Int               e;
   NALU_HYPRE_Int               m_e;

   /* reverse permutation array */
   NALU_HYPRE_Int               *rperm;
   NALU_HYPRE_Int               *perm, *qperm;

   /* start setup */
   /* check input and get problem size */
   n =  nalu_hypre_CSRMatrixNumRows(A_diag);
   if (nLU < 0 || nLU > n)
   {
      nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_ARG, "WARNING: nLU out of range.\n");
   }
   m = n - nLU;
   e = nI - nLU;
   m_e = n - nI;
   if (e < 0)
   {
      nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_ARG, "WARNING: nLU should not exceed nI.\n");
   }

   /* Init I array anyway. S's might be freed later */
   D_data = nalu_hypre_CTAlloc(NALU_HYPRE_Real, n, memory_location);
   L_diag_i = nalu_hypre_CTAlloc(NALU_HYPRE_Int, (n + 1), memory_location);
   U_diag_i = nalu_hypre_CTAlloc(NALU_HYPRE_Int, (n + 1), memory_location);
   S_diag_i = nalu_hypre_CTAlloc(NALU_HYPRE_Int, (m + 1), memory_location);

   /* set Comm_Pkg if not yet built */
   nalu_hypre_MPI_Comm_size(comm, &num_procs);
   nalu_hypre_MPI_Comm_rank(comm, &my_id);
   comm_pkg = nalu_hypre_ParCSRMatrixCommPkg(A);
   if (!comm_pkg)
   {
      nalu_hypre_MatvecCommPkgCreate(A);
      comm_pkg = nalu_hypre_ParCSRMatrixCommPkg(A);
   }

   /*
    * 2: Symbolic factorization
    * setup iw and rperm first
    */
   /* allocate work arrays */
   iw = nalu_hypre_CTAlloc(NALU_HYPRE_Int, 4 * n, NALU_HYPRE_MEMORY_HOST);
   rperm = iw + 3 * n;
   L_diag_i[0] = U_diag_i[0] = S_diag_i[0] = 0;
   /* get reverse permutation (rperm).
    * rperm holds the reordered indexes.
    */

   if (!permp)
   {
      perm = nalu_hypre_TAlloc(NALU_HYPRE_Int, n, memory_location);
      for (i = 0; i < n; i++)
      {
         perm[i] = i;
      }
   }
   else
   {
      perm = permp;
   }

   if (!qpermp)
   {
      qperm = nalu_hypre_TAlloc(NALU_HYPRE_Int, n, memory_location);
      for (i = 0; i < n; i++)
      {
         qperm[i] = i;
      }
   }
   else
   {
      qperm = qpermp;
   }

   for (i = 0; i < n; i++)
   {
      rperm[qperm[i]] = i;
   }

   /* do symbolic factorization */
   nalu_hypre_ILUSetupILUKSymbolic(n, A_diag_i, A_diag_j, lfil, perm, rperm, iw,
                              nLU, L_diag_i, U_diag_i, S_diag_i, &L_diag_j, &U_diag_j, &S_diag_j, u_end);

   /*
    * after this, we have our I,J for L, U and S ready, and L sorted
    * iw are still -1 after symbolic factorization
    * now setup helper array here
    */
   if (L_diag_i[n])
   {
      L_diag_data = nalu_hypre_CTAlloc(NALU_HYPRE_Real, L_diag_i[n], memory_location);
   }
   if (U_diag_i[n])
   {
      U_diag_data = nalu_hypre_CTAlloc(NALU_HYPRE_Real, U_diag_i[n], memory_location);
   }
   if (S_diag_i[m])
   {
      S_diag_data = nalu_hypre_CTAlloc(NALU_HYPRE_Real, S_diag_i[m], memory_location);
   }

   /*
    * 3: Begin real factorization
    * we already have L and U structure ready, so no extra working array needed
    */
   /* first loop for upper part */
   for (ii = 0; ii < nLU; ii++)
   {
      // get row i
      i = perm[ii];
      kl = L_diag_i[ii + 1];
      ku = U_diag_i[ii + 1];
      k1 = A_diag_i[i];
      k2 = A_diag_i[i + 1];
      /* set up working arrays */
      for (j = L_diag_i[ii]; j < kl; j++)
      {
         col = L_diag_j[j];
         iw[col] = j;
      }
      D_data[ii] = 0.0;
      iw[ii] = ii;
      for (j = U_diag_i[ii]; j < ku; j++)
      {
         col = U_diag_j[j];
         iw[col] = j;
      }
      /* copy data from A into L, D and U */
      for (j = k1; j < k2; j++)
      {
         /* compute everything in new index */
         col = rperm[A_diag_j[j]];
         icol = iw[col];
         /* A for sure to be inside the pattern */
         if (col < ii)
         {
            L_diag_data[icol] = A_diag_data[j];
         }
         else if (col == ii)
         {
            D_data[ii] = A_diag_data[j];
         }
         else
         {
            U_diag_data[icol] = A_diag_data[j];
         }
      }
      /* elimination */
      for (j = L_diag_i[ii]; j < kl; j++)
      {
         jpiv = L_diag_j[j];
         L_diag_data[j] *= D_data[jpiv];
         ku = U_diag_i[jpiv + 1];

         for (k = U_diag_i[jpiv]; k < ku; k++)
         {
            col = U_diag_j[k];
            icol = iw[col];
            if (icol < 0)
            {
               /* not in partern */
               continue;
            }
            if (col < ii)
            {
               /* L part */
               L_diag_data[icol] -= L_diag_data[j] * U_diag_data[k];
            }
            else if (col == ii)
            {
               /* diag part */
               D_data[icol] -= L_diag_data[j] * U_diag_data[k];
            }
            else
            {
               /* U part */
               U_diag_data[icol] -= L_diag_data[j] * U_diag_data[k];
            }
         }
      }
      /* reset working array */
      ku = U_diag_i[ii + 1];
      for (j = L_diag_i[ii]; j < kl; j++)
      {
         col = L_diag_j[j];
         iw[col] = -1;
      }
      iw[ii] = -1;
      for (j = U_diag_i[ii]; j < ku ; j++)
      {
         col = U_diag_j[j];
         iw[col] = -1;
      }

      /* diagonal part (we store the inverse) */
      if (nalu_hypre_abs(D_data[ii]) < MAT_TOL)
      {
         D_data[ii] = 1.0e-06;
      }
      D_data[ii] = 1. / D_data[ii];
   }

   /* Now lower part for Schur complement */
   for (ii = nLU; ii < n; ii++)
   {
      // get row i
      i = perm[ii];
      kl = L_diag_i[ii + 1];
      ku = S_diag_i[ii - nLU + 1];
      k1 = A_diag_i[i];
      k2 = A_diag_i[i + 1];
      /* set up working arrays */
      for (j = L_diag_i[ii]; j < kl; j++)
      {
         col = L_diag_j[j];
         iw[col] = j;
      }
      for (j = S_diag_i[ii - nLU]; j < ku; j++)
      {
         col = S_diag_j[j];
         iw[col] = j;
      }
      /* copy data from A into L, and S */
      for (j = k1; j < k2; j++)
      {
         /* compute everything in new index */
         col = rperm[A_diag_j[j]];
         icol = iw[col];
         /* A for sure to be inside the pattern */
         if (col < nLU)
         {
            L_diag_data[icol] = A_diag_data[j];
         }
         else
         {
            S_diag_data[icol] = A_diag_data[j];
         }
      }
      /* elimination */
      for (j = L_diag_i[ii]; j < kl; j++)
      {
         jpiv = L_diag_j[j];
         L_diag_data[j] *= D_data[jpiv];
         ku = U_diag_i[jpiv + 1];
         for (k = U_diag_i[jpiv]; k < ku; k++)
         {
            col = U_diag_j[k];
            icol = iw[col];
            if (icol < 0)
            {
               /* not in partern */
               continue;
            }
            if (col < nLU)
            {
               /* L part */
               L_diag_data[icol] -= L_diag_data[j] * U_diag_data[k];
            }
            else
            {
               /* S part */
               S_diag_data[icol] -= L_diag_data[j] * U_diag_data[k];
            }
         }
      }
      /* reset working array */
      for (j = L_diag_i[ii]; j < kl ; j++)
      {
         col = L_diag_j[j];
         iw[col] = -1;
      }
      ku = S_diag_i[ii - nLU + 1];
      for (j = S_diag_i[ii - nLU]; j < ku; j++)
      {
         col = S_diag_j[j];
         iw[col] = -1;
         /* remember to update index, S is smaller! */
         S_diag_j[j] -= nLU;
      }
   }

   /*
    * 4: Finishing up and free
    */

   /* First create Schur complement if necessary
    * Check if we need to create Schur complement
    */
   NALU_HYPRE_BigInt big_m = (NALU_HYPRE_BigInt)m;
   nalu_hypre_MPI_Allreduce(&big_m, &total_rows, 1, NALU_HYPRE_MPI_BIG_INT, nalu_hypre_MPI_SUM, comm);
   /* only form when total_rows > 0 */
   if ( total_rows > 0 )
   {
      /* now create S */
      /* need to get new column start */
      {
         NALU_HYPRE_BigInt global_start;
         nalu_hypre_MPI_Scan(&big_m, &global_start, 1, NALU_HYPRE_MPI_BIG_INT, nalu_hypre_MPI_SUM, comm);
         col_starts[0] = global_start - m;
         col_starts[1] = global_start;
      }

      /* We did nothing to A_offd, so all the data kept, just reorder them
       * The create function takes comm, global num rows/cols,
       *    row/col start, num cols offd, nnz diag, nnz offd
       */
      S_offd_nnz = nalu_hypre_CSRMatrixNumNonzeros(A_offd);
      S_offd_ncols = nalu_hypre_CSRMatrixNumCols(A_offd);

      matS = nalu_hypre_ParCSRMatrixCreate( comm,
                                       total_rows,
                                       total_rows,
                                       col_starts,
                                       col_starts,
                                       S_offd_ncols,
                                       S_diag_i[m],
                                       S_offd_nnz);

      /* first put diagonal data in */
      S_diag = nalu_hypre_ParCSRMatrixDiag(matS);

      nalu_hypre_CSRMatrixI(S_diag) = S_diag_i;
      nalu_hypre_CSRMatrixData(S_diag) = S_diag_data;
      nalu_hypre_CSRMatrixJ(S_diag) = S_diag_j;

      /* now start to construct offdiag of S */
      S_offd = nalu_hypre_ParCSRMatrixOffd(matS);
      S_offd_i = nalu_hypre_TAlloc(NALU_HYPRE_Int, m + 1, memory_location);
      S_offd_j = nalu_hypre_TAlloc(NALU_HYPRE_Int, S_offd_nnz, memory_location);
      S_offd_data = nalu_hypre_TAlloc(NALU_HYPRE_Real, S_offd_nnz, memory_location);
      S_offd_colmap = nalu_hypre_CTAlloc(NALU_HYPRE_BigInt, S_offd_ncols, NALU_HYPRE_MEMORY_HOST);

      /* simply use a loop to copy data from A_offd */
      S_offd_i[0] = 0;
      k3 = 0;
      for (i = 1; i <= e; i++)
      {
         S_offd_i[i + 1] = k3;
      }
      for (i = 0; i < m_e; i++)
      {
         col = perm[i + nI];
         k1 = A_offd_i[col];
         k2 = A_offd_i[col + 1];
         for (j = k1; j < k2; j++)
         {
            S_offd_j[k3] = A_offd_j[j];
            S_offd_data[k3++] = A_offd_data[j];
         }
         S_offd_i[i + e + 1] = k3;
      }

      /* give I, J, DATA to S_offd */
      nalu_hypre_CSRMatrixI(S_offd) = S_offd_i;
      nalu_hypre_CSRMatrixJ(S_offd) = S_offd_j;
      nalu_hypre_CSRMatrixData(S_offd) = S_offd_data;

      /* now we need to update S_offd_colmap */

      /* get total num of send */
      NALU_HYPRE_Int num_sends = nalu_hypre_ParCSRCommPkgNumSends(comm_pkg);
      NALU_HYPRE_Int begin = nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, 0);
      NALU_HYPRE_Int end = nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends);
      send_buf = nalu_hypre_TAlloc(NALU_HYPRE_BigInt, end - begin, NALU_HYPRE_MEMORY_HOST);

      /* copy new index into send_buf */
      for (i = begin; i < end; i++)
      {
         send_buf[i - begin] = rperm[nalu_hypre_ParCSRCommPkgSendMapElmt(comm_pkg, i)] - nLU + col_starts[0];
      }

      /* main communication */
      comm_handle = nalu_hypre_ParCSRCommHandleCreate(21, comm_pkg, send_buf, S_offd_colmap);
      nalu_hypre_ParCSRCommHandleDestroy(comm_handle);

      /* setup index */
      nalu_hypre_ParCSRMatrixColMapOffd(matS) = S_offd_colmap;

      nalu_hypre_ILUSortOffdColmap(matS);

      /* free */
      nalu_hypre_TFree(send_buf, NALU_HYPRE_MEMORY_HOST);
   } /* end of forming S */

   /* Assemble LDU matrices */
   /* zero out unfactored rows */
   for (k = nLU; k < n; k++)
   {
      D_data[k] = 1.;
   }

   matL = nalu_hypre_ParCSRMatrixCreate( comm,
                                    nalu_hypre_ParCSRMatrixGlobalNumRows(A),
                                    nalu_hypre_ParCSRMatrixGlobalNumRows(A),
                                    nalu_hypre_ParCSRMatrixRowStarts(A),
                                    nalu_hypre_ParCSRMatrixColStarts(A),
                                    0 /* num_cols_offd */,
                                    L_diag_i[n],
                                    0 /* num_nonzeros_offd */);

   L_diag = nalu_hypre_ParCSRMatrixDiag(matL);
   nalu_hypre_CSRMatrixI(L_diag) = L_diag_i;
   if (L_diag_i[n] > 0)
   {
      nalu_hypre_CSRMatrixData(L_diag) = L_diag_data;
      nalu_hypre_CSRMatrixJ(L_diag) = L_diag_j;
   }
   else
   {
      /* we allocated some initial length, so free them */
      nalu_hypre_TFree(L_diag_j, memory_location);
   }
   /* store (global) total number of nonzeros */
   local_nnz = (NALU_HYPRE_Real) (L_diag_i[n]);
   nalu_hypre_MPI_Allreduce(&local_nnz, &total_nnz, 1, NALU_HYPRE_MPI_REAL, nalu_hypre_MPI_SUM, comm);
   nalu_hypre_ParCSRMatrixDNumNonzeros(matL) = total_nnz;

   matU = nalu_hypre_ParCSRMatrixCreate( comm,
                                    nalu_hypre_ParCSRMatrixGlobalNumRows(A),
                                    nalu_hypre_ParCSRMatrixGlobalNumRows(A),
                                    nalu_hypre_ParCSRMatrixRowStarts(A),
                                    nalu_hypre_ParCSRMatrixColStarts(A),
                                    0,
                                    U_diag_i[n],
                                    0 );

   U_diag = nalu_hypre_ParCSRMatrixDiag(matU);
   nalu_hypre_CSRMatrixI(U_diag) = U_diag_i;
   if (U_diag_i[n] > 0)
   {
      nalu_hypre_CSRMatrixData(U_diag) = U_diag_data;
      nalu_hypre_CSRMatrixJ(U_diag) = U_diag_j;
   }
   else
   {
      /* we allocated some initial length, so free them */
      nalu_hypre_TFree(U_diag_j, memory_location);
   }
   /* store (global) total number of nonzeros */
   local_nnz = (NALU_HYPRE_Real) (U_diag_i[n]);
   nalu_hypre_MPI_Allreduce(&local_nnz, &total_nnz, 1, NALU_HYPRE_MPI_REAL, nalu_hypre_MPI_SUM, comm);
   nalu_hypre_ParCSRMatrixDNumNonzeros(matU) = total_nnz;

   /* free */
   nalu_hypre_TFree(iw, NALU_HYPRE_MEMORY_HOST);
   if (!matS)
   {
      /* we allocate some memory for S, need to free if unused */
      nalu_hypre_TFree(S_diag_i, memory_location);
   }

   if (!permp)
   {
      nalu_hypre_TFree(perm, memory_location);
   }

   if (!qpermp)
   {
      nalu_hypre_TFree(qperm, memory_location);
   }

   /* set matrix pointers */
   *Lptr = matL;
   *Dptr = D_data;
   *Uptr = matU;
   *Sptr = matS;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ILUSetupILUT
 *
 * Setup ILU(t) numeric factorization
 *
 * A: input matrix
 * lfil: maximum nnz per row in L and U
 * tol: droptol array in ILUT
 *    tol[0]: matrix B
 *    tol[1]: matrix E and F
 *    tol[2]: matrix S
 * perm: permutation array indicating ordering of factorization.
 *       Perm could come from a CF_marker array or a reordering routine.
 * qperm: permutation array for column
 * nLU: size of computed LDU factorization.
 *      If nLU < n, Schur complement will be formed
 * nI: number of interial unknowns. nLU should obey nLU <= nI.
 * Lptr, Dptr, Uptr: L, D, U factors.
 * Sptr: Schur complement
 *
 * Keep the largest lfil entries that is greater than some tol relative
 *    to the input tol and the norm of that row in both L and U
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ILUSetupILUT(nalu_hypre_ParCSRMatrix  *A,
                   NALU_HYPRE_Int            lfil,
                   NALU_HYPRE_Real          *tol,
                   NALU_HYPRE_Int           *permp,
                   NALU_HYPRE_Int           *qpermp,
                   NALU_HYPRE_Int            nLU,
                   NALU_HYPRE_Int            nI,
                   nalu_hypre_ParCSRMatrix **Lptr,
                   NALU_HYPRE_Real         **Dptr,
                   nalu_hypre_ParCSRMatrix **Uptr,
                   nalu_hypre_ParCSRMatrix **Sptr,
                   NALU_HYPRE_Int          **u_end)
{
   /*
    * 1: Setup and create buffers
    * matL/U: the ParCSR matrix for L and U
    * L/U_diag: the diagonal csr matrix of matL/U
    * A_diag_*: tempory pointer for the diagonal matrix of A and its '*' slot
    * ii = outer loop from 0 to nLU - 1
    * i = the real col number in diag inside the outer loop
    * iw =  working array store the reverse of active col number
    * iL = working array store the active col number
    */
   NALU_HYPRE_Real               local_nnz, total_nnz;
   NALU_HYPRE_Int                i, ii, j, k, k1, k2, k3, kl, ku, col, icol, lenl, lenu, lenhu, lenhlr,
                            lenhll, jpos, jrow;
   NALU_HYPRE_Real               inorm, itolb, itolef, itols, dpiv, lxu;
   NALU_HYPRE_Int                *iw, *iL;
   NALU_HYPRE_Real               *w;

   /* memory management */
   NALU_HYPRE_Int                ctrL;
   NALU_HYPRE_Int                ctrU;
   NALU_HYPRE_Int                initial_alloc = 0;
   NALU_HYPRE_Int                capacity_L;
   NALU_HYPRE_Int                capacity_U;
   NALU_HYPRE_Int                ctrS;
   NALU_HYPRE_Int                capacity_S;
   NALU_HYPRE_Int                nnz_A;

   /* communication stuffs for S */
   MPI_Comm                 comm             = nalu_hypre_ParCSRMatrixComm(A);
   NALU_HYPRE_Int                S_offd_nnz, S_offd_ncols;
   nalu_hypre_ParCSRCommPkg      *comm_pkg;
   nalu_hypre_ParCSRCommHandle   *comm_handle;
   NALU_HYPRE_Int                num_procs, my_id;
   NALU_HYPRE_BigInt             col_starts[2];
   NALU_HYPRE_BigInt             total_rows;
   NALU_HYPRE_Int                num_sends;
   NALU_HYPRE_Int                begin, end;

   /* data objects for A */
   nalu_hypre_CSRMatrix          *A_diag          = nalu_hypre_ParCSRMatrixDiag(A);
   nalu_hypre_CSRMatrix          *A_offd          = nalu_hypre_ParCSRMatrixOffd(A);
   NALU_HYPRE_Real               *A_diag_data     = nalu_hypre_CSRMatrixData(A_diag);
   NALU_HYPRE_Int                *A_diag_i        = nalu_hypre_CSRMatrixI(A_diag);
   NALU_HYPRE_Int                *A_diag_j        = nalu_hypre_CSRMatrixJ(A_diag);
   NALU_HYPRE_Int                *A_offd_i        = nalu_hypre_CSRMatrixI(A_offd);
   NALU_HYPRE_Int                *A_offd_j        = nalu_hypre_CSRMatrixJ(A_offd);
   NALU_HYPRE_Real               *A_offd_data     = nalu_hypre_CSRMatrixData(A_offd);
   NALU_HYPRE_MemoryLocation      memory_location = nalu_hypre_ParCSRMatrixMemoryLocation(A);

   /* data objects for L, D, U */
   nalu_hypre_ParCSRMatrix       *matL;
   nalu_hypre_ParCSRMatrix       *matU;
   nalu_hypre_CSRMatrix          *L_diag;
   nalu_hypre_CSRMatrix          *U_diag;
   NALU_HYPRE_Real               *D_data;
   NALU_HYPRE_Real               *L_diag_data     = NULL;
   NALU_HYPRE_Int                *L_diag_i;
   NALU_HYPRE_Int                *L_diag_j        = NULL;
   NALU_HYPRE_Real               *U_diag_data     = NULL;
   NALU_HYPRE_Int                *U_diag_i;
   NALU_HYPRE_Int                *U_diag_j        = NULL;

   /* data objects for S */
   nalu_hypre_ParCSRMatrix       *matS            = NULL;
   nalu_hypre_CSRMatrix          *S_diag;
   nalu_hypre_CSRMatrix          *S_offd;
   NALU_HYPRE_Real               *S_diag_data     = NULL;
   NALU_HYPRE_Int                *S_diag_i        = NULL;
   NALU_HYPRE_Int                *S_diag_j        = NULL;
   NALU_HYPRE_Int                *S_offd_i        = NULL;
   NALU_HYPRE_Int                *S_offd_j        = NULL;
   NALU_HYPRE_BigInt                *S_offd_colmap   = NULL;
   NALU_HYPRE_Real               *S_offd_data;
   NALU_HYPRE_BigInt                *send_buf        = NULL;
   NALU_HYPRE_Int                *u_end_array;

   /* reverse permutation */
   NALU_HYPRE_Int                *rperm;
   NALU_HYPRE_Int                *perm, *qperm;

   /* problem size
    * m is n - nLU, num of rows of local Schur system
    * m_e is the size of interface nodes
    * e is the number of interial rows in local Schur Complement
    */
   NALU_HYPRE_Int                n;
   NALU_HYPRE_Int                m;
   NALU_HYPRE_Int                e;
   NALU_HYPRE_Int                m_e;

   /* start setup
    * check input first
    */
   n = nalu_hypre_CSRMatrixNumRows(A_diag);
   if (nLU < 0 || nLU > n)
   {
      nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_ARG, "WARNING: nLU out of range.\n");
   }
   m = n - nLU;
   e = nI - nLU;
   m_e = n - nI;
   if (e < 0)
   {
      nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_ARG, "WARNING: nLU should not exceed nI.\n");
   }

   u_end_array = nalu_hypre_TAlloc(NALU_HYPRE_Int, nLU, NALU_HYPRE_MEMORY_HOST);

   /* start set up
    * setup communication stuffs first
    */
   nalu_hypre_MPI_Comm_size(comm, &num_procs);
   nalu_hypre_MPI_Comm_rank(comm, &my_id);
   comm_pkg = nalu_hypre_ParCSRMatrixCommPkg(A);
   /* create if not yet built */
   if (!comm_pkg)
   {
      nalu_hypre_MatvecCommPkgCreate(A);
      comm_pkg = nalu_hypre_ParCSRMatrixCommPkg(A);
   }

   /* setup initial memory, in ILUT, just guess with max nnz per row */
   nnz_A = A_diag_i[nLU];
   if (n > 0)
   {
      initial_alloc = (NALU_HYPRE_Int)(nalu_hypre_min(nLU + nalu_hypre_ceil((nnz_A / 2.0) * nLU / n),
                                            nLU * lfil));
   }
   capacity_L = initial_alloc;
   capacity_U = initial_alloc;

   D_data = nalu_hypre_CTAlloc(NALU_HYPRE_Real, n, memory_location);
   L_diag_i = nalu_hypre_CTAlloc(NALU_HYPRE_Int, (n + 1), memory_location);
   U_diag_i = nalu_hypre_CTAlloc(NALU_HYPRE_Int, (n + 1), memory_location);

   L_diag_j = nalu_hypre_CTAlloc(NALU_HYPRE_Int, capacity_L, memory_location);
   U_diag_j = nalu_hypre_CTAlloc(NALU_HYPRE_Int, capacity_U, memory_location);
   L_diag_data = nalu_hypre_CTAlloc(NALU_HYPRE_Real, capacity_L, memory_location);
   U_diag_data = nalu_hypre_CTAlloc(NALU_HYPRE_Real, capacity_U, memory_location);

   ctrL = ctrU = 0;

   ctrS = 0;
   S_diag_i = nalu_hypre_CTAlloc(NALU_HYPRE_Int, (m + 1), memory_location);
   S_diag_i[0] = 0;

   /* only setup S part when n > nLU */
   if (m > 0)
   {
      capacity_S = (NALU_HYPRE_Int)(nalu_hypre_min(m + nalu_hypre_ceil((nnz_A / 2.0) * m / n), m * lfil));
      S_diag_j = nalu_hypre_CTAlloc(NALU_HYPRE_Int, capacity_S, memory_location);
      S_diag_data = nalu_hypre_CTAlloc(NALU_HYPRE_Real, capacity_S, memory_location);
   }

   /* setting up working array */
   iw = nalu_hypre_CTAlloc(NALU_HYPRE_Int, 3 * n, NALU_HYPRE_MEMORY_HOST);
   iL = iw + n;
   w = nalu_hypre_CTAlloc(NALU_HYPRE_Real, n, NALU_HYPRE_MEMORY_HOST);
   for (i = 0; i < n; i++)
   {
      iw[i] = -1;
   }
   L_diag_i[0] = U_diag_i[0] = 0;
   /* get reverse permutation (rperm).
    * rperm holds the reordered indexes.
    * rperm[old] -> new
    * perm[new]  -> old
    */
   rperm = iw + 2 * n;

   if (!permp)
   {
      perm = nalu_hypre_TAlloc(NALU_HYPRE_Int, n, memory_location);
      for (i = 0; i < n; i++)
      {
         perm[i] = i;
      }
   }
   else
   {
      perm = permp;
   }

   if (!qpermp)
   {
      qperm = nalu_hypre_TAlloc(NALU_HYPRE_Int, n, memory_location);
      for (i = 0; i < n; i++)
      {
         qperm[i] = i;
      }
   }
   else
   {
      qperm = qpermp;
   }

   for (i = 0; i < n; i++)
   {
      rperm[perm[i]] = i;
   }
   /*
    * 2: Main loop of elimination
    * maintain two heaps
    * |----->*********<-----|-----*********|
    * |col heap***value heap|value in U****|
    */

   /* main outer loop for upper part */
   for (ii = 0; ii < nLU; ii++)
   {
      /* get real row with perm */
      i = perm[ii];
      k1 = A_diag_i[i];
      k2 = A_diag_i[i + 1];
      kl = ii - 1;
      /* reset row norm of ith row */
      inorm = .0;
      for (j = k1; j < k2; j++)
      {
         inorm += nalu_hypre_abs(A_diag_data[j]);
      }
      if (inorm == .0)
      {
         nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_ARG, "WARNING: ILUT with zero row.\n");
      }
      inorm /= (NALU_HYPRE_Real)(k2 - k1);
      /* set the scaled tol for that row */
      itolb = tol[0] * inorm;
      itolef = tol[1] * inorm;

      /* reset displacement */
      lenhll = lenhlr = lenu = 0;
      w[ii] = 0.0;
      iw[ii] = ii;
      /* copy in data from A */
      for (j = k1; j < k2; j++)
      {
         /* get now col number */
         col = rperm[A_diag_j[j]];
         if (col < ii)
         {
            /* L part of it */
            iL[lenhll] = col;
            w[lenhll] = A_diag_data[j];
            iw[col] = lenhll++;
            /* add to heap, by col number */
            nalu_hypre_ILUMinHeapAddIRIi(iL, w, iw, lenhll);
         }
         else if (col == ii)
         {
            w[ii] = A_diag_data[j];
         }
         else
         {
            lenu++;
            jpos = lenu + ii;
            iL[jpos] = col;
            w[jpos] = A_diag_data[j];
            iw[col] = jpos;
         }
      }

      /*
       * main elimination
       * need to maintain 2 heaps for L, one heap for col and one heaps for value
       * maintian an array for U, and do qsplit with quick sort after that
       * while the heap of col is greater than zero
       */
      while (lenhll > 0)
      {

         /* get the next row from top of the heap */
         jrow = iL[0];
         dpiv = w[0] * D_data[jrow];
         w[0] = dpiv;
         /* now remove it from the top of the heap */
         nalu_hypre_ILUMinHeapRemoveIRIi(iL, w, iw, lenhll);
         lenhll--;
         /*
          * reset the drop part to -1
          * we don't need this iw anymore
          */
         iw[jrow] = -1;
         /* need to keep this one, move to the end of the heap */
         /* no longer need to maintain iw */
         nalu_hypre_swap2(iL, w, lenhll, kl - lenhlr);
         lenhlr++;
         nalu_hypre_ILUMaxrHeapAddRabsI(w + kl, iL + kl, lenhlr);
         /* loop for elimination */
         ku = U_diag_i[jrow + 1];
         for (j = U_diag_i[jrow]; j < ku; j++)
         {
            col = U_diag_j[j];
            icol = iw[col];
            lxu = - dpiv * U_diag_data[j];
            /* we don't want to fill small number to empty place */
            if ((icol == -1) &&
                ((col < nLU && nalu_hypre_abs(lxu) < itolb) || (col >= nLU && nalu_hypre_abs(lxu) < itolef)))
            {
               continue;
            }
            if (icol == -1)
            {
               if (col < ii)
               {
                  /* L part
                   * not already in L part
                   * put it to the end of heap
                   * might overwrite some small entries, no issue
                   */
                  iL[lenhll] = col;
                  w[lenhll] = lxu;
                  iw[col] = lenhll++;
                  /* add to heap, by col number */
                  nalu_hypre_ILUMinHeapAddIRIi(iL, w, iw, lenhll);
               }
               else if (col == ii)
               {
                  w[ii] += lxu;
               }
               else
               {
                  /*
                   * not already in U part
                   * put is to the end of heap
                   */
                  lenu++;
                  jpos = lenu + ii;
                  iL[jpos] = col;
                  w[jpos] = lxu;
                  iw[col] = jpos;
               }
            }
            else
            {
               w[icol] += lxu;
            }
         }
      }/* while loop for the elimination of current row */

      if (nalu_hypre_abs(w[ii]) < MAT_TOL)
      {
         w[ii] = 1.0e-06;
      }
      D_data[ii] = 1. / w[ii];
      iw[ii] = -1;

      /*
       * now pick up the largest lfil from L
       * L part is guarantee to be larger than itol
       */

      lenl = lenhlr < lfil ? lenhlr : lfil;
      L_diag_i[ii + 1] = L_diag_i[ii] + lenl;
      if (lenl > 0)
      {
         /* test if memory is enough */
         while (ctrL + lenl > capacity_L)
         {
            NALU_HYPRE_Int tmp = capacity_L;
            capacity_L = (NALU_HYPRE_Int)(capacity_L * EXPAND_FACT + 1);
            L_diag_j = nalu_hypre_TReAlloc_v2(L_diag_j, NALU_HYPRE_Int, tmp, NALU_HYPRE_Int,
                                         capacity_L, memory_location);
            L_diag_data = nalu_hypre_TReAlloc_v2(L_diag_data, NALU_HYPRE_Real, tmp, NALU_HYPRE_Real,
                                            capacity_L, memory_location);
         }
         ctrL += lenl;

         /* copy large data in */
         for (j = L_diag_i[ii]; j < ctrL; j++)
         {
            L_diag_j[j] = iL[kl];
            L_diag_data[j] = w[kl];
            nalu_hypre_ILUMaxrHeapRemoveRabsI(w + kl, iL + kl, lenhlr);
            lenhlr--;
         }
      }
      /*
       * now reset working array
       * L part already reset when move out of heap, only U part
       */
      ku = lenu + ii;
      for (j = ii + 1; j <= ku; j++)
      {
         iw[iL[j]] = -1;
      }

      if (lenu < lfil)
      {
         /* we simply keep all of the data, no need to sort */
         lenhu = lenu;
      }
      else
      {
         /* need to sort the first small(hopefully) part of it */
         lenhu = lfil;
         /* quick split, only sort the first small part of the array */
         nalu_hypre_ILUMaxQSplitRabsI(w, iL, ii + 1, ii + lenhu, ii + lenu);
      }

      U_diag_i[ii + 1] = U_diag_i[ii] + lenhu;
      if (lenhu > 0)
      {
         /* test if memory is enough */
         while (ctrU + lenhu > capacity_U)
         {
            NALU_HYPRE_Int tmp = capacity_U;
            capacity_U = (NALU_HYPRE_Int)(capacity_U * EXPAND_FACT + 1);
            U_diag_j = nalu_hypre_TReAlloc_v2(U_diag_j, NALU_HYPRE_Int, tmp, NALU_HYPRE_Int,
                                         capacity_U, memory_location);
            U_diag_data = nalu_hypre_TReAlloc_v2(U_diag_data, NALU_HYPRE_Real, tmp, NALU_HYPRE_Real,
                                            capacity_U, memory_location);
         }
         ctrU += lenhu;
         /* copy large data in */
         for (j = U_diag_i[ii]; j < ctrU; j++)
         {
            jpos = ii + 1 + j - U_diag_i[ii];
            U_diag_j[j] = iL[jpos];
            U_diag_data[j] = w[jpos];
         }
      }
      /* check and build u_end array */
      if (m > 0)
      {
         nalu_hypre_qsort1(U_diag_j, U_diag_data, U_diag_i[ii], U_diag_i[ii + 1] - 1);
         nalu_hypre_BinarySearch2(U_diag_j, nLU, U_diag_i[ii], U_diag_i[ii + 1] - 1, u_end_array + ii);
      }
      else
      {
         /* Everything is in U */
         u_end_array[ii] = ctrU;
      }
   }/* end of ii loop from 0 to nLU-1 */


   /* now main loop for Schur comlement part */
   for (ii = nLU; ii < n; ii++)
   {
      /* get real row with perm */
      i = perm[ii];
      k1 = A_diag_i[i];
      k2 = A_diag_i[i + 1];
      kl = nLU - 1;
      /* reset row norm of ith row */
      inorm = .0;
      for (j = k1; j < k2; j++)
      {
         inorm += nalu_hypre_abs(A_diag_data[j]);
      }
      if (inorm == .0)
      {
         nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_ARG, "WARNING: ILUT with zero row.\n");
      }
      inorm /= (NALU_HYPRE_Real)(k2 - k1);
      /* set the scaled tol for that row */
      itols = tol[2] * inorm;
      itolef = tol[1] * inorm;

      /* reset displacement */
      lenhll = lenhlr = lenu = 0;
      /* copy in data from A */
      for (j = k1; j < k2; j++)
      {
         /* get now col number */
         col = rperm[A_diag_j[j]];
         if (col < nLU)
         {
            /* L part of it */
            iL[lenhll] = col;
            w[lenhll] = A_diag_data[j];
            iw[col] = lenhll++;
            /* add to heap, by col number */
            nalu_hypre_ILUMinHeapAddIRIi(iL, w, iw, lenhll);
         }
         else if (col == ii)
         {
            /* the diagonla entry of S */
            iL[nLU] = col;
            w[nLU] = A_diag_data[j];
            iw[col] = nLU;
         }
         else
         {
            /* S part of it */
            lenu++;
            jpos = lenu + nLU;
            iL[jpos] = col;
            w[jpos] = A_diag_data[j];
            iw[col] = jpos;
         }
      }

      /*
       * main elimination
       * need to maintain 2 heaps for L, one heap for col and one heaps for value
       * maintian an array for S, and do qsplit with quick sort after that
       * while the heap of col is greater than zero
       */
      while (lenhll > 0)
      {
         /* get the next row from top of the heap */
         jrow = iL[0];
         dpiv = w[0] * D_data[jrow];
         w[0] = dpiv;
         /* now remove it from the top of the heap */
         nalu_hypre_ILUMinHeapRemoveIRIi(iL, w, iw, lenhll);
         lenhll--;
         /*
          * reset the drop part to -1
          * we don't need this iw anymore
          */
         iw[jrow] = -1;
         /* need to keep this one, move to the end of the heap */
         /* no longer need to maintain iw */
         nalu_hypre_swap2(iL, w, lenhll, kl - lenhlr);
         lenhlr++;
         nalu_hypre_ILUMaxrHeapAddRabsI(w + kl, iL + kl, lenhlr);
         /* loop for elimination */
         ku = U_diag_i[jrow + 1];
         for (j = U_diag_i[jrow]; j < ku; j++)
         {
            col = U_diag_j[j];
            icol = iw[col];
            lxu = - dpiv * U_diag_data[j];
            /* we don't want to fill small number to empty place */
            if ((icol == -1) &&
                ((col < nLU  && nalu_hypre_abs(lxu) < itolef) ||
                 (col >= nLU && nalu_hypre_abs(lxu) < itols )))
            {
               continue;
            }
            if (icol == -1)
            {
               if (col < nLU)
               {
                  /* L part
                   * not already in L part
                   * put it to the end of heap
                   * might overwrite some small entries, no issue
                   */
                  iL[lenhll] = col;
                  w[lenhll] = lxu;
                  iw[col] = lenhll++;
                  /* add to heap, by col number */
                  nalu_hypre_ILUMinHeapAddIRIi(iL, w, iw, lenhll);
               }
               else if (col == ii)
               {
                  /* the diagonla entry of S */
                  iL[nLU] = col;
                  w[nLU] = A_diag_data[j];
                  iw[col] = nLU;
               }
               else
               {
                  /*
                   * not already in S part
                   * put is to the end of heap
                   */
                  lenu++;
                  jpos = lenu + nLU;
                  iL[jpos] = col;
                  w[jpos] = lxu;
                  iw[col] = jpos;
               }
            }
            else
            {
               w[icol] += lxu;
            }
         }
      }/* while loop for the elimination of current row */

      /*
       * now pick up the largest lfil from L
       * L part is guarantee to be larger than itol
       */

      lenl = lenhlr < lfil ? lenhlr : lfil;
      L_diag_i[ii + 1] = L_diag_i[ii] + lenl;
      if (lenl > 0)
      {
         /* test if memory is enough */
         while (ctrL + lenl > capacity_L)
         {
            NALU_HYPRE_Int tmp = capacity_L;
            capacity_L = (NALU_HYPRE_Int)(capacity_L * EXPAND_FACT + 1);
            L_diag_j = nalu_hypre_TReAlloc_v2(L_diag_j, NALU_HYPRE_Int, tmp, NALU_HYPRE_Int,
                                         capacity_L, memory_location);
            L_diag_data = nalu_hypre_TReAlloc_v2(L_diag_data, NALU_HYPRE_Real, tmp, NALU_HYPRE_Real,
                                            capacity_L, memory_location);
         }
         ctrL += lenl;

         /* copy large data in */
         for (j = L_diag_i[ii]; j < ctrL; j++)
         {
            L_diag_j[j] = iL[kl];
            L_diag_data[j] = w[kl];
            nalu_hypre_ILUMaxrHeapRemoveRabsI(w + kl, iL + kl, lenhlr);
            lenhlr--;
         }
      }
      /*
       * now reset working array
       * L part already reset when move out of heap, only S part
       */
      ku = lenu + nLU;
      for (j = nLU; j <= ku; j++)
      {
         iw[iL[j]] = -1;
      }

      /* no dropping at this point of time for S */
      //lenhu = lenu < lfil ? lenu : lfil;
      lenhu = lenu;
      /* quick split, only sort the first small part of the array */
      nalu_hypre_ILUMaxQSplitRabsI(w, iL, nLU + 1, nLU + lenhu, nLU + lenu);
      /* we have diagonal in S anyway */
      /* test if memory is enough */
      while (ctrS + lenhu + 1 > capacity_S)
      {
         NALU_HYPRE_Int tmp = capacity_S;
         capacity_S = (NALU_HYPRE_Int)(capacity_S * EXPAND_FACT + 1);
         S_diag_j = nalu_hypre_TReAlloc_v2(S_diag_j, NALU_HYPRE_Int, tmp,
                                      NALU_HYPRE_Int, capacity_S, memory_location);
         S_diag_data = nalu_hypre_TReAlloc_v2(S_diag_data, NALU_HYPRE_Real, tmp,
                                         NALU_HYPRE_Real, capacity_S, memory_location);
      }

      ctrS += (lenhu + 1);
      S_diag_i[ii - nLU + 1] = ctrS;

      /* copy large data in, diagonal first */
      S_diag_j[S_diag_i[ii - nLU]] = iL[nLU] - nLU;
      S_diag_data[S_diag_i[ii - nLU]] = w[nLU];
      for (j = S_diag_i[ii - nLU] + 1; j < ctrS; j++)
      {
         jpos = nLU + j - S_diag_i[ii - nLU];
         S_diag_j[j] = iL[jpos] - nLU;
         S_diag_data[j] = w[jpos];
      }
   }/* end of ii loop from nLU to n-1 */

   /*
    * 3: Finishing up and free
    */

   /* First create Schur complement if necessary
    * Check if we need to create Schur complement
    */
   NALU_HYPRE_BigInt big_m = (NALU_HYPRE_BigInt)m;
   nalu_hypre_MPI_Allreduce(&big_m, &total_rows, 1, NALU_HYPRE_MPI_BIG_INT, nalu_hypre_MPI_SUM, comm);

   /* only form when total_rows > 0 */
   if ( total_rows > 0 )
   {
      /* now create S */
      /* need to get new column start */
      {
         NALU_HYPRE_BigInt global_start;
         nalu_hypre_MPI_Scan(&big_m, &global_start, 1, NALU_HYPRE_MPI_BIG_INT, nalu_hypre_MPI_SUM, comm);
         col_starts[0] = global_start - m;
         col_starts[1] = global_start;
      }
      /* We did nothing to A_offd, so all the data kept, just reorder them
       * The create function takes comm, global num rows/cols,
       *    row/col start, num cols offd, nnz diag, nnz offd
       */
      S_offd_nnz = nalu_hypre_CSRMatrixNumNonzeros(A_offd);
      S_offd_ncols = nalu_hypre_CSRMatrixNumCols(A_offd);

      matS = nalu_hypre_ParCSRMatrixCreate( comm,
                                       total_rows,
                                       total_rows,
                                       col_starts,
                                       col_starts,
                                       S_offd_ncols,
                                       S_diag_i[m],
                                       S_offd_nnz);

      /* first put diagonal data in */
      S_diag = nalu_hypre_ParCSRMatrixDiag(matS);

      nalu_hypre_CSRMatrixI(S_diag) = S_diag_i;
      nalu_hypre_CSRMatrixData(S_diag) = S_diag_data;
      nalu_hypre_CSRMatrixJ(S_diag) = S_diag_j;

      /* now start to construct offdiag of S */
      S_offd = nalu_hypre_ParCSRMatrixOffd(matS);
      S_offd_i = nalu_hypre_TAlloc(NALU_HYPRE_Int, m + 1, memory_location);
      S_offd_j = nalu_hypre_TAlloc(NALU_HYPRE_Int, S_offd_nnz, memory_location);
      S_offd_data = nalu_hypre_TAlloc(NALU_HYPRE_Real, S_offd_nnz, memory_location);
      S_offd_colmap = nalu_hypre_CTAlloc(NALU_HYPRE_BigInt, S_offd_ncols, NALU_HYPRE_MEMORY_HOST);

      /* simply use a loop to copy data from A_offd */
      S_offd_i[0] = 0;
      k3 = 0;
      for (i = 1; i <= e; i++)
      {
         S_offd_i[i] = k3;
      }
      for (i = 0; i < m_e; i++)
      {
         col = perm[i + nI];
         k1 = A_offd_i[col];
         k2 = A_offd_i[col + 1];
         for (j = k1; j < k2; j++)
         {
            S_offd_j[k3] = A_offd_j[j];
            S_offd_data[k3++] = A_offd_data[j];
         }
         S_offd_i[i + e + 1] = k3;
      }

      /* give I, J, DATA to S_offd */
      nalu_hypre_CSRMatrixI(S_offd) = S_offd_i;
      nalu_hypre_CSRMatrixJ(S_offd) = S_offd_j;
      nalu_hypre_CSRMatrixData(S_offd) = S_offd_data;

      /* now we need to update S_offd_colmap */

      /* get total num of send */
      num_sends = nalu_hypre_ParCSRCommPkgNumSends(comm_pkg);
      begin = nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, 0);
      end = nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends);
      send_buf = nalu_hypre_TAlloc(NALU_HYPRE_BigInt, end - begin, NALU_HYPRE_MEMORY_HOST);
      /* copy new index into send_buf */
      for (i = begin; i < end; i++)
      {
         send_buf[i - begin] = rperm[nalu_hypre_ParCSRCommPkgSendMapElmt(comm_pkg, i)] - nLU + col_starts[0];
      }

      /* main communication */
      comm_handle = nalu_hypre_ParCSRCommHandleCreate(21, comm_pkg, send_buf, S_offd_colmap);
      /* need this to synchronize, Isend & Irecv used in above functions */
      nalu_hypre_ParCSRCommHandleDestroy(comm_handle);

      /* setup index */
      nalu_hypre_ParCSRMatrixColMapOffd(matS) = S_offd_colmap;

      nalu_hypre_ILUSortOffdColmap(matS);

      /* free */
      nalu_hypre_TFree(send_buf, NALU_HYPRE_MEMORY_HOST);
   } /* end of forming S */

   /* now start to construct L and U */
   for (k = nLU; k < n; k++)
   {
      /* set U after nLU to be 0, and diag to be one */
      U_diag_i[k + 1] = U_diag_i[nLU];
      D_data[k] = 1.;
   }

   /* create parcsr matrix */
   matL = nalu_hypre_ParCSRMatrixCreate( comm,
                                    nalu_hypre_ParCSRMatrixGlobalNumRows(A),
                                    nalu_hypre_ParCSRMatrixGlobalNumRows(A),
                                    nalu_hypre_ParCSRMatrixRowStarts(A),
                                    nalu_hypre_ParCSRMatrixColStarts(A),
                                    0,
                                    L_diag_i[n],
                                    0 );

   L_diag = nalu_hypre_ParCSRMatrixDiag(matL);
   nalu_hypre_CSRMatrixI(L_diag) = L_diag_i;
   if (L_diag_i[n] > 0)
   {
      nalu_hypre_CSRMatrixData(L_diag) = L_diag_data;
      nalu_hypre_CSRMatrixJ(L_diag) = L_diag_j;
   }
   else
   {
      /* we initialized some anyway, so remove if unused */
      nalu_hypre_TFree(L_diag_j, memory_location);
      nalu_hypre_TFree(L_diag_data, memory_location);
   }
   /* store (global) total number of nonzeros */
   local_nnz = (NALU_HYPRE_Real) (L_diag_i[n]);
   nalu_hypre_MPI_Allreduce(&local_nnz, &total_nnz, 1, NALU_HYPRE_MPI_REAL, nalu_hypre_MPI_SUM, comm);
   nalu_hypre_ParCSRMatrixDNumNonzeros(matL) = total_nnz;

   matU = nalu_hypre_ParCSRMatrixCreate( comm,
                                    nalu_hypre_ParCSRMatrixGlobalNumRows(A),
                                    nalu_hypre_ParCSRMatrixGlobalNumRows(A),
                                    nalu_hypre_ParCSRMatrixRowStarts(A),
                                    nalu_hypre_ParCSRMatrixColStarts(A),
                                    0,
                                    U_diag_i[n],
                                    0 );

   U_diag = nalu_hypre_ParCSRMatrixDiag(matU);
   nalu_hypre_CSRMatrixI(U_diag) = U_diag_i;
   if (U_diag_i[n] > 0)
   {
      nalu_hypre_CSRMatrixData(U_diag) = U_diag_data;
      nalu_hypre_CSRMatrixJ(U_diag) = U_diag_j;
   }
   else
   {
      /* we initialized some anyway, so remove if unused */
      nalu_hypre_TFree(U_diag_j, memory_location);
      nalu_hypre_TFree(U_diag_data, memory_location);
   }
   /* store (global) total number of nonzeros */
   local_nnz = (NALU_HYPRE_Real) (U_diag_i[n]);
   nalu_hypre_MPI_Allreduce(&local_nnz, &total_nnz, 1, NALU_HYPRE_MPI_REAL, nalu_hypre_MPI_SUM, comm);
   nalu_hypre_ParCSRMatrixDNumNonzeros(matU) = total_nnz;

   /* free working array */
   nalu_hypre_TFree(iw, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(w, NALU_HYPRE_MEMORY_HOST);

   if (!matS)
   {
      nalu_hypre_TFree(S_diag_i, memory_location);
   }

   if (!permp)
   {
      nalu_hypre_TFree(perm, memory_location);
   }

   if (!qpermp)
   {
      nalu_hypre_TFree(qperm, memory_location);
   }

   /* set matrix pointers */
   *Lptr = matL;
   *Dptr = D_data;
   *Uptr = matU;
   *Sptr = matS;
   *u_end = u_end_array;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_NSHSetup
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_NSHSetup( void               *nsh_vdata,
                nalu_hypre_ParCSRMatrix *A,
                nalu_hypre_ParVector    *f,
                nalu_hypre_ParVector    *u )
{
   MPI_Comm             comm              = nalu_hypre_ParCSRMatrixComm(A);
   nalu_hypre_ParNSHData     *nsh_data         = (nalu_hypre_ParNSHData*) nsh_vdata;

   /* Pointers to NSH data */
   NALU_HYPRE_Int             logging          = nalu_hypre_ParNSHDataLogging(nsh_data);
   NALU_HYPRE_Int             print_level      = nalu_hypre_ParNSHDataPrintLevel(nsh_data);
   nalu_hypre_ParCSRMatrix   *matA             = nalu_hypre_ParNSHDataMatA(nsh_data);
   nalu_hypre_ParCSRMatrix   *matM             = nalu_hypre_ParNSHDataMatM(nsh_data);
   nalu_hypre_ParVector      *Utemp;
   nalu_hypre_ParVector      *Ftemp;
   nalu_hypre_ParVector      *F_array          = nalu_hypre_ParNSHDataF(nsh_data);
   nalu_hypre_ParVector      *U_array          = nalu_hypre_ParNSHDataU(nsh_data);
   nalu_hypre_ParVector      *residual         = nalu_hypre_ParNSHDataResidual(nsh_data);
   NALU_HYPRE_Real           *rel_res_norms    = nalu_hypre_ParNSHDataRelResNorms(nsh_data);

   /* Solver setting */
   NALU_HYPRE_Real           *droptol          = nalu_hypre_ParNSHDataDroptol(nsh_data);
   NALU_HYPRE_Real            mr_tol           = nalu_hypre_ParNSHDataMRTol(nsh_data);
   NALU_HYPRE_Int             mr_max_row_nnz   = nalu_hypre_ParNSHDataMRMaxRowNnz(nsh_data);
   NALU_HYPRE_Int             mr_max_iter      = nalu_hypre_ParNSHDataMRMaxIter(nsh_data);
   NALU_HYPRE_Int             mr_col_version   = nalu_hypre_ParNSHDataMRColVersion(nsh_data);
   NALU_HYPRE_Real            nsh_tol          = nalu_hypre_ParNSHDataNSHTol(nsh_data);
   NALU_HYPRE_Int             nsh_max_row_nnz  = nalu_hypre_ParNSHDataNSHMaxRowNnz(nsh_data);
   NALU_HYPRE_Int             nsh_max_iter     = nalu_hypre_ParNSHDataNSHMaxIter(nsh_data);
   NALU_HYPRE_Int             num_procs,  my_id;

   nalu_hypre_MPI_Comm_size(comm, &num_procs);
   nalu_hypre_MPI_Comm_rank(comm, &my_id);

   /* Free Previously allocated data, if any not destroyed */
   nalu_hypre_TFree(matM, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(nalu_hypre_ParNSHDataL1Norms(nsh_data), NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_ParVectorDestroy(nalu_hypre_ParNSHDataUTemp(nsh_data));
   nalu_hypre_ParVectorDestroy(nalu_hypre_ParNSHDataFTemp(nsh_data));
   nalu_hypre_ParVectorDestroy(nalu_hypre_ParNSHDataResidual(nsh_data));
   nalu_hypre_TFree(nalu_hypre_ParNSHDataRelResNorms(nsh_data), NALU_HYPRE_MEMORY_HOST);

   matM = NULL;
   nalu_hypre_ParNSHDataL1Norms(nsh_data)     = NULL;
   nalu_hypre_ParNSHDataUTemp(nsh_data)       = NULL;
   nalu_hypre_ParNSHDataFTemp(nsh_data)       = NULL;
   nalu_hypre_ParNSHDataResidual(nsh_data)    = NULL;
   nalu_hypre_ParNSHDataRelResNorms(nsh_data) = NULL;

   /* start to create working vectors */
   Utemp = nalu_hypre_ParVectorCreate(nalu_hypre_ParCSRMatrixComm(A),
                                 nalu_hypre_ParCSRMatrixGlobalNumRows(A),
                                 nalu_hypre_ParCSRMatrixRowStarts(A));
   nalu_hypre_ParVectorInitialize(Utemp);
   nalu_hypre_ParNSHDataUTemp(nsh_data) = Utemp;

   Ftemp = nalu_hypre_ParVectorCreate(nalu_hypre_ParCSRMatrixComm(A),
                                 nalu_hypre_ParCSRMatrixGlobalNumRows(A),
                                 nalu_hypre_ParCSRMatrixRowStarts(A));
   nalu_hypre_ParVectorInitialize(Ftemp);
   nalu_hypre_ParNSHDataFTemp(nsh_data) = Ftemp;

   /* Set matrix, solution and rhs pointers */
   matA = A;
   F_array = f;
   U_array = u;

   /* NSH compute approximate inverse, see par_ilu.c */
   nalu_hypre_ILUParCSRInverseNSH(matA, &matM, droptol, mr_tol, nsh_tol, NALU_HYPRE_REAL_MIN,
                             mr_max_row_nnz, nsh_max_row_nnz, mr_max_iter, nsh_max_iter,
                             mr_col_version, print_level);

   /* Set pointers to NSH data */
   nalu_hypre_ParNSHDataMatA(nsh_data) = matA;
   nalu_hypre_ParNSHDataF(nsh_data)    = F_array;
   nalu_hypre_ParNSHDataU(nsh_data)    = U_array;
   nalu_hypre_ParNSHDataMatM(nsh_data) = matM;

   /* Compute operator complexity */
   nalu_hypre_ParCSRMatrixSetDNumNonzeros(matA);
   nalu_hypre_ParCSRMatrixSetDNumNonzeros(matM);

   /* Compute complexity */
   nalu_hypre_ParNSHDataOperatorComplexity(nsh_data) = nalu_hypre_ParCSRMatrixDNumNonzeros(matM) /
                                                  nalu_hypre_ParCSRMatrixDNumNonzeros(matA);
   if (my_id == 0 && print_level > 0)
   {
      nalu_hypre_printf("NSH SETUP: operator complexity = %f  \n",
                   nalu_hypre_ParNSHDataOperatorComplexity(nsh_data));
   }

   if (logging > 1)
   {
      residual = nalu_hypre_ParVectorCreate(nalu_hypre_ParCSRMatrixComm(matA),
                                       nalu_hypre_ParCSRMatrixGlobalNumRows(matA),
                                       nalu_hypre_ParCSRMatrixRowStarts(matA));
      nalu_hypre_ParVectorInitialize(residual);
      nalu_hypre_ParNSHDataResidual(nsh_data) = residual;
   }
   else
   {
      nalu_hypre_ParNSHDataResidual(nsh_data) = NULL;
   }

   rel_res_norms = nalu_hypre_CTAlloc(NALU_HYPRE_Real, nalu_hypre_ParNSHDataMaxIter(nsh_data),
                                 NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_ParNSHDataRelResNorms(nsh_data) = rel_res_norms;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ILUSetupILU0RAS
 *
 * ILU(0) for RAS, has some external rows
 *
 * A = input matrix
 * perm = permutation array indicating ordering of factorization.
 *        Perm could come from a CF_marker array or a reordering routine.
 * nLU = size of computed LDU factorization.
 * Lptr, Dptr, Uptr, Sptr = L, D, U, S factors.
 * will form global Schur Matrix if nLU < n
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ILUSetupILU0RAS(nalu_hypre_ParCSRMatrix  *A,
                      NALU_HYPRE_Int           *perm,
                      NALU_HYPRE_Int            nLU,
                      nalu_hypre_ParCSRMatrix **Lptr,
                      NALU_HYPRE_Real         **Dptr,
                      nalu_hypre_ParCSRMatrix **Uptr)
{
   /* communication stuffs for S */
   MPI_Comm                 comm          = nalu_hypre_ParCSRMatrixComm(A);
   NALU_HYPRE_Int                num_procs;
   nalu_hypre_ParCSRCommPkg      *comm_pkg;

   /* data objects for A */
   nalu_hypre_CSRMatrix          *A_diag       = nalu_hypre_ParCSRMatrixDiag(A);
   nalu_hypre_CSRMatrix          *A_offd       = nalu_hypre_ParCSRMatrixOffd(A);
   NALU_HYPRE_Real               *A_diag_data  = nalu_hypre_CSRMatrixData(A_diag);
   NALU_HYPRE_Int                *A_diag_i     = nalu_hypre_CSRMatrixI(A_diag);
   NALU_HYPRE_Int                *A_diag_j     = nalu_hypre_CSRMatrixJ(A_diag);
   NALU_HYPRE_Real               *A_offd_data  = nalu_hypre_CSRMatrixData(A_offd);
   NALU_HYPRE_Int                *A_offd_i     = nalu_hypre_CSRMatrixI(A_offd);
   NALU_HYPRE_Int                *A_offd_j     = nalu_hypre_CSRMatrixJ(A_offd);
   NALU_HYPRE_MemoryLocation      memory_location = nalu_hypre_ParCSRMatrixMemoryLocation(A);

   /* size of problem and external matrix */
   NALU_HYPRE_Int                n             =  nalu_hypre_CSRMatrixNumRows(A_diag);
   NALU_HYPRE_Int                ext           = nalu_hypre_CSRMatrixNumCols(A_offd);
   NALU_HYPRE_Int                total_rows    = n + ext;
   NALU_HYPRE_BigInt             col_starts[2];
   NALU_HYPRE_BigInt             global_num_rows;
   NALU_HYPRE_Real               local_nnz, total_nnz;

   /* data objects for L, D, U */
   nalu_hypre_ParCSRMatrix       *matL;
   nalu_hypre_ParCSRMatrix       *matU;
   nalu_hypre_CSRMatrix          *L_diag;
   nalu_hypre_CSRMatrix          *U_diag;
   NALU_HYPRE_Real               *D_data;
   NALU_HYPRE_Real               *L_diag_data;
   NALU_HYPRE_Int                *L_diag_i;
   NALU_HYPRE_Int                *L_diag_j;
   NALU_HYPRE_Real               *U_diag_data;
   NALU_HYPRE_Int                *U_diag_i;
   NALU_HYPRE_Int                *U_diag_j;

   /* data objects for E, external matrix */
   NALU_HYPRE_Int                *E_i;
   NALU_HYPRE_Int                *E_j;
   NALU_HYPRE_Real               *E_data;

   /* memory management */
   NALU_HYPRE_Int                initial_alloc = 0;
   NALU_HYPRE_Int                capacity_L;
   NALU_HYPRE_Int                capacity_U;
   NALU_HYPRE_Int                nnz_A = A_diag_i[n];

   /* reverse permutation array */
   NALU_HYPRE_Int                *rperm;

   /* the original permutation array */
   NALU_HYPRE_Int                *perm_old;

   NALU_HYPRE_Int                i, ii, j, k, k1, k2, ctrU, ctrL, lenl, lenu, jpiv, col, jpos;
   NALU_HYPRE_Int                *iw, *iL, *iU;
   NALU_HYPRE_Real               dd, t, dpiv, lxu, *wU, *wL;

   /* start setup
    * get communication stuffs first
    */
   nalu_hypre_MPI_Comm_size(comm, &num_procs);
   comm_pkg = nalu_hypre_ParCSRMatrixCommPkg(A);

   /* Setup if not yet built */
   if (!comm_pkg)
   {
      nalu_hypre_MatvecCommPkgCreate(A);
      comm_pkg = nalu_hypre_ParCSRMatrixCommPkg(A);
   }

   /* check for correctness */
   if (nLU < 0 || nLU > n)
   {
      nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_ARG, "WARNING: nLU out of range.\n");
   }

   /* Allocate memory for L,D,U,S factors */
   if (n > 0)
   {
      initial_alloc = (NALU_HYPRE_Int)((n + ext) + nalu_hypre_ceil((nnz_A / 2.0) * total_rows / n));
   }
   capacity_L = initial_alloc;
   capacity_U = initial_alloc;

   D_data      = nalu_hypre_TAlloc(NALU_HYPRE_Real, total_rows, memory_location);
   L_diag_i    = nalu_hypre_TAlloc(NALU_HYPRE_Int, total_rows + 1, memory_location);
   L_diag_j    = nalu_hypre_TAlloc(NALU_HYPRE_Int, capacity_L, memory_location);
   L_diag_data = nalu_hypre_TAlloc(NALU_HYPRE_Real, capacity_L, memory_location);
   U_diag_i    = nalu_hypre_TAlloc(NALU_HYPRE_Int, total_rows + 1, memory_location);
   U_diag_j    = nalu_hypre_TAlloc(NALU_HYPRE_Int, capacity_U, memory_location);
   U_diag_data = nalu_hypre_TAlloc(NALU_HYPRE_Real, capacity_U, memory_location);

   /* allocate working arrays */
   iw          = nalu_hypre_TAlloc(NALU_HYPRE_Int, 4 * total_rows, NALU_HYPRE_MEMORY_HOST);
   iL          = iw + total_rows;
   rperm       = iw + 2 * total_rows;
   perm_old    = perm;
   perm        = iw + 3 * total_rows;
   wL          = nalu_hypre_TAlloc(NALU_HYPRE_Real, total_rows, NALU_HYPRE_MEMORY_HOST);
   ctrU = ctrL = 0;
   L_diag_i[0] = U_diag_i[0] = 0;

   /* set marker array iw to -1 */
   for (i = 0; i < total_rows; i++)
   {
      iw[i] = -1;
   }

   /* expand perm to suit extra data, remember to free */
   for (i = 0; i < n; i++)
   {
      perm[i] = perm_old[i];
   }
   for (i = n; i < total_rows; i++)
   {
      perm[i] = i;
   }

   /* get reverse permutation (rperm).
    * rperm holds the reordered indexes.
    */
   for (i = 0; i < total_rows; i++)
   {
      rperm[perm[i]] = i;
   }

   /* get external rows */
   nalu_hypre_ILUBuildRASExternalMatrix(A, rperm, &E_i, &E_j, &E_data);

   /*---------  Begin Factorization. Work in permuted space  ----
    * this is the first part, without offd
    */
   for (ii = 0; ii < nLU; ii++)
   {
      // get row i
      i = perm[ii];
      // get extents of row i
      k1 = A_diag_i[i];
      k2 = A_diag_i[i + 1];

      /*-------------------- unpack L & U-parts of row of A in arrays w */
      iU = iL + ii;
      wU = wL + ii;
      /*--------------------  diagonal entry */
      dd = 0.0;
      lenl  = lenu = 0;
      iw[ii] = ii;
      /*-------------------- scan & unwrap column */
      for (j = k1; j < k2; j++)
      {
         col = rperm[A_diag_j[j]];
         t = A_diag_data[j];
         if ( col < ii )
         {
            iw[col] = lenl;
            iL[lenl] = col;
            wL[lenl++] = t;
         }
         else if (col > ii)
         {
            iw[col] = lenu;
            iU[lenu] = col;
            wU[lenu++] = t;
         }
         else
         {
            dd = t;
         }
      }

      /* eliminate row */
      /*-------------------------------------------------------------------------
       *  In order to do the elimination in the correct order we must select the
       *  smallest column index among iL[k], k = j, j+1, ..., lenl-1. For ILU(0),
       *  no new fill-ins are expect, so we can pre-sort iL and wL prior to the
       *  entering the elimination loop.
       *-----------------------------------------------------------------------*/
      //      nalu_hypre_quickSortIR(iL, wL, iw, 0, (lenl-1));
      nalu_hypre_qsort3ir(iL, wL, iw, 0, (lenl - 1));
      for (j = 0; j < lenl; j++)
      {
         jpiv = iL[j];
         /* get factor/ pivot element */
         dpiv = wL[j] * D_data[jpiv];
         /* store entry in L */
         wL[j] = dpiv;

         /* zero out element - reset pivot */
         iw[jpiv] = -1;
         /* combine current row and pivot row */
         for (k = U_diag_i[jpiv]; k < U_diag_i[jpiv + 1]; k++)
         {
            col = U_diag_j[k];
            jpos = iw[col];

            /* Only fill-in nonzero pattern (jpos != 0) */
            if (jpos < 0)
            {
               continue;
            }

            lxu = - U_diag_data[k] * dpiv;
            if (col < ii)
            {
               /* dealing with L part */
               wL[jpos] += lxu;
            }
            else if (col > ii)
            {
               /* dealing with U part */
               wU[jpos] += lxu;
            }
            else
            {
               /* diagonal update */
               dd += lxu;
            }
         }
      }
      /* restore iw (only need to restore diagonal and U part */
      iw[ii] = -1;
      for (j = 0; j < lenu; j++)
      {
         iw[iU[j]] = -1;
      }

      /* Update LDU factors */
      /* L part */
      /* Check that memory is sufficient */
      while ((ctrL + lenl) > capacity_L)
      {
         NALU_HYPRE_Int tmp = capacity_L;
         capacity_L = (NALU_HYPRE_Int)(capacity_L * EXPAND_FACT + 1);
         L_diag_j = nalu_hypre_TReAlloc_v2(L_diag_j, NALU_HYPRE_Int, tmp,
                                      NALU_HYPRE_Int, capacity_L, memory_location);
         L_diag_data = nalu_hypre_TReAlloc_v2(L_diag_data, NALU_HYPRE_Real, tmp,
                                         NALU_HYPRE_Real, capacity_L, memory_location);
      }
      nalu_hypre_TMemcpy(&(L_diag_j)[ctrL], iL, NALU_HYPRE_Int, lenl, memory_location, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TMemcpy(&(L_diag_data)[ctrL], wL, NALU_HYPRE_Real, lenl, memory_location, NALU_HYPRE_MEMORY_HOST);
      L_diag_i[ii + 1] = (ctrL += lenl);

      /* diagonal part (we store the inverse) */
      if (nalu_hypre_abs(dd) < MAT_TOL)
      {
         dd = 1.0e-6;
      }
      D_data[ii] = 1. / dd;

      /* U part */
      /* Check that memory is sufficient */
      while ((ctrU + lenu) > capacity_U)
      {
         NALU_HYPRE_Int tmp = capacity_U;
         capacity_U = (NALU_HYPRE_Int)(capacity_U * EXPAND_FACT + 1);
         U_diag_j = nalu_hypre_TReAlloc_v2(U_diag_j, NALU_HYPRE_Int, tmp,
                                      NALU_HYPRE_Int, capacity_U, memory_location);
         U_diag_data = nalu_hypre_TReAlloc_v2(U_diag_data, NALU_HYPRE_Real, tmp,
                                         NALU_HYPRE_Real, capacity_U, memory_location);
      }
      nalu_hypre_TMemcpy(&(U_diag_j)[ctrU], iU, NALU_HYPRE_Int, lenu, memory_location, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TMemcpy(&(U_diag_data)[ctrU], wU, NALU_HYPRE_Real, lenu, memory_location, NALU_HYPRE_MEMORY_HOST);
      U_diag_i[ii + 1] = (ctrU += lenu);
   }

   /*---------  Begin Factorization in lower part  ----
    * here we need to get off diagonals in
    */
   for (ii = nLU; ii < n; ii++)
   {
      // get row i
      i = perm[ii];
      // get extents of row i
      k1 = A_diag_i[i];
      k2 = A_diag_i[i + 1];

      /*-------------------- unpack L & U-parts of row of A in arrays w */
      iU = iL + ii;
      wU = wL + ii;
      /*--------------------  diagonal entry */
      dd = 0.0;
      lenl  = lenu = 0;
      iw[ii] = ii;
      /*-------------------- scan & unwrap column */
      for (j = k1; j < k2; j++)
      {
         col = rperm[A_diag_j[j]];
         t = A_diag_data[j];
         if (col < ii)
         {
            iw[col] = lenl;
            iL[lenl] = col;
            wL[lenl++] = t;
         }
         else if (col > ii)
         {
            iw[col] = lenu;
            iU[lenu] = col;
            wU[lenu++] = t;
         }
         else
         {
            dd = t;
         }
      }

      /*------------------ sjcan offd*/
      k1 = A_offd_i[i];
      k2 = A_offd_i[i + 1];
      for (j = k1; j < k2; j++)
      {
         /* add offd to U part, all offd are U for this part */
         col = A_offd_j[j] + n;
         t = A_offd_data[j];
         iw[col] = lenu;
         iU[lenu] = col;
         wU[lenu++] = t;
      }

      /* eliminate row */
      /*-------------------------------------------------------------------------
       *  In order to do the elimination in the correct order we must select the
       *  smallest column index among iL[k], k = j, j+1, ..., lenl-1. For ILU(0),
       *  no new fill-ins are expect, so we can pre-sort iL and wL prior to the
       *  entering the elimination loop.
       *-----------------------------------------------------------------------*/
      //      nalu_hypre_quickSortIR(iL, wL, iw, 0, (lenl-1));
      nalu_hypre_qsort3ir(iL, wL, iw, 0, (lenl - 1));
      for (j = 0; j < lenl; j++)
      {
         jpiv = iL[j];
         /* get factor/ pivot element */
         dpiv = wL[j] * D_data[jpiv];
         /* store entry in L */
         wL[j] = dpiv;

         /* zero out element - reset pivot */
         iw[jpiv] = -1;
         /* combine current row and pivot row */
         for (k = U_diag_i[jpiv]; k < U_diag_i[jpiv + 1]; k++)
         {
            col = U_diag_j[k];
            jpos = iw[col];

            /* Only fill-in nonzero pattern (jpos != 0) */
            if (jpos < 0)
            {
               continue;
            }

            lxu = - U_diag_data[k] * dpiv;
            if (col < ii)
            {
               /* dealing with L part */
               wL[jpos] += lxu;
            }
            else if (col > ii)
            {
               /* dealing with U part */
               wU[jpos] += lxu;
            }
            else
            {
               /* diagonal update */
               dd += lxu;
            }
         }
      }
      /* restore iw (only need to restore diagonal and U part */
      iw[ii] = -1;
      for (j = 0; j < lenu; j++)
      {
         iw[iU[j]] = -1;
      }

      /* Update LDU factors */
      /* L part */
      /* Check that memory is sufficient */
      while ((ctrL + lenl) > capacity_L)
      {
         NALU_HYPRE_Int tmp = capacity_L;
         capacity_L = (NALU_HYPRE_Int)(capacity_L * EXPAND_FACT + 1);
         L_diag_j = nalu_hypre_TReAlloc_v2(L_diag_j, NALU_HYPRE_Int, tmp,
                                      NALU_HYPRE_Int, capacity_L, memory_location);
         L_diag_data = nalu_hypre_TReAlloc_v2(L_diag_data, NALU_HYPRE_Real, tmp,
                                         NALU_HYPRE_Real, capacity_L, memory_location);
      }
      nalu_hypre_TMemcpy(&(L_diag_j)[ctrL], iL, NALU_HYPRE_Int, lenl, memory_location, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TMemcpy(&(L_diag_data)[ctrL], wL, NALU_HYPRE_Real, lenl, memory_location, NALU_HYPRE_MEMORY_HOST);
      L_diag_i[ii + 1] = (ctrL += lenl);

      /* diagonal part (we store the inverse) */
      if (nalu_hypre_abs(dd) < MAT_TOL)
      {
         dd = 1.0e-6;
      }
      D_data[ii] = 1. / dd;

      /* U part */
      /* Check that memory is sufficient */
      while ((ctrU + lenu) > capacity_U)
      {
         NALU_HYPRE_Int tmp = capacity_U;
         capacity_U = (NALU_HYPRE_Int)(capacity_U * EXPAND_FACT + 1);
         U_diag_j = nalu_hypre_TReAlloc_v2(U_diag_j, NALU_HYPRE_Int, tmp,
                                      NALU_HYPRE_Int, capacity_U, memory_location);
         U_diag_data = nalu_hypre_TReAlloc_v2(U_diag_data, NALU_HYPRE_Real, tmp,
                                         NALU_HYPRE_Real, capacity_U, memory_location);
      }
      nalu_hypre_TMemcpy(&(U_diag_j)[ctrU], iU, NALU_HYPRE_Int, lenu, memory_location, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TMemcpy(&(U_diag_data)[ctrU], wU, NALU_HYPRE_Real, lenu, memory_location, NALU_HYPRE_MEMORY_HOST);
      U_diag_i[ii + 1] = (ctrU += lenu);
   }

   /*---------  Begin Factorization in external part  ----
    * here we need to get off diagonals in
    */
   for (ii = n ; ii < total_rows ; ii++)
   {
      // get row i
      i = ii - n;
      // get extents of row i
      k1 = E_i[i];
      k2 = E_i[i + 1];

      /*-------------------- unpack L & U-parts of row of A in arrays w */
      iU = iL + ii;
      wU = wL + ii;
      /*--------------------  diagonal entry */
      dd = 0.0;
      lenl  = lenu = 0;
      iw[ii] = ii;
      /*-------------------- scan & unwrap column */
      for (j = k1; j < k2; j++)
      {
         col = rperm[E_j[j]];
         t = E_data[j];
         if (col < ii)
         {
            iw[col] = lenl;
            iL[lenl] = col;
            wL[lenl++] = t;
         }
         else if (col > ii)
         {
            iw[col] = lenu;
            iU[lenu] = col;
            wU[lenu++] = t;
         }
         else
         {
            dd = t;
         }
      }

      /* eliminate row */
      /*-------------------------------------------------------------------------
       *  In order to do the elimination in the correct order we must select the
       *  smallest column index among iL[k], k = j, j+1, ..., lenl-1. For ILU(0),
       *  no new fill-ins are expect, so we can pre-sort iL and wL prior to the
       *  entering the elimination loop.
       *-----------------------------------------------------------------------*/
      //      nalu_hypre_quickSortIR(iL, wL, iw, 0, (lenl-1));
      nalu_hypre_qsort3ir(iL, wL, iw, 0, (lenl - 1));
      for (j = 0; j < lenl; j++)
      {
         jpiv = iL[j];
         /* get factor/ pivot element */
         dpiv = wL[j] * D_data[jpiv];
         /* store entry in L */
         wL[j] = dpiv;

         /* zero out element - reset pivot */
         iw[jpiv] = -1;
         /* combine current row and pivot row */
         for (k = U_diag_i[jpiv]; k < U_diag_i[jpiv + 1]; k++)
         {
            col = U_diag_j[k];
            jpos = iw[col];

            /* Only fill-in nonzero pattern (jpos != 0) */
            if (jpos < 0)
            {
               continue;
            }

            lxu = - U_diag_data[k] * dpiv;
            if (col < ii)
            {
               /* dealing with L part */
               wL[jpos] += lxu;
            }
            else if (col > ii)
            {
               /* dealing with U part */
               wU[jpos] += lxu;
            }
            else
            {
               /* diagonal update */
               dd += lxu;
            }
         }
      }
      /* restore iw (only need to restore diagonal and U part */
      iw[ii] = -1;
      for (j = 0; j < lenu; j++)
      {
         iw[iU[j]] = -1;
      }

      /* Update LDU factors */
      /* L part */
      /* Check that memory is sufficient */
      while ((ctrL + lenl) > capacity_L)
      {
         NALU_HYPRE_Int tmp = capacity_L;
         capacity_L = (NALU_HYPRE_Int)(capacity_L * EXPAND_FACT + 1);
         L_diag_j = nalu_hypre_TReAlloc_v2(L_diag_j, NALU_HYPRE_Int, tmp,
                                      NALU_HYPRE_Int, capacity_L, memory_location);
         L_diag_data = nalu_hypre_TReAlloc_v2(L_diag_data, NALU_HYPRE_Real, tmp,
                                         NALU_HYPRE_Real, capacity_L, memory_location);
      }
      nalu_hypre_TMemcpy(&(L_diag_j)[ctrL], iL, NALU_HYPRE_Int, lenl, memory_location, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TMemcpy(&(L_diag_data)[ctrL], wL, NALU_HYPRE_Real, lenl, memory_location, NALU_HYPRE_MEMORY_HOST);
      L_diag_i[ii + 1] = (ctrL += lenl);

      /* diagonal part (we store the inverse) */
      if (nalu_hypre_abs(dd) < MAT_TOL)
      {
         dd = 1.0e-6;
      }
      D_data[ii] = 1. / dd;

      /* U part */
      /* Check that memory is sufficient */
      while ((ctrU + lenu) > capacity_U)
      {
         NALU_HYPRE_Int tmp = capacity_U;
         capacity_U = (NALU_HYPRE_Int)(capacity_U * EXPAND_FACT + 1);
         U_diag_j = nalu_hypre_TReAlloc_v2(U_diag_j, NALU_HYPRE_Int, tmp,
                                      NALU_HYPRE_Int, capacity_U, memory_location);
         U_diag_data = nalu_hypre_TReAlloc_v2(U_diag_data, NALU_HYPRE_Real, tmp,
                                         NALU_HYPRE_Real, capacity_U, memory_location);
      }
      nalu_hypre_TMemcpy(&(U_diag_j)[ctrU], iU, NALU_HYPRE_Int, lenu, memory_location, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TMemcpy(&(U_diag_data)[ctrU], wU, NALU_HYPRE_Real, lenu, memory_location, NALU_HYPRE_MEMORY_HOST);
      U_diag_i[ii + 1] = (ctrU += lenu);
   }

   NALU_HYPRE_BigInt big_total_rows = (NALU_HYPRE_BigInt)total_rows;
   nalu_hypre_MPI_Allreduce(&big_total_rows, &global_num_rows, 1, NALU_HYPRE_MPI_BIG_INT, nalu_hypre_MPI_SUM, comm);

   /* need to get new column start */
   {
      NALU_HYPRE_BigInt global_start;
      nalu_hypre_MPI_Scan(&big_total_rows, &global_start, 1, NALU_HYPRE_MPI_BIG_INT, nalu_hypre_MPI_SUM, comm);
      col_starts[0] = global_start - total_rows;
      col_starts[1] = global_start;
   }

   matL = nalu_hypre_ParCSRMatrixCreate( comm,
                                    global_num_rows,
                                    global_num_rows,
                                    col_starts,
                                    col_starts,
                                    0,
                                    ctrL,
                                    0 );

   L_diag = nalu_hypre_ParCSRMatrixDiag(matL);
   nalu_hypre_CSRMatrixI(L_diag) = L_diag_i;
   if (ctrL)
   {
      nalu_hypre_CSRMatrixData(L_diag) = L_diag_data;
      nalu_hypre_CSRMatrixJ(L_diag) = L_diag_j;
   }
   else
   {
      /* we've allocated some memory, so free if not used */
      nalu_hypre_TFree(L_diag_j, memory_location);
      nalu_hypre_TFree(L_diag_data, memory_location);
   }
   /* store (global) total number of nonzeros */
   local_nnz = (NALU_HYPRE_Real) ctrL;
   nalu_hypre_MPI_Allreduce(&local_nnz, &total_nnz, 1, NALU_HYPRE_MPI_REAL, nalu_hypre_MPI_SUM, comm);
   nalu_hypre_ParCSRMatrixDNumNonzeros(matL) = total_nnz;

   matU = nalu_hypre_ParCSRMatrixCreate( comm,
                                    global_num_rows,
                                    global_num_rows,
                                    col_starts,
                                    col_starts,
                                    0,
                                    ctrU,
                                    0 );

   U_diag = nalu_hypre_ParCSRMatrixDiag(matU);
   nalu_hypre_CSRMatrixI(U_diag) = U_diag_i;
   if (ctrU)
   {
      nalu_hypre_CSRMatrixData(U_diag) = U_diag_data;
      nalu_hypre_CSRMatrixJ(U_diag) = U_diag_j;
   }
   else
   {
      /* we've allocated some memory, so free if not used */
      nalu_hypre_TFree(U_diag_j, memory_location);
      nalu_hypre_TFree(U_diag_data, memory_location);
   }
   /* store (global) total number of nonzeros */
   local_nnz = (NALU_HYPRE_Real) ctrU;
   nalu_hypre_MPI_Allreduce(&local_nnz, &total_nnz, 1, NALU_HYPRE_MPI_REAL, nalu_hypre_MPI_SUM, comm);
   nalu_hypre_ParCSRMatrixDNumNonzeros(matU) = total_nnz;
   /* free memory */
   nalu_hypre_TFree(wL, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(iw, NALU_HYPRE_MEMORY_HOST);

   /* free external data */
   if (E_i)
   {
      nalu_hypre_TFree(E_i, NALU_HYPRE_MEMORY_HOST);
   }
   if (E_j)
   {
      nalu_hypre_TFree(E_j, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(E_data, NALU_HYPRE_MEMORY_HOST);
   }

   /* set matrix pointers */
   *Lptr = matL;
   *Dptr = D_data;
   *Uptr = matU;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ILUSetupILUKRASSymbolic
 *
 * ILU(k) symbolic factorization for RAS
 *
 * n = total rows of input
 * lfil = level of fill-in, the k in ILU(k)
 * perm = permutation array indicating ordering of factorization.
 * rperm = reverse permutation array, used here to avoid duplicate memory allocation
 * iw = working array, used here to avoid duplicate memory allocation
 * nLU = size of computed LDU factorization.
 * A/L/U/E_i = the I slot of A, L, U and E
 * A/L/U/E_j = the J slot of A, L, U and E
 *
 * Will form global Schur Matrix if nLU < n
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ILUSetupILUKRASSymbolic(NALU_HYPRE_Int   n,
                              NALU_HYPRE_Int  *A_diag_i,
                              NALU_HYPRE_Int  *A_diag_j,
                              NALU_HYPRE_Int  *A_offd_i,
                              NALU_HYPRE_Int  *A_offd_j,
                              NALU_HYPRE_Int  *E_i,
                              NALU_HYPRE_Int  *E_j,
                              NALU_HYPRE_Int   ext,
                              NALU_HYPRE_Int   lfil,
                              NALU_HYPRE_Int  *perm,
                              NALU_HYPRE_Int  *rperm,
                              NALU_HYPRE_Int  *iw,
                              NALU_HYPRE_Int   nLU,
                              NALU_HYPRE_Int  *L_diag_i,
                              NALU_HYPRE_Int  *U_diag_i,
                              NALU_HYPRE_Int **L_diag_j,
                              NALU_HYPRE_Int **U_diag_j)
{
   /*
    * 1: Setup and create buffers
    * A_diag_*: tempory pointer for the diagonal matrix of A and its '*' slot
    * ii: outer loop from 0 to nLU - 1
    * i: the real col number in diag inside the outer loop
    * iw:  working array store the reverse of active col number
    * iL: working array store the active col number
    * iLev: working array store the active level of current row
    * lenl/u: current position in iw and so
    * ctrL/U/S: global position in J
    */

   NALU_HYPRE_Int      *temp_L_diag_j, *temp_U_diag_j, *u_levels;
   NALU_HYPRE_Int      *iL, *iLev;
   NALU_HYPRE_Int      ii, i, j, k, ku, lena, lenl, lenu, lenh, ilev, lev, col, icol;
   //   NALU_HYPRE_Int      m = n - nLU;
   NALU_HYPRE_Int      total_rows = ext + n;

   /* memory management */
   NALU_HYPRE_Int      ctrL;
   NALU_HYPRE_Int      ctrU;
   NALU_HYPRE_Int      capacity_L;
   NALU_HYPRE_Int      capacity_U;
   NALU_HYPRE_Int      initial_alloc = 0;
   NALU_HYPRE_Int      nnz_A;
   NALU_HYPRE_MemoryLocation memory_location;

   /* Get default memory location */
   NALU_HYPRE_GetMemoryLocation(&memory_location);

   /* set iL and iLev to right place in iw array */
   iL             = iw + total_rows;
   iLev           = iw + 2 * total_rows;

   /* setup initial memory used */
   nnz_A          = A_diag_i[n];
   if (n > 0)
   {
      initial_alloc  = (NALU_HYPRE_Int)((n + ext) + nalu_hypre_ceil((nnz_A / 2.0) * total_rows / n));
   }
   capacity_L     = initial_alloc;
   capacity_U     = initial_alloc;

   /* allocate other memory for L and U struct */
   temp_L_diag_j  = nalu_hypre_CTAlloc(NALU_HYPRE_Int, capacity_L, memory_location);
   temp_U_diag_j  = nalu_hypre_CTAlloc(NALU_HYPRE_Int, capacity_U, memory_location);

   u_levels       = nalu_hypre_CTAlloc(NALU_HYPRE_Int, capacity_U, NALU_HYPRE_MEMORY_HOST);
   ctrL = ctrU = 0;

   /* set initial value for working array */
   for (ii = 0; ii < total_rows; ii++)
   {
      iw[ii] = -1;
   }

   /*
    * 2: Start of main loop
    * those in iL are NEW col index (after permutation)
    */
   for (ii = 0; ii < nLU; ii++)
   {
      i = perm[ii];
      lenl = 0;
      lenh = 0;/* this is the current length of heap */
      lenu = ii;
      lena = A_diag_i[i + 1];
      /* put those already inside original pattern, and set their level to 0 */
      for (j = A_diag_i[i]; j < lena; j++)
      {
         /* get the neworder of that col */
         col = rperm[A_diag_j[j]];
         if (col < ii)
         {
            /*
             * this is an entry in L
             * we maintain a heap structure for L part
             */
            iL[lenh] = col;
            iLev[lenh] = 0;
            iw[col] = lenh++;
            /*now miantian a heap structure*/
            nalu_hypre_ILUMinHeapAddIIIi(iL, iLev, iw, lenh);
         }
         else if (col > ii)
         {
            /* this is an entry in U */
            iL[lenu] = col;
            iLev[lenu] = 0;
            iw[col] = lenu++;
         }
      }/* end of j loop for adding pattern in original matrix */

      /*
       * search lower part of current row and update pattern based on level
       */
      while (lenh > 0)
      {
         /*
          * k is now the new col index after permutation
          * the first element of the heap is the smallest
          */
         k = iL[0];
         ilev = iLev[0];
         /*
          * we now need to maintain the heap structure
          */
         nalu_hypre_ILUMinHeapRemoveIIIi(iL, iLev, iw, lenh);
         lenh--;
         /* copy to the end of array */
         lenl++;
         /* reset iw for that, not using anymore */
         iw[k] = -1;
         nalu_hypre_swap2i(iL, iLev, ii - lenl, lenh);
         /*
          * now the elimination on current row could start.
          * eliminate row k (new index) from current row
          */
         ku = U_diag_i[k + 1];
         for (j = U_diag_i[k]; j < ku; j++)
         {
            col = temp_U_diag_j[j];
            lev = u_levels[j] + ilev + 1;
            /* ignore large level */
            icol = iw[col];
            /* skill large level */
            if (lev > lfil)
            {
               continue;
            }
            if (icol < 0)
            {
               /* not yet in */
               if (col < ii)
               {
                  /*
                   * if we add to the left L, we need to maintian the
                   *    heap structure
                   */
                  iL[lenh] = col;
                  iLev[lenh] = lev;
                  iw[col] = lenh++;
                  /*swap it with the element right after the heap*/

                  /* maintain the heap */
                  nalu_hypre_ILUMinHeapAddIIIi(iL, iLev, iw, lenh);
               }
               else if (col > ii)
               {
                  iL[lenu] = col;
                  iLev[lenu] = lev;
                  iw[col] = lenu++;
               }
            }
            else
            {
               iLev[icol] = nalu_hypre_min(lev, iLev[icol]);
            }
         }/* end of loop j for level update */
      }/* end of while loop for iith row */

      /* now update everything, indices, levels and so */
      L_diag_i[ii + 1] = L_diag_i[ii] + lenl;
      if (lenl > 0)
      {
         /* check if memory is enough */
         while (ctrL + lenl > capacity_L)
         {
            NALU_HYPRE_Int tmp = capacity_L;
            capacity_L = (NALU_HYPRE_Int)(capacity_L * EXPAND_FACT + 1);
            temp_L_diag_j = nalu_hypre_TReAlloc_v2(temp_L_diag_j, NALU_HYPRE_Int, tmp, NALU_HYPRE_Int, capacity_L,
                                              memory_location);
         }
         /* now copy L data, reverse order */
         for (j = 0; j < lenl; j++)
         {
            temp_L_diag_j[ctrL + j] = iL[ii - j - 1];
         }
         ctrL += lenl;
      }
      k = lenu - ii;
      U_diag_i[ii + 1] = U_diag_i[ii] + k;
      if (k > 0)
      {
         /* check if memory is enough */
         while (ctrU + k > capacity_U)
         {
            NALU_HYPRE_Int tmp = capacity_U;
            capacity_U = (NALU_HYPRE_Int)(capacity_U * EXPAND_FACT + 1);
            temp_U_diag_j = nalu_hypre_TReAlloc_v2(temp_U_diag_j, NALU_HYPRE_Int, tmp, NALU_HYPRE_Int, capacity_U,
                                              memory_location);
            u_levels = nalu_hypre_TReAlloc_v2(u_levels, NALU_HYPRE_Int, tmp, NALU_HYPRE_Int, capacity_U, NALU_HYPRE_MEMORY_HOST);
         }
         //nalu_hypre_TMemcpy(temp_U_diag_j+ctrU,iL+ii,NALU_HYPRE_Int,k,memory_location,NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TMemcpy(temp_U_diag_j + ctrU, iL + ii, NALU_HYPRE_Int, k,
                       memory_location, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TMemcpy(u_levels + ctrU, iLev + ii, NALU_HYPRE_Int, k,
                       NALU_HYPRE_MEMORY_HOST, NALU_HYPRE_MEMORY_HOST);
         ctrU += k;
      }

      /* reset iw */
      for (j = ii; j < lenu; j++)
      {
         iw[iL[j]] = -1;
      }

   }/* end of main loop ii from 0 to nLU-1 */

   /*
    * Offd part
    */
   for (ii = nLU; ii < n; ii++)
   {
      i = perm[ii];
      lenl = 0;
      lenh = 0;/* this is the current length of heap */
      lenu = ii;
      lena = A_diag_i[i + 1];
      /* put those already inside original pattern, and set their level to 0 */
      for (j = A_diag_i[i]; j < lena; j++)
      {
         /* get the neworder of that col */
         col = rperm[A_diag_j[j]];
         if (col < ii)
         {
            /*
             * this is an entry in L
             * we maintain a heap structure for L part
             */
            iL[lenh] = col;
            iLev[lenh] = 0;
            iw[col] = lenh++;
            /*now miantian a heap structure*/
            nalu_hypre_ILUMinHeapAddIIIi(iL, iLev, iw, lenh);
         }
         else if (col > ii)
         {
            /* this is an entry in U */
            iL[lenu] = col;
            iLev[lenu] = 0;
            iw[col] = lenu++;
         }
      }/* end of j loop for adding pattern in original matrix */

      /* put those already inside offd pattern in, and set their level to 0 */
      lena = A_offd_i[i + 1];
      for (j = A_offd_i[i]; j < lena; j++)
      {
         /* the offd cols are in order */
         col = A_offd_j[j] + n;
         /* col for sure to be greater than ii */
         iL[lenu] = col;
         iLev[lenu] = 0;
         iw[col] = lenu++;
      }

      /*
       * search lower part of current row and update pattern based on level
       */
      while (lenh > 0)
      {
         /*
          * k is now the new col index after permutation
          * the first element of the heap is the smallest
          */
         k = iL[0];
         ilev = iLev[0];
         /*
          * we now need to maintain the heap structure
          */
         nalu_hypre_ILUMinHeapRemoveIIIi(iL, iLev, iw, lenh);
         lenh--;
         /* copy to the end of array */
         lenl++;
         /* reset iw for that, not using anymore */
         iw[k] = -1;
         nalu_hypre_swap2i(iL, iLev, ii - lenl, lenh);
         /*
          * now the elimination on current row could start.
          * eliminate row k (new index) from current row
          */
         ku = U_diag_i[k + 1];
         for (j = U_diag_i[k]; j < ku; j++)
         {
            col = temp_U_diag_j[j];
            lev = u_levels[j] + ilev + 1;
            /* ignore large level */
            icol = iw[col];
            /* skill large level */
            if (lev > lfil)
            {
               continue;
            }
            if (icol < 0)
            {
               /* not yet in */
               if (col < ii)
               {
                  /*
                   * if we add to the left L, we need to maintian the
                   *    heap structure
                   */
                  iL[lenh] = col;
                  iLev[lenh] = lev;
                  iw[col] = lenh++;
                  /*swap it with the element right after the heap*/

                  /* maintain the heap */
                  nalu_hypre_ILUMinHeapAddIIIi(iL, iLev, iw, lenh);
               }
               else if (col > ii)
               {
                  iL[lenu] = col;
                  iLev[lenu] = lev;
                  iw[col] = lenu++;
               }
            }
            else
            {
               iLev[icol] = nalu_hypre_min(lev, iLev[icol]);
            }
         }/* end of loop j for level update */
      }/* end of while loop for iith row */

      /* now update everything, indices, levels and so */
      L_diag_i[ii + 1] = L_diag_i[ii] + lenl;
      if (lenl > 0)
      {
         /* check if memory is enough */
         while (ctrL + lenl > capacity_L)
         {
            NALU_HYPRE_Int tmp = capacity_L;
            capacity_L = (NALU_HYPRE_Int)(capacity_L * EXPAND_FACT + 1);
            temp_L_diag_j = nalu_hypre_TReAlloc_v2(temp_L_diag_j, NALU_HYPRE_Int, tmp, NALU_HYPRE_Int, capacity_L,
                                              memory_location);
         }
         /* now copy L data, reverse order */
         for (j = 0; j < lenl; j++)
         {
            temp_L_diag_j[ctrL + j] = iL[ii - j - 1];
         }
         ctrL += lenl;
      }
      k = lenu - ii;
      U_diag_i[ii + 1] = U_diag_i[ii] + k;
      if (k > 0)
      {
         /* check if memory is enough */
         while (ctrU + k > capacity_U)
         {
            NALU_HYPRE_Int tmp = capacity_U;
            capacity_U = (NALU_HYPRE_Int)(capacity_U * EXPAND_FACT + 1);
            temp_U_diag_j = nalu_hypre_TReAlloc_v2(temp_U_diag_j, NALU_HYPRE_Int, tmp, NALU_HYPRE_Int, capacity_U,
                                              memory_location);
            u_levels = nalu_hypre_TReAlloc_v2(u_levels, NALU_HYPRE_Int, tmp, NALU_HYPRE_Int, capacity_U, NALU_HYPRE_MEMORY_HOST);
         }
         nalu_hypre_TMemcpy(temp_U_diag_j + ctrU, iL + ii, NALU_HYPRE_Int, k, memory_location, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TMemcpy(u_levels + ctrU, iLev + ii, NALU_HYPRE_Int, k, NALU_HYPRE_MEMORY_HOST, NALU_HYPRE_MEMORY_HOST);
         ctrU += k;
      }

      /* reset iw */
      for (j = ii; j < lenu; j++)
      {
         iw[iL[j]] = -1;
      }
   } /* end of main loop ii from nLU to n */

   /* external part matrix */
   for (ii = n; ii < total_rows; ii++)
   {
      i = ii - n;
      lenl = 0;
      lenh = 0;/* this is the current length of heap */
      lenu = ii;
      lena = E_i[i + 1];
      /* put those already inside original pattern, and set their level to 0 */
      for (j = E_i[i]; j < lena; j++)
      {
         /* get the neworder of that col */
         col = E_j[j];
         if (col < ii)
         {
            /*
             * this is an entry in L
             * we maintain a heap structure for L part
             */
            iL[lenh] = col;
            iLev[lenh] = 0;
            iw[col] = lenh++;
            /*now miantian a heap structure*/
            nalu_hypre_ILUMinHeapAddIIIi(iL, iLev, iw, lenh);
         }
         else if (col > ii)
         {
            /* this is an entry in U */
            iL[lenu] = col;
            iLev[lenu] = 0;
            iw[col] = lenu++;
         }
      }/* end of j loop for adding pattern in original matrix */

      /*
       * search lower part of current row and update pattern based on level
       */
      while (lenh > 0)
      {
         /*
          * k is now the new col index after permutation
          * the first element of the heap is the smallest
          */
         k = iL[0];
         ilev = iLev[0];
         /*
          * we now need to maintain the heap structure
          */
         nalu_hypre_ILUMinHeapRemoveIIIi(iL, iLev, iw, lenh);
         lenh--;
         /* copy to the end of array */
         lenl++;
         /* reset iw for that, not using anymore */
         iw[k] = -1;
         nalu_hypre_swap2i(iL, iLev, ii - lenl, lenh);
         /*
          * now the elimination on current row could start.
          * eliminate row k (new index) from current row
          */
         ku = U_diag_i[k + 1];
         for (j = U_diag_i[k]; j < ku; j++)
         {
            col = temp_U_diag_j[j];
            lev = u_levels[j] + ilev + 1;
            /* ignore large level */
            icol = iw[col];
            /* skill large level */
            if (lev > lfil)
            {
               continue;
            }
            if (icol < 0)
            {
               /* not yet in */
               if (col < ii)
               {
                  /*
                   * if we add to the left L, we need to maintian the
                   *    heap structure
                   */
                  iL[lenh] = col;
                  iLev[lenh] = lev;
                  iw[col] = lenh++;
                  /*swap it with the element right after the heap*/

                  /* maintain the heap */
                  nalu_hypre_ILUMinHeapAddIIIi(iL, iLev, iw, lenh);
               }
               else if (col > ii)
               {
                  iL[lenu] = col;
                  iLev[lenu] = lev;
                  iw[col] = lenu++;
               }
            }
            else
            {
               iLev[icol] = nalu_hypre_min(lev, iLev[icol]);
            }
         }/* end of loop j for level update */
      }/* end of while loop for iith row */

      /* now update everything, indices, levels and so */
      L_diag_i[ii + 1] = L_diag_i[ii] + lenl;
      if (lenl > 0)
      {
         /* check if memory is enough */
         while (ctrL + lenl > capacity_L)
         {
            NALU_HYPRE_Int tmp = capacity_L;
            capacity_L = (NALU_HYPRE_Int)(capacity_L * EXPAND_FACT + 1);
            temp_L_diag_j = nalu_hypre_TReAlloc_v2(temp_L_diag_j, NALU_HYPRE_Int, tmp, NALU_HYPRE_Int, capacity_L,
                                              memory_location);
         }
         /* now copy L data, reverse order */
         for (j = 0; j < lenl; j++)
         {
            temp_L_diag_j[ctrL + j] = iL[ii - j - 1];
         }
         ctrL += lenl;
      }
      k = lenu - ii;
      U_diag_i[ii + 1] = U_diag_i[ii] + k;
      if (k > 0)
      {
         /* check if memory is enough */
         while (ctrU + k > capacity_U)
         {
            NALU_HYPRE_Int tmp = capacity_U;
            capacity_U = (NALU_HYPRE_Int)(capacity_U * EXPAND_FACT + 1);
            temp_U_diag_j = nalu_hypre_TReAlloc_v2(temp_U_diag_j, NALU_HYPRE_Int, tmp, NALU_HYPRE_Int, capacity_U,
                                              memory_location);
            u_levels = nalu_hypre_TReAlloc_v2(u_levels, NALU_HYPRE_Int, tmp, NALU_HYPRE_Int, capacity_U, NALU_HYPRE_MEMORY_HOST);
         }
         nalu_hypre_TMemcpy(temp_U_diag_j + ctrU, iL + ii, NALU_HYPRE_Int, k, memory_location, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TMemcpy(u_levels + ctrU, iLev + ii, NALU_HYPRE_Int, k, NALU_HYPRE_MEMORY_HOST, NALU_HYPRE_MEMORY_HOST);
         ctrU += k;
      }

      /* reset iw */
      for (j = ii; j < lenu; j++)
      {
         iw[iL[j]] = -1;
      }

   }/* end of main loop ii from n to total_rows */

   /*
    * 3: Finishing up and free memory
    */
   nalu_hypre_TFree(u_levels, NALU_HYPRE_MEMORY_HOST);

   *L_diag_j = temp_L_diag_j;
   *U_diag_j = temp_U_diag_j;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ILUSetupILUKRAS
 *
 * ILU(k) numeric factorization for RAS
 *
 * A: input matrix
 * lfil: level of fill-in, the k in ILU(k)
 * perm: permutation array indicating ordering of factorization.
 *       Perm could come from a CF_marker array or a reordering routine.
 * nLU: size of computed LDU factorization.
 * Lptr, Dptr, Uptr: L, D, U factors.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ILUSetupILUKRAS(nalu_hypre_ParCSRMatrix  *A,
                      NALU_HYPRE_Int            lfil,
                      NALU_HYPRE_Int           *perm,
                      NALU_HYPRE_Int            nLU,
                      nalu_hypre_ParCSRMatrix **Lptr,
                      NALU_HYPRE_Real         **Dptr,
                      nalu_hypre_ParCSRMatrix **Uptr)
{
   /*
    * 1: Setup and create buffers
    * matL/U: the ParCSR matrix for L and U
    * L/U_diag: the diagonal csr matrix of matL/U
    * A_diag_*: tempory pointer for the diagonal matrix of A and its '*' slot
    * ii = outer loop from 0 to nLU - 1
    * i = the real col number in diag inside the outer loop
    * iw =  working array store the reverse of active col number
    * iL = working array store the active col number
    */

   /* call ILU0 if lfil is 0 */
   if (lfil == 0)
   {
      return nalu_hypre_ILUSetupILU0RAS(A, perm, nLU, Lptr, Dptr, Uptr);
   }

   NALU_HYPRE_Int               i, ii, j, k, k1, k2, kl, ku, jpiv, col, icol;
   NALU_HYPRE_Int               *iw;
   MPI_Comm                comm           = nalu_hypre_ParCSRMatrixComm(A);
   NALU_HYPRE_Int               num_procs;

   /* data objects for A */
   nalu_hypre_CSRMatrix         *A_diag        = nalu_hypre_ParCSRMatrixDiag(A);
   nalu_hypre_CSRMatrix         *A_offd        = nalu_hypre_ParCSRMatrixOffd(A);
   NALU_HYPRE_Real              *A_diag_data   = nalu_hypre_CSRMatrixData(A_diag);
   NALU_HYPRE_Int               *A_diag_i      = nalu_hypre_CSRMatrixI(A_diag);
   NALU_HYPRE_Int               *A_diag_j      = nalu_hypre_CSRMatrixJ(A_diag);
   NALU_HYPRE_Real              *A_offd_data   = nalu_hypre_CSRMatrixData(A_offd);
   NALU_HYPRE_Int               *A_offd_i      = nalu_hypre_CSRMatrixI(A_offd);
   NALU_HYPRE_Int               *A_offd_j      = nalu_hypre_CSRMatrixJ(A_offd);
   NALU_HYPRE_MemoryLocation     memory_location = nalu_hypre_ParCSRMatrixMemoryLocation(A);

   /* data objects for L, D, U */
   nalu_hypre_ParCSRMatrix      *matL;
   nalu_hypre_ParCSRMatrix      *matU;
   nalu_hypre_CSRMatrix         *L_diag;
   nalu_hypre_CSRMatrix         *U_diag;
   NALU_HYPRE_Real              *D_data;
   NALU_HYPRE_Real              *L_diag_data   = NULL;
   NALU_HYPRE_Int               *L_diag_i;
   NALU_HYPRE_Int               *L_diag_j      = NULL;
   NALU_HYPRE_Real              *U_diag_data   = NULL;
   NALU_HYPRE_Int               *U_diag_i;
   NALU_HYPRE_Int               *U_diag_j      = NULL;

   /* size of problem and external matrix */
   NALU_HYPRE_Int               n              = nalu_hypre_CSRMatrixNumRows(A_diag);
   NALU_HYPRE_Int               ext            = nalu_hypre_CSRMatrixNumCols(A_offd);
   NALU_HYPRE_Int               total_rows     = n + ext;
   NALU_HYPRE_BigInt            global_num_rows;
   NALU_HYPRE_BigInt            col_starts[2];
   NALU_HYPRE_Real              local_nnz, total_nnz;

   /* data objects for E, external matrix */
   NALU_HYPRE_Int               *E_i;
   NALU_HYPRE_Int               *E_j;
   NALU_HYPRE_Real              *E_data;

   /* communication */
   nalu_hypre_ParCSRCommPkg     *comm_pkg;
   nalu_hypre_MPI_Comm_size(comm, &num_procs);

   /* reverse permutation array */
   NALU_HYPRE_Int               *rperm;
   /* temp array for old permutation */
   NALU_HYPRE_Int               *perm_old;

   /* start setup */
   /* check input and get problem size */
   n =  nalu_hypre_CSRMatrixNumRows(A_diag);
   if (nLU < 0 || nLU > n)
   {
      nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_ARG, "WARNING: nLU out of range.\n");
   }

   /* Init I array anyway. S's might be freed later */
   D_data   = nalu_hypre_CTAlloc(NALU_HYPRE_Real, total_rows, memory_location);
   L_diag_i = nalu_hypre_CTAlloc(NALU_HYPRE_Int, (total_rows + 1), memory_location);
   U_diag_i = nalu_hypre_CTAlloc(NALU_HYPRE_Int, (total_rows + 1), memory_location);

   /* set Comm_Pkg if not yet built */
   comm_pkg = nalu_hypre_ParCSRMatrixCommPkg(A);
   if (!comm_pkg)
   {
      nalu_hypre_MatvecCommPkgCreate(A);
      comm_pkg = nalu_hypre_ParCSRMatrixCommPkg(A);
   }

   /*
    * 2: Symbolic factorization
    * setup iw and rperm first
    */
   /* allocate work arrays */
   iw          = nalu_hypre_CTAlloc(NALU_HYPRE_Int, 5 * total_rows, NALU_HYPRE_MEMORY_HOST);
   rperm       = iw + 3 * total_rows;
   perm_old    = perm;
   perm        = iw + 4 * total_rows;
   L_diag_i[0] = U_diag_i[0] = 0;
   /* get reverse permutation (rperm).
    * rperm holds the reordered indexes.
    */
   for (i = 0; i < n; i++)
   {
      perm[i] = perm_old[i];
   }
   for (i = n; i < total_rows; i++)
   {
      perm[i] = i;
   }
   for (i = 0; i < total_rows; i++)
   {
      rperm[perm[i]] = i;
   }

   /* get external rows */
   nalu_hypre_ILUBuildRASExternalMatrix(A, rperm, &E_i, &E_j, &E_data);
   /* do symbolic factorization */
   nalu_hypre_ILUSetupILUKRASSymbolic(n, A_diag_i, A_diag_j, A_offd_i, A_offd_j, E_i, E_j, ext, lfil, perm,
                                 rperm, iw,
                                 nLU, L_diag_i, U_diag_i, &L_diag_j, &U_diag_j);

   /*
    * after this, we have our I,J for L, U and S ready, and L sorted
    * iw are still -1 after symbolic factorization
    * now setup helper array here
    */
   if (L_diag_i[total_rows])
   {
      L_diag_data = nalu_hypre_CTAlloc(NALU_HYPRE_Real, L_diag_i[total_rows], memory_location);
   }
   if (U_diag_i[total_rows])
   {
      U_diag_data = nalu_hypre_CTAlloc(NALU_HYPRE_Real, U_diag_i[total_rows], memory_location);
   }

   /*
    * 3: Begin real factorization
    * we already have L and U structure ready, so no extra working array needed
    */
   /* first loop for upper part */
   for (ii = 0; ii < nLU; ii++)
   {
      // get row i
      i = perm[ii];
      kl = L_diag_i[ii + 1];
      ku = U_diag_i[ii + 1];
      k1 = A_diag_i[i];
      k2 = A_diag_i[i + 1];
      /* set up working arrays */
      for (j = L_diag_i[ii]; j < kl; j++)
      {
         col = L_diag_j[j];
         iw[col] = j;
      }
      D_data[ii] = 0.0;
      iw[ii] = ii;
      for (j = U_diag_i[ii]; j < ku; j++)
      {
         col = U_diag_j[j];
         iw[col] = j;
      }
      /* copy data from A into L, D and U */
      for (j = k1; j < k2; j++)
      {
         /* compute everything in new index */
         col = rperm[A_diag_j[j]];
         icol = iw[col];
         /* A for sure to be inside the pattern */
         if (col < ii)
         {
            L_diag_data[icol] = A_diag_data[j];
         }
         else if (col == ii)
         {
            D_data[ii] = A_diag_data[j];
         }
         else
         {
            U_diag_data[icol] = A_diag_data[j];
         }
      }
      /* elimination */
      for (j = L_diag_i[ii]; j < kl; j++)
      {
         jpiv = L_diag_j[j];
         L_diag_data[j] *= D_data[jpiv];
         ku = U_diag_i[jpiv + 1];

         for (k = U_diag_i[jpiv]; k < ku; k++)
         {
            col = U_diag_j[k];
            icol = iw[col];
            if (icol < 0)
            {
               /* not in partern */
               continue;
            }
            if (col < ii)
            {
               /* L part */
               L_diag_data[icol] -= L_diag_data[j] * U_diag_data[k];
            }
            else if (col == ii)
            {
               /* diag part */
               D_data[icol] -= L_diag_data[j] * U_diag_data[k];
            }
            else
            {
               /* U part */
               U_diag_data[icol] -= L_diag_data[j] * U_diag_data[k];
            }
         }
      }
      /* reset working array */
      ku = U_diag_i[ii + 1];
      for (j = L_diag_i[ii]; j < kl; j++)
      {
         col = L_diag_j[j];
         iw[col] = -1;
      }
      iw[ii] = -1;
      for (j = U_diag_i[ii]; j < ku; j++)
      {
         col = U_diag_j[j];
         iw[col] = -1;
      }

      /* diagonal part (we store the inverse) */
      if (nalu_hypre_abs(D_data[ii]) < MAT_TOL)
      {
         D_data[ii] = 1.0e-06;
      }
      D_data[ii] = 1. / D_data[ii];

   }/* end of loop for upper part */

   /* first loop for upper part */
   for (ii = nLU; ii < n; ii++)
   {
      // get row i
      i = perm[ii];
      kl = L_diag_i[ii + 1];
      ku = U_diag_i[ii + 1];
      /* set up working arrays */
      for (j = L_diag_i[ii]; j < kl; j++)
      {
         col = L_diag_j[j];
         iw[col] = j;
      }
      D_data[ii] = 0.0;
      iw[ii] = ii;
      for (j = U_diag_i[ii]; j < ku; j++)
      {
         col = U_diag_j[j];
         iw[col] = j;
      }
      /* copy data from A into L, D and U */
      k1 = A_diag_i[i];
      k2 = A_diag_i[i + 1];
      for (j = k1; j < k2; j++)
      {
         /* compute everything in new index */
         col = rperm[A_diag_j[j]];
         icol = iw[col];
         /* A for sure to be inside the pattern */
         if (col < ii)
         {
            L_diag_data[icol] = A_diag_data[j];
         }
         else if (col == ii)
         {
            D_data[ii] = A_diag_data[j];
         }
         else
         {
            U_diag_data[icol] = A_diag_data[j];
         }
      }
      /* copy data from A_offd into L, D and U */
      k1 = A_offd_i[i];
      k2 = A_offd_i[i + 1];
      for (j = k1; j < k2; j++)
      {
         /* compute everything in new index */
         col = A_offd_j[j] + n;
         icol = iw[col];
         U_diag_data[icol] = A_offd_data[j];
      }
      /* elimination */
      for (j = L_diag_i[ii]; j < kl; j++)
      {
         jpiv = L_diag_j[j];
         L_diag_data[j] *= D_data[jpiv];
         ku = U_diag_i[jpiv + 1];

         for (k = U_diag_i[jpiv]; k < ku; k++)
         {
            col = U_diag_j[k];
            icol = iw[col];
            if (icol < 0)
            {
               /* not in partern */
               continue;
            }
            if (col < ii)
            {
               /* L part */
               L_diag_data[icol] -= L_diag_data[j] * U_diag_data[k];
            }
            else if (col == ii)
            {
               /* diag part */
               D_data[icol] -= L_diag_data[j] * U_diag_data[k];
            }
            else
            {
               /* U part */
               U_diag_data[icol] -= L_diag_data[j] * U_diag_data[k];
            }
         }
      }
      /* reset working array */
      ku = U_diag_i[ii + 1];
      for (j = L_diag_i[ii]; j < kl; j++)
      {
         col = L_diag_j[j];
         iw[col] = -1;
      }
      iw[ii] = -1;
      for (j = U_diag_i[ii]; j < ku; j++)
      {
         col = U_diag_j[j];
         iw[col] = -1;
      }

      /* diagonal part (we store the inverse) */
      if (nalu_hypre_abs(D_data[ii]) < MAT_TOL)
      {
         D_data[ii] = 1.0e-06;
      }
      D_data[ii] = 1. / D_data[ii];

   }/* end of loop for lower part */

   /* last loop through external */
   for (ii = n; ii < total_rows; ii++)
   {
      // get row i
      i = ii - n;
      kl = L_diag_i[ii + 1];
      ku = U_diag_i[ii + 1];
      k1 = E_i[i];
      k2 = E_i[i + 1];
      /* set up working arrays */
      for (j = L_diag_i[ii]; j < kl; j++)
      {
         col = L_diag_j[j];
         iw[col] = j;
      }
      D_data[ii] = 0.0;
      iw[ii] = ii;
      for (j = U_diag_i[ii]; j < ku; j++)
      {
         col = U_diag_j[j];
         iw[col] = j;
      }
      /* copy data from E into L, D and U */
      for (j = k1; j < k2; j++)
      {
         /* compute everything in new index */
         col = E_j[j];
         icol = iw[col];
         /* A for sure to be inside the pattern */
         if (col < ii)
         {
            L_diag_data[icol] = E_data[j];
         }
         else if (col == ii)
         {
            D_data[ii] = E_data[j];
         }
         else
         {
            U_diag_data[icol] = E_data[j];
         }
      }
      /* elimination */
      for (j = L_diag_i[ii]; j < kl; j++)
      {
         jpiv = L_diag_j[j];
         L_diag_data[j] *= D_data[jpiv];
         ku = U_diag_i[jpiv + 1];

         for (k = U_diag_i[jpiv]; k < ku; k++)
         {
            col = U_diag_j[k];
            icol = iw[col];
            if (icol < 0)
            {
               /* not in partern */
               continue;
            }
            if (col < ii)
            {
               /* L part */
               L_diag_data[icol] -= L_diag_data[j] * U_diag_data[k];
            }
            else if (col == ii)
            {
               /* diag part */
               D_data[icol] -= L_diag_data[j] * U_diag_data[k];
            }
            else
            {
               /* U part */
               U_diag_data[icol] -= L_diag_data[j] * U_diag_data[k];
            }
         }
      }
      /* reset working array */
      ku = U_diag_i[ii + 1];
      for (j = L_diag_i[ii]; j < kl; j++)
      {
         col = L_diag_j[j];
         iw[col] = -1;
      }
      iw[ii] = -1;
      for (j = U_diag_i[ii]; j < ku; j++)
      {
         col = U_diag_j[j];
         iw[col] = -1;
      }

      /* diagonal part (we store the inverse) */
      if (nalu_hypre_abs(D_data[ii]) < MAT_TOL)
      {
         D_data[ii] = 1.0e-06;
      }
      D_data[ii] = 1. / D_data[ii];

   }/* end of loop for external loop */

   /*
    * 4: Finishing up and free
    */
   NALU_HYPRE_BigInt big_total_rows = (NALU_HYPRE_BigInt)total_rows;
   nalu_hypre_MPI_Allreduce(&big_total_rows, &global_num_rows, 1, NALU_HYPRE_MPI_BIG_INT,
                       nalu_hypre_MPI_SUM, comm);
   /* need to get new column start */
   {
      NALU_HYPRE_BigInt global_start;
      nalu_hypre_MPI_Scan(&big_total_rows, &global_start, 1, NALU_HYPRE_MPI_BIG_INT, nalu_hypre_MPI_SUM, comm);
      col_starts[0] = global_start - total_rows;
      col_starts[1] = global_start;
   }
   /* Assemble LDU matrices */
   matL = nalu_hypre_ParCSRMatrixCreate( comm,
                                    global_num_rows,
                                    global_num_rows,
                                    col_starts,
                                    col_starts,
                                    0 /* num_cols_offd */,
                                    L_diag_i[total_rows],
                                    0 /* num_nonzeros_offd */);

   L_diag = nalu_hypre_ParCSRMatrixDiag(matL);
   nalu_hypre_CSRMatrixI(L_diag) = L_diag_i;
   if (L_diag_i[total_rows] > 0)
   {
      nalu_hypre_CSRMatrixData(L_diag) = L_diag_data;
      nalu_hypre_CSRMatrixJ(L_diag) = L_diag_j;
   }
   else
   {
      /* we allocated some initial length, so free them */
      nalu_hypre_TFree(L_diag_j, memory_location);
   }
   /* store (global) total number of nonzeros */
   local_nnz = (NALU_HYPRE_Real) (L_diag_i[total_rows]);
   nalu_hypre_MPI_Allreduce(&local_nnz, &total_nnz, 1, NALU_HYPRE_MPI_REAL, nalu_hypre_MPI_SUM, comm);
   nalu_hypre_ParCSRMatrixDNumNonzeros(matL) = total_nnz;

   matU = nalu_hypre_ParCSRMatrixCreate( comm,
                                    global_num_rows,
                                    global_num_rows,
                                    col_starts,
                                    col_starts,
                                    0,
                                    U_diag_i[total_rows],
                                    0 );

   U_diag = nalu_hypre_ParCSRMatrixDiag(matU);
   nalu_hypre_CSRMatrixI(U_diag) = U_diag_i;
   if (U_diag_i[n] > 0)
   {
      nalu_hypre_CSRMatrixData(U_diag) = U_diag_data;
      nalu_hypre_CSRMatrixJ(U_diag) = U_diag_j;
   }
   else
   {
      /* we allocated some initial length, so free them */
      nalu_hypre_TFree(U_diag_j, memory_location);
   }
   /* store (global) total number of nonzeros */
   local_nnz = (NALU_HYPRE_Real) (U_diag_i[total_rows]);
   nalu_hypre_MPI_Allreduce(&local_nnz, &total_nnz, 1, NALU_HYPRE_MPI_REAL, nalu_hypre_MPI_SUM, comm);
   nalu_hypre_ParCSRMatrixDNumNonzeros(matU) = total_nnz;

   /* free */
   nalu_hypre_TFree(iw, NALU_HYPRE_MEMORY_HOST);

   /* free external data */
   if (E_i)
   {
      nalu_hypre_TFree(E_i, NALU_HYPRE_MEMORY_HOST);
   }
   if (E_j)
   {
      nalu_hypre_TFree(E_j, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(E_data, NALU_HYPRE_MEMORY_HOST);
   }

   /* set matrix pointers */
   *Lptr = matL;
   *Dptr = D_data;
   *Uptr = matU;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ILUSetupILUTRAS
 *
 * ILUT for RAS
 *
 * A: input matrix
 * lfil: level of fill-in, the k in ILU(k)
 * tol: droptol array in ILUT
 *    tol[0]: matrix B
 *    tol[1]: matrix E and F
 *    tol[2]: matrix S
 * perm: permutation array indicating ordering of factorization.
 *       Perm could come from a CF_marker: array or a reordering routine.
 * nLU: size of computed LDU factorization. If nLU < n, Schur compelemnt will be formed
 * Lptr, Dptr, Uptr: L, D, U factors.
 * Sptr: Schur complement
 *
 * Keep the largest lfil entries that is greater than some tol relative
 *    to the input tol and the norm of that row in both L and U
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ILUSetupILUTRAS(nalu_hypre_ParCSRMatrix  *A,
                      NALU_HYPRE_Int            lfil,
                      NALU_HYPRE_Real          *tol,
                      NALU_HYPRE_Int           *perm,
                      NALU_HYPRE_Int            nLU,
                      nalu_hypre_ParCSRMatrix **Lptr,
                      NALU_HYPRE_Real         **Dptr,
                      nalu_hypre_ParCSRMatrix **Uptr)
{
   /*
    * 1: Setup and create buffers
    * matL/U: the ParCSR matrix for L and U
    * L/U_diag: the diagonal csr matrix of matL/U
    * A_diag_*: tempory pointer for the diagonal matrix of A and its '*' slot
    * ii = outer loop from 0 to nLU - 1
    * i = the real col number in diag inside the outer loop
    * iw =  working array store the reverse of active col number
    * iL = working array store the active col number
    */
   NALU_HYPRE_Real               local_nnz, total_nnz;
   NALU_HYPRE_Int                i, ii, j, k1, k2, k12, k22, kl, ku, col, icol, lenl, lenu, lenhu, lenhlr,
                            lenhll, jpos, jrow;
   NALU_HYPRE_Real               inorm, itolb, itolef, dpiv, lxu;
   NALU_HYPRE_Int                *iw, *iL;
   NALU_HYPRE_Real               *w;

   /* memory management */
   NALU_HYPRE_Int                ctrL;
   NALU_HYPRE_Int                ctrU;
   NALU_HYPRE_Int                initial_alloc = 0;
   NALU_HYPRE_Int                capacity_L;
   NALU_HYPRE_Int                capacity_U;
   NALU_HYPRE_Int                nnz_A;

   /* communication stuffs for S */
   MPI_Comm                 comm          = nalu_hypre_ParCSRMatrixComm(A);
   NALU_HYPRE_Int                num_procs;
   nalu_hypre_ParCSRCommPkg      *comm_pkg;
   NALU_HYPRE_BigInt             col_starts[2];

   /* data objects for A */
   nalu_hypre_CSRMatrix          *A_diag       = nalu_hypre_ParCSRMatrixDiag(A);
   nalu_hypre_CSRMatrix          *A_offd       = nalu_hypre_ParCSRMatrixOffd(A);
   NALU_HYPRE_Real               *A_diag_data  = nalu_hypre_CSRMatrixData(A_diag);
   NALU_HYPRE_Int                *A_diag_i     = nalu_hypre_CSRMatrixI(A_diag);
   NALU_HYPRE_Int                *A_diag_j     = nalu_hypre_CSRMatrixJ(A_diag);
   NALU_HYPRE_Int                *A_offd_i     = nalu_hypre_CSRMatrixI(A_offd);
   NALU_HYPRE_Int                *A_offd_j     = nalu_hypre_CSRMatrixJ(A_offd);
   NALU_HYPRE_Real               *A_offd_data  = nalu_hypre_CSRMatrixData(A_offd);
   NALU_HYPRE_MemoryLocation      memory_location = nalu_hypre_ParCSRMatrixMemoryLocation(A);

   /* data objects for L, D, U */
   nalu_hypre_ParCSRMatrix       *matL;
   nalu_hypre_ParCSRMatrix       *matU;
   nalu_hypre_CSRMatrix          *L_diag;
   nalu_hypre_CSRMatrix          *U_diag;
   NALU_HYPRE_Real               *D_data;
   NALU_HYPRE_Real               *L_diag_data  = NULL;
   NALU_HYPRE_Int                *L_diag_i;
   NALU_HYPRE_Int                *L_diag_j     = NULL;
   NALU_HYPRE_Real               *U_diag_data  = NULL;
   NALU_HYPRE_Int                *U_diag_i;
   NALU_HYPRE_Int                *U_diag_j     = NULL;

   /* size of problem and external matrix */
   NALU_HYPRE_Int                n             = nalu_hypre_CSRMatrixNumRows(A_diag);
   NALU_HYPRE_Int                ext           = nalu_hypre_CSRMatrixNumCols(A_offd);
   NALU_HYPRE_Int                total_rows    = n + ext;
   NALU_HYPRE_BigInt              global_num_rows;

   /* data objects for E, external matrix */
   NALU_HYPRE_Int                *E_i;
   NALU_HYPRE_Int                *E_j;
   NALU_HYPRE_Real               *E_data;

   /* reverse permutation */
   NALU_HYPRE_Int                *rperm;
   /* old permutation */
   NALU_HYPRE_Int                *perm_old;

   /* start setup
    * check input first
    */
   n = nalu_hypre_CSRMatrixNumRows(A_diag);
   if (nLU < 0 || nLU > n)
   {
      nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_ARG, "WARNING: nLU out of range.\n");
   }

   /* start set up
    * setup communication stuffs first
    */
   nalu_hypre_MPI_Comm_size(comm, &num_procs);
   comm_pkg = nalu_hypre_ParCSRMatrixCommPkg(A);
   /* create if not yet built */
   if (!comm_pkg)
   {
      nalu_hypre_MatvecCommPkgCreate(A);
      comm_pkg = nalu_hypre_ParCSRMatrixCommPkg(A);
   }

   /* setup initial memory */
   nnz_A = A_diag_i[nLU];
   if (n > 0)
   {
      initial_alloc = (NALU_HYPRE_Int)(nLU + nalu_hypre_ceil((NALU_HYPRE_Real)(nnz_A / 2.0)));
   }
   capacity_L = initial_alloc;
   capacity_U = initial_alloc;

   D_data = nalu_hypre_CTAlloc(NALU_HYPRE_Real, total_rows, memory_location);
   L_diag_i = nalu_hypre_CTAlloc(NALU_HYPRE_Int, (total_rows + 1), memory_location);
   U_diag_i = nalu_hypre_CTAlloc(NALU_HYPRE_Int, (total_rows + 1), memory_location);

   L_diag_j = nalu_hypre_CTAlloc(NALU_HYPRE_Int, capacity_L, memory_location);
   U_diag_j = nalu_hypre_CTAlloc(NALU_HYPRE_Int, capacity_U, memory_location);
   L_diag_data = nalu_hypre_CTAlloc(NALU_HYPRE_Real, capacity_L, memory_location);
   U_diag_data = nalu_hypre_CTAlloc(NALU_HYPRE_Real, capacity_U, memory_location);

   ctrL = ctrU = 0;

   /* setting up working array */
   iw = nalu_hypre_CTAlloc(NALU_HYPRE_Int, 4 * total_rows, NALU_HYPRE_MEMORY_HOST);
   iL = iw + total_rows;
   w = nalu_hypre_CTAlloc(NALU_HYPRE_Real, total_rows, NALU_HYPRE_MEMORY_HOST);
   for (i = 0; i < total_rows; i++)
   {
      iw[i] = -1;
   }
   L_diag_i[0] = U_diag_i[0] = 0;
   /* get reverse permutation (rperm).
    * rperm holds the reordered indexes.
    * rperm[old] -> new
    * perm[new]  -> old
    */
   rperm = iw + 2 * total_rows;
   perm_old = perm;
   perm = iw + 3 * total_rows;
   for (i = 0; i < n; i++)
   {
      perm[i] = perm_old[i];
   }
   for (i = n; i < total_rows; i++)
   {
      perm[i] = i;
   }
   for (i = 0; i < total_rows; i++)
   {
      rperm[perm[i]] = i;
   }
   /* get external matrix */
   nalu_hypre_ILUBuildRASExternalMatrix(A, rperm, &E_i, &E_j, &E_data);

   /*
    * 2: Main loop of elimination
    * maintain two heaps
    * |----->*********<-----|-----*********|
    * |col heap***value heap|value in U****|
    */

   /* main outer loop for upper part */
   for (ii = 0 ; ii < nLU; ii++)
   {
      /* get real row with perm */
      i = perm[ii];
      k1 = A_diag_i[i];
      k2 = A_diag_i[i + 1];
      kl = ii - 1;
      /* reset row norm of ith row */
      inorm = .0;
      for (j = k1; j < k2; j++)
      {
         inorm += nalu_hypre_abs(A_diag_data[j]);
      }
      if (inorm == .0)
      {
         nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_ARG, "WARNING: ILUT with zero row.\n");
      }
      inorm /= (NALU_HYPRE_Real)(k2 - k1);
      /* set the scaled tol for that row */
      itolb = tol[0] * inorm;
      itolef = tol[1] * inorm;

      /* reset displacement */
      lenhll = lenhlr = lenu = 0;
      w[ii] = 0.0;
      iw[ii] = ii;
      /* copy in data from A */
      for (j = k1; j < k2; j++)
      {
         /* get now col number */
         col = rperm[A_diag_j[j]];
         if (col < ii)
         {
            /* L part of it */
            iL[lenhll] = col;
            w[lenhll] = A_diag_data[j];
            iw[col] = lenhll++;
            /* add to heap, by col number */
            nalu_hypre_ILUMinHeapAddIRIi(iL, w, iw, lenhll);
         }
         else if (col == ii)
         {
            w[ii] = A_diag_data[j];
         }
         else
         {
            lenu++;
            jpos = lenu + ii;
            iL[jpos] = col;
            w[jpos] = A_diag_data[j];
            iw[col] = jpos;
         }
      }

      /*
       * main elimination
       * need to maintain 2 heaps for L, one heap for col and one heaps for value
       * maintian an array for U, and do qsplit with quick sort after that
       * while the heap of col is greater than zero
       */
      while (lenhll > 0)
      {

         /* get the next row from top of the heap */
         jrow = iL[0];
         dpiv = w[0] * D_data[jrow];
         w[0] = dpiv;
         /* now remove it from the top of the heap */
         nalu_hypre_ILUMinHeapRemoveIRIi(iL, w, iw, lenhll);
         lenhll--;
         /*
          * reset the drop part to -1
          * we don't need this iw anymore
          */
         iw[jrow] = -1;

         /* need to keep this one, move to the end of the heap */
         /* no longer need to maintain iw */
         nalu_hypre_swap2(iL, w, lenhll, kl - lenhlr);
         lenhlr++;
         nalu_hypre_ILUMaxrHeapAddRabsI(w + kl, iL + kl, lenhlr);

         /* loop for elimination */
         ku = U_diag_i[jrow + 1];
         for (j = U_diag_i[jrow]; j < ku; j++)
         {
            col  = U_diag_j[j];
            icol = iw[col];
            lxu  = - dpiv * U_diag_data[j];

            /* we don't want to fill small number to empty place */
            if ((icol == -1) &&
                ((col <  nLU && nalu_hypre_abs(lxu) < itolb) ||
                 (col >= nLU && nalu_hypre_abs(lxu) < itolef)))
            {
               continue;
            }

            if (icol == -1)
            {
               if (col < ii)
               {
                  /* L part
                   * not already in L part
                   * put it to the end of heap
                   * might overwrite some small entries, no issue
                   */
                  iL[lenhll] = col;
                  w[lenhll] = lxu;
                  iw[col] = lenhll++;
                  /* add to heap, by col number */
                  nalu_hypre_ILUMinHeapAddIRIi(iL, w, iw, lenhll);
               }
               else if (col == ii)
               {
                  w[ii] += lxu;
               }
               else
               {
                  /*
                   * not already in U part
                   * put is to the end of heap
                   */
                  lenu++;
                  jpos = lenu + ii;
                  iL[jpos] = col;
                  w[jpos] = lxu;
                  iw[col] = jpos;
               }
            }
            else
            {
               w[icol] += lxu;
            }
         }
      }/* while loop for the elimination of current row */

      if (nalu_hypre_abs(w[ii]) < MAT_TOL)
      {
         w[ii] = 1.0e-06;
      }
      D_data[ii] = 1. / w[ii];
      iw[ii] = -1;

      /*
       * now pick up the largest lfil from L
       * L part is guarantee to be larger than itol
       */

      lenl = lenhlr < lfil ? lenhlr : lfil;
      L_diag_i[ii + 1] = L_diag_i[ii] + lenl;
      if (lenl > 0)
      {
         /* test if memory is enough */
         while (ctrL + lenl > capacity_L)
         {
            NALU_HYPRE_Int tmp = capacity_L;

            capacity_L = (NALU_HYPRE_Int)(capacity_L * EXPAND_FACT + 1);
            L_diag_j = nalu_hypre_TReAlloc_v2(L_diag_j, NALU_HYPRE_Int, tmp,
                                         NALU_HYPRE_Int, capacity_L, memory_location);
            L_diag_data = nalu_hypre_TReAlloc_v2(L_diag_data, NALU_HYPRE_Real, tmp,
                                            NALU_HYPRE_Real, capacity_L, memory_location);
         }
         ctrL += lenl;
         /* copy large data in */
         for (j = L_diag_i[ii]; j < ctrL; j++)
         {
            L_diag_j[j] = iL[kl];
            L_diag_data[j] = w[kl];
            nalu_hypre_ILUMaxrHeapRemoveRabsI(w + kl, iL + kl, lenhlr);
            lenhlr--;
         }
      }
      /*
       * now reset working array
       * L part already reset when move out of heap, only U part
       */
      ku = lenu + ii;
      for (j = ii + 1; j <= ku; j++)
      {
         iw[iL[j]] = -1;
      }

      if (lenu < lfil)
      {
         /* we simply keep all of the data, no need to sort */
         lenhu = lenu;
      }
      else
      {
         /* need to sort the first small(hopefully) part of it */
         lenhu = lfil;
         /* quick split, only sort the first small part of the array */
         nalu_hypre_ILUMaxQSplitRabsI(w, iL, ii + 1, ii + lenhu, ii + lenu);
      }

      U_diag_i[ii + 1] = U_diag_i[ii] + lenhu;
      if (lenhu > 0)
      {
         /* test if memory is enough */
         while (ctrU + lenhu > capacity_U)
         {
            NALU_HYPRE_Int tmp = capacity_U;
            capacity_U = (NALU_HYPRE_Int)(capacity_U * EXPAND_FACT + 1);
            U_diag_j = nalu_hypre_TReAlloc_v2(U_diag_j, NALU_HYPRE_Int, tmp,
                                         NALU_HYPRE_Int, capacity_U, memory_location);
            U_diag_data = nalu_hypre_TReAlloc_v2(U_diag_data, NALU_HYPRE_Real, tmp,
                                            NALU_HYPRE_Real, capacity_U, memory_location);
         }
         ctrU += lenhu;
         /* copy large data in */
         for (j = U_diag_i[ii]; j < ctrU; j++)
         {
            jpos = ii + 1 + j - U_diag_i[ii];
            U_diag_j[j] = iL[jpos];
            U_diag_data[j] = w[jpos];
         }
      }
   }/* end of ii loop from 0 to nLU-1 */

   /* second outer loop for lower part */
   for (ii = nLU; ii < n; ii++)
   {
      /* get real row with perm */
      i = perm[ii];
      k1 = A_diag_i[i];
      k2 = A_diag_i[i + 1];
      k12 = A_offd_i[i];
      k22 = A_offd_i[i + 1];
      kl = ii - 1;
      /* reset row norm of ith row */
      inorm = .0;
      for (j = k1; j < k2; j++)
      {
         inorm += nalu_hypre_abs(A_diag_data[j]);
      }
      for (j = k12; j < k22; j++)
      {
         inorm += nalu_hypre_abs(A_offd_data[j]);
      }
      if (inorm == .0)
      {
         nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_ARG, "WARNING: ILUT with zero row.\n");
      }
      inorm /= (NALU_HYPRE_Real)(k2 + k22 - k1 - k12);
      /* set the scaled tol for that row */
      itolb = tol[0] * inorm;
      itolef = tol[1] * inorm;

      /* reset displacement */
      lenhll = lenhlr = lenu = 0;
      w[ii] = 0.0;
      iw[ii] = ii;
      /* copy in data from A_diag */
      for (j = k1; j < k2; j++)
      {
         /* get now col number */
         col = rperm[A_diag_j[j]];
         if (col < ii)
         {
            /* L part of it */
            iL[lenhll] = col;
            w[lenhll] = A_diag_data[j];
            iw[col] = lenhll++;
            /* add to heap, by col number */
            nalu_hypre_ILUMinHeapAddIRIi(iL, w, iw, lenhll);
         }
         else if (col == ii)
         {
            w[ii] = A_diag_data[j];
         }
         else
         {
            lenu++;
            jpos = lenu + ii;
            iL[jpos] = col;
            w[jpos] = A_diag_data[j];
            iw[col] = jpos;
         }
      }
      /* copy in data from A_offd */
      for (j = k12; j < k22; j++)
      {
         /* get now col number */
         col = A_offd_j[j] + n;
         /* all should greater than ii in lower part */
         lenu++;
         jpos = lenu + ii;
         iL[jpos] = col;
         w[jpos] = A_offd_data[j];
         iw[col] = jpos;
      }

      /*
       * main elimination
       * need to maintain 2 heaps for L, one heap for col and one heaps for value
       * maintian an array for U, and do qsplit with quick sort after that
       * while the heap of col is greater than zero
       */
      while (lenhll > 0)
      {

         /* get the next row from top of the heap */
         jrow = iL[0];
         dpiv = w[0] * D_data[jrow];
         w[0] = dpiv;
         /* now remove it from the top of the heap */
         nalu_hypre_ILUMinHeapRemoveIRIi(iL, w, iw, lenhll);
         lenhll--;
         /*
          * reset the drop part to -1
          * we don't need this iw anymore
          */
         iw[jrow] = -1;
         /* need to keep this one, move to the end of the heap */
         /* no longer need to maintain iw */
         nalu_hypre_swap2(iL, w, lenhll, kl - lenhlr);
         lenhlr++;
         nalu_hypre_ILUMaxrHeapAddRabsI(w + kl, iL + kl, lenhlr);
         /* loop for elimination */
         ku = U_diag_i[jrow + 1];
         for (j = U_diag_i[jrow]; j < ku; j++)
         {
            col = U_diag_j[j];
            icol = iw[col];
            lxu = - dpiv * U_diag_data[j];
            /* we don't want to fill small number to empty place */
            if ((icol == -1) &&
                ((col < nLU && nalu_hypre_abs(lxu) < itolb) || (col >= nLU && nalu_hypre_abs(lxu) < itolef)))
            {
               continue;
            }
            if (icol == -1)
            {
               if (col < ii)
               {
                  /* L part
                   * not already in L part
                   * put it to the end of heap
                   * might overwrite some small entries, no issue
                   */
                  iL[lenhll] = col;
                  w[lenhll] = lxu;
                  iw[col] = lenhll++;
                  /* add to heap, by col number */
                  nalu_hypre_ILUMinHeapAddIRIi(iL, w, iw, lenhll);
               }
               else if (col == ii)
               {
                  w[ii] += lxu;
               }
               else
               {
                  /*
                   * not already in U part
                   * put is to the end of heap
                   */
                  lenu++;
                  jpos = lenu + ii;
                  iL[jpos] = col;
                  w[jpos] = lxu;
                  iw[col] = jpos;
               }
            }
            else
            {
               w[icol] += lxu;
            }
         }
      }/* while loop for the elimination of current row */

      if (nalu_hypre_abs(w[ii]) < MAT_TOL)
      {
         w[ii] = 1.0e-06;
      }
      D_data[ii] = 1. / w[ii];
      iw[ii] = -1;

      /*
       * now pick up the largest lfil from L
       * L part is guarantee to be larger than itol
       */

      lenl = lenhlr < lfil ? lenhlr : lfil;
      L_diag_i[ii + 1] = L_diag_i[ii] + lenl;
      if (lenl > 0)
      {
         /* test if memory is enough */
         while (ctrL + lenl > capacity_L)
         {
            NALU_HYPRE_Int tmp = capacity_L;
            capacity_L = (NALU_HYPRE_Int)(capacity_L * EXPAND_FACT + 1);
            L_diag_j = nalu_hypre_TReAlloc_v2(L_diag_j, NALU_HYPRE_Int, tmp,
                                         NALU_HYPRE_Int, capacity_L, memory_location);
            L_diag_data = nalu_hypre_TReAlloc_v2(L_diag_data, NALU_HYPRE_Real, tmp,
                                            NALU_HYPRE_Real, capacity_L, memory_location);
         }
         ctrL += lenl;
         /* copy large data in */
         for (j = L_diag_i[ii]; j < ctrL; j++)
         {
            L_diag_j[j] = iL[kl];
            L_diag_data[j] = w[kl];
            nalu_hypre_ILUMaxrHeapRemoveRabsI(w + kl, iL + kl, lenhlr);
            lenhlr--;
         }
      }
      /*
       * now reset working array
       * L part already reset when move out of heap, only U part
       */
      ku = lenu + ii;
      for (j = ii + 1; j <= ku; j++)
      {
         iw[iL[j]] = -1;
      }

      if (lenu < lfil)
      {
         /* we simply keep all of the data, no need to sort */
         lenhu = lenu;
      }
      else
      {
         /* need to sort the first small(hopefully) part of it */
         lenhu = lfil;
         /* quick split, only sort the first small part of the array */
         nalu_hypre_ILUMaxQSplitRabsI(w, iL, ii + 1, ii + lenhu, ii + lenu);
      }

      U_diag_i[ii + 1] = U_diag_i[ii] + lenhu;
      if (lenhu > 0)
      {
         /* test if memory is enough */
         while (ctrU + lenhu > capacity_U)
         {
            NALU_HYPRE_Int tmp = capacity_U;
            capacity_U = (NALU_HYPRE_Int)(capacity_U * EXPAND_FACT + 1);
            U_diag_j = nalu_hypre_TReAlloc_v2(U_diag_j, NALU_HYPRE_Int, tmp,
                                         NALU_HYPRE_Int, capacity_U, memory_location);
            U_diag_data = nalu_hypre_TReAlloc_v2(U_diag_data, NALU_HYPRE_Real, tmp,
                                            NALU_HYPRE_Real, capacity_U, memory_location);
         }
         ctrU += lenhu;
         /* copy large data in */
         for (j = U_diag_i[ii]; j < ctrU; j++)
         {
            jpos = ii + 1 + j - U_diag_i[ii];
            U_diag_j[j] = iL[jpos];
            U_diag_data[j] = w[jpos];
         }
      }
   }/* end of ii loop from nLU to n */


   /* main outer loop for upper part */
   for (ii = n; ii < total_rows; ii++)
   {
      /* get real row with perm */
      i = ii - n;
      k1 = E_i[i];
      k2 = E_i[i + 1];
      kl = ii - 1;
      /* reset row norm of ith row */
      inorm = .0;
      for (j = k1; j < k2; j++)
      {
         inorm += nalu_hypre_abs(E_data[j]);
      }
      if (inorm == .0)
      {
         nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_ARG, "WARNING: ILUT with zero row.\n");
      }
      inorm /= (NALU_HYPRE_Real)(k2 - k1);
      /* set the scaled tol for that row */
      itolb = tol[0] * inorm;
      itolef = tol[1] * inorm;

      /* reset displacement */
      lenhll = lenhlr = lenu = 0;
      w[ii] = 0.0;
      iw[ii] = ii;
      /* copy in data from A */
      for (j = k1; j < k2; j++)
      {
         /* get now col number */
         col = rperm[E_j[j]];
         if (col < ii)
         {
            /* L part of it */
            iL[lenhll] = col;
            w[lenhll] = E_data[j];
            iw[col] = lenhll++;
            /* add to heap, by col number */
            nalu_hypre_ILUMinHeapAddIRIi(iL, w, iw, lenhll);
         }
         else if (col == ii)
         {
            w[ii] = E_data[j];
         }
         else
         {
            lenu++;
            jpos = lenu + ii;
            iL[jpos] = col;
            w[jpos] = E_data[j];
            iw[col] = jpos;
         }
      }

      /*
       * main elimination
       * need to maintain 2 heaps for L, one heap for col and one heaps for value
       * maintian an array for U, and do qsplit with quick sort after that
       * while the heap of col is greater than zero
       */
      while (lenhll > 0)
      {

         /* get the next row from top of the heap */
         jrow = iL[0];
         dpiv = w[0] * D_data[jrow];
         w[0] = dpiv;
         /* now remove it from the top of the heap */
         nalu_hypre_ILUMinHeapRemoveIRIi(iL, w, iw, lenhll);
         lenhll--;
         /*
          * reset the drop part to -1
          * we don't need this iw anymore
          */
         iw[jrow] = -1;
         /* need to keep this one, move to the end of the heap */
         /* no longer need to maintain iw */
         nalu_hypre_swap2(iL, w, lenhll, kl - lenhlr);
         lenhlr++;
         nalu_hypre_ILUMaxrHeapAddRabsI(w + kl, iL + kl, lenhlr);
         /* loop for elimination */
         ku = U_diag_i[jrow + 1];
         for (j = U_diag_i[jrow]; j < ku; j++)
         {
            col = U_diag_j[j];
            icol = iw[col];
            lxu = - dpiv * U_diag_data[j];
            /* we don't want to fill small number to empty place */
            if ((icol == -1) &&
                ((col < nLU && nalu_hypre_abs(lxu) < itolb) || (col >= nLU && nalu_hypre_abs(lxu) < itolef)))
            {
               continue;
            }
            if (icol == -1)
            {
               if (col < ii)
               {
                  /* L part
                   * not already in L part
                   * put it to the end of heap
                   * might overwrite some small entries, no issue
                   */
                  iL[lenhll] = col;
                  w[lenhll] = lxu;
                  iw[col] = lenhll++;
                  /* add to heap, by col number */
                  nalu_hypre_ILUMinHeapAddIRIi(iL, w, iw, lenhll);
               }
               else if (col == ii)
               {
                  w[ii] += lxu;
               }
               else
               {
                  /*
                   * not already in U part
                   * put is to the end of heap
                   */
                  lenu++;
                  jpos = lenu + ii;
                  iL[jpos] = col;
                  w[jpos] = lxu;
                  iw[col] = jpos;
               }
            }
            else
            {
               w[icol] += lxu;
            }
         }
      }/* while loop for the elimination of current row */

      if (nalu_hypre_abs(w[ii]) < MAT_TOL)
      {
         w[ii] = 1.0e-06;
      }
      D_data[ii] = 1. / w[ii];
      iw[ii] = -1;

      /*
       * now pick up the largest lfil from L
       * L part is guarantee to be larger than itol
       */

      lenl = lenhlr < lfil ? lenhlr : lfil;
      L_diag_i[ii + 1] = L_diag_i[ii] + lenl;
      if (lenl > 0)
      {
         /* test if memory is enough */
         while (ctrL + lenl > capacity_L)
         {
            NALU_HYPRE_Int tmp = capacity_L;
            capacity_L = (NALU_HYPRE_Int)(capacity_L * EXPAND_FACT + 1);
            L_diag_j = nalu_hypre_TReAlloc_v2(L_diag_j, NALU_HYPRE_Int, tmp,
                                         NALU_HYPRE_Int, capacity_L, memory_location);
            L_diag_data = nalu_hypre_TReAlloc_v2(L_diag_data, NALU_HYPRE_Real, tmp,
                                            NALU_HYPRE_Real, capacity_L, memory_location);
         }
         ctrL += lenl;
         /* copy large data in */
         for (j = L_diag_i[ii]; j < ctrL; j++)
         {
            L_diag_j[j] = iL[kl];
            L_diag_data[j] = w[kl];
            nalu_hypre_ILUMaxrHeapRemoveRabsI(w + kl, iL + kl, lenhlr);
            lenhlr--;
         }
      }
      /*
       * now reset working array
       * L part already reset when move out of heap, only U part
       */
      ku = lenu + ii;
      for (j = ii + 1; j <= ku; j++)
      {
         iw[iL[j]] = -1;
      }

      if (lenu < lfil)
      {
         /* we simply keep all of the data, no need to sort */
         lenhu = lenu;
      }
      else
      {
         /* need to sort the first small(hopefully) part of it */
         lenhu = lfil;
         /* quick split, only sort the first small part of the array */
         nalu_hypre_ILUMaxQSplitRabsI(w, iL, ii + 1, ii + lenhu, ii + lenu);
      }

      U_diag_i[ii + 1] = U_diag_i[ii] + lenhu;
      if (lenhu > 0)
      {
         /* test if memory is enough */
         while (ctrU + lenhu > capacity_U)
         {
            NALU_HYPRE_Int tmp = capacity_U;
            capacity_U = (NALU_HYPRE_Int)(capacity_U * EXPAND_FACT + 1);
            U_diag_j = nalu_hypre_TReAlloc_v2(U_diag_j, NALU_HYPRE_Int, tmp,
                                         NALU_HYPRE_Int, capacity_U, memory_location);
            U_diag_data = nalu_hypre_TReAlloc_v2(U_diag_data, NALU_HYPRE_Real, tmp,
                                            NALU_HYPRE_Real, capacity_U, memory_location);
         }
         ctrU += lenhu;
         /* copy large data in */
         for (j = U_diag_i[ii]; j < ctrU; j++)
         {
            jpos = ii + 1 + j - U_diag_i[ii];
            U_diag_j[j] = iL[jpos];
            U_diag_data[j] = w[jpos];
         }
      }
   }/* end of ii loop from nLU to total_rows */

   /*
    * 3: Finishing up and free
    */
   NALU_HYPRE_BigInt big_total_rows = (NALU_HYPRE_BigInt)total_rows;
   nalu_hypre_MPI_Allreduce(&big_total_rows, &global_num_rows, 1, NALU_HYPRE_MPI_BIG_INT,
                       nalu_hypre_MPI_SUM, comm);
   /* need to get new column start */
   {
      NALU_HYPRE_BigInt global_start;
      nalu_hypre_MPI_Scan(&big_total_rows, &global_start, 1, NALU_HYPRE_MPI_BIG_INT, nalu_hypre_MPI_SUM, comm);
      col_starts[0] = global_start - total_rows;
      col_starts[1] = global_start;
   }

   /* create parcsr matrix */
   matL = nalu_hypre_ParCSRMatrixCreate( comm,
                                    global_num_rows,
                                    global_num_rows,
                                    col_starts,
                                    col_starts,
                                    0,
                                    L_diag_i[total_rows],
                                    0 );

   L_diag = nalu_hypre_ParCSRMatrixDiag(matL);
   nalu_hypre_CSRMatrixI(L_diag) = L_diag_i;
   if (L_diag_i[total_rows] > 0)
   {
      nalu_hypre_CSRMatrixData(L_diag) = L_diag_data;
      nalu_hypre_CSRMatrixJ(L_diag) = L_diag_j;
   }
   else
   {
      /* we initialized some anyway, so remove if unused */
      nalu_hypre_TFree(L_diag_j, memory_location);
      nalu_hypre_TFree(L_diag_data, memory_location);
   }
   /* store (global) total number of nonzeros */
   local_nnz = (NALU_HYPRE_Real) (L_diag_i[total_rows]);
   nalu_hypre_MPI_Allreduce(&local_nnz, &total_nnz, 1, NALU_HYPRE_MPI_REAL, nalu_hypre_MPI_SUM, comm);
   nalu_hypre_ParCSRMatrixDNumNonzeros(matL) = total_nnz;

   matU = nalu_hypre_ParCSRMatrixCreate( comm,
                                    global_num_rows,
                                    global_num_rows,
                                    col_starts,
                                    col_starts,
                                    0,
                                    U_diag_i[total_rows],
                                    0 );

   U_diag = nalu_hypre_ParCSRMatrixDiag(matU);
   nalu_hypre_CSRMatrixI(U_diag) = U_diag_i;
   if (U_diag_i[total_rows] > 0)
   {
      nalu_hypre_CSRMatrixData(U_diag) = U_diag_data;
      nalu_hypre_CSRMatrixJ(U_diag) = U_diag_j;
   }
   else
   {
      /* we initialized some anyway, so remove if unused */
      nalu_hypre_TFree(U_diag_j, memory_location);
      nalu_hypre_TFree(U_diag_data, memory_location);
   }
   /* store (global) total number of nonzeros */
   local_nnz = (NALU_HYPRE_Real) (U_diag_i[total_rows]);
   nalu_hypre_MPI_Allreduce(&local_nnz, &total_nnz, 1, NALU_HYPRE_MPI_REAL, nalu_hypre_MPI_SUM, comm);
   nalu_hypre_ParCSRMatrixDNumNonzeros(matU) = total_nnz;

   /* free working array */
   nalu_hypre_TFree(iw, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(w, NALU_HYPRE_MEMORY_HOST);

   /* free external data */
   if (E_i)
   {
      nalu_hypre_TFree(E_i, NALU_HYPRE_MEMORY_HOST);
   }
   if (E_j)
   {
      nalu_hypre_TFree(E_j, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(E_data, NALU_HYPRE_MEMORY_HOST);
   }

   /* set matrix pointers */
   *Lptr = matL;
   *Dptr = D_data;
   *Uptr = matU;

   return nalu_hypre_error_flag;
}
