/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_nalu_hypre_parcsr_ls.h"
#include "float.h"
#include "ams.h"
#include "temp_multivector.h"
#include "lobpcg.h"
#include "ame.h"
#include "_nalu_hypre_utilities.hpp"

/*--------------------------------------------------------------------------
 * nalu_hypre_AMECreate
 *
 * Allocate the AMS eigensolver structure.
 *--------------------------------------------------------------------------*/

void * nalu_hypre_AMECreate()
{
   nalu_hypre_AMEData *ame_data;

   ame_data = nalu_hypre_CTAlloc(nalu_hypre_AMEData,  1, NALU_HYPRE_MEMORY_HOST);

   /* Default parameters */

   ame_data -> block_size = 1;  /* compute 1 eigenvector */
   ame_data -> maxit = 100;     /* perform at most 100 iterations */
   ame_data -> atol = 1e-6;     /* absolute convergence tolerance */
   ame_data -> rtol = 1e-6;     /* relative convergence tolerance */
   ame_data -> print_level = 1; /* print max residual norm at each step */

   /* These will be computed during setup */

   ame_data -> eigenvectors = NULL;
   ame_data -> eigenvalues  = NULL;
   ame_data -> interpreter  = NULL;
   ame_data -> G            = NULL;
   ame_data -> A_G          = NULL;
   ame_data -> B1_G         = NULL;
   ame_data -> B2_G         = NULL;
   ame_data -> t1           = NULL;
   ame_data -> t2           = NULL;
   ame_data -> t3           = NULL;

   /* The rest of the fields are initialized using the Set functions */

   ame_data -> precond      = NULL;
   ame_data -> M            = NULL;

   return (void *) ame_data;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_AMEDestroy
 *
 * Deallocate the AMS eigensolver structure. If nalu_hypre_AMEGetEigenvectors()
 * has been called, the eigenvalue/vector data will not be destroyed.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int nalu_hypre_AMEDestroy(void *esolver)
{
   nalu_hypre_AMEData *ame_data = (nalu_hypre_AMEData *) esolver;
   nalu_hypre_AMSData *ams_data;
   mv_InterfaceInterpreter* interpreter;
   mv_MultiVectorPtr eigenvectors;

   if (!ame_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   ams_data = ame_data -> precond;
   interpreter = (mv_InterfaceInterpreter*) ame_data -> interpreter;
   eigenvectors = (mv_MultiVectorPtr) ame_data -> eigenvectors;
   if (!ams_data || !interpreter || !eigenvectors)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   if (ame_data -> G)
   {
      nalu_hypre_ParCSRMatrixDestroy(ame_data -> G);
   }
   if (ame_data -> A_G)
   {
      nalu_hypre_ParCSRMatrixDestroy(ame_data -> A_G);
   }
   if (ame_data -> B1_G)
   {
      NALU_HYPRE_BoomerAMGDestroy(ame_data -> B1_G);
   }
   if (ame_data -> B2_G)
   {
      NALU_HYPRE_ParCSRPCGDestroy(ame_data -> B2_G);
   }

   if (ame_data -> eigenvalues)
   {
      nalu_hypre_TFree(ame_data -> eigenvalues, NALU_HYPRE_MEMORY_HOST);
   }
   if (eigenvectors)
   {
      mv_MultiVectorDestroy(eigenvectors);
   }

   if (interpreter)
   {
      nalu_hypre_TFree(interpreter, NALU_HYPRE_MEMORY_HOST);
   }

   if (ams_data ->  beta_is_zero)
   {
      if (ame_data -> t1)
      {
         nalu_hypre_ParVectorDestroy(ame_data -> t1);
      }
      if (ame_data -> t2)
      {
         nalu_hypre_ParVectorDestroy(ame_data -> t2);
      }
   }

   if (ame_data)
   {
      nalu_hypre_TFree(ame_data, NALU_HYPRE_MEMORY_HOST);
   }

   /* Fields initialized using the Set functions are not destroyed */

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_AMESetAMSSolver
 *
 * Sets the AMS solver to be used as a preconditioner in the eigensolver.
 * This function should be called before nalu_hypre_AMESetup()!
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int nalu_hypre_AMESetAMSSolver(void *esolver,
                                void *ams_solver)
{
   nalu_hypre_AMEData *ame_data = (nalu_hypre_AMEData *) esolver;
   ame_data -> precond = (nalu_hypre_AMSData*) ams_solver;
   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_AMESetMassMatrix
 *
 * Sets the edge mass matrix, which appear on the rhs of the eigenproblem.
 * This function should be called before nalu_hypre_AMESetup()!
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int nalu_hypre_AMESetMassMatrix(void *esolver,
                                 nalu_hypre_ParCSRMatrix *M)
{
   nalu_hypre_AMEData *ame_data = (nalu_hypre_AMEData *) esolver;
   ame_data -> M = M;
   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_AMESetBlockSize
 *
 * Sets the block size -- the number of eigenvalues/eigenvectors to be
 * computed. This function should be called before nalu_hypre_AMESetup()!
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int nalu_hypre_AMESetBlockSize(void *esolver,
                                NALU_HYPRE_Int block_size)
{
   nalu_hypre_AMEData *ame_data = (nalu_hypre_AMEData *) esolver;
   ame_data -> block_size = block_size;
   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_AMESetMaxIter
 *
 * Set the maximum number of iterations. The default value is 100.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int nalu_hypre_AMESetMaxIter(void *esolver,
                              NALU_HYPRE_Int maxit)
{
   nalu_hypre_AMEData *ame_data = (nalu_hypre_AMEData *) esolver;
   ame_data -> maxit = maxit;
   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_AMESetTol
 *
 * Set the absolute convergence tolerance. The default value is 1e-6.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int nalu_hypre_AMESetTol(void *esolver,
                          NALU_HYPRE_Real tol)
{
   nalu_hypre_AMEData *ame_data = (nalu_hypre_AMEData *) esolver;
   ame_data -> atol = tol;
   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_AMESetRTol
 *
 * Set the relative convergence tolerance. The default value is 1e-6.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int nalu_hypre_AMESetRTol(void *esolver,
                           NALU_HYPRE_Real tol)
{
   nalu_hypre_AMEData *ame_data = (nalu_hypre_AMEData *) esolver;
   ame_data -> rtol = tol;
   return nalu_hypre_error_flag;
}
/*--------------------------------------------------------------------------
 * nalu_hypre_AMESetPrintLevel
 *
 * Control how much information is printed during the solution iterations.
 * The default values is 1.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int nalu_hypre_AMESetPrintLevel(void *esolver,
                                 NALU_HYPRE_Int print_level)
{
   nalu_hypre_AMEData *ame_data = (nalu_hypre_AMEData *) esolver;
   ame_data -> print_level = print_level;
   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_AMESetup
 *
 * Construct an eigensolver based on existing AMS solver. The number of
 * desired (minimal nonzero) eigenvectors is set by nalu_hypre_AMESetBlockSize().
 *
 * The following functions need to be called before nalu_hypre_AMSSetup():
 * - nalu_hypre_AMESetAMSSolver()
 * - nalu_hypre_AMESetMassMatrix()
 *--------------------------------------------------------------------------*/
#if defined(NALU_HYPRE_USING_CUDA) || defined(NALU_HYPRE_USING_HIP)
__global__ void
hypreGPUKernel_GtEliminateBoundary( nalu_hypre_DeviceItem    &item,
                                    NALU_HYPRE_Int      nrows,
                                    NALU_HYPRE_Int     *Gt_diag_i,
                                    NALU_HYPRE_Int     *Gt_diag_j,
                                    NALU_HYPRE_Complex *Gt_diag_data,
                                    NALU_HYPRE_Int     *Gt_offd_i,
                                    NALU_HYPRE_Int     *Gt_offd_j,
                                    NALU_HYPRE_Complex *Gt_offd_data,
                                    NALU_HYPRE_Int     *edge_bc,
                                    NALU_HYPRE_Int     *edge_bc_offd)
{
   NALU_HYPRE_Int row_i = nalu_hypre_gpu_get_grid_warp_id<1, 1>(item);

   if (row_i >= nrows)
   {
      return;
   }

   NALU_HYPRE_Int lane = nalu_hypre_gpu_get_lane_id<1>(item);
   NALU_HYPRE_Int p1 = 0, q1, p2 = 0, q2 = 0;
   bool nonempty_offd = Gt_offd_j != NULL;
   bool bdr = false;

   if (lane < 2)
   {
      p1 = read_only_load(Gt_diag_i + row_i + lane);
      if (nonempty_offd)
      {
         p2 = read_only_load(Gt_offd_i + row_i + lane);
      }
   }

   q1 = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, p1, 1);
   p1 = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, p1, 0);
   if (nonempty_offd)
   {
      q2 = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, p2, 1);
      p2 = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, p2, 0);
   }

   for (NALU_HYPRE_Int j = p1 + lane; warp_any_sync(item, NALU_HYPRE_WARP_FULL_MASK, j < q1);
        j += NALU_HYPRE_WARP_SIZE)
   {
      const nalu_hypre_int k = j < q1 && read_only_load(&edge_bc[read_only_load(&Gt_diag_j[j])]) != 0;
      if ( warp_any_sync(item, NALU_HYPRE_WARP_FULL_MASK, k) )
      {
         bdr = true;
         break;
      }
   }

   if (!bdr)
   {
      for (NALU_HYPRE_Int j = p2 + lane; warp_any_sync(item, NALU_HYPRE_WARP_FULL_MASK, j < q2);
           j += NALU_HYPRE_WARP_SIZE)
      {
         const nalu_hypre_int k = j < q2 && read_only_load(&edge_bc_offd[read_only_load(&Gt_offd_j[j])]) != 0;
         if ( warp_any_sync(item, NALU_HYPRE_WARP_FULL_MASK, k) )
         {
            bdr = true;
            break;
         }
      }
   }

   if (bdr)
   {
      for (NALU_HYPRE_Int j = p1 + lane; j < q1; j += NALU_HYPRE_WARP_SIZE)
      {
         Gt_diag_data[j] = 0.0;
      }
      for (NALU_HYPRE_Int j = p2 + lane; j < q2; j += NALU_HYPRE_WARP_SIZE)
      {
         Gt_offd_data[j] = 0.0;
      }
   }
}
#endif

NALU_HYPRE_Int nalu_hypre_AMESetup(void *esolver)
{
   NALU_HYPRE_Int ne, *edge_bc;

   nalu_hypre_AMEData *ame_data = (nalu_hypre_AMEData *) esolver;
   nalu_hypre_AMSData *ams_data = ame_data -> precond;

   if (ams_data -> beta_is_zero)
   {
      ame_data -> t1 = nalu_hypre_ParVectorInDomainOf(ams_data -> G);
      ame_data -> t2 = nalu_hypre_ParVectorInDomainOf(ams_data -> G);
   }
   else
   {
      ame_data -> t1 = ams_data -> r1;
      ame_data -> t2 = ams_data -> g1;
   }
   ame_data -> t3 = ams_data -> r0;

   NALU_HYPRE_MemoryLocation memory_location = nalu_hypre_ParCSRMatrixMemoryLocation(ams_data -> A);
#if defined(NALU_HYPRE_USING_CUDA) || defined(NALU_HYPRE_USING_HIP)
   NALU_HYPRE_ExecutionPolicy exec = nalu_hypre_GetExecPolicy1(memory_location);
#endif

   /* Eliminate boundary conditions in G = [Gii, Gib; 0, Gbb], i.e.,
      compute [Gii, 0; 0, 0] */
   {
      NALU_HYPRE_Int i, j, k, nv;
      NALU_HYPRE_Int *offd_edge_bc;

      nalu_hypre_ParCSRMatrix *Gt;

      nv = nalu_hypre_ParCSRMatrixNumCols(ams_data -> G);
      ne = nalu_hypre_ParCSRMatrixNumRows(ams_data -> G);

      edge_bc = nalu_hypre_CTAlloc(NALU_HYPRE_Int, ne, memory_location);

      /* Find boundary (eliminated) edges */
      {
         nalu_hypre_CSRMatrix *Ad = nalu_hypre_ParCSRMatrixDiag(ams_data -> A);
         NALU_HYPRE_Int *AdI = nalu_hypre_CSRMatrixI(Ad);
         NALU_HYPRE_Int *AdJ = nalu_hypre_CSRMatrixJ(Ad);
         NALU_HYPRE_Real *AdA = nalu_hypre_CSRMatrixData(Ad);
         nalu_hypre_CSRMatrix *Ao = nalu_hypre_ParCSRMatrixOffd(ams_data -> A);
         NALU_HYPRE_Int *AoI = nalu_hypre_CSRMatrixI(Ao);
         NALU_HYPRE_Real *AoA = nalu_hypre_CSRMatrixData(Ao);

         /* A row (edge) is boundary if its off-diag l1 norm is less than eps */
         NALU_HYPRE_Real eps = DBL_EPSILON * 1e+4;

#if defined(NALU_HYPRE_USING_CUDA) || defined(NALU_HYPRE_USING_HIP)
         if (exec == NALU_HYPRE_EXEC_DEVICE)
         {
            NALU_HYPRE_Real *l1norm_arr = nalu_hypre_TAlloc(NALU_HYPRE_Real, ne, memory_location);
            nalu_hypre_CSRMatrixExtractDiagonalDevice(Ad, l1norm_arr, 1);
            NALU_HYPRE_THRUST_CALL( transform,
                               l1norm_arr,
                               l1norm_arr + ne,
                               l1norm_arr,
                               thrust::negate<NALU_HYPRE_Real>() );
            nalu_hypre_CSRMatrixComputeRowSumDevice(Ad, NULL, NULL, l1norm_arr, 1, 1.0, "add");
            if (AoA)
            {
               nalu_hypre_CSRMatrixComputeRowSumDevice(Ao, NULL, NULL, l1norm_arr, 1, 1.0, "add");
            }
            NALU_HYPRE_THRUST_CALL( replace_if,
                               edge_bc,
                               edge_bc + ne,
                               l1norm_arr,
                               less_than<NALU_HYPRE_Real>(eps),
                               1 );
            nalu_hypre_TFree(l1norm_arr, memory_location);
         }
         else
#endif
         {
            NALU_HYPRE_Real l1_norm;
            for (i = 0; i < ne; i++)
            {
               l1_norm = 0.0;
               for (j = AdI[i]; j < AdI[i + 1]; j++)
                  if (AdJ[j] != i)
                  {
                     l1_norm += fabs(AdA[j]);
                  }
               if (AoI)
                  for (j = AoI[i]; j < AoI[i + 1]; j++)
                  {
                     l1_norm += fabs(AoA[j]);
                  }
               if (l1_norm < eps)
               {
                  edge_bc[i] = 1;
               }
            }
         }
      }

      nalu_hypre_ParCSRMatrixTranspose(ams_data->G, &Gt, 1);

      nalu_hypre_assert( nalu_hypre_ParCSRMatrixMemoryLocation(ams_data->G) == memory_location);

      /* Use a Matvec communication to find which of the edges
         connected to local vertices are on the boundary */
      {
         nalu_hypre_ParCSRCommHandle *comm_handle;
         nalu_hypre_ParCSRCommPkg *comm_pkg;
         NALU_HYPRE_Int num_sends, *int_buf_data;
         NALU_HYPRE_Int index, start;

         offd_edge_bc = nalu_hypre_TAlloc(NALU_HYPRE_Int, nalu_hypre_CSRMatrixNumCols(nalu_hypre_ParCSRMatrixOffd(Gt)),
                                     memory_location);

         nalu_hypre_MatvecCommPkgCreate(Gt);
         comm_pkg = nalu_hypre_ParCSRMatrixCommPkg(Gt);

         num_sends = nalu_hypre_ParCSRCommPkgNumSends(comm_pkg);
         int_buf_data = nalu_hypre_TAlloc(NALU_HYPRE_Int, nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends),
                                     memory_location );

#if defined(NALU_HYPRE_USING_CUDA) || defined(NALU_HYPRE_USING_HIP)
         if (exec == NALU_HYPRE_EXEC_DEVICE)
         {
            nalu_hypre_ParCSRCommPkgCopySendMapElmtsToDevice(comm_pkg);

            NALU_HYPRE_THRUST_CALL( gather,
                               nalu_hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg),
                               nalu_hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg) + nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg,
                                     num_sends),
                               edge_bc,
                               int_buf_data );

#if defined(NALU_HYPRE_WITH_GPU_AWARE_MPI) && THRUST_CALL_BLOCKING == 0
            /* RL: make sure int_buf_data is ready before issuing GPU-GPU MPI */
            nalu_hypre_ForceSyncComputeStream(nalu_hypre_handle());
#endif
         }
         else
#endif
         {
            index = 0;
            for (i = 0; i < num_sends; i++)
            {
               start = nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
               for (j = start; j < nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, i + 1); j++)
               {
                  k = nalu_hypre_ParCSRCommPkgSendMapElmt(comm_pkg, j);
                  int_buf_data[index++] = edge_bc[k];
               }
            }
         }

         comm_handle = nalu_hypre_ParCSRCommHandleCreate_v2(11, comm_pkg,
                                                       memory_location, int_buf_data,
                                                       memory_location, offd_edge_bc);
         nalu_hypre_ParCSRCommHandleDestroy(comm_handle);
         nalu_hypre_TFree(int_buf_data, memory_location);
      }

      /* Eliminate boundary vertex entries in G^t */
      {
         nalu_hypre_CSRMatrix *Gtd = nalu_hypre_ParCSRMatrixDiag(Gt);
         NALU_HYPRE_Int *GtdI = nalu_hypre_CSRMatrixI(Gtd);
         NALU_HYPRE_Int *GtdJ = nalu_hypre_CSRMatrixJ(Gtd);
         NALU_HYPRE_Real *GtdA = nalu_hypre_CSRMatrixData(Gtd);
         nalu_hypre_CSRMatrix *Gto = nalu_hypre_ParCSRMatrixOffd(Gt);
         NALU_HYPRE_Int *GtoI = nalu_hypre_CSRMatrixI(Gto);
         NALU_HYPRE_Int *GtoJ = nalu_hypre_CSRMatrixJ(Gto);
         NALU_HYPRE_Real *GtoA = nalu_hypre_CSRMatrixData(Gto);

         NALU_HYPRE_Int bdr;

#if defined(NALU_HYPRE_USING_CUDA) || defined(NALU_HYPRE_USING_HIP)
         if (exec == NALU_HYPRE_EXEC_DEVICE)
         {
            dim3 bDim = nalu_hypre_GetDefaultDeviceBlockDimension();
            dim3 gDim = nalu_hypre_GetDefaultDeviceGridDimension(nv, "warp", bDim);
            NALU_HYPRE_GPU_LAUNCH( hypreGPUKernel_GtEliminateBoundary, gDim, bDim,
                              nv, GtdI, GtdJ, GtdA, GtoI, GtoJ, GtoA, edge_bc, offd_edge_bc );
         }
         else
#endif
         {
            for (i = 0; i < nv; i++)
            {
               bdr = 0;
               /* A vertex is boundary if it belongs to a boundary edge */
               for (j = GtdI[i]; j < GtdI[i + 1]; j++)
                  if (edge_bc[GtdJ[j]]) { bdr = 1; break; }
               if (!bdr && GtoI)
                  for (j = GtoI[i]; j < GtoI[i + 1]; j++)
                     if (offd_edge_bc[GtoJ[j]]) { bdr = 1; break; }

               if (bdr)
               {
                  for (j = GtdI[i]; j < GtdI[i + 1]; j++)
                     /* if (!edge_bc[GtdJ[j]]) */
                  {
                     GtdA[j] = 0.0;
                  }
                  if (GtoI)
                     for (j = GtoI[i]; j < GtoI[i + 1]; j++)
                        /* if (!offd_edge_bc[GtoJ[j]]) */
                     {
                        GtoA[j] = 0.0;
                     }
               }
            }
         }
      }

      nalu_hypre_ParCSRMatrixTranspose(Gt, &ame_data -> G, 1);

      nalu_hypre_ParCSRMatrixDestroy(Gt);
      nalu_hypre_TFree(offd_edge_bc, memory_location);
   }

   /* Compute G^t M G */
   {
      if (!nalu_hypre_ParCSRMatrixCommPkg(ame_data -> G))
      {
         nalu_hypre_MatvecCommPkgCreate(ame_data -> G);
      }

      if (!nalu_hypre_ParCSRMatrixCommPkg(ame_data -> M))
      {
         nalu_hypre_MatvecCommPkgCreate(ame_data -> M);
      }

#if defined(NALU_HYPRE_USING_CUDA) || defined(NALU_HYPRE_USING_HIP)
      if (exec == NALU_HYPRE_EXEC_DEVICE)
      {
         ame_data -> A_G = nalu_hypre_ParCSRMatrixRAPKT(ame_data -> G, ame_data -> M, ame_data -> G, 1);
      }
      else
#endif
      {
         nalu_hypre_BoomerAMGBuildCoarseOperator(ame_data -> G,
                                            ame_data -> M,
                                            ame_data -> G,
                                            &ame_data -> A_G);
      }

      nalu_hypre_ParCSRMatrixFixZeroRows(ame_data -> A_G);
   }

   /* Create AMG preconditioner and PCG-AMG solver for G^tMG */
   {
      NALU_HYPRE_BoomerAMGCreate(&ame_data -> B1_G);
      NALU_HYPRE_BoomerAMGSetCoarsenType(ame_data -> B1_G, ams_data -> B_G_coarsen_type);
      NALU_HYPRE_BoomerAMGSetAggNumLevels(ame_data -> B1_G, ams_data -> B_G_agg_levels);
      NALU_HYPRE_BoomerAMGSetRelaxType(ame_data -> B1_G, ams_data -> B_G_relax_type);
      NALU_HYPRE_BoomerAMGSetNumSweeps(ame_data -> B1_G, 1);
      NALU_HYPRE_BoomerAMGSetMaxLevels(ame_data -> B1_G, 25);
      NALU_HYPRE_BoomerAMGSetTol(ame_data -> B1_G, 0.0);
      NALU_HYPRE_BoomerAMGSetMaxIter(ame_data -> B1_G, 1);
      NALU_HYPRE_BoomerAMGSetStrongThreshold(ame_data -> B1_G, ams_data -> B_G_theta);
      /* don't use exact solve on the coarsest level (matrix may be singular) */
      NALU_HYPRE_BoomerAMGSetCycleRelaxType(ame_data -> B1_G,
                                       ams_data -> B_G_relax_type,
                                       3);

      NALU_HYPRE_ParCSRPCGCreate(nalu_hypre_ParCSRMatrixComm(ame_data->A_G),
                            &ame_data -> B2_G);
      NALU_HYPRE_PCGSetPrintLevel(ame_data -> B2_G, 0);
      NALU_HYPRE_PCGSetTol(ame_data -> B2_G, 1e-12);
      NALU_HYPRE_PCGSetMaxIter(ame_data -> B2_G, 20);

      NALU_HYPRE_PCGSetPrecond(ame_data -> B2_G,
                          (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_BoomerAMGSolve,
                          (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_BoomerAMGSetup,
                          ame_data -> B1_G);

      NALU_HYPRE_ParCSRPCGSetup(ame_data -> B2_G,
                           (NALU_HYPRE_ParCSRMatrix)ame_data->A_G,
                           (NALU_HYPRE_ParVector)ame_data->t1,
                           (NALU_HYPRE_ParVector)ame_data->t2);
   }

   /* Setup LOBPCG */
   {
      NALU_HYPRE_Int seed = 75;
      mv_InterfaceInterpreter* interpreter;
      mv_MultiVectorPtr eigenvectors;

      ame_data -> interpreter = nalu_hypre_CTAlloc(mv_InterfaceInterpreter, 1, NALU_HYPRE_MEMORY_HOST);
      interpreter = (mv_InterfaceInterpreter*) ame_data -> interpreter;
      NALU_HYPRE_ParCSRSetupInterpreter(interpreter);

      ame_data -> eigenvalues = nalu_hypre_CTAlloc(NALU_HYPRE_Real,  ame_data -> block_size, NALU_HYPRE_MEMORY_HOST);

      ame_data -> eigenvectors =
         mv_MultiVectorCreateFromSampleVector(interpreter,
                                              ame_data -> block_size,
                                              ame_data -> t3);
      eigenvectors = (mv_MultiVectorPtr) ame_data -> eigenvectors;

      mv_MultiVectorSetRandom (eigenvectors, seed);

      /* Make the initial vectors discretely divergence free */
      {
         NALU_HYPRE_Int i, j;
         NALU_HYPRE_Real *data;

         mv_TempMultiVector* tmp = (mv_TempMultiVector*) mv_MultiVectorGetData(eigenvectors);
         NALU_HYPRE_ParVector *v = (NALU_HYPRE_ParVector*)(tmp -> vector);
         nalu_hypre_ParVector *vi;

         for (i = 0; i < ame_data -> block_size; i++)
         {
            vi = (nalu_hypre_ParVector*) v[i];
            data = nalu_hypre_VectorData(nalu_hypre_ParVectorLocalVector(vi));
#if defined(NALU_HYPRE_USING_CUDA) || defined(NALU_HYPRE_USING_HIP)
            if (exec == NALU_HYPRE_EXEC_DEVICE)
            {
               NALU_HYPRE_THRUST_CALL( replace_if,
                                  data,
                                  data + ne,
                                  edge_bc,
                                  thrust::identity<NALU_HYPRE_Int>(),
                                  0.0 );
            }
            else
#endif
            {
               for (j = 0; j < ne; j++)
                  if (edge_bc[j])
                  {
                     data[j] = 0.0;
                  }
            }
            nalu_hypre_AMEDiscrDivFreeComponent(esolver, vi);
         }
      }
   }

   nalu_hypre_TFree(edge_bc, memory_location);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_AMSDiscrDivFreeComponent
 *
 * Remove the component of b in the range of G, i.e., compute
 *              b = (I - G (G^t M G)^{-1} G^t M) b
 * This way b will be orthogonal to gradients of linear functions.
 * The problem with G^t M G is solved only approximately by PCG-AMG.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int nalu_hypre_AMEDiscrDivFreeComponent(void *esolver, nalu_hypre_ParVector *b)
{
   nalu_hypre_AMEData *ame_data = (nalu_hypre_AMEData *) esolver;

   /* t3 = M b */
   nalu_hypre_ParCSRMatrixMatvec(1.0, ame_data -> M, b, 0.0, ame_data -> t3);

   /* t1 = G^t t3 */
   nalu_hypre_ParCSRMatrixMatvecT(1.0, ame_data -> G, ame_data -> t3, 0.0, ame_data -> t1);

   /* (G^t M G) t2 = t1 */
   nalu_hypre_ParVectorSetConstantValues(ame_data -> t2, 0.0);
   NALU_HYPRE_ParCSRPCGSolve(ame_data -> B2_G,
                        (NALU_HYPRE_ParCSRMatrix)ame_data -> A_G,
                        (NALU_HYPRE_ParVector)ame_data -> t1,
                        (NALU_HYPRE_ParVector)ame_data -> t2);

   /* b = b - G t2 */
   nalu_hypre_ParCSRMatrixMatvec(-1.0, ame_data -> G, ame_data -> t2, 1.0, b);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_AMEOperatorA and nalu_hypre_AMEMultiOperatorA
 *
 * The stiffness matrix considered as an operator on (multi)vectors.
 *--------------------------------------------------------------------------*/

void nalu_hypre_AMEOperatorA(void *data, void* x, void* y)
{
   nalu_hypre_AMEData *ame_data = (nalu_hypre_AMEData *) data;
   nalu_hypre_AMSData *ams_data = ame_data -> precond;
   nalu_hypre_ParCSRMatrixMatvec(1.0, ams_data -> A, (nalu_hypre_ParVector*)x,
                            0.0, (nalu_hypre_ParVector*)y);
}

void nalu_hypre_AMEMultiOperatorA(void *data, void* x, void* y)
{
   nalu_hypre_AMEData *ame_data = (nalu_hypre_AMEData *) data;
   mv_InterfaceInterpreter*
   interpreter = (mv_InterfaceInterpreter*) ame_data -> interpreter;
   interpreter -> Eval(nalu_hypre_AMEOperatorA, data, x, y);
}

/*--------------------------------------------------------------------------
 * nalu_hypre_AMEOperatorM and nalu_hypre_AMEMultiOperatorM
 *
 * The mass matrix considered as an operator on (multi)vectors.
 *--------------------------------------------------------------------------*/

void nalu_hypre_AMEOperatorM(void *data, void* x, void* y)
{
   nalu_hypre_AMEData *ame_data = (nalu_hypre_AMEData *) data;
   nalu_hypre_ParCSRMatrixMatvec(1.0, ame_data -> M, (nalu_hypre_ParVector*)x,
                            0.0, (nalu_hypre_ParVector*)y);
}

void nalu_hypre_AMEMultiOperatorM(void *data, void* x, void* y)
{
   nalu_hypre_AMEData *ame_data = (nalu_hypre_AMEData *) data;
   mv_InterfaceInterpreter*
   interpreter = (mv_InterfaceInterpreter*) ame_data -> interpreter;
   interpreter -> Eval(nalu_hypre_AMEOperatorM, data, x, y);
}

/*--------------------------------------------------------------------------
 * nalu_hypre_AMEOperatorB and nalu_hypre_AMEMultiOperatorB
 *
 * The AMS method considered as an operator on (multi)vectors.
 * Make sure that the result is discr. div. free.
 *--------------------------------------------------------------------------*/

void nalu_hypre_AMEOperatorB(void *data, void* x, void* y)
{
   nalu_hypre_AMEData *ame_data = (nalu_hypre_AMEData *) data;
   nalu_hypre_AMSData *ams_data = ame_data -> precond;

   nalu_hypre_ParVectorSetConstantValues((nalu_hypre_ParVector*)y, 0.0);
   nalu_hypre_AMSSolve(ame_data -> precond, ams_data -> A, (nalu_hypre_ParVector*) x, (nalu_hypre_ParVector*) y);

   nalu_hypre_AMEDiscrDivFreeComponent(data, (nalu_hypre_ParVector *)y);
}

void nalu_hypre_AMEMultiOperatorB(void *data, void* x, void* y)
{
   nalu_hypre_AMEData *ame_data = (nalu_hypre_AMEData *) data;
   mv_InterfaceInterpreter*
   interpreter = (mv_InterfaceInterpreter*) ame_data -> interpreter;
   interpreter -> Eval(nalu_hypre_AMEOperatorB, data, x, y);
}

/*--------------------------------------------------------------------------
 * nalu_hypre_AMESolve
 *
 * Solve the eigensystem A u = lambda M u, G^t u = 0 using a subspace
 * version of LOBPCG (i.e. we iterate in the discr. div. free space).
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int nalu_hypre_AMESolve(void *esolver)
{
   nalu_hypre_AMEData *ame_data = (nalu_hypre_AMEData *) esolver;

   NALU_HYPRE_Int nit;
   lobpcg_BLASLAPACKFunctions blap_fn;
   lobpcg_Tolerance lobpcg_tol;
   NALU_HYPRE_Real *residuals;

   blap_fn.dsygv  = nalu_hypre_dsygv;
   blap_fn.dpotrf = nalu_hypre_dpotrf;
   lobpcg_tol.relative = ame_data -> rtol;
   lobpcg_tol.absolute = ame_data -> atol;
   residuals = nalu_hypre_TAlloc(NALU_HYPRE_Real,  ame_data -> block_size, NALU_HYPRE_MEMORY_HOST);

   lobpcg_solve((mv_MultiVectorPtr) ame_data -> eigenvectors,
                esolver, nalu_hypre_AMEMultiOperatorA,
                esolver, nalu_hypre_AMEMultiOperatorM,
                esolver, nalu_hypre_AMEMultiOperatorB,
                NULL, blap_fn, lobpcg_tol, ame_data -> maxit,
                ame_data -> print_level, &nit,
                ame_data -> eigenvalues,
                NULL, ame_data -> block_size,
                residuals,
                NULL, ame_data -> block_size);

   nalu_hypre_TFree(residuals, NALU_HYPRE_MEMORY_HOST);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_AMEGetEigenvectors
 *
 * Return a pointer to the computed eigenvectors.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int nalu_hypre_AMEGetEigenvectors(void *esolver,
                                   NALU_HYPRE_ParVector **eigenvectors_ptr)
{
   nalu_hypre_AMEData *ame_data = (nalu_hypre_AMEData *) esolver;
   mv_MultiVectorPtr
   eigenvectors = (mv_MultiVectorPtr) ame_data -> eigenvectors;
   mv_TempMultiVector* tmp = (mv_TempMultiVector*) mv_MultiVectorGetData(eigenvectors);

   *eigenvectors_ptr = (NALU_HYPRE_ParVector*)(tmp -> vector);
   tmp -> vector = NULL;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_AMEGetEigenvalues
 *
 * Return a pointer to the computed eigenvalues.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int nalu_hypre_AMEGetEigenvalues(void *esolver,
                                  NALU_HYPRE_Real **eigenvalues_ptr)
{
   nalu_hypre_AMEData *ame_data = (nalu_hypre_AMEData *) esolver;
   *eigenvalues_ptr = ame_data -> eigenvalues;
   ame_data -> eigenvalues = NULL;
   return nalu_hypre_error_flag;
}
