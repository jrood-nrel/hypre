/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * a few more relaxation schemes: Chebychev, FCF-Jacobi, CG  -
 * these do not go through the CF interface (nalu_hypre_BoomerAMGRelaxIF)
 *
 *****************************************************************************/

#include "_nalu_hypre_parcsr_ls.h"

#if defined(NALU_HYPRE_USING_CUDA) || defined(NALU_HYPRE_USING_HIP)
#include "_nalu_hypre_utilities.hpp"

/**
 * @brief Calculates row sums and other metrics of a matrix on the device
 * to be used for the MaxEigEstimate
 */
__global__ void
hypreGPUKernel_CSRMaxEigEstimate(nalu_hypre_DeviceItem    &item,
                                 NALU_HYPRE_Int      nrows,
                                 NALU_HYPRE_Int     *diag_ia,
                                 NALU_HYPRE_Int     *diag_ja,
                                 NALU_HYPRE_Complex *diag_aa,
                                 NALU_HYPRE_Int     *offd_ia,
                                 NALU_HYPRE_Int     *offd_ja,
                                 NALU_HYPRE_Complex *offd_aa,
                                 NALU_HYPRE_Complex *row_sum_lower,
                                 NALU_HYPRE_Complex *row_sum_upper,
                                 NALU_HYPRE_Int      scale)
{
   NALU_HYPRE_Int row_i = nalu_hypre_gpu_get_grid_warp_id<1, 1>(item);

   if (row_i >= nrows)
   {
      return;
   }

   NALU_HYPRE_Int lane = nalu_hypre_gpu_get_lane_id<1>(item);
   NALU_HYPRE_Int p = 0, q;

   NALU_HYPRE_Complex diag_value = 0.0;
   NALU_HYPRE_Complex row_sum_i  = 0.0;
   NALU_HYPRE_Complex lower, upper;

   if (lane < 2)
   {
      p = read_only_load(diag_ia + row_i + lane);
   }
   q = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, p, 1);
   p = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, p, 0);

   for (NALU_HYPRE_Int j = p + lane; j < q; j += NALU_HYPRE_WARP_SIZE)
   {
      NALU_HYPRE_Complex aij = read_only_load(&diag_aa[j]);
      if ( read_only_load(&diag_ja[j]) == row_i )
      {
         diag_value = aij;
      }
      else
      {
         row_sum_i += fabs(aij);
      }
   }

   if (lane < 2)
   {
      p = read_only_load(offd_ia + row_i + lane);
   }
   q = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, p, 1);
   p = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, p, 0);

   for (NALU_HYPRE_Int j = p + lane; j < q; j += NALU_HYPRE_WARP_SIZE)
   {
      NALU_HYPRE_Complex aij = read_only_load(&offd_aa[j]);
      row_sum_i += fabs(aij);
   }

   // Get the row_sum and diagonal value on lane 0
   row_sum_i = warp_reduce_sum(item, row_sum_i);

   diag_value = warp_reduce_sum(item, diag_value);

   if (lane == 0)
   {
      lower = diag_value - row_sum_i;
      upper = diag_value + row_sum_i;

      if (scale)
      {
         lower /= nalu_hypre_abs(diag_value);
         upper /= nalu_hypre_abs(diag_value);
      }

      row_sum_upper[row_i] = upper;
      row_sum_lower[row_i] = lower;
   }
}

/**
 * @brief Estimates the max eigenvalue using infinity norm on the device
 *
 * @param[in] A Matrix to relax with
 * @param[in] to scale by diagonal
 * @param[out] Maximum eigenvalue
 */
NALU_HYPRE_Int
nalu_hypre_ParCSRMaxEigEstimateDevice( nalu_hypre_ParCSRMatrix *A,
                                  NALU_HYPRE_Int           scale,
                                  NALU_HYPRE_Real         *max_eig,
                                  NALU_HYPRE_Real         *min_eig )
{
   NALU_HYPRE_Real e_max;
   NALU_HYPRE_Real e_min;
   NALU_HYPRE_Int  A_num_rows;


   NALU_HYPRE_Real *A_diag_data;
   NALU_HYPRE_Real *A_offd_data;
   NALU_HYPRE_Int  *A_diag_i;
   NALU_HYPRE_Int  *A_offd_i;
   NALU_HYPRE_Int  *A_diag_j;
   NALU_HYPRE_Int  *A_offd_j;


   A_num_rows = nalu_hypre_CSRMatrixNumRows(nalu_hypre_ParCSRMatrixDiag(A));

   NALU_HYPRE_Real *rowsums_lower = nalu_hypre_TAlloc(NALU_HYPRE_Real, A_num_rows,
                                            nalu_hypre_ParCSRMatrixMemoryLocation(A));
   NALU_HYPRE_Real *rowsums_upper = nalu_hypre_TAlloc(NALU_HYPRE_Real, A_num_rows,
                                            nalu_hypre_ParCSRMatrixMemoryLocation(A));

   A_diag_i    = nalu_hypre_CSRMatrixI(nalu_hypre_ParCSRMatrixDiag(A));
   A_diag_j    = nalu_hypre_CSRMatrixJ(nalu_hypre_ParCSRMatrixDiag(A));
   A_diag_data = nalu_hypre_CSRMatrixData(nalu_hypre_ParCSRMatrixDiag(A));
   A_offd_i    = nalu_hypre_CSRMatrixI(nalu_hypre_ParCSRMatrixOffd(A));
   A_offd_j    = nalu_hypre_CSRMatrixJ(nalu_hypre_ParCSRMatrixOffd(A));
   A_offd_data = nalu_hypre_CSRMatrixData(nalu_hypre_ParCSRMatrixOffd(A));

   dim3 bDim, gDim;

   bDim = nalu_hypre_GetDefaultDeviceBlockDimension();
   gDim = nalu_hypre_GetDefaultDeviceGridDimension(A_num_rows, "warp", bDim);
   NALU_HYPRE_GPU_LAUNCH(hypreGPUKernel_CSRMaxEigEstimate,
                    gDim,
                    bDim,
                    A_num_rows,
                    A_diag_i,
                    A_diag_j,
                    A_diag_data,
                    A_offd_i,
                    A_offd_j,
                    A_offd_data,
                    rowsums_lower,
                    rowsums_upper,
                    scale);

   nalu_hypre_SyncComputeStream(nalu_hypre_handle());

   e_min = NALU_HYPRE_THRUST_CALL(reduce, rowsums_lower, rowsums_lower + A_num_rows, (NALU_HYPRE_Real)0,
                             thrust::minimum<NALU_HYPRE_Real>());
   e_max = NALU_HYPRE_THRUST_CALL(reduce, rowsums_upper, rowsums_upper + A_num_rows, (NALU_HYPRE_Real)0,
                             thrust::maximum<NALU_HYPRE_Real>());

   /* Same as nalu_hypre_ParCSRMaxEigEstimateHost */

   NALU_HYPRE_Real send_buf[2];
   NALU_HYPRE_Real recv_buf[2];

   send_buf[0] = -e_min;
   send_buf[1] = e_max;

   nalu_hypre_MPI_Allreduce(send_buf, recv_buf, 2, NALU_HYPRE_MPI_REAL, nalu_hypre_MPI_MAX,
                       nalu_hypre_ParCSRMatrixComm(A));

   /* return */
   if ( nalu_hypre_abs(e_min) > nalu_hypre_abs(e_max) )
   {
      *min_eig = e_min;
      *max_eig = nalu_hypre_min(0.0, e_max);
   }
   else
   {
      *min_eig = nalu_hypre_max(e_min, 0.0);
      *max_eig = e_max;
   }

   nalu_hypre_TFree(rowsums_lower, nalu_hypre_ParCSRMatrixMemoryLocation(A));
   nalu_hypre_TFree(rowsums_upper, nalu_hypre_ParCSRMatrixMemoryLocation(A));

   return nalu_hypre_error_flag;
}

/**
 *  @brief Uses CG to get the eigenvalue estimate on the device
 *
 *  @param[in] A Matrix to relax with
 *  @param[in] scale Gets the eigenvalue est of D^{-1/2} A D^{-1/2}
 *  @param[in] max_iter Maximum number of CG iterations
 *  @param[out] max_eig Estimated max eigenvalue
 *  @param[out] min_eig Estimated min eigenvalue
 */
NALU_HYPRE_Int
nalu_hypre_ParCSRMaxEigEstimateCGDevice(nalu_hypre_ParCSRMatrix *A,     /* matrix to relax with */
                                   NALU_HYPRE_Int           scale, /* scale by diagonal?*/
                                   NALU_HYPRE_Int           max_iter,
                                   NALU_HYPRE_Real         *max_eig,
                                   NALU_HYPRE_Real         *min_eig)
{
#if defined(NALU_HYPRE_USING_CUDA)
   nalu_hypre_GpuProfilingPushRange("ParCSRMaxEigEstimate_Setup");
#endif
   NALU_HYPRE_Int        i, err;
   nalu_hypre_ParVector *p;
   nalu_hypre_ParVector *s;
   nalu_hypre_ParVector *r;
   nalu_hypre_ParVector *ds;
   nalu_hypre_ParVector *u;

   NALU_HYPRE_Real *tridiag = NULL;
   NALU_HYPRE_Real *trioffd = NULL;

   NALU_HYPRE_Real  lambda_max;
   NALU_HYPRE_Real  beta, gamma = 0.0, alpha, sdotp, gamma_old, alphainv;
   NALU_HYPRE_Real  lambda_min;
   NALU_HYPRE_Real *s_data, *p_data, *ds_data, *u_data, *r_data;
   NALU_HYPRE_Int   local_size = nalu_hypre_CSRMatrixNumRows(nalu_hypre_ParCSRMatrixDiag(A));

   /* check the size of A - don't iterate more than the size */
   NALU_HYPRE_BigInt size = nalu_hypre_ParCSRMatrixGlobalNumRows(A);

   if (size < (NALU_HYPRE_BigInt)max_iter)
   {
      max_iter = (NALU_HYPRE_Int)size;
   }

#if defined(NALU_HYPRE_USING_CUDA)
   nalu_hypre_GpuProfilingPushRange("ParCSRMaxEigEstimate_Setup_DataAlloc");
#endif
   /* create some temp vectors: p, s, r , ds, u*/
   r = nalu_hypre_ParVectorCreate(nalu_hypre_ParCSRMatrixComm(A),
                             nalu_hypre_ParCSRMatrixGlobalNumRows(A),
                             nalu_hypre_ParCSRMatrixRowStarts(A));
   nalu_hypre_ParVectorInitialize_v2(r, nalu_hypre_ParCSRMatrixMemoryLocation(A));

   p = nalu_hypre_ParVectorCreate(nalu_hypre_ParCSRMatrixComm(A),
                             nalu_hypre_ParCSRMatrixGlobalNumRows(A),
                             nalu_hypre_ParCSRMatrixRowStarts(A));
   nalu_hypre_ParVectorInitialize_v2(p, nalu_hypre_ParCSRMatrixMemoryLocation(A));

   s = nalu_hypre_ParVectorCreate(nalu_hypre_ParCSRMatrixComm(A),
                             nalu_hypre_ParCSRMatrixGlobalNumRows(A),
                             nalu_hypre_ParCSRMatrixRowStarts(A));
   nalu_hypre_ParVectorInitialize_v2(s, nalu_hypre_ParCSRMatrixMemoryLocation(A));

   /* DS Starts on host to be populated, then transferred to device */
   ds = nalu_hypre_ParVectorCreate(nalu_hypre_ParCSRMatrixComm(A),
                              nalu_hypre_ParCSRMatrixGlobalNumRows(A),
                              nalu_hypre_ParCSRMatrixRowStarts(A));
   nalu_hypre_ParVectorInitialize_v2(ds, nalu_hypre_ParCSRMatrixMemoryLocation(A));
   ds_data = nalu_hypre_VectorData(nalu_hypre_ParVectorLocalVector(ds));

   u = nalu_hypre_ParVectorCreate(nalu_hypre_ParCSRMatrixComm(A),
                             nalu_hypre_ParCSRMatrixGlobalNumRows(A),
                             nalu_hypre_ParCSRMatrixRowStarts(A));
   nalu_hypre_ParVectorInitialize_v2(u, nalu_hypre_ParCSRMatrixMemoryLocation(A));

   /* point to local data */
   s_data = nalu_hypre_VectorData(nalu_hypre_ParVectorLocalVector(s));
   p_data = nalu_hypre_VectorData(nalu_hypre_ParVectorLocalVector(p));
   u_data = nalu_hypre_VectorData(nalu_hypre_ParVectorLocalVector(u));
   r_data = nalu_hypre_VectorData(nalu_hypre_ParVectorLocalVector(r));

#if defined(NALU_HYPRE_USING_CUDA)
   nalu_hypre_GpuProfilingPopRange(); /*Setup Data Alloc*/
#endif

#if defined(NALU_HYPRE_USING_CUDA)
   nalu_hypre_GpuProfilingPushRange("ParCSRMaxEigEstimate_Setup_CPUAlloc_Setup");
#endif

#if defined(NALU_HYPRE_USING_CUDA)
   nalu_hypre_GpuProfilingPushRange("ParCSRMaxEigEstimate_Setup_CPUAlloc_Setup_Alloc");
#endif

   /* make room for tri-diag matrix */
   tridiag = nalu_hypre_CTAlloc(NALU_HYPRE_Real, max_iter + 1, NALU_HYPRE_MEMORY_HOST);
   trioffd = nalu_hypre_CTAlloc(NALU_HYPRE_Real, max_iter + 1, NALU_HYPRE_MEMORY_HOST);
#if defined(NALU_HYPRE_USING_CUDA)
   nalu_hypre_GpuProfilingPopRange(); /*SETUP_Alloc*/
#endif

#if defined(NALU_HYPRE_USING_CUDA)
   nalu_hypre_GpuProfilingPushRange("ParCSRMaxEigEstimate_Setup_CPUAlloc_Zeroing");
#endif
   for (i = 0; i < max_iter + 1; i++)
   {
      tridiag[i] = 0;
      trioffd[i] = 0;
   }
#if defined(NALU_HYPRE_USING_CUDA)
   nalu_hypre_GpuProfilingPopRange(); /*Zeroing */
#endif

#if defined(NALU_HYPRE_USING_CUDA)
   nalu_hypre_GpuProfilingPushRange("ParCSRMaxEigEstimate_Setup_CPUAlloc_Random");
#endif

   /* set residual to random */
   nalu_hypre_CurandUniform(local_size, r_data, 0, 0, 0, 0);

   nalu_hypre_SyncComputeStream(nalu_hypre_handle());

   NALU_HYPRE_THRUST_CALL(transform,
                     r_data, r_data + local_size, r_data,
                     2.0 * _1 - 1.0);

#if defined(NALU_HYPRE_USING_CUDA)
   nalu_hypre_GpuProfilingPopRange(); /*CPUAlloc_Random*/
#endif

#if defined(NALU_HYPRE_USING_CUDA)
   nalu_hypre_GpuProfilingPushRange("ParCSRMaxEigEstimate_Setup_CPUAlloc_Diag");
#endif

   if (scale)
   {
      nalu_hypre_CSRMatrixExtractDiagonal(nalu_hypre_ParCSRMatrixDiag(A), ds_data, 4);
   }
   else
   {
      /* set ds to 1 */
      nalu_hypre_ParVectorSetConstantValues(ds, 1.0);
   }

#if defined(NALU_HYPRE_USING_CUDA)
   nalu_hypre_GpuProfilingPopRange(); /*Setup_CPUAlloc__Diag */
#endif
#if defined(NALU_HYPRE_USING_CUDA) /*CPUAlloc_Setup */
   nalu_hypre_GpuProfilingPopRange();
#endif

#if defined(NALU_HYPRE_USING_CUDA)
   nalu_hypre_GpuProfilingPopRange(); /* Setup */
#endif
#if defined(NALU_HYPRE_USING_CUDA)
   nalu_hypre_GpuProfilingPushRange("ParCSRMaxEigEstimate_Iter");
#endif

   /* gamma = <r,Cr> */
   gamma = nalu_hypre_ParVectorInnerProd(r, p);

   /* for the initial filling of the tridiag matrix */
   beta = 1.0;

   i = 0;
   while (i < max_iter)
   {
      /* s = C*r */
      /* TO DO:  C = diag scale */
      nalu_hypre_ParVectorCopy(r, s);

      /*gamma = <r,Cr> */
      gamma_old = gamma;
      gamma     = nalu_hypre_ParVectorInnerProd(r, s);

      if (gamma < NALU_HYPRE_REAL_EPSILON)
      {
         break;
      }

      if (i == 0)
      {
         beta = 1.0;
         /* p_0 = C*r */
         nalu_hypre_ParVectorCopy(s, p);
      }
      else
      {
         /* beta = gamma / gamma_old */
         beta = gamma / gamma_old;

         /* p = s + beta p */
         hypreDevice_ComplexAxpyn(p_data, local_size, s_data, p_data, beta);
      }

      if (scale)
      {
         /* s = D^{-1/2}A*D^{-1/2}*p */

         /* u = ds .* p */
         NALU_HYPRE_THRUST_CALL( transform, ds_data, ds_data + local_size, p_data, u_data, _1 * _2 );

         nalu_hypre_ParCSRMatrixMatvec(1.0, A, u, 0.0, s);

         /* s = ds .* s */
         NALU_HYPRE_THRUST_CALL( transform, ds_data, ds_data + local_size, s_data, s_data, _1 * _2 );
      }
      else
      {
         /* s = A*p */
         nalu_hypre_ParCSRMatrixMatvec(1.0, A, p, 0.0, s);
      }

      /* <s,p> */
      sdotp = nalu_hypre_ParVectorInnerProd(s, p);

      /* alpha = gamma / <s,p> */
      alpha = gamma / sdotp;

      /* get tridiagonal matrix */
      alphainv = 1.0 / alpha;

      tridiag[i + 1] = alphainv;
      tridiag[i] *= beta;
      tridiag[i] += alphainv;

      trioffd[i + 1] = alphainv;
      trioffd[i] *= sqrt(beta);

      /* x = x + alpha*p */
      /* don't need */

      /* r = r - alpha*s */
      nalu_hypre_ParVectorAxpy(-alpha, s, r);

      i++;
   }

   /* GPU NOTE:
    * There is a CUDA whitepaper on calculating the eigenvalues of a symmetric
    * tridiagonal matrix via bisection
    * https://docs.nvidia.com/cuda/samples/6_Advanced/eigenvalues/doc/eigenvalues.pdf
    * As well as code in their sample code
    * https://docs.nvidia.com/cuda/cuda-samples/index.html#eigenvalues
    * They claim that all code is available under a permissive license
    * https://developer.nvidia.com/cuda-code-samples
    * I believe the applicable license is available at
    * https://docs.nvidia.com/cuda/eula/index.html#license-driver
    * but I am not certain, nor do I have the legal knowledge to know if the
    * license is compatible with that which HYPRE is released under.
    */
#if defined(NALU_HYPRE_USING_CUDA)
   nalu_hypre_GpuProfilingPopRange();
#endif
#if defined(NALU_HYPRE_USING_CUDA)
   nalu_hypre_GpuProfilingPushRange("ParCSRMaxEigEstimate_TriDiagEigenSolve");
#endif

   /* eispack routine - eigenvalues return in tridiag and ordered*/
   nalu_hypre_LINPACKcgtql1(&i, tridiag, trioffd, &err);

   lambda_max = tridiag[i - 1];
   lambda_min = tridiag[0];
#if defined(NALU_HYPRE_USING_CUDA)
   nalu_hypre_GpuProfilingPopRange();
#endif
   /* nalu_hypre_printf("linpack max eig est = %g\n", lambda_max);*/
   /* nalu_hypre_printf("linpack min eig est = %g\n", lambda_min);*/

   nalu_hypre_TFree(tridiag, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(trioffd, NALU_HYPRE_MEMORY_HOST);

   nalu_hypre_ParVectorDestroy(r);
   nalu_hypre_ParVectorDestroy(s);
   nalu_hypre_ParVectorDestroy(p);
   nalu_hypre_ParVectorDestroy(ds);
   nalu_hypre_ParVectorDestroy(u);

   /* return */
   *max_eig = lambda_max;
   *min_eig = lambda_min;

   return nalu_hypre_error_flag;
}

#endif
