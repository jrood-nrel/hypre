/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_nalu_hypre_onedpl.hpp"
#include "_nalu_hypre_parcsr_ls.h"
#include "float.h"
#include "ams.h"
#include "_nalu_hypre_utilities.hpp"

/*--------------------------------------------------------------------------
 * nalu_hypre_ParCSRRelax
 *
 * Relaxation on the ParCSR matrix A with right-hand side f and
 * initial guess u. Possible values for relax_type are:
 *
 * 1 = l1-scaled (or weighted) Jacobi
 * 2 = l1-scaled block Gauss-Seidel/SSOR
 * 3 = Kaczmarz
 * 4 = truncated version of 2 (Remark 6.2 in smoothers paper)
 * x = BoomerAMG relaxation with relax_type = |x|
 * (16 = Cheby)
 *
 * The default value of relax_type is 2.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ParCSRRelax( nalu_hypre_ParCSRMatrix *A,              /* matrix to relax with */
                   nalu_hypre_ParVector    *f,              /* right-hand side */
                   NALU_HYPRE_Int           relax_type,     /* relaxation type */
                   NALU_HYPRE_Int           relax_times,    /* number of sweeps */
                   NALU_HYPRE_Real         *l1_norms,       /* l1 norms of the rows of A */
                   NALU_HYPRE_Real          relax_weight,   /* damping coefficient (usually <= 1) */
                   NALU_HYPRE_Real          omega,          /* SOR parameter (usually in (0,2) */
                   NALU_HYPRE_Real          max_eig_est,    /* for cheby smoothers */
                   NALU_HYPRE_Real          min_eig_est,
                   NALU_HYPRE_Int           cheby_order,
                   NALU_HYPRE_Real          cheby_fraction,
                   nalu_hypre_ParVector    *u,              /* initial/updated approximation */
                   nalu_hypre_ParVector    *v,              /* temporary vector */
                   nalu_hypre_ParVector    *z               /* temporary vector */ )
{
   NALU_HYPRE_Int sweep;

   for (sweep = 0; sweep < relax_times; sweep++)
   {
      if (relax_type == 1) /* l1-scaled Jacobi */
      {
         nalu_hypre_BoomerAMGRelax(A, f, NULL, 7, 0, relax_weight, 1.0, l1_norms, u, v, z);
      }
      else if (relax_type == 2 || relax_type == 4) /* offd-l1-scaled block GS */
      {
         /* !!! Note: relax_weight and omega flipped !!! */
         nalu_hypre_BoomerAMGRelaxHybridSOR(A, f, NULL, 0, omega,
                                       relax_weight, l1_norms, u, v, z, 1, 1, 0, 1);
      }
      else if (relax_type == 3) /* Kaczmarz */
      {
         nalu_hypre_BoomerAMGRelax(A, f, NULL, 20, 0, relax_weight, omega, l1_norms, u, v, z);
      }
      else /* call BoomerAMG relaxation */
      {
         if (relax_type == 16)
         {
            nalu_hypre_ParCSRRelax_Cheby(A, f, max_eig_est, min_eig_est, cheby_fraction, cheby_order, 1,
                                    0, u, v, z);
         }
         else
         {
            nalu_hypre_BoomerAMGRelax(A, f, NULL, nalu_hypre_abs(relax_type), 0, relax_weight,
                                 omega, l1_norms, u, v, z);
         }
      }
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ParVectorInRangeOf
 *
 * Return a vector that belongs to the range of a given matrix.
 *--------------------------------------------------------------------------*/

nalu_hypre_ParVector *nalu_hypre_ParVectorInRangeOf(nalu_hypre_ParCSRMatrix *A)
{
   nalu_hypre_ParVector *x;

   x = nalu_hypre_ParVectorCreate(nalu_hypre_ParCSRMatrixComm(A),
                             nalu_hypre_ParCSRMatrixGlobalNumRows(A),
                             nalu_hypre_ParCSRMatrixRowStarts(A));
   nalu_hypre_ParVectorInitialize(x);
   nalu_hypre_ParVectorOwnsData(x) = 1;

   return x;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ParVectorInDomainOf
 *
 * Return a vector that belongs to the domain of a given matrix.
 *--------------------------------------------------------------------------*/

nalu_hypre_ParVector *nalu_hypre_ParVectorInDomainOf(nalu_hypre_ParCSRMatrix *A)
{
   nalu_hypre_ParVector *x;

   x = nalu_hypre_ParVectorCreate(nalu_hypre_ParCSRMatrixComm(A),
                             nalu_hypre_ParCSRMatrixGlobalNumCols(A),
                             nalu_hypre_ParCSRMatrixColStarts(A));
   nalu_hypre_ParVectorInitialize(x);
   nalu_hypre_ParVectorOwnsData(x) = 1;

   return x;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ParVectorBlockSplit
 *
 * Extract the dim sub-vectors x_0,...,x_{dim-1} composing a parallel
 * block vector x. It is assumed that &x[i] = [x_0[i],...,x_{dim-1}[i]].
 *--------------------------------------------------------------------------*/
#if defined(NALU_HYPRE_USING_GPU)
template<NALU_HYPRE_Int dir>
__global__ void
hypreGPUKernel_ParVectorBlockSplitGather(nalu_hypre_DeviceItem &item,
                                         NALU_HYPRE_Int   size,
                                         NALU_HYPRE_Int   dim,
                                         NALU_HYPRE_Real *x0,
                                         NALU_HYPRE_Real *x1,
                                         NALU_HYPRE_Real *x2,
                                         NALU_HYPRE_Real *x)
{
   const NALU_HYPRE_Int i = nalu_hypre_gpu_get_grid_thread_id<1, 1>(item);

   if (i >= size * dim)
   {
      return;
   }

   NALU_HYPRE_Real *xx[3];

   xx[0] = x0;
   xx[1] = x1;
   xx[2] = x2;

   const NALU_HYPRE_Int d = i % dim;
   const NALU_HYPRE_Int k = i / dim;

   if (dir == 0)
   {
      xx[d][k] = x[i];
   }
   else if (dir == 1)
   {
      x[i] = xx[d][k];
   }
}
#endif

NALU_HYPRE_Int nalu_hypre_ParVectorBlockSplit(nalu_hypre_ParVector *x,
                                    nalu_hypre_ParVector *x_[3],
                                    NALU_HYPRE_Int dim)
{
   NALU_HYPRE_Int i, d, size_;
   NALU_HYPRE_Real *x_data, *x_data_[3];

#if defined(NALU_HYPRE_USING_GPU)
   NALU_HYPRE_ExecutionPolicy exec = nalu_hypre_GetExecPolicy1( nalu_hypre_ParVectorMemoryLocation(x) );
#endif

   size_ = nalu_hypre_VectorSize(nalu_hypre_ParVectorLocalVector(x_[0]));

   x_data = nalu_hypre_VectorData(nalu_hypre_ParVectorLocalVector(x));
   for (d = 0; d < dim; d++)
   {
      x_data_[d] = nalu_hypre_VectorData(nalu_hypre_ParVectorLocalVector(x_[d]));
   }

#if defined(NALU_HYPRE_USING_GPU)
   if (exec == NALU_HYPRE_EXEC_DEVICE)
   {
      dim3 bDim = nalu_hypre_GetDefaultDeviceBlockDimension();
      dim3 gDim = nalu_hypre_GetDefaultDeviceGridDimension(size_ * dim, "thread", bDim);
      NALU_HYPRE_GPU_LAUNCH( hypreGPUKernel_ParVectorBlockSplitGather<0>, gDim, bDim,
                        size_, dim, x_data_[0], x_data_[1], x_data_[2], x_data);
   }
   else
#endif
   {
      for (i = 0; i < size_; i++)
         for (d = 0; d < dim; d++)
         {
            x_data_[d][i] = x_data[dim * i + d];
         }
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ParVectorBlockGather
 *
 * Compose a parallel block vector x from dim given sub-vectors
 * x_0,...,x_{dim-1}, such that &x[i] = [x_0[i],...,x_{dim-1}[i]].
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int nalu_hypre_ParVectorBlockGather(nalu_hypre_ParVector *x,
                                     nalu_hypre_ParVector *x_[3],
                                     NALU_HYPRE_Int dim)
{
   NALU_HYPRE_Int i, d, size_;
   NALU_HYPRE_Real *x_data, *x_data_[3];

#if defined(NALU_HYPRE_USING_GPU)
   NALU_HYPRE_ExecutionPolicy exec = nalu_hypre_GetExecPolicy1( nalu_hypre_ParVectorMemoryLocation(x) );
#endif

   size_ = nalu_hypre_VectorSize(nalu_hypre_ParVectorLocalVector(x_[0]));

   x_data = nalu_hypre_VectorData(nalu_hypre_ParVectorLocalVector(x));
   for (d = 0; d < dim; d++)
   {
      x_data_[d] = nalu_hypre_VectorData(nalu_hypre_ParVectorLocalVector(x_[d]));
   }

#if defined(NALU_HYPRE_USING_GPU)
   if (exec == NALU_HYPRE_EXEC_DEVICE)
   {
      dim3 bDim = nalu_hypre_GetDefaultDeviceBlockDimension();
      dim3 gDim = nalu_hypre_GetDefaultDeviceGridDimension(size_ * dim, "thread", bDim);
      NALU_HYPRE_GPU_LAUNCH( hypreGPUKernel_ParVectorBlockSplitGather<1>, gDim, bDim,
                        size_, dim, x_data_[0], x_data_[1], x_data_[2], x_data);
   }
   else
#endif
   {
      for (i = 0; i < size_; i++)
         for (d = 0; d < dim; d++)
         {
            x_data[dim * i + d] = x_data_[d][i];
         }
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_BoomerAMGBlockSolve
 *
 * Apply the block-diagonal solver diag(B) to the system diag(A) x = b.
 * Here B is a given BoomerAMG solver for A, while x and b are "block"
 * parallel vectors.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int nalu_hypre_BoomerAMGBlockSolve(void *B,
                                    nalu_hypre_ParCSRMatrix *A,
                                    nalu_hypre_ParVector *b,
                                    nalu_hypre_ParVector *x)
{
   NALU_HYPRE_Int d, dim = 1;

   nalu_hypre_ParVector *b_[3];
   nalu_hypre_ParVector *x_[3];

   dim = nalu_hypre_ParVectorGlobalSize(x) / nalu_hypre_ParCSRMatrixGlobalNumRows(A);

   if (dim == 1)
   {
      nalu_hypre_BoomerAMGSolve(B, A, b, x);
      return nalu_hypre_error_flag;
   }

   for (d = 0; d < dim; d++)
   {
      b_[d] = nalu_hypre_ParVectorInRangeOf(A);
      x_[d] = nalu_hypre_ParVectorInRangeOf(A);
   }

   nalu_hypre_ParVectorBlockSplit(b, b_, dim);
   nalu_hypre_ParVectorBlockSplit(x, x_, dim);

   for (d = 0; d < dim; d++)
   {
      nalu_hypre_BoomerAMGSolve(B, A, b_[d], x_[d]);
   }

   nalu_hypre_ParVectorBlockGather(x, x_, dim);

   for (d = 0; d < dim; d++)
   {
      nalu_hypre_ParVectorDestroy(b_[d]);
      nalu_hypre_ParVectorDestroy(x_[d]);
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ParCSRMatrixFixZeroRows
 *
 * For every zero row in the matrix: set the diagonal element to 1.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int nalu_hypre_ParCSRMatrixFixZeroRowsHost(nalu_hypre_ParCSRMatrix *A)
{
   NALU_HYPRE_Int i, j;
   NALU_HYPRE_Real l1_norm;
   NALU_HYPRE_Int num_rows = nalu_hypre_ParCSRMatrixNumRows(A);

   nalu_hypre_CSRMatrix *A_diag = nalu_hypre_ParCSRMatrixDiag(A);
   NALU_HYPRE_Int *A_diag_I = nalu_hypre_CSRMatrixI(A_diag);
   NALU_HYPRE_Int *A_diag_J = nalu_hypre_CSRMatrixJ(A_diag);
   NALU_HYPRE_Real *A_diag_data = nalu_hypre_CSRMatrixData(A_diag);

   nalu_hypre_CSRMatrix *A_offd = nalu_hypre_ParCSRMatrixOffd(A);
   NALU_HYPRE_Int *A_offd_I = nalu_hypre_CSRMatrixI(A_offd);
   NALU_HYPRE_Real *A_offd_data = nalu_hypre_CSRMatrixData(A_offd);
   NALU_HYPRE_Int num_cols_offd = nalu_hypre_CSRMatrixNumCols(A_offd);

   /* a row will be considered zero if its l1 norm is less than eps */
   NALU_HYPRE_Real eps = 0.0; /* DBL_EPSILON * 1e+4; */

   for (i = 0; i < num_rows; i++)
   {
      l1_norm = 0.0;
      for (j = A_diag_I[i]; j < A_diag_I[i + 1]; j++)
      {
         l1_norm += nalu_hypre_abs(A_diag_data[j]);
      }
      if (num_cols_offd)
         for (j = A_offd_I[i]; j < A_offd_I[i + 1]; j++)
         {
            l1_norm += nalu_hypre_abs(A_offd_data[j]);
         }

      if (l1_norm <= eps)
      {
         for (j = A_diag_I[i]; j < A_diag_I[i + 1]; j++)
            if (A_diag_J[j] == i)
            {
               A_diag_data[j] = 1.0;
            }
            else
            {
               A_diag_data[j] = 0.0;
            }
         if (num_cols_offd)
            for (j = A_offd_I[i]; j < A_offd_I[i + 1]; j++)
            {
               A_offd_data[j] = 0.0;
            }
      }
   }

   return nalu_hypre_error_flag;
}

#if defined(NALU_HYPRE_USING_GPU)
__global__ void
hypreGPUKernel_ParCSRMatrixFixZeroRows( nalu_hypre_DeviceItem    &item,
                                        NALU_HYPRE_Int      nrows,
                                        NALU_HYPRE_Int     *A_diag_i,
                                        NALU_HYPRE_Int     *A_diag_j,
                                        NALU_HYPRE_Complex *A_diag_data,
                                        NALU_HYPRE_Int     *A_offd_i,
                                        NALU_HYPRE_Complex *A_offd_data,
                                        NALU_HYPRE_Int      num_cols_offd)
{
   NALU_HYPRE_Int row_i = nalu_hypre_gpu_get_grid_warp_id<1, 1>(item);

   if (row_i >= nrows)
   {
      return;
   }

   NALU_HYPRE_Int lane = nalu_hypre_gpu_get_lane_id<1>(item);
   NALU_HYPRE_Real eps = 0.0; /* DBL_EPSILON * 1e+4; */
   NALU_HYPRE_Real l1_norm = 0.0;
   NALU_HYPRE_Int p1 = 0, q1, p2 = 0, q2 = 0;

   if (lane < 2)
   {
      p1 = read_only_load(A_diag_i + row_i + lane);
      if (num_cols_offd)
      {
         p2 = read_only_load(A_offd_i + row_i + lane);
      }
   }

   q1 = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, p1, 1);
   p1 = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, p1, 0);
   if (num_cols_offd)
   {
      q2 = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, p2, 1);
      p2 = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, p2, 0);
   }

   for (NALU_HYPRE_Int j = p1 + lane; j < q1; j += NALU_HYPRE_WARP_SIZE)
   {
      l1_norm += nalu_hypre_abs(A_diag_data[j]);
   }

   for (NALU_HYPRE_Int j = p2 + lane; j < q2; j += NALU_HYPRE_WARP_SIZE)
   {
      l1_norm += nalu_hypre_abs(A_offd_data[j]);
   }

   l1_norm = warp_allreduce_sum(item, l1_norm);

   if (l1_norm <= eps)
   {
      for (NALU_HYPRE_Int j = p1 + lane; j < q1; j += NALU_HYPRE_WARP_SIZE)
      {
         if (row_i == read_only_load(&A_diag_j[j]))
         {
            A_diag_data[j] = 1.0;
         }
         else
         {
            A_diag_data[j] = 0.0;
         }
      }

      for (NALU_HYPRE_Int j = p2 + lane; j < q2; j += NALU_HYPRE_WARP_SIZE)
      {
         A_offd_data[j] = 0.0;
      }
   }
}

NALU_HYPRE_Int nalu_hypre_ParCSRMatrixFixZeroRowsDevice(nalu_hypre_ParCSRMatrix *A)
{
   NALU_HYPRE_Int        nrows         = nalu_hypre_ParCSRMatrixNumRows(A);
   nalu_hypre_CSRMatrix *A_diag        = nalu_hypre_ParCSRMatrixDiag(A);
   NALU_HYPRE_Real      *A_diag_data   = nalu_hypre_CSRMatrixData(A_diag);
   NALU_HYPRE_Int       *A_diag_i      = nalu_hypre_CSRMatrixI(A_diag);
   NALU_HYPRE_Int       *A_diag_j      = nalu_hypre_CSRMatrixJ(A_diag);
   nalu_hypre_CSRMatrix *A_offd        = nalu_hypre_ParCSRMatrixOffd(A);
   NALU_HYPRE_Real      *A_offd_data   = nalu_hypre_CSRMatrixData(A_offd);
   NALU_HYPRE_Int       *A_offd_i      = nalu_hypre_CSRMatrixI(A_offd);
   NALU_HYPRE_Int        num_cols_offd = nalu_hypre_CSRMatrixNumCols(A_offd);

   dim3 bDim = nalu_hypre_GetDefaultDeviceBlockDimension();
   dim3 gDim = nalu_hypre_GetDefaultDeviceGridDimension(nrows, "warp", bDim);

   NALU_HYPRE_GPU_LAUNCH(hypreGPUKernel_ParCSRMatrixFixZeroRows, gDim, bDim,
                    nrows, A_diag_i, A_diag_j, A_diag_data, A_offd_i, A_offd_data, num_cols_offd);

   //nalu_hypre_SyncComputeStream(nalu_hypre_handle());

   return nalu_hypre_error_flag;
}
#endif

NALU_HYPRE_Int nalu_hypre_ParCSRMatrixFixZeroRows(nalu_hypre_ParCSRMatrix *A)
{
#if defined(NALU_HYPRE_USING_GPU)
   NALU_HYPRE_ExecutionPolicy exec = nalu_hypre_GetExecPolicy1( nalu_hypre_ParCSRMatrixMemoryLocation(A) );
   if (exec == NALU_HYPRE_EXEC_DEVICE)
   {
      return nalu_hypre_ParCSRMatrixFixZeroRowsDevice(A);
   }
   else
#endif
   {
      return nalu_hypre_ParCSRMatrixFixZeroRowsHost(A);
   }
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ParCSRComputeL1Norms
 *
 * Compute the l1 norms of the rows of a given matrix, depending on
 * the option parameter:
 *
 * option 1 = Compute the l1 norm of the rows
 * option 2 = Compute the l1 norm of the (processor) off-diagonal
 *            part of the rows plus the diagonal of A
 * option 3 = Compute the l2 norm^2 of the rows
 * option 4 = Truncated version of option 2 based on Remark 6.2 in "Multigrid
 *            Smoothers for Ultra-Parallel Computing"
 *
 * The above computations are done in a CF manner, whenever the provided
 * cf_marker is not NULL.
 *--------------------------------------------------------------------------*/

#if defined(NALU_HYPRE_USING_GPU)
#if defined(NALU_HYPRE_USING_SYCL)
struct l1_norm_op1
#else
struct l1_norm_op1 : public thrust::binary_function<NALU_HYPRE_Complex, NALU_HYPRE_Complex, NALU_HYPRE_Complex>
#endif
{
   __host__ __device__
   NALU_HYPRE_Complex operator()(const NALU_HYPRE_Complex &x, const NALU_HYPRE_Complex &y) const
   {
      return x <= 4.0 / 3.0 * y ? y : x;
   }
};
#endif

NALU_HYPRE_Int nalu_hypre_ParCSRComputeL1Norms(nalu_hypre_ParCSRMatrix  *A,
                                     NALU_HYPRE_Int            option,
                                     NALU_HYPRE_Int           *cf_marker,
                                     NALU_HYPRE_Real         **l1_norm_ptr)
{
   NALU_HYPRE_Int i;
   NALU_HYPRE_Int num_rows = nalu_hypre_ParCSRMatrixNumRows(A);
   nalu_hypre_CSRMatrix *A_diag = nalu_hypre_ParCSRMatrixDiag(A);
   nalu_hypre_CSRMatrix *A_offd = nalu_hypre_ParCSRMatrixOffd(A);
   NALU_HYPRE_Int num_cols_offd = nalu_hypre_CSRMatrixNumCols(A_offd);

   NALU_HYPRE_MemoryLocation memory_location_l1 = nalu_hypre_ParCSRMatrixMemoryLocation(A);

   NALU_HYPRE_ExecutionPolicy exec = nalu_hypre_GetExecPolicy1( memory_location_l1 );

   if (exec == NALU_HYPRE_EXEC_HOST)
   {
      NALU_HYPRE_Int num_threads = nalu_hypre_NumThreads();
      if (num_threads > 1)
      {
         return nalu_hypre_ParCSRComputeL1NormsThreads(A, option, num_threads, cf_marker, l1_norm_ptr);
      }
   }

   NALU_HYPRE_Real *l1_norm = nalu_hypre_TAlloc(NALU_HYPRE_Real, num_rows, memory_location_l1);

   NALU_HYPRE_MemoryLocation memory_location_tmp =
      exec == NALU_HYPRE_EXEC_HOST ? NALU_HYPRE_MEMORY_HOST : NALU_HYPRE_MEMORY_DEVICE;

   NALU_HYPRE_Real *diag_tmp = NULL;

   NALU_HYPRE_Int *cf_marker_offd = NULL;

   /* collect the cf marker data from other procs */
   if (cf_marker != NULL)
   {
      NALU_HYPRE_Int num_sends;
      NALU_HYPRE_Int *int_buf_data = NULL;

      nalu_hypre_ParCSRCommPkg  *comm_pkg = nalu_hypre_ParCSRMatrixCommPkg(A);
      nalu_hypre_ParCSRCommHandle *comm_handle;

      if (num_cols_offd)
      {
         cf_marker_offd = nalu_hypre_CTAlloc(NALU_HYPRE_Int, num_cols_offd, memory_location_tmp);
      }
      num_sends = nalu_hypre_ParCSRCommPkgNumSends(comm_pkg);
      if (nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends))
      {
         int_buf_data = nalu_hypre_CTAlloc(NALU_HYPRE_Int, nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends),
                                      memory_location_tmp);
      }
#if defined(NALU_HYPRE_USING_GPU)
      if (exec == NALU_HYPRE_EXEC_DEVICE)
      {
         nalu_hypre_ParCSRCommPkgCopySendMapElmtsToDevice(comm_pkg);
#if defined(NALU_HYPRE_USING_SYCL)
         hypreSycl_gather( nalu_hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg),
                           nalu_hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg) + nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg,
                                                                                                             num_sends),
                           cf_marker,
                           int_buf_data );
#else
         NALU_HYPRE_THRUST_CALL( gather,
                            nalu_hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg),
                            nalu_hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg) + nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg,
                                  num_sends),
                            cf_marker,
                            int_buf_data );
#endif

#if defined(NALU_HYPRE_WITH_GPU_AWARE_MPI) && defined(NALU_HYPRE_USING_THRUST_NOSYNC)
         /* RL: make sure int_buf_data is ready before issuing GPU-GPU MPI */
         nalu_hypre_ForceSyncComputeStream(nalu_hypre_handle());
#endif
      }
      else
#endif
      {
         NALU_HYPRE_Int index = 0;
         NALU_HYPRE_Int start;
         NALU_HYPRE_Int j;
         for (i = 0; i < num_sends; i++)
         {
            start = nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
            for (j = start; j < nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, i + 1); j++)
            {
               int_buf_data[index++] = cf_marker[nalu_hypre_ParCSRCommPkgSendMapElmt(comm_pkg, j)];
            }
         }
      }

      comm_handle = nalu_hypre_ParCSRCommHandleCreate_v2(11, comm_pkg, memory_location_tmp, int_buf_data,
                                                    memory_location_tmp, cf_marker_offd);
      nalu_hypre_ParCSRCommHandleDestroy(comm_handle);
      nalu_hypre_TFree(int_buf_data, memory_location_tmp);
   }

   if (option == 1)
   {
      /* Set the l1 norm of the diag part */
      nalu_hypre_CSRMatrixComputeRowSum(A_diag, cf_marker, cf_marker, l1_norm, 1, 1.0, "set");

      /* Add the l1 norm of the offd part */
      if (num_cols_offd)
      {
         nalu_hypre_CSRMatrixComputeRowSum(A_offd, cf_marker, cf_marker_offd, l1_norm, 1, 1.0, "add");
      }
   }
   else if (option == 2)
   {
      /* Set the abs(diag) element */
      nalu_hypre_CSRMatrixExtractDiagonal(A_diag, l1_norm, 1);
      /* Add the l1 norm of the offd part */
      if (num_cols_offd)
      {
         nalu_hypre_CSRMatrixComputeRowSum(A_offd, cf_marker, cf_marker, l1_norm, 1, 1.0, "add");
      }
   }
   else if (option == 3)
   {
      /* Set the CF l2 norm of the diag part */
      nalu_hypre_CSRMatrixComputeRowSum(A_diag, NULL, NULL, l1_norm, 2, 1.0, "set");
      /* Add the CF l2 norm of the offd part */
      if (num_cols_offd)
      {
         nalu_hypre_CSRMatrixComputeRowSum(A_offd, NULL, NULL, l1_norm, 2, 1.0, "add");
      }
   }
   else if (option == 4)
   {
      /* Set the abs(diag) element */
      nalu_hypre_CSRMatrixExtractDiagonal(A_diag, l1_norm, 1);

      diag_tmp = nalu_hypre_TAlloc(NALU_HYPRE_Real, num_rows, memory_location_tmp);
      nalu_hypre_TMemcpy(diag_tmp, l1_norm, NALU_HYPRE_Real, num_rows, memory_location_tmp, memory_location_l1);

      /* Add the scaled l1 norm of the offd part */
      if (num_cols_offd)
      {
         nalu_hypre_CSRMatrixComputeRowSum(A_offd, cf_marker, cf_marker_offd, l1_norm, 1, 0.5, "add");
      }

      /* Truncate according to Remark 6.2 */
#if defined(NALU_HYPRE_USING_GPU)
      if (exec == NALU_HYPRE_EXEC_DEVICE)
      {
#if defined(NALU_HYPRE_USING_SYCL)
         NALU_HYPRE_ONEDPL_CALL( std::transform, l1_norm, l1_norm + num_rows, diag_tmp, l1_norm, l1_norm_op1() );
#else
         NALU_HYPRE_THRUST_CALL( transform, l1_norm, l1_norm + num_rows, diag_tmp, l1_norm, l1_norm_op1() );
#endif
      }
      else
#endif
      {
         for (i = 0; i < num_rows; i++)
         {
            if (l1_norm[i] <= 4.0 / 3.0 * diag_tmp[i])
            {
               l1_norm[i] = diag_tmp[i];
            }
         }
      }
   }
   else if (option == 5) /*stores diagonal of A for Jacobi using matvec, rlx 7 */
   {
      /* Set the diag element */
      nalu_hypre_CSRMatrixExtractDiagonal(A_diag, l1_norm, 0);

#if defined(NALU_HYPRE_USING_GPU)
      if ( exec == NALU_HYPRE_EXEC_DEVICE)
      {
#if defined(NALU_HYPRE_USING_SYCL)
         NALU_HYPRE_ONEDPL_CALL( std::replace_if, l1_norm, l1_norm + num_rows, [] (const auto & x) {return !x;},
         1.0 );
#else
         thrust::identity<NALU_HYPRE_Complex> identity;
         NALU_HYPRE_THRUST_CALL( replace_if, l1_norm, l1_norm + num_rows, thrust::not1(identity), 1.0 );
#endif
      }
      else
#endif
      {
         for (i = 0; i < num_rows; i++)
         {
            if (l1_norm[i] == 0.0)
            {
               l1_norm[i] = 1.0;
            }
         }
      }

      *l1_norm_ptr = l1_norm;

      return nalu_hypre_error_flag;
   }

   /* Handle negative definite matrices */
   if (!diag_tmp)
   {
      diag_tmp = nalu_hypre_TAlloc(NALU_HYPRE_Real, num_rows, memory_location_tmp);
   }

   /* Set the diag element */
   nalu_hypre_CSRMatrixExtractDiagonal(A_diag, diag_tmp, 0);

#if defined(NALU_HYPRE_USING_GPU)
   if (exec == NALU_HYPRE_EXEC_DEVICE)
   {
#if defined(NALU_HYPRE_USING_SYCL)
      hypreSycl_transform_if( l1_norm, l1_norm + num_rows, diag_tmp, l1_norm,
                              std::negate<NALU_HYPRE_Real>(),
                              is_negative<NALU_HYPRE_Real>() );
      bool any_zero = 0.0 == NALU_HYPRE_ONEDPL_CALL( std::reduce, l1_norm, l1_norm + num_rows, 1.0,
                                                oneapi::dpl::minimum<NALU_HYPRE_Real>() );
#else
      NALU_HYPRE_THRUST_CALL( transform_if, l1_norm, l1_norm + num_rows, diag_tmp, l1_norm,
                         thrust::negate<NALU_HYPRE_Real>(),
                         is_negative<NALU_HYPRE_Real>() );
      //bool any_zero = NALU_HYPRE_THRUST_CALL( any_of, l1_norm, l1_norm + num_rows, thrust::not1(thrust::identity<NALU_HYPRE_Complex>()) );
      bool any_zero = 0.0 == NALU_HYPRE_THRUST_CALL( reduce, l1_norm, l1_norm + num_rows, 1.0,
                                                thrust::minimum<NALU_HYPRE_Real>() );
#endif
      if ( any_zero )
      {
         nalu_hypre_error_in_arg(1);
      }
   }
   else
#endif
   {
      for (i = 0; i < num_rows; i++)
      {
         if (diag_tmp[i] < 0.0)
         {
            l1_norm[i] = -l1_norm[i];
         }
      }

      for (i = 0; i < num_rows; i++)
      {
         /* if (nalu_hypre_abs(l1_norm[i]) < DBL_EPSILON) */
         if (nalu_hypre_abs(l1_norm[i]) == 0.0)
         {
            nalu_hypre_error_in_arg(1);
            break;
         }
      }
   }

   nalu_hypre_TFree(cf_marker_offd, memory_location_tmp);
   nalu_hypre_TFree(diag_tmp, memory_location_tmp);

   *l1_norm_ptr = l1_norm;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ParCSRMatrixSetDiagRows
 *
 * For every row containing only a diagonal element: set it to d.
 *--------------------------------------------------------------------------*/
#if defined(NALU_HYPRE_USING_GPU)
__global__ void
hypreGPUKernel_ParCSRMatrixSetDiagRows(nalu_hypre_DeviceItem    &item,
                                       NALU_HYPRE_Int      nrows,
                                       NALU_HYPRE_Int     *A_diag_I,
                                       NALU_HYPRE_Int     *A_diag_J,
                                       NALU_HYPRE_Complex *A_diag_data,
                                       NALU_HYPRE_Int     *A_offd_I,
                                       NALU_HYPRE_Int      num_cols_offd,
                                       NALU_HYPRE_Real     d)
{
   const NALU_HYPRE_Int i = nalu_hypre_gpu_get_grid_thread_id<1, 1>(item);
   if (i >= nrows)
   {
      return;
   }

   NALU_HYPRE_Int j = read_only_load(&A_diag_I[i]);

   if ( (read_only_load(&A_diag_I[i + 1]) == j + 1) && (read_only_load(&A_diag_J[j]) == i) &&
        (!num_cols_offd || (read_only_load(&A_offd_I[i + 1]) == read_only_load(&A_offd_I[i]))) )
   {
      A_diag_data[j] = d;
   }
}
#endif

NALU_HYPRE_Int nalu_hypre_ParCSRMatrixSetDiagRows(nalu_hypre_ParCSRMatrix *A, NALU_HYPRE_Real d)
{
   NALU_HYPRE_Int i, j;
   NALU_HYPRE_Int num_rows = nalu_hypre_ParCSRMatrixNumRows(A);

   nalu_hypre_CSRMatrix *A_diag = nalu_hypre_ParCSRMatrixDiag(A);
   NALU_HYPRE_Int *A_diag_I = nalu_hypre_CSRMatrixI(A_diag);
   NALU_HYPRE_Int *A_diag_J = nalu_hypre_CSRMatrixJ(A_diag);
   NALU_HYPRE_Real *A_diag_data = nalu_hypre_CSRMatrixData(A_diag);

   nalu_hypre_CSRMatrix *A_offd = nalu_hypre_ParCSRMatrixOffd(A);
   NALU_HYPRE_Int *A_offd_I = nalu_hypre_CSRMatrixI(A_offd);
   NALU_HYPRE_Int num_cols_offd = nalu_hypre_CSRMatrixNumCols(A_offd);

#if defined(NALU_HYPRE_USING_GPU)
   NALU_HYPRE_ExecutionPolicy exec = nalu_hypre_GetExecPolicy1( nalu_hypre_ParCSRMatrixMemoryLocation(A) );
   if (exec == NALU_HYPRE_EXEC_DEVICE)
   {
      dim3 bDim = nalu_hypre_GetDefaultDeviceBlockDimension();
      dim3 gDim = nalu_hypre_GetDefaultDeviceGridDimension(num_rows, "thread", bDim);
      NALU_HYPRE_GPU_LAUNCH( hypreGPUKernel_ParCSRMatrixSetDiagRows, gDim, bDim,
                        num_rows, A_diag_I, A_diag_J, A_diag_data, A_offd_I, num_cols_offd, d);
   }
   else
#endif
   {
      for (i = 0; i < num_rows; i++)
      {
         j = A_diag_I[i];
         if ((A_diag_I[i + 1] == j + 1) && (A_diag_J[j] == i) &&
             (!num_cols_offd || (A_offd_I[i + 1] == A_offd_I[i])))
         {
            A_diag_data[j] = d;
         }
      }
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_AMSCreate
 *
 * Allocate the AMS solver structure.
 *--------------------------------------------------------------------------*/

void * nalu_hypre_AMSCreate(void)
{
   nalu_hypre_AMSData *ams_data;

   ams_data = nalu_hypre_CTAlloc(nalu_hypre_AMSData,  1, NALU_HYPRE_MEMORY_HOST);

   /* Default parameters */

   ams_data -> dim = 3;                /* 3D problem */
   ams_data -> maxit = 20;             /* perform at most 20 iterations */
   ams_data -> tol = 1e-6;             /* convergence tolerance */
   ams_data -> print_level = 1;        /* print residual norm at each step */
   ams_data -> cycle_type = 1;         /* a 3-level multiplicative solver */
   ams_data -> A_relax_type = 2;       /* offd-l1-scaled GS */
   ams_data -> A_relax_times = 1;      /* one relaxation sweep */
   ams_data -> A_relax_weight = 1.0;   /* damping parameter */
   ams_data -> A_omega = 1.0;          /* SSOR coefficient */
   ams_data -> A_cheby_order = 2;      /* Cheby: order (1 -4 are vaild) */
   ams_data -> A_cheby_fraction = .3;  /* Cheby: fraction of spectrum to smooth */

   ams_data -> B_G_coarsen_type = 10;  /* HMIS coarsening */
   ams_data -> B_G_agg_levels = 1;     /* Levels of aggressive coarsening */
   ams_data -> B_G_relax_type = 3;     /* hybrid G-S/Jacobi */
   ams_data -> B_G_theta = 0.25;       /* strength threshold */
   ams_data -> B_G_interp_type = 0;    /* interpolation type */
   ams_data -> B_G_Pmax = 0;           /* max nonzero elements in interp. rows */
   ams_data -> B_Pi_coarsen_type = 10; /* HMIS coarsening */
   ams_data -> B_Pi_agg_levels = 1;    /* Levels of aggressive coarsening */
   ams_data -> B_Pi_relax_type = 3;    /* hybrid G-S/Jacobi */
   ams_data -> B_Pi_theta = 0.25;      /* strength threshold */
   ams_data -> B_Pi_interp_type = 0;   /* interpolation type */
   ams_data -> B_Pi_Pmax = 0;          /* max nonzero elements in interp. rows */
   ams_data -> beta_is_zero = 0;       /* the problem has a mass term */

   /* By default, do l1-GS smoothing on the coarsest grid */
   ams_data -> B_G_coarse_relax_type  = 8;
   ams_data -> B_Pi_coarse_relax_type = 8;

   /* The rest of the fields are initialized using the Set functions */

   ams_data -> A    = NULL;
   ams_data -> G    = NULL;
   ams_data -> A_G  = NULL;
   ams_data -> B_G  = 0;
   ams_data -> Pi   = NULL;
   ams_data -> A_Pi = NULL;
   ams_data -> B_Pi = 0;
   ams_data -> x    = NULL;
   ams_data -> y    = NULL;
   ams_data -> z    = NULL;
   ams_data -> Gx   = NULL;
   ams_data -> Gy   = NULL;
   ams_data -> Gz   = NULL;

   ams_data -> r0  = NULL;
   ams_data -> g0  = NULL;
   ams_data -> r1  = NULL;
   ams_data -> g1  = NULL;
   ams_data -> r2  = NULL;
   ams_data -> g2  = NULL;
   ams_data -> zz  = NULL;

   ams_data -> Pix    = NULL;
   ams_data -> Piy    = NULL;
   ams_data -> Piz    = NULL;
   ams_data -> A_Pix  = NULL;
   ams_data -> A_Piy  = NULL;
   ams_data -> A_Piz  = NULL;
   ams_data -> B_Pix  = 0;
   ams_data -> B_Piy  = 0;
   ams_data -> B_Piz  = 0;

   ams_data -> interior_nodes       = NULL;
   ams_data -> G0                   = NULL;
   ams_data -> A_G0                 = NULL;
   ams_data -> B_G0                 = 0;
   ams_data -> projection_frequency = 5;

   ams_data -> A_l1_norms = NULL;
   ams_data -> A_max_eig_est = 0;
   ams_data -> A_min_eig_est = 0;

   ams_data -> owns_Pi   = 1;
   ams_data -> owns_A_G  = 0;
   ams_data -> owns_A_Pi = 0;

   return (void *) ams_data;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_AMSDestroy
 *
 * Deallocate the AMS solver structure. Note that the input data (given
 * through the Set functions) is not destroyed.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int nalu_hypre_AMSDestroy(void *solver)
{
   nalu_hypre_AMSData *ams_data = (nalu_hypre_AMSData *) solver;

   if (!ams_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   if (ams_data -> owns_A_G)
      if (ams_data -> A_G)
      {
         nalu_hypre_ParCSRMatrixDestroy(ams_data -> A_G);
      }
   if (!ams_data -> beta_is_zero)
      if (ams_data -> B_G)
      {
         NALU_HYPRE_BoomerAMGDestroy(ams_data -> B_G);
      }

   if (ams_data -> owns_Pi && ams_data -> Pi)
   {
      nalu_hypre_ParCSRMatrixDestroy(ams_data -> Pi);
   }
   if (ams_data -> owns_A_Pi)
      if (ams_data -> A_Pi)
      {
         nalu_hypre_ParCSRMatrixDestroy(ams_data -> A_Pi);
      }
   if (ams_data -> B_Pi)
   {
      NALU_HYPRE_BoomerAMGDestroy(ams_data -> B_Pi);
   }

   if (ams_data -> owns_Pi && ams_data -> Pix)
   {
      nalu_hypre_ParCSRMatrixDestroy(ams_data -> Pix);
   }
   if (ams_data -> A_Pix)
   {
      nalu_hypre_ParCSRMatrixDestroy(ams_data -> A_Pix);
   }
   if (ams_data -> B_Pix)
   {
      NALU_HYPRE_BoomerAMGDestroy(ams_data -> B_Pix);
   }
   if (ams_data -> owns_Pi && ams_data -> Piy)
   {
      nalu_hypre_ParCSRMatrixDestroy(ams_data -> Piy);
   }
   if (ams_data -> A_Piy)
   {
      nalu_hypre_ParCSRMatrixDestroy(ams_data -> A_Piy);
   }
   if (ams_data -> B_Piy)
   {
      NALU_HYPRE_BoomerAMGDestroy(ams_data -> B_Piy);
   }
   if (ams_data -> owns_Pi && ams_data -> Piz)
   {
      nalu_hypre_ParCSRMatrixDestroy(ams_data -> Piz);
   }
   if (ams_data -> A_Piz)
   {
      nalu_hypre_ParCSRMatrixDestroy(ams_data -> A_Piz);
   }
   if (ams_data -> B_Piz)
   {
      NALU_HYPRE_BoomerAMGDestroy(ams_data -> B_Piz);
   }

   if (ams_data -> r0)
   {
      nalu_hypre_ParVectorDestroy(ams_data -> r0);
   }
   if (ams_data -> g0)
   {
      nalu_hypre_ParVectorDestroy(ams_data -> g0);
   }
   if (ams_data -> r1)
   {
      nalu_hypre_ParVectorDestroy(ams_data -> r1);
   }
   if (ams_data -> g1)
   {
      nalu_hypre_ParVectorDestroy(ams_data -> g1);
   }
   if (ams_data -> r2)
   {
      nalu_hypre_ParVectorDestroy(ams_data -> r2);
   }
   if (ams_data -> g2)
   {
      nalu_hypre_ParVectorDestroy(ams_data -> g2);
   }
   if (ams_data -> zz)
   {
      nalu_hypre_ParVectorDestroy(ams_data -> zz);
   }

   if (ams_data -> G0)
   {
      nalu_hypre_ParCSRMatrixDestroy(ams_data -> A);
   }
   if (ams_data -> G0)
   {
      nalu_hypre_ParCSRMatrixDestroy(ams_data -> G0);
   }
   if (ams_data -> A_G0)
   {
      nalu_hypre_ParCSRMatrixDestroy(ams_data -> A_G0);
   }
   if (ams_data -> B_G0)
   {
      NALU_HYPRE_BoomerAMGDestroy(ams_data -> B_G0);
   }

   nalu_hypre_SeqVectorDestroy(ams_data -> A_l1_norms);

   /* G, x, y ,z, Gx, Gy and Gz are not destroyed */

   if (ams_data)
   {
      nalu_hypre_TFree(ams_data, NALU_HYPRE_MEMORY_HOST);
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_AMSSetDimension
 *
 * Set problem dimension (2 or 3). By default we assume dim = 3.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int nalu_hypre_AMSSetDimension(void *solver,
                                NALU_HYPRE_Int dim)
{
   nalu_hypre_AMSData *ams_data = (nalu_hypre_AMSData *) solver;

   if (dim != 1 && dim != 2 && dim != 3)
   {
      nalu_hypre_error_in_arg(2);
   }

   ams_data -> dim = dim;
   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_AMSSetDiscreteGradient
 *
 * Set the discrete gradient matrix G.
 * This function should be called before nalu_hypre_AMSSetup()!
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int nalu_hypre_AMSSetDiscreteGradient(void *solver,
                                       nalu_hypre_ParCSRMatrix *G)
{
   nalu_hypre_AMSData *ams_data = (nalu_hypre_AMSData *) solver;
   ams_data -> G = G;
   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_AMSSetCoordinateVectors
 *
 * Set the x, y and z coordinates of the vertices in the mesh.
 *
 * Either SetCoordinateVectors or SetEdgeConstantVectors should be
 * called before nalu_hypre_AMSSetup()!
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int nalu_hypre_AMSSetCoordinateVectors(void *solver,
                                        nalu_hypre_ParVector *x,
                                        nalu_hypre_ParVector *y,
                                        nalu_hypre_ParVector *z)
{
   nalu_hypre_AMSData *ams_data = (nalu_hypre_AMSData *) solver;
   ams_data -> x = x;
   ams_data -> y = y;
   ams_data -> z = z;
   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_AMSSetEdgeConstantVectors
 *
 * Set the vectors Gx, Gy and Gz which give the representations of
 * the constant vector fields (1,0,0), (0,1,0) and (0,0,1) in the
 * edge element basis.
 *
 * Either SetCoordinateVectors or SetEdgeConstantVectors should be
 * called before nalu_hypre_AMSSetup()!
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int nalu_hypre_AMSSetEdgeConstantVectors(void *solver,
                                          nalu_hypre_ParVector *Gx,
                                          nalu_hypre_ParVector *Gy,
                                          nalu_hypre_ParVector *Gz)
{
   nalu_hypre_AMSData *ams_data = (nalu_hypre_AMSData *) solver;
   ams_data -> Gx = Gx;
   ams_data -> Gy = Gy;
   ams_data -> Gz = Gz;
   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_AMSSetInterpolations
 *
 * Set the (components of) the Nedelec interpolation matrix Pi=[Pix,Piy,Piz].
 *
 * This function is generally intended to be used only for high-order Nedelec
 * discretizations (in the lowest order case, Pi is constructed internally in
 * AMS from the discreet gradient matrix and the coordinates of the vertices),
 * though it can also be used in the lowest-order case or for other types of
 * discretizations (e.g. ones based on the second family of Nedelec elements).
 *
 * By definition, Pi is the matrix representation of the linear operator that
 * interpolates (high-order) vector nodal finite elements into the (high-order)
 * Nedelec space. The component matrices are defined as Pix phi = Pi (phi,0,0)
 * and similarly for Piy and Piz. Note that all these operators depend on the
 * choice of the basis and degrees of freedom in the high-order spaces.
 *
 * The column numbering of Pi should be node-based, i.e. the x/y/z components of
 * the first node (vertex or high-order dof) should be listed first, followed by
 * the x/y/z components of the second node and so on (see the documentation of
 * NALU_HYPRE_BoomerAMGSetDofFunc).
 *
 * If used, this function should be called before nalu_hypre_AMSSetup() and there is
 * no need to provide the vertex coordinates. Furthermore, only one of the sets
 * {Pi} and {Pix,Piy,Piz} needs to be specified (though it is OK to provide
 * both).  If Pix is NULL, then scalar Pi-based AMS cycles, i.e. those with
 * cycle_type > 10, will be unavailable.  Similarly, AMS cycles based on
 * monolithic Pi (cycle_type < 10) require that Pi is not NULL.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int nalu_hypre_AMSSetInterpolations(void *solver,
                                     nalu_hypre_ParCSRMatrix *Pi,
                                     nalu_hypre_ParCSRMatrix *Pix,
                                     nalu_hypre_ParCSRMatrix *Piy,
                                     nalu_hypre_ParCSRMatrix *Piz)
{
   nalu_hypre_AMSData *ams_data = (nalu_hypre_AMSData *) solver;
   ams_data -> Pi = Pi;
   ams_data -> Pix = Pix;
   ams_data -> Piy = Piy;
   ams_data -> Piz = Piz;
   ams_data -> owns_Pi = 0;
   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_AMSSetAlphaPoissonMatrix
 *
 * Set the matrix corresponding to the Poisson problem with coefficient
 * alpha (the curl-curl term coefficient in the Maxwell problem).
 *
 * If this function is called, the coarse space solver on the range
 * of Pi^T is a block-diagonal version of A_Pi. If this function is not
 * called, the coarse space solver on the range of Pi^T is constructed
 * as Pi^T A Pi in nalu_hypre_AMSSetup().
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int nalu_hypre_AMSSetAlphaPoissonMatrix(void *solver,
                                         nalu_hypre_ParCSRMatrix *A_Pi)
{
   nalu_hypre_AMSData *ams_data = (nalu_hypre_AMSData *) solver;
   ams_data -> A_Pi = A_Pi;

   /* Penalize the eliminated degrees of freedom */
   nalu_hypre_ParCSRMatrixSetDiagRows(A_Pi, NALU_HYPRE_REAL_MAX);

   /* Make sure that the first entry in each row is the diagonal one. */
   /* nalu_hypre_CSRMatrixReorder(nalu_hypre_ParCSRMatrixDiag(A_Pi)); */

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_AMSSetBetaPoissonMatrix
 *
 * Set the matrix corresponding to the Poisson problem with coefficient
 * beta (the mass term coefficient in the Maxwell problem).
 *
 * This function call is optional - if not given, the Poisson matrix will
 * be computed in nalu_hypre_AMSSetup(). If the given matrix is NULL, we assume
 * that beta is 0 and use two-level (instead of three-level) methods.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int nalu_hypre_AMSSetBetaPoissonMatrix(void *solver,
                                        nalu_hypre_ParCSRMatrix *A_G)
{
   nalu_hypre_AMSData *ams_data = (nalu_hypre_AMSData *) solver;
   ams_data -> A_G = A_G;
   if (!A_G)
   {
      ams_data -> beta_is_zero = 1;
   }
   else
   {
      /* Penalize the eliminated degrees of freedom */
      nalu_hypre_ParCSRMatrixSetDiagRows(A_G, NALU_HYPRE_REAL_MAX);

      /* Make sure that the first entry in each row is the diagonal one. */
      /* nalu_hypre_CSRMatrixReorder(nalu_hypre_ParCSRMatrixDiag(A_G)); */
   }
   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_AMSSetInteriorNodes
 *
 * Set the list of nodes which are interior to the zero-conductivity region.
 * A node is interior if interior_nodes[i] == 1.0.
 *
 * Should be called before nalu_hypre_AMSSetup()!
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int nalu_hypre_AMSSetInteriorNodes(void *solver,
                                    nalu_hypre_ParVector *interior_nodes)
{
   nalu_hypre_AMSData *ams_data = (nalu_hypre_AMSData *) solver;
   ams_data -> interior_nodes = interior_nodes;
   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_AMSSetProjectionFrequency
 *
 * How often to project the r.h.s. onto the compatible sub-space Ker(G0^T),
 * when iterating with the solver.
 *
 * The default value is every 5th iteration.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int nalu_hypre_AMSSetProjectionFrequency(void *solver,
                                          NALU_HYPRE_Int projection_frequency)
{
   nalu_hypre_AMSData *ams_data = (nalu_hypre_AMSData *) solver;
   ams_data -> projection_frequency = projection_frequency;
   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_AMSSetMaxIter
 *
 * Set the maximum number of iterations in the three-level method.
 * The default value is 20. To use the AMS solver as a preconditioner,
 * set maxit to 1, tol to 0.0 and print_level to 0.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int nalu_hypre_AMSSetMaxIter(void *solver,
                              NALU_HYPRE_Int maxit)
{
   nalu_hypre_AMSData *ams_data = (nalu_hypre_AMSData *) solver;
   ams_data -> maxit = maxit;
   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_AMSSetTol
 *
 * Set the convergence tolerance (if the method is used as a solver).
 * The default value is 1e-6.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int nalu_hypre_AMSSetTol(void *solver,
                          NALU_HYPRE_Real tol)
{
   nalu_hypre_AMSData *ams_data = (nalu_hypre_AMSData *) solver;
   ams_data -> tol = tol;
   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_AMSSetCycleType
 *
 * Choose which three-level solver to use. Possible values are:
 *
 *   1 = 3-level multipl. solver (01210)      <-- small solution time
 *   2 = 3-level additive solver (0+1+2)
 *   3 = 3-level multipl. solver (02120)
 *   4 = 3-level additive solver (010+2)
 *   5 = 3-level multipl. solver (0102010)    <-- small solution time
 *   6 = 3-level additive solver (1+020)
 *   7 = 3-level multipl. solver (0201020)    <-- small number of iterations
 *   8 = 3-level additive solver (0(1+2)0)    <-- small solution time
 *   9 = 3-level multipl. solver (01210) with discrete divergence
 *  11 = 5-level multipl. solver (013454310)  <-- small solution time, memory
 *  12 = 5-level additive solver (0+1+3+4+5)
 *  13 = 5-level multipl. solver (034515430)  <-- small solution time, memory
 *  14 = 5-level additive solver (01(3+4+5)10)
 *  20 = 2-level multipl. solver (0[12]0)
 *
 *   0 = a Hiptmair-like smoother (010)
 *
 * The default value is 1.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int nalu_hypre_AMSSetCycleType(void *solver,
                                NALU_HYPRE_Int cycle_type)
{
   nalu_hypre_AMSData *ams_data = (nalu_hypre_AMSData *) solver;
   ams_data -> cycle_type = cycle_type;
   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_AMSSetPrintLevel
 *
 * Control how much information is printed during the solution iterations.
 * The defaut values is 1 (print residual norm at each step).
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int nalu_hypre_AMSSetPrintLevel(void *solver,
                                 NALU_HYPRE_Int print_level)
{
   nalu_hypre_AMSData *ams_data = (nalu_hypre_AMSData *) solver;
   ams_data -> print_level = print_level;
   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_AMSSetSmoothingOptions
 *
 * Set relaxation parameters for A. Default values: 2, 1, 1.0, 1.0.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int nalu_hypre_AMSSetSmoothingOptions(void *solver,
                                       NALU_HYPRE_Int A_relax_type,
                                       NALU_HYPRE_Int A_relax_times,
                                       NALU_HYPRE_Real A_relax_weight,
                                       NALU_HYPRE_Real A_omega)
{
   nalu_hypre_AMSData *ams_data = (nalu_hypre_AMSData *) solver;
   ams_data -> A_relax_type = A_relax_type;
   ams_data -> A_relax_times = A_relax_times;
   ams_data -> A_relax_weight = A_relax_weight;
   ams_data -> A_omega = A_omega;
   return nalu_hypre_error_flag;
}
/*--------------------------------------------------------------------------
 * nalu_hypre_AMSSetChebySmoothingOptions
 *  AB: note: this could be added to the above,
 *      but I didn't want to change parameter list)
 * Set parameters for chebyshev smoother for A. Default values: 2,.3.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_AMSSetChebySmoothingOptions(void       *solver,
                                  NALU_HYPRE_Int   A_cheby_order,
                                  NALU_HYPRE_Real  A_cheby_fraction)
{
   nalu_hypre_AMSData *ams_data = (nalu_hypre_AMSData *) solver;
   ams_data -> A_cheby_order =  A_cheby_order;
   ams_data -> A_cheby_fraction =  A_cheby_fraction;

   return nalu_hypre_error_flag;
}
/*--------------------------------------------------------------------------
 * nalu_hypre_AMSSetAlphaAMGOptions
 *
 * Set AMG parameters for B_Pi. Default values: 10, 1, 3, 0.25, 0, 0.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int nalu_hypre_AMSSetAlphaAMGOptions(void *solver,
                                      NALU_HYPRE_Int B_Pi_coarsen_type,
                                      NALU_HYPRE_Int B_Pi_agg_levels,
                                      NALU_HYPRE_Int B_Pi_relax_type,
                                      NALU_HYPRE_Real B_Pi_theta,
                                      NALU_HYPRE_Int B_Pi_interp_type,
                                      NALU_HYPRE_Int B_Pi_Pmax)
{
   nalu_hypre_AMSData *ams_data = (nalu_hypre_AMSData *) solver;
   ams_data -> B_Pi_coarsen_type = B_Pi_coarsen_type;
   ams_data -> B_Pi_agg_levels = B_Pi_agg_levels;
   ams_data -> B_Pi_relax_type = B_Pi_relax_type;
   ams_data -> B_Pi_theta = B_Pi_theta;
   ams_data -> B_Pi_interp_type = B_Pi_interp_type;
   ams_data -> B_Pi_Pmax = B_Pi_Pmax;
   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_AMSSetAlphaAMGCoarseRelaxType
 *
 * Set the AMG coarsest level relaxation for B_Pi. Default value: 8.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int nalu_hypre_AMSSetAlphaAMGCoarseRelaxType(void *solver,
                                              NALU_HYPRE_Int B_Pi_coarse_relax_type)
{
   nalu_hypre_AMSData *ams_data =  (nalu_hypre_AMSData *)solver;
   ams_data -> B_Pi_coarse_relax_type = B_Pi_coarse_relax_type;
   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_AMSSetBetaAMGOptions
 *
 * Set AMG parameters for B_G. Default values: 10, 1, 3, 0.25, 0, 0.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int nalu_hypre_AMSSetBetaAMGOptions(void *solver,
                                     NALU_HYPRE_Int B_G_coarsen_type,
                                     NALU_HYPRE_Int B_G_agg_levels,
                                     NALU_HYPRE_Int B_G_relax_type,
                                     NALU_HYPRE_Real B_G_theta,
                                     NALU_HYPRE_Int B_G_interp_type,
                                     NALU_HYPRE_Int B_G_Pmax)
{
   nalu_hypre_AMSData *ams_data = (nalu_hypre_AMSData *) solver;
   ams_data -> B_G_coarsen_type = B_G_coarsen_type;
   ams_data -> B_G_agg_levels = B_G_agg_levels;
   ams_data -> B_G_relax_type = B_G_relax_type;
   ams_data -> B_G_theta = B_G_theta;
   ams_data -> B_G_interp_type = B_G_interp_type;
   ams_data -> B_G_Pmax = B_G_Pmax;
   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_AMSSetBetaAMGCoarseRelaxType
 *
 * Set the AMG coarsest level relaxation for B_G. Default value: 8.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int nalu_hypre_AMSSetBetaAMGCoarseRelaxType(void *solver,
                                             NALU_HYPRE_Int B_G_coarse_relax_type)
{
   nalu_hypre_AMSData *ams_data = (nalu_hypre_AMSData *) solver;
   ams_data -> B_G_coarse_relax_type = B_G_coarse_relax_type;
   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_AMSComputePi
 *
 * Construct the Pi interpolation matrix, which maps the space of vector
 * linear finite elements to the space of edge finite elements.
 *
 * The construction is based on the fact that Pi = [Pi_x, Pi_y, Pi_z],
 * where each block has the same sparsity structure as G, and the entries
 * can be computed from the vectors Gx, Gy, Gz.
 *--------------------------------------------------------------------------*/

#if defined(NALU_HYPRE_USING_GPU)
__global__ void
hypreGPUKernel_AMSComputePi_copy1(nalu_hypre_DeviceItem &item,
                                  NALU_HYPRE_Int  nnz,
                                  NALU_HYPRE_Int  dim,
                                  NALU_HYPRE_Int *j_in,
                                  NALU_HYPRE_Int *j_out)
{
   const NALU_HYPRE_Int i = nalu_hypre_gpu_get_grid_thread_id<1, 1>(item);

   if (i < nnz)
   {
      const NALU_HYPRE_Int j = dim * i;

      for (NALU_HYPRE_Int d = 0; d < dim; d++)
      {
         j_out[j + d] = dim * read_only_load(&j_in[i]) + d;
      }
   }
}

__global__ void
hypreGPUKernel_AMSComputePi_copy2(nalu_hypre_DeviceItem &item,
                                  NALU_HYPRE_Int   nrows,
                                  NALU_HYPRE_Int   dim,
                                  NALU_HYPRE_Int  *i_in,
                                  NALU_HYPRE_Real *data_in,
                                  NALU_HYPRE_Real *Gx_data,
                                  NALU_HYPRE_Real *Gy_data,
                                  NALU_HYPRE_Real *Gz_data,
                                  NALU_HYPRE_Real *data_out)
{
   const NALU_HYPRE_Int i = nalu_hypre_gpu_get_grid_warp_id<1, 1>(item);

   if (i >= nrows)
   {
      return;
   }

   const NALU_HYPRE_Int lane_id = nalu_hypre_gpu_get_lane_id<1>(item);
   NALU_HYPRE_Int j = 0, istart, iend;
   NALU_HYPRE_Real t, G[3], *Gdata[3];

   Gdata[0] = Gx_data;
   Gdata[1] = Gy_data;
   Gdata[2] = Gz_data;

   if (lane_id < 2)
   {
      j = read_only_load(i_in + i + lane_id);
   }

   istart = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, j, 0);
   iend   = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, j, 1);

   if (lane_id < dim)
   {
      t = read_only_load(Gdata[lane_id] + i);
   }

   for (NALU_HYPRE_Int d = 0; d < dim; d++)
   {
      G[d] = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, t, d);
   }

   for (j = istart + lane_id; j < iend; j += NALU_HYPRE_WARP_SIZE)
   {
      const NALU_HYPRE_Real v = data_in ? nalu_hypre_abs(read_only_load(&data_in[j])) * 0.5 : 1.0;
      const NALU_HYPRE_Int k = j * dim;

      for (NALU_HYPRE_Int d = 0; d < dim; d++)
      {
         data_out[k + d] = v * G[d];
      }
   }
}

#endif

NALU_HYPRE_Int nalu_hypre_AMSComputePi(nalu_hypre_ParCSRMatrix *A,
                             nalu_hypre_ParCSRMatrix *G,
                             nalu_hypre_ParVector *Gx,
                             nalu_hypre_ParVector *Gy,
                             nalu_hypre_ParVector *Gz,
                             NALU_HYPRE_Int dim,
                             nalu_hypre_ParCSRMatrix **Pi_ptr)
{
   nalu_hypre_ParCSRMatrix *Pi;

   /* Compute Pi = [Pi_x, Pi_y, Pi_z] */
   {
      NALU_HYPRE_Int i, j, d;

      NALU_HYPRE_Real *Gx_data, *Gy_data, *Gz_data;

      MPI_Comm comm = nalu_hypre_ParCSRMatrixComm(G);
      NALU_HYPRE_BigInt global_num_rows = nalu_hypre_ParCSRMatrixGlobalNumRows(G);
      NALU_HYPRE_BigInt global_num_cols = dim * nalu_hypre_ParCSRMatrixGlobalNumCols(G);
      NALU_HYPRE_BigInt *row_starts = nalu_hypre_ParCSRMatrixRowStarts(G);
      NALU_HYPRE_BigInt *col_starts;
      NALU_HYPRE_Int col_starts_size;
      NALU_HYPRE_Int num_cols_offd = dim * nalu_hypre_CSRMatrixNumCols(nalu_hypre_ParCSRMatrixOffd(G));
      NALU_HYPRE_Int num_nonzeros_diag = dim * nalu_hypre_CSRMatrixNumNonzeros(nalu_hypre_ParCSRMatrixDiag(G));
      NALU_HYPRE_Int num_nonzeros_offd = dim * nalu_hypre_CSRMatrixNumNonzeros(nalu_hypre_ParCSRMatrixOffd(G));
      NALU_HYPRE_BigInt *col_starts_G = nalu_hypre_ParCSRMatrixColStarts(G);

      col_starts_size = 2;
      col_starts = nalu_hypre_TAlloc(NALU_HYPRE_BigInt, col_starts_size, NALU_HYPRE_MEMORY_HOST);
      for (i = 0; i < col_starts_size; i++)
      {
         col_starts[i] = (NALU_HYPRE_BigInt)dim * col_starts_G[i];
      }

      Pi = nalu_hypre_ParCSRMatrixCreate(comm,
                                    global_num_rows,
                                    global_num_cols,
                                    row_starts,
                                    col_starts,
                                    num_cols_offd,
                                    num_nonzeros_diag,
                                    num_nonzeros_offd);

      nalu_hypre_ParCSRMatrixOwnsData(Pi) = 1;
      nalu_hypre_ParCSRMatrixInitialize(Pi);
      nalu_hypre_TFree(col_starts, NALU_HYPRE_MEMORY_HOST);

      Gx_data = nalu_hypre_VectorData(nalu_hypre_ParVectorLocalVector(Gx));
      if (dim >= 2)
      {
         Gy_data = nalu_hypre_VectorData(nalu_hypre_ParVectorLocalVector(Gy));
      }
      if (dim == 3)
      {
         Gz_data = nalu_hypre_VectorData(nalu_hypre_ParVectorLocalVector(Gz));
      }

#if defined(NALU_HYPRE_USING_GPU)
      NALU_HYPRE_ExecutionPolicy exec = nalu_hypre_GetExecPolicy2( nalu_hypre_ParCSRMatrixMemoryLocation(G),
                                                         nalu_hypre_ParCSRMatrixMemoryLocation(Pi) );
#endif

      /* Fill-in the diagonal part */
      {
         nalu_hypre_CSRMatrix *G_diag = nalu_hypre_ParCSRMatrixDiag(G);
         NALU_HYPRE_Int *G_diag_I = nalu_hypre_CSRMatrixI(G_diag);
         NALU_HYPRE_Int *G_diag_J = nalu_hypre_CSRMatrixJ(G_diag);
         NALU_HYPRE_Real *G_diag_data = nalu_hypre_CSRMatrixData(G_diag);

         NALU_HYPRE_Int G_diag_nrows = nalu_hypre_CSRMatrixNumRows(G_diag);
         NALU_HYPRE_Int G_diag_nnz = nalu_hypre_CSRMatrixNumNonzeros(G_diag);

         nalu_hypre_CSRMatrix *Pi_diag = nalu_hypre_ParCSRMatrixDiag(Pi);
         NALU_HYPRE_Int *Pi_diag_I = nalu_hypre_CSRMatrixI(Pi_diag);
         NALU_HYPRE_Int *Pi_diag_J = nalu_hypre_CSRMatrixJ(Pi_diag);
         NALU_HYPRE_Real *Pi_diag_data = nalu_hypre_CSRMatrixData(Pi_diag);

#if defined(NALU_HYPRE_USING_GPU)
         if (exec == NALU_HYPRE_EXEC_DEVICE)
         {
            hypreDevice_IntScalen( G_diag_I, G_diag_nrows + 1, Pi_diag_I, dim );

            dim3 bDim = nalu_hypre_GetDefaultDeviceBlockDimension();
            dim3 gDim = nalu_hypre_GetDefaultDeviceGridDimension(G_diag_nnz, "thread", bDim);

            NALU_HYPRE_GPU_LAUNCH( hypreGPUKernel_AMSComputePi_copy1, gDim, bDim,
                              G_diag_nnz, dim, G_diag_J, Pi_diag_J );

            gDim = nalu_hypre_GetDefaultDeviceGridDimension(G_diag_nrows, "warp", bDim);

            NALU_HYPRE_GPU_LAUNCH( hypreGPUKernel_AMSComputePi_copy2, gDim, bDim,
                              G_diag_nrows, dim, G_diag_I, G_diag_data, Gx_data, Gy_data, Gz_data,
                              Pi_diag_data );
         }
         else
#endif
         {
            for (i = 0; i < G_diag_nrows + 1; i++)
            {
               Pi_diag_I[i] = dim * G_diag_I[i];
            }

            for (i = 0; i < G_diag_nnz; i++)
               for (d = 0; d < dim; d++)
               {
                  Pi_diag_J[dim * i + d] = dim * G_diag_J[i] + d;
               }

            for (i = 0; i < G_diag_nrows; i++)
               for (j = G_diag_I[i]; j < G_diag_I[i + 1]; j++)
               {
                  *Pi_diag_data++ = nalu_hypre_abs(G_diag_data[j]) * 0.5 * Gx_data[i];
                  if (dim >= 2)
                  {
                     *Pi_diag_data++ = nalu_hypre_abs(G_diag_data[j]) * 0.5 * Gy_data[i];
                  }
                  if (dim == 3)
                  {
                     *Pi_diag_data++ = nalu_hypre_abs(G_diag_data[j]) * 0.5 * Gz_data[i];
                  }
               }
         }
      }

      /* Fill-in the off-diagonal part */
      {
         nalu_hypre_CSRMatrix *G_offd = nalu_hypre_ParCSRMatrixOffd(G);
         NALU_HYPRE_Int *G_offd_I = nalu_hypre_CSRMatrixI(G_offd);
         NALU_HYPRE_Int *G_offd_J = nalu_hypre_CSRMatrixJ(G_offd);
         NALU_HYPRE_Real *G_offd_data = nalu_hypre_CSRMatrixData(G_offd);

         NALU_HYPRE_Int G_offd_nrows = nalu_hypre_CSRMatrixNumRows(G_offd);
         NALU_HYPRE_Int G_offd_ncols = nalu_hypre_CSRMatrixNumCols(G_offd);
         NALU_HYPRE_Int G_offd_nnz = nalu_hypre_CSRMatrixNumNonzeros(G_offd);

         nalu_hypre_CSRMatrix *Pi_offd = nalu_hypre_ParCSRMatrixOffd(Pi);
         NALU_HYPRE_Int *Pi_offd_I = nalu_hypre_CSRMatrixI(Pi_offd);
         NALU_HYPRE_Int *Pi_offd_J = nalu_hypre_CSRMatrixJ(Pi_offd);
         NALU_HYPRE_Real *Pi_offd_data = nalu_hypre_CSRMatrixData(Pi_offd);

         NALU_HYPRE_BigInt *G_cmap = nalu_hypre_ParCSRMatrixColMapOffd(G);
         NALU_HYPRE_BigInt *Pi_cmap = nalu_hypre_ParCSRMatrixColMapOffd(Pi);

#if defined(NALU_HYPRE_USING_GPU)
         if (exec == NALU_HYPRE_EXEC_DEVICE)
         {
            if (G_offd_ncols)
            {
               hypreDevice_IntScalen( G_offd_I, G_offd_nrows + 1, Pi_offd_I, dim );
            }

            dim3 bDim = nalu_hypre_GetDefaultDeviceBlockDimension();
            dim3 gDim = nalu_hypre_GetDefaultDeviceGridDimension(G_offd_nnz, "thread", bDim);

            NALU_HYPRE_GPU_LAUNCH( hypreGPUKernel_AMSComputePi_copy1, gDim, bDim,
                              G_offd_nnz, dim, G_offd_J, Pi_offd_J );

            gDim = nalu_hypre_GetDefaultDeviceGridDimension(G_offd_nrows, "warp", bDim);

            NALU_HYPRE_GPU_LAUNCH( hypreGPUKernel_AMSComputePi_copy2, gDim, bDim,
                              G_offd_nrows, dim, G_offd_I, G_offd_data, Gx_data, Gy_data, Gz_data,
                              Pi_offd_data );
         }
         else
#endif
         {
            if (G_offd_ncols)
               for (i = 0; i < G_offd_nrows + 1; i++)
               {
                  Pi_offd_I[i] = dim * G_offd_I[i];
               }

            for (i = 0; i < G_offd_nnz; i++)
               for (d = 0; d < dim; d++)
               {
                  Pi_offd_J[dim * i + d] = dim * G_offd_J[i] + d;
               }

            for (i = 0; i < G_offd_nrows; i++)
               for (j = G_offd_I[i]; j < G_offd_I[i + 1]; j++)
               {
                  *Pi_offd_data++ = nalu_hypre_abs(G_offd_data[j]) * 0.5 * Gx_data[i];
                  if (dim >= 2)
                  {
                     *Pi_offd_data++ = nalu_hypre_abs(G_offd_data[j]) * 0.5 * Gy_data[i];
                  }
                  if (dim == 3)
                  {
                     *Pi_offd_data++ = nalu_hypre_abs(G_offd_data[j]) * 0.5 * Gz_data[i];
                  }
               }
         }

         for (i = 0; i < G_offd_ncols; i++)
            for (d = 0; d < dim; d++)
            {
               Pi_cmap[dim * i + d] = (NALU_HYPRE_BigInt)dim * G_cmap[i] + (NALU_HYPRE_BigInt)d;
            }
      }
   }

   *Pi_ptr = Pi;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_AMSComputePixyz
 *
 * Construct the components Pix, Piy, Piz of the interpolation matrix Pi,
 * which maps the space of vector linear finite elements to the space of
 * edge finite elements.
 *
 * The construction is based on the fact that each component has the same
 * sparsity structure as G, and the entries can be computed from the vectors
 * Gx, Gy, Gz.
 *--------------------------------------------------------------------------*/

#if defined(NALU_HYPRE_USING_GPU)
__global__ void
hypreGPUKernel_AMSComputePixyz_copy(nalu_hypre_DeviceItem &item,
                                    NALU_HYPRE_Int   nrows,
                                    NALU_HYPRE_Int   dim,
                                    NALU_HYPRE_Int  *i_in,
                                    NALU_HYPRE_Real *data_in,
                                    NALU_HYPRE_Real *Gx_data,
                                    NALU_HYPRE_Real *Gy_data,
                                    NALU_HYPRE_Real *Gz_data,
                                    NALU_HYPRE_Real *data_x_out,
                                    NALU_HYPRE_Real *data_y_out,
                                    NALU_HYPRE_Real *data_z_out )
{
   const NALU_HYPRE_Int i = nalu_hypre_gpu_get_grid_warp_id<1, 1>(item);

   if (i >= nrows)
   {
      return;
   }

   const NALU_HYPRE_Int lane_id = nalu_hypre_gpu_get_lane_id<1>(item);
   NALU_HYPRE_Int j = 0, istart, iend;
   NALU_HYPRE_Real t, G[3], *Gdata[3], *Odata[3];

   Gdata[0] = Gx_data;
   Gdata[1] = Gy_data;
   Gdata[2] = Gz_data;

   Odata[0] = data_x_out;
   Odata[1] = data_y_out;
   Odata[2] = data_z_out;

   if (lane_id < 2)
   {
      j = read_only_load(i_in + i + lane_id);
   }

   istart = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, j, 0);
   iend   = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, j, 1);

   if (lane_id < dim)
   {
      t = read_only_load(Gdata[lane_id] + i);
   }

   for (NALU_HYPRE_Int d = 0; d < dim; d++)
   {
      G[d] = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, t, d);
   }

   for (j = istart + lane_id; j < iend; j += NALU_HYPRE_WARP_SIZE)
   {
      const NALU_HYPRE_Real v = data_in ? nalu_hypre_abs(read_only_load(&data_in[j])) * 0.5 : 1.0;

      for (NALU_HYPRE_Int d = 0; d < dim; d++)
      {
         Odata[d][j] = v * G[d];
      }
   }
}
#endif

NALU_HYPRE_Int nalu_hypre_AMSComputePixyz(nalu_hypre_ParCSRMatrix *A,
                                nalu_hypre_ParCSRMatrix *G,
                                nalu_hypre_ParVector *Gx,
                                nalu_hypre_ParVector *Gy,
                                nalu_hypre_ParVector *Gz,
                                NALU_HYPRE_Int dim,
                                nalu_hypre_ParCSRMatrix **Pix_ptr,
                                nalu_hypre_ParCSRMatrix **Piy_ptr,
                                nalu_hypre_ParCSRMatrix **Piz_ptr)
{
   nalu_hypre_ParCSRMatrix *Pix, *Piy, *Piz;

#if defined(NALU_HYPRE_USING_GPU)
   NALU_HYPRE_ExecutionPolicy exec = nalu_hypre_GetExecPolicy1( nalu_hypre_ParCSRMatrixMemoryLocation(G) );
#endif

   /* Compute Pix, Piy, Piz  */
   {
      NALU_HYPRE_Int i, j;

      NALU_HYPRE_Real *Gx_data, *Gy_data, *Gz_data;

      MPI_Comm comm = nalu_hypre_ParCSRMatrixComm(G);
      NALU_HYPRE_BigInt global_num_rows = nalu_hypre_ParCSRMatrixGlobalNumRows(G);
      NALU_HYPRE_BigInt global_num_cols = nalu_hypre_ParCSRMatrixGlobalNumCols(G);
      NALU_HYPRE_BigInt *row_starts = nalu_hypre_ParCSRMatrixRowStarts(G);
      NALU_HYPRE_BigInt *col_starts = nalu_hypre_ParCSRMatrixColStarts(G);
      NALU_HYPRE_Int num_cols_offd = nalu_hypre_CSRMatrixNumCols(nalu_hypre_ParCSRMatrixOffd(G));
      NALU_HYPRE_Int num_nonzeros_diag = nalu_hypre_CSRMatrixNumNonzeros(nalu_hypre_ParCSRMatrixDiag(G));
      NALU_HYPRE_Int num_nonzeros_offd = nalu_hypre_CSRMatrixNumNonzeros(nalu_hypre_ParCSRMatrixOffd(G));

      Pix = nalu_hypre_ParCSRMatrixCreate(comm,
                                     global_num_rows,
                                     global_num_cols,
                                     row_starts,
                                     col_starts,
                                     num_cols_offd,
                                     num_nonzeros_diag,
                                     num_nonzeros_offd);
      nalu_hypre_ParCSRMatrixOwnsData(Pix) = 1;
      nalu_hypre_ParCSRMatrixInitialize(Pix);

      if (dim >= 2)
      {
         Piy = nalu_hypre_ParCSRMatrixCreate(comm,
                                        global_num_rows,
                                        global_num_cols,
                                        row_starts,
                                        col_starts,
                                        num_cols_offd,
                                        num_nonzeros_diag,
                                        num_nonzeros_offd);
         nalu_hypre_ParCSRMatrixOwnsData(Piy) = 1;
         nalu_hypre_ParCSRMatrixInitialize(Piy);
      }

      if (dim == 3)
      {
         Piz = nalu_hypre_ParCSRMatrixCreate(comm,
                                        global_num_rows,
                                        global_num_cols,
                                        row_starts,
                                        col_starts,
                                        num_cols_offd,
                                        num_nonzeros_diag,
                                        num_nonzeros_offd);
         nalu_hypre_ParCSRMatrixOwnsData(Piz) = 1;
         nalu_hypre_ParCSRMatrixInitialize(Piz);
      }

      Gx_data = nalu_hypre_VectorData(nalu_hypre_ParVectorLocalVector(Gx));
      if (dim >= 2)
      {
         Gy_data = nalu_hypre_VectorData(nalu_hypre_ParVectorLocalVector(Gy));
      }
      if (dim == 3)
      {
         Gz_data = nalu_hypre_VectorData(nalu_hypre_ParVectorLocalVector(Gz));
      }

      /* Fill-in the diagonal part */
      if (dim == 3)
      {
         nalu_hypre_CSRMatrix *G_diag = nalu_hypre_ParCSRMatrixDiag(G);
         NALU_HYPRE_Int *G_diag_I = nalu_hypre_CSRMatrixI(G_diag);
         NALU_HYPRE_Int *G_diag_J = nalu_hypre_CSRMatrixJ(G_diag);
         NALU_HYPRE_Real *G_diag_data = nalu_hypre_CSRMatrixData(G_diag);

         NALU_HYPRE_Int G_diag_nrows = nalu_hypre_CSRMatrixNumRows(G_diag);
         NALU_HYPRE_Int G_diag_nnz = nalu_hypre_CSRMatrixNumNonzeros(G_diag);

         nalu_hypre_CSRMatrix *Pix_diag = nalu_hypre_ParCSRMatrixDiag(Pix);
         NALU_HYPRE_Int *Pix_diag_I = nalu_hypre_CSRMatrixI(Pix_diag);
         NALU_HYPRE_Int *Pix_diag_J = nalu_hypre_CSRMatrixJ(Pix_diag);
         NALU_HYPRE_Real *Pix_diag_data = nalu_hypre_CSRMatrixData(Pix_diag);

         nalu_hypre_CSRMatrix *Piy_diag = nalu_hypre_ParCSRMatrixDiag(Piy);
         NALU_HYPRE_Int *Piy_diag_I = nalu_hypre_CSRMatrixI(Piy_diag);
         NALU_HYPRE_Int *Piy_diag_J = nalu_hypre_CSRMatrixJ(Piy_diag);
         NALU_HYPRE_Real *Piy_diag_data = nalu_hypre_CSRMatrixData(Piy_diag);

         nalu_hypre_CSRMatrix *Piz_diag = nalu_hypre_ParCSRMatrixDiag(Piz);
         NALU_HYPRE_Int *Piz_diag_I = nalu_hypre_CSRMatrixI(Piz_diag);
         NALU_HYPRE_Int *Piz_diag_J = nalu_hypre_CSRMatrixJ(Piz_diag);
         NALU_HYPRE_Real *Piz_diag_data = nalu_hypre_CSRMatrixData(Piz_diag);

#if defined(NALU_HYPRE_USING_GPU)
         if (exec == NALU_HYPRE_EXEC_DEVICE)
         {
#if defined(NALU_HYPRE_USING_SYCL)
            NALU_HYPRE_ONEDPL_CALL( std::copy_n,
                               oneapi::dpl::make_zip_iterator(G_diag_I, G_diag_I, G_diag_I),
                               G_diag_nrows + 1,
                               oneapi::dpl::make_zip_iterator(Pix_diag_I, Piy_diag_I, Piz_diag_I) );

            NALU_HYPRE_ONEDPL_CALL( std::copy_n,
                               oneapi::dpl::make_zip_iterator(G_diag_J, G_diag_J, G_diag_J),
                               G_diag_nnz,
                               oneapi::dpl::make_zip_iterator(Pix_diag_J, Piy_diag_J, Piz_diag_J) );
#else
            NALU_HYPRE_THRUST_CALL( copy_n,
                               thrust::make_zip_iterator(thrust::make_tuple(G_diag_I, G_diag_I, G_diag_I)),
                               G_diag_nrows + 1,
                               thrust::make_zip_iterator(thrust::make_tuple(Pix_diag_I, Piy_diag_I, Piz_diag_I)) );

            NALU_HYPRE_THRUST_CALL( copy_n,
                               thrust::make_zip_iterator(thrust::make_tuple(G_diag_J, G_diag_J, G_diag_J)),
                               G_diag_nnz,
                               thrust::make_zip_iterator(thrust::make_tuple(Pix_diag_J, Piy_diag_J, Piz_diag_J)) );
#endif

            dim3 bDim = nalu_hypre_GetDefaultDeviceBlockDimension();
            dim3 gDim = nalu_hypre_GetDefaultDeviceGridDimension(G_diag_nrows, "warp", bDim);

            NALU_HYPRE_GPU_LAUNCH( hypreGPUKernel_AMSComputePixyz_copy, gDim, bDim,
                              G_diag_nrows, dim, G_diag_I, G_diag_data, Gx_data, Gy_data, Gz_data,
                              Pix_diag_data, Piy_diag_data, Piz_diag_data );
         }
         else
#endif
         {
            for (i = 0; i < G_diag_nrows + 1; i++)
            {
               Pix_diag_I[i] = G_diag_I[i];
               Piy_diag_I[i] = G_diag_I[i];
               Piz_diag_I[i] = G_diag_I[i];
            }

            for (i = 0; i < G_diag_nnz; i++)
            {
               Pix_diag_J[i] = G_diag_J[i];
               Piy_diag_J[i] = G_diag_J[i];
               Piz_diag_J[i] = G_diag_J[i];
            }

            for (i = 0; i < G_diag_nrows; i++)
               for (j = G_diag_I[i]; j < G_diag_I[i + 1]; j++)
               {
                  *Pix_diag_data++ = nalu_hypre_abs(G_diag_data[j]) * 0.5 * Gx_data[i];
                  *Piy_diag_data++ = nalu_hypre_abs(G_diag_data[j]) * 0.5 * Gy_data[i];
                  *Piz_diag_data++ = nalu_hypre_abs(G_diag_data[j]) * 0.5 * Gz_data[i];
               }
         }
      }
      else if (dim == 2)
      {
         nalu_hypre_CSRMatrix *G_diag = nalu_hypre_ParCSRMatrixDiag(G);
         NALU_HYPRE_Int *G_diag_I = nalu_hypre_CSRMatrixI(G_diag);
         NALU_HYPRE_Int *G_diag_J = nalu_hypre_CSRMatrixJ(G_diag);
         NALU_HYPRE_Real *G_diag_data = nalu_hypre_CSRMatrixData(G_diag);

         NALU_HYPRE_Int G_diag_nrows = nalu_hypre_CSRMatrixNumRows(G_diag);
         NALU_HYPRE_Int G_diag_nnz = nalu_hypre_CSRMatrixNumNonzeros(G_diag);

         nalu_hypre_CSRMatrix *Pix_diag = nalu_hypre_ParCSRMatrixDiag(Pix);
         NALU_HYPRE_Int *Pix_diag_I = nalu_hypre_CSRMatrixI(Pix_diag);
         NALU_HYPRE_Int *Pix_diag_J = nalu_hypre_CSRMatrixJ(Pix_diag);
         NALU_HYPRE_Real *Pix_diag_data = nalu_hypre_CSRMatrixData(Pix_diag);

         nalu_hypre_CSRMatrix *Piy_diag = nalu_hypre_ParCSRMatrixDiag(Piy);
         NALU_HYPRE_Int *Piy_diag_I = nalu_hypre_CSRMatrixI(Piy_diag);
         NALU_HYPRE_Int *Piy_diag_J = nalu_hypre_CSRMatrixJ(Piy_diag);
         NALU_HYPRE_Real *Piy_diag_data = nalu_hypre_CSRMatrixData(Piy_diag);

#if defined(NALU_HYPRE_USING_GPU)
         if (exec == NALU_HYPRE_EXEC_DEVICE)
         {
#if defined(NALU_HYPRE_USING_SYCL)
            NALU_HYPRE_ONEDPL_CALL( std::copy_n,
                               oneapi::dpl::make_zip_iterator(G_diag_I, G_diag_I),
                               G_diag_nrows + 1,
                               oneapi::dpl::make_zip_iterator(Pix_diag_I, Piy_diag_I) );

            NALU_HYPRE_ONEDPL_CALL( std::copy_n,
                               oneapi::dpl::make_zip_iterator(G_diag_J, G_diag_J),
                               G_diag_nnz,
                               oneapi::dpl::make_zip_iterator(Pix_diag_J, Piy_diag_J) );
#else
            NALU_HYPRE_THRUST_CALL( copy_n,
                               thrust::make_zip_iterator(thrust::make_tuple(G_diag_I, G_diag_I)),
                               G_diag_nrows + 1,
                               thrust::make_zip_iterator(thrust::make_tuple(Pix_diag_I, Piy_diag_I)) );

            NALU_HYPRE_THRUST_CALL( copy_n,
                               thrust::make_zip_iterator(thrust::make_tuple(G_diag_J, G_diag_J)),
                               G_diag_nnz,
                               thrust::make_zip_iterator(thrust::make_tuple(Pix_diag_J, Piy_diag_J)) );
#endif

            dim3 bDim = nalu_hypre_GetDefaultDeviceBlockDimension();
            dim3 gDim = nalu_hypre_GetDefaultDeviceGridDimension(G_diag_nrows, "warp", bDim);

            NALU_HYPRE_GPU_LAUNCH( hypreGPUKernel_AMSComputePixyz_copy, gDim, bDim,
                              G_diag_nrows, dim, G_diag_I, G_diag_data, Gx_data, Gy_data, NULL,
                              Pix_diag_data, Piy_diag_data, NULL );
         }
         else
#endif
         {
            for (i = 0; i < G_diag_nrows + 1; i++)
            {
               Pix_diag_I[i] = G_diag_I[i];
               Piy_diag_I[i] = G_diag_I[i];
            }

            for (i = 0; i < G_diag_nnz; i++)
            {
               Pix_diag_J[i] = G_diag_J[i];
               Piy_diag_J[i] = G_diag_J[i];
            }

            for (i = 0; i < G_diag_nrows; i++)
               for (j = G_diag_I[i]; j < G_diag_I[i + 1]; j++)
               {
                  *Pix_diag_data++ = nalu_hypre_abs(G_diag_data[j]) * 0.5 * Gx_data[i];
                  *Piy_diag_data++ = nalu_hypre_abs(G_diag_data[j]) * 0.5 * Gy_data[i];
               }
         }
      }
      else
      {
         nalu_hypre_CSRMatrix *G_diag = nalu_hypre_ParCSRMatrixDiag(G);
         NALU_HYPRE_Int *G_diag_I = nalu_hypre_CSRMatrixI(G_diag);
         NALU_HYPRE_Int *G_diag_J = nalu_hypre_CSRMatrixJ(G_diag);
         NALU_HYPRE_Real *G_diag_data = nalu_hypre_CSRMatrixData(G_diag);

         NALU_HYPRE_Int G_diag_nrows = nalu_hypre_CSRMatrixNumRows(G_diag);
         NALU_HYPRE_Int G_diag_nnz = nalu_hypre_CSRMatrixNumNonzeros(G_diag);

         nalu_hypre_CSRMatrix *Pix_diag = nalu_hypre_ParCSRMatrixDiag(Pix);
         NALU_HYPRE_Int *Pix_diag_I = nalu_hypre_CSRMatrixI(Pix_diag);
         NALU_HYPRE_Int *Pix_diag_J = nalu_hypre_CSRMatrixJ(Pix_diag);
         NALU_HYPRE_Real *Pix_diag_data = nalu_hypre_CSRMatrixData(Pix_diag);

#if defined(NALU_HYPRE_USING_GPU)
         if (exec == NALU_HYPRE_EXEC_DEVICE)
         {
#if defined(NALU_HYPRE_USING_SYCL)
            NALU_HYPRE_ONEDPL_CALL( std::copy_n,
                               G_diag_I,
                               G_diag_nrows + 1,
                               Pix_diag_I );

            NALU_HYPRE_ONEDPL_CALL( std::copy_n,
                               G_diag_J,
                               G_diag_nnz,
                               Pix_diag_J );
#else
            NALU_HYPRE_THRUST_CALL( copy_n,
                               G_diag_I,
                               G_diag_nrows + 1,
                               Pix_diag_I );

            NALU_HYPRE_THRUST_CALL( copy_n,
                               G_diag_J,
                               G_diag_nnz,
                               Pix_diag_J );
#endif

            dim3 bDim = nalu_hypre_GetDefaultDeviceBlockDimension();
            dim3 gDim = nalu_hypre_GetDefaultDeviceGridDimension(G_diag_nrows, "warp", bDim);

            NALU_HYPRE_GPU_LAUNCH( hypreGPUKernel_AMSComputePixyz_copy, gDim, bDim,
                              G_diag_nrows, dim, G_diag_I, G_diag_data, Gx_data, NULL, NULL,
                              Pix_diag_data, NULL, NULL );
         }
         else
#endif
         {
            for (i = 0; i < G_diag_nrows + 1; i++)
            {
               Pix_diag_I[i] = G_diag_I[i];
            }

            for (i = 0; i < G_diag_nnz; i++)
            {
               Pix_diag_J[i] = G_diag_J[i];
            }

            for (i = 0; i < G_diag_nrows; i++)
               for (j = G_diag_I[i]; j < G_diag_I[i + 1]; j++)
               {
                  *Pix_diag_data++ = nalu_hypre_abs(G_diag_data[j]) * 0.5 * Gx_data[i];
               }
         }
      }


      /* Fill-in the off-diagonal part */
      if (dim == 3)
      {
         nalu_hypre_CSRMatrix *G_offd = nalu_hypre_ParCSRMatrixOffd(G);
         NALU_HYPRE_Int *G_offd_I = nalu_hypre_CSRMatrixI(G_offd);
         NALU_HYPRE_Int *G_offd_J = nalu_hypre_CSRMatrixJ(G_offd);
         NALU_HYPRE_Real *G_offd_data = nalu_hypre_CSRMatrixData(G_offd);

         NALU_HYPRE_Int G_offd_nrows = nalu_hypre_CSRMatrixNumRows(G_offd);
         NALU_HYPRE_Int G_offd_ncols = nalu_hypre_CSRMatrixNumCols(G_offd);
         NALU_HYPRE_Int G_offd_nnz = nalu_hypre_CSRMatrixNumNonzeros(G_offd);

         nalu_hypre_CSRMatrix *Pix_offd = nalu_hypre_ParCSRMatrixOffd(Pix);
         NALU_HYPRE_Int *Pix_offd_I = nalu_hypre_CSRMatrixI(Pix_offd);
         NALU_HYPRE_Int *Pix_offd_J = nalu_hypre_CSRMatrixJ(Pix_offd);
         NALU_HYPRE_Real *Pix_offd_data = nalu_hypre_CSRMatrixData(Pix_offd);

         nalu_hypre_CSRMatrix *Piy_offd = nalu_hypre_ParCSRMatrixOffd(Piy);
         NALU_HYPRE_Int *Piy_offd_I = nalu_hypre_CSRMatrixI(Piy_offd);
         NALU_HYPRE_Int *Piy_offd_J = nalu_hypre_CSRMatrixJ(Piy_offd);
         NALU_HYPRE_Real *Piy_offd_data = nalu_hypre_CSRMatrixData(Piy_offd);

         nalu_hypre_CSRMatrix *Piz_offd = nalu_hypre_ParCSRMatrixOffd(Piz);
         NALU_HYPRE_Int *Piz_offd_I = nalu_hypre_CSRMatrixI(Piz_offd);
         NALU_HYPRE_Int *Piz_offd_J = nalu_hypre_CSRMatrixJ(Piz_offd);
         NALU_HYPRE_Real *Piz_offd_data = nalu_hypre_CSRMatrixData(Piz_offd);

         NALU_HYPRE_BigInt *G_cmap = nalu_hypre_ParCSRMatrixColMapOffd(G);
         NALU_HYPRE_BigInt *Pix_cmap = nalu_hypre_ParCSRMatrixColMapOffd(Pix);
         NALU_HYPRE_BigInt *Piy_cmap = nalu_hypre_ParCSRMatrixColMapOffd(Piy);
         NALU_HYPRE_BigInt *Piz_cmap = nalu_hypre_ParCSRMatrixColMapOffd(Piz);

#if defined(NALU_HYPRE_USING_GPU)
         if (exec == NALU_HYPRE_EXEC_DEVICE)
         {
#if defined(NALU_HYPRE_USING_SYCL)
            if (G_offd_ncols)
            {
               NALU_HYPRE_ONEDPL_CALL( std::copy_n,
                                  oneapi::dpl::make_zip_iterator(G_offd_I, G_offd_I, G_offd_I),
                                  G_offd_nrows + 1,
                                  oneapi::dpl::make_zip_iterator(Pix_offd_I, Piy_offd_I, Piz_offd_I) );
            }

            NALU_HYPRE_ONEDPL_CALL( std::copy_n,
                               oneapi::dpl::make_zip_iterator(G_offd_J, G_offd_J, G_offd_J),
                               G_offd_nnz,
                               oneapi::dpl::make_zip_iterator(Pix_offd_J, Piy_offd_J, Piz_offd_J) );
#else
            if (G_offd_ncols)
            {
               NALU_HYPRE_THRUST_CALL( copy_n,
                                  thrust::make_zip_iterator(thrust::make_tuple(G_offd_I, G_offd_I, G_offd_I)),
                                  G_offd_nrows + 1,
                                  thrust::make_zip_iterator(thrust::make_tuple(Pix_offd_I, Piy_offd_I, Piz_offd_I)) );
            }

            NALU_HYPRE_THRUST_CALL( copy_n,
                               thrust::make_zip_iterator(thrust::make_tuple(G_offd_J, G_offd_J, G_offd_J)),
                               G_offd_nnz,
                               thrust::make_zip_iterator(thrust::make_tuple(Pix_offd_J, Piy_offd_J, Piz_offd_J)) );
#endif

            dim3 bDim = nalu_hypre_GetDefaultDeviceBlockDimension();
            dim3 gDim = nalu_hypre_GetDefaultDeviceGridDimension(G_offd_nrows, "warp", bDim);

            NALU_HYPRE_GPU_LAUNCH( hypreGPUKernel_AMSComputePixyz_copy, gDim, bDim,
                              G_offd_nrows, dim, G_offd_I, G_offd_data, Gx_data, Gy_data, Gz_data,
                              Pix_offd_data, Piy_offd_data, Piz_offd_data );
         }
         else
#endif
         {
            if (G_offd_ncols)
               for (i = 0; i < G_offd_nrows + 1; i++)
               {
                  Pix_offd_I[i] = G_offd_I[i];
                  Piy_offd_I[i] = G_offd_I[i];
                  Piz_offd_I[i] = G_offd_I[i];
               }

            for (i = 0; i < G_offd_nnz; i++)
            {
               Pix_offd_J[i] = G_offd_J[i];
               Piy_offd_J[i] = G_offd_J[i];
               Piz_offd_J[i] = G_offd_J[i];
            }

            for (i = 0; i < G_offd_nrows; i++)
               for (j = G_offd_I[i]; j < G_offd_I[i + 1]; j++)
               {
                  *Pix_offd_data++ = nalu_hypre_abs(G_offd_data[j]) * 0.5 * Gx_data[i];
                  *Piy_offd_data++ = nalu_hypre_abs(G_offd_data[j]) * 0.5 * Gy_data[i];
                  *Piz_offd_data++ = nalu_hypre_abs(G_offd_data[j]) * 0.5 * Gz_data[i];
               }
         }

         for (i = 0; i < G_offd_ncols; i++)
         {
            Pix_cmap[i] = G_cmap[i];
            Piy_cmap[i] = G_cmap[i];
            Piz_cmap[i] = G_cmap[i];
         }
      }
      else if (dim == 2)
      {
         nalu_hypre_CSRMatrix *G_offd = nalu_hypre_ParCSRMatrixOffd(G);
         NALU_HYPRE_Int *G_offd_I = nalu_hypre_CSRMatrixI(G_offd);
         NALU_HYPRE_Int *G_offd_J = nalu_hypre_CSRMatrixJ(G_offd);
         NALU_HYPRE_Real *G_offd_data = nalu_hypre_CSRMatrixData(G_offd);

         NALU_HYPRE_Int G_offd_nrows = nalu_hypre_CSRMatrixNumRows(G_offd);
         NALU_HYPRE_Int G_offd_ncols = nalu_hypre_CSRMatrixNumCols(G_offd);
         NALU_HYPRE_Int G_offd_nnz = nalu_hypre_CSRMatrixNumNonzeros(G_offd);

         nalu_hypre_CSRMatrix *Pix_offd = nalu_hypre_ParCSRMatrixOffd(Pix);
         NALU_HYPRE_Int *Pix_offd_I = nalu_hypre_CSRMatrixI(Pix_offd);
         NALU_HYPRE_Int *Pix_offd_J = nalu_hypre_CSRMatrixJ(Pix_offd);
         NALU_HYPRE_Real *Pix_offd_data = nalu_hypre_CSRMatrixData(Pix_offd);

         nalu_hypre_CSRMatrix *Piy_offd = nalu_hypre_ParCSRMatrixOffd(Piy);
         NALU_HYPRE_Int *Piy_offd_I = nalu_hypre_CSRMatrixI(Piy_offd);
         NALU_HYPRE_Int *Piy_offd_J = nalu_hypre_CSRMatrixJ(Piy_offd);
         NALU_HYPRE_Real *Piy_offd_data = nalu_hypre_CSRMatrixData(Piy_offd);

         NALU_HYPRE_BigInt *G_cmap = nalu_hypre_ParCSRMatrixColMapOffd(G);
         NALU_HYPRE_BigInt *Pix_cmap = nalu_hypre_ParCSRMatrixColMapOffd(Pix);
         NALU_HYPRE_BigInt *Piy_cmap = nalu_hypre_ParCSRMatrixColMapOffd(Piy);

#if defined(NALU_HYPRE_USING_GPU)
         if (exec == NALU_HYPRE_EXEC_DEVICE)
         {
#if defined(NALU_HYPRE_USING_SYCL)
            if (G_offd_ncols)
            {
               NALU_HYPRE_ONEDPL_CALL( std::copy_n,
                                  oneapi::dpl::make_zip_iterator(G_offd_I, G_offd_I),
                                  G_offd_nrows + 1,
                                  oneapi::dpl::make_zip_iterator(Pix_offd_I, Piy_offd_I) );
            }

            NALU_HYPRE_ONEDPL_CALL( std::copy_n,
                               oneapi::dpl::make_zip_iterator(G_offd_J, G_offd_J),
                               G_offd_nnz,
                               oneapi::dpl::make_zip_iterator(Pix_offd_J, Piy_offd_J) );
#else
            if (G_offd_ncols)
            {
               NALU_HYPRE_THRUST_CALL( copy_n,
                                  thrust::make_zip_iterator(thrust::make_tuple(G_offd_I, G_offd_I)),
                                  G_offd_nrows + 1,
                                  thrust::make_zip_iterator(thrust::make_tuple(Pix_offd_I, Piy_offd_I)) );
            }

            NALU_HYPRE_THRUST_CALL( copy_n,
                               thrust::make_zip_iterator(thrust::make_tuple(G_offd_J, G_offd_J)),
                               G_offd_nnz,
                               thrust::make_zip_iterator(thrust::make_tuple(Pix_offd_J, Piy_offd_J)) );
#endif

            dim3 bDim = nalu_hypre_GetDefaultDeviceBlockDimension();
            dim3 gDim = nalu_hypre_GetDefaultDeviceGridDimension(G_offd_nrows, "warp", bDim);

            NALU_HYPRE_GPU_LAUNCH( hypreGPUKernel_AMSComputePixyz_copy, gDim, bDim,
                              G_offd_nrows, dim, G_offd_I, G_offd_data, Gx_data, Gy_data, NULL,
                              Pix_offd_data, Piy_offd_data, NULL );
         }
         else
#endif
         {
            if (G_offd_ncols)
               for (i = 0; i < G_offd_nrows + 1; i++)
               {
                  Pix_offd_I[i] = G_offd_I[i];
                  Piy_offd_I[i] = G_offd_I[i];
               }

            for (i = 0; i < G_offd_nnz; i++)
            {
               Pix_offd_J[i] = G_offd_J[i];
               Piy_offd_J[i] = G_offd_J[i];
            }

            for (i = 0; i < G_offd_nrows; i++)
               for (j = G_offd_I[i]; j < G_offd_I[i + 1]; j++)
               {
                  *Pix_offd_data++ = nalu_hypre_abs(G_offd_data[j]) * 0.5 * Gx_data[i];
                  *Piy_offd_data++ = nalu_hypre_abs(G_offd_data[j]) * 0.5 * Gy_data[i];
               }
         }

         for (i = 0; i < G_offd_ncols; i++)
         {
            Pix_cmap[i] = G_cmap[i];
            Piy_cmap[i] = G_cmap[i];
         }
      }
      else
      {
         nalu_hypre_CSRMatrix *G_offd = nalu_hypre_ParCSRMatrixOffd(G);
         NALU_HYPRE_Int *G_offd_I = nalu_hypre_CSRMatrixI(G_offd);
         NALU_HYPRE_Int *G_offd_J = nalu_hypre_CSRMatrixJ(G_offd);
         NALU_HYPRE_Real *G_offd_data = nalu_hypre_CSRMatrixData(G_offd);

         NALU_HYPRE_Int G_offd_nrows = nalu_hypre_CSRMatrixNumRows(G_offd);
         NALU_HYPRE_Int G_offd_ncols = nalu_hypre_CSRMatrixNumCols(G_offd);
         NALU_HYPRE_Int G_offd_nnz = nalu_hypre_CSRMatrixNumNonzeros(G_offd);

         nalu_hypre_CSRMatrix *Pix_offd = nalu_hypre_ParCSRMatrixOffd(Pix);
         NALU_HYPRE_Int *Pix_offd_I = nalu_hypre_CSRMatrixI(Pix_offd);
         NALU_HYPRE_Int *Pix_offd_J = nalu_hypre_CSRMatrixJ(Pix_offd);
         NALU_HYPRE_Real *Pix_offd_data = nalu_hypre_CSRMatrixData(Pix_offd);

         NALU_HYPRE_BigInt *G_cmap = nalu_hypre_ParCSRMatrixColMapOffd(G);
         NALU_HYPRE_BigInt *Pix_cmap = nalu_hypre_ParCSRMatrixColMapOffd(Pix);

#if defined(NALU_HYPRE_USING_GPU)
         if (exec == NALU_HYPRE_EXEC_DEVICE)
         {
#if defined(NALU_HYPRE_USING_SYCL)
            if (G_offd_ncols)
            {
               NALU_HYPRE_ONEDPL_CALL( std::copy_n,
                                  G_offd_I,
                                  G_offd_nrows + 1,
                                  Pix_offd_I );
            }

            NALU_HYPRE_ONEDPL_CALL( std::copy_n,
                               G_offd_J,
                               G_offd_nnz,
                               Pix_offd_J );
#else
            if (G_offd_ncols)
            {
               NALU_HYPRE_THRUST_CALL( copy_n,
                                  G_offd_I,
                                  G_offd_nrows + 1,
                                  Pix_offd_I );
            }

            NALU_HYPRE_THRUST_CALL( copy_n,
                               G_offd_J,
                               G_offd_nnz,
                               Pix_offd_J );
#endif

            dim3 bDim = nalu_hypre_GetDefaultDeviceBlockDimension();
            dim3 gDim = nalu_hypre_GetDefaultDeviceGridDimension(G_offd_nrows, "warp", bDim);

            NALU_HYPRE_GPU_LAUNCH( hypreGPUKernel_AMSComputePixyz_copy, gDim, bDim,
                              G_offd_nrows, dim, G_offd_I, G_offd_data, Gx_data, NULL, NULL,
                              Pix_offd_data, NULL, NULL );
         }
         else
#endif
         {
            if (G_offd_ncols)
               for (i = 0; i < G_offd_nrows + 1; i++)
               {
                  Pix_offd_I[i] = G_offd_I[i];
               }

            for (i = 0; i < G_offd_nnz; i++)
            {
               Pix_offd_J[i] = G_offd_J[i];
            }

            for (i = 0; i < G_offd_nrows; i++)
               for (j = G_offd_I[i]; j < G_offd_I[i + 1]; j++)
               {
                  *Pix_offd_data++ = nalu_hypre_abs(G_offd_data[j]) * 0.5 * Gx_data[i];
               }
         }

         for (i = 0; i < G_offd_ncols; i++)
         {
            Pix_cmap[i] = G_cmap[i];
         }
      }
   }

   *Pix_ptr = Pix;
   if (dim >= 2)
   {
      *Piy_ptr = Piy;
   }
   if (dim == 3)
   {
      *Piz_ptr = Piz;
   }

   return nalu_hypre_error_flag;
}

#if defined(NALU_HYPRE_USING_GPU)
__global__ void
hypreGPUKernel_AMSComputeGPi_copy2(nalu_hypre_DeviceItem &item,
                                   NALU_HYPRE_Int   nrows,
                                   NALU_HYPRE_Int   dim,
                                   NALU_HYPRE_Int  *i_in,
                                   NALU_HYPRE_Real *data_in,
                                   NALU_HYPRE_Real *Gx_data,
                                   NALU_HYPRE_Real *Gy_data,
                                   NALU_HYPRE_Real *Gz_data,
                                   NALU_HYPRE_Real *data_out)
{
   const NALU_HYPRE_Int i = nalu_hypre_gpu_get_grid_warp_id<1, 1>(item);

   if (i >= nrows)
   {
      return;
   }

   const NALU_HYPRE_Int lane_id = nalu_hypre_gpu_get_lane_id<1>(item);
   NALU_HYPRE_Int j = 0, istart, iend;
   NALU_HYPRE_Real t, G[3], *Gdata[3];

   Gdata[0] = Gx_data;
   Gdata[1] = Gy_data;
   Gdata[2] = Gz_data;

   if (lane_id < 2)
   {
      j = read_only_load(i_in + i + lane_id);
   }

   istart = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, j, 0);
   iend   = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, j, 1);

   if (lane_id < dim - 1)
   {
      t = read_only_load(Gdata[lane_id] + i);
   }

   for (NALU_HYPRE_Int d = 0; d < dim - 1; d++)
   {
      G[d] = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, t, d);
   }

   for (j = istart + lane_id; j < iend; j += NALU_HYPRE_WARP_SIZE)
   {
      const NALU_HYPRE_Real u = read_only_load(&data_in[j]);
      const NALU_HYPRE_Real v = nalu_hypre_abs(u) * 0.5;
      const NALU_HYPRE_Int k = j * dim;

      data_out[k] = u;
      for (NALU_HYPRE_Int d = 0; d < dim - 1; d++)
      {
         data_out[k + d + 1] = v * G[d];
      }
   }
}
#endif

/*--------------------------------------------------------------------------
 * nalu_hypre_AMSComputeGPi
 *
 * Construct the matrix [G,Pi] which can be considered an interpolation
 * matrix from S_h^4 (4 copies of the scalar linear finite element space)
 * to the edge finite elements space.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int nalu_hypre_AMSComputeGPi(nalu_hypre_ParCSRMatrix *A,
                              nalu_hypre_ParCSRMatrix *G,
                              nalu_hypre_ParVector *Gx,
                              nalu_hypre_ParVector *Gy,
                              nalu_hypre_ParVector *Gz,
                              NALU_HYPRE_Int dim,
                              nalu_hypre_ParCSRMatrix **GPi_ptr)
{
   nalu_hypre_ParCSRMatrix *GPi;

   /* Take into account G */
   dim++;

   /* Compute GPi = [Pi_x, Pi_y, Pi_z, G] */
   {
      NALU_HYPRE_Int i, j, d;

      NALU_HYPRE_Real *Gx_data, *Gy_data, *Gz_data;

      MPI_Comm comm = nalu_hypre_ParCSRMatrixComm(G);
      NALU_HYPRE_BigInt global_num_rows = nalu_hypre_ParCSRMatrixGlobalNumRows(G);
      NALU_HYPRE_BigInt global_num_cols = dim * nalu_hypre_ParCSRMatrixGlobalNumCols(G);
      NALU_HYPRE_BigInt *row_starts = nalu_hypre_ParCSRMatrixRowStarts(G);
      NALU_HYPRE_BigInt *col_starts;
      NALU_HYPRE_Int col_starts_size;
      NALU_HYPRE_Int num_cols_offd = dim * nalu_hypre_CSRMatrixNumCols(nalu_hypre_ParCSRMatrixOffd(G));
      NALU_HYPRE_Int num_nonzeros_diag = dim * nalu_hypre_CSRMatrixNumNonzeros(nalu_hypre_ParCSRMatrixDiag(G));
      NALU_HYPRE_Int num_nonzeros_offd = dim * nalu_hypre_CSRMatrixNumNonzeros(nalu_hypre_ParCSRMatrixOffd(G));
      NALU_HYPRE_BigInt *col_starts_G = nalu_hypre_ParCSRMatrixColStarts(G);
      col_starts_size = 2;
      col_starts = nalu_hypre_TAlloc(NALU_HYPRE_BigInt, col_starts_size, NALU_HYPRE_MEMORY_HOST);
      for (i = 0; i < col_starts_size; i++)
      {
         col_starts[i] = (NALU_HYPRE_BigInt) dim * col_starts_G[i];
      }

      GPi = nalu_hypre_ParCSRMatrixCreate(comm,
                                     global_num_rows,
                                     global_num_cols,
                                     row_starts,
                                     col_starts,
                                     num_cols_offd,
                                     num_nonzeros_diag,
                                     num_nonzeros_offd);

      nalu_hypre_ParCSRMatrixOwnsData(GPi) = 1;
      nalu_hypre_ParCSRMatrixInitialize(GPi);

      Gx_data = nalu_hypre_VectorData(nalu_hypre_ParVectorLocalVector(Gx));
      if (dim >= 3)
      {
         Gy_data = nalu_hypre_VectorData(nalu_hypre_ParVectorLocalVector(Gy));
      }
      if (dim == 4)
      {
         Gz_data = nalu_hypre_VectorData(nalu_hypre_ParVectorLocalVector(Gz));
      }

#if defined(NALU_HYPRE_USING_GPU)
      NALU_HYPRE_ExecutionPolicy exec = nalu_hypre_GetExecPolicy2( nalu_hypre_ParCSRMatrixMemoryLocation(G),
                                                         nalu_hypre_ParCSRMatrixMemoryLocation(GPi) );
#endif

      /* Fill-in the diagonal part */
      {
         nalu_hypre_CSRMatrix *G_diag = nalu_hypre_ParCSRMatrixDiag(G);
         NALU_HYPRE_Int *G_diag_I = nalu_hypre_CSRMatrixI(G_diag);
         NALU_HYPRE_Int *G_diag_J = nalu_hypre_CSRMatrixJ(G_diag);
         NALU_HYPRE_Real *G_diag_data = nalu_hypre_CSRMatrixData(G_diag);

         NALU_HYPRE_Int G_diag_nrows = nalu_hypre_CSRMatrixNumRows(G_diag);
         NALU_HYPRE_Int G_diag_nnz = nalu_hypre_CSRMatrixNumNonzeros(G_diag);

         nalu_hypre_CSRMatrix *GPi_diag = nalu_hypre_ParCSRMatrixDiag(GPi);
         NALU_HYPRE_Int *GPi_diag_I = nalu_hypre_CSRMatrixI(GPi_diag);
         NALU_HYPRE_Int *GPi_diag_J = nalu_hypre_CSRMatrixJ(GPi_diag);
         NALU_HYPRE_Real *GPi_diag_data = nalu_hypre_CSRMatrixData(GPi_diag);

#if defined(NALU_HYPRE_USING_GPU)
         if (exec == NALU_HYPRE_EXEC_DEVICE)
         {
            hypreDevice_IntScalen( G_diag_I, G_diag_nrows + 1, GPi_diag_I, dim );

            dim3 bDim = nalu_hypre_GetDefaultDeviceBlockDimension();
            dim3 gDim = nalu_hypre_GetDefaultDeviceGridDimension(G_diag_nnz, "thread", bDim);

            NALU_HYPRE_GPU_LAUNCH( hypreGPUKernel_AMSComputePi_copy1, gDim, bDim,
                              G_diag_nnz, dim, G_diag_J, GPi_diag_J );

            gDim = nalu_hypre_GetDefaultDeviceGridDimension(G_diag_nrows, "warp", bDim);

            NALU_HYPRE_GPU_LAUNCH( hypreGPUKernel_AMSComputeGPi_copy2, gDim, bDim,
                              G_diag_nrows, dim, G_diag_I, G_diag_data, Gx_data, Gy_data, Gz_data,
                              GPi_diag_data );
         }
         else
#endif
         {
            for (i = 0; i < G_diag_nrows + 1; i++)
            {
               GPi_diag_I[i] = dim * G_diag_I[i];
            }

            for (i = 0; i < G_diag_nnz; i++)
               for (d = 0; d < dim; d++)
               {
                  GPi_diag_J[dim * i + d] = dim * G_diag_J[i] + d;
               }

            for (i = 0; i < G_diag_nrows; i++)
               for (j = G_diag_I[i]; j < G_diag_I[i + 1]; j++)
               {
                  *GPi_diag_data++ = G_diag_data[j];
                  *GPi_diag_data++ = nalu_hypre_abs(G_diag_data[j]) * 0.5 * Gx_data[i];
                  if (dim >= 3)
                  {
                     *GPi_diag_data++ = nalu_hypre_abs(G_diag_data[j]) * 0.5 * Gy_data[i];
                  }
                  if (dim == 4)
                  {
                     *GPi_diag_data++ = nalu_hypre_abs(G_diag_data[j]) * 0.5 * Gz_data[i];
                  }
               }
         }
      }

      /* Fill-in the off-diagonal part */
      {
         nalu_hypre_CSRMatrix *G_offd = nalu_hypre_ParCSRMatrixOffd(G);
         NALU_HYPRE_Int *G_offd_I = nalu_hypre_CSRMatrixI(G_offd);
         NALU_HYPRE_Int *G_offd_J = nalu_hypre_CSRMatrixJ(G_offd);
         NALU_HYPRE_Real *G_offd_data = nalu_hypre_CSRMatrixData(G_offd);

         NALU_HYPRE_Int G_offd_nrows = nalu_hypre_CSRMatrixNumRows(G_offd);
         NALU_HYPRE_Int G_offd_ncols = nalu_hypre_CSRMatrixNumCols(G_offd);
         NALU_HYPRE_Int G_offd_nnz = nalu_hypre_CSRMatrixNumNonzeros(G_offd);

         nalu_hypre_CSRMatrix *GPi_offd = nalu_hypre_ParCSRMatrixOffd(GPi);
         NALU_HYPRE_Int *GPi_offd_I = nalu_hypre_CSRMatrixI(GPi_offd);
         NALU_HYPRE_Int *GPi_offd_J = nalu_hypre_CSRMatrixJ(GPi_offd);
         NALU_HYPRE_Real *GPi_offd_data = nalu_hypre_CSRMatrixData(GPi_offd);

         NALU_HYPRE_BigInt *G_cmap = nalu_hypre_ParCSRMatrixColMapOffd(G);
         NALU_HYPRE_BigInt *GPi_cmap = nalu_hypre_ParCSRMatrixColMapOffd(GPi);

#if defined(NALU_HYPRE_USING_GPU)
         if (exec == NALU_HYPRE_EXEC_DEVICE)
         {
            if (G_offd_ncols)
            {
               hypreDevice_IntScalen( G_offd_I, G_offd_nrows + 1, GPi_offd_I, dim );
            }

            dim3 bDim = nalu_hypre_GetDefaultDeviceBlockDimension();
            dim3 gDim = nalu_hypre_GetDefaultDeviceGridDimension(G_offd_nnz, "thread", bDim);

            NALU_HYPRE_GPU_LAUNCH( hypreGPUKernel_AMSComputePi_copy1, gDim, bDim,
                              G_offd_nnz, dim, G_offd_J, GPi_offd_J );

            gDim = nalu_hypre_GetDefaultDeviceGridDimension(G_offd_nrows, "warp", bDim);

            NALU_HYPRE_GPU_LAUNCH( hypreGPUKernel_AMSComputeGPi_copy2, gDim, bDim,
                              G_offd_nrows, dim, G_offd_I, G_offd_data, Gx_data, Gy_data, Gz_data,
                              GPi_offd_data );
         }
         else
#endif
         {
            if (G_offd_ncols)
               for (i = 0; i < G_offd_nrows + 1; i++)
               {
                  GPi_offd_I[i] = dim * G_offd_I[i];
               }

            for (i = 0; i < G_offd_nnz; i++)
               for (d = 0; d < dim; d++)
               {
                  GPi_offd_J[dim * i + d] = dim * G_offd_J[i] + d;
               }

            for (i = 0; i < G_offd_nrows; i++)
               for (j = G_offd_I[i]; j < G_offd_I[i + 1]; j++)
               {
                  *GPi_offd_data++ = G_offd_data[j];
                  *GPi_offd_data++ = nalu_hypre_abs(G_offd_data[j]) * 0.5 * Gx_data[i];
                  if (dim >= 3)
                  {
                     *GPi_offd_data++ = nalu_hypre_abs(G_offd_data[j]) * 0.5 * Gy_data[i];
                  }
                  if (dim == 4)
                  {
                     *GPi_offd_data++ = nalu_hypre_abs(G_offd_data[j]) * 0.5 * Gz_data[i];
                  }
               }
         }

         for (i = 0; i < G_offd_ncols; i++)
            for (d = 0; d < dim; d++)
            {
               GPi_cmap[dim * i + d] = dim * G_cmap[i] + d;
            }
      }

   }

   *GPi_ptr = GPi;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_AMSSetup
 *
 * Construct the AMS solver components.
 *
 * The following functions need to be called before nalu_hypre_AMSSetup():
 * - nalu_hypre_AMSSetDimension() (if solving a 2D problem)
 * - nalu_hypre_AMSSetDiscreteGradient()
 * - nalu_hypre_AMSSetCoordinateVectors() or nalu_hypre_AMSSetEdgeConstantVectors
 *--------------------------------------------------------------------------*/
#if defined(NALU_HYPRE_USING_GPU)
__global__ void
hypreGPUKernel_FixInterNodes( nalu_hypre_DeviceItem    &item,
                              NALU_HYPRE_Int      nrows,
                              NALU_HYPRE_Int     *G0t_diag_i,
                              NALU_HYPRE_Complex *G0t_diag_data,
                              NALU_HYPRE_Int     *G0t_offd_i,
                              NALU_HYPRE_Complex *G0t_offd_data,
                              NALU_HYPRE_Real    *interior_nodes_data)
{
   NALU_HYPRE_Int row_i = nalu_hypre_gpu_get_grid_warp_id<1, 1>(item);

   if (row_i >= nrows)
   {
      return;
   }

   NALU_HYPRE_Int lane = nalu_hypre_gpu_get_lane_id<1>(item);
   NALU_HYPRE_Int not1 = 0;

   if (lane == 0)
   {
      not1 = read_only_load(&interior_nodes_data[row_i]) != 1.0;
   }

   not1 = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, not1, 0);

   if (!not1)
   {
      return;
   }

   NALU_HYPRE_Int p1 = 0, q1, p2 = 0, q2 = 0;
   bool nonempty_offd = G0t_offd_data != NULL;

   if (lane < 2)
   {
      p1 = read_only_load(G0t_diag_i + row_i + lane);
      if (nonempty_offd)
      {
         p2 = read_only_load(G0t_offd_i + row_i + lane);
      }
   }

   q1 = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, p1, 1);
   p1 = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, p1, 0);
   if (nonempty_offd)
   {
      q2 = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, p2, 1);
      p2 = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, p2, 0);
   }

   for (NALU_HYPRE_Int j = p1 + lane; j < q1; j += NALU_HYPRE_WARP_SIZE)
   {
      G0t_diag_data[j] = 0.0;
   }
   for (NALU_HYPRE_Int j = p2 + lane; j < q2; j += NALU_HYPRE_WARP_SIZE)
   {
      G0t_offd_data[j] = 0.0;
   }
}

__global__ void
hypreGPUKernel_AMSSetupScaleGGt( nalu_hypre_DeviceItem &item,
                                 NALU_HYPRE_Int   Gt_num_rows,
                                 NALU_HYPRE_Int  *Gt_diag_i,
                                 NALU_HYPRE_Int  *Gt_diag_j,
                                 NALU_HYPRE_Real *Gt_diag_data,
                                 NALU_HYPRE_Int  *Gt_offd_i,
                                 NALU_HYPRE_Real *Gt_offd_data,
                                 NALU_HYPRE_Real *Gx_data,
                                 NALU_HYPRE_Real *Gy_data,
                                 NALU_HYPRE_Real *Gz_data )
{
   NALU_HYPRE_Int row_i = nalu_hypre_gpu_get_grid_warp_id<1, 1>(item);

   if (row_i >= Gt_num_rows)
   {
      return;
   }

   NALU_HYPRE_Int lane = nalu_hypre_gpu_get_lane_id<1>(item);
   NALU_HYPRE_Real h2 = 0.0;
   NALU_HYPRE_Int ne, p1 = 0, q1, p2 = 0, q2 = 0;

   if (lane < 2)
   {
      p1 = read_only_load(Gt_diag_i + row_i + lane);
   }
   q1 = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, p1, 1);
   p1 = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, p1, 0);
   ne = q1 - p1;

   if (ne == 0)
   {
      return;
   }

   if (Gt_offd_data != NULL)
   {
      if (lane < 2)
      {
         p2 = read_only_load(Gt_offd_i + row_i + lane);
      }
      q2 = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, p2, 1);
      p2 = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, p2, 0);
   }

   for (NALU_HYPRE_Int j = p1 + lane; j < q1; j += NALU_HYPRE_WARP_SIZE)
   {
      const NALU_HYPRE_Int k = read_only_load(&Gt_diag_j[j]);
      const NALU_HYPRE_Real Gx = read_only_load(&Gx_data[k]);
      const NALU_HYPRE_Real Gy = read_only_load(&Gy_data[k]);
      const NALU_HYPRE_Real Gz = read_only_load(&Gz_data[k]);

      h2 += Gx * Gx + Gy * Gy + Gz * Gz;
   }

   h2 = warp_allreduce_sum(item, h2) / ne;

   for (NALU_HYPRE_Int j = p1 + lane; j < q1; j += NALU_HYPRE_WARP_SIZE)
   {
      Gt_diag_data[j] *= h2;
   }

   for (NALU_HYPRE_Int j = p2 + lane; j < q2; j += NALU_HYPRE_WARP_SIZE)
   {
      Gt_offd_data[j] *= h2;
   }
}
#endif

NALU_HYPRE_Int nalu_hypre_AMSSetup(void *solver,
                         nalu_hypre_ParCSRMatrix *A,
                         nalu_hypre_ParVector *b,
                         nalu_hypre_ParVector *x)
{
#if defined(NALU_HYPRE_USING_GPU)
   NALU_HYPRE_ExecutionPolicy exec = nalu_hypre_GetExecPolicy1( nalu_hypre_ParCSRMatrixMemoryLocation(A) );
#endif

   nalu_hypre_AMSData *ams_data = (nalu_hypre_AMSData *) solver;

   NALU_HYPRE_Int input_info = 0;

   ams_data -> A = A;

   /* Modifications for problems with zero-conductivity regions */
   if (ams_data -> interior_nodes)
   {
      nalu_hypre_ParCSRMatrix *G0t, *Aorig = A;

      /* Make sure that multiple Setup()+Solve() give identical results */
      ams_data -> solve_counter = 0;

      /* Construct the discrete gradient matrix for the zero-conductivity region
         by eliminating the zero-conductivity nodes from G^t. The range of G0
         represents the kernel of A, i.e. the gradients of nodal basis functions
         supported in zero-conductivity regions. */
      nalu_hypre_ParCSRMatrixTranspose(ams_data -> G, &G0t, 1);

      {
         NALU_HYPRE_Int i, j;
         NALU_HYPRE_Int nv = nalu_hypre_ParCSRMatrixNumCols(ams_data -> G);
         nalu_hypre_CSRMatrix *G0td = nalu_hypre_ParCSRMatrixDiag(G0t);
         NALU_HYPRE_Int *G0tdI = nalu_hypre_CSRMatrixI(G0td);
         NALU_HYPRE_Real *G0tdA = nalu_hypre_CSRMatrixData(G0td);
         nalu_hypre_CSRMatrix *G0to = nalu_hypre_ParCSRMatrixOffd(G0t);
         NALU_HYPRE_Int *G0toI = nalu_hypre_CSRMatrixI(G0to);
         NALU_HYPRE_Real *G0toA = nalu_hypre_CSRMatrixData(G0to);
         NALU_HYPRE_Real *interior_nodes_data = nalu_hypre_VectorData(
                                              nalu_hypre_ParVectorLocalVector((nalu_hypre_ParVector*) ams_data -> interior_nodes));

#if defined(NALU_HYPRE_USING_GPU)
         if (exec == NALU_HYPRE_EXEC_DEVICE)
         {
            dim3 bDim = nalu_hypre_GetDefaultDeviceBlockDimension();
            dim3 gDim = nalu_hypre_GetDefaultDeviceGridDimension(nv, "warp", bDim);
            NALU_HYPRE_GPU_LAUNCH( hypreGPUKernel_FixInterNodes, gDim, bDim,
                              nv, G0tdI, G0tdA, G0toI, G0toA, interior_nodes_data );
         }
         else
#endif
         {
            for (i = 0; i < nv; i++)
            {
               if (interior_nodes_data[i] != 1)
               {
                  for (j = G0tdI[i]; j < G0tdI[i + 1]; j++)
                  {
                     G0tdA[j] = 0.0;
                  }
                  if (G0toI)
                     for (j = G0toI[i]; j < G0toI[i + 1]; j++)
                     {
                        G0toA[j] = 0.0;
                     }
               }
            }
         }
      }
      nalu_hypre_ParCSRMatrixTranspose(G0t, & ams_data -> G0, 1);

      /* Construct the subspace matrix A_G0 = G0^T G0 */
#if defined(NALU_HYPRE_USING_GPU)
      if (exec == NALU_HYPRE_EXEC_DEVICE)
      {
         ams_data -> A_G0 = nalu_hypre_ParCSRMatMat(G0t, ams_data -> G0);
      }
      else
#endif
      {
         ams_data -> A_G0 = nalu_hypre_ParMatmul(G0t, ams_data -> G0);
      }
      nalu_hypre_ParCSRMatrixFixZeroRows(ams_data -> A_G0);

      /* Create AMG solver for A_G0 */
      NALU_HYPRE_BoomerAMGCreate(&ams_data -> B_G0);
      NALU_HYPRE_BoomerAMGSetCoarsenType(ams_data -> B_G0, ams_data -> B_G_coarsen_type);
      NALU_HYPRE_BoomerAMGSetAggNumLevels(ams_data -> B_G0, ams_data -> B_G_agg_levels);
      NALU_HYPRE_BoomerAMGSetRelaxType(ams_data -> B_G0, ams_data -> B_G_relax_type);
      NALU_HYPRE_BoomerAMGSetNumSweeps(ams_data -> B_G0, 1);
      NALU_HYPRE_BoomerAMGSetMaxLevels(ams_data -> B_G0, 25);
      NALU_HYPRE_BoomerAMGSetTol(ams_data -> B_G0, 0.0);
      NALU_HYPRE_BoomerAMGSetMaxIter(ams_data -> B_G0, 3); /* use just a few V-cycles */
      NALU_HYPRE_BoomerAMGSetStrongThreshold(ams_data -> B_G0, ams_data -> B_G_theta);
      NALU_HYPRE_BoomerAMGSetInterpType(ams_data -> B_G0, ams_data -> B_G_interp_type);
      NALU_HYPRE_BoomerAMGSetPMaxElmts(ams_data -> B_G0, ams_data -> B_G_Pmax);
      NALU_HYPRE_BoomerAMGSetMinCoarseSize(ams_data -> B_G0, 2); /* don't coarsen to 0 */
      /* Generally, don't use exact solve on the coarsest level (matrix may be singular) */
      NALU_HYPRE_BoomerAMGSetCycleRelaxType(ams_data -> B_G0, ams_data -> B_G_coarse_relax_type, 3);
      NALU_HYPRE_BoomerAMGSetup(ams_data -> B_G0,
                           (NALU_HYPRE_ParCSRMatrix)ams_data -> A_G0,
                           0, 0);

      /* Construct the preconditioner for ams_data->A = A + G0 G0^T.
         NOTE: this can be optimized significantly by taking into account that
         the sparsity pattern of A is subset of the sparsity pattern of G0 G0^T */
      {
#if defined(NALU_HYPRE_USING_GPU)
         nalu_hypre_ParCSRMatrix *A;
         if (exec == NALU_HYPRE_EXEC_DEVICE)
         {
            A = nalu_hypre_ParCSRMatMat(ams_data -> G0, G0t);
         }
         else
#endif
         {
            A = nalu_hypre_ParMatmul(ams_data -> G0, G0t);
         }
         nalu_hypre_ParCSRMatrix *B = Aorig;
         nalu_hypre_ParCSRMatrix **C_ptr = &ams_data -> A;
         nalu_hypre_ParCSRMatrix *C;
         NALU_HYPRE_Real factor, lfactor;
         /* scale (penalize) G0 G0^T before adding it to the matrix */
         {
            NALU_HYPRE_Int i;
            NALU_HYPRE_Int B_num_rows = nalu_hypre_CSRMatrixNumRows(nalu_hypre_ParCSRMatrixDiag(B));
            NALU_HYPRE_Real *B_diag_data = nalu_hypre_CSRMatrixData(nalu_hypre_ParCSRMatrixDiag(B));
            NALU_HYPRE_Real *B_offd_data = nalu_hypre_CSRMatrixData(nalu_hypre_ParCSRMatrixOffd(B));
            NALU_HYPRE_Int *B_diag_i = nalu_hypre_CSRMatrixI(nalu_hypre_ParCSRMatrixDiag(B));
            NALU_HYPRE_Int *B_offd_i = nalu_hypre_CSRMatrixI(nalu_hypre_ParCSRMatrixOffd(B));
            lfactor = -1;
#if defined(NALU_HYPRE_USING_GPU)
            if (exec == NALU_HYPRE_EXEC_DEVICE)
            {
               NALU_HYPRE_Int nnz_diag = nalu_hypre_CSRMatrixNumNonzeros(nalu_hypre_ParCSRMatrixDiag(B));
               NALU_HYPRE_Int nnz_offd = nalu_hypre_CSRMatrixNumNonzeros(nalu_hypre_ParCSRMatrixOffd(B));
#if defined(NALU_HYPRE_DEBUG)
               NALU_HYPRE_Int nnz;
               nalu_hypre_TMemcpy(&nnz, &B_diag_i[B_num_rows], NALU_HYPRE_Int, 1, NALU_HYPRE_MEMORY_HOST, NALU_HYPRE_MEMORY_DEVICE);
               nalu_hypre_assert(nnz == nnz_diag);
               nalu_hypre_TMemcpy(&nnz, &B_offd_i[B_num_rows], NALU_HYPRE_Int, 1, NALU_HYPRE_MEMORY_HOST, NALU_HYPRE_MEMORY_DEVICE);
               nalu_hypre_assert(nnz == nnz_offd);
#endif
#if defined(NALU_HYPRE_USING_SYCL)
               if (nnz_diag)
               {
                  lfactor = NALU_HYPRE_ONEDPL_CALL( std::reduce,
                                               oneapi::dpl::make_transform_iterator(B_diag_data,            absolute_value<NALU_HYPRE_Real>()),
                                               oneapi::dpl::make_transform_iterator(B_diag_data + nnz_diag, absolute_value<NALU_HYPRE_Real>()),
                                               -1.0,
                                               sycl::maximum<NALU_HYPRE_Real>() );
               }

               if (nnz_offd)
               {
                  lfactor = NALU_HYPRE_ONEDPL_CALL( std::reduce,
                                               oneapi::dpl::make_transform_iterator(B_offd_data,            absolute_value<NALU_HYPRE_Real>()),
                                               oneapi::dpl::make_transform_iterator(B_offd_data + nnz_offd, absolute_value<NALU_HYPRE_Real>()),
                                               lfactor,
                                               sycl::maximum<NALU_HYPRE_Real>() );

               }
#else
               if (nnz_diag)
               {
                  lfactor = NALU_HYPRE_THRUST_CALL( reduce,
                                               thrust::make_transform_iterator(B_diag_data,            absolute_value<NALU_HYPRE_Real>()),
                                               thrust::make_transform_iterator(B_diag_data + nnz_diag, absolute_value<NALU_HYPRE_Real>()),
                                               -1.0,
                                               thrust::maximum<NALU_HYPRE_Real>() );
               }

               if (nnz_offd)
               {
                  lfactor = NALU_HYPRE_THRUST_CALL( reduce,
                                               thrust::make_transform_iterator(B_offd_data,            absolute_value<NALU_HYPRE_Real>()),
                                               thrust::make_transform_iterator(B_offd_data + nnz_offd, absolute_value<NALU_HYPRE_Real>()),
                                               lfactor,
                                               thrust::maximum<NALU_HYPRE_Real>() );

               }
#endif
            }
            else
#endif
            {
               for (i = 0; i < B_diag_i[B_num_rows]; i++)
                  if (nalu_hypre_abs(B_diag_data[i]) > lfactor)
                  {
                     lfactor = nalu_hypre_abs(B_diag_data[i]);
                  }
               for (i = 0; i < B_offd_i[B_num_rows]; i++)
                  if (nalu_hypre_abs(B_offd_data[i]) > lfactor)
                  {
                     lfactor = nalu_hypre_abs(B_offd_data[i]);
                  }
            }

            lfactor *= 1e-10; /* scaling factor: max|A_ij|*1e-10 */
            nalu_hypre_MPI_Allreduce(&lfactor, &factor, 1, NALU_HYPRE_MPI_REAL, nalu_hypre_MPI_MAX, nalu_hypre_ParCSRMatrixComm(A));
         }

         nalu_hypre_ParCSRMatrixAdd(factor, A, 1.0, B, &C);

         /*nalu_hypre_CSRMatrix *A_local, *B_local, *C_local, *C_tmp;

         MPI_Comm comm = nalu_hypre_ParCSRMatrixComm(A);
         NALU_HYPRE_BigInt global_num_rows = nalu_hypre_ParCSRMatrixGlobalNumRows(A);
         NALU_HYPRE_BigInt global_num_cols = nalu_hypre_ParCSRMatrixGlobalNumCols(A);
         NALU_HYPRE_BigInt *row_starts = nalu_hypre_ParCSRMatrixRowStarts(A);
         NALU_HYPRE_BigInt *col_starts = nalu_hypre_ParCSRMatrixColStarts(A);
         NALU_HYPRE_Int A_num_cols_offd = nalu_hypre_CSRMatrixNumCols(nalu_hypre_ParCSRMatrixOffd(A));
         NALU_HYPRE_Int A_num_nonzeros_diag = nalu_hypre_CSRMatrixNumNonzeros(nalu_hypre_ParCSRMatrixDiag(A));
         NALU_HYPRE_Int A_num_nonzeros_offd = nalu_hypre_CSRMatrixNumNonzeros(nalu_hypre_ParCSRMatrixOffd(A));
         NALU_HYPRE_Int B_num_cols_offd = nalu_hypre_CSRMatrixNumCols(nalu_hypre_ParCSRMatrixOffd(B));
         NALU_HYPRE_Int B_num_nonzeros_diag = nalu_hypre_CSRMatrixNumNonzeros(nalu_hypre_ParCSRMatrixDiag(B));
         NALU_HYPRE_Int B_num_nonzeros_offd = nalu_hypre_CSRMatrixNumNonzeros(nalu_hypre_ParCSRMatrixOffd(B));

         A_local = nalu_hypre_MergeDiagAndOffd(A);
         B_local = nalu_hypre_MergeDiagAndOffd(B);*/
         /* scale (penalize) G0 G0^T before adding it to the matrix */
         /*{
            NALU_HYPRE_Int i, nnz = nalu_hypre_CSRMatrixNumNonzeros(A_local);
            NALU_HYPRE_Real *data = nalu_hypre_CSRMatrixData(A_local);
            NALU_HYPRE_Real *dataB = nalu_hypre_CSRMatrixData(B_local);
            NALU_HYPRE_Int nnzB = nalu_hypre_CSRMatrixNumNonzeros(B_local);
            NALU_HYPRE_Real factor, lfactor;
            lfactor = -1;
            for (i = 0; i < nnzB; i++)
               if (nalu_hypre_abs(dataB[i]) > lfactor)
                  lfactor = nalu_hypre_abs(dataB[i]);
            lfactor *= 1e-10;
            nalu_hypre_MPI_Allreduce(&lfactor, &factor, 1, NALU_HYPRE_MPI_REAL, nalu_hypre_MPI_MAX,
                                nalu_hypre_ParCSRMatrixComm(A));
            for (i = 0; i < nnz; i++)
               data[i] *= factor;
         }
         C_tmp = nalu_hypre_CSRMatrixBigAdd(A_local, B_local);
         C_local = nalu_hypre_CSRMatrixBigDeleteZeros(C_tmp,0.0);
         if (C_local)
            nalu_hypre_CSRMatrixDestroy(C_tmp);
         else
            C_local = C_tmp;

         C = nalu_hypre_ParCSRMatrixCreate (comm,
                                       global_num_rows,
                                       global_num_cols,
                                       row_starts,
                                       col_starts,
                                       A_num_cols_offd + B_num_cols_offd,
                                       A_num_nonzeros_diag + B_num_nonzeros_diag,
                                       A_num_nonzeros_offd + B_num_nonzeros_offd);
         GenerateDiagAndOffd(C_local, C,
                             nalu_hypre_ParCSRMatrixFirstColDiag(A),
                             nalu_hypre_ParCSRMatrixLastColDiag(A));

         nalu_hypre_CSRMatrixDestroy(A_local);
         nalu_hypre_CSRMatrixDestroy(B_local);
         nalu_hypre_CSRMatrixDestroy(C_local);
         */

         nalu_hypre_ParCSRMatrixDestroy(A);

         *C_ptr = C;
      }

      nalu_hypre_ParCSRMatrixDestroy(G0t);
   }

   /* Make sure that the first entry in each row is the diagonal one. */
   /* nalu_hypre_CSRMatrixReorder(nalu_hypre_ParCSRMatrixDiag(ams_data -> A)); */

   /* Compute the l1 norm of the rows of A */
   if (ams_data -> A_relax_type >= 1 && ams_data -> A_relax_type <= 4)
   {
      NALU_HYPRE_Real *l1_norm_data = NULL;

      nalu_hypre_ParCSRComputeL1Norms(ams_data -> A, ams_data -> A_relax_type, NULL, &l1_norm_data);

      ams_data -> A_l1_norms = nalu_hypre_SeqVectorCreate(nalu_hypre_ParCSRMatrixNumRows(ams_data -> A));
      nalu_hypre_VectorData(ams_data -> A_l1_norms) = l1_norm_data;
      nalu_hypre_SeqVectorInitialize_v2(ams_data -> A_l1_norms,
                                   nalu_hypre_ParCSRMatrixMemoryLocation(ams_data -> A));
   }

   /* Chebyshev? */
   if (ams_data -> A_relax_type == 16)
   {
      nalu_hypre_ParCSRMaxEigEstimateCG(ams_data->A, 1, 10,
                                   &ams_data->A_max_eig_est,
                                   &ams_data->A_min_eig_est);
   }

   /* If not given, compute Gx, Gy and Gz */
   {
      if (ams_data -> x != NULL &&
          (ams_data -> dim == 1 || ams_data -> y != NULL) &&
          (ams_data -> dim <= 2 || ams_data -> z != NULL))
      {
         input_info = 1;
      }

      if (ams_data -> Gx != NULL &&
          (ams_data -> dim == 1 || ams_data -> Gy != NULL) &&
          (ams_data -> dim <= 2 || ams_data -> Gz != NULL))
      {
         input_info = 2;
      }

      if (input_info == 1)
      {
         ams_data -> Gx = nalu_hypre_ParVectorInRangeOf(ams_data -> G);
         nalu_hypre_ParCSRMatrixMatvec (1.0, ams_data -> G, ams_data -> x, 0.0, ams_data -> Gx);
         if (ams_data -> dim >= 2)
         {
            ams_data -> Gy = nalu_hypre_ParVectorInRangeOf(ams_data -> G);
            nalu_hypre_ParCSRMatrixMatvec (1.0, ams_data -> G, ams_data -> y, 0.0, ams_data -> Gy);
         }
         if (ams_data -> dim == 3)
         {
            ams_data -> Gz = nalu_hypre_ParVectorInRangeOf(ams_data -> G);
            nalu_hypre_ParCSRMatrixMatvec (1.0, ams_data -> G, ams_data -> z, 0.0, ams_data -> Gz);
         }
      }
   }

   if (ams_data -> Pi == NULL && ams_data -> Pix == NULL)
   {
      if (ams_data -> cycle_type == 20)
         /* Construct the combined interpolation matrix [G,Pi] */
         nalu_hypre_AMSComputeGPi(ams_data -> A,
                             ams_data -> G,
                             ams_data -> Gx,
                             ams_data -> Gy,
                             ams_data -> Gz,
                             ams_data -> dim,
                             &ams_data -> Pi);
      else if (ams_data -> cycle_type > 10)
         /* Construct Pi{x,y,z} instead of Pi = [Pix,Piy,Piz] */
         nalu_hypre_AMSComputePixyz(ams_data -> A,
                               ams_data -> G,
                               ams_data -> Gx,
                               ams_data -> Gy,
                               ams_data -> Gz,
                               ams_data -> dim,
                               &ams_data -> Pix,
                               &ams_data -> Piy,
                               &ams_data -> Piz);
      else
         /* Construct the Pi interpolation matrix */
         nalu_hypre_AMSComputePi(ams_data -> A,
                            ams_data -> G,
                            ams_data -> Gx,
                            ams_data -> Gy,
                            ams_data -> Gz,
                            ams_data -> dim,
                            &ams_data -> Pi);
   }

   /* Keep Gx, Gy and Gz only if use the method with discrete divergence
      stabilization (where we use them to compute the local mesh size). */
   if (input_info == 1 && ams_data -> cycle_type != 9)
   {
      nalu_hypre_ParVectorDestroy(ams_data -> Gx);
      if (ams_data -> dim >= 2)
      {
         nalu_hypre_ParVectorDestroy(ams_data -> Gy);
      }
      if (ams_data -> dim == 3)
      {
         nalu_hypre_ParVectorDestroy(ams_data -> Gz);
      }
   }

   /* Create the AMG solver on the range of G^T */
   if (!ams_data -> beta_is_zero && ams_data -> cycle_type != 20)
   {
      NALU_HYPRE_BoomerAMGCreate(&ams_data -> B_G);
      NALU_HYPRE_BoomerAMGSetCoarsenType(ams_data -> B_G, ams_data -> B_G_coarsen_type);
      NALU_HYPRE_BoomerAMGSetAggNumLevels(ams_data -> B_G, ams_data -> B_G_agg_levels);
      NALU_HYPRE_BoomerAMGSetRelaxType(ams_data -> B_G, ams_data -> B_G_relax_type);
      NALU_HYPRE_BoomerAMGSetNumSweeps(ams_data -> B_G, 1);
      NALU_HYPRE_BoomerAMGSetMaxLevels(ams_data -> B_G, 25);
      NALU_HYPRE_BoomerAMGSetTol(ams_data -> B_G, 0.0);
      NALU_HYPRE_BoomerAMGSetMaxIter(ams_data -> B_G, 1);
      NALU_HYPRE_BoomerAMGSetStrongThreshold(ams_data -> B_G, ams_data -> B_G_theta);
      NALU_HYPRE_BoomerAMGSetInterpType(ams_data -> B_G, ams_data -> B_G_interp_type);
      NALU_HYPRE_BoomerAMGSetPMaxElmts(ams_data -> B_G, ams_data -> B_G_Pmax);
      NALU_HYPRE_BoomerAMGSetMinCoarseSize(ams_data -> B_G, 2); /* don't coarsen to 0 */

      /* Generally, don't use exact solve on the coarsest level (matrix may be singular) */
      NALU_HYPRE_BoomerAMGSetCycleRelaxType(ams_data -> B_G, ams_data -> B_G_coarse_relax_type, 3);

      if (ams_data -> cycle_type == 0)
      {
         NALU_HYPRE_BoomerAMGSetMaxLevels(ams_data -> B_G, 2);
      }

      /* If not given, construct the coarse space matrix by RAP */
      if (!ams_data -> A_G)
      {
         if (!nalu_hypre_ParCSRMatrixCommPkg(ams_data -> G))
         {
            nalu_hypre_MatvecCommPkgCreate(ams_data -> G);
         }

         if (!nalu_hypre_ParCSRMatrixCommPkg(ams_data -> A))
         {
            nalu_hypre_MatvecCommPkgCreate(ams_data -> A);
         }

#if defined(NALU_HYPRE_USING_GPU)
         if (exec == NALU_HYPRE_EXEC_DEVICE)
         {
            ams_data -> A_G = nalu_hypre_ParCSRMatrixRAPKT(ams_data -> G,
                                                      ams_data -> A,
                                                      ams_data -> G, 1);
         }
         else
#endif
         {
            nalu_hypre_BoomerAMGBuildCoarseOperator(ams_data -> G,
                                               ams_data -> A,
                                               ams_data -> G,
                                               &ams_data -> A_G);
         }

         /* Make sure that A_G has no zero rows (this can happen
            if beta is zero in part of the domain). */
         nalu_hypre_ParCSRMatrixFixZeroRows(ams_data -> A_G);
         ams_data -> owns_A_G = 1;
      }

      NALU_HYPRE_BoomerAMGSetup(ams_data -> B_G,
                           (NALU_HYPRE_ParCSRMatrix)ams_data -> A_G,
                           NULL, NULL);
   }

   if (ams_data -> cycle_type > 10 && ams_data -> cycle_type != 20)
      /* Create the AMG solvers on the range of Pi{x,y,z}^T */
   {
      NALU_HYPRE_BoomerAMGCreate(&ams_data -> B_Pix);
      NALU_HYPRE_BoomerAMGSetCoarsenType(ams_data -> B_Pix, ams_data -> B_Pi_coarsen_type);
      NALU_HYPRE_BoomerAMGSetAggNumLevels(ams_data -> B_Pix, ams_data -> B_Pi_agg_levels);
      NALU_HYPRE_BoomerAMGSetRelaxType(ams_data -> B_Pix, ams_data -> B_Pi_relax_type);
      NALU_HYPRE_BoomerAMGSetNumSweeps(ams_data -> B_Pix, 1);
      NALU_HYPRE_BoomerAMGSetMaxLevels(ams_data -> B_Pix, 25);
      NALU_HYPRE_BoomerAMGSetTol(ams_data -> B_Pix, 0.0);
      NALU_HYPRE_BoomerAMGSetMaxIter(ams_data -> B_Pix, 1);
      NALU_HYPRE_BoomerAMGSetStrongThreshold(ams_data -> B_Pix, ams_data -> B_Pi_theta);
      NALU_HYPRE_BoomerAMGSetInterpType(ams_data -> B_Pix, ams_data -> B_Pi_interp_type);
      NALU_HYPRE_BoomerAMGSetPMaxElmts(ams_data -> B_Pix, ams_data -> B_Pi_Pmax);
      NALU_HYPRE_BoomerAMGSetMinCoarseSize(ams_data -> B_Pix, 2);

      NALU_HYPRE_BoomerAMGCreate(&ams_data -> B_Piy);
      NALU_HYPRE_BoomerAMGSetCoarsenType(ams_data -> B_Piy, ams_data -> B_Pi_coarsen_type);
      NALU_HYPRE_BoomerAMGSetAggNumLevels(ams_data -> B_Piy, ams_data -> B_Pi_agg_levels);
      NALU_HYPRE_BoomerAMGSetRelaxType(ams_data -> B_Piy, ams_data -> B_Pi_relax_type);
      NALU_HYPRE_BoomerAMGSetNumSweeps(ams_data -> B_Piy, 1);
      NALU_HYPRE_BoomerAMGSetMaxLevels(ams_data -> B_Piy, 25);
      NALU_HYPRE_BoomerAMGSetTol(ams_data -> B_Piy, 0.0);
      NALU_HYPRE_BoomerAMGSetMaxIter(ams_data -> B_Piy, 1);
      NALU_HYPRE_BoomerAMGSetStrongThreshold(ams_data -> B_Piy, ams_data -> B_Pi_theta);
      NALU_HYPRE_BoomerAMGSetInterpType(ams_data -> B_Piy, ams_data -> B_Pi_interp_type);
      NALU_HYPRE_BoomerAMGSetPMaxElmts(ams_data -> B_Piy, ams_data -> B_Pi_Pmax);
      NALU_HYPRE_BoomerAMGSetMinCoarseSize(ams_data -> B_Piy, 2);

      NALU_HYPRE_BoomerAMGCreate(&ams_data -> B_Piz);
      NALU_HYPRE_BoomerAMGSetCoarsenType(ams_data -> B_Piz, ams_data -> B_Pi_coarsen_type);
      NALU_HYPRE_BoomerAMGSetAggNumLevels(ams_data -> B_Piz, ams_data -> B_Pi_agg_levels);
      NALU_HYPRE_BoomerAMGSetRelaxType(ams_data -> B_Piz, ams_data -> B_Pi_relax_type);
      NALU_HYPRE_BoomerAMGSetNumSweeps(ams_data -> B_Piz, 1);
      NALU_HYPRE_BoomerAMGSetMaxLevels(ams_data -> B_Piz, 25);
      NALU_HYPRE_BoomerAMGSetTol(ams_data -> B_Piz, 0.0);
      NALU_HYPRE_BoomerAMGSetMaxIter(ams_data -> B_Piz, 1);
      NALU_HYPRE_BoomerAMGSetStrongThreshold(ams_data -> B_Piz, ams_data -> B_Pi_theta);
      NALU_HYPRE_BoomerAMGSetInterpType(ams_data -> B_Piz, ams_data -> B_Pi_interp_type);
      NALU_HYPRE_BoomerAMGSetPMaxElmts(ams_data -> B_Piz, ams_data -> B_Pi_Pmax);
      NALU_HYPRE_BoomerAMGSetMinCoarseSize(ams_data -> B_Piz, 2);

      /* Generally, don't use exact solve on the coarsest level (matrices may be singular) */
      NALU_HYPRE_BoomerAMGSetCycleRelaxType(ams_data -> B_Pix, ams_data -> B_Pi_coarse_relax_type, 3);
      NALU_HYPRE_BoomerAMGSetCycleRelaxType(ams_data -> B_Piy, ams_data -> B_Pi_coarse_relax_type, 3);
      NALU_HYPRE_BoomerAMGSetCycleRelaxType(ams_data -> B_Piz, ams_data -> B_Pi_coarse_relax_type, 3);

      if (ams_data -> cycle_type == 0)
      {
         NALU_HYPRE_BoomerAMGSetMaxLevels(ams_data -> B_Pix, 2);
         NALU_HYPRE_BoomerAMGSetMaxLevels(ams_data -> B_Piy, 2);
         NALU_HYPRE_BoomerAMGSetMaxLevels(ams_data -> B_Piz, 2);
      }

      /* Construct the coarse space matrices by RAP */
      if (!nalu_hypre_ParCSRMatrixCommPkg(ams_data -> Pix))
      {
         nalu_hypre_MatvecCommPkgCreate(ams_data -> Pix);
      }

#if defined(NALU_HYPRE_USING_GPU)
      if (exec == NALU_HYPRE_EXEC_DEVICE)
      {
         ams_data -> A_Pix = nalu_hypre_ParCSRMatrixRAPKT(ams_data -> Pix, ams_data -> A, ams_data -> Pix, 1);
      }
      else
#endif
      {
         nalu_hypre_BoomerAMGBuildCoarseOperator(ams_data -> Pix,
                                            ams_data -> A,
                                            ams_data -> Pix,
                                            &ams_data -> A_Pix);
      }

      /* Make sure that A_Pix has no zero rows (this can happen
         for some kinds of boundary conditions with contact). */
      nalu_hypre_ParCSRMatrixFixZeroRows(ams_data -> A_Pix);

      NALU_HYPRE_BoomerAMGSetup(ams_data -> B_Pix,
                           (NALU_HYPRE_ParCSRMatrix)ams_data -> A_Pix,
                           NULL, NULL);

      if (ams_data -> Piy)
      {
         if (!nalu_hypre_ParCSRMatrixCommPkg(ams_data -> Piy))
         {
            nalu_hypre_MatvecCommPkgCreate(ams_data -> Piy);
         }

#if defined(NALU_HYPRE_USING_GPU)
         if (exec == NALU_HYPRE_EXEC_DEVICE)
         {
            ams_data -> A_Piy = nalu_hypre_ParCSRMatrixRAPKT(ams_data -> Piy,
                                                        ams_data -> A,
                                                        ams_data -> Piy, 1);
         }
         else
#endif
         {
            nalu_hypre_BoomerAMGBuildCoarseOperator(ams_data -> Piy,
                                               ams_data -> A,
                                               ams_data -> Piy,
                                               &ams_data -> A_Piy);
         }

         /* Make sure that A_Piy has no zero rows (this can happen
            for some kinds of boundary conditions with contact). */
         nalu_hypre_ParCSRMatrixFixZeroRows(ams_data -> A_Piy);

         NALU_HYPRE_BoomerAMGSetup(ams_data -> B_Piy,
                              (NALU_HYPRE_ParCSRMatrix)ams_data -> A_Piy,
                              NULL, NULL);
      }

      if (ams_data -> Piz)
      {
         if (!nalu_hypre_ParCSRMatrixCommPkg(ams_data -> Piz))
         {
            nalu_hypre_MatvecCommPkgCreate(ams_data -> Piz);
         }

#if defined(NALU_HYPRE_USING_GPU)
         if (exec == NALU_HYPRE_EXEC_DEVICE)
         {
            ams_data -> A_Piz = nalu_hypre_ParCSRMatrixRAPKT(ams_data -> Piz,
                                                        ams_data -> A,
                                                        ams_data -> Piz, 1);
         }
         else
#endif
         {
            nalu_hypre_BoomerAMGBuildCoarseOperator(ams_data -> Piz,
                                               ams_data -> A,
                                               ams_data -> Piz,
                                               &ams_data -> A_Piz);
         }

         /* Make sure that A_Piz has no zero rows (this can happen
            for some kinds of boundary conditions with contact). */
         nalu_hypre_ParCSRMatrixFixZeroRows(ams_data -> A_Piz);

         NALU_HYPRE_BoomerAMGSetup(ams_data -> B_Piz,
                              (NALU_HYPRE_ParCSRMatrix)ams_data -> A_Piz,
                              NULL, NULL);
      }
   }
   else
      /* Create the AMG solver on the range of Pi^T */
   {
      NALU_HYPRE_BoomerAMGCreate(&ams_data -> B_Pi);
      NALU_HYPRE_BoomerAMGSetCoarsenType(ams_data -> B_Pi, ams_data -> B_Pi_coarsen_type);
      NALU_HYPRE_BoomerAMGSetAggNumLevels(ams_data -> B_Pi, ams_data -> B_Pi_agg_levels);
      NALU_HYPRE_BoomerAMGSetRelaxType(ams_data -> B_Pi, ams_data -> B_Pi_relax_type);
      NALU_HYPRE_BoomerAMGSetNumSweeps(ams_data -> B_Pi, 1);
      NALU_HYPRE_BoomerAMGSetMaxLevels(ams_data -> B_Pi, 25);
      NALU_HYPRE_BoomerAMGSetTol(ams_data -> B_Pi, 0.0);
      NALU_HYPRE_BoomerAMGSetMaxIter(ams_data -> B_Pi, 1);
      NALU_HYPRE_BoomerAMGSetStrongThreshold(ams_data -> B_Pi, ams_data -> B_Pi_theta);
      NALU_HYPRE_BoomerAMGSetInterpType(ams_data -> B_Pi, ams_data -> B_Pi_interp_type);
      NALU_HYPRE_BoomerAMGSetPMaxElmts(ams_data -> B_Pi, ams_data -> B_Pi_Pmax);
      NALU_HYPRE_BoomerAMGSetMinCoarseSize(ams_data -> B_Pi, 2); /* don't coarsen to 0 */

      /* Generally, don't use exact solve on the coarsest level (matrix may be singular) */
      NALU_HYPRE_BoomerAMGSetCycleRelaxType(ams_data -> B_Pi, ams_data -> B_Pi_coarse_relax_type, 3);

      if (ams_data -> cycle_type == 0)
      {
         NALU_HYPRE_BoomerAMGSetMaxLevels(ams_data -> B_Pi, 2);
      }

      /* If not given, construct the coarse space matrix by RAP and
         notify BoomerAMG that this is a dim x dim block system. */
      if (!ams_data -> A_Pi)
      {
         if (!nalu_hypre_ParCSRMatrixCommPkg(ams_data -> Pi))
         {
            nalu_hypre_MatvecCommPkgCreate(ams_data -> Pi);
         }

         if (!nalu_hypre_ParCSRMatrixCommPkg(ams_data -> A))
         {
            nalu_hypre_MatvecCommPkgCreate(ams_data -> A);
         }

         if (ams_data -> cycle_type == 9)
         {
            /* Add a discrete divergence term to A before computing  Pi^t A Pi */
            {
               nalu_hypre_ParCSRMatrix *Gt, *GGt = NULL, *ApGGt;
               nalu_hypre_ParCSRMatrixTranspose(ams_data -> G, &Gt, 1);

               /* scale GGt by h^2 */
               {
                  NALU_HYPRE_Real h2;
                  NALU_HYPRE_Int i, j, k, ne;

                  nalu_hypre_CSRMatrix *Gt_diag = nalu_hypre_ParCSRMatrixDiag(Gt);
                  NALU_HYPRE_Int Gt_num_rows = nalu_hypre_CSRMatrixNumRows(Gt_diag);
                  NALU_HYPRE_Int *Gt_diag_I = nalu_hypre_CSRMatrixI(Gt_diag);
                  NALU_HYPRE_Int *Gt_diag_J = nalu_hypre_CSRMatrixJ(Gt_diag);
                  NALU_HYPRE_Real *Gt_diag_data = nalu_hypre_CSRMatrixData(Gt_diag);

                  nalu_hypre_CSRMatrix *Gt_offd = nalu_hypre_ParCSRMatrixOffd(Gt);
                  NALU_HYPRE_Int *Gt_offd_I = nalu_hypre_CSRMatrixI(Gt_offd);
                  NALU_HYPRE_Real *Gt_offd_data = nalu_hypre_CSRMatrixData(Gt_offd);

                  NALU_HYPRE_Real *Gx_data = nalu_hypre_VectorData(nalu_hypre_ParVectorLocalVector(ams_data -> Gx));
                  NALU_HYPRE_Real *Gy_data = nalu_hypre_VectorData(nalu_hypre_ParVectorLocalVector(ams_data -> Gy));
                  NALU_HYPRE_Real *Gz_data = nalu_hypre_VectorData(nalu_hypre_ParVectorLocalVector(ams_data -> Gz));

#if defined(NALU_HYPRE_USING_GPU)
                  if (exec == NALU_HYPRE_EXEC_DEVICE)
                  {
                     dim3 bDim = nalu_hypre_GetDefaultDeviceBlockDimension();
                     dim3 gDim = nalu_hypre_GetDefaultDeviceGridDimension(Gt_num_rows, "warp", bDim);
                     NALU_HYPRE_GPU_LAUNCH( hypreGPUKernel_AMSSetupScaleGGt, gDim, bDim,
                                       Gt_num_rows, Gt_diag_I, Gt_diag_J, Gt_diag_data, Gt_offd_I, Gt_offd_data,
                                       Gx_data, Gy_data, Gz_data );
                  }
                  else
#endif
                  {
                     for (i = 0; i < Gt_num_rows; i++)
                     {
                        /* determine the characteristic mesh size for vertex i */
                        h2 = 0.0;
                        ne = 0;
                        for (j = Gt_diag_I[i]; j < Gt_diag_I[i + 1]; j++)
                        {
                           k = Gt_diag_J[j];
                           h2 += Gx_data[k] * Gx_data[k] + Gy_data[k] * Gy_data[k] + Gz_data[k] * Gz_data[k];
                           ne++;
                        }

                        if (ne != 0)
                        {
                           h2 /= ne;
                           for (j = Gt_diag_I[i]; j < Gt_diag_I[i + 1]; j++)
                           {
                              Gt_diag_data[j] *= h2;
                           }
                           for (j = Gt_offd_I[i]; j < Gt_offd_I[i + 1]; j++)
                           {
                              Gt_offd_data[j] *= h2;
                           }
                        }
                     }
                  }
               }

               /* we only needed Gx, Gy and Gz to compute the local mesh size */
               if (input_info == 1)
               {
                  nalu_hypre_ParVectorDestroy(ams_data -> Gx);
                  if (ams_data -> dim >= 2)
                  {
                     nalu_hypre_ParVectorDestroy(ams_data -> Gy);
                  }
                  if (ams_data -> dim == 3)
                  {
                     nalu_hypre_ParVectorDestroy(ams_data -> Gz);
                  }
               }

#if defined(NALU_HYPRE_USING_GPU)
               if (exec == NALU_HYPRE_EXEC_DEVICE)
               {
                  GGt = nalu_hypre_ParCSRMatMat(ams_data -> G, Gt);
               }
               else
#endif
               {
                  GGt = nalu_hypre_ParMatmul(ams_data -> G, Gt);
               }
               nalu_hypre_ParCSRMatrixDestroy(Gt);

               /* nalu_hypre_ParCSRMatrixAdd(GGt, A, &ams_data -> A); */
               nalu_hypre_ParCSRMatrixAdd(1.0, GGt, 1.0, ams_data -> A, &ApGGt);
               /*{
                  nalu_hypre_ParCSRMatrix *A = GGt;
                  nalu_hypre_ParCSRMatrix *B = ams_data -> A;
                  nalu_hypre_ParCSRMatrix **C_ptr = &ApGGt;

                  nalu_hypre_ParCSRMatrix *C;
                  nalu_hypre_CSRMatrix *A_local, *B_local, *C_local;

                  MPI_Comm comm = nalu_hypre_ParCSRMatrixComm(A);
                  NALU_HYPRE_BigInt global_num_rows = nalu_hypre_ParCSRMatrixGlobalNumRows(A);
                  NALU_HYPRE_BigInt global_num_cols = nalu_hypre_ParCSRMatrixGlobalNumCols(A);
                  NALU_HYPRE_BigInt *row_starts = nalu_hypre_ParCSRMatrixRowStarts(A);
                  NALU_HYPRE_BigInt *col_starts = nalu_hypre_ParCSRMatrixColStarts(A);
                  NALU_HYPRE_Int A_num_cols_offd = nalu_hypre_CSRMatrixNumCols(nalu_hypre_ParCSRMatrixOffd(A));
                  NALU_HYPRE_Int A_num_nonzeros_diag = nalu_hypre_CSRMatrixNumNonzeros(nalu_hypre_ParCSRMatrixDiag(A));
                  NALU_HYPRE_Int A_num_nonzeros_offd = nalu_hypre_CSRMatrixNumNonzeros(nalu_hypre_ParCSRMatrixOffd(A));
                  NALU_HYPRE_Int B_num_cols_offd = nalu_hypre_CSRMatrixNumCols(nalu_hypre_ParCSRMatrixOffd(B));
                  NALU_HYPRE_Int B_num_nonzeros_diag = nalu_hypre_CSRMatrixNumNonzeros(nalu_hypre_ParCSRMatrixDiag(B));
                  NALU_HYPRE_Int B_num_nonzeros_offd = nalu_hypre_CSRMatrixNumNonzeros(nalu_hypre_ParCSRMatrixOffd(B));

                  A_local = nalu_hypre_MergeDiagAndOffd(A);
                  B_local = nalu_hypre_MergeDiagAndOffd(B);
                  C_local = nalu_hypre_CSRMatrixBigAdd(A_local, B_local);
                  nalu_hypre_CSRMatrixBigJtoJ(C_local);

                  C = nalu_hypre_ParCSRMatrixCreate (comm,
                                                global_num_rows,
                                                global_num_cols,
                                                row_starts,
                                                col_starts,
                                                A_num_cols_offd + B_num_cols_offd,
                                                A_num_nonzeros_diag + B_num_nonzeros_diag,
                                                A_num_nonzeros_offd + B_num_nonzeros_offd);
                  GenerateDiagAndOffd(C_local, C,
                                      nalu_hypre_ParCSRMatrixFirstColDiag(A),
                                      nalu_hypre_ParCSRMatrixLastColDiag(A));

                  nalu_hypre_CSRMatrixDestroy(A_local);
                  nalu_hypre_CSRMatrixDestroy(B_local);
                  nalu_hypre_CSRMatrixDestroy(C_local);

                  *C_ptr = C;
               }*/

               nalu_hypre_ParCSRMatrixDestroy(GGt);

#if defined(NALU_HYPRE_USING_GPU)
               if (exec == NALU_HYPRE_EXEC_DEVICE)
               {
                  ams_data -> A_Pi = nalu_hypre_ParCSRMatrixRAPKT(ams_data -> Pi, ApGGt, ams_data -> Pi, 1);
               }
               else
#endif
               {
                  nalu_hypre_BoomerAMGBuildCoarseOperator(ams_data -> Pi,
                                                     ApGGt,
                                                     ams_data -> Pi,
                                                     &ams_data -> A_Pi);
               }
            }
         }
         else
         {
#if defined(NALU_HYPRE_USING_GPU)
            if (exec == NALU_HYPRE_EXEC_DEVICE)
            {
               ams_data -> A_Pi = nalu_hypre_ParCSRMatrixRAPKT(ams_data -> Pi, ams_data -> A, ams_data -> Pi, 1);
            }
            else
#endif
            {
               nalu_hypre_BoomerAMGBuildCoarseOperator(ams_data -> Pi,
                                                  ams_data -> A,
                                                  ams_data -> Pi,
                                                  &ams_data -> A_Pi);
            }
         }

         ams_data -> owns_A_Pi = 1;

         if (ams_data -> cycle_type != 20)
         {
            NALU_HYPRE_BoomerAMGSetNumFunctions(ams_data -> B_Pi, ams_data -> dim);
         }
         else
         {
            NALU_HYPRE_BoomerAMGSetNumFunctions(ams_data -> B_Pi, ams_data -> dim + 1);
         }
         /* NALU_HYPRE_BoomerAMGSetNodal(ams_data -> B_Pi, 1); */
      }

      /* Make sure that A_Pi has no zero rows (this can happen for
         some kinds of boundary conditions with contact). */
      nalu_hypre_ParCSRMatrixFixZeroRows(ams_data -> A_Pi);

      NALU_HYPRE_BoomerAMGSetup(ams_data -> B_Pi,
                           (NALU_HYPRE_ParCSRMatrix)ams_data -> A_Pi,
                           0, 0);
   }

   /* Allocate temporary vectors */
   ams_data -> r0 = nalu_hypre_ParVectorInRangeOf(ams_data -> A);
   ams_data -> g0 = nalu_hypre_ParVectorInRangeOf(ams_data -> A);
   if (ams_data -> A_G)
   {
      ams_data -> r1 = nalu_hypre_ParVectorInRangeOf(ams_data -> A_G);
      ams_data -> g1 = nalu_hypre_ParVectorInRangeOf(ams_data -> A_G);
   }
   if (ams_data -> r1 == NULL && ams_data -> A_Pix)
   {
      ams_data -> r1 = nalu_hypre_ParVectorInRangeOf(ams_data -> A_Pix);
      ams_data -> g1 = nalu_hypre_ParVectorInRangeOf(ams_data -> A_Pix);
   }
   if (ams_data -> Pi)
   {
      ams_data -> r2 = nalu_hypre_ParVectorInDomainOf(ams_data -> Pi);
      ams_data -> g2 = nalu_hypre_ParVectorInDomainOf(ams_data -> Pi);
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_AMSSolve
 *
 * Solve the system A x = b.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int nalu_hypre_AMSSolve(void *solver,
                         nalu_hypre_ParCSRMatrix *A,
                         nalu_hypre_ParVector *b,
                         nalu_hypre_ParVector *x)
{
   nalu_hypre_AMSData *ams_data = (nalu_hypre_AMSData *) solver;

   NALU_HYPRE_Int i, my_id = -1;
   NALU_HYPRE_Real r0_norm, r_norm, b_norm, relative_resid = 0, old_resid;

   char cycle[30];
   nalu_hypre_ParCSRMatrix *Ai[5], *Pi[5];
   NALU_HYPRE_Solver Bi[5];
   NALU_HYPRE_PtrToSolverFcn HBi[5];
   nalu_hypre_ParVector *ri[5], *gi[5];
   NALU_HYPRE_Int needZ = 0;

   nalu_hypre_ParVector *z = ams_data -> zz;

   Ai[0] = ams_data -> A_G;    Pi[0] = ams_data -> G;
   Ai[1] = ams_data -> A_Pi;   Pi[1] = ams_data -> Pi;
   Ai[2] = ams_data -> A_Pix;  Pi[2] = ams_data -> Pix;
   Ai[3] = ams_data -> A_Piy;  Pi[3] = ams_data -> Piy;
   Ai[4] = ams_data -> A_Piz;  Pi[4] = ams_data -> Piz;

   Bi[0] = ams_data -> B_G;    HBi[0] = (NALU_HYPRE_PtrToSolverFcn) nalu_hypre_BoomerAMGSolve;
   Bi[1] = ams_data -> B_Pi;   HBi[1] = (NALU_HYPRE_PtrToSolverFcn) nalu_hypre_BoomerAMGBlockSolve;
   Bi[2] = ams_data -> B_Pix;  HBi[2] = (NALU_HYPRE_PtrToSolverFcn) nalu_hypre_BoomerAMGSolve;
   Bi[3] = ams_data -> B_Piy;  HBi[3] = (NALU_HYPRE_PtrToSolverFcn) nalu_hypre_BoomerAMGSolve;
   Bi[4] = ams_data -> B_Piz;  HBi[4] = (NALU_HYPRE_PtrToSolverFcn) nalu_hypre_BoomerAMGSolve;

   ri[0] = ams_data -> r1;     gi[0] = ams_data -> g1;
   ri[1] = ams_data -> r2;     gi[1] = ams_data -> g2;
   ri[2] = ams_data -> r1;     gi[2] = ams_data -> g1;
   ri[3] = ams_data -> r1;     gi[3] = ams_data -> g1;
   ri[4] = ams_data -> r1;     gi[4] = ams_data -> g1;

   /* may need to create an additional temporary vector for relaxation */
#if defined(NALU_HYPRE_USING_GPU)
   NALU_HYPRE_ExecutionPolicy exec = nalu_hypre_GetExecPolicy1( nalu_hypre_ParCSRMatrixMemoryLocation(A) );

   if (exec == NALU_HYPRE_EXEC_DEVICE)
   {
      needZ = ams_data -> A_relax_type == 2 || ams_data -> A_relax_type == 4 ||
              ams_data -> A_relax_type == 16;
   }
   else
#endif
   {
      needZ = nalu_hypre_NumThreads() > 1 || ams_data -> A_relax_type == 16;
   }

   if (needZ && !z)
   {
      z = nalu_hypre_ParVectorCreate(nalu_hypre_ParCSRMatrixComm(A),
                                nalu_hypre_ParCSRMatrixGlobalNumRows(A),
                                nalu_hypre_ParCSRMatrixRowStarts(A));
      nalu_hypre_ParVectorInitialize(z);
      ams_data -> zz = z;
   }

   if (ams_data -> print_level > 0)
   {
      nalu_hypre_MPI_Comm_rank(nalu_hypre_ParCSRMatrixComm(A), &my_id);
   }

   /* Compatible subspace projection for problems with zero-conductivity regions.
      Note that this modifies the input (r.h.s.) vector b! */
   if ( (ams_data -> B_G0) &&
        (++ams_data->solve_counter % ( ams_data -> projection_frequency ) == 0) )
   {
      /* nalu_hypre_printf("Projecting onto the compatible subspace...\n"); */
      nalu_hypre_AMSProjectOutGradients(ams_data, b);
   }

   if (ams_data -> beta_is_zero)
   {
      switch (ams_data -> cycle_type)
      {
         case 0:
            nalu_hypre_sprintf(cycle, "%s", "0");
            break;
         case 1:
         case 3:
         case 5:
         case 7:
         default:
            nalu_hypre_sprintf(cycle, "%s", "020");
            break;
         case 2:
         case 4:
         case 6:
         case 8:
            nalu_hypre_sprintf(cycle, "%s", "(0+2)");
            break;
         case 11:
         case 13:
            nalu_hypre_sprintf(cycle, "%s", "0345430");
            break;
         case 12:
            nalu_hypre_sprintf(cycle, "%s", "(0+3+4+5)");
            break;
         case 14:
            nalu_hypre_sprintf(cycle, "%s", "0(+3+4+5)0");
            break;
      }
   }
   else
   {
      switch (ams_data -> cycle_type)
      {
         case 0:
            nalu_hypre_sprintf(cycle, "%s", "010");
            break;
         case 1:
         default:
            nalu_hypre_sprintf(cycle, "%s", "01210");
            break;
         case 2:
            nalu_hypre_sprintf(cycle, "%s", "(0+1+2)");
            break;
         case 3:
            nalu_hypre_sprintf(cycle, "%s", "02120");
            break;
         case 4:
            nalu_hypre_sprintf(cycle, "%s", "(010+2)");
            break;
         case 5:
            nalu_hypre_sprintf(cycle, "%s", "0102010");
            break;
         case 6:
            nalu_hypre_sprintf(cycle, "%s", "(020+1)");
            break;
         case 7:
            nalu_hypre_sprintf(cycle, "%s", "0201020");
            break;
         case 8:
            nalu_hypre_sprintf(cycle, "%s", "0(+1+2)0");
            break;
         case 9:
            nalu_hypre_sprintf(cycle, "%s", "01210");
            break;
         case 11:
            nalu_hypre_sprintf(cycle, "%s", "013454310");
            break;
         case 12:
            nalu_hypre_sprintf(cycle, "%s", "(0+1+3+4+5)");
            break;
         case 13:
            nalu_hypre_sprintf(cycle, "%s", "034515430");
            break;
         case 14:
            nalu_hypre_sprintf(cycle, "%s", "01(+3+4+5)10");
            break;
         case 20:
            nalu_hypre_sprintf(cycle, "%s", "020");
            break;
      }
   }

   for (i = 0; i < ams_data -> maxit; i++)
   {
      /* Compute initial residual norms */
      if (ams_data -> maxit > 1 && i == 0)
      {
         nalu_hypre_ParVectorCopy(b, ams_data -> r0);
         nalu_hypre_ParCSRMatrixMatvec(-1.0, ams_data -> A, x, 1.0, ams_data -> r0);
         r_norm = nalu_hypre_sqrt(nalu_hypre_ParVectorInnerProd(ams_data -> r0, ams_data -> r0));
         r0_norm = r_norm;
         b_norm = nalu_hypre_sqrt(nalu_hypre_ParVectorInnerProd(b, b));
         if (b_norm)
         {
            relative_resid = r_norm / b_norm;
         }
         else
         {
            relative_resid = r_norm;
         }
         if (my_id == 0 && ams_data -> print_level > 0)
         {
            nalu_hypre_printf("                                            relative\n");
            nalu_hypre_printf("               residual        factor       residual\n");
            nalu_hypre_printf("               --------        ------       --------\n");
            nalu_hypre_printf("    Initial    %e                 %e\n",
                         r_norm, relative_resid);
         }
      }

      /* Apply the preconditioner */
      nalu_hypre_ParCSRSubspacePrec(ams_data -> A,
                               ams_data -> A_relax_type,
                               ams_data -> A_relax_times,
                               ams_data -> A_l1_norms ? nalu_hypre_VectorData(ams_data -> A_l1_norms) : NULL,
                               ams_data -> A_relax_weight,
                               ams_data -> A_omega,
                               ams_data -> A_max_eig_est,
                               ams_data -> A_min_eig_est,
                               ams_data -> A_cheby_order,
                               ams_data -> A_cheby_fraction,
                               Ai, Bi, HBi, Pi, ri, gi,
                               b, x,
                               ams_data -> r0,
                               ams_data -> g0,
                               cycle,
                               z);

      /* Compute new residual norms */
      if (ams_data -> maxit > 1)
      {
         old_resid = r_norm;
         nalu_hypre_ParVectorCopy(b, ams_data -> r0);
         nalu_hypre_ParCSRMatrixMatvec(-1.0, ams_data -> A, x, 1.0, ams_data -> r0);
         r_norm = nalu_hypre_sqrt(nalu_hypre_ParVectorInnerProd(ams_data -> r0, ams_data -> r0));
         if (b_norm)
         {
            relative_resid = r_norm / b_norm;
         }
         else
         {
            relative_resid = r_norm;
         }
         if (my_id == 0 && ams_data -> print_level > 0)
            nalu_hypre_printf("    Cycle %2d   %e    %f     %e \n",
                         i + 1, r_norm, r_norm / old_resid, relative_resid);
      }

      if (relative_resid < ams_data -> tol)
      {
         i++;
         break;
      }
   }

   if (my_id == 0 && ams_data -> print_level > 0 && ams_data -> maxit > 1)
      nalu_hypre_printf("\n\n Average Convergence Factor = %f\n\n",
                   nalu_hypre_pow((r_norm / r0_norm), (1.0 / (NALU_HYPRE_Real) i)));

   ams_data -> num_iterations = i;
   ams_data -> rel_resid_norm = relative_resid;

   if (ams_data -> num_iterations == ams_data -> maxit && ams_data -> tol > 0.0)
   {
      nalu_hypre_error(NALU_HYPRE_ERROR_CONV);
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ParCSRSubspacePrec
 *
 * General subspace preconditioner for A0 y = x, based on ParCSR storage.
 *
 * P[i] and A[i] are the interpolation and coarse grid matrices for
 * the (i+1)'th subspace. B[i] is an AMG solver for A[i]. r[i] and g[i]
 * are temporary vectors. A0_* are the fine grid smoothing parameters.
 *
 * The default mode is multiplicative, '+' changes the next correction
 * to additive, based on residual computed at '('.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int nalu_hypre_ParCSRSubspacePrec(/* fine space matrix */
   nalu_hypre_ParCSRMatrix *A0,
   /* relaxation parameters */
   NALU_HYPRE_Int A0_relax_type,
   NALU_HYPRE_Int A0_relax_times,
   NALU_HYPRE_Real *A0_l1_norms,
   NALU_HYPRE_Real A0_relax_weight,
   NALU_HYPRE_Real A0_omega,
   NALU_HYPRE_Real A0_max_eig_est,
   NALU_HYPRE_Real A0_min_eig_est,
   NALU_HYPRE_Int A0_cheby_order,
   NALU_HYPRE_Real A0_cheby_fraction,
   /* subspace matrices */
   nalu_hypre_ParCSRMatrix **A,
   /* subspace preconditioners */
   NALU_HYPRE_Solver *B,
   /* hypre solver functions for B */
   NALU_HYPRE_PtrToSolverFcn *HB,
   /* subspace interpolations */
   nalu_hypre_ParCSRMatrix **P,
   /* temporary subspace vectors */
   nalu_hypre_ParVector **r,
   nalu_hypre_ParVector **g,
   /* right-hand side */
   nalu_hypre_ParVector *x,
   /* current approximation */
   nalu_hypre_ParVector *y,
   /* current residual */
   nalu_hypre_ParVector *r0,
   /* temporary vector */
   nalu_hypre_ParVector *g0,
   char *cycle,
   /* temporary vector */
   nalu_hypre_ParVector *z)
{
   char *op;
   NALU_HYPRE_Int use_saved_residual = 0;

   for (op = cycle; *op != '\0'; op++)
   {
      /* do nothing */
      if (*op == ')')
      {
         continue;
      }

      /* compute the residual: r = x - Ay */
      else if (*op == '(')
      {
         nalu_hypre_ParVectorCopy(x, r0);
         nalu_hypre_ParCSRMatrixMatvec(-1.0, A0, y, 1.0, r0);
      }

      /* switch to additive correction */
      else if (*op == '+')
      {
         use_saved_residual = 1;
         continue;
      }

      /* smooth: y += S (x - Ay) */
      else if (*op == '0')
      {
         nalu_hypre_ParCSRRelax(A0, x,
                           A0_relax_type,
                           A0_relax_times,
                           A0_l1_norms,
                           A0_relax_weight,
                           A0_omega,
                           A0_max_eig_est,
                           A0_min_eig_est,
                           A0_cheby_order,
                           A0_cheby_fraction,
                           y, g0, z);
      }

      /* subspace correction: y += P B^{-1} P^t r */
      else
      {
         NALU_HYPRE_Int i = *op - '1';
         if (i < 0)
         {
            nalu_hypre_error_in_arg(16);
         }

         /* skip empty subspaces */
         if (!A[i]) { continue; }

         /* compute the residual? */
         if (use_saved_residual)
         {
            use_saved_residual = 0;
            nalu_hypre_ParCSRMatrixMatvecT(1.0, P[i], r0, 0.0, r[i]);
         }
         else
         {
            nalu_hypre_ParVectorCopy(x, g0);
            nalu_hypre_ParCSRMatrixMatvec(-1.0, A0, y, 1.0, g0);
            nalu_hypre_ParCSRMatrixMatvecT(1.0, P[i], g0, 0.0, r[i]);
         }

         nalu_hypre_ParVectorSetConstantValues(g[i], 0.0);
         (*HB[i]) (B[i], (NALU_HYPRE_Matrix)A[i],
                   (NALU_HYPRE_Vector)r[i], (NALU_HYPRE_Vector)g[i]);
         nalu_hypre_ParCSRMatrixMatvec(1.0, P[i], g[i], 0.0, g0);
         nalu_hypre_ParVectorAxpy(1.0, g0, y);
      }
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_AMSGetNumIterations
 *
 * Get the number of AMS iterations.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int nalu_hypre_AMSGetNumIterations(void *solver,
                                    NALU_HYPRE_Int *num_iterations)
{
   nalu_hypre_AMSData *ams_data = (nalu_hypre_AMSData *) solver;
   *num_iterations = ams_data -> num_iterations;
   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_AMSGetFinalRelativeResidualNorm
 *
 * Get the final relative residual norm in AMS.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int nalu_hypre_AMSGetFinalRelativeResidualNorm(void *solver,
                                                NALU_HYPRE_Real *rel_resid_norm)
{
   nalu_hypre_AMSData *ams_data = (nalu_hypre_AMSData *) solver;
   *rel_resid_norm = ams_data -> rel_resid_norm;
   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_AMSProjectOutGradients
 *
 * For problems with zero-conductivity regions, project the vector onto the
 * compatible subspace: x = (I - G0 (G0^t G0)^{-1} G0^T) x, where G0 is the
 * discrete gradient restricted to the interior nodes of the regions with
 * zero conductivity. This ensures that x is orthogonal to the gradients in
 * the range of G0.
 *
 * This function is typically called after the solution iteration is complete,
 * in order to facilitate the visualization of the computed field. Without it
 * the values in the zero-conductivity regions contain kernel components.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int nalu_hypre_AMSProjectOutGradients(void *solver,
                                       nalu_hypre_ParVector *x)
{
   nalu_hypre_AMSData *ams_data = (nalu_hypre_AMSData *) solver;

   if (ams_data -> B_G0)
   {
      nalu_hypre_ParCSRMatrixMatvecT(1.0, ams_data -> G0, x, 0.0, ams_data -> r1);
      nalu_hypre_ParVectorSetConstantValues(ams_data -> g1, 0.0);
      nalu_hypre_BoomerAMGSolve(ams_data -> B_G0, ams_data -> A_G0, ams_data -> r1, ams_data -> g1);
      nalu_hypre_ParCSRMatrixMatvec(1.0, ams_data -> G0, ams_data -> g1, 0.0, ams_data -> g0);
      nalu_hypre_ParVectorAxpy(-1.0, ams_data -> g0, x);
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_AMSConstructDiscreteGradient
 *
 * Construct and return the lowest-order discrete gradient matrix G, based on:
 * - a matrix on the egdes (e.g. the stiffness matrix A)
 * - a vector on the vertices (e.g. the x coordinates)
 * - the array edge_vertex, which lists the global indexes of the
 *   vertices of the local edges.
 *
 * We assume that edge_vertex lists the edge vertices consecutively,
 * and that the orientation of all edges is consistent. More specificaly:
 * If edge_orientation = 1, the edges are already oriented.
 * If edge_orientation = 2, the orientation of edge i depends only on the
 *                          sign of edge_vertex[2*i+1] - edge_vertex[2*i].
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int nalu_hypre_AMSConstructDiscreteGradient(nalu_hypre_ParCSRMatrix *A,
                                             nalu_hypre_ParVector *x_coord,
                                             NALU_HYPRE_BigInt *edge_vertex,
                                             NALU_HYPRE_Int edge_orientation,
                                             nalu_hypre_ParCSRMatrix **G_ptr)
{
   nalu_hypre_ParCSRMatrix *G;

   NALU_HYPRE_Int nedges;

   nedges = nalu_hypre_ParCSRMatrixNumRows(A);

   /* Construct the local part of G based on edge_vertex and the edge
      and vertex partitionings from A and x_coord */
   {
      NALU_HYPRE_Int i, *I = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  nedges + 1, NALU_HYPRE_MEMORY_HOST);
      NALU_HYPRE_Real *data = nalu_hypre_CTAlloc(NALU_HYPRE_Real,  2 * nedges, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_CSRMatrix *local = nalu_hypre_CSRMatrixCreate (nedges,
                                                      nalu_hypre_ParVectorGlobalSize(x_coord),
                                                      2 * nedges);

      for (i = 0; i <= nedges; i++)
      {
         I[i] = 2 * i;
      }

      if (edge_orientation == 1)
      {
         /* Assume that the edges are already oriented */
         for (i = 0; i < 2 * nedges; i += 2)
         {
            data[i]   = -1.0;
            data[i + 1] =  1.0;
         }
      }
      else if (edge_orientation == 2)
      {
         /* Assume that the edge orientation is based on the vertex indexes */
         for (i = 0; i < 2 * nedges; i += 2)
         {
            if (edge_vertex[i] < edge_vertex[i + 1])
            {
               data[i]   = -1.0;
               data[i + 1] =  1.0;
            }
            else
            {
               data[i]   =  1.0;
               data[i + 1] = -1.0;
            }
         }
      }
      else
      {
         nalu_hypre_error_in_arg(4);
      }

      nalu_hypre_CSRMatrixI(local) = I;
      nalu_hypre_CSRMatrixBigJ(local) = edge_vertex;
      nalu_hypre_CSRMatrixData(local) = data;

      nalu_hypre_CSRMatrixRownnz(local) = NULL;
      nalu_hypre_CSRMatrixOwnsData(local) = 1;
      nalu_hypre_CSRMatrixNumRownnz(local) = nedges;

      /* Generate the discrete gradient matrix */
      G = nalu_hypre_ParCSRMatrixCreate(nalu_hypre_ParCSRMatrixComm(A),
                                   nalu_hypre_ParCSRMatrixGlobalNumRows(A),
                                   nalu_hypre_ParVectorGlobalSize(x_coord),
                                   nalu_hypre_ParCSRMatrixRowStarts(A),
                                   nalu_hypre_ParVectorPartitioning(x_coord),
                                   0, 0, 0);
      nalu_hypre_CSRMatrixBigJtoJ(local);
      GenerateDiagAndOffd(local, G,
                          nalu_hypre_ParVectorFirstIndex(x_coord),
                          nalu_hypre_ParVectorLastIndex(x_coord));


      /* Account for empty rows in G. These may appear when A includes only
         the interior (non-Dirichlet b.c.) edges. */
      {
         nalu_hypre_CSRMatrix *G_diag = nalu_hypre_ParCSRMatrixDiag(G);
         G_diag->num_cols = nalu_hypre_VectorSize(nalu_hypre_ParVectorLocalVector(x_coord));
      }

      /* Free the local matrix */
      nalu_hypre_CSRMatrixDestroy(local);
   }

   *G_ptr = G;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_AMSFEISetup
 *
 * Construct an AMS solver object based on the following data:
 *
 *    A              - the edge element stiffness matrix
 *    num_vert       - number of vertices (nodes) in the processor
 *    num_local_vert - number of vertices owned by the processor
 *    vert_number    - global indexes of the vertices in the processor
 *    vert_coord     - coordinates of the vertices in the processor
 *    num_edges      - number of edges owned by the processor
 *    edge_vertex    - the vertices of the edges owned by the processor.
 *                     Vertices are in local numbering (the same as in
 *                     vert_number), and edge orientation is always from
 *                     the first to the second vertex.
 *
 * Here we distinguish between vertices that belong to elements in the
 * current processor, and the subset of these vertices that is owned by
 * the processor.
 *
 * This function is written specifically for input from the FEI and should
 * be called before nalu_hypre_AMSSetup().
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int nalu_hypre_AMSFEISetup(void *solver,
                            nalu_hypre_ParCSRMatrix *A,
                            nalu_hypre_ParVector *b,
                            nalu_hypre_ParVector *x,
                            NALU_HYPRE_Int num_vert,
                            NALU_HYPRE_Int num_local_vert,
                            NALU_HYPRE_BigInt *vert_number,
                            NALU_HYPRE_Real *vert_coord,
                            NALU_HYPRE_Int num_edges,
                            NALU_HYPRE_BigInt *edge_vertex)
{
   nalu_hypre_AMSData *ams_data = (nalu_hypre_AMSData *) solver;

   NALU_HYPRE_Int i, j;

   nalu_hypre_ParCSRMatrix *G;
   nalu_hypre_ParVector *x_coord, *y_coord, *z_coord;
   NALU_HYPRE_Real *x_data, *y_data, *z_data;

   MPI_Comm comm = nalu_hypre_ParCSRMatrixComm(A);
   NALU_HYPRE_BigInt vert_part[2], num_global_vert;
   NALU_HYPRE_BigInt vert_start, vert_end;
   NALU_HYPRE_BigInt big_local_vert = (NALU_HYPRE_BigInt) num_local_vert;

   /* Find the processor partitioning of the vertices */
   nalu_hypre_MPI_Scan(&big_local_vert, &vert_part[1], 1, NALU_HYPRE_MPI_BIG_INT, nalu_hypre_MPI_SUM, comm);
   vert_part[0] = vert_part[1] - big_local_vert;
   nalu_hypre_MPI_Allreduce(&big_local_vert, &num_global_vert, 1, NALU_HYPRE_MPI_BIG_INT, nalu_hypre_MPI_SUM, comm);

   /* Construct hypre parallel vectors for the vertex coordinates */
   x_coord = nalu_hypre_ParVectorCreate(comm, num_global_vert, vert_part);
   nalu_hypre_ParVectorInitialize(x_coord);
   nalu_hypre_ParVectorOwnsData(x_coord) = 1;
   x_data = nalu_hypre_VectorData(nalu_hypre_ParVectorLocalVector(x_coord));

   y_coord = nalu_hypre_ParVectorCreate(comm, num_global_vert, vert_part);
   nalu_hypre_ParVectorInitialize(y_coord);
   nalu_hypre_ParVectorOwnsData(y_coord) = 1;
   y_data = nalu_hypre_VectorData(nalu_hypre_ParVectorLocalVector(y_coord));

   z_coord = nalu_hypre_ParVectorCreate(comm, num_global_vert, vert_part);
   nalu_hypre_ParVectorInitialize(z_coord);
   nalu_hypre_ParVectorOwnsData(z_coord) = 1;
   z_data = nalu_hypre_VectorData(nalu_hypre_ParVectorLocalVector(z_coord));

   vert_start = nalu_hypre_ParVectorFirstIndex(x_coord);
   vert_end   = nalu_hypre_ParVectorLastIndex(x_coord);

   /* Save coordinates of locally owned vertices */
   for (i = 0; i < num_vert; i++)
   {
      if (vert_number[i] >= vert_start && vert_number[i] <= vert_end)
      {
         j = (NALU_HYPRE_Int)(vert_number[i] - vert_start);
         x_data[j] = vert_coord[3 * i];
         y_data[j] = vert_coord[3 * i + 1];
         z_data[j] = vert_coord[3 * i + 2];
      }
   }

   /* Change vertex numbers from local to global */
   for (i = 0; i < 2 * num_edges; i++)
   {
      edge_vertex[i] = vert_number[edge_vertex[i]];
   }

   /* Construct the local part of G based on edge_vertex */
   {
      /* NALU_HYPRE_Int num_edges = nalu_hypre_ParCSRMatrixNumRows(A); */
      NALU_HYPRE_Int *I = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_edges + 1, NALU_HYPRE_MEMORY_HOST);
      NALU_HYPRE_Real *data = nalu_hypre_CTAlloc(NALU_HYPRE_Real,  2 * num_edges, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_CSRMatrix *local = nalu_hypre_CSRMatrixCreate (num_edges,
                                                      num_global_vert,
                                                      2 * num_edges);

      for (i = 0; i <= num_edges; i++)
      {
         I[i] = 2 * i;
      }

      /* Assume that the edge orientation is based on the vertex indexes */
      for (i = 0; i < 2 * num_edges; i += 2)
      {
         data[i]   =  1.0;
         data[i + 1] = -1.0;
      }

      nalu_hypre_CSRMatrixI(local) = I;
      nalu_hypre_CSRMatrixBigJ(local) = edge_vertex;
      nalu_hypre_CSRMatrixData(local) = data;

      nalu_hypre_CSRMatrixRownnz(local) = NULL;
      nalu_hypre_CSRMatrixOwnsData(local) = 1;
      nalu_hypre_CSRMatrixNumRownnz(local) = num_edges;

      G = nalu_hypre_ParCSRMatrixCreate(comm,
                                   nalu_hypre_ParCSRMatrixGlobalNumRows(A),
                                   num_global_vert,
                                   nalu_hypre_ParCSRMatrixRowStarts(A),
                                   vert_part,
                                   0, 0, 0);
      nalu_hypre_CSRMatrixBigJtoJ(local);
      GenerateDiagAndOffd(local, G, vert_start, vert_end);

      //nalu_hypre_CSRMatrixJ(local) = NULL;
      nalu_hypre_CSRMatrixDestroy(local);
   }

   ams_data -> G = G;

   ams_data -> x = x_coord;
   ams_data -> y = y_coord;
   ams_data -> z = z_coord;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_AMSFEIDestroy
 *
 * Free the additional memory allocated in nalu_hypre_AMSFEISetup().
 *
 * This function is written specifically for input from the FEI and should
 * be called before nalu_hypre_AMSDestroy().
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int nalu_hypre_AMSFEIDestroy(void *solver)
{
   nalu_hypre_AMSData *ams_data = (nalu_hypre_AMSData *) solver;

   if (ams_data -> G)
   {
      nalu_hypre_ParCSRMatrixDestroy(ams_data -> G);
   }

   if (ams_data -> x)
   {
      nalu_hypre_ParVectorDestroy(ams_data -> x);
   }
   if (ams_data -> y)
   {
      nalu_hypre_ParVectorDestroy(ams_data -> y);
   }
   if (ams_data -> z)
   {
      nalu_hypre_ParVectorDestroy(ams_data -> z);
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ParCSRComputeL1Norms Threads
 *
 * Compute the l1 norms of the rows of a given matrix, depending on
 * the option parameter:
 *
 * option 1 = Compute the l1 norm of the rows
 * option 2 = Compute the l1 norm of the (processor) off-diagonal
 *            part of the rows plus the diagonal of A
 * option 3 = Compute the l2 norm^2 of the rows
 * option 4 = Truncated version of option 2 based on Remark 6.2 in "Multigrid
 *            Smoothers for Ultra-Parallel Computing"
 *
 * The above computations are done in a CF manner, whenever the provided
 * cf_marker is not NULL.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int nalu_hypre_ParCSRComputeL1NormsThreads(nalu_hypre_ParCSRMatrix *A,
                                            NALU_HYPRE_Int           option,
                                            NALU_HYPRE_Int           num_threads,
                                            NALU_HYPRE_Int          *cf_marker,
                                            NALU_HYPRE_Real        **l1_norm_ptr)
{
   NALU_HYPRE_Int i, j, k;
   NALU_HYPRE_Int num_rows = nalu_hypre_ParCSRMatrixNumRows(A);

   nalu_hypre_CSRMatrix *A_diag = nalu_hypre_ParCSRMatrixDiag(A);
   NALU_HYPRE_Int *A_diag_I = nalu_hypre_CSRMatrixI(A_diag);
   NALU_HYPRE_Int *A_diag_J = nalu_hypre_CSRMatrixJ(A_diag);
   NALU_HYPRE_Real *A_diag_data = nalu_hypre_CSRMatrixData(A_diag);

   nalu_hypre_CSRMatrix *A_offd = nalu_hypre_ParCSRMatrixOffd(A);
   NALU_HYPRE_Int *A_offd_I = nalu_hypre_CSRMatrixI(A_offd);
   NALU_HYPRE_Int *A_offd_J = nalu_hypre_CSRMatrixJ(A_offd);
   NALU_HYPRE_Real *A_offd_data = nalu_hypre_CSRMatrixData(A_offd);
   NALU_HYPRE_Int num_cols_offd = nalu_hypre_CSRMatrixNumCols(A_offd);

   NALU_HYPRE_Real diag;
   NALU_HYPRE_Real *l1_norm = nalu_hypre_TAlloc(NALU_HYPRE_Real, num_rows, nalu_hypre_ParCSRMatrixMemoryLocation(A));
   NALU_HYPRE_Int ii, ns, ne, rest, size;

   NALU_HYPRE_Int *cf_marker_offd = NULL;
   NALU_HYPRE_Int cf_diag;

   /* collect the cf marker data from other procs */
   if (cf_marker != NULL)
   {
      NALU_HYPRE_Int index;
      NALU_HYPRE_Int num_sends;
      NALU_HYPRE_Int start;
      NALU_HYPRE_Int *int_buf_data = NULL;

      nalu_hypre_ParCSRCommPkg  *comm_pkg = nalu_hypre_ParCSRMatrixCommPkg(A);
      nalu_hypre_ParCSRCommHandle *comm_handle;

      if (num_cols_offd)
      {
         cf_marker_offd = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_cols_offd, NALU_HYPRE_MEMORY_HOST);
      }
      num_sends = nalu_hypre_ParCSRCommPkgNumSends(comm_pkg);
      if (nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends))
         int_buf_data = nalu_hypre_CTAlloc(NALU_HYPRE_Int,
                                      nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg,  num_sends), NALU_HYPRE_MEMORY_HOST);
      index = 0;
      for (i = 0; i < num_sends; i++)
      {
         start = nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
         for (j = start; j < nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, i + 1); j++)
         {
            int_buf_data[index++] = cf_marker[nalu_hypre_ParCSRCommPkgSendMapElmt(comm_pkg, j)];
         }
      }
      comm_handle = nalu_hypre_ParCSRCommHandleCreate(11, comm_pkg, int_buf_data,
                                                 cf_marker_offd);
      nalu_hypre_ParCSRCommHandleDestroy(comm_handle);
      nalu_hypre_TFree(int_buf_data, NALU_HYPRE_MEMORY_HOST);
   }

#ifdef NALU_HYPRE_USING_OPENMP
   #pragma omp parallel for private(i,ii,j,k,ns,ne,rest,size,diag,cf_diag) NALU_HYPRE_SMP_SCHEDULE
#endif
   for (k = 0; k < num_threads; k++)
   {
      size = num_rows / num_threads;
      rest = num_rows - size * num_threads;
      if (k < rest)
      {
         ns = k * size + k;
         ne = (k + 1) * size + k + 1;
      }
      else
      {
         ns = k * size + rest;
         ne = (k + 1) * size + rest;
      }

      if (option == 1)
      {
         for (i = ns; i < ne; i++)
         {
            l1_norm[i] = 0.0;
            if (cf_marker == NULL)
            {
               /* Add the l1 norm of the diag part of the ith row */
               for (j = A_diag_I[i]; j < A_diag_I[i + 1]; j++)
               {
                  l1_norm[i] += nalu_hypre_abs(A_diag_data[j]);
               }
               /* Add the l1 norm of the offd part of the ith row */
               if (num_cols_offd)
               {
                  for (j = A_offd_I[i]; j < A_offd_I[i + 1]; j++)
                  {
                     l1_norm[i] += nalu_hypre_abs(A_offd_data[j]);
                  }
               }
            }
            else
            {
               cf_diag = cf_marker[i];
               /* Add the CF l1 norm of the diag part of the ith row */
               for (j = A_diag_I[i]; j < A_diag_I[i + 1]; j++)
                  if (cf_diag == cf_marker[A_diag_J[j]])
                  {
                     l1_norm[i] += nalu_hypre_abs(A_diag_data[j]);
                  }
               /* Add the CF l1 norm of the offd part of the ith row */
               if (num_cols_offd)
               {
                  for (j = A_offd_I[i]; j < A_offd_I[i + 1]; j++)
                     if (cf_diag == cf_marker_offd[A_offd_J[j]])
                     {
                        l1_norm[i] += nalu_hypre_abs(A_offd_data[j]);
                     }
               }
            }
         }
      }
      else if (option == 2)
      {
         for (i = ns; i < ne; i++)
         {
            l1_norm[i] = 0.0;
            if (cf_marker == NULL)
            {
               /* Add the diagonal and the local off-thread part of the ith row */
               for (j = A_diag_I[i]; j < A_diag_I[i + 1]; j++)
               {
                  ii = A_diag_J[j];
                  if (ii == i || ii < ns || ii >= ne)
                  {
                     l1_norm[i] += nalu_hypre_abs(A_diag_data[j]);
                  }
               }
               /* Add the l1 norm of the offd part of the ith row */
               if (num_cols_offd)
               {
                  for (j = A_offd_I[i]; j < A_offd_I[i + 1]; j++)
                  {
                     l1_norm[i] += nalu_hypre_abs(A_offd_data[j]);
                  }
               }
            }
            else
            {
               cf_diag = cf_marker[i];
               /* Add the diagonal and the local off-thread part of the ith row */
               for (j = A_diag_I[i]; j < A_diag_I[i + 1]; j++)
               {
                  ii = A_diag_J[j];
                  if ((ii == i || ii < ns || ii >= ne) &&
                      (cf_diag == cf_marker[A_diag_J[j]]))
                  {
                     l1_norm[i] += nalu_hypre_abs(A_diag_data[j]);
                  }
               }
               /* Add the CF l1 norm of the offd part of the ith row */
               if (num_cols_offd)
               {
                  for (j = A_offd_I[i]; j < A_offd_I[i + 1]; j++)
                     if (cf_diag == cf_marker_offd[A_offd_J[j]])
                     {
                        l1_norm[i] += nalu_hypre_abs(A_offd_data[j]);
                     }
               }
            }
         }
      }
      else if (option == 3)
      {
         for (i = ns; i < ne; i++)
         {
            l1_norm[i] = 0.0;
            for (j = A_diag_I[i]; j < A_diag_I[i + 1]; j++)
            {
               l1_norm[i] += A_diag_data[j] * A_diag_data[j];
            }
            if (num_cols_offd)
               for (j = A_offd_I[i]; j < A_offd_I[i + 1]; j++)
               {
                  l1_norm[i] += A_offd_data[j] * A_offd_data[j];
               }
         }
      }
      else if (option == 4)
      {
         for (i = ns; i < ne; i++)
         {
            l1_norm[i] = 0.0;
            if (cf_marker == NULL)
            {
               /* Add the diagonal and the local off-thread part of the ith row */
               for (j = A_diag_I[i]; j < A_diag_I[i + 1]; j++)
               {
                  ii = A_diag_J[j];
                  if (ii == i || ii < ns || ii >= ne)
                  {
                     if (ii == i)
                     {
                        diag = nalu_hypre_abs(A_diag_data[j]);
                        l1_norm[i] += nalu_hypre_abs(A_diag_data[j]);
                     }
                     else
                     {
                        l1_norm[i] += 0.5 * nalu_hypre_abs(A_diag_data[j]);
                     }
                  }
               }
               /* Add the l1 norm of the offd part of the ith row */
               if (num_cols_offd)
               {
                  for (j = A_offd_I[i]; j < A_offd_I[i + 1]; j++)
                  {
                     l1_norm[i] += 0.5 * nalu_hypre_abs(A_offd_data[j]);
                  }
               }
            }
            else
            {
               cf_diag = cf_marker[i];
               /* Add the diagonal and the local off-thread part of the ith row */
               for (j = A_diag_I[i]; j < A_diag_I[i + 1]; j++)
               {
                  ii = A_diag_J[j];
                  if ((ii == i || ii < ns || ii >= ne) &&
                      (cf_diag == cf_marker[A_diag_J[j]]))
                  {
                     if (ii == i)
                     {
                        diag = nalu_hypre_abs(A_diag_data[j]);
                        l1_norm[i] += nalu_hypre_abs(A_diag_data[j]);
                     }
                     else
                     {
                        l1_norm[i] += 0.5 * nalu_hypre_abs(A_diag_data[j]);
                     }
                  }
               }
               /* Add the CF l1 norm of the offd part of the ith row */
               if (num_cols_offd)
               {
                  for (j = A_offd_I[i]; j < A_offd_I[i + 1]; j++)
                     if (cf_diag == cf_marker_offd[A_offd_J[j]])
                     {
                        l1_norm[i] += 0.5 * nalu_hypre_abs(A_offd_data[j]);
                     }
               }
            }

            /* Truncate according to Remark 6.2 */
            if (l1_norm[i] <= 4.0 / 3.0 * diag)
            {
               l1_norm[i] = diag;
            }
         }
      }

      else if (option == 5) /*stores diagonal of A for Jacobi using matvec, rlx 7 */
      {
         /* Set the diag element */
         for (i = ns; i < ne; i++)
         {
            l1_norm[i] =  A_diag_data[A_diag_I[i]];
            if (l1_norm[i] == 0) { l1_norm[i] = 1.0; }
         }
      }

      if (option < 5)
      {
         /* Handle negative definite matrices */
         for (i = ns; i < ne; i++)
            if (A_diag_data[A_diag_I[i]] < 0)
            {
               l1_norm[i] = -l1_norm[i];
            }

         for (i = ns; i < ne; i++)
            /* if (nalu_hypre_abs(l1_norm[i]) < DBL_EPSILON) */
            if (nalu_hypre_abs(l1_norm[i]) == 0.0)
            {
               nalu_hypre_error_in_arg(1);
               break;
            }
      }

   }

   nalu_hypre_TFree(cf_marker_offd, NALU_HYPRE_MEMORY_HOST);

   *l1_norm_ptr = l1_norm;

   return nalu_hypre_error_flag;
}
