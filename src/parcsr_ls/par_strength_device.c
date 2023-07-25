/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_nalu_hypre_onedpl.hpp"
#include "_nalu_hypre_parcsr_ls.h"
#include "_nalu_hypre_utilities.hpp"

#if defined(NALU_HYPRE_USING_GPU)

__global__ void nalu_hypre_BoomerAMGCreateS_rowcount( nalu_hypre_DeviceItem &item,
                                                 NALU_HYPRE_Int nr_of_rows,
                                                 NALU_HYPRE_Real max_row_sum, NALU_HYPRE_Real strength_threshold,
                                                 NALU_HYPRE_Real* A_diag_data, NALU_HYPRE_Int* A_diag_i, NALU_HYPRE_Int* A_diag_j,
                                                 NALU_HYPRE_Real* A_offd_data, NALU_HYPRE_Int* A_offd_i, NALU_HYPRE_Int* A_offd_j,
                                                 NALU_HYPRE_Int* S_temp_diag_j, NALU_HYPRE_Int* S_temp_offd_j,
                                                 NALU_HYPRE_Int num_functions, NALU_HYPRE_Int* dof_func, NALU_HYPRE_Int* dof_func_offd,
                                                 NALU_HYPRE_Int* jS_diag, NALU_HYPRE_Int* jS_offd );
__global__ void nalu_hypre_BoomerAMGCreateSabs_rowcount( nalu_hypre_DeviceItem &item,
                                                    NALU_HYPRE_Int nr_of_rows,
                                                    NALU_HYPRE_Real max_row_sum, NALU_HYPRE_Real strength_threshold,
                                                    NALU_HYPRE_Real* A_diag_data, NALU_HYPRE_Int* A_diag_i, NALU_HYPRE_Int* A_diag_j,
                                                    NALU_HYPRE_Real* A_offd_data, NALU_HYPRE_Int* A_offd_i, NALU_HYPRE_Int* A_offd_j,
                                                    NALU_HYPRE_Int* S_temp_diag_j, NALU_HYPRE_Int* S_temp_offd_j,
                                                    NALU_HYPRE_Int num_functions, NALU_HYPRE_Int* dof_func, NALU_HYPRE_Int* dof_func_offd,
                                                    NALU_HYPRE_Int* jS_diag, NALU_HYPRE_Int* jS_offd );


/*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -*/
NALU_HYPRE_Int
nalu_hypre_BoomerAMGCreateSDevice(nalu_hypre_ParCSRMatrix    *A,
                             NALU_HYPRE_Int              abs_soc,
                             NALU_HYPRE_Real             strength_threshold,
                             NALU_HYPRE_Real             max_row_sum,
                             NALU_HYPRE_Int              num_functions,
                             NALU_HYPRE_Int             *dof_func,
                             nalu_hypre_ParCSRMatrix   **S_ptr)
{
#ifdef NALU_HYPRE_PROFILE
   nalu_hypre_profile_times[NALU_HYPRE_TIMER_ID_CREATES] -= nalu_hypre_MPI_Wtime();
#endif

   MPI_Comm                 comm            = nalu_hypre_ParCSRMatrixComm(A);
   nalu_hypre_ParCSRCommPkg     *comm_pkg        = nalu_hypre_ParCSRMatrixCommPkg(A);
   nalu_hypre_ParCSRCommHandle  *comm_handle;
   nalu_hypre_CSRMatrix         *A_diag          = nalu_hypre_ParCSRMatrixDiag(A);
   NALU_HYPRE_Int               *A_diag_i        = nalu_hypre_CSRMatrixI(A_diag);
   NALU_HYPRE_Real              *A_diag_data     = nalu_hypre_CSRMatrixData(A_diag);
   nalu_hypre_CSRMatrix         *A_offd          = nalu_hypre_ParCSRMatrixOffd(A);
   NALU_HYPRE_Int               *A_offd_i        = nalu_hypre_CSRMatrixI(A_offd);
   NALU_HYPRE_Real              *A_offd_data     = nalu_hypre_CSRMatrixData(A_offd);
   NALU_HYPRE_Int               *A_diag_j        = nalu_hypre_CSRMatrixJ(A_diag);
   NALU_HYPRE_Int               *A_offd_j        = nalu_hypre_CSRMatrixJ(A_offd);
   NALU_HYPRE_BigInt            *row_starts      = nalu_hypre_ParCSRMatrixRowStarts(A);
   NALU_HYPRE_Int                num_variables   = nalu_hypre_CSRMatrixNumRows(A_diag);
   NALU_HYPRE_BigInt             global_num_vars = nalu_hypre_ParCSRMatrixGlobalNumRows(A);
   NALU_HYPRE_Int                num_nonzeros_diag;
   NALU_HYPRE_Int                num_nonzeros_offd;
   NALU_HYPRE_Int                num_cols_offd = nalu_hypre_CSRMatrixNumCols(A_offd);
   nalu_hypre_ParCSRMatrix      *S;
   nalu_hypre_CSRMatrix         *S_diag;
   NALU_HYPRE_Int               *S_diag_i;
   NALU_HYPRE_Int               *S_diag_j, *S_temp_diag_j;
   /* NALU_HYPRE_Real           *S_diag_data; */
   nalu_hypre_CSRMatrix         *S_offd;
   NALU_HYPRE_Int               *S_offd_i = NULL;
   NALU_HYPRE_Int               *S_offd_j = NULL, *S_temp_offd_j = NULL;
   /* NALU_HYPRE_Real           *S_offd_data; */
   NALU_HYPRE_Int                ierr = 0;
   NALU_HYPRE_Int               *dof_func_offd_dev = NULL;
   NALU_HYPRE_Int                num_sends;

   NALU_HYPRE_MemoryLocation     memory_location = nalu_hypre_ParCSRMatrixMemoryLocation(A);

   /*--------------------------------------------------------------
    * Compute a  ParCSR strength matrix, S.
    *
    * Default "strength" of dependence/influence is defined in
    * the following way: i depends on j if
    *     aij > nalu_hypre_max (k != i) aik,    aii < 0
    * or
    *     aij < nalu_hypre_min (k != i) aik,    aii >= 0
    * Then S_ij = 1, else S_ij = 0.
    *
    * If abs_soc != 0, then use an absolute strength of connection:
    * i depends on j if
    *     abs(aij) > nalu_hypre_max (k != i) abs(aik)
    *
    * NOTE: the entries are negative initially, corresponding
    * to "unaccounted-for" dependence.
    *----------------------------------------------------------------*/

   num_nonzeros_diag = nalu_hypre_CSRMatrixNumNonzeros(A_diag);
   num_nonzeros_offd = nalu_hypre_CSRMatrixNumNonzeros(A_offd);

   S_diag_i = nalu_hypre_TAlloc(NALU_HYPRE_Int, num_variables + 1, memory_location);
   S_offd_i = nalu_hypre_TAlloc(NALU_HYPRE_Int, num_variables + 1, memory_location);
   S_temp_diag_j = nalu_hypre_TAlloc(NALU_HYPRE_Int, num_nonzeros_diag, NALU_HYPRE_MEMORY_DEVICE);
   S_temp_offd_j = nalu_hypre_TAlloc(NALU_HYPRE_Int, num_nonzeros_offd, NALU_HYPRE_MEMORY_DEVICE);

   if (num_functions > 1)
   {
      dof_func_offd_dev = nalu_hypre_TAlloc(NALU_HYPRE_Int, num_cols_offd, NALU_HYPRE_MEMORY_DEVICE);
   }

   /*-------------------------------------------------------------------
     * Get the dof_func data for the off-processor columns
     *-------------------------------------------------------------------*/
   if (!comm_pkg)
   {
      nalu_hypre_MatvecCommPkgCreate(A);
      comm_pkg = nalu_hypre_ParCSRMatrixCommPkg(A);
   }

   num_sends = nalu_hypre_ParCSRCommPkgNumSends(comm_pkg);

   if (num_functions > 1)
   {
      NALU_HYPRE_Int *int_buf_data = nalu_hypre_TAlloc(NALU_HYPRE_Int, nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg,
                                                                                        num_sends), NALU_HYPRE_MEMORY_DEVICE);

      nalu_hypre_ParCSRCommPkgCopySendMapElmtsToDevice(comm_pkg);
#if defined(NALU_HYPRE_USING_SYCL)
      hypreSycl_gather( nalu_hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg),
                        nalu_hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg) +
                        nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends),
                        dof_func,
                        int_buf_data );
#else
      NALU_HYPRE_THRUST_CALL( gather,
                         nalu_hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg),
                         nalu_hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg) +
                         nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends),
                         dof_func,
                         int_buf_data );
#endif

#if defined(NALU_HYPRE_WITH_GPU_AWARE_MPI) && defined(NALU_HYPRE_USING_THRUST_NOSYNC)
      /* RL: make sure int_buf_data is ready before issuing GPU-GPU MPI */
      nalu_hypre_ForceSyncComputeStream(nalu_hypre_handle());
#endif

      comm_handle = nalu_hypre_ParCSRCommHandleCreate_v2(11, comm_pkg, NALU_HYPRE_MEMORY_DEVICE, int_buf_data,
                                                    NALU_HYPRE_MEMORY_DEVICE, dof_func_offd_dev);
      nalu_hypre_ParCSRCommHandleDestroy(comm_handle);

      nalu_hypre_TFree(int_buf_data, NALU_HYPRE_MEMORY_DEVICE);
   }

   /* count the row nnz of S */
   dim3 bDim = nalu_hypre_GetDefaultDeviceBlockDimension();
   dim3 gDim = nalu_hypre_GetDefaultDeviceGridDimension(num_variables, "warp", bDim);

   if (abs_soc)
   {
      NALU_HYPRE_GPU_LAUNCH( nalu_hypre_BoomerAMGCreateSabs_rowcount, gDim, bDim,
                        num_variables, max_row_sum, strength_threshold,
                        A_diag_data, A_diag_i, A_diag_j,
                        A_offd_data, A_offd_i, A_offd_j,
                        S_temp_diag_j, S_temp_offd_j,
                        num_functions, dof_func, dof_func_offd_dev,
                        S_diag_i, S_offd_i );
   }
   else
   {
      NALU_HYPRE_GPU_LAUNCH( nalu_hypre_BoomerAMGCreateS_rowcount, gDim, bDim,
                        num_variables, max_row_sum, strength_threshold,
                        A_diag_data, A_diag_i, A_diag_j,
                        A_offd_data, A_offd_i, A_offd_j,
                        S_temp_diag_j, S_temp_offd_j,
                        num_functions, dof_func, dof_func_offd_dev,
                        S_diag_i, S_offd_i );
   }

   nalu_hypre_Memset(S_diag_i + num_variables, 0, sizeof(NALU_HYPRE_Int), NALU_HYPRE_MEMORY_DEVICE);
   nalu_hypre_Memset(S_offd_i + num_variables, 0, sizeof(NALU_HYPRE_Int), NALU_HYPRE_MEMORY_DEVICE);

   hypreDevice_IntegerExclusiveScan(num_variables + 1, S_diag_i);
   hypreDevice_IntegerExclusiveScan(num_variables + 1, S_offd_i);

   NALU_HYPRE_Int *tmp, S_num_nonzeros_diag, S_num_nonzeros_offd;

   nalu_hypre_TMemcpy(&S_num_nonzeros_diag, &S_diag_i[num_variables], NALU_HYPRE_Int, 1, NALU_HYPRE_MEMORY_HOST,
                 memory_location);
   nalu_hypre_TMemcpy(&S_num_nonzeros_offd, &S_offd_i[num_variables], NALU_HYPRE_Int, 1, NALU_HYPRE_MEMORY_HOST,
                 memory_location);

   S_diag_j = nalu_hypre_TAlloc(NALU_HYPRE_Int, S_num_nonzeros_diag, memory_location);
   S_offd_j = nalu_hypre_TAlloc(NALU_HYPRE_Int, S_num_nonzeros_offd, memory_location);

#if defined(NALU_HYPRE_USING_SYCL)
   tmp = NALU_HYPRE_ONEDPL_CALL(std::copy_if, S_temp_diag_j, S_temp_diag_j + num_nonzeros_diag, S_diag_j,
                           is_nonnegative<NALU_HYPRE_Int>());
#else
   tmp = NALU_HYPRE_THRUST_CALL(copy_if, S_temp_diag_j, S_temp_diag_j + num_nonzeros_diag, S_diag_j,
                           is_nonnegative<NALU_HYPRE_Int>());
#endif

   nalu_hypre_assert(S_num_nonzeros_diag == tmp - S_diag_j);

#if defined(NALU_HYPRE_USING_SYCL)
   tmp = NALU_HYPRE_ONEDPL_CALL(std::copy_if, S_temp_offd_j, S_temp_offd_j + num_nonzeros_offd, S_offd_j,
                           is_nonnegative<NALU_HYPRE_Int>());
#else
   tmp = NALU_HYPRE_THRUST_CALL(copy_if, S_temp_offd_j, S_temp_offd_j + num_nonzeros_offd, S_offd_j,
                           is_nonnegative<NALU_HYPRE_Int>());
#endif

   nalu_hypre_assert(S_num_nonzeros_offd == tmp - S_offd_j);

   S = nalu_hypre_ParCSRMatrixCreate(comm, global_num_vars, global_num_vars, row_starts, row_starts,
                                num_cols_offd, num_nonzeros_diag, num_nonzeros_offd);

   S_diag = nalu_hypre_ParCSRMatrixDiag(S);
   S_offd = nalu_hypre_ParCSRMatrixOffd(S);

   nalu_hypre_CSRMatrixNumNonzeros(S_diag) = S_num_nonzeros_diag;
   nalu_hypre_CSRMatrixNumNonzeros(S_offd) = S_num_nonzeros_offd;
   nalu_hypre_CSRMatrixI(S_diag) = S_diag_i;
   nalu_hypre_CSRMatrixJ(S_diag) = S_diag_j;
   nalu_hypre_CSRMatrixI(S_offd) = S_offd_i;
   nalu_hypre_CSRMatrixJ(S_offd) = S_offd_j;
   nalu_hypre_CSRMatrixMemoryLocation(S_diag) = memory_location;
   nalu_hypre_CSRMatrixMemoryLocation(S_offd) = memory_location;

   nalu_hypre_ParCSRMatrixCommPkg(S) = NULL;

   nalu_hypre_ParCSRMatrixColMapOffd(S) = nalu_hypre_TAlloc(NALU_HYPRE_BigInt, num_cols_offd, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TMemcpy(nalu_hypre_ParCSRMatrixColMapOffd(S), nalu_hypre_ParCSRMatrixColMapOffd(A),
                 NALU_HYPRE_BigInt, num_cols_offd, NALU_HYPRE_MEMORY_HOST, NALU_HYPRE_MEMORY_HOST);

   nalu_hypre_ParCSRMatrixSocDiagJ(S) = S_temp_diag_j;
   nalu_hypre_ParCSRMatrixSocOffdJ(S) = S_temp_offd_j;

   *S_ptr = S;

   nalu_hypre_TFree(dof_func_offd_dev, NALU_HYPRE_MEMORY_DEVICE);
   /*
   nalu_hypre_TFree(S_temp_diag_j,     NALU_HYPRE_MEMORY_DEVICE);
   nalu_hypre_TFree(S_temp_offd_j,     NALU_HYPRE_MEMORY_DEVICE);
   */

#ifdef NALU_HYPRE_PROFILE
   nalu_hypre_profile_times[NALU_HYPRE_TIMER_ID_CREATES] += nalu_hypre_MPI_Wtime();
#endif

   return (ierr);
}

/*-----------------------------------------------------------------------*/
__global__ void nalu_hypre_BoomerAMGCreateS_rowcount( nalu_hypre_DeviceItem &item,
                                                 NALU_HYPRE_Int   nr_of_rows,
                                                 NALU_HYPRE_Real  max_row_sum,
                                                 NALU_HYPRE_Real  strength_threshold,
                                                 NALU_HYPRE_Real *A_diag_data,
                                                 NALU_HYPRE_Int  *A_diag_i,
                                                 NALU_HYPRE_Int  *A_diag_j,
                                                 NALU_HYPRE_Real *A_offd_data,
                                                 NALU_HYPRE_Int  *A_offd_i,
                                                 NALU_HYPRE_Int  *A_offd_j,
                                                 NALU_HYPRE_Int  *S_temp_diag_j,
                                                 NALU_HYPRE_Int  *S_temp_offd_j,
                                                 NALU_HYPRE_Int   num_functions,
                                                 NALU_HYPRE_Int  *dof_func,
                                                 NALU_HYPRE_Int  *dof_func_offd,
                                                 NALU_HYPRE_Int  *jS_diag,
                                                 NALU_HYPRE_Int  *jS_offd )
{
   /*-----------------------------------------------------------------------*/
   /*
      Input: nr_of_rows - Number of rows in matrix (local in processor)
             A_diag_data, A_diag_i, A_diag_j - CSR representation of A_diag
             A_offd_data, A_offd_i, A_offd_j - CSR representation of A_offd
             num_function  - Number of degrees of freedom per grid point
             dof_func      - vector over nonzero elements of A_diag, indicating the degree of freedom
             dof_func_offd - vector over nonzero elements of A_offd, indicating the degree of freedom

      Output: S_temp_diag_j - S_diag_j vector before compression, i.e.,elements that are < 0 should be removed
                              strong connections: same as A_diag_j; weak: -1; diagonal: -2
              S_temp_offd_j - S_offd_j vector before compression, i.e.,elements that are < 0 should be removed
                              strong connections: same as A_offd_j; weak: -1;
              jS_diag       - row nnz vector for compressed S_diag
              jS_offd       - row nnz vector for compressed S_offd
    */
   /*-----------------------------------------------------------------------*/

   NALU_HYPRE_Real row_scale = 0.0, row_sum = 0.0, row_max = 0.0, row_min = 0.0, diag = 0.0;
   NALU_HYPRE_Int row_nnz_diag = 0, row_nnz_offd = 0, diag_pos = -1;

   NALU_HYPRE_Int row = nalu_hypre_gpu_get_grid_warp_id<1, 1>(item);

   if (row >= nr_of_rows)
   {
      return;
   }

   NALU_HYPRE_Int lane = nalu_hypre_gpu_get_lane_id<1>(item);
   NALU_HYPRE_Int p_diag = 0, q_diag, p_offd = 0, q_offd;

   /* diag part */
   if (lane < 2)
   {
      p_diag = read_only_load(A_diag_i + row + lane);
   }
   q_diag = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, p_diag, 1);
   p_diag = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, p_diag, 0);

   for (NALU_HYPRE_Int i = p_diag + lane; i < q_diag; i += NALU_HYPRE_WARP_SIZE)
   {
      const NALU_HYPRE_Int col = read_only_load(&A_diag_j[i]);

      if ( num_functions == 1 || row == col ||
           read_only_load(&dof_func[row]) == read_only_load(&dof_func[col]) )
      {
         const NALU_HYPRE_Real v = read_only_load(&A_diag_data[i]);
         row_sum += v;
         if (row == col)
         {
            diag = v;
            diag_pos = i;
         }
         else
         {
            row_max = nalu_hypre_max(row_max, v);
            row_min = nalu_hypre_min(row_min, v);
         }
      }
   }

   /* offd part */
   if (lane < 2)
   {
      p_offd = read_only_load(A_offd_i + row + lane);
   }
   q_offd = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, p_offd, 1);
   p_offd = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, p_offd, 0);

   for (NALU_HYPRE_Int i = p_offd + lane; i < q_offd; i += NALU_HYPRE_WARP_SIZE)
   {
      if ( num_functions == 1 ||
           read_only_load(&dof_func[row]) == read_only_load(&dof_func_offd[read_only_load(&A_offd_j[i])]) )
      {
         const NALU_HYPRE_Real v = read_only_load(&A_offd_data[i]);
         row_sum += v;
         row_max = nalu_hypre_max(row_max, v);
         row_min = nalu_hypre_min(row_min, v);
      }
   }

   diag = warp_allreduce_sum(item, diag);

   /* sign of diag */
   const NALU_HYPRE_Int sdiag = diag > 0.0 ? 1 : -1;

   /* compute scaling factor and row sum */
   row_sum = warp_allreduce_sum(item, row_sum);

   if (diag > 0.0)
   {
      row_scale = warp_allreduce_min(item, row_min);
   }
   else
   {
      row_scale = warp_allreduce_max(item, row_max);
   }

   /* compute row of S */
   NALU_HYPRE_Int all_weak = max_row_sum < 1.0 && nalu_hypre_abs(row_sum) > nalu_hypre_abs(diag) * max_row_sum;
   const NALU_HYPRE_Real thresh = sdiag * strength_threshold * row_scale;

   for (NALU_HYPRE_Int i = p_diag + lane; i < q_diag; i += NALU_HYPRE_WARP_SIZE)
   {
      const NALU_HYPRE_Int cond = all_weak == 0 && diag_pos != i &&
                             ( num_functions == 1 || read_only_load(&dof_func[row]) ==
                               read_only_load(&dof_func[read_only_load(&A_diag_j[i])]) ) &&
                             sdiag * read_only_load(&A_diag_data[i]) < thresh;
      S_temp_diag_j[i] = cond * (1 + read_only_load(&A_diag_j[i])) - 1;
      row_nnz_diag += cond;
   }

   /* !!! mark diagonal as -2 !!! */
   if (diag_pos >= 0)
   {
      S_temp_diag_j[diag_pos] = -2;
   }

   for (NALU_HYPRE_Int i = p_offd + lane; i < q_offd; i += NALU_HYPRE_WARP_SIZE)
   {
      const NALU_HYPRE_Int cond = all_weak == 0 &&
                             ( num_functions == 1 || read_only_load(&dof_func[row]) ==
                               read_only_load(&dof_func_offd[read_only_load(&A_offd_j[i])]) ) &&
                             sdiag * read_only_load(&A_offd_data[i]) < thresh;
      S_temp_offd_j[i] = cond * (1 + read_only_load(&A_offd_j[i])) - 1;
      row_nnz_offd += cond;
   }

   row_nnz_diag = warp_reduce_sum(item, row_nnz_diag);
   row_nnz_offd = warp_reduce_sum(item, row_nnz_offd);

   if (0 == lane)
   {
      jS_diag[row] = row_nnz_diag;
      jS_offd[row] = row_nnz_offd;
   }
}

NALU_HYPRE_Int
nalu_hypre_BoomerAMGMakeSocFromSDevice( nalu_hypre_ParCSRMatrix *A,
                                   nalu_hypre_ParCSRMatrix *S)
{
   if (!nalu_hypre_ParCSRMatrixSocDiagJ(S))
   {
      nalu_hypre_CSRMatrix *A_diag = nalu_hypre_ParCSRMatrixDiag(A);
      nalu_hypre_CSRMatrix *S_diag = nalu_hypre_ParCSRMatrixDiag(S);
      NALU_HYPRE_Int nnz_diag = nalu_hypre_CSRMatrixNumNonzeros(A_diag);
      NALU_HYPRE_Int *soc_diag = nalu_hypre_TAlloc(NALU_HYPRE_Int, nnz_diag, NALU_HYPRE_MEMORY_DEVICE);
      nalu_hypre_CSRMatrixIntersectPattern(A_diag, S_diag, soc_diag, 1);
      nalu_hypre_ParCSRMatrixSocDiagJ(S) = soc_diag;
   }

   if (!nalu_hypre_ParCSRMatrixSocOffdJ(S))
   {
      nalu_hypre_CSRMatrix *A_offd = nalu_hypre_ParCSRMatrixOffd(A);
      nalu_hypre_CSRMatrix *S_offd = nalu_hypre_ParCSRMatrixOffd(S);
      NALU_HYPRE_Int nnz_offd = nalu_hypre_CSRMatrixNumNonzeros(A_offd);
      NALU_HYPRE_Int *soc_offd = nalu_hypre_TAlloc(NALU_HYPRE_Int, nnz_offd, NALU_HYPRE_MEMORY_DEVICE);
      nalu_hypre_CSRMatrixIntersectPattern(A_offd, S_offd, soc_offd, 0);
      nalu_hypre_ParCSRMatrixSocOffdJ(S) = soc_offd;
   }

   return nalu_hypre_error_flag;
}

/*-----------------------------------------------------------------------*/
__global__ void nalu_hypre_BoomerAMGCreateSabs_rowcount( nalu_hypre_DeviceItem &item,
                                                    NALU_HYPRE_Int   nr_of_rows,
                                                    NALU_HYPRE_Real  max_row_sum,
                                                    NALU_HYPRE_Real  strength_threshold,
                                                    NALU_HYPRE_Real *A_diag_data,
                                                    NALU_HYPRE_Int  *A_diag_i,
                                                    NALU_HYPRE_Int  *A_diag_j,
                                                    NALU_HYPRE_Real *A_offd_data,
                                                    NALU_HYPRE_Int  *A_offd_i,
                                                    NALU_HYPRE_Int  *A_offd_j,
                                                    NALU_HYPRE_Int  *S_temp_diag_j,
                                                    NALU_HYPRE_Int  *S_temp_offd_j,
                                                    NALU_HYPRE_Int   num_functions,
                                                    NALU_HYPRE_Int  *dof_func,
                                                    NALU_HYPRE_Int  *dof_func_offd,
                                                    NALU_HYPRE_Int  *jS_diag,
                                                    NALU_HYPRE_Int  *jS_offd )
{
   /*-----------------------------------------------------------------------*/
   /*
      Input: nr_of_rows - Number of rows in matrix (local in processor)
             A_diag_data, A_diag_i, A_diag_j - CSR representation of A_diag
             A_offd_data, A_offd_i, A_offd_j - CSR representation of A_offd
             num_function  - Number of degrees of freedom per grid point
             dof_func      - vector over nonzero elements of A_diag, indicating the degree of freedom
             dof_func_offd - vector over nonzero elements of A_offd, indicating the degree of freedom

      Output: S_temp_diag_j - S_diag_j vector before compression, i.e.,elements that are < 0 should be removed
                              strong connections: same as A_diag_j; weak: -1; diagonal: -2
              S_temp_offd_j - S_offd_j vector before compression, i.e.,elements that are < 0 should be removed
                              strong connections: same as A_offd_j; weak: -1;
              jS_diag       - row nnz vector for compressed S_diag
              jS_offd       - row nnz vector for compressed S_offd
    */
   /*-----------------------------------------------------------------------*/

   NALU_HYPRE_Real row_scale = 0.0, row_sum = 0.0, diag = 0.0;
   NALU_HYPRE_Int row_nnz_diag = 0, row_nnz_offd = 0, diag_pos = -1;

   NALU_HYPRE_Int row = nalu_hypre_gpu_get_grid_warp_id<1, 1>(item);

   if (row >= nr_of_rows)
   {
      return;
   }

   NALU_HYPRE_Int lane = nalu_hypre_gpu_get_lane_id<1>(item);
   NALU_HYPRE_Int p_diag = 0, q_diag, p_offd = 0, q_offd;

   /* diag part */
   if (lane < 2)
   {
      p_diag = read_only_load(A_diag_i + row + lane);
   }
   q_diag = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, p_diag, 1);
   p_diag = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, p_diag, 0);

   for (NALU_HYPRE_Int i = p_diag + lane; i < q_diag; i += NALU_HYPRE_WARP_SIZE)
   {
      const NALU_HYPRE_Int col = read_only_load(&A_diag_j[i]);

      if ( num_functions == 1 || row == col ||
           read_only_load(&dof_func[row]) == read_only_load(&dof_func[col]) )
      {
         const NALU_HYPRE_Real v = nalu_hypre_cabs( read_only_load(&A_diag_data[i]) );
         row_sum += v;
         if (row == col)
         {
            diag = v;
            diag_pos = i;
         }
         else
         {
            row_scale = nalu_hypre_max(row_scale, v);
         }
      }
   }

   /* offd part */
   if (lane < 2)
   {
      p_offd = read_only_load(A_offd_i + row + lane);
   }
   q_offd = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, p_offd, 1);
   p_offd = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, p_offd, 0);

   for (NALU_HYPRE_Int i = p_offd + lane; i < q_offd; i += NALU_HYPRE_WARP_SIZE)
   {
      if ( num_functions == 1 ||
           read_only_load(&dof_func[row]) == read_only_load(&dof_func_offd[read_only_load(&A_offd_j[i])]) )
      {
         const NALU_HYPRE_Real v = nalu_hypre_cabs( read_only_load(&A_offd_data[i]) );
         row_sum += v;
         row_scale = nalu_hypre_max(row_scale, v);
      }
   }

   diag = warp_allreduce_sum(item, diag);

   /* compute scaling factor and row sum */
   row_sum = warp_allreduce_sum(item, row_sum);
   row_scale = warp_allreduce_max(item, row_scale);

   /* compute row of S */
   NALU_HYPRE_Int all_weak = max_row_sum < 1.0 &&
                        nalu_hypre_abs(row_sum) < nalu_hypre_abs(diag) * (2.0 - max_row_sum);
   const NALU_HYPRE_Real thresh = strength_threshold * row_scale;

   for (NALU_HYPRE_Int i = p_diag + lane; i < q_diag;
        i += NALU_HYPRE_WARP_SIZE)
   {
      const NALU_HYPRE_Int cond = all_weak == 0 && diag_pos != i &&
                             ( num_functions == 1 || read_only_load(&dof_func[row]) ==
                               read_only_load(&dof_func[read_only_load(&A_diag_j[i])]) ) &&
                             nalu_hypre_cabs( read_only_load(&A_diag_data[i]) ) > thresh;
      S_temp_diag_j[i] = cond * (1 + read_only_load(&A_diag_j[i])) - 1;
      row_nnz_diag += cond;
   }

   /* !!! mark diagonal as -2 !!! */
   if (diag_pos >= 0)
   {
      S_temp_diag_j[diag_pos] = -2;
   }

   for (NALU_HYPRE_Int i = p_offd + lane; i < q_offd; i += NALU_HYPRE_WARP_SIZE)
   {
      const NALU_HYPRE_Int cond = all_weak == 0 &&
                             ( num_functions == 1 || read_only_load(&dof_func[row]) ==
                               read_only_load(&dof_func_offd[read_only_load(&A_offd_j[i])]) ) &&
                             nalu_hypre_cabs( read_only_load(&A_offd_data[i]) ) > thresh;
      S_temp_offd_j[i] = cond * (1 + read_only_load(&A_offd_j[i])) - 1;
      row_nnz_offd += cond;
   }

   row_nnz_diag = warp_reduce_sum(item, row_nnz_diag);
   row_nnz_offd = warp_reduce_sum(item, row_nnz_offd);

   if (0 == lane)
   {
      jS_diag[row] = row_nnz_diag;
      jS_offd[row] = row_nnz_offd;
   }
}

/*--------------------------------------------------------------------------
 * nalu_hypre_BoomerAMGCorrectCFMarker : corrects CF_marker after aggr. coarsening
 *--------------------------------------------------------------------------*/
NALU_HYPRE_Int
nalu_hypre_BoomerAMGCorrectCFMarkerDevice(nalu_hypre_IntArray *CF_marker, nalu_hypre_IntArray *new_CF_marker)
{

   NALU_HYPRE_Int n_fine     = nalu_hypre_IntArraySize(CF_marker);
   NALU_HYPRE_Int n_coarse   = nalu_hypre_IntArraySize(new_CF_marker);

   NALU_HYPRE_Int *indices   = nalu_hypre_CTAlloc(NALU_HYPRE_Int, n_coarse, NALU_HYPRE_MEMORY_DEVICE);
   NALU_HYPRE_Int *CF_C      = nalu_hypre_CTAlloc(NALU_HYPRE_Int, n_coarse, NALU_HYPRE_MEMORY_DEVICE);

#if defined(NALU_HYPRE_USING_SYCL)
   /* save CF_marker values at C points in CF_C and C point indices */
   NALU_HYPRE_ONEDPL_CALL( std::copy_if,
                      nalu_hypre_IntArrayData(CF_marker),
                      nalu_hypre_IntArrayData(CF_marker) + n_fine,
                      CF_C,
                      is_positive<NALU_HYPRE_Int>() );
   hypreSycl_copy_if( oneapi::dpl::counting_iterator<NALU_HYPRE_Int>(0),
                      oneapi::dpl::counting_iterator<NALU_HYPRE_Int>(n_fine),
                      nalu_hypre_IntArrayData(CF_marker),
                      indices,
                      is_positive<NALU_HYPRE_Int>() );

   /* replace CF_marker at C points with 1 */
   NALU_HYPRE_ONEDPL_CALL( std::replace_if,
                      nalu_hypre_IntArrayData(CF_marker),
                      nalu_hypre_IntArrayData(CF_marker) + n_fine,
                      is_positive<NALU_HYPRE_Int>(),
                      1 );

   /* update with new_CF_marker wherever C point value was initially 1 */
   hypreSycl_scatter_if( nalu_hypre_IntArrayData(new_CF_marker),
                         nalu_hypre_IntArrayData(new_CF_marker) + n_coarse,
                         indices,
                         CF_C,
                         nalu_hypre_IntArrayData(CF_marker),
                         equal<NALU_HYPRE_Int>(1) );
#else
   /* save CF_marker values at C points in CF_C and C point indices */
   NALU_HYPRE_THRUST_CALL( copy_if,
                      nalu_hypre_IntArrayData(CF_marker),
                      nalu_hypre_IntArrayData(CF_marker) + n_fine,
                      CF_C,
                      is_positive<NALU_HYPRE_Int>() );
   NALU_HYPRE_THRUST_CALL( copy_if,
                      thrust::counting_iterator<NALU_HYPRE_Int>(0),
                      thrust::counting_iterator<NALU_HYPRE_Int>(n_fine),
                      nalu_hypre_IntArrayData(CF_marker),
                      indices,
                      is_positive<NALU_HYPRE_Int>() );

   /* replace CF_marker at C points with 1 */
   NALU_HYPRE_THRUST_CALL( replace_if,
                      nalu_hypre_IntArrayData(CF_marker),
                      nalu_hypre_IntArrayData(CF_marker) + n_fine,
                      is_positive<NALU_HYPRE_Int>(),
                      1 );

   /* update with new_CF_marker wherever C point value was initially 1 */
   NALU_HYPRE_THRUST_CALL( scatter_if,
                      nalu_hypre_IntArrayData(new_CF_marker),
                      nalu_hypre_IntArrayData(new_CF_marker) + n_coarse,
                      indices,
                      CF_C,
                      nalu_hypre_IntArrayData(CF_marker),
                      equal<NALU_HYPRE_Int>(1) );
#endif

   nalu_hypre_TFree(indices, NALU_HYPRE_MEMORY_DEVICE);
   nalu_hypre_TFree(CF_C, NALU_HYPRE_MEMORY_DEVICE);

   return 0;
}
/*--------------------------------------------------------------------------
 * nalu_hypre_BoomerAMGCorrectCFMarker2 : corrects CF_marker after aggr. coarsening,
 * but marks new F-points (previous C-points) as -2
 *--------------------------------------------------------------------------*/
NALU_HYPRE_Int
nalu_hypre_BoomerAMGCorrectCFMarker2Device(nalu_hypre_IntArray *CF_marker, nalu_hypre_IntArray *new_CF_marker)
{

   NALU_HYPRE_Int n_fine     = nalu_hypre_IntArraySize(CF_marker);
   NALU_HYPRE_Int n_coarse   = nalu_hypre_IntArraySize(new_CF_marker);

   NALU_HYPRE_Int *indices   = nalu_hypre_CTAlloc(NALU_HYPRE_Int, n_coarse, NALU_HYPRE_MEMORY_DEVICE);

#if defined(NALU_HYPRE_USING_SYCL)
   /* save C point indices */
   hypreSycl_copy_if( oneapi::dpl::counting_iterator<NALU_HYPRE_Int>(0),
                      oneapi::dpl::counting_iterator<NALU_HYPRE_Int>(n_fine),
                      nalu_hypre_IntArrayData(CF_marker),
                      indices,
                      is_positive<NALU_HYPRE_Int>() );

   /* replace CF_marker at C points with 1 */
   NALU_HYPRE_ONEDPL_CALL( std::replace_if,
                      nalu_hypre_IntArrayData(CF_marker),
                      nalu_hypre_IntArrayData(CF_marker) + n_fine,
                      is_positive<NALU_HYPRE_Int>(),
                      1 );

   /* update values in CF_marker to -2 wherever new_CF_marker == -1 */
   hypreSycl_transform_if( oneapi::dpl::make_permutation_iterator(nalu_hypre_IntArrayData(CF_marker),
                                                                  indices),
                           oneapi::dpl::make_permutation_iterator(nalu_hypre_IntArrayData(CF_marker), indices) + n_coarse,
                           nalu_hypre_IntArrayData(new_CF_marker),
                           oneapi::dpl::make_permutation_iterator(nalu_hypre_IntArrayData(CF_marker), indices),
   [] (const auto & x) { return -2; },
   equal<NALU_HYPRE_Int>(-1) );
#else
   /* save C point indices */
   NALU_HYPRE_THRUST_CALL( copy_if,
                      thrust::counting_iterator<NALU_HYPRE_Int>(0),
                      thrust::counting_iterator<NALU_HYPRE_Int>(n_fine),
                      nalu_hypre_IntArrayData(CF_marker),
                      indices,
                      is_positive<NALU_HYPRE_Int>() );

   /* replace CF_marker at C points with 1 */
   NALU_HYPRE_THRUST_CALL( replace_if,
                      nalu_hypre_IntArrayData(CF_marker),
                      nalu_hypre_IntArrayData(CF_marker) + n_fine,
                      is_positive<NALU_HYPRE_Int>(),
                      1 );

   /* update values in CF_marker to -2 wherever new_CF_marker == -1 */
   NALU_HYPRE_THRUST_CALL( scatter_if,
                      thrust::make_constant_iterator(-2),
                      thrust::make_constant_iterator(-2) + n_coarse,
                      indices,
                      nalu_hypre_IntArrayData(new_CF_marker),
                      nalu_hypre_IntArrayData(CF_marker),
                      equal<NALU_HYPRE_Int>(-1) );
#endif

   nalu_hypre_TFree(indices, NALU_HYPRE_MEMORY_DEVICE);

   return 0;
}

#endif /* #if defined(NALU_HYPRE_USING_GPU) */
