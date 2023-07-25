/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_nalu_hypre_onedpl.hpp"
#include "_nalu_hypre_parcsr_ls.h"
#include "_nalu_hypre_utilities.hpp"

#if defined(NALU_HYPRE_USING_GPU)

#if defined(NALU_HYPRE_USING_SYCL)
template<typename T>
struct tuple_plus
{
   __host__ __device__
   std::tuple<T, T> operator()( const std::tuple<T, T> & x1, const std::tuple<T, T> & x2) const
   {
      return std::make_tuple( std::get<0>(x1) + std::get<0>(x2),
                              std::get<1>(x1) + std::get<1>(x2) );
   }
};

struct local_equal_plus_constant
{
   NALU_HYPRE_BigInt _value;

   local_equal_plus_constant(NALU_HYPRE_BigInt value) : _value(value) {}

   __host__ __device__ NALU_HYPRE_BigInt operator()(NALU_HYPRE_BigInt /*x*/, NALU_HYPRE_BigInt y) const
   { return y + _value; }
};

/* transform from local C index to global C index */
struct globalC_functor
{
   NALU_HYPRE_BigInt C_first;

   globalC_functor(NALU_HYPRE_BigInt C_first_)
   {
      C_first = C_first_;
   }

   __host__ __device__
   NALU_HYPRE_BigInt operator()(const NALU_HYPRE_Int x) const
   {
      return ( (NALU_HYPRE_BigInt) x + C_first );
   }
};
#else
template<typename T>
struct tuple_plus : public
   thrust::binary_function<thrust::tuple<T, T>, thrust::tuple<T, T>, thrust::tuple<T, T> >
{
   __host__ __device__
   thrust::tuple<T, T> operator()( const thrust::tuple<T, T> & x1, const thrust::tuple<T, T> & x2)
   {
      return thrust::make_tuple( thrust::get<0>(x1) + thrust::get<0>(x2),
                                 thrust::get<1>(x1) + thrust::get<1>(x2) );
   }
};

template<typename T>
struct tuple_minus : public
   thrust::binary_function<thrust::tuple<T, T>, thrust::tuple<T, T>, thrust::tuple<T, T> >
{
   __host__ __device__
   thrust::tuple<T, T> operator()( const thrust::tuple<T, T> & x1, const thrust::tuple<T, T> & x2)
   {
      return thrust::make_tuple( thrust::get<0>(x1) - thrust::get<0>(x2),
                                 thrust::get<1>(x1) - thrust::get<1>(x2) );
   }
};

struct local_equal_plus_constant : public
   thrust::binary_function<NALU_HYPRE_BigInt, NALU_HYPRE_BigInt, NALU_HYPRE_BigInt>
{
   NALU_HYPRE_BigInt _value;

   local_equal_plus_constant(NALU_HYPRE_BigInt value) : _value(value) {}

   __host__ __device__ NALU_HYPRE_BigInt operator()(NALU_HYPRE_BigInt /*x*/, NALU_HYPRE_BigInt y)
   { return y + _value; }
};

/* transform from local C index to global C index */
struct globalC_functor : public thrust::unary_function<NALU_HYPRE_Int, NALU_HYPRE_BigInt>
{
   NALU_HYPRE_BigInt C_first;

   globalC_functor(NALU_HYPRE_BigInt C_first_)
   {
      C_first = C_first_;
   }

   __host__ __device__
   NALU_HYPRE_BigInt operator()(const NALU_HYPRE_Int x) const
   {
      return ( (NALU_HYPRE_BigInt) x + C_first );
   }
};
#endif

void nalu_hypre_modmp_init_fine_to_coarse( NALU_HYPRE_Int n_fine, NALU_HYPRE_Int *pass_marker, NALU_HYPRE_Int color,
                                      NALU_HYPRE_Int *fine_to_coarse );

void nalu_hypre_modmp_compute_num_cols_offd_fine_to_coarse( NALU_HYPRE_Int * pass_marker_offd,
                                                       NALU_HYPRE_Int color, NALU_HYPRE_Int num_cols_offd_A, NALU_HYPRE_Int & num_cols_offd,
                                                       NALU_HYPRE_Int ** fine_to_coarse_offd );

__global__ void hypreGPUKernel_cfmarker_masked_rowsum( nalu_hypre_DeviceItem &item, NALU_HYPRE_Int nrows,
                                                       NALU_HYPRE_Int *A_diag_i,
                                                       NALU_HYPRE_Int *A_diag_j, NALU_HYPRE_Complex *A_diag_data, NALU_HYPRE_Int *A_offd_i, NALU_HYPRE_Int *A_offd_j,
                                                       NALU_HYPRE_Complex *A_offd_data, NALU_HYPRE_Int *CF_marker, NALU_HYPRE_Int *dof_func, NALU_HYPRE_Int *dof_func_offd,
                                                       NALU_HYPRE_Complex *row_sums );

__global__ void hypreGPUKernel_generate_Pdiag_i_Poffd_i( nalu_hypre_DeviceItem &item,
                                                         NALU_HYPRE_Int num_points,
                                                         NALU_HYPRE_Int color,
                                                         NALU_HYPRE_Int *pass_order, NALU_HYPRE_Int *pass_marker, NALU_HYPRE_Int *pass_marker_offd, NALU_HYPRE_Int *S_diag_i,
                                                         NALU_HYPRE_Int *S_diag_j, NALU_HYPRE_Int *S_offd_i, NALU_HYPRE_Int *S_offd_j, NALU_HYPRE_Int *P_diag_i,
                                                         NALU_HYPRE_Int *P_offd_i );

__global__ void hypreGPUKernel_generate_Pdiag_j_Poffd_j( nalu_hypre_DeviceItem &item,
                                                         NALU_HYPRE_Int num_points,
                                                         NALU_HYPRE_Int color,
                                                         NALU_HYPRE_Int *pass_order, NALU_HYPRE_Int *pass_marker, NALU_HYPRE_Int *pass_marker_offd,
                                                         NALU_HYPRE_Int *fine_to_coarse, NALU_HYPRE_Int *fine_to_coarse_offd, NALU_HYPRE_Int *A_diag_i, NALU_HYPRE_Int *A_diag_j,
                                                         NALU_HYPRE_Complex *A_diag_data, NALU_HYPRE_Int *A_offd_i, NALU_HYPRE_Int *A_offd_j, NALU_HYPRE_Complex *A_offd_data,
                                                         NALU_HYPRE_Int *Soc_diag_j, NALU_HYPRE_Int *Soc_offd_j, NALU_HYPRE_Int *P_diag_i, NALU_HYPRE_Int *P_offd_i,
                                                         NALU_HYPRE_Int *P_diag_j, NALU_HYPRE_Complex *P_diag_data, NALU_HYPRE_Int *P_offd_j, NALU_HYPRE_Complex *P_offd_data,
                                                         NALU_HYPRE_Complex *row_sums );

__global__ void hypreGPUKernel_insert_remaining_weights( nalu_hypre_DeviceItem &item, NALU_HYPRE_Int start,
                                                         NALU_HYPRE_Int stop,
                                                         NALU_HYPRE_Int *pass_order, NALU_HYPRE_Int *Pi_diag_i, NALU_HYPRE_Int *Pi_diag_j, NALU_HYPRE_Real *Pi_diag_data,
                                                         NALU_HYPRE_Int *P_diag_i, NALU_HYPRE_Int *P_diag_j, NALU_HYPRE_Real *P_diag_data, NALU_HYPRE_Int *Pi_offd_i,
                                                         NALU_HYPRE_Int *Pi_offd_j, NALU_HYPRE_Real *Pi_offd_data, NALU_HYPRE_Int *P_offd_i, NALU_HYPRE_Int *P_offd_j,
                                                         NALU_HYPRE_Real *P_offd_data );

__global__ void hypreGPUKernel_generate_Qdiag_j_Qoffd_j( nalu_hypre_DeviceItem &item,
                                                         NALU_HYPRE_Int num_points,
                                                         NALU_HYPRE_Int color,
                                                         NALU_HYPRE_Int *pass_order, NALU_HYPRE_Int *pass_marker, NALU_HYPRE_Int *pass_marker_offd,
                                                         NALU_HYPRE_Int *fine_to_coarse, NALU_HYPRE_Int *fine_to_coarse_offd, NALU_HYPRE_Int *A_diag_i, NALU_HYPRE_Int *A_diag_j,
                                                         NALU_HYPRE_Complex *A_diag_data, NALU_HYPRE_Int *A_offd_i, NALU_HYPRE_Int *A_offd_j, NALU_HYPRE_Complex *A_offd_data,
                                                         NALU_HYPRE_Int *Soc_diag_j, NALU_HYPRE_Int *Soc_offd_j, NALU_HYPRE_Int *Q_diag_i, NALU_HYPRE_Int *Q_offd_i,
                                                         NALU_HYPRE_Int *Q_diag_j, NALU_HYPRE_Complex *Q_diag_data, NALU_HYPRE_Int *Q_offd_j, NALU_HYPRE_Complex *Q_offd_data,
                                                         NALU_HYPRE_Complex *w_row_sum, NALU_HYPRE_Int num_functions, NALU_HYPRE_Int *dof_func, NALU_HYPRE_Int *dof_func_offd );

__global__ void hypreGPUKernel_mutli_pi_rowsum( nalu_hypre_DeviceItem &item, NALU_HYPRE_Int num_points,
                                                NALU_HYPRE_Int *pass_order,
                                                NALU_HYPRE_Int *A_diag_i, NALU_HYPRE_Complex *A_diag_data, NALU_HYPRE_Int *Pi_diag_i, NALU_HYPRE_Complex *Pi_diag_data,
                                                NALU_HYPRE_Int *Pi_offd_i, NALU_HYPRE_Complex *Pi_offd_data, NALU_HYPRE_Complex *w_row_sum );

__global__ void hypreGPUKernel_pass_order_count( nalu_hypre_DeviceItem &item, NALU_HYPRE_Int num_points,
                                                 NALU_HYPRE_Int color,
                                                 NALU_HYPRE_Int *points_left, NALU_HYPRE_Int *pass_marker, NALU_HYPRE_Int *pass_marker_offd, NALU_HYPRE_Int *S_diag_i,
                                                 NALU_HYPRE_Int *S_diag_j, NALU_HYPRE_Int *S_offd_i, NALU_HYPRE_Int *S_offd_j, NALU_HYPRE_Int *diag_shifts );

__global__ void hypreGPUKernel_populate_big_P_offd_j( nalu_hypre_DeviceItem &item, NALU_HYPRE_Int start,
                                                      NALU_HYPRE_Int stop,
                                                      NALU_HYPRE_Int *pass_order, NALU_HYPRE_Int *P_offd_i, NALU_HYPRE_Int *P_offd_j, NALU_HYPRE_BigInt *col_map_offd_Pi,
                                                      NALU_HYPRE_BigInt *big_P_offd_j );

/*--------------------------------------------------------------------------
 * nalu_hypre_ParAMGBuildModMultipass
 * This routine implements Stuben's direct interpolation with multiple passes.
 * expressed with matrix matrix multiplications
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_BoomerAMGBuildModMultipassDevice( nalu_hypre_ParCSRMatrix  *A,
                                        NALU_HYPRE_Int           *CF_marker,
                                        nalu_hypre_ParCSRMatrix  *S,
                                        NALU_HYPRE_BigInt        *num_cpts_global,
                                        NALU_HYPRE_Real           trunc_factor,
                                        NALU_HYPRE_Int            P_max_elmts,
                                        NALU_HYPRE_Int            interp_type,
                                        NALU_HYPRE_Int            num_functions,
                                        NALU_HYPRE_Int           *dof_func,
                                        nalu_hypre_ParCSRMatrix **P_ptr )
{
#ifdef NALU_HYPRE_PROFILE
   nalu_hypre_profile_times[NALU_HYPRE_TIMER_ID_MULTIPASS_INTERP] -= nalu_hypre_MPI_Wtime();
#endif

   nalu_hypre_assert( nalu_hypre_ParCSRMatrixMemoryLocation(A) == NALU_HYPRE_MEMORY_DEVICE );
   nalu_hypre_assert( nalu_hypre_ParCSRMatrixMemoryLocation(S) == NALU_HYPRE_MEMORY_DEVICE );

   MPI_Comm                comm     = nalu_hypre_ParCSRMatrixComm(A);
   nalu_hypre_ParCSRCommPkg    *comm_pkg = nalu_hypre_ParCSRMatrixCommPkg(A);
   nalu_hypre_ParCSRCommHandle *comm_handle;

   NALU_HYPRE_Int        n_fine          = nalu_hypre_ParCSRMatrixNumRows(A);
   nalu_hypre_CSRMatrix *A_diag          = nalu_hypre_ParCSRMatrixDiag(A);
   NALU_HYPRE_Real      *A_diag_data     = nalu_hypre_CSRMatrixData(A_diag);
   NALU_HYPRE_Int       *A_diag_i        = nalu_hypre_CSRMatrixI(A_diag);
   NALU_HYPRE_Int       *A_diag_j        = nalu_hypre_CSRMatrixJ(A_diag);
   nalu_hypre_CSRMatrix *A_offd          = nalu_hypre_ParCSRMatrixOffd(A);
   NALU_HYPRE_Int       *A_offd_i        = nalu_hypre_CSRMatrixI(A_offd);
   NALU_HYPRE_Int       *A_offd_j        = nalu_hypre_CSRMatrixJ(A_offd);
   NALU_HYPRE_Real      *A_offd_data     = nalu_hypre_CSRMatrixData(A_offd);
   NALU_HYPRE_Int        num_cols_offd_A = nalu_hypre_CSRMatrixNumCols(A_offd);

   nalu_hypre_CSRMatrix *S_diag       = nalu_hypre_ParCSRMatrixDiag(S);
   NALU_HYPRE_Int       *S_diag_i     = nalu_hypre_CSRMatrixI(S_diag);
   NALU_HYPRE_Int       *S_diag_j     = nalu_hypre_CSRMatrixJ(S_diag);
   nalu_hypre_CSRMatrix *S_offd       = nalu_hypre_ParCSRMatrixOffd(S);
   NALU_HYPRE_Int       *S_offd_i     = nalu_hypre_CSRMatrixI(S_offd);
   NALU_HYPRE_Int       *S_offd_j     = nalu_hypre_CSRMatrixJ(S_offd);

   nalu_hypre_ParCSRMatrix **Pi;
   nalu_hypre_ParCSRMatrix  *P;
   nalu_hypre_CSRMatrix     *P_diag;
   NALU_HYPRE_Real          *P_diag_data;
   NALU_HYPRE_Int           *P_diag_i;
   NALU_HYPRE_Int           *P_diag_j;
   nalu_hypre_CSRMatrix     *P_offd;
   NALU_HYPRE_Real          *P_offd_data = NULL;
   NALU_HYPRE_Int           *P_offd_i;
   NALU_HYPRE_Int           *P_offd_j = NULL;
   NALU_HYPRE_BigInt        *col_map_offd_P = NULL;
   NALU_HYPRE_BigInt        *col_map_offd_P_host = NULL;
   NALU_HYPRE_Int            num_cols_offd_P = 0;
   NALU_HYPRE_Int            num_sends = nalu_hypre_ParCSRCommPkgNumSends(comm_pkg);
   NALU_HYPRE_Int            num_elem_send = nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends);
   NALU_HYPRE_Int           *int_buf_data = NULL;
   NALU_HYPRE_Int            P_diag_size = 0, P_offd_size = 0;

   NALU_HYPRE_Int       *pass_starts;
   NALU_HYPRE_Int       *fine_to_coarse;
   NALU_HYPRE_Int       *points_left;
   NALU_HYPRE_Int       *pass_marker;
   NALU_HYPRE_Int       *pass_marker_offd = NULL;
   NALU_HYPRE_Int       *pass_order;

   NALU_HYPRE_Int        i;
   NALU_HYPRE_Int        num_passes, p, remaining;
   NALU_HYPRE_Int        pass_starts_p1, pass_starts_p2;
   NALU_HYPRE_BigInt     remaining_big; /* tmp variable for reducing global_remaining */
   NALU_HYPRE_BigInt     global_remaining;
   NALU_HYPRE_Int        cnt, cnt_old, cnt_rem, current_pass;

   NALU_HYPRE_BigInt     total_global_cpts;
   NALU_HYPRE_Int        my_id, num_procs;

   NALU_HYPRE_Int       *dof_func_offd = NULL;
   NALU_HYPRE_Real      *row_sums = NULL;

   nalu_hypre_GpuProfilingPushRange("Section1");

   /* MPI size and rank*/
   nalu_hypre_MPI_Comm_size(comm, &num_procs);
   nalu_hypre_MPI_Comm_rank(comm, &my_id);

   if (num_procs > 1)
   {
      if (my_id == num_procs - 1)
      {
         total_global_cpts = num_cpts_global[1];
      }
      nalu_hypre_MPI_Bcast(&total_global_cpts, 1, NALU_HYPRE_MPI_BIG_INT, num_procs - 1, comm);
   }
   else
   {
      total_global_cpts = num_cpts_global[1];
   }

   if (!total_global_cpts)
   {
      *P_ptr = NULL;
      return nalu_hypre_error_flag;
   }

   nalu_hypre_BoomerAMGMakeSocFromSDevice(A, S);

   /* Generate pass marker array */
   /* contains pass numbers for each variable according to original order */
   pass_marker = nalu_hypre_CTAlloc(NALU_HYPRE_Int, n_fine, NALU_HYPRE_MEMORY_DEVICE);
   /* contains row numbers according to new order, pass 1 followed by pass 2 etc */
   pass_order = nalu_hypre_CTAlloc(NALU_HYPRE_Int, n_fine, NALU_HYPRE_MEMORY_DEVICE);
   /* F2C mapping */
   /* reverse of pass_order, keeps track where original numbers go */
   fine_to_coarse = nalu_hypre_TAlloc(NALU_HYPRE_Int, n_fine, NALU_HYPRE_MEMORY_DEVICE);
   /* contains row numbers of remaining points, auxiliary */
   points_left = nalu_hypre_CTAlloc(NALU_HYPRE_Int, n_fine, NALU_HYPRE_MEMORY_DEVICE);
   P_diag_i = nalu_hypre_CTAlloc(NALU_HYPRE_Int, n_fine + 1, NALU_HYPRE_MEMORY_DEVICE);
   P_offd_i = nalu_hypre_CTAlloc(NALU_HYPRE_Int, n_fine + 1, NALU_HYPRE_MEMORY_DEVICE);

#if defined(NALU_HYPRE_USING_SYCL)
   /* Fpts; number of F pts */
   oneapi::dpl::counting_iterator<NALU_HYPRE_Int> count(0);
   NALU_HYPRE_Int *points_end = hypreSycl_copy_if( count,
                                              count + n_fine,
                                              CF_marker,
                                              points_left,
   [] (const auto & x) {return x != 1;} );
   remaining = points_end - points_left;

   /* Cpts; number of C pts */
   NALU_HYPRE_Int *pass_end = hypreSycl_copy_if( count,
                                            count + n_fine,
                                            CF_marker,
                                            pass_order,
                                            equal<NALU_HYPRE_Int>(1) );

   P_diag_size = cnt = pass_end - pass_order;

   /* mark C points pass-1; row nnz of C-diag = 1, C-offd = 0 */
   auto zip0 = oneapi::dpl::make_zip_iterator( pass_marker, P_diag_i, P_offd_i );
   hypreSycl_transform_if( zip0,
                           zip0 + n_fine,
                           CF_marker,
                           zip0,
   [] (const auto & x) {return std::make_tuple(NALU_HYPRE_Int(1), NALU_HYPRE_Int(1), NALU_HYPRE_Int(0));},
   equal<NALU_HYPRE_Int>(1) );

   NALU_HYPRE_ONEDPL_CALL( std::exclusive_scan,
                      oneapi::dpl::make_transform_iterator(CF_marker,          equal<NALU_HYPRE_Int>(1)),
                      oneapi::dpl::make_transform_iterator(CF_marker + n_fine, equal<NALU_HYPRE_Int>(1)),
                      fine_to_coarse,
                      NALU_HYPRE_Int(0) );
#else
   /* Fpts; number of F pts */
   NALU_HYPRE_Int *points_end = NALU_HYPRE_THRUST_CALL( copy_if,
                                              thrust::make_counting_iterator(0),
                                              thrust::make_counting_iterator(n_fine),
                                              CF_marker,
                                              points_left,
                                              thrust::not1(equal<NALU_HYPRE_Int>(1)) );
   remaining = points_end - points_left;

   /* Cpts; number of C pts */
   NALU_HYPRE_Int *pass_end = NALU_HYPRE_THRUST_CALL( copy_if,
                                            thrust::make_counting_iterator(0),
                                            thrust::make_counting_iterator(n_fine),
                                            CF_marker,
                                            pass_order,
                                            equal<NALU_HYPRE_Int>(1) );

   P_diag_size = cnt = pass_end - pass_order;

   /* mark C points pass-1; row nnz of C-diag = 1, C-offd = 0 */
   NALU_HYPRE_THRUST_CALL( replace_if,
                      thrust::make_zip_iterator( thrust::make_tuple(pass_marker, P_diag_i, P_offd_i) ),
                      thrust::make_zip_iterator( thrust::make_tuple(pass_marker, P_diag_i, P_offd_i) ) + n_fine,
                      CF_marker,
                      equal<NALU_HYPRE_Int>(1),
                      thrust::make_tuple(NALU_HYPRE_Int(1), NALU_HYPRE_Int(1), NALU_HYPRE_Int(0)) );

   NALU_HYPRE_THRUST_CALL( exclusive_scan,
                      thrust::make_transform_iterator(CF_marker,          equal<NALU_HYPRE_Int>(1)),
                      thrust::make_transform_iterator(CF_marker + n_fine, equal<NALU_HYPRE_Int>(1)),
                      fine_to_coarse,
                      NALU_HYPRE_Int(0) );
#endif

   /* contains beginning for each pass in pass_order field, assume no more than 10 passes */
   pass_starts = nalu_hypre_CTAlloc(NALU_HYPRE_Int, 11, NALU_HYPRE_MEMORY_HOST);
   /* first pass is C */
   pass_starts[0] = 0;
   pass_starts[1] = cnt;

   /* communicate dof_func */
   if (num_procs > 1 && num_functions > 1)
   {
      int_buf_data = nalu_hypre_TAlloc(NALU_HYPRE_Int, num_elem_send, NALU_HYPRE_MEMORY_DEVICE);

      nalu_hypre_ParCSRCommPkgCopySendMapElmtsToDevice(comm_pkg);
#if defined(NALU_HYPRE_USING_SYCL)
      hypreSycl_gather( nalu_hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg),
                        nalu_hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg) + num_elem_send,
                        dof_func,
                        int_buf_data );
#else
      NALU_HYPRE_THRUST_CALL( gather,
                         nalu_hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg),
                         nalu_hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg) + num_elem_send,
                         dof_func,
                         int_buf_data );
#endif

#if defined(NALU_HYPRE_WITH_GPU_AWARE_MPI) && defined(NALU_HYPRE_USING_THRUST_NOSYNC)
      /* RL: make sure int_buf_data is ready before issuing GPU-GPU MPI */
      nalu_hypre_ForceSyncComputeStream(nalu_hypre_handle());
#endif

      dof_func_offd = nalu_hypre_TAlloc(NALU_HYPRE_Int, num_cols_offd_A, NALU_HYPRE_MEMORY_DEVICE);

      comm_handle = nalu_hypre_ParCSRCommHandleCreate_v2(11, comm_pkg, NALU_HYPRE_MEMORY_DEVICE, int_buf_data,
                                                    NALU_HYPRE_MEMORY_DEVICE, dof_func_offd);
      nalu_hypre_ParCSRCommHandleDestroy(comm_handle);
   }

   /* communicate pass_marker */
   if (num_procs > 1)
   {
      if (!int_buf_data)
      {
         int_buf_data = nalu_hypre_CTAlloc(NALU_HYPRE_Int, num_elem_send, NALU_HYPRE_MEMORY_DEVICE);
      }

      nalu_hypre_ParCSRCommPkgCopySendMapElmtsToDevice(comm_pkg);

#if defined(NALU_HYPRE_USING_SYCL)
      hypreSycl_gather( nalu_hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg),
                        nalu_hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg) + num_elem_send,
                        pass_marker,
                        int_buf_data );
#else
      NALU_HYPRE_THRUST_CALL( gather,
                         nalu_hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg),
                         nalu_hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg) + num_elem_send,
                         pass_marker,
                         int_buf_data );
#endif

#if defined(NALU_HYPRE_WITH_GPU_AWARE_MPI) && defined(NALU_HYPRE_USING_THRUST_NOSYNC)
      /* RL: make sure int_buf_data is ready before issuing GPU-GPU MPI */
      nalu_hypre_ForceSyncComputeStream(nalu_hypre_handle());
#endif

      /* allocate one more see comments in nalu_hypre_modmp_compute_num_cols_offd_fine_to_coarse */
      pass_marker_offd = nalu_hypre_CTAlloc(NALU_HYPRE_Int, num_cols_offd_A + 1, NALU_HYPRE_MEMORY_DEVICE);

      /* create a handle to start communication. 11: for integer */
      comm_handle = nalu_hypre_ParCSRCommHandleCreate_v2(11, comm_pkg, NALU_HYPRE_MEMORY_DEVICE, int_buf_data,
                                                    NALU_HYPRE_MEMORY_DEVICE, pass_marker_offd);

      /* destroy the handle to finish communication */
      nalu_hypre_ParCSRCommHandleDestroy(comm_handle);
   }

   current_pass = 1;
   num_passes = 1;
   /* color points according to pass number */
   remaining_big = remaining;
   nalu_hypre_MPI_Allreduce(&remaining_big, &global_remaining, 1, NALU_HYPRE_MPI_BIG_INT, nalu_hypre_MPI_SUM, comm);

   nalu_hypre_GpuProfilingPopRange();

   nalu_hypre_GpuProfilingPushRange("Section2");

   NALU_HYPRE_Int *points_left_old = nalu_hypre_TAlloc(NALU_HYPRE_Int, remaining, NALU_HYPRE_MEMORY_DEVICE);
   NALU_HYPRE_Int *diag_shifts     = nalu_hypre_TAlloc(NALU_HYPRE_Int, remaining, NALU_HYPRE_MEMORY_DEVICE);

   while (global_remaining > 0)
   {
      cnt_rem = 0;
      cnt_old = cnt;

      {
         dim3 bDim = nalu_hypre_GetDefaultDeviceBlockDimension();
         dim3 gDim = nalu_hypre_GetDefaultDeviceGridDimension(remaining, "warp", bDim);

         /* output diag_shifts is 0/1 indicating if points_left_dev[i] is picked in this pass */
         NALU_HYPRE_GPU_LAUNCH( hypreGPUKernel_pass_order_count,
                           gDim, bDim,
                           remaining,
                           current_pass,
                           points_left,
                           pass_marker,
                           pass_marker_offd,
                           S_diag_i,
                           S_diag_j,
                           S_offd_i,
                           S_offd_j,
                           diag_shifts );

#if defined(NALU_HYPRE_USING_SYCL)
         cnt = NALU_HYPRE_ONEDPL_CALL( std::reduce,
                                  diag_shifts,
                                  diag_shifts + remaining,
                                  cnt_old,
                                  std::plus<NALU_HYPRE_Int>() );

         cnt_rem = remaining - (cnt - cnt_old);

         auto perm0 = oneapi::dpl::make_permutation_iterator(pass_marker, points_left);
         hypreSycl_transform_if( perm0,
                                 perm0 + remaining,
                                 diag_shifts,
                                 perm0,
         [current_pass = current_pass] (const auto & x) {return current_pass + 1;},
         [] (const auto & x) {return x;} );

         nalu_hypre_TMemcpy(points_left_old, points_left, NALU_HYPRE_Int, remaining, NALU_HYPRE_MEMORY_DEVICE,
                       NALU_HYPRE_MEMORY_DEVICE);

         NALU_HYPRE_Int *new_end;
         new_end = hypreSycl_copy_if( points_left_old,
                                      points_left_old + remaining,
                                      diag_shifts,
                                      pass_order + cnt_old,
         [] (const auto & x) {return x;} );

         nalu_hypre_assert(new_end - pass_order == cnt);

         new_end = hypreSycl_copy_if( points_left_old,
                                      points_left_old + remaining,
                                      diag_shifts,
                                      points_left,
         [] (const auto & x) {return !x;} );
#else
         cnt = NALU_HYPRE_THRUST_CALL( reduce,
                                  diag_shifts,
                                  diag_shifts + remaining,
                                  cnt_old,
                                  thrust::plus<NALU_HYPRE_Int>() );

         cnt_rem = remaining - (cnt - cnt_old);

         NALU_HYPRE_THRUST_CALL( replace_if,
                            thrust::make_permutation_iterator(pass_marker, points_left),
                            thrust::make_permutation_iterator(pass_marker, points_left + remaining),
                            diag_shifts,
                            thrust::identity<NALU_HYPRE_Int>(),
                            current_pass + 1 );

         nalu_hypre_TMemcpy(points_left_old, points_left, NALU_HYPRE_Int, remaining, NALU_HYPRE_MEMORY_DEVICE,
                       NALU_HYPRE_MEMORY_DEVICE);

         NALU_HYPRE_Int *new_end;
         new_end = NALU_HYPRE_THRUST_CALL( copy_if,
                                      points_left_old,
                                      points_left_old + remaining,
                                      diag_shifts,
                                      pass_order + cnt_old,
                                      thrust::identity<NALU_HYPRE_Int>() );

         nalu_hypre_assert(new_end - pass_order == cnt);

         new_end = NALU_HYPRE_THRUST_CALL( copy_if,
                                      points_left_old,
                                      points_left_old + remaining,
                                      diag_shifts,
                                      points_left,
                                      thrust::not1(thrust::identity<NALU_HYPRE_Int>()) );
#endif

         nalu_hypre_assert(new_end - points_left == cnt_rem);
      }

      remaining = cnt_rem;
      current_pass++;
      num_passes++;

      if (num_passes > 9)
      {
         nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC, " Warning!!! too many passes! out of range!\n");
         break;
      }

      pass_starts[num_passes] = cnt;

      /* update pass_marker_offd */
      if (num_procs > 1)
      {
#if defined(NALU_HYPRE_USING_SYCL)
         hypreSycl_gather( nalu_hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg),
                           nalu_hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg) + num_elem_send,
                           pass_marker,
                           int_buf_data );
#else
         NALU_HYPRE_THRUST_CALL( gather,
                            nalu_hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg),
                            nalu_hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg) + num_elem_send,
                            pass_marker,
                            int_buf_data );
#endif

#if defined(NALU_HYPRE_WITH_GPU_AWARE_MPI) && defined(NALU_HYPRE_USING_THRUST_NOSYNC)
         /* RL: make sure int_buf_data is ready before issuing GPU-GPU MPI */
         nalu_hypre_ForceSyncComputeStream(nalu_hypre_handle());
#endif

         /* create a handle to start communication. 11: for integer */
         comm_handle = nalu_hypre_ParCSRCommHandleCreate_v2(11, comm_pkg, NALU_HYPRE_MEMORY_DEVICE, int_buf_data,
                                                       NALU_HYPRE_MEMORY_DEVICE, pass_marker_offd);

         /* destroy the handle to finish communication */
         nalu_hypre_ParCSRCommHandleDestroy(comm_handle);
      }

      NALU_HYPRE_BigInt old_global_remaining = global_remaining;

      remaining_big = remaining;
      nalu_hypre_MPI_Allreduce(&remaining_big, &global_remaining, 1, NALU_HYPRE_MPI_BIG_INT, nalu_hypre_MPI_SUM, comm);

      /* if the number of remaining points does not change, we have a situation of isolated areas of
       * fine points that are not connected to any C-points, and the pass generation process breaks
       * down. Those points can be ignored, i.e. the corresponding rows in P will just be 0
       * and can be ignored for the algorithm. */
      if (old_global_remaining == global_remaining)
      {
         break;
      }

   } // while (global_remaining > 0)

   nalu_hypre_TFree(diag_shifts,     NALU_HYPRE_MEMORY_DEVICE);
   nalu_hypre_TFree(points_left_old, NALU_HYPRE_MEMORY_DEVICE);
   nalu_hypre_TFree(int_buf_data,    NALU_HYPRE_MEMORY_DEVICE);
   nalu_hypre_TFree(points_left,     NALU_HYPRE_MEMORY_DEVICE);

   /* generate row sum of weak points and C-points to be ignored */
   row_sums = nalu_hypre_CTAlloc(NALU_HYPRE_Real, n_fine, NALU_HYPRE_MEMORY_DEVICE);

   {
      dim3 bDim = nalu_hypre_GetDefaultDeviceBlockDimension();
      dim3 gDim = nalu_hypre_GetDefaultDeviceGridDimension(n_fine, "warp", bDim);

      NALU_HYPRE_GPU_LAUNCH( hypreGPUKernel_cfmarker_masked_rowsum, gDim, bDim,
                        n_fine, A_diag_i, A_diag_j, A_diag_data,
                        A_offd_i, A_offd_j, A_offd_data,
                        CF_marker,
                        num_functions > 1 ? dof_func : NULL,
                        num_functions > 1 ? dof_func_offd : NULL,
                        row_sums );
   }

   nalu_hypre_GpuProfilingPopRange();

   nalu_hypre_GpuProfilingPushRange("MultipassPiDevice");

   Pi = nalu_hypre_CTAlloc(nalu_hypre_ParCSRMatrix*, num_passes, NALU_HYPRE_MEMORY_HOST);

   nalu_hypre_GenerateMultipassPiDevice(A, S, num_cpts_global, &pass_order[pass_starts[1]],
                                   pass_marker, pass_marker_offd,
                                   pass_starts[2] - pass_starts[1], 1, row_sums, &Pi[0]);

   nalu_hypre_GpuProfilingPopRange();

   if (interp_type == 8)
   {
      for (i = 1; i < num_passes - 1; i++)
      {
         nalu_hypre_GpuProfilingPushRange(std::string("MultipassPiDevice Loop" + std::to_string(i)).c_str());

         nalu_hypre_ParCSRMatrix *Q;
         NALU_HYPRE_BigInt *c_pts_starts = nalu_hypre_ParCSRMatrixRowStarts(Pi[i - 1]);

         nalu_hypre_GenerateMultipassPiDevice(A, S, c_pts_starts, &pass_order[pass_starts[i + 1]],
                                         pass_marker, pass_marker_offd,
                                         pass_starts[i + 2] - pass_starts[i + 1], i + 1, row_sums, &Q);

         nalu_hypre_GpuProfilingPopRange();
         Pi[i] = nalu_hypre_ParCSRMatMat(Q, Pi[i - 1]);

         nalu_hypre_ParCSRMatrixDestroy(Q);
      }
   }
   else if (interp_type == 9)
   {
      for (i = 1; i < num_passes - 1; i++)
      {
         nalu_hypre_GpuProfilingPushRange(std::string("MultiPiDevice Loop" + std::to_string(i)).c_str());
         NALU_HYPRE_BigInt *c_pts_starts = nalu_hypre_ParCSRMatrixRowStarts(Pi[i - 1]);

         nalu_hypre_GenerateMultiPiDevice(A, S, Pi[i - 1], c_pts_starts, &pass_order[pass_starts[i + 1]],
                                     pass_marker, pass_marker_offd,
                                     pass_starts[i + 2] - pass_starts[i + 1], i + 1,
                                     num_functions, dof_func, dof_func_offd, &Pi[i] );

         nalu_hypre_GpuProfilingPopRange();
      }
   }

   nalu_hypre_GpuProfilingPushRange("Section3");

   // We don't need the row sums anymore
   nalu_hypre_TFree(row_sums, NALU_HYPRE_MEMORY_DEVICE);

   /* populate P_diag_i/P_offd_i[i] with nnz of i-th row */
   for (i = 0; i < num_passes - 1; i++)
   {
      NALU_HYPRE_Int *Pi_diag_i = nalu_hypre_CSRMatrixI(nalu_hypre_ParCSRMatrixDiag(Pi[i]));
      NALU_HYPRE_Int *Pi_offd_i = nalu_hypre_CSRMatrixI(nalu_hypre_ParCSRMatrixOffd(Pi[i]));

      NALU_HYPRE_Int start = pass_starts[i + 1];
      NALU_HYPRE_Int stop  = pass_starts[i + 2];

#if defined(NALU_HYPRE_USING_SYCL)
      NALU_HYPRE_ONEDPL_CALL( std::transform,
                         Pi_diag_i + 1,
                         Pi_diag_i + stop - start + 1,
                         Pi_diag_i,
                         oneapi::dpl::make_permutation_iterator( P_diag_i, pass_order + start ),
                         std::minus<NALU_HYPRE_Int>() );
      NALU_HYPRE_ONEDPL_CALL( std::transform,
                         Pi_offd_i + 1,
                         Pi_offd_i + stop - start + 1,
                         Pi_offd_i,
                         oneapi::dpl::make_permutation_iterator( P_offd_i, pass_order + start ),
                         std::minus<NALU_HYPRE_Int>() );
#else
      NALU_HYPRE_THRUST_CALL( transform,
                         thrust::make_zip_iterator(thrust::make_tuple(Pi_diag_i, Pi_offd_i)) + 1,
                         thrust::make_zip_iterator(thrust::make_tuple(Pi_diag_i, Pi_offd_i)) + stop - start + 1,
                         thrust::make_zip_iterator(thrust::make_tuple(Pi_diag_i, Pi_offd_i)),
                         thrust::make_permutation_iterator( thrust::make_zip_iterator(thrust::make_tuple(P_diag_i,
                                                                                                         P_offd_i)), pass_order + start ),
                         tuple_minus<NALU_HYPRE_Int>() );
#endif

      P_diag_size += nalu_hypre_CSRMatrixNumNonzeros(nalu_hypre_ParCSRMatrixDiag(Pi[i]));
      P_offd_size += nalu_hypre_CSRMatrixNumNonzeros(nalu_hypre_ParCSRMatrixOffd(Pi[i]));
   }

#if defined(NALU_HYPRE_USING_SYCL)
   /* WM: todo - this is a workaround since oneDPL's exclusive_scan gives incorrect results when doing the scan in place */
   auto zip2 = oneapi::dpl::make_zip_iterator( P_diag_i, P_offd_i );
   NALU_HYPRE_Int *P_diag_i_tmp = nalu_hypre_CTAlloc(NALU_HYPRE_Int, n_fine + 1, NALU_HYPRE_MEMORY_DEVICE);
   NALU_HYPRE_Int *P_offd_i_tmp = nalu_hypre_CTAlloc(NALU_HYPRE_Int, n_fine + 1, NALU_HYPRE_MEMORY_DEVICE);
   NALU_HYPRE_ONEDPL_CALL( std::exclusive_scan,
                      zip2,
                      zip2 + n_fine + 1,
                      oneapi::dpl::make_zip_iterator(P_diag_i_tmp, P_offd_i_tmp),
                      std::make_tuple(NALU_HYPRE_Int(0), NALU_HYPRE_Int(0)),
                      tuple_plus<NALU_HYPRE_Int>() );
   nalu_hypre_TMemcpy(P_diag_i, P_diag_i_tmp, NALU_HYPRE_Int, n_fine + 1, NALU_HYPRE_MEMORY_DEVICE,
                 NALU_HYPRE_MEMORY_DEVICE);
   nalu_hypre_TMemcpy(P_offd_i, P_offd_i_tmp, NALU_HYPRE_Int, n_fine + 1, NALU_HYPRE_MEMORY_DEVICE,
                 NALU_HYPRE_MEMORY_DEVICE);
   nalu_hypre_TFree(P_diag_i_tmp, NALU_HYPRE_MEMORY_DEVICE);
   nalu_hypre_TFree(P_offd_i_tmp, NALU_HYPRE_MEMORY_DEVICE);
#else
   NALU_HYPRE_THRUST_CALL( exclusive_scan,
                      thrust::make_zip_iterator( thrust::make_tuple(P_diag_i, P_offd_i) ),
                      thrust::make_zip_iterator( thrust::make_tuple(P_diag_i, P_offd_i) ) + n_fine + 1,
                      thrust::make_zip_iterator( thrust::make_tuple(P_diag_i, P_offd_i) ),
                      thrust::make_tuple(NALU_HYPRE_Int(0), NALU_HYPRE_Int(0)),
                      tuple_plus<NALU_HYPRE_Int>() );
#endif

#ifdef NALU_HYPRE_DEBUG
   {
      NALU_HYPRE_Int tmp;
      nalu_hypre_TMemcpy(&tmp, &P_diag_i[n_fine], NALU_HYPRE_Int, 1, NALU_HYPRE_MEMORY_HOST, NALU_HYPRE_MEMORY_DEVICE);
      nalu_hypre_assert(tmp == P_diag_size);
      nalu_hypre_TMemcpy(&tmp, &P_offd_i[n_fine], NALU_HYPRE_Int, 1, NALU_HYPRE_MEMORY_HOST, NALU_HYPRE_MEMORY_DEVICE);
      nalu_hypre_assert(tmp == P_offd_size);
   }
#endif

   P_diag_j    = nalu_hypre_TAlloc(NALU_HYPRE_Int,  P_diag_size, NALU_HYPRE_MEMORY_DEVICE);
   P_diag_data = nalu_hypre_TAlloc(NALU_HYPRE_Real, P_diag_size, NALU_HYPRE_MEMORY_DEVICE);
   P_offd_j    = nalu_hypre_TAlloc(NALU_HYPRE_Int,  P_offd_size, NALU_HYPRE_MEMORY_DEVICE);
   P_offd_data = nalu_hypre_TAlloc(NALU_HYPRE_Real, P_offd_size, NALU_HYPRE_MEMORY_DEVICE);

   /* insert weights for coarse points */
   {
#if defined(NALU_HYPRE_USING_SYCL)
      auto perm1 = oneapi::dpl::make_permutation_iterator( fine_to_coarse, pass_order );
      hypreSycl_scatter( perm1,
                         perm1 + pass_starts[1],
                         oneapi::dpl::make_permutation_iterator( P_diag_i, pass_order ),
                         P_diag_j );

      auto perm2 = oneapi::dpl::make_permutation_iterator( P_diag_i, pass_order );
      auto perm3 = oneapi::dpl::make_permutation_iterator( P_diag_data, perm2 );
      NALU_HYPRE_ONEDPL_CALL( std::transform,
                         perm3,
                         perm3 + pass_starts[1],
                         perm3,
      [] (const auto & x) {return 1.0;} );
#else
      NALU_HYPRE_THRUST_CALL( scatter,
                         thrust::make_permutation_iterator( fine_to_coarse, pass_order ),
                         thrust::make_permutation_iterator( fine_to_coarse, pass_order ) + pass_starts[1],
                         thrust::make_permutation_iterator( P_diag_i, pass_order ),
                         P_diag_j );

      NALU_HYPRE_THRUST_CALL( scatter,
                         thrust::make_constant_iterator<NALU_HYPRE_Real>(1.0),
                         thrust::make_constant_iterator<NALU_HYPRE_Real>(1.0) + pass_starts[1],
                         thrust::make_permutation_iterator( P_diag_i, pass_order ),
                         P_diag_data );
#endif
   }

   /* generate col_map_offd_P by combining all col_map_offd_Pi
    * and reompute indices if needed */

   /* insert remaining weights */
   for (p = 0; p < num_passes - 1; p++)
   {
      NALU_HYPRE_Int  *Pi_diag_i    = nalu_hypre_CSRMatrixI(nalu_hypre_ParCSRMatrixDiag(Pi[p]));
      NALU_HYPRE_Int  *Pi_offd_i    = nalu_hypre_CSRMatrixI(nalu_hypre_ParCSRMatrixOffd(Pi[p]));
      NALU_HYPRE_Int  *Pi_diag_j    = nalu_hypre_CSRMatrixJ(nalu_hypre_ParCSRMatrixDiag(Pi[p]));
      NALU_HYPRE_Int  *Pi_offd_j    = nalu_hypre_CSRMatrixJ(nalu_hypre_ParCSRMatrixOffd(Pi[p]));
      NALU_HYPRE_Real *Pi_diag_data = nalu_hypre_CSRMatrixData(nalu_hypre_ParCSRMatrixDiag(Pi[p]));
      NALU_HYPRE_Real *Pi_offd_data = nalu_hypre_CSRMatrixData(nalu_hypre_ParCSRMatrixOffd(Pi[p]));

      NALU_HYPRE_Int num_points = pass_starts[p + 2] - pass_starts[p + 1];

      dim3 bDim = nalu_hypre_GetDefaultDeviceBlockDimension();
      dim3 gDim = nalu_hypre_GetDefaultDeviceGridDimension(num_points, "warp", bDim);

      pass_starts_p1 = pass_starts[p + 1];
      pass_starts_p2 = pass_starts[p + 2];
      NALU_HYPRE_GPU_LAUNCH( hypreGPUKernel_insert_remaining_weights, gDim, bDim,
                        pass_starts_p1, pass_starts_p2, pass_order,
                        Pi_diag_i, Pi_diag_j, Pi_diag_data,
                        P_diag_i, P_diag_j, P_diag_data,
                        Pi_offd_i, Pi_offd_j, Pi_offd_data,
                        P_offd_i, P_offd_j, P_offd_data );
   }

   /* Note that col indices in P_offd_j probably not consistent,
      this gets fixed after truncation */
   P = nalu_hypre_ParCSRMatrixCreate(comm,
                                nalu_hypre_ParCSRMatrixGlobalNumRows(A),
                                total_global_cpts,
                                nalu_hypre_ParCSRMatrixRowStarts(A),
                                num_cpts_global,
                                num_cols_offd_P,
                                P_diag_size,
                                P_offd_size);

   P_diag = nalu_hypre_ParCSRMatrixDiag(P);
   nalu_hypre_CSRMatrixData(P_diag) = P_diag_data;
   nalu_hypre_CSRMatrixI(P_diag)    = P_diag_i;
   nalu_hypre_CSRMatrixJ(P_diag)    = P_diag_j;

   P_offd = nalu_hypre_ParCSRMatrixOffd(P);
   nalu_hypre_CSRMatrixData(P_offd) = P_offd_data;
   nalu_hypre_CSRMatrixI(P_offd)    = P_offd_i;
   nalu_hypre_CSRMatrixJ(P_offd)    = P_offd_j;

   /* Compress P, removing coefficients smaller than trunc_factor * Max */
   if (trunc_factor != 0.0 || P_max_elmts > 0)
   {
      nalu_hypre_BoomerAMGInterpTruncationDevice(P, trunc_factor, P_max_elmts);
      P_diag_data = nalu_hypre_CSRMatrixData(P_diag);
      P_diag_i = nalu_hypre_CSRMatrixI(P_diag);
      P_diag_j = nalu_hypre_CSRMatrixJ(P_diag);
      P_offd_data = nalu_hypre_CSRMatrixData(P_offd);
      P_offd_i = nalu_hypre_CSRMatrixI(P_offd);
      P_offd_j = nalu_hypre_CSRMatrixJ(P_offd);
      P_offd_size = nalu_hypre_CSRMatrixNumNonzeros(P_offd);
   }

   nalu_hypre_GpuProfilingPopRange();

   num_cols_offd_P = 0;

   if (P_offd_size)
   {
      nalu_hypre_GpuProfilingPushRange("Section4");

      NALU_HYPRE_BigInt *big_P_offd_j = nalu_hypre_TAlloc(NALU_HYPRE_BigInt, P_offd_size, NALU_HYPRE_MEMORY_DEVICE);

      for (p = 0; p < num_passes - 1; p++)
      {
         NALU_HYPRE_BigInt *col_map_offd_Pi = nalu_hypre_ParCSRMatrixDeviceColMapOffd(Pi[p]);

         NALU_HYPRE_Int npoints = pass_starts[p + 2] - pass_starts[p + 1];
         dim3 bDim = nalu_hypre_GetDefaultDeviceBlockDimension();
         dim3 gDim = nalu_hypre_GetDefaultDeviceGridDimension(npoints, "warp", bDim);

         pass_starts_p1 = pass_starts[p + 1];
         pass_starts_p2 = pass_starts[p + 2];
         NALU_HYPRE_GPU_LAUNCH( hypreGPUKernel_populate_big_P_offd_j, gDim, bDim,
                           pass_starts_p1,
                           pass_starts_p2,
                           pass_order,
                           P_offd_i,
                           P_offd_j,
                           col_map_offd_Pi,
                           big_P_offd_j );

      } // end num_passes for loop

      NALU_HYPRE_BigInt *tmp_P_offd_j = nalu_hypre_TAlloc(NALU_HYPRE_BigInt, P_offd_size, NALU_HYPRE_MEMORY_DEVICE);
      nalu_hypre_TMemcpy(tmp_P_offd_j, big_P_offd_j, NALU_HYPRE_BigInt, P_offd_size, NALU_HYPRE_MEMORY_DEVICE,
                    NALU_HYPRE_MEMORY_DEVICE);

#if defined(NALU_HYPRE_USING_SYCL)
      NALU_HYPRE_ONEDPL_CALL( std::sort,
                         tmp_P_offd_j,
                         tmp_P_offd_j + P_offd_size );

      NALU_HYPRE_BigInt *new_end = NALU_HYPRE_ONEDPL_CALL( std::unique,
                                                 tmp_P_offd_j,
                                                 tmp_P_offd_j + P_offd_size );
#else
      NALU_HYPRE_THRUST_CALL( sort,
                         tmp_P_offd_j,
                         tmp_P_offd_j + P_offd_size );

      NALU_HYPRE_BigInt *new_end = NALU_HYPRE_THRUST_CALL( unique,
                                                 tmp_P_offd_j,
                                                 tmp_P_offd_j + P_offd_size );
#endif

      num_cols_offd_P = new_end - tmp_P_offd_j;
      col_map_offd_P = nalu_hypre_TAlloc(NALU_HYPRE_BigInt, num_cols_offd_P, NALU_HYPRE_MEMORY_DEVICE);
      nalu_hypre_TMemcpy(col_map_offd_P, tmp_P_offd_j, NALU_HYPRE_BigInt, num_cols_offd_P, NALU_HYPRE_MEMORY_DEVICE,
                    NALU_HYPRE_MEMORY_DEVICE);
      nalu_hypre_TFree(tmp_P_offd_j, NALU_HYPRE_MEMORY_DEVICE);

      // PB: It seems we still need this on the host??
      col_map_offd_P_host = nalu_hypre_TAlloc(NALU_HYPRE_BigInt, num_cols_offd_P, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TMemcpy(col_map_offd_P_host, col_map_offd_P, NALU_HYPRE_BigInt, num_cols_offd_P, NALU_HYPRE_MEMORY_HOST,
                    NALU_HYPRE_MEMORY_DEVICE);

#if defined(NALU_HYPRE_USING_SYCL)
      NALU_HYPRE_ONEDPL_CALL( oneapi::dpl::lower_bound,
                         col_map_offd_P,
                         col_map_offd_P + num_cols_offd_P,
                         big_P_offd_j,
                         big_P_offd_j + P_offd_size,
                         P_offd_j );
#else
      NALU_HYPRE_THRUST_CALL( lower_bound,
                         col_map_offd_P,
                         col_map_offd_P + num_cols_offd_P,
                         big_P_offd_j,
                         big_P_offd_j + P_offd_size,
                         P_offd_j );
#endif

      nalu_hypre_TFree(big_P_offd_j, NALU_HYPRE_MEMORY_DEVICE);

      nalu_hypre_GpuProfilingPopRange();
   } // if (P_offd_size)

   nalu_hypre_GpuProfilingPushRange("Section5");

   nalu_hypre_ParCSRMatrixColMapOffd(P)       = col_map_offd_P_host;
   nalu_hypre_ParCSRMatrixDeviceColMapOffd(P) = col_map_offd_P;
   nalu_hypre_CSRMatrixNumCols(P_offd)        = num_cols_offd_P;

   nalu_hypre_CSRMatrixMemoryLocation(P_diag) = NALU_HYPRE_MEMORY_DEVICE;
   nalu_hypre_CSRMatrixMemoryLocation(P_offd) = NALU_HYPRE_MEMORY_DEVICE;

   nalu_hypre_MatvecCommPkgCreate(P);

   for (i = 0; i < num_passes - 1; i++)
   {
      nalu_hypre_ParCSRMatrixDestroy(Pi[i]);
   }

   nalu_hypre_TFree(Pi,               NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(dof_func_offd,    NALU_HYPRE_MEMORY_DEVICE);
   nalu_hypre_TFree(pass_starts,      NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(pass_marker,      NALU_HYPRE_MEMORY_DEVICE);
   nalu_hypre_TFree(pass_marker_offd, NALU_HYPRE_MEMORY_DEVICE);
   nalu_hypre_TFree(pass_order,       NALU_HYPRE_MEMORY_DEVICE);
   nalu_hypre_TFree(fine_to_coarse,   NALU_HYPRE_MEMORY_DEVICE);

#if defined(NALU_HYPRE_USING_SYCL)
   NALU_HYPRE_ONEDPL_CALL( std::replace_if,
                      CF_marker,
                      CF_marker + n_fine,
                      equal<NALU_HYPRE_Int>(-3),
                      static_cast<NALU_HYPRE_Int>(-1) );
#else
   NALU_HYPRE_THRUST_CALL( replace_if,
                      CF_marker,
                      CF_marker + n_fine,
                      equal<NALU_HYPRE_Int>(-3),
                      static_cast<NALU_HYPRE_Int>(-1) );
#endif

   *P_ptr = P;

   nalu_hypre_GpuProfilingPopRange();

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_GenerateMultipassPiDevice( nalu_hypre_ParCSRMatrix  *A,
                                 nalu_hypre_ParCSRMatrix  *S,
                                 NALU_HYPRE_BigInt        *c_pts_starts,
                                 NALU_HYPRE_Int           *pass_order,
                                 NALU_HYPRE_Int           *pass_marker,
                                 NALU_HYPRE_Int           *pass_marker_offd,
                                 NALU_HYPRE_Int            num_points, /* |F| */
                                 NALU_HYPRE_Int            color, /* C-color */
                                 NALU_HYPRE_Real          *row_sums,
                                 nalu_hypre_ParCSRMatrix **P_ptr)
{
   MPI_Comm                comm     = nalu_hypre_ParCSRMatrixComm(A);
   nalu_hypre_ParCSRCommPkg    *comm_pkg = nalu_hypre_ParCSRMatrixCommPkg(A);
   nalu_hypre_ParCSRCommHandle *comm_handle;

   nalu_hypre_CSRMatrix *A_diag      = nalu_hypre_ParCSRMatrixDiag(A);
   NALU_HYPRE_Real      *A_diag_data = nalu_hypre_CSRMatrixData(A_diag);
   NALU_HYPRE_Int       *A_diag_i    = nalu_hypre_CSRMatrixI(A_diag);
   NALU_HYPRE_Int       *A_diag_j    = nalu_hypre_CSRMatrixJ(A_diag);
   NALU_HYPRE_Int        n_fine      = nalu_hypre_CSRMatrixNumRows(A_diag);

   nalu_hypre_CSRMatrix *A_offd          = nalu_hypre_ParCSRMatrixOffd(A);
   NALU_HYPRE_Int       *A_offd_i        = nalu_hypre_CSRMatrixI(A_offd);
   NALU_HYPRE_Int       *A_offd_j        = nalu_hypre_CSRMatrixJ(A_offd);
   NALU_HYPRE_Real      *A_offd_data     = nalu_hypre_CSRMatrixData(A_offd);
   NALU_HYPRE_Int        num_cols_offd_A = nalu_hypre_CSRMatrixNumCols(A_offd);

   nalu_hypre_CSRMatrix *S_diag   = nalu_hypre_ParCSRMatrixDiag(S);
   NALU_HYPRE_Int       *S_diag_i = nalu_hypre_CSRMatrixI(S_diag);
   NALU_HYPRE_Int       *S_diag_j = nalu_hypre_CSRMatrixJ(S_diag);

   nalu_hypre_CSRMatrix *S_offd   = nalu_hypre_ParCSRMatrixOffd(S);
   NALU_HYPRE_Int       *S_offd_i = nalu_hypre_CSRMatrixI(S_offd);
   NALU_HYPRE_Int       *S_offd_j = nalu_hypre_CSRMatrixJ(S_offd);

   NALU_HYPRE_Int *Soc_diag_j = nalu_hypre_ParCSRMatrixSocDiagJ(S);
   NALU_HYPRE_Int *Soc_offd_j = nalu_hypre_ParCSRMatrixSocOffdJ(S);

   NALU_HYPRE_BigInt    *col_map_offd_P     = NULL;
   NALU_HYPRE_BigInt    *col_map_offd_P_dev = NULL;
   NALU_HYPRE_Int        num_cols_offd_P;
   NALU_HYPRE_Int        nnz_diag, nnz_offd;

   nalu_hypre_ParCSRMatrix *P;
   nalu_hypre_CSRMatrix    *P_diag;
   NALU_HYPRE_Real         *P_diag_data;
   NALU_HYPRE_Int          *P_diag_i; /*at first counter of nonzero cols for each row,
                                   finally will be pointer to start of row */
   NALU_HYPRE_Int          *P_diag_j;
   nalu_hypre_CSRMatrix    *P_offd;
   NALU_HYPRE_Real         *P_offd_data = NULL;
   NALU_HYPRE_Int          *P_offd_i; /*at first counter of nonzero cols for each row,
                                   finally will be pointer to start of row */
   NALU_HYPRE_Int          *P_offd_j = NULL;

   NALU_HYPRE_Int       *fine_to_coarse;
   NALU_HYPRE_Int       *fine_to_coarse_offd = NULL;
   NALU_HYPRE_BigInt     f_pts_starts[2];
   NALU_HYPRE_Int        my_id, num_procs;
   NALU_HYPRE_BigInt     total_global_fpts;
   NALU_HYPRE_BigInt     total_global_cpts;
   NALU_HYPRE_BigInt    *big_convert_offd = NULL;
   NALU_HYPRE_BigInt    *big_buf_data = NULL;

   /* MPI size and rank*/
   nalu_hypre_MPI_Comm_size(comm, &num_procs);
   nalu_hypre_MPI_Comm_rank(comm, &my_id);

   fine_to_coarse = nalu_hypre_TAlloc(NALU_HYPRE_Int, n_fine, NALU_HYPRE_MEMORY_DEVICE);

   nalu_hypre_modmp_init_fine_to_coarse(n_fine, pass_marker, color, fine_to_coarse);

   if (num_procs > 1)
   {
      NALU_HYPRE_BigInt big_Fpts = num_points;

      nalu_hypre_MPI_Scan(&big_Fpts, f_pts_starts + 1, 1, NALU_HYPRE_MPI_BIG_INT, nalu_hypre_MPI_SUM, comm);

      f_pts_starts[0] = f_pts_starts[1] - big_Fpts;

      if (my_id == num_procs - 1)
      {
         total_global_fpts = f_pts_starts[1];
         total_global_cpts = c_pts_starts[1];
      }
      nalu_hypre_MPI_Bcast(&total_global_fpts, 1, NALU_HYPRE_MPI_BIG_INT, num_procs - 1, comm);
      nalu_hypre_MPI_Bcast(&total_global_cpts, 1, NALU_HYPRE_MPI_BIG_INT, num_procs - 1, comm);
   }
   else
   {
      f_pts_starts[0] = 0;
      f_pts_starts[1] = num_points;
      total_global_fpts = f_pts_starts[1];
      total_global_cpts = c_pts_starts[1];
   }

   num_cols_offd_P = 0;

   if (num_procs > 1)
   {
      NALU_HYPRE_Int num_sends = nalu_hypre_ParCSRCommPkgNumSends(comm_pkg);
      NALU_HYPRE_Int num_elem_send = nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends);

      big_convert_offd = nalu_hypre_TAlloc(NALU_HYPRE_BigInt, num_cols_offd_A, NALU_HYPRE_MEMORY_DEVICE);
      big_buf_data     = nalu_hypre_TAlloc(NALU_HYPRE_BigInt, num_elem_send,   NALU_HYPRE_MEMORY_DEVICE);

      globalC_functor functor(c_pts_starts[0]);

#if defined(NALU_HYPRE_USING_SYCL)
      hypreSycl_gather( nalu_hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg),
                        nalu_hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg) + num_elem_send,
                        oneapi::dpl::make_transform_iterator(fine_to_coarse, functor),
                        big_buf_data );
#else
      NALU_HYPRE_THRUST_CALL( gather,
                         nalu_hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg),
                         nalu_hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg) + num_elem_send,
                         thrust::make_transform_iterator(fine_to_coarse, functor),
                         big_buf_data );
#endif

#if defined(NALU_HYPRE_WITH_GPU_AWARE_MPI) && defined(NALU_HYPRE_USING_THRUST_NOSYNC)
      /* RL: make sure big_buf_data is ready before issuing GPU-GPU MPI */
      nalu_hypre_ForceSyncComputeStream(nalu_hypre_handle());
#endif

      comm_handle = nalu_hypre_ParCSRCommHandleCreate_v2(21, comm_pkg, NALU_HYPRE_MEMORY_DEVICE, big_buf_data,
                                                    NALU_HYPRE_MEMORY_DEVICE, big_convert_offd);

      nalu_hypre_ParCSRCommHandleDestroy(comm_handle);

      // This will allocate fine_to_coarse_offd
      nalu_hypre_modmp_compute_num_cols_offd_fine_to_coarse( pass_marker_offd, color, num_cols_offd_A,
                                                        num_cols_offd_P, &fine_to_coarse_offd );

      //FIXME: Clean this up when we don't need the host pointer anymore
      col_map_offd_P     = nalu_hypre_TAlloc(NALU_HYPRE_BigInt, num_cols_offd_P, NALU_HYPRE_MEMORY_HOST);
      col_map_offd_P_dev = nalu_hypre_TAlloc(NALU_HYPRE_BigInt, num_cols_offd_P, NALU_HYPRE_MEMORY_DEVICE);

#if defined(NALU_HYPRE_USING_SYCL)
      NALU_HYPRE_BigInt *col_map_end = hypreSycl_copy_if( big_convert_offd,
                                                     big_convert_offd + num_cols_offd_A,
                                                     pass_marker_offd,
                                                     col_map_offd_P_dev,
                                                     equal<NALU_HYPRE_Int>(color) );
#else
      NALU_HYPRE_BigInt *col_map_end = NALU_HYPRE_THRUST_CALL( copy_if,
                                                     big_convert_offd,
                                                     big_convert_offd + num_cols_offd_A,
                                                     pass_marker_offd,
                                                     col_map_offd_P_dev,
                                                     equal<NALU_HYPRE_Int>(color) );
#endif

      nalu_hypre_assert(num_cols_offd_P == col_map_end - col_map_offd_P_dev);

      //FIXME: Clean this up when we don't need the host pointer anymore
      nalu_hypre_TMemcpy(col_map_offd_P, col_map_offd_P_dev, NALU_HYPRE_BigInt, num_cols_offd_P, NALU_HYPRE_MEMORY_HOST,
                    NALU_HYPRE_MEMORY_DEVICE);

      nalu_hypre_TFree(big_convert_offd, NALU_HYPRE_MEMORY_DEVICE);
      nalu_hypre_TFree(big_buf_data,     NALU_HYPRE_MEMORY_DEVICE);
   }

   P_diag_i = nalu_hypre_TAlloc(NALU_HYPRE_Int, num_points + 1, NALU_HYPRE_MEMORY_DEVICE);
   P_offd_i = nalu_hypre_TAlloc(NALU_HYPRE_Int, num_points + 1, NALU_HYPRE_MEMORY_DEVICE);

   /* generate P_diag_i and P_offd_i */
   {
      dim3 bDim = nalu_hypre_GetDefaultDeviceBlockDimension();
      dim3 gDim = nalu_hypre_GetDefaultDeviceGridDimension(num_points, "warp", bDim);

      NALU_HYPRE_GPU_LAUNCH( hypreGPUKernel_generate_Pdiag_i_Poffd_i, gDim, bDim,
                        num_points, color, pass_order, pass_marker, pass_marker_offd,
                        S_diag_i, S_diag_j, S_offd_i, S_offd_j,
                        P_diag_i, P_offd_i );

      nalu_hypre_Memset(P_diag_i + num_points, 0, sizeof(NALU_HYPRE_Int), NALU_HYPRE_MEMORY_DEVICE);
      nalu_hypre_Memset(P_offd_i + num_points, 0, sizeof(NALU_HYPRE_Int), NALU_HYPRE_MEMORY_DEVICE);

#if defined(NALU_HYPRE_USING_SYCL)
      /* WM: todo - this is a workaround since oneDPL's exclusive_scan gives incorrect results when doing the scan in place */
      auto zip3 = oneapi::dpl::make_zip_iterator( P_diag_i, P_offd_i );
      NALU_HYPRE_Int *P_diag_i_tmp = nalu_hypre_CTAlloc(NALU_HYPRE_Int, num_points + 1, NALU_HYPRE_MEMORY_DEVICE);
      NALU_HYPRE_Int *P_offd_i_tmp = nalu_hypre_CTAlloc(NALU_HYPRE_Int, num_points + 1, NALU_HYPRE_MEMORY_DEVICE);
      NALU_HYPRE_ONEDPL_CALL( std::exclusive_scan,
                         zip3,
                         zip3 + num_points + 1,
                         oneapi::dpl::make_zip_iterator( P_diag_i_tmp, P_offd_i_tmp ),
                         std::make_tuple(NALU_HYPRE_Int(0), NALU_HYPRE_Int(0)),
                         tuple_plus<NALU_HYPRE_Int>() );
      nalu_hypre_TMemcpy(P_diag_i, P_diag_i_tmp, NALU_HYPRE_Int, num_points + 1, NALU_HYPRE_MEMORY_DEVICE,
                    NALU_HYPRE_MEMORY_DEVICE);
      nalu_hypre_TMemcpy(P_offd_i, P_offd_i_tmp, NALU_HYPRE_Int, num_points + 1, NALU_HYPRE_MEMORY_DEVICE,
                    NALU_HYPRE_MEMORY_DEVICE);
      nalu_hypre_TFree(P_diag_i_tmp, NALU_HYPRE_MEMORY_DEVICE);
      nalu_hypre_TFree(P_offd_i_tmp, NALU_HYPRE_MEMORY_DEVICE);
#else
      NALU_HYPRE_THRUST_CALL( exclusive_scan,
                         thrust::make_zip_iterator( thrust::make_tuple(P_diag_i, P_offd_i) ),
                         thrust::make_zip_iterator( thrust::make_tuple(P_diag_i, P_offd_i) ) + num_points + 1,
                         thrust::make_zip_iterator( thrust::make_tuple(P_diag_i, P_offd_i) ),
                         thrust::make_tuple(NALU_HYPRE_Int(0), NALU_HYPRE_Int(0)),
                         tuple_plus<NALU_HYPRE_Int>() );
#endif

      nalu_hypre_TMemcpy(&nnz_diag, &P_diag_i[num_points], NALU_HYPRE_Int, 1, NALU_HYPRE_MEMORY_HOST,
                    NALU_HYPRE_MEMORY_DEVICE);
      nalu_hypre_TMemcpy(&nnz_offd, &P_offd_i[num_points], NALU_HYPRE_Int, 1, NALU_HYPRE_MEMORY_HOST,
                    NALU_HYPRE_MEMORY_DEVICE);
   }

   /* generate P_diag_j/data and P_offd_j/data */
   P_diag_j    = nalu_hypre_TAlloc(NALU_HYPRE_Int,  nnz_diag, NALU_HYPRE_MEMORY_DEVICE);
   P_diag_data = nalu_hypre_TAlloc(NALU_HYPRE_Real, nnz_diag, NALU_HYPRE_MEMORY_DEVICE);
   P_offd_j    = nalu_hypre_TAlloc(NALU_HYPRE_Int,  nnz_offd, NALU_HYPRE_MEMORY_DEVICE);
   P_offd_data = nalu_hypre_TAlloc(NALU_HYPRE_Real, nnz_offd, NALU_HYPRE_MEMORY_DEVICE);

   {
      dim3 bDim = nalu_hypre_GetDefaultDeviceBlockDimension();
      dim3 gDim = nalu_hypre_GetDefaultDeviceGridDimension(num_points, "warp", bDim);

      NALU_HYPRE_GPU_LAUNCH( hypreGPUKernel_generate_Pdiag_j_Poffd_j, gDim, bDim,
                        num_points,
                        color,
                        pass_order,
                        pass_marker,
                        pass_marker_offd,
                        fine_to_coarse,
                        fine_to_coarse_offd,
                        A_diag_i,
                        A_diag_j,
                        A_diag_data,
                        A_offd_i,
                        A_offd_j,
                        A_offd_data,
                        Soc_diag_j,
                        Soc_offd_j,
                        P_diag_i,
                        P_offd_i,
                        P_diag_j,
                        P_diag_data,
                        P_offd_j,
                        P_offd_data,
                        row_sums );
   }

   nalu_hypre_TFree(fine_to_coarse,      NALU_HYPRE_MEMORY_DEVICE);
   nalu_hypre_TFree(fine_to_coarse_offd, NALU_HYPRE_MEMORY_DEVICE);

   P = nalu_hypre_ParCSRMatrixCreate(comm,
                                total_global_fpts,
                                total_global_cpts,
                                f_pts_starts,
                                c_pts_starts,
                                num_cols_offd_P,
                                nnz_diag,
                                nnz_offd);

   P_diag = nalu_hypre_ParCSRMatrixDiag(P);
   nalu_hypre_CSRMatrixData(P_diag) = P_diag_data;
   nalu_hypre_CSRMatrixI(P_diag)    = P_diag_i;
   nalu_hypre_CSRMatrixJ(P_diag)    = P_diag_j;

   P_offd = nalu_hypre_ParCSRMatrixOffd(P);
   nalu_hypre_CSRMatrixData(P_offd) = P_offd_data;
   nalu_hypre_CSRMatrixI(P_offd)    = P_offd_i;
   nalu_hypre_CSRMatrixJ(P_offd)    = P_offd_j;

   nalu_hypre_ParCSRMatrixColMapOffd(P) = col_map_offd_P;
   nalu_hypre_ParCSRMatrixDeviceColMapOffd(P) = col_map_offd_P_dev;

   nalu_hypre_CSRMatrixMemoryLocation(P_diag) = NALU_HYPRE_MEMORY_DEVICE;
   nalu_hypre_CSRMatrixMemoryLocation(P_offd) = NALU_HYPRE_MEMORY_DEVICE;

   nalu_hypre_MatvecCommPkgCreate(P);

   *P_ptr = P;

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_GenerateMultiPiDevice( nalu_hypre_ParCSRMatrix  *A,
                             nalu_hypre_ParCSRMatrix  *S,
                             nalu_hypre_ParCSRMatrix  *P,
                             NALU_HYPRE_BigInt        *c_pts_starts,
                             NALU_HYPRE_Int           *pass_order,
                             NALU_HYPRE_Int           *pass_marker,
                             NALU_HYPRE_Int           *pass_marker_offd,
                             NALU_HYPRE_Int            num_points,
                             NALU_HYPRE_Int            color,
                             NALU_HYPRE_Int            num_functions,
                             NALU_HYPRE_Int           *dof_func,
                             NALU_HYPRE_Int           *dof_func_offd,
                             nalu_hypre_ParCSRMatrix **Pi_ptr )
{
   MPI_Comm                comm = nalu_hypre_ParCSRMatrixComm(A);
   nalu_hypre_ParCSRCommPkg    *comm_pkg = nalu_hypre_ParCSRMatrixCommPkg(A);
   nalu_hypre_ParCSRCommHandle *comm_handle;

   nalu_hypre_CSRMatrix *A_diag      = nalu_hypre_ParCSRMatrixDiag(A);
   NALU_HYPRE_Real      *A_diag_data = nalu_hypre_CSRMatrixData(A_diag);
   NALU_HYPRE_Int       *A_diag_i    = nalu_hypre_CSRMatrixI(A_diag);
   NALU_HYPRE_Int       *A_diag_j    = nalu_hypre_CSRMatrixJ(A_diag);
   NALU_HYPRE_Int        n_fine      = nalu_hypre_CSRMatrixNumRows(A_diag);

   nalu_hypre_CSRMatrix *A_offd          = nalu_hypre_ParCSRMatrixOffd(A);
   NALU_HYPRE_Int       *A_offd_i        = nalu_hypre_CSRMatrixI(A_offd);
   NALU_HYPRE_Int       *A_offd_j        = nalu_hypre_CSRMatrixJ(A_offd);
   NALU_HYPRE_Real      *A_offd_data     = nalu_hypre_CSRMatrixData(A_offd);
   NALU_HYPRE_Int        num_cols_offd_A = nalu_hypre_CSRMatrixNumCols(A_offd);

   nalu_hypre_CSRMatrix *S_diag   = nalu_hypre_ParCSRMatrixDiag(S);
   NALU_HYPRE_Int       *S_diag_i = nalu_hypre_CSRMatrixI(S_diag);
   NALU_HYPRE_Int       *S_diag_j = nalu_hypre_CSRMatrixJ(S_diag);

   nalu_hypre_CSRMatrix *S_offd   = nalu_hypre_ParCSRMatrixOffd(S);
   NALU_HYPRE_Int       *S_offd_i = nalu_hypre_CSRMatrixI(S_offd);
   NALU_HYPRE_Int       *S_offd_j = nalu_hypre_CSRMatrixJ(S_offd);

   NALU_HYPRE_Int *Soc_diag_j = nalu_hypre_ParCSRMatrixSocDiagJ(S);
   NALU_HYPRE_Int *Soc_offd_j = nalu_hypre_ParCSRMatrixSocOffdJ(S);

   NALU_HYPRE_BigInt    *col_map_offd_Q     = NULL;
   NALU_HYPRE_BigInt    *col_map_offd_Q_dev = NULL;
   NALU_HYPRE_Int        num_cols_offd_Q;

   nalu_hypre_ParCSRMatrix *Pi;
   nalu_hypre_CSRMatrix    *Pi_diag;
   NALU_HYPRE_Int          *Pi_diag_i;
   NALU_HYPRE_Real         *Pi_diag_data;
   nalu_hypre_CSRMatrix    *Pi_offd;
   NALU_HYPRE_Int          *Pi_offd_i;
   NALU_HYPRE_Real         *Pi_offd_data;

   NALU_HYPRE_Int           nnz_diag, nnz_offd;

   nalu_hypre_ParCSRMatrix *Q;
   nalu_hypre_CSRMatrix    *Q_diag;
   NALU_HYPRE_Real         *Q_diag_data;
   NALU_HYPRE_Int          *Q_diag_i; /*at first counter of nonzero cols for each row,
                                   finally will be pointer to start of row */
   NALU_HYPRE_Int          *Q_diag_j;
   nalu_hypre_CSRMatrix    *Q_offd;
   NALU_HYPRE_Real         *Q_offd_data = NULL;
   NALU_HYPRE_Int          *Q_offd_i; /*at first counter of nonzero cols for each row,
                                  finally will be pointer to start of row */
   NALU_HYPRE_Int          *Q_offd_j = NULL;

   NALU_HYPRE_Int       *fine_to_coarse;
   NALU_HYPRE_Int       *fine_to_coarse_offd = NULL;
   NALU_HYPRE_BigInt     f_pts_starts[2];
   NALU_HYPRE_Int        my_id, num_procs;
   NALU_HYPRE_BigInt     total_global_fpts;
   NALU_HYPRE_BigInt     total_global_cpts;
   NALU_HYPRE_BigInt    *big_convert_offd = NULL;
   NALU_HYPRE_BigInt    *big_buf_data = NULL;
   NALU_HYPRE_Real      *w_row_sum;

   /* MPI size and rank*/
   nalu_hypre_MPI_Comm_size(comm, &num_procs);
   nalu_hypre_MPI_Comm_rank(comm, &my_id);

   fine_to_coarse = nalu_hypre_TAlloc(NALU_HYPRE_Int, n_fine, NALU_HYPRE_MEMORY_DEVICE);

   nalu_hypre_modmp_init_fine_to_coarse(n_fine, pass_marker, color, fine_to_coarse);

   if (num_procs > 1)
   {
      NALU_HYPRE_BigInt big_Fpts = num_points;

      nalu_hypre_MPI_Scan(&big_Fpts, f_pts_starts + 1, 1, NALU_HYPRE_MPI_BIG_INT, nalu_hypre_MPI_SUM, comm);

      f_pts_starts[0] = f_pts_starts[1] - big_Fpts;

      if (my_id == num_procs - 1)
      {
         total_global_fpts = f_pts_starts[1];
         total_global_cpts = c_pts_starts[1];
      }
      nalu_hypre_MPI_Bcast(&total_global_fpts, 1, NALU_HYPRE_MPI_BIG_INT, num_procs - 1, comm);
      nalu_hypre_MPI_Bcast(&total_global_cpts, 1, NALU_HYPRE_MPI_BIG_INT, num_procs - 1, comm);
   }
   else
   {
      f_pts_starts[0] = 0;
      f_pts_starts[1] = num_points;
      total_global_fpts = f_pts_starts[1];
      total_global_cpts = c_pts_starts[1];
   }

   num_cols_offd_Q = 0;

   if (num_procs > 1)
   {
      NALU_HYPRE_Int num_sends = nalu_hypre_ParCSRCommPkgNumSends(comm_pkg);
      NALU_HYPRE_Int num_elem_send = nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends);

      big_convert_offd = nalu_hypre_TAlloc(NALU_HYPRE_BigInt, num_cols_offd_A, NALU_HYPRE_MEMORY_DEVICE);
      big_buf_data     = nalu_hypre_TAlloc(NALU_HYPRE_BigInt, num_elem_send,   NALU_HYPRE_MEMORY_DEVICE);

      globalC_functor functor(c_pts_starts[0]);

#if defined(NALU_HYPRE_USING_SYCL)
      hypreSycl_gather( nalu_hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg),
                        nalu_hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg) + num_elem_send,
                        oneapi::dpl::make_transform_iterator(fine_to_coarse, functor),
                        big_buf_data );
#else
      NALU_HYPRE_THRUST_CALL( gather,
                         nalu_hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg),
                         nalu_hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg) + num_elem_send,
                         thrust::make_transform_iterator(fine_to_coarse, functor),
                         big_buf_data );
#endif

#if defined(NALU_HYPRE_WITH_GPU_AWARE_MPI) && defined(NALU_HYPRE_USING_THRUST_NOSYNC)
      /* RL: make sure big_buf_data is ready before issuing GPU-GPU MPI */
      nalu_hypre_ForceSyncComputeStream(nalu_hypre_handle());
#endif

      comm_handle = nalu_hypre_ParCSRCommHandleCreate_v2(21, comm_pkg, NALU_HYPRE_MEMORY_DEVICE, big_buf_data,
                                                    NALU_HYPRE_MEMORY_DEVICE, big_convert_offd);

      nalu_hypre_ParCSRCommHandleDestroy(comm_handle);

      // This will allocate fine_to_coarse_offd_dev
      nalu_hypre_modmp_compute_num_cols_offd_fine_to_coarse( pass_marker_offd, color, num_cols_offd_A,
                                                        num_cols_offd_Q, &fine_to_coarse_offd );

      //FIXME: PB: It seems we need the host value too?!?!
      col_map_offd_Q     = nalu_hypre_TAlloc(NALU_HYPRE_BigInt, num_cols_offd_Q, NALU_HYPRE_MEMORY_HOST);
      col_map_offd_Q_dev = nalu_hypre_TAlloc(NALU_HYPRE_BigInt, num_cols_offd_Q, NALU_HYPRE_MEMORY_DEVICE);

#if defined(NALU_HYPRE_USING_SYCL)
      NALU_HYPRE_BigInt *col_map_end = hypreSycl_copy_if( big_convert_offd,
                                                     big_convert_offd + num_cols_offd_A,
                                                     pass_marker_offd,
                                                     col_map_offd_Q_dev,
                                                     equal<NALU_HYPRE_Int>(color) );
#else
      NALU_HYPRE_BigInt *col_map_end = NALU_HYPRE_THRUST_CALL( copy_if,
                                                     big_convert_offd,
                                                     big_convert_offd + num_cols_offd_A,
                                                     pass_marker_offd,
                                                     col_map_offd_Q_dev,
                                                     equal<NALU_HYPRE_Int>(color) );
#endif

      nalu_hypre_assert(num_cols_offd_Q == col_map_end - col_map_offd_Q_dev);

      //FIXME: PB: It seems like we're required to have a host version of this??
      nalu_hypre_TMemcpy(col_map_offd_Q, col_map_offd_Q_dev, NALU_HYPRE_BigInt, num_cols_offd_Q, NALU_HYPRE_MEMORY_HOST,
                    NALU_HYPRE_MEMORY_DEVICE);

      nalu_hypre_TFree(big_convert_offd, NALU_HYPRE_MEMORY_DEVICE );
      nalu_hypre_TFree(big_buf_data, NALU_HYPRE_MEMORY_DEVICE);
   }

   Q_diag_i = nalu_hypre_TAlloc(NALU_HYPRE_Int, num_points + 1, NALU_HYPRE_MEMORY_DEVICE);
   Q_offd_i = nalu_hypre_TAlloc(NALU_HYPRE_Int, num_points + 1, NALU_HYPRE_MEMORY_DEVICE);

   /* generate Q_diag_i and Q_offd_i */
   {
      dim3 bDim = nalu_hypre_GetDefaultDeviceBlockDimension();
      dim3 gDim = nalu_hypre_GetDefaultDeviceGridDimension(num_points, "warp", bDim);

      NALU_HYPRE_GPU_LAUNCH( hypreGPUKernel_generate_Pdiag_i_Poffd_i, gDim, bDim,
                        num_points, color, pass_order, pass_marker, pass_marker_offd,
                        S_diag_i, S_diag_j, S_offd_i, S_offd_j,
                        Q_diag_i, Q_offd_i );

      nalu_hypre_Memset(Q_diag_i + num_points, 0, sizeof(NALU_HYPRE_Int), NALU_HYPRE_MEMORY_DEVICE);
      nalu_hypre_Memset(Q_offd_i + num_points, 0, sizeof(NALU_HYPRE_Int), NALU_HYPRE_MEMORY_DEVICE);

#if defined(NALU_HYPRE_USING_SYCL)
      /* WM: todo - this is a workaround since oneDPL's exclusive_scan gives incorrect results when doing the scan in place */
      auto zip4 = oneapi::dpl::make_zip_iterator( Q_diag_i, Q_offd_i );
      NALU_HYPRE_Int *Q_diag_i_tmp = nalu_hypre_CTAlloc(NALU_HYPRE_Int, num_points + 1, NALU_HYPRE_MEMORY_DEVICE);
      NALU_HYPRE_Int *Q_offd_i_tmp = nalu_hypre_CTAlloc(NALU_HYPRE_Int, num_points + 1, NALU_HYPRE_MEMORY_DEVICE);
      NALU_HYPRE_ONEDPL_CALL( std::exclusive_scan,
                         zip4,
                         zip4 + num_points + 1,
                         oneapi::dpl::make_zip_iterator( Q_diag_i_tmp, Q_offd_i_tmp ),
                         std::make_tuple(NALU_HYPRE_Int(0), NALU_HYPRE_Int(0)),
                         tuple_plus<NALU_HYPRE_Int>() );
      nalu_hypre_TMemcpy(Q_diag_i, Q_diag_i_tmp, NALU_HYPRE_Int, num_points + 1, NALU_HYPRE_MEMORY_DEVICE,
                    NALU_HYPRE_MEMORY_DEVICE);
      nalu_hypre_TMemcpy(Q_offd_i, Q_offd_i_tmp, NALU_HYPRE_Int, num_points + 1, NALU_HYPRE_MEMORY_DEVICE,
                    NALU_HYPRE_MEMORY_DEVICE);
      nalu_hypre_TFree(Q_diag_i_tmp, NALU_HYPRE_MEMORY_DEVICE);
      nalu_hypre_TFree(Q_offd_i_tmp, NALU_HYPRE_MEMORY_DEVICE);
#else
      NALU_HYPRE_THRUST_CALL( exclusive_scan,
                         thrust::make_zip_iterator( thrust::make_tuple(Q_diag_i, Q_offd_i) ),
                         thrust::make_zip_iterator( thrust::make_tuple(Q_diag_i, Q_offd_i) ) + num_points + 1,
                         thrust::make_zip_iterator( thrust::make_tuple(Q_diag_i, Q_offd_i) ),
                         thrust::make_tuple(NALU_HYPRE_Int(0), NALU_HYPRE_Int(0)),
                         tuple_plus<NALU_HYPRE_Int>() );
#endif

      nalu_hypre_TMemcpy(&nnz_diag, &Q_diag_i[num_points], NALU_HYPRE_Int, 1, NALU_HYPRE_MEMORY_HOST,
                    NALU_HYPRE_MEMORY_DEVICE);
      nalu_hypre_TMemcpy(&nnz_offd, &Q_offd_i[num_points], NALU_HYPRE_Int, 1, NALU_HYPRE_MEMORY_HOST,
                    NALU_HYPRE_MEMORY_DEVICE);
   }

   /* generate P_diag_j/data and P_offd_j/data */
   Q_diag_j    = nalu_hypre_TAlloc(NALU_HYPRE_Int,  nnz_diag,   NALU_HYPRE_MEMORY_DEVICE);
   Q_diag_data = nalu_hypre_TAlloc(NALU_HYPRE_Real, nnz_diag,   NALU_HYPRE_MEMORY_DEVICE);
   Q_offd_j    = nalu_hypre_TAlloc(NALU_HYPRE_Int,  nnz_offd,   NALU_HYPRE_MEMORY_DEVICE);
   Q_offd_data = nalu_hypre_TAlloc(NALU_HYPRE_Real, nnz_offd,   NALU_HYPRE_MEMORY_DEVICE);
   w_row_sum   = nalu_hypre_TAlloc(NALU_HYPRE_Real, num_points, NALU_HYPRE_MEMORY_DEVICE);

   {
      dim3 bDim = nalu_hypre_GetDefaultDeviceBlockDimension();
      dim3 gDim = nalu_hypre_GetDefaultDeviceGridDimension(num_points, "warp", bDim);

      NALU_HYPRE_GPU_LAUNCH( hypreGPUKernel_generate_Qdiag_j_Qoffd_j, gDim, bDim,
                        num_points,
                        color,
                        pass_order,
                        pass_marker,
                        pass_marker_offd,
                        fine_to_coarse,
                        fine_to_coarse_offd,
                        A_diag_i,
                        A_diag_j,
                        A_diag_data,
                        A_offd_i,
                        A_offd_j,
                        A_offd_data,
                        Soc_diag_j,
                        Soc_offd_j,
                        Q_diag_i,
                        Q_offd_i,
                        Q_diag_j,
                        Q_diag_data,
                        Q_offd_j,
                        Q_offd_data,
                        w_row_sum,
                        num_functions,
                        dof_func,
                        dof_func_offd );
   }

   nalu_hypre_TFree(fine_to_coarse,      NALU_HYPRE_MEMORY_DEVICE);
   nalu_hypre_TFree(fine_to_coarse_offd, NALU_HYPRE_MEMORY_DEVICE);

   Q = nalu_hypre_ParCSRMatrixCreate(comm,
                                total_global_fpts,
                                total_global_cpts,
                                f_pts_starts,
                                c_pts_starts,
                                num_cols_offd_Q,
                                nnz_diag,
                                nnz_offd);

   Q_diag = nalu_hypre_ParCSRMatrixDiag(Q);
   nalu_hypre_CSRMatrixData(Q_diag) = Q_diag_data;
   nalu_hypre_CSRMatrixI(Q_diag)    = Q_diag_i;
   nalu_hypre_CSRMatrixJ(Q_diag)    = Q_diag_j;

   Q_offd = nalu_hypre_ParCSRMatrixOffd(Q);
   nalu_hypre_CSRMatrixData(Q_offd) = Q_offd_data;
   nalu_hypre_CSRMatrixI(Q_offd)    = Q_offd_i;
   nalu_hypre_CSRMatrixJ(Q_offd)    = Q_offd_j;

   nalu_hypre_ParCSRMatrixColMapOffd(Q) = col_map_offd_Q;
   nalu_hypre_ParCSRMatrixDeviceColMapOffd(Q) = col_map_offd_Q_dev;

   nalu_hypre_CSRMatrixMemoryLocation(Q_diag) = NALU_HYPRE_MEMORY_DEVICE;
   nalu_hypre_CSRMatrixMemoryLocation(Q_offd) = NALU_HYPRE_MEMORY_DEVICE;

   nalu_hypre_MatvecCommPkgCreate(Q);

   Pi = nalu_hypre_ParCSRMatMat(Q, P);

   Pi_diag = nalu_hypre_ParCSRMatrixDiag(Pi);
   Pi_diag_data = nalu_hypre_CSRMatrixData(Pi_diag);
   Pi_diag_i = nalu_hypre_CSRMatrixI(Pi_diag);
   Pi_offd = nalu_hypre_ParCSRMatrixOffd(Pi);
   Pi_offd_data = nalu_hypre_CSRMatrixData(Pi_offd);
   Pi_offd_i = nalu_hypre_CSRMatrixI(Pi_offd);

   {
      dim3 bDim = nalu_hypre_GetDefaultDeviceBlockDimension();
      dim3 gDim = nalu_hypre_GetDefaultDeviceGridDimension(num_points, "warp", bDim);

      NALU_HYPRE_GPU_LAUNCH( hypreGPUKernel_mutli_pi_rowsum, gDim, bDim,
                        num_points, pass_order, A_diag_i, A_diag_data,
                        Pi_diag_i, Pi_diag_data, Pi_offd_i, Pi_offd_data,
                        w_row_sum );
   }

   nalu_hypre_TFree(w_row_sum, NALU_HYPRE_MEMORY_DEVICE);

   nalu_hypre_ParCSRMatrixDestroy(Q);

   *Pi_ptr = Pi;

   return nalu_hypre_error_flag;
}

void nalu_hypre_modmp_init_fine_to_coarse( NALU_HYPRE_Int  n_fine,
                                      NALU_HYPRE_Int *pass_marker,
                                      NALU_HYPRE_Int  color,
                                      NALU_HYPRE_Int *fine_to_coarse )
{
   // n_fine == pass_marker.size()
   // Host code this is replacing:
   // n_cpts = 0;
   // for (NALU_HYPRE_Int i=0; i < n_fine; i++)
   //  {
   //    if (pass_marker[i] == color)
   //      fine_to_coarse[i] = n_cpts++;
   //    else
   //      fine_to_coarse[i] = -1;
   //  }

   if (n_fine == 0)
   {
      return;
   }

#if defined(NALU_HYPRE_USING_SYCL)
   NALU_HYPRE_ONEDPL_CALL( std::exclusive_scan,
                      oneapi::dpl::make_transform_iterator(pass_marker,          equal<NALU_HYPRE_Int>(color)),
                      oneapi::dpl::make_transform_iterator(pass_marker + n_fine, equal<NALU_HYPRE_Int>(color)),
                      fine_to_coarse,
                      NALU_HYPRE_Int(0) );

   hypreSycl_transform_if( fine_to_coarse,
                           fine_to_coarse + n_fine,
                           pass_marker,
                           fine_to_coarse,
   [] (const auto & x) {return -1;},
   [color = color] (const auto & x) {return x != color;} );
#else
   NALU_HYPRE_THRUST_CALL( exclusive_scan,
                      thrust::make_transform_iterator(pass_marker,          equal<NALU_HYPRE_Int>(color)),
                      thrust::make_transform_iterator(pass_marker + n_fine, equal<NALU_HYPRE_Int>(color)),
                      fine_to_coarse,
                      NALU_HYPRE_Int(0) );

   NALU_HYPRE_THRUST_CALL( replace_if,
                      fine_to_coarse,
                      fine_to_coarse + n_fine,
                      pass_marker,
                      thrust::not1(equal<NALU_HYPRE_Int>(color)),
                      -1 );
#endif
}

void
nalu_hypre_modmp_compute_num_cols_offd_fine_to_coarse( NALU_HYPRE_Int  *pass_marker_offd,
                                                  NALU_HYPRE_Int   color,
                                                  NALU_HYPRE_Int   num_cols_offd_A,
                                                  NALU_HYPRE_Int  &num_cols_offd,
                                                  NALU_HYPRE_Int **fine_to_coarse_offd_ptr )
{
   // We allocate with a "+1" because the host version of this code incremented the counter
   // even on the last match, so we create an extra entry the exclusive_scan will reflect this
   // and we can read off the last entry and only do 1 kernel call and 1 memcpy
   // RL: this trick requires pass_marker_offd has 1 more space allocated too
   NALU_HYPRE_Int *fine_to_coarse_offd = nalu_hypre_TAlloc(NALU_HYPRE_Int, num_cols_offd_A + 1, NALU_HYPRE_MEMORY_DEVICE);

#if defined(NALU_HYPRE_USING_SYCL)
   NALU_HYPRE_ONEDPL_CALL( std::exclusive_scan,
                      oneapi::dpl::make_transform_iterator(pass_marker_offd,
                                                           equal<NALU_HYPRE_Int>(color)),
                      oneapi::dpl::make_transform_iterator(pass_marker_offd + num_cols_offd_A + 1,
                                                           equal<NALU_HYPRE_Int>(color)),
                      fine_to_coarse_offd,
                      NALU_HYPRE_Int(0) );
#else
   NALU_HYPRE_THRUST_CALL( exclusive_scan,
                      thrust::make_transform_iterator(pass_marker_offd,                       equal<NALU_HYPRE_Int>(color)),
                      thrust::make_transform_iterator(pass_marker_offd + num_cols_offd_A + 1, equal<NALU_HYPRE_Int>(color)),
                      fine_to_coarse_offd,
                      NALU_HYPRE_Int(0) );
#endif

   nalu_hypre_TMemcpy( &num_cols_offd, fine_to_coarse_offd + num_cols_offd_A, NALU_HYPRE_Int, 1,
                  NALU_HYPRE_MEMORY_HOST, NALU_HYPRE_MEMORY_DEVICE);

   *fine_to_coarse_offd_ptr = fine_to_coarse_offd;
}

__global__
void hypreGPUKernel_cfmarker_masked_rowsum( nalu_hypre_DeviceItem    &item,
                                            NALU_HYPRE_Int      nrows,
                                            NALU_HYPRE_Int     *A_diag_i,
                                            NALU_HYPRE_Int     *A_diag_j,
                                            NALU_HYPRE_Complex *A_diag_data,
                                            NALU_HYPRE_Int     *A_offd_i,
                                            NALU_HYPRE_Int     *A_offd_j,
                                            NALU_HYPRE_Complex *A_offd_data,
                                            NALU_HYPRE_Int     *CF_marker,
                                            NALU_HYPRE_Int     *dof_func,
                                            NALU_HYPRE_Int     *dof_func_offd,
                                            NALU_HYPRE_Complex *row_sums )
{
   NALU_HYPRE_Int row_i = nalu_hypre_gpu_get_grid_warp_id<1, 1>(item);

   if (row_i >= nrows || read_only_load(&CF_marker[row_i]) >= 0)
   {
      return;
   }

   NALU_HYPRE_Int lane = nalu_hypre_gpu_get_lane_id<1>(item);
   NALU_HYPRE_Int p = 0;
   NALU_HYPRE_Int q = 0;
   NALU_HYPRE_Int func_i = dof_func ? read_only_load(&dof_func[row_i]) : 0;

   // A_diag part
   if (lane < 2)
   {
      p = read_only_load(A_diag_i + row_i + lane);
   }
   q = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, p, 1);
   p = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, p, 0);

   NALU_HYPRE_Complex row_sum_i = 0.0;

   // exclude diagonal: do not assume it is the first entry
   for (NALU_HYPRE_Int j = p + lane; j < q; j += NALU_HYPRE_WARP_SIZE)
   {
      NALU_HYPRE_Int col = read_only_load(&A_diag_j[j]);

      if (row_i != col)
      {
         NALU_HYPRE_Int func_j = dof_func ? read_only_load(&dof_func[col]) : 0;

         if (func_i == func_j)
         {
            NALU_HYPRE_Complex value = read_only_load(&A_diag_data[j]);
            row_sum_i += value;
         }
      }
   }

   // A_offd part
   if (lane < 2)
   {
      p = read_only_load(A_offd_i + row_i + lane);
   }
   q = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, p, 1);
   p = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, p, 0);

   for (NALU_HYPRE_Int j = p + lane; j < q; j += NALU_HYPRE_WARP_SIZE)
   {
      NALU_HYPRE_Int func_j = 0;
      if (dof_func_offd)
      {
         NALU_HYPRE_Int col = read_only_load(&A_offd_j[j]);
         func_j = read_only_load(&dof_func_offd[col]);
      }

      if (func_i == func_j)
      {
         NALU_HYPRE_Complex value = read_only_load(&A_offd_data[j]);
         row_sum_i += value;
      }
   }

   row_sum_i = warp_reduce_sum(item, row_sum_i);

   if (lane == 0)
   {
      row_sums[row_i] = row_sum_i;
   }
}

__global__
void hypreGPUKernel_mutli_pi_rowsum( nalu_hypre_DeviceItem    &item,
                                     NALU_HYPRE_Int      num_points,
                                     NALU_HYPRE_Int     *pass_order,
                                     NALU_HYPRE_Int     *A_diag_i,
                                     NALU_HYPRE_Complex *A_diag_data,
                                     NALU_HYPRE_Int     *Pi_diag_i,
                                     NALU_HYPRE_Complex *Pi_diag_data,
                                     NALU_HYPRE_Int     *Pi_offd_i,
                                     NALU_HYPRE_Complex *Pi_offd_data,
                                     NALU_HYPRE_Complex *w_row_sum )
{
   NALU_HYPRE_Int row_i = nalu_hypre_gpu_get_grid_warp_id<1, 1>(item);

   if (row_i >= num_points)
   {
      return;
   }

   NALU_HYPRE_Int lane = nalu_hypre_gpu_get_lane_id<1>(item);
   NALU_HYPRE_Int p_diag = 0, q_diag = 0, p_offd = 0, q_offd = 0;
   NALU_HYPRE_Real row_sum_C = 0.0;

   // Pi_diag
   if (lane < 2)
   {
      p_diag = read_only_load(Pi_diag_i + row_i + lane);
   }
   q_diag = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, p_diag, 1);
   p_diag = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, p_diag, 0);

   for (NALU_HYPRE_Int j = p_diag + lane; j < q_diag; j += NALU_HYPRE_WARP_SIZE)
   {
      row_sum_C += read_only_load(&Pi_diag_data[j]);
   }

   // Pi_offd
   if (lane < 2)
   {
      p_offd = read_only_load(Pi_offd_i + row_i + lane);
   }
   q_offd = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, p_offd, 1);
   p_offd = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, p_offd, 0);

   for (NALU_HYPRE_Int j = p_offd + lane; j < q_offd; j += NALU_HYPRE_WARP_SIZE)
   {
      row_sum_C += read_only_load(&Pi_offd_data[j]);
   }

   row_sum_C = warp_reduce_sum(item, row_sum_C);

   if ( lane == 0 )
   {
      const NALU_HYPRE_Int i1 = read_only_load(&pass_order[row_i]);
      const NALU_HYPRE_Int j1 = read_only_load(&A_diag_i[i1]);
      //XXX RL: rely on diagonal is the first of row [FIX?]
      const NALU_HYPRE_Real diagonal = read_only_load(&A_diag_data[j1]);
      const NALU_HYPRE_Real value = row_sum_C * diagonal;
      row_sum_C += read_only_load(&w_row_sum[row_i]);

      if ( value != 0.0 )
      {
         row_sum_C /= value;
      }
   }

   row_sum_C = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, row_sum_C, 0);

   // Pi_diag
   for (NALU_HYPRE_Int j = p_diag + lane; j < q_diag; j += NALU_HYPRE_WARP_SIZE)
   {
      Pi_diag_data[j] *= -row_sum_C;
   }

   // Pi_offd
   for (NALU_HYPRE_Int j = p_offd + lane; j < q_offd; j += NALU_HYPRE_WARP_SIZE)
   {
      Pi_offd_data[j] *= -row_sum_C;
   }
}

__global__
void hypreGPUKernel_generate_Pdiag_i_Poffd_i( nalu_hypre_DeviceItem &item,
                                              NALU_HYPRE_Int  num_points,
                                              NALU_HYPRE_Int  color,
                                              NALU_HYPRE_Int *pass_order,
                                              NALU_HYPRE_Int *pass_marker,
                                              NALU_HYPRE_Int *pass_marker_offd,
                                              NALU_HYPRE_Int *S_diag_i,
                                              NALU_HYPRE_Int *S_diag_j,
                                              NALU_HYPRE_Int *S_offd_i,
                                              NALU_HYPRE_Int *S_offd_j,
                                              NALU_HYPRE_Int *P_diag_i,
                                              NALU_HYPRE_Int *P_offd_i )
{
   /*
    nnz_diag = 0;
    nnz_offd = 0;
    for (i=0; i < num_points; i++)
    {
      i1 = pass_order[i];
      for (j=S_diag_i[i1]; j < S_diag_i[i1+1]; j++)
      {
         j1 = S_diag_j[j];
         if (pass_marker[j1] == color)
         {
             P_diag_i[i]++;
             nnz_diag++;
         }
      }
      for (j=S_offd_i[i1]; j < S_offd_i[i1+1]; j++)
      {
         j1 = S_offd_j[j];
         if (pass_marker_offd[j1] == color)
         {
             P_offd_i[i]++;
             nnz_offd++;
         }
      }
    }
   */

   NALU_HYPRE_Int row_i = nalu_hypre_gpu_get_grid_warp_id<1, 1>(item);

   if (row_i >= num_points)
   {
      return;
   }

   NALU_HYPRE_Int i1 = read_only_load(&pass_order[row_i]);
   NALU_HYPRE_Int lane = nalu_hypre_gpu_get_lane_id<1>(item);
   NALU_HYPRE_Int p = 0;
   NALU_HYPRE_Int q = 0;
   NALU_HYPRE_Int diag_increment = 0;
   NALU_HYPRE_Int offd_increment = 0;

   // S_diag
   if (lane < 2)
   {
      p = read_only_load(S_diag_i + i1 + lane);
   }
   q = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, p, 1);
   p = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, p, 0);

   for (NALU_HYPRE_Int j = p + lane; j < q; j += NALU_HYPRE_WARP_SIZE)
   {
      const NALU_HYPRE_Int j1 = read_only_load(&S_diag_j[j]);
      const NALU_HYPRE_Int marker = read_only_load(&pass_marker[j1]);

      diag_increment += marker == color;
   }

   diag_increment = warp_reduce_sum(item, diag_increment);

   // Increment P_diag_i, but then we need to also do a block reduction
   // on diag_increment to log the total nnz_diag for the block
   // Then after the kernel, we'll accumulate nnz_diag for each block
   if (lane == 0)
   {
      P_diag_i[row_i] = diag_increment;
   }

   // S_offd
   if (lane < 2)
   {
      p = read_only_load(S_offd_i + i1 + lane);
   }
   q = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, p, 1);
   p = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, p, 0);

   for (NALU_HYPRE_Int j = p + lane; j < q; j += NALU_HYPRE_WARP_SIZE)
   {
      const NALU_HYPRE_Int j1 = read_only_load(&S_offd_j[j]);
      const NALU_HYPRE_Int marker = read_only_load(&pass_marker_offd[j1]);

      offd_increment += marker == color;
   }

   offd_increment = warp_reduce_sum(item, offd_increment);

   // Increment P_offd_i, but then we need to also do a block reduction
   // on offd_increment to log the total nnz_offd for the block
   // Then after the kernel, we'll accumulate nnz_offd for each block
   if (lane == 0)
   {
      P_offd_i[row_i] = offd_increment;
   }
}

__global__
void hypreGPUKernel_generate_Pdiag_j_Poffd_j( nalu_hypre_DeviceItem    &item,
                                              NALU_HYPRE_Int      num_points,
                                              NALU_HYPRE_Int      color,
                                              NALU_HYPRE_Int     *pass_order,
                                              NALU_HYPRE_Int     *pass_marker,
                                              NALU_HYPRE_Int     *pass_marker_offd,
                                              NALU_HYPRE_Int     *fine_to_coarse,
                                              NALU_HYPRE_Int     *fine_to_coarse_offd,
                                              NALU_HYPRE_Int     *A_diag_i,
                                              NALU_HYPRE_Int     *A_diag_j,
                                              NALU_HYPRE_Complex *A_diag_data,
                                              NALU_HYPRE_Int     *A_offd_i,
                                              NALU_HYPRE_Int     *A_offd_j,
                                              NALU_HYPRE_Complex *A_offd_data,
                                              NALU_HYPRE_Int     *Soc_diag_j,
                                              NALU_HYPRE_Int     *Soc_offd_j,
                                              NALU_HYPRE_Int     *P_diag_i,
                                              NALU_HYPRE_Int     *P_offd_i,
                                              NALU_HYPRE_Int     *P_diag_j,
                                              NALU_HYPRE_Complex *P_diag_data,
                                              NALU_HYPRE_Int     *P_offd_j,
                                              NALU_HYPRE_Complex *P_offd_data,
                                              NALU_HYPRE_Complex *row_sums )
{
   NALU_HYPRE_Int row_i = nalu_hypre_gpu_get_grid_warp_id<1, 1>(item);

   if (row_i >= num_points)
   {
      return;
   }

   NALU_HYPRE_Int i1 = read_only_load(&pass_order[row_i]);
   NALU_HYPRE_Int lane = nalu_hypre_gpu_get_lane_id<1>(item);
   NALU_HYPRE_Int p_diag_A = 0, q_diag_A, p_diag_P = 0, q_diag_P;
   NALU_HYPRE_Int k;
   NALU_HYPRE_Complex row_sum_C = 0.0, diagonal = 0.0;

   // S_diag
   if (lane < 2)
   {
      p_diag_A = read_only_load(A_diag_i + i1 + lane);
      p_diag_P = read_only_load(P_diag_i + row_i + lane);
   }
   q_diag_A = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, p_diag_A, 1);
   p_diag_A = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, p_diag_A, 0);
   q_diag_P = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, p_diag_P, 1);
   p_diag_P = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, p_diag_P, 0);

   k = p_diag_P;
   for (NALU_HYPRE_Int j = p_diag_A + lane; warp_any_sync(item, NALU_HYPRE_WARP_FULL_MASK, j < q_diag_A);
        j += NALU_HYPRE_WARP_SIZE)
   {
      NALU_HYPRE_Int equal = 0;
      NALU_HYPRE_Int sum = 0;
      NALU_HYPRE_Int j1 = -1;

      if ( j < q_diag_A )
      {
         j1 = read_only_load(&Soc_diag_j[j]);
         equal = j1 > -1 && read_only_load(&pass_marker[j1]) == color;
      }

      NALU_HYPRE_Int pos = warp_prefix_sum(item, lane, equal, sum);

      if (equal)
      {
         P_diag_j[k + pos] = read_only_load(&fine_to_coarse[j1]);
         NALU_HYPRE_Complex val = read_only_load(&A_diag_data[j]);
         P_diag_data[k + pos] = val;
         row_sum_C += val;
      }

      if (j1 == -2)
      {
         diagonal = read_only_load(&A_diag_data[j]);
      }

      k += sum;
   }

   nalu_hypre_device_assert(k == q_diag_P);

   // S_offd
   NALU_HYPRE_Int p_offd_A = 0, q_offd_A, p_offd_P = 0, q_offd_P;

   if (lane < 2)
   {
      p_offd_A = read_only_load(A_offd_i + i1 + lane);
      p_offd_P = read_only_load(P_offd_i + row_i + lane);
   }
   q_offd_A = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, p_offd_A, 1);
   p_offd_A = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, p_offd_A, 0);
   q_offd_P = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, p_offd_P, 1);
   p_offd_P = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, p_offd_P, 0);

   k = p_offd_P;
   for (NALU_HYPRE_Int j = p_offd_A + lane; warp_any_sync(item, NALU_HYPRE_WARP_FULL_MASK, j < q_offd_A);
        j += NALU_HYPRE_WARP_SIZE)
   {
      NALU_HYPRE_Int equal = 0;
      NALU_HYPRE_Int sum = 0;
      NALU_HYPRE_Int j1 = -1;

      if ( j < q_offd_A )
      {
         j1 = read_only_load(&Soc_offd_j[j]);
         equal = j1 > -1 && read_only_load(&pass_marker_offd[j1]) == color;
      }

      NALU_HYPRE_Int pos = warp_prefix_sum(item, lane, equal, sum);

      if (equal)
      {
         P_offd_j[k + pos] = read_only_load(&fine_to_coarse_offd[j1]);
         NALU_HYPRE_Complex val = read_only_load(&A_offd_data[j]);
         P_offd_data[k + pos] = val;
         row_sum_C += val;
      }

      k += sum;
   }

   nalu_hypre_device_assert(k == q_offd_P);

   row_sum_C = warp_reduce_sum(item, row_sum_C);
   diagonal = warp_reduce_sum(item, diagonal);
   NALU_HYPRE_Complex value = row_sum_C * diagonal;
   NALU_HYPRE_Complex row_sum_i = 0.0;

   if (lane == 0)
   {
      row_sum_i = read_only_load(&row_sums[i1]);

      if (value)
      {
         row_sum_i /= value;
         row_sums[i1] = row_sum_i;
      }
   }

   row_sum_i = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, row_sum_i, 0);

   for (NALU_HYPRE_Int j = p_diag_P + lane; j < q_diag_P; j += NALU_HYPRE_WARP_SIZE)
   {
      P_diag_data[j] = -P_diag_data[j] * row_sum_i;
   }

   for (NALU_HYPRE_Int j = p_offd_P + lane; j < q_offd_P; j += NALU_HYPRE_WARP_SIZE)
   {
      P_offd_data[j] = -P_offd_data[j] * row_sum_i;
   }
}

__global__
void hypreGPUKernel_insert_remaining_weights( nalu_hypre_DeviceItem &item,
                                              NALU_HYPRE_Int   start,
                                              NALU_HYPRE_Int   stop,
                                              NALU_HYPRE_Int  *pass_order,
                                              NALU_HYPRE_Int  *Pi_diag_i,
                                              NALU_HYPRE_Int  *Pi_diag_j,
                                              NALU_HYPRE_Real *Pi_diag_data,
                                              NALU_HYPRE_Int  *P_diag_i,
                                              NALU_HYPRE_Int  *P_diag_j,
                                              NALU_HYPRE_Real *P_diag_data,
                                              NALU_HYPRE_Int  *Pi_offd_i,
                                              NALU_HYPRE_Int  *Pi_offd_j,
                                              NALU_HYPRE_Real *Pi_offd_data,
                                              NALU_HYPRE_Int  *P_offd_i,
                                              NALU_HYPRE_Int  *P_offd_j,
                                              NALU_HYPRE_Real *P_offd_data )
{
   NALU_HYPRE_Int row_i = nalu_hypre_gpu_get_grid_warp_id<1, 1>(item);

   if (row_i >= stop - start)
   {
      return;
   }

   NALU_HYPRE_Int i1 = read_only_load(&pass_order[row_i + start]);
   NALU_HYPRE_Int lane = nalu_hypre_gpu_get_lane_id<1>(item);
   NALU_HYPRE_Int p = 0;
   NALU_HYPRE_Int q = 0;
   NALU_HYPRE_Int i2;

   // P_diag
   if (lane < 2)
   {
      p = read_only_load(P_diag_i + i1 + lane);
   }
   q = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, p, 1);
   p = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, p, 0);

   i2 = read_only_load(&Pi_diag_i[row_i]) - p;
   for (NALU_HYPRE_Int j = p + lane; j < q; j += NALU_HYPRE_WARP_SIZE)
   {
      P_diag_j[j] = Pi_diag_j[j + i2];
      P_diag_data[j] = Pi_diag_data[j + i2];
   }

   // P_offd
   if (lane < 2)
   {
      p = read_only_load(P_offd_i + i1 + lane);
   }
   q = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, p, 1);
   p = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, p, 0);

   i2 = read_only_load(&Pi_offd_i[row_i]) - p;
   for (NALU_HYPRE_Int j = p + lane; j < q; j += NALU_HYPRE_WARP_SIZE)
   {
      P_offd_j[j] = Pi_offd_j[j + i2];
      P_offd_data[j] = Pi_offd_data[j + i2];
   }
}


__global__
void hypreGPUKernel_generate_Qdiag_j_Qoffd_j( nalu_hypre_DeviceItem    &item,
                                              NALU_HYPRE_Int      num_points,
                                              NALU_HYPRE_Int      color,
                                              NALU_HYPRE_Int     *pass_order,
                                              NALU_HYPRE_Int     *pass_marker,
                                              NALU_HYPRE_Int     *pass_marker_offd,
                                              NALU_HYPRE_Int     *fine_to_coarse,
                                              NALU_HYPRE_Int     *fine_to_coarse_offd,
                                              NALU_HYPRE_Int     *A_diag_i,
                                              NALU_HYPRE_Int     *A_diag_j,
                                              NALU_HYPRE_Complex *A_diag_data,
                                              NALU_HYPRE_Int     *A_offd_i,
                                              NALU_HYPRE_Int     *A_offd_j,
                                              NALU_HYPRE_Complex *A_offd_data,
                                              NALU_HYPRE_Int     *Soc_diag_j,
                                              NALU_HYPRE_Int     *Soc_offd_j,
                                              NALU_HYPRE_Int     *Q_diag_i,
                                              NALU_HYPRE_Int     *Q_offd_i,
                                              NALU_HYPRE_Int     *Q_diag_j,
                                              NALU_HYPRE_Complex *Q_diag_data,
                                              NALU_HYPRE_Int     *Q_offd_j,
                                              NALU_HYPRE_Complex *Q_offd_data,
                                              NALU_HYPRE_Complex *w_row_sum,
                                              NALU_HYPRE_Int      num_functions,
                                              NALU_HYPRE_Int     *dof_func,
                                              NALU_HYPRE_Int     *dof_func_offd )
{
   NALU_HYPRE_Int row_i = nalu_hypre_gpu_get_grid_warp_id<1, 1>(item);

   if (row_i >= num_points)
   {
      return;
   }

   NALU_HYPRE_Int i1 = read_only_load(&pass_order[row_i]);
   NALU_HYPRE_Int lane = nalu_hypre_gpu_get_lane_id<1>(item);
   NALU_HYPRE_Int p_diag_A = 0, q_diag_A, p_diag_P = 0;
#ifdef NALU_HYPRE_DEBUG
   NALU_HYPRE_Int q_diag_P;
#endif
   NALU_HYPRE_Int k;
   NALU_HYPRE_Complex w_row_sum_i = 0.0;
   NALU_HYPRE_Int dof_func_i1 = -1;

   if (num_functions > 1)
   {
      if (lane == 0)
      {
         dof_func_i1 = read_only_load(&dof_func[i1]);
      }
      dof_func_i1 = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, dof_func_i1, 0);
   }

   // S_diag
#ifdef NALU_HYPRE_DEBUG
   if (lane < 2)
   {
      p_diag_A = read_only_load(A_diag_i + i1 + lane);
      p_diag_P = read_only_load(Q_diag_i + row_i + lane);
   }
   q_diag_A = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, p_diag_A, 1);
   p_diag_A = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, p_diag_A, 0);
   q_diag_P = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, p_diag_P, 1);
   p_diag_P = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, p_diag_P, 0);
#else
   if (lane < 2)
   {
      p_diag_A = read_only_load(A_diag_i + i1 + lane);
   }
   q_diag_A = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, p_diag_A, 1);
   p_diag_A = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, p_diag_A, 0);
   if (lane == 0)
   {
      p_diag_P = read_only_load(Q_diag_i + row_i);
   }
   p_diag_P = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, p_diag_P, 0);
#endif

   k = p_diag_P;
   for (NALU_HYPRE_Int j = p_diag_A + lane; warp_any_sync(item, NALU_HYPRE_WARP_FULL_MASK, j < q_diag_A);
        j += NALU_HYPRE_WARP_SIZE)
   {
      NALU_HYPRE_Int equal = 0;
      NALU_HYPRE_Int sum = 0;
      NALU_HYPRE_Int j1 = -1;

      if ( j < q_diag_A )
      {
         j1 = read_only_load(&Soc_diag_j[j]);
         equal = j1 > -1 && read_only_load(&pass_marker[j1]) == color;
      }

      NALU_HYPRE_Int pos = warp_prefix_sum(item, lane, equal, sum);

      if (equal)
      {
         Q_diag_j[k + pos] = read_only_load(&fine_to_coarse[j1]);
         Q_diag_data[k + pos] = read_only_load(&A_diag_data[j]);
      }
      else if (j < q_diag_A && j1 != -2)
      {
         if (num_functions > 1)
         {
            const NALU_HYPRE_Int col = read_only_load(&A_diag_j[j]);
            if ( dof_func_i1 == read_only_load(&dof_func[col]) )
            {
               w_row_sum_i += read_only_load(&A_diag_data[j]);
            }
         }
         else
         {
            w_row_sum_i += read_only_load(&A_diag_data[j]);
         }
      }

      k += sum;
   }

#ifdef NALU_HYPRE_DEBUG
   nalu_hypre_device_assert(k == q_diag_P);
#endif

   // S_offd
   NALU_HYPRE_Int p_offd_A = 0, q_offd_A, p_offd_P = 0;
#ifdef NALU_HYPRE_DEBUG
   NALU_HYPRE_Int q_offd_P;
#endif

#ifdef NALU_HYPRE_DEBUG
   if (lane < 2)
   {
      p_offd_A = read_only_load(A_offd_i + i1 + lane);
      p_offd_P = read_only_load(Q_offd_i + row_i + lane);
   }
   q_offd_A = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, p_offd_A, 1);
   p_offd_A = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, p_offd_A, 0);
   q_offd_P = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, p_offd_P, 1);
   p_offd_P = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, p_offd_P, 0);
#else
   if (lane < 2)
   {
      p_offd_A = read_only_load(A_offd_i + i1 + lane);
   }
   q_offd_A = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, p_offd_A, 1);
   p_offd_A = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, p_offd_A, 0);
   if (lane == 0)
   {
      p_offd_P = read_only_load(Q_offd_i + row_i);
   }
   p_offd_P = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, p_offd_P, 0);
#endif

   k = p_offd_P;
   for (NALU_HYPRE_Int j = p_offd_A + lane; warp_any_sync(item, NALU_HYPRE_WARP_FULL_MASK, j < q_offd_A);
        j += NALU_HYPRE_WARP_SIZE)
   {
      NALU_HYPRE_Int equal = 0;
      NALU_HYPRE_Int sum = 0;
      NALU_HYPRE_Int j1 = -1;

      if ( j < q_offd_A )
      {
         j1 = read_only_load(&Soc_offd_j[j]);
         equal = j1 > -1 && read_only_load(&pass_marker_offd[j1]) == color;
      }

      NALU_HYPRE_Int pos = warp_prefix_sum(item, lane, equal, sum);

      if (equal)
      {
         Q_offd_j[k + pos] = read_only_load(&fine_to_coarse_offd[j1]);
         Q_offd_data[k + pos] = read_only_load(&A_offd_data[j]);
      }
      else if (j < q_offd_A)
      {
         if (num_functions > 1)
         {
            const NALU_HYPRE_Int col = read_only_load(&A_offd_j[j]);
            if ( dof_func_i1 == read_only_load(&dof_func_offd[col]) )
            {
               w_row_sum_i += read_only_load(&A_offd_data[j]);
            }
         }
         else
         {
            w_row_sum_i += read_only_load(&A_offd_data[j]);
         }
      }

      k += sum;
   }

#ifdef NALU_HYPRE_DEBUG
   nalu_hypre_device_assert(k == q_offd_P);
#endif

   w_row_sum_i = warp_reduce_sum(item, w_row_sum_i);

   if (lane == 0)
   {
      w_row_sum[row_i] = w_row_sum_i;
   }
}

__global__
void hypreGPUKernel_pass_order_count( nalu_hypre_DeviceItem &item,
                                      NALU_HYPRE_Int  num_points,
                                      NALU_HYPRE_Int  color,
                                      NALU_HYPRE_Int *points_left,
                                      NALU_HYPRE_Int *pass_marker,
                                      NALU_HYPRE_Int *pass_marker_offd,
                                      NALU_HYPRE_Int *S_diag_i,
                                      NALU_HYPRE_Int *S_diag_j,
                                      NALU_HYPRE_Int *S_offd_i,
                                      NALU_HYPRE_Int *S_offd_j,
                                      NALU_HYPRE_Int *diag_shifts )
{
   NALU_HYPRE_Int row_i = nalu_hypre_gpu_get_grid_warp_id<1, 1>(item);

   if (row_i >= num_points)
   {
      return;
   }

   NALU_HYPRE_Int i1 = read_only_load(&points_left[row_i]);
   NALU_HYPRE_Int lane = nalu_hypre_gpu_get_lane_id<1>(item);
   NALU_HYPRE_Int p = 0;
   NALU_HYPRE_Int q = 0;
   nalu_hypre_int brk = 0;

   // S_diag
   if (lane < 2)
   {
      p = read_only_load(S_diag_i + i1 + lane);
   }
   q = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, p, 1);
   p = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, p, 0);

   for (NALU_HYPRE_Int j = p + lane; warp_any_sync(item, NALU_HYPRE_WARP_FULL_MASK, j < q); j += NALU_HYPRE_WARP_SIZE)
   {
      if (j < q)
      {
         NALU_HYPRE_Int j1 = read_only_load(&S_diag_j[j]);
         if ( read_only_load(&pass_marker[j1]) == color )
         {
            brk = 1;
         }
      }

      brk = warp_any_sync(item, NALU_HYPRE_WARP_FULL_MASK, brk);

      if (brk)
      {
         break;
      }
   }

   if (brk)
   {
      // Only one thread can increment because of the break
      // so we just need to increment by 1
      if (lane == 0)
      {
         diag_shifts[row_i] = 1;
      }

      return;
   }

   // S_offd
   if (lane < 2)
   {
      p = read_only_load(S_offd_i + i1 + lane);
   }
   q = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, p, 1);
   p = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, p, 0);

   for (NALU_HYPRE_Int j = p + lane; warp_any_sync(item, NALU_HYPRE_WARP_FULL_MASK, j < q); j += NALU_HYPRE_WARP_SIZE)
   {
      if (j < q)
      {
         NALU_HYPRE_Int j1 = read_only_load(&S_offd_j[j]);
         if ( read_only_load(&pass_marker_offd[j1]) == color )
         {
            brk = 1;
         }
      }

      brk = warp_any_sync(item, NALU_HYPRE_WARP_FULL_MASK, brk);

      if (brk)
      {
         break;
      }
   }

   // Only one thread can increment because of the break
   // so we just need to increment by 1
   if (lane == 0)
   {
      diag_shifts[row_i] = (brk != 0);
   }
}

__global__
void hypreGPUKernel_populate_big_P_offd_j( nalu_hypre_DeviceItem   &item,
                                           NALU_HYPRE_Int     start,
                                           NALU_HYPRE_Int     stop,
                                           NALU_HYPRE_Int    *pass_order,
                                           NALU_HYPRE_Int    *P_offd_i,
                                           NALU_HYPRE_Int    *P_offd_j,
                                           NALU_HYPRE_BigInt *col_map_offd_Pi,
                                           NALU_HYPRE_BigInt *big_P_offd_j )
{
   NALU_HYPRE_Int i = nalu_hypre_gpu_get_grid_warp_id<1, 1>(item) + start;

   if (i >= stop)
   {
      return;
   }

   NALU_HYPRE_Int lane = nalu_hypre_gpu_get_lane_id<1>(item);
   NALU_HYPRE_Int i1 = read_only_load(&pass_order[i]);
   NALU_HYPRE_Int p = 0;
   NALU_HYPRE_Int q = 0;

   if (lane < 2)
   {
      p = read_only_load(P_offd_i + i1 + lane);
   }
   q = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, p, 1);
   p = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, p, 0);

   for (NALU_HYPRE_Int j = p + lane; j < q; j += NALU_HYPRE_WARP_SIZE)
   {
      NALU_HYPRE_Int col = read_only_load(&P_offd_j[j]);
      big_P_offd_j[j] = read_only_load(&col_map_offd_Pi[col]);
   }
}

#endif // defined(NALU_HYPRE_USING_GPU)
