/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_nalu_hypre_onedpl.hpp"
#include "_nalu_hypre_utilities.h"
#include "_nalu_hypre_parcsr_mv.h"
#include "_nalu_hypre_utilities.hpp"

#if defined(NALU_HYPRE_USING_GPU)

#if defined(NALU_HYPRE_USING_SYCL)
namespace thrust = std;
#endif

typedef thrust::tuple<NALU_HYPRE_Int, NALU_HYPRE_Int> Tuple;

/* transform from local F/C index to global F/C index,
 * where F index "x" are saved as "-x-1"
 */
#if defined(NALU_HYPRE_USING_SYCL)
struct FFFC_functor
#else
struct FFFC_functor : public thrust::unary_function<Tuple, NALU_HYPRE_BigInt>
#endif
{
   NALU_HYPRE_BigInt CF_first[2];

   FFFC_functor(NALU_HYPRE_BigInt F_first_, NALU_HYPRE_BigInt C_first_)
   {
      CF_first[1] = F_first_;
      CF_first[0] = C_first_;
   }

   __host__ __device__
   NALU_HYPRE_BigInt operator()(const Tuple& t) const
   {
      const NALU_HYPRE_Int local_idx = thrust::get<0>(t);
      const NALU_HYPRE_Int cf_marker = thrust::get<1>(t);
      const NALU_HYPRE_Int s = cf_marker < 0;
      const NALU_HYPRE_Int m = 1 - 2 * s;
      return m * (local_idx + CF_first[s] + s);
   }
};

/* this predicate selects A^s_{FF} */
template<typename T>
#if defined(NALU_HYPRE_USING_SYCL)
struct FF_pred
#else
struct FF_pred : public thrust::unary_function<Tuple, bool>
#endif
{
   NALU_HYPRE_Int  option;
   NALU_HYPRE_Int *row_CF_marker;
   T         *col_CF_marker;

   FF_pred(NALU_HYPRE_Int option_, NALU_HYPRE_Int *row_CF_marker_, T *col_CF_marker_)
   {
      option = option_;
      row_CF_marker = row_CF_marker_;
      col_CF_marker = col_CF_marker_;
   }

   __host__ __device__
   bool operator()(const Tuple& t) const
   {
      const NALU_HYPRE_Int i = thrust::get<0>(t);
      const NALU_HYPRE_Int j = thrust::get<1>(t);

      if (option == 1)
      {
         /* A_{F,F} */
         return row_CF_marker[i] <   0 && (j == -2 || (j >= 0 && col_CF_marker[j] < 0));
      }
      else
      {
         /* A_{F2, F} */
         return row_CF_marker[i] == -2 && (j == -2 || (j >= 0 && col_CF_marker[j] < 0));
      }
   }
};

/* this predicate selects A^s_{FC} */
template<typename T>
#if defined(NALU_HYPRE_USING_SYCL)
struct FC_pred
#else
struct FC_pred : public thrust::unary_function<Tuple, bool>
#endif
{
   NALU_HYPRE_Int *row_CF_marker;
   T         *col_CF_marker;

   FC_pred(NALU_HYPRE_Int *row_CF_marker_, T *col_CF_marker_)
   {
      row_CF_marker = row_CF_marker_;
      col_CF_marker = col_CF_marker_;
   }

   __host__ __device__
   bool operator()(const Tuple& t) const
   {
      const NALU_HYPRE_Int i = thrust::get<0>(t);
      const NALU_HYPRE_Int j = thrust::get<1>(t);

      return row_CF_marker[i] < 0 && (j >= 0 && col_CF_marker[j] >= 0);
   }
};

/* this predicate selects A^s_{CF} */
template<typename T>
#if defined(NALU_HYPRE_USING_SYCL)
struct CF_pred
#else
struct CF_pred : public thrust::unary_function<Tuple, bool>
#endif
{
   NALU_HYPRE_Int *row_CF_marker;
   T         *col_CF_marker;

   CF_pred(NALU_HYPRE_Int *row_CF_marker_, T *col_CF_marker_)
   {
      row_CF_marker = row_CF_marker_;
      col_CF_marker = col_CF_marker_;
   }

   __host__ __device__
   bool operator()(const Tuple& t) const
   {
      const NALU_HYPRE_Int i = thrust::get<0>(t);
      const NALU_HYPRE_Int j = thrust::get<1>(t);

      return row_CF_marker[i] >= 0 && (j >= 0 && col_CF_marker[j] < 0);
   }
};

/* this predicate selects A^s_{CC} */
template<typename T>
#if defined(NALU_HYPRE_USING_SYCL)
struct CC_pred
#else
struct CC_pred : public thrust::unary_function<Tuple, bool>
#endif
{
   NALU_HYPRE_Int *row_CF_marker;
   T         *col_CF_marker;

   CC_pred(NALU_HYPRE_Int *row_CF_marker_, T *col_CF_marker_)
   {
      row_CF_marker = row_CF_marker_;
      col_CF_marker = col_CF_marker_;
   }

   __host__ __device__
   bool operator()(const Tuple& t) const
   {
      const NALU_HYPRE_Int i = thrust::get<0>(t);
      const NALU_HYPRE_Int j = thrust::get<1>(t);

      return row_CF_marker[i] >= 0 && (j == -2 || (j >= 0 && col_CF_marker[j] >= 0));
   }
};

/* this predicate selects A^s_{C,:} */
#if defined(NALU_HYPRE_USING_SYCL)
struct CX_pred
#else
struct CX_pred : public thrust::unary_function<Tuple, bool>
#endif
{
   NALU_HYPRE_Int *row_CF_marker;

   CX_pred(NALU_HYPRE_Int *row_CF_marker_)
   {
      row_CF_marker = row_CF_marker_;
   }

   __host__ __device__
   bool operator()(const Tuple& t) const
   {
      const NALU_HYPRE_Int i = thrust::get<0>(t);
      const NALU_HYPRE_Int j = thrust::get<1>(t);

      return row_CF_marker[i] >= 0 && (j == -2 || j >= 0);
   }
};

/* this predicate selects A^s_{:,C} */
template<typename T>
#if defined(NALU_HYPRE_USING_SYCL)
struct XC_pred
#else
struct XC_pred : public thrust::unary_function<Tuple, bool>
#endif
{
   T         *col_CF_marker;

   XC_pred(T *col_CF_marker_)
   {
      col_CF_marker = col_CF_marker_;
   }

   __host__ __device__
   bool operator()(const Tuple& t) const
   {
      const NALU_HYPRE_Int i = thrust::get<0>(t);
      const NALU_HYPRE_Int j = thrust::get<1>(t);

      return (j == -2 && col_CF_marker[i] >= 0) || (j >= 0 && col_CF_marker[j] >= 0);
   }
};

/* Option = 1:
 *    F is marked as -1, C is +1
 *    | AFF AFC |
 *    | ACF ACC |
 *
 * Option = 2 (for aggressive coarsening):
 *    F_2 is marked as -2 in CF_marker, F_1 as -1, and C_2 as +1
 *    | AF1F1 AF1F2 AF1C2 |
 *    | AF2F1 AF2F2 AF2C2 |
 *    | AC2F1 AC2F2 AC2C2 |
 *    F = F1 + F2
 *    AFC: A_{F, C2}
 *    AFF: A_{F2, F}
 *    ACF: A_{C2, F}
 *    ACC: A_{C2, C2}
 */

NALU_HYPRE_Int
nalu_hypre_ParCSRMatrixGenerateFFFCDevice_core( nalu_hypre_ParCSRMatrix  *A,
                                           NALU_HYPRE_Int           *CF_marker,
                                           NALU_HYPRE_BigInt        *cpts_starts,
                                           nalu_hypre_ParCSRMatrix  *S,
                                           nalu_hypre_ParCSRMatrix **AFC_ptr,
                                           nalu_hypre_ParCSRMatrix **AFF_ptr,
                                           nalu_hypre_ParCSRMatrix **ACF_ptr,
                                           nalu_hypre_ParCSRMatrix **ACC_ptr,
                                           NALU_HYPRE_Int            option )
{
   MPI_Comm                 comm     = nalu_hypre_ParCSRMatrixComm(A);
   if (!nalu_hypre_ParCSRMatrixCommPkg(A))
   {
      nalu_hypre_MatvecCommPkgCreate(A);
   }
   nalu_hypre_ParCSRCommPkg     *comm_pkg = nalu_hypre_ParCSRMatrixCommPkg(A);
   nalu_hypre_ParCSRCommHandle  *comm_handle;
   NALU_HYPRE_Int                num_sends     = nalu_hypre_ParCSRCommPkgNumSends(comm_pkg);
   NALU_HYPRE_Int                num_elem_send = nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends);
   //NALU_HYPRE_MemoryLocation     memory_location = nalu_hypre_ParCSRMatrixMemoryLocation(A);
   /* diag part of A */
   nalu_hypre_CSRMatrix    *A_diag   = nalu_hypre_ParCSRMatrixDiag(A);
   NALU_HYPRE_Complex      *A_diag_a = nalu_hypre_CSRMatrixData(A_diag);
   NALU_HYPRE_Int          *A_diag_i = nalu_hypre_CSRMatrixI(A_diag);
   NALU_HYPRE_Int          *A_diag_j = nalu_hypre_CSRMatrixJ(A_diag);
   NALU_HYPRE_Int           A_diag_nnz = nalu_hypre_CSRMatrixNumNonzeros(A_diag);
   /* offd part of A */
   nalu_hypre_CSRMatrix    *A_offd   = nalu_hypre_ParCSRMatrixOffd(A);
   NALU_HYPRE_Complex      *A_offd_a = nalu_hypre_CSRMatrixData(A_offd);
   NALU_HYPRE_Int          *A_offd_i = nalu_hypre_CSRMatrixI(A_offd);
   NALU_HYPRE_Int          *A_offd_j = nalu_hypre_CSRMatrixJ(A_offd);
   NALU_HYPRE_Int           A_offd_nnz = nalu_hypre_CSRMatrixNumNonzeros(A_offd);
   NALU_HYPRE_Int           num_cols_A_offd = nalu_hypre_CSRMatrixNumCols(A_offd);
   /* SoC */
   NALU_HYPRE_Int          *Soc_diag_j = S ? nalu_hypre_ParCSRMatrixSocDiagJ(S) : A_diag_j;
   NALU_HYPRE_Int          *Soc_offd_j = S ? nalu_hypre_ParCSRMatrixSocOffdJ(S) : A_offd_j;
   /* MPI size and rank */
   NALU_HYPRE_Int           my_id, num_procs;
   /* nF and nC */
   NALU_HYPRE_Int           n_local, nF_local, nC_local, nF2_local = 0;
   NALU_HYPRE_BigInt        fpts_starts[2], *row_starts, f2pts_starts[2];
   NALU_HYPRE_BigInt        nF_global, nC_global, nF2_global = 0;
   NALU_HYPRE_BigInt        F_first, C_first;
   /* work arrays */
   NALU_HYPRE_Int          *map2FC, *map2F2 = NULL, *itmp, *A_diag_ii, *A_offd_ii, *offd_mark;
   NALU_HYPRE_BigInt       *send_buf, *recv_buf;

   nalu_hypre_GpuProfilingPushRange("ParCSRMatrixGenerateFFFC");

   nalu_hypre_MPI_Comm_size(comm, &num_procs);
   nalu_hypre_MPI_Comm_rank(comm, &my_id);

   n_local    = nalu_hypre_ParCSRMatrixNumRows(A);
   row_starts = nalu_hypre_ParCSRMatrixRowStarts(A);

   if (my_id == (num_procs - 1))
   {
      nC_global = cpts_starts[1];
   }
   nalu_hypre_MPI_Bcast(&nC_global, 1, NALU_HYPRE_MPI_BIG_INT, num_procs - 1, comm);
   nC_local = (NALU_HYPRE_Int) (cpts_starts[1] - cpts_starts[0]);
   fpts_starts[0] = row_starts[0] - cpts_starts[0];
   fpts_starts[1] = row_starts[1] - cpts_starts[1];
   F_first = fpts_starts[0];
   C_first = cpts_starts[0];
   nF_local = n_local - nC_local;
   nF_global = nalu_hypre_ParCSRMatrixGlobalNumRows(A) - nC_global;

   map2FC     = nalu_hypre_TAlloc(NALU_HYPRE_Int,    n_local,         NALU_HYPRE_MEMORY_DEVICE);
   itmp       = nalu_hypre_TAlloc(NALU_HYPRE_Int,    n_local,         NALU_HYPRE_MEMORY_DEVICE);
   recv_buf   = nalu_hypre_TAlloc(NALU_HYPRE_BigInt, num_cols_A_offd, NALU_HYPRE_MEMORY_DEVICE);

   if (option == 2)
   {
#if defined(NALU_HYPRE_USING_SYCL)
      nF2_local = NALU_HYPRE_ONEDPL_CALL( std::count,
                                     CF_marker,
                                     CF_marker + n_local,
                                     -2 );
#else
      nF2_local = NALU_HYPRE_THRUST_CALL( count,
                                     CF_marker,
                                     CF_marker + n_local,
                                     -2 );
#endif

      NALU_HYPRE_BigInt nF2_local_big = nF2_local;

      nalu_hypre_MPI_Scan(&nF2_local_big, f2pts_starts + 1, 1, NALU_HYPRE_MPI_BIG_INT, nalu_hypre_MPI_SUM, comm);
      f2pts_starts[0] = f2pts_starts[1] - nF2_local_big;
      if (my_id == (num_procs - 1))
      {
         nF2_global = f2pts_starts[1];
      }
      nalu_hypre_MPI_Bcast(&nF2_global, 1, NALU_HYPRE_MPI_BIG_INT, num_procs - 1, comm);
   }

   /* map from all points (i.e, F+C) to F/C indices */
#if defined(NALU_HYPRE_USING_SYCL)
   NALU_HYPRE_ONEDPL_CALL( std::exclusive_scan,
                      oneapi::dpl::make_transform_iterator(CF_marker,           is_negative<NALU_HYPRE_Int>()),
                      oneapi::dpl::make_transform_iterator(CF_marker + n_local, is_negative<NALU_HYPRE_Int>()),
                      map2FC, /* F */
                      NALU_HYPRE_Int(0) );

   NALU_HYPRE_ONEDPL_CALL( std::exclusive_scan,
                      oneapi::dpl::make_transform_iterator(CF_marker,           is_nonnegative<NALU_HYPRE_Int>()),
                      oneapi::dpl::make_transform_iterator(CF_marker + n_local, is_nonnegative<NALU_HYPRE_Int>()),
                      itmp, /* C */
                      NALU_HYPRE_Int(0) );

   hypreSycl_scatter_if( itmp,
                         itmp + n_local,
                         oneapi::dpl::counting_iterator<NALU_HYPRE_Int>(0),
                         CF_marker,
                         map2FC,
                         is_nonnegative<NALU_HYPRE_Int>() ); /* FC combined */
#else
   NALU_HYPRE_THRUST_CALL( exclusive_scan,
                      thrust::make_transform_iterator(CF_marker,           is_negative<NALU_HYPRE_Int>()),
                      thrust::make_transform_iterator(CF_marker + n_local, is_negative<NALU_HYPRE_Int>()),
                      map2FC, /* F */
                      NALU_HYPRE_Int(0) ); /* *MUST* pass init value since input and output types diff. */

   NALU_HYPRE_THRUST_CALL( exclusive_scan,
                      thrust::make_transform_iterator(CF_marker,           is_nonnegative<NALU_HYPRE_Int>()),
                      thrust::make_transform_iterator(CF_marker + n_local, is_nonnegative<NALU_HYPRE_Int>()),
                      itmp, /* C */
                      NALU_HYPRE_Int(0) ); /* *MUST* pass init value since input and output types diff. */

   NALU_HYPRE_THRUST_CALL( scatter_if,
                      itmp,
                      itmp + n_local,
                      thrust::counting_iterator<NALU_HYPRE_Int>(0),
                      thrust::make_transform_iterator(CF_marker, is_nonnegative<NALU_HYPRE_Int>()),
                      map2FC ); /* FC combined */
#endif

   nalu_hypre_TFree(itmp, NALU_HYPRE_MEMORY_DEVICE);

   if (option == 2)
   {
      map2F2 = nalu_hypre_TAlloc(NALU_HYPRE_Int, n_local, NALU_HYPRE_MEMORY_DEVICE);
#if defined(NALU_HYPRE_USING_SYCL)
      NALU_HYPRE_ONEDPL_CALL( std::exclusive_scan,
                         oneapi::dpl::make_transform_iterator(CF_marker,           equal<NALU_HYPRE_Int>(-2)),
                         oneapi::dpl::make_transform_iterator(CF_marker + n_local, equal<NALU_HYPRE_Int>(-2)),
                         map2F2, /* F2 */
                         NALU_HYPRE_Int(0) ); /* *MUST* pass init value since input and output types diff. */
#else
      NALU_HYPRE_THRUST_CALL( exclusive_scan,
                         thrust::make_transform_iterator(CF_marker,           equal<NALU_HYPRE_Int>(-2)),
                         thrust::make_transform_iterator(CF_marker + n_local, equal<NALU_HYPRE_Int>(-2)),
                         map2F2, /* F2 */
                         NALU_HYPRE_Int(0) ); /* *MUST* pass init value since input and output types diff. */
#endif
   }

   /* send_buf: global F/C indices. Note F-pts "x" are saved as "-x-1" */
   send_buf = nalu_hypre_TAlloc(NALU_HYPRE_BigInt, num_elem_send, NALU_HYPRE_MEMORY_DEVICE);

   nalu_hypre_ParCSRCommPkgCopySendMapElmtsToDevice(comm_pkg);

   FFFC_functor functor(F_first, C_first);
#if defined(NALU_HYPRE_USING_SYCL)
   hypreSycl_gather( nalu_hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg),
                     nalu_hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg) + num_elem_send,
                     oneapi::dpl::make_transform_iterator(
                        oneapi::dpl::make_zip_iterator(map2FC, CF_marker), functor),
                     send_buf );
#else
   NALU_HYPRE_THRUST_CALL( gather,
                      nalu_hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg),
                      nalu_hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg) + num_elem_send,
                      thrust::make_transform_iterator(thrust::make_zip_iterator(thrust::make_tuple(map2FC, CF_marker)),
                                                      functor),
                      send_buf );
#endif

#if defined(NALU_HYPRE_WITH_GPU_AWARE_MPI) && defined(NALU_HYPRE_USING_THRUST_NOSYNC)
   /* RL: make sure send_buf is ready before issuing GPU-GPU MPI */
   nalu_hypre_ForceSyncComputeStream(nalu_hypre_handle());
#endif

   comm_handle = nalu_hypre_ParCSRCommHandleCreate_v2(21, comm_pkg, NALU_HYPRE_MEMORY_DEVICE, send_buf,
                                                 NALU_HYPRE_MEMORY_DEVICE, recv_buf);
   nalu_hypre_ParCSRCommHandleDestroy(comm_handle);

   nalu_hypre_TFree(send_buf, NALU_HYPRE_MEMORY_DEVICE);

   A_diag_ii = nalu_hypre_TAlloc(NALU_HYPRE_Int, A_diag_nnz,      NALU_HYPRE_MEMORY_DEVICE);
   A_offd_ii = nalu_hypre_TAlloc(NALU_HYPRE_Int, A_offd_nnz,      NALU_HYPRE_MEMORY_DEVICE);
   offd_mark = nalu_hypre_TAlloc(NALU_HYPRE_Int, num_cols_A_offd, NALU_HYPRE_MEMORY_DEVICE);

   hypreDevice_CsrRowPtrsToIndices_v2(n_local, A_diag_nnz, A_diag_i, A_diag_ii);
   hypreDevice_CsrRowPtrsToIndices_v2(n_local, A_offd_nnz, A_offd_i, A_offd_ii);

   if (AFF_ptr)
   {
      NALU_HYPRE_Int           AFF_diag_nnz, AFF_offd_nnz;
      NALU_HYPRE_Int          *AFF_diag_ii, *AFF_diag_i, *AFF_diag_j;
      NALU_HYPRE_Complex      *AFF_diag_a;
      NALU_HYPRE_Int          *AFF_offd_ii, *AFF_offd_i, *AFF_offd_j;
      NALU_HYPRE_Complex      *AFF_offd_a;
      nalu_hypre_ParCSRMatrix *AFF;
      nalu_hypre_CSRMatrix    *AFF_diag, *AFF_offd;
      NALU_HYPRE_BigInt       *col_map_offd_AFF;
      NALU_HYPRE_Int           num_cols_AFF_offd;

      /* AFF Diag */
      FF_pred<NALU_HYPRE_Int> AFF_pred_diag(option, CF_marker, CF_marker);
#if defined(NALU_HYPRE_USING_SYCL)
      AFF_diag_nnz = NALU_HYPRE_ONEDPL_CALL( std::count_if,
                                        oneapi::dpl::make_zip_iterator(A_diag_ii, Soc_diag_j),
                                        oneapi::dpl::make_zip_iterator(A_diag_ii, Soc_diag_j) + A_diag_nnz,
                                        AFF_pred_diag );
#else
      AFF_diag_nnz = NALU_HYPRE_THRUST_CALL( count_if,
                                        thrust::make_zip_iterator(thrust::make_tuple(A_diag_ii, Soc_diag_j)),
                                        thrust::make_zip_iterator(thrust::make_tuple(A_diag_ii, Soc_diag_j)) + A_diag_nnz,
                                        AFF_pred_diag );
#endif

      AFF_diag_ii = nalu_hypre_TAlloc(NALU_HYPRE_Int,     AFF_diag_nnz, NALU_HYPRE_MEMORY_DEVICE);
      AFF_diag_j  = nalu_hypre_TAlloc(NALU_HYPRE_Int,     AFF_diag_nnz, NALU_HYPRE_MEMORY_DEVICE);
      AFF_diag_a  = nalu_hypre_TAlloc(NALU_HYPRE_Complex, AFF_diag_nnz, NALU_HYPRE_MEMORY_DEVICE);

#if defined(NALU_HYPRE_USING_SYCL)
      /* Notice that we cannot use Soc_diag_j in the first two arguments since the diagonal is marked as -2 */
      auto new_end = hypreSycl_copy_if( oneapi::dpl::make_zip_iterator(A_diag_ii, A_diag_j, A_diag_a),
                                        oneapi::dpl::make_zip_iterator(A_diag_ii, A_diag_j, A_diag_a) + A_diag_nnz,
                                        oneapi::dpl::make_zip_iterator(A_diag_ii, Soc_diag_j),
                                        oneapi::dpl::make_zip_iterator(AFF_diag_ii, AFF_diag_j, AFF_diag_a),
                                        AFF_pred_diag );

      nalu_hypre_assert( std::get<0>(new_end.base()) == AFF_diag_ii + AFF_diag_nnz );

      hypreSycl_gather( AFF_diag_j,
                        AFF_diag_j + AFF_diag_nnz,
                        map2FC,
                        AFF_diag_j );

      hypreSycl_gather( AFF_diag_ii,
                        AFF_diag_ii + AFF_diag_nnz,
                        option == 1 ? map2FC : map2F2,
                        AFF_diag_ii );

#else
      /* Notice that we cannot use Soc_diag_j in the first two arguments since the diagonal is marked as -2 */
      auto new_end = NALU_HYPRE_THRUST_CALL( copy_if,
                                        thrust::make_zip_iterator(thrust::make_tuple(A_diag_ii, A_diag_j, A_diag_a)),
                                        thrust::make_zip_iterator(thrust::make_tuple(A_diag_ii, A_diag_j, A_diag_a)) + A_diag_nnz,
                                        thrust::make_zip_iterator(thrust::make_tuple(A_diag_ii, Soc_diag_j)),
                                        thrust::make_zip_iterator(thrust::make_tuple(AFF_diag_ii, AFF_diag_j, AFF_diag_a)),
                                        AFF_pred_diag );

      nalu_hypre_assert( thrust::get<0>(new_end.get_iterator_tuple()) == AFF_diag_ii + AFF_diag_nnz );

      NALU_HYPRE_THRUST_CALL ( gather,
                          AFF_diag_j,
                          AFF_diag_j + AFF_diag_nnz,
                          map2FC,
                          AFF_diag_j );

      NALU_HYPRE_THRUST_CALL ( gather,
                          AFF_diag_ii,
                          AFF_diag_ii + AFF_diag_nnz,
                          option == 1 ? map2FC : map2F2,
                          AFF_diag_ii );
#endif

      AFF_diag_i = hypreDevice_CsrRowIndicesToPtrs(option == 1 ? nF_local : nF2_local, AFF_diag_nnz,
                                                   AFF_diag_ii);
      nalu_hypre_TFree(AFF_diag_ii, NALU_HYPRE_MEMORY_DEVICE);

      /* AFF Offd */
      FF_pred<NALU_HYPRE_BigInt> AFF_pred_offd(option, CF_marker, recv_buf);
#if defined(NALU_HYPRE_USING_SYCL)
      AFF_offd_nnz = NALU_HYPRE_ONEDPL_CALL( std::count_if,
                                        oneapi::dpl::make_zip_iterator(A_offd_ii, Soc_offd_j),
                                        oneapi::dpl::make_zip_iterator(A_offd_ii, Soc_offd_j) + A_offd_nnz,
                                        AFF_pred_offd );
#else
      AFF_offd_nnz = NALU_HYPRE_THRUST_CALL( count_if,
                                        thrust::make_zip_iterator(thrust::make_tuple(A_offd_ii, Soc_offd_j)),
                                        thrust::make_zip_iterator(thrust::make_tuple(A_offd_ii, Soc_offd_j)) + A_offd_nnz,
                                        AFF_pred_offd );
#endif

      AFF_offd_ii = nalu_hypre_TAlloc(NALU_HYPRE_Int,     AFF_offd_nnz, NALU_HYPRE_MEMORY_DEVICE);
      AFF_offd_j  = nalu_hypre_TAlloc(NALU_HYPRE_Int,     AFF_offd_nnz, NALU_HYPRE_MEMORY_DEVICE);
      AFF_offd_a  = nalu_hypre_TAlloc(NALU_HYPRE_Complex, AFF_offd_nnz, NALU_HYPRE_MEMORY_DEVICE);

#if defined(NALU_HYPRE_USING_SYCL)
      new_end = hypreSycl_copy_if( oneapi::dpl::make_zip_iterator(A_offd_ii, Soc_offd_j, A_offd_a),
                                   oneapi::dpl::make_zip_iterator(A_offd_ii, Soc_offd_j, A_offd_a) + A_offd_nnz,
                                   oneapi::dpl::make_zip_iterator(A_offd_ii, Soc_offd_j),
                                   oneapi::dpl::make_zip_iterator(AFF_offd_ii, AFF_offd_j, AFF_offd_a),
                                   AFF_pred_offd );

      nalu_hypre_assert( std::get<0>(new_end.base()) == AFF_offd_ii + AFF_offd_nnz );

      hypreSycl_gather( AFF_offd_ii,
                        AFF_offd_ii + AFF_offd_nnz,
                        option == 1 ? map2FC : map2F2,
                        AFF_offd_ii );
#else
      new_end = NALU_HYPRE_THRUST_CALL( copy_if,
                                   thrust::make_zip_iterator(thrust::make_tuple(A_offd_ii, Soc_offd_j, A_offd_a)),
                                   thrust::make_zip_iterator(thrust::make_tuple(A_offd_ii, Soc_offd_j, A_offd_a)) + A_offd_nnz,
                                   thrust::make_zip_iterator(thrust::make_tuple(A_offd_ii, Soc_offd_j)),
                                   thrust::make_zip_iterator(thrust::make_tuple(AFF_offd_ii, AFF_offd_j, AFF_offd_a)),
                                   AFF_pred_offd );

      nalu_hypre_assert( thrust::get<0>(new_end.get_iterator_tuple()) == AFF_offd_ii + AFF_offd_nnz );

      NALU_HYPRE_THRUST_CALL ( gather,
                          AFF_offd_ii,
                          AFF_offd_ii + AFF_offd_nnz,
                          option == 1 ? map2FC : map2F2,
                          AFF_offd_ii );
#endif

      AFF_offd_i = hypreDevice_CsrRowIndicesToPtrs(option == 1 ? nF_local : nF2_local, AFF_offd_nnz,
                                                   AFF_offd_ii);
      nalu_hypre_TFree(AFF_offd_ii, NALU_HYPRE_MEMORY_DEVICE);

      /* col_map_offd_AFF */
      NALU_HYPRE_Int *tmp_j = nalu_hypre_TAlloc(NALU_HYPRE_Int, nalu_hypre_max(AFF_offd_nnz, num_cols_A_offd),
                                      NALU_HYPRE_MEMORY_DEVICE);
      nalu_hypre_TMemcpy(tmp_j, AFF_offd_j, NALU_HYPRE_Int, AFF_offd_nnz, NALU_HYPRE_MEMORY_DEVICE, NALU_HYPRE_MEMORY_DEVICE);
#if defined(NALU_HYPRE_USING_SYCL)
      NALU_HYPRE_ONEDPL_CALL( std::sort,
                         tmp_j,
                         tmp_j + AFF_offd_nnz );
      NALU_HYPRE_Int *tmp_end = NALU_HYPRE_ONEDPL_CALL( std::unique,
                                              tmp_j,
                                              tmp_j + AFF_offd_nnz );
      num_cols_AFF_offd = tmp_end - tmp_j;
      NALU_HYPRE_ONEDPL_CALL( std::fill_n,
                         offd_mark,
                         num_cols_A_offd,
                         0 );
      hypreDevice_ScatterConstant(offd_mark, num_cols_AFF_offd, tmp_j, (NALU_HYPRE_Int) 1);
      NALU_HYPRE_ONEDPL_CALL( std::exclusive_scan,
                         offd_mark,
                         offd_mark + num_cols_A_offd,
                         tmp_j,
                         0 );
      hypreSycl_gather( AFF_offd_j,
                        AFF_offd_j + AFF_offd_nnz,
                        tmp_j,
                        AFF_offd_j );
      col_map_offd_AFF = nalu_hypre_TAlloc(NALU_HYPRE_BigInt, num_cols_AFF_offd, NALU_HYPRE_MEMORY_DEVICE);
      NALU_HYPRE_BigInt *tmp_end_big = hypreSycl_copy_if( recv_buf,
                                                     recv_buf + num_cols_A_offd,
                                                     offd_mark,
                                                     col_map_offd_AFF,
      [] (const auto & x) {return x;} );
      NALU_HYPRE_ONEDPL_CALL( std::transform,
                         col_map_offd_AFF,
                         col_map_offd_AFF + num_cols_AFF_offd,
                         col_map_offd_AFF,
      [] (auto const & x) { return -x - 1; } );
      nalu_hypre_assert(tmp_end_big - col_map_offd_AFF == num_cols_AFF_offd);
#else
      NALU_HYPRE_THRUST_CALL( sort,
                         tmp_j,
                         tmp_j + AFF_offd_nnz );
      NALU_HYPRE_Int *tmp_end = NALU_HYPRE_THRUST_CALL( unique,
                                              tmp_j,
                                              tmp_j + AFF_offd_nnz );
      num_cols_AFF_offd = tmp_end - tmp_j;
      hypreDevice_IntFilln( offd_mark, num_cols_A_offd, 0 );
      hypreDevice_ScatterConstant(offd_mark, num_cols_AFF_offd, tmp_j, (NALU_HYPRE_Int) 1);
      NALU_HYPRE_THRUST_CALL( exclusive_scan,
                         offd_mark,
                         offd_mark + num_cols_A_offd,
                         tmp_j );
      NALU_HYPRE_THRUST_CALL( gather,
                         AFF_offd_j,
                         AFF_offd_j + AFF_offd_nnz,
                         tmp_j,
                         AFF_offd_j );
      col_map_offd_AFF = nalu_hypre_TAlloc(NALU_HYPRE_BigInt, num_cols_AFF_offd, NALU_HYPRE_MEMORY_DEVICE);
      NALU_HYPRE_BigInt *tmp_end_big = NALU_HYPRE_THRUST_CALL( copy_if,
                                                     thrust::make_transform_iterator(recv_buf, -_1 - 1),
                                                     thrust::make_transform_iterator(recv_buf, -_1 - 1) + num_cols_A_offd,
                                                     offd_mark,
                                                     col_map_offd_AFF,
                                                     thrust::identity<NALU_HYPRE_Int>() );
      nalu_hypre_assert(tmp_end_big - col_map_offd_AFF == num_cols_AFF_offd);
#endif
      nalu_hypre_TFree(tmp_j, NALU_HYPRE_MEMORY_DEVICE);

      AFF = nalu_hypre_ParCSRMatrixCreate(comm,
                                     option == 1 ? nF_global : nF2_global,
                                     nF_global,
                                     option == 1 ? fpts_starts : f2pts_starts,
                                     fpts_starts,
                                     num_cols_AFF_offd,
                                     AFF_diag_nnz,
                                     AFF_offd_nnz);

      AFF_diag = nalu_hypre_ParCSRMatrixDiag(AFF);
      nalu_hypre_CSRMatrixData(AFF_diag) = AFF_diag_a;
      nalu_hypre_CSRMatrixI(AFF_diag)    = AFF_diag_i;
      nalu_hypre_CSRMatrixJ(AFF_diag)    = AFF_diag_j;

      AFF_offd = nalu_hypre_ParCSRMatrixOffd(AFF);
      nalu_hypre_CSRMatrixData(AFF_offd) = AFF_offd_a;
      nalu_hypre_CSRMatrixI(AFF_offd)    = AFF_offd_i;
      nalu_hypre_CSRMatrixJ(AFF_offd)    = AFF_offd_j;

      nalu_hypre_CSRMatrixMemoryLocation(AFF_diag) = NALU_HYPRE_MEMORY_DEVICE;
      nalu_hypre_CSRMatrixMemoryLocation(AFF_offd) = NALU_HYPRE_MEMORY_DEVICE;

      nalu_hypre_ParCSRMatrixDeviceColMapOffd(AFF) = col_map_offd_AFF;
      nalu_hypre_ParCSRMatrixColMapOffd(AFF) = nalu_hypre_TAlloc(NALU_HYPRE_BigInt, num_cols_AFF_offd,
                                                       NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TMemcpy(nalu_hypre_ParCSRMatrixColMapOffd(AFF), col_map_offd_AFF, NALU_HYPRE_BigInt, num_cols_AFF_offd,
                    NALU_HYPRE_MEMORY_HOST, NALU_HYPRE_MEMORY_DEVICE);

      nalu_hypre_ParCSRMatrixSetNumNonzeros(AFF);
      nalu_hypre_ParCSRMatrixDNumNonzeros(AFF) = (NALU_HYPRE_Real) nalu_hypre_ParCSRMatrixNumNonzeros(AFF);
      nalu_hypre_MatvecCommPkgCreate(AFF);

      *AFF_ptr = AFF;
   }

   if (AFC_ptr)
   {
      NALU_HYPRE_Int           AFC_diag_nnz, AFC_offd_nnz;
      NALU_HYPRE_Int          *AFC_diag_ii, *AFC_diag_i, *AFC_diag_j;
      NALU_HYPRE_Complex      *AFC_diag_a;
      NALU_HYPRE_Int          *AFC_offd_ii, *AFC_offd_i, *AFC_offd_j;
      NALU_HYPRE_Complex      *AFC_offd_a;
      nalu_hypre_ParCSRMatrix *AFC;
      nalu_hypre_CSRMatrix    *AFC_diag, *AFC_offd;
      NALU_HYPRE_BigInt       *col_map_offd_AFC;
      NALU_HYPRE_Int           num_cols_AFC_offd;

      /* AFC Diag */
      FC_pred<NALU_HYPRE_Int> AFC_pred_diag(CF_marker, CF_marker);
#if defined(NALU_HYPRE_USING_SYCL)
      AFC_diag_nnz = NALU_HYPRE_ONEDPL_CALL( std::count_if,
                                        oneapi::dpl::make_zip_iterator(A_diag_ii, Soc_diag_j),
                                        oneapi::dpl::make_zip_iterator(A_diag_ii, Soc_diag_j) + A_diag_nnz,
                                        AFC_pred_diag );
#else
      AFC_diag_nnz = NALU_HYPRE_THRUST_CALL( count_if,
                                        thrust::make_zip_iterator(thrust::make_tuple(A_diag_ii, Soc_diag_j)),
                                        thrust::make_zip_iterator(thrust::make_tuple(A_diag_ii, Soc_diag_j)) + A_diag_nnz,
                                        AFC_pred_diag );
#endif

      AFC_diag_ii = nalu_hypre_TAlloc(NALU_HYPRE_Int,     AFC_diag_nnz, NALU_HYPRE_MEMORY_DEVICE);
      AFC_diag_j  = nalu_hypre_TAlloc(NALU_HYPRE_Int,     AFC_diag_nnz, NALU_HYPRE_MEMORY_DEVICE);
      AFC_diag_a  = nalu_hypre_TAlloc(NALU_HYPRE_Complex, AFC_diag_nnz, NALU_HYPRE_MEMORY_DEVICE);

#if defined(NALU_HYPRE_USING_SYCL)
      auto new_end = hypreSycl_copy_if( oneapi::dpl::make_zip_iterator(A_diag_ii, Soc_diag_j, A_diag_a),
                                        oneapi::dpl::make_zip_iterator(A_diag_ii, Soc_diag_j, A_diag_a) + A_diag_nnz,
                                        oneapi::dpl::make_zip_iterator(A_diag_ii, Soc_diag_j),
                                        oneapi::dpl::make_zip_iterator(AFC_diag_ii, AFC_diag_j, AFC_diag_a),
                                        AFC_pred_diag );

      nalu_hypre_assert( std::get<0>(new_end.base()) == AFC_diag_ii + AFC_diag_nnz );

      hypreSycl_gather( AFC_diag_j,
                        AFC_diag_j + AFC_diag_nnz,
                        map2FC,
                        AFC_diag_j );

      hypreSycl_gather( AFC_diag_ii,
                        AFC_diag_ii + AFC_diag_nnz,
                        map2FC,
                        AFC_diag_ii );
#else
      auto new_end = NALU_HYPRE_THRUST_CALL( copy_if,
                                        thrust::make_zip_iterator(thrust::make_tuple(A_diag_ii, Soc_diag_j, A_diag_a)),
                                        thrust::make_zip_iterator(thrust::make_tuple(A_diag_ii, Soc_diag_j, A_diag_a)) + A_diag_nnz,
                                        thrust::make_zip_iterator(thrust::make_tuple(A_diag_ii, Soc_diag_j)),
                                        thrust::make_zip_iterator(thrust::make_tuple(AFC_diag_ii, AFC_diag_j, AFC_diag_a)),
                                        AFC_pred_diag );

      nalu_hypre_assert( thrust::get<0>(new_end.get_iterator_tuple()) == AFC_diag_ii + AFC_diag_nnz );

      NALU_HYPRE_THRUST_CALL ( gather,
                          AFC_diag_j,
                          AFC_diag_j + AFC_diag_nnz,
                          map2FC,
                          AFC_diag_j );

      NALU_HYPRE_THRUST_CALL ( gather,
                          AFC_diag_ii,
                          AFC_diag_ii + AFC_diag_nnz,
                          map2FC,
                          AFC_diag_ii );
#endif

      AFC_diag_i = hypreDevice_CsrRowIndicesToPtrs(nF_local, AFC_diag_nnz, AFC_diag_ii);
      nalu_hypre_TFree(AFC_diag_ii, NALU_HYPRE_MEMORY_DEVICE);

      /* AFC Offd */
      FC_pred<NALU_HYPRE_BigInt> AFC_pred_offd(CF_marker, recv_buf);
#if defined(NALU_HYPRE_USING_SYCL)
      AFC_offd_nnz = NALU_HYPRE_ONEDPL_CALL( std::count_if,
                                        oneapi::dpl::make_zip_iterator(A_offd_ii, Soc_offd_j),
                                        oneapi::dpl::make_zip_iterator(A_offd_ii, Soc_offd_j) + A_offd_nnz,
                                        AFC_pred_offd );
#else
      AFC_offd_nnz = NALU_HYPRE_THRUST_CALL( count_if,
                                        thrust::make_zip_iterator(thrust::make_tuple(A_offd_ii, Soc_offd_j)),
                                        thrust::make_zip_iterator(thrust::make_tuple(A_offd_ii, Soc_offd_j)) + A_offd_nnz,
                                        AFC_pred_offd );
#endif

      AFC_offd_ii = nalu_hypre_TAlloc(NALU_HYPRE_Int,     AFC_offd_nnz, NALU_HYPRE_MEMORY_DEVICE);
      AFC_offd_j  = nalu_hypre_TAlloc(NALU_HYPRE_Int,     AFC_offd_nnz, NALU_HYPRE_MEMORY_DEVICE);
      AFC_offd_a  = nalu_hypre_TAlloc(NALU_HYPRE_Complex, AFC_offd_nnz, NALU_HYPRE_MEMORY_DEVICE);

#if defined(NALU_HYPRE_USING_SYCL)
      new_end = hypreSycl_copy_if( oneapi::dpl::make_zip_iterator(A_offd_ii, Soc_offd_j, A_offd_a),
                                   oneapi::dpl::make_zip_iterator(A_offd_ii, Soc_offd_j, A_offd_a) + A_offd_nnz,
                                   oneapi::dpl::make_zip_iterator(A_offd_ii, Soc_offd_j),
                                   oneapi::dpl::make_zip_iterator(AFC_offd_ii, AFC_offd_j, AFC_offd_a),
                                   AFC_pred_offd );

      nalu_hypre_assert( std::get<0>(new_end.base()) == AFC_offd_ii + AFC_offd_nnz );

      hypreSycl_gather( AFC_offd_ii,
                        AFC_offd_ii + AFC_offd_nnz,
                        map2FC,
                        AFC_offd_ii );
#else
      new_end = NALU_HYPRE_THRUST_CALL( copy_if,
                                   thrust::make_zip_iterator(thrust::make_tuple(A_offd_ii, Soc_offd_j, A_offd_a)),
                                   thrust::make_zip_iterator(thrust::make_tuple(A_offd_ii, Soc_offd_j, A_offd_a)) + A_offd_nnz,
                                   thrust::make_zip_iterator(thrust::make_tuple(A_offd_ii, Soc_offd_j)),
                                   thrust::make_zip_iterator(thrust::make_tuple(AFC_offd_ii, AFC_offd_j, AFC_offd_a)),
                                   AFC_pred_offd );

      nalu_hypre_assert( thrust::get<0>(new_end.get_iterator_tuple()) == AFC_offd_ii + AFC_offd_nnz );

      NALU_HYPRE_THRUST_CALL ( gather,
                          AFC_offd_ii,
                          AFC_offd_ii + AFC_offd_nnz,
                          map2FC,
                          AFC_offd_ii );
#endif

      AFC_offd_i = hypreDevice_CsrRowIndicesToPtrs(nF_local, AFC_offd_nnz, AFC_offd_ii);
      nalu_hypre_TFree(AFC_offd_ii, NALU_HYPRE_MEMORY_DEVICE);

      /* col_map_offd_AFC */
      NALU_HYPRE_Int *tmp_j = nalu_hypre_TAlloc(NALU_HYPRE_Int, nalu_hypre_max(AFC_offd_nnz, num_cols_A_offd),
                                      NALU_HYPRE_MEMORY_DEVICE);
      nalu_hypre_TMemcpy(tmp_j, AFC_offd_j, NALU_HYPRE_Int, AFC_offd_nnz, NALU_HYPRE_MEMORY_DEVICE, NALU_HYPRE_MEMORY_DEVICE);
#if defined(NALU_HYPRE_USING_SYCL)
      NALU_HYPRE_ONEDPL_CALL( std::sort,
                         tmp_j,
                         tmp_j + AFC_offd_nnz );
      NALU_HYPRE_Int *tmp_end = NALU_HYPRE_ONEDPL_CALL( std::unique,
                                              tmp_j,
                                              tmp_j + AFC_offd_nnz );
      num_cols_AFC_offd = tmp_end - tmp_j;
      NALU_HYPRE_ONEDPL_CALL( std::fill_n,
                         offd_mark,
                         num_cols_A_offd,
                         0 );
      hypreDevice_ScatterConstant(offd_mark, num_cols_AFC_offd, tmp_j, (NALU_HYPRE_Int) 1);
      NALU_HYPRE_ONEDPL_CALL( std::exclusive_scan,
                         offd_mark,
                         offd_mark + num_cols_A_offd,
                         tmp_j,
                         0);
      hypreSycl_gather( AFC_offd_j,
                        AFC_offd_j + AFC_offd_nnz,
                        tmp_j,
                        AFC_offd_j );
      col_map_offd_AFC = nalu_hypre_TAlloc(NALU_HYPRE_BigInt, num_cols_AFC_offd, NALU_HYPRE_MEMORY_DEVICE);
      NALU_HYPRE_BigInt *tmp_end_big = hypreSycl_copy_if( recv_buf,
                                                     recv_buf + num_cols_A_offd,
                                                     offd_mark,
                                                     col_map_offd_AFC,
      [] (const auto & x) {return x;});
#else
      NALU_HYPRE_THRUST_CALL( sort,
                         tmp_j,
                         tmp_j + AFC_offd_nnz );
      NALU_HYPRE_Int *tmp_end = NALU_HYPRE_THRUST_CALL( unique,
                                              tmp_j,
                                              tmp_j + AFC_offd_nnz );
      num_cols_AFC_offd = tmp_end - tmp_j;
      hypreDevice_IntFilln( offd_mark, num_cols_A_offd, 0 );
      hypreDevice_ScatterConstant(offd_mark, num_cols_AFC_offd, tmp_j, (NALU_HYPRE_Int) 1);
      NALU_HYPRE_THRUST_CALL( exclusive_scan,
                         offd_mark,
                         offd_mark + num_cols_A_offd,
                         tmp_j);
      NALU_HYPRE_THRUST_CALL( gather,
                         AFC_offd_j,
                         AFC_offd_j + AFC_offd_nnz,
                         tmp_j,
                         AFC_offd_j );
      col_map_offd_AFC = nalu_hypre_TAlloc(NALU_HYPRE_BigInt, num_cols_AFC_offd, NALU_HYPRE_MEMORY_DEVICE);
      NALU_HYPRE_BigInt *tmp_end_big = NALU_HYPRE_THRUST_CALL( copy_if,
                                                     recv_buf,
                                                     recv_buf + num_cols_A_offd,
                                                     offd_mark,
                                                     col_map_offd_AFC,
                                                     thrust::identity<NALU_HYPRE_Int>());
#endif
      nalu_hypre_assert(tmp_end_big - col_map_offd_AFC == num_cols_AFC_offd);
      nalu_hypre_TFree(tmp_j, NALU_HYPRE_MEMORY_DEVICE);

      /* AFC */
      AFC = nalu_hypre_ParCSRMatrixCreate(comm,
                                     nF_global,
                                     nC_global,
                                     fpts_starts,
                                     cpts_starts,
                                     num_cols_AFC_offd,
                                     AFC_diag_nnz,
                                     AFC_offd_nnz);

      AFC_diag = nalu_hypre_ParCSRMatrixDiag(AFC);
      nalu_hypre_CSRMatrixData(AFC_diag) = AFC_diag_a;
      nalu_hypre_CSRMatrixI(AFC_diag)    = AFC_diag_i;
      nalu_hypre_CSRMatrixJ(AFC_diag)    = AFC_diag_j;

      AFC_offd = nalu_hypre_ParCSRMatrixOffd(AFC);
      nalu_hypre_CSRMatrixData(AFC_offd) = AFC_offd_a;
      nalu_hypre_CSRMatrixI(AFC_offd)    = AFC_offd_i;
      nalu_hypre_CSRMatrixJ(AFC_offd)    = AFC_offd_j;

      nalu_hypre_CSRMatrixMemoryLocation(AFC_diag) = NALU_HYPRE_MEMORY_DEVICE;
      nalu_hypre_CSRMatrixMemoryLocation(AFC_offd) = NALU_HYPRE_MEMORY_DEVICE;

      nalu_hypre_ParCSRMatrixDeviceColMapOffd(AFC) = col_map_offd_AFC;
      nalu_hypre_ParCSRMatrixColMapOffd(AFC) = nalu_hypre_TAlloc(NALU_HYPRE_BigInt, num_cols_AFC_offd,
                                                       NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TMemcpy(nalu_hypre_ParCSRMatrixColMapOffd(AFC), col_map_offd_AFC, NALU_HYPRE_BigInt, num_cols_AFC_offd,
                    NALU_HYPRE_MEMORY_HOST, NALU_HYPRE_MEMORY_DEVICE);

      nalu_hypre_ParCSRMatrixSetNumNonzeros(AFC);
      nalu_hypre_ParCSRMatrixDNumNonzeros(AFC) = (NALU_HYPRE_Real) nalu_hypre_ParCSRMatrixNumNonzeros(AFC);
      nalu_hypre_MatvecCommPkgCreate(AFC);

      *AFC_ptr = AFC;
   }

   if (ACF_ptr)
   {
      NALU_HYPRE_Int           ACF_diag_nnz, ACF_offd_nnz;
      NALU_HYPRE_Int          *ACF_diag_ii, *ACF_diag_i, *ACF_diag_j;
      NALU_HYPRE_Complex      *ACF_diag_a;
      NALU_HYPRE_Int          *ACF_offd_ii, *ACF_offd_i, *ACF_offd_j;
      NALU_HYPRE_Complex      *ACF_offd_a;
      nalu_hypre_ParCSRMatrix *ACF;
      nalu_hypre_CSRMatrix    *ACF_diag, *ACF_offd;
      NALU_HYPRE_BigInt       *col_map_offd_ACF;
      NALU_HYPRE_Int           num_cols_ACF_offd;

      /* ACF Diag */
      CF_pred<NALU_HYPRE_Int> ACF_pred_diag(CF_marker, CF_marker);
#if defined(NALU_HYPRE_USING_SYCL)
      ACF_diag_nnz = NALU_HYPRE_ONEDPL_CALL( std::count_if,
                                        oneapi::dpl::make_zip_iterator(A_diag_ii, Soc_diag_j),
                                        oneapi::dpl::make_zip_iterator(A_diag_ii, Soc_diag_j) + A_diag_nnz,
                                        ACF_pred_diag );
#else
      ACF_diag_nnz = NALU_HYPRE_THRUST_CALL( count_if,
                                        thrust::make_zip_iterator(thrust::make_tuple(A_diag_ii, Soc_diag_j)),
                                        thrust::make_zip_iterator(thrust::make_tuple(A_diag_ii, Soc_diag_j)) + A_diag_nnz,
                                        ACF_pred_diag );
#endif

      ACF_diag_ii = nalu_hypre_TAlloc(NALU_HYPRE_Int,     ACF_diag_nnz, NALU_HYPRE_MEMORY_DEVICE);
      ACF_diag_j  = nalu_hypre_TAlloc(NALU_HYPRE_Int,     ACF_diag_nnz, NALU_HYPRE_MEMORY_DEVICE);
      ACF_diag_a  = nalu_hypre_TAlloc(NALU_HYPRE_Complex, ACF_diag_nnz, NALU_HYPRE_MEMORY_DEVICE);

#if defined(NALU_HYPRE_USING_SYCL)
      auto new_end = hypreSycl_copy_if( oneapi::dpl::make_zip_iterator(A_diag_ii, Soc_diag_j, A_diag_a),
                                        oneapi::dpl::make_zip_iterator(A_diag_ii, Soc_diag_j, A_diag_a) + A_diag_nnz,
                                        oneapi::dpl::make_zip_iterator(A_diag_ii, Soc_diag_j),
                                        oneapi::dpl::make_zip_iterator(ACF_diag_ii, ACF_diag_j, ACF_diag_a),
                                        ACF_pred_diag );

      nalu_hypre_assert( std::get<0>(new_end.base()) == ACF_diag_ii + ACF_diag_nnz );

      hypreSycl_gather( ACF_diag_j,
                        ACF_diag_j + ACF_diag_nnz,
                        map2FC,
                        ACF_diag_j );

      hypreSycl_gather( ACF_diag_ii,
                        ACF_diag_ii + ACF_diag_nnz,
                        map2FC,
                        ACF_diag_ii );
#else
      auto new_end = NALU_HYPRE_THRUST_CALL( copy_if,
                                        thrust::make_zip_iterator(thrust::make_tuple(A_diag_ii, Soc_diag_j, A_diag_a)),
                                        thrust::make_zip_iterator(thrust::make_tuple(A_diag_ii, Soc_diag_j, A_diag_a)) + A_diag_nnz,
                                        thrust::make_zip_iterator(thrust::make_tuple(A_diag_ii, Soc_diag_j)),
                                        thrust::make_zip_iterator(thrust::make_tuple(ACF_diag_ii, ACF_diag_j, ACF_diag_a)),
                                        ACF_pred_diag );

      nalu_hypre_assert( thrust::get<0>(new_end.get_iterator_tuple()) == ACF_diag_ii + ACF_diag_nnz );

      NALU_HYPRE_THRUST_CALL ( gather,
                          ACF_diag_j,
                          ACF_diag_j + ACF_diag_nnz,
                          map2FC,
                          ACF_diag_j );

      NALU_HYPRE_THRUST_CALL ( gather,
                          ACF_diag_ii,
                          ACF_diag_ii + ACF_diag_nnz,
                          map2FC,
                          ACF_diag_ii );
#endif

      ACF_diag_i = hypreDevice_CsrRowIndicesToPtrs(nC_local, ACF_diag_nnz, ACF_diag_ii);
      nalu_hypre_TFree(ACF_diag_ii, NALU_HYPRE_MEMORY_DEVICE);

      /* ACF Offd */
      CF_pred<NALU_HYPRE_BigInt> ACF_pred_offd(CF_marker, recv_buf);
#if defined(NALU_HYPRE_USING_SYCL)
      ACF_offd_nnz = NALU_HYPRE_ONEDPL_CALL( std::count_if,
                                        oneapi::dpl::make_zip_iterator(A_offd_ii, Soc_offd_j),
                                        oneapi::dpl::make_zip_iterator(A_offd_ii, Soc_offd_j) + A_offd_nnz,
                                        ACF_pred_offd );
#else
      ACF_offd_nnz = NALU_HYPRE_THRUST_CALL( count_if,
                                        thrust::make_zip_iterator(thrust::make_tuple(A_offd_ii, Soc_offd_j)),
                                        thrust::make_zip_iterator(thrust::make_tuple(A_offd_ii, Soc_offd_j)) + A_offd_nnz,
                                        ACF_pred_offd );
#endif

      ACF_offd_ii = nalu_hypre_TAlloc(NALU_HYPRE_Int,     ACF_offd_nnz, NALU_HYPRE_MEMORY_DEVICE);
      ACF_offd_j  = nalu_hypre_TAlloc(NALU_HYPRE_Int,     ACF_offd_nnz, NALU_HYPRE_MEMORY_DEVICE);
      ACF_offd_a  = nalu_hypre_TAlloc(NALU_HYPRE_Complex, ACF_offd_nnz, NALU_HYPRE_MEMORY_DEVICE);

#if defined(NALU_HYPRE_USING_SYCL)
      new_end = hypreSycl_copy_if( oneapi::dpl::make_zip_iterator(A_offd_ii, Soc_offd_j, A_offd_a),
                                   oneapi::dpl::make_zip_iterator(A_offd_ii, Soc_offd_j, A_offd_a) + A_offd_nnz,
                                   oneapi::dpl::make_zip_iterator(A_offd_ii, Soc_offd_j),
                                   oneapi::dpl::make_zip_iterator(ACF_offd_ii, ACF_offd_j, ACF_offd_a),
                                   ACF_pred_offd );

      nalu_hypre_assert( std::get<0>(new_end.base()) == ACF_offd_ii + ACF_offd_nnz );

      hypreSycl_gather( ACF_offd_ii,
                        ACF_offd_ii + ACF_offd_nnz,
                        map2FC,
                        ACF_offd_ii );
#else
      new_end = NALU_HYPRE_THRUST_CALL( copy_if,
                                   thrust::make_zip_iterator(thrust::make_tuple(A_offd_ii, Soc_offd_j, A_offd_a)),
                                   thrust::make_zip_iterator(thrust::make_tuple(A_offd_ii, Soc_offd_j, A_offd_a)) + A_offd_nnz,
                                   thrust::make_zip_iterator(thrust::make_tuple(A_offd_ii, Soc_offd_j)),
                                   thrust::make_zip_iterator(thrust::make_tuple(ACF_offd_ii, ACF_offd_j, ACF_offd_a)),
                                   ACF_pred_offd );

      nalu_hypre_assert( thrust::get<0>(new_end.get_iterator_tuple()) == ACF_offd_ii + ACF_offd_nnz );

      NALU_HYPRE_THRUST_CALL ( gather,
                          ACF_offd_ii,
                          ACF_offd_ii + ACF_offd_nnz,
                          map2FC,
                          ACF_offd_ii );
#endif

      ACF_offd_i = hypreDevice_CsrRowIndicesToPtrs(nC_local, ACF_offd_nnz, ACF_offd_ii);
      nalu_hypre_TFree(ACF_offd_ii, NALU_HYPRE_MEMORY_DEVICE);

      /* col_map_offd_ACF */
      NALU_HYPRE_Int *tmp_j = nalu_hypre_TAlloc(NALU_HYPRE_Int, nalu_hypre_max(ACF_offd_nnz, num_cols_A_offd),
                                      NALU_HYPRE_MEMORY_DEVICE);
      nalu_hypre_TMemcpy(tmp_j, ACF_offd_j, NALU_HYPRE_Int, ACF_offd_nnz, NALU_HYPRE_MEMORY_DEVICE, NALU_HYPRE_MEMORY_DEVICE);
#if defined(NALU_HYPRE_USING_SYCL)
      NALU_HYPRE_ONEDPL_CALL( std::sort,
                         tmp_j,
                         tmp_j + ACF_offd_nnz );
      NALU_HYPRE_Int *tmp_end = NALU_HYPRE_ONEDPL_CALL( std::unique,
                                              tmp_j,
                                              tmp_j + ACF_offd_nnz );
      num_cols_ACF_offd = tmp_end - tmp_j;
      NALU_HYPRE_ONEDPL_CALL( std::fill_n,
                         offd_mark,
                         num_cols_A_offd,
                         0 );
      hypreDevice_ScatterConstant(offd_mark, num_cols_ACF_offd, tmp_j, (NALU_HYPRE_Int) 1);
      NALU_HYPRE_ONEDPL_CALL( std::exclusive_scan,
                         offd_mark,
                         offd_mark + num_cols_A_offd,
                         tmp_j,
                         0);
      hypreSycl_gather( ACF_offd_j,
                        ACF_offd_j + ACF_offd_nnz,
                        tmp_j,
                        ACF_offd_j );
      col_map_offd_ACF = nalu_hypre_TAlloc(NALU_HYPRE_BigInt, num_cols_ACF_offd, NALU_HYPRE_MEMORY_DEVICE);
      NALU_HYPRE_BigInt *tmp_end_big = hypreSycl_copy_if( recv_buf,
                                                     recv_buf + num_cols_A_offd,
                                                     offd_mark,
                                                     col_map_offd_ACF,
      [] (const auto & x) {return x;} );
      NALU_HYPRE_ONEDPL_CALL( std::transform,
                         col_map_offd_ACF,
                         col_map_offd_ACF + num_cols_ACF_offd,
                         col_map_offd_ACF,
      [] (const auto & x) {return -x - 1;} );
#else
      NALU_HYPRE_THRUST_CALL( sort,
                         tmp_j,
                         tmp_j + ACF_offd_nnz );
      NALU_HYPRE_Int *tmp_end = NALU_HYPRE_THRUST_CALL( unique,
                                              tmp_j,
                                              tmp_j + ACF_offd_nnz );
      num_cols_ACF_offd = tmp_end - tmp_j;
      hypreDevice_IntFilln( offd_mark, num_cols_A_offd, 0 );
      hypreDevice_ScatterConstant(offd_mark, num_cols_ACF_offd, tmp_j, (NALU_HYPRE_Int) 1);
      NALU_HYPRE_THRUST_CALL( exclusive_scan,
                         offd_mark,
                         offd_mark + num_cols_A_offd,
                         tmp_j);
      NALU_HYPRE_THRUST_CALL( gather,
                         ACF_offd_j,
                         ACF_offd_j + ACF_offd_nnz,
                         tmp_j,
                         ACF_offd_j );
      col_map_offd_ACF = nalu_hypre_TAlloc(NALU_HYPRE_BigInt, num_cols_ACF_offd, NALU_HYPRE_MEMORY_DEVICE);
      NALU_HYPRE_BigInt *tmp_end_big = NALU_HYPRE_THRUST_CALL( copy_if,
                                                     thrust::make_transform_iterator(recv_buf, -_1 - 1),
                                                     thrust::make_transform_iterator(recv_buf, -_1 - 1) + num_cols_A_offd,
                                                     offd_mark,
                                                     col_map_offd_ACF,
                                                     thrust::identity<NALU_HYPRE_Int>());
#endif
      nalu_hypre_assert(tmp_end_big - col_map_offd_ACF == num_cols_ACF_offd);
      nalu_hypre_TFree(tmp_j, NALU_HYPRE_MEMORY_DEVICE);

      /* ACF */
      ACF = nalu_hypre_ParCSRMatrixCreate(comm,
                                     nC_global,
                                     nF_global,
                                     cpts_starts,
                                     fpts_starts,
                                     num_cols_ACF_offd,
                                     ACF_diag_nnz,
                                     ACF_offd_nnz);

      ACF_diag = nalu_hypre_ParCSRMatrixDiag(ACF);
      nalu_hypre_CSRMatrixData(ACF_diag) = ACF_diag_a;
      nalu_hypre_CSRMatrixI(ACF_diag)    = ACF_diag_i;
      nalu_hypre_CSRMatrixJ(ACF_diag)    = ACF_diag_j;

      ACF_offd = nalu_hypre_ParCSRMatrixOffd(ACF);
      nalu_hypre_CSRMatrixData(ACF_offd) = ACF_offd_a;
      nalu_hypre_CSRMatrixI(ACF_offd)    = ACF_offd_i;
      nalu_hypre_CSRMatrixJ(ACF_offd)    = ACF_offd_j;

      nalu_hypre_CSRMatrixMemoryLocation(ACF_diag) = NALU_HYPRE_MEMORY_DEVICE;
      nalu_hypre_CSRMatrixMemoryLocation(ACF_offd) = NALU_HYPRE_MEMORY_DEVICE;

      nalu_hypre_ParCSRMatrixDeviceColMapOffd(ACF) = col_map_offd_ACF;
      nalu_hypre_ParCSRMatrixColMapOffd(ACF) = nalu_hypre_TAlloc(NALU_HYPRE_BigInt, num_cols_ACF_offd,
                                                       NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TMemcpy(nalu_hypre_ParCSRMatrixColMapOffd(ACF), col_map_offd_ACF, NALU_HYPRE_BigInt, num_cols_ACF_offd,
                    NALU_HYPRE_MEMORY_HOST, NALU_HYPRE_MEMORY_DEVICE);

      nalu_hypre_ParCSRMatrixSetNumNonzeros(ACF);
      nalu_hypre_ParCSRMatrixDNumNonzeros(ACF) = (NALU_HYPRE_Real) nalu_hypre_ParCSRMatrixNumNonzeros(ACF);
      nalu_hypre_MatvecCommPkgCreate(ACF);

      *ACF_ptr = ACF;
   }

   if (ACC_ptr)
   {
      NALU_HYPRE_Int           ACC_diag_nnz, ACC_offd_nnz;
      NALU_HYPRE_Int          *ACC_diag_ii, *ACC_diag_i, *ACC_diag_j;
      NALU_HYPRE_Complex      *ACC_diag_a;
      NALU_HYPRE_Int          *ACC_offd_ii, *ACC_offd_i, *ACC_offd_j;
      NALU_HYPRE_Complex      *ACC_offd_a;
      nalu_hypre_ParCSRMatrix *ACC;
      nalu_hypre_CSRMatrix    *ACC_diag, *ACC_offd;
      NALU_HYPRE_BigInt       *col_map_offd_ACC;
      NALU_HYPRE_Int           num_cols_ACC_offd;

      /* ACC Diag */
      CC_pred<NALU_HYPRE_Int> ACC_pred_diag(CF_marker, CF_marker);
#if defined(NALU_HYPRE_USING_SYCL)
      ACC_diag_nnz = NALU_HYPRE_ONEDPL_CALL( std::count_if,
                                        oneapi::dpl::make_zip_iterator(A_diag_ii, Soc_diag_j),
                                        oneapi::dpl::make_zip_iterator(A_diag_ii, Soc_diag_j) + A_diag_nnz,
                                        ACC_pred_diag );
#else
      ACC_diag_nnz = NALU_HYPRE_THRUST_CALL( count_if,
                                        thrust::make_zip_iterator(thrust::make_tuple(A_diag_ii, Soc_diag_j)),
                                        thrust::make_zip_iterator(thrust::make_tuple(A_diag_ii, Soc_diag_j)) + A_diag_nnz,
                                        ACC_pred_diag );
#endif

      ACC_diag_ii = nalu_hypre_TAlloc(NALU_HYPRE_Int,     ACC_diag_nnz, NALU_HYPRE_MEMORY_DEVICE);
      ACC_diag_j  = nalu_hypre_TAlloc(NALU_HYPRE_Int,     ACC_diag_nnz, NALU_HYPRE_MEMORY_DEVICE);
      ACC_diag_a  = nalu_hypre_TAlloc(NALU_HYPRE_Complex, ACC_diag_nnz, NALU_HYPRE_MEMORY_DEVICE);

      /* Notice that we cannot use Soc_diag_j in the first two arguments since the diagonal is marked as -2 */
#if defined(NALU_HYPRE_USING_SYCL)
      auto new_end = hypreSycl_copy_if( oneapi::dpl::make_zip_iterator(A_diag_ii, A_diag_j, A_diag_a),
                                        oneapi::dpl::make_zip_iterator(A_diag_ii, A_diag_j, A_diag_a) + A_diag_nnz,
                                        oneapi::dpl::make_zip_iterator(A_diag_ii, Soc_diag_j),
                                        oneapi::dpl::make_zip_iterator(ACC_diag_ii, ACC_diag_j, ACC_diag_a),
                                        ACC_pred_diag );

      nalu_hypre_assert( std::get<0>(new_end.base()) == ACC_diag_ii + ACC_diag_nnz );

      hypreSycl_gather( ACC_diag_j,
                        ACC_diag_j + ACC_diag_nnz,
                        map2FC,
                        ACC_diag_j );

      hypreSycl_gather( ACC_diag_ii,
                        ACC_diag_ii + ACC_diag_nnz,
                        map2FC,
                        ACC_diag_ii );
#else
      auto new_end = NALU_HYPRE_THRUST_CALL( copy_if,
                                        thrust::make_zip_iterator(thrust::make_tuple(A_diag_ii, A_diag_j, A_diag_a)),
                                        thrust::make_zip_iterator(thrust::make_tuple(A_diag_ii, A_diag_j, A_diag_a)) + A_diag_nnz,
                                        thrust::make_zip_iterator(thrust::make_tuple(A_diag_ii, Soc_diag_j)),
                                        thrust::make_zip_iterator(thrust::make_tuple(ACC_diag_ii, ACC_diag_j, ACC_diag_a)),
                                        ACC_pred_diag );

      nalu_hypre_assert( thrust::get<0>(new_end.get_iterator_tuple()) == ACC_diag_ii + ACC_diag_nnz );

      NALU_HYPRE_THRUST_CALL ( gather,
                          ACC_diag_j,
                          ACC_diag_j + ACC_diag_nnz,
                          map2FC,
                          ACC_diag_j );

      NALU_HYPRE_THRUST_CALL ( gather,
                          ACC_diag_ii,
                          ACC_diag_ii + ACC_diag_nnz,
                          map2FC,
                          ACC_diag_ii );
#endif

      ACC_diag_i = hypreDevice_CsrRowIndicesToPtrs(nC_local, ACC_diag_nnz, ACC_diag_ii);
      nalu_hypre_TFree(ACC_diag_ii, NALU_HYPRE_MEMORY_DEVICE);

      /* ACC Offd */
      CC_pred<NALU_HYPRE_BigInt> ACC_pred_offd(CF_marker, recv_buf);
#if defined(NALU_HYPRE_USING_SYCL)
      ACC_offd_nnz = NALU_HYPRE_ONEDPL_CALL( std::count_if,
                                        oneapi::dpl::make_zip_iterator(A_offd_ii, Soc_offd_j),
                                        oneapi::dpl::make_zip_iterator(A_offd_ii, Soc_offd_j) + A_offd_nnz,
                                        ACC_pred_offd );
#else
      ACC_offd_nnz = NALU_HYPRE_THRUST_CALL( count_if,
                                        thrust::make_zip_iterator(thrust::make_tuple(A_offd_ii, Soc_offd_j)),
                                        thrust::make_zip_iterator(thrust::make_tuple(A_offd_ii, Soc_offd_j)) + A_offd_nnz,
                                        ACC_pred_offd );
#endif

      ACC_offd_ii = nalu_hypre_TAlloc(NALU_HYPRE_Int,     ACC_offd_nnz, NALU_HYPRE_MEMORY_DEVICE);
      ACC_offd_j  = nalu_hypre_TAlloc(NALU_HYPRE_Int,     ACC_offd_nnz, NALU_HYPRE_MEMORY_DEVICE);
      ACC_offd_a  = nalu_hypre_TAlloc(NALU_HYPRE_Complex, ACC_offd_nnz, NALU_HYPRE_MEMORY_DEVICE);

#if defined(NALU_HYPRE_USING_SYCL)
      new_end = hypreSycl_copy_if( oneapi::dpl::make_zip_iterator(A_offd_ii, Soc_offd_j, A_offd_a),
                                   oneapi::dpl::make_zip_iterator(A_offd_ii, Soc_offd_j, A_offd_a) + A_offd_nnz,
                                   oneapi::dpl::make_zip_iterator(A_offd_ii, Soc_offd_j),
                                   oneapi::dpl::make_zip_iterator(ACC_offd_ii, ACC_offd_j, ACC_offd_a),
                                   ACC_pred_offd );

      nalu_hypre_assert( std::get<0>(new_end.base()) == ACC_offd_ii + ACC_offd_nnz );

      hypreSycl_gather( ACC_offd_ii,
                        ACC_offd_ii + ACC_offd_nnz,
                        map2FC,
                        ACC_offd_ii );
#else
      new_end = NALU_HYPRE_THRUST_CALL( copy_if,
                                   thrust::make_zip_iterator(thrust::make_tuple(A_offd_ii, Soc_offd_j, A_offd_a)),
                                   thrust::make_zip_iterator(thrust::make_tuple(A_offd_ii, Soc_offd_j, A_offd_a)) + A_offd_nnz,
                                   thrust::make_zip_iterator(thrust::make_tuple(A_offd_ii, Soc_offd_j)),
                                   thrust::make_zip_iterator(thrust::make_tuple(ACC_offd_ii, ACC_offd_j, ACC_offd_a)),
                                   ACC_pred_offd );

      nalu_hypre_assert( thrust::get<0>(new_end.get_iterator_tuple()) == ACC_offd_ii + ACC_offd_nnz );

      NALU_HYPRE_THRUST_CALL ( gather,
                          ACC_offd_ii,
                          ACC_offd_ii + ACC_offd_nnz,
                          map2FC,
                          ACC_offd_ii );
#endif

      ACC_offd_i = hypreDevice_CsrRowIndicesToPtrs(nC_local, ACC_offd_nnz, ACC_offd_ii);
      nalu_hypre_TFree(ACC_offd_ii, NALU_HYPRE_MEMORY_DEVICE);

      /* col_map_offd_ACC */
      NALU_HYPRE_Int *tmp_j = nalu_hypre_TAlloc(NALU_HYPRE_Int, nalu_hypre_max(ACC_offd_nnz, num_cols_A_offd),
                                      NALU_HYPRE_MEMORY_DEVICE);
      nalu_hypre_TMemcpy(tmp_j, ACC_offd_j, NALU_HYPRE_Int, ACC_offd_nnz, NALU_HYPRE_MEMORY_DEVICE, NALU_HYPRE_MEMORY_DEVICE);
#if defined(NALU_HYPRE_USING_SYCL)
      NALU_HYPRE_ONEDPL_CALL( std::sort,
                         tmp_j,
                         tmp_j + ACC_offd_nnz );
      NALU_HYPRE_Int *tmp_end = NALU_HYPRE_ONEDPL_CALL( std::unique,
                                              tmp_j,
                                              tmp_j + ACC_offd_nnz );
      num_cols_ACC_offd = tmp_end - tmp_j;
      NALU_HYPRE_ONEDPL_CALL( std::fill_n,
                         offd_mark,
                         num_cols_A_offd,
                         0 );
      hypreDevice_ScatterConstant(offd_mark, num_cols_ACC_offd, tmp_j, (NALU_HYPRE_Int) 1);
      NALU_HYPRE_ONEDPL_CALL( std::exclusive_scan,
                         offd_mark,
                         offd_mark + num_cols_A_offd,
                         tmp_j,
                         0);
      hypreSycl_gather( ACC_offd_j,
                        ACC_offd_j + ACC_offd_nnz,
                        tmp_j,
                        ACC_offd_j );
      col_map_offd_ACC = nalu_hypre_TAlloc(NALU_HYPRE_BigInt, num_cols_ACC_offd, NALU_HYPRE_MEMORY_DEVICE);
      NALU_HYPRE_BigInt *tmp_end_big = hypreSycl_copy_if( recv_buf,
                                                     recv_buf + num_cols_A_offd,
                                                     offd_mark,
                                                     col_map_offd_ACC,
      [] (const auto & x) {return x;} );
#else
      NALU_HYPRE_THRUST_CALL( sort,
                         tmp_j,
                         tmp_j + ACC_offd_nnz );
      NALU_HYPRE_Int *tmp_end = NALU_HYPRE_THRUST_CALL( unique,
                                              tmp_j,
                                              tmp_j + ACC_offd_nnz );
      num_cols_ACC_offd = tmp_end - tmp_j;
      hypreDevice_IntFilln( offd_mark, num_cols_A_offd, 0 );
      hypreDevice_ScatterConstant(offd_mark, num_cols_ACC_offd, tmp_j, (NALU_HYPRE_Int) 1);
      NALU_HYPRE_THRUST_CALL( exclusive_scan,
                         offd_mark,
                         offd_mark + num_cols_A_offd,
                         tmp_j);
      NALU_HYPRE_THRUST_CALL( gather,
                         ACC_offd_j,
                         ACC_offd_j + ACC_offd_nnz,
                         tmp_j,
                         ACC_offd_j );
      col_map_offd_ACC = nalu_hypre_TAlloc(NALU_HYPRE_BigInt, num_cols_ACC_offd, NALU_HYPRE_MEMORY_DEVICE);
      NALU_HYPRE_BigInt *tmp_end_big = NALU_HYPRE_THRUST_CALL( copy_if,
                                                     recv_buf,
                                                     recv_buf + num_cols_A_offd,
                                                     offd_mark,
                                                     col_map_offd_ACC,
                                                     thrust::identity<NALU_HYPRE_Int>());
#endif
      nalu_hypre_assert(tmp_end_big - col_map_offd_ACC == num_cols_ACC_offd);
      nalu_hypre_TFree(tmp_j, NALU_HYPRE_MEMORY_DEVICE);

      /* ACC */
      ACC = nalu_hypre_ParCSRMatrixCreate(comm,
                                     nC_global,
                                     nC_global,
                                     cpts_starts,
                                     cpts_starts,
                                     num_cols_ACC_offd,
                                     ACC_diag_nnz,
                                     ACC_offd_nnz);

      ACC_diag = nalu_hypre_ParCSRMatrixDiag(ACC);
      nalu_hypre_CSRMatrixData(ACC_diag) = ACC_diag_a;
      nalu_hypre_CSRMatrixI(ACC_diag)    = ACC_diag_i;
      nalu_hypre_CSRMatrixJ(ACC_diag)    = ACC_diag_j;

      ACC_offd = nalu_hypre_ParCSRMatrixOffd(ACC);
      nalu_hypre_CSRMatrixData(ACC_offd) = ACC_offd_a;
      nalu_hypre_CSRMatrixI(ACC_offd)    = ACC_offd_i;
      nalu_hypre_CSRMatrixJ(ACC_offd)    = ACC_offd_j;

      nalu_hypre_CSRMatrixMemoryLocation(ACC_diag) = NALU_HYPRE_MEMORY_DEVICE;
      nalu_hypre_CSRMatrixMemoryLocation(ACC_offd) = NALU_HYPRE_MEMORY_DEVICE;

      nalu_hypre_ParCSRMatrixDeviceColMapOffd(ACC) = col_map_offd_ACC;
      nalu_hypre_ParCSRMatrixColMapOffd(ACC) = nalu_hypre_TAlloc(NALU_HYPRE_BigInt, num_cols_ACC_offd,
                                                       NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TMemcpy(nalu_hypre_ParCSRMatrixColMapOffd(ACC), col_map_offd_ACC, NALU_HYPRE_BigInt, num_cols_ACC_offd,
                    NALU_HYPRE_MEMORY_HOST, NALU_HYPRE_MEMORY_DEVICE);

      nalu_hypre_ParCSRMatrixSetNumNonzeros(ACC);
      nalu_hypre_ParCSRMatrixDNumNonzeros(ACC) = (NALU_HYPRE_Real) nalu_hypre_ParCSRMatrixNumNonzeros(ACC);
      nalu_hypre_MatvecCommPkgCreate(ACC);

      *ACC_ptr = ACC;
   }

   nalu_hypre_TFree(A_diag_ii, NALU_HYPRE_MEMORY_DEVICE);
   nalu_hypre_TFree(A_offd_ii, NALU_HYPRE_MEMORY_DEVICE);
   nalu_hypre_TFree(offd_mark, NALU_HYPRE_MEMORY_DEVICE);
   nalu_hypre_TFree(map2FC,    NALU_HYPRE_MEMORY_DEVICE);
   nalu_hypre_TFree(map2F2,    NALU_HYPRE_MEMORY_DEVICE);
   nalu_hypre_TFree(recv_buf,  NALU_HYPRE_MEMORY_DEVICE);

   nalu_hypre_GpuProfilingPopRange();

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ParCSRMatrixGenerateFFFCDevice
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ParCSRMatrixGenerateFFFCDevice( nalu_hypre_ParCSRMatrix  *A,
                                      NALU_HYPRE_Int           *CF_marker,
                                      NALU_HYPRE_BigInt        *cpts_starts,
                                      nalu_hypre_ParCSRMatrix  *S,
                                      nalu_hypre_ParCSRMatrix **AFC_ptr,
                                      nalu_hypre_ParCSRMatrix **AFF_ptr )
{
   return nalu_hypre_ParCSRMatrixGenerateFFFCDevice_core(A, CF_marker, cpts_starts, S,
                                                    AFC_ptr, AFF_ptr,
                                                    NULL, NULL, 1);
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ParCSRMatrixGenerateFFFC3Device
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ParCSRMatrixGenerateFFFC3Device( nalu_hypre_ParCSRMatrix  *A,
                                       NALU_HYPRE_Int           *CF_marker,
                                       NALU_HYPRE_BigInt        *cpts_starts,
                                       nalu_hypre_ParCSRMatrix  *S,
                                       nalu_hypre_ParCSRMatrix **AFC_ptr,
                                       nalu_hypre_ParCSRMatrix **AFF_ptr)
{
   return nalu_hypre_ParCSRMatrixGenerateFFFCDevice_core(A, CF_marker, cpts_starts, S,
                                                    AFC_ptr, AFF_ptr,
                                                    NULL, NULL, 2);
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ParCSRMatrixGenerateFFCFDevice
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ParCSRMatrixGenerateFFCFDevice( nalu_hypre_ParCSRMatrix  *A,
                                      NALU_HYPRE_Int           *CF_marker,
                                      NALU_HYPRE_BigInt        *cpts_starts,
                                      nalu_hypre_ParCSRMatrix  *S,
                                      nalu_hypre_ParCSRMatrix **ACF_ptr,
                                      nalu_hypre_ParCSRMatrix **AFF_ptr )
{
   return nalu_hypre_ParCSRMatrixGenerateFFFCDevice_core(A, CF_marker, cpts_starts, S,
                                                    NULL, AFF_ptr,
                                                    ACF_ptr, NULL, 1);
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ParCSRMatrixGenerateCFDevice
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ParCSRMatrixGenerateCFDevice( nalu_hypre_ParCSRMatrix  *A,
                                    NALU_HYPRE_Int           *CF_marker,
                                    NALU_HYPRE_BigInt        *cpts_starts,
                                    nalu_hypre_ParCSRMatrix  *S,
                                    nalu_hypre_ParCSRMatrix **ACF_ptr)
{
   return nalu_hypre_ParCSRMatrixGenerateFFFCDevice_core(A, CF_marker, cpts_starts, S,
                                                    NULL, NULL,
                                                    ACF_ptr, NULL, 1);
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ParCSRMatrixGenerateCCDevice
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ParCSRMatrixGenerateCCDevice( nalu_hypre_ParCSRMatrix  *A,
                                    NALU_HYPRE_Int           *CF_marker,
                                    NALU_HYPRE_BigInt        *cpts_starts,
                                    nalu_hypre_ParCSRMatrix  *S,
                                    nalu_hypre_ParCSRMatrix **ACC_ptr)
{
   return nalu_hypre_ParCSRMatrixGenerateFFFCDevice_core(A, CF_marker, cpts_starts, S,
                                                    NULL, NULL,
                                                    NULL, ACC_ptr, 1);
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ParCSRMatrixGenerateCCCFDevice
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ParCSRMatrixGenerateCCCFDevice( nalu_hypre_ParCSRMatrix  *A,
                                      NALU_HYPRE_Int           *CF_marker,
                                      NALU_HYPRE_BigInt        *cpts_starts,
                                      nalu_hypre_ParCSRMatrix  *S,
                                      nalu_hypre_ParCSRMatrix **ACF_ptr,
                                      nalu_hypre_ParCSRMatrix **ACC_ptr)
{
   return nalu_hypre_ParCSRMatrixGenerateFFFCDevice_core(A, CF_marker, cpts_starts, S,
                                                    NULL, NULL,
                                                    ACF_ptr, ACC_ptr, 1);
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ParCSRMatrixGenerate1DCFDevice
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ParCSRMatrixGenerate1DCFDevice( nalu_hypre_ParCSRMatrix  *A,
                                      NALU_HYPRE_Int           *CF_marker,
                                      NALU_HYPRE_BigInt        *cpts_starts,
                                      nalu_hypre_ParCSRMatrix  *S,
                                      nalu_hypre_ParCSRMatrix **ACX_ptr,
                                      nalu_hypre_ParCSRMatrix **AXC_ptr )
{
   MPI_Comm                 comm     = nalu_hypre_ParCSRMatrixComm(A);
   nalu_hypre_ParCSRCommPkg     *comm_pkg = nalu_hypre_ParCSRMatrixCommPkg(A);
   nalu_hypre_ParCSRCommHandle  *comm_handle;
   NALU_HYPRE_Int                num_sends     = nalu_hypre_ParCSRCommPkgNumSends(comm_pkg);
   NALU_HYPRE_Int                num_elem_send = nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends);
   //NALU_HYPRE_MemoryLocation     memory_location = nalu_hypre_ParCSRMatrixMemoryLocation(A);
   /* diag part of A */
   nalu_hypre_CSRMatrix    *A_diag   = nalu_hypre_ParCSRMatrixDiag(A);
   NALU_HYPRE_Complex      *A_diag_a = nalu_hypre_CSRMatrixData(A_diag);
   NALU_HYPRE_Int          *A_diag_i = nalu_hypre_CSRMatrixI(A_diag);
   NALU_HYPRE_Int          *A_diag_j = nalu_hypre_CSRMatrixJ(A_diag);
   NALU_HYPRE_Int           A_diag_nnz = nalu_hypre_CSRMatrixNumNonzeros(A_diag);
   /* offd part of A */
   nalu_hypre_CSRMatrix    *A_offd   = nalu_hypre_ParCSRMatrixOffd(A);
   NALU_HYPRE_Complex      *A_offd_a = nalu_hypre_CSRMatrixData(A_offd);
   NALU_HYPRE_Int          *A_offd_i = nalu_hypre_CSRMatrixI(A_offd);
   NALU_HYPRE_Int          *A_offd_j = nalu_hypre_CSRMatrixJ(A_offd);
   NALU_HYPRE_Int           A_offd_nnz = nalu_hypre_CSRMatrixNumNonzeros(A_offd);
   NALU_HYPRE_BigInt       *col_map_offd_A = nalu_hypre_ParCSRMatrixDeviceColMapOffd(A);
   NALU_HYPRE_Int           num_cols_A_offd = nalu_hypre_CSRMatrixNumCols(A_offd);
   /* SoC */
   NALU_HYPRE_Int          *Soc_diag_j = S ? nalu_hypre_ParCSRMatrixSocDiagJ(S) : A_diag_j;
   NALU_HYPRE_Int          *Soc_offd_j = S ? nalu_hypre_ParCSRMatrixSocOffdJ(S) : A_offd_j;
   /* MPI size and rank */
   NALU_HYPRE_Int           my_id, num_procs;
   /* nF and nC */
   NALU_HYPRE_Int           n_local, /*nF_local,*/ nC_local;
   NALU_HYPRE_BigInt        fpts_starts[2], *row_starts;
   NALU_HYPRE_BigInt        /*nF_global,*/ nC_global;
   NALU_HYPRE_BigInt        F_first, C_first;
   /* work arrays */
   NALU_HYPRE_Int          *map2FC, *itmp, *A_diag_ii, *A_offd_ii, *offd_mark;
   NALU_HYPRE_BigInt       *send_buf, *recv_buf;

   nalu_hypre_MPI_Comm_size(comm, &num_procs);
   nalu_hypre_MPI_Comm_rank(comm, &my_id);

   n_local    = nalu_hypre_ParCSRMatrixNumRows(A);
   row_starts = nalu_hypre_ParCSRMatrixRowStarts(A);

   if (!col_map_offd_A)
   {
      col_map_offd_A = nalu_hypre_TAlloc(NALU_HYPRE_BigInt, num_cols_A_offd, NALU_HYPRE_MEMORY_DEVICE);
      nalu_hypre_TMemcpy(col_map_offd_A, nalu_hypre_ParCSRMatrixColMapOffd(A), NALU_HYPRE_BigInt, num_cols_A_offd,
                    NALU_HYPRE_MEMORY_DEVICE, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_ParCSRMatrixDeviceColMapOffd(A) = col_map_offd_A;
   }

   if (my_id == (num_procs - 1))
   {
      nC_global = cpts_starts[1];
   }
   nalu_hypre_MPI_Bcast(&nC_global, 1, NALU_HYPRE_MPI_BIG_INT, num_procs - 1, comm);
   nC_local = (NALU_HYPRE_Int) (cpts_starts[1] - cpts_starts[0]);
   fpts_starts[0] = row_starts[0] - cpts_starts[0];
   fpts_starts[1] = row_starts[1] - cpts_starts[1];
   F_first = fpts_starts[0];
   C_first = cpts_starts[0];
   /*
   nF_local = n_local - nC_local;
   nF_global = nalu_hypre_ParCSRMatrixGlobalNumRows(A) - nC_global;
   */

   map2FC     = nalu_hypre_TAlloc(NALU_HYPRE_Int,    n_local,         NALU_HYPRE_MEMORY_DEVICE);
   itmp       = nalu_hypre_TAlloc(NALU_HYPRE_Int,    n_local,         NALU_HYPRE_MEMORY_DEVICE);
   recv_buf   = nalu_hypre_TAlloc(NALU_HYPRE_BigInt, num_cols_A_offd, NALU_HYPRE_MEMORY_DEVICE);

#if defined(NALU_HYPRE_USING_SYCL)
   /* map from all points (i.e, F+C) to F/C indices */
   NALU_HYPRE_ONEDPL_CALL( std::exclusive_scan,
                      oneapi::dpl::make_transform_iterator(CF_marker,           is_negative<NALU_HYPRE_Int>()),
                      oneapi::dpl::make_transform_iterator(CF_marker + n_local, is_negative<NALU_HYPRE_Int>()),
                      map2FC, /* F */
                      NALU_HYPRE_Int(0) ); /* *MUST* pass init value since input and output types diff. */

   NALU_HYPRE_ONEDPL_CALL( std::exclusive_scan,
                      oneapi::dpl::make_transform_iterator(CF_marker,           is_nonnegative<NALU_HYPRE_Int>()),
                      oneapi::dpl::make_transform_iterator(CF_marker + n_local, is_nonnegative<NALU_HYPRE_Int>()),
                      itmp, /* C */
                      NALU_HYPRE_Int(0) ); /* *MUST* pass init value since input and output types diff. */

   hypreSycl_scatter_if( itmp,
                         itmp + n_local,
                         oneapi::dpl::counting_iterator<NALU_HYPRE_Int>(0),
                         CF_marker,
                         map2FC, /* FC combined */
                         is_nonnegative<NALU_HYPRE_Int>() );
#else
   /* map from all points (i.e, F+C) to F/C indices */
   NALU_HYPRE_THRUST_CALL( exclusive_scan,
                      thrust::make_transform_iterator(CF_marker,           is_negative<NALU_HYPRE_Int>()),
                      thrust::make_transform_iterator(CF_marker + n_local, is_negative<NALU_HYPRE_Int>()),
                      map2FC, /* F */
                      NALU_HYPRE_Int(0) ); /* *MUST* pass init value since input and output types diff. */

   NALU_HYPRE_THRUST_CALL( exclusive_scan,
                      thrust::make_transform_iterator(CF_marker,           is_nonnegative<NALU_HYPRE_Int>()),
                      thrust::make_transform_iterator(CF_marker + n_local, is_nonnegative<NALU_HYPRE_Int>()),
                      itmp, /* C */
                      NALU_HYPRE_Int(0) ); /* *MUST* pass init value since input and output types diff. */

   NALU_HYPRE_THRUST_CALL( scatter_if,
                      itmp,
                      itmp + n_local,
                      thrust::counting_iterator<NALU_HYPRE_Int>(0),
                      thrust::make_transform_iterator(CF_marker, is_nonnegative<NALU_HYPRE_Int>()),
                      map2FC ); /* FC combined */
#endif

   nalu_hypre_TFree(itmp, NALU_HYPRE_MEMORY_DEVICE);

   /* send_buf: global F/C indices. Note F-pts "x" are saved as "-x-1" */
   send_buf = nalu_hypre_TAlloc(NALU_HYPRE_BigInt, num_elem_send, NALU_HYPRE_MEMORY_DEVICE);

   nalu_hypre_ParCSRCommPkgCopySendMapElmtsToDevice(comm_pkg);

   FFFC_functor functor(F_first, C_first);
#if defined(NALU_HYPRE_USING_SYCL)
   auto zip = oneapi::dpl::make_zip_iterator(map2FC, CF_marker);
   hypreSycl_gather( nalu_hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg),
                     nalu_hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg) + num_elem_send,
                     oneapi::dpl::make_transform_iterator(zip, functor),
                     send_buf );
#else
   NALU_HYPRE_THRUST_CALL( gather,
                      nalu_hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg),
                      nalu_hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg) + num_elem_send,
                      thrust::make_transform_iterator(thrust::make_zip_iterator(thrust::make_tuple(map2FC, CF_marker)),
                                                      functor),
                      send_buf );
#endif

#if defined(NALU_HYPRE_WITH_GPU_AWARE_MPI) && defined(NALU_HYPRE_USING_THRUST_NOSYNC)
   /* RL: make sure send_buf is ready before issuing GPU-GPU MPI */
   nalu_hypre_ForceSyncComputeStream(nalu_hypre_handle());
#endif

   comm_handle = nalu_hypre_ParCSRCommHandleCreate_v2(21, comm_pkg, NALU_HYPRE_MEMORY_DEVICE, send_buf,
                                                 NALU_HYPRE_MEMORY_DEVICE, recv_buf);
   nalu_hypre_ParCSRCommHandleDestroy(comm_handle);

   nalu_hypre_TFree(send_buf, NALU_HYPRE_MEMORY_DEVICE);

   A_diag_ii = nalu_hypre_TAlloc(NALU_HYPRE_Int, A_diag_nnz,      NALU_HYPRE_MEMORY_DEVICE);
   A_offd_ii = nalu_hypre_TAlloc(NALU_HYPRE_Int, A_offd_nnz,      NALU_HYPRE_MEMORY_DEVICE);
   offd_mark = nalu_hypre_TAlloc(NALU_HYPRE_Int, num_cols_A_offd, NALU_HYPRE_MEMORY_DEVICE);

   hypreDevice_CsrRowPtrsToIndices_v2(n_local, A_diag_nnz, A_diag_i, A_diag_ii);
   hypreDevice_CsrRowPtrsToIndices_v2(n_local, A_offd_nnz, A_offd_i, A_offd_ii);

   if (ACX_ptr)
   {
      NALU_HYPRE_Int           ACX_diag_nnz, ACX_offd_nnz;
      NALU_HYPRE_Int          *ACX_diag_ii, *ACX_diag_i, *ACX_diag_j;
      NALU_HYPRE_Complex      *ACX_diag_a;
      NALU_HYPRE_Int          *ACX_offd_ii, *ACX_offd_i, *ACX_offd_j;
      NALU_HYPRE_Complex      *ACX_offd_a;
      nalu_hypre_ParCSRMatrix *ACX;
      nalu_hypre_CSRMatrix    *ACX_diag, *ACX_offd;
      NALU_HYPRE_BigInt       *col_map_offd_ACX;
      NALU_HYPRE_Int           num_cols_ACX_offd;

      /* ACX Diag */
      CX_pred ACX_pred(CF_marker);
#if defined(NALU_HYPRE_USING_SYCL)
      ACX_diag_nnz = NALU_HYPRE_ONEDPL_CALL( std::count_if,
                                        oneapi::dpl::make_zip_iterator(A_diag_ii, Soc_diag_j),
                                        oneapi::dpl::make_zip_iterator(A_diag_ii, Soc_diag_j) + A_diag_nnz,
                                        ACX_pred );
#else
      ACX_diag_nnz = NALU_HYPRE_THRUST_CALL( count_if,
                                        thrust::make_zip_iterator(thrust::make_tuple(A_diag_ii, Soc_diag_j)),
                                        thrust::make_zip_iterator(thrust::make_tuple(A_diag_ii, Soc_diag_j)) + A_diag_nnz,
                                        ACX_pred );
#endif

      ACX_diag_ii = nalu_hypre_TAlloc(NALU_HYPRE_Int,     ACX_diag_nnz, NALU_HYPRE_MEMORY_DEVICE);
      ACX_diag_j  = nalu_hypre_TAlloc(NALU_HYPRE_Int,     ACX_diag_nnz, NALU_HYPRE_MEMORY_DEVICE);
      ACX_diag_a  = nalu_hypre_TAlloc(NALU_HYPRE_Complex, ACX_diag_nnz, NALU_HYPRE_MEMORY_DEVICE);

      /* Notice that we cannot use Soc_diag_j in the first two arguments since the diagonal is marked as -2 */
#if defined(NALU_HYPRE_USING_SYCL)
      auto new_end = hypreSycl_copy_if( oneapi::dpl::make_zip_iterator(A_diag_ii, A_diag_j, A_diag_a),
                                        oneapi::dpl::make_zip_iterator(A_diag_ii, A_diag_j, A_diag_a) + A_diag_nnz,
                                        oneapi::dpl::make_zip_iterator(A_diag_ii, Soc_diag_j),
                                        oneapi::dpl::make_zip_iterator(ACX_diag_ii, ACX_diag_j, ACX_diag_a),
                                        ACX_pred );

      nalu_hypre_assert( std::get<0>(new_end.base()) == ACX_diag_ii + ACX_diag_nnz );

      hypreSycl_gather( ACX_diag_ii,
                        ACX_diag_ii + ACX_diag_nnz,
                        map2FC,
                        ACX_diag_ii );
#else
      auto new_end = NALU_HYPRE_THRUST_CALL( copy_if,
                                        thrust::make_zip_iterator(thrust::make_tuple(A_diag_ii, A_diag_j, A_diag_a)),
                                        thrust::make_zip_iterator(thrust::make_tuple(A_diag_ii, A_diag_j, A_diag_a)) + A_diag_nnz,
                                        thrust::make_zip_iterator(thrust::make_tuple(A_diag_ii, Soc_diag_j)),
                                        thrust::make_zip_iterator(thrust::make_tuple(ACX_diag_ii, ACX_diag_j, ACX_diag_a)),
                                        ACX_pred );

      nalu_hypre_assert( thrust::get<0>(new_end.get_iterator_tuple()) == ACX_diag_ii + ACX_diag_nnz );

      NALU_HYPRE_THRUST_CALL ( gather,
                          ACX_diag_ii,
                          ACX_diag_ii + ACX_diag_nnz,
                          map2FC,
                          ACX_diag_ii );
#endif

      ACX_diag_i = hypreDevice_CsrRowIndicesToPtrs(nC_local, ACX_diag_nnz, ACX_diag_ii);
      nalu_hypre_TFree(ACX_diag_ii, NALU_HYPRE_MEMORY_DEVICE);

      /* ACX Offd */
#if defined(NALU_HYPRE_USING_SYCL)
      ACX_offd_nnz = NALU_HYPRE_ONEDPL_CALL( std::count_if,
                                        oneapi::dpl::make_zip_iterator(A_offd_ii, Soc_offd_j),
                                        oneapi::dpl::make_zip_iterator(A_offd_ii, Soc_offd_j) + A_offd_nnz,
                                        ACX_pred );
#else
      ACX_offd_nnz = NALU_HYPRE_THRUST_CALL( count_if,
                                        thrust::make_zip_iterator(thrust::make_tuple(A_offd_ii, Soc_offd_j)),
                                        thrust::make_zip_iterator(thrust::make_tuple(A_offd_ii, Soc_offd_j)) + A_offd_nnz,
                                        ACX_pred );
#endif

      ACX_offd_ii = nalu_hypre_TAlloc(NALU_HYPRE_Int,     ACX_offd_nnz, NALU_HYPRE_MEMORY_DEVICE);
      ACX_offd_j  = nalu_hypre_TAlloc(NALU_HYPRE_Int,     ACX_offd_nnz, NALU_HYPRE_MEMORY_DEVICE);
      ACX_offd_a  = nalu_hypre_TAlloc(NALU_HYPRE_Complex, ACX_offd_nnz, NALU_HYPRE_MEMORY_DEVICE);

#if defined(NALU_HYPRE_USING_SYCL)
      new_end = hypreSycl_copy_if( oneapi::dpl::make_zip_iterator(A_offd_ii, Soc_offd_j, A_offd_a),
                                   oneapi::dpl::make_zip_iterator(A_offd_ii, Soc_offd_j, A_offd_a) + A_offd_nnz,
                                   oneapi::dpl::make_zip_iterator(A_offd_ii, Soc_offd_j),
                                   oneapi::dpl::make_zip_iterator(ACX_offd_ii, ACX_offd_j, ACX_offd_a),
                                   ACX_pred );

      nalu_hypre_assert( std::get<0>(new_end.base()) == ACX_offd_ii + ACX_offd_nnz );

      hypreSycl_gather( ACX_offd_ii,
                        ACX_offd_ii + ACX_offd_nnz,
                        map2FC,
                        ACX_offd_ii );
#else
      new_end = NALU_HYPRE_THRUST_CALL( copy_if,
                                   thrust::make_zip_iterator(thrust::make_tuple(A_offd_ii, Soc_offd_j, A_offd_a)),
                                   thrust::make_zip_iterator(thrust::make_tuple(A_offd_ii, Soc_offd_j, A_offd_a)) + A_offd_nnz,
                                   thrust::make_zip_iterator(thrust::make_tuple(A_offd_ii, Soc_offd_j)),
                                   thrust::make_zip_iterator(thrust::make_tuple(ACX_offd_ii, ACX_offd_j, ACX_offd_a)),
                                   ACX_pred );

      nalu_hypre_assert( thrust::get<0>(new_end.get_iterator_tuple()) == ACX_offd_ii + ACX_offd_nnz );

      NALU_HYPRE_THRUST_CALL ( gather,
                          ACX_offd_ii,
                          ACX_offd_ii + ACX_offd_nnz,
                          map2FC,
                          ACX_offd_ii );
#endif

      ACX_offd_i = hypreDevice_CsrRowIndicesToPtrs(nC_local, ACX_offd_nnz, ACX_offd_ii);
      nalu_hypre_TFree(ACX_offd_ii, NALU_HYPRE_MEMORY_DEVICE);

      /* col_map_offd_ACX */
      NALU_HYPRE_Int *tmp_j = nalu_hypre_TAlloc(NALU_HYPRE_Int, nalu_hypre_max(ACX_offd_nnz, num_cols_A_offd),
                                      NALU_HYPRE_MEMORY_DEVICE);
      nalu_hypre_TMemcpy(tmp_j, ACX_offd_j, NALU_HYPRE_Int, ACX_offd_nnz, NALU_HYPRE_MEMORY_DEVICE, NALU_HYPRE_MEMORY_DEVICE);
#if defined(NALU_HYPRE_USING_SYCL)
      NALU_HYPRE_ONEDPL_CALL( std::sort,
                         tmp_j,
                         tmp_j + ACX_offd_nnz );
      NALU_HYPRE_Int *tmp_end = NALU_HYPRE_ONEDPL_CALL( std::unique,
                                              tmp_j,
                                              tmp_j + ACX_offd_nnz );
#else
      NALU_HYPRE_THRUST_CALL( sort,
                         tmp_j,
                         tmp_j + ACX_offd_nnz );
      NALU_HYPRE_Int *tmp_end = NALU_HYPRE_THRUST_CALL( unique,
                                              tmp_j,
                                              tmp_j + ACX_offd_nnz );
#endif
      num_cols_ACX_offd = tmp_end - tmp_j;
      hypreDevice_IntFilln( offd_mark, num_cols_A_offd, 0 );
      hypreDevice_ScatterConstant(offd_mark, num_cols_ACX_offd, tmp_j, (NALU_HYPRE_Int) 1);
#if defined(NALU_HYPRE_USING_SYCL)
      NALU_HYPRE_ONEDPL_CALL( std::exclusive_scan,
                         offd_mark,
                         offd_mark + num_cols_A_offd,
                         tmp_j,
                         0 );
      hypreSycl_gather( ACX_offd_j,
                        ACX_offd_j + ACX_offd_nnz,
                        tmp_j,
                        ACX_offd_j );
      col_map_offd_ACX = nalu_hypre_TAlloc(NALU_HYPRE_BigInt, num_cols_ACX_offd, NALU_HYPRE_MEMORY_DEVICE);
      NALU_HYPRE_BigInt *tmp_end_big = hypreSycl_copy_if( col_map_offd_A,
                                                     col_map_offd_A + num_cols_A_offd,
                                                     offd_mark,
                                                     col_map_offd_ACX,
      [] (const auto & x) {return x;} );
#else
      NALU_HYPRE_THRUST_CALL( exclusive_scan,
                         offd_mark,
                         offd_mark + num_cols_A_offd,
                         tmp_j);
      NALU_HYPRE_THRUST_CALL( gather,
                         ACX_offd_j,
                         ACX_offd_j + ACX_offd_nnz,
                         tmp_j,
                         ACX_offd_j );
      col_map_offd_ACX = nalu_hypre_TAlloc(NALU_HYPRE_BigInt, num_cols_ACX_offd, NALU_HYPRE_MEMORY_DEVICE);
      NALU_HYPRE_BigInt *tmp_end_big = NALU_HYPRE_THRUST_CALL( copy_if,
                                                     col_map_offd_A,
                                                     col_map_offd_A + num_cols_A_offd,
                                                     offd_mark,
                                                     col_map_offd_ACX,
                                                     thrust::identity<NALU_HYPRE_Int>());
#endif
      nalu_hypre_assert(tmp_end_big - col_map_offd_ACX == num_cols_ACX_offd);
      nalu_hypre_TFree(tmp_j, NALU_HYPRE_MEMORY_DEVICE);

      /* ACX */
      ACX = nalu_hypre_ParCSRMatrixCreate(comm,
                                     nC_global,
                                     nalu_hypre_ParCSRMatrixGlobalNumCols(A),
                                     cpts_starts,
                                     nalu_hypre_ParCSRMatrixColStarts(A),
                                     num_cols_ACX_offd,
                                     ACX_diag_nnz,
                                     ACX_offd_nnz);

      ACX_diag = nalu_hypre_ParCSRMatrixDiag(ACX);
      nalu_hypre_CSRMatrixData(ACX_diag) = ACX_diag_a;
      nalu_hypre_CSRMatrixI(ACX_diag)    = ACX_diag_i;
      nalu_hypre_CSRMatrixJ(ACX_diag)    = ACX_diag_j;

      ACX_offd = nalu_hypre_ParCSRMatrixOffd(ACX);
      nalu_hypre_CSRMatrixData(ACX_offd) = ACX_offd_a;
      nalu_hypre_CSRMatrixI(ACX_offd)    = ACX_offd_i;
      nalu_hypre_CSRMatrixJ(ACX_offd)    = ACX_offd_j;

      nalu_hypre_CSRMatrixMemoryLocation(ACX_diag) = NALU_HYPRE_MEMORY_DEVICE;
      nalu_hypre_CSRMatrixMemoryLocation(ACX_offd) = NALU_HYPRE_MEMORY_DEVICE;

      nalu_hypre_ParCSRMatrixDeviceColMapOffd(ACX) = col_map_offd_ACX;
      nalu_hypre_ParCSRMatrixColMapOffd(ACX) = nalu_hypre_TAlloc(NALU_HYPRE_BigInt, num_cols_ACX_offd,
                                                       NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TMemcpy(nalu_hypre_ParCSRMatrixColMapOffd(ACX), col_map_offd_ACX, NALU_HYPRE_BigInt, num_cols_ACX_offd,
                    NALU_HYPRE_MEMORY_HOST, NALU_HYPRE_MEMORY_DEVICE);

      nalu_hypre_ParCSRMatrixSetNumNonzeros(ACX);
      nalu_hypre_ParCSRMatrixDNumNonzeros(ACX) = (NALU_HYPRE_Real) nalu_hypre_ParCSRMatrixNumNonzeros(ACX);
      nalu_hypre_MatvecCommPkgCreate(ACX);

      *ACX_ptr = ACX;
   }

   if (AXC_ptr)
   {
      NALU_HYPRE_Int           AXC_diag_nnz, AXC_offd_nnz;
      NALU_HYPRE_Int          *AXC_diag_ii, *AXC_diag_i, *AXC_diag_j;
      NALU_HYPRE_Complex      *AXC_diag_a;
      NALU_HYPRE_Int          *AXC_offd_ii, *AXC_offd_i, *AXC_offd_j;
      NALU_HYPRE_Complex      *AXC_offd_a;
      nalu_hypre_ParCSRMatrix *AXC;
      nalu_hypre_CSRMatrix    *AXC_diag, *AXC_offd;
      NALU_HYPRE_BigInt       *col_map_offd_AXC;
      NALU_HYPRE_Int           num_cols_AXC_offd;

      /* AXC Diag */
      XC_pred<NALU_HYPRE_Int> AXC_pred_diag(CF_marker);
#if defined(NALU_HYPRE_USING_SYCL)
      AXC_diag_nnz = NALU_HYPRE_ONEDPL_CALL( std::count_if,
                                        oneapi::dpl::make_zip_iterator(A_diag_ii, Soc_diag_j),
                                        oneapi::dpl::make_zip_iterator(A_diag_ii, Soc_diag_j) + A_diag_nnz,
                                        AXC_pred_diag );
#else
      AXC_diag_nnz = NALU_HYPRE_THRUST_CALL( count_if,
                                        thrust::make_zip_iterator(thrust::make_tuple(A_diag_ii, Soc_diag_j)),
                                        thrust::make_zip_iterator(thrust::make_tuple(A_diag_ii, Soc_diag_j)) + A_diag_nnz,
                                        AXC_pred_diag );
#endif

      AXC_diag_ii = nalu_hypre_TAlloc(NALU_HYPRE_Int,     AXC_diag_nnz, NALU_HYPRE_MEMORY_DEVICE);
      AXC_diag_j  = nalu_hypre_TAlloc(NALU_HYPRE_Int,     AXC_diag_nnz, NALU_HYPRE_MEMORY_DEVICE);
      AXC_diag_a  = nalu_hypre_TAlloc(NALU_HYPRE_Complex, AXC_diag_nnz, NALU_HYPRE_MEMORY_DEVICE);

      /* Notice that we cannot use Soc_diag_j in the first two arguments since the diagonal is marked as -2 */
#if defined(NALU_HYPRE_USING_SYCL)
      auto new_end = hypreSycl_copy_if( oneapi::dpl::make_zip_iterator(A_diag_ii, A_diag_j, A_diag_a),
                                        oneapi::dpl::make_zip_iterator(A_diag_ii, A_diag_j, A_diag_a) + A_diag_nnz,
                                        oneapi::dpl::make_zip_iterator(A_diag_ii, Soc_diag_j),
                                        oneapi::dpl::make_zip_iterator(AXC_diag_ii, AXC_diag_j, AXC_diag_a),
                                        AXC_pred_diag );

      nalu_hypre_assert( std::get<0>(new_end.base()) == AXC_diag_ii + AXC_diag_nnz );

      hypreSycl_gather( AXC_diag_j,
                        AXC_diag_j + AXC_diag_nnz,
                        map2FC,
                        AXC_diag_j );
#else
      auto new_end = NALU_HYPRE_THRUST_CALL( copy_if,
                                        thrust::make_zip_iterator(thrust::make_tuple(A_diag_ii, A_diag_j, A_diag_a)),
                                        thrust::make_zip_iterator(thrust::make_tuple(A_diag_ii, A_diag_j, A_diag_a)) + A_diag_nnz,
                                        thrust::make_zip_iterator(thrust::make_tuple(A_diag_ii, Soc_diag_j)),
                                        thrust::make_zip_iterator(thrust::make_tuple(AXC_diag_ii, AXC_diag_j, AXC_diag_a)),
                                        AXC_pred_diag );

      nalu_hypre_assert( thrust::get<0>(new_end.get_iterator_tuple()) == AXC_diag_ii + AXC_diag_nnz );

      NALU_HYPRE_THRUST_CALL ( gather,
                          AXC_diag_j,
                          AXC_diag_j + AXC_diag_nnz,
                          map2FC,
                          AXC_diag_j );
#endif

      AXC_diag_i = hypreDevice_CsrRowIndicesToPtrs(n_local, AXC_diag_nnz, AXC_diag_ii);
      nalu_hypre_TFree(AXC_diag_ii, NALU_HYPRE_MEMORY_DEVICE);

      /* AXC Offd */
      XC_pred<NALU_HYPRE_BigInt> AXC_pred_offd(recv_buf);
#if defined(NALU_HYPRE_USING_SYCL)
      AXC_offd_nnz = NALU_HYPRE_ONEDPL_CALL( std::count_if,
                                        oneapi::dpl::make_zip_iterator(A_offd_ii, Soc_offd_j),
                                        oneapi::dpl::make_zip_iterator(A_offd_ii, Soc_offd_j) + A_offd_nnz,
                                        AXC_pred_offd );
#else
      AXC_offd_nnz = NALU_HYPRE_THRUST_CALL( count_if,
                                        thrust::make_zip_iterator(thrust::make_tuple(A_offd_ii, Soc_offd_j)),
                                        thrust::make_zip_iterator(thrust::make_tuple(A_offd_ii, Soc_offd_j)) + A_offd_nnz,
                                        AXC_pred_offd );
#endif

      AXC_offd_ii = nalu_hypre_TAlloc(NALU_HYPRE_Int,     AXC_offd_nnz, NALU_HYPRE_MEMORY_DEVICE);
      AXC_offd_j  = nalu_hypre_TAlloc(NALU_HYPRE_Int,     AXC_offd_nnz, NALU_HYPRE_MEMORY_DEVICE);
      AXC_offd_a  = nalu_hypre_TAlloc(NALU_HYPRE_Complex, AXC_offd_nnz, NALU_HYPRE_MEMORY_DEVICE);

#if defined(NALU_HYPRE_USING_SYCL)
      new_end = hypreSycl_copy_if( oneapi::dpl::make_zip_iterator(A_offd_ii, Soc_offd_j, A_offd_a),
                                   oneapi::dpl::make_zip_iterator(A_offd_ii, Soc_offd_j, A_offd_a) + A_offd_nnz,
                                   oneapi::dpl::make_zip_iterator(A_offd_ii, Soc_offd_j),
                                   oneapi::dpl::make_zip_iterator(AXC_offd_ii, AXC_offd_j, AXC_offd_a),
                                   AXC_pred_offd );

      nalu_hypre_assert( std::get<0>(new_end.base()) == AXC_offd_ii + AXC_offd_nnz );
#else
      new_end = NALU_HYPRE_THRUST_CALL( copy_if,
                                   thrust::make_zip_iterator(thrust::make_tuple(A_offd_ii, Soc_offd_j, A_offd_a)),
                                   thrust::make_zip_iterator(thrust::make_tuple(A_offd_ii, Soc_offd_j, A_offd_a)) + A_offd_nnz,
                                   thrust::make_zip_iterator(thrust::make_tuple(A_offd_ii, Soc_offd_j)),
                                   thrust::make_zip_iterator(thrust::make_tuple(AXC_offd_ii, AXC_offd_j, AXC_offd_a)),
                                   AXC_pred_offd );

      nalu_hypre_assert( thrust::get<0>(new_end.get_iterator_tuple()) == AXC_offd_ii + AXC_offd_nnz );
#endif

      AXC_offd_i = hypreDevice_CsrRowIndicesToPtrs(n_local, AXC_offd_nnz, AXC_offd_ii);
      nalu_hypre_TFree(AXC_offd_ii, NALU_HYPRE_MEMORY_DEVICE);

      /* col_map_offd_AXC */
      NALU_HYPRE_Int *tmp_j = nalu_hypre_TAlloc(NALU_HYPRE_Int, nalu_hypre_max(AXC_offd_nnz, num_cols_A_offd),
                                      NALU_HYPRE_MEMORY_DEVICE);
      nalu_hypre_TMemcpy(tmp_j, AXC_offd_j, NALU_HYPRE_Int, AXC_offd_nnz, NALU_HYPRE_MEMORY_DEVICE, NALU_HYPRE_MEMORY_DEVICE);
#if defined(NALU_HYPRE_USING_SYCL)
      NALU_HYPRE_ONEDPL_CALL( std::sort,
                         tmp_j,
                         tmp_j + AXC_offd_nnz );
      NALU_HYPRE_Int *tmp_end = NALU_HYPRE_ONEDPL_CALL( std::unique,
                                              tmp_j,
                                              tmp_j + AXC_offd_nnz );
#else
      NALU_HYPRE_THRUST_CALL( sort,
                         tmp_j,
                         tmp_j + AXC_offd_nnz );
      NALU_HYPRE_Int *tmp_end = NALU_HYPRE_THRUST_CALL( unique,
                                              tmp_j,
                                              tmp_j + AXC_offd_nnz );
#endif
      num_cols_AXC_offd = tmp_end - tmp_j;
      hypreDevice_IntFilln( offd_mark, num_cols_A_offd, 0 );
      hypreDevice_ScatterConstant(offd_mark, num_cols_AXC_offd, tmp_j, (NALU_HYPRE_Int) 1);
#if defined(NALU_HYPRE_USING_SYCL)
      NALU_HYPRE_ONEDPL_CALL( std::exclusive_scan,
                         offd_mark,
                         offd_mark + num_cols_A_offd,
                         tmp_j,
                         0 );
      hypreSycl_gather( AXC_offd_j,
                        AXC_offd_j + AXC_offd_nnz,
                        tmp_j,
                        AXC_offd_j );
      col_map_offd_AXC = nalu_hypre_TAlloc(NALU_HYPRE_BigInt, num_cols_AXC_offd, NALU_HYPRE_MEMORY_DEVICE);
      NALU_HYPRE_BigInt *tmp_end_big = hypreSycl_copy_if( recv_buf,
                                                     recv_buf + num_cols_A_offd,
                                                     offd_mark,
                                                     col_map_offd_AXC,
      [] (const auto & x) {return x;} );
#else
      NALU_HYPRE_THRUST_CALL( exclusive_scan,
                         offd_mark,
                         offd_mark + num_cols_A_offd,
                         tmp_j);
      NALU_HYPRE_THRUST_CALL( gather,
                         AXC_offd_j,
                         AXC_offd_j + AXC_offd_nnz,
                         tmp_j,
                         AXC_offd_j );
      col_map_offd_AXC = nalu_hypre_TAlloc(NALU_HYPRE_BigInt, num_cols_AXC_offd, NALU_HYPRE_MEMORY_DEVICE);
      NALU_HYPRE_BigInt *tmp_end_big = NALU_HYPRE_THRUST_CALL( copy_if,
                                                     recv_buf,
                                                     recv_buf + num_cols_A_offd,
                                                     offd_mark,
                                                     col_map_offd_AXC,
                                                     thrust::identity<NALU_HYPRE_Int>());
#endif
      nalu_hypre_assert(tmp_end_big - col_map_offd_AXC == num_cols_AXC_offd);
      nalu_hypre_TFree(tmp_j, NALU_HYPRE_MEMORY_DEVICE);

      /* AXC */
      AXC = nalu_hypre_ParCSRMatrixCreate(comm,
                                     nalu_hypre_ParCSRMatrixGlobalNumRows(A),
                                     nC_global,
                                     row_starts,
                                     cpts_starts,
                                     num_cols_AXC_offd,
                                     AXC_diag_nnz,
                                     AXC_offd_nnz);

      AXC_diag = nalu_hypre_ParCSRMatrixDiag(AXC);
      nalu_hypre_CSRMatrixData(AXC_diag) = AXC_diag_a;
      nalu_hypre_CSRMatrixI(AXC_diag)    = AXC_diag_i;
      nalu_hypre_CSRMatrixJ(AXC_diag)    = AXC_diag_j;

      AXC_offd = nalu_hypre_ParCSRMatrixOffd(AXC);
      nalu_hypre_CSRMatrixData(AXC_offd) = AXC_offd_a;
      nalu_hypre_CSRMatrixI(AXC_offd)    = AXC_offd_i;
      nalu_hypre_CSRMatrixJ(AXC_offd)    = AXC_offd_j;

      nalu_hypre_CSRMatrixMemoryLocation(AXC_diag) = NALU_HYPRE_MEMORY_DEVICE;
      nalu_hypre_CSRMatrixMemoryLocation(AXC_offd) = NALU_HYPRE_MEMORY_DEVICE;

      nalu_hypre_ParCSRMatrixDeviceColMapOffd(AXC) = col_map_offd_AXC;
      nalu_hypre_ParCSRMatrixColMapOffd(AXC) = nalu_hypre_TAlloc(NALU_HYPRE_BigInt, num_cols_AXC_offd,
                                                       NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TMemcpy(nalu_hypre_ParCSRMatrixColMapOffd(AXC), col_map_offd_AXC, NALU_HYPRE_BigInt, num_cols_AXC_offd,
                    NALU_HYPRE_MEMORY_HOST, NALU_HYPRE_MEMORY_DEVICE);

      nalu_hypre_ParCSRMatrixSetNumNonzeros(AXC);
      nalu_hypre_ParCSRMatrixDNumNonzeros(AXC) = (NALU_HYPRE_Real) nalu_hypre_ParCSRMatrixNumNonzeros(AXC);
      nalu_hypre_MatvecCommPkgCreate(AXC);

      *AXC_ptr = AXC;
   }

   nalu_hypre_TFree(A_diag_ii, NALU_HYPRE_MEMORY_DEVICE);
   nalu_hypre_TFree(A_offd_ii, NALU_HYPRE_MEMORY_DEVICE);
   nalu_hypre_TFree(offd_mark, NALU_HYPRE_MEMORY_DEVICE);
   nalu_hypre_TFree(map2FC,    NALU_HYPRE_MEMORY_DEVICE);
   nalu_hypre_TFree(recv_buf,  NALU_HYPRE_MEMORY_DEVICE);

   return nalu_hypre_error_flag;
}

#endif // #if defined(NALU_HYPRE_USING_GPU)
