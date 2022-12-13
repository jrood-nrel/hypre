/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * IJMatrix_ParCSR interface
 *
 *****************************************************************************/

#include "_hypre_onedpl.hpp"
#include "_hypre_IJ_mv.h"
#include "_hypre_utilities.hpp"

#if defined(NALU_HYPRE_USING_CUDA) || defined(NALU_HYPRE_USING_HIP) || defined(NALU_HYPRE_USING_SYCL)

__global__ void
hypreGPUKernel_IJMatrixValues_dev1(hypre_DeviceItem &item, NALU_HYPRE_Int n, NALU_HYPRE_Int *rowind,
                                   NALU_HYPRE_Int *row_ptr,
                                   NALU_HYPRE_Int *row_len, NALU_HYPRE_Int *mark)
{
   NALU_HYPRE_Int global_thread_id = hypre_gpu_get_grid_thread_id<1, 1>(item);

   if (global_thread_id < n)
   {
      NALU_HYPRE_Int row = rowind[global_thread_id];
      if (global_thread_id < read_only_load(&row_ptr[row]) + read_only_load(&row_len[row]))
      {
         mark[global_thread_id] = 0;
      }
      else
      {
         mark[global_thread_id] = -1;
      }
   }
}

/* E.g. nrows = 3
 *      ncols = 2 3 4
 *      rows  = 10 20 30
 *      rows_indexes = 0 4 9
 *              (0 1 2 3 | 4 5 6 7 8 | 9 10 11 12 13)
 *      cols   = x x ! ! | * * * ! ! | +  +  +  +  !
 *      values = . . ! ! | . . . ! ! | .  .  .  .  !
 */

NALU_HYPRE_Int
hypre_IJMatrixSetAddValuesParCSRDevice( hypre_IJMatrix       *matrix,
                                        NALU_HYPRE_Int             nrows,
                                        NALU_HYPRE_Int            *ncols,        /* if NULL, == all ones */
                                        const NALU_HYPRE_BigInt   *rows,
                                        const NALU_HYPRE_Int      *row_indexes,  /* if NULL, == ex_scan of ncols, i.e, no gap */
                                        const NALU_HYPRE_BigInt   *cols,
                                        const NALU_HYPRE_Complex  *values,
                                        const char           *action )
{
   NALU_HYPRE_BigInt *row_partitioning = hypre_IJMatrixRowPartitioning(matrix);
   NALU_HYPRE_BigInt *col_partitioning = hypre_IJMatrixColPartitioning(matrix);
   NALU_HYPRE_BigInt row_start = row_partitioning[0];
   NALU_HYPRE_BigInt row_end   = row_partitioning[1];
   NALU_HYPRE_BigInt col_start = col_partitioning[0];
   NALU_HYPRE_BigInt col_end   = col_partitioning[1];
   NALU_HYPRE_Int num_local_rows = row_end - row_start;
   NALU_HYPRE_Int num_local_cols = col_end - col_start;
   const char SorA = action[0] == 's' ? 1 : 0;

   hypre_AuxParCSRMatrix *aux_matrix = (hypre_AuxParCSRMatrix *) hypre_IJMatrixTranslator(matrix);

   NALU_HYPRE_Int  nelms;
   NALU_HYPRE_Int *row_ptr = NULL;

   /* expand rows into full expansion of rows based on ncols
    * if ncols == NULL, ncols is all ones, so rows are indeed full expansion */
   if (ncols)
   {
      row_ptr = hypre_TAlloc(NALU_HYPRE_Int, nrows + 1, NALU_HYPRE_MEMORY_DEVICE);
      hypre_TMemcpy(row_ptr, ncols, NALU_HYPRE_Int, nrows, NALU_HYPRE_MEMORY_DEVICE, NALU_HYPRE_MEMORY_DEVICE);
      /* RL: have to init the last entry !!! */
      hypre_Memset(row_ptr + nrows, 0, sizeof(NALU_HYPRE_Int), NALU_HYPRE_MEMORY_DEVICE);
      hypreDevice_IntegerExclusiveScan(nrows + 1, row_ptr);
      hypre_TMemcpy(&nelms, row_ptr + nrows, NALU_HYPRE_Int, 1, NALU_HYPRE_MEMORY_HOST, NALU_HYPRE_MEMORY_DEVICE);
   }
   else
   {
      nelms = nrows;
   }

   if (nelms <= 0)
   {
      hypre_TFree(row_ptr, NALU_HYPRE_MEMORY_DEVICE);
      return hypre_error_flag;
   }

   if (!aux_matrix)
   {
      hypre_AuxParCSRMatrixCreate(&aux_matrix, num_local_rows, num_local_cols, NULL);
      hypre_AuxParCSRMatrixInitialize_v2(aux_matrix, NALU_HYPRE_MEMORY_DEVICE);
      hypre_IJMatrixTranslator(matrix) = aux_matrix;
   }

   NALU_HYPRE_Int      stack_elmts_max      = hypre_AuxParCSRMatrixMaxStackElmts(aux_matrix);
   NALU_HYPRE_Int      stack_elmts_current  = hypre_AuxParCSRMatrixCurrentStackElmts(aux_matrix);
   NALU_HYPRE_Int      stack_elmts_required = stack_elmts_current + nelms;
   NALU_HYPRE_BigInt  *stack_i              = hypre_AuxParCSRMatrixStackI(aux_matrix);
   NALU_HYPRE_BigInt  *stack_j              = hypre_AuxParCSRMatrixStackJ(aux_matrix);
   NALU_HYPRE_Complex *stack_data           = hypre_AuxParCSRMatrixStackData(aux_matrix);
   char          *stack_sora           = hypre_AuxParCSRMatrixStackSorA(aux_matrix);

   if ( stack_elmts_max < stack_elmts_required )
   {
      NALU_HYPRE_Int stack_elmts_max_new = hypre_max(hypre_AuxParCSRMatrixUsrOnProcElmts (aux_matrix), 0) +
                                      hypre_max(hypre_AuxParCSRMatrixUsrOffProcElmts(aux_matrix), 0);
      if ( hypre_AuxParCSRMatrixUsrOnProcElmts (aux_matrix) < 0 ||
           hypre_AuxParCSRMatrixUsrOffProcElmts(aux_matrix) < 0 )
      {
         stack_elmts_max_new = hypre_max(num_local_rows * hypre_AuxParCSRMatrixInitAllocFactor(aux_matrix),
                                         stack_elmts_max_new);
         stack_elmts_max_new = hypre_max(stack_elmts_max * hypre_AuxParCSRMatrixGrowFactor(aux_matrix),
                                         stack_elmts_max_new);
      }
      stack_elmts_max_new = hypre_max(stack_elmts_required, stack_elmts_max_new);

      hypre_AuxParCSRMatrixStackI(aux_matrix)    = stack_i    = hypre_TReAlloc_v2(stack_i,
                                                                                  NALU_HYPRE_BigInt,  stack_elmts_max, NALU_HYPRE_BigInt,  stack_elmts_max_new, NALU_HYPRE_MEMORY_DEVICE);
      hypre_AuxParCSRMatrixStackJ(aux_matrix)    = stack_j    = hypre_TReAlloc_v2(stack_j,
                                                                                  NALU_HYPRE_BigInt,  stack_elmts_max, NALU_HYPRE_BigInt,  stack_elmts_max_new, NALU_HYPRE_MEMORY_DEVICE);
      hypre_AuxParCSRMatrixStackData(aux_matrix) = stack_data = hypre_TReAlloc_v2(stack_data,
                                                                                  NALU_HYPRE_Complex, stack_elmts_max, NALU_HYPRE_Complex, stack_elmts_max_new, NALU_HYPRE_MEMORY_DEVICE);
      hypre_AuxParCSRMatrixStackSorA(aux_matrix) = stack_sora = hypre_TReAlloc_v2(stack_sora,
                                                                                  char, stack_elmts_max,          char, stack_elmts_max_new, NALU_HYPRE_MEMORY_DEVICE);
      hypre_AuxParCSRMatrixMaxStackElmts(aux_matrix) = stack_elmts_max_new;
   }

   hypreDevice_CharFilln(stack_sora + stack_elmts_current, nelms, SorA);

   if (ncols)
   {
      hypreDevice_CsrRowPtrsToIndicesWithRowNum(nrows, nelms, row_ptr, (NALU_HYPRE_BigInt *) rows,
                                                stack_i + stack_elmts_current);
   }
   else
   {
      hypre_TMemcpy(stack_i + stack_elmts_current, rows, NALU_HYPRE_BigInt, nelms, NALU_HYPRE_MEMORY_DEVICE,
                    NALU_HYPRE_MEMORY_DEVICE);
   }

   if (row_indexes)
   {
      NALU_HYPRE_Int len, len1;
      hypre_TMemcpy(&len1, &row_indexes[nrows - 1], NALU_HYPRE_Int, 1, NALU_HYPRE_MEMORY_HOST, NALU_HYPRE_MEMORY_DEVICE);
      if (ncols)
      {
         hypre_TMemcpy(&len, &ncols[nrows - 1], NALU_HYPRE_Int, 1, NALU_HYPRE_MEMORY_HOST, NALU_HYPRE_MEMORY_DEVICE);
      }
      else
      {
         len = 1;
      }
      /* this is the *effective* length of cols and values */
      len += len1;
      NALU_HYPRE_Int *indicator = hypre_CTAlloc(NALU_HYPRE_Int, len, NALU_HYPRE_MEMORY_DEVICE);
      hypreDevice_CsrRowPtrsToIndices_v2(nrows - 1, len1, (NALU_HYPRE_Int *) row_indexes, indicator);
      /* mark unwanted elements as -1 */
      dim3 bDim = hypre_GetDefaultDeviceBlockDimension();
      dim3 gDim = hypre_GetDefaultDeviceGridDimension(len1, "thread", bDim);
      NALU_HYPRE_GPU_LAUNCH( hypreGPUKernel_IJMatrixValues_dev1, gDim, bDim, len1, indicator,
                        (NALU_HYPRE_Int *) row_indexes, ncols, indicator );

#if defined(NALU_HYPRE_USING_SYCL)
      auto zip_in = oneapi::dpl::make_zip_iterator(cols, values);
      auto zip_out = oneapi::dpl::make_zip_iterator(stack_j + stack_elmts_current,
                                                    stack_data + stack_elmts_current);
      auto new_end = hypreSycl_copy_if( zip_in,
                                        zip_in + len,
                                        indicator,
                                        zip_out,
                                        is_nonnegative<NALU_HYPRE_Int>() );

      NALU_HYPRE_Int nnz_tmp = std::get<0>(new_end.base()) - (stack_j + stack_elmts_current);
#else
      auto new_end = NALU_HYPRE_THRUST_CALL(
                        copy_if,
                        thrust::make_zip_iterator(thrust::make_tuple(cols,       values)),
                        thrust::make_zip_iterator(thrust::make_tuple(cols + len, values + len)),
                        indicator,
                        thrust::make_zip_iterator(thrust::make_tuple(stack_j    + stack_elmts_current,
                                                                     stack_data + stack_elmts_current)),
                        is_nonnegative<NALU_HYPRE_Int>() );

      NALU_HYPRE_Int nnz_tmp = thrust::get<0>(new_end.get_iterator_tuple()) - (stack_j + stack_elmts_current);
#endif

      hypre_assert(nnz_tmp == nelms);

      hypre_TFree(indicator, NALU_HYPRE_MEMORY_DEVICE);
   }
   else
   {
      hypre_TMemcpy(stack_j    + stack_elmts_current, cols,   NALU_HYPRE_BigInt,  nelms, NALU_HYPRE_MEMORY_DEVICE,
                    NALU_HYPRE_MEMORY_DEVICE);
      hypre_TMemcpy(stack_data + stack_elmts_current, values, NALU_HYPRE_Complex, nelms, NALU_HYPRE_MEMORY_DEVICE,
                    NALU_HYPRE_MEMORY_DEVICE);
   }

   hypre_AuxParCSRMatrixCurrentStackElmts(aux_matrix) += nelms;

   hypre_TFree(row_ptr, NALU_HYPRE_MEMORY_DEVICE);

   return hypre_error_flag;
}

#if defined(NALU_HYPRE_USING_SYCL)
template<typename T1, typename T2>
struct hypre_IJMatrixAssembleFunctor
{
   typedef std::tuple<T1, T2> Tuple;

   Tuple operator()(const Tuple& x, const Tuple& y ) const
   {
      return std::make_tuple( hypre_max(std::get<0>(x), std::get<0>(y)),
                              std::get<1>(x) + std::get<1>(y) );
   }
};
#else
template<typename T1, typename T2>
struct hypre_IJMatrixAssembleFunctor : public
   thrust::binary_function< thrust::tuple<T1, T2>, thrust::tuple<T1, T2>, thrust::tuple<T1, T2> >
{
   typedef thrust::tuple<T1, T2> Tuple;

   __device__ Tuple operator()(const Tuple& x, const Tuple& y )
   {
      return thrust::make_tuple( hypre_max(thrust::get<0>(x), thrust::get<0>(y)),
                                 thrust::get<1>(x) + thrust::get<1>(y) );
   }
};
#endif

/* helper routine used in hypre_IJMatrixAssembleParCSRDevice:
 * 1. sort (X0, A0) with key (I0, J0)
 *    [put the diagonal first; see the comments in cuda_utils.c]
 * 2. for each segment in (I0, J0), zero out in A0 all before the last `set'
 * 3. reduce A0 [with sum] and reduce X0 [with max]
 * N0: input size; N1: size after reduction (<= N0)
 * Note: (I1, J1, X1, A1) are not resized to N1 but have size N0
 */
NALU_HYPRE_Int
hypre_IJMatrixAssembleSortAndReduce1(NALU_HYPRE_Int  N0, NALU_HYPRE_BigInt  *I0, NALU_HYPRE_BigInt  *J0, char  *X0,
                                     NALU_HYPRE_Complex  *A0,
                                     NALU_HYPRE_Int *N1, NALU_HYPRE_BigInt **I1, NALU_HYPRE_BigInt **J1, char **X1, NALU_HYPRE_Complex **A1 )
{
   hypreDevice_StableSortTupleByTupleKey(N0, I0, J0, X0, A0, 2);

   NALU_HYPRE_BigInt  *I = hypre_TAlloc(NALU_HYPRE_BigInt,  N0, NALU_HYPRE_MEMORY_DEVICE);
   NALU_HYPRE_BigInt  *J = hypre_TAlloc(NALU_HYPRE_BigInt,  N0, NALU_HYPRE_MEMORY_DEVICE);
   char          *X = hypre_TAlloc(char,          N0, NALU_HYPRE_MEMORY_DEVICE);
   NALU_HYPRE_Complex *A = hypre_TAlloc(NALU_HYPRE_Complex, N0, NALU_HYPRE_MEMORY_DEVICE);

   /*
   dim3 bDim = hypre_GetDefaultDeviceBlockDimension();
   dim3 gDim = hypre_GetDefaultDeviceGridDimension(N0, "thread", bDim);
   NALU_HYPRE_GPU_LAUNCH( hypreGPUKernel_IJMatrixAssembleSortAndReduce1, gDim, bDim, N0, I0, J0, X0, A0 );
   */

   /* output X: 0: keep, 1: zero-out */
#if defined(NALU_HYPRE_USING_SYCL)
   /* WM: oneDPL currently does not have a reverse iterator */
   /*     should be able to do this with a reverse operation defined in a struct */
   /*     instead of explicitly allocating and generating the reverse_perm, */
   /*     but I can't get that to work for some reason */
   NALU_HYPRE_Int *reverse_perm = hypre_TAlloc(NALU_HYPRE_Int, N0, NALU_HYPRE_MEMORY_DEVICE);
   NALU_HYPRE_ONEDPL_CALL( std::transform,
                      oneapi::dpl::counting_iterator(0),
                      oneapi::dpl::counting_iterator(N0),
                      reverse_perm,
   [N0] (auto i) { return N0 - i - 1; });

   auto I0_J0_reversed = oneapi::dpl::make_permutation_iterator(
                            oneapi::dpl::make_zip_iterator(I0, J0), reverse_perm);
   auto X0_reversed = oneapi::dpl::make_permutation_iterator(X0, reverse_perm);
   auto X_reversed = oneapi::dpl::make_permutation_iterator(X, reverse_perm);

   NALU_HYPRE_ONEDPL_CALL( oneapi::dpl::exclusive_scan_by_segment,
                      I0_J0_reversed,      /* key begin */
                      I0_J0_reversed + N0, /* key end */
                      X0_reversed,      /* input value begin */
                      X_reversed,       /* output value begin */
                      char(0),          /* init */
                      std::equal_to< std::tuple<NALU_HYPRE_BigInt, NALU_HYPRE_BigInt> >(),
                      oneapi::dpl::maximum<char>() );

   hypre_TFree(reverse_perm, NALU_HYPRE_MEMORY_DEVICE);

   hypreSycl_transform_if(A0,
                          A0 + N0,
                          X,
                          A0,
   [] (const auto & x) {return x;},
   [] (const auto & x) {return 0.0;} );

   auto I0_J0_zip = oneapi::dpl::make_zip_iterator(I0, J0);
   auto new_end = NALU_HYPRE_ONEDPL_CALL( oneapi::dpl::reduce_by_segment,
                                     I0_J0_zip,                                                    /* keys_first */
                                     I0_J0_zip + N0,                                               /* keys_last */
                                     oneapi::dpl::make_zip_iterator(X0, A0),                       /* values_first */
                                     oneapi::dpl::make_zip_iterator(I, J),                         /* keys_output */
                                     oneapi::dpl::make_zip_iterator(X, A),                         /* values_output */
                                     std::equal_to< std::tuple<NALU_HYPRE_BigInt, NALU_HYPRE_BigInt> >(),    /* binary_pred */
                                     hypre_IJMatrixAssembleFunctor<char, NALU_HYPRE_Complex>()          /* binary_op */);

   *N1 = std::get<0>(new_end.first.base()) - I;
#else
   NALU_HYPRE_THRUST_CALL(
      exclusive_scan_by_key,
      make_reverse_iterator(thrust::make_zip_iterator(thrust::make_tuple(I0 + N0, J0 + N0))),
      make_reverse_iterator(thrust::make_zip_iterator(thrust::make_tuple(I0,    J0))),
      make_reverse_iterator(thrust::device_pointer_cast<char>(X0) + N0),
      make_reverse_iterator(thrust::device_pointer_cast<char>(X) + N0),
      char(0),
      thrust::equal_to< thrust::tuple<NALU_HYPRE_BigInt, NALU_HYPRE_BigInt> >(),
      thrust::maximum<char>() );

   NALU_HYPRE_THRUST_CALL(replace_if, A0, A0 + N0, X, thrust::identity<char>(), 0.0);

   auto new_end = NALU_HYPRE_THRUST_CALL(
                     reduce_by_key,
                     thrust::make_zip_iterator(thrust::make_tuple(I0,      J0     )), /* keys_first */
                     thrust::make_zip_iterator(thrust::make_tuple(I0 + N0, J0 + N0)), /* keys_last */
                     thrust::make_zip_iterator(thrust::make_tuple(X0,      A0     )), /* values_first */
                     thrust::make_zip_iterator(thrust::make_tuple(I,       J      )), /* keys_output */
                     thrust::make_zip_iterator(thrust::make_tuple(X,       A      )), /* values_output */
                     thrust::equal_to< thrust::tuple<NALU_HYPRE_BigInt, NALU_HYPRE_BigInt> >(), /* binary_pred */
                     hypre_IJMatrixAssembleFunctor<char, NALU_HYPRE_Complex>()             /* binary_op */);

   *N1 = thrust::get<0>(new_end.first.get_iterator_tuple()) - I;
#endif
   *I1 = I;
   *J1 = J;
   *X1 = X;
   *A1 = A;

   return hypre_error_flag;
}

#if defined(NALU_HYPRE_USING_SYCL)
template<typename T1, typename T2>
struct hypre_IJMatrixAssembleFunctor2
{
   typedef std::tuple<T1, T2> Tuple;

   __device__ Tuple operator()(const Tuple& x, const Tuple& y) const
   {
      const char          tx = std::get<0>(x);
      const char          ty = std::get<0>(y);
      const NALU_HYPRE_Complex vx = std::get<1>(x);
      const NALU_HYPRE_Complex vy = std::get<1>(y);
      const NALU_HYPRE_Complex vz = tx == 0 && ty == 0 ? vx + vy : tx ? vx : vy;
      return std::make_tuple(0, vz);
   }
};
#else
template<typename T1, typename T2>
struct hypre_IJMatrixAssembleFunctor2 : public
   thrust::binary_function< thrust::tuple<T1, T2>, thrust::tuple<T1, T2>, thrust::tuple<T1, T2> >
{
   typedef thrust::tuple<T1, T2> Tuple;

   __device__ Tuple operator()(const Tuple& x, const Tuple& y)
   {
      const char          tx = thrust::get<0>(x);
      const char          ty = thrust::get<0>(y);
      const NALU_HYPRE_Complex vx = thrust::get<1>(x);
      const NALU_HYPRE_Complex vy = thrust::get<1>(y);
      const NALU_HYPRE_Complex vz = tx == 0 && ty == 0 ? vx + vy : tx ? vx : vy;
      return thrust::make_tuple(0, vz);
   }
};
#endif

NALU_HYPRE_Int
hypre_IJMatrixAssembleSortAndReduce2(NALU_HYPRE_Int  N0, NALU_HYPRE_Int  *I0, NALU_HYPRE_Int  *J0, char  *X0,
                                     NALU_HYPRE_Complex  *A0,
                                     NALU_HYPRE_Int *N1, NALU_HYPRE_Int **I1, NALU_HYPRE_Int **J1,            NALU_HYPRE_Complex **A1,
                                     NALU_HYPRE_Int  opt )
{
   hypreDevice_StableSortTupleByTupleKey(N0, I0, J0, X0, A0, opt);

   NALU_HYPRE_Int     *I = hypre_TAlloc(NALU_HYPRE_Int,     N0, NALU_HYPRE_MEMORY_DEVICE);
   NALU_HYPRE_Int     *J = hypre_TAlloc(NALU_HYPRE_Int,     N0, NALU_HYPRE_MEMORY_DEVICE);
   char          *X = hypre_TAlloc(char,          N0, NALU_HYPRE_MEMORY_DEVICE);
   NALU_HYPRE_Complex *A = hypre_TAlloc(NALU_HYPRE_Complex, N0, NALU_HYPRE_MEMORY_DEVICE);

#if defined(NALU_HYPRE_USING_SYCL)
   auto new_end = NALU_HYPRE_ONEDPL_CALL( oneapi::dpl::reduce_by_segment,
                                     oneapi::dpl::make_zip_iterator(I0, J0),                 /* keys_first */
                                     oneapi::dpl::make_zip_iterator(I0 + N0, J0 + N0),       /* keys_last */
                                     oneapi::dpl::make_zip_iterator(X0, A0),                 /* values_first */
                                     oneapi::dpl::make_zip_iterator(I, J),                   /* keys_output */
                                     oneapi::dpl::make_zip_iterator(X, A),                   /* values_output */
                                     std::equal_to< std::tuple<NALU_HYPRE_Int, NALU_HYPRE_Int> >(),    /* binary_pred */
                                     hypre_IJMatrixAssembleFunctor2<char, NALU_HYPRE_Complex>()   /* binary_op */);

   *N1 = std::get<0>(new_end.first.base()) - I;
#else
   auto new_end = NALU_HYPRE_THRUST_CALL(
                     reduce_by_key,
                     thrust::make_zip_iterator(thrust::make_tuple(I0,      J0     )), /* keys_first */
                     thrust::make_zip_iterator(thrust::make_tuple(I0 + N0, J0 + N0)), /* keys_last */
                     thrust::make_zip_iterator(thrust::make_tuple(X0,      A0     )), /* values_first */
                     thrust::make_zip_iterator(thrust::make_tuple(I,       J      )), /* keys_output */
                     thrust::make_zip_iterator(thrust::make_tuple(X,       A      )), /* values_output */
                     thrust::equal_to< thrust::tuple<NALU_HYPRE_Int, NALU_HYPRE_Int> >(),       /* binary_pred */
                     hypre_IJMatrixAssembleFunctor2<char, NALU_HYPRE_Complex>()            /* binary_op */);

   *N1 = thrust::get<0>(new_end.first.get_iterator_tuple()) - I;
#endif
   *I1 = I;
   *J1 = J;
   *A1 = A;

   hypre_TFree(X, NALU_HYPRE_MEMORY_DEVICE);

   return hypre_error_flag;
}

NALU_HYPRE_Int
hypre_IJMatrixAssembleSortAndReduce3(NALU_HYPRE_Int  N0, NALU_HYPRE_BigInt  *I0, NALU_HYPRE_BigInt  *J0,  char *X0,
                                     NALU_HYPRE_Complex  *A0,
                                     NALU_HYPRE_Int *N1)
{
   hypreDevice_StableSortTupleByTupleKey(N0, I0, J0, X0, A0, 0);

   NALU_HYPRE_BigInt  *I = hypre_TAlloc(NALU_HYPRE_BigInt,  N0, NALU_HYPRE_MEMORY_DEVICE);
   NALU_HYPRE_BigInt  *J = hypre_TAlloc(NALU_HYPRE_BigInt,  N0, NALU_HYPRE_MEMORY_DEVICE);
   NALU_HYPRE_Complex *A = hypre_TAlloc(NALU_HYPRE_Complex, N0, NALU_HYPRE_MEMORY_DEVICE);

#if defined(NALU_HYPRE_USING_SYCL)
   /* WM: oneDPL currently does not have a reverse iterator */
   /*     should be able to do this with a reverse operation defined in a struct */
   /*     instead of explicitly allocating and generating the reverse_perm, */
   /*     but I can't get that to work for some reason */
   NALU_HYPRE_Int *reverse_perm = hypre_TAlloc(NALU_HYPRE_Int, N0, NALU_HYPRE_MEMORY_DEVICE);
   NALU_HYPRE_ONEDPL_CALL( std::transform,
                      oneapi::dpl::counting_iterator(0),
                      oneapi::dpl::counting_iterator(N0),
                      reverse_perm,
   [N0] (auto i) { return N0 - i - 1; });

   auto I0_J0_reversed = oneapi::dpl::make_permutation_iterator(
                            oneapi::dpl::make_zip_iterator(I0, J0), reverse_perm);
   auto X0_reversed = oneapi::dpl::make_permutation_iterator(X0, reverse_perm);

   NALU_HYPRE_ONEDPL_CALL( oneapi::dpl::inclusive_scan_by_segment,
                      I0_J0_reversed,      /* key begin */
                      I0_J0_reversed + N0, /* key end */
                      X0_reversed,         /* input value begin */
                      X0_reversed,         /* output value begin */
                      std::equal_to< std::tuple<NALU_HYPRE_BigInt, NALU_HYPRE_BigInt> >(),
                      oneapi::dpl::maximum<char>() );

   hypre_TFree(reverse_perm, NALU_HYPRE_MEMORY_DEVICE);

   hypreSycl_transform_if(A0,
                          A0 + N0,
                          X0,
                          A0,
   [] (const auto & x) {return x;},
   [] (const auto & x) {return 0.0;} );

   auto I0_J0_zip = oneapi::dpl::make_zip_iterator(I0, J0);

   auto new_end = NALU_HYPRE_ONEDPL_CALL( oneapi::dpl::reduce_by_segment,
                                     I0_J0_zip,                                                    /* keys_first */
                                     I0_J0_zip + N0,                                               /* keys_last */
                                     A0,                                                           /* values_first */
                                     oneapi::dpl::make_zip_iterator(I, J),                         /* keys_output */
                                     A,                                                            /* values_output */
                                     std::equal_to< std::tuple<NALU_HYPRE_BigInt, NALU_HYPRE_BigInt> >()     /* binary_pred */);
#else
   /* output in X0: 0: keep, 1: zero-out */
   NALU_HYPRE_THRUST_CALL(
      inclusive_scan_by_key,
      make_reverse_iterator(thrust::make_zip_iterator(thrust::make_tuple(I0 + N0, J0 + N0))),
      make_reverse_iterator(thrust::make_zip_iterator(thrust::make_tuple(I0,    J0))),
      make_reverse_iterator(thrust::device_pointer_cast<char>(X0) + N0),
      make_reverse_iterator(thrust::device_pointer_cast<char>(X0) + N0),
      thrust::equal_to< thrust::tuple<NALU_HYPRE_BigInt, NALU_HYPRE_BigInt> >(),
      thrust::maximum<char>() );

   NALU_HYPRE_THRUST_CALL(replace_if, A0, A0 + N0, X0, thrust::identity<char>(), 0.0);

   auto new_end = NALU_HYPRE_THRUST_CALL(
                     reduce_by_key,
                     thrust::make_zip_iterator(thrust::make_tuple(I0,      J0     )), /* keys_first */
                     thrust::make_zip_iterator(thrust::make_tuple(I0 + N0, J0 + N0)), /* keys_last */
                     A0,                                                              /* values_first */
                     thrust::make_zip_iterator(thrust::make_tuple(I,       J      )), /* keys_output */
                     A,                                                               /* values_output */
                     thrust::equal_to< thrust::tuple<NALU_HYPRE_Int, NALU_HYPRE_Int> >()        /* binary_pred */);
#endif

   NALU_HYPRE_Int Nt = new_end.second - A;

   hypre_assert(Nt <= N0);

   /* remove numrical zeros */
#if defined(NALU_HYPRE_USING_SYCL)
   auto new_end2 = hypreSycl_copy_if( oneapi::dpl::make_zip_iterator(I, J, A),
                                      oneapi::dpl::make_zip_iterator(I + Nt, J + Nt, A + Nt),
                                      A,
                                      oneapi::dpl::make_zip_iterator(I0, J0, A0),
   [] (const auto & x) {return x;});

   *N1 = std::get<0>(new_end2.base()) - I0;
#else
   auto new_end2 = NALU_HYPRE_THRUST_CALL( copy_if,
                                      thrust::make_zip_iterator(thrust::make_tuple(I,    J,    A)),
                                      thrust::make_zip_iterator(thrust::make_tuple(I + Nt, J + Nt, A + Nt)),
                                      A,
                                      thrust::make_zip_iterator(thrust::make_tuple(I0, J0, A0)),
                                      thrust::identity<NALU_HYPRE_Complex>() );

   *N1 = thrust::get<0>(new_end2.get_iterator_tuple()) - I0;
#endif

   hypre_assert(*N1 <= Nt);

   hypre_TFree(I, NALU_HYPRE_MEMORY_DEVICE);
   hypre_TFree(J, NALU_HYPRE_MEMORY_DEVICE);
   hypre_TFree(A, NALU_HYPRE_MEMORY_DEVICE);

   return hypre_error_flag;
}

#if 0
NALU_HYPRE_Int
hypre_IJMatrixAssembleSortAndRemove(NALU_HYPRE_Int N0, NALU_HYPRE_BigInt *I0, NALU_HYPRE_BigInt *J0, char *X0,
                                    NALU_HYPRE_Complex *A0)
{
   hypreDevice_StableSortTupleByTupleKey(N0, I0, J0, X0, A0, 0);

   /* output in X0: 0: keep, 1: remove */
   NALU_HYPRE_THRUST_CALL(
      inclusive_scan_by_key,
      make_reverse_iterator(thrust::make_zip_iterator(thrust::make_tuple(I0 + N0, J0 + N0))),
      make_reverse_iterator(thrust::make_zip_iterator(thrust::make_tuple(I0,    J0))),
      make_reverse_iterator(thrust::device_pointer_cast<char>(X0) + N0),
      make_reverse_iterator(thrust::device_pointer_cast<char>(X0) + N0),
      thrust::equal_to< thrust::tuple<NALU_HYPRE_BigInt, NALU_HYPRE_BigInt> >(),
      thrust::maximum<char>() );

   auto new_end = NALU_HYPRE_THRUST_CALL(
                     remove_if,
                     thrust::make_zip_iterator(thrust::make_tuple(I0,    J0,    A0)),
                     thrust::make_zip_iterator(thrust::make_tuple(I0 + N0, J0 + N0, A0 + N0)),
                     X0,
                     thrust::identity<char>());

   NALU_HYPRE_Int N1 = thrust::get<0>(new_end.get_iterator_tuple()) - I0;

   hypre_assert(N1 >= 0 && N1 <= N0);

   return N1;
}
#endif

NALU_HYPRE_Int
hypre_IJMatrixAssembleParCSRDevice(hypre_IJMatrix *matrix)
{
   MPI_Comm comm = hypre_IJMatrixComm(matrix);
   NALU_HYPRE_BigInt *row_partitioning = hypre_IJMatrixRowPartitioning(matrix);
   NALU_HYPRE_BigInt *col_partitioning = hypre_IJMatrixColPartitioning(matrix);
   NALU_HYPRE_BigInt row_start = row_partitioning[0];
   NALU_HYPRE_BigInt row_end   = row_partitioning[1];
   NALU_HYPRE_BigInt col_start = col_partitioning[0];
   NALU_HYPRE_BigInt col_end   = col_partitioning[1];
   NALU_HYPRE_Int nrows = row_end - row_start;
   NALU_HYPRE_Int ncols = col_end - col_start;

   hypre_ParCSRMatrix    *par_matrix = (hypre_ParCSRMatrix*)    hypre_IJMatrixObject(matrix);
   hypre_AuxParCSRMatrix *aux_matrix = (hypre_AuxParCSRMatrix*) hypre_IJMatrixTranslator(matrix);

   if (!aux_matrix)
   {
      return hypre_error_flag;
   }

   if (!par_matrix)
   {
      return hypre_error_flag;
   }

   NALU_HYPRE_Int      nelms      = hypre_AuxParCSRMatrixCurrentStackElmts(aux_matrix);
   NALU_HYPRE_BigInt  *stack_i    = hypre_AuxParCSRMatrixStackI(aux_matrix);
   NALU_HYPRE_BigInt  *stack_j    = hypre_AuxParCSRMatrixStackJ(aux_matrix);
   NALU_HYPRE_Complex *stack_data = hypre_AuxParCSRMatrixStackData(aux_matrix);
   char          *stack_sora = hypre_AuxParCSRMatrixStackSorA(aux_matrix);

   in_range<NALU_HYPRE_BigInt> pred(row_start, row_end - 1);
#if defined(NALU_HYPRE_USING_SYCL)
   NALU_HYPRE_Int nelms_on = NALU_HYPRE_ONEDPL_CALL(std::count_if, stack_i, stack_i + nelms, pred);
#else
   NALU_HYPRE_Int nelms_on = NALU_HYPRE_THRUST_CALL(count_if, stack_i, stack_i + nelms, pred);
#endif
   NALU_HYPRE_Int nelms_off = nelms - nelms_on;
   NALU_HYPRE_Int nelms_off_max;
   hypre_MPI_Allreduce(&nelms_off, &nelms_off_max, 1, NALU_HYPRE_MPI_INT, hypre_MPI_MAX, comm);

   /* communicate for aux off-proc and add to remote aux on-proc */
   if (nelms_off_max)
   {
      NALU_HYPRE_Int      new_nnz       = 0;
      NALU_HYPRE_BigInt  *off_proc_i    = NULL;
      NALU_HYPRE_BigInt  *off_proc_j    = NULL;
      NALU_HYPRE_Complex *off_proc_data = NULL;

      if (nelms_off)
      {
         /* copy off-proc entries out of stack and remove from stack */
         off_proc_i          = hypre_TAlloc(NALU_HYPRE_BigInt,  nelms_off, NALU_HYPRE_MEMORY_DEVICE);
         off_proc_j          = hypre_TAlloc(NALU_HYPRE_BigInt,  nelms_off, NALU_HYPRE_MEMORY_DEVICE);
         off_proc_data       = hypre_TAlloc(NALU_HYPRE_Complex, nelms_off, NALU_HYPRE_MEMORY_DEVICE);
         char *off_proc_sora = hypre_TAlloc(char,          nelms_off, NALU_HYPRE_MEMORY_DEVICE);
         char *is_on_proc    = hypre_TAlloc(char,          nelms,     NALU_HYPRE_MEMORY_DEVICE);

#if defined(NALU_HYPRE_USING_SYCL)
         NALU_HYPRE_ONEDPL_CALL(std::transform, stack_i, stack_i + nelms, is_on_proc, pred);
         /* WM: HERE */
         auto zip_in = oneapi::dpl::make_zip_iterator(stack_i, stack_j, stack_data, stack_sora);
         auto zip_out = oneapi::dpl::make_zip_iterator(off_proc_i, off_proc_j, off_proc_data, off_proc_sora);
         auto new_end1 = hypreSycl_copy_if( zip_in,         /* first */
                                            zip_in + nelms, /* last */
                                            is_on_proc,     /* stencil */
                                            zip_out,        /* result */
         [] (const auto & x) {return !x;} );

         hypre_assert(std::get<0>(new_end1.base()) - off_proc_i == nelms_off);

         /* remove off-proc entries from stack */
         auto new_end2 = hypreSycl_remove_if( zip_in,          /* first */
                                              zip_in + nelms,  /* last */
                                              is_on_proc,      /* stencil */
         [] (const auto & x) {return !x;} );

         hypre_assert(std::get<0>(new_end2.base()) - stack_i == nelms_on);
#else
         NALU_HYPRE_THRUST_CALL(transform, stack_i, stack_i + nelms, is_on_proc, pred);

         auto new_end1 = NALU_HYPRE_THRUST_CALL(
                            copy_if,
                            thrust::make_zip_iterator(thrust::make_tuple(stack_i,         stack_j,         stack_data,
                                                                         stack_sora        )),  /* first */
                            thrust::make_zip_iterator(thrust::make_tuple(stack_i + nelms, stack_j + nelms, stack_data + nelms,
                                                                         stack_sora + nelms)),  /* last */
                            is_on_proc,                                                                                                               /* stencil */
                            thrust::make_zip_iterator(thrust::make_tuple(off_proc_i,      off_proc_j,      off_proc_data,
                                                                         off_proc_sora)),       /* result */
                            thrust::not1(thrust::identity<char>()) );

         hypre_assert(thrust::get<0>(new_end1.get_iterator_tuple()) - off_proc_i == nelms_off);

         /* remove off-proc entries from stack */
         auto new_end2 = NALU_HYPRE_THRUST_CALL(
                            remove_if,
                            thrust::make_zip_iterator(thrust::make_tuple(stack_i,         stack_j,         stack_data,
                                                                         stack_sora        )),  /* first */
                            thrust::make_zip_iterator(thrust::make_tuple(stack_i + nelms, stack_j + nelms, stack_data + nelms,
                                                                         stack_sora + nelms)),  /* last */
                            is_on_proc,                                                                                                               /* stencil */
                            thrust::not1(thrust::identity<char>()) );

         hypre_assert(thrust::get<0>(new_end2.get_iterator_tuple()) - stack_i == nelms_on);
#endif

         hypre_AuxParCSRMatrixCurrentStackElmts(aux_matrix) = nelms_on;

         hypre_TFree(is_on_proc, NALU_HYPRE_MEMORY_DEVICE);

         /* sort and reduce */
         hypre_IJMatrixAssembleSortAndReduce3(nelms_off, off_proc_i, off_proc_j, off_proc_sora,
                                              off_proc_data, &new_nnz);
         // new_nnz = hypre_IJMatrixAssembleSortAndRemove(nelms_off, off_proc_i, off_proc_j, off_proc_sora, off_proc_data);

         hypre_TFree(off_proc_sora, NALU_HYPRE_MEMORY_DEVICE);
      }

      /* send new_i/j/data to remote processes and the receivers call addtovalues */
      hypre_IJMatrixAssembleOffProcValsParCSR(matrix, -1, -1, new_nnz, NALU_HYPRE_MEMORY_DEVICE, off_proc_i,
                                              off_proc_j, off_proc_data);

      hypre_TFree(off_proc_i,    NALU_HYPRE_MEMORY_DEVICE);
      hypre_TFree(off_proc_j,    NALU_HYPRE_MEMORY_DEVICE);
      hypre_TFree(off_proc_data, NALU_HYPRE_MEMORY_DEVICE);
   }

   /* Note: the stack might have been changed in hypre_IJMatrixAssembleOffProcValsParCSR,
    * so must get the size and the pointers again */
   nelms      = hypre_AuxParCSRMatrixCurrentStackElmts(aux_matrix);
   stack_i    = hypre_AuxParCSRMatrixStackI(aux_matrix);
   stack_j    = hypre_AuxParCSRMatrixStackJ(aux_matrix);
   stack_data = hypre_AuxParCSRMatrixStackData(aux_matrix);
   stack_sora = hypre_AuxParCSRMatrixStackSorA(aux_matrix);

#ifdef NALU_HYPRE_DEBUG
   /* the stack should only have on-proc elements now */
#if defined(NALU_HYPRE_USING_SYCL)
   NALU_HYPRE_Int tmp = NALU_HYPRE_ONEDPL_CALL(std::count_if, stack_i, stack_i + nelms, pred);
#else
   NALU_HYPRE_Int tmp = NALU_HYPRE_THRUST_CALL(count_if, stack_i, stack_i + nelms, pred);
#endif
   hypre_assert(nelms == tmp);
#endif

   if (nelms)
   {
      NALU_HYPRE_Int      new_nnz;
      NALU_HYPRE_BigInt  *new_i;
      NALU_HYPRE_BigInt  *new_j;
      NALU_HYPRE_Complex *new_data;
      char          *new_sora;

      /* sort and reduce */
      hypre_IJMatrixAssembleSortAndReduce1(nelms, stack_i, stack_j, stack_sora, stack_data,
                                           &new_nnz, &new_i, &new_j, &new_sora, &new_data);

      /* adjust row indices from global to local */
      NALU_HYPRE_Int *new_i_local = hypre_TAlloc(NALU_HYPRE_Int, new_nnz, NALU_HYPRE_MEMORY_DEVICE);
#if defined(NALU_HYPRE_USING_SYCL)
      NALU_HYPRE_ONEDPL_CALL( std::transform,
                         new_i,
                         new_i + new_nnz,
                         new_i_local,
      [row_start = row_start] (const auto & x) {return x - row_start;} );
#else
      NALU_HYPRE_THRUST_CALL( transform,
                         new_i,
                         new_i + new_nnz,
                         new_i_local,
                         _1 - row_start );
#endif

      hypre_TFree(new_i, NALU_HYPRE_MEMORY_DEVICE);

      NALU_HYPRE_Int      num_cols_offd_new;
      NALU_HYPRE_BigInt  *col_map_offd_new;
      NALU_HYPRE_Int     *col_map_offd_map;
      NALU_HYPRE_Int      diag_nnz_new;
      NALU_HYPRE_Int     *diag_i_new = NULL;
      NALU_HYPRE_Int     *diag_j_new = NULL;
      NALU_HYPRE_Complex *diag_a_new = NULL;
      char          *diag_sora_new = NULL;
      NALU_HYPRE_Int      offd_nnz_new;
      NALU_HYPRE_Int     *offd_i_new = NULL;
      NALU_HYPRE_Int     *offd_j_new = NULL;
      NALU_HYPRE_Complex *offd_a_new = NULL;
      char          *offd_sora_new = NULL;

      NALU_HYPRE_Int diag_nnz_existed = hypre_CSRMatrixNumNonzeros(hypre_ParCSRMatrixDiag(par_matrix));
      NALU_HYPRE_Int offd_nnz_existed = hypre_CSRMatrixNumNonzeros(hypre_ParCSRMatrixOffd(par_matrix));

      hypre_CSRMatrixSplitDevice_core( 0,
                                       nrows,
                                       new_nnz,
                                       NULL,
                                       new_j,
                                       NULL,
                                       NULL,
                                       col_start,
                                       col_end - 1,
                                       hypre_CSRMatrixNumCols(hypre_ParCSRMatrixOffd(par_matrix)),
                                       NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                       &diag_nnz_new,
                                       NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                       &offd_nnz_new,
                                       NULL,
                                       NULL,
                                       NULL,
                                       NULL );

      if (diag_nnz_new)
      {
         diag_i_new = hypre_TAlloc(NALU_HYPRE_Int,     diag_nnz_existed + diag_nnz_new, NALU_HYPRE_MEMORY_DEVICE);
         diag_j_new = hypre_TAlloc(NALU_HYPRE_Int,     diag_nnz_existed + diag_nnz_new, NALU_HYPRE_MEMORY_DEVICE);
         diag_a_new = hypre_TAlloc(NALU_HYPRE_Complex, diag_nnz_existed + diag_nnz_new, NALU_HYPRE_MEMORY_DEVICE);
         if (diag_nnz_existed)
         {
            diag_sora_new = hypre_TAlloc(char,    diag_nnz_existed + diag_nnz_new, NALU_HYPRE_MEMORY_DEVICE);
         }
      }

      if (offd_nnz_new)
      {
         offd_i_new = hypre_TAlloc(NALU_HYPRE_Int,     offd_nnz_existed + offd_nnz_new, NALU_HYPRE_MEMORY_DEVICE);
         offd_j_new = hypre_TAlloc(NALU_HYPRE_Int,     offd_nnz_existed + offd_nnz_new, NALU_HYPRE_MEMORY_DEVICE);
         offd_a_new = hypre_TAlloc(NALU_HYPRE_Complex, offd_nnz_existed + offd_nnz_new, NALU_HYPRE_MEMORY_DEVICE);
         if (offd_nnz_existed)
         {
            offd_sora_new = hypre_TAlloc(char,    offd_nnz_existed + offd_nnz_new, NALU_HYPRE_MEMORY_DEVICE);
         }
      }

      /* split IJ into diag and offd */
      hypre_CSRMatrixSplitDevice_core( 1,
                                       nrows,
                                       new_nnz,
                                       new_i_local,
                                       new_j,
                                       new_data,
                                       diag_nnz_existed || offd_nnz_existed ? new_sora : NULL,
                                       col_start,
                                       col_end - 1,
                                       hypre_CSRMatrixNumCols(hypre_ParCSRMatrixOffd(par_matrix)),
                                       hypre_ParCSRMatrixDeviceColMapOffd(par_matrix),
                                       &col_map_offd_map,
                                       &num_cols_offd_new,
                                       &col_map_offd_new,
                                       &diag_nnz_new,
                                       diag_i_new + diag_nnz_existed,
                                       diag_j_new + diag_nnz_existed,
                                       diag_a_new + diag_nnz_existed,
                                       diag_nnz_existed ? diag_sora_new + diag_nnz_existed : NULL,
                                       &offd_nnz_new,
                                       offd_i_new + offd_nnz_existed,
                                       offd_j_new + offd_nnz_existed,
                                       offd_a_new + offd_nnz_existed,
                                       offd_nnz_existed ? offd_sora_new + offd_nnz_existed : NULL );

      hypre_TFree(new_i_local, NALU_HYPRE_MEMORY_DEVICE);
      hypre_TFree(new_j,       NALU_HYPRE_MEMORY_DEVICE);
      hypre_TFree(new_data,    NALU_HYPRE_MEMORY_DEVICE);
      hypre_TFree(new_sora,    NALU_HYPRE_MEMORY_DEVICE);

      NALU_HYPRE_Int      nnz_new;
      NALU_HYPRE_Int     *tmp_i;
      NALU_HYPRE_Int     *tmp_j;
      NALU_HYPRE_Complex *tmp_a;

      /* expand the existing diag/offd and compress with the new one */
      if (diag_nnz_new > 0)
      {
         if (diag_nnz_existed)
         {
            /* the existing parcsr should come first and the entries are "add" */
            hypreDevice_CsrRowPtrsToIndices_v2(nrows, diag_nnz_existed,
                                               hypre_CSRMatrixI(hypre_ParCSRMatrixDiag(par_matrix)), diag_i_new);

            hypre_TMemcpy(diag_j_new, hypre_CSRMatrixJ(hypre_ParCSRMatrixDiag(par_matrix)), NALU_HYPRE_Int,
                          diag_nnz_existed, NALU_HYPRE_MEMORY_DEVICE, NALU_HYPRE_MEMORY_DEVICE);

            hypre_TMemcpy(diag_a_new, hypre_CSRMatrixData(hypre_ParCSRMatrixDiag(par_matrix)), NALU_HYPRE_Complex,
                          diag_nnz_existed, NALU_HYPRE_MEMORY_DEVICE, NALU_HYPRE_MEMORY_DEVICE);

            hypreDevice_CharFilln(diag_sora_new, diag_nnz_existed, 0);

            hypre_IJMatrixAssembleSortAndReduce2(diag_nnz_existed + diag_nnz_new, diag_i_new, diag_j_new,
                                                 diag_sora_new, diag_a_new,
                                                 &nnz_new, &tmp_i, &tmp_j, &tmp_a, 2);

            hypre_TFree(diag_i_new,    NALU_HYPRE_MEMORY_DEVICE);
            hypre_TFree(diag_j_new,    NALU_HYPRE_MEMORY_DEVICE);
            hypre_TFree(diag_sora_new, NALU_HYPRE_MEMORY_DEVICE);
            hypre_TFree(diag_a_new,    NALU_HYPRE_MEMORY_DEVICE);

            tmp_j = hypre_TReAlloc_v2(tmp_j, NALU_HYPRE_Int,     diag_nnz_existed + diag_nnz_new, NALU_HYPRE_Int,
                                      nnz_new, NALU_HYPRE_MEMORY_DEVICE);
            tmp_a = hypre_TReAlloc_v2(tmp_a, NALU_HYPRE_Complex, diag_nnz_existed + diag_nnz_new, NALU_HYPRE_Complex,
                                      nnz_new, NALU_HYPRE_MEMORY_DEVICE);

            diag_nnz_new = nnz_new;
            diag_i_new   = tmp_i;
            diag_j_new   = tmp_j;
            diag_a_new   = tmp_a;
         }

         hypre_CSRMatrix *diag               = hypre_CSRMatrixCreate(nrows, ncols, diag_nnz_new);
         hypre_CSRMatrixI(diag)              = hypreDevice_CsrRowIndicesToPtrs(nrows, diag_nnz_new,
                                                                               diag_i_new);
         hypre_CSRMatrixJ(diag)              = diag_j_new;
         hypre_CSRMatrixData(diag)           = diag_a_new;
         hypre_CSRMatrixMemoryLocation(diag) = NALU_HYPRE_MEMORY_DEVICE;

         hypre_TFree(diag_i_new, NALU_HYPRE_MEMORY_DEVICE);

         hypre_CSRMatrixDestroy(hypre_ParCSRMatrixDiag(par_matrix));
         hypre_ParCSRMatrixDiag(par_matrix) = diag;
      }

      if (offd_nnz_new > 0)
      {
         if (offd_nnz_existed)
         {
            /* the existing parcsr should come first and the entries are "add" */
            hypreDevice_CsrRowPtrsToIndices_v2(nrows, offd_nnz_existed,
                                               hypre_CSRMatrixI(hypre_ParCSRMatrixOffd(par_matrix)), offd_i_new);

            /* adjust with the new col_map_offd_map */
#if defined(NALU_HYPRE_USING_SYCL)
            hypreSycl_gather( hypre_CSRMatrixJ(hypre_ParCSRMatrixOffd(par_matrix)),
                              hypre_CSRMatrixJ(hypre_ParCSRMatrixOffd(par_matrix)) + offd_nnz_existed,
                              col_map_offd_map,
                              offd_j_new );
#else
            NALU_HYPRE_THRUST_CALL( gather,
                               hypre_CSRMatrixJ(hypre_ParCSRMatrixOffd(par_matrix)),
                               hypre_CSRMatrixJ(hypre_ParCSRMatrixOffd(par_matrix)) + offd_nnz_existed,
                               col_map_offd_map,
                               offd_j_new );
#endif

            hypre_TMemcpy(offd_a_new, hypre_CSRMatrixData(hypre_ParCSRMatrixOffd(par_matrix)), NALU_HYPRE_Complex,
                          offd_nnz_existed, NALU_HYPRE_MEMORY_DEVICE, NALU_HYPRE_MEMORY_DEVICE);

            hypreDevice_CharFilln(offd_sora_new, offd_nnz_existed, 0);

            hypre_IJMatrixAssembleSortAndReduce2(offd_nnz_existed + offd_nnz_new, offd_i_new, offd_j_new,
                                                 offd_sora_new, offd_a_new,
                                                 &nnz_new, &tmp_i, &tmp_j, &tmp_a, 0);

            hypre_TFree(offd_i_new,    NALU_HYPRE_MEMORY_DEVICE);
            hypre_TFree(offd_j_new,    NALU_HYPRE_MEMORY_DEVICE);
            hypre_TFree(offd_sora_new, NALU_HYPRE_MEMORY_DEVICE);
            hypre_TFree(offd_a_new,    NALU_HYPRE_MEMORY_DEVICE);

            tmp_j = hypre_TReAlloc_v2(tmp_j, NALU_HYPRE_Int,     offd_nnz_existed + offd_nnz_new, NALU_HYPRE_Int,
                                      nnz_new, NALU_HYPRE_MEMORY_DEVICE);
            tmp_a = hypre_TReAlloc_v2(tmp_a, NALU_HYPRE_Complex, offd_nnz_existed + offd_nnz_new, NALU_HYPRE_Complex,
                                      nnz_new, NALU_HYPRE_MEMORY_DEVICE);

            offd_nnz_new = nnz_new;
            offd_i_new   = tmp_i;
            offd_j_new   = tmp_j;
            offd_a_new   = tmp_a;
         }

         hypre_CSRMatrix *offd               = hypre_CSRMatrixCreate(nrows, num_cols_offd_new, offd_nnz_new);
         hypre_CSRMatrixI(offd)              = hypreDevice_CsrRowIndicesToPtrs(nrows, offd_nnz_new,
                                                                               offd_i_new);
         hypre_CSRMatrixJ(offd)              = offd_j_new;
         hypre_CSRMatrixData(offd)           = offd_a_new;
         hypre_CSRMatrixMemoryLocation(offd) = NALU_HYPRE_MEMORY_DEVICE;

         hypre_TFree(offd_i_new, NALU_HYPRE_MEMORY_DEVICE);

         hypre_CSRMatrixDestroy(hypre_ParCSRMatrixOffd(par_matrix));
         hypre_ParCSRMatrixOffd(par_matrix) = offd;

         hypre_TFree(hypre_ParCSRMatrixDeviceColMapOffd(par_matrix), NALU_HYPRE_MEMORY_DEVICE);
         hypre_ParCSRMatrixDeviceColMapOffd(par_matrix) = col_map_offd_new;

         hypre_TFree(hypre_ParCSRMatrixColMapOffd(par_matrix), NALU_HYPRE_MEMORY_HOST);
         hypre_ParCSRMatrixColMapOffd(par_matrix) = hypre_TAlloc(NALU_HYPRE_BigInt, num_cols_offd_new,
                                                                 NALU_HYPRE_MEMORY_HOST);
         hypre_TMemcpy(hypre_ParCSRMatrixColMapOffd(par_matrix), col_map_offd_new, NALU_HYPRE_BigInt,
                       num_cols_offd_new,
                       NALU_HYPRE_MEMORY_HOST, NALU_HYPRE_MEMORY_DEVICE);

         col_map_offd_new = NULL;
      }

      hypre_TFree(col_map_offd_map, NALU_HYPRE_MEMORY_DEVICE);
      hypre_TFree(col_map_offd_new, NALU_HYPRE_MEMORY_DEVICE);
   } /* if (nelms) */

   hypre_IJMatrixAssembleFlag(matrix) = 1;
   hypre_AuxParCSRMatrixDestroy(aux_matrix);
   hypre_IJMatrixTranslator(matrix) = NULL;

   return hypre_error_flag;
}

NALU_HYPRE_Int
hypre_IJMatrixSetConstantValuesParCSRDevice( hypre_IJMatrix *matrix,
                                             NALU_HYPRE_Complex   value )
{
   hypre_ParCSRMatrix *par_matrix = (hypre_ParCSRMatrix *) hypre_IJMatrixObject( matrix );
   hypre_CSRMatrix    *diag       = hypre_ParCSRMatrixDiag(par_matrix);
   hypre_CSRMatrix    *offd       = hypre_ParCSRMatrixOffd(par_matrix);
   NALU_HYPRE_Complex      *diag_data  = hypre_CSRMatrixData(diag);
   NALU_HYPRE_Complex      *offd_data  = hypre_CSRMatrixData(offd);
   NALU_HYPRE_Int           nnz_diag   = hypre_CSRMatrixNumNonzeros(diag);
   NALU_HYPRE_Int           nnz_offd   = hypre_CSRMatrixNumNonzeros(offd);

   hypreDevice_ComplexFilln( diag_data, nnz_diag, value );
   hypreDevice_ComplexFilln( offd_data, nnz_offd, value );

   return hypre_error_flag;
}

#endif
