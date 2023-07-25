/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * IJVector_ParCSR interface
 *
 *****************************************************************************/

#include "_nalu_hypre_onedpl.hpp"
#include "_nalu_hypre_IJ_mv.h"
#include "_nalu_hypre_utilities.hpp"

#if defined(NALU_HYPRE_USING_GPU)

/*--------------------------------------------------------------------
 * nalu_hypre_IJVectorAssembleFunctor
 *--------------------------------------------------------------------*/

#if defined(NALU_HYPRE_USING_SYCL)
template<typename T1, typename T2>
struct nalu_hypre_IJVectorAssembleFunctor
{
   typedef std::tuple<T1, T2> Tuple;

   __device__ Tuple operator() (const Tuple& x, const Tuple& y ) const
   {
      return std::make_tuple( nalu_hypre_max(std::get<0>(x), std::get<0>(y)),
                              std::get<1>(x) + std::get<1>(y) );
   }
};
#else
template<typename T1, typename T2>
struct nalu_hypre_IJVectorAssembleFunctor : public
   thrust::binary_function< thrust::tuple<T1, T2>, thrust::tuple<T1, T2>, thrust::tuple<T1, T2> >
{
   typedef thrust::tuple<T1, T2> Tuple;

   __device__ Tuple operator() (const Tuple& x, const Tuple& y )
   {
      return thrust::make_tuple( nalu_hypre_max(thrust::get<0>(x), thrust::get<0>(y)),
                                 thrust::get<1>(x) + thrust::get<1>(y) );
   }
};
#endif

/*--------------------------------------------------------------------
 * nalu_hypre_IJVectorAssembleSortAndReduce1
 *
 * helper routine used in nalu_hypre_IJVectorAssembleParCSRDevice:
 *   1. sort (X0, A0) with key I0
 *   2. for each segment in I0, zero out in A0 all before the last `set'
 *   3. reduce A0 [with sum] and reduce X0 [with max]
 *
 * N0: input size; N1: size after reduction (<= N0)
 * Note: (I1, X1, A1) are not resized to N1 but have size N0
 *--------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_IJVectorAssembleSortAndReduce1( NALU_HYPRE_Int       N0,
                                      NALU_HYPRE_BigInt   *I0,
                                      char           *X0,
                                      NALU_HYPRE_Complex  *A0,
                                      NALU_HYPRE_Int      *N1,
                                      NALU_HYPRE_BigInt  **I1,
                                      char          **X1,
                                      NALU_HYPRE_Complex **A1 )
{
#if defined(NALU_HYPRE_USING_SYCL)
   auto zipped_begin = oneapi::dpl::make_zip_iterator(I0, X0, A0);
   NALU_HYPRE_ONEDPL_CALL( std::stable_sort,
                      zipped_begin, zipped_begin + N0,
   [](auto lhs, auto rhs) { return std::get<0>(lhs) < std::get<0>(rhs); } );
#else
   NALU_HYPRE_THRUST_CALL( stable_sort_by_key,
                      I0,
                      I0 + N0,
                      thrust::make_zip_iterator(thrust::make_tuple(X0, A0)) );
#endif

   NALU_HYPRE_BigInt  *I = nalu_hypre_TAlloc(NALU_HYPRE_BigInt,  N0, NALU_HYPRE_MEMORY_DEVICE);
   char          *X = nalu_hypre_TAlloc(char,          N0, NALU_HYPRE_MEMORY_DEVICE);
   NALU_HYPRE_Complex *A = nalu_hypre_TAlloc(NALU_HYPRE_Complex, N0, NALU_HYPRE_MEMORY_DEVICE);

   /* output X: 0: keep, 1: zero-out */
#if defined(NALU_HYPRE_USING_SYCL)
   /* WM: todo - oneDPL currently does not have a reverse iterator */
   /*     should be able to do this with a reverse operation defined in a struct */
   /*     instead of explicitly allocating and generating the reverse_perm, */
   /*     but I can't get that to work for some reason */
   NALU_HYPRE_Int *reverse_perm = nalu_hypre_TAlloc(NALU_HYPRE_Int, N0, NALU_HYPRE_MEMORY_DEVICE);
   NALU_HYPRE_ONEDPL_CALL( std::transform,
                      oneapi::dpl::counting_iterator<NALU_HYPRE_Int>(0),
                      oneapi::dpl::counting_iterator<NALU_HYPRE_Int>(N0),
                      reverse_perm,
   [N0] (auto i) { return N0 - i - 1; });

   auto I0_reversed = oneapi::dpl::make_permutation_iterator(I0, reverse_perm);
   auto X0_reversed = oneapi::dpl::make_permutation_iterator(X0, reverse_perm);
   auto X_reversed = oneapi::dpl::make_permutation_iterator(X, reverse_perm);

   NALU_HYPRE_ONEDPL_CALL( oneapi::dpl::exclusive_scan_by_segment,
                      I0_reversed,      /* key begin */
                      I0_reversed + N0, /* key end */
                      X0_reversed,      /* input value begin */
                      X_reversed,       /* output value begin */
                      char(0),          /* init */
                      std::equal_to<NALU_HYPRE_BigInt>(),
                      oneapi::dpl::maximum<char>() );

   nalu_hypre_TFree(reverse_perm, NALU_HYPRE_MEMORY_DEVICE);

   hypreSycl_transform_if(A0,
                          A0 + N0,
                          X,
                          A0,
   [] (const auto & x) {return 0.0;},
   [] (const auto & x) {return x;} );

   auto new_end = NALU_HYPRE_ONEDPL_CALL( oneapi::dpl::reduce_by_segment,
                                     I0,                                                         /* keys_first */
                                     I0 + N0,                                                    /* keys_last */
                                     oneapi::dpl::make_zip_iterator(X0, A0),                     /* values_first */
                                     I,                                                          /* keys_output */
                                     oneapi::dpl::make_zip_iterator(X, A),                       /* values_output */
                                     std::equal_to<NALU_HYPRE_BigInt>(),                              /* binary_pred */
                                     nalu_hypre_IJVectorAssembleFunctor<char, NALU_HYPRE_Complex>()        /* binary_op */);
#else
   NALU_HYPRE_THRUST_CALL(
      exclusive_scan_by_key,
      make_reverse_iterator(thrust::device_pointer_cast<NALU_HYPRE_BigInt>(I0) + N0), /* key begin */
      make_reverse_iterator(thrust::device_pointer_cast<NALU_HYPRE_BigInt>(I0)),      /* key end */
      make_reverse_iterator(thrust::device_pointer_cast<char>(X0) + N0),         /* input value begin */
      make_reverse_iterator(thrust::device_pointer_cast<char>(X) + N0),          /* output value begin */
      char(0),                                                                   /* init */
      thrust::equal_to<NALU_HYPRE_BigInt>(),
      thrust::maximum<char>() );

   NALU_HYPRE_THRUST_CALL(replace_if, A0, A0 + N0, X, thrust::identity<char>(), 0.0);

   auto new_end = NALU_HYPRE_THRUST_CALL(
                     reduce_by_key,
                     I0,                                                              /* keys_first */
                     I0 + N0,                                                         /* keys_last */
                     thrust::make_zip_iterator(thrust::make_tuple(X0,      A0     )), /* values_first */
                     I,                                                               /* keys_output */
                     thrust::make_zip_iterator(thrust::make_tuple(X,       A      )), /* values_output */
                     thrust::equal_to<NALU_HYPRE_BigInt>(),                                /* binary_pred */
                     nalu_hypre_IJVectorAssembleFunctor<char, NALU_HYPRE_Complex>()             /* binary_op */);
#endif

   *N1 = new_end.first - I;
   *I1 = I;
   *X1 = X;
   *A1 = A;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------
 * nalu_hypre_IJVectorAssembleSortAndReduce3
 *--------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_IJVectorAssembleSortAndReduce3( NALU_HYPRE_Int      N0,
                                      NALU_HYPRE_BigInt  *I0,
                                      char          *X0,
                                      NALU_HYPRE_Complex *A0,
                                      NALU_HYPRE_Int     *N1 )
{
#if defined(NALU_HYPRE_USING_SYCL)
   auto zipped_begin = oneapi::dpl::make_zip_iterator(I0, X0, A0);
   NALU_HYPRE_ONEDPL_CALL( std::stable_sort,
                      zipped_begin, zipped_begin + N0,
   [](auto lhs, auto rhs) { return std::get<0>(lhs) < std::get<0>(rhs); } );
#else
   NALU_HYPRE_THRUST_CALL( stable_sort_by_key,
                      I0,
                      I0 + N0,
                      thrust::make_zip_iterator(thrust::make_tuple(X0, A0)) );
#endif

   NALU_HYPRE_BigInt  *I = nalu_hypre_TAlloc(NALU_HYPRE_BigInt,  N0, NALU_HYPRE_MEMORY_DEVICE);
   NALU_HYPRE_Complex *A = nalu_hypre_TAlloc(NALU_HYPRE_Complex, N0, NALU_HYPRE_MEMORY_DEVICE);

   /* output in X0: 0: keep, 1: zero-out */
#if defined(NALU_HYPRE_USING_SYCL)
   /* WM: todo - oneDPL currently does not have a reverse iterator */
   /*     should be able to do this with a reverse operation defined in a struct */
   /*     instead of explicitly allocating and generating the reverse_perm, */
   /*     but I can't get that to work for some reason */
   NALU_HYPRE_Int *reverse_perm = nalu_hypre_TAlloc(NALU_HYPRE_Int, N0, NALU_HYPRE_MEMORY_DEVICE);
   NALU_HYPRE_ONEDPL_CALL( std::transform,
                      oneapi::dpl::counting_iterator<NALU_HYPRE_Int>(0),
                      oneapi::dpl::counting_iterator<NALU_HYPRE_Int>(N0),
                      reverse_perm,
   [N0] (auto i) { return N0 - i - 1; });

   auto I0_reversed = oneapi::dpl::make_permutation_iterator(I0, reverse_perm);
   auto X0_reversed = oneapi::dpl::make_permutation_iterator(X0, reverse_perm);

   NALU_HYPRE_ONEDPL_CALL( oneapi::dpl::inclusive_scan_by_segment,
                      I0_reversed,      /* key begin */
                      I0_reversed + N0, /* key end */
                      X0_reversed,      /* input value begin */
                      X0_reversed,      /* output value begin */
                      std::equal_to<NALU_HYPRE_BigInt>(),
                      oneapi::dpl::maximum<char>() );

   nalu_hypre_TFree(reverse_perm, NALU_HYPRE_MEMORY_DEVICE);

   hypreSycl_transform_if(A0,
                          A0 + N0,
                          X0,
                          A0,
   [] (const auto & x) {return 0.0;},
   [] (const auto & x) {return x;} );

   /* WM: todo - why don't I use the NALU_HYPRE_ONEDPL_CALL macro here? Compile issue? */
   auto new_end = oneapi::dpl::reduce_by_segment(
                     oneapi::dpl::execution::make_device_policy<class devutils>(*nalu_hypre_HandleComputeStream(
                                                                                   nalu_hypre_handle())),
                     I0,      /* keys_first */
                     I0 + N0, /* keys_last */
                     A0,      /* values_first */
                     I,       /* keys_output */
                     A        /* values_output */);
#else
   NALU_HYPRE_THRUST_CALL(
      inclusive_scan_by_key,
      make_reverse_iterator(thrust::device_pointer_cast<NALU_HYPRE_BigInt>(I0) + N0), /* key begin */
      make_reverse_iterator(thrust::device_pointer_cast<NALU_HYPRE_BigInt>(I0)),    /* key end */
      make_reverse_iterator(thrust::device_pointer_cast<char>(X0) + N0),       /* input value begin */
      make_reverse_iterator(thrust::device_pointer_cast<char>(X0) + N0),       /* output value begin */
      thrust::equal_to<NALU_HYPRE_BigInt>(),
      thrust::maximum<char>() );

   NALU_HYPRE_THRUST_CALL(replace_if, A0, A0 + N0, X0, thrust::identity<char>(), 0.0);

   auto new_end = NALU_HYPRE_THRUST_CALL(
                     reduce_by_key,
                     I0,      /* keys_first */
                     I0 + N0, /* keys_last */
                     A0,      /* values_first */
                     I,       /* keys_output */
                     A        /* values_output */);
#endif

   NALU_HYPRE_Int Nt = new_end.second - A;

   nalu_hypre_assert(Nt <= N0);

   /* remove numerical zeros */
#if defined(NALU_HYPRE_USING_SYCL)
   auto new_end2 = hypreSycl_copy_if( oneapi::dpl::make_zip_iterator(I, A),
                                      oneapi::dpl::make_zip_iterator(I, A) + Nt,
                                      A,
                                      oneapi::dpl::make_zip_iterator(I0, A0),
   [] (const auto & x) {return x;} );

   *N1 = std::get<0>(new_end2.base()) - I0;
#else
   auto new_end2 = NALU_HYPRE_THRUST_CALL( copy_if,
                                      thrust::make_zip_iterator(thrust::make_tuple(I, A)),
                                      thrust::make_zip_iterator(thrust::make_tuple(I, A)) + Nt,
                                      A,
                                      thrust::make_zip_iterator(thrust::make_tuple(I0, A0)),
                                      thrust::identity<NALU_HYPRE_Complex>() );

   *N1 = thrust::get<0>(new_end2.get_iterator_tuple()) - I0;
#endif

   nalu_hypre_assert(*N1 <= Nt);

   nalu_hypre_TFree(I, NALU_HYPRE_MEMORY_DEVICE);
   nalu_hypre_TFree(A, NALU_HYPRE_MEMORY_DEVICE);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------
 * hypreGPUKernel_IJVectorAssemblePar
 *
 * y[map[i]-offset] = x[i] or y[map[i]] += x[i] depending on SorA,
 * same index cannot appear more than once in map
 *--------------------------------------------------------------------*/

__global__ void
hypreGPUKernel_IJVectorAssemblePar( nalu_hypre_DeviceItem &item,
                                    NALU_HYPRE_Int         n,
                                    NALU_HYPRE_Complex    *x,
                                    NALU_HYPRE_BigInt     *map,
                                    NALU_HYPRE_BigInt      offset,
                                    char             *SorA,
                                    NALU_HYPRE_Complex    *y )
{
   NALU_HYPRE_Int i = nalu_hypre_gpu_get_grid_thread_id<1, 1>(item);

   if (i >= n)
   {
      return;
   }

   if (SorA[i])
   {
      y[map[i] - offset] = x[i];
   }
   else
   {
      y[map[i] - offset] += x[i];
   }
}

/*--------------------------------------------------------------------
 * nalu_hypre_IJVectorSetAddValuesParDevice
 *--------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_IJVectorSetAddValuesParDevice(nalu_hypre_IJVector       *vector,
                                    NALU_HYPRE_Int             num_values,
                                    const NALU_HYPRE_BigInt   *indices,
                                    const NALU_HYPRE_Complex  *values,
                                    const char           *action)
{
   NALU_HYPRE_BigInt    *IJpartitioning = nalu_hypre_IJVectorPartitioning(vector);
   NALU_HYPRE_BigInt     vec_start      = IJpartitioning[0];

   nalu_hypre_ParVector *par_vector     = (nalu_hypre_ParVector*) nalu_hypre_IJVectorObject(vector);
   nalu_hypre_Vector    *local_vector   = nalu_hypre_ParVectorLocalVector(par_vector);
   NALU_HYPRE_Int        size           = nalu_hypre_VectorSize(local_vector);
   NALU_HYPRE_Int        num_vectors    = nalu_hypre_VectorNumVectors(local_vector);
   NALU_HYPRE_Int        component      = nalu_hypre_VectorComponent(local_vector);
   NALU_HYPRE_Int        vecstride      = nalu_hypre_VectorVectorStride(local_vector);

   const char       SorA           = action[0] == 's' ? 1 : 0;

   if (num_values <= 0)
   {
      return nalu_hypre_error_flag;
   }

   /* this is a special use to set/add local values */
   if (!indices)
   {
      NALU_HYPRE_Int     num_values2 = nalu_hypre_min(size, num_values);
      NALU_HYPRE_BigInt *indices2    = nalu_hypre_TAlloc(NALU_HYPRE_BigInt, num_values2, NALU_HYPRE_MEMORY_DEVICE);

#if defined(NALU_HYPRE_USING_SYCL)
      hypreSycl_sequence(indices2, indices2 + num_values2, vec_start);
#else
      NALU_HYPRE_THRUST_CALL(sequence, indices2, indices2 + num_values2, vec_start);
#endif

      nalu_hypre_IJVectorSetAddValuesParDevice(vector, num_values2, indices2, values, action);

      nalu_hypre_TFree(indices2, NALU_HYPRE_MEMORY_DEVICE);

      return nalu_hypre_error_flag;
   }

   nalu_hypre_AuxParVector *aux_vector = (nalu_hypre_AuxParVector*) nalu_hypre_IJVectorTranslator(vector);

   if (!aux_vector)
   {
      nalu_hypre_AuxParVectorCreate(&aux_vector);
      nalu_hypre_AuxParVectorInitialize_v2(aux_vector, NALU_HYPRE_MEMORY_DEVICE);
      nalu_hypre_IJVectorTranslator(vector) = aux_vector;
   }

   NALU_HYPRE_Int      stack_elmts_max      = nalu_hypre_AuxParVectorMaxStackElmts(aux_vector);
   NALU_HYPRE_Int      stack_elmts_current  = nalu_hypre_AuxParVectorCurrentStackElmts(aux_vector);
   NALU_HYPRE_Int      stack_elmts_required = stack_elmts_current + num_values;
   NALU_HYPRE_BigInt  *stack_i              = nalu_hypre_AuxParVectorStackI(aux_vector);
   NALU_HYPRE_BigInt  *stack_voff           = nalu_hypre_AuxParVectorStackVoff(aux_vector);
   NALU_HYPRE_Complex *stack_data           = nalu_hypre_AuxParVectorStackData(aux_vector);
   char          *stack_sora           = nalu_hypre_AuxParVectorStackSorA(aux_vector);

   if (stack_elmts_max < stack_elmts_required)
   {
      NALU_HYPRE_Int stack_elmts_max_new = size * nalu_hypre_AuxParVectorInitAllocFactor(aux_vector);

      if (nalu_hypre_AuxParVectorUsrOffProcElmts(aux_vector) >= 0)
      {
         stack_elmts_max_new += nalu_hypre_AuxParVectorUsrOffProcElmts(aux_vector);
      }
      stack_elmts_max_new = nalu_hypre_max(stack_elmts_max * nalu_hypre_AuxParVectorGrowFactor(aux_vector),
                                      stack_elmts_max_new);
      stack_elmts_max_new = nalu_hypre_max(stack_elmts_required, stack_elmts_max_new);

      stack_i    = nalu_hypre_TReAlloc_v2(stack_i,     NALU_HYPRE_BigInt, stack_elmts_max,  NALU_HYPRE_BigInt,
                                     stack_elmts_max_new, NALU_HYPRE_MEMORY_DEVICE);
      stack_data = nalu_hypre_TReAlloc_v2(stack_data, NALU_HYPRE_Complex, stack_elmts_max, NALU_HYPRE_Complex,
                                     stack_elmts_max_new, NALU_HYPRE_MEMORY_DEVICE);
      stack_sora = nalu_hypre_TReAlloc_v2(stack_sora,          char, stack_elmts_max,          char,
                                     stack_elmts_max_new, NALU_HYPRE_MEMORY_DEVICE);

      if (num_vectors > 1)
      {
         stack_voff = nalu_hypre_TReAlloc_v2(stack_voff, NALU_HYPRE_BigInt, stack_elmts_max, NALU_HYPRE_BigInt,
                                        stack_elmts_max_new, NALU_HYPRE_MEMORY_DEVICE);
      }

      nalu_hypre_AuxParVectorStackI(aux_vector)        = stack_i;
      nalu_hypre_AuxParVectorStackVoff(aux_vector)     = stack_voff;
      nalu_hypre_AuxParVectorStackData(aux_vector)     = stack_data;
      nalu_hypre_AuxParVectorStackSorA(aux_vector)     = stack_sora;
      nalu_hypre_AuxParVectorMaxStackElmts(aux_vector) = stack_elmts_max_new;
   }

   hypreDevice_CharFilln(stack_sora + stack_elmts_current, num_values, SorA);
   if (num_vectors > 1)
   {
      hypreDevice_BigIntFilln(stack_voff + stack_elmts_current, num_values,
                              (NALU_HYPRE_BigInt) component * vecstride);
   }

   nalu_hypre_TMemcpy(stack_i    + stack_elmts_current, indices, NALU_HYPRE_BigInt,  num_values,
                 NALU_HYPRE_MEMORY_DEVICE, NALU_HYPRE_MEMORY_DEVICE);
   nalu_hypre_TMemcpy(stack_data + stack_elmts_current, values,  NALU_HYPRE_Complex, num_values,
                 NALU_HYPRE_MEMORY_DEVICE, NALU_HYPRE_MEMORY_DEVICE);

   nalu_hypre_AuxParVectorCurrentStackElmts(aux_vector) += num_values;

   return nalu_hypre_error_flag;
}

/******************************************************************************
 * nalu_hypre_IJVectorAssembleParDevice
 *****************************************************************************/

NALU_HYPRE_Int
nalu_hypre_IJVectorAssembleParDevice(nalu_hypre_IJVector *vector)
{
   MPI_Comm            comm           = nalu_hypre_IJVectorComm(vector);
   nalu_hypre_ParVector    *par_vector     = (nalu_hypre_ParVector*) nalu_hypre_IJVectorObject(vector);
   nalu_hypre_AuxParVector *aux_vector     = (nalu_hypre_AuxParVector*) nalu_hypre_IJVectorTranslator(vector);
   NALU_HYPRE_BigInt       *IJpartitioning = nalu_hypre_IJVectorPartitioning(vector);
   NALU_HYPRE_BigInt        vec_start      = IJpartitioning[0];
   NALU_HYPRE_BigInt        vec_stop       = IJpartitioning[1] - 1;

   nalu_hypre_Vector       *local_vector   = nalu_hypre_ParVectorLocalVector(par_vector);
   NALU_HYPRE_Int           num_vectors    = nalu_hypre_VectorNumVectors(local_vector);
   NALU_HYPRE_Complex      *data           = nalu_hypre_VectorData(local_vector);

   if (!aux_vector)
   {
      return nalu_hypre_error_flag;
   }

   if (!par_vector)
   {
      return nalu_hypre_error_flag;
   }

   NALU_HYPRE_Int      nelms      = nalu_hypre_AuxParVectorCurrentStackElmts(aux_vector);
   NALU_HYPRE_BigInt  *stack_i    = nalu_hypre_AuxParVectorStackI(aux_vector);
   NALU_HYPRE_BigInt  *stack_voff = nalu_hypre_AuxParVectorStackVoff(aux_vector);
   NALU_HYPRE_Complex *stack_data = nalu_hypre_AuxParVectorStackData(aux_vector);
   char          *stack_sora = nalu_hypre_AuxParVectorStackSorA(aux_vector);

   in_range<NALU_HYPRE_BigInt> pred(vec_start, vec_stop);
#if defined(NALU_HYPRE_USING_SYCL)
   NALU_HYPRE_Int nelms_on = NALU_HYPRE_ONEDPL_CALL(std::count_if, stack_i, stack_i + nelms, pred);
#else
   NALU_HYPRE_Int nelms_on = NALU_HYPRE_THRUST_CALL(count_if, stack_i, stack_i + nelms, pred);
#endif
   NALU_HYPRE_Int nelms_off = nelms - nelms_on;
   NALU_HYPRE_Int nelms_off_max;
   nalu_hypre_MPI_Allreduce(&nelms_off, &nelms_off_max, 1, NALU_HYPRE_MPI_INT, nalu_hypre_MPI_MAX, comm);

   /* communicate for aux off-proc and add to remote aux on-proc */
   if (nelms_off_max)
   {
      NALU_HYPRE_Int      new_nnz       = 0;
      NALU_HYPRE_BigInt  *off_proc_i    = NULL;
      NALU_HYPRE_Complex *off_proc_data = NULL;

      if (num_vectors > 1)
      {
         nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC,
                           "Off proc IJVectorAssembleParDevice not implemented for multivectors!\n");
         return nalu_hypre_error_flag;
      }

      if (nelms_off)
      {
         /* copy off-proc entries out of stack and remove from stack */
         off_proc_i          = nalu_hypre_TAlloc(NALU_HYPRE_BigInt,  nelms_off, NALU_HYPRE_MEMORY_DEVICE);
         off_proc_data       = nalu_hypre_TAlloc(NALU_HYPRE_Complex, nelms_off, NALU_HYPRE_MEMORY_DEVICE);
         char *off_proc_sora = nalu_hypre_TAlloc(char,          nelms_off, NALU_HYPRE_MEMORY_DEVICE);
         char *is_on_proc    = nalu_hypre_TAlloc(char,          nelms,     NALU_HYPRE_MEMORY_DEVICE);

#if defined(NALU_HYPRE_USING_SYCL)
         NALU_HYPRE_ONEDPL_CALL(std::transform, stack_i, stack_i + nelms, is_on_proc, pred);
         auto zip_in = oneapi::dpl::make_zip_iterator(stack_i, stack_data, stack_sora);
         auto zip_out = oneapi::dpl::make_zip_iterator(off_proc_i, off_proc_data, off_proc_sora);
         auto new_end1 = hypreSycl_copy_if( zip_in,  /* first */
                                            zip_in + nelms, /* last */
                                            is_on_proc, /* stencil */
                                            zip_out, /* result */
         [] (const auto & x) {return !x;} );

         nalu_hypre_assert(std::get<0>(new_end1.base()) - off_proc_i == nelms_off);

         /* remove off-proc entries from stack */
         auto new_end2 = hypreSycl_remove_if( zip_in,         /* first */
                                              zip_in + nelms, /* last */
                                              is_on_proc,     /* stencil */
         [] (const auto & x) {return !x;} );

         nalu_hypre_assert(std::get<0>(new_end2.base()) - stack_i == nelms_on);
#else
         NALU_HYPRE_THRUST_CALL(transform, stack_i, stack_i + nelms, is_on_proc, pred);

         auto new_end1 = NALU_HYPRE_THRUST_CALL(
                            copy_if,
                            thrust::make_zip_iterator(thrust::make_tuple(stack_i,         stack_data,
                                                                         stack_sora        )),  /* first */
                            thrust::make_zip_iterator(thrust::make_tuple(stack_i + nelms, stack_data + nelms,
                                                                         stack_sora + nelms)),  /* last */
                            is_on_proc,                                                         /* stencil */
                            thrust::make_zip_iterator(thrust::make_tuple(off_proc_i,      off_proc_data,
                                                                         off_proc_sora)),       /* result */
                            thrust::not1(thrust::identity<char>()) );

         nalu_hypre_assert(thrust::get<0>(new_end1.get_iterator_tuple()) - off_proc_i == nelms_off);

         /* remove off-proc entries from stack */
         auto new_end2 = NALU_HYPRE_THRUST_CALL(
                            remove_if,
                            thrust::make_zip_iterator(thrust::make_tuple(stack_i,         stack_data,
                                                                         stack_sora        )),  /* first */
                            thrust::make_zip_iterator(thrust::make_tuple(stack_i + nelms, stack_data + nelms,
                                                                         stack_sora + nelms)),  /* last */
                            is_on_proc,                                                         /* stencil */
                            thrust::not1(thrust::identity<char>()) );

         nalu_hypre_assert(thrust::get<0>(new_end2.get_iterator_tuple()) - stack_i == nelms_on);
#endif

         nalu_hypre_AuxParVectorCurrentStackElmts(aux_vector) = nelms_on;

         nalu_hypre_TFree(is_on_proc, NALU_HYPRE_MEMORY_DEVICE);

         /* sort and reduce */
         nalu_hypre_IJVectorAssembleSortAndReduce3(nelms_off, off_proc_i, off_proc_sora, off_proc_data, &new_nnz);

         nalu_hypre_TFree(off_proc_sora, NALU_HYPRE_MEMORY_DEVICE);
      }

      /* send off_proc_i/data to remote processes and the receivers call addtovalues */
      nalu_hypre_IJVectorAssembleOffProcValsPar(vector, -1, new_nnz, NALU_HYPRE_MEMORY_DEVICE,
                                           off_proc_i, off_proc_data);

      nalu_hypre_TFree(off_proc_i,    NALU_HYPRE_MEMORY_DEVICE);
      nalu_hypre_TFree(off_proc_data, NALU_HYPRE_MEMORY_DEVICE);
   }

   /* Note: the stack might have been changed in nalu_hypre_IJVectorAssembleOffProcValsPar,
    * so must get the size and the pointers again */
   nelms      = nalu_hypre_AuxParVectorCurrentStackElmts(aux_vector);
   stack_i    = nalu_hypre_AuxParVectorStackI(aux_vector);
   stack_voff = nalu_hypre_AuxParVectorStackVoff(aux_vector);
   stack_data = nalu_hypre_AuxParVectorStackData(aux_vector);
   stack_sora = nalu_hypre_AuxParVectorStackSorA(aux_vector);

#ifdef NALU_HYPRE_DEBUG
   /* the stack should only have on-proc elements now */
#if defined(NALU_HYPRE_USING_SYCL)
   NALU_HYPRE_Int tmp = NALU_HYPRE_ONEDPL_CALL(std::count_if, stack_i, stack_i + nelms, pred);
#else
   NALU_HYPRE_Int tmp = NALU_HYPRE_THRUST_CALL(count_if, stack_i, stack_i + nelms, pred);
#endif
   nalu_hypre_assert(nelms == tmp);
#endif

   if (nelms)
   {
      NALU_HYPRE_Int      new_nnz;
      NALU_HYPRE_BigInt  *new_i;
      NALU_HYPRE_Complex *new_data;
      char          *new_sora;

      /* Shift stack_i with multivector component offsets */
      if (num_vectors > 1)
      {
         hypreDevice_BigIntAxpyn(stack_voff, nelms, stack_i, stack_i, 1);
      }

      /* sort and reduce */
      nalu_hypre_IJVectorAssembleSortAndReduce1(nelms, stack_i, stack_sora, stack_data,
                                           &new_nnz, &new_i, &new_sora, &new_data);

      /* set/add to local vector */
      dim3 bDim = nalu_hypre_GetDefaultDeviceBlockDimension();
      dim3 gDim = nalu_hypre_GetDefaultDeviceGridDimension(new_nnz, "thread", bDim);
      NALU_HYPRE_GPU_LAUNCH( hypreGPUKernel_IJVectorAssemblePar, gDim, bDim,
                        new_nnz, new_data, new_i,
                        vec_start, new_sora,
                        data );

      nalu_hypre_TFree(new_i,    NALU_HYPRE_MEMORY_DEVICE);
      nalu_hypre_TFree(new_data, NALU_HYPRE_MEMORY_DEVICE);
      nalu_hypre_TFree(new_sora, NALU_HYPRE_MEMORY_DEVICE);
   }

   nalu_hypre_AuxParVectorDestroy(aux_vector);
   nalu_hypre_IJVectorTranslator(vector) = NULL;

   return nalu_hypre_error_flag;
}

__global__ void
hypreCUDAKernel_IJVectorUpdateValues( nalu_hypre_DeviceItem    &item,
                                      NALU_HYPRE_Int            n,
                                      const NALU_HYPRE_Complex *x,
                                      const NALU_HYPRE_BigInt  *indices,
                                      NALU_HYPRE_BigInt         start,
                                      NALU_HYPRE_BigInt         stop,
                                      NALU_HYPRE_Int            action,
                                      NALU_HYPRE_Complex       *y )
{
   NALU_HYPRE_Int i = nalu_hypre_gpu_get_grid_thread_id<1, 1>(item);

   if (i >= n)
   {
      return;
   }

   NALU_HYPRE_Int j;

   if (indices)
   {
      j = (NALU_HYPRE_Int) (read_only_load(&indices[i]) - start);
   }
   else
   {
      j = i;
   }

   if (j < 0 || j > (NALU_HYPRE_Int) (stop - start))
   {
      return;
   }

   if (action)
   {
      y[j] = x[i];
   }
   else
   {
      y[j] += x[i];
   }
}

NALU_HYPRE_Int
nalu_hypre_IJVectorUpdateValuesDevice( nalu_hypre_IJVector      *vector,
                                  NALU_HYPRE_Int            num_values,
                                  const NALU_HYPRE_BigInt  *indices,
                                  const NALU_HYPRE_Complex *values,
                                  NALU_HYPRE_Int            action)
{
   NALU_HYPRE_BigInt *IJpartitioning = nalu_hypre_IJVectorPartitioning(vector);
   NALU_HYPRE_BigInt  vec_start = IJpartitioning[0];
   NALU_HYPRE_BigInt  vec_stop  = IJpartitioning[1] - 1;

   if (!indices)
   {
      num_values = vec_stop - vec_start + 1;
   }

   if (num_values <= 0)
   {
      return nalu_hypre_error_flag;
   }

   /* set/add to local vector */
   dim3 bDim = nalu_hypre_GetDefaultDeviceBlockDimension();
   dim3 gDim = nalu_hypre_GetDefaultDeviceGridDimension(num_values, "thread", bDim);

   nalu_hypre_ParVector *par_vector = (nalu_hypre_ParVector*) nalu_hypre_IJVectorObject(vector);

   NALU_HYPRE_GPU_LAUNCH( hypreCUDAKernel_IJVectorUpdateValues,
                     gDim, bDim,
                     num_values, values, indices,
                     vec_start, vec_stop, action,
                     nalu_hypre_VectorData(nalu_hypre_ParVectorLocalVector(par_vector)) );

   return nalu_hypre_error_flag;
}

#endif

