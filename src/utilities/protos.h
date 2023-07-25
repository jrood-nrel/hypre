/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/* amg_linklist.c */
void nalu_hypre_dispose_elt ( nalu_hypre_LinkList element_ptr );
void nalu_hypre_remove_point ( nalu_hypre_LinkList *LoL_head_ptr, nalu_hypre_LinkList *LoL_tail_ptr,
                          NALU_HYPRE_Int measure, NALU_HYPRE_Int index, NALU_HYPRE_Int *lists, NALU_HYPRE_Int *where );
nalu_hypre_LinkList nalu_hypre_create_elt ( NALU_HYPRE_Int Item );
void nalu_hypre_enter_on_lists ( nalu_hypre_LinkList *LoL_head_ptr, nalu_hypre_LinkList *LoL_tail_ptr,
                            NALU_HYPRE_Int measure, NALU_HYPRE_Int index, NALU_HYPRE_Int *lists, NALU_HYPRE_Int *where );

/* binsearch.c */
NALU_HYPRE_Int nalu_hypre_BinarySearch ( NALU_HYPRE_Int *list, NALU_HYPRE_Int value, NALU_HYPRE_Int list_length );
NALU_HYPRE_Int nalu_hypre_BigBinarySearch ( NALU_HYPRE_BigInt *list, NALU_HYPRE_BigInt value, NALU_HYPRE_Int list_length );
NALU_HYPRE_Int nalu_hypre_BinarySearch2 ( NALU_HYPRE_Int *list, NALU_HYPRE_Int value, NALU_HYPRE_Int low, NALU_HYPRE_Int high,
                                NALU_HYPRE_Int *spot );
NALU_HYPRE_Int *nalu_hypre_LowerBound( NALU_HYPRE_Int *first, NALU_HYPRE_Int *last, NALU_HYPRE_Int value );
NALU_HYPRE_BigInt *nalu_hypre_BigLowerBound( NALU_HYPRE_BigInt *first, NALU_HYPRE_BigInt *last, NALU_HYPRE_BigInt value );

/* log.c */
NALU_HYPRE_Int nalu_hypre_Log2( NALU_HYPRE_Int p );

/* complex.c */
#ifdef NALU_HYPRE_COMPLEX
NALU_HYPRE_Complex nalu_hypre_conj( NALU_HYPRE_Complex value );
NALU_HYPRE_Real    nalu_hypre_cabs( NALU_HYPRE_Complex value );
NALU_HYPRE_Real    nalu_hypre_creal( NALU_HYPRE_Complex value );
NALU_HYPRE_Real    nalu_hypre_cimag( NALU_HYPRE_Complex value );
NALU_HYPRE_Complex nalu_hypre_csqrt( NALU_HYPRE_Complex value );
#else
#define nalu_hypre_conj(value)  value
#define nalu_hypre_cabs(value)  nalu_hypre_abs(value)
#define nalu_hypre_creal(value) value
#define nalu_hypre_cimag(value) 0.0
#define nalu_hypre_csqrt(value) nalu_hypre_sqrt(value)
#endif

/* state.c */
NALU_HYPRE_Int nalu_hypre_Initialized( void );
NALU_HYPRE_Int nalu_hypre_Finalized( void );
NALU_HYPRE_Int nalu_hypre_SetInitialized( void );
NALU_HYPRE_Int nalu_hypre_SetFinalized( void );

/* general.c */
nalu_hypre_Handle* nalu_hypre_handle(void);
nalu_hypre_Handle* nalu_hypre_HandleCreate(void);
NALU_HYPRE_Int nalu_hypre_HandleDestroy(nalu_hypre_Handle *nalu_hypre_handle_);
NALU_HYPRE_Int nalu_hypre_SetDevice(nalu_hypre_int device_id, nalu_hypre_Handle *nalu_hypre_handle_);
NALU_HYPRE_Int nalu_hypre_GetDevice(nalu_hypre_int *device_id);
NALU_HYPRE_Int nalu_hypre_GetDeviceCount(nalu_hypre_int *device_count);
NALU_HYPRE_Int nalu_hypre_GetDeviceLastError(void);
NALU_HYPRE_Int nalu_hypre_UmpireInit(nalu_hypre_Handle *nalu_hypre_handle_);
NALU_HYPRE_Int nalu_hypre_UmpireFinalize(nalu_hypre_Handle *nalu_hypre_handle_);

/* qsort.c */
void nalu_hypre_swap ( NALU_HYPRE_Int *v, NALU_HYPRE_Int i, NALU_HYPRE_Int j );
void nalu_hypre_swap_c ( NALU_HYPRE_Complex *v, NALU_HYPRE_Int i, NALU_HYPRE_Int j );
void nalu_hypre_swap2 ( NALU_HYPRE_Int *v, NALU_HYPRE_Real *w, NALU_HYPRE_Int i, NALU_HYPRE_Int j );
void nalu_hypre_BigSwap2 ( NALU_HYPRE_BigInt *v, NALU_HYPRE_Real *w, NALU_HYPRE_Int i, NALU_HYPRE_Int j );
void nalu_hypre_swap2i ( NALU_HYPRE_Int *v, NALU_HYPRE_Int *w, NALU_HYPRE_Int i, NALU_HYPRE_Int j );
void nalu_hypre_BigSwap2i ( NALU_HYPRE_BigInt *v, NALU_HYPRE_Int *w, NALU_HYPRE_Int i, NALU_HYPRE_Int j );
void nalu_hypre_swap3i ( NALU_HYPRE_Int *v, NALU_HYPRE_Int *w, NALU_HYPRE_Int *z, NALU_HYPRE_Int i, NALU_HYPRE_Int j );
void nalu_hypre_swap3_d ( NALU_HYPRE_Real *v, NALU_HYPRE_Int *w, NALU_HYPRE_Int *z, NALU_HYPRE_Int i, NALU_HYPRE_Int j );
void nalu_hypre_swap3_d_perm(NALU_HYPRE_Int  *v, NALU_HYPRE_Real  *w, NALU_HYPRE_Int  *z, NALU_HYPRE_Int  i, NALU_HYPRE_Int  j );
void nalu_hypre_BigSwap4_d ( NALU_HYPRE_Real *v, NALU_HYPRE_BigInt *w, NALU_HYPRE_Int *z, NALU_HYPRE_Int *y, NALU_HYPRE_Int i,
                        NALU_HYPRE_Int j );
void nalu_hypre_swap_d ( NALU_HYPRE_Real *v, NALU_HYPRE_Int i, NALU_HYPRE_Int j );
void nalu_hypre_qsort0 ( NALU_HYPRE_Int *v, NALU_HYPRE_Int left, NALU_HYPRE_Int right );
void nalu_hypre_qsort1 ( NALU_HYPRE_Int *v, NALU_HYPRE_Real *w, NALU_HYPRE_Int left, NALU_HYPRE_Int right );
void nalu_hypre_BigQsort1 ( NALU_HYPRE_BigInt *v, NALU_HYPRE_Real *w, NALU_HYPRE_Int left, NALU_HYPRE_Int right );
void nalu_hypre_qsort2i ( NALU_HYPRE_Int *v, NALU_HYPRE_Int *w, NALU_HYPRE_Int left, NALU_HYPRE_Int right );
void nalu_hypre_BigQsort2i( NALU_HYPRE_BigInt *v, NALU_HYPRE_Int *w, NALU_HYPRE_Int  left, NALU_HYPRE_Int  right );
void nalu_hypre_qsort2 ( NALU_HYPRE_Int *v, NALU_HYPRE_Real *w, NALU_HYPRE_Int left, NALU_HYPRE_Int right );
void nalu_hypre_qsort2_abs ( NALU_HYPRE_Int *v, NALU_HYPRE_Real *w, NALU_HYPRE_Int left, NALU_HYPRE_Int right );
void nalu_hypre_qsort3i ( NALU_HYPRE_Int *v, NALU_HYPRE_Int *w, NALU_HYPRE_Int *z, NALU_HYPRE_Int left, NALU_HYPRE_Int right );
void nalu_hypre_qsort3ir ( NALU_HYPRE_Int *v, NALU_HYPRE_Real *w, NALU_HYPRE_Int *z, NALU_HYPRE_Int left, NALU_HYPRE_Int right );
void nalu_hypre_qsort3( NALU_HYPRE_Real *v, NALU_HYPRE_Int *w, NALU_HYPRE_Int *z, NALU_HYPRE_Int  left, NALU_HYPRE_Int  right );
void nalu_hypre_qsort3_abs ( NALU_HYPRE_Real *v, NALU_HYPRE_Int *w, NALU_HYPRE_Int *z, NALU_HYPRE_Int left,
                        NALU_HYPRE_Int right );
void nalu_hypre_BigQsort4_abs ( NALU_HYPRE_Real *v, NALU_HYPRE_BigInt *w, NALU_HYPRE_Int *z, NALU_HYPRE_Int *y,
                           NALU_HYPRE_Int left, NALU_HYPRE_Int right );
void nalu_hypre_qsort_abs ( NALU_HYPRE_Real *w, NALU_HYPRE_Int left, NALU_HYPRE_Int right );
void nalu_hypre_BigSwapbi(NALU_HYPRE_BigInt  *v, NALU_HYPRE_Int  *w, NALU_HYPRE_Int  i, NALU_HYPRE_Int  j );
void nalu_hypre_BigQsortbi( NALU_HYPRE_BigInt *v, NALU_HYPRE_Int *w, NALU_HYPRE_Int  left, NALU_HYPRE_Int  right );
void nalu_hypre_BigSwapLoc(NALU_HYPRE_BigInt  *v, NALU_HYPRE_Int  *w, NALU_HYPRE_Int  i, NALU_HYPRE_Int  j );
void nalu_hypre_BigQsortbLoc( NALU_HYPRE_BigInt *v, NALU_HYPRE_Int *w, NALU_HYPRE_Int  left, NALU_HYPRE_Int  right );
void nalu_hypre_BigSwapb2i(NALU_HYPRE_BigInt  *v, NALU_HYPRE_Int  *w, NALU_HYPRE_Int  *z, NALU_HYPRE_Int  i, NALU_HYPRE_Int  j );
void nalu_hypre_BigQsortb2i( NALU_HYPRE_BigInt *v, NALU_HYPRE_Int *w, NALU_HYPRE_Int *z, NALU_HYPRE_Int  left,
                        NALU_HYPRE_Int  right );
void nalu_hypre_BigSwap( NALU_HYPRE_BigInt *v, NALU_HYPRE_Int  i, NALU_HYPRE_Int  j );
void nalu_hypre_BigQsort0( NALU_HYPRE_BigInt *v, NALU_HYPRE_Int  left, NALU_HYPRE_Int  right );
void nalu_hypre_topo_sort(const NALU_HYPRE_Int *row_ptr, const NALU_HYPRE_Int *col_inds, const NALU_HYPRE_Complex *data,
                     NALU_HYPRE_Int *ordering, NALU_HYPRE_Int n);
void nalu_hypre_dense_topo_sort(const NALU_HYPRE_Complex *L, NALU_HYPRE_Int *ordering, NALU_HYPRE_Int n,
                           NALU_HYPRE_Int is_col_major);

/* qsplit.c */
NALU_HYPRE_Int nalu_hypre_DoubleQuickSplit ( NALU_HYPRE_Real *values, NALU_HYPRE_Int *indices, NALU_HYPRE_Int list_length,
                                   NALU_HYPRE_Int NumberKept );

/* random.c */
/* NALU_HYPRE_CUDA_GLOBAL */ void nalu_hypre_SeedRand ( NALU_HYPRE_Int seed );
/* NALU_HYPRE_CUDA_GLOBAL */ NALU_HYPRE_Int nalu_hypre_RandI ( void );
/* NALU_HYPRE_CUDA_GLOBAL */ NALU_HYPRE_Real nalu_hypre_Rand ( void );

/* prefix_sum.c */
/**
 * Assumed to be called within an omp region.
 * Let x_i be the input of ith thread.
 * The output of ith thread y_i = x_0 + x_1 + ... + x_{i-1}
 * Additionally, sum = x_0 + x_1 + ... + x_{nthreads - 1}
 * Note that always y_0 = 0
 *
 * @param workspace at least with length (nthreads+1)
 *                  workspace[tid] will contain result for tid
 *                  workspace[nthreads] will contain sum
 */
void nalu_hypre_prefix_sum(NALU_HYPRE_Int *in_out, NALU_HYPRE_Int *sum, NALU_HYPRE_Int *workspace);
/**
 * This version does prefix sum in pair.
 * Useful when we prefix sum of diag and offd in tandem.
 *
 * @param worksapce at least with length 2*(nthreads+1)
 *                  workspace[2*tid] and workspace[2*tid+1] will contain results for tid
 *                  workspace[3*nthreads] and workspace[3*nthreads + 1] will contain sums
 */
void nalu_hypre_prefix_sum_pair(NALU_HYPRE_Int *in_out1, NALU_HYPRE_Int *sum1, NALU_HYPRE_Int *in_out2, NALU_HYPRE_Int *sum2,
                           NALU_HYPRE_Int *workspace);
/**
 * @param workspace at least with length 3*(nthreads+1)
 *                  workspace[3*tid:3*tid+3) will contain results for tid
 */
void nalu_hypre_prefix_sum_triple(NALU_HYPRE_Int *in_out1, NALU_HYPRE_Int *sum1, NALU_HYPRE_Int *in_out2,
                             NALU_HYPRE_Int *sum2, NALU_HYPRE_Int *in_out3, NALU_HYPRE_Int *sum3, NALU_HYPRE_Int *workspace);

/**
 * n prefix-sums together.
 * workspace[n*tid:n*(tid+1)) will contain results for tid
 * workspace[nthreads*tid:nthreads*(tid+1)) will contain sums
 *
 * @param workspace at least with length n*(nthreads+1)
 */
void nalu_hypre_prefix_sum_multiple(NALU_HYPRE_Int *in_out, NALU_HYPRE_Int *sum, NALU_HYPRE_Int n,
                               NALU_HYPRE_Int *workspace);

/* hopscotch_hash.c */

#ifdef NALU_HYPRE_USING_OPENMP

/* Check if atomic operations are available to use concurrent hopscotch hash table */
#if defined(__GNUC__) && defined(__GNUC_MINOR__) && defined(__GNUC_PATCHLEVEL__) && (__GNUC__ * 10000 + __GNUC_MINOR__ * 100 + __GNUC_PATCHLEVEL__) > 40100
#define NALU_HYPRE_USING_ATOMIC
//#elif defined _MSC_VER // JSP: haven't tested, so comment out for now
//#define NALU_HYPRE_USING_ATOMIC
//#elif defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L && !defined(__STDC_NO_ATOMICS__)
// JSP: not many compilers have implemented this, so comment out for now
//#define NALU_HYPRE_USING_ATOMIC
//#include <stdatomic.h>
#endif

#endif // NALU_HYPRE_USING_OPENMP

#ifdef NALU_HYPRE_HOPSCOTCH
#ifdef NALU_HYPRE_USING_ATOMIC
// concurrent hopscotch hashing is possible only with atomic supports
#define NALU_HYPRE_CONCURRENT_HOPSCOTCH
#endif
#endif

#ifdef NALU_HYPRE_CONCURRENT_HOPSCOTCH
typedef struct
{
   NALU_HYPRE_Int volatile timestamp;
   omp_lock_t         lock;
} nalu_hypre_HopscotchSegment;
#endif

/**
 * The current typical use case of unordered set is putting input sequence
 * with lots of duplication (putting all colidx received from other ranks),
 * followed by one sweep of enumeration.
 * Since the capacity is set to the number of inputs, which is much larger
 * than the number of unique elements, we optimize for initialization and
 * enumeration whose time is proportional to the capacity.
 * For initialization and enumeration, structure of array (SoA) is better
 * for vectorization, cache line utilization, and so on.
 */
typedef struct
{
   NALU_HYPRE_Int  volatile              segmentMask;
   NALU_HYPRE_Int  volatile              bucketMask;
#ifdef NALU_HYPRE_CONCURRENT_HOPSCOTCH
   nalu_hypre_HopscotchSegment* volatile segments;
#endif
   NALU_HYPRE_Int *volatile              key;
   nalu_hypre_uint *volatile             hopInfo;
   NALU_HYPRE_Int *volatile              hash;
} nalu_hypre_UnorderedIntSet;

typedef struct
{
   NALU_HYPRE_Int volatile            segmentMask;
   NALU_HYPRE_Int volatile            bucketMask;
#ifdef NALU_HYPRE_CONCURRENT_HOPSCOTCH
   nalu_hypre_HopscotchSegment* volatile segments;
#endif
   NALU_HYPRE_BigInt *volatile           key;
   nalu_hypre_uint *volatile             hopInfo;
   NALU_HYPRE_BigInt *volatile           hash;
} nalu_hypre_UnorderedBigIntSet;

typedef struct
{
   nalu_hypre_uint volatile hopInfo;
   NALU_HYPRE_Int  volatile hash;
   NALU_HYPRE_Int  volatile key;
   NALU_HYPRE_Int  volatile data;
} nalu_hypre_HopscotchBucket;

typedef struct
{
   nalu_hypre_uint volatile hopInfo;
   NALU_HYPRE_BigInt  volatile hash;
   NALU_HYPRE_BigInt  volatile key;
   NALU_HYPRE_Int  volatile data;
} nalu_hypre_BigHopscotchBucket;

/**
 * The current typical use case of unoredered map is putting input sequence
 * with no duplication (inverse map of a bijective mapping) followed by
 * lots of lookups.
 * For lookup, array of structure (AoS) gives better cache line utilization.
 */
typedef struct
{
   NALU_HYPRE_Int  volatile              segmentMask;
   NALU_HYPRE_Int  volatile              bucketMask;
#ifdef NALU_HYPRE_CONCURRENT_HOPSCOTCH
   nalu_hypre_HopscotchSegment* volatile segments;
#endif
   nalu_hypre_HopscotchBucket* volatile table;
} nalu_hypre_UnorderedIntMap;

typedef struct
{
   NALU_HYPRE_Int  volatile segmentMask;
   NALU_HYPRE_Int  volatile bucketMask;
#ifdef NALU_HYPRE_CONCURRENT_HOPSCOTCH
   nalu_hypre_HopscotchSegment* volatile segments;
#endif
   nalu_hypre_BigHopscotchBucket* volatile table;
} nalu_hypre_UnorderedBigIntMap;

/* merge_sort.c */
/**
 * Why merge sort?
 * 1) Merge sort can take advantage of eliminating duplicates.
 * 2) Merge sort is more efficiently parallelizable than qsort
 */
NALU_HYPRE_Int nalu_hypre_IntArrayMergeOrdered( nalu_hypre_IntArray *array1, nalu_hypre_IntArray *array2,
                                      nalu_hypre_IntArray *array3 );
void nalu_hypre_union2(NALU_HYPRE_Int n1, NALU_HYPRE_BigInt *arr1, NALU_HYPRE_Int n2, NALU_HYPRE_BigInt *arr2, NALU_HYPRE_Int *n3,
                  NALU_HYPRE_BigInt *arr3, NALU_HYPRE_Int *map1, NALU_HYPRE_Int *map2);
void nalu_hypre_merge_sort(NALU_HYPRE_Int *in, NALU_HYPRE_Int *temp, NALU_HYPRE_Int len, NALU_HYPRE_Int **sorted);
void nalu_hypre_big_merge_sort(NALU_HYPRE_BigInt *in, NALU_HYPRE_BigInt *temp, NALU_HYPRE_Int len,
                          NALU_HYPRE_BigInt **sorted);
void nalu_hypre_sort_and_create_inverse_map(NALU_HYPRE_Int *in, NALU_HYPRE_Int len, NALU_HYPRE_Int **out,
                                       nalu_hypre_UnorderedIntMap *inverse_map);
void nalu_hypre_big_sort_and_create_inverse_map(NALU_HYPRE_BigInt *in, NALU_HYPRE_Int len, NALU_HYPRE_BigInt **out,
                                           nalu_hypre_UnorderedBigIntMap *inverse_map);

/* device_utils.c */
#if defined(NALU_HYPRE_USING_GPU)
NALU_HYPRE_Int nalu_hypre_SyncComputeStream(nalu_hypre_Handle *nalu_hypre_handle);
NALU_HYPRE_Int nalu_hypre_SyncCudaDevice(nalu_hypre_Handle *nalu_hypre_handle);
NALU_HYPRE_Int nalu_hypre_ResetCudaDevice(nalu_hypre_Handle *nalu_hypre_handle);
NALU_HYPRE_Int hypreDevice_DiagScaleVector(NALU_HYPRE_Int num_vectors, NALU_HYPRE_Int num_rows,
                                      NALU_HYPRE_Int *A_i, NALU_HYPRE_Complex *A_data,
                                      NALU_HYPRE_Complex *x, NALU_HYPRE_Complex beta,
                                      NALU_HYPRE_Complex *y);
NALU_HYPRE_Int hypreDevice_DiagScaleVector2(NALU_HYPRE_Int num_vectors, NALU_HYPRE_Int num_rows,
                                       NALU_HYPRE_Complex *diag, NALU_HYPRE_Complex *x,
                                       NALU_HYPRE_Complex beta, NALU_HYPRE_Complex *y,
                                       NALU_HYPRE_Complex *z, NALU_HYPRE_Int computeY);
NALU_HYPRE_Int hypreDevice_ComplexArrayToArrayOfPtrs(NALU_HYPRE_Int n, NALU_HYPRE_Int m,
                                                NALU_HYPRE_Complex *data, NALU_HYPRE_Complex **data_aop);
NALU_HYPRE_Int hypreDevice_zeqxmydd(NALU_HYPRE_Int n, NALU_HYPRE_Complex *x, NALU_HYPRE_Complex alpha,
                               NALU_HYPRE_Complex *y, NALU_HYPRE_Complex *z, NALU_HYPRE_Complex *d);
NALU_HYPRE_Int hypreDevice_IVAXPY(NALU_HYPRE_Int n, NALU_HYPRE_Complex *a, NALU_HYPRE_Complex *x, NALU_HYPRE_Complex *y);
NALU_HYPRE_Int hypreDevice_IVAXPYMarked(NALU_HYPRE_Int n, NALU_HYPRE_Complex *a, NALU_HYPRE_Complex *x,
                                   NALU_HYPRE_Complex *y, NALU_HYPRE_Int *marker, NALU_HYPRE_Int marker_val);
NALU_HYPRE_Int hypreDevice_IVAMXPMY(NALU_HYPRE_Int m, NALU_HYPRE_Int n, NALU_HYPRE_Complex *a,
                               NALU_HYPRE_Complex *x, NALU_HYPRE_Complex *y);
NALU_HYPRE_Int hypreDevice_IntFilln(NALU_HYPRE_Int *d_x, size_t n, NALU_HYPRE_Int v);
NALU_HYPRE_Int hypreDevice_BigIntFilln(NALU_HYPRE_BigInt *d_x, size_t n, NALU_HYPRE_BigInt v);
NALU_HYPRE_Int hypreDevice_ComplexFilln(NALU_HYPRE_Complex *d_x, size_t n, NALU_HYPRE_Complex v);
NALU_HYPRE_Int hypreDevice_CharFilln(char *d_x, size_t n, char v);
NALU_HYPRE_Int hypreDevice_IntStridedCopy ( NALU_HYPRE_Int size, NALU_HYPRE_Int stride,
                                       NALU_HYPRE_Int *in, NALU_HYPRE_Int *out );
NALU_HYPRE_Int hypreDevice_ComplexStridedCopy ( NALU_HYPRE_Int size, NALU_HYPRE_Int stride,
                                           NALU_HYPRE_Complex *in, NALU_HYPRE_Complex *out );
NALU_HYPRE_Int hypreDevice_IntScalen(NALU_HYPRE_Int *d_x, size_t n, NALU_HYPRE_Int *d_y, NALU_HYPRE_Int v);
NALU_HYPRE_Int hypreDevice_ComplexScalen(NALU_HYPRE_Complex *d_x, size_t n, NALU_HYPRE_Complex *d_y,
                                    NALU_HYPRE_Complex v);
NALU_HYPRE_Int hypreDevice_ComplexAxpyn(NALU_HYPRE_Complex *d_x, size_t n, NALU_HYPRE_Complex *d_y,
                                   NALU_HYPRE_Complex *d_z, NALU_HYPRE_Complex a);
NALU_HYPRE_Int hypreDevice_ComplexAxpyzn(NALU_HYPRE_Int n, NALU_HYPRE_Complex *d_x, NALU_HYPRE_Complex *d_y,
                                    NALU_HYPRE_Complex *d_z, NALU_HYPRE_Complex a, NALU_HYPRE_Complex b);
NALU_HYPRE_Int hypreDevice_IntAxpyn(NALU_HYPRE_Int *d_x, size_t n, NALU_HYPRE_Int *d_y, NALU_HYPRE_Int *d_z,
                               NALU_HYPRE_Int a);
NALU_HYPRE_Int hypreDevice_BigIntAxpyn(NALU_HYPRE_BigInt *d_x, size_t n, NALU_HYPRE_BigInt *d_y,
                                  NALU_HYPRE_BigInt *d_z, NALU_HYPRE_BigInt a);
NALU_HYPRE_Int* hypreDevice_CsrRowPtrsToIndices(NALU_HYPRE_Int nrows, NALU_HYPRE_Int nnz, NALU_HYPRE_Int *d_row_ptr);
NALU_HYPRE_Int hypreDevice_CsrRowPtrsToIndices_v2(NALU_HYPRE_Int nrows, NALU_HYPRE_Int nnz, NALU_HYPRE_Int *d_row_ptr,
                                             NALU_HYPRE_Int *d_row_ind);
NALU_HYPRE_Int* hypreDevice_CsrRowIndicesToPtrs(NALU_HYPRE_Int nrows, NALU_HYPRE_Int nnz, NALU_HYPRE_Int *d_row_ind);
NALU_HYPRE_Int hypreDevice_CsrRowIndicesToPtrs_v2(NALU_HYPRE_Int nrows, NALU_HYPRE_Int nnz, NALU_HYPRE_Int *d_row_ind,
                                             NALU_HYPRE_Int *d_row_ptr);
NALU_HYPRE_Int hypreDevice_GetRowNnz(NALU_HYPRE_Int nrows, NALU_HYPRE_Int *d_row_indices, NALU_HYPRE_Int *d_diag_ia,
                                NALU_HYPRE_Int *d_offd_ia, NALU_HYPRE_Int *d_rownnz);
NALU_HYPRE_Int hypreDevice_CopyParCSRRows(NALU_HYPRE_Int nrows, NALU_HYPRE_Int *d_row_indices, NALU_HYPRE_Int job,
                                     NALU_HYPRE_Int has_offd, NALU_HYPRE_BigInt first_col,
                                     NALU_HYPRE_BigInt *d_col_map_offd_A, NALU_HYPRE_Int *d_diag_i,
                                     NALU_HYPRE_Int *d_diag_j, NALU_HYPRE_Complex *d_diag_a,
                                     NALU_HYPRE_Int *d_offd_i, NALU_HYPRE_Int *d_offd_j,
                                     NALU_HYPRE_Complex *d_offd_a, NALU_HYPRE_Int *d_ib,
                                     NALU_HYPRE_BigInt *d_jb, NALU_HYPRE_Complex *d_ab);
NALU_HYPRE_Int hypreDevice_IntegerReduceSum(NALU_HYPRE_Int m, NALU_HYPRE_Int *d_i);
NALU_HYPRE_Complex hypreDevice_ComplexReduceSum(NALU_HYPRE_Int m, NALU_HYPRE_Complex *d_x);
NALU_HYPRE_Int hypreDevice_IntegerInclusiveScan(NALU_HYPRE_Int n, NALU_HYPRE_Int *d_i);
NALU_HYPRE_Int hypreDevice_IntegerExclusiveScan(NALU_HYPRE_Int n, NALU_HYPRE_Int *d_i);
NALU_HYPRE_Int nalu_hypre_CudaCompileFlagCheck(void);
#endif

NALU_HYPRE_Int nalu_hypre_CurandUniform( NALU_HYPRE_Int n, NALU_HYPRE_Real *urand, NALU_HYPRE_Int set_seed,
                               nalu_hypre_ulonglongint seed, NALU_HYPRE_Int set_offset, nalu_hypre_ulonglongint offset);
NALU_HYPRE_Int nalu_hypre_CurandUniformSingle( NALU_HYPRE_Int n, float *urand, NALU_HYPRE_Int set_seed,
                                     nalu_hypre_ulonglongint seed, NALU_HYPRE_Int set_offset, nalu_hypre_ulonglongint offset);

NALU_HYPRE_Int nalu_hypre_ResetDeviceRandGenerator( nalu_hypre_ulonglongint seed, nalu_hypre_ulonglongint offset );

NALU_HYPRE_Int nalu_hypre_bind_device(NALU_HYPRE_Int myid, NALU_HYPRE_Int nproc, MPI_Comm comm);

/* nvtx.c */
void nalu_hypre_GpuProfilingPushRangeColor(const char *name, NALU_HYPRE_Int cid);
void nalu_hypre_GpuProfilingPushRange(const char *name);
void nalu_hypre_GpuProfilingPopRange(void);

/* utilities.c */
NALU_HYPRE_Int nalu_hypre_multmod(NALU_HYPRE_Int a, NALU_HYPRE_Int b, NALU_HYPRE_Int mod);
void nalu_hypre_partition1D(NALU_HYPRE_Int n, NALU_HYPRE_Int p, NALU_HYPRE_Int j, NALU_HYPRE_Int *s, NALU_HYPRE_Int *e);
char *nalu_hypre_strcpy(char *destination, const char *source);

NALU_HYPRE_Int nalu_hypre_SetSyncCudaCompute(NALU_HYPRE_Int action);
NALU_HYPRE_Int nalu_hypre_RestoreSyncCudaCompute(void);
NALU_HYPRE_Int nalu_hypre_GetSyncCudaCompute(NALU_HYPRE_Int *cuda_compute_stream_sync_ptr);
NALU_HYPRE_Int nalu_hypre_ForceSyncComputeStream(nalu_hypre_Handle *nalu_hypre_handle);

/* handle.c */
NALU_HYPRE_Int nalu_hypre_SetSpTransUseVendor( NALU_HYPRE_Int use_vendor );
NALU_HYPRE_Int nalu_hypre_SetSpMVUseVendor( NALU_HYPRE_Int use_vendor );
NALU_HYPRE_Int nalu_hypre_SetSpGemmUseVendor( NALU_HYPRE_Int use_vendor );
NALU_HYPRE_Int nalu_hypre_SetSpGemmAlgorithm( NALU_HYPRE_Int value );
NALU_HYPRE_Int nalu_hypre_SetSpGemmBinned( NALU_HYPRE_Int value );
NALU_HYPRE_Int nalu_hypre_SetSpGemmRownnzEstimateMethod( NALU_HYPRE_Int value );
NALU_HYPRE_Int nalu_hypre_SetSpGemmRownnzEstimateNSamples( NALU_HYPRE_Int value );
NALU_HYPRE_Int nalu_hypre_SetSpGemmRownnzEstimateMultFactor( NALU_HYPRE_Real value );
NALU_HYPRE_Int nalu_hypre_SetSpGemmHashType( char value );
NALU_HYPRE_Int nalu_hypre_SetUseGpuRand( NALU_HYPRE_Int use_gpurand );
NALU_HYPRE_Int nalu_hypre_SetGaussSeidelMethod( NALU_HYPRE_Int gs_method );
NALU_HYPRE_Int nalu_hypre_SetUserDeviceMalloc(GPUMallocFunc func);
NALU_HYPRE_Int nalu_hypre_SetUserDeviceMfree(GPUMfreeFunc func);

/* int_array.c */
nalu_hypre_IntArray* nalu_hypre_IntArrayCreate( NALU_HYPRE_Int size );
NALU_HYPRE_Int nalu_hypre_IntArrayDestroy( nalu_hypre_IntArray *array );
NALU_HYPRE_Int nalu_hypre_IntArrayInitialize_v2( nalu_hypre_IntArray *array,
                                       NALU_HYPRE_MemoryLocation memory_location );
NALU_HYPRE_Int nalu_hypre_IntArrayInitialize( nalu_hypre_IntArray *array );
NALU_HYPRE_Int nalu_hypre_IntArrayCopy( nalu_hypre_IntArray *x, nalu_hypre_IntArray *y );
nalu_hypre_IntArray* nalu_hypre_IntArrayCloneDeep_v2( nalu_hypre_IntArray *x,
                                            NALU_HYPRE_MemoryLocation memory_location );
nalu_hypre_IntArray* nalu_hypre_IntArrayCloneDeep( nalu_hypre_IntArray *x );
NALU_HYPRE_Int nalu_hypre_IntArrayMigrate( nalu_hypre_IntArray *v, NALU_HYPRE_MemoryLocation memory_location );
NALU_HYPRE_Int nalu_hypre_IntArrayPrint( MPI_Comm comm, nalu_hypre_IntArray *array, const char *filename );
NALU_HYPRE_Int nalu_hypre_IntArrayRead( MPI_Comm comm, const char *filename, nalu_hypre_IntArray **array_ptr );
NALU_HYPRE_Int nalu_hypre_IntArraySetConstantValuesHost( nalu_hypre_IntArray *v, NALU_HYPRE_Int value );
NALU_HYPRE_Int nalu_hypre_IntArraySetConstantValues( nalu_hypre_IntArray *v, NALU_HYPRE_Int value );
NALU_HYPRE_Int nalu_hypre_IntArrayCountHost( nalu_hypre_IntArray *v, NALU_HYPRE_Int value,
                                   NALU_HYPRE_Int *num_values_ptr );
NALU_HYPRE_Int nalu_hypre_IntArrayCount( nalu_hypre_IntArray *v, NALU_HYPRE_Int value,
                               NALU_HYPRE_Int *num_values_ptr );
NALU_HYPRE_Int nalu_hypre_IntArrayInverseMapping( nalu_hypre_IntArray *v, nalu_hypre_IntArray **w_ptr );
NALU_HYPRE_Int nalu_hypre_IntArrayNegate( nalu_hypre_IntArray *v );

/* int_array_device.c */
#if defined(NALU_HYPRE_USING_GPU) || defined(NALU_HYPRE_USING_DEVICE_OPENMP)
NALU_HYPRE_Int nalu_hypre_IntArraySetConstantValuesDevice( nalu_hypre_IntArray *v, NALU_HYPRE_Int value );
NALU_HYPRE_Int nalu_hypre_IntArrayCountDevice ( nalu_hypre_IntArray *v, NALU_HYPRE_Int value,
                                      NALU_HYPRE_Int *num_values_ptr );
NALU_HYPRE_Int nalu_hypre_IntArrayInverseMappingDevice( nalu_hypre_IntArray *v, nalu_hypre_IntArray *w );
NALU_HYPRE_Int nalu_hypre_IntArrayNegateDevice( nalu_hypre_IntArray *v );
#endif

/* memory_tracker.c */
#ifdef NALU_HYPRE_USING_MEMORY_TRACKER
nalu_hypre_MemoryTracker* nalu_hypre_memory_tracker(void);
nalu_hypre_MemoryTracker * nalu_hypre_MemoryTrackerCreate(void);
void nalu_hypre_MemoryTrackerDestroy(nalu_hypre_MemoryTracker *tracker);
void nalu_hypre_MemoryTrackerInsert1(const char *action, void *ptr, size_t nbytes,
                                nalu_hypre_MemoryLocation memory_location, const char *filename,
                                const char *function, NALU_HYPRE_Int line);
void nalu_hypre_MemoryTrackerInsert2(const char *action, void *ptr, void *ptr2, size_t nbytes,
                                nalu_hypre_MemoryLocation memory_location,
                                nalu_hypre_MemoryLocation memory_location2,
                                const char *filename,
                                const char *function, NALU_HYPRE_Int line);
NALU_HYPRE_Int nalu_hypre_PrintMemoryTracker( size_t *totl_bytes_o, size_t *peak_bytes_o,
                                    size_t *curr_bytes_o, NALU_HYPRE_Int do_print, const char *fname );
NALU_HYPRE_Int nalu_hypre_MemoryTrackerSetPrint(NALU_HYPRE_Int do_print);
NALU_HYPRE_Int nalu_hypre_MemoryTrackerSetFileName(const char *file_name);
#endif

/* magma.c */
#if defined(NALU_HYPRE_USING_MAGMA)
NALU_HYPRE_Int nalu_hypre_MagmaInitialize(void);
NALU_HYPRE_Int nalu_hypre_MagmaFinalize(void);
#endif
