/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/*--------------------------------------------------------------------------
 * nalu_hypre_RedBlackGSData data structure
 *--------------------------------------------------------------------------*/

typedef struct
{
   MPI_Comm                comm;

   NALU_HYPRE_Real              tol;                /* not yet used */
   NALU_HYPRE_Int               max_iter;
   NALU_HYPRE_Int               rel_change;         /* not yet used */
   NALU_HYPRE_Int               zero_guess;
   NALU_HYPRE_Int               rb_start;

   nalu_hypre_StructMatrix     *A;
   nalu_hypre_StructVector     *b;
   nalu_hypre_StructVector     *x;

   NALU_HYPRE_Int               diag_rank;

   nalu_hypre_ComputePkg       *compute_pkg;

   /* log info (always logged) */
   NALU_HYPRE_Int               num_iterations;
   NALU_HYPRE_Int               time_index;
   NALU_HYPRE_Int               flops;

} nalu_hypre_RedBlackGSData;

#ifdef NALU_HYPRE_USING_RAJA

#define nalu_hypre_RedBlackLoopInit()
#define nalu_hypre_RedBlackLoopBegin(ni,nj,nk,redblack,     \
                                Astart,Ani,Anj,Ai,     \
                                bstart,bni,bnj,bi,     \
                                xstart,xni,xnj,xi)     \
{                                                      \
   NALU_HYPRE_Int nalu_hypre__tot = nk*nj*((ni+1)/2);            \
   forall< nalu_hypre_raja_exec_policy >(RangeSegment(0, nalu_hypre__tot), [=] nalu_hypre_RAJA_DEVICE (NALU_HYPRE_Int idx) \
   {                                                   \
      NALU_HYPRE_Int idx_local = idx;                       \
      NALU_HYPRE_Int ii,jj,kk,Ai,bi,xi;                     \
      NALU_HYPRE_Int local_ii;                              \
      kk = idx_local % nk;                             \
      idx_local = idx_local / nk;                      \
      jj = idx_local % nj;                             \
      idx_local = idx_local / nj;                      \
      local_ii = (kk + jj + redblack) % 2;             \
      ii = 2*idx_local + local_ii;                     \
      if (ii < ni)                                     \
      {                                                \
         Ai = Astart + kk*Anj*Ani + jj*Ani + ii;       \
         bi = bstart + kk*bnj*bni + jj*bni + ii;       \
         xi = xstart + kk*xnj*xni + jj*xni + ii;       \

#define nalu_hypre_RedBlackLoopEnd()                        \
      }                                                \
   });                                                 \
   nalu_hypre_fence();                                      \
}

#define nalu_hypre_RedBlackConstantcoefLoopBegin(ni,nj,nk,redblack, \
                                            bstart,bni,bnj,bi, \
                                            xstart,xni,xnj,xi) \
{                                                              \
   NALU_HYPRE_Int nalu_hypre__tot = nk*nj*((ni+1)/2);                    \
   forall< nalu_hypre_raja_exec_policy >(RangeSegment(0, nalu_hypre__tot), [=] nalu_hypre_RAJA_DEVICE (NALU_HYPRE_Int idx) \
   {                                                           \
      NALU_HYPRE_Int idx_local = idx;                               \
      NALU_HYPRE_Int ii,jj,kk,bi,xi;                                \
      NALU_HYPRE_Int local_ii;                                      \
      kk = idx_local % nk;                                     \
      idx_local = idx_local / nk;                              \
      jj = idx_local % nj;                                     \
      idx_local = idx_local / nj;                              \
      local_ii = (kk + jj + redblack) % 2;                     \
      ii = 2*idx_local + local_ii;                             \
      if (ii < ni)                                             \
      {                                                        \
          bi = bstart + kk*bnj*bni + jj*bni + ii;              \
          xi = xstart + kk*xnj*xni + jj*xni + ii;              \

#define nalu_hypre_RedBlackConstantcoefLoopEnd()                    \
      }                                                        \
   });                                                         \
   nalu_hypre_fence();                                              \
}

#elif defined(NALU_HYPRE_USING_KOKKOS)

#define nalu_hypre_RedBlackLoopInit()
#define nalu_hypre_RedBlackLoopBegin(ni,nj,nk,redblack,                  \
                                Astart,Ani,Anj,Ai,                  \
                                bstart,bni,bnj,bi,                  \
                                xstart,xni,xnj,xi)                  \
{                                                                   \
   NALU_HYPRE_Int nalu_hypre__tot = nk*nj*((ni+1)/2);                         \
   Kokkos::parallel_for (nalu_hypre__tot, KOKKOS_LAMBDA (NALU_HYPRE_Int idx)  \
   {                                                                \
      NALU_HYPRE_Int idx_local = idx;                                    \
      NALU_HYPRE_Int ii,jj,kk,Ai,bi,xi;                                  \
      NALU_HYPRE_Int local_ii;                                           \
      kk = idx_local % nk;                                          \
      idx_local = idx_local / nk;                                   \
      jj = idx_local % nj;                                          \
      idx_local = idx_local / nj;                                   \
      local_ii = (kk + jj + redblack) % 2;                          \
      ii = 2*idx_local + local_ii;                                  \
      if (ii < ni)                                                  \
      {                                                             \
         Ai = Astart + kk*Anj*Ani + jj*Ani + ii;                    \
         bi = bstart + kk*bnj*bni + jj*bni + ii;                    \
         xi = xstart + kk*xnj*xni + jj*xni + ii;                    \

#define nalu_hypre_RedBlackLoopEnd()                                     \
      }                                                             \
   });                                                              \
   nalu_hypre_fence();                                                   \
}

#define nalu_hypre_RedBlackConstantcoefLoopBegin(ni,nj,nk,redblack,      \
                                            bstart,bni,bnj,bi,      \
                                            xstart,xni,xnj,xi)      \
{                                                                   \
   NALU_HYPRE_Int nalu_hypre__tot = nk*nj*((ni+1)/2);                         \
   Kokkos::parallel_for (nalu_hypre__tot, KOKKOS_LAMBDA (NALU_HYPRE_Int idx)  \
   {                                                                \
      NALU_HYPRE_Int idx_local = idx;                                    \
      NALU_HYPRE_Int ii,jj,kk,bi,xi;                                     \
      NALU_HYPRE_Int local_ii;                                           \
      kk = idx_local % nk;                                          \
      idx_local = idx_local / nk;                                   \
      jj = idx_local % nj;                                          \
      idx_local = idx_local / nj;                                   \
      local_ii = (kk + jj + redblack) % 2;                          \
      ii = 2*idx_local + local_ii;                                  \
      if (ii < ni)                                                  \
      {                                                             \
         bi = bstart + kk*bnj*bni + jj*bni + ii;                    \
         xi = xstart + kk*xnj*xni + jj*xni + ii;                    \

#define nalu_hypre_RedBlackConstantcoefLoopEnd()                         \
      }                                                             \
   });                                                              \
   nalu_hypre_fence();                                                   \
}

#elif defined(NALU_HYPRE_USING_CUDA) || defined(NALU_HYPRE_USING_HIP)

#define nalu_hypre_RedBlackLoopInit()
#define nalu_hypre_RedBlackLoopBegin(ni,nj,nk,redblack,        \
                                Astart,Ani,Anj,Ai,        \
                                bstart,bni,bnj,bi,        \
                                xstart,xni,xnj,xi)        \
{                                                         \
   NALU_HYPRE_Int nalu_hypre__tot = nk*nj*((ni+1)/2);               \
   BoxLoopforall(nalu_hypre__tot, NALU_HYPRE_LAMBDA (NALU_HYPRE_Int idx) \
   {                                                      \
      NALU_HYPRE_Int idx_local = idx;                          \
      NALU_HYPRE_Int ii,jj,kk,Ai,bi,xi;                        \
      NALU_HYPRE_Int local_ii;                                 \
      kk = idx_local % nk;                                \
      idx_local = idx_local / nk;                         \
      jj = idx_local % nj;                                \
      idx_local = idx_local / nj;                         \
      local_ii = (kk + jj + redblack) % 2;                \
      ii = 2*idx_local + local_ii;                        \
      if (ii < ni)                                        \
      {                                                   \
         Ai = Astart + kk*Anj*Ani + jj*Ani + ii;          \
         bi = bstart + kk*bnj*bni + jj*bni + ii;          \
         xi = xstart + kk*xnj*xni + jj*xni + ii;          \

#define nalu_hypre_RedBlackLoopEnd()                           \
      }                                                   \
   });                                                    \
}

#define nalu_hypre_RedBlackConstantcoefLoopBegin(ni,nj,nk,redblack,      \
                                            bstart,bni,bnj,bi,      \
                                            xstart,xni,xnj,xi)      \
{                                                                   \
   NALU_HYPRE_Int nalu_hypre__tot = nk*nj*((ni+1)/2);                         \
   BoxLoopforall(nalu_hypre__tot, NALU_HYPRE_LAMBDA (NALU_HYPRE_Int idx)           \
   {                                                                \
      NALU_HYPRE_Int idx_local = idx;                                    \
      NALU_HYPRE_Int ii,jj,kk,bi,xi;                                     \
      NALU_HYPRE_Int local_ii;                                           \
      kk = idx_local % nk;                                          \
      idx_local = idx_local / nk;                                   \
      jj = idx_local % nj;                                          \
      idx_local = idx_local / nj;                                   \
      local_ii = (kk + jj + redblack) % 2;                          \
      ii = 2*idx_local + local_ii;                                  \
      if (ii < ni)                                                  \
      {                                                             \
         bi = bstart + kk*bnj*bni + jj*bni + ii;                    \
         xi = xstart + kk*xnj*xni + jj*xni + ii;                    \

#define nalu_hypre_RedBlackConstantcoefLoopEnd()                         \
      }                                                             \
   });                                                              \
}

#elif defined(NALU_HYPRE_USING_SYCL)

#define nalu_hypre_RedBlackLoopInit()
#define nalu_hypre_RedBlackLoopBegin(ni,nj,nk,redblack,                  \
                                Astart,Ani,Anj,Ai,                  \
                                bstart,bni,bnj,bi,                  \
                                xstart,xni,xnj,xi)                  \
{                                                                   \
   NALU_HYPRE_Int nalu_hypre__tot = nk*nj*((ni+1)/2);                         \
   BoxLoopforall(nalu_hypre__tot, [=] (sycl::nd_item<3> item)            \
   {                                                                \
      NALU_HYPRE_Int idx = (NALU_HYPRE_Int) item.get_global_linear_id();      \
      NALU_HYPRE_Int idx_local = idx;                                    \
      NALU_HYPRE_Int ii,jj,kk,Ai,bi,xi;                                  \
      NALU_HYPRE_Int local_ii;                                           \
      kk = idx_local % nk;                                          \
      idx_local = idx_local / nk;                                   \
      jj = idx_local % nj;                                          \
      idx_local = idx_local / nj;                                   \
      local_ii = (kk + jj + redblack) % 2;                          \
      ii = 2*idx_local + local_ii;                                  \
      if (ii < ni)                                                  \
      {                                                             \
         Ai = Astart + kk*Anj*Ani + jj*Ani + ii;                    \
         bi = bstart + kk*bnj*bni + jj*bni + ii;                    \
         xi = xstart + kk*xnj*xni + jj*xni + ii;                    \

#define nalu_hypre_RedBlackLoopEnd()                                     \
      }                                                             \
   });                                                              \
}

#define nalu_hypre_RedBlackConstantcoefLoopBegin(ni,nj,nk,redblack,      \
                                            bstart,bni,bnj,bi,      \
                                            xstart,xni,xnj,xi)      \
{                                                                   \
   NALU_HYPRE_Int nalu_hypre__tot = nk*nj*((ni+1)/2);                         \
   BoxLoopforall(nalu_hypre__tot, [=] (sycl::nd_item<3> item)            \
   {                                                                \
      NALU_HYPRE_Int idx = (NALU_HYPRE_Int) item.get_global_linear_id();      \
      NALU_HYPRE_Int idx_local = idx;                                    \
      NALU_HYPRE_Int ii,jj,kk,bi,xi;                                     \
      NALU_HYPRE_Int local_ii;                                           \
      kk = idx_local % nk;                                          \
      idx_local = idx_local / nk;                                   \
      jj = idx_local % nj;                                          \
      idx_local = idx_local / nj;                                   \
      local_ii = (kk + jj + redblack) % 2;                          \
      ii = 2*idx_local + local_ii;                                  \
      if (ii < ni)                                                  \
      {                                                             \
         bi = bstart + kk*bnj*bni + jj*bni + ii;                    \
         xi = xstart + kk*xnj*xni + jj*xni + ii;                    \

#define nalu_hypre_RedBlackConstantcoefLoopEnd()                         \
      }                                                             \
   });                                                              \
}

#elif defined(NALU_HYPRE_USING_DEVICE_OPENMP)

/* BEGIN OF OMP 4.5 */
/* #define IF_CLAUSE if (nalu_hypre__global_offload) */

/* stringification:
 * _Pragma(string-literal), so we need to cast argument to a string
 * The three dots as last argument of the macro tells compiler that this is a variadic macro.
 * I.e. this is a macro that receives variable number of arguments.
 */
//#define NALU_HYPRE_STR(s...) #s
//#define NALU_HYPRE_XSTR(s...) NALU_HYPRE_STR(s)

#define nalu_hypre_RedBlackLoopInit()

#define nalu_hypre_RedBlackLoopBegin(ni,nj,nk,redblack,                      \
                                Astart,Ani,Anj,Ai,                      \
                                bstart,bni,bnj,bi,                      \
                                xstart,xni,xnj,xi)                      \
{                                                                       \
   NALU_HYPRE_Int nalu_hypre__thread, nalu_hypre__tot = nk*nj*((ni+1)/2);              \
   NALU_HYPRE_BOXLOOP_ENTRY_PRINT                                            \
   /* device code: */                                                   \
   _Pragma (NALU_HYPRE_XSTR(omp target teams distribute parallel for IF_CLAUSE IS_DEVICE_CLAUSE)) \
   for (nalu_hypre__thread=0; nalu_hypre__thread<nalu_hypre__tot; nalu_hypre__thread++)     \
   {                                                                    \
        NALU_HYPRE_Int idx_local = nalu_hypre__thread;                            \
        NALU_HYPRE_Int ii,jj,kk,Ai,bi,xi;                                    \
        NALU_HYPRE_Int local_ii;                                             \
        kk = idx_local % nk;                                            \
        idx_local = idx_local / nk;                                     \
        jj = idx_local % nj;                                            \
        idx_local = idx_local / nj;                                     \
        local_ii = (kk + jj + redblack) % 2;                            \
        ii = 2*idx_local + local_ii;                                    \
        if (ii < ni)                                                    \
        {                                                               \
            Ai = Astart + kk*Anj*Ani + jj*Ani + ii;                     \
            bi = bstart + kk*bnj*bni + jj*bni + ii;                     \
            xi = xstart + kk*xnj*xni + jj*xni + ii;                     \

#define nalu_hypre_RedBlackLoopEnd()                                         \
        }                                                               \
     }                                                                  \
}



#define nalu_hypre_RedBlackConstantcoefLoopBegin(ni,nj,nk,redblack,        \
                                            bstart,bni,bnj,bi,        \
                                            xstart,xni,xnj,xi)        \
{                                                                     \
   NALU_HYPRE_Int nalu_hypre__thread, nalu_hypre__tot = nk*nj*((ni+1)/2);            \
   NALU_HYPRE_BOXLOOP_ENTRY_PRINT                                          \
   /* device code: */                                                 \
   _Pragma (NALU_HYPRE_XSTR(omp target teams distribute parallel for IF_CLAUSE IS_DEVICE_CLAUSE)) \
   for (nalu_hypre__thread=0; nalu_hypre__thread<nalu_hypre__tot; nalu_hypre__thread++)   \
   {                                                                  \
        NALU_HYPRE_Int idx_local = nalu_hypre__thread;                          \
        NALU_HYPRE_Int ii,jj,kk,bi,xi;                                     \
        NALU_HYPRE_Int local_ii;                                           \
        kk = idx_local % nk;                                          \
        idx_local = idx_local / nk;                                   \
        jj = idx_local % nj;                                          \
        idx_local = idx_local / nj;                                   \
        local_ii = (kk + jj + redblack) % 2;                          \
        ii = 2*idx_local + local_ii;                                  \
        if (ii < ni)                                                  \
        {                                                             \
            bi = bstart + kk*bnj*bni + jj*bni + ii;                   \
            xi = xstart + kk*xnj*xni + jj*xni + ii;                   \

#define nalu_hypre_RedBlackConstantcoefLoopEnd()                           \
         }                                                            \
     }                                                                \
}
/* END OF OMP 4.5 */

#else

/* CPU */
#define NALU_HYPRE_REDBLACK_PRIVATE nalu_hypre__kk

#define nalu_hypre_RedBlackLoopInit()\
{\
   NALU_HYPRE_Int nalu_hypre__kk;

#ifdef NALU_HYPRE_USING_OPENMP
#define NALU_HYPRE_BOX_REDUCTION
#if defined(WIN32) && defined(_MSC_VER)
#define Pragma(x) __pragma(NALU_HYPRE_XSTR(x))
#else
#define Pragma(x) _Pragma(NALU_HYPRE_XSTR(x))
#endif
#define OMPRB1 Pragma(omp parallel for private(NALU_HYPRE_REDBLACK_PRIVATE) NALU_HYPRE_BOX_REDUCTION NALU_HYPRE_SMP_SCHEDULE)
#else
#define OMPRB1
#endif

#define nalu_hypre_RedBlackLoopBegin(ni,nj,nk,redblack,  \
                                Astart,Ani,Anj,Ai,  \
                                bstart,bni,bnj,bi,  \
                                xstart,xni,xnj,xi)  \
   OMPRB1 \
   for (nalu_hypre__kk = 0; nalu_hypre__kk < nk; nalu_hypre__kk++) \
   {\
      NALU_HYPRE_Int ii,jj,Ai,bi,xi;\
      for (jj = 0; jj < nj; jj++)\
      {\
         ii = (nalu_hypre__kk + jj + redblack) % 2;\
         Ai = Astart + nalu_hypre__kk*Anj*Ani + jj*Ani + ii; \
         bi = bstart + nalu_hypre__kk*bnj*bni + jj*bni + ii; \
         xi = xstart + nalu_hypre__kk*xnj*xni + jj*xni + ii; \
         for (; ii < ni; ii+=2, Ai+=2, bi+=2, xi+=2)\
         {

#define nalu_hypre_RedBlackLoopEnd()\
         }\
      }\
   }\
}

#define nalu_hypre_RedBlackConstantcoefLoopBegin(ni,nj,nk,redblack, \
                                            bstart,bni,bnj,bi, \
                                            xstart,xni,xnj,xi) \
   OMPRB1 \
   for (nalu_hypre__kk = 0; nalu_hypre__kk < nk; nalu_hypre__kk++)\
   {\
      NALU_HYPRE_Int ii,jj,bi,xi;\
      for (jj = 0; jj < nj; jj++)\
      {\
         ii = (nalu_hypre__kk + jj + redblack) % 2;\
         bi = bstart + nalu_hypre__kk*bnj*bni + jj*bni + ii;\
         xi = xstart + nalu_hypre__kk*xnj*xni + jj*xni + ii;\
         for (; ii < ni; ii+=2, Ai+=2, bi+=2, xi+=2)\
         {

#define nalu_hypre_RedBlackConstantcoefLoopEnd()\
         }\
      }\
   }\
}
#endif
