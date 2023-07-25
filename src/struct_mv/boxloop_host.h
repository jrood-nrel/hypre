/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Header info for the BoxLoop
 *
 *****************************************************************************/

/*--------------------------------------------------------------------------
 * BoxLoop macros:
 *--------------------------------------------------------------------------*/

#ifndef NALU_HYPRE_BOXLOOP_HOST_HEADER
#define NALU_HYPRE_BOXLOOP_HOST_HEADER

#if defined(NALU_HYPRE_USING_OPENMP)
#define NALU_HYPRE_BOX_REDUCTION
#define NALU_HYPRE_OMP_CLAUSE
#if defined(WIN32) && defined(_MSC_VER)
#define Pragma(x) __pragma(NALU_HYPRE_XSTR(x))
#else
#define Pragma(x) _Pragma(NALU_HYPRE_XSTR(x))
#endif
#define OMP0 Pragma(omp parallel for NALU_HYPRE_OMP_CLAUSE NALU_HYPRE_BOX_REDUCTION NALU_HYPRE_SMP_SCHEDULE)
#define OMP1 Pragma(omp parallel for private(NALU_HYPRE_BOX_PRIVATE) NALU_HYPRE_OMP_CLAUSE NALU_HYPRE_BOX_REDUCTION NALU_HYPRE_SMP_SCHEDULE)
#else /* #if defined(NALU_HYPRE_USING_OPENMP) */
#define OMP0
#define OMP1
#endif /* #if defined(NALU_HYPRE_USING_OPENMP) */

#define zypre_BoxLoop0Begin(ndim, loop_size)                                  \
{                                                                             \
   zypre_BoxLoopDeclare();                                                    \
   zypre_BoxLoopInit(ndim, loop_size);                                        \
   OMP1                                                                       \
   for (nalu_hypre__block = 0; nalu_hypre__block < nalu_hypre__num_blocks; nalu_hypre__block++)   \
   {                                                                          \
      zypre_BoxLoopSet();                                                     \
      for (nalu_hypre__J = 0; nalu_hypre__J < nalu_hypre__JN; nalu_hypre__J++)                    \
      {                                                                       \
         for (nalu_hypre__I = 0; nalu_hypre__I < nalu_hypre__IN; nalu_hypre__I++)                 \
         {

#define zypre_BoxLoop0End()                                                   \
         }                                                                    \
         zypre_BoxLoopInc1();                                                 \
         zypre_BoxLoopInc2();                                                 \
      }                                                                       \
   }                                                                          \
}

#define zypre_BoxLoop1Begin(ndim, loop_size,                                  \
                            dbox1, start1, stride1, i1)                       \
{                                                                             \
   NALU_HYPRE_Int i1;                                                              \
   zypre_BoxLoopDeclare();                                                    \
   zypre_BoxLoopDeclareK(1);                                                  \
   zypre_BoxLoopInit(ndim, loop_size);                                        \
   zypre_BoxLoopInitK(1, dbox1, start1, stride1, i1);                         \
   OMP1                                                                       \
   for (nalu_hypre__block = 0; nalu_hypre__block < nalu_hypre__num_blocks; nalu_hypre__block++)   \
   {                                                                          \
      NALU_HYPRE_Int i1;                                                           \
      zypre_BoxLoopSet();                                                     \
      zypre_BoxLoopSetK(1, i1);                                               \
      for (nalu_hypre__J = 0; nalu_hypre__J < nalu_hypre__JN; nalu_hypre__J++)                    \
      {                                                                       \
         for (nalu_hypre__I = 0; nalu_hypre__I < nalu_hypre__IN; nalu_hypre__I++)                 \
         {

#define zypre_BoxLoop1End(i1)                                                 \
            i1 += nalu_hypre__i0inc1;                                              \
         }                                                                    \
         zypre_BoxLoopInc1();                                                 \
         i1 += nalu_hypre__ikinc1[nalu_hypre__d];                                       \
         zypre_BoxLoopInc2();                                                 \
      }                                                                       \
   }                                                                          \
}


#define zypre_BoxLoop2Begin(ndim, loop_size,                                  \
                            dbox1, start1, stride1, i1,                       \
                            dbox2, start2, stride2, i2)                       \
{                                                                             \
   NALU_HYPRE_Int i1, i2;                                                          \
   zypre_BoxLoopDeclare();                                                    \
   zypre_BoxLoopDeclareK(1);                                                  \
   zypre_BoxLoopDeclareK(2);                                                  \
   zypre_BoxLoopInit(ndim, loop_size);                                        \
   zypre_BoxLoopInitK(1, dbox1, start1, stride1, i1);                         \
   zypre_BoxLoopInitK(2, dbox2, start2, stride2, i2);                         \
   OMP1                                                                       \
   for (nalu_hypre__block = 0; nalu_hypre__block < nalu_hypre__num_blocks; nalu_hypre__block++)   \
   {                                                                          \
      NALU_HYPRE_Int i1, i2;                                                       \
      zypre_BoxLoopSet();                                                     \
      zypre_BoxLoopSetK(1, i1);                                               \
      zypre_BoxLoopSetK(2, i2);                                               \
      for (nalu_hypre__J = 0; nalu_hypre__J < nalu_hypre__JN; nalu_hypre__J++)                    \
      {                                                                       \
         for (nalu_hypre__I = 0; nalu_hypre__I < nalu_hypre__IN; nalu_hypre__I++)                 \
         {

#define zypre_BoxLoop2End(i1, i2)                                             \
            i1 += nalu_hypre__i0inc1;                                              \
            i2 += nalu_hypre__i0inc2;                                              \
         }                                                                    \
         zypre_BoxLoopInc1();                                                 \
         i1 += nalu_hypre__ikinc1[nalu_hypre__d];                                       \
         i2 += nalu_hypre__ikinc2[nalu_hypre__d];                                       \
         zypre_BoxLoopInc2();                                                 \
      }                                                                       \
   }                                                                          \
}


#define zypre_BoxLoop3Begin(ndim, loop_size,                                  \
                            dbox1, start1, stride1, i1,                       \
                            dbox2, start2, stride2, i2,                       \
                            dbox3, start3, stride3, i3)                       \
{                                                                             \
   NALU_HYPRE_Int i1, i2, i3;                                                      \
   zypre_BoxLoopDeclare();                                                    \
   zypre_BoxLoopDeclareK(1);                                                  \
   zypre_BoxLoopDeclareK(2);                                                  \
   zypre_BoxLoopDeclareK(3);                                                  \
   zypre_BoxLoopInit(ndim, loop_size);                                        \
   zypre_BoxLoopInitK(1, dbox1, start1, stride1, i1);                         \
   zypre_BoxLoopInitK(2, dbox2, start2, stride2, i2);                         \
   zypre_BoxLoopInitK(3, dbox3, start3, stride3, i3);                         \
   OMP1                                                                       \
   for (nalu_hypre__block = 0; nalu_hypre__block < nalu_hypre__num_blocks; nalu_hypre__block++)   \
   {                                                                          \
      NALU_HYPRE_Int i1, i2, i3;                                                   \
      zypre_BoxLoopSet();                                                     \
      zypre_BoxLoopSetK(1, i1);                                               \
      zypre_BoxLoopSetK(2, i2);                                               \
      zypre_BoxLoopSetK(3, i3);                                               \
      for (nalu_hypre__J = 0; nalu_hypre__J < nalu_hypre__JN; nalu_hypre__J++)                    \
      {                                                                       \
         for (nalu_hypre__I = 0; nalu_hypre__I < nalu_hypre__IN; nalu_hypre__I++)                 \
         {

#define zypre_BoxLoop3End(i1, i2, i3)                                         \
            i1 += nalu_hypre__i0inc1;                                              \
            i2 += nalu_hypre__i0inc2;                                              \
            i3 += nalu_hypre__i0inc3;                                              \
         }                                                                    \
         zypre_BoxLoopInc1();                                                 \
         i1 += nalu_hypre__ikinc1[nalu_hypre__d];                                       \
         i2 += nalu_hypre__ikinc2[nalu_hypre__d];                                       \
         i3 += nalu_hypre__ikinc3[nalu_hypre__d];                                       \
         zypre_BoxLoopInc2();                                                 \
      }                                                                       \
   }                                                                          \
}

#define zypre_BoxLoop4Begin(ndim, loop_size,                                  \
                            dbox1, start1, stride1, i1,                       \
                            dbox2, start2, stride2, i2,                       \
                            dbox3, start3, stride3, i3,                       \
                            dbox4, start4, stride4, i4)                       \
{                                                                             \
   NALU_HYPRE_Int i1, i2, i3, i4;                                                  \
   zypre_BoxLoopDeclare();                                                    \
   zypre_BoxLoopDeclareK(1);                                                  \
   zypre_BoxLoopDeclareK(2);                                                  \
   zypre_BoxLoopDeclareK(3);                                                  \
   zypre_BoxLoopDeclareK(4);                                                  \
   zypre_BoxLoopInit(ndim, loop_size);                                        \
   zypre_BoxLoopInitK(1, dbox1, start1, stride1, i1);                         \
   zypre_BoxLoopInitK(2, dbox2, start2, stride2, i2);                         \
   zypre_BoxLoopInitK(3, dbox3, start3, stride3, i3);                         \
   zypre_BoxLoopInitK(4, dbox4, start4, stride4, i4);                         \
   OMP1                                                                       \
   for (nalu_hypre__block = 0; nalu_hypre__block < nalu_hypre__num_blocks; nalu_hypre__block++)   \
   {                                                                          \
      NALU_HYPRE_Int i1, i2, i3, i4;                                               \
      zypre_BoxLoopSet();                                                     \
      zypre_BoxLoopSetK(1, i1);                                               \
      zypre_BoxLoopSetK(2, i2);                                               \
      zypre_BoxLoopSetK(3, i3);                                               \
      zypre_BoxLoopSetK(4, i4);                                               \
      for (nalu_hypre__J = 0; nalu_hypre__J < nalu_hypre__JN; nalu_hypre__J++)                    \
      {                                                                       \
         for (nalu_hypre__I = 0; nalu_hypre__I < nalu_hypre__IN; nalu_hypre__I++)                 \
         {

#define zypre_BoxLoop4End(i1, i2, i3, i4)                                     \
            i1 += nalu_hypre__i0inc1;                                              \
            i2 += nalu_hypre__i0inc2;                                              \
            i3 += nalu_hypre__i0inc3;                                              \
            i4 += nalu_hypre__i0inc4;                                              \
         }                                                                    \
         zypre_BoxLoopInc1();                                                 \
         i1 += nalu_hypre__ikinc1[nalu_hypre__d];                                       \
         i2 += nalu_hypre__ikinc2[nalu_hypre__d];                                       \
         i3 += nalu_hypre__ikinc3[nalu_hypre__d];                                       \
         i4 += nalu_hypre__ikinc4[nalu_hypre__d];                                       \
         zypre_BoxLoopInc2();                                                 \
      }                                                                       \
   }                                                                          \
}

#define zypre_BasicBoxLoop1Begin(ndim, loop_size,                             \
                                 stride1, i1)                                 \
{                                                                             \
   zypre_BoxLoopDeclare();                                                    \
   zypre_BoxLoopDeclareK(1);                                                  \
   zypre_BoxLoopInit(ndim, loop_size);                                        \
   zypre_BasicBoxLoopInitK(1, stride1);                                       \
   OMP1                                                                       \
   for (nalu_hypre__block = 0; nalu_hypre__block < nalu_hypre__num_blocks; nalu_hypre__block++)   \
   {                                                                          \
      NALU_HYPRE_Int i1;                                                           \
      zypre_BoxLoopSet();                                                     \
      zypre_BoxLoopSetK(1, i1);                                               \
      for (nalu_hypre__J = 0; nalu_hypre__J < nalu_hypre__JN; nalu_hypre__J++)                    \
      {                                                                       \
         for (nalu_hypre__I = 0; nalu_hypre__I < nalu_hypre__IN; nalu_hypre__I++)                 \
         {

#define zypre_BasicBoxLoop2Begin(ndim, loop_size,                             \
                                 stride1, i1,                                 \
                                 stride2, i2)                                 \
{                                                                             \
   zypre_BoxLoopDeclare();                                                    \
   zypre_BoxLoopDeclareK(1);                                                  \
   zypre_BoxLoopDeclareK(2);                                                  \
   zypre_BoxLoopInit(ndim, loop_size);                                        \
   zypre_BasicBoxLoopInitK(1, stride1);                                       \
   zypre_BasicBoxLoopInitK(2, stride2);                                       \
   OMP1                                                                       \
   for (nalu_hypre__block = 0; nalu_hypre__block < nalu_hypre__num_blocks; nalu_hypre__block++)   \
   {                                                                          \
      NALU_HYPRE_Int i1, i2;                                                       \
      zypre_BoxLoopSet();                                                     \
      zypre_BoxLoopSetK(1, i1);                                               \
      zypre_BoxLoopSetK(2, i2);                                               \
      for (nalu_hypre__J = 0; nalu_hypre__J < nalu_hypre__JN; nalu_hypre__J++)                    \
      {                                                                       \
         for (nalu_hypre__I = 0; nalu_hypre__I < nalu_hypre__IN; nalu_hypre__I++)                 \
         {


#define zypre_LoopBegin(size, idx)                                            \
{                                                                             \
   NALU_HYPRE_Int idx;                                                             \
   OMP0                                                                       \
   for (idx = 0; idx < size; idx ++)                                          \
   {

#define zypre_LoopEnd()                                                       \
   }                                                                          \
}

/*--------------------------------------------------------------------------
 * Serial BoxLoop macros:
 * [same as the ones above (without OMP and with SetOneBlock)]
 * TODO: combine them
 *--------------------------------------------------------------------------*/
#define nalu_hypre_SerialBoxLoop0Begin(ndim, loop_size)                            \
{                                                                             \
   zypre_BoxLoopDeclare();                                                    \
   zypre_BoxLoopInit(ndim, loop_size);                                        \
   zypre_BoxLoopSetOneBlock();                                                \
   for (nalu_hypre__block = 0; nalu_hypre__block < nalu_hypre__num_blocks; nalu_hypre__block++)   \
   {                                                                          \
      zypre_BoxLoopSet();                                                     \
      for (nalu_hypre__J = 0; nalu_hypre__J < nalu_hypre__JN; nalu_hypre__J++)                    \
      {                                                                       \
         for (nalu_hypre__I = 0; nalu_hypre__I < nalu_hypre__IN; nalu_hypre__I++)                 \
         {

#define nalu_hypre_SerialBoxLoop0End()                                             \
         }                                                                    \
         zypre_BoxLoopInc1();                                                 \
         zypre_BoxLoopInc2();                                                 \
      }                                                                       \
   }                                                                          \
}

#define nalu_hypre_SerialBoxLoop1Begin(ndim, loop_size,                            \
                                  dbox1, start1, stride1, i1)                 \
{                                                                             \
   NALU_HYPRE_Int i1;                                                              \
   zypre_BoxLoopDeclare();                                                    \
   zypre_BoxLoopDeclareK(1);                                                  \
   zypre_BoxLoopInit(ndim, loop_size);                                        \
   zypre_BoxLoopInitK(1, dbox1, start1, stride1, i1);                         \
   zypre_BoxLoopSetOneBlock();                                                \
   for (nalu_hypre__block = 0; nalu_hypre__block < nalu_hypre__num_blocks; nalu_hypre__block++)   \
   {                                                                          \
      zypre_BoxLoopSet();                                                     \
      zypre_BoxLoopSetK(1, i1);                                               \
      for (nalu_hypre__J = 0; nalu_hypre__J < nalu_hypre__JN; nalu_hypre__J++)                    \
      {                                                                       \
         for (nalu_hypre__I = 0; nalu_hypre__I < nalu_hypre__IN; nalu_hypre__I++)                 \
         {

#define nalu_hypre_SerialBoxLoop1End(i1)  zypre_BoxLoop1End(i1)

#define nalu_hypre_SerialBoxLoop2Begin(ndim, loop_size,                            \
                                  dbox1, start1, stride1, i1,                 \
                                  dbox2, start2, stride2, i2)                 \
{                                                                             \
   NALU_HYPRE_Int i1,i2;                                                           \
   zypre_BoxLoopDeclare();                                                    \
   zypre_BoxLoopDeclareK(1);                                                  \
   zypre_BoxLoopDeclareK(2);                                                  \
   zypre_BoxLoopInit(ndim, loop_size);                                        \
   zypre_BoxLoopInitK(1, dbox1, start1, stride1, i1);                         \
   zypre_BoxLoopInitK(2, dbox2, start2, stride2, i2);                         \
   zypre_BoxLoopSetOneBlock();                                                \
   for (nalu_hypre__block = 0; nalu_hypre__block < nalu_hypre__num_blocks; nalu_hypre__block++)   \
   {                                                                          \
      zypre_BoxLoopSet();                                                     \
      zypre_BoxLoopSetK(1, i1);                                               \
      zypre_BoxLoopSetK(2, i2);                                               \
      for (nalu_hypre__J = 0; nalu_hypre__J < nalu_hypre__JN; nalu_hypre__J++)                    \
      {                                                                       \
         for (nalu_hypre__I = 0; nalu_hypre__I < nalu_hypre__IN; nalu_hypre__I++)                 \
         {

#define nalu_hypre_SerialBoxLoop2End(i1, i2) zypre_BoxLoop2End(i1, i2)

/* Reduction BoxLoop1 */
#define zypre_BoxLoop1ReductionBegin(ndim, loop_size, dbox1, start1, stride1, i1, reducesum) \
        zypre_BoxLoop1Begin(ndim, loop_size, dbox1, start1, stride1, i1)

#define zypre_BoxLoop1ReductionEnd(i1, reducesum) zypre_BoxLoop1End(i1)

/* Reduction BoxLoop2 */
#define zypre_BoxLoop2ReductionBegin(ndim, loop_size, dbox1, start1, stride1, i1,            \
                                                      dbox2, start2, stride2, i2, reducesum) \
        zypre_BoxLoop2Begin(ndim, loop_size, dbox1, start1, stride1, i1,                     \
                                             dbox2, start2, stride2, i2)

#define zypre_BoxLoop2ReductionEnd(i1, i2, reducesum) zypre_BoxLoop2End(i1, i2)


/* Renaming */
#define nalu_hypre_BoxLoopGetIndexHost          zypre_BoxLoopGetIndex
#define nalu_hypre_BoxLoopBlockHost             zypre_BoxLoopBlock
#define nalu_hypre_BoxLoop0BeginHost            zypre_BoxLoop0Begin
#define nalu_hypre_BoxLoop0EndHost              zypre_BoxLoop0End
#define nalu_hypre_BoxLoop1BeginHost            zypre_BoxLoop1Begin
#define nalu_hypre_BoxLoop1EndHost              zypre_BoxLoop1End
#define nalu_hypre_BoxLoop2BeginHost            zypre_BoxLoop2Begin
#define nalu_hypre_BoxLoop2EndHost              zypre_BoxLoop2End
#define nalu_hypre_BoxLoop3BeginHost            zypre_BoxLoop3Begin
#define nalu_hypre_BoxLoop3EndHost              zypre_BoxLoop3End
#define nalu_hypre_BoxLoop4BeginHost            zypre_BoxLoop4Begin
#define nalu_hypre_BoxLoop4EndHost              zypre_BoxLoop4End
#define nalu_hypre_BasicBoxLoop1BeginHost       zypre_BasicBoxLoop1Begin
#define nalu_hypre_BasicBoxLoop2BeginHost       zypre_BasicBoxLoop2Begin
#define nalu_hypre_LoopBeginHost                zypre_LoopBegin
#define nalu_hypre_LoopEndHost                  zypre_LoopEnd
#define nalu_hypre_BoxLoop1ReductionBeginHost   zypre_BoxLoop1ReductionBegin
#define nalu_hypre_BoxLoop1ReductionEndHost     zypre_BoxLoop1ReductionEnd
#define nalu_hypre_BoxLoop2ReductionBeginHost   zypre_BoxLoop2ReductionBegin
#define nalu_hypre_BoxLoop2ReductionEndHost     zypre_BoxLoop2ReductionEnd

//TODO TEMP FIX
#if !defined(NALU_HYPRE_USING_RAJA) && !defined(NALU_HYPRE_USING_KOKKOS) && !defined(NALU_HYPRE_USING_CUDA) && !defined(NALU_HYPRE_USING_HIP) && !defined(NALU_HYPRE_USING_DEVICE_OPENMP) && !defined(NALU_HYPRE_USING_SYCL)
#define nalu_hypre_BoxLoopGetIndex          nalu_hypre_BoxLoopGetIndexHost
#define nalu_hypre_BoxLoopBlock()           0
#define nalu_hypre_BoxLoop0Begin            nalu_hypre_BoxLoop0BeginHost
#define nalu_hypre_BoxLoop0End              nalu_hypre_BoxLoop0EndHost
#define nalu_hypre_BoxLoop1Begin            nalu_hypre_BoxLoop1BeginHost
#define nalu_hypre_BoxLoop1End              nalu_hypre_BoxLoop1EndHost
#define nalu_hypre_BoxLoop2Begin            nalu_hypre_BoxLoop2BeginHost
#define nalu_hypre_BoxLoop2End              nalu_hypre_BoxLoop2EndHost
#define nalu_hypre_BoxLoop3Begin            nalu_hypre_BoxLoop3BeginHost
#define nalu_hypre_BoxLoop3End              nalu_hypre_BoxLoop3EndHost
#define nalu_hypre_BoxLoop4Begin            nalu_hypre_BoxLoop4BeginHost
#define nalu_hypre_BoxLoop4End              nalu_hypre_BoxLoop4EndHost
#define nalu_hypre_BasicBoxLoop1Begin       nalu_hypre_BasicBoxLoop1BeginHost
#define nalu_hypre_BasicBoxLoop2Begin       nalu_hypre_BasicBoxLoop2BeginHost
#define nalu_hypre_LoopBegin                nalu_hypre_LoopBeginHost
#define nalu_hypre_LoopEnd                  nalu_hypre_LoopEndHost
#define nalu_hypre_BoxLoop1ReductionBegin   nalu_hypre_BoxLoop1ReductionBeginHost
#define nalu_hypre_BoxLoop1ReductionEnd     nalu_hypre_BoxLoop1ReductionEndHost
#define nalu_hypre_BoxLoop2ReductionBegin   nalu_hypre_BoxLoop2ReductionBeginHost
#define nalu_hypre_BoxLoop2ReductionEnd     nalu_hypre_BoxLoop2ReductionEndHost
#endif

#endif /* #ifndef NALU_HYPRE_BOXLOOP_HOST_HEADER */

