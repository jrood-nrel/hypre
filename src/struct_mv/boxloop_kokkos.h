/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
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

#ifndef NALU_HYPRE_BOXLOOP_KOKKOS_HEADER
#define NALU_HYPRE_BOXLOOP_KOKKOS_HEADER

#if defined(NALU_HYPRE_USING_KOKKOS)

#ifdef __cplusplus
extern "C++"
{
#endif

#include <Kokkos_Core.hpp>
   using namespace Kokkos;

#ifdef __cplusplus
}
#endif

#if defined( KOKKOS_HAVE_MPI )
#include <mpi.h>
#endif

typedef struct nalu_hypre_Boxloop_struct
{
   NALU_HYPRE_Int lsize0, lsize1, lsize2;
   NALU_HYPRE_Int strides0, strides1, strides2;
   NALU_HYPRE_Int bstart0, bstart1, bstart2;
   NALU_HYPRE_Int bsize0, bsize1, bsize2;
} nalu_hypre_Boxloop;


#define nalu_hypre_fence()
/*
#define nalu_hypre_fence()                                \
   cudaError err = cudaGetLastError();               \
   if ( cudaSuccess != err ) {                                                \
     printf("\n ERROR nalu_hypre_newBoxLoop: %s in %s(%d) function %s\n", cudaGetErrorString(err),__FILE__,__LINE__,__FUNCTION__); \
   }                                                                        \
   nalu_hypre_CheckErrorDevice(cudaDeviceSynchronize());
*/


#define nalu_hypre_newBoxLoopInit(ndim,loop_size)                                \
   NALU_HYPRE_Int nalu_hypre__tot = 1;                                                \
   for (NALU_HYPRE_Int d = 0;d < ndim;d ++)                                      \
      nalu_hypre__tot *= loop_size[d];


#define nalu_hypre_BoxLoopIncK(k,box,nalu_hypre__i)                                                \
   NALU_HYPRE_Int nalu_hypre_boxD##k = 1;                                                          \
   NALU_HYPRE_Int nalu_hypre__i = 0;                                                               \
   nalu_hypre__i += (nalu_hypre_IndexD(local_idx, 0)*box.strides0 + box.bstart0) * nalu_hypre_boxD##k;  \
   nalu_hypre_boxD##k *= nalu_hypre_max(0, box.bsize0 + 1);                                        \
   nalu_hypre__i += (nalu_hypre_IndexD(local_idx, 1)*box.strides1 + box.bstart1) * nalu_hypre_boxD##k;  \
   nalu_hypre_boxD##k *= nalu_hypre_max(0, box.bsize1 + 1);                                        \
   nalu_hypre__i += (nalu_hypre_IndexD(local_idx, 2)*box.strides2 + box.bstart2) * nalu_hypre_boxD##k;  \
   nalu_hypre_boxD##k *= nalu_hypre_max(0, box.bsize2 + 1);                                        \

#define nalu_hypre_newBoxLoopDeclare(box)                                        \
  nalu_hypre_Index local_idx;                                                    \
  NALU_HYPRE_Int idx_local = idx;                                                \
  nalu_hypre_IndexD(local_idx, 0)  = idx_local % box.lsize0;                     \
  idx_local = idx_local / box.lsize0;                                       \
  nalu_hypre_IndexD(local_idx, 1)  = idx_local % box.lsize1;                     \
  idx_local = idx_local / box.lsize1;                                       \
  nalu_hypre_IndexD(local_idx, 2)  = idx_local % box.lsize2;

#define nalu_hypre_BoxLoopDataDeclareK(k,ndim,loop_size,dbox,start,stride)       \
   nalu_hypre_Boxloop databox##k;                                                \
   databox##k.lsize0 = loop_size[0];                                        \
   databox##k.strides0 = stride[0];                                         \
   databox##k.bstart0  = start[0] - dbox->imin[0];                          \
   databox##k.bsize0   = dbox->imax[0]-dbox->imin[0];                       \
   if (ndim > 1)                                                            \
   {                                                                        \
      databox##k.lsize1 = loop_size[1];                                     \
      databox##k.strides1 = stride[1];                                      \
      databox##k.bstart1  = start[1] - dbox->imin[1];                       \
      databox##k.bsize1   = dbox->imax[1]-dbox->imin[1];                    \
   }                                                                        \
   else                                                                     \
   {                                                                        \
      databox##k.lsize1 = 1;                                                \
      databox##k.strides1 = 0;                                              \
      databox##k.bstart1  = 0;                                              \
      databox##k.bsize1   = 0;                                              \
   }                                                                        \
   if (ndim == 3)                                                           \
   {                                                                        \
      databox##k.lsize2 = loop_size[2];                                     \
      databox##k.strides2 = stride[2];                                      \
      databox##k.bstart2  = start[2] - dbox->imin[2];                       \
      databox##k.bsize2   = dbox->imax[2]-dbox->imin[2];                    \
   }                                                                        \
   else                                                                     \
   {                                                                        \
      databox##k.lsize2 = 1;                                                \
      databox##k.strides2 = 0;                                              \
      databox##k.bstart2  = 0;                                              \
      databox##k.bsize2   = 0;                                              \
   }

#define nalu_hypre_newBoxLoop0Begin(ndim, loop_size)                         \
{                                                                       \
   nalu_hypre_newBoxLoopInit(ndim,loop_size);                                \
   Kokkos::parallel_for (nalu_hypre__tot, KOKKOS_LAMBDA (NALU_HYPRE_Int idx)      \
   {


#define nalu_hypre_newBoxLoop0End(i1)                                        \
   });                                                                  \
   nalu_hypre_fence();                                                       \
}


#define nalu_hypre_newBoxLoop1Begin(ndim, loop_size,                         \
                               dbox1, start1, stride1, i1)              \
{                                                                       \
   nalu_hypre_newBoxLoopInit(ndim,loop_size)                                 \
   nalu_hypre_BoxLoopDataDeclareK(1,ndim,loop_size,dbox1,start1,stride1);    \
   Kokkos::parallel_for (nalu_hypre__tot, KOKKOS_LAMBDA (NALU_HYPRE_Int idx)      \
   {                                                                    \
      nalu_hypre_newBoxLoopDeclare(databox1);                                \
      nalu_hypre_BoxLoopIncK(1,databox1,i1);


#define nalu_hypre_newBoxLoop1End(i1)                                        \
   });                                                                  \
     nalu_hypre_fence();                                                     \
 }


#define nalu_hypre_newBoxLoop2Begin(ndim, loop_size,                         \
                               dbox1, start1, stride1, i1,              \
                               dbox2, start2, stride2, i2)              \
{                                                                       \
   nalu_hypre_newBoxLoopInit(ndim,loop_size);                                \
   nalu_hypre_BoxLoopDataDeclareK(1,ndim,loop_size,dbox1,start1,stride1);    \
   nalu_hypre_BoxLoopDataDeclareK(2,ndim,loop_size,dbox2,start2,stride2);    \
   Kokkos::parallel_for (nalu_hypre__tot, KOKKOS_LAMBDA (NALU_HYPRE_Int idx)      \
   {                                                                    \
      nalu_hypre_newBoxLoopDeclare(databox1)                                 \
      nalu_hypre_BoxLoopIncK(1,databox1,i1);                                 \
      nalu_hypre_BoxLoopIncK(2,databox2,i2);

#define nalu_hypre_newBoxLoop2End(i1, i2)                                    \
   });                                                                  \
   nalu_hypre_fence();                                                       \
}


#define nalu_hypre_newBoxLoop3Begin(ndim, loop_size,                         \
                               dbox1, start1, stride1, i1,              \
                               dbox2, start2, stride2, i2,              \
                               dbox3, start3, stride3, i3)              \
{                                                                       \
   nalu_hypre_newBoxLoopInit(ndim,loop_size);                                \
   nalu_hypre_BoxLoopDataDeclareK(1,ndim,loop_size,dbox1,start1,stride1);    \
   nalu_hypre_BoxLoopDataDeclareK(2,ndim,loop_size,dbox2,start2,stride2);    \
   nalu_hypre_BoxLoopDataDeclareK(3,ndim,loop_size,dbox3,start3,stride3);    \
   Kokkos::parallel_for (nalu_hypre__tot, KOKKOS_LAMBDA (NALU_HYPRE_Int idx)      \
   {                                                                    \
      nalu_hypre_newBoxLoopDeclare(databox1);                                \
      nalu_hypre_BoxLoopIncK(1,databox1,i1);                                 \
      nalu_hypre_BoxLoopIncK(2,databox2,i2);                                 \
      nalu_hypre_BoxLoopIncK(3,databox3,i3);

#define nalu_hypre_newBoxLoop3End(i1, i2, i3)                                \
   });                                                                  \
   nalu_hypre_fence();                                                       \
}

#define nalu_hypre_newBoxLoop4Begin(ndim, loop_size,                         \
                               dbox1, start1, stride1, i1,              \
                               dbox2, start2, stride2, i2,              \
                               dbox3, start3, stride3, i3,              \
                               dbox4, start4, stride4, i4)              \
{                                                                       \
   nalu_hypre_newBoxLoopInit(ndim,loop_size);                                \
   nalu_hypre_BoxLoopDataDeclareK(1,ndim,loop_size,dbox1,start1,stride1);    \
   nalu_hypre_BoxLoopDataDeclareK(2,ndim,loop_size,dbox2,start2,stride2);    \
   nalu_hypre_BoxLoopDataDeclareK(3,ndim,loop_size,dbox3,start3,stride3);    \
   nalu_hypre_BoxLoopDataDeclareK(4,ndim,loop_size,dbox4,start4,stride4);    \
   Kokkos::parallel_for (nalu_hypre__tot, KOKKOS_LAMBDA (NALU_HYPRE_Int idx)      \
   {                                                                    \
      nalu_hypre_newBoxLoopDeclare(databox1);                                \
      nalu_hypre_BoxLoopIncK(1,databox1,i1);                                 \
      nalu_hypre_BoxLoopIncK(2,databox2,i2);                                 \
      nalu_hypre_BoxLoopIncK(3,databox3,i3);                                 \
      nalu_hypre_BoxLoopIncK(4,databox4,i4);


#define nalu_hypre_newBoxLoop4End(i1, i2, i3, i4)                            \
   });                                                                  \
   nalu_hypre_fence();                                                       \
}

#define nalu_hypre_BasicBoxLoopDataDeclareK(k,ndim,loop_size,stride)         \
        nalu_hypre_Boxloop databox##k;                                       \
        databox##k.lsize0 = loop_size[0];                               \
        databox##k.strides0 = stride[0];                                \
        databox##k.bstart0  = 0;                                        \
        databox##k.bsize0   = 0;                                        \
        if (ndim > 1)                                                   \
        {                                                               \
            databox##k.lsize1 = loop_size[1];                           \
            databox##k.strides1 = stride[1];                            \
            databox##k.bstart1  = 0;                                    \
            databox##k.bsize1   = 0;                                    \
        }                                                               \
        else                                                            \
        {                                                               \
                databox##k.lsize1 = 1;                                  \
                databox##k.strides1 = 0;                                \
                databox##k.bstart1  = 0;                                \
                databox##k.bsize1   = 0;                                \
        }                                                               \
        if (ndim == 3)                                                  \
        {                                                               \
            databox##k.lsize2 = loop_size[2];                           \
            databox##k.strides2 = stride[2];                            \
            databox##k.bstart2  = 0;                                    \
            databox##k.bsize2   = 0;                                    \
        }                                                               \
        else                                                            \
        {                                                               \
            databox##k.lsize2 = 1;                                      \
            databox##k.strides2 = 0;                                    \
            databox##k.bstart2  = 0;                                    \
            databox##k.bsize2   = 0;                                    \
        }

#define nalu_hypre_newBasicBoxLoop2Begin(ndim, loop_size,                    \
                                    stride1, i1,                        \
                                    stride2, i2)                        \
{                                                                       \
   nalu_hypre_newBoxLoopInit(ndim,loop_size);                                \
   nalu_hypre_BasicBoxLoopDataDeclareK(1,ndim,loop_size,stride1);            \
   nalu_hypre_BasicBoxLoopDataDeclareK(2,ndim,loop_size,stride2);            \
   Kokkos::parallel_for (nalu_hypre__tot, KOKKOS_LAMBDA (NALU_HYPRE_Int idx)      \
   {                                                                    \
      nalu_hypre_newBoxLoopDeclare(databox1);                                \
      nalu_hypre_BoxLoopIncK(1,databox1,i1);                                 \
      nalu_hypre_BoxLoopIncK(2,databox2,i2);                                 \

#define nalu_hypre_BoxLoop1ReductionBegin(ndim, loop_size,                   \
                                     dbox1, start1, stride1, i1,        \
                                     NALU_HYPRE_BOX_REDUCTION)               \
 {                                                                      \
     NALU_HYPRE_Real __nalu_hypre_sum_tmp = NALU_HYPRE_BOX_REDUCTION;                  \
     NALU_HYPRE_BOX_REDUCTION = 0.0;                                         \
     nalu_hypre_newBoxLoopInit(ndim,loop_size);                              \
     nalu_hypre_BoxLoopDataDeclareK(1,ndim,loop_size,dbox1,start1,stride1);  \
     Kokkos::parallel_reduce (nalu_hypre__tot, KOKKOS_LAMBDA (NALU_HYPRE_Int idx, \
                              NALU_HYPRE_Real &NALU_HYPRE_BOX_REDUCTION)          \
     {                                                                  \
        nalu_hypre_newBoxLoopDeclare(databox1);                              \
        nalu_hypre_BoxLoopIncK(1,databox1,i1);                               \



#define nalu_hypre_BoxLoop1ReductionEnd(i1, NALU_HYPRE_BOX_REDUCTION)            \
     }, NALU_HYPRE_BOX_REDUCTION);                                           \
     nalu_hypre_fence();                                                     \
     NALU_HYPRE_BOX_REDUCTION += __nalu_hypre_sum_tmp;                            \
 }

#define nalu_hypre_BoxLoop2ReductionBegin(ndim, loop_size,                  \
                                      dbox1, start1, stride1, i1,       \
                                      dbox2, start2, stride2, i2,       \
                                      NALU_HYPRE_BOX_REDUCTION)              \
 {                                                                      \
     NALU_HYPRE_Real __nalu_hypre_sum_tmp = NALU_HYPRE_BOX_REDUCTION;                  \
     NALU_HYPRE_BOX_REDUCTION = 0.0;                                         \
     nalu_hypre_newBoxLoopInit(ndim,loop_size);                              \
     nalu_hypre_BoxLoopDataDeclareK(1,ndim,loop_size,dbox1,start1,stride1);  \
     nalu_hypre_BoxLoopDataDeclareK(2,ndim,loop_size,dbox2,start2,stride2);  \
     Kokkos::parallel_reduce (nalu_hypre__tot, KOKKOS_LAMBDA (NALU_HYPRE_Int idx, \
                              NALU_HYPRE_Real &NALU_HYPRE_BOX_REDUCTION)          \
     {                                                                  \
         nalu_hypre_newBoxLoopDeclare(databox1);                             \
         nalu_hypre_BoxLoopIncK(1,databox1,i1);                              \
         nalu_hypre_BoxLoopIncK(2,databox2,i2);                              \

#define nalu_hypre_BoxLoop2ReductionEnd(i1, i2, NALU_HYPRE_BOX_REDUCTION)        \
     }, NALU_HYPRE_BOX_REDUCTION);                                           \
     nalu_hypre_fence();                                                     \
     NALU_HYPRE_BOX_REDUCTION += __nalu_hypre_sum_tmp;                            \
 }

#define nalu_hypre_LoopBegin(size,idx)                                       \
{                                                                       \
   Kokkos::parallel_for(size, KOKKOS_LAMBDA (NALU_HYPRE_Int idx)             \
   {

#define nalu_hypre_LoopEnd()                                                 \
   });                                                                  \
   nalu_hypre_fence();                                                       \
}

/*
extern "C++"
{
struct ColumnSums
{
  typedef NALU_HYPRE_Real value_type[];
  typedef View<NALU_HYPRE_Real**>::size_type size_type;
  size_type value_count;
  View<NALU_HYPRE_Real**> X_;
  ColumnSums(const View<NALU_HYPRE_Real**>& X):value_count(X.dimension_1()),X_(X){}
  KOKKOS_INLINE_FUNCTION void
  operator()(const size_type i,value_type sum) const
  {
    for (size_type j = 0;j < value_count;j++)
    {
       sum[j] += X_(i,j);
    }
  }
  KOKKOS_INLINE_FUNCTION void
  join (volatile value_type dst,volatile value_type src) const
  {
    for (size_type j= 0;j < value_count;j++)
    {
      dst[j] +=src[j];
    }
  }
  KOKKOS_INLINE_FUNCTION void init(value_type sum) const
  {
    for (size_type j= 0;j < value_count;j++)
    {
      sum[j] += 0.0;
    }
  }
};
}
*/

#define nalu_hypre_BoxLoopGetIndex(index)     \
  index[0] = nalu_hypre_IndexD(local_idx, 0); \
  index[1] = nalu_hypre_IndexD(local_idx, 1); \
  index[2] = nalu_hypre_IndexD(local_idx, 2);

#define nalu_hypre_BoxLoopBlock()       0
#define nalu_hypre_BoxLoop0Begin      nalu_hypre_newBoxLoop0Begin
#define nalu_hypre_BoxLoop0For        nalu_hypre_newBoxLoop0For
#define nalu_hypre_BoxLoop0End        nalu_hypre_newBoxLoop0End
#define nalu_hypre_BoxLoop1Begin      nalu_hypre_newBoxLoop1Begin
#define nalu_hypre_BoxLoop1For        nalu_hypre_newBoxLoop1For
#define nalu_hypre_BoxLoop1End        nalu_hypre_newBoxLoop1End
#define nalu_hypre_BoxLoop2Begin      nalu_hypre_newBoxLoop2Begin
#define nalu_hypre_BoxLoop2For        nalu_hypre_newBoxLoop2For
#define nalu_hypre_BoxLoop2End        nalu_hypre_newBoxLoop2End
#define nalu_hypre_BoxLoop3Begin      nalu_hypre_newBoxLoop3Begin
#define nalu_hypre_BoxLoop3For        nalu_hypre_newBoxLoop3For
#define nalu_hypre_BoxLoop3End        nalu_hypre_newBoxLoop3End
#define nalu_hypre_BoxLoop4Begin      nalu_hypre_newBoxLoop4Begin
#define nalu_hypre_BoxLoop4For        nalu_hypre_newBoxLoop4For
#define nalu_hypre_BoxLoop4End        nalu_hypre_newBoxLoop4End

#define nalu_hypre_BasicBoxLoop2Begin nalu_hypre_newBasicBoxLoop2Begin

#endif

#endif /* #ifndef NALU_HYPRE_BOXLOOP_KOKKOS_HEADER */

