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

#ifndef NALU_HYPRE_BOXLOOP_SYCL_HEADER
#define NALU_HYPRE_BOXLOOP_SYCL_HEADER

#if defined(NALU_HYPRE_USING_SYCL) && !defined(NALU_HYPRE_USING_RAJA) && !defined(NALU_HYPRE_USING_KOKKOS)

typedef struct nalu_hypre_Boxloop_struct
{
   NALU_HYPRE_Int lsize0, lsize1, lsize2;
   NALU_HYPRE_Int strides0, strides1, strides2;
   NALU_HYPRE_Int bstart0, bstart1, bstart2;
   NALU_HYPRE_Int bsize0, bsize1, bsize2;
} nalu_hypre_Boxloop;


#ifdef __cplusplus
extern "C++"
{
#endif

   /*********************************************************************
    * wrapper functions calling sycl parallel_for
    * WM: todo - add runtime switch between CPU/GPU execution
    *********************************************************************/

   template<typename LOOP_BODY>
   void
   BoxLoopforall( NALU_HYPRE_Int length,
                  LOOP_BODY loop_body)
   {
      if (length <= 0)
      {
         return;
      }
      dim3 bDim = nalu_hypre_GetDefaultDeviceBlockDimension();
      dim3 gDim = nalu_hypre_GetDefaultDeviceGridDimension(length, "thread", bDim);

      nalu_hypre_HandleComputeStream(nalu_hypre_handle())->submit([&] (sycl::handler & cgh)
      {
         cgh.parallel_for(sycl::nd_range<3>(gDim * bDim, bDim), loop_body);
      }).wait_and_throw();
   }

   template<typename LOOP_BODY>
   void
   ReductionBoxLoopforall( LOOP_BODY  loop_body,
                           NALU_HYPRE_Int length,
                           NALU_HYPRE_Real * shared_sum_var )
   {
      if (length <= 0)
      {
         return;
      }
      dim3 bDim = nalu_hypre_GetDefaultDeviceBlockDimension();
      dim3 gDim = nalu_hypre_GetDefaultDeviceGridDimension(length, "thread", bDim);

      nalu_hypre_HandleComputeStream(nalu_hypre_handle())->submit([&] (sycl::handler & cgh)
      {
         cgh.parallel_for(sycl::nd_range<3>(gDim * bDim, bDim), sycl::reduction(shared_sum_var,
                                                                                std::plus<>()), loop_body);
      }).wait_and_throw();
   }

#ifdef __cplusplus
}
#endif


/*********************************************************************
 * Init/Declare/IncK etc.
 *********************************************************************/

/* Get 1-D length of the loop, in nalu_hypre__tot */
#define nalu_hypre_newBoxLoopInit(ndim, loop_size)              \
   NALU_HYPRE_Int nalu_hypre__tot = 1;                               \
   for (NALU_HYPRE_Int nalu_hypre_d = 0; nalu_hypre_d < ndim; nalu_hypre_d ++) \
   {                                                       \
      nalu_hypre__tot *= loop_size[nalu_hypre_d];                    \
   }

/* Initialize struct for box-k */
#define nalu_hypre_BoxLoopDataDeclareK(k, ndim, loop_size, dbox, start, stride) \
   nalu_hypre_Boxloop databox##k;                                               \
   databox##k.lsize0   = loop_size[0];                                     \
   databox##k.strides0 = stride[0];                                        \
   databox##k.bstart0  = start[0] - dbox->imin[0];                         \
   databox##k.bsize0   = dbox->imax[0] - dbox->imin[0];                    \
   if (ndim > 1)                                                           \
   {                                                                       \
      databox##k.lsize1   = loop_size[1];                                  \
      databox##k.strides1 = stride[1];                                     \
      databox##k.bstart1  = start[1] - dbox->imin[1];                      \
      databox##k.bsize1   = dbox->imax[1] - dbox->imin[1];                 \
   }                                                                       \
   else                                                                    \
   {                                                                       \
      databox##k.lsize1   = 1;                                             \
      databox##k.strides1 = 0;                                             \
      databox##k.bstart1  = 0;                                             \
      databox##k.bsize1   = 0;                                             \
   }                                                                       \
   if (ndim == 3)                                                          \
   {                                                                       \
      databox##k.lsize2   = loop_size[2];                                  \
      databox##k.strides2 = stride[2];                                     \
      databox##k.bstart2  = start[2] - dbox->imin[2];                      \
      databox##k.bsize2   = dbox->imax[2] - dbox->imin[2];                 \
   }                                                                       \
   else                                                                    \
   {                                                                       \
      databox##k.lsize2   = 1;                                             \
      databox##k.strides2 = 0;                                             \
      databox##k.bstart2  = 0;                                             \
      databox##k.bsize2   = 0;                                             \
   }

#define nalu_hypre_BasicBoxLoopDataDeclareK(k,ndim,loop_size,stride) \
nalu_hypre_Boxloop databox##k;                                       \
databox##k.lsize0   = loop_size[0];                             \
databox##k.strides0 = stride[0];                                \
databox##k.bstart0  = 0;                                        \
databox##k.bsize0   = 0;                                        \
if (ndim > 1)                                                   \
{                                                               \
   databox##k.lsize1   = loop_size[1];                          \
   databox##k.strides1 = stride[1];                             \
   databox##k.bstart1  = 0;                                     \
   databox##k.bsize1   = 0;                                     \
}                                                               \
else                                                            \
{                                                               \
   databox##k.lsize1   = 1;                                     \
   databox##k.strides1 = 0;                                     \
   databox##k.bstart1  = 0;                                     \
   databox##k.bsize1   = 0;                                     \
}                                                               \
if (ndim == 3)                                                  \
{                                                               \
   databox##k.lsize2   = loop_size[2];                          \
   databox##k.strides2 = stride[2];                             \
   databox##k.bstart2  = 0;                                     \
   databox##k.bsize2   = 0;                                     \
}                                                               \
else                                                            \
{                                                               \
    databox##k.lsize2   = 1;                                    \
    databox##k.strides2 = 0;                                    \
    databox##k.bstart2  = 0;                                    \
    databox##k.bsize2   = 0;                                    \
}

/* Given input 1-D 'idx' in box, get 3-D 'local_idx' in loop_size */
#define nalu_hypre_newBoxLoopDeclare(box)                     \
   nalu_hypre_Index local_idx;                                \
   NALU_HYPRE_Int idx_local = idx;                            \
   nalu_hypre_IndexD(local_idx, 0)  = idx_local % box.lsize0; \
   idx_local = idx_local / box.lsize0;                   \
   nalu_hypre_IndexD(local_idx, 1)  = idx_local % box.lsize1; \
   idx_local = idx_local / box.lsize1;                   \
   nalu_hypre_IndexD(local_idx, 2)  = idx_local % box.lsize2; \

/* Given input 3-D 'local_idx', get 1-D 'nalu_hypre__i' in 'box' */
#define nalu_hypre_BoxLoopIncK(k, box, nalu_hypre__i)                                               \
   NALU_HYPRE_Int nalu_hypre_boxD##k = 1;                                                           \
   NALU_HYPRE_Int nalu_hypre__i = 0;                                                                \
   nalu_hypre__i += (nalu_hypre_IndexD(local_idx, 0) * box.strides0 + box.bstart0) * nalu_hypre_boxD##k; \
   nalu_hypre_boxD##k *= nalu_hypre_max(0, box.bsize0 + 1);                                         \
   nalu_hypre__i += (nalu_hypre_IndexD(local_idx, 1) * box.strides1 + box.bstart1) * nalu_hypre_boxD##k; \
   nalu_hypre_boxD##k *= nalu_hypre_max(0, box.bsize1 + 1);                                         \
   nalu_hypre__i += (nalu_hypre_IndexD(local_idx, 2) * box.strides2 + box.bstart2) * nalu_hypre_boxD##k; \
   nalu_hypre_boxD##k *= nalu_hypre_max(0, box.bsize2 + 1);

/* get 3-D local_idx into 'index' */
#define nalu_hypre_BoxLoopGetIndex(index)      \
   index[0] = nalu_hypre_IndexD(local_idx, 0); \
   index[1] = nalu_hypre_IndexD(local_idx, 1); \
   index[2] = nalu_hypre_IndexD(local_idx, 2);


/*********************************************************************
 * Boxloops
 *********************************************************************/

/* BoxLoop 0 */
#define nalu_hypre_newBoxLoop0Begin(ndim, loop_size)                                                       \
{                                                                                                     \
   nalu_hypre_newBoxLoopInit(ndim, loop_size);                                                             \
   BoxLoopforall(nalu_hypre__tot, [=] (sycl::nd_item<3> item)                                              \
   {                                                                                                  \
      NALU_HYPRE_Int idx = (NALU_HYPRE_Int) item.get_global_linear_id();                                        \
      if (idx < nalu_hypre__tot)                                                                           \
      {                                                                                               \

#define nalu_hypre_newBoxLoop0End()                                                                        \
      }                                                                                               \
   });                                                                                                \
}

/* BoxLoop 1 */
#define nalu_hypre_newBoxLoop1Begin(ndim, loop_size, dbox1, start1, stride1, i1)                           \
{                                                                                                     \
   nalu_hypre_newBoxLoopInit(ndim, loop_size);                                                             \
   nalu_hypre_BoxLoopDataDeclareK(1, ndim, loop_size, dbox1, start1, stride1);                             \
   BoxLoopforall(nalu_hypre__tot, [=] (sycl::nd_item<3> item)                                              \
   {                                                                                                  \
      NALU_HYPRE_Int idx = (NALU_HYPRE_Int) item.get_global_linear_id();                                        \
      if (idx < nalu_hypre__tot)                                                                           \
      {                                                                                               \
         nalu_hypre_newBoxLoopDeclare(databox1);                                                           \
         nalu_hypre_BoxLoopIncK(1, databox1, i1);

#define nalu_hypre_newBoxLoop1End(i1)                                                                      \
      }                                                                                               \
   });                                                                                                \
}

/* BoxLoop 2 */
#define nalu_hypre_newBoxLoop2Begin(ndim, loop_size, dbox1, start1, stride1, i1,                           \
                                                dbox2, start2, stride2, i2)                           \
{                                                                                                     \
   nalu_hypre_newBoxLoopInit(ndim, loop_size);                                                             \
   nalu_hypre_BoxLoopDataDeclareK(1, ndim, loop_size, dbox1, start1, stride1);                             \
   nalu_hypre_BoxLoopDataDeclareK(2, ndim, loop_size, dbox2, start2, stride2);                             \
   BoxLoopforall(nalu_hypre__tot, [=] (sycl::nd_item<3> item)                                              \
   {                                                                                                  \
      NALU_HYPRE_Int idx = (NALU_HYPRE_Int) item.get_global_linear_id();                                        \
      if (idx < nalu_hypre__tot)                                                                           \
      {                                                                                               \
         nalu_hypre_newBoxLoopDeclare(databox1);                                                           \
         nalu_hypre_BoxLoopIncK(1, databox1, i1);                                                          \
         nalu_hypre_BoxLoopIncK(2, databox2, i2);

#define nalu_hypre_newBoxLoop2End(i1, i2)                                                                  \
      }                                                                                               \
   });                                                                                                \
}

/* BoxLoop 3 */
#define nalu_hypre_newBoxLoop3Begin(ndim, loop_size, dbox1, start1, stride1, i1,                           \
                                                dbox2, start2, stride2, i2,                           \
                                                dbox3, start3, stride3, i3)                           \
{                                                                                                     \
   nalu_hypre_newBoxLoopInit(ndim, loop_size);                                                             \
   nalu_hypre_BoxLoopDataDeclareK(1, ndim,loop_size, dbox1, start1, stride1);                              \
   nalu_hypre_BoxLoopDataDeclareK(2, ndim,loop_size, dbox2, start2, stride2);                              \
   nalu_hypre_BoxLoopDataDeclareK(3, ndim,loop_size, dbox3, start3, stride3);                              \
   BoxLoopforall(nalu_hypre__tot, [=] (sycl::nd_item<3> item)                                              \
   {                                                                                                  \
      NALU_HYPRE_Int idx = (NALU_HYPRE_Int) item.get_global_linear_id();                                        \
      if (idx < nalu_hypre__tot)                                                                           \
      {                                                                                               \
         nalu_hypre_newBoxLoopDeclare(databox1);                                                           \
         nalu_hypre_BoxLoopIncK(1, databox1, i1);                                                          \
         nalu_hypre_BoxLoopIncK(2, databox2, i2);                                                          \
         nalu_hypre_BoxLoopIncK(3, databox3, i3);

#define nalu_hypre_newBoxLoop3End(i1, i2, i3)                                                              \
      }                                                                                               \
   });                                                                                                \
}

/* BoxLoop 4 */
#define nalu_hypre_newBoxLoop4Begin(ndim, loop_size, dbox1, start1, stride1, i1,                           \
                                                dbox2, start2, stride2, i2,                           \
                                                dbox3, start3, stride3, i3,                           \
                                                dbox4, start4, stride4, i4)                           \
{                                                                                                     \
   nalu_hypre_newBoxLoopInit(ndim, loop_size);                                                             \
   nalu_hypre_BoxLoopDataDeclareK(1, ndim, loop_size, dbox1, start1, stride1);                             \
   nalu_hypre_BoxLoopDataDeclareK(2, ndim, loop_size, dbox2, start2, stride2);                             \
   nalu_hypre_BoxLoopDataDeclareK(3, ndim, loop_size, dbox3, start3, stride3);                             \
   nalu_hypre_BoxLoopDataDeclareK(4, ndim, loop_size, dbox4, start4, stride4);                             \
   BoxLoopforall(nalu_hypre__tot, [=] (sycl::nd_item<3> item)                                              \
   {                                                                                                  \
      NALU_HYPRE_Int idx = (NALU_HYPRE_Int) item.get_global_linear_id();                                        \
      if (idx < nalu_hypre__tot)                                                                           \
      {                                                                                               \
         nalu_hypre_newBoxLoopDeclare(databox1);                                                           \
         nalu_hypre_BoxLoopIncK(1, databox1, i1);                                                          \
         nalu_hypre_BoxLoopIncK(2, databox2, i2);                                                          \
         nalu_hypre_BoxLoopIncK(3, databox3, i3);                                                          \
         nalu_hypre_BoxLoopIncK(4, databox4, i4);

#define nalu_hypre_newBoxLoop4End(i1, i2, i3, i4)                                                          \
      }                                                                                               \
   });                                                                                                \
}


/* Basic BoxLoops have no boxes */
/* BoxLoop 1 */
#define nalu_hypre_newBasicBoxLoop1Begin(ndim, loop_size, stride1, i1)                                     \
{                                                                                                     \
   nalu_hypre_newBoxLoopInit(ndim, loop_size);                                                             \
   nalu_hypre_BasicBoxLoopDataDeclareK(1, ndim, loop_size, stride1);                                       \
   BoxLoopforall(nalu_hypre__tot, [=] (sycl::nd_item<3> item)                                              \
   {                                                                                                  \
      NALU_HYPRE_Int idx = (NALU_HYPRE_Int) item.get_global_linear_id();                                        \
      if (idx < nalu_hypre__tot)                                                                           \
      {                                                                                               \
         nalu_hypre_newBoxLoopDeclare(databox1);                                                           \
         nalu_hypre_BoxLoopIncK(1, databox1, i1);

/* BoxLoop 2 */
#define nalu_hypre_newBasicBoxLoop2Begin(ndim, loop_size, stride1, i1, stride2, i2)                        \
{                                                                                                     \
   nalu_hypre_newBoxLoopInit(ndim, loop_size);                                                             \
   nalu_hypre_BasicBoxLoopDataDeclareK(1, ndim, loop_size, stride1);                                       \
   nalu_hypre_BasicBoxLoopDataDeclareK(2, ndim, loop_size, stride2);                                       \
   BoxLoopforall(nalu_hypre__tot, [=] (sycl::nd_item<3> item)                                              \
   {                                                                                                  \
      NALU_HYPRE_Int idx = (NALU_HYPRE_Int) item.get_global_linear_id();                                        \
      if (idx < nalu_hypre__tot)                                                                           \
      {                                                                                               \
         nalu_hypre_newBoxLoopDeclare(databox1);                                                           \
         nalu_hypre_BoxLoopIncK(1, databox1, i1);                                                          \
         nalu_hypre_BoxLoopIncK(2, databox2, i2);


/* Reduction BoxLoop1 */
#define nalu_hypre_newBoxLoop1ReductionBegin(ndim, loop_size, dbox1, start1, stride1, i1, sum_var)         \
{                                                                                                     \
   nalu_hypre_newBoxLoopInit(ndim, loop_size);                                                             \
   nalu_hypre_BoxLoopDataDeclareK(1, ndim, loop_size, dbox1, start1, stride1);                             \
   NALU_HYPRE_Real *shared_sum_var = nalu_hypre_CTAlloc(NALU_HYPRE_Real, 1, NALU_HYPRE_MEMORY_DEVICE);                    \
   nalu_hypre_TMemcpy(shared_sum_var, &sum_var, NALU_HYPRE_Real, 1, NALU_HYPRE_MEMORY_DEVICE, NALU_HYPRE_MEMORY_HOST);    \
   ReductionBoxLoopforall( [=,nalu_hypre_unused_var=sum_var] (sycl::nd_item<3> item, auto &sum_var)        \
   {                                                                                                  \
      NALU_HYPRE_Int idx = (NALU_HYPRE_Int) item.get_global_linear_id();                                        \
      if (idx < nalu_hypre__tot)                                                                           \
      {                                                                                               \
         nalu_hypre_newBoxLoopDeclare(databox1);                                                           \
         nalu_hypre_BoxLoopIncK(1, databox1, i1);

#define nalu_hypre_newBoxLoop1ReductionEnd(i1, sum_var)                                                    \
      }                                                                                               \
   }, nalu_hypre__tot, shared_sum_var);                                                                    \
   nalu_hypre_TMemcpy(&sum_var, shared_sum_var, NALU_HYPRE_Real, 1, NALU_HYPRE_MEMORY_HOST, NALU_HYPRE_MEMORY_DEVICE);    \
   nalu_hypre_TFree(shared_sum_var, NALU_HYPRE_MEMORY_DEVICE);                                                  \
}

/* Reduction BoxLoop2 */
#define nalu_hypre_newBoxLoop2ReductionBegin(ndim, loop_size, dbox1, start1, stride1, i1,                  \
                                                      dbox2, start2, stride2, i2, sum_var)            \
{                                                                                                     \
   nalu_hypre_newBoxLoopInit(ndim, loop_size);                                                             \
   nalu_hypre_BoxLoopDataDeclareK(1, ndim, loop_size, dbox1, start1, stride1);                             \
   nalu_hypre_BoxLoopDataDeclareK(2, ndim, loop_size, dbox2, start2, stride2);                             \
   NALU_HYPRE_Real *shared_sum_var = nalu_hypre_CTAlloc(NALU_HYPRE_Real, 1, NALU_HYPRE_MEMORY_DEVICE);                    \
   nalu_hypre_TMemcpy(shared_sum_var, &sum_var, NALU_HYPRE_Real, 1, NALU_HYPRE_MEMORY_DEVICE, NALU_HYPRE_MEMORY_HOST);    \
   ReductionBoxLoopforall( [=,nalu_hypre_unused_var=sum_var] (sycl::nd_item<3> item, auto &sum_var)        \
   {                                                                                                  \
      NALU_HYPRE_Int idx = (NALU_HYPRE_Int) item.get_global_linear_id();                                        \
      if (idx < nalu_hypre__tot)                                                                           \
      {                                                                                               \
         nalu_hypre_newBoxLoopDeclare(databox1);                                                           \
         nalu_hypre_BoxLoopIncK(1, databox1, i1);                                                          \
         nalu_hypre_BoxLoopIncK(2, databox2, i2);

#define nalu_hypre_newBoxLoop2ReductionEnd(i1, i2, sum_var)                                                \
      }                                                                                               \
   }, nalu_hypre__tot, shared_sum_var);                                                                    \
   nalu_hypre_TMemcpy(&sum_var, shared_sum_var, NALU_HYPRE_Real, 1, NALU_HYPRE_MEMORY_HOST, NALU_HYPRE_MEMORY_DEVICE);    \
   nalu_hypre_TFree(shared_sum_var, NALU_HYPRE_MEMORY_DEVICE);                                                  \
}

/* Plain parallel_for loop */
#define nalu_hypre_LoopBegin(size, idx)                                                                    \
{                                                                                                     \
   NALU_HYPRE_Int nalu_hypre__tot = size;                                                                       \
   BoxLoopforall(nalu_hypre__tot, [=] (sycl::nd_item<3> item)                                              \
   {                                                                                                  \
      NALU_HYPRE_Int idx = (NALU_HYPRE_Int) item.get_global_linear_id();                                        \
      if (idx < nalu_hypre__tot)                                                                           \
      {                                                                                               \

#define nalu_hypre_LoopEnd()                                                                               \
      }                                                                                               \
   });                                                                                                \
}


/*********************************************************************
 * renamings
 *********************************************************************/

#define nalu_hypre_BoxLoopBlock()       0

#define nalu_hypre_BoxLoop0Begin      nalu_hypre_newBoxLoop0Begin
#define nalu_hypre_BoxLoop0End        nalu_hypre_newBoxLoop0End
#define nalu_hypre_BoxLoop1Begin      nalu_hypre_newBoxLoop1Begin
#define nalu_hypre_BoxLoop1End        nalu_hypre_newBoxLoop1End
#define nalu_hypre_BoxLoop2Begin      nalu_hypre_newBoxLoop2Begin
#define nalu_hypre_BoxLoop2End        nalu_hypre_newBoxLoop2End
#define nalu_hypre_BoxLoop3Begin      nalu_hypre_newBoxLoop3Begin
#define nalu_hypre_BoxLoop3End        nalu_hypre_newBoxLoop3End
#define nalu_hypre_BoxLoop4Begin      nalu_hypre_newBoxLoop4Begin
#define nalu_hypre_BoxLoop4End        nalu_hypre_newBoxLoop4End

#define nalu_hypre_BasicBoxLoop1Begin nalu_hypre_newBasicBoxLoop1Begin
#define nalu_hypre_BasicBoxLoop2Begin nalu_hypre_newBasicBoxLoop2Begin

/* Reduction */
#define nalu_hypre_BoxLoop1ReductionBegin(ndim, loop_size, dbox1, start1, stride1, i1, reducesum) \
        nalu_hypre_newBoxLoop1ReductionBegin(ndim, loop_size, dbox1, start1, stride1, i1, reducesum)

#define nalu_hypre_BoxLoop1ReductionEnd(i1, reducesum) \
        nalu_hypre_newBoxLoop1ReductionEnd(i1, reducesum)

#define nalu_hypre_BoxLoop2ReductionBegin(ndim, loop_size, dbox1, start1, stride1, i1, \
                                                      dbox2, start2, stride2, i2, reducesum) \
        nalu_hypre_newBoxLoop2ReductionBegin(ndim, loop_size, dbox1, start1, stride1, i1, \
                                                         dbox2, start2, stride2, i2, reducesum)

#define nalu_hypre_BoxLoop2ReductionEnd(i1, i2, reducesum) \
        nalu_hypre_newBoxLoop2ReductionEnd(i1, i2, reducesum)

#endif

#endif /* #ifndef NALU_HYPRE_BOXLOOP_SYCL_HEADER */

