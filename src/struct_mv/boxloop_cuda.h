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

#ifndef NALU_HYPRE_BOXLOOP_CUDA_HEADER
#define NALU_HYPRE_BOXLOOP_CUDA_HEADER

#if (defined(NALU_HYPRE_USING_CUDA) || defined(NALU_HYPRE_USING_HIP)) && !defined(NALU_HYPRE_USING_RAJA) && !defined(NALU_HYPRE_USING_KOKKOS)

#define NALU_HYPRE_LAMBDA [=] __host__  __device__

/* TODO: RL: support 4-D */
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

   /* -------------------------
    *     parfor-loop
    * ------------------------*/

   template <typename LOOP_BODY>
   __global__ void
   forall_kernel( nalu_hypre_DeviceItem & item,
                  LOOP_BODY loop_body,
                  NALU_HYPRE_Int length )
   {
      const NALU_HYPRE_Int idx = nalu_hypre_gpu_get_grid_thread_id<1, 1>(item);
      /* const NALU_HYPRE_Int number_threads = nalu_hypre_gpu_get_grid_num_threads<1,1>(item); */

      if (idx < length)
      {
         loop_body(idx);
      }
   }

   template<typename LOOP_BODY>
   void
   BoxLoopforall( NALU_HYPRE_Int length,
                  LOOP_BODY loop_body )
   {
      const NALU_HYPRE_MemoryLocation memory_location = nalu_hypre_HandleMemoryLocation(nalu_hypre_handle());
      const NALU_HYPRE_ExecutionPolicy exec_policy = nalu_hypre_GetExecPolicy1(memory_location);

      if (exec_policy == NALU_HYPRE_EXEC_HOST)
      {
#ifdef NALU_HYPRE_USING_OPENMP
         #pragma omp parallel for NALU_HYPRE_SMP_SCHEDULE
#endif
         for (NALU_HYPRE_Int idx = 0; idx < length; idx++)
         {
            loop_body(idx);
         }
      }
      else if (exec_policy == NALU_HYPRE_EXEC_DEVICE)
      {
         const dim3 bDim = nalu_hypre_GetDefaultDeviceBlockDimension();
         const dim3 gDim = nalu_hypre_GetDefaultDeviceGridDimension(length, "thread", bDim);

         NALU_HYPRE_GPU_LAUNCH( forall_kernel, gDim, bDim, loop_body, length );
      }
   }

   /* ------------------------------
    *     parforreduction-loop
    * -----------------------------*/

   template <typename LOOP_BODY, typename REDUCER>
   __global__ void
   reductionforall_kernel( nalu_hypre_DeviceItem & item,
                           NALU_HYPRE_Int length,
                           REDUCER   reducer,
                           LOOP_BODY loop_body )
   {
      const NALU_HYPRE_Int thread_id = nalu_hypre_gpu_get_grid_thread_id<1, 1>(item);
      const NALU_HYPRE_Int n_threads = nalu_hypre_gpu_get_grid_num_threads<1, 1>(item);

      for (NALU_HYPRE_Int idx = thread_id; idx < length; idx += n_threads)
      {
         loop_body(idx, reducer);
      }

      /* reduction in block-level and the save the results in reducer */
      reducer.BlockReduce();
   }

   template<typename LOOP_BODY, typename REDUCER>
   void
   ReductionBoxLoopforall( NALU_HYPRE_Int  length,
                           REDUCER   & reducer,
                           LOOP_BODY  loop_body )
   {
      if (length <= 0)
      {
         return;
      }

      const NALU_HYPRE_MemoryLocation memory_location = nalu_hypre_HandleMemoryLocation(nalu_hypre_handle());
      const NALU_HYPRE_ExecutionPolicy exec_policy = nalu_hypre_GetExecPolicy1(memory_location);

      if (exec_policy == NALU_HYPRE_EXEC_HOST)
      {
         for (NALU_HYPRE_Int idx = 0; idx < length; idx++)
         {
            loop_body(idx, reducer);
         }
      }
      else if (exec_policy == NALU_HYPRE_EXEC_DEVICE)
      {
         /* Assume gDim cannot exceed NALU_HYPRE_MAX_NTHREADS_BLOCK (the max size for the 2nd reduction)
          * and bDim <= WARP * WARP (because we use 1 warp fro the block-level reduction) */
         const dim3 bDim = nalu_hypre_GetDefaultDeviceBlockDimension();
         dim3 gDim = nalu_hypre_GetDefaultDeviceGridDimension(length, "thread", bDim);
         gDim.x = nalu_hypre_min(gDim.x, NALU_HYPRE_MAX_NTHREADS_BLOCK);

         reducer.nblocks = gDim.x;

         reducer.Allocate2ndPhaseBuffer();

         /*
         nalu_hypre_printf("length= %d, blocksize = %d, gridsize = %d\n", length, bDim.x, gDim.x);
         */

         NALU_HYPRE_GPU_LAUNCH( reductionforall_kernel, gDim, bDim, length, reducer, loop_body );
      }
   }

#ifdef __cplusplus
}
#endif

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
   /* dim 0 */                                                             \
   databox##k.lsize0   = loop_size[0];                                     \
   databox##k.strides0 = stride[0];                                        \
   databox##k.bstart0  = start[0] - dbox->imin[0];                         \
   databox##k.bsize0   = dbox->imax[0] - dbox->imin[0];                    \
   /* dim 1 */                                                             \
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
   /* dim 2 */                                                             \
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

/* RL: TODO loop_size out of box struct, bsize +1 */
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
#define nalu_hypre_BoxLoopGetIndexCUDA(index)                                                              \
   index[0] = nalu_hypre_IndexD(local_idx, 0);                                                             \
   index[1] = nalu_hypre_IndexD(local_idx, 1);                                                             \
   index[2] = nalu_hypre_IndexD(local_idx, 2);

/* BoxLoop 0 */
#define nalu_hypre_BoxLoop0BeginCUDA(ndim, loop_size)                                                      \
{                                                                                                     \
   nalu_hypre_newBoxLoopInit(ndim, loop_size);                                                             \
   BoxLoopforall(nalu_hypre__tot, NALU_HYPRE_LAMBDA (NALU_HYPRE_Int idx)                                             \
   {

#define nalu_hypre_BoxLoop0EndCUDA()                                                                       \
   });                                                                                                \
}

/* BoxLoop 1 */
#define nalu_hypre_BoxLoop1BeginCUDA(ndim, loop_size, dbox1, start1, stride1, i1)                          \
{                                                                                                     \
   nalu_hypre_newBoxLoopInit(ndim, loop_size);                                                             \
   nalu_hypre_BoxLoopDataDeclareK(1, ndim, loop_size, dbox1, start1, stride1);                             \
   BoxLoopforall(nalu_hypre__tot, NALU_HYPRE_LAMBDA (NALU_HYPRE_Int idx)                                             \
   {                                                                                                  \
      nalu_hypre_newBoxLoopDeclare(databox1);                                                              \
      nalu_hypre_BoxLoopIncK(1, databox1, i1);

#define nalu_hypre_BoxLoop1EndCUDA(i1)                                                                     \
   });                                                                                                \
}

/* BoxLoop 2 */
#define nalu_hypre_BoxLoop2BeginCUDA(ndim, loop_size, dbox1, start1, stride1, i1,                          \
                                                 dbox2, start2, stride2, i2)                          \
{                                                                                                     \
   nalu_hypre_newBoxLoopInit(ndim, loop_size);                                                             \
   nalu_hypre_BoxLoopDataDeclareK(1, ndim, loop_size, dbox1, start1, stride1);                             \
   nalu_hypre_BoxLoopDataDeclareK(2, ndim, loop_size, dbox2, start2, stride2);                             \
   BoxLoopforall(nalu_hypre__tot, NALU_HYPRE_LAMBDA (NALU_HYPRE_Int idx)                                             \
   {                                                                                                  \
      nalu_hypre_newBoxLoopDeclare(databox1);                                                              \
      nalu_hypre_BoxLoopIncK(1, databox1, i1);                                                             \
      nalu_hypre_BoxLoopIncK(2, databox2, i2);

#define nalu_hypre_BoxLoop2EndCUDA(i1, i2)                                                                 \
   });                                                                                                \
}

/* BoxLoop 3 */
#define nalu_hypre_BoxLoop3BeginCUDA(ndim, loop_size, dbox1, start1, stride1, i1,                          \
                                                 dbox2, start2, stride2, i2,                          \
                                                 dbox3, start3, stride3, i3)                          \
{                                                                                                     \
   nalu_hypre_newBoxLoopInit(ndim, loop_size);                                                             \
   nalu_hypre_BoxLoopDataDeclareK(1, ndim,loop_size, dbox1, start1, stride1);                              \
   nalu_hypre_BoxLoopDataDeclareK(2, ndim,loop_size, dbox2, start2, stride2);                              \
   nalu_hypre_BoxLoopDataDeclareK(3, ndim,loop_size, dbox3, start3, stride3);                              \
   BoxLoopforall(nalu_hypre__tot, NALU_HYPRE_LAMBDA (NALU_HYPRE_Int idx)                                             \
   {                                                                                                  \
      nalu_hypre_newBoxLoopDeclare(databox1);                                                              \
      nalu_hypre_BoxLoopIncK(1, databox1, i1);                                                             \
      nalu_hypre_BoxLoopIncK(2, databox2, i2);                                                             \
      nalu_hypre_BoxLoopIncK(3, databox3, i3);

#define nalu_hypre_BoxLoop3EndCUDA(i1, i2, i3)                                                             \
   });                                                                                                \
}

/* BoxLoop 4 */
#define nalu_hypre_BoxLoop4BeginCUDA(ndim, loop_size, dbox1, start1, stride1, i1,                          \
                                                 dbox2, start2, stride2, i2,                          \
                                                 dbox3, start3, stride3, i3,                          \
                                                 dbox4, start4, stride4, i4)                          \
{                                                                                                     \
   nalu_hypre_newBoxLoopInit(ndim, loop_size);                                                             \
   nalu_hypre_BoxLoopDataDeclareK(1, ndim, loop_size, dbox1, start1, stride1);                             \
   nalu_hypre_BoxLoopDataDeclareK(2, ndim, loop_size, dbox2, start2, stride2);                             \
   nalu_hypre_BoxLoopDataDeclareK(3, ndim, loop_size, dbox3, start3, stride3);                             \
   nalu_hypre_BoxLoopDataDeclareK(4, ndim, loop_size, dbox4, start4, stride4);                             \
   BoxLoopforall(nalu_hypre__tot, NALU_HYPRE_LAMBDA (NALU_HYPRE_Int idx)                                             \
   {                                                                                                  \
      nalu_hypre_newBoxLoopDeclare(databox1);                                                              \
      nalu_hypre_BoxLoopIncK(1, databox1, i1);                                                             \
      nalu_hypre_BoxLoopIncK(2, databox2, i2);                                                             \
      nalu_hypre_BoxLoopIncK(3, databox3, i3);                                                             \
      nalu_hypre_BoxLoopIncK(4, databox4, i4);

#define nalu_hypre_BoxLoop4EndCUDA(i1, i2, i3, i4)                                                         \
   });                                                                                                \
}

/* Basic BoxLoops have no boxes */
/* BoxLoop 1 */
#define nalu_hypre_BasicBoxLoop1BeginCUDA(ndim, loop_size, stride1, i1)                                    \
{                                                                                                     \
   nalu_hypre_newBoxLoopInit(ndim, loop_size);                                                             \
   nalu_hypre_BasicBoxLoopDataDeclareK(1, ndim, loop_size, stride1);                                       \
   BoxLoopforall(nalu_hypre__tot, NALU_HYPRE_LAMBDA (NALU_HYPRE_Int idx)                                             \
   {                                                                                                  \
      nalu_hypre_newBoxLoopDeclare(databox1);                                                              \
      nalu_hypre_BoxLoopIncK(1, databox1, i1);

/* BoxLoop 2 */
#define nalu_hypre_BasicBoxLoop2BeginCUDA(ndim, loop_size, stride1, i1, stride2, i2)                       \
{                                                                                                     \
   nalu_hypre_newBoxLoopInit(ndim, loop_size);                                                             \
   nalu_hypre_BasicBoxLoopDataDeclareK(1, ndim, loop_size, stride1);                                       \
   nalu_hypre_BasicBoxLoopDataDeclareK(2, ndim, loop_size, stride2);                                       \
   BoxLoopforall(nalu_hypre__tot, NALU_HYPRE_LAMBDA (NALU_HYPRE_Int idx)                                             \
   {                                                                                                  \
      nalu_hypre_newBoxLoopDeclare(databox1);                                                              \
      nalu_hypre_BoxLoopIncK(1, databox1, i1);                                                             \
      nalu_hypre_BoxLoopIncK(2, databox2, i2);                                                             \

/* Parallel for-loop */
#define nalu_hypre_LoopBeginCUDA(size, idx)                                                                \
{                                                                                                     \
   BoxLoopforall(size, NALU_HYPRE_LAMBDA (NALU_HYPRE_Int idx)                                                   \
   {

#define nalu_hypre_LoopEndCUDA()                                                                           \
   });                                                                                                \
}

/* Reduction BoxLoop1 */
#define nalu_hypre_BoxLoop1ReductionBeginCUDA(ndim, loop_size, dbox1, start1, stride1, i1, reducesum)                 \
{                                                                                                                \
   nalu_hypre_newBoxLoopInit(ndim, loop_size);                                                                        \
   nalu_hypre_BoxLoopDataDeclareK(1, ndim, loop_size, dbox1, start1, stride1);                                        \
   ReductionBoxLoopforall(nalu_hypre__tot, reducesum, NALU_HYPRE_LAMBDA (NALU_HYPRE_Int idx, decltype(reducesum) &reducesum)    \
   {                                                                                                             \
      nalu_hypre_newBoxLoopDeclare(databox1);                                                                         \
      nalu_hypre_BoxLoopIncK(1, databox1, i1);

#define nalu_hypre_BoxLoop1ReductionEndCUDA(i1, reducesum)                                                            \
   });                                                                                                           \
}

/* Reduction BoxLoop2 */
#define nalu_hypre_BoxLoop2ReductionBeginCUDA(ndim, loop_size, dbox1, start1, stride1, i1,                            \
                                                          dbox2, start2, stride2, i2, reducesum)                 \
{                                                                                                                \
   nalu_hypre_newBoxLoopInit(ndim, loop_size);                                                                        \
   nalu_hypre_BoxLoopDataDeclareK(1, ndim, loop_size, dbox1, start1, stride1);                                        \
   nalu_hypre_BoxLoopDataDeclareK(2, ndim, loop_size, dbox2, start2, stride2);                                        \
   ReductionBoxLoopforall(nalu_hypre__tot, reducesum, NALU_HYPRE_LAMBDA (NALU_HYPRE_Int idx, decltype(reducesum) &reducesum)    \
   {                                                                                                             \
      nalu_hypre_newBoxLoopDeclare(databox1);                                                                         \
      nalu_hypre_BoxLoopIncK(1, databox1, i1);                                                                        \
      nalu_hypre_BoxLoopIncK(2, databox2, i2);

#define nalu_hypre_BoxLoop2ReductionEndCUDA(i1, i2, reducesum)                                                        \
   });                                                                                                           \
}

/* Renamings */
#define nalu_hypre_BoxLoopGetIndexDevice          nalu_hypre_BoxLoopGetIndexCUDA
#define nalu_hypre_BoxLoopBlockDevice()           0
#define nalu_hypre_BoxLoop0BeginDevice            nalu_hypre_BoxLoop0BeginCUDA
#define nalu_hypre_BoxLoop0EndDevice              nalu_hypre_BoxLoop0EndCUDA
#define nalu_hypre_BoxLoop1BeginDevice            nalu_hypre_BoxLoop1BeginCUDA
#define nalu_hypre_BoxLoop1EndDevice              nalu_hypre_BoxLoop1EndCUDA
#define nalu_hypre_BoxLoop2BeginDevice            nalu_hypre_BoxLoop2BeginCUDA
#define nalu_hypre_BoxLoop2EndDevice              nalu_hypre_BoxLoop2EndCUDA
#define nalu_hypre_BoxLoop3BeginDevice            nalu_hypre_BoxLoop3BeginCUDA
#define nalu_hypre_BoxLoop3EndDevice              nalu_hypre_BoxLoop3EndCUDA
#define nalu_hypre_BoxLoop4BeginDevice            nalu_hypre_BoxLoop4BeginCUDA
#define nalu_hypre_BoxLoop4EndDevice              nalu_hypre_BoxLoop4EndCUDA
#define nalu_hypre_BasicBoxLoop1BeginDevice       nalu_hypre_BasicBoxLoop1BeginCUDA
#define nalu_hypre_BasicBoxLoop2BeginDevice       nalu_hypre_BasicBoxLoop2BeginCUDA
#define nalu_hypre_LoopBeginDevice                nalu_hypre_LoopBeginCUDA
#define nalu_hypre_LoopEndDevice                  nalu_hypre_LoopEndCUDA
#define nalu_hypre_BoxLoop1ReductionBeginDevice   nalu_hypre_BoxLoop1ReductionBeginCUDA
#define nalu_hypre_BoxLoop1ReductionEndDevice     nalu_hypre_BoxLoop1ReductionEndCUDA
#define nalu_hypre_BoxLoop2ReductionBeginDevice   nalu_hypre_BoxLoop2ReductionBeginCUDA
#define nalu_hypre_BoxLoop2ReductionEndDevice     nalu_hypre_BoxLoop2ReductionEndCUDA


//TODO TEMP FIX
#define nalu_hypre_BoxLoopGetIndex          nalu_hypre_BoxLoopGetIndexDevice
#define nalu_hypre_BoxLoopBlock()           0
#define nalu_hypre_BoxLoop0Begin            nalu_hypre_BoxLoop0BeginDevice
#define nalu_hypre_BoxLoop0End              nalu_hypre_BoxLoop0EndDevice
#define nalu_hypre_BoxLoop1Begin            nalu_hypre_BoxLoop1BeginDevice
#define nalu_hypre_BoxLoop1End              nalu_hypre_BoxLoop1EndDevice
#define nalu_hypre_BoxLoop2Begin            nalu_hypre_BoxLoop2BeginDevice
#define nalu_hypre_BoxLoop2End              nalu_hypre_BoxLoop2EndDevice
#define nalu_hypre_BoxLoop3Begin            nalu_hypre_BoxLoop3BeginDevice
#define nalu_hypre_BoxLoop3End              nalu_hypre_BoxLoop3EndDevice
#define nalu_hypre_BoxLoop4Begin            nalu_hypre_BoxLoop4BeginDevice
#define nalu_hypre_BoxLoop4End              nalu_hypre_BoxLoop4EndDevice
#define nalu_hypre_BasicBoxLoop1Begin       nalu_hypre_BasicBoxLoop1BeginDevice
#define nalu_hypre_BasicBoxLoop2Begin       nalu_hypre_BasicBoxLoop2BeginDevice
#define nalu_hypre_LoopBegin                nalu_hypre_LoopBeginDevice
#define nalu_hypre_LoopEnd                  nalu_hypre_LoopEndDevice
#define nalu_hypre_BoxLoop1ReductionBegin   nalu_hypre_BoxLoop1ReductionBeginDevice
#define nalu_hypre_BoxLoop1ReductionEnd     nalu_hypre_BoxLoop1ReductionEndDevice
#define nalu_hypre_BoxLoop2ReductionBegin   nalu_hypre_BoxLoop2ReductionBeginDevice
#define nalu_hypre_BoxLoop2ReductionEnd     nalu_hypre_BoxLoop2ReductionEndDevice

#endif

#endif /* #ifndef NALU_HYPRE_BOXLOOP_CUDA_HEADER */

