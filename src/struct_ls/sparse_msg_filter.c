/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_nalu_hypre_struct_ls.h"
#include "_nalu_hypre_struct_mv.hpp"

#if 0

/*--------------------------------------------------------------------------
 * nalu_hypre_SparseMSGFilterSetup
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SparseMSGFilterSetup( nalu_hypre_StructMatrix *A,
                            NALU_HYPRE_Int          *num_grids,
                            NALU_HYPRE_Int           lx,
                            NALU_HYPRE_Int           ly,
                            NALU_HYPRE_Int           lz,
                            NALU_HYPRE_Int           jump,
                            nalu_hypre_StructVector *visitx,
                            nalu_hypre_StructVector *visity,
                            nalu_hypre_StructVector *visitz    )
{
   NALU_HYPRE_Int             ierr = 0;

   nalu_hypre_BoxArray        *compute_boxes;
   nalu_hypre_Box             *compute_box;

   nalu_hypre_Box             *A_dbox;
   nalu_hypre_Box             *v_dbox;

   NALU_HYPRE_Int              Ai;
   NALU_HYPRE_Int              vi;

   NALU_HYPRE_Real            *Ap;
   NALU_HYPRE_Real            *vxp;
   NALU_HYPRE_Real            *vyp;
   NALU_HYPRE_Real            *vzp;
   NALU_HYPRE_Real             lambdax;
   NALU_HYPRE_Real             lambday;
   NALU_HYPRE_Real             lambdaz;
   NALU_HYPRE_Real             lambda_max;

   nalu_hypre_StructStencil   *stencil;
   nalu_hypre_Index           *stencil_shape;
   NALU_HYPRE_Int              stencil_size;

   NALU_HYPRE_Int              Astenc;

   nalu_hypre_Index            loop_size;
   nalu_hypre_Index            cindex;
   nalu_hypre_IndexRef         start;
   nalu_hypre_Index            startv;
   nalu_hypre_Index            stride;
   nalu_hypre_Index            stridev;

   NALU_HYPRE_Int              i, si, dir, k, l;

   /*----------------------------------------------------------
    * Initialize some things
    *----------------------------------------------------------*/

   stencil       = nalu_hypre_StructMatrixStencil(A);
   stencil_shape = nalu_hypre_StructStencilShape(stencil);
   stencil_size  = nalu_hypre_StructStencilSize(stencil);

   /*-----------------------------------------------------
    * Compute encoding digit and strides
    *-----------------------------------------------------*/

   nalu_hypre_SetIndex3(stride, 1, 1, 1);

   l = lx + ly + lz;
   if ((l >= 1) && (l <= jump))
   {
      k = 1 >> l;
      nalu_hypre_SetIndex3(stridev, (1 >> lx), (1 >> ly), (1 >> lz));
   }
   else
   {
      k = 1;
      nalu_hypre_SetIndex3(stridev, 1, 1, 1);

      nalu_hypre_StructVectorSetConstantValues(visitx, 0.0);
      nalu_hypre_StructVectorSetConstantValues(visity, 0.0);
      nalu_hypre_StructVectorSetConstantValues(visitz, 0.0);
   }

   /*-----------------------------------------------------
    * Compute visit vectors
    *-----------------------------------------------------*/

   nalu_hypre_SetIndex3(cindex, 0, 0, 0);

   compute_boxes = nalu_hypre_StructGridBoxes(nalu_hypre_StructMatrixGrid(A));
   nalu_hypre_ForBoxI(i, compute_boxes)
   {
      compute_box = nalu_hypre_BoxArrayBox(compute_boxes, i);

      A_dbox = nalu_hypre_BoxArrayBox(nalu_hypre_StructMatrixDataSpace(A), i);
      v_dbox = nalu_hypre_BoxArrayBox(nalu_hypre_StructVectorDataSpace(visitx), i);

      vxp = nalu_hypre_StructVectorBoxData(visitx, i);
      vyp = nalu_hypre_StructVectorBoxData(visity, i);
      vzp = nalu_hypre_StructVectorBoxData(visitz, i);

      start = nalu_hypre_BoxIMin(compute_box);
      nalu_hypre_StructMapCoarseToFine(start, cindex, stridev, startv);
      nalu_hypre_BoxGetSize(compute_box, loop_size);

      nalu_hypre_BoxLoop2Begin(nalu_hypre_StructMatrixNDim(A), loop_size,
                          A_dbox, start,  stride,  Ai,
                          v_dbox, startv, stridev, vi);
      {
         NALU_HYPRE_Real lambdax = 0.0;
         NALU_HYPRE_Real lambday = 0.0;
         NALU_HYPRE_Real lambdaz = 0.0;
         NALU_HYPRE_Int si, dir, Astenc;
         NALU_HYPRE_Real *Ap, lambda_max;

         for (si = 0; si < stencil_size; si++)
         {
            Ap = nalu_hypre_StructMatrixBoxData(A, i, si);

            /* compute lambdax */
            Astenc = nalu_hypre_IndexD(stencil_shape[si], 0);
            if (Astenc == 0)
            {
               lambdax += Ap[Ai];
            }
            else
            {
               lambdax -= Ap[Ai];
            }

            /* compute lambday */
            Astenc = nalu_hypre_IndexD(stencil_shape[si], 1);
            if (Astenc == 0)
            {
               lambday += Ap[Ai];
            }
            else
            {
               lambday -= Ap[Ai];
            }

            /* compute lambdaz */
            Astenc = nalu_hypre_IndexD(stencil_shape[si], 2);
            if (Astenc == 0)
            {
               lambdaz += Ap[Ai];
            }
            else
            {
               lambdaz -= Ap[Ai];
            }
         }

         lambdax *= lambdax;
         lambday *= lambday;
         lambdaz *= lambdaz;

         lambda_max = 0;
         dir = -1;
         if ((lx < num_grids[0] - 1) && (lambdax > lambda_max))
         {
            lambda_max = lambdax;
            dir = 0;
         }
         if ((ly < num_grids[1] - 1) && (lambday > lambda_max))
         {
            lambda_max = lambday;
            dir = 1;
         }
         if ((lz < num_grids[2] - 1) && (lambdaz > lambda_max))
         {
            lambda_max = lambdaz;
            dir = 2;
         }

         if (dir == 0)
         {
            vxp[vi] = (NALU_HYPRE_Real) ( ((NALU_HYPRE_Int) vxp[vi]) | k );
         }
         else if (dir == 1)
         {
            vyp[vi] = (NALU_HYPRE_Real) ( ((NALU_HYPRE_Int) vyp[vi]) | k );
         }
         else if (dir == 2)
         {
            vzp[vi] = (NALU_HYPRE_Real) ( ((NALU_HYPRE_Int) vzp[vi]) | k );
         }
      }
      nalu_hypre_BoxLoop2End(Ai, vi);
   }

   return ierr;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_SparseMSGFilter
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SparseMSGFilter( nalu_hypre_StructVector *visit,
                       nalu_hypre_StructVector *e,
                       NALU_HYPRE_Int           lx,
                       NALU_HYPRE_Int           ly,
                       NALU_HYPRE_Int           lz,
                       NALU_HYPRE_Int           jump  )
{
   NALU_HYPRE_Int             ierr = 0;

   nalu_hypre_BoxArray        *compute_boxes;
   nalu_hypre_Box             *compute_box;

   nalu_hypre_Box             *e_dbox;
   nalu_hypre_Box             *v_dbox;

   NALU_HYPRE_Int              ei;
   NALU_HYPRE_Int              vi;

   NALU_HYPRE_Real            *ep;
   NALU_HYPRE_Real            *vp;

   nalu_hypre_Index            loop_size;
   nalu_hypre_Index            cindex;
   nalu_hypre_IndexRef         start;
   nalu_hypre_Index            startv;
   nalu_hypre_Index            stride;
   nalu_hypre_Index            stridev;

   NALU_HYPRE_Int              i, k, l;

   /*-----------------------------------------------------
    * Compute encoding digit and strides
    *-----------------------------------------------------*/

   nalu_hypre_SetIndex3(stride, 1, 1, 1);

   l = lx + ly + lz;
   if ((l >= 1) && (l <= jump))
   {
      k = 1 >> l;
      nalu_hypre_SetIndex3(stridev, (1 >> lx), (1 >> ly), (1 >> lz));
   }
   else
   {
      k = 1;
      nalu_hypre_SetIndex3(stridev, 1, 1, 1);
   }

   /*-----------------------------------------------------
    * Filter interpolated error
    *-----------------------------------------------------*/

   nalu_hypre_SetIndex3(cindex, 0, 0, 0);

   compute_boxes = nalu_hypre_StructGridBoxes(nalu_hypre_StructVectorGrid(e));
   nalu_hypre_ForBoxI(i, compute_boxes)
   {
      compute_box = nalu_hypre_BoxArrayBox(compute_boxes, i);

      e_dbox = nalu_hypre_BoxArrayBox(nalu_hypre_StructVectorDataSpace(e), i);
      v_dbox = nalu_hypre_BoxArrayBox(nalu_hypre_StructVectorDataSpace(visit), i);

      ep = nalu_hypre_StructVectorBoxData(e, i);
      vp = nalu_hypre_StructVectorBoxData(visit, i);

      start = nalu_hypre_BoxIMin(compute_box);
      nalu_hypre_StructMapCoarseToFine(start, cindex, stridev, startv);
      nalu_hypre_BoxGetSize(compute_box, loop_size);

      nalu_hypre_BoxLoop2Begin(nalu_hypre_StructVectorNDim(e), loop_size,
                          e_dbox, start,  stride,  ei,
                          v_dbox, startv, stridev, vi);
      {
         if ( !(((NALU_HYPRE_Int) vp[vi]) & k) )
         {
            ep[ei] = 0.0;
         }
      }
      nalu_hypre_BoxLoop2End(ei, vi);
   }

   return ierr;
}

#else

/*--------------------------------------------------------------------------
 * nalu_hypre_SparseMSGFilterSetup
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SparseMSGFilterSetup( nalu_hypre_StructMatrix *A,
                            NALU_HYPRE_Int          *num_grids,
                            NALU_HYPRE_Int           lx,
                            NALU_HYPRE_Int           ly,
                            NALU_HYPRE_Int           lz,
                            NALU_HYPRE_Int           jump,
                            nalu_hypre_StructVector *visitx,
                            nalu_hypre_StructVector *visity,
                            nalu_hypre_StructVector *visitz    )
{
   NALU_HYPRE_Int             ierr = 0;

   nalu_hypre_BoxArray        *compute_boxes;
   nalu_hypre_Box             *compute_box;

   nalu_hypre_Box             *A_dbox;
   nalu_hypre_Box             *v_dbox;

   NALU_HYPRE_Real            *vxp;
   NALU_HYPRE_Real            *vyp;
   NALU_HYPRE_Real            *vzp;

   nalu_hypre_StructStencil   *stencil;
   nalu_hypre_Index           *stencil_shape;
   NALU_HYPRE_Int              stencil_size;

   nalu_hypre_Index            loop_size;
   nalu_hypre_Index            cindex;
   nalu_hypre_IndexRef         start;
   nalu_hypre_Index            startv;
   nalu_hypre_Index            stride;
   nalu_hypre_Index            stridev;
   NALU_HYPRE_Int              i;
   NALU_HYPRE_MemoryLocation   memory_location = nalu_hypre_StructMatrixMemoryLocation(A);

   /*----------------------------------------------------------
    * Initialize some things
    *----------------------------------------------------------*/

   stencil       = nalu_hypre_StructMatrixStencil(A);
   stencil_shape = nalu_hypre_StructStencilShape(stencil);
   stencil_size  = nalu_hypre_StructStencilSize(stencil);

   /*-----------------------------------------------------
    * Compute encoding digit and strides
    *-----------------------------------------------------*/

   nalu_hypre_SetIndex3(stride, 1, 1, 1);
   nalu_hypre_SetIndex3(stridev, 1, 1, 1);

   /*-----------------------------------------------------
    * Compute visit vectors
    *-----------------------------------------------------*/

   nalu_hypre_SetIndex3(cindex, 0, 0, 0);

   compute_boxes = nalu_hypre_StructGridBoxes(nalu_hypre_StructMatrixGrid(A));

   NALU_HYPRE_Int     **data_indices = nalu_hypre_StructMatrixDataIndices(A);
   NALU_HYPRE_Complex  *matrixA_data = nalu_hypre_StructMatrixData(A);
   NALU_HYPRE_Int      *data_indices_d; /* On device */
   nalu_hypre_Index    *stencil_shape_d;

   if (nalu_hypre_GetExecPolicy1(memory_location) == NALU_HYPRE_EXEC_DEVICE)
   {
      NALU_HYPRE_Int nboxes = nalu_hypre_BoxArraySize(compute_boxes);
      data_indices_d  = nalu_hypre_TAlloc(NALU_HYPRE_Int, stencil_size * nboxes, memory_location);
      stencil_shape_d = nalu_hypre_TAlloc(nalu_hypre_Index, stencil_size, memory_location);
      nalu_hypre_TMemcpy(data_indices_d, data_indices[0], NALU_HYPRE_Int, stencil_size * nboxes,
                    memory_location, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TMemcpy(stencil_shape_d, stencil_shape, nalu_hypre_Index, stencil_size,
                    memory_location, NALU_HYPRE_MEMORY_HOST);
   }
   else
   {
      data_indices_d = data_indices[0];
      stencil_shape_d = stencil_shape;
   }

   nalu_hypre_ForBoxI(i, compute_boxes)
   {
      compute_box = nalu_hypre_BoxArrayBox(compute_boxes, i);

      A_dbox = nalu_hypre_BoxArrayBox(nalu_hypre_StructMatrixDataSpace(A), i);
      v_dbox = nalu_hypre_BoxArrayBox(nalu_hypre_StructVectorDataSpace(visitx), i);

      vxp = nalu_hypre_StructVectorBoxData(visitx, i);
      vyp = nalu_hypre_StructVectorBoxData(visity, i);
      vzp = nalu_hypre_StructVectorBoxData(visitz, i);

      start = nalu_hypre_BoxIMin(compute_box);
      nalu_hypre_StructMapCoarseToFine(start, cindex, stridev, startv);
      nalu_hypre_BoxGetSize(compute_box, loop_size);

#define DEVICE_VAR is_device_ptr(stencil_shape_d,vxp,vyp,vzp,data_indices_d,matrixA_data)
      nalu_hypre_BoxLoop2Begin(nalu_hypre_StructMatrixNDim(A), loop_size,
                          A_dbox, start,  stride,  Ai,
                          v_dbox, startv, stridev, vi);
      {
         NALU_HYPRE_Real lambdax, lambday, lambdaz;
         NALU_HYPRE_Real *Ap;
         NALU_HYPRE_Int si, Astenc;

         lambdax = 0.0;
         lambday = 0.0;
         lambdaz = 0.0;

         for (si = 0; si < stencil_size; si++)
         {
            Ap = matrixA_data + data_indices_d[i * stencil_size + si];

            /* compute lambdax */
            Astenc = nalu_hypre_IndexD(stencil_shape_d[si], 0);
            if (Astenc == 0)
            {
               lambdax += Ap[Ai];
            }
            else
            {
               lambdax -= Ap[Ai];
            }

            /* compute lambday */
            Astenc = nalu_hypre_IndexD(stencil_shape_d[si], 1);
            if (Astenc == 0)
            {
               lambday += Ap[Ai];
            }
            else
            {
               lambday -= Ap[Ai];
            }

            /* compute lambdaz */
            Astenc = nalu_hypre_IndexD(stencil_shape_d[si], 2);
            if (Astenc == 0)
            {
               lambdaz += Ap[Ai];
            }
            else
            {
               lambdaz -= Ap[Ai];
            }
         }

         lambdax *= lambdax;
         lambday *= lambday;
         lambdaz *= lambdaz;

         vxp[vi] = lambdax / (lambdax + lambday + lambdaz);
         vyp[vi] = lambday / (lambdax + lambday + lambdaz);
         vzp[vi] = lambdaz / (lambdax + lambday + lambdaz);
      }
      nalu_hypre_BoxLoop2End(Ai, vi);
#undef DEVICE_VAR
   }

   if (nalu_hypre_GetExecPolicy1(memory_location) == NALU_HYPRE_EXEC_DEVICE)
   {
      nalu_hypre_TFree(data_indices_d, memory_location);
      nalu_hypre_TFree(stencil_shape_d, memory_location);
   }

   return ierr;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_SparseMSGFilter
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SparseMSGFilter( nalu_hypre_StructVector *visit,
                       nalu_hypre_StructVector *e,
                       NALU_HYPRE_Int           lx,
                       NALU_HYPRE_Int           ly,
                       NALU_HYPRE_Int           lz,
                       NALU_HYPRE_Int           jump  )
{
   NALU_HYPRE_Int             ierr = 0;

   nalu_hypre_BoxArray        *compute_boxes;
   nalu_hypre_Box             *compute_box;

   nalu_hypre_Box             *e_dbox;
   nalu_hypre_Box             *v_dbox;

   NALU_HYPRE_Real            *ep;
   NALU_HYPRE_Real            *vp;

   nalu_hypre_Index            loop_size;
   nalu_hypre_Index            cindex;
   nalu_hypre_IndexRef         start;
   nalu_hypre_Index            startv;
   nalu_hypre_Index            stride;
   nalu_hypre_Index            stridev;

   NALU_HYPRE_Int              i;

   /*-----------------------------------------------------
    * Compute encoding digit and strides
    *-----------------------------------------------------*/

   nalu_hypre_SetIndex3(stride, 1, 1, 1);
   nalu_hypre_SetIndex3(stridev, 1, 1, 1);

   /*-----------------------------------------------------
    * Filter interpolated error
    *-----------------------------------------------------*/

   nalu_hypre_SetIndex3(cindex, 0, 0, 0);

   compute_boxes = nalu_hypre_StructGridBoxes(nalu_hypre_StructVectorGrid(e));
   nalu_hypre_ForBoxI(i, compute_boxes)
   {
      compute_box = nalu_hypre_BoxArrayBox(compute_boxes, i);

      e_dbox = nalu_hypre_BoxArrayBox(nalu_hypre_StructVectorDataSpace(e), i);
      v_dbox = nalu_hypre_BoxArrayBox(nalu_hypre_StructVectorDataSpace(visit), i);

      ep = nalu_hypre_StructVectorBoxData(e, i);
      vp = nalu_hypre_StructVectorBoxData(visit, i);

      start = nalu_hypre_BoxIMin(compute_box);
      nalu_hypre_StructMapCoarseToFine(start, cindex, stridev, startv);
      nalu_hypre_BoxGetSize(compute_box, loop_size);

#define DEVICE_VAR is_device_ptr(ep,vp)
      nalu_hypre_BoxLoop2Begin(nalu_hypre_StructVectorNDim(e), loop_size,
                          e_dbox, start,  stride,  ei,
                          v_dbox, startv, stridev, vi);
      {
         ep[ei] *= vp[vi];
      }
      nalu_hypre_BoxLoop2End(ei, vi);
#undef DEVICE_VAR
   }

   return ierr;
}

#endif
