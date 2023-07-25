/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_nalu_hypre_struct_ls.h"
#include "temp_multivector.h"
#include "_nalu_hypre_struct_mv.hpp"

NALU_HYPRE_Int
nalu_hypre_StructVectorSetRandomValues( nalu_hypre_StructVector *vector,
                                   NALU_HYPRE_Int           seed )
{
   nalu_hypre_Box           *v_data_box;
   NALU_HYPRE_Real          *vp;
   nalu_hypre_BoxArray      *boxes;
   nalu_hypre_Box           *box;
   nalu_hypre_Index          loop_size;
   nalu_hypre_IndexRef       start;
   nalu_hypre_Index          unit_stride;
   NALU_HYPRE_Int            i;
   NALU_HYPRE_Complex       *data            = nalu_hypre_StructVectorData(vector);
   NALU_HYPRE_Complex       *data_host       = NULL;
   NALU_HYPRE_Int            data_size       = nalu_hypre_StructVectorDataSize(vector);
   NALU_HYPRE_MemoryLocation memory_location = nalu_hypre_StructVectorMemoryLocation(vector);

   /*-----------------------------------------------------------------------
    * Set the vector coefficients
    *-----------------------------------------------------------------------*/

   //   srand( seed );
   nalu_hypre_SeedRand(seed);

   nalu_hypre_SetIndex3(unit_stride, 1, 1, 1);

   boxes = nalu_hypre_StructGridBoxes(nalu_hypre_StructVectorGrid(vector));

   if (nalu_hypre_GetExecPolicy1(memory_location) == NALU_HYPRE_EXEC_DEVICE)
   {
      data_host = nalu_hypre_CTAlloc(NALU_HYPRE_Complex, data_size, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_StructVectorData(vector) = data_host;
   }

   nalu_hypre_ForBoxI(i, boxes)
   {
      box   = nalu_hypre_BoxArrayBox(boxes, i);
      start = nalu_hypre_BoxIMin(box);

      v_data_box = nalu_hypre_BoxArrayBox(nalu_hypre_StructVectorDataSpace(vector), i);
      vp = nalu_hypre_StructVectorBoxData(vector, i);

      nalu_hypre_BoxGetSize(box, loop_size);

      nalu_hypre_SerialBoxLoop1Begin(nalu_hypre_StructVectorNDim(vector), loop_size,
                                v_data_box, start, unit_stride, vi);
      {
         vp[vi] = 2.0 * nalu_hypre_Rand() - 1.0;
      }
      nalu_hypre_SerialBoxLoop1End(vi);
   }

   if (data_host)
   {
      nalu_hypre_TMemcpy(data, data_host, NALU_HYPRE_Complex, data_size, memory_location, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_StructVectorData(vector) = data;
      nalu_hypre_TFree(data_host, NALU_HYPRE_MEMORY_HOST);
   }

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_StructSetRandomValues( void* v, NALU_HYPRE_Int seed )
{

   return nalu_hypre_StructVectorSetRandomValues( (nalu_hypre_StructVector*)v, seed );
}

NALU_HYPRE_Int
NALU_HYPRE_StructSetupInterpreter( mv_InterfaceInterpreter *i )
{
   i->CreateVector = nalu_hypre_StructKrylovCreateVector;
   i->DestroyVector = nalu_hypre_StructKrylovDestroyVector;
   i->InnerProd = nalu_hypre_StructKrylovInnerProd;
   i->CopyVector = nalu_hypre_StructKrylovCopyVector;
   i->ClearVector = nalu_hypre_StructKrylovClearVector;
   i->SetRandomValues = nalu_hypre_StructSetRandomValues;
   i->ScaleVector = nalu_hypre_StructKrylovScaleVector;
   i->Axpy = nalu_hypre_StructKrylovAxpy;

   i->CreateMultiVector = mv_TempMultiVectorCreateFromSampleVector;
   i->CopyCreateMultiVector = mv_TempMultiVectorCreateCopy;
   i->DestroyMultiVector = mv_TempMultiVectorDestroy;

   i->Width = mv_TempMultiVectorWidth;
   i->Height = mv_TempMultiVectorHeight;
   i->SetMask = mv_TempMultiVectorSetMask;
   i->CopyMultiVector = mv_TempMultiVectorCopy;
   i->ClearMultiVector = mv_TempMultiVectorClear;
   i->SetRandomVectors = mv_TempMultiVectorSetRandom;
   i->MultiInnerProd = mv_TempMultiVectorByMultiVector;
   i->MultiInnerProdDiag = mv_TempMultiVectorByMultiVectorDiag;
   i->MultiVecMat = mv_TempMultiVectorByMatrix;
   i->MultiVecMatDiag = mv_TempMultiVectorByDiagonal;
   i->MultiAxpy = mv_TempMultiVectorAxpy;
   i->MultiXapy = mv_TempMultiVectorXapy;
   i->Eval = mv_TempMultiVectorEval;

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
NALU_HYPRE_StructSetupMatvec(NALU_HYPRE_MatvecFunctions * mv)
{
   mv->MatvecCreate = nalu_hypre_StructKrylovMatvecCreate;
   mv->Matvec = nalu_hypre_StructKrylovMatvec;
   mv->MatvecDestroy = nalu_hypre_StructKrylovMatvecDestroy;

   mv->MatMultiVecCreate = NULL;
   mv->MatMultiVec = NULL;
   mv->MatMultiVecDestroy = NULL;

   return nalu_hypre_error_flag;
}
