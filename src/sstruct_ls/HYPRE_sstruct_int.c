/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_nalu_hypre_sstruct_ls.h"
#include "interpreter.h"
#include "NALU_HYPRE_MatvecFunctions.h"
#include "temp_multivector.h"


NALU_HYPRE_Int
nalu_hypre_SStructPVectorSetRandomValues( nalu_hypre_SStructPVector *pvector, NALU_HYPRE_Int seed )
{
   NALU_HYPRE_Int ierr = 0;
   NALU_HYPRE_Int           nvars = nalu_hypre_SStructPVectorNVars(pvector);
   nalu_hypre_StructVector *svector;
   NALU_HYPRE_Int           var;

   nalu_hypre_SeedRand( seed );

   for (var = 0; var < nvars; var++)
   {
      svector = nalu_hypre_SStructPVectorSVector(pvector, var);
      seed = nalu_hypre_RandI();
      nalu_hypre_StructVectorSetRandomValues(svector, seed);
   }

   return ierr;
}

NALU_HYPRE_Int
nalu_hypre_SStructVectorSetRandomValues( nalu_hypre_SStructVector *vector, NALU_HYPRE_Int seed )
{
   NALU_HYPRE_Int ierr = 0;
   NALU_HYPRE_Int             nparts = nalu_hypre_SStructVectorNParts(vector);
   nalu_hypre_SStructPVector *pvector;
   NALU_HYPRE_Int             part;

   nalu_hypre_SeedRand( seed );

   for (part = 0; part < nparts; part++)
   {
      pvector = nalu_hypre_SStructVectorPVector(vector, part);
      seed = nalu_hypre_RandI();
      nalu_hypre_SStructPVectorSetRandomValues(pvector, seed);
   }

   return ierr;
}

NALU_HYPRE_Int
nalu_hypre_SStructSetRandomValues( void* v, NALU_HYPRE_Int seed )
{

   return nalu_hypre_SStructVectorSetRandomValues( (nalu_hypre_SStructVector*)v, seed );
}

NALU_HYPRE_Int
NALU_HYPRE_SStructSetupInterpreter( mv_InterfaceInterpreter *i )
{
   i->CreateVector = nalu_hypre_SStructKrylovCreateVector;
   i->DestroyVector = nalu_hypre_SStructKrylovDestroyVector;
   i->InnerProd = nalu_hypre_SStructKrylovInnerProd;
   i->CopyVector = nalu_hypre_SStructKrylovCopyVector;
   i->ClearVector = nalu_hypre_SStructKrylovClearVector;
   i->SetRandomValues = nalu_hypre_SStructSetRandomValues;
   i->ScaleVector = nalu_hypre_SStructKrylovScaleVector;
   i->Axpy = nalu_hypre_SStructKrylovAxpy;

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

   return 0;
}

NALU_HYPRE_Int
NALU_HYPRE_SStructSetupMatvec(NALU_HYPRE_MatvecFunctions * mv)
{
   mv->MatvecCreate = nalu_hypre_SStructKrylovMatvecCreate;
   mv->Matvec = nalu_hypre_SStructKrylovMatvec;
   mv->MatvecDestroy = nalu_hypre_SStructKrylovMatvecDestroy;

   mv->MatMultiVecCreate = NULL;
   mv->MatMultiVec = NULL;
   mv->MatMultiVecDestroy = NULL;

   return 0;
}
