/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * NALU_HYPRE_SStructVector interface
 *
 *****************************************************************************/

#include "_nalu_hypre_sstruct_mv.h"
#include "fortran.h"

#ifdef __cplusplus
extern "C" {
#endif

/*--------------------------------------------------------------------------
 *  NALU_HYPRE_SStructVectorCreate
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_sstructvectorcreate, NALU_HYPRE_SSTRUCTVECTORCREATE)
(nalu_hypre_F90_Comm *comm,
 nalu_hypre_F90_Obj *grid,
 nalu_hypre_F90_Obj *vector_ptr,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           (NALU_HYPRE_SStructVectorCreate(
               nalu_hypre_F90_PassComm (comm),
               nalu_hypre_F90_PassObj (NALU_HYPRE_SStructGrid, grid),
               nalu_hypre_F90_PassObjRef (NALU_HYPRE_SStructVector, vector_ptr) ) );
}

/*--------------------------------------------------------------------------
  NALU_HYPRE_SStructVectorDestroy
  *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_sstructvectordestroy, NALU_HYPRE_SSTRUCTVECTORDESTROY)
(nalu_hypre_F90_Obj *vector,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           (NALU_HYPRE_SStructVectorDestroy(
               nalu_hypre_F90_PassObj (NALU_HYPRE_SStructVector, vector) ) );
}

/*---------------------------------------------------------
  NALU_HYPRE_SStructVectorInitialize
  * ----------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_sstructvectorinitialize, NALU_HYPRE_SSTRUCTVECTORINITIALIZE)
(nalu_hypre_F90_Obj *vector,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           (NALU_HYPRE_SStructVectorInitialize(
               nalu_hypre_F90_PassObj (NALU_HYPRE_SStructVector, vector) ));
}

/*--------------------------------------------------------------------------
 *  NALU_HYPRE_SStructVectorSetValues
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_sstructvectorsetvalues, NALU_HYPRE_SSTRUCTVECTORSETVALUES)
(nalu_hypre_F90_Obj *vector,
 nalu_hypre_F90_Int *part,
 nalu_hypre_F90_IntArray *index,
 nalu_hypre_F90_Int *var,
 nalu_hypre_F90_Complex *value,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           (NALU_HYPRE_SStructVectorSetValues(
               nalu_hypre_F90_PassObj (NALU_HYPRE_SStructVector, vector),
               nalu_hypre_F90_PassInt (part),
               nalu_hypre_F90_PassIntArray (index),
               nalu_hypre_F90_PassInt (var),
               nalu_hypre_F90_PassComplexRef (value) ) );
}

/*--------------------------------------------------------------------------
 *  NALU_HYPRE_SStructVectorAddToValues
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_sstructvectoraddtovalues, NALU_HYPRE_SSTRUCTVECTORADDTOVALUES)
(nalu_hypre_F90_Obj *vector,
 nalu_hypre_F90_Int *part,
 nalu_hypre_F90_IntArray *index,
 nalu_hypre_F90_Int *var,
 nalu_hypre_F90_Complex *value,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           (NALU_HYPRE_SStructVectorAddToValues(
               nalu_hypre_F90_PassObj (NALU_HYPRE_SStructVector, vector),
               nalu_hypre_F90_PassInt (part),
               nalu_hypre_F90_PassIntArray (index),
               nalu_hypre_F90_PassInt (var),
               nalu_hypre_F90_PassComplexRef (value) ) );
}

/*--------------------------------------------------------------------------
 *  NALU_HYPRE_SStructVectorGetValues
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_sstructvectorgetvalues, NALU_HYPRE_SSTRUCTVECTORGETVALUES)
(nalu_hypre_F90_Obj *vector,
 nalu_hypre_F90_Int *part,
 nalu_hypre_F90_IntArray *index,
 nalu_hypre_F90_Int *var,
 nalu_hypre_F90_Complex *value,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           (NALU_HYPRE_SStructVectorGetValues(
               nalu_hypre_F90_PassObj (NALU_HYPRE_SStructVector, vector),
               nalu_hypre_F90_PassInt (part),
               nalu_hypre_F90_PassIntArray (index),
               nalu_hypre_F90_PassInt (var),
               nalu_hypre_F90_PassComplexRef (value) ));
}

/*--------------------------------------------------------------------------
 *  NALU_HYPRE_SStructVectorSetBoxValues
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_sstructvectorsetboxvalues, NALU_HYPRE_SSTRUCTVECTORSETBOXVALUES)
(nalu_hypre_F90_Obj *vector,
 nalu_hypre_F90_Int *part,
 nalu_hypre_F90_IntArray *ilower,
 nalu_hypre_F90_IntArray *iupper,
 nalu_hypre_F90_Int *var,
 nalu_hypre_F90_ComplexArray *values,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           (NALU_HYPRE_SStructVectorSetBoxValues(
               nalu_hypre_F90_PassObj (NALU_HYPRE_SStructVector, vector),
               nalu_hypre_F90_PassInt (part),
               nalu_hypre_F90_PassIntArray (ilower),
               nalu_hypre_F90_PassIntArray (iupper),
               nalu_hypre_F90_PassInt (var),
               nalu_hypre_F90_PassComplexArray (values) ) );
}

/*--------------------------------------------------------------------------
 *  NALU_HYPRE_SStructVectorAddToBoxValues
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_sstructvectoraddtoboxvalu, NALU_HYPRE_SSTRUCTVECTORADDTOBOXVALU)
(nalu_hypre_F90_Obj *vector,
 nalu_hypre_F90_Int *part,
 nalu_hypre_F90_IntArray *ilower,
 nalu_hypre_F90_IntArray *iupper,
 nalu_hypre_F90_Int *var,
 nalu_hypre_F90_ComplexArray *values,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           (NALU_HYPRE_SStructVectorAddToBoxValues(
               nalu_hypre_F90_PassObj (NALU_HYPRE_SStructVector, vector),
               nalu_hypre_F90_PassInt (part),
               nalu_hypre_F90_PassIntArray (ilower),
               nalu_hypre_F90_PassIntArray (iupper),
               nalu_hypre_F90_PassInt (var),
               nalu_hypre_F90_PassComplexArray (values) ) );
}

/*--------------------------------------------------------------------------
 *  NALU_HYPRE_SStructVectorGetBoxValues
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_sstructvectorgetboxvalues, NALU_HYPRE_SSTRUCTVECTORGETBOXVALUES)
(nalu_hypre_F90_Obj *vector,
 nalu_hypre_F90_Int *part,
 nalu_hypre_F90_IntArray *ilower,
 nalu_hypre_F90_IntArray *iupper,
 nalu_hypre_F90_Int *var,
 nalu_hypre_F90_ComplexArray *values,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           (NALU_HYPRE_SStructVectorGetBoxValues(
               (NALU_HYPRE_SStructVector ) * vector,
               nalu_hypre_F90_PassInt (part),
               nalu_hypre_F90_PassIntArray (ilower),
               nalu_hypre_F90_PassIntArray (iupper),
               nalu_hypre_F90_PassInt (var),
               nalu_hypre_F90_PassComplexArray (values) ) );
}

/*--------------------------------------------------------------------------
 *  NALU_HYPRE_SStructVectorAssemble
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_sstructvectorassemble, NALU_HYPRE_SSTRUCTVECTORASSEMBLE)
(nalu_hypre_F90_Obj *vector,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           (NALU_HYPRE_SStructVectorAssemble(
               nalu_hypre_F90_PassObj (NALU_HYPRE_SStructVector, vector) ));
}

/*--------------------------------------------------------------------------
 *  NALU_HYPRE_SStructVectorGather
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_sstructvectorgather, NALU_HYPRE_SSTRUCTVECTORGATHER)
(nalu_hypre_F90_Obj *vector,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           (NALU_HYPRE_SStructVectorGather(
               nalu_hypre_F90_PassObj (NALU_HYPRE_SStructVector, vector) ));
}

/*--------------------------------------------------------------------------
 *  NALU_HYPRE_SStructVectorSetConstantValues
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_sstructvectorsetconstantv, NALU_HYPRE_SSTRUCTVECTORSETCONSTANTV)
(nalu_hypre_F90_Obj *vector,
 nalu_hypre_F90_Complex *value,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           (NALU_HYPRE_SStructVectorSetConstantValues(
               nalu_hypre_F90_PassObj (NALU_HYPRE_SStructVector, vector),
               nalu_hypre_F90_PassComplex (value)));
}

/*--------------------------------------------------------------------------
 *  NALU_HYPRE_SStructVectorSetObjectType
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_sstructvectorsetobjecttyp, NALU_HYPRE_SSTRUCTVECTORSETOBJECTTYP)
(nalu_hypre_F90_Obj *vector,
 nalu_hypre_F90_Int *type,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           (NALU_HYPRE_SStructVectorSetObjectType(
               nalu_hypre_F90_PassObj (NALU_HYPRE_SStructVector, vector),
               nalu_hypre_F90_PassInt (type) ));
}

/*--------------------------------------------------------------------------
 *  NALU_HYPRE_SStructVectorGetObject
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_sstructvectorgetobject, NALU_HYPRE_SSTRUCTVECTORGETOBJECT)
(nalu_hypre_F90_Obj *vector,
 nalu_hypre_F90_Obj *object,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           (NALU_HYPRE_SStructVectorGetObject(
               nalu_hypre_F90_PassObj (NALU_HYPRE_SStructVector, vector),
               (void **)              object ));
}

/*--------------------------------------------------------------------------
 *  NALU_HYPRE_SStructVectorPrint
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_sstructvectorprint, NALU_HYPRE_SSTRUCTVECTORPRINT)
(char *filename,
 nalu_hypre_F90_Obj *vector,
 nalu_hypre_F90_Int *all,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           (NALU_HYPRE_SStructVectorPrint(
               (char * )        filename,
               nalu_hypre_F90_PassObj (NALU_HYPRE_SStructVector, vector),
               nalu_hypre_F90_PassInt (all) ) );
}

/*--------------------------------------------------------------------------
 *  NALU_HYPRE_SStructVectorCopy
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_sstructvectorcopy, NALU_HYPRE_SSTRUCTVECTORCOPY)
(nalu_hypre_F90_Obj *x,
 nalu_hypre_F90_Obj *y,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           (NALU_HYPRE_SStructVectorCopy(
               nalu_hypre_F90_PassObj (NALU_HYPRE_SStructVector, x),
               nalu_hypre_F90_PassObj (NALU_HYPRE_SStructVector, y) ) );
}

/*--------------------------------------------------------------------------
 *  NALU_HYPRE_SStructVectorScale
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_sstructvectorscale, NALU_HYPRE_SSTRUCTVECTORSCALE)
(nalu_hypre_F90_Complex *alpha,
 nalu_hypre_F90_Obj *y,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           (NALU_HYPRE_SStructVectorScale(
               nalu_hypre_F90_PassComplex (alpha),
               nalu_hypre_F90_PassObj (NALU_HYPRE_SStructVector, y) ) );
}

/*--------------------------------------------------------------------------
 *  NALU_HYPRE_SStructInnerProd
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_sstructinnerprod, NALU_HYPRE_SSTRUCTINNERPROD)
(nalu_hypre_F90_Obj *x,
 nalu_hypre_F90_Obj *y,
 nalu_hypre_F90_Complex *result,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           (NALU_HYPRE_SStructInnerProd(
               nalu_hypre_F90_PassObj (NALU_HYPRE_SStructVector, x),
               nalu_hypre_F90_PassObj (NALU_HYPRE_SStructVector, y),
               nalu_hypre_F90_PassComplexRef (result) ) );
}

/*--------------------------------------------------------------------------
 *  NALU_HYPRE_SStructAxpy
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_sstructaxpy, NALU_HYPRE_SSTRUCTAXPY)
(nalu_hypre_F90_Complex *alpha,
 nalu_hypre_F90_Obj *x,
 nalu_hypre_F90_Obj *y,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           (NALU_HYPRE_SStructAxpy(
               nalu_hypre_F90_PassComplex (alpha),
               nalu_hypre_F90_PassObj (NALU_HYPRE_SStructVector, x),
               nalu_hypre_F90_PassObj (NALU_HYPRE_SStructVector, y) ) );
}

#ifdef __cplusplus
}
#endif
