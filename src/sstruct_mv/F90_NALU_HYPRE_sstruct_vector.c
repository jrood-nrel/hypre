/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * NALU_HYPRE_SStructVector interface
 *
 *****************************************************************************/

#include "_hypre_sstruct_mv.h"
#include "fortran.h"

#ifdef __cplusplus
extern "C" {
#endif

/*--------------------------------------------------------------------------
 *  NALU_HYPRE_SStructVectorCreate
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructvectorcreate, NALU_HYPRE_SSTRUCTVECTORCREATE)
(hypre_F90_Comm *comm,
 hypre_F90_Obj *grid,
 hypre_F90_Obj *vector_ptr,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (NALU_HYPRE_SStructVectorCreate(
               hypre_F90_PassComm (comm),
               hypre_F90_PassObj (NALU_HYPRE_SStructGrid, grid),
               hypre_F90_PassObjRef (NALU_HYPRE_SStructVector, vector_ptr) ) );
}

/*--------------------------------------------------------------------------
  NALU_HYPRE_SStructVectorDestroy
  *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructvectordestroy, NALU_HYPRE_SSTRUCTVECTORDESTROY)
(hypre_F90_Obj *vector,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (NALU_HYPRE_SStructVectorDestroy(
               hypre_F90_PassObj (NALU_HYPRE_SStructVector, vector) ) );
}

/*---------------------------------------------------------
  NALU_HYPRE_SStructVectorInitialize
  * ----------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructvectorinitialize, NALU_HYPRE_SSTRUCTVECTORINITIALIZE)
(hypre_F90_Obj *vector,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (NALU_HYPRE_SStructVectorInitialize(
               hypre_F90_PassObj (NALU_HYPRE_SStructVector, vector) ));
}

/*--------------------------------------------------------------------------
 *  NALU_HYPRE_SStructVectorSetValues
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructvectorsetvalues, NALU_HYPRE_SSTRUCTVECTORSETVALUES)
(hypre_F90_Obj *vector,
 hypre_F90_Int *part,
 hypre_F90_IntArray *index,
 hypre_F90_Int *var,
 hypre_F90_Complex *value,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (NALU_HYPRE_SStructVectorSetValues(
               hypre_F90_PassObj (NALU_HYPRE_SStructVector, vector),
               hypre_F90_PassInt (part),
               hypre_F90_PassIntArray (index),
               hypre_F90_PassInt (var),
               hypre_F90_PassComplexRef (value) ) );
}

/*--------------------------------------------------------------------------
 *  NALU_HYPRE_SStructVectorAddToValues
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructvectoraddtovalues, NALU_HYPRE_SSTRUCTVECTORADDTOVALUES)
(hypre_F90_Obj *vector,
 hypre_F90_Int *part,
 hypre_F90_IntArray *index,
 hypre_F90_Int *var,
 hypre_F90_Complex *value,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (NALU_HYPRE_SStructVectorAddToValues(
               hypre_F90_PassObj (NALU_HYPRE_SStructVector, vector),
               hypre_F90_PassInt (part),
               hypre_F90_PassIntArray (index),
               hypre_F90_PassInt (var),
               hypre_F90_PassComplexRef (value) ) );
}

/*--------------------------------------------------------------------------
 *  NALU_HYPRE_SStructVectorGetValues
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructvectorgetvalues, NALU_HYPRE_SSTRUCTVECTORGETVALUES)
(hypre_F90_Obj *vector,
 hypre_F90_Int *part,
 hypre_F90_IntArray *index,
 hypre_F90_Int *var,
 hypre_F90_Complex *value,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (NALU_HYPRE_SStructVectorGetValues(
               hypre_F90_PassObj (NALU_HYPRE_SStructVector, vector),
               hypre_F90_PassInt (part),
               hypre_F90_PassIntArray (index),
               hypre_F90_PassInt (var),
               hypre_F90_PassComplexRef (value) ));
}

/*--------------------------------------------------------------------------
 *  NALU_HYPRE_SStructVectorSetBoxValues
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructvectorsetboxvalues, NALU_HYPRE_SSTRUCTVECTORSETBOXVALUES)
(hypre_F90_Obj *vector,
 hypre_F90_Int *part,
 hypre_F90_IntArray *ilower,
 hypre_F90_IntArray *iupper,
 hypre_F90_Int *var,
 hypre_F90_ComplexArray *values,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (NALU_HYPRE_SStructVectorSetBoxValues(
               hypre_F90_PassObj (NALU_HYPRE_SStructVector, vector),
               hypre_F90_PassInt (part),
               hypre_F90_PassIntArray (ilower),
               hypre_F90_PassIntArray (iupper),
               hypre_F90_PassInt (var),
               hypre_F90_PassComplexArray (values) ) );
}

/*--------------------------------------------------------------------------
 *  NALU_HYPRE_SStructVectorAddToBoxValues
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructvectoraddtoboxvalu, NALU_HYPRE_SSTRUCTVECTORADDTOBOXVALU)
(hypre_F90_Obj *vector,
 hypre_F90_Int *part,
 hypre_F90_IntArray *ilower,
 hypre_F90_IntArray *iupper,
 hypre_F90_Int *var,
 hypre_F90_ComplexArray *values,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (NALU_HYPRE_SStructVectorAddToBoxValues(
               hypre_F90_PassObj (NALU_HYPRE_SStructVector, vector),
               hypre_F90_PassInt (part),
               hypre_F90_PassIntArray (ilower),
               hypre_F90_PassIntArray (iupper),
               hypre_F90_PassInt (var),
               hypre_F90_PassComplexArray (values) ) );
}

/*--------------------------------------------------------------------------
 *  NALU_HYPRE_SStructVectorGetBoxValues
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructvectorgetboxvalues, NALU_HYPRE_SSTRUCTVECTORGETBOXVALUES)
(hypre_F90_Obj *vector,
 hypre_F90_Int *part,
 hypre_F90_IntArray *ilower,
 hypre_F90_IntArray *iupper,
 hypre_F90_Int *var,
 hypre_F90_ComplexArray *values,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (NALU_HYPRE_SStructVectorGetBoxValues(
               (NALU_HYPRE_SStructVector ) * vector,
               hypre_F90_PassInt (part),
               hypre_F90_PassIntArray (ilower),
               hypre_F90_PassIntArray (iupper),
               hypre_F90_PassInt (var),
               hypre_F90_PassComplexArray (values) ) );
}

/*--------------------------------------------------------------------------
 *  NALU_HYPRE_SStructVectorAssemble
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructvectorassemble, NALU_HYPRE_SSTRUCTVECTORASSEMBLE)
(hypre_F90_Obj *vector,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (NALU_HYPRE_SStructVectorAssemble(
               hypre_F90_PassObj (NALU_HYPRE_SStructVector, vector) ));
}

/*--------------------------------------------------------------------------
 *  NALU_HYPRE_SStructVectorGather
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructvectorgather, NALU_HYPRE_SSTRUCTVECTORGATHER)
(hypre_F90_Obj *vector,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (NALU_HYPRE_SStructVectorGather(
               hypre_F90_PassObj (NALU_HYPRE_SStructVector, vector) ));
}

/*--------------------------------------------------------------------------
 *  NALU_HYPRE_SStructVectorSetConstantValues
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructvectorsetconstantv, NALU_HYPRE_SSTRUCTVECTORSETCONSTANTV)
(hypre_F90_Obj *vector,
 hypre_F90_Complex *value,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (NALU_HYPRE_SStructVectorSetConstantValues(
               hypre_F90_PassObj (NALU_HYPRE_SStructVector, vector),
               hypre_F90_PassComplex (value)));
}

/*--------------------------------------------------------------------------
 *  NALU_HYPRE_SStructVectorSetObjectType
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructvectorsetobjecttyp, NALU_HYPRE_SSTRUCTVECTORSETOBJECTTYP)
(hypre_F90_Obj *vector,
 hypre_F90_Int *type,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (NALU_HYPRE_SStructVectorSetObjectType(
               hypre_F90_PassObj (NALU_HYPRE_SStructVector, vector),
               hypre_F90_PassInt (type) ));
}

/*--------------------------------------------------------------------------
 *  NALU_HYPRE_SStructVectorGetObject
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructvectorgetobject, NALU_HYPRE_SSTRUCTVECTORGETOBJECT)
(hypre_F90_Obj *vector,
 hypre_F90_Obj *object,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (NALU_HYPRE_SStructVectorGetObject(
               hypre_F90_PassObj (NALU_HYPRE_SStructVector, vector),
               (void **)              object ));
}

/*--------------------------------------------------------------------------
 *  NALU_HYPRE_SStructVectorPrint
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructvectorprint, NALU_HYPRE_SSTRUCTVECTORPRINT)
(char *filename,
 hypre_F90_Obj *vector,
 hypre_F90_Int *all,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (NALU_HYPRE_SStructVectorPrint(
               (char * )        filename,
               hypre_F90_PassObj (NALU_HYPRE_SStructVector, vector),
               hypre_F90_PassInt (all) ) );
}

/*--------------------------------------------------------------------------
 *  NALU_HYPRE_SStructVectorCopy
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructvectorcopy, NALU_HYPRE_SSTRUCTVECTORCOPY)
(hypre_F90_Obj *x,
 hypre_F90_Obj *y,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (NALU_HYPRE_SStructVectorCopy(
               hypre_F90_PassObj (NALU_HYPRE_SStructVector, x),
               hypre_F90_PassObj (NALU_HYPRE_SStructVector, y) ) );
}

/*--------------------------------------------------------------------------
 *  NALU_HYPRE_SStructVectorScale
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructvectorscale, NALU_HYPRE_SSTRUCTVECTORSCALE)
(hypre_F90_Complex *alpha,
 hypre_F90_Obj *y,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (NALU_HYPRE_SStructVectorScale(
               hypre_F90_PassComplex (alpha),
               hypre_F90_PassObj (NALU_HYPRE_SStructVector, y) ) );
}

/*--------------------------------------------------------------------------
 *  NALU_HYPRE_SStructInnerProd
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructinnerprod, NALU_HYPRE_SSTRUCTINNERPROD)
(hypre_F90_Obj *x,
 hypre_F90_Obj *y,
 hypre_F90_Complex *result,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (NALU_HYPRE_SStructInnerProd(
               hypre_F90_PassObj (NALU_HYPRE_SStructVector, x),
               hypre_F90_PassObj (NALU_HYPRE_SStructVector, y),
               hypre_F90_PassComplexRef (result) ) );
}

/*--------------------------------------------------------------------------
 *  NALU_HYPRE_SStructAxpy
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructaxpy, NALU_HYPRE_SSTRUCTAXPY)
(hypre_F90_Complex *alpha,
 hypre_F90_Obj *x,
 hypre_F90_Obj *y,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (NALU_HYPRE_SStructAxpy(
               hypre_F90_PassComplex (alpha),
               hypre_F90_PassObj (NALU_HYPRE_SStructVector, x),
               hypre_F90_PassObj (NALU_HYPRE_SStructVector, y) ) );
}

#ifdef __cplusplus
}
#endif
