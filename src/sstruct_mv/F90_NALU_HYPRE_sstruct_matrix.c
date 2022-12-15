/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * NALU_HYPRE_SStructMatrix interface
 *
 *****************************************************************************/

#include "_nalu_hypre_sstruct_mv.h"
#include "fortran.h"

#ifdef __cplusplus
extern "C" {
#endif

/*--------------------------------------------------------------------------
 *  NALU_HYPRE_SStructMatrixCreate
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_sstructmatrixcreate, NALU_HYPRE_SSTRUCTMATRIXCREATE)
(nalu_hypre_F90_Comm *comm,
 nalu_hypre_F90_Obj *graph,
 nalu_hypre_F90_Obj *matrix_ptr,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           (NALU_HYPRE_SStructMatrixCreate(
               nalu_hypre_F90_PassComm (comm),
               nalu_hypre_F90_PassObj (NALU_HYPRE_SStructGraph, graph),
               nalu_hypre_F90_PassObjRef (NALU_HYPRE_SStructMatrix, matrix_ptr) ) );
}

/*--------------------------------------------------------------------------
 *  NALU_HYPRE_SStructMatrixDestroy
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_sstructmatrixdestroy, NALU_HYPRE_SSTRUCTMATRIXDESTROY)
(nalu_hypre_F90_Obj *matrix,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           (NALU_HYPRE_SStructMatrixDestroy(
               nalu_hypre_F90_PassObj (NALU_HYPRE_SStructMatrix, matrix) ));
}

/*--------------------------------------------------------------------------
 *  NALU_HYPRE_SStructMatrixInitialize
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_sstructmatrixinitialize, NALU_HYPRE_SSTRUCTMATRIXINITIALIZE)
(nalu_hypre_F90_Obj *matrix,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           (NALU_HYPRE_SStructMatrixInitialize(
               nalu_hypre_F90_PassObj (NALU_HYPRE_SStructMatrix, matrix) ));
}

/*--------------------------------------------------------------------------
 *  NALU_HYPRE_SStructMatrixSetValues
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_sstructmatrixsetvalues, NALU_HYPRE_SSTRUCTMATRIXSETVALUES)
(nalu_hypre_F90_Obj *matrix,
 nalu_hypre_F90_Int *part,
 nalu_hypre_F90_IntArray *index,
 nalu_hypre_F90_Int *var,
 nalu_hypre_F90_Int *nentries,
 nalu_hypre_F90_IntArray *entries,
 nalu_hypre_F90_ComplexArray *values,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           (NALU_HYPRE_SStructMatrixSetValues(
               nalu_hypre_F90_PassObj (NALU_HYPRE_SStructMatrix, matrix),
               nalu_hypre_F90_PassInt (part),
               nalu_hypre_F90_PassIntArray (index),
               nalu_hypre_F90_PassInt (var),
               nalu_hypre_F90_PassInt (nentries),
               nalu_hypre_F90_PassIntArray (entries),
               nalu_hypre_F90_PassComplexArray (values) ));
}

/*--------------------------------------------------------------------------
 *  NALU_HYPRE_SStructMatrixAddToValues
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_sstructmatrixaddtovalues, NALU_HYPRE_SSTRUCTMATRIXADDTOVALUES)
(nalu_hypre_F90_Obj *matrix,
 nalu_hypre_F90_Int *part,
 nalu_hypre_F90_IntArray *index,
 nalu_hypre_F90_Int *var,
 nalu_hypre_F90_Int *nentries,
 nalu_hypre_F90_IntArray *entries,
 nalu_hypre_F90_ComplexArray *values,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           (NALU_HYPRE_SStructMatrixAddToValues(
               nalu_hypre_F90_PassObj (NALU_HYPRE_SStructMatrix, matrix),
               nalu_hypre_F90_PassInt (part),
               nalu_hypre_F90_PassIntArray (index),
               nalu_hypre_F90_PassInt (var),
               nalu_hypre_F90_PassInt (nentries),
               nalu_hypre_F90_PassIntArray (entries),
               nalu_hypre_F90_PassComplexArray (values)) );
}

/*--------------------------------------------------------------------------
 *  NALU_HYPRE_SStructMatrixAddFEMValues
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_sstructmatrixaddfemvalues, NALU_HYPRE_SSTRUCTMATRIXADDFEMVALUES)
(nalu_hypre_F90_Obj *matrix,
 nalu_hypre_F90_Int *part,
 nalu_hypre_F90_IntArray *index,
 nalu_hypre_F90_ComplexArray *values,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           (NALU_HYPRE_SStructMatrixAddFEMValues(
               nalu_hypre_F90_PassObj (NALU_HYPRE_SStructMatrix, matrix),
               nalu_hypre_F90_PassInt (part),
               nalu_hypre_F90_PassIntArray (index),
               nalu_hypre_F90_PassComplexArray (values)) );
}

/*--------------------------------------------------------------------------
 *  NALU_HYPRE_SStructMatrixGetValues
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_sstructmatrixgetvalues, NALU_HYPRE_SSTRUCTMATRIXGETVALUES)
(nalu_hypre_F90_Obj *matrix,
 nalu_hypre_F90_Int *part,
 nalu_hypre_F90_IntArray *index,
 nalu_hypre_F90_Int *var,
 nalu_hypre_F90_Int *nentries,
 nalu_hypre_F90_IntArray *entries,
 nalu_hypre_F90_ComplexArray *values,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           (NALU_HYPRE_SStructMatrixGetValues(
               nalu_hypre_F90_PassObj (NALU_HYPRE_SStructMatrix, matrix),
               nalu_hypre_F90_PassInt (part),
               nalu_hypre_F90_PassIntArray (index),
               nalu_hypre_F90_PassInt (var),
               nalu_hypre_F90_PassInt (nentries),
               nalu_hypre_F90_PassIntArray (entries),
               nalu_hypre_F90_PassComplexArray (values)) );
}

/*--------------------------------------------------------------------------
 *  NALU_HYPRE_SStructMatrixSetBoxValues
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_sstructmatrixsetboxvalues, NALU_HYPRE_SSTRUCTMATRIXSETBOXVALUES)
(nalu_hypre_F90_Obj *matrix,
 nalu_hypre_F90_Int *part,
 nalu_hypre_F90_IntArray *ilower,
 nalu_hypre_F90_IntArray *iupper,
 nalu_hypre_F90_Int *var,
 nalu_hypre_F90_Int *nentries,
 nalu_hypre_F90_IntArray *entries,
 nalu_hypre_F90_ComplexArray *values,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           (NALU_HYPRE_SStructMatrixSetBoxValues(
               nalu_hypre_F90_PassObj (NALU_HYPRE_SStructMatrix, matrix),
               nalu_hypre_F90_PassInt (part),
               nalu_hypre_F90_PassIntArray (ilower),
               nalu_hypre_F90_PassIntArray (iupper),
               nalu_hypre_F90_PassInt (var),
               nalu_hypre_F90_PassInt (nentries),
               nalu_hypre_F90_PassIntArray (entries),
               nalu_hypre_F90_PassComplexArray (values)));
}

/*--------------------------------------------------------------------------
 *  NALU_HYPRE_SStructMatrixAddToBoxValues
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_sstructmatrixaddtoboxvalu, NALU_HYPRE_SSTRUCTMATRIXADDTOBOXVALU)
(nalu_hypre_F90_Obj *matrix,
 nalu_hypre_F90_Int *part,
 nalu_hypre_F90_IntArray *ilower,
 nalu_hypre_F90_IntArray *iupper,
 nalu_hypre_F90_Int *var,
 nalu_hypre_F90_Int *nentries,
 nalu_hypre_F90_IntArray *entries,
 nalu_hypre_F90_ComplexArray *values,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           (NALU_HYPRE_SStructMatrixAddToBoxValues(
               nalu_hypre_F90_PassObj (NALU_HYPRE_SStructMatrix, matrix),
               nalu_hypre_F90_PassInt (part),
               nalu_hypre_F90_PassIntArray (ilower),
               nalu_hypre_F90_PassIntArray (iupper),
               nalu_hypre_F90_PassInt (var),
               nalu_hypre_F90_PassInt (nentries),
               nalu_hypre_F90_PassIntArray (entries),
               nalu_hypre_F90_PassComplexArray (values) ) );
}

/*--------------------------------------------------------------------------
 *  NALU_HYPRE_SStructMatrixGetBoxValues
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_sstructmatrixgetboxvalues, NALU_HYPRE_SSTRUCTMATRIXGETBOXVALUES)
(nalu_hypre_F90_Obj *matrix,
 nalu_hypre_F90_Int *part,
 nalu_hypre_F90_IntArray *ilower,
 nalu_hypre_F90_IntArray *iupper,
 nalu_hypre_F90_Int *var,
 nalu_hypre_F90_Int *nentries,
 nalu_hypre_F90_IntArray *entries,
 nalu_hypre_F90_ComplexArray *values,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           (NALU_HYPRE_SStructMatrixGetBoxValues(
               nalu_hypre_F90_PassObj (NALU_HYPRE_SStructMatrix, matrix),
               nalu_hypre_F90_PassInt (part),
               nalu_hypre_F90_PassIntArray (ilower),
               nalu_hypre_F90_PassIntArray (iupper),
               nalu_hypre_F90_PassInt (var),
               nalu_hypre_F90_PassInt (nentries),
               nalu_hypre_F90_PassIntArray (entries),
               nalu_hypre_F90_PassComplexArray (values)));
}

/*--------------------------------------------------------------------------
 *  NALU_HYPRE_SStructMatrixAssemble
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_sstructmatrixassemble, NALU_HYPRE_SSTRUCTMATRIXASSEMBLE)
(nalu_hypre_F90_Obj *matrix,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           (NALU_HYPRE_SStructMatrixAssemble(
               nalu_hypre_F90_PassObj (NALU_HYPRE_SStructMatrix, matrix) ));
}

/*--------------------------------------------------------------------------
 *  NALU_HYPRE_SStructMatrixSetSymmetric
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_sstructmatrixsetsymmetric, NALU_HYPRE_SSTRUCTMATRIXSETSYMMETRIC)
(nalu_hypre_F90_Obj *matrix,
 nalu_hypre_F90_Int *part,
 nalu_hypre_F90_Int *var,
 nalu_hypre_F90_Int *to_var,
 nalu_hypre_F90_Int *symmetric,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           (NALU_HYPRE_SStructMatrixSetSymmetric(
               nalu_hypre_F90_PassObj (NALU_HYPRE_SStructMatrix, matrix),
               nalu_hypre_F90_PassInt (part),
               nalu_hypre_F90_PassInt (var),
               nalu_hypre_F90_PassInt (to_var),
               nalu_hypre_F90_PassInt (symmetric) ));
}

/*--------------------------------------------------------------------------
 *  NALU_HYPRE_SStructMatrixSetNSSymmetric
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_sstructmatrixsetnssymmetr, NALU_HYPRE_SSTRUCTMATRIXSETNSSYMMETR)
(nalu_hypre_F90_Obj *matrix,
 nalu_hypre_F90_Int *symmetric,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           (NALU_HYPRE_SStructMatrixSetNSSymmetric(
               nalu_hypre_F90_PassObj (NALU_HYPRE_SStructMatrix, matrix),
               nalu_hypre_F90_PassInt (symmetric) ));
}

/*--------------------------------------------------------------------------
 *  NALU_HYPRE_SStructMatrixSetObjectType
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_sstructmatrixsetobjecttyp, NALU_HYPRE_SSTRUCTMATRIXSETOBJECTTYP)
(nalu_hypre_F90_Obj *matrix,
 nalu_hypre_F90_Int *type,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           (NALU_HYPRE_SStructMatrixSetObjectType(
               nalu_hypre_F90_PassObj (NALU_HYPRE_SStructMatrix, matrix),
               nalu_hypre_F90_PassInt (type) ));
}

/*--------------------------------------------------------------------------
 *  NALU_HYPRE_SStructMatrixGetObject
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_sstructmatrixgetobject, NALU_HYPRE_SSTRUCTMATRIXGETOBJECT)
(nalu_hypre_F90_Obj *matrix,
 nalu_hypre_F90_Obj *object,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           (NALU_HYPRE_SStructMatrixGetObject(
               nalu_hypre_F90_PassObj (NALU_HYPRE_SStructMatrix, matrix),
               (void **)              object )) ;
}

/*--------------------------------------------------------------------------
 *  NALU_HYPRE_SStructMatrixPrint
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_sstructmatrixprint, NALU_HYPRE_SSTRUCTMATRIXPRINT)
(char *filename,
 nalu_hypre_F90_Obj *matrix,
 nalu_hypre_F90_Int *all,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           (NALU_HYPRE_SStructMatrixPrint(
               (char *)           filename,
               nalu_hypre_F90_PassObj (NALU_HYPRE_SStructMatrix, matrix),
               nalu_hypre_F90_PassInt (all) ) );
}

/*--------------------------------------------------------------------------
 *  NALU_HYPRE_SStructMatrixMatvec
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_sstructmatrixmatvec, NALU_HYPRE_SSTRUCTMATRIXMATVEC)
(nalu_hypre_F90_Complex *alpha,
 nalu_hypre_F90_Obj *A,
 nalu_hypre_F90_Obj *x,
 nalu_hypre_F90_Complex *beta,
 nalu_hypre_F90_Obj *y,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           (NALU_HYPRE_SStructMatrixMatvec(
               nalu_hypre_F90_PassComplex (alpha),
               nalu_hypre_F90_PassObj (NALU_HYPRE_SStructMatrix, A),
               nalu_hypre_F90_PassObj (NALU_HYPRE_SStructVector, x),
               nalu_hypre_F90_PassComplex (beta),
               nalu_hypre_F90_PassObj (NALU_HYPRE_SStructVector, y) )) ;
}

#ifdef __cplusplus
}
#endif
