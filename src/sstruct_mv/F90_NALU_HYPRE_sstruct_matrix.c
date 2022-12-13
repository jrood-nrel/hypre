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

#include "_hypre_sstruct_mv.h"
#include "fortran.h"

#ifdef __cplusplus
extern "C" {
#endif

/*--------------------------------------------------------------------------
 *  NALU_HYPRE_SStructMatrixCreate
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructmatrixcreate, NALU_HYPRE_SSTRUCTMATRIXCREATE)
(hypre_F90_Comm *comm,
 hypre_F90_Obj *graph,
 hypre_F90_Obj *matrix_ptr,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (NALU_HYPRE_SStructMatrixCreate(
               hypre_F90_PassComm (comm),
               hypre_F90_PassObj (NALU_HYPRE_SStructGraph, graph),
               hypre_F90_PassObjRef (NALU_HYPRE_SStructMatrix, matrix_ptr) ) );
}

/*--------------------------------------------------------------------------
 *  NALU_HYPRE_SStructMatrixDestroy
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructmatrixdestroy, NALU_HYPRE_SSTRUCTMATRIXDESTROY)
(hypre_F90_Obj *matrix,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (NALU_HYPRE_SStructMatrixDestroy(
               hypre_F90_PassObj (NALU_HYPRE_SStructMatrix, matrix) ));
}

/*--------------------------------------------------------------------------
 *  NALU_HYPRE_SStructMatrixInitialize
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructmatrixinitialize, NALU_HYPRE_SSTRUCTMATRIXINITIALIZE)
(hypre_F90_Obj *matrix,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (NALU_HYPRE_SStructMatrixInitialize(
               hypre_F90_PassObj (NALU_HYPRE_SStructMatrix, matrix) ));
}

/*--------------------------------------------------------------------------
 *  NALU_HYPRE_SStructMatrixSetValues
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructmatrixsetvalues, NALU_HYPRE_SSTRUCTMATRIXSETVALUES)
(hypre_F90_Obj *matrix,
 hypre_F90_Int *part,
 hypre_F90_IntArray *index,
 hypre_F90_Int *var,
 hypre_F90_Int *nentries,
 hypre_F90_IntArray *entries,
 hypre_F90_ComplexArray *values,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (NALU_HYPRE_SStructMatrixSetValues(
               hypre_F90_PassObj (NALU_HYPRE_SStructMatrix, matrix),
               hypre_F90_PassInt (part),
               hypre_F90_PassIntArray (index),
               hypre_F90_PassInt (var),
               hypre_F90_PassInt (nentries),
               hypre_F90_PassIntArray (entries),
               hypre_F90_PassComplexArray (values) ));
}

/*--------------------------------------------------------------------------
 *  NALU_HYPRE_SStructMatrixAddToValues
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructmatrixaddtovalues, NALU_HYPRE_SSTRUCTMATRIXADDTOVALUES)
(hypre_F90_Obj *matrix,
 hypre_F90_Int *part,
 hypre_F90_IntArray *index,
 hypre_F90_Int *var,
 hypre_F90_Int *nentries,
 hypre_F90_IntArray *entries,
 hypre_F90_ComplexArray *values,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (NALU_HYPRE_SStructMatrixAddToValues(
               hypre_F90_PassObj (NALU_HYPRE_SStructMatrix, matrix),
               hypre_F90_PassInt (part),
               hypre_F90_PassIntArray (index),
               hypre_F90_PassInt (var),
               hypre_F90_PassInt (nentries),
               hypre_F90_PassIntArray (entries),
               hypre_F90_PassComplexArray (values)) );
}

/*--------------------------------------------------------------------------
 *  NALU_HYPRE_SStructMatrixAddFEMValues
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructmatrixaddfemvalues, NALU_HYPRE_SSTRUCTMATRIXADDFEMVALUES)
(hypre_F90_Obj *matrix,
 hypre_F90_Int *part,
 hypre_F90_IntArray *index,
 hypre_F90_ComplexArray *values,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (NALU_HYPRE_SStructMatrixAddFEMValues(
               hypre_F90_PassObj (NALU_HYPRE_SStructMatrix, matrix),
               hypre_F90_PassInt (part),
               hypre_F90_PassIntArray (index),
               hypre_F90_PassComplexArray (values)) );
}

/*--------------------------------------------------------------------------
 *  NALU_HYPRE_SStructMatrixGetValues
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructmatrixgetvalues, NALU_HYPRE_SSTRUCTMATRIXGETVALUES)
(hypre_F90_Obj *matrix,
 hypre_F90_Int *part,
 hypre_F90_IntArray *index,
 hypre_F90_Int *var,
 hypre_F90_Int *nentries,
 hypre_F90_IntArray *entries,
 hypre_F90_ComplexArray *values,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (NALU_HYPRE_SStructMatrixGetValues(
               hypre_F90_PassObj (NALU_HYPRE_SStructMatrix, matrix),
               hypre_F90_PassInt (part),
               hypre_F90_PassIntArray (index),
               hypre_F90_PassInt (var),
               hypre_F90_PassInt (nentries),
               hypre_F90_PassIntArray (entries),
               hypre_F90_PassComplexArray (values)) );
}

/*--------------------------------------------------------------------------
 *  NALU_HYPRE_SStructMatrixSetBoxValues
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructmatrixsetboxvalues, NALU_HYPRE_SSTRUCTMATRIXSETBOXVALUES)
(hypre_F90_Obj *matrix,
 hypre_F90_Int *part,
 hypre_F90_IntArray *ilower,
 hypre_F90_IntArray *iupper,
 hypre_F90_Int *var,
 hypre_F90_Int *nentries,
 hypre_F90_IntArray *entries,
 hypre_F90_ComplexArray *values,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (NALU_HYPRE_SStructMatrixSetBoxValues(
               hypre_F90_PassObj (NALU_HYPRE_SStructMatrix, matrix),
               hypre_F90_PassInt (part),
               hypre_F90_PassIntArray (ilower),
               hypre_F90_PassIntArray (iupper),
               hypre_F90_PassInt (var),
               hypre_F90_PassInt (nentries),
               hypre_F90_PassIntArray (entries),
               hypre_F90_PassComplexArray (values)));
}

/*--------------------------------------------------------------------------
 *  NALU_HYPRE_SStructMatrixAddToBoxValues
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructmatrixaddtoboxvalu, NALU_HYPRE_SSTRUCTMATRIXADDTOBOXVALU)
(hypre_F90_Obj *matrix,
 hypre_F90_Int *part,
 hypre_F90_IntArray *ilower,
 hypre_F90_IntArray *iupper,
 hypre_F90_Int *var,
 hypre_F90_Int *nentries,
 hypre_F90_IntArray *entries,
 hypre_F90_ComplexArray *values,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (NALU_HYPRE_SStructMatrixAddToBoxValues(
               hypre_F90_PassObj (NALU_HYPRE_SStructMatrix, matrix),
               hypre_F90_PassInt (part),
               hypre_F90_PassIntArray (ilower),
               hypre_F90_PassIntArray (iupper),
               hypre_F90_PassInt (var),
               hypre_F90_PassInt (nentries),
               hypre_F90_PassIntArray (entries),
               hypre_F90_PassComplexArray (values) ) );
}

/*--------------------------------------------------------------------------
 *  NALU_HYPRE_SStructMatrixGetBoxValues
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructmatrixgetboxvalues, NALU_HYPRE_SSTRUCTMATRIXGETBOXVALUES)
(hypre_F90_Obj *matrix,
 hypre_F90_Int *part,
 hypre_F90_IntArray *ilower,
 hypre_F90_IntArray *iupper,
 hypre_F90_Int *var,
 hypre_F90_Int *nentries,
 hypre_F90_IntArray *entries,
 hypre_F90_ComplexArray *values,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (NALU_HYPRE_SStructMatrixGetBoxValues(
               hypre_F90_PassObj (NALU_HYPRE_SStructMatrix, matrix),
               hypre_F90_PassInt (part),
               hypre_F90_PassIntArray (ilower),
               hypre_F90_PassIntArray (iupper),
               hypre_F90_PassInt (var),
               hypre_F90_PassInt (nentries),
               hypre_F90_PassIntArray (entries),
               hypre_F90_PassComplexArray (values)));
}

/*--------------------------------------------------------------------------
 *  NALU_HYPRE_SStructMatrixAssemble
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructmatrixassemble, NALU_HYPRE_SSTRUCTMATRIXASSEMBLE)
(hypre_F90_Obj *matrix,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (NALU_HYPRE_SStructMatrixAssemble(
               hypre_F90_PassObj (NALU_HYPRE_SStructMatrix, matrix) ));
}

/*--------------------------------------------------------------------------
 *  NALU_HYPRE_SStructMatrixSetSymmetric
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructmatrixsetsymmetric, NALU_HYPRE_SSTRUCTMATRIXSETSYMMETRIC)
(hypre_F90_Obj *matrix,
 hypre_F90_Int *part,
 hypre_F90_Int *var,
 hypre_F90_Int *to_var,
 hypre_F90_Int *symmetric,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (NALU_HYPRE_SStructMatrixSetSymmetric(
               hypre_F90_PassObj (NALU_HYPRE_SStructMatrix, matrix),
               hypre_F90_PassInt (part),
               hypre_F90_PassInt (var),
               hypre_F90_PassInt (to_var),
               hypre_F90_PassInt (symmetric) ));
}

/*--------------------------------------------------------------------------
 *  NALU_HYPRE_SStructMatrixSetNSSymmetric
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructmatrixsetnssymmetr, NALU_HYPRE_SSTRUCTMATRIXSETNSSYMMETR)
(hypre_F90_Obj *matrix,
 hypre_F90_Int *symmetric,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (NALU_HYPRE_SStructMatrixSetNSSymmetric(
               hypre_F90_PassObj (NALU_HYPRE_SStructMatrix, matrix),
               hypre_F90_PassInt (symmetric) ));
}

/*--------------------------------------------------------------------------
 *  NALU_HYPRE_SStructMatrixSetObjectType
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructmatrixsetobjecttyp, NALU_HYPRE_SSTRUCTMATRIXSETOBJECTTYP)
(hypre_F90_Obj *matrix,
 hypre_F90_Int *type,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (NALU_HYPRE_SStructMatrixSetObjectType(
               hypre_F90_PassObj (NALU_HYPRE_SStructMatrix, matrix),
               hypre_F90_PassInt (type) ));
}

/*--------------------------------------------------------------------------
 *  NALU_HYPRE_SStructMatrixGetObject
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructmatrixgetobject, NALU_HYPRE_SSTRUCTMATRIXGETOBJECT)
(hypre_F90_Obj *matrix,
 hypre_F90_Obj *object,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (NALU_HYPRE_SStructMatrixGetObject(
               hypre_F90_PassObj (NALU_HYPRE_SStructMatrix, matrix),
               (void **)              object )) ;
}

/*--------------------------------------------------------------------------
 *  NALU_HYPRE_SStructMatrixPrint
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructmatrixprint, NALU_HYPRE_SSTRUCTMATRIXPRINT)
(char *filename,
 hypre_F90_Obj *matrix,
 hypre_F90_Int *all,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (NALU_HYPRE_SStructMatrixPrint(
               (char *)           filename,
               hypre_F90_PassObj (NALU_HYPRE_SStructMatrix, matrix),
               hypre_F90_PassInt (all) ) );
}

/*--------------------------------------------------------------------------
 *  NALU_HYPRE_SStructMatrixMatvec
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructmatrixmatvec, NALU_HYPRE_SSTRUCTMATRIXMATVEC)
(hypre_F90_Complex *alpha,
 hypre_F90_Obj *A,
 hypre_F90_Obj *x,
 hypre_F90_Complex *beta,
 hypre_F90_Obj *y,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (NALU_HYPRE_SStructMatrixMatvec(
               hypre_F90_PassComplex (alpha),
               hypre_F90_PassObj (NALU_HYPRE_SStructMatrix, A),
               hypre_F90_PassObj (NALU_HYPRE_SStructVector, x),
               hypre_F90_PassComplex (beta),
               hypre_F90_PassObj (NALU_HYPRE_SStructVector, y) )) ;
}

#ifdef __cplusplus
}
#endif
