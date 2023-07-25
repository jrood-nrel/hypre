/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * NALU_HYPRE_StructVector interface
 *
 *****************************************************************************/

#include "_nalu_hypre_struct_mv.h"
#include "fortran.h"

#ifdef __cplusplus
extern "C" {
#endif

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructVectorCreate
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_structvectorcreate, NALU_HYPRE_STRUCTVECTORCREATE)
( nalu_hypre_F90_Comm *comm,
  nalu_hypre_F90_Obj *grid,
  nalu_hypre_F90_Obj *vector,
  nalu_hypre_F90_Int *ierr    )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_StructVectorCreate(
                nalu_hypre_F90_PassComm (comm),
                nalu_hypre_F90_PassObj (NALU_HYPRE_StructGrid, grid),
                nalu_hypre_F90_PassObjRef (NALU_HYPRE_StructVector, vector)   ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructVectorDestroy
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_structvectordestroy, NALU_HYPRE_STRUCTVECTORDESTROY)
( nalu_hypre_F90_Obj *vector,
  nalu_hypre_F90_Int *ierr   )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_StructVectorDestroy(
                nalu_hypre_F90_PassObj (NALU_HYPRE_StructVector, vector) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructVectorInitialize
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_structvectorinitialize, NALU_HYPRE_STRUCTVECTORINITIALIZE)
( nalu_hypre_F90_Obj *vector,
  nalu_hypre_F90_Int *ierr   )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_StructVectorInitialize(
                nalu_hypre_F90_PassObj (NALU_HYPRE_StructVector, vector) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructVectorSetValues
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_structvectorsetvalues, NALU_HYPRE_STRUCTVECTORSETVALUES)
( nalu_hypre_F90_Obj *vector,
  nalu_hypre_F90_IntArray *grid_index,
  nalu_hypre_F90_Complex *values,
  nalu_hypre_F90_Int *ierr       )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_StructVectorSetValues(
                nalu_hypre_F90_PassObj (NALU_HYPRE_StructVector, vector),
                nalu_hypre_F90_PassIntArray (grid_index),
                nalu_hypre_F90_PassComplex (values)     ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructVectorSetBoxValues
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_structvectorsetboxvalues, NALU_HYPRE_STRUCTVECTORSETBOXVALUES)
( nalu_hypre_F90_Obj *vector,
  nalu_hypre_F90_IntArray *ilower,
  nalu_hypre_F90_IntArray *iupper,
  nalu_hypre_F90_ComplexArray *values,
  nalu_hypre_F90_Int *ierr   )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_StructVectorSetBoxValues(
                nalu_hypre_F90_PassObj (NALU_HYPRE_StructVector, vector),
                nalu_hypre_F90_PassIntArray (ilower),
                nalu_hypre_F90_PassIntArray (iupper),
                nalu_hypre_F90_PassComplexArray (values)  ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructVectorAddToValues
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_structvectoraddtovalues, NALU_HYPRE_STRUCTVECTORADDTOVALUES)
( nalu_hypre_F90_Obj *vector,
  nalu_hypre_F90_IntArray *grid_index,
  nalu_hypre_F90_Complex *values,
  nalu_hypre_F90_Int *ierr       )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_StructVectorAddToValues(
                nalu_hypre_F90_PassObj (NALU_HYPRE_StructVector, vector),
                nalu_hypre_F90_PassIntArray (grid_index),
                nalu_hypre_F90_PassComplex (values)     ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructVectorAddToBoxValues
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_structvectoraddtoboxvalue, NALU_HYPRE_STRUCTVECTORADDTOBOXVALUE)
( nalu_hypre_F90_Obj *vector,
  nalu_hypre_F90_IntArray *ilower,
  nalu_hypre_F90_IntArray *iupper,
  nalu_hypre_F90_ComplexArray *values,
  nalu_hypre_F90_Int *ierr   )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_StructVectorAddToBoxValues(
                nalu_hypre_F90_PassObj (NALU_HYPRE_StructVector, vector),
                nalu_hypre_F90_PassIntArray (ilower),
                nalu_hypre_F90_PassIntArray (iupper),
                nalu_hypre_F90_PassComplexArray (values)  ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructVectorScaleValues
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_structvectorscalevalues, NALU_HYPRE_STRUCTVECTORSCALEVALUES)
( nalu_hypre_F90_Obj *vector,
  nalu_hypre_F90_Complex *factor,
  nalu_hypre_F90_Int *ierr       )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_StructVectorScaleValues(
                nalu_hypre_F90_PassObj (NALU_HYPRE_StructVector, vector),
                nalu_hypre_F90_PassComplex (factor) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructVectorGetValues
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_structvectorgetvalues, NALU_HYPRE_STRUCTVECTORGETVALUES)
( nalu_hypre_F90_Obj *vector,
  nalu_hypre_F90_IntArray *grid_index,
  nalu_hypre_F90_Complex *values_ptr,
  nalu_hypre_F90_Int *ierr       )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_StructVectorGetValues(
                nalu_hypre_F90_PassObj (NALU_HYPRE_StructVector, vector),
                nalu_hypre_F90_PassIntArray (grid_index),
                nalu_hypre_F90_PassComplexRef (values_ptr) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructVectorGetBoxValues
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_structvectorgetboxvalues, NALU_HYPRE_STRUCTVECTORGETBOXVALUES)
( nalu_hypre_F90_Obj *vector,
  nalu_hypre_F90_IntArray *ilower,
  nalu_hypre_F90_IntArray *iupper,
  nalu_hypre_F90_ComplexArray *values,
  nalu_hypre_F90_Int *ierr   )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_StructVectorGetBoxValues(
                nalu_hypre_F90_PassObj (NALU_HYPRE_StructVector, vector),
                nalu_hypre_F90_PassIntArray (ilower),
                nalu_hypre_F90_PassIntArray (iupper),
                nalu_hypre_F90_PassComplexArray (values)  ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructVectorAssemble
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_structvectorassemble, NALU_HYPRE_STRUCTVECTORASSEMBLE)
( nalu_hypre_F90_Obj *vector,
  nalu_hypre_F90_Int *ierr   )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_StructVectorAssemble(
                nalu_hypre_F90_PassObj (NALU_HYPRE_StructVector, vector) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructVectorSetNumGhost
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_structvectorsetnumghost, NALU_HYPRE_STRUCTVECTORSETNUMGHOST)
( nalu_hypre_F90_Obj *vector,
  nalu_hypre_F90_IntArray *num_ghost,
  nalu_hypre_F90_Int *ierr      )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_StructVectorSetNumGhost(
                nalu_hypre_F90_PassObj (NALU_HYPRE_StructVector, vector),
                nalu_hypre_F90_PassIntArray (num_ghost) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructVectorCopy
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_structvectorcopy, NALU_HYPRE_STRUCTVECTORCOPY)
( nalu_hypre_F90_Obj *x,
  nalu_hypre_F90_Obj *y,
  nalu_hypre_F90_Int *ierr )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_StructVectorCopy(
                nalu_hypre_F90_PassObj (NALU_HYPRE_StructVector, x),
                nalu_hypre_F90_PassObj (NALU_HYPRE_StructVector, y) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructVectorSetConstantValues
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_structvectorsetconstantva, NALU_HYPRE_STRUCTVECTORSETCONSTANTVA)
( nalu_hypre_F90_Obj *vector,
  nalu_hypre_F90_Complex *values,
  nalu_hypre_F90_Int *ierr   )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_StructVectorSetConstantValues(
                nalu_hypre_F90_PassObj (NALU_HYPRE_StructVector, vector),
                nalu_hypre_F90_PassComplex (values) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructVectorGetMigrateCommPkg
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_structvectorgetmigratecom, NALU_HYPRE_STRUCTVECTORGETMIGRATECOM)
( nalu_hypre_F90_Obj *from_vector,
  nalu_hypre_F90_Obj *to_vector,
  nalu_hypre_F90_Obj *comm_pkg,
  nalu_hypre_F90_Int *ierr        )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_StructVectorGetMigrateCommPkg(
                nalu_hypre_F90_PassObj (NALU_HYPRE_StructVector, from_vector),
                nalu_hypre_F90_PassObj (NALU_HYPRE_StructVector, to_vector),
                nalu_hypre_F90_PassObjRef (NALU_HYPRE_CommPkg, comm_pkg)    ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructVectorMigrate
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_structvectormigrate, NALU_HYPRE_STRUCTVECTORMIGRATE)
( nalu_hypre_F90_Obj *comm_pkg,
  nalu_hypre_F90_Obj *from_vector,
  nalu_hypre_F90_Obj *to_vector,
  nalu_hypre_F90_Int *ierr        )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_StructVectorMigrate(
                nalu_hypre_F90_PassObj (NALU_HYPRE_CommPkg, comm_pkg),
                nalu_hypre_F90_PassObj (NALU_HYPRE_StructVector, from_vector),
                nalu_hypre_F90_PassObj (NALU_HYPRE_StructVector, to_vector)   ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_CommPkgDestroy
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_destroycommpkg, NALU_HYPRE_DESTROYCOMMPKG)
( nalu_hypre_F90_Obj *comm_pkg,
  nalu_hypre_F90_Int *ierr     )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_CommPkgDestroy(
                nalu_hypre_F90_PassObj (NALU_HYPRE_CommPkg, comm_pkg) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructVectorPrint
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_structvectorprint, NALU_HYPRE_STRUCTVECTORPRINT)
(
   nalu_hypre_F90_Obj *vector,
   nalu_hypre_F90_Int *all,
   nalu_hypre_F90_Int *ierr )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_StructVectorPrint(
                "NALU_HYPRE_StructVector.out",
                nalu_hypre_F90_PassObj (NALU_HYPRE_StructVector, vector),
                nalu_hypre_F90_PassInt (all)) );
}

#ifdef __cplusplus
}
#endif
