/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * NALU_HYPRE_StructVector interface
 *
 *****************************************************************************/

#include "_hypre_struct_mv.h"
#include "fortran.h"

#ifdef __cplusplus
extern "C" {
#endif

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructVectorCreate
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structvectorcreate, NALU_HYPRE_STRUCTVECTORCREATE)
( hypre_F90_Comm *comm,
  hypre_F90_Obj *grid,
  hypre_F90_Obj *vector,
  hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_StructVectorCreate(
                hypre_F90_PassComm (comm),
                hypre_F90_PassObj (NALU_HYPRE_StructGrid, grid),
                hypre_F90_PassObjRef (NALU_HYPRE_StructVector, vector)   ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructVectorDestroy
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structvectordestroy, NALU_HYPRE_STRUCTVECTORDESTROY)
( hypre_F90_Obj *vector,
  hypre_F90_Int *ierr   )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_StructVectorDestroy(
                hypre_F90_PassObj (NALU_HYPRE_StructVector, vector) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructVectorInitialize
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structvectorinitialize, NALU_HYPRE_STRUCTVECTORINITIALIZE)
( hypre_F90_Obj *vector,
  hypre_F90_Int *ierr   )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_StructVectorInitialize(
                hypre_F90_PassObj (NALU_HYPRE_StructVector, vector) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructVectorSetValues
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structvectorsetvalues, NALU_HYPRE_STRUCTVECTORSETVALUES)
( hypre_F90_Obj *vector,
  hypre_F90_IntArray *grid_index,
  hypre_F90_Complex *values,
  hypre_F90_Int *ierr       )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_StructVectorSetValues(
                hypre_F90_PassObj (NALU_HYPRE_StructVector, vector),
                hypre_F90_PassIntArray (grid_index),
                hypre_F90_PassComplex (values)     ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructVectorSetBoxValues
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structvectorsetboxvalues, NALU_HYPRE_STRUCTVECTORSETBOXVALUES)
( hypre_F90_Obj *vector,
  hypre_F90_IntArray *ilower,
  hypre_F90_IntArray *iupper,
  hypre_F90_ComplexArray *values,
  hypre_F90_Int *ierr   )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_StructVectorSetBoxValues(
                hypre_F90_PassObj (NALU_HYPRE_StructVector, vector),
                hypre_F90_PassIntArray (ilower),
                hypre_F90_PassIntArray (iupper),
                hypre_F90_PassComplexArray (values)  ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructVectorAddToValues
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structvectoraddtovalues, NALU_HYPRE_STRUCTVECTORADDTOVALUES)
( hypre_F90_Obj *vector,
  hypre_F90_IntArray *grid_index,
  hypre_F90_Complex *values,
  hypre_F90_Int *ierr       )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_StructVectorAddToValues(
                hypre_F90_PassObj (NALU_HYPRE_StructVector, vector),
                hypre_F90_PassIntArray (grid_index),
                hypre_F90_PassComplex (values)     ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructVectorAddToBoxValues
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structvectoraddtoboxvalue, NALU_HYPRE_STRUCTVECTORADDTOBOXVALUE)
( hypre_F90_Obj *vector,
  hypre_F90_IntArray *ilower,
  hypre_F90_IntArray *iupper,
  hypre_F90_ComplexArray *values,
  hypre_F90_Int *ierr   )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_StructVectorAddToBoxValues(
                hypre_F90_PassObj (NALU_HYPRE_StructVector, vector),
                hypre_F90_PassIntArray (ilower),
                hypre_F90_PassIntArray (iupper),
                hypre_F90_PassComplexArray (values)  ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructVectorScaleValues
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structvectorscalevalues, NALU_HYPRE_STRUCTVECTORSCALEVALUES)
( hypre_F90_Obj *vector,
  hypre_F90_Complex *factor,
  hypre_F90_Int *ierr       )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_StructVectorScaleValues(
                hypre_F90_PassObj (NALU_HYPRE_StructVector, vector),
                hypre_F90_PassComplex (factor) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructVectorGetValues
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structvectorgetvalues, NALU_HYPRE_STRUCTVECTORGETVALUES)
( hypre_F90_Obj *vector,
  hypre_F90_IntArray *grid_index,
  hypre_F90_Complex *values_ptr,
  hypre_F90_Int *ierr       )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_StructVectorGetValues(
                hypre_F90_PassObj (NALU_HYPRE_StructVector, vector),
                hypre_F90_PassIntArray (grid_index),
                hypre_F90_PassComplexRef (values_ptr) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructVectorGetBoxValues
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structvectorgetboxvalues, NALU_HYPRE_STRUCTVECTORGETBOXVALUES)
( hypre_F90_Obj *vector,
  hypre_F90_IntArray *ilower,
  hypre_F90_IntArray *iupper,
  hypre_F90_ComplexArray *values,
  hypre_F90_Int *ierr   )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_StructVectorGetBoxValues(
                hypre_F90_PassObj (NALU_HYPRE_StructVector, vector),
                hypre_F90_PassIntArray (ilower),
                hypre_F90_PassIntArray (iupper),
                hypre_F90_PassComplexArray (values)  ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructVectorAssemble
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structvectorassemble, NALU_HYPRE_STRUCTVECTORASSEMBLE)
( hypre_F90_Obj *vector,
  hypre_F90_Int *ierr   )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_StructVectorAssemble(
                hypre_F90_PassObj (NALU_HYPRE_StructVector, vector) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructVectorSetNumGhost
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structvectorsetnumghost, NALU_HYPRE_STRUCTVECTORSETNUMGHOST)
( hypre_F90_Obj *vector,
  hypre_F90_IntArray *num_ghost,
  hypre_F90_Int *ierr      )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_StructVectorSetNumGhost(
                hypre_F90_PassObj (NALU_HYPRE_StructVector, vector),
                hypre_F90_PassIntArray (num_ghost) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructVectorCopy
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structvectorcopy, NALU_HYPRE_STRUCTVECTORCOPY)
( hypre_F90_Obj *x,
  hypre_F90_Obj *y,
  hypre_F90_Int *ierr )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_StructVectorCopy(
                hypre_F90_PassObj (NALU_HYPRE_StructVector, x),
                hypre_F90_PassObj (NALU_HYPRE_StructVector, y) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructVectorSetConstantValues
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structvectorsetconstantva, NALU_HYPRE_STRUCTVECTORSETCONSTANTVA)
( hypre_F90_Obj *vector,
  hypre_F90_Complex *values,
  hypre_F90_Int *ierr   )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_StructVectorSetConstantValues(
                hypre_F90_PassObj (NALU_HYPRE_StructVector, vector),
                hypre_F90_PassComplex (values) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructVectorGetMigrateCommPkg
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structvectorgetmigratecom, NALU_HYPRE_STRUCTVECTORGETMIGRATECOM)
( hypre_F90_Obj *from_vector,
  hypre_F90_Obj *to_vector,
  hypre_F90_Obj *comm_pkg,
  hypre_F90_Int *ierr        )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_StructVectorGetMigrateCommPkg(
                hypre_F90_PassObj (NALU_HYPRE_StructVector, from_vector),
                hypre_F90_PassObj (NALU_HYPRE_StructVector, to_vector),
                hypre_F90_PassObjRef (NALU_HYPRE_CommPkg, comm_pkg)    ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructVectorMigrate
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structvectormigrate, NALU_HYPRE_STRUCTVECTORMIGRATE)
( hypre_F90_Obj *comm_pkg,
  hypre_F90_Obj *from_vector,
  hypre_F90_Obj *to_vector,
  hypre_F90_Int *ierr        )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_StructVectorMigrate(
                hypre_F90_PassObj (NALU_HYPRE_CommPkg, comm_pkg),
                hypre_F90_PassObj (NALU_HYPRE_StructVector, from_vector),
                hypre_F90_PassObj (NALU_HYPRE_StructVector, to_vector)   ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_CommPkgDestroy
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_destroycommpkg, NALU_HYPRE_DESTROYCOMMPKG)
( hypre_F90_Obj *comm_pkg,
  hypre_F90_Int *ierr     )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_CommPkgDestroy(
                hypre_F90_PassObj (NALU_HYPRE_CommPkg, comm_pkg) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructVectorPrint
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structvectorprint, NALU_HYPRE_STRUCTVECTORPRINT)
(
   hypre_F90_Obj *vector,
   hypre_F90_Int *all,
   hypre_F90_Int *ierr )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_StructVectorPrint(
                "NALU_HYPRE_StructVector.out",
                hypre_F90_PassObj (NALU_HYPRE_StructVector, vector),
                hypre_F90_PassInt (all)) );
}

#ifdef __cplusplus
}
#endif
