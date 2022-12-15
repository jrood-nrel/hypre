/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * NALU_HYPRE_IJMatrix Fortran interface
 *
 *****************************************************************************/

#include "./_nalu_hypre_IJ_mv.h"
#include "fortran.h"

#ifdef __cplusplus
extern "C" {
#endif

/*--------------------------------------------------------------------------
 * NALU_HYPRE_IJVectorCreate
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_ijvectorcreate, NALU_HYPRE_IJVECTORCREATE)
( nalu_hypre_F90_Comm *comm,
  nalu_hypre_F90_BigInt *jlower,
  nalu_hypre_F90_BigInt *jupper,
  nalu_hypre_F90_Obj *vector,
  nalu_hypre_F90_Int *ierr    )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_IJVectorCreate(
                nalu_hypre_F90_PassComm (comm),
                nalu_hypre_F90_PassBigInt (jlower),
                nalu_hypre_F90_PassBigInt (jupper),
                nalu_hypre_F90_PassObjRef (NALU_HYPRE_IJVector, vector)  ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_IJVectorDestroy
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_ijvectordestroy, NALU_HYPRE_IJVECTORDESTROY)
( nalu_hypre_F90_Obj *vector,
  nalu_hypre_F90_Int *ierr    )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_IJVectorDestroy(
                nalu_hypre_F90_PassObj (NALU_HYPRE_IJVector, vector) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_IJVectorInitialize
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_ijvectorinitialize, NALU_HYPRE_IJVECTORINITIALIZE)
( nalu_hypre_F90_Obj *vector,
  nalu_hypre_F90_Int *ierr    )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_IJVectorInitialize(
                nalu_hypre_F90_PassObj (NALU_HYPRE_IJVector, vector) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_IJVectorSetValues
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_ijvectorsetvalues, NALU_HYPRE_IJVECTORSETVALUES)
( nalu_hypre_F90_Obj *vector,
  nalu_hypre_F90_Int *num_values,
  nalu_hypre_F90_BigIntArray *indices,
  nalu_hypre_F90_ComplexArray *values,
  nalu_hypre_F90_Int *ierr        )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_IJVectorSetValues(
                nalu_hypre_F90_PassObj (NALU_HYPRE_IJVector, vector),
                nalu_hypre_F90_PassInt (num_values),
                nalu_hypre_F90_PassBigIntArray (indices),
                nalu_hypre_F90_PassComplexArray (values)      ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_IJVectorAddToValues
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_ijvectoraddtovalues, NALU_HYPRE_IJVECTORADDTOVALUES)
( nalu_hypre_F90_Obj *vector,
  nalu_hypre_F90_Int *num_values,
  nalu_hypre_F90_BigIntArray *indices,
  nalu_hypre_F90_ComplexArray *values,
  nalu_hypre_F90_Int *ierr        )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_IJVectorAddToValues(
                nalu_hypre_F90_PassObj (NALU_HYPRE_IJVector, vector),
                nalu_hypre_F90_PassInt (num_values),
                nalu_hypre_F90_PassBigIntArray (indices),
                nalu_hypre_F90_PassComplexArray (values)      ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_IJVectorAssemble
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_ijvectorassemble, NALU_HYPRE_IJVECTORASSEMBLE)
( nalu_hypre_F90_Obj *vector,
  nalu_hypre_F90_Int *ierr    )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_IJVectorAssemble(
                nalu_hypre_F90_PassObj (NALU_HYPRE_IJVector, vector) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_IJVectorGetValues
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_ijvectorgetvalues, NALU_HYPRE_IJVECTORGETVALUES)
( nalu_hypre_F90_Obj *vector,
  nalu_hypre_F90_Int *num_values,
  nalu_hypre_F90_BigIntArray *indices,
  nalu_hypre_F90_ComplexArray *values,
  nalu_hypre_F90_Int *ierr        )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_IJVectorGetValues(
                nalu_hypre_F90_PassObj (NALU_HYPRE_IJVector, vector),
                nalu_hypre_F90_PassInt (num_values),
                nalu_hypre_F90_PassBigIntArray (indices),
                nalu_hypre_F90_PassComplexArray (values)      ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_IJVectorSetMaxOffProcElmts
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_ijvectorsetmaxoffprocelmt, NALU_HYPRE_IJVECTORSETMAXOFFPROCELMT)
( nalu_hypre_F90_Obj *vector,
  nalu_hypre_F90_Int *max_off_proc_elmts,
  nalu_hypre_F90_Int *ierr    )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_IJVectorSetMaxOffProcElmts(
                nalu_hypre_F90_PassObj (NALU_HYPRE_IJVector, vector),
                nalu_hypre_F90_PassInt (max_off_proc_elmts) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_IJVectorSetObjectType
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_ijvectorsetobjecttype, NALU_HYPRE_IJVECTORSETOBJECTTYPE)
( nalu_hypre_F90_Obj *vector,
  nalu_hypre_F90_Int *type,
  nalu_hypre_F90_Int *ierr    )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_IJVectorSetObjectType(
                nalu_hypre_F90_PassObj (NALU_HYPRE_IJVector, vector),
                nalu_hypre_F90_PassInt (type)    ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_IJVectorGetObjectType
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_ijvectorgetobjecttype, NALU_HYPRE_IJVECTORGETOBJECTTYPE)
( nalu_hypre_F90_Obj *vector,
  nalu_hypre_F90_Int *type,
  nalu_hypre_F90_Int *ierr    )
{
   *ierr = (nalu_hypre_F90_Int)
           (NALU_HYPRE_IJVectorGetObjectType(
               nalu_hypre_F90_PassObj (NALU_HYPRE_IJVector, vector),
               nalu_hypre_F90_PassIntRef (type)    ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_IJVectorGetLocalRange
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_ijvectorgetlocalrange, NALU_HYPRE_IJVECTORGETLOCALRANGE)
( nalu_hypre_F90_Obj *vector,
  nalu_hypre_F90_BigInt *jlower,
  nalu_hypre_F90_BigInt *jupper,
  nalu_hypre_F90_Int *ierr    )
{
   *ierr = (nalu_hypre_F90_Int)
           (NALU_HYPRE_IJVectorGetLocalRange(
               nalu_hypre_F90_PassObj (NALU_HYPRE_IJVector, vector),
               nalu_hypre_F90_PassBigIntRef (jlower),
               nalu_hypre_F90_PassBigIntRef (jupper)  ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_IJVectorGetObject
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_ijvectorgetobject, NALU_HYPRE_IJVECTORGETOBJECT)
( nalu_hypre_F90_Obj *vector,
  nalu_hypre_F90_Obj *object,
  nalu_hypre_F90_Int *ierr    )
{
   *ierr = (nalu_hypre_F90_Int)
           (NALU_HYPRE_IJVectorGetObject(
               nalu_hypre_F90_PassObj (NALU_HYPRE_IJVector, vector),
               (void **)         object  ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_IJVectorRead
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_ijvectorread, NALU_HYPRE_IJVECTORREAD)
( char     *filename,
  nalu_hypre_F90_Comm *comm,
  nalu_hypre_F90_Int *object_type,
  nalu_hypre_F90_Obj *vector,
  nalu_hypre_F90_Int *ierr      )
{
   *ierr = (nalu_hypre_F90_Int)
           (NALU_HYPRE_IJVectorRead(
               (char *)            filename,
               nalu_hypre_F90_PassComm (comm),
               nalu_hypre_F90_PassInt (object_type),
               nalu_hypre_F90_PassObjRef (NALU_HYPRE_IJVector, vector)       ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_IJVectorPrint
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_ijvectorprint, NALU_HYPRE_IJVECTORPRINT)
( nalu_hypre_F90_Obj *vector,
  char     *filename,
  nalu_hypre_F90_Int *ierr      )
{
   *ierr = (nalu_hypre_F90_Int)
           (NALU_HYPRE_IJVectorPrint(
               nalu_hypre_F90_PassObj (NALU_HYPRE_IJVector, vector),
               (char *)          filename ) );
}

#ifdef __cplusplus
}
#endif
