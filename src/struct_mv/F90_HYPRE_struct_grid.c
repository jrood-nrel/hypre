/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * NALU_HYPRE_StructGrid interface
 *
 *****************************************************************************/

#include "_nalu_hypre_struct_mv.h"
#include "fortran.h"

#ifdef __cplusplus
extern "C" {
#endif

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructGridCreate
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_structgridcreate, NALU_HYPRE_STRUCTGRIDCREATE)
( nalu_hypre_F90_Comm *comm,
  nalu_hypre_F90_Int *dim,
  nalu_hypre_F90_Obj *grid,
  nalu_hypre_F90_Int *ierr )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_StructGridCreate(
                nalu_hypre_F90_PassComm (comm),
                nalu_hypre_F90_PassInt (dim),
                nalu_hypre_F90_PassObjRef (NALU_HYPRE_StructGrid, grid) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructGridDestroy
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_structgriddestroy, NALU_HYPRE_STRUCTGRIDDESTROY)
( nalu_hypre_F90_Obj *grid,
  nalu_hypre_F90_Int *ierr )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_StructGridDestroy(
                nalu_hypre_F90_PassObj (NALU_HYPRE_StructGrid, grid) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructGridSetExtents
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_structgridsetextents, NALU_HYPRE_STRUCTGRIDSETEXTENTS)
( nalu_hypre_F90_Obj *grid,
  nalu_hypre_F90_IntArray *ilower,
  nalu_hypre_F90_IntArray *iupper,
  nalu_hypre_F90_Int *ierr )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_StructGridSetExtents(
                nalu_hypre_F90_PassObj (NALU_HYPRE_StructGrid, grid),
                nalu_hypre_F90_PassIntArray (ilower),
                nalu_hypre_F90_PassIntArray (iupper) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SetStructGridPeriodicity
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_structgridsetperiodic, NALU_HYPRE_STRUCTGRIDSETPERIODIC)
( nalu_hypre_F90_Obj *grid,
  nalu_hypre_F90_IntArray *periodic,
  nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_StructGridSetPeriodic(
                nalu_hypre_F90_PassObj (NALU_HYPRE_StructGrid, grid),
                nalu_hypre_F90_PassIntArray (periodic)) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructGridAssemble
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_structgridassemble, NALU_HYPRE_STRUCTGRIDASSEMBLE)
( nalu_hypre_F90_Obj *grid,
  nalu_hypre_F90_Int *ierr )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_StructGridAssemble(
                nalu_hypre_F90_PassObj (NALU_HYPRE_StructGrid, grid)) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructGridSetNumGhost
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_structgridsetnumghost, NALU_HYPRE_STRUCTGRIDSETNUMGHOST)
( nalu_hypre_F90_Obj *grid,
  nalu_hypre_F90_IntArray *num_ghost,
  nalu_hypre_F90_Int *ierr )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_StructGridSetNumGhost(
                nalu_hypre_F90_PassObj (NALU_HYPRE_StructGrid, grid),
                nalu_hypre_F90_PassIntArray (num_ghost)) );
}

#ifdef __cplusplus
}
#endif
