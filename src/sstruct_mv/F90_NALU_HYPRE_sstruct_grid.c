/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * NALU_HYPRE_SStructGrid interface
 *
 *****************************************************************************/

#include "_nalu_hypre_sstruct_mv.h"
#include "fortran.h"

#ifdef __cplusplus
extern "C" {
#endif

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructGridCreate
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_sstructgridcreate, NALU_HYPRE_SSTRUCTGRIDCREATE)
(nalu_hypre_F90_Comm   *comm,
 nalu_hypre_F90_Int    *ndim,
 nalu_hypre_F90_Int    *nparts,
 nalu_hypre_F90_ObjRef *grid_ptr,
 nalu_hypre_F90_Int    *ierr)
{
   *ierr = (nalu_hypre_F90_Int) NALU_HYPRE_SStructGridCreate(
              nalu_hypre_F90_PassComm   (comm),
              nalu_hypre_F90_PassInt    (ndim),
              nalu_hypre_F90_PassInt    (nparts),
              nalu_hypre_F90_PassObjRef (NALU_HYPRE_SStructGrid, grid_ptr) );
}

/*--------------------------------------------------------------------------
 *  NALU_HYPRE_SStructGridDestroy
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_sstructgriddestroy, NALU_HYPRE_SSTRUCTGRIDDESTROY)
(nalu_hypre_F90_Obj *grid,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int) NALU_HYPRE_SStructGridDestroy(
              nalu_hypre_F90_PassObj (NALU_HYPRE_SStructGrid, grid) );
}

/*--------------------------------------------------------------------------
 *  NALU_HYPRE_SStructGridSetExtents
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_sstructgridsetextents, NALU_HYPRE_SSTRUCTGRIDSETEXTENTS)
(nalu_hypre_F90_Obj      *grid,
 nalu_hypre_F90_Int      *part,
 nalu_hypre_F90_IntArray *ilower,
 nalu_hypre_F90_IntArray *iupper,
 nalu_hypre_F90_Int      *ierr)
{
   *ierr = (nalu_hypre_F90_Int) NALU_HYPRE_SStructGridSetExtents(
              nalu_hypre_F90_PassObj      (NALU_HYPRE_SStructGrid, grid),
              nalu_hypre_F90_PassInt      (part),
              nalu_hypre_F90_PassIntArray (ilower),
              nalu_hypre_F90_PassIntArray (iupper) );
}

/*--------------------------------------------------------------------------
 *  NALU_HYPRE_SStructGridSetVariables
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_sstructgridsetvariables, NALU_HYPRE_SSTRUCTGRIDSETVARIABLES)
(nalu_hypre_F90_Obj      *grid,
 nalu_hypre_F90_Int      *part,
 nalu_hypre_F90_Int      *nvars,
 nalu_hypre_F90_IntArray *vartypes,
 nalu_hypre_F90_Int      *ierr)
{
   *ierr = (nalu_hypre_F90_Int) NALU_HYPRE_SStructGridSetVariables(
              nalu_hypre_F90_PassObj      (NALU_HYPRE_SStructGrid, grid),
              nalu_hypre_F90_PassInt      (part),
              nalu_hypre_F90_PassInt      (nvars),
              nalu_hypre_F90_PassIntArray (vartypes) );
}

/*--------------------------------------------------------------------------
 *  NALU_HYPRE_SStructGridAddVariables
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_sstructgridaddvariables, NALU_HYPRE_SSTRUCTGRIDADDVARIABLES)
(nalu_hypre_F90_Obj      *grid,
 nalu_hypre_F90_Int      *part,
 nalu_hypre_F90_IntArray *index,
 nalu_hypre_F90_Int      *nvars,
 nalu_hypre_F90_IntArray *vartypes,
 nalu_hypre_F90_Int      *ierr)
{
   *ierr = (nalu_hypre_F90_Int) NALU_HYPRE_SStructGridAddVariables(
              nalu_hypre_F90_PassObj(NALU_HYPRE_SStructGrid, grid),
              nalu_hypre_F90_PassInt(part),
              nalu_hypre_F90_PassIntArray(index),
              nalu_hypre_F90_PassInt(nvars),
              nalu_hypre_F90_PassIntArray(vartypes));
}

/*--------------------------------------------------------------------------
 *  NALU_HYPRE_SStructGridSetFEMOrdering
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_sstructgridsetfemordering, NALU_HYPRE_SSTRUCTGRIDSETFEMORDERING)
(nalu_hypre_F90_Obj      *grid,
 nalu_hypre_F90_Int      *part,
 nalu_hypre_F90_IntArray *ordering,
 nalu_hypre_F90_Int      *ierr)
{
   *ierr = (nalu_hypre_F90_Int) NALU_HYPRE_SStructGridSetFEMOrdering(
              nalu_hypre_F90_PassObj      (NALU_HYPRE_SStructGrid, grid),
              nalu_hypre_F90_PassInt      (part),
              nalu_hypre_F90_PassIntArray (ordering) );
}

/*--------------------------------------------------------------------------
 *  NALU_HYPRE_SStructGridSetNeighborPart
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_sstructgridsetneighborpart, NALU_HYPRE_SSTRUCTGRIDSETNEIGHBORPART)
(nalu_hypre_F90_Obj      *grid,
 nalu_hypre_F90_Int      *part,
 nalu_hypre_F90_IntArray *ilower,
 nalu_hypre_F90_IntArray *iupper,
 nalu_hypre_F90_Int      *nbor_part,
 nalu_hypre_F90_IntArray *nbor_ilower,
 nalu_hypre_F90_IntArray *nbor_iupper,
 nalu_hypre_F90_IntArray *index_map,
 nalu_hypre_F90_IntArray *index_dir,
 nalu_hypre_F90_Int      *ierr)
{
   *ierr = (nalu_hypre_F90_Int) NALU_HYPRE_SStructGridSetNeighborPart(
              nalu_hypre_F90_PassObj      (NALU_HYPRE_SStructGrid, grid),
              nalu_hypre_F90_PassInt      (part),
              nalu_hypre_F90_PassIntArray (ilower),
              nalu_hypre_F90_PassIntArray (iupper),
              nalu_hypre_F90_PassInt      (nbor_part),
              nalu_hypre_F90_PassIntArray (nbor_ilower),
              nalu_hypre_F90_PassIntArray (nbor_iupper),
              nalu_hypre_F90_PassIntArray (index_map),
              nalu_hypre_F90_PassIntArray (index_dir) );
}

/*--------------------------------------------------------------------------
 *  NALU_HYPRE_SStructGridSetSharedPart
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_sstructgridsetsharedpart, NALU_HYPRE_SSTRUCTGRIDSETSHAREDPART)
(nalu_hypre_F90_Obj      *grid,
 nalu_hypre_F90_Int      *part,
 nalu_hypre_F90_IntArray *ilower,
 nalu_hypre_F90_IntArray *iupper,
 nalu_hypre_F90_IntArray *offset,
 nalu_hypre_F90_Int      *shared_part,
 nalu_hypre_F90_IntArray *shared_ilower,
 nalu_hypre_F90_IntArray *shared_iupper,
 nalu_hypre_F90_IntArray *shared_offset,
 nalu_hypre_F90_IntArray *index_map,
 nalu_hypre_F90_IntArray *index_dir,
 nalu_hypre_F90_Int      *ierr)
{
   *ierr = (nalu_hypre_F90_Int) NALU_HYPRE_SStructGridSetSharedPart(
              nalu_hypre_F90_PassObj      (NALU_HYPRE_SStructGrid, grid),
              nalu_hypre_F90_PassInt      (part),
              nalu_hypre_F90_PassIntArray (ilower),
              nalu_hypre_F90_PassIntArray (iupper),
              nalu_hypre_F90_PassIntArray (offset),
              nalu_hypre_F90_PassInt      (shared_part),
              nalu_hypre_F90_PassIntArray (shared_ilower),
              nalu_hypre_F90_PassIntArray (shared_iupper),
              nalu_hypre_F90_PassIntArray (shared_offset),
              nalu_hypre_F90_PassIntArray (index_map),
              nalu_hypre_F90_PassIntArray (index_dir) );
}

/*--------------------------------------------------------------------------
 * *** placeholder ***
 *  NALU_HYPRE_SStructGridAddUnstructuredPart
 *--------------------------------------------------------------------------*/

#if 0

void
nalu_hypre_F90_IFACE(nalu_hypre_sstructgridaddunstructure, NALU_HYPRE_SSTRUCTGRIDADDUNSTRUCTURE)
(nalu_hypre_F90_Obj *grid,
 nalu_hypre_F90_Int *ilower,
 nalu_hypre_F90_Int *iupper,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int) NALU_HYPRE_SStructGridAddUnstructuredPart(
              nalu_hypre_F90_PassObj (NALU_HYPRE_SStructGrid, grid),
              nalu_hypre_F90_PassInt (ilower),
              nalu_hypre_F90_PassInt (iupper) );
}
#endif

/*--------------------------------------------------------------------------
 *  NALU_HYPRE_SStructGridAssemble
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_sstructgridassemble, NALU_HYPRE_SSTRUCTGRIDASSEMBLE)
(nalu_hypre_F90_Obj *grid,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int) NALU_HYPRE_SStructGridAssemble(
              nalu_hypre_F90_PassObj (NALU_HYPRE_SStructGrid, grid) );
}

/*--------------------------------------------------------------------------
 *  NALU_HYPRE_SStructGridSetPeriodic
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_sstructgridsetperiodic, NALU_HYPRE_SSTRUCTGRIDSETPERIODIC)
(nalu_hypre_F90_Obj      *grid,
 nalu_hypre_F90_Int      *part,
 nalu_hypre_F90_IntArray *periodic,
 nalu_hypre_F90_Int      *ierr)
{
   *ierr = (nalu_hypre_F90_Int) NALU_HYPRE_SStructGridSetPeriodic(
              nalu_hypre_F90_PassObj      (NALU_HYPRE_SStructGrid, grid),
              nalu_hypre_F90_PassInt      (part),
              nalu_hypre_F90_PassIntArray (periodic) );
}

/*--------------------------------------------------------------------------
 *  NALU_HYPRE_SStructGridSetNumGhost
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_sstructgridsetnumghost, NALU_HYPRE_SSTRUCTGRIDSETNUMGHOST)
(nalu_hypre_F90_Obj      *grid,
 nalu_hypre_F90_IntArray *num_ghost,
 nalu_hypre_F90_Int      *ierr)
{
   *ierr = (nalu_hypre_F90_Int) NALU_HYPRE_SStructGridSetNumGhost(
              nalu_hypre_F90_PassObj      (NALU_HYPRE_SStructGrid, grid),
              nalu_hypre_F90_PassIntArray (num_ghost) );
}

#ifdef __cplusplus
}
#endif
