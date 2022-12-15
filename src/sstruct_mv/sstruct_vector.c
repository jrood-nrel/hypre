/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Member functions for nalu_hypre_SStructVector class.
 *
 *****************************************************************************/

#include "_nalu_hypre_sstruct_mv.h"
#include "_nalu_hypre_struct_mv.hpp"

/*==========================================================================
 * SStructPVector routines
 *==========================================================================*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SStructPVectorRef( nalu_hypre_SStructPVector  *vector,
                         nalu_hypre_SStructPVector **vector_ref )
{
   nalu_hypre_SStructPVectorRefCount(vector) ++;
   *vector_ref = vector;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SStructPVectorCreate( MPI_Comm               comm,
                            nalu_hypre_SStructPGrid    *pgrid,
                            nalu_hypre_SStructPVector **pvector_ptr)
{
   nalu_hypre_SStructPVector  *pvector;
   NALU_HYPRE_Int              nvars;
   nalu_hypre_StructVector   **svectors;
   nalu_hypre_CommPkg        **comm_pkgs;
   nalu_hypre_StructGrid      *sgrid;
   NALU_HYPRE_Int              var;

   pvector = nalu_hypre_TAlloc(nalu_hypre_SStructPVector,  1, NALU_HYPRE_MEMORY_HOST);

   nalu_hypre_SStructPVectorComm(pvector)  = comm;
   nalu_hypre_SStructPVectorPGrid(pvector) = pgrid;
   nvars = nalu_hypre_SStructPGridNVars(pgrid);
   nalu_hypre_SStructPVectorNVars(pvector) = nvars;
   svectors = nalu_hypre_TAlloc(nalu_hypre_StructVector *,  nvars, NALU_HYPRE_MEMORY_HOST);

   for (var = 0; var < nvars; var++)
   {
      sgrid = nalu_hypre_SStructPGridSGrid(pgrid, var);
      svectors[var] = nalu_hypre_StructVectorCreate(comm, sgrid);
   }
   nalu_hypre_SStructPVectorSVectors(pvector) = svectors;
   comm_pkgs = nalu_hypre_TAlloc(nalu_hypre_CommPkg *,  nvars, NALU_HYPRE_MEMORY_HOST);
   for (var = 0; var < nvars; var++)
   {
      comm_pkgs[var] = NULL;
   }
   nalu_hypre_SStructPVectorCommPkgs(pvector) = comm_pkgs;
   nalu_hypre_SStructPVectorRefCount(pvector) = 1;

   /* GEC inclusion of dataindices   */
   nalu_hypre_SStructPVectorDataIndices(pvector) = NULL ;

   *pvector_ptr = pvector;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SStructPVectorDestroy( nalu_hypre_SStructPVector *pvector )
{
   NALU_HYPRE_Int            nvars;
   nalu_hypre_StructVector **svectors;
   nalu_hypre_CommPkg      **comm_pkgs;
   NALU_HYPRE_Int            var;

   /* GEC destroying dataindices and data in pvector   */

   NALU_HYPRE_Int          *dataindices;

   if (pvector)
   {
      nalu_hypre_SStructPVectorRefCount(pvector) --;
      if (nalu_hypre_SStructPVectorRefCount(pvector) == 0)
      {
         nvars     = nalu_hypre_SStructPVectorNVars(pvector);
         svectors = nalu_hypre_SStructPVectorSVectors(pvector);
         comm_pkgs = nalu_hypre_SStructPVectorCommPkgs(pvector);
         dataindices = nalu_hypre_SStructPVectorDataIndices(pvector);
         for (var = 0; var < nvars; var++)
         {
            nalu_hypre_StructVectorDestroy(svectors[var]);
            nalu_hypre_CommPkgDestroy(comm_pkgs[var]);
         }

         nalu_hypre_TFree(dataindices, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(svectors, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(comm_pkgs, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(pvector, NALU_HYPRE_MEMORY_HOST);
      }
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SStructPVectorInitialize( nalu_hypre_SStructPVector *pvector )
{
   nalu_hypre_SStructPGrid    *pgrid     = nalu_hypre_SStructPVectorPGrid(pvector);
   NALU_HYPRE_Int              nvars     = nalu_hypre_SStructPVectorNVars(pvector);
   NALU_HYPRE_SStructVariable *vartypes  = nalu_hypre_SStructPGridVarTypes(pgrid);
   nalu_hypre_StructVector    *svector;
   NALU_HYPRE_Int              var;

   for (var = 0; var < nvars; var++)
   {
      svector = nalu_hypre_SStructPVectorSVector(pvector, var);
      nalu_hypre_StructVectorInitialize(svector);
      if (vartypes[var] > 0)
      {
         /* needed to get AddTo accumulation correct between processors */
         nalu_hypre_StructVectorClearGhostValues(svector);
      }
   }

   nalu_hypre_SStructPVectorAccumulated(pvector) = 0;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * (action > 0): add-to values
 * (action = 0): set values
 * (action < 0): get values
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SStructPVectorSetValues( nalu_hypre_SStructPVector *pvector,
                               nalu_hypre_Index           index,
                               NALU_HYPRE_Int             var,
                               NALU_HYPRE_Complex        *value,
                               NALU_HYPRE_Int             action )
{
   nalu_hypre_StructVector *svector = nalu_hypre_SStructPVectorSVector(pvector, var);
   NALU_HYPRE_Int           ndim = nalu_hypre_StructVectorNDim(svector);
   nalu_hypre_BoxArray     *grid_boxes;
   nalu_hypre_Box          *box, *grow_box;
   NALU_HYPRE_Int           i;

   /* set values inside the grid */
   nalu_hypre_StructVectorSetValues(svector, index, value, action, -1, 0);

   /* set (AddTo/Get) or clear (Set) values outside the grid in ghost zones */
   if (action != 0)
   {
      /* AddTo/Get */
      nalu_hypre_SStructPGrid *pgrid = nalu_hypre_SStructPVectorPGrid(pvector);
      nalu_hypre_Index         varoffset;
      NALU_HYPRE_Int           done = 0;

      grid_boxes = nalu_hypre_StructGridBoxes(nalu_hypre_StructVectorGrid(svector));

      nalu_hypre_ForBoxI(i, grid_boxes)
      {
         box = nalu_hypre_BoxArrayBox(grid_boxes, i);
         if (nalu_hypre_IndexInBox(index, box))
         {
            done = 1;
            break;
         }
      }

      if (!done)
      {
         grow_box = nalu_hypre_BoxCreate(ndim);
         nalu_hypre_SStructVariableGetOffset(
            nalu_hypre_SStructPGridVarType(pgrid, var), ndim, varoffset);
         nalu_hypre_ForBoxI(i, grid_boxes)
         {
            box = nalu_hypre_BoxArrayBox(grid_boxes, i);
            nalu_hypre_CopyBox(box, grow_box);
            nalu_hypre_BoxGrowByIndex(grow_box, varoffset);
            if (nalu_hypre_IndexInBox(index, grow_box))
            {
               nalu_hypre_StructVectorSetValues(svector, index, value, action, i, 1);
               break;
            }
         }
         nalu_hypre_BoxDestroy(grow_box);
      }
   }
   else
   {
      /* Set */
      grid_boxes = nalu_hypre_StructGridBoxes(nalu_hypre_StructVectorGrid(svector));

      nalu_hypre_ForBoxI(i, grid_boxes)
      {
         box = nalu_hypre_BoxArrayBox(grid_boxes, i);
         if (!nalu_hypre_IndexInBox(index, box))
         {
            nalu_hypre_StructVectorClearValues(svector, index, i, 1);
         }
      }
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * (action > 0): add-to values
 * (action = 0): set values
 * (action < 0): get values
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SStructPVectorSetBoxValues( nalu_hypre_SStructPVector *pvector,
                                  nalu_hypre_Box            *set_box,
                                  NALU_HYPRE_Int             var,
                                  nalu_hypre_Box            *value_box,
                                  NALU_HYPRE_Complex        *values,
                                  NALU_HYPRE_Int             action )
{
   nalu_hypre_StructVector *svector = nalu_hypre_SStructPVectorSVector(pvector, var);
   NALU_HYPRE_Int           ndim = nalu_hypre_StructVectorNDim(svector);
   nalu_hypre_BoxArray     *grid_boxes;
   NALU_HYPRE_Int           i, j;

   /* set values inside the grid */
   nalu_hypre_StructVectorSetBoxValues(svector, set_box, value_box, values, action, -1, 0);

   /* TODO: Why need DeviceSync? */
#if defined(NALU_HYPRE_USING_GPU)
   nalu_hypre_SyncCudaDevice(nalu_hypre_handle());
#endif
   /* set (AddTo/Get) or clear (Set) values outside the grid in ghost zones */
   if (action != 0)
   {
      /* AddTo/Get */
      nalu_hypre_SStructPGrid  *pgrid = nalu_hypre_SStructPVectorPGrid(pvector);
      nalu_hypre_Index          varoffset;
      nalu_hypre_BoxArray      *left_boxes, *done_boxes, *temp_boxes;
      nalu_hypre_Box           *left_box, *done_box, *int_box;

      nalu_hypre_SStructVariableGetOffset(
         nalu_hypre_SStructPGridVarType(pgrid, var), ndim, varoffset);
      grid_boxes = nalu_hypre_StructGridBoxes(nalu_hypre_StructVectorGrid(svector));

      left_boxes = nalu_hypre_BoxArrayCreate(1, ndim);
      done_boxes = nalu_hypre_BoxArrayCreate(2, ndim);
      temp_boxes = nalu_hypre_BoxArrayCreate(0, ndim);

      /* done_box always points to the first box in done_boxes */
      done_box = nalu_hypre_BoxArrayBox(done_boxes, 0);
      /* int_box always points to the second box in done_boxes */
      int_box = nalu_hypre_BoxArrayBox(done_boxes, 1);

      nalu_hypre_CopyBox(set_box, nalu_hypre_BoxArrayBox(left_boxes, 0));
      nalu_hypre_BoxArraySetSize(left_boxes, 1);
      nalu_hypre_SubtractBoxArrays(left_boxes, grid_boxes, temp_boxes);

      nalu_hypre_BoxArraySetSize(done_boxes, 0);
      nalu_hypre_ForBoxI(i, grid_boxes)
      {
         nalu_hypre_SubtractBoxArrays(left_boxes, done_boxes, temp_boxes);
         nalu_hypre_BoxArraySetSize(done_boxes, 1);
         nalu_hypre_CopyBox(nalu_hypre_BoxArrayBox(grid_boxes, i), done_box);
         nalu_hypre_BoxGrowByIndex(done_box, varoffset);
         nalu_hypre_ForBoxI(j, left_boxes)
         {
            left_box = nalu_hypre_BoxArrayBox(left_boxes, j);
            nalu_hypre_IntersectBoxes(left_box, done_box, int_box);
            nalu_hypre_StructVectorSetBoxValues(svector, int_box, value_box,
                                           values, action, i, 1);
         }
      }

      nalu_hypre_BoxArrayDestroy(left_boxes);
      nalu_hypre_BoxArrayDestroy(done_boxes);
      nalu_hypre_BoxArrayDestroy(temp_boxes);
   }
   else
   {
      /* Set */
      nalu_hypre_BoxArray  *diff_boxes;
      nalu_hypre_Box       *grid_box, *diff_box;

      grid_boxes = nalu_hypre_StructGridBoxes(nalu_hypre_StructVectorGrid(svector));
      diff_boxes = nalu_hypre_BoxArrayCreate(0, ndim);

      nalu_hypre_ForBoxI(i, grid_boxes)
      {
         grid_box = nalu_hypre_BoxArrayBox(grid_boxes, i);
         nalu_hypre_BoxArraySetSize(diff_boxes, 0);
         nalu_hypre_SubtractBoxes(set_box, grid_box, diff_boxes);

         nalu_hypre_ForBoxI(j, diff_boxes)
         {
            diff_box = nalu_hypre_BoxArrayBox(diff_boxes, j);
            nalu_hypre_StructVectorClearBoxValues(svector, diff_box, i, 1);
         }
      }
      nalu_hypre_BoxArrayDestroy(diff_boxes);
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SStructPVectorAccumulate( nalu_hypre_SStructPVector *pvector )
{
   nalu_hypre_SStructPGrid    *pgrid     = nalu_hypre_SStructPVectorPGrid(pvector);
   NALU_HYPRE_Int              nvars     = nalu_hypre_SStructPVectorNVars(pvector);
   nalu_hypre_StructVector   **svectors  = nalu_hypre_SStructPVectorSVectors(pvector);
   nalu_hypre_CommPkg        **comm_pkgs = nalu_hypre_SStructPVectorCommPkgs(pvector);

   nalu_hypre_CommInfo        *comm_info;
   nalu_hypre_CommPkg         *comm_pkg;
   nalu_hypre_CommHandle      *comm_handle;

   NALU_HYPRE_Int              ndim      = nalu_hypre_SStructPGridNDim(pgrid);
   NALU_HYPRE_SStructVariable *vartypes  = nalu_hypre_SStructPGridVarTypes(pgrid);

   nalu_hypre_Index            varoffset;
   NALU_HYPRE_Int              num_ghost[2 * NALU_HYPRE_MAXDIM];
   nalu_hypre_StructGrid      *sgrid;
   NALU_HYPRE_Int              var, d;

   /* if values already accumulated, just return */
   if (nalu_hypre_SStructPVectorAccumulated(pvector))
   {
      return nalu_hypre_error_flag;
   }

   for (var = 0; var < nvars; var++)
   {
      if (vartypes[var] > 0)
      {
         sgrid = nalu_hypre_StructVectorGrid(svectors[var]);
         nalu_hypre_SStructVariableGetOffset(vartypes[var], ndim, varoffset);
         for (d = 0; d < ndim; d++)
         {
            num_ghost[2 * d]   = num_ghost[2 * d + 1] = nalu_hypre_IndexD(varoffset, d);
         }

         nalu_hypre_CreateCommInfoFromNumGhost(sgrid, num_ghost, &comm_info);
         nalu_hypre_CommPkgDestroy(comm_pkgs[var]);
         nalu_hypre_CommPkgCreate(comm_info,
                             nalu_hypre_StructVectorDataSpace(svectors[var]),
                             nalu_hypre_StructVectorDataSpace(svectors[var]),
                             1, NULL, 0, nalu_hypre_StructVectorComm(svectors[var]),
                             &comm_pkgs[var]);

         /* accumulate values from AddTo */
         nalu_hypre_CommPkgCreate(comm_info,
                             nalu_hypre_StructVectorDataSpace(svectors[var]),
                             nalu_hypre_StructVectorDataSpace(svectors[var]),
                             1, NULL, 1, nalu_hypre_StructVectorComm(svectors[var]),
                             &comm_pkg);
         nalu_hypre_InitializeCommunication(comm_pkg,
                                       nalu_hypre_StructVectorData(svectors[var]),
                                       nalu_hypre_StructVectorData(svectors[var]), 1, 0,
                                       &comm_handle);
         nalu_hypre_FinalizeCommunication(comm_handle);

         nalu_hypre_CommInfoDestroy(comm_info);
         nalu_hypre_CommPkgDestroy(comm_pkg);
      }
   }

   nalu_hypre_SStructPVectorAccumulated(pvector) = 1;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SStructPVectorAssemble( nalu_hypre_SStructPVector *pvector )
{
   NALU_HYPRE_Int              nvars     = nalu_hypre_SStructPVectorNVars(pvector);
   nalu_hypre_StructVector   **svectors  = nalu_hypre_SStructPVectorSVectors(pvector);
   NALU_HYPRE_Int              var;

   nalu_hypre_SStructPVectorAccumulate(pvector);

   for (var = 0; var < nvars; var++)
   {
      nalu_hypre_StructVectorClearGhostValues(svectors[var]);
      nalu_hypre_StructVectorAssemble(svectors[var]);
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SStructPVectorGather( nalu_hypre_SStructPVector *pvector )
{
   NALU_HYPRE_Int              nvars     = nalu_hypre_SStructPVectorNVars(pvector);
   nalu_hypre_StructVector   **svectors  = nalu_hypre_SStructPVectorSVectors(pvector);
   nalu_hypre_CommPkg        **comm_pkgs = nalu_hypre_SStructPVectorCommPkgs(pvector);
   nalu_hypre_CommHandle      *comm_handle;
   NALU_HYPRE_Int              var;

   for (var = 0; var < nvars; var++)
   {
      if (comm_pkgs[var] != NULL)
      {
         nalu_hypre_InitializeCommunication(comm_pkgs[var],
                                       nalu_hypre_StructVectorData(svectors[var]),
                                       nalu_hypre_StructVectorData(svectors[var]), 0, 0,
                                       &comm_handle);
         nalu_hypre_FinalizeCommunication(comm_handle);
      }
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SStructPVectorGetValues( nalu_hypre_SStructPVector *pvector,
                               nalu_hypre_Index           index,
                               NALU_HYPRE_Int             var,
                               NALU_HYPRE_Complex        *value )
{
   nalu_hypre_SStructPGrid *pgrid     = nalu_hypre_SStructPVectorPGrid(pvector);
   nalu_hypre_StructVector *svector   = nalu_hypre_SStructPVectorSVector(pvector, var);
   nalu_hypre_StructGrid   *sgrid     = nalu_hypre_StructVectorGrid(svector);
   nalu_hypre_BoxArray     *iboxarray = nalu_hypre_SStructPGridIBoxArray(pgrid, var);
   nalu_hypre_BoxArray     *tboxarray;

   /* temporarily swap out sgrid boxes in order to get boundary data */
   tboxarray = nalu_hypre_StructGridBoxes(sgrid);
   nalu_hypre_StructGridBoxes(sgrid) = iboxarray;
   nalu_hypre_StructVectorSetValues(svector, index, value, -1, -1, 0);
   nalu_hypre_StructGridBoxes(sgrid) = tboxarray;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SStructPVectorGetBoxValues( nalu_hypre_SStructPVector *pvector,
                                  nalu_hypre_Box            *set_box,
                                  NALU_HYPRE_Int             var,
                                  nalu_hypre_Box            *value_box,
                                  NALU_HYPRE_Complex        *values )
{
   nalu_hypre_SStructPGrid *pgrid     = nalu_hypre_SStructPVectorPGrid(pvector);
   nalu_hypre_StructVector *svector   = nalu_hypre_SStructPVectorSVector(pvector, var);
   nalu_hypre_StructGrid   *sgrid     = nalu_hypre_StructVectorGrid(svector);
   nalu_hypre_BoxArray     *iboxarray = nalu_hypre_SStructPGridIBoxArray(pgrid, var);
   nalu_hypre_BoxArray     *tboxarray;

   /* temporarily swap out sgrid boxes in order to get boundary data */
   tboxarray = nalu_hypre_StructGridBoxes(sgrid);
   nalu_hypre_StructGridBoxes(sgrid) = iboxarray;
   nalu_hypre_StructVectorSetBoxValues(svector, set_box, value_box, values, -1, -1, 0);
   nalu_hypre_StructGridBoxes(sgrid) = tboxarray;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SStructPVectorSetConstantValues( nalu_hypre_SStructPVector *pvector,
                                       NALU_HYPRE_Complex         value )
{
   NALU_HYPRE_Int           nvars = nalu_hypre_SStructPVectorNVars(pvector);
   nalu_hypre_StructVector *svector;
   NALU_HYPRE_Int           var;

   for (var = 0; var < nvars; var++)
   {
      svector = nalu_hypre_SStructPVectorSVector(pvector, var);
      nalu_hypre_StructVectorSetConstantValues(svector, value);
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * For now, just print multiple files
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SStructPVectorPrint( const char           *filename,
                           nalu_hypre_SStructPVector *pvector,
                           NALU_HYPRE_Int             all )
{
   NALU_HYPRE_Int  nvars = nalu_hypre_SStructPVectorNVars(pvector);
   NALU_HYPRE_Int  var;
   char new_filename[255];

   for (var = 0; var < nvars; var++)
   {
      nalu_hypre_sprintf(new_filename, "%s.%02d", filename, var);
      nalu_hypre_StructVectorPrint(new_filename,
                              nalu_hypre_SStructPVectorSVector(pvector, var),
                              all);
   }

   return nalu_hypre_error_flag;
}

/*==========================================================================
 * SStructVector routines
 *==========================================================================*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SStructVectorRef( nalu_hypre_SStructVector  *vector,
                        nalu_hypre_SStructVector **vector_ref )
{
   nalu_hypre_SStructVectorRefCount(vector) ++;
   *vector_ref = vector;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SStructVectorSetConstantValues( nalu_hypre_SStructVector *vector,
                                      NALU_HYPRE_Complex        value )
{
   NALU_HYPRE_Int             nparts = nalu_hypre_SStructVectorNParts(vector);
   nalu_hypre_SStructPVector *pvector;
   NALU_HYPRE_Int             part;

   for (part = 0; part < nparts; part++)
   {
      pvector = nalu_hypre_SStructVectorPVector(vector, part);
      nalu_hypre_SStructPVectorSetConstantValues(pvector, value);
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Here the address of the parvector inside the semistructured vector
 * is provided to the "outside". It assumes that the vector type
 * is NALU_HYPRE_SSTRUCT
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SStructVectorConvert( nalu_hypre_SStructVector  *vector,
                            nalu_hypre_ParVector     **parvector_ptr )
{
   *parvector_ptr = nalu_hypre_SStructVectorParVector(vector);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Copy values from vector to parvector and provide the address
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SStructVectorParConvert( nalu_hypre_SStructVector  *vector,
                               nalu_hypre_ParVector     **parvector_ptr )
{
   nalu_hypre_ParVector      *parvector;
   NALU_HYPRE_Complex        *pardata;
   NALU_HYPRE_Int             pari;

   nalu_hypre_SStructPVector *pvector;
   nalu_hypre_StructVector   *y;
   nalu_hypre_Box            *y_data_box;
   NALU_HYPRE_Complex        *yp;
   nalu_hypre_BoxArray       *boxes;
   nalu_hypre_Box            *box;
   nalu_hypre_Index           loop_size;
   nalu_hypre_IndexRef        start;
   nalu_hypre_Index           stride;

   NALU_HYPRE_Int             nparts, nvars;
   NALU_HYPRE_Int             part, var, i;

   nalu_hypre_SetIndex(stride, 1);

   parvector = nalu_hypre_SStructVectorParVector(vector);
   pardata = nalu_hypre_VectorData(nalu_hypre_ParVectorLocalVector(parvector));
   pari = 0;
   nparts = nalu_hypre_SStructVectorNParts(vector);
   for (part = 0; part < nparts; part++)
   {
      pvector = nalu_hypre_SStructVectorPVector(vector, part);
      nvars = nalu_hypre_SStructPVectorNVars(pvector);
      for (var = 0; var < nvars; var++)
      {
         y = nalu_hypre_SStructPVectorSVector(pvector, var);

         boxes = nalu_hypre_StructGridBoxes(nalu_hypre_StructVectorGrid(y));
         nalu_hypre_ForBoxI(i, boxes)
         {
            box   = nalu_hypre_BoxArrayBox(boxes, i);
            start = nalu_hypre_BoxIMin(box);

            y_data_box =
               nalu_hypre_BoxArrayBox(nalu_hypre_StructVectorDataSpace(y), i);
            yp = nalu_hypre_StructVectorBoxData(y, i);

            nalu_hypre_BoxGetSize(box, loop_size);

#undef DEVICE_VAR
#define DEVICE_VAR is_device_ptr(pardata,yp)
            nalu_hypre_BoxLoop2Begin(nalu_hypre_SStructVectorNDim(vector), loop_size,
                                y_data_box, start, stride, yi,
                                box,        start, stride, bi);
            {
               pardata[pari + bi] = yp[yi];
            }
            nalu_hypre_BoxLoop2End(yi, bi);
#undef DEVICE_VAR
#define DEVICE_VAR

            pari += nalu_hypre_BoxVolume(box);
         }
      }
   }

   *parvector_ptr = nalu_hypre_SStructVectorParVector(vector);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Used for NALU_HYPRE_SSTRUCT type semi structured vectors.
 * A dummy function to indicate that the struct vector part will be used.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SStructVectorRestore( nalu_hypre_SStructVector *vector,
                            nalu_hypre_ParVector     *parvector )
{
   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Copy values from parvector to vector
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SStructVectorParRestore( nalu_hypre_SStructVector *vector,
                               nalu_hypre_ParVector     *parvector )
{
   NALU_HYPRE_Complex        *pardata;
   NALU_HYPRE_Int             pari;

   nalu_hypre_SStructPVector *pvector;
   nalu_hypre_StructVector   *y;
   nalu_hypre_Box            *y_data_box;
   NALU_HYPRE_Complex        *yp;
   nalu_hypre_BoxArray       *boxes;
   nalu_hypre_Box            *box;
   nalu_hypre_Index           loop_size;
   nalu_hypre_IndexRef        start;
   nalu_hypre_Index           stride;

   NALU_HYPRE_Int             nparts, nvars;
   NALU_HYPRE_Int             part, var, i;

   if (parvector != NULL)
   {
      nalu_hypre_SetIndex(stride, 1);

      parvector = nalu_hypre_SStructVectorParVector(vector);
      pardata = nalu_hypre_VectorData(nalu_hypre_ParVectorLocalVector(parvector));
      pari = 0;
      nparts = nalu_hypre_SStructVectorNParts(vector);
      for (part = 0; part < nparts; part++)
      {
         pvector = nalu_hypre_SStructVectorPVector(vector, part);
         nvars = nalu_hypre_SStructPVectorNVars(pvector);
         for (var = 0; var < nvars; var++)
         {
            y = nalu_hypre_SStructPVectorSVector(pvector, var);

            boxes = nalu_hypre_StructGridBoxes(nalu_hypre_StructVectorGrid(y));
            nalu_hypre_ForBoxI(i, boxes)
            {
               box   = nalu_hypre_BoxArrayBox(boxes, i);
               start = nalu_hypre_BoxIMin(box);

               y_data_box =
                  nalu_hypre_BoxArrayBox(nalu_hypre_StructVectorDataSpace(y), i);
               yp = nalu_hypre_StructVectorBoxData(y, i);

               nalu_hypre_BoxGetSize(box, loop_size);

#undef DEVICE_VAR
#define DEVICE_VAR is_device_ptr(yp,pardata)
               nalu_hypre_BoxLoop2Begin(nalu_hypre_SStructVectorNDim(vector), loop_size,
                                   y_data_box, start, stride, yi,
                                   box,        start, stride, bi);
               {
                  yp[yi] = pardata[pari + bi];
               }
               nalu_hypre_BoxLoop2End(yi, bi);
#undef DEVICE_VAR
#define DEVICE_VAR

               pari += nalu_hypre_BoxVolume(box);
            }
         }
      }
   }

   return nalu_hypre_error_flag;
}
/*------------------------------------------------------------------
 *  GEC1002 shell initialization of a pvector
 *   if the pvector exists. This function will set the dataindices
 *  and datasize of the pvector. Datasize is the sum of the sizes
 *  of each svector and dataindices is defined as
 *  dataindices[var]= aggregated initial size of the pvector[var]
 *  When ucvars are present we need to modify adding nucvars.
 *----------------------------------------------------------------*/
NALU_HYPRE_Int
nalu_hypre_SStructPVectorInitializeShell( nalu_hypre_SStructPVector *pvector)
{
   NALU_HYPRE_Int            nvars = nalu_hypre_SStructPVectorNVars(pvector);
   NALU_HYPRE_Int            var;
   NALU_HYPRE_Int            pdatasize;
   NALU_HYPRE_Int            svectdatasize;
   NALU_HYPRE_Int           *pdataindices;
   NALU_HYPRE_Int            nucvars = 0;
   nalu_hypre_StructVector  *svector;

   pdatasize = 0;
   pdataindices = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  nvars, NALU_HYPRE_MEMORY_HOST);

   for (var = 0; var < nvars; var++)
   {
      svector = nalu_hypre_SStructPVectorSVector(pvector, var);
      nalu_hypre_StructVectorInitializeShell(svector);
      pdataindices[var] = pdatasize ;
      svectdatasize = nalu_hypre_StructVectorDataSize(svector);
      pdatasize += svectdatasize;
   }

   /* GEC1002 assuming that the ucvars are located at the end, after the
    * the size of the vars has been included we add the number of uvar
    * for this part                                                  */

   nalu_hypre_SStructPVectorDataIndices(pvector) = pdataindices;
   nalu_hypre_SStructPVectorDataSize(pvector) = pdatasize + nucvars ;

   nalu_hypre_SStructPVectorAccumulated(pvector) = 0;

   return nalu_hypre_error_flag;
}

/*------------------------------------------------------------------
 *  GEC1002 shell initialization of a sstructvector
 *  if the vector exists. This function will set the
 *  dataindices and datasize of the vector. When ucvars
 *  are present at the end of all the parts we need to modify adding pieces
 *  for ucvars.
 *----------------------------------------------------------------*/
NALU_HYPRE_Int
nalu_hypre_SStructVectorInitializeShell( nalu_hypre_SStructVector *vector)
{
   NALU_HYPRE_Int                part  ;
   NALU_HYPRE_Int                datasize;
   NALU_HYPRE_Int                pdatasize;
   NALU_HYPRE_Int                nparts = nalu_hypre_SStructVectorNParts(vector);
   nalu_hypre_SStructPVector    *pvector;
   NALU_HYPRE_Int               *dataindices;

   datasize = 0;
   dataindices = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  nparts, NALU_HYPRE_MEMORY_HOST);
   for (part = 0; part < nparts; part++)
   {
      pvector = nalu_hypre_SStructVectorPVector(vector, part) ;
      nalu_hypre_SStructPVectorInitializeShell(pvector);
      pdatasize = nalu_hypre_SStructPVectorDataSize(pvector);
      dataindices[part] = datasize ;
      datasize        += pdatasize ;
   }
   nalu_hypre_SStructVectorDataIndices(vector) = dataindices;
   nalu_hypre_SStructVectorDataSize(vector) = datasize ;

   return nalu_hypre_error_flag;
}


NALU_HYPRE_Int
nalu_hypre_SStructVectorClearGhostValues(nalu_hypre_SStructVector *vector)
{
   NALU_HYPRE_Int              nparts = nalu_hypre_SStructVectorNParts(vector);
   nalu_hypre_SStructPVector  *pvector;
   nalu_hypre_StructVector    *svector;

   NALU_HYPRE_Int    part;
   NALU_HYPRE_Int    nvars, var;

   for (part = 0; part < nparts; part++)
   {
      pvector = nalu_hypre_SStructVectorPVector(vector, part);
      nvars  = nalu_hypre_SStructPVectorNVars(pvector);

      for (var = 0; var < nvars; var++)
      {
         svector = nalu_hypre_SStructPVectorSVector(pvector, var);
         nalu_hypre_StructVectorClearGhostValues(svector);
      }
   }

   return nalu_hypre_error_flag;
}

NALU_HYPRE_MemoryLocation
nalu_hypre_SStructVectorMemoryLocation(nalu_hypre_SStructVector *vector)
{
   NALU_HYPRE_Int type = nalu_hypre_SStructVectorObjectType(vector);

   if (type == NALU_HYPRE_SSTRUCT)
   {
      nalu_hypre_ParVector *parvector;
      nalu_hypre_SStructVectorConvert(vector, &parvector);
      return nalu_hypre_ParVectorMemoryLocation(parvector);
   }

   void *object;
   NALU_HYPRE_SStructVectorGetObject(vector, &object);

   if (type == NALU_HYPRE_PARCSR)
   {
      return nalu_hypre_ParVectorMemoryLocation((nalu_hypre_ParVector *) object);
   }

   if (type == NALU_HYPRE_STRUCT)
   {
      return nalu_hypre_StructVectorMemoryLocation((nalu_hypre_StructVector *) object);
   }

   return NALU_HYPRE_MEMORY_UNDEFINED;
}

