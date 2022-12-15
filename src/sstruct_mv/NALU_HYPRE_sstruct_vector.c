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

#include "_nalu_hypre_sstruct_mv.h"

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructVectorCreate( MPI_Comm              comm,
                           NALU_HYPRE_SStructGrid     grid,
                           NALU_HYPRE_SStructVector  *vector_ptr )
{
   nalu_hypre_SStructVector   *vector;
   NALU_HYPRE_Int              nparts;
   nalu_hypre_SStructPVector **pvectors;
   MPI_Comm               pcomm;
   nalu_hypre_SStructPGrid    *pgrid;
   NALU_HYPRE_Int              part;

   vector = nalu_hypre_TAlloc(nalu_hypre_SStructVector, 1, NALU_HYPRE_MEMORY_HOST);

   nalu_hypre_SStructVectorComm(vector) = comm;
   nalu_hypre_SStructVectorNDim(vector) = nalu_hypre_SStructGridNDim(grid);
   nalu_hypre_SStructGridRef(grid, &nalu_hypre_SStructVectorGrid(vector));
   nalu_hypre_SStructVectorObjectType(vector) = NALU_HYPRE_SSTRUCT;
   nparts = nalu_hypre_SStructGridNParts(grid);
   nalu_hypre_SStructVectorNParts(vector) = nparts;
   pvectors = nalu_hypre_TAlloc(nalu_hypre_SStructPVector *, nparts, NALU_HYPRE_MEMORY_HOST);
   for (part = 0; part < nparts; part++)
   {
      pgrid = nalu_hypre_SStructGridPGrid(grid, part);
      pcomm = nalu_hypre_SStructPGridComm(pgrid);
      nalu_hypre_SStructPVectorCreate(pcomm, pgrid, &pvectors[part]);
   }
   nalu_hypre_SStructVectorPVectors(vector)   = pvectors;
   nalu_hypre_SStructVectorIJVector(vector)   = NULL;

   /* GEC1002 initializing to NULL */

   nalu_hypre_SStructVectorDataIndices(vector) = NULL;
   nalu_hypre_SStructVectorData(vector)        = NULL;

   /* GEC1002 moving the creation of the ijvector the the initialize part
    *   ilower = nalu_hypre_SStructGridStartRank(grid);
    *   iupper = ilower + nalu_hypre_SStructGridLocalSize(grid) - 1;
    *  NALU_HYPRE_IJVectorCreate(comm, ilowergh, iuppergh,
    *                  &nalu_hypre_SStructVectorIJVector(vector)); */

   nalu_hypre_SStructVectorIJVector(vector)   = NULL;
   nalu_hypre_SStructVectorParVector(vector)  = NULL;
   nalu_hypre_SStructVectorGlobalSize(vector) = 0;
   nalu_hypre_SStructVectorRefCount(vector)   = 1;
   nalu_hypre_SStructVectorDataSize(vector)   = 0;
   nalu_hypre_SStructVectorObjectType(vector) = NALU_HYPRE_SSTRUCT;

   *vector_ptr = vector;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructVectorDestroy( NALU_HYPRE_SStructVector vector )
{
   NALU_HYPRE_Int              nparts;
   nalu_hypre_SStructPVector **pvectors;
   NALU_HYPRE_Int              part;
   NALU_HYPRE_Int              vector_type;
   NALU_HYPRE_MemoryLocation   memory_location = nalu_hypre_SStructVectorMemoryLocation(vector);

   /* GEC1002 destroying data indices and data in vector  */

   if (vector)
   {
      vector_type = nalu_hypre_SStructVectorObjectType(vector);
      nalu_hypre_SStructVectorRefCount(vector) --;
      if (nalu_hypre_SStructVectorRefCount(vector) == 0)
      {
         NALU_HYPRE_SStructGridDestroy(nalu_hypre_SStructVectorGrid(vector));
         nparts   = nalu_hypre_SStructVectorNParts(vector);
         pvectors = nalu_hypre_SStructVectorPVectors(vector);
         for (part = 0; part < nparts; part++)
         {
            nalu_hypre_SStructPVectorDestroy(pvectors[part]);
         }
         nalu_hypre_TFree(pvectors, NALU_HYPRE_MEMORY_HOST);
         NALU_HYPRE_IJVectorDestroy(nalu_hypre_SStructVectorIJVector(vector));

         /* GEC1002 the ijdestroy takes care of the data when the
          * vector is type NALU_HYPRE_SSTRUCT. This is a result that the
          * ijvector does not use the owndata flag in the data structure
          * unlike the struct vector                               */

         /* GEC if data has been allocated then free the pointer */
         nalu_hypre_TFree(nalu_hypre_SStructVectorDataIndices(vector), NALU_HYPRE_MEMORY_HOST);

         if (nalu_hypre_SStructVectorData(vector) && (vector_type == NALU_HYPRE_PARCSR))
         {
            nalu_hypre_TFree(nalu_hypre_SStructVectorData(vector), memory_location);
         }

         nalu_hypre_TFree(vector, NALU_HYPRE_MEMORY_HOST);
      }
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * GEC1002 changes to initialize the vector with a data chunk
 * that includes all the part,var pieces instead of just svector-var
 * pieces. In case of pure unstruct-variables (ucvar), which are at the
 * end of each part, we might need to modify initialize shell vector
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructVectorInitialize( NALU_HYPRE_SStructVector vector )
{
   NALU_HYPRE_Int               datasize;
   NALU_HYPRE_Int               nvars ;
   NALU_HYPRE_Int               nparts = nalu_hypre_SStructVectorNParts(vector) ;
   NALU_HYPRE_Int               var, part  ;
   NALU_HYPRE_Complex          *data ;
   NALU_HYPRE_Complex          *pdata ;
   NALU_HYPRE_Complex          *sdata  ;
   nalu_hypre_SStructPVector   *pvector;
   nalu_hypre_StructVector     *svector;
   NALU_HYPRE_Int              *dataindices;
   NALU_HYPRE_Int              *pdataindices;
   NALU_HYPRE_Int               vector_type = nalu_hypre_SStructVectorObjectType(vector);
   nalu_hypre_SStructGrid      *grid =  nalu_hypre_SStructVectorGrid(vector);
   MPI_Comm                comm = nalu_hypre_SStructVectorComm(vector);
   NALU_HYPRE_IJVector          ijvector;
   nalu_hypre_SStructPGrid     *pgrid;
   NALU_HYPRE_SStructVariable  *vartypes;
   NALU_HYPRE_MemoryLocation    memory_location = nalu_hypre_HandleMemoryLocation(nalu_hypre_handle());

   /* GEC0902 addition of variables for ilower and iupper   */
   NALU_HYPRE_Int               ilower, iupper;
   nalu_hypre_ParVector        *par_vector;
   nalu_hypre_Vector           *parlocal_vector;


   /* GEC0902 getting the datasizes and indices we need  */

   nalu_hypre_SStructVectorInitializeShell(vector);

   datasize = nalu_hypre_SStructVectorDataSize(vector);

   data = nalu_hypre_CTAlloc(NALU_HYPRE_Complex, datasize, memory_location);

   dataindices = nalu_hypre_SStructVectorDataIndices(vector);

   nalu_hypre_SStructVectorData(vector) = data;

   for (part = 0; part < nparts; part++)
   {
      pvector = nalu_hypre_SStructVectorPVector(vector, part);
      pdataindices = nalu_hypre_SStructPVectorDataIndices(pvector);
      /* shift-num   = dataindices[part]; */
      pdata = data + dataindices[part];
      nvars = nalu_hypre_SStructPVectorNVars(pvector);

      pgrid    = nalu_hypre_SStructPVectorPGrid(pvector);
      vartypes = nalu_hypre_SStructPGridVarTypes(pgrid);
      for (var = 0; var < nvars; var++)
      {
         svector = nalu_hypre_SStructPVectorSVector(pvector, var);
         /*  shift-pnum    = pdataindices[var]; */
         sdata   = pdata + pdataindices[var];

         /* GEC1002 initialization of inside data pointer of a svector
          * because no data is alloced, we make sure the flag is zero. This
          * affects the destroy */
         nalu_hypre_StructVectorInitializeData(svector, sdata);
         nalu_hypre_StructVectorDataAlloced(svector) = 0;
         if (vartypes[var] > 0)
         {
            /* needed to get AddTo accumulation correct between processors */
            nalu_hypre_StructVectorClearGhostValues(svector);
         }
      }
   }

   /* GEC1002 this is now the creation of the ijmatrix and the initialization
    * by checking the type of the vector */

   if (vector_type == NALU_HYPRE_PARCSR )
   {
      ilower = nalu_hypre_SStructGridStartRank(grid);
      iupper = ilower + nalu_hypre_SStructGridLocalSize(grid) - 1;
   }

   if (vector_type == NALU_HYPRE_SSTRUCT || vector_type == NALU_HYPRE_STRUCT)
   {
      ilower = nalu_hypre_SStructGridGhstartRank(grid);
      iupper = ilower + nalu_hypre_SStructGridGhlocalSize(grid) - 1;
   }

   NALU_HYPRE_IJVectorCreate(comm, ilower, iupper,
                        &nalu_hypre_SStructVectorIJVector(vector));

   /* GEC1002, once the partitioning is done, it is time for the actual
    * initialization                                                 */


   /* u-vector: the type is for the parvector inside the ijvector */

   ijvector = nalu_hypre_SStructVectorIJVector(vector);

   NALU_HYPRE_IJVectorSetObjectType(ijvector, NALU_HYPRE_PARCSR);

   NALU_HYPRE_IJVectorInitialize(ijvector);


   /* GEC1002 for NALU_HYPRE_SSTRUCT type of vector, we do not need data allocated
    * inside the parvector piece of the structure. We make that pointer within
    * the localvector to point to the outside "data". Before redirecting the
    * local pointer to point to the true data chunk for NALU_HYPRE_SSTRUCT: we
    * destroy and assign.  We now have two entries of the data structure
    * pointing to the same chunk if we have a NALU_HYPRE_SSTRUCT vector We do not
    * need the IJVectorInitializePar, we have to undoit for the SStruct case in
    * a sense it is a desinitializepar */

   if (vector_type == NALU_HYPRE_SSTRUCT || vector_type == NALU_HYPRE_STRUCT)
   {
      par_vector = (nalu_hypre_ParVector *) nalu_hypre_IJVectorObject(ijvector);
      parlocal_vector = nalu_hypre_ParVectorLocalVector(par_vector);
      nalu_hypre_TFree(nalu_hypre_VectorData(parlocal_vector), nalu_hypre_VectorMemoryLocation(parlocal_vector));
      nalu_hypre_VectorData(parlocal_vector) = data ;
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructVectorSetValues( NALU_HYPRE_SStructVector  vector,
                              NALU_HYPRE_Int            part,
                              NALU_HYPRE_Int           *index,
                              NALU_HYPRE_Int            var,
                              NALU_HYPRE_Complex       *value )
{
   NALU_HYPRE_Int             ndim    = nalu_hypre_SStructVectorNDim(vector);
   nalu_hypre_SStructPVector *pvector = nalu_hypre_SStructVectorPVector(vector, part);
   nalu_hypre_Index           cindex;

   nalu_hypre_CopyToCleanIndex(index, ndim, cindex);

   if (var < nalu_hypre_SStructPVectorNVars(pvector))
   {
      nalu_hypre_SStructPVectorSetValues(pvector, cindex, var, value, 0);
   }
   else
   {
      /* TODO */
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructVectorAddToValues( NALU_HYPRE_SStructVector  vector,
                                NALU_HYPRE_Int            part,
                                NALU_HYPRE_Int           *index,
                                NALU_HYPRE_Int            var,
                                NALU_HYPRE_Complex       *value )
{
   NALU_HYPRE_Int             ndim    = nalu_hypre_SStructVectorNDim(vector);
   nalu_hypre_SStructPVector *pvector = nalu_hypre_SStructVectorPVector(vector, part);
   nalu_hypre_Index           cindex;

   nalu_hypre_CopyToCleanIndex(index, ndim, cindex);

   if (var < nalu_hypre_SStructPVectorNVars(pvector))
   {
      nalu_hypre_SStructPVectorSetValues(pvector, cindex, var, value, 1);
   }
   else
   {
      /* TODO */
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/* ONLY3D */

NALU_HYPRE_Int
NALU_HYPRE_SStructVectorAddFEMValues( NALU_HYPRE_SStructVector  vector,
                                 NALU_HYPRE_Int            part,
                                 NALU_HYPRE_Int           *index,
                                 NALU_HYPRE_Complex       *values )
{
   NALU_HYPRE_Int           ndim         = nalu_hypre_SStructVectorNDim(vector);
   nalu_hypre_SStructGrid  *grid         = nalu_hypre_SStructVectorGrid(vector);
   NALU_HYPRE_Int           fem_nvars    = nalu_hypre_SStructGridFEMPNVars(grid, part);
   NALU_HYPRE_Int          *fem_vars     = nalu_hypre_SStructGridFEMPVars(grid, part);
   nalu_hypre_Index        *fem_offsets  = nalu_hypre_SStructGridFEMPOffsets(grid, part);
   NALU_HYPRE_Int           i, d, vindex[3];

   for (i = 0; i < fem_nvars; i++)
   {
      for (d = 0; d < ndim; d++)
      {
         /* note: these offsets are different from what the user passes in */
         vindex[d] = index[d] + nalu_hypre_IndexD(fem_offsets[i], d);
      }
      NALU_HYPRE_SStructVectorAddToValues(
         vector, part, vindex, fem_vars[i], &values[i]);
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructVectorGetValues( NALU_HYPRE_SStructVector  vector,
                              NALU_HYPRE_Int            part,
                              NALU_HYPRE_Int           *index,
                              NALU_HYPRE_Int            var,
                              NALU_HYPRE_Complex       *value )
{
   NALU_HYPRE_Int             ndim    = nalu_hypre_SStructVectorNDim(vector);
   nalu_hypre_SStructPVector *pvector = nalu_hypre_SStructVectorPVector(vector, part);
   nalu_hypre_Index           cindex;

   nalu_hypre_CopyToCleanIndex(index, ndim, cindex);

   if (var < nalu_hypre_SStructPVectorNVars(pvector))
   {
      nalu_hypre_SStructPVectorGetValues(pvector, cindex, var, value);
   }
   else
   {
      /* TODO */
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/* ONLY3D */

NALU_HYPRE_Int
NALU_HYPRE_SStructVectorGetFEMValues( NALU_HYPRE_SStructVector  vector,
                                 NALU_HYPRE_Int            part,
                                 NALU_HYPRE_Int           *index,
                                 NALU_HYPRE_Complex       *values )
{
   NALU_HYPRE_Int             ndim         = nalu_hypre_SStructVectorNDim(vector);
   nalu_hypre_SStructGrid    *grid         = nalu_hypre_SStructVectorGrid(vector);
   nalu_hypre_SStructPVector *pvector      = nalu_hypre_SStructVectorPVector(vector, part);
   NALU_HYPRE_Int             fem_nvars    = nalu_hypre_SStructGridFEMPNVars(grid, part);
   NALU_HYPRE_Int            *fem_vars     = nalu_hypre_SStructGridFEMPVars(grid, part);
   nalu_hypre_Index          *fem_offsets  = nalu_hypre_SStructGridFEMPOffsets(grid, part);
   NALU_HYPRE_Int             i, d, vindex[3];

   nalu_hypre_SetIndex(vindex, 0);
   for (i = 0; i < fem_nvars; i++)
   {
      for (d = 0; d < ndim; d++)
      {
         /* note: these offsets are different from what the user passes in */
         vindex[d] = index[d] + nalu_hypre_IndexD(fem_offsets[i], d);
      }
      nalu_hypre_SStructPVectorGetValues(pvector, vindex, fem_vars[i], &values[i]);
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructVectorSetBoxValues( NALU_HYPRE_SStructVector  vector,
                                 NALU_HYPRE_Int            part,
                                 NALU_HYPRE_Int           *ilower,
                                 NALU_HYPRE_Int           *iupper,
                                 NALU_HYPRE_Int            var,
                                 NALU_HYPRE_Complex       *values )
{
   NALU_HYPRE_SStructVectorSetBoxValues2(vector, part, ilower, iupper, var,
                                    ilower, iupper, values);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructVectorAddToBoxValues( NALU_HYPRE_SStructVector  vector,
                                   NALU_HYPRE_Int            part,
                                   NALU_HYPRE_Int           *ilower,
                                   NALU_HYPRE_Int           *iupper,
                                   NALU_HYPRE_Int            var,
                                   NALU_HYPRE_Complex       *values )
{
   NALU_HYPRE_SStructVectorAddToBoxValues2(vector, part, ilower, iupper, var,
                                      ilower, iupper, values);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructVectorGetBoxValues(NALU_HYPRE_SStructVector  vector,
                                NALU_HYPRE_Int            part,
                                NALU_HYPRE_Int           *ilower,
                                NALU_HYPRE_Int           *iupper,
                                NALU_HYPRE_Int            var,
                                NALU_HYPRE_Complex       *values )
{
   NALU_HYPRE_SStructVectorGetBoxValues2(vector, part, ilower, iupper, var,
                                    ilower, iupper, values);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructVectorSetBoxValues2( NALU_HYPRE_SStructVector  vector,
                                  NALU_HYPRE_Int            part,
                                  NALU_HYPRE_Int           *ilower,
                                  NALU_HYPRE_Int           *iupper,
                                  NALU_HYPRE_Int            var,
                                  NALU_HYPRE_Int           *vilower,
                                  NALU_HYPRE_Int           *viupper,
                                  NALU_HYPRE_Complex       *values )
{
   nalu_hypre_SStructPVector *pvector = nalu_hypre_SStructVectorPVector(vector, part);
   nalu_hypre_Box            *set_box, *value_box;
   NALU_HYPRE_Int             d, ndim = nalu_hypre_SStructVectorNDim(vector);

   /* This creates boxes with zeroed-out extents */
   set_box = nalu_hypre_BoxCreate(ndim);
   value_box = nalu_hypre_BoxCreate(ndim);

   for (d = 0; d < ndim; d++)
   {
      nalu_hypre_BoxIMinD(set_box, d) = ilower[d];
      nalu_hypre_BoxIMaxD(set_box, d) = iupper[d];
      nalu_hypre_BoxIMinD(value_box, d) = vilower[d];
      nalu_hypre_BoxIMaxD(value_box, d) = viupper[d];
   }

   nalu_hypre_SStructPVectorSetBoxValues(pvector, set_box, var, value_box, values, 0);

   nalu_hypre_BoxDestroy(set_box);
   nalu_hypre_BoxDestroy(value_box);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructVectorAddToBoxValues2( NALU_HYPRE_SStructVector  vector,
                                    NALU_HYPRE_Int            part,
                                    NALU_HYPRE_Int           *ilower,
                                    NALU_HYPRE_Int           *iupper,
                                    NALU_HYPRE_Int            var,
                                    NALU_HYPRE_Int           *vilower,
                                    NALU_HYPRE_Int           *viupper,
                                    NALU_HYPRE_Complex       *values )
{
   nalu_hypre_SStructPVector *pvector = nalu_hypre_SStructVectorPVector(vector, part);
   nalu_hypre_Box            *set_box, *value_box;
   NALU_HYPRE_Int             d, ndim = nalu_hypre_SStructVectorNDim(vector);

   /* This creates boxes with zeroed-out extents */
   set_box = nalu_hypre_BoxCreate(ndim);
   value_box = nalu_hypre_BoxCreate(ndim);

   for (d = 0; d < ndim; d++)
   {
      nalu_hypre_BoxIMinD(set_box, d) = ilower[d];
      nalu_hypre_BoxIMaxD(set_box, d) = iupper[d];
      nalu_hypre_BoxIMinD(value_box, d) = vilower[d];
      nalu_hypre_BoxIMaxD(value_box, d) = viupper[d];
   }

   nalu_hypre_SStructPVectorSetBoxValues(pvector, set_box, var, value_box, values, 1);

   nalu_hypre_BoxDestroy(set_box);
   nalu_hypre_BoxDestroy(value_box);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructVectorGetBoxValues2(NALU_HYPRE_SStructVector  vector,
                                 NALU_HYPRE_Int            part,
                                 NALU_HYPRE_Int           *ilower,
                                 NALU_HYPRE_Int           *iupper,
                                 NALU_HYPRE_Int            var,
                                 NALU_HYPRE_Int           *vilower,
                                 NALU_HYPRE_Int           *viupper,
                                 NALU_HYPRE_Complex       *values )
{
   nalu_hypre_SStructPVector *pvector = nalu_hypre_SStructVectorPVector(vector, part);
   nalu_hypre_Box            *set_box, *value_box;
   NALU_HYPRE_Int             d, ndim = nalu_hypre_SStructVectorNDim(vector);

   /* This creates boxes with zeroed-out extents */
   set_box = nalu_hypre_BoxCreate(ndim);
   value_box = nalu_hypre_BoxCreate(ndim);

   for (d = 0; d < ndim; d++)
   {
      nalu_hypre_BoxIMinD(set_box, d) = ilower[d];
      nalu_hypre_BoxIMaxD(set_box, d) = iupper[d];
      nalu_hypre_BoxIMinD(value_box, d) = vilower[d];
      nalu_hypre_BoxIMaxD(value_box, d) = viupper[d];
   }

   nalu_hypre_SStructPVectorGetBoxValues(pvector, set_box, var, value_box, values);

   nalu_hypre_BoxDestroy(set_box);
   nalu_hypre_BoxDestroy(value_box);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructVectorAssemble( NALU_HYPRE_SStructVector vector )
{
   nalu_hypre_SStructGrid      *grid            = nalu_hypre_SStructVectorGrid(vector);
   NALU_HYPRE_Int               nparts          = nalu_hypre_SStructVectorNParts(vector);
   NALU_HYPRE_IJVector          ijvector        = nalu_hypre_SStructVectorIJVector(vector);
   nalu_hypre_SStructCommInfo **vnbor_comm_info = nalu_hypre_SStructGridVNborCommInfo(grid);
   NALU_HYPRE_Int               vnbor_ncomms    = nalu_hypre_SStructGridVNborNComms(grid);
   NALU_HYPRE_Int               part;

   nalu_hypre_CommInfo         *comm_info;
   NALU_HYPRE_Int               send_part,    recv_part;
   NALU_HYPRE_Int               send_var,     recv_var;
   nalu_hypre_StructVector     *send_vector, *recv_vector;
   nalu_hypre_CommPkg          *comm_pkg;
   nalu_hypre_CommHandle       *comm_handle;
   NALU_HYPRE_Int               ci;

   /*------------------------------------------------------
    * Communicate and accumulate within parts
    *------------------------------------------------------*/

   for (part = 0; part < nparts; part++)
   {
      nalu_hypre_SStructPVectorAccumulate(nalu_hypre_SStructVectorPVector(vector, part));
   }

   /*------------------------------------------------------
    * Communicate and accumulate between parts
    *------------------------------------------------------*/

   for (ci = 0; ci < vnbor_ncomms; ci++)
   {
      comm_info = nalu_hypre_SStructCommInfoCommInfo(vnbor_comm_info[ci]);
      send_part = nalu_hypre_SStructCommInfoSendPart(vnbor_comm_info[ci]);
      recv_part = nalu_hypre_SStructCommInfoRecvPart(vnbor_comm_info[ci]);
      send_var  = nalu_hypre_SStructCommInfoSendVar(vnbor_comm_info[ci]);
      recv_var  = nalu_hypre_SStructCommInfoRecvVar(vnbor_comm_info[ci]);

      send_vector = nalu_hypre_SStructPVectorSVector(
                       nalu_hypre_SStructVectorPVector(vector, send_part), send_var);
      recv_vector = nalu_hypre_SStructPVectorSVector(
                       nalu_hypre_SStructVectorPVector(vector, recv_part), recv_var);

      /* want to communicate and add ghost data to real data */
      nalu_hypre_CommPkgCreate(comm_info,
                          nalu_hypre_StructVectorDataSpace(send_vector),
                          nalu_hypre_StructVectorDataSpace(recv_vector),
                          1, NULL, 1, nalu_hypre_StructVectorComm(send_vector),
                          &comm_pkg);
      /* note reversal of send/recv data here */
      nalu_hypre_InitializeCommunication(comm_pkg,
                                    nalu_hypre_StructVectorData(recv_vector),
                                    nalu_hypre_StructVectorData(send_vector),
                                    1, 0, &comm_handle);
      nalu_hypre_FinalizeCommunication(comm_handle);
      nalu_hypre_CommPkgDestroy(comm_pkg);
   }

   /*------------------------------------------------------
    * Assemble P and U vectors
    *------------------------------------------------------*/

   for (part = 0; part < nparts; part++)
   {
      nalu_hypre_SStructPVectorAssemble(nalu_hypre_SStructVectorPVector(vector, part));
   }

   /* u-vector */
   NALU_HYPRE_IJVectorAssemble(ijvector);

   NALU_HYPRE_IJVectorGetObject(ijvector,
                           (void **) &nalu_hypre_SStructVectorParVector(vector));

   /*------------------------------------------------------
    *------------------------------------------------------*/

   /* if the object type is parcsr, then convert the sstruct vector which has ghost
      layers to a parcsr vector without ghostlayers. */
   if (nalu_hypre_SStructVectorObjectType(vector) == NALU_HYPRE_PARCSR)
   {
      nalu_hypre_SStructVectorParConvert(vector,
                                    &nalu_hypre_SStructVectorParVector(vector));
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * RDF: I don't think this will work correctly in the case where a processor's
 * data is shared entirely with other processors.  The code in PGridAssemble
 * ensures that data is uniquely distributed, so the data box for this processor
 * would be empty and there would be no ghost zones to fill in Gather.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructVectorGather( NALU_HYPRE_SStructVector vector )
{
   nalu_hypre_SStructGrid      *grid            = nalu_hypre_SStructVectorGrid(vector);
   NALU_HYPRE_Int               nparts          = nalu_hypre_SStructVectorNParts(vector);
   nalu_hypre_SStructCommInfo **vnbor_comm_info = nalu_hypre_SStructGridVNborCommInfo(grid);
   NALU_HYPRE_Int               vnbor_ncomms    = nalu_hypre_SStructGridVNborNComms(grid);
   NALU_HYPRE_Int               part;

   nalu_hypre_CommInfo         *comm_info;
   NALU_HYPRE_Int               send_part,    recv_part;
   NALU_HYPRE_Int               send_var,     recv_var;
   nalu_hypre_StructVector     *send_vector, *recv_vector;
   nalu_hypre_CommPkg          *comm_pkg;
   nalu_hypre_CommHandle       *comm_handle;
   NALU_HYPRE_Int               ci;

   /* GEC1102 we change the name of the restore-->parrestore  */

   if (nalu_hypre_SStructVectorObjectType(vector) == NALU_HYPRE_PARCSR)
   {
      nalu_hypre_SStructVectorParRestore(vector, nalu_hypre_SStructVectorParVector(vector));
   }

   for (part = 0; part < nparts; part++)
   {
      nalu_hypre_SStructPVectorGather(nalu_hypre_SStructVectorPVector(vector, part));
   }

   /* gather shared data from other parts */

   for (ci = 0; ci < vnbor_ncomms; ci++)
   {
      comm_info = nalu_hypre_SStructCommInfoCommInfo(vnbor_comm_info[ci]);
      send_part = nalu_hypre_SStructCommInfoSendPart(vnbor_comm_info[ci]);
      recv_part = nalu_hypre_SStructCommInfoRecvPart(vnbor_comm_info[ci]);
      send_var  = nalu_hypre_SStructCommInfoSendVar(vnbor_comm_info[ci]);
      recv_var  = nalu_hypre_SStructCommInfoRecvVar(vnbor_comm_info[ci]);

      send_vector = nalu_hypre_SStructPVectorSVector(
                       nalu_hypre_SStructVectorPVector(vector, send_part), send_var);
      recv_vector = nalu_hypre_SStructPVectorSVector(
                       nalu_hypre_SStructVectorPVector(vector, recv_part), recv_var);

      /* want to communicate real data to ghost data */
      nalu_hypre_CommPkgCreate(comm_info,
                          nalu_hypre_StructVectorDataSpace(send_vector),
                          nalu_hypre_StructVectorDataSpace(recv_vector),
                          1, NULL, 0, nalu_hypre_StructVectorComm(send_vector),
                          &comm_pkg);
      nalu_hypre_InitializeCommunication(comm_pkg,
                                    nalu_hypre_StructVectorData(send_vector),
                                    nalu_hypre_StructVectorData(recv_vector),
                                    0, 0, &comm_handle);
      nalu_hypre_FinalizeCommunication(comm_handle);
      nalu_hypre_CommPkgDestroy(comm_pkg);

      /* boundary ghost values may not be clear */
      nalu_hypre_StructVectorBGhostNotClear(recv_vector) = 1;
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructVectorSetConstantValues( NALU_HYPRE_SStructVector vector,
                                      NALU_HYPRE_Complex       value )
{
   nalu_hypre_SStructPVector *pvector;
   NALU_HYPRE_Int part;
   NALU_HYPRE_Int nparts   = nalu_hypre_SStructVectorNParts(vector);

   for ( part = 0; part < nparts; part++ )
   {
      pvector = nalu_hypre_SStructVectorPVector( vector, part );
      nalu_hypre_SStructPVectorSetConstantValues( pvector, value );
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructVectorSetObjectType( NALU_HYPRE_SStructVector  vector,
                                  NALU_HYPRE_Int            type )
{
   /* this implements only NALU_HYPRE_PARCSR, which is always available */
   nalu_hypre_SStructVectorObjectType(vector) = type;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructVectorGetObject( NALU_HYPRE_SStructVector   vector,
                              void                **object )
{
   NALU_HYPRE_Int             type = nalu_hypre_SStructVectorObjectType(vector);
   nalu_hypre_SStructPVector *pvector;
   nalu_hypre_StructVector   *svector;
   NALU_HYPRE_Int             part, var;

   if (type == NALU_HYPRE_SSTRUCT)
   {
      *object = vector;
   }
   else if (type == NALU_HYPRE_PARCSR)
   {
      *object = nalu_hypre_SStructVectorParVector(vector);
   }
   else if (type == NALU_HYPRE_STRUCT)
   {
      /* only one part & one variable */
      part = 0;
      var = 0;
      pvector = nalu_hypre_SStructVectorPVector(vector, part);
      svector = nalu_hypre_SStructPVectorSVector(pvector, var);
      *object = svector;
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructVectorPrint
 *
 * This function prints a SStructVector to file. For the assumptions used
 * here, see NALU_HYPRE_SStructMatrixPrint.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructVectorPrint( const char          *filename,
                          NALU_HYPRE_SStructVector  vector,
                          NALU_HYPRE_Int            all )
{
   /* Vector variables */
   MPI_Comm              comm = nalu_hypre_SStructVectorComm(vector);
   NALU_HYPRE_Int             nparts = nalu_hypre_SStructVectorNParts(vector);
   nalu_hypre_SStructGrid    *grid = nalu_hypre_SStructVectorGrid(vector);

   /* Local variables */
   nalu_hypre_SStructPVector *pvector;
   nalu_hypre_StructVector   *svector;

   FILE                 *file;
   NALU_HYPRE_Int             myid;
   NALU_HYPRE_Int             part, var, nvars;
   char                  new_filename[255];

   /* Print auxiliary data */
   nalu_hypre_MPI_Comm_rank(comm, &myid);
   nalu_hypre_sprintf(new_filename, "%s.%05d", filename, myid);
   if ((file = fopen(new_filename, "w")) == NULL)
   {
      nalu_hypre_printf("Error: can't open output file %s\n", new_filename);
      nalu_hypre_error_in_arg(1);

      return nalu_hypre_error_flag;
   }

   nalu_hypre_fprintf(file, "SStructVector\n");
   nalu_hypre_SStructGridPrint(file, grid);

   /* Print (part, var) vectors */
   for (part = 0; part < nparts; part++)
   {
      pvector = nalu_hypre_SStructVectorPVector(vector, part);
      nvars = nalu_hypre_SStructPVectorNVars(pvector);
      for (var = 0; var < nvars; var++)
      {
         svector = nalu_hypre_SStructPVectorSVector(pvector, var);

         nalu_hypre_fprintf(file, "\nData - (Part %d, Var %d):\n", part, var);
         nalu_hypre_StructVectorPrintData(file, svector, all);
      }
   }

   fclose(file);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructVectorRead
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructVectorRead( MPI_Comm             comm,
                         const char          *filename,
                         NALU_HYPRE_SStructVector *vector_ptr )
{
   /* Vector variables */
   NALU_HYPRE_SStructVector    vector;
   nalu_hypre_SStructPVector  *pvector;
   nalu_hypre_StructVector    *svector;
   nalu_hypre_SStructGrid     *grid;
   NALU_HYPRE_Int              nparts;
   NALU_HYPRE_Int              nvars;

   /* Local variables */
   FILE                  *file;
   char                   new_filename[255];
   NALU_HYPRE_Int              p, v, part, var;
   NALU_HYPRE_Int              myid;

   /* Read auxiliary data */
   nalu_hypre_MPI_Comm_rank(comm, &myid);
   nalu_hypre_sprintf(new_filename, "%s.%05d", filename, myid);
   if ((file = fopen(new_filename, "r")) == NULL)
   {
      nalu_hypre_printf("Error: can't open input file %s\n", new_filename);
      nalu_hypre_error_in_arg(2);

      return nalu_hypre_error_flag;
   }

   nalu_hypre_fscanf(file, "SStructVector\n");
   nalu_hypre_SStructGridRead(comm, file, &grid);

   /* Create and initialize vector */
   NALU_HYPRE_SStructVectorCreate(comm, grid, &vector);
   NALU_HYPRE_SStructVectorInitialize(vector);

   /* Read values from file */
   nparts = nalu_hypre_SStructVectorNParts(vector);
   for (p = 0; p < nparts; p++)
   {
      pvector = nalu_hypre_SStructVectorPVector(vector, p);
      nvars = nalu_hypre_SStructPVectorNVars(pvector);

      for (v = 0; v < nvars; v++)
      {
         nalu_hypre_fscanf(file, "\nData - (Part %d, Var %d):\n", &part, &var);

         pvector = nalu_hypre_SStructVectorPVector(vector, part);
         svector = nalu_hypre_SStructPVectorSVector(pvector, var);

         nalu_hypre_StructVectorReadData(file, svector);
      }
   }
   fclose(file);

   /* Assemble vector */
   NALU_HYPRE_SStructVectorAssemble(vector);

   /* Decrease ref counters */
   NALU_HYPRE_SStructGridDestroy(grid);

   *vector_ptr = vector;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * copy x to y, y should already exist and be the same size
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructVectorCopy( NALU_HYPRE_SStructVector x,
                         NALU_HYPRE_SStructVector y )
{
   nalu_hypre_SStructCopy(x, y);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * y = a*y, for vector y and scalar a
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructVectorScale( NALU_HYPRE_Complex       alpha,
                          NALU_HYPRE_SStructVector y )
{
   nalu_hypre_SStructScale( alpha, (nalu_hypre_SStructVector *)y );

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * inner or dot product, result = < x, y >
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructInnerProd( NALU_HYPRE_SStructVector x,
                        NALU_HYPRE_SStructVector y,
                        NALU_HYPRE_Real         *result )
{
   nalu_hypre_SStructInnerProd(x, y, result);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * y = y + alpha*x for vectors y, x and scalar alpha
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructAxpy( NALU_HYPRE_Complex       alpha,
                   NALU_HYPRE_SStructVector x,
                   NALU_HYPRE_SStructVector y )
{
   nalu_hypre_SStructAxpy(alpha, x, y);

   return nalu_hypre_error_flag;
}
