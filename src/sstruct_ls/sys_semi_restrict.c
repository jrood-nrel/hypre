/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_nalu_hypre_sstruct_ls.h"

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

typedef struct
{
   NALU_HYPRE_Int           nvars;
   void              **srestrict_data;
} nalu_hypre_SysSemiRestrictData;

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SysSemiRestrictCreate( void **sys_restrict_vdata_ptr)
{
   nalu_hypre_SysSemiRestrictData *sys_restrict_data;

   sys_restrict_data = nalu_hypre_CTAlloc(nalu_hypre_SysSemiRestrictData,  1, NALU_HYPRE_MEMORY_HOST);
   *sys_restrict_vdata_ptr = (void *) sys_restrict_data;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SysSemiRestrictSetup( void                 *sys_restrict_vdata,
                            nalu_hypre_SStructPMatrix *R,
                            NALU_HYPRE_Int             R_stored_as_transpose,
                            nalu_hypre_SStructPVector *r,
                            nalu_hypre_SStructPVector *rc,
                            nalu_hypre_Index           cindex,
                            nalu_hypre_Index           findex,
                            nalu_hypre_Index           stride                )
{
   nalu_hypre_SysSemiRestrictData  *sys_restrict_data = (nalu_hypre_SysSemiRestrictData  *)sys_restrict_vdata;
   void                      **srestrict_data;

   NALU_HYPRE_Int                   nvars;

   nalu_hypre_StructMatrix         *R_s;
   nalu_hypre_StructVector         *rc_s;
   nalu_hypre_StructVector         *r_s;

   NALU_HYPRE_Int                   vi;

   nvars = nalu_hypre_SStructPMatrixNVars(R);
   srestrict_data = nalu_hypre_CTAlloc(void *,  nvars, NALU_HYPRE_MEMORY_HOST);

   for (vi = 0; vi < nvars; vi++)
   {
      R_s  = nalu_hypre_SStructPMatrixSMatrix(R, vi, vi);
      rc_s = nalu_hypre_SStructPVectorSVector(rc, vi);
      r_s  = nalu_hypre_SStructPVectorSVector(r, vi);
      srestrict_data[vi] = nalu_hypre_SemiRestrictCreate( );
      nalu_hypre_SemiRestrictSetup( srestrict_data[vi], R_s, R_stored_as_transpose,
                               r_s, rc_s, cindex, findex, stride);
   }

   (sys_restrict_data -> nvars)        = nvars;
   (sys_restrict_data -> srestrict_data) = srestrict_data;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SysSemiRestrict( void                 *sys_restrict_vdata,
                       nalu_hypre_SStructPMatrix *R,
                       nalu_hypre_SStructPVector *r,
                       nalu_hypre_SStructPVector *rc             )
{
   nalu_hypre_SysSemiRestrictData  *sys_restrict_data = (nalu_hypre_SysSemiRestrictData  *)sys_restrict_vdata;
   void                      **srestrict_data
      = (sys_restrict_data -> srestrict_data);
   NALU_HYPRE_Int                   nvars = (sys_restrict_data -> nvars);

   void                       *sdata;
   nalu_hypre_StructMatrix         *R_s;
   nalu_hypre_StructVector         *rc_s;
   nalu_hypre_StructVector         *r_s;

   NALU_HYPRE_Int                   vi;

   for (vi = 0; vi < nvars; vi++)
   {
      sdata = srestrict_data[vi];
      R_s  = nalu_hypre_SStructPMatrixSMatrix(R, vi, vi);
      rc_s = nalu_hypre_SStructPVectorSVector(rc, vi);
      r_s  = nalu_hypre_SStructPVectorSVector(r, vi);
      nalu_hypre_SemiRestrict(sdata, R_s, r_s, rc_s);
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SysSemiRestrictDestroy( void *sys_restrict_vdata )
{
   nalu_hypre_SysSemiRestrictData *sys_restrict_data = (nalu_hypre_SysSemiRestrictData  *)sys_restrict_vdata;

   NALU_HYPRE_Int               nvars;
   void                  **srestrict_data;
   NALU_HYPRE_Int               vi;

   if (sys_restrict_data)
   {
      nvars        = (sys_restrict_data -> nvars);
      srestrict_data = (sys_restrict_data -> srestrict_data);
      for (vi = 0; vi < nvars; vi++)
      {
         if (srestrict_data[vi] != NULL)
         {
            nalu_hypre_SemiRestrictDestroy(srestrict_data[vi]);
         }
      }
      nalu_hypre_TFree(srestrict_data, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(sys_restrict_data, NALU_HYPRE_MEMORY_HOST);
   }

   return nalu_hypre_error_flag;
}

