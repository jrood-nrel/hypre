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
   void              **sinterp_data;

} nalu_hypre_SysSemiInterpData;

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SysSemiInterpCreate( void **sys_interp_vdata_ptr )
{
   nalu_hypre_SysSemiInterpData *sys_interp_data;

   sys_interp_data = nalu_hypre_CTAlloc(nalu_hypre_SysSemiInterpData,  1, NALU_HYPRE_MEMORY_HOST);
   *sys_interp_vdata_ptr = (void *) sys_interp_data;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SysSemiInterpSetup( void                 *sys_interp_vdata,
                          nalu_hypre_SStructPMatrix *P,
                          NALU_HYPRE_Int             P_stored_as_transpose,
                          nalu_hypre_SStructPVector *xc,
                          nalu_hypre_SStructPVector *e,
                          nalu_hypre_Index           cindex,
                          nalu_hypre_Index           findex,
                          nalu_hypre_Index           stride       )
{
   nalu_hypre_SysSemiInterpData  *sys_interp_data = (nalu_hypre_SysSemiInterpData  *)sys_interp_vdata;
   void                    **sinterp_data;

   NALU_HYPRE_Int                 nvars;

   nalu_hypre_StructMatrix       *P_s;
   nalu_hypre_StructVector       *xc_s;
   nalu_hypre_StructVector       *e_s;

   NALU_HYPRE_Int                 vi;

   nvars = nalu_hypre_SStructPMatrixNVars(P);
   sinterp_data = nalu_hypre_CTAlloc(void *,  nvars, NALU_HYPRE_MEMORY_HOST);

   for (vi = 0; vi < nvars; vi++)
   {
      P_s  = nalu_hypre_SStructPMatrixSMatrix(P, vi, vi);
      xc_s = nalu_hypre_SStructPVectorSVector(xc, vi);
      e_s  = nalu_hypre_SStructPVectorSVector(e, vi);
      sinterp_data[vi] = nalu_hypre_SemiInterpCreate( );
      nalu_hypre_SemiInterpSetup( sinterp_data[vi], P_s, P_stored_as_transpose,
                             xc_s, e_s, cindex, findex, stride);
   }

   (sys_interp_data -> nvars)        = nvars;
   (sys_interp_data -> sinterp_data) = sinterp_data;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SysSemiInterp( void                 *sys_interp_vdata,
                     nalu_hypre_SStructPMatrix *P,
                     nalu_hypre_SStructPVector *xc,
                     nalu_hypre_SStructPVector *e            )
{
   nalu_hypre_SysSemiInterpData  *sys_interp_data = (nalu_hypre_SysSemiInterpData  *)sys_interp_vdata;
   void                    **sinterp_data = (sys_interp_data -> sinterp_data);
   NALU_HYPRE_Int                 nvars = (sys_interp_data -> nvars);

   void                     *sdata;
   nalu_hypre_StructMatrix       *P_s;
   nalu_hypre_StructVector       *xc_s;
   nalu_hypre_StructVector       *e_s;

   NALU_HYPRE_Int                 vi;

   for (vi = 0; vi < nvars; vi++)
   {
      sdata = sinterp_data[vi];
      P_s  = nalu_hypre_SStructPMatrixSMatrix(P, vi, vi);
      xc_s = nalu_hypre_SStructPVectorSVector(xc, vi);
      e_s  = nalu_hypre_SStructPVectorSVector(e, vi);
      nalu_hypre_SemiInterp(sdata, P_s, xc_s, e_s);
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SysSemiInterpDestroy( void *sys_interp_vdata )
{
   nalu_hypre_SysSemiInterpData *sys_interp_data = (nalu_hypre_SysSemiInterpData  *)sys_interp_vdata;

   NALU_HYPRE_Int             nvars;
   void                **sinterp_data;
   NALU_HYPRE_Int             vi;

   if (sys_interp_data)
   {
      nvars        = (sys_interp_data -> nvars);
      sinterp_data = (sys_interp_data -> sinterp_data);
      for (vi = 0; vi < nvars; vi++)
      {
         if (sinterp_data[vi] != NULL)
         {
            nalu_hypre_SemiInterpDestroy(sinterp_data[vi]);
         }
      }
      nalu_hypre_TFree(sinterp_data, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(sys_interp_data, NALU_HYPRE_MEMORY_HOST);
   }

   return nalu_hypre_error_flag;
}

