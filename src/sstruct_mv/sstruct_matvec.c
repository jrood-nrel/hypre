/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * SStruct matrix-vector multiply routine
 *
 *****************************************************************************/

#include "_nalu_hypre_sstruct_mv.h"

/*==========================================================================
 * PMatvec routines
 *==========================================================================*/

/*--------------------------------------------------------------------------
 * nalu_hypre_SStructPMatvecData data structure
 *--------------------------------------------------------------------------*/

typedef struct
{
   NALU_HYPRE_Int     nvars;
   void ***smatvec_data;

} nalu_hypre_SStructPMatvecData;

/*--------------------------------------------------------------------------
 * nalu_hypre_SStructPMatvecCreate
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SStructPMatvecCreate( void **pmatvec_vdata_ptr )
{
   nalu_hypre_SStructPMatvecData *pmatvec_data;

   pmatvec_data = nalu_hypre_CTAlloc(nalu_hypre_SStructPMatvecData,  1, NALU_HYPRE_MEMORY_HOST);
   *pmatvec_vdata_ptr = (void *) pmatvec_data;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_SStructPMatvecSetup
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SStructPMatvecSetup( void                 *pmatvec_vdata,
                           nalu_hypre_SStructPMatrix *pA,
                           nalu_hypre_SStructPVector *px )
{
   nalu_hypre_SStructPMatvecData   *pmatvec_data = (nalu_hypre_SStructPMatvecData   *)pmatvec_vdata;
   NALU_HYPRE_Int                   nvars;
   void                     ***smatvec_data;
   nalu_hypre_StructMatrix         *sA;
   nalu_hypre_StructVector         *sx;
   NALU_HYPRE_Int                   vi, vj;

   nvars = nalu_hypre_SStructPMatrixNVars(pA);
   smatvec_data = nalu_hypre_TAlloc(void **,  nvars, NALU_HYPRE_MEMORY_HOST);
   for (vi = 0; vi < nvars; vi++)
   {
      smatvec_data[vi] = nalu_hypre_TAlloc(void *,  nvars, NALU_HYPRE_MEMORY_HOST);
      for (vj = 0; vj < nvars; vj++)
      {
         sA = nalu_hypre_SStructPMatrixSMatrix(pA, vi, vj);
         sx = nalu_hypre_SStructPVectorSVector(px, vj);
         smatvec_data[vi][vj] = NULL;
         if (sA != NULL)
         {
            smatvec_data[vi][vj] = nalu_hypre_StructMatvecCreate();
            nalu_hypre_StructMatvecSetup(smatvec_data[vi][vj], sA, sx);
         }
      }
   }
   (pmatvec_data -> nvars)        = nvars;
   (pmatvec_data -> smatvec_data) = smatvec_data;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_SStructPMatvecCompute
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SStructPMatvecCompute( void                 *pmatvec_vdata,
                             NALU_HYPRE_Complex         alpha,
                             nalu_hypre_SStructPMatrix *pA,
                             nalu_hypre_SStructPVector *px,
                             NALU_HYPRE_Complex         beta,
                             nalu_hypre_SStructPVector *py )
{
   nalu_hypre_SStructPMatvecData   *pmatvec_data = (nalu_hypre_SStructPMatvecData   *)pmatvec_vdata;
   NALU_HYPRE_Int                   nvars        = (pmatvec_data -> nvars);
   void                     ***smatvec_data = (pmatvec_data -> smatvec_data);

   void                       *sdata;
   nalu_hypre_StructMatrix         *sA;
   nalu_hypre_StructVector         *sx;
   nalu_hypre_StructVector         *sy;

   NALU_HYPRE_Int                  vi, vj;

   for (vi = 0; vi < nvars; vi++)
   {
      sy = nalu_hypre_SStructPVectorSVector(py, vi);

      /* diagonal block computation */
      if (smatvec_data[vi][vi] != NULL)
      {
         sdata = smatvec_data[vi][vi];
         sA = nalu_hypre_SStructPMatrixSMatrix(pA, vi, vi);
         sx = nalu_hypre_SStructPVectorSVector(px, vi);
         nalu_hypre_StructMatvecCompute(sdata, alpha, sA, sx, beta, sy);
      }
      else
      {
         nalu_hypre_StructScale(beta, sy);
      }

      /* off-diagonal block computation */
      for (vj = 0; vj < nvars; vj++)
      {
         if ((smatvec_data[vi][vj] != NULL) && (vj != vi))
         {
            sdata = smatvec_data[vi][vj];
            sA = nalu_hypre_SStructPMatrixSMatrix(pA, vi, vj);
            sx = nalu_hypre_SStructPVectorSVector(px, vj);
            nalu_hypre_StructMatvecCompute(sdata, alpha, sA, sx, 1.0, sy);
         }
      }
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_SStructPMatvecDestroy
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SStructPMatvecDestroy( void *pmatvec_vdata )
{
   nalu_hypre_SStructPMatvecData   *pmatvec_data = (nalu_hypre_SStructPMatvecData   *)pmatvec_vdata;
   NALU_HYPRE_Int                   nvars;
   void                     ***smatvec_data;
   NALU_HYPRE_Int                   vi, vj;

   if (pmatvec_data)
   {
      nvars        = (pmatvec_data -> nvars);
      smatvec_data = (pmatvec_data -> smatvec_data);
      for (vi = 0; vi < nvars; vi++)
      {
         for (vj = 0; vj < nvars; vj++)
         {
            if (smatvec_data[vi][vj] != NULL)
            {
               nalu_hypre_StructMatvecDestroy(smatvec_data[vi][vj]);
            }
         }
         nalu_hypre_TFree(smatvec_data[vi], NALU_HYPRE_MEMORY_HOST);
      }
      nalu_hypre_TFree(smatvec_data, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(pmatvec_data, NALU_HYPRE_MEMORY_HOST);
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_SStructPMatvec
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SStructPMatvec( NALU_HYPRE_Complex         alpha,
                      nalu_hypre_SStructPMatrix *pA,
                      nalu_hypre_SStructPVector *px,
                      NALU_HYPRE_Complex         beta,
                      nalu_hypre_SStructPVector *py )
{
   void *pmatvec_data;

   nalu_hypre_SStructPMatvecCreate(&pmatvec_data);
   nalu_hypre_SStructPMatvecSetup(pmatvec_data, pA, px);
   nalu_hypre_SStructPMatvecCompute(pmatvec_data, alpha, pA, px, beta, py);
   nalu_hypre_SStructPMatvecDestroy(pmatvec_data);

   return nalu_hypre_error_flag;
}

/*==========================================================================
 * Matvec routines
 *==========================================================================*/

/*--------------------------------------------------------------------------
 * nalu_hypre_SStructMatvecData data structure
 *--------------------------------------------------------------------------*/

typedef struct
{
   NALU_HYPRE_Int    nparts;
   void **pmatvec_data;

} nalu_hypre_SStructMatvecData;

/*--------------------------------------------------------------------------
 * nalu_hypre_SStructMatvecCreate
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SStructMatvecCreate( void **matvec_vdata_ptr )
{
   nalu_hypre_SStructMatvecData *matvec_data;

   matvec_data = nalu_hypre_CTAlloc(nalu_hypre_SStructMatvecData,  1, NALU_HYPRE_MEMORY_HOST);
   *matvec_vdata_ptr = (void *) matvec_data;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_SStructMatvecSetup
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SStructMatvecSetup( void                *matvec_vdata,
                          nalu_hypre_SStructMatrix *A,
                          nalu_hypre_SStructVector *x )
{
   nalu_hypre_SStructMatvecData  *matvec_data = (nalu_hypre_SStructMatvecData   *)matvec_vdata;
   NALU_HYPRE_Int                 nparts;
   void                    **pmatvec_data;
   nalu_hypre_SStructPMatrix     *pA;
   nalu_hypre_SStructPVector     *px;
   NALU_HYPRE_Int                 part;

   nparts = nalu_hypre_SStructMatrixNParts(A);
   pmatvec_data = nalu_hypre_TAlloc(void *,  nparts, NALU_HYPRE_MEMORY_HOST);
   for (part = 0; part < nparts; part++)
   {
      nalu_hypre_SStructPMatvecCreate(&pmatvec_data[part]);
      pA = nalu_hypre_SStructMatrixPMatrix(A, part);
      px = nalu_hypre_SStructVectorPVector(x, part);
      nalu_hypre_SStructPMatvecSetup(pmatvec_data[part], pA, px);
   }
   (matvec_data -> nparts)       = nparts;
   (matvec_data -> pmatvec_data) = pmatvec_data;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_SStructMatvecCompute
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SStructMatvecCompute( void                *matvec_vdata,
                            NALU_HYPRE_Complex        alpha,
                            nalu_hypre_SStructMatrix *A,
                            nalu_hypre_SStructVector *x,
                            NALU_HYPRE_Complex        beta,
                            nalu_hypre_SStructVector *y )
{
   nalu_hypre_SStructMatvecData  *matvec_data  = (nalu_hypre_SStructMatvecData   *)matvec_vdata;
   NALU_HYPRE_Int                 nparts       = (matvec_data -> nparts);
   void                    **pmatvec_data = (matvec_data -> pmatvec_data);

   void                     *pdata;
   nalu_hypre_SStructPMatrix     *pA;
   nalu_hypre_SStructPVector     *px;
   nalu_hypre_SStructPVector     *py;

   nalu_hypre_ParCSRMatrix       *parcsrA = nalu_hypre_SStructMatrixParCSRMatrix(A);
   nalu_hypre_ParVector          *parx;
   nalu_hypre_ParVector          *pary;

   NALU_HYPRE_Int                 part;
   NALU_HYPRE_Int                 x_object_type = nalu_hypre_SStructVectorObjectType(x);
   NALU_HYPRE_Int                 A_object_type = nalu_hypre_SStructMatrixObjectType(A);

   if (x_object_type != A_object_type)
   {
      nalu_hypre_error_in_arg(2);
      nalu_hypre_error_in_arg(3);
      return nalu_hypre_error_flag;
   }

   if ( (x_object_type == NALU_HYPRE_SSTRUCT) || (x_object_type == NALU_HYPRE_STRUCT) )
   {
      /* do S-matrix computations */
      for (part = 0; part < nparts; part++)
      {
         pdata = pmatvec_data[part];
         pA = nalu_hypre_SStructMatrixPMatrix(A, part);
         px = nalu_hypre_SStructVectorPVector(x, part);
         py = nalu_hypre_SStructVectorPVector(y, part);
         nalu_hypre_SStructPMatvecCompute(pdata, alpha, pA, px, beta, py);
      }

      if (x_object_type == NALU_HYPRE_SSTRUCT)
      {

         /* do U-matrix computations */

         /* GEC1002 the data chunk pointed by the local-parvectors
          *  inside the semistruct vectors x and y is now identical to the
          *  data chunk of the structure vectors x and y. The role of the function
          *  convert is to pass the addresses of the data chunk
          *  to the parx and pary. */

         nalu_hypre_SStructVectorConvert(x, &parx);
         nalu_hypre_SStructVectorConvert(y, &pary);

         nalu_hypre_ParCSRMatrixMatvec(alpha, parcsrA, parx, 1.0, pary);

         /* dummy functions since there is nothing to restore  */

         nalu_hypre_SStructVectorRestore(x, NULL);
         nalu_hypre_SStructVectorRestore(y, pary);

         parx = NULL;
      }

   }

   else if (x_object_type == NALU_HYPRE_PARCSR)
   {
      nalu_hypre_SStructVectorConvert(x, &parx);
      nalu_hypre_SStructVectorConvert(y, &pary);

      nalu_hypre_ParCSRMatrixMatvec(alpha, parcsrA, parx, beta, pary);

      nalu_hypre_SStructVectorRestore(x, NULL);
      nalu_hypre_SStructVectorRestore(y, pary);

      parx = NULL;
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_SStructMatvecDestroy
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SStructMatvecDestroy( void *matvec_vdata )
{
   nalu_hypre_SStructMatvecData  *matvec_data = (nalu_hypre_SStructMatvecData   *)matvec_vdata;
   NALU_HYPRE_Int                 nparts;
   void                    **pmatvec_data;
   NALU_HYPRE_Int                 part;

   if (matvec_data)
   {
      nparts       = (matvec_data -> nparts);
      pmatvec_data = (matvec_data -> pmatvec_data);
      for (part = 0; part < nparts; part++)
      {
         nalu_hypre_SStructPMatvecDestroy(pmatvec_data[part]);
      }
      nalu_hypre_TFree(pmatvec_data, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(matvec_data, NALU_HYPRE_MEMORY_HOST);
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_SStructMatvec
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SStructMatvec( NALU_HYPRE_Complex        alpha,
                     nalu_hypre_SStructMatrix *A,
                     nalu_hypre_SStructVector *x,
                     NALU_HYPRE_Complex        beta,
                     nalu_hypre_SStructVector *y )
{
   void *matvec_data;

   nalu_hypre_SStructMatvecCreate(&matvec_data);
   nalu_hypre_SStructMatvecSetup(matvec_data, A, x);
   nalu_hypre_SStructMatvecCompute(matvec_data, alpha, A, x, beta, y);
   nalu_hypre_SStructMatvecDestroy(matvec_data);

   return nalu_hypre_error_flag;
}
