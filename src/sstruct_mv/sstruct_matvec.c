/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * SStruct matrix-vector multiply routine
 *
 *****************************************************************************/

#include "_hypre_sstruct_mv.h"

/*==========================================================================
 * PMatvec routines
 *==========================================================================*/

/*--------------------------------------------------------------------------
 * hypre_SStructPMatvecData data structure
 *--------------------------------------------------------------------------*/

typedef struct
{
   NALU_HYPRE_Int     nvars;
   void ***smatvec_data;

} hypre_SStructPMatvecData;

/*--------------------------------------------------------------------------
 * hypre_SStructPMatvecCreate
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
hypre_SStructPMatvecCreate( void **pmatvec_vdata_ptr )
{
   hypre_SStructPMatvecData *pmatvec_data;

   pmatvec_data = hypre_CTAlloc(hypre_SStructPMatvecData,  1, NALU_HYPRE_MEMORY_HOST);
   *pmatvec_vdata_ptr = (void *) pmatvec_data;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_SStructPMatvecSetup
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
hypre_SStructPMatvecSetup( void                 *pmatvec_vdata,
                           hypre_SStructPMatrix *pA,
                           hypre_SStructPVector *px )
{
   hypre_SStructPMatvecData   *pmatvec_data = (hypre_SStructPMatvecData   *)pmatvec_vdata;
   NALU_HYPRE_Int                   nvars;
   void                     ***smatvec_data;
   hypre_StructMatrix         *sA;
   hypre_StructVector         *sx;
   NALU_HYPRE_Int                   vi, vj;

   nvars = hypre_SStructPMatrixNVars(pA);
   smatvec_data = hypre_TAlloc(void **,  nvars, NALU_HYPRE_MEMORY_HOST);
   for (vi = 0; vi < nvars; vi++)
   {
      smatvec_data[vi] = hypre_TAlloc(void *,  nvars, NALU_HYPRE_MEMORY_HOST);
      for (vj = 0; vj < nvars; vj++)
      {
         sA = hypre_SStructPMatrixSMatrix(pA, vi, vj);
         sx = hypre_SStructPVectorSVector(px, vj);
         smatvec_data[vi][vj] = NULL;
         if (sA != NULL)
         {
            smatvec_data[vi][vj] = hypre_StructMatvecCreate();
            hypre_StructMatvecSetup(smatvec_data[vi][vj], sA, sx);
         }
      }
   }
   (pmatvec_data -> nvars)        = nvars;
   (pmatvec_data -> smatvec_data) = smatvec_data;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_SStructPMatvecCompute
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
hypre_SStructPMatvecCompute( void                 *pmatvec_vdata,
                             NALU_HYPRE_Complex         alpha,
                             hypre_SStructPMatrix *pA,
                             hypre_SStructPVector *px,
                             NALU_HYPRE_Complex         beta,
                             hypre_SStructPVector *py )
{
   hypre_SStructPMatvecData   *pmatvec_data = (hypre_SStructPMatvecData   *)pmatvec_vdata;
   NALU_HYPRE_Int                   nvars        = (pmatvec_data -> nvars);
   void                     ***smatvec_data = (pmatvec_data -> smatvec_data);

   void                       *sdata;
   hypre_StructMatrix         *sA;
   hypre_StructVector         *sx;
   hypre_StructVector         *sy;

   NALU_HYPRE_Int                  vi, vj;

   for (vi = 0; vi < nvars; vi++)
   {
      sy = hypre_SStructPVectorSVector(py, vi);

      /* diagonal block computation */
      if (smatvec_data[vi][vi] != NULL)
      {
         sdata = smatvec_data[vi][vi];
         sA = hypre_SStructPMatrixSMatrix(pA, vi, vi);
         sx = hypre_SStructPVectorSVector(px, vi);
         hypre_StructMatvecCompute(sdata, alpha, sA, sx, beta, sy);
      }
      else
      {
         hypre_StructScale(beta, sy);
      }

      /* off-diagonal block computation */
      for (vj = 0; vj < nvars; vj++)
      {
         if ((smatvec_data[vi][vj] != NULL) && (vj != vi))
         {
            sdata = smatvec_data[vi][vj];
            sA = hypre_SStructPMatrixSMatrix(pA, vi, vj);
            sx = hypre_SStructPVectorSVector(px, vj);
            hypre_StructMatvecCompute(sdata, alpha, sA, sx, 1.0, sy);
         }
      }
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_SStructPMatvecDestroy
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
hypre_SStructPMatvecDestroy( void *pmatvec_vdata )
{
   hypre_SStructPMatvecData   *pmatvec_data = (hypre_SStructPMatvecData   *)pmatvec_vdata;
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
               hypre_StructMatvecDestroy(smatvec_data[vi][vj]);
            }
         }
         hypre_TFree(smatvec_data[vi], NALU_HYPRE_MEMORY_HOST);
      }
      hypre_TFree(smatvec_data, NALU_HYPRE_MEMORY_HOST);
      hypre_TFree(pmatvec_data, NALU_HYPRE_MEMORY_HOST);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_SStructPMatvec
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
hypre_SStructPMatvec( NALU_HYPRE_Complex         alpha,
                      hypre_SStructPMatrix *pA,
                      hypre_SStructPVector *px,
                      NALU_HYPRE_Complex         beta,
                      hypre_SStructPVector *py )
{
   void *pmatvec_data;

   hypre_SStructPMatvecCreate(&pmatvec_data);
   hypre_SStructPMatvecSetup(pmatvec_data, pA, px);
   hypre_SStructPMatvecCompute(pmatvec_data, alpha, pA, px, beta, py);
   hypre_SStructPMatvecDestroy(pmatvec_data);

   return hypre_error_flag;
}

/*==========================================================================
 * Matvec routines
 *==========================================================================*/

/*--------------------------------------------------------------------------
 * hypre_SStructMatvecData data structure
 *--------------------------------------------------------------------------*/

typedef struct
{
   NALU_HYPRE_Int    nparts;
   void **pmatvec_data;

} hypre_SStructMatvecData;

/*--------------------------------------------------------------------------
 * hypre_SStructMatvecCreate
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
hypre_SStructMatvecCreate( void **matvec_vdata_ptr )
{
   hypre_SStructMatvecData *matvec_data;

   matvec_data = hypre_CTAlloc(hypre_SStructMatvecData,  1, NALU_HYPRE_MEMORY_HOST);
   *matvec_vdata_ptr = (void *) matvec_data;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_SStructMatvecSetup
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
hypre_SStructMatvecSetup( void                *matvec_vdata,
                          hypre_SStructMatrix *A,
                          hypre_SStructVector *x )
{
   hypre_SStructMatvecData  *matvec_data = (hypre_SStructMatvecData   *)matvec_vdata;
   NALU_HYPRE_Int                 nparts;
   void                    **pmatvec_data;
   hypre_SStructPMatrix     *pA;
   hypre_SStructPVector     *px;
   NALU_HYPRE_Int                 part;

   nparts = hypre_SStructMatrixNParts(A);
   pmatvec_data = hypre_TAlloc(void *,  nparts, NALU_HYPRE_MEMORY_HOST);
   for (part = 0; part < nparts; part++)
   {
      hypre_SStructPMatvecCreate(&pmatvec_data[part]);
      pA = hypre_SStructMatrixPMatrix(A, part);
      px = hypre_SStructVectorPVector(x, part);
      hypre_SStructPMatvecSetup(pmatvec_data[part], pA, px);
   }
   (matvec_data -> nparts)       = nparts;
   (matvec_data -> pmatvec_data) = pmatvec_data;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_SStructMatvecCompute
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
hypre_SStructMatvecCompute( void                *matvec_vdata,
                            NALU_HYPRE_Complex        alpha,
                            hypre_SStructMatrix *A,
                            hypre_SStructVector *x,
                            NALU_HYPRE_Complex        beta,
                            hypre_SStructVector *y )
{
   hypre_SStructMatvecData  *matvec_data  = (hypre_SStructMatvecData   *)matvec_vdata;
   NALU_HYPRE_Int                 nparts       = (matvec_data -> nparts);
   void                    **pmatvec_data = (matvec_data -> pmatvec_data);

   void                     *pdata;
   hypre_SStructPMatrix     *pA;
   hypre_SStructPVector     *px;
   hypre_SStructPVector     *py;

   hypre_ParCSRMatrix       *parcsrA = hypre_SStructMatrixParCSRMatrix(A);
   hypre_ParVector          *parx;
   hypre_ParVector          *pary;

   NALU_HYPRE_Int                 part;
   NALU_HYPRE_Int                 x_object_type = hypre_SStructVectorObjectType(x);
   NALU_HYPRE_Int                 A_object_type = hypre_SStructMatrixObjectType(A);

   if (x_object_type != A_object_type)
   {
      hypre_error_in_arg(2);
      hypre_error_in_arg(3);
      return hypre_error_flag;
   }

   if ( (x_object_type == NALU_HYPRE_SSTRUCT) || (x_object_type == NALU_HYPRE_STRUCT) )
   {
      /* do S-matrix computations */
      for (part = 0; part < nparts; part++)
      {
         pdata = pmatvec_data[part];
         pA = hypre_SStructMatrixPMatrix(A, part);
         px = hypre_SStructVectorPVector(x, part);
         py = hypre_SStructVectorPVector(y, part);
         hypre_SStructPMatvecCompute(pdata, alpha, pA, px, beta, py);
      }

      if (x_object_type == NALU_HYPRE_SSTRUCT)
      {

         /* do U-matrix computations */

         /* GEC1002 the data chunk pointed by the local-parvectors
          *  inside the semistruct vectors x and y is now identical to the
          *  data chunk of the structure vectors x and y. The role of the function
          *  convert is to pass the addresses of the data chunk
          *  to the parx and pary. */

         hypre_SStructVectorConvert(x, &parx);
         hypre_SStructVectorConvert(y, &pary);

         hypre_ParCSRMatrixMatvec(alpha, parcsrA, parx, 1.0, pary);

         /* dummy functions since there is nothing to restore  */

         hypre_SStructVectorRestore(x, NULL);
         hypre_SStructVectorRestore(y, pary);

         parx = NULL;
      }

   }

   else if (x_object_type == NALU_HYPRE_PARCSR)
   {
      hypre_SStructVectorConvert(x, &parx);
      hypre_SStructVectorConvert(y, &pary);

      hypre_ParCSRMatrixMatvec(alpha, parcsrA, parx, beta, pary);

      hypre_SStructVectorRestore(x, NULL);
      hypre_SStructVectorRestore(y, pary);

      parx = NULL;
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_SStructMatvecDestroy
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
hypre_SStructMatvecDestroy( void *matvec_vdata )
{
   hypre_SStructMatvecData  *matvec_data = (hypre_SStructMatvecData   *)matvec_vdata;
   NALU_HYPRE_Int                 nparts;
   void                    **pmatvec_data;
   NALU_HYPRE_Int                 part;

   if (matvec_data)
   {
      nparts       = (matvec_data -> nparts);
      pmatvec_data = (matvec_data -> pmatvec_data);
      for (part = 0; part < nparts; part++)
      {
         hypre_SStructPMatvecDestroy(pmatvec_data[part]);
      }
      hypre_TFree(pmatvec_data, NALU_HYPRE_MEMORY_HOST);
      hypre_TFree(matvec_data, NALU_HYPRE_MEMORY_HOST);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_SStructMatvec
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
hypre_SStructMatvec( NALU_HYPRE_Complex        alpha,
                     hypre_SStructMatrix *A,
                     hypre_SStructVector *x,
                     NALU_HYPRE_Complex        beta,
                     hypre_SStructVector *y )
{
   void *matvec_data;

   hypre_SStructMatvecCreate(&matvec_data);
   hypre_SStructMatvecSetup(matvec_data, A, x);
   hypre_SStructMatvecCompute(matvec_data, alpha, A, x, beta, y);
   hypre_SStructMatvecDestroy(matvec_data);

   return hypre_error_flag;
}
