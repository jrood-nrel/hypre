/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_nalu_hypre_struct_ls.h"

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

typedef struct
{
   void                   *relax_data;
   void                   *rb_relax_data;
   NALU_HYPRE_Int               relax_type;
   NALU_HYPRE_Real              jacobi_weight;

} nalu_hypre_PFMGRelaxData;

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

void *
nalu_hypre_PFMGRelaxCreate( MPI_Comm  comm )
{
   nalu_hypre_PFMGRelaxData *pfmg_relax_data;

   pfmg_relax_data = nalu_hypre_CTAlloc(nalu_hypre_PFMGRelaxData,  1, NALU_HYPRE_MEMORY_HOST);
   (pfmg_relax_data -> relax_data) = nalu_hypre_PointRelaxCreate(comm);
   (pfmg_relax_data -> rb_relax_data) = nalu_hypre_RedBlackGSCreate(comm);
   (pfmg_relax_data -> relax_type) = 0;        /* Weighted Jacobi */
   (pfmg_relax_data -> jacobi_weight) = 0.0;

   return (void *) pfmg_relax_data;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_PFMGRelaxDestroy( void *pfmg_relax_vdata )
{
   nalu_hypre_PFMGRelaxData *pfmg_relax_data = (nalu_hypre_PFMGRelaxData *)pfmg_relax_vdata;

   if (pfmg_relax_data)
   {
      nalu_hypre_PointRelaxDestroy(pfmg_relax_data -> relax_data);
      nalu_hypre_RedBlackGSDestroy(pfmg_relax_data -> rb_relax_data);
      nalu_hypre_TFree(pfmg_relax_data, NALU_HYPRE_MEMORY_HOST);
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_PFMGRelax( void               *pfmg_relax_vdata,
                 nalu_hypre_StructMatrix *A,
                 nalu_hypre_StructVector *b,
                 nalu_hypre_StructVector *x                )
{
   nalu_hypre_PFMGRelaxData *pfmg_relax_data = (nalu_hypre_PFMGRelaxData *)pfmg_relax_vdata;
   NALU_HYPRE_Int    relax_type = (pfmg_relax_data -> relax_type);
   NALU_HYPRE_Int    constant_coefficient = nalu_hypre_StructMatrixConstantCoefficient(A);

   switch (relax_type)
   {
      case 0:
      case 1:
         nalu_hypre_PointRelax((pfmg_relax_data -> relax_data), A, b, x);
         break;
      case 2:
      case 3:
         if (constant_coefficient)
         {
            nalu_hypre_RedBlackConstantCoefGS((pfmg_relax_data -> rb_relax_data),
                                         A, b, x);
         }
         else
         {
            nalu_hypre_RedBlackGS((pfmg_relax_data -> rb_relax_data), A, b, x);
         }

         break;
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_PFMGRelaxSetup( void               *pfmg_relax_vdata,
                      nalu_hypre_StructMatrix *A,
                      nalu_hypre_StructVector *b,
                      nalu_hypre_StructVector *x                )
{
   nalu_hypre_PFMGRelaxData *pfmg_relax_data  = (nalu_hypre_PFMGRelaxData *)pfmg_relax_vdata;
   NALU_HYPRE_Int            relax_type       = (pfmg_relax_data -> relax_type);
   NALU_HYPRE_Real           jacobi_weight    = (pfmg_relax_data -> jacobi_weight);

   switch (relax_type)
   {
      case 0:
      case 1:
         nalu_hypre_PointRelaxSetup((pfmg_relax_data -> relax_data), A, b, x);
         break;
      case 2:
      case 3:
         nalu_hypre_RedBlackGSSetup((pfmg_relax_data -> rb_relax_data), A, b, x);
         break;
   }

   if (relax_type == 1)
   {
      nalu_hypre_PointRelaxSetWeight(pfmg_relax_data -> relax_data, jacobi_weight);
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_PFMGRelaxSetType( void  *pfmg_relax_vdata,
                        NALU_HYPRE_Int    relax_type       )
{
   nalu_hypre_PFMGRelaxData *pfmg_relax_data = (nalu_hypre_PFMGRelaxData *)pfmg_relax_vdata;
   void                *relax_data = (pfmg_relax_data -> relax_data);

   (pfmg_relax_data -> relax_type) = relax_type;

   switch (relax_type)
   {
      case 0: /* Jacobi */
      {
         nalu_hypre_Index  stride;
         nalu_hypre_Index  indices[1];

         nalu_hypre_PointRelaxSetWeight(relax_data, 1.0);
         nalu_hypre_PointRelaxSetNumPointsets(relax_data, 1);

         nalu_hypre_SetIndex3(stride, 1, 1, 1);
         nalu_hypre_SetIndex3(indices[0], 0, 0, 0);
         nalu_hypre_PointRelaxSetPointset(relax_data, 0, 1, stride, indices);
      }
      break;

      case 2: /* Red-Black Gauss-Seidel */
      case 3: /* Red-Black Gauss-Seidel (non-symmetric) */
         break;
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_PFMGRelaxSetJacobiWeight(void  *pfmg_relax_vdata,
                               NALU_HYPRE_Real weight)
{
   nalu_hypre_PFMGRelaxData *pfmg_relax_data = (nalu_hypre_PFMGRelaxData *)pfmg_relax_vdata;

   (pfmg_relax_data -> jacobi_weight)    = weight;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_PFMGRelaxSetPreRelax( void  *pfmg_relax_vdata )
{
   nalu_hypre_PFMGRelaxData *pfmg_relax_data = (nalu_hypre_PFMGRelaxData *)pfmg_relax_vdata;
   NALU_HYPRE_Int            relax_type = (pfmg_relax_data -> relax_type);

   switch (relax_type)
   {
      case 1: /* Weighted Jacobi */
      case 0: /* Jacobi */
         break;

      case 2: /* Red-Black Gauss-Seidel */
         nalu_hypre_RedBlackGSSetStartRed((pfmg_relax_data -> rb_relax_data));
         break;

      case 3: /* Red-Black Gauss-Seidel (non-symmetric) */
         nalu_hypre_RedBlackGSSetStartRed((pfmg_relax_data -> rb_relax_data));
         break;
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_PFMGRelaxSetPostRelax( void  *pfmg_relax_vdata )
{
   nalu_hypre_PFMGRelaxData *pfmg_relax_data = (nalu_hypre_PFMGRelaxData *)pfmg_relax_vdata;
   NALU_HYPRE_Int            relax_type = (pfmg_relax_data -> relax_type);

   switch (relax_type)
   {
      case 1: /* Weighted Jacobi */
      case 0: /* Jacobi */
         break;

      case 2: /* Red-Black Gauss-Seidel */
         nalu_hypre_RedBlackGSSetStartBlack((pfmg_relax_data -> rb_relax_data));
         break;

      case 3: /* Red-Black Gauss-Seidel (non-symmetric) */
         nalu_hypre_RedBlackGSSetStartRed((pfmg_relax_data -> rb_relax_data));
         break;
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_PFMGRelaxSetTol( void   *pfmg_relax_vdata,
                       NALU_HYPRE_Real  tol              )
{
   nalu_hypre_PFMGRelaxData *pfmg_relax_data = (nalu_hypre_PFMGRelaxData *)pfmg_relax_vdata;

   nalu_hypre_PointRelaxSetTol((pfmg_relax_data -> relax_data), tol);
   nalu_hypre_RedBlackGSSetTol((pfmg_relax_data -> rb_relax_data), tol);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_PFMGRelaxSetMaxIter( void  *pfmg_relax_vdata,
                           NALU_HYPRE_Int    max_iter         )
{
   nalu_hypre_PFMGRelaxData *pfmg_relax_data = (nalu_hypre_PFMGRelaxData *)pfmg_relax_vdata;

   nalu_hypre_PointRelaxSetMaxIter((pfmg_relax_data -> relax_data), max_iter);
   nalu_hypre_RedBlackGSSetMaxIter((pfmg_relax_data -> rb_relax_data), max_iter);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_PFMGRelaxSetZeroGuess( void  *pfmg_relax_vdata,
                             NALU_HYPRE_Int    zero_guess       )
{
   nalu_hypre_PFMGRelaxData *pfmg_relax_data = (nalu_hypre_PFMGRelaxData *)pfmg_relax_vdata;

   nalu_hypre_PointRelaxSetZeroGuess((pfmg_relax_data -> relax_data), zero_guess);
   nalu_hypre_RedBlackGSSetZeroGuess((pfmg_relax_data -> rb_relax_data), zero_guess);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_PFMGRelaxSetTempVec( void               *pfmg_relax_vdata,
                           nalu_hypre_StructVector *t                )
{
   nalu_hypre_PFMGRelaxData *pfmg_relax_data = (nalu_hypre_PFMGRelaxData *)pfmg_relax_vdata;

   nalu_hypre_PointRelaxSetTempVec((pfmg_relax_data -> relax_data), t);

   return nalu_hypre_error_flag;
}

