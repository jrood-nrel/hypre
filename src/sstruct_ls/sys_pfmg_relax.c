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
   void                   *relax_data;
   NALU_HYPRE_Int               relax_type;
   NALU_HYPRE_Real              jacobi_weight;

} nalu_hypre_SysPFMGRelaxData;

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

void *
nalu_hypre_SysPFMGRelaxCreate( MPI_Comm  comm )
{
   nalu_hypre_SysPFMGRelaxData *sys_pfmg_relax_data;

   sys_pfmg_relax_data = nalu_hypre_CTAlloc(nalu_hypre_SysPFMGRelaxData,  1, NALU_HYPRE_MEMORY_HOST);
   (sys_pfmg_relax_data -> relax_data) = nalu_hypre_NodeRelaxCreate(comm);
   (sys_pfmg_relax_data -> relax_type) = 0;        /* Weighted Jacobi */

   return (void *) sys_pfmg_relax_data;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SysPFMGRelaxDestroy( void *sys_pfmg_relax_vdata )
{
   nalu_hypre_SysPFMGRelaxData *sys_pfmg_relax_data = (nalu_hypre_SysPFMGRelaxData *)sys_pfmg_relax_vdata;

   if (sys_pfmg_relax_data)
   {
      nalu_hypre_NodeRelaxDestroy(sys_pfmg_relax_data -> relax_data);
      nalu_hypre_TFree(sys_pfmg_relax_data, NALU_HYPRE_MEMORY_HOST);
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SysPFMGRelax( void                 *sys_pfmg_relax_vdata,
                    nalu_hypre_SStructPMatrix *A,
                    nalu_hypre_SStructPVector *b,
                    nalu_hypre_SStructPVector *x                )
{
   nalu_hypre_SysPFMGRelaxData *sys_pfmg_relax_data = (nalu_hypre_SysPFMGRelaxData *)sys_pfmg_relax_vdata;

   nalu_hypre_NodeRelax((sys_pfmg_relax_data -> relax_data), A, b, x);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SysPFMGRelaxSetup( void                 *sys_pfmg_relax_vdata,
                         nalu_hypre_SStructPMatrix *A,
                         nalu_hypre_SStructPVector *b,
                         nalu_hypre_SStructPVector *x                )
{
   nalu_hypre_SysPFMGRelaxData *sys_pfmg_relax_data = (nalu_hypre_SysPFMGRelaxData *)sys_pfmg_relax_vdata;
   void                   *relax_data    = (sys_pfmg_relax_data -> relax_data);
   NALU_HYPRE_Int               relax_type    = (sys_pfmg_relax_data -> relax_type);
   NALU_HYPRE_Real              jacobi_weight = (sys_pfmg_relax_data -> jacobi_weight);

   if (relax_type == 1)
   {
      nalu_hypre_NodeRelaxSetWeight(relax_data, jacobi_weight);
   }

   nalu_hypre_NodeRelaxSetup((sys_pfmg_relax_data -> relax_data), A, b, x);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SysPFMGRelaxSetType( void  *sys_pfmg_relax_vdata,
                           NALU_HYPRE_Int    relax_type       )
{
   nalu_hypre_SysPFMGRelaxData *sys_pfmg_relax_data = (nalu_hypre_SysPFMGRelaxData *)sys_pfmg_relax_vdata;
   void                   *relax_data = (sys_pfmg_relax_data -> relax_data);

   (sys_pfmg_relax_data -> relax_type) = relax_type;

   switch (relax_type)
   {
      case 0: /* Jacobi */
      {
         nalu_hypre_Index  stride;
         nalu_hypre_Index  indices[1];

         nalu_hypre_NodeRelaxSetWeight(relax_data, 1.0);
         nalu_hypre_NodeRelaxSetNumNodesets(relax_data, 1);

         nalu_hypre_SetIndex3(stride, 1, 1, 1);
         nalu_hypre_SetIndex3(indices[0], 0, 0, 0);
         nalu_hypre_NodeRelaxSetNodeset(relax_data, 0, 1, stride, indices);
      }
      break;

      case 2: /* Red-Black Gauss-Seidel */
      {
         nalu_hypre_Index  stride;
         nalu_hypre_Index  indices[4];

         nalu_hypre_NodeRelaxSetNumNodesets(relax_data, 2);

         nalu_hypre_SetIndex3(stride, 2, 2, 2);

         /* define red points (point set 0) */
         nalu_hypre_SetIndex3(indices[0], 1, 0, 0);
         nalu_hypre_SetIndex3(indices[1], 0, 1, 0);
         nalu_hypre_SetIndex3(indices[2], 0, 0, 1);
         nalu_hypre_SetIndex3(indices[3], 1, 1, 1);
         nalu_hypre_NodeRelaxSetNodeset(relax_data, 0, 4, stride, indices);

         /* define black points (point set 1) */
         nalu_hypre_SetIndex3(indices[0], 0, 0, 0);
         nalu_hypre_SetIndex3(indices[1], 1, 1, 0);
         nalu_hypre_SetIndex3(indices[2], 1, 0, 1);
         nalu_hypre_SetIndex3(indices[3], 0, 1, 1);
         nalu_hypre_NodeRelaxSetNodeset(relax_data, 1, 4, stride, indices);
      }
      break;
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SysPFMGRelaxSetJacobiWeight(void  *sys_pfmg_relax_vdata,
                                  NALU_HYPRE_Real weight)
{
   nalu_hypre_SysPFMGRelaxData *sys_pfmg_relax_data = (nalu_hypre_SysPFMGRelaxData *)sys_pfmg_relax_vdata;

   (sys_pfmg_relax_data -> jacobi_weight)    = weight;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SysPFMGRelaxSetPreRelax( void  *sys_pfmg_relax_vdata )
{
   nalu_hypre_SysPFMGRelaxData *sys_pfmg_relax_data = (nalu_hypre_SysPFMGRelaxData *)sys_pfmg_relax_vdata;
   void                   *relax_data = (sys_pfmg_relax_data -> relax_data);
   NALU_HYPRE_Int               relax_type = (sys_pfmg_relax_data -> relax_type);

   switch (relax_type)
   {
      case 1: /* Weighted Jacobi */
      case 0: /* Jacobi */
         break;

      case 2: /* Red-Black Gauss-Seidel */
      {
         nalu_hypre_NodeRelaxSetNodesetRank(relax_data, 0, 0);
         nalu_hypre_NodeRelaxSetNodesetRank(relax_data, 1, 1);
      }
      break;
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SysPFMGRelaxSetPostRelax( void  *sys_pfmg_relax_vdata )
{
   nalu_hypre_SysPFMGRelaxData *sys_pfmg_relax_data = (nalu_hypre_SysPFMGRelaxData *)sys_pfmg_relax_vdata;
   void                   *relax_data = (sys_pfmg_relax_data -> relax_data);
   NALU_HYPRE_Int               relax_type = (sys_pfmg_relax_data -> relax_type);

   switch (relax_type)
   {
      case 1: /* Weighted Jacobi */
      case 0: /* Jacobi */
         break;

      case 2: /* Red-Black Gauss-Seidel */
      {
         nalu_hypre_NodeRelaxSetNodesetRank(relax_data, 0, 1);
         nalu_hypre_NodeRelaxSetNodesetRank(relax_data, 1, 0);
      }
      break;
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SysPFMGRelaxSetTol( void   *sys_pfmg_relax_vdata,
                          NALU_HYPRE_Real  tol              )
{
   nalu_hypre_SysPFMGRelaxData *sys_pfmg_relax_data = (nalu_hypre_SysPFMGRelaxData *)sys_pfmg_relax_vdata;

   nalu_hypre_NodeRelaxSetTol((sys_pfmg_relax_data -> relax_data), tol);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SysPFMGRelaxSetMaxIter( void  *sys_pfmg_relax_vdata,
                              NALU_HYPRE_Int    max_iter         )
{
   nalu_hypre_SysPFMGRelaxData *sys_pfmg_relax_data = (nalu_hypre_SysPFMGRelaxData *)sys_pfmg_relax_vdata;

   nalu_hypre_NodeRelaxSetMaxIter((sys_pfmg_relax_data -> relax_data), max_iter);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SysPFMGRelaxSetZeroGuess( void  *sys_pfmg_relax_vdata,
                                NALU_HYPRE_Int    zero_guess       )
{
   nalu_hypre_SysPFMGRelaxData *sys_pfmg_relax_data = (nalu_hypre_SysPFMGRelaxData *)sys_pfmg_relax_vdata;

   nalu_hypre_NodeRelaxSetZeroGuess((sys_pfmg_relax_data -> relax_data), zero_guess);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SysPFMGRelaxSetTempVec( void               *sys_pfmg_relax_vdata,
                              nalu_hypre_SStructPVector *t                )
{
   nalu_hypre_SysPFMGRelaxData *sys_pfmg_relax_data = (nalu_hypre_SysPFMGRelaxData *)sys_pfmg_relax_vdata;

   nalu_hypre_NodeRelaxSetTempVec((sys_pfmg_relax_data -> relax_data), t);

   return nalu_hypre_error_flag;
}

