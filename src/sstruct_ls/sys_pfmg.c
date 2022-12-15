/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_nalu_hypre_sstruct_ls.h"
#include "sys_pfmg.h"

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

void *
nalu_hypre_SysPFMGCreate( MPI_Comm  comm )
{
   nalu_hypre_SysPFMGData *sys_pfmg_data;

   sys_pfmg_data = nalu_hypre_CTAlloc(nalu_hypre_SysPFMGData,  1, NALU_HYPRE_MEMORY_HOST);

   (sys_pfmg_data -> comm)       = comm;
   (sys_pfmg_data -> time_index) = nalu_hypre_InitializeTiming("SYS_PFMG");

   /* set defaults */
   (sys_pfmg_data -> tol)              = 1.0e-06;
   (sys_pfmg_data -> max_iter  )       = 200;
   (sys_pfmg_data -> rel_change)       = 0;
   (sys_pfmg_data -> zero_guess)       = 0;
   (sys_pfmg_data -> max_levels)       = 0;
   (sys_pfmg_data -> dxyz)[0]          = 0.0;
   (sys_pfmg_data -> dxyz)[1]          = 0.0;
   (sys_pfmg_data -> dxyz)[2]          = 0.0;
   (sys_pfmg_data -> relax_type)       = 1;       /* weighted Jacobi */
   (sys_pfmg_data -> jacobi_weight)    = 0.0;
   (sys_pfmg_data -> usr_jacobi_weight) = 0;
   (sys_pfmg_data -> num_pre_relax)    = 1;
   (sys_pfmg_data -> num_post_relax)   = 1;
   (sys_pfmg_data -> skip_relax)       = 1;
   (sys_pfmg_data -> logging)          = 0;
   (sys_pfmg_data -> print_level)      = 0;

   /* initialize */
   (sys_pfmg_data -> num_levels) = -1;

   return (void *) sys_pfmg_data;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SysPFMGDestroy( void *sys_pfmg_vdata )
{
   nalu_hypre_SysPFMGData *sys_pfmg_data = (nalu_hypre_SysPFMGData *)sys_pfmg_vdata;

   NALU_HYPRE_Int l;

   if (sys_pfmg_data)
   {
      if ((sys_pfmg_data -> logging) > 0)
      {
         nalu_hypre_TFree(sys_pfmg_data -> norms, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(sys_pfmg_data -> rel_norms, NALU_HYPRE_MEMORY_HOST);
      }

      if ((sys_pfmg_data -> num_levels) > -1)
      {
         for (l = 0; l < (sys_pfmg_data -> num_levels); l++)
         {
            nalu_hypre_SysPFMGRelaxDestroy(sys_pfmg_data -> relax_data_l[l]);
            nalu_hypre_SStructPMatvecDestroy(sys_pfmg_data -> matvec_data_l[l]);
         }
         for (l = 0; l < ((sys_pfmg_data -> num_levels) - 1); l++)
         {
            nalu_hypre_SysSemiRestrictDestroy(sys_pfmg_data -> restrict_data_l[l]);
            nalu_hypre_SysSemiInterpDestroy(sys_pfmg_data -> interp_data_l[l]);
         }
         nalu_hypre_TFree(sys_pfmg_data -> relax_data_l, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(sys_pfmg_data -> matvec_data_l, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(sys_pfmg_data -> restrict_data_l, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(sys_pfmg_data -> interp_data_l, NALU_HYPRE_MEMORY_HOST);

         nalu_hypre_SStructPVectorDestroy(sys_pfmg_data -> tx_l[0]);
         /*nalu_hypre_SStructPGridDestroy(sys_pfmg_data -> grid_l[0]);*/
         nalu_hypre_SStructPMatrixDestroy(sys_pfmg_data -> A_l[0]);
         nalu_hypre_SStructPVectorDestroy(sys_pfmg_data -> b_l[0]);
         nalu_hypre_SStructPVectorDestroy(sys_pfmg_data -> x_l[0]);
         for (l = 0; l < ((sys_pfmg_data -> num_levels) - 1); l++)
         {
            nalu_hypre_SStructPGridDestroy(sys_pfmg_data -> grid_l[l + 1]);
            nalu_hypre_SStructPGridDestroy(sys_pfmg_data -> P_grid_l[l + 1]);
            nalu_hypre_SStructPMatrixDestroy(sys_pfmg_data -> A_l[l + 1]);
            nalu_hypre_SStructPMatrixDestroy(sys_pfmg_data -> P_l[l]);
            nalu_hypre_SStructPVectorDestroy(sys_pfmg_data -> b_l[l + 1]);
            nalu_hypre_SStructPVectorDestroy(sys_pfmg_data -> x_l[l + 1]);
            nalu_hypre_SStructPVectorDestroy(sys_pfmg_data -> tx_l[l + 1]);
         }
         nalu_hypre_TFree(sys_pfmg_data -> data, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(sys_pfmg_data -> cdir_l, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(sys_pfmg_data -> active_l, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(sys_pfmg_data -> grid_l, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(sys_pfmg_data -> P_grid_l, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(sys_pfmg_data -> A_l, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(sys_pfmg_data -> P_l, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(sys_pfmg_data -> RT_l, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(sys_pfmg_data -> b_l, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(sys_pfmg_data -> x_l, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(sys_pfmg_data -> tx_l, NALU_HYPRE_MEMORY_HOST);
      }

      nalu_hypre_FinalizeTiming(sys_pfmg_data -> time_index);
      nalu_hypre_TFree(sys_pfmg_data, NALU_HYPRE_MEMORY_HOST);
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SysPFMGSetTol( void   *sys_pfmg_vdata,
                     NALU_HYPRE_Real  tol       )
{
   nalu_hypre_SysPFMGData *sys_pfmg_data = (nalu_hypre_SysPFMGData *)sys_pfmg_vdata;

   (sys_pfmg_data -> tol) = tol;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SysPFMGSetMaxIter( void *sys_pfmg_vdata,
                         NALU_HYPRE_Int   max_iter  )
{
   nalu_hypre_SysPFMGData *sys_pfmg_data = (nalu_hypre_SysPFMGData *)sys_pfmg_vdata;

   (sys_pfmg_data -> max_iter) = max_iter;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SysPFMGSetRelChange( void *sys_pfmg_vdata,
                           NALU_HYPRE_Int   rel_change  )
{
   nalu_hypre_SysPFMGData *sys_pfmg_data = (nalu_hypre_SysPFMGData *)sys_pfmg_vdata;

   (sys_pfmg_data -> rel_change) = rel_change;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SysPFMGSetZeroGuess( void *sys_pfmg_vdata,
                           NALU_HYPRE_Int   zero_guess )
{
   nalu_hypre_SysPFMGData *sys_pfmg_data = (nalu_hypre_SysPFMGData *)sys_pfmg_vdata;

   (sys_pfmg_data -> zero_guess) = zero_guess;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SysPFMGSetRelaxType( void *sys_pfmg_vdata,
                           NALU_HYPRE_Int   relax_type )
{
   nalu_hypre_SysPFMGData *sys_pfmg_data = (nalu_hypre_SysPFMGData *)sys_pfmg_vdata;

   (sys_pfmg_data -> relax_type) = relax_type;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/
NALU_HYPRE_Int
nalu_hypre_SysPFMGSetJacobiWeight( void  *sys_pfmg_vdata,
                              NALU_HYPRE_Real weight )
{
   nalu_hypre_SysPFMGData *sys_pfmg_data = (nalu_hypre_SysPFMGData *)sys_pfmg_vdata;

   (sys_pfmg_data -> jacobi_weight)    = weight;
   (sys_pfmg_data -> usr_jacobi_weight) = 1;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SysPFMGSetNumPreRelax( void *sys_pfmg_vdata,
                             NALU_HYPRE_Int   num_pre_relax )
{
   nalu_hypre_SysPFMGData *sys_pfmg_data = (nalu_hypre_SysPFMGData *)sys_pfmg_vdata;

   (sys_pfmg_data -> num_pre_relax) = num_pre_relax;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SysPFMGSetNumPostRelax( void *sys_pfmg_vdata,
                              NALU_HYPRE_Int   num_post_relax )
{
   nalu_hypre_SysPFMGData *sys_pfmg_data = (nalu_hypre_SysPFMGData *)sys_pfmg_vdata;

   (sys_pfmg_data -> num_post_relax) = num_post_relax;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SysPFMGSetSkipRelax( void *sys_pfmg_vdata,
                           NALU_HYPRE_Int  skip_relax )
{
   nalu_hypre_SysPFMGData *sys_pfmg_data = (nalu_hypre_SysPFMGData *)sys_pfmg_vdata;

   (sys_pfmg_data -> skip_relax) = skip_relax;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SysPFMGSetDxyz( void   *sys_pfmg_vdata,
                      NALU_HYPRE_Real *dxyz       )
{
   nalu_hypre_SysPFMGData *sys_pfmg_data = (nalu_hypre_SysPFMGData *)sys_pfmg_vdata;

   (sys_pfmg_data -> dxyz[0]) = dxyz[0];
   (sys_pfmg_data -> dxyz[1]) = dxyz[1];
   (sys_pfmg_data -> dxyz[2]) = dxyz[2];

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SysPFMGSetLogging( void *sys_pfmg_vdata,
                         NALU_HYPRE_Int   logging)
{
   nalu_hypre_SysPFMGData *sys_pfmg_data = (nalu_hypre_SysPFMGData *)sys_pfmg_vdata;

   (sys_pfmg_data -> logging) = logging;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SysPFMGSetPrintLevel( void *sys_pfmg_vdata,
                            NALU_HYPRE_Int   print_level)
{
   nalu_hypre_SysPFMGData *sys_pfmg_data = (nalu_hypre_SysPFMGData *)sys_pfmg_vdata;

   (sys_pfmg_data -> print_level) = print_level;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SysPFMGGetNumIterations( void *sys_pfmg_vdata,
                               NALU_HYPRE_Int  *num_iterations )
{
   nalu_hypre_SysPFMGData *sys_pfmg_data = (nalu_hypre_SysPFMGData *)sys_pfmg_vdata;

   *num_iterations = (sys_pfmg_data -> num_iterations);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SysPFMGPrintLogging( void *sys_pfmg_vdata,
                           NALU_HYPRE_Int   myid)
{
   nalu_hypre_SysPFMGData *sys_pfmg_data = (nalu_hypre_SysPFMGData *)sys_pfmg_vdata;
   NALU_HYPRE_Int          i;
   NALU_HYPRE_Int          num_iterations  = (sys_pfmg_data -> num_iterations);
   NALU_HYPRE_Int          logging   = (sys_pfmg_data -> logging);
   NALU_HYPRE_Int          print_level   = (sys_pfmg_data -> print_level);
   NALU_HYPRE_Real        *norms     = (sys_pfmg_data -> norms);
   NALU_HYPRE_Real        *rel_norms = (sys_pfmg_data -> rel_norms);

   if (myid == 0)
   {
      if (print_level > 0 )
      {
         if (logging > 0)
         {
            for (i = 0; i < num_iterations; i++)
            {
               nalu_hypre_printf("Residual norm[%d] = %e   ", i, norms[i]);
               nalu_hypre_printf("Relative residual norm[%d] = %e\n", i, rel_norms[i]);
            }
         }
      }
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SysPFMGGetFinalRelativeResidualNorm( void   *sys_pfmg_vdata,
                                           NALU_HYPRE_Real *relative_residual_norm )
{
   nalu_hypre_SysPFMGData *sys_pfmg_data = (nalu_hypre_SysPFMGData *)sys_pfmg_vdata;

   NALU_HYPRE_Int          max_iter        = (sys_pfmg_data -> max_iter);
   NALU_HYPRE_Int          num_iterations  = (sys_pfmg_data -> num_iterations);
   NALU_HYPRE_Int          logging         = (sys_pfmg_data -> logging);
   NALU_HYPRE_Real        *rel_norms       = (sys_pfmg_data -> rel_norms);

   if (logging > 0)
   {
      if (max_iter == 0)
      {
         nalu_hypre_error_in_arg(1);
      }
      else if (num_iterations == max_iter)
      {
         *relative_residual_norm = rel_norms[num_iterations - 1];
      }
      else
      {
         *relative_residual_norm = rel_norms[num_iterations];
      }
   }

   return nalu_hypre_error_flag;
}


