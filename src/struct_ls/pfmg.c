/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_nalu_hypre_struct_ls.h"
#include "pfmg.h"

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

void *
nalu_hypre_PFMGCreate( MPI_Comm  comm )
{
   nalu_hypre_PFMGData *pfmg_data;

   pfmg_data = nalu_hypre_CTAlloc(nalu_hypre_PFMGData,  1, NALU_HYPRE_MEMORY_HOST);

   (pfmg_data -> comm)       = comm;
   (pfmg_data -> time_index) = nalu_hypre_InitializeTiming("PFMG");

   /* set defaults */
   (pfmg_data -> tol)               = 1.0e-06;
   (pfmg_data -> max_iter)          = 200;
   (pfmg_data -> rel_change)        = 0;
   (pfmg_data -> zero_guess)        = 0;
   (pfmg_data -> max_levels)        = 0;
   (pfmg_data -> dxyz)[0]           = 0.0;
   (pfmg_data -> dxyz)[1]           = 0.0;
   (pfmg_data -> dxyz)[2]           = 0.0;
   (pfmg_data -> relax_type)        = 1;       /* weighted Jacobi */
   (pfmg_data -> jacobi_weight)     = 0.0;
   (pfmg_data -> usr_jacobi_weight) = 0;    /* no user Jacobi weight */
   (pfmg_data -> rap_type)          = 0;
   (pfmg_data -> num_pre_relax)     = 1;
   (pfmg_data -> num_post_relax)    = 1;
   (pfmg_data -> skip_relax)        = 1;
   (pfmg_data -> logging)           = 0;
   (pfmg_data -> print_level)       = 0;

   (pfmg_data -> memory_location)   = nalu_hypre_HandleMemoryLocation(nalu_hypre_handle());

   /* initialize */
   (pfmg_data -> num_levels)  = -1;

   return (void *) pfmg_data;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_PFMGDestroy( void *pfmg_vdata )
{
   nalu_hypre_PFMGData *pfmg_data = (nalu_hypre_PFMGData *)pfmg_vdata;

   NALU_HYPRE_Int l;

   NALU_HYPRE_ANNOTATE_FUNC_BEGIN;

   if (pfmg_data)
   {
      if ((pfmg_data -> logging) > 0)
      {
         nalu_hypre_TFree(pfmg_data -> norms, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(pfmg_data -> rel_norms, NALU_HYPRE_MEMORY_HOST);
      }

      NALU_HYPRE_MemoryLocation memory_location = pfmg_data -> memory_location;

      if ((pfmg_data -> num_levels) > -1)
      {
         for (l = 0; l < (pfmg_data -> num_levels); l++)
         {
            if (pfmg_data -> active_l[l])
            {
               nalu_hypre_PFMGRelaxDestroy(pfmg_data -> relax_data_l[l]);
            }
            nalu_hypre_StructMatvecDestroy(pfmg_data -> matvec_data_l[l]);
         }
         for (l = 0; l < ((pfmg_data -> num_levels) - 1); l++)
         {
            nalu_hypre_SemiRestrictDestroy(pfmg_data -> restrict_data_l[l]);
            nalu_hypre_SemiInterpDestroy(pfmg_data -> interp_data_l[l]);
         }
         nalu_hypre_TFree(pfmg_data -> relax_data_l, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(pfmg_data -> matvec_data_l, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(pfmg_data -> restrict_data_l, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(pfmg_data -> interp_data_l, NALU_HYPRE_MEMORY_HOST);

         nalu_hypre_StructVectorDestroy(pfmg_data -> tx_l[0]);
         nalu_hypre_StructGridDestroy(pfmg_data -> grid_l[0]);
         nalu_hypre_StructMatrixDestroy(pfmg_data -> A_l[0]);
         nalu_hypre_StructVectorDestroy(pfmg_data -> b_l[0]);
         nalu_hypre_StructVectorDestroy(pfmg_data -> x_l[0]);
         for (l = 0; l < ((pfmg_data -> num_levels) - 1); l++)
         {
            nalu_hypre_StructGridDestroy(pfmg_data -> grid_l[l + 1]);
            nalu_hypre_StructGridDestroy(pfmg_data -> P_grid_l[l + 1]);
            nalu_hypre_StructMatrixDestroy(pfmg_data -> A_l[l + 1]);
            nalu_hypre_StructMatrixDestroy(pfmg_data -> P_l[l]);
            nalu_hypre_StructVectorDestroy(pfmg_data -> b_l[l + 1]);
            nalu_hypre_StructVectorDestroy(pfmg_data -> x_l[l + 1]);
            nalu_hypre_StructVectorDestroy(pfmg_data -> tx_l[l + 1]);
         }

         nalu_hypre_TFree(pfmg_data -> data, memory_location);
         nalu_hypre_TFree(pfmg_data -> data_const, NALU_HYPRE_MEMORY_HOST);

         nalu_hypre_TFree(pfmg_data -> cdir_l, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(pfmg_data -> active_l, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(pfmg_data -> grid_l, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(pfmg_data -> P_grid_l, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(pfmg_data -> A_l, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(pfmg_data -> P_l, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(pfmg_data -> RT_l, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(pfmg_data -> b_l, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(pfmg_data -> x_l, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(pfmg_data -> tx_l, NALU_HYPRE_MEMORY_HOST);
      }

      nalu_hypre_FinalizeTiming(pfmg_data -> time_index);
      nalu_hypre_TFree(pfmg_data, NALU_HYPRE_MEMORY_HOST);
   }

   NALU_HYPRE_ANNOTATE_FUNC_END;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_PFMGSetTol( void   *pfmg_vdata,
                  NALU_HYPRE_Real  tol       )
{
   nalu_hypre_PFMGData *pfmg_data = (nalu_hypre_PFMGData *)pfmg_vdata;

   (pfmg_data -> tol) = tol;

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_PFMGGetTol( void   *pfmg_vdata,
                  NALU_HYPRE_Real *tol       )
{
   nalu_hypre_PFMGData *pfmg_data = (nalu_hypre_PFMGData *)pfmg_vdata;

   *tol = (pfmg_data -> tol);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_PFMGSetMaxIter( void *pfmg_vdata,
                      NALU_HYPRE_Int   max_iter  )
{
   nalu_hypre_PFMGData *pfmg_data = (nalu_hypre_PFMGData *)pfmg_vdata;

   (pfmg_data -> max_iter) = max_iter;

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_PFMGGetMaxIter( void *pfmg_vdata,
                      NALU_HYPRE_Int * max_iter  )
{
   nalu_hypre_PFMGData *pfmg_data = (nalu_hypre_PFMGData *)pfmg_vdata;

   *max_iter = (pfmg_data -> max_iter);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_PFMGSetMaxLevels( void *pfmg_vdata,
                        NALU_HYPRE_Int   max_levels  )
{
   nalu_hypre_PFMGData *pfmg_data = (nalu_hypre_PFMGData *)pfmg_vdata;

   (pfmg_data -> max_levels) = max_levels;

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_PFMGGetMaxLevels( void *pfmg_vdata,
                        NALU_HYPRE_Int * max_levels  )
{
   nalu_hypre_PFMGData *pfmg_data = (nalu_hypre_PFMGData *)pfmg_vdata;

   *max_levels = (pfmg_data -> max_levels);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_PFMGSetRelChange( void *pfmg_vdata,
                        NALU_HYPRE_Int   rel_change  )
{
   nalu_hypre_PFMGData *pfmg_data = (nalu_hypre_PFMGData *)pfmg_vdata;

   (pfmg_data -> rel_change) = rel_change;

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_PFMGGetRelChange( void *pfmg_vdata,
                        NALU_HYPRE_Int * rel_change  )
{
   nalu_hypre_PFMGData *pfmg_data = (nalu_hypre_PFMGData *)pfmg_vdata;

   *rel_change = (pfmg_data -> rel_change);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_PFMGSetZeroGuess( void *pfmg_vdata,
                        NALU_HYPRE_Int   zero_guess )
{
   nalu_hypre_PFMGData *pfmg_data = (nalu_hypre_PFMGData *)pfmg_vdata;

   (pfmg_data -> zero_guess) = zero_guess;

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_PFMGGetZeroGuess( void *pfmg_vdata,
                        NALU_HYPRE_Int * zero_guess )
{
   nalu_hypre_PFMGData *pfmg_data = (nalu_hypre_PFMGData *)pfmg_vdata;

   *zero_guess = (pfmg_data -> zero_guess);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_PFMGSetRelaxType( void *pfmg_vdata,
                        NALU_HYPRE_Int   relax_type )
{
   nalu_hypre_PFMGData *pfmg_data = (nalu_hypre_PFMGData *)pfmg_vdata;

   (pfmg_data -> relax_type) = relax_type;

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_PFMGGetRelaxType( void *pfmg_vdata,
                        NALU_HYPRE_Int * relax_type )
{
   nalu_hypre_PFMGData *pfmg_data = (nalu_hypre_PFMGData *)pfmg_vdata;

   *relax_type = (pfmg_data -> relax_type);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/
NALU_HYPRE_Int
nalu_hypre_PFMGSetJacobiWeight( void  *pfmg_vdata,
                           NALU_HYPRE_Real weight )
{
   nalu_hypre_PFMGData *pfmg_data = (nalu_hypre_PFMGData *)pfmg_vdata;

   (pfmg_data -> jacobi_weight)    = weight;
   (pfmg_data -> usr_jacobi_weight) = 1;

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_PFMGGetJacobiWeight( void  *pfmg_vdata,
                           NALU_HYPRE_Real *weight )
{
   nalu_hypre_PFMGData *pfmg_data = (nalu_hypre_PFMGData *)pfmg_vdata;

   *weight = (pfmg_data -> jacobi_weight);

   return nalu_hypre_error_flag;
}


/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_PFMGSetRAPType( void *pfmg_vdata,
                      NALU_HYPRE_Int   rap_type )
{
   nalu_hypre_PFMGData *pfmg_data = (nalu_hypre_PFMGData *)pfmg_vdata;

   (pfmg_data -> rap_type) = rap_type;

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_PFMGGetRAPType( void *pfmg_vdata,
                      NALU_HYPRE_Int * rap_type )
{
   nalu_hypre_PFMGData *pfmg_data = (nalu_hypre_PFMGData *)pfmg_vdata;

   *rap_type = (pfmg_data -> rap_type);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_PFMGSetNumPreRelax( void *pfmg_vdata,
                          NALU_HYPRE_Int   num_pre_relax )
{
   nalu_hypre_PFMGData *pfmg_data = (nalu_hypre_PFMGData *)pfmg_vdata;

   (pfmg_data -> num_pre_relax) = num_pre_relax;

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_PFMGGetNumPreRelax( void *pfmg_vdata,
                          NALU_HYPRE_Int * num_pre_relax )
{
   nalu_hypre_PFMGData *pfmg_data = (nalu_hypre_PFMGData *)pfmg_vdata;

   *num_pre_relax = (pfmg_data -> num_pre_relax);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_PFMGSetNumPostRelax( void *pfmg_vdata,
                           NALU_HYPRE_Int   num_post_relax )
{
   nalu_hypre_PFMGData *pfmg_data = (nalu_hypre_PFMGData *)pfmg_vdata;

   (pfmg_data -> num_post_relax) = num_post_relax;

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_PFMGGetNumPostRelax( void *pfmg_vdata,
                           NALU_HYPRE_Int * num_post_relax )
{
   nalu_hypre_PFMGData *pfmg_data = (nalu_hypre_PFMGData *)pfmg_vdata;

   *num_post_relax = (pfmg_data -> num_post_relax);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_PFMGSetSkipRelax( void *pfmg_vdata,
                        NALU_HYPRE_Int  skip_relax )
{
   nalu_hypre_PFMGData *pfmg_data = (nalu_hypre_PFMGData *)pfmg_vdata;

   (pfmg_data -> skip_relax) = skip_relax;

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_PFMGGetSkipRelax( void *pfmg_vdata,
                        NALU_HYPRE_Int *skip_relax )
{
   nalu_hypre_PFMGData *pfmg_data = (nalu_hypre_PFMGData *)pfmg_vdata;

   *skip_relax = (pfmg_data -> skip_relax);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_PFMGSetDxyz( void   *pfmg_vdata,
                   NALU_HYPRE_Real *dxyz       )
{
   nalu_hypre_PFMGData *pfmg_data = (nalu_hypre_PFMGData *)pfmg_vdata;

   (pfmg_data -> dxyz[0]) = dxyz[0];
   (pfmg_data -> dxyz[1]) = dxyz[1];
   (pfmg_data -> dxyz[2]) = dxyz[2];

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_PFMGSetLogging( void *pfmg_vdata,
                      NALU_HYPRE_Int   logging)
{
   nalu_hypre_PFMGData *pfmg_data = (nalu_hypre_PFMGData *)pfmg_vdata;

   (pfmg_data -> logging) = logging;

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_PFMGGetLogging( void *pfmg_vdata,
                      NALU_HYPRE_Int * logging)
{
   nalu_hypre_PFMGData *pfmg_data = (nalu_hypre_PFMGData *)pfmg_vdata;

   *logging = (pfmg_data -> logging);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_PFMGSetPrintLevel( void *pfmg_vdata,
                         NALU_HYPRE_Int   print_level)
{
   nalu_hypre_PFMGData *pfmg_data = (nalu_hypre_PFMGData *)pfmg_vdata;

   (pfmg_data -> print_level) = print_level;

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_PFMGGetPrintLevel( void *pfmg_vdata,
                         NALU_HYPRE_Int * print_level)
{
   nalu_hypre_PFMGData *pfmg_data = (nalu_hypre_PFMGData *)pfmg_vdata;

   *print_level = (pfmg_data -> print_level);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_PFMGGetNumIterations( void *pfmg_vdata,
                            NALU_HYPRE_Int  *num_iterations )
{
   nalu_hypre_PFMGData *pfmg_data = (nalu_hypre_PFMGData *)pfmg_vdata;

   *num_iterations = (pfmg_data -> num_iterations);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_PFMGPrintLogging( void *pfmg_vdata,
                        NALU_HYPRE_Int   myid)
{
   nalu_hypre_PFMGData *pfmg_data = (nalu_hypre_PFMGData *)pfmg_vdata;
   NALU_HYPRE_Int       i;
   NALU_HYPRE_Int       num_iterations  = (pfmg_data -> num_iterations);
   NALU_HYPRE_Int       logging   = (pfmg_data -> logging);
   NALU_HYPRE_Int    print_level  = (pfmg_data -> print_level);
   NALU_HYPRE_Real     *norms     = (pfmg_data -> norms);
   NALU_HYPRE_Real     *rel_norms = (pfmg_data -> rel_norms);

   if (myid == 0)
   {
      if (print_level > 0)
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
nalu_hypre_PFMGGetFinalRelativeResidualNorm( void   *pfmg_vdata,
                                        NALU_HYPRE_Real *relative_residual_norm )
{
   nalu_hypre_PFMGData *pfmg_data = (nalu_hypre_PFMGData *)pfmg_vdata;

   NALU_HYPRE_Int       max_iter        = (pfmg_data -> max_iter);
   NALU_HYPRE_Int       num_iterations  = (pfmg_data -> num_iterations);
   NALU_HYPRE_Int       logging         = (pfmg_data -> logging);
   NALU_HYPRE_Real     *rel_norms       = (pfmg_data -> rel_norms);

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

#if 0 //defined(NALU_HYPRE_USING_CUDA) || defined(NALU_HYPRE_USING_HIP)
NALU_HYPRE_Int
nalu_hypre_PFMGSetDeviceLevel( void *pfmg_vdata,
                          NALU_HYPRE_Int   device_level  )
{
   nalu_hypre_PFMGData *pfmg_data = (nalu_hypre_PFMGData *)pfmg_vdata;

   (pfmg_data -> devicelevel) = device_level;

   return nalu_hypre_error_flag;
}
#endif
