/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_nalu_hypre_struct_ls.h"
#include "_nalu_hypre_struct_mv.hpp"
#include "smg.h"

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

void *
nalu_hypre_SMGCreate( MPI_Comm  comm )
{
   nalu_hypre_SMGData *smg_data;

   smg_data = nalu_hypre_CTAlloc(nalu_hypre_SMGData, 1, NALU_HYPRE_MEMORY_HOST);

   (smg_data -> comm)        = comm;
   (smg_data -> time_index)  = nalu_hypre_InitializeTiming("SMG");

   /* set defaults */
   (smg_data -> memory_use) = 0;
   (smg_data -> tol)        = 1.0e-06;
   (smg_data -> max_iter)   = 200;
   (smg_data -> rel_change) = 0;
   (smg_data -> zero_guess) = 0;
   (smg_data -> max_levels) = 0;
   (smg_data -> num_pre_relax)  = 1;
   (smg_data -> num_post_relax) = 1;
   (smg_data -> cdir) = 2;
   nalu_hypre_SetIndex3((smg_data -> base_index), 0, 0, 0);
   nalu_hypre_SetIndex3((smg_data -> base_stride), 1, 1, 1);
   (smg_data -> logging) = 0;
   (smg_data -> print_level) = 0;

   (smg_data -> memory_location) = nalu_hypre_HandleMemoryLocation(nalu_hypre_handle());

   /* initialize */
   (smg_data -> num_levels) = -1;

   return (void *) smg_data;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SMGDestroy( void *smg_vdata )
{
   nalu_hypre_SMGData *smg_data = (nalu_hypre_SMGData *)smg_vdata;

   NALU_HYPRE_Int l;

   NALU_HYPRE_ANNOTATE_FUNC_BEGIN;

   if (smg_data)
   {
      if ((smg_data -> logging) > 0)
      {
         nalu_hypre_TFree(smg_data -> norms, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(smg_data -> rel_norms, NALU_HYPRE_MEMORY_HOST);
      }

      NALU_HYPRE_MemoryLocation memory_location = smg_data -> memory_location;

      if ((smg_data -> num_levels) > -1)
      {
         for (l = 0; l < ((smg_data -> num_levels) - 1); l++)
         {
            nalu_hypre_SMGRelaxDestroy(smg_data -> relax_data_l[l]);
            nalu_hypre_SMGResidualDestroy(smg_data -> residual_data_l[l]);
            nalu_hypre_SemiRestrictDestroy(smg_data -> restrict_data_l[l]);
            nalu_hypre_SemiInterpDestroy(smg_data -> interp_data_l[l]);
         }
         nalu_hypre_SMGRelaxDestroy(smg_data -> relax_data_l[l]);
         if (l == 0)
         {
            nalu_hypre_SMGResidualDestroy(smg_data -> residual_data_l[l]);
         }
         nalu_hypre_TFree(smg_data -> relax_data_l, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(smg_data -> residual_data_l, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(smg_data -> restrict_data_l, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(smg_data -> interp_data_l, NALU_HYPRE_MEMORY_HOST);

         nalu_hypre_StructVectorDestroy(smg_data -> tb_l[0]);
         nalu_hypre_StructVectorDestroy(smg_data -> tx_l[0]);
         nalu_hypre_StructGridDestroy(smg_data -> grid_l[0]);
         nalu_hypre_StructMatrixDestroy(smg_data -> A_l[0]);
         nalu_hypre_StructVectorDestroy(smg_data -> b_l[0]);
         nalu_hypre_StructVectorDestroy(smg_data -> x_l[0]);
         for (l = 0; l < ((smg_data -> num_levels) - 1); l++)
         {
            nalu_hypre_StructGridDestroy(smg_data -> grid_l[l + 1]);
            nalu_hypre_StructGridDestroy(smg_data -> PT_grid_l[l + 1]);
            nalu_hypre_StructMatrixDestroy(smg_data -> A_l[l + 1]);
            if (smg_data -> PT_l[l] == smg_data -> R_l[l])
            {
               nalu_hypre_StructMatrixDestroy(smg_data -> PT_l[l]);
            }
            else
            {
               nalu_hypre_StructMatrixDestroy(smg_data -> PT_l[l]);
               nalu_hypre_StructMatrixDestroy(smg_data -> R_l[l]);
            }
            nalu_hypre_StructVectorDestroy(smg_data -> b_l[l + 1]);
            nalu_hypre_StructVectorDestroy(smg_data -> x_l[l + 1]);
            nalu_hypre_StructVectorDestroy(smg_data -> tb_l[l + 1]);
            nalu_hypre_StructVectorDestroy(smg_data -> tx_l[l + 1]);
         }
         nalu_hypre_TFree(smg_data -> data, memory_location);
         nalu_hypre_TFree(smg_data -> grid_l, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(smg_data -> PT_grid_l, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(smg_data -> A_l, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(smg_data -> PT_l, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(smg_data -> R_l, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(smg_data -> b_l, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(smg_data -> x_l, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(smg_data -> tb_l, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(smg_data -> tx_l, NALU_HYPRE_MEMORY_HOST);
      }

      nalu_hypre_FinalizeTiming(smg_data -> time_index);
      nalu_hypre_TFree(smg_data, NALU_HYPRE_MEMORY_HOST);
   }

   NALU_HYPRE_ANNOTATE_FUNC_END;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SMGSetMemoryUse( void *smg_vdata,
                       NALU_HYPRE_Int   memory_use )
{
   nalu_hypre_SMGData *smg_data = (nalu_hypre_SMGData *)smg_vdata;

   (smg_data -> memory_use) = memory_use;

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_SMGGetMemoryUse( void *smg_vdata,
                       NALU_HYPRE_Int * memory_use )
{
   nalu_hypre_SMGData *smg_data = (nalu_hypre_SMGData *)smg_vdata;

   *memory_use = (smg_data -> memory_use);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SMGSetTol( void   *smg_vdata,
                 NALU_HYPRE_Real  tol       )
{
   nalu_hypre_SMGData *smg_data = (nalu_hypre_SMGData *)smg_vdata;

   (smg_data -> tol) = tol;

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_SMGGetTol( void   *smg_vdata,
                 NALU_HYPRE_Real *tol       )
{
   nalu_hypre_SMGData *smg_data = (nalu_hypre_SMGData *)smg_vdata;

   *tol = (smg_data -> tol);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SMGSetMaxIter( void *smg_vdata,
                     NALU_HYPRE_Int   max_iter  )
{
   nalu_hypre_SMGData *smg_data = (nalu_hypre_SMGData *)smg_vdata;

   (smg_data -> max_iter) = max_iter;

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_SMGGetMaxIter( void *smg_vdata,
                     NALU_HYPRE_Int * max_iter  )
{
   nalu_hypre_SMGData *smg_data = (nalu_hypre_SMGData *)smg_vdata;

   *max_iter = (smg_data -> max_iter);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SMGSetRelChange( void *smg_vdata,
                       NALU_HYPRE_Int   rel_change  )
{
   nalu_hypre_SMGData *smg_data = (nalu_hypre_SMGData *)smg_vdata;

   (smg_data -> rel_change) = rel_change;

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_SMGGetRelChange( void *smg_vdata,
                       NALU_HYPRE_Int * rel_change  )
{
   nalu_hypre_SMGData *smg_data = (nalu_hypre_SMGData *)smg_vdata;

   *rel_change = (smg_data -> rel_change);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SMGSetZeroGuess( void *smg_vdata,
                       NALU_HYPRE_Int   zero_guess )
{
   nalu_hypre_SMGData *smg_data = (nalu_hypre_SMGData *)smg_vdata;

   (smg_data -> zero_guess) = zero_guess;

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_SMGGetZeroGuess( void *smg_vdata,
                       NALU_HYPRE_Int * zero_guess )
{
   nalu_hypre_SMGData *smg_data = (nalu_hypre_SMGData *)smg_vdata;

   *zero_guess = (smg_data -> zero_guess);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Note that we require at least 1 pre-relax sweep.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SMGSetNumPreRelax( void *smg_vdata,
                         NALU_HYPRE_Int   num_pre_relax )
{
   nalu_hypre_SMGData *smg_data = (nalu_hypre_SMGData *)smg_vdata;

   (smg_data -> num_pre_relax) = nalu_hypre_max(num_pre_relax, 1);

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_SMGGetNumPreRelax( void *smg_vdata,
                         NALU_HYPRE_Int * num_pre_relax )
{
   nalu_hypre_SMGData *smg_data = (nalu_hypre_SMGData *)smg_vdata;

   *num_pre_relax = (smg_data -> num_pre_relax);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SMGSetNumPostRelax( void *smg_vdata,
                          NALU_HYPRE_Int   num_post_relax )
{
   nalu_hypre_SMGData *smg_data = (nalu_hypre_SMGData *)smg_vdata;

   (smg_data -> num_post_relax) = num_post_relax;

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_SMGGetNumPostRelax( void *smg_vdata,
                          NALU_HYPRE_Int * num_post_relax )
{
   nalu_hypre_SMGData *smg_data = (nalu_hypre_SMGData *)smg_vdata;

   *num_post_relax = (smg_data -> num_post_relax);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SMGSetBase( void        *smg_vdata,
                  nalu_hypre_Index  base_index,
                  nalu_hypre_Index  base_stride )
{
   nalu_hypre_SMGData *smg_data = (nalu_hypre_SMGData *)smg_vdata;
   NALU_HYPRE_Int      d;

   for (d = 0; d < 3; d++)
   {
      nalu_hypre_IndexD((smg_data -> base_index),  d) =
         nalu_hypre_IndexD(base_index,  d);
      nalu_hypre_IndexD((smg_data -> base_stride), d) =
         nalu_hypre_IndexD(base_stride, d);
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SMGSetLogging( void *smg_vdata,
                     NALU_HYPRE_Int   logging)
{
   nalu_hypre_SMGData *smg_data = (nalu_hypre_SMGData *)smg_vdata;

   (smg_data -> logging) = logging;

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_SMGGetLogging( void *smg_vdata,
                     NALU_HYPRE_Int * logging)
{
   nalu_hypre_SMGData *smg_data = (nalu_hypre_SMGData *)smg_vdata;

   *logging = (smg_data -> logging);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SMGSetPrintLevel( void *smg_vdata,
                        NALU_HYPRE_Int   print_level)
{
   nalu_hypre_SMGData *smg_data = (nalu_hypre_SMGData *)smg_vdata;

   (smg_data -> print_level) = print_level;

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_SMGGetPrintLevel( void *smg_vdata,
                        NALU_HYPRE_Int * print_level)
{
   nalu_hypre_SMGData *smg_data = (nalu_hypre_SMGData *)smg_vdata;

   *print_level = (smg_data -> print_level);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SMGGetNumIterations( void *smg_vdata,
                           NALU_HYPRE_Int  *num_iterations )
{
   nalu_hypre_SMGData *smg_data = (nalu_hypre_SMGData *)smg_vdata;

   *num_iterations = (smg_data -> num_iterations);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SMGPrintLogging( void *smg_vdata,
                       NALU_HYPRE_Int   myid)
{
   nalu_hypre_SMGData *smg_data = (nalu_hypre_SMGData *)smg_vdata;
   NALU_HYPRE_Int    i;
   NALU_HYPRE_Int    num_iterations  = (smg_data -> num_iterations);
   NALU_HYPRE_Int    logging   = (smg_data -> logging);
   NALU_HYPRE_Int    print_level  = (smg_data -> print_level);
   NALU_HYPRE_Real  *norms     = (smg_data -> norms);
   NALU_HYPRE_Real  *rel_norms = (smg_data -> rel_norms);


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
nalu_hypre_SMGGetFinalRelativeResidualNorm( void   *smg_vdata,
                                       NALU_HYPRE_Real *relative_residual_norm )
{
   nalu_hypre_SMGData *smg_data = (nalu_hypre_SMGData *)smg_vdata;

   NALU_HYPRE_Int      max_iter        = (smg_data -> max_iter);
   NALU_HYPRE_Int      num_iterations  = (smg_data -> num_iterations);
   NALU_HYPRE_Int      logging         = (smg_data -> logging);
   NALU_HYPRE_Real    *rel_norms       = (smg_data -> rel_norms);

   if (logging > 0)
   {
      if (num_iterations == max_iter)
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

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SMGSetStructVectorConstantValues( nalu_hypre_StructVector *vector,
                                        NALU_HYPRE_Real          values,
                                        nalu_hypre_BoxArray     *box_array,
                                        nalu_hypre_Index         stride    )
{
   nalu_hypre_Box          *v_data_box;

   NALU_HYPRE_Real         *vp;

   nalu_hypre_Box          *box;
   nalu_hypre_Index         loop_size;
   nalu_hypre_IndexRef      start;

   NALU_HYPRE_Int           i;

   /*-----------------------------------------------------------------------
    * Set the vector coefficients
    *-----------------------------------------------------------------------*/

   nalu_hypre_ForBoxI(i, box_array)
   {
      box   = nalu_hypre_BoxArrayBox(box_array, i);
      start = nalu_hypre_BoxIMin(box);

      v_data_box =
         nalu_hypre_BoxArrayBox(nalu_hypre_StructVectorDataSpace(vector), i);
      vp = nalu_hypre_StructVectorBoxData(vector, i);

      nalu_hypre_BoxGetStrideSize(box, stride, loop_size);

#define DEVICE_VAR is_device_ptr(vp)
      nalu_hypre_BoxLoop1Begin(nalu_hypre_StructVectorNDim(vector), loop_size,
                          v_data_box, start, stride, vi);
      {
         vp[vi] = values;
      }
      nalu_hypre_BoxLoop1End(vi);
#undef DEVICE_VAR
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_StructSMGSetMaxLevel( void   *smg_vdata,
                            NALU_HYPRE_Int   max_level  )
{
   nalu_hypre_SMGData *smg_data = (nalu_hypre_SMGData *)smg_vdata;

   (smg_data -> max_levels) = max_level;

   return nalu_hypre_error_flag;
}

#if 0 //defined(NALU_HYPRE_USING_CUDA) || defined(NALU_HYPRE_USING_HIP)
NALU_HYPRE_Int
nalu_hypre_StructSMGSetDeviceLevel( void   *smg_vdata,
                               NALU_HYPRE_Int   device_level  )
{
   nalu_hypre_SMGData *smg_data = (nalu_hypre_SMGData *)smg_vdata;

   (smg_data -> devicelevel) = device_level;

   return nalu_hypre_error_flag;
}
#endif
