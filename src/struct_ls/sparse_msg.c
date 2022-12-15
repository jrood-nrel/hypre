/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 *
 *****************************************************************************/

#include "_nalu_hypre_struct_ls.h"
#include "sparse_msg.h"

/*--------------------------------------------------------------------------
 * nalu_hypre_SparseMSGCreate
 *--------------------------------------------------------------------------*/

void *
nalu_hypre_SparseMSGCreate( MPI_Comm  comm )
{
   nalu_hypre_SparseMSGData *smsg_data;

   smsg_data = nalu_hypre_CTAlloc(nalu_hypre_SparseMSGData,  1, NALU_HYPRE_MEMORY_HOST);

   (smsg_data -> comm)       = comm;
   (smsg_data -> time_index) = nalu_hypre_InitializeTiming("SparseMSG");

   /* set defaults */
   (smsg_data -> tol)              = 1.0e-06;
   (smsg_data -> max_iter)         = 200;
   (smsg_data -> rel_change)       = 0;
   (smsg_data -> zero_guess)       = 0;
   (smsg_data -> jump)             = 0;
   (smsg_data -> relax_type)       = 1;       /* weighted Jacobi */
   (smsg_data -> jacobi_weight)    = 0.0;
   (smsg_data -> usr_jacobi_weight) = 0;    /* no user Jacobi weight */
   (smsg_data -> num_pre_relax)    = 1;
   (smsg_data -> num_post_relax)   = 1;
   (smsg_data -> num_fine_relax)   = 1;
   (smsg_data -> logging)          = 0;
   (smsg_data -> print_level)      = 0;

   /* initialize */
   (smsg_data -> num_grids[0])     = 1;
   (smsg_data -> num_grids[1])     = 1;
   (smsg_data -> num_grids[2])     = 1;

   (smsg_data -> memory_location)  = nalu_hypre_HandleMemoryLocation(nalu_hypre_handle());

   return (void *) smsg_data;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_SparseMSGDestroy
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SparseMSGDestroy( void *smsg_vdata )
{
   NALU_HYPRE_Int ierr = 0;

   /* RDF */
#if 0
   nalu_hypre_SparseMSGData *smsg_data = smsg_vdata;

   NALU_HYPRE_Int fi, l;

   if (smsg_data)
   {
      if ((smsg_data -> logging) > 0)
      {
         nalu_hypre_TFree(smsg_data -> norms, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(smsg_data -> rel_norms, NALU_HYPRE_MEMORY_HOST);
      }

      if ((smsg_data -> num_levels) > 1)
      {
         for (fi = 0; fi < (smsg_data -> num_all_grids); fi++)
         {
            nalu_hypre_PFMGRelaxDestroy(smsg_data -> relax_array[fi]);
            nalu_hypre_StructMatvecDestroy(smsg_data -> matvec_array[fi]);
            nalu_hypre_SemiRestrictDestroy(smsg_data -> restrictx_array[fi]);
            nalu_hypre_SemiRestrictDestroy(smsg_data -> restricty_array[fi]);
            nalu_hypre_SemiRestrictDestroy(smsg_data -> restrictz_array[fi]);
            nalu_hypre_SemiInterpDestroy(smsg_data -> interpx_array[fi]);
            nalu_hypre_SemiInterpDestroy(smsg_data -> interpy_array[fi]);
            nalu_hypre_SemiInterpDestroy(smsg_data -> interpz_array[fi]);
            nalu_hypre_StructMatrixDestroy(smsg_data -> A_array[fi]);
            nalu_hypre_StructVectorDestroy(smsg_data -> b_array[fi]);
            nalu_hypre_StructVectorDestroy(smsg_data -> x_array[fi]);
            nalu_hypre_StructVectorDestroy(smsg_data -> t_array[fi]);
            nalu_hypre_StructVectorDestroy(smsg_data -> r_array[fi]);
            nalu_hypre_StructVectorDestroy(smsg_data -> visitx_array[fi]);
            nalu_hypre_StructVectorDestroy(smsg_data -> visity_array[fi]);
            nalu_hypre_StructVectorDestroy(smsg_data -> visitz_array[fi]);
            nalu_hypre_StructGridDestroy(smsg_data -> grid_array[fi]);
         }

         for (l = 0; l < (smsg_data -> num_grids[0]) - 1; l++)
         {
            nalu_hypre_StructMatrixDestroy(smsg_data -> Px_array[l]);
            nalu_hypre_StructGridDestroy(smsg_data -> Px_grid_array[l]);
         }
         for (l = 0; l < (smsg_data -> num_grids[1]) - 1; l++)
         {
            nalu_hypre_StructMatrixDestroy(smsg_data -> Py_array[l]);
            nalu_hypre_StructGridDestroy(smsg_data -> Py_grid_array[l]);
         }
         for (l = 0; l < (smsg_data -> num_grids[2]) - 1; l++)
         {
            nalu_hypre_StructMatrixDestroy(smsg_data -> Pz_array[l]);
            nalu_hypre_StructGridDestroy(smsg_data -> Pz_grid_array[l]);
         }

         nalu_hypre_TFree(smsg_data -> data, NALU_HYPRE_MEMORY_HOST);

         nalu_hypre_TFree(smsg_data -> relax_array, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(smsg_data -> matvec_array, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(smsg_data -> restrictx_array, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(smsg_data -> restricty_array, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(smsg_data -> restrictz_array, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(smsg_data -> interpx_array, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(smsg_data -> interpy_array, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(smsg_data -> interpz_array, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(smsg_data -> A_array, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(smsg_data -> Px_array, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(smsg_data -> Py_array, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(smsg_data -> Pz_array, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(smsg_data -> RTx_array, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(smsg_data -> RTy_array, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(smsg_data -> RTz_array, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(smsg_data -> b_array, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(smsg_data -> x_array, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(smsg_data -> t_array, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(smsg_data -> r_array, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(smsg_data -> grid_array, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(smsg_data -> Px_grid_array, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(smsg_data -> Py_grid_array, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(smsg_data -> Pz_grid_array, NALU_HYPRE_MEMORY_HOST);
      }

      nalu_hypre_FinalizeTiming(smsg_data -> time_index);
      nalu_hypre_TFree(smsg_data, NALU_HYPRE_MEMORY_HOST);
   }
#endif
   /* RDF */

   return ierr;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_SparseMSGSetTol
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SparseMSGSetTol( void   *smsg_vdata,
                       NALU_HYPRE_Real  tol        )
{
   nalu_hypre_SparseMSGData *smsg_data = (nalu_hypre_SparseMSGData *)smsg_vdata;
   NALU_HYPRE_Int       ierr = 0;

   (smsg_data -> tol) = tol;

   return ierr;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_SparseMSGSetMaxIter
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SparseMSGSetMaxIter( void *smsg_vdata,
                           NALU_HYPRE_Int   max_iter   )
{
   nalu_hypre_SparseMSGData *smsg_data = (nalu_hypre_SparseMSGData *)smsg_vdata;
   NALU_HYPRE_Int       ierr = 0;

   (smsg_data -> max_iter) = max_iter;

   return ierr;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_SparseMSGSetJump
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SparseMSGSetJump(  void *smsg_vdata,
                         NALU_HYPRE_Int   jump       )

{
   nalu_hypre_SparseMSGData *smsg_data = (nalu_hypre_SparseMSGData *)smsg_vdata;
   NALU_HYPRE_Int            ierr = 0;

   (smsg_data -> jump) = jump;

   return ierr;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_SparseMSGSetRelChange
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SparseMSGSetRelChange( void *smsg_vdata,
                             NALU_HYPRE_Int   rel_change )
{
   nalu_hypre_SparseMSGData *smsg_data = (nalu_hypre_SparseMSGData *)smsg_vdata;
   NALU_HYPRE_Int       ierr = 0;

   (smsg_data -> rel_change) = rel_change;

   return ierr;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_SparseMSGSetZeroGuess
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SparseMSGSetZeroGuess( void *smsg_vdata,
                             NALU_HYPRE_Int   zero_guess )
{
   nalu_hypre_SparseMSGData *smsg_data = (nalu_hypre_SparseMSGData *)smsg_vdata;
   NALU_HYPRE_Int       ierr = 0;

   (smsg_data -> zero_guess) = zero_guess;

   return ierr;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_SparseMSGSetRelaxType
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SparseMSGSetRelaxType( void *smsg_vdata,
                             NALU_HYPRE_Int   relax_type )
{
   nalu_hypre_SparseMSGData *smsg_data = (nalu_hypre_SparseMSGData *)smsg_vdata;
   NALU_HYPRE_Int       ierr = 0;

   (smsg_data -> relax_type) = relax_type;

   return ierr;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_SparseMSGSetJacobiWeight
 *--------------------------------------------------------------------------*/
NALU_HYPRE_Int
nalu_hypre_SparseMSGSetJacobiWeight( void  *smsg_vdata,
                                NALU_HYPRE_Real weight )
{
   nalu_hypre_SparseMSGData *smsg_data = (nalu_hypre_SparseMSGData *)smsg_vdata;

   (smsg_data -> jacobi_weight)    = weight;
   (smsg_data -> usr_jacobi_weight) = 1;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_SparseMSGSetNumPreRelax
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SparseMSGSetNumPreRelax( void *smsg_vdata,
                               NALU_HYPRE_Int   num_pre_relax )
{
   nalu_hypre_SparseMSGData *smsg_data = (nalu_hypre_SparseMSGData *)smsg_vdata;
   NALU_HYPRE_Int       ierr = 0;

   (smsg_data -> num_pre_relax) = num_pre_relax;

   return ierr;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_SparseMSGSetNumPostRelax
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SparseMSGSetNumPostRelax( void *smsg_vdata,
                                NALU_HYPRE_Int   num_post_relax )
{
   nalu_hypre_SparseMSGData *smsg_data = (nalu_hypre_SparseMSGData *)smsg_vdata;
   NALU_HYPRE_Int       ierr = 0;

   (smsg_data -> num_post_relax) = num_post_relax;

   return ierr;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_SparseMSGSetNumFineRelax
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SparseMSGSetNumFineRelax( void *smsg_vdata,
                                NALU_HYPRE_Int   num_fine_relax )
{
   nalu_hypre_SparseMSGData *smsg_data = (nalu_hypre_SparseMSGData *)smsg_vdata;
   NALU_HYPRE_Int       ierr = 0;

   (smsg_data -> num_fine_relax) = num_fine_relax;

   return ierr;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_SparseMSGSetLogging
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SparseMSGSetLogging( void *smsg_vdata,
                           NALU_HYPRE_Int   logging    )
{
   nalu_hypre_SparseMSGData *smsg_data = (nalu_hypre_SparseMSGData *)smsg_vdata;
   NALU_HYPRE_Int       ierr = 0;

   (smsg_data -> logging) = logging;

   return ierr;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_SparseMSGSetPrintLevel
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SparseMSGSetPrintLevel( void *smsg_vdata,
                              NALU_HYPRE_Int   print_level    )
{
   nalu_hypre_SparseMSGData *smsg_data = (nalu_hypre_SparseMSGData *)smsg_vdata;
   NALU_HYPRE_Int       ierr = 0;

   (smsg_data -> print_level) = print_level;

   return ierr;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_SparseMSGGetNumIterations
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SparseMSGGetNumIterations( void *smsg_vdata,
                                 NALU_HYPRE_Int  *num_iterations )
{
   nalu_hypre_SparseMSGData *smsg_data = (nalu_hypre_SparseMSGData *)smsg_vdata;
   NALU_HYPRE_Int       ierr = 0;

   *num_iterations = (smsg_data -> num_iterations);

   return ierr;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_SparseMSGPrintLogging
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SparseMSGPrintLogging( void *smsg_vdata,
                             NALU_HYPRE_Int   myid       )
{
   nalu_hypre_SparseMSGData *smsg_data = (nalu_hypre_SparseMSGData *)smsg_vdata;
   NALU_HYPRE_Int       ierr = 0;
   NALU_HYPRE_Int       i;
   NALU_HYPRE_Int       num_iterations  = (smsg_data -> num_iterations);
   NALU_HYPRE_Int       logging   = (smsg_data -> logging);
   NALU_HYPRE_Int     print_level = (smsg_data -> print_level);
   NALU_HYPRE_Real     *norms     = (smsg_data -> norms);
   NALU_HYPRE_Real     *rel_norms = (smsg_data -> rel_norms);

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

   return ierr;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_SparseMSGGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SparseMSGGetFinalRelativeResidualNorm( void   *smsg_vdata,
                                             NALU_HYPRE_Real *relative_residual_norm )
{
   nalu_hypre_SparseMSGData *smsg_data = (nalu_hypre_SparseMSGData *)smsg_vdata;

   NALU_HYPRE_Int       max_iter        = (smsg_data -> max_iter);
   NALU_HYPRE_Int       num_iterations  = (smsg_data -> num_iterations);
   NALU_HYPRE_Int       logging         = (smsg_data -> logging);
   NALU_HYPRE_Real     *rel_norms       = (smsg_data -> rel_norms);

   NALU_HYPRE_Int       ierr = 0;


   if (logging > 0)
   {
      if (max_iter == 0)
      {
         ierr = 1;
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

   return ierr;
}


