/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_nalu_hypre_sstruct_ls.h"
#include "fac.h"

/*--------------------------------------------------------------------------
 * nalu_hypre_FACCreate
 *--------------------------------------------------------------------------*/
void *
nalu_hypre_FACCreate( MPI_Comm  comm )
{
   nalu_hypre_FACData *fac_data;

   fac_data = nalu_hypre_CTAlloc(nalu_hypre_FACData,  1, NALU_HYPRE_MEMORY_HOST);

   (fac_data -> comm)       = comm;
   (fac_data -> time_index) = nalu_hypre_InitializeTiming("FAC");

   /* set defaults */
   (fac_data -> tol)              = 1.0e-06;
   (fac_data -> max_cycles)       = 200;
   (fac_data -> zero_guess)       = 0;
   (fac_data -> max_levels)       = 0;
   (fac_data -> relax_type)       = 2; /*  1 Jacobi; 2 Gauss-Seidel */
   (fac_data -> jacobi_weight)    = 0.0;
   (fac_data -> usr_jacobi_weight) = 0;
   (fac_data -> num_pre_smooth)   = 1;
   (fac_data -> num_post_smooth)  = 1;
   (fac_data -> csolver_type)     = 1;
   (fac_data -> logging)          = 0;

   return (void *) fac_data;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_FACDestroy
 *--------------------------------------------------------------------------*/
NALU_HYPRE_Int
nalu_hypre_FACDestroy2(void *fac_vdata)
{
   nalu_hypre_FACData *fac_data = (nalu_hypre_FACData *)fac_vdata;

   NALU_HYPRE_Int level;
   NALU_HYPRE_Int ierr = 0;

   if (fac_data)
   {
      nalu_hypre_TFree((fac_data ->plevels), NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree((fac_data ->prefinements), NALU_HYPRE_MEMORY_HOST);

      NALU_HYPRE_SStructGraphDestroy(nalu_hypre_SStructMatrixGraph((fac_data -> A_rap)));
      NALU_HYPRE_SStructMatrixDestroy((fac_data -> A_rap));
      for (level = 0; level <= (fac_data -> max_levels); level++)
      {
         NALU_HYPRE_SStructMatrixDestroy( (fac_data -> A_level[level]) );
         NALU_HYPRE_SStructVectorDestroy( (fac_data -> x_level[level]) );
         NALU_HYPRE_SStructVectorDestroy( (fac_data -> b_level[level]) );
         NALU_HYPRE_SStructVectorDestroy( (fac_data -> r_level[level]) );
         NALU_HYPRE_SStructVectorDestroy( (fac_data -> e_level[level]) );
         nalu_hypre_SStructPVectorDestroy( (fac_data -> tx_level[level]) );

         NALU_HYPRE_SStructGraphDestroy( (fac_data -> graph_level[level]) );
         NALU_HYPRE_SStructGridDestroy(  (fac_data -> grid_level[level]) );

         nalu_hypre_SStructMatvecDestroy( (fac_data   -> matvec_data_level[level]) );
         nalu_hypre_SStructPMatvecDestroy((fac_data  -> pmatvec_data_level[level]) );

         nalu_hypre_SysPFMGRelaxDestroy( (fac_data -> relax_data_level[level]) );

         if (level > 0)
         {
            nalu_hypre_FacSemiRestrictDestroy2( (fac_data -> restrict_data_level[level]) );
         }

         if (level < (fac_data -> max_levels))
         {
            nalu_hypre_FacSemiInterpDestroy2( (fac_data -> interp_data_level[level]) );
         }
      }
      nalu_hypre_SStructMatvecDestroy( (fac_data -> matvec_data) );

      nalu_hypre_TFree(fac_data -> A_level, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(fac_data -> x_level, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(fac_data -> b_level, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(fac_data -> r_level, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(fac_data -> e_level, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(fac_data -> tx_level, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(fac_data -> relax_data_level, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(fac_data -> restrict_data_level, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(fac_data -> matvec_data_level, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(fac_data -> pmatvec_data_level, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(fac_data -> interp_data_level, NALU_HYPRE_MEMORY_HOST);

      nalu_hypre_TFree(fac_data -> grid_level, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(fac_data -> graph_level, NALU_HYPRE_MEMORY_HOST);

      NALU_HYPRE_SStructVectorDestroy(fac_data -> tx);

      nalu_hypre_TFree(fac_data -> level_to_part, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(fac_data -> part_to_level, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(fac_data -> refine_factors, NALU_HYPRE_MEMORY_HOST);

      if ( (fac_data -> csolver_type) == 1)
      {
         NALU_HYPRE_SStructPCGDestroy(fac_data -> csolver);
         NALU_HYPRE_SStructSysPFMGDestroy(fac_data -> cprecond);
      }
      else if ((fac_data -> csolver_type) == 2)
      {
         NALU_HYPRE_SStructSysPFMGDestroy(fac_data -> csolver);
      }

      if ((fac_data -> logging) > 0)
      {
         nalu_hypre_TFree(fac_data -> norms, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(fac_data -> rel_norms, NALU_HYPRE_MEMORY_HOST);
      }

      nalu_hypre_FinalizeTiming(fac_data -> time_index);

      nalu_hypre_TFree(fac_data, NALU_HYPRE_MEMORY_HOST);
   }

   return (ierr);
}

NALU_HYPRE_Int
nalu_hypre_FACSetTol( void   *fac_vdata,
                 NALU_HYPRE_Real  tol       )
{
   nalu_hypre_FACData *fac_data = (nalu_hypre_FACData *)fac_vdata;
   NALU_HYPRE_Int          ierr = 0;

   (fac_data -> tol) = tol;

   return ierr;
}


/*--------------------------------------------------------------------------
 * nalu_hypre_FACSetPLevels
 *--------------------------------------------------------------------------*/
NALU_HYPRE_Int
nalu_hypre_FACSetPLevels( void *fac_vdata,
                     NALU_HYPRE_Int   nparts,
                     NALU_HYPRE_Int  *plevels)
{
   nalu_hypre_FACData *fac_data   = (nalu_hypre_FACData *)fac_vdata;
   NALU_HYPRE_Int     *fac_plevels;
   NALU_HYPRE_Int      ierr       = 0;
   NALU_HYPRE_Int      i;

   fac_plevels = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  nparts, NALU_HYPRE_MEMORY_HOST);

   for (i = 0; i < nparts; i++)
   {
      fac_plevels[i] = plevels[i];
   }

   (fac_data -> plevels) =  fac_plevels;

   return ierr;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_FACSetPRefinements
 *--------------------------------------------------------------------------*/
NALU_HYPRE_Int
nalu_hypre_FACSetPRefinements( void         *fac_vdata,
                          NALU_HYPRE_Int     nparts,
                          nalu_hypre_Index  *prefinements )
{
   nalu_hypre_FACData *fac_data   = (nalu_hypre_FACData *)fac_vdata;
   nalu_hypre_Index   *fac_prefinements;
   NALU_HYPRE_Int      ierr       = 0;
   NALU_HYPRE_Int      i;

   fac_prefinements = nalu_hypre_TAlloc(nalu_hypre_Index,  nparts, NALU_HYPRE_MEMORY_HOST);

   for (i = 0; i < nparts; i++)
   {
      nalu_hypre_CopyIndex( prefinements[i], fac_prefinements[i] );
   }

   (fac_data -> prefinements) =  fac_prefinements;

   return ierr;
}
/*--------------------------------------------------------------------------
 * nalu_hypre_FACSetMaxLevels
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_FACSetMaxLevels( void *fac_vdata,
                       NALU_HYPRE_Int   nparts )
{
   nalu_hypre_FACData *fac_data = (nalu_hypre_FACData *)fac_vdata;
   NALU_HYPRE_Int          ierr = 0;

   (fac_data -> max_levels) = nparts - 1;

   return ierr;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_FACSetMaxIter
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_FACSetMaxIter( void *fac_vdata,
                     NALU_HYPRE_Int   max_iter  )
{
   nalu_hypre_FACData *fac_data = (nalu_hypre_FACData *)fac_vdata;
   NALU_HYPRE_Int          ierr = 0;

   (fac_data -> max_cycles) = max_iter;

   return ierr;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_FACSetRelChange
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_FACSetRelChange( void *fac_vdata,
                       NALU_HYPRE_Int   rel_change  )
{
   nalu_hypre_FACData *fac_data = (nalu_hypre_FACData *)fac_vdata;
   NALU_HYPRE_Int          ierr = 0;

   (fac_data -> rel_change) = rel_change;

   return ierr;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_FACSetZeroGuess
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_FACSetZeroGuess( void *fac_vdata,
                       NALU_HYPRE_Int   zero_guess )
{
   nalu_hypre_FACData *fac_data = (nalu_hypre_FACData *)fac_vdata;
   NALU_HYPRE_Int          ierr = 0;

   (fac_data -> zero_guess) = zero_guess;

   return ierr;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_FACSetRelaxType
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_FACSetRelaxType( void *fac_vdata,
                       NALU_HYPRE_Int   relax_type )
{
   nalu_hypre_FACData *fac_data = (nalu_hypre_FACData *)fac_vdata;
   NALU_HYPRE_Int          ierr = 0;

   (fac_data -> relax_type) = relax_type;

   return ierr;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_FACSetJacobiWeight
 *--------------------------------------------------------------------------*/
NALU_HYPRE_Int
nalu_hypre_FACSetJacobiWeight( void  *fac_vdata,
                          NALU_HYPRE_Real weight )
{
   nalu_hypre_FACData *fac_data = (nalu_hypre_FACData *)fac_vdata;

   (fac_data -> jacobi_weight)    = weight;
   (fac_data -> usr_jacobi_weight) = 1;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_FACSetNumPreRelax
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_FACSetNumPreSmooth( void *fac_vdata,
                          NALU_HYPRE_Int   num_pre_smooth )
{
   nalu_hypre_FACData *fac_data = (nalu_hypre_FACData *)fac_vdata;
   NALU_HYPRE_Int          ierr = 0;

   (fac_data -> num_pre_smooth) = num_pre_smooth;

   return ierr;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_FACSetNumPostRelax
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_FACSetNumPostSmooth( void *fac_vdata,
                           NALU_HYPRE_Int   num_post_smooth )
{
   nalu_hypre_FACData *fac_data = (nalu_hypre_FACData *)fac_vdata;
   NALU_HYPRE_Int          ierr = 0;

   (fac_data -> num_post_smooth) = num_post_smooth;

   return ierr;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_FACSetCoarseSolverType
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_FACSetCoarseSolverType( void *fac_vdata,
                              NALU_HYPRE_Int   csolver_type)
{
   nalu_hypre_FACData *fac_data = (nalu_hypre_FACData *)fac_vdata;
   NALU_HYPRE_Int          ierr = 0;

   (fac_data -> csolver_type) = csolver_type;

   return ierr;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_FACSetLogging
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_FACSetLogging( void *fac_vdata,
                     NALU_HYPRE_Int   logging)
{
   nalu_hypre_FACData *fac_data = (nalu_hypre_FACData *)fac_vdata;
   NALU_HYPRE_Int          ierr = 0;

   (fac_data -> logging) = logging;

   return ierr;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_SysFACGetNumIterations
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_FACGetNumIterations( void *fac_vdata,
                           NALU_HYPRE_Int  *num_iterations )
{
   nalu_hypre_FACData *fac_data = (nalu_hypre_FACData *)fac_vdata;
   NALU_HYPRE_Int          ierr = 0;

   *num_iterations = (fac_data -> num_iterations);

   return ierr;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_FACPrintLogging
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_FACPrintLogging( void *fac_vdata,
                       NALU_HYPRE_Int   myid)
{
   nalu_hypre_FACData *fac_data = (nalu_hypre_FACData *)fac_vdata;
   NALU_HYPRE_Int          ierr = 0;
   NALU_HYPRE_Int          i;
   NALU_HYPRE_Int          num_iterations  = (fac_data -> num_iterations);
   NALU_HYPRE_Int          logging   = (fac_data -> logging);
   NALU_HYPRE_Real        *norms     = (fac_data -> norms);
   NALU_HYPRE_Real        *rel_norms = (fac_data -> rel_norms);

   if (myid == 0)
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

   return ierr;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_FACGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_FACGetFinalRelativeResidualNorm( void   *fac_vdata,
                                       NALU_HYPRE_Real *relative_residual_norm )
{
   nalu_hypre_FACData *fac_data = (nalu_hypre_FACData *)fac_vdata;

   NALU_HYPRE_Int          max_iter        = (fac_data -> max_cycles);
   NALU_HYPRE_Int          num_iterations  = (fac_data -> num_iterations);
   NALU_HYPRE_Int          logging         = (fac_data -> logging);
   NALU_HYPRE_Real        *rel_norms       = (fac_data -> rel_norms);

   NALU_HYPRE_Int          ierr = 0;


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

