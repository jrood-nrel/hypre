/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_sstruct_ls.h"
#include "maxwell_TV.h"

/*--------------------------------------------------------------------------
 * hypre_MaxwellTVCreate
 *--------------------------------------------------------------------------*/

void *
hypre_MaxwellTVCreate( MPI_Comm  comm )
{
   hypre_MaxwellData *maxwell_data;
   hypre_Index       *maxwell_rfactor;

   maxwell_data = hypre_CTAlloc(hypre_MaxwellData,  1, NALU_HYPRE_MEMORY_HOST);

   (maxwell_data -> comm)       = comm;
   (maxwell_data -> time_index) = hypre_InitializeTiming("Maxwell_Solver");

   /* set defaults */
   (maxwell_data -> tol)            = 1.0e-06;
   (maxwell_data -> max_iter)       = 200;
   (maxwell_data -> rel_change)     = 0;
   (maxwell_data -> zero_guess)     = 0;
   (maxwell_data -> num_pre_relax)  = 1;
   (maxwell_data -> num_post_relax) = 1;
   (maxwell_data -> constant_coef)  = 0;
   (maxwell_data -> print_level)    = 0;
   (maxwell_data -> logging)        = 0;

   maxwell_rfactor = hypre_TAlloc(hypre_Index,  1, NALU_HYPRE_MEMORY_HOST);
   hypre_SetIndex3(maxwell_rfactor[0], 2, 2, 2);
   (maxwell_data -> rfactor) = maxwell_rfactor;


   return (void *) maxwell_data;
}

/*--------------------------------------------------------------------------
 * hypre_MaxwellTVDestroy
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
hypre_MaxwellTVDestroy( void *maxwell_vdata )
{
   hypre_MaxwellData *maxwell_data = (hypre_MaxwellData *)maxwell_vdata;

   NALU_HYPRE_Int l;
   NALU_HYPRE_Int ierr = 0;

   if (maxwell_data)
   {
      hypre_TFree(maxwell_data-> rfactor, NALU_HYPRE_MEMORY_HOST);

      if ((maxwell_data -> logging) > 0)
      {
         hypre_TFree(maxwell_data -> norms, NALU_HYPRE_MEMORY_HOST);
         hypre_TFree(maxwell_data -> rel_norms, NALU_HYPRE_MEMORY_HOST);
      }

      if ((maxwell_data -> edge_numlevels) > 0)
      {
         for (l = 0; l < (maxwell_data-> edge_numlevels); l++)
         {
            NALU_HYPRE_SStructGridDestroy(maxwell_data-> egrid_l[l]);
            hypre_ParVectorDestroy(maxwell_data-> rese_l[l]);
            hypre_ParVectorDestroy(maxwell_data-> ee_l[l]);
            hypre_ParVectorDestroy(maxwell_data-> eVtemp_l[l]);
            hypre_ParVectorDestroy(maxwell_data-> eVtemp2_l[l]);
            hypre_TFree(maxwell_data -> eCF_marker_l[l], NALU_HYPRE_MEMORY_HOST);

            /* Cannot destroy Aee_l[0] since it points to the user
               Aee_in. */
            if (l)
            {
               hypre_ParCSRMatrixDestroy(maxwell_data-> Aee_l[l]);
               hypre_ParVectorDestroy(maxwell_data-> be_l[l]);
               hypre_ParVectorDestroy(maxwell_data-> xe_l[l]);
            }

            if (l < (maxwell_data-> edge_numlevels) - 1)
            {
               NALU_HYPRE_IJMatrixDestroy(
                  (NALU_HYPRE_IJMatrix)  (maxwell_data-> Pe_l[l]));
            }

            hypre_TFree(maxwell_data-> BdryRanks_l[l], NALU_HYPRE_MEMORY_HOST);
         }
         hypre_TFree(maxwell_data-> egrid_l, NALU_HYPRE_MEMORY_HOST);
         hypre_TFree(maxwell_data-> Aee_l, NALU_HYPRE_MEMORY_HOST);
         hypre_TFree(maxwell_data-> be_l, NALU_HYPRE_MEMORY_HOST);
         hypre_TFree(maxwell_data-> xe_l, NALU_HYPRE_MEMORY_HOST);
         hypre_TFree(maxwell_data-> rese_l, NALU_HYPRE_MEMORY_HOST);
         hypre_TFree(maxwell_data-> ee_l, NALU_HYPRE_MEMORY_HOST);
         hypre_TFree(maxwell_data-> eVtemp_l, NALU_HYPRE_MEMORY_HOST);
         hypre_TFree(maxwell_data-> eVtemp2_l, NALU_HYPRE_MEMORY_HOST);
         hypre_TFree(maxwell_data-> Pe_l, NALU_HYPRE_MEMORY_HOST);
         hypre_TFree(maxwell_data-> ReT_l, NALU_HYPRE_MEMORY_HOST);
         hypre_TFree(maxwell_data-> eCF_marker_l, NALU_HYPRE_MEMORY_HOST);
         hypre_TFree(maxwell_data-> erelax_weight, NALU_HYPRE_MEMORY_HOST);
         hypre_TFree(maxwell_data-> eomega, NALU_HYPRE_MEMORY_HOST);

         hypre_TFree(maxwell_data-> BdryRanks_l, NALU_HYPRE_MEMORY_HOST);
         hypre_TFree(maxwell_data-> BdryRanksCnts_l, NALU_HYPRE_MEMORY_HOST);
      }

      if ((maxwell_data -> node_numlevels) > 0)
      {
         for (l = 0; l < (maxwell_data-> node_numlevels); l++)
         {
            hypre_ParVectorDestroy(maxwell_data-> resn_l[l]);
            hypre_ParVectorDestroy(maxwell_data-> en_l[l]);
            hypre_ParVectorDestroy(maxwell_data-> nVtemp_l[l]);
            hypre_ParVectorDestroy(maxwell_data-> nVtemp2_l[l]);
         }
         hypre_BoomerAMGDestroy(maxwell_data-> amg_vdata);

         hypre_TFree(maxwell_data-> Ann_l, NALU_HYPRE_MEMORY_HOST);
         hypre_TFree(maxwell_data-> Pn_l, NALU_HYPRE_MEMORY_HOST);
         hypre_TFree(maxwell_data-> RnT_l, NALU_HYPRE_MEMORY_HOST);
         hypre_TFree(maxwell_data-> bn_l, NALU_HYPRE_MEMORY_HOST);
         hypre_TFree(maxwell_data-> xn_l, NALU_HYPRE_MEMORY_HOST);
         hypre_TFree(maxwell_data-> resn_l, NALU_HYPRE_MEMORY_HOST);
         hypre_TFree(maxwell_data-> en_l, NALU_HYPRE_MEMORY_HOST);
         hypre_TFree(maxwell_data-> nVtemp_l, NALU_HYPRE_MEMORY_HOST);
         hypre_TFree(maxwell_data-> nVtemp2_l, NALU_HYPRE_MEMORY_HOST);
         hypre_TFree(maxwell_data-> nCF_marker_l, NALU_HYPRE_MEMORY_HOST);
         hypre_TFree(maxwell_data-> nrelax_weight, NALU_HYPRE_MEMORY_HOST);
         hypre_TFree(maxwell_data-> nomega, NALU_HYPRE_MEMORY_HOST);
      }

      NALU_HYPRE_SStructStencilDestroy(maxwell_data-> Ann_stencils[0]);
      hypre_TFree(maxwell_data-> Ann_stencils, NALU_HYPRE_MEMORY_HOST);

      if ((maxwell_data -> en_numlevels) > 0)
      {
         for (l = 1; l < (maxwell_data-> en_numlevels); l++)
         {
            hypre_ParCSRMatrixDestroy(maxwell_data-> Aen_l[l]);
         }
      }
      hypre_TFree(maxwell_data-> Aen_l, NALU_HYPRE_MEMORY_HOST);

      NALU_HYPRE_SStructVectorDestroy(
         (NALU_HYPRE_SStructVector) maxwell_data-> bn);
      NALU_HYPRE_SStructVectorDestroy(
         (NALU_HYPRE_SStructVector) maxwell_data-> xn);
      NALU_HYPRE_SStructMatrixDestroy(
         (NALU_HYPRE_SStructMatrix) maxwell_data-> Ann);
      NALU_HYPRE_IJMatrixDestroy(maxwell_data-> Aen);

      hypre_ParCSRMatrixDestroy(maxwell_data-> T_transpose);

      hypre_FinalizeTiming(maxwell_data -> time_index);
      hypre_TFree(maxwell_data, NALU_HYPRE_MEMORY_HOST);
   }

   return (ierr);
}

/*--------------------------------------------------------------------------
 * hypre_MaxwellSetRfactors
 *--------------------------------------------------------------------------*/
NALU_HYPRE_Int
hypre_MaxwellSetRfactors(void         *maxwell_vdata,
                         NALU_HYPRE_Int     rfactor[3] )
{
   hypre_MaxwellData *maxwell_data   = (hypre_MaxwellData *)maxwell_vdata;
   hypre_Index       *maxwell_rfactor = (maxwell_data -> rfactor);
   NALU_HYPRE_Int          ierr       = 0;

   hypre_CopyIndex(rfactor, maxwell_rfactor[0]);

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_MaxwellSetGrad
 *--------------------------------------------------------------------------*/
NALU_HYPRE_Int
hypre_MaxwellSetGrad(void               *maxwell_vdata,
                     hypre_ParCSRMatrix *T )
{
   hypre_MaxwellData *maxwell_data = (hypre_MaxwellData *)maxwell_vdata;
   NALU_HYPRE_Int          ierr       = 0;

   (maxwell_data -> Tgrad) =  T;

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_MaxwellSetConstantCoef
 *--------------------------------------------------------------------------*/
NALU_HYPRE_Int
hypre_MaxwellSetConstantCoef( void   *maxwell_vdata,
                              NALU_HYPRE_Int     constant_coef)
{
   hypre_MaxwellData *maxwell_data = (hypre_MaxwellData *)maxwell_vdata;
   NALU_HYPRE_Int          ierr        = 0;

   (maxwell_data -> constant_coef) = constant_coef;

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_MaxwellSetTol
 *--------------------------------------------------------------------------*/
NALU_HYPRE_Int
hypre_MaxwellSetTol( void   *maxwell_vdata,
                     NALU_HYPRE_Real  tol       )
{
   hypre_MaxwellData *maxwell_data = (hypre_MaxwellData *)maxwell_vdata;
   NALU_HYPRE_Int          ierr        = 0;

   (maxwell_data -> tol) = tol;

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_MaxwellSetMaxIter
 *--------------------------------------------------------------------------*/
NALU_HYPRE_Int
hypre_MaxwellSetMaxIter( void *maxwell_vdata,
                         NALU_HYPRE_Int   max_iter  )
{
   hypre_MaxwellData *maxwell_data = (hypre_MaxwellData *)maxwell_vdata;
   NALU_HYPRE_Int          ierr = 0;

   (maxwell_data -> max_iter) = max_iter;

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_MaxwellSetRelChange
 *--------------------------------------------------------------------------*/
NALU_HYPRE_Int
hypre_MaxwellSetRelChange( void *maxwell_vdata,
                           NALU_HYPRE_Int   rel_change  )
{
   hypre_MaxwellData *maxwell_data = (hypre_MaxwellData *)maxwell_vdata;
   NALU_HYPRE_Int          ierr = 0;

   (maxwell_data -> rel_change) = rel_change;

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_MaxwellNumPreRelax
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
hypre_MaxwellSetNumPreRelax( void *maxwell_vdata,
                             NALU_HYPRE_Int   num_pre_relax )
{
   hypre_MaxwellData *maxwell_data = (hypre_MaxwellData *)maxwell_vdata;
   NALU_HYPRE_Int          ierr = 0;

   (maxwell_data -> num_pre_relax) = num_pre_relax;

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_MaxwellSetNumPostRelax
 *--------------------------------------------------------------------------*/
NALU_HYPRE_Int
hypre_MaxwellSetNumPostRelax( void *maxwell_vdata,
                              NALU_HYPRE_Int   num_post_relax )
{
   hypre_MaxwellData *maxwell_data = (hypre_MaxwellData *)maxwell_vdata;
   NALU_HYPRE_Int          ierr = 0;

   (maxwell_data -> num_post_relax) = num_post_relax;

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_MaxwellGetNumIterations
 *--------------------------------------------------------------------------*/
NALU_HYPRE_Int
hypre_MaxwellGetNumIterations( void *maxwell_vdata,
                               NALU_HYPRE_Int  *num_iterations )
{
   hypre_MaxwellData *maxwell_data = (hypre_MaxwellData *)maxwell_vdata;
   NALU_HYPRE_Int          ierr = 0;

   *num_iterations = (maxwell_data -> num_iterations);

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_MaxwellSetPrintLevel
 *--------------------------------------------------------------------------*/
NALU_HYPRE_Int
hypre_MaxwellSetPrintLevel( void *maxwell_vdata,
                            NALU_HYPRE_Int   print_level)
{
   hypre_MaxwellData *maxwell_data = (hypre_MaxwellData *)maxwell_vdata;
   NALU_HYPRE_Int          ierr = 0;

   (maxwell_data -> print_level) = print_level;

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_MaxwellSetLogging
 *--------------------------------------------------------------------------*/
NALU_HYPRE_Int
hypre_MaxwellSetLogging( void *maxwell_vdata,
                         NALU_HYPRE_Int   logging)
{
   hypre_MaxwellData *maxwell_data = (hypre_MaxwellData *)maxwell_vdata;
   NALU_HYPRE_Int          ierr = 0;

   (maxwell_data -> logging) = logging;

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_MaxwellPrintLogging
 *--------------------------------------------------------------------------*/
NALU_HYPRE_Int
hypre_MaxwellPrintLogging( void *maxwell_vdata,
                           NALU_HYPRE_Int   myid)
{
   hypre_MaxwellData *maxwell_data = (hypre_MaxwellData *)maxwell_vdata;
   NALU_HYPRE_Int          ierr = 0;
   NALU_HYPRE_Int          i;
   NALU_HYPRE_Int          num_iterations = (maxwell_data -> num_iterations);
   NALU_HYPRE_Int          logging       = (maxwell_data -> logging);
   NALU_HYPRE_Int          print_level   = (maxwell_data -> print_level);
   NALU_HYPRE_Real        *norms         = (maxwell_data -> norms);
   NALU_HYPRE_Real        *rel_norms     = (maxwell_data -> rel_norms);

   if (myid == 0)
   {
      if (print_level > 0 )
      {
         if (logging > 0)
         {
            for (i = 0; i < num_iterations; i++)
            {
               hypre_printf("Residual norm[%d] = %e   ", i, norms[i]);
               hypre_printf("Relative residual norm[%d] = %e\n", i, rel_norms[i]);
            }
         }
      }
   }

   return ierr;
}

NALU_HYPRE_Int
hypre_MaxwellGetFinalRelativeResidualNorm( void   *maxwell_vdata,
                                           NALU_HYPRE_Real *relative_residual_norm )
{
   hypre_MaxwellData *maxwell_data = (hypre_MaxwellData *)maxwell_vdata;

   NALU_HYPRE_Int          max_iter        = (maxwell_data -> max_iter);
   NALU_HYPRE_Int          num_iterations  = (maxwell_data -> num_iterations);
   NALU_HYPRE_Int          logging         = (maxwell_data -> logging);
   NALU_HYPRE_Real        *rel_norms       = (maxwell_data -> rel_norms);

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
