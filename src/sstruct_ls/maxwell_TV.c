/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_nalu_hypre_sstruct_ls.h"
#include "maxwell_TV.h"

/*--------------------------------------------------------------------------
 * nalu_hypre_MaxwellTVCreate
 *--------------------------------------------------------------------------*/

void *
nalu_hypre_MaxwellTVCreate( MPI_Comm  comm )
{
   nalu_hypre_MaxwellData *maxwell_data;
   nalu_hypre_Index       *maxwell_rfactor;

   maxwell_data = nalu_hypre_CTAlloc(nalu_hypre_MaxwellData,  1, NALU_HYPRE_MEMORY_HOST);

   (maxwell_data -> comm)       = comm;
   (maxwell_data -> time_index) = nalu_hypre_InitializeTiming("Maxwell_Solver");

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

   maxwell_rfactor = nalu_hypre_TAlloc(nalu_hypre_Index,  1, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_SetIndex3(maxwell_rfactor[0], 2, 2, 2);
   (maxwell_data -> rfactor) = maxwell_rfactor;


   return (void *) maxwell_data;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_MaxwellTVDestroy
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_MaxwellTVDestroy( void *maxwell_vdata )
{
   nalu_hypre_MaxwellData *maxwell_data = (nalu_hypre_MaxwellData *)maxwell_vdata;

   NALU_HYPRE_Int l;
   NALU_HYPRE_Int ierr = 0;

   if (maxwell_data)
   {
      nalu_hypre_TFree(maxwell_data-> rfactor, NALU_HYPRE_MEMORY_HOST);

      if ((maxwell_data -> logging) > 0)
      {
         nalu_hypre_TFree(maxwell_data -> norms, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(maxwell_data -> rel_norms, NALU_HYPRE_MEMORY_HOST);
      }

      if ((maxwell_data -> edge_numlevels) > 0)
      {
         for (l = 0; l < (maxwell_data-> edge_numlevels); l++)
         {
            NALU_HYPRE_SStructGridDestroy(maxwell_data-> egrid_l[l]);
            nalu_hypre_ParVectorDestroy(maxwell_data-> rese_l[l]);
            nalu_hypre_ParVectorDestroy(maxwell_data-> ee_l[l]);
            nalu_hypre_ParVectorDestroy(maxwell_data-> eVtemp_l[l]);
            nalu_hypre_ParVectorDestroy(maxwell_data-> eVtemp2_l[l]);
            nalu_hypre_TFree(maxwell_data -> eCF_marker_l[l], NALU_HYPRE_MEMORY_HOST);

            /* Cannot destroy Aee_l[0] since it points to the user
               Aee_in. */
            if (l)
            {
               nalu_hypre_ParCSRMatrixDestroy(maxwell_data-> Aee_l[l]);
               nalu_hypre_ParVectorDestroy(maxwell_data-> be_l[l]);
               nalu_hypre_ParVectorDestroy(maxwell_data-> xe_l[l]);
            }

            if (l < (maxwell_data-> edge_numlevels) - 1)
            {
               NALU_HYPRE_IJMatrixDestroy(
                  (NALU_HYPRE_IJMatrix)  (maxwell_data-> Pe_l[l]));
            }

            nalu_hypre_TFree(maxwell_data-> BdryRanks_l[l], NALU_HYPRE_MEMORY_HOST);
         }
         nalu_hypre_TFree(maxwell_data-> egrid_l, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(maxwell_data-> Aee_l, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(maxwell_data-> be_l, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(maxwell_data-> xe_l, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(maxwell_data-> rese_l, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(maxwell_data-> ee_l, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(maxwell_data-> eVtemp_l, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(maxwell_data-> eVtemp2_l, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(maxwell_data-> Pe_l, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(maxwell_data-> ReT_l, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(maxwell_data-> eCF_marker_l, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(maxwell_data-> erelax_weight, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(maxwell_data-> eomega, NALU_HYPRE_MEMORY_HOST);

         nalu_hypre_TFree(maxwell_data-> BdryRanks_l, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(maxwell_data-> BdryRanksCnts_l, NALU_HYPRE_MEMORY_HOST);
      }

      if ((maxwell_data -> node_numlevels) > 0)
      {
         for (l = 0; l < (maxwell_data-> node_numlevels); l++)
         {
            nalu_hypre_ParVectorDestroy(maxwell_data-> resn_l[l]);
            nalu_hypre_ParVectorDestroy(maxwell_data-> en_l[l]);
            nalu_hypre_ParVectorDestroy(maxwell_data-> nVtemp_l[l]);
            nalu_hypre_ParVectorDestroy(maxwell_data-> nVtemp2_l[l]);
         }
         nalu_hypre_BoomerAMGDestroy(maxwell_data-> amg_vdata);

         nalu_hypre_TFree(maxwell_data-> Ann_l, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(maxwell_data-> Pn_l, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(maxwell_data-> RnT_l, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(maxwell_data-> bn_l, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(maxwell_data-> xn_l, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(maxwell_data-> resn_l, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(maxwell_data-> en_l, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(maxwell_data-> nVtemp_l, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(maxwell_data-> nVtemp2_l, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(maxwell_data-> nCF_marker_l, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(maxwell_data-> nrelax_weight, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(maxwell_data-> nomega, NALU_HYPRE_MEMORY_HOST);
      }

      NALU_HYPRE_SStructStencilDestroy(maxwell_data-> Ann_stencils[0]);
      nalu_hypre_TFree(maxwell_data-> Ann_stencils, NALU_HYPRE_MEMORY_HOST);

      if ((maxwell_data -> en_numlevels) > 0)
      {
         for (l = 1; l < (maxwell_data-> en_numlevels); l++)
         {
            nalu_hypre_ParCSRMatrixDestroy(maxwell_data-> Aen_l[l]);
         }
      }
      nalu_hypre_TFree(maxwell_data-> Aen_l, NALU_HYPRE_MEMORY_HOST);

      NALU_HYPRE_SStructVectorDestroy(
         (NALU_HYPRE_SStructVector) maxwell_data-> bn);
      NALU_HYPRE_SStructVectorDestroy(
         (NALU_HYPRE_SStructVector) maxwell_data-> xn);
      NALU_HYPRE_SStructMatrixDestroy(
         (NALU_HYPRE_SStructMatrix) maxwell_data-> Ann);
      NALU_HYPRE_IJMatrixDestroy(maxwell_data-> Aen);

      nalu_hypre_ParCSRMatrixDestroy(maxwell_data-> T_transpose);

      nalu_hypre_FinalizeTiming(maxwell_data -> time_index);
      nalu_hypre_TFree(maxwell_data, NALU_HYPRE_MEMORY_HOST);
   }

   return (ierr);
}

/*--------------------------------------------------------------------------
 * nalu_hypre_MaxwellSetRfactors
 *--------------------------------------------------------------------------*/
NALU_HYPRE_Int
nalu_hypre_MaxwellSetRfactors(void         *maxwell_vdata,
                         NALU_HYPRE_Int     rfactor[3] )
{
   nalu_hypre_MaxwellData *maxwell_data   = (nalu_hypre_MaxwellData *)maxwell_vdata;
   nalu_hypre_Index       *maxwell_rfactor = (maxwell_data -> rfactor);
   NALU_HYPRE_Int          ierr       = 0;

   nalu_hypre_CopyIndex(rfactor, maxwell_rfactor[0]);

   return ierr;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_MaxwellSetGrad
 *--------------------------------------------------------------------------*/
NALU_HYPRE_Int
nalu_hypre_MaxwellSetGrad(void               *maxwell_vdata,
                     nalu_hypre_ParCSRMatrix *T )
{
   nalu_hypre_MaxwellData *maxwell_data = (nalu_hypre_MaxwellData *)maxwell_vdata;
   NALU_HYPRE_Int          ierr       = 0;

   (maxwell_data -> Tgrad) =  T;

   return ierr;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_MaxwellSetConstantCoef
 *--------------------------------------------------------------------------*/
NALU_HYPRE_Int
nalu_hypre_MaxwellSetConstantCoef( void   *maxwell_vdata,
                              NALU_HYPRE_Int     constant_coef)
{
   nalu_hypre_MaxwellData *maxwell_data = (nalu_hypre_MaxwellData *)maxwell_vdata;
   NALU_HYPRE_Int          ierr        = 0;

   (maxwell_data -> constant_coef) = constant_coef;

   return ierr;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_MaxwellSetTol
 *--------------------------------------------------------------------------*/
NALU_HYPRE_Int
nalu_hypre_MaxwellSetTol( void   *maxwell_vdata,
                     NALU_HYPRE_Real  tol       )
{
   nalu_hypre_MaxwellData *maxwell_data = (nalu_hypre_MaxwellData *)maxwell_vdata;
   NALU_HYPRE_Int          ierr        = 0;

   (maxwell_data -> tol) = tol;

   return ierr;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_MaxwellSetMaxIter
 *--------------------------------------------------------------------------*/
NALU_HYPRE_Int
nalu_hypre_MaxwellSetMaxIter( void *maxwell_vdata,
                         NALU_HYPRE_Int   max_iter  )
{
   nalu_hypre_MaxwellData *maxwell_data = (nalu_hypre_MaxwellData *)maxwell_vdata;
   NALU_HYPRE_Int          ierr = 0;

   (maxwell_data -> max_iter) = max_iter;

   return ierr;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_MaxwellSetRelChange
 *--------------------------------------------------------------------------*/
NALU_HYPRE_Int
nalu_hypre_MaxwellSetRelChange( void *maxwell_vdata,
                           NALU_HYPRE_Int   rel_change  )
{
   nalu_hypre_MaxwellData *maxwell_data = (nalu_hypre_MaxwellData *)maxwell_vdata;
   NALU_HYPRE_Int          ierr = 0;

   (maxwell_data -> rel_change) = rel_change;

   return ierr;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_MaxwellNumPreRelax
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_MaxwellSetNumPreRelax( void *maxwell_vdata,
                             NALU_HYPRE_Int   num_pre_relax )
{
   nalu_hypre_MaxwellData *maxwell_data = (nalu_hypre_MaxwellData *)maxwell_vdata;
   NALU_HYPRE_Int          ierr = 0;

   (maxwell_data -> num_pre_relax) = num_pre_relax;

   return ierr;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_MaxwellSetNumPostRelax
 *--------------------------------------------------------------------------*/
NALU_HYPRE_Int
nalu_hypre_MaxwellSetNumPostRelax( void *maxwell_vdata,
                              NALU_HYPRE_Int   num_post_relax )
{
   nalu_hypre_MaxwellData *maxwell_data = (nalu_hypre_MaxwellData *)maxwell_vdata;
   NALU_HYPRE_Int          ierr = 0;

   (maxwell_data -> num_post_relax) = num_post_relax;

   return ierr;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_MaxwellGetNumIterations
 *--------------------------------------------------------------------------*/
NALU_HYPRE_Int
nalu_hypre_MaxwellGetNumIterations( void *maxwell_vdata,
                               NALU_HYPRE_Int  *num_iterations )
{
   nalu_hypre_MaxwellData *maxwell_data = (nalu_hypre_MaxwellData *)maxwell_vdata;
   NALU_HYPRE_Int          ierr = 0;

   *num_iterations = (maxwell_data -> num_iterations);

   return ierr;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_MaxwellSetPrintLevel
 *--------------------------------------------------------------------------*/
NALU_HYPRE_Int
nalu_hypre_MaxwellSetPrintLevel( void *maxwell_vdata,
                            NALU_HYPRE_Int   print_level)
{
   nalu_hypre_MaxwellData *maxwell_data = (nalu_hypre_MaxwellData *)maxwell_vdata;
   NALU_HYPRE_Int          ierr = 0;

   (maxwell_data -> print_level) = print_level;

   return ierr;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_MaxwellSetLogging
 *--------------------------------------------------------------------------*/
NALU_HYPRE_Int
nalu_hypre_MaxwellSetLogging( void *maxwell_vdata,
                         NALU_HYPRE_Int   logging)
{
   nalu_hypre_MaxwellData *maxwell_data = (nalu_hypre_MaxwellData *)maxwell_vdata;
   NALU_HYPRE_Int          ierr = 0;

   (maxwell_data -> logging) = logging;

   return ierr;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_MaxwellPrintLogging
 *--------------------------------------------------------------------------*/
NALU_HYPRE_Int
nalu_hypre_MaxwellPrintLogging( void *maxwell_vdata,
                           NALU_HYPRE_Int   myid)
{
   nalu_hypre_MaxwellData *maxwell_data = (nalu_hypre_MaxwellData *)maxwell_vdata;
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
               nalu_hypre_printf("Residual norm[%d] = %e   ", i, norms[i]);
               nalu_hypre_printf("Relative residual norm[%d] = %e\n", i, rel_norms[i]);
            }
         }
      }
   }

   return ierr;
}

NALU_HYPRE_Int
nalu_hypre_MaxwellGetFinalRelativeResidualNorm( void   *maxwell_vdata,
                                           NALU_HYPRE_Real *relative_residual_norm )
{
   nalu_hypre_MaxwellData *maxwell_data = (nalu_hypre_MaxwellData *)maxwell_vdata;

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
