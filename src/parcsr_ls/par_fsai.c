/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 *******************************************************************************/

#include "_nalu_hypre_parcsr_ls.h"
#include "par_fsai.h"

/******************************************************************************
 * NALU_HYPRE_FSAICreate
 ******************************************************************************/

void *
nalu_hypre_FSAICreate()
{
   nalu_hypre_ParFSAIData    *fsai_data;

   /* setup params */
   NALU_HYPRE_Int            algo_type;
   NALU_HYPRE_Int            max_steps;
   NALU_HYPRE_Int            max_step_size;
   NALU_HYPRE_Real           kap_tolerance;

   /* solver params */
   NALU_HYPRE_Int            eig_max_iters;
   NALU_HYPRE_Int            max_iterations;
   NALU_HYPRE_Int            num_iterations;
   NALU_HYPRE_Real           tolerance;
   NALU_HYPRE_Real           omega;

   /* log info */
   NALU_HYPRE_Int            logging;

   /* output params */
   NALU_HYPRE_Int            print_level;

   /*-----------------------------------------------------------------------
    * Setup default values for parameters
    *-----------------------------------------------------------------------*/
   fsai_data = nalu_hypre_CTAlloc(nalu_hypre_ParFSAIData, 1, NALU_HYPRE_MEMORY_HOST);

   /* setup params */
   algo_type = nalu_hypre_NumThreads() > 4 ? 2 : 1;
   max_steps = 3;
   max_step_size = 5;
   kap_tolerance = 1.0e-3;

   /* solver params */
   eig_max_iters = 0;
   max_iterations = 20;
   tolerance = 1.0e-6;
   omega = 1.0;

   /* log info */
   logging = 0;
   num_iterations = 0;

   /* output params */
   print_level = 0;

   NALU_HYPRE_ANNOTATE_FUNC_BEGIN;

   /*-----------------------------------------------------------------------
    * Create the nalu_hypre_ParFSAIData structure and return
    *-----------------------------------------------------------------------*/

   nalu_hypre_ParFSAIDataGmat(fsai_data)      = NULL;
   nalu_hypre_ParFSAIDataGTmat(fsai_data)     = NULL;
   nalu_hypre_ParFSAIDataRWork(fsai_data)     = NULL;
   nalu_hypre_ParFSAIDataZWork(fsai_data)     = NULL;
   nalu_hypre_ParFSAIDataZeroGuess(fsai_data) = 0;

   nalu_hypre_FSAISetAlgoType(fsai_data, algo_type);
   nalu_hypre_FSAISetMaxSteps(fsai_data, max_steps);
   nalu_hypre_FSAISetMaxStepSize(fsai_data, max_step_size);
   nalu_hypre_FSAISetKapTolerance(fsai_data, kap_tolerance);

   nalu_hypre_FSAISetMaxIterations(fsai_data, max_iterations);
   nalu_hypre_FSAISetEigMaxIters(fsai_data, eig_max_iters);
   nalu_hypre_FSAISetTolerance(fsai_data, tolerance);
   nalu_hypre_FSAISetOmega(fsai_data, omega);

   nalu_hypre_FSAISetLogging(fsai_data, logging);
   nalu_hypre_FSAISetNumIterations(fsai_data, num_iterations);

   nalu_hypre_FSAISetPrintLevel(fsai_data, print_level);

   NALU_HYPRE_ANNOTATE_FUNC_END;

   return (void *) fsai_data;
}

/******************************************************************************
 * NALU_HYPRE_FSAIDestroy
 ******************************************************************************/

NALU_HYPRE_Int
nalu_hypre_FSAIDestroy( void *data )
{
   nalu_hypre_ParFSAIData *fsai_data = (nalu_hypre_ParFSAIData*)data;

   NALU_HYPRE_ANNOTATE_FUNC_BEGIN;

   if (fsai_data)
   {
      if (nalu_hypre_ParFSAIDataGmat(fsai_data))
      {
         nalu_hypre_ParCSRMatrixDestroy(nalu_hypre_ParFSAIDataGmat(fsai_data));
      }

      if (nalu_hypre_ParFSAIDataGTmat(fsai_data))
      {
         nalu_hypre_ParCSRMatrixDestroy(nalu_hypre_ParFSAIDataGTmat(fsai_data));
      }

      nalu_hypre_ParVectorDestroy(nalu_hypre_ParFSAIDataRWork(fsai_data));
      nalu_hypre_ParVectorDestroy(nalu_hypre_ParFSAIDataZWork(fsai_data));

      nalu_hypre_TFree(fsai_data, NALU_HYPRE_MEMORY_HOST);
   }

   NALU_HYPRE_ANNOTATE_FUNC_END;

   return nalu_hypre_error_flag;
}

/******************************************************************************
 * Routines to SET the setup phase parameters
 ******************************************************************************/

NALU_HYPRE_Int
nalu_hypre_FSAISetAlgoType( void      *data,
                       NALU_HYPRE_Int  algo_type )
{
   nalu_hypre_ParFSAIData  *fsai_data = (nalu_hypre_ParFSAIData*) data;

   if (!fsai_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   if (algo_type < 0)
   {
      nalu_hypre_error_in_arg(2);
      return nalu_hypre_error_flag;
   }

   nalu_hypre_ParFSAIDataAlgoType(fsai_data) = algo_type;

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_FSAISetMaxSteps( void      *data,
                       NALU_HYPRE_Int  max_steps )
{
   nalu_hypre_ParFSAIData  *fsai_data = (nalu_hypre_ParFSAIData*) data;

   if (!fsai_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   if (max_steps < 0)
   {
      nalu_hypre_error_in_arg(2);
      return nalu_hypre_error_flag;
   }

   nalu_hypre_ParFSAIDataMaxSteps(fsai_data) = max_steps;

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_FSAISetMaxStepSize( void      *data,
                          NALU_HYPRE_Int  max_step_size )
{
   nalu_hypre_ParFSAIData  *fsai_data = (nalu_hypre_ParFSAIData*) data;

   if (!fsai_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   if (max_step_size < 0)
   {
      nalu_hypre_error_in_arg(2);
      return nalu_hypre_error_flag;
   }

   nalu_hypre_ParFSAIDataMaxStepSize(fsai_data) = max_step_size;

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_FSAISetKapTolerance( void       *data,
                           NALU_HYPRE_Real  kap_tolerance )
{
   nalu_hypre_ParFSAIData  *fsai_data = (nalu_hypre_ParFSAIData*) data;

   if (!fsai_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   if (kap_tolerance < 0)
   {
      nalu_hypre_error_in_arg(2);
      return nalu_hypre_error_flag;
   }

   nalu_hypre_ParFSAIDataKapTolerance(fsai_data) = kap_tolerance;

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_FSAISetMaxIterations( void      *data,
                            NALU_HYPRE_Int  max_iterations )
{
   nalu_hypre_ParFSAIData  *fsai_data = (nalu_hypre_ParFSAIData*) data;

   if (!fsai_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   if (max_iterations < 0)
   {
      nalu_hypre_error_in_arg(2);
      return nalu_hypre_error_flag;
   }

   nalu_hypre_ParFSAIDataMaxIterations(fsai_data) = max_iterations;

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_FSAISetEigMaxIters( void      *data,
                          NALU_HYPRE_Int  eig_max_iters )
{
   nalu_hypre_ParFSAIData  *fsai_data = (nalu_hypre_ParFSAIData*) data;

   if (!fsai_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   if (eig_max_iters < 0)
   {
      nalu_hypre_error_in_arg(2);
      return nalu_hypre_error_flag;
   }

   nalu_hypre_ParFSAIDataEigMaxIters(fsai_data) = eig_max_iters;

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_FSAISetZeroGuess( void     *data,
                        NALU_HYPRE_Int zero_guess )
{
   nalu_hypre_ParFSAIData  *fsai_data = (nalu_hypre_ParFSAIData*) data;

   if (!fsai_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   if (zero_guess != 0)
   {
      nalu_hypre_ParFSAIDataZeroGuess(fsai_data) = 1;
   }
   else
   {
      nalu_hypre_ParFSAIDataZeroGuess(fsai_data) = 0;
   }

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_FSAISetTolerance( void       *data,
                        NALU_HYPRE_Real  tolerance )
{
   nalu_hypre_ParFSAIData  *fsai_data = (nalu_hypre_ParFSAIData*) data;

   if (!fsai_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   if (tolerance < 0)
   {
      nalu_hypre_error_in_arg(2);
      return nalu_hypre_error_flag;
   }

   nalu_hypre_ParFSAIDataTolerance(fsai_data) = tolerance;

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_FSAISetOmega( void       *data,
                    NALU_HYPRE_Real  omega )
{
   nalu_hypre_ParFSAIData  *fsai_data = (nalu_hypre_ParFSAIData*) data;

   if (!fsai_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   if (omega < 0)
   {
      nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC, "Negative omega not allowed!");
      return nalu_hypre_error_flag;
   }

   nalu_hypre_ParFSAIDataOmega(fsai_data) = omega;

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_FSAISetLogging( void      *data,
                      NALU_HYPRE_Int  logging )
{
   /*   This function should be called before Setup.  Logging changes
    *    may require allocation or freeing of arrays, which is presently
    *    only done there.
    *    It may be possible to support logging changes at other times,
    *    but there is little need.
    */
   nalu_hypre_ParFSAIData  *fsai_data = (nalu_hypre_ParFSAIData*) data;

   if (!fsai_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   if (logging < 0)
   {
      nalu_hypre_error_in_arg(2);
      return nalu_hypre_error_flag;
   }

   nalu_hypre_ParFSAIDataLogging(fsai_data) = logging;

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_FSAISetNumIterations( void      *data,
                            NALU_HYPRE_Int  num_iterations )
{
   nalu_hypre_ParFSAIData  *fsai_data = (nalu_hypre_ParFSAIData*) data;

   if (!fsai_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   if (num_iterations < 0)
   {
      nalu_hypre_error_in_arg(2);
      return nalu_hypre_error_flag;
   }

   nalu_hypre_ParFSAIDataNumIterations(fsai_data) = num_iterations;

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_FSAISetPrintLevel( void      *data,
                         NALU_HYPRE_Int  print_level )
{
   nalu_hypre_ParFSAIData  *fsai_data = (nalu_hypre_ParFSAIData*) data;

   if (!fsai_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   if (print_level < 0)
   {
      nalu_hypre_error_in_arg(2);
      return nalu_hypre_error_flag;
   }

   nalu_hypre_ParFSAIDataPrintLevel(fsai_data) = print_level;

   return nalu_hypre_error_flag;
}

/******************************************************************************
 * Routines to GET the setup phase parameters
 ******************************************************************************/

NALU_HYPRE_Int
nalu_hypre_FSAIGetAlgoType( void      *data,
                       NALU_HYPRE_Int *algo_type )
{
   nalu_hypre_ParFSAIData  *fsai_data = (nalu_hypre_ParFSAIData*) data;

   if (!fsai_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   *algo_type = nalu_hypre_ParFSAIDataAlgoType(fsai_data);

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_FSAIGetMaxSteps( void      *data,
                       NALU_HYPRE_Int *algo_type )
{
   nalu_hypre_ParFSAIData  *fsai_data = (nalu_hypre_ParFSAIData*) data;

   if (!fsai_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   *algo_type = nalu_hypre_ParFSAIDataMaxSteps(fsai_data);

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_FSAIGetMaxStepSize( void      *data,
                          NALU_HYPRE_Int *max_step_size )
{
   nalu_hypre_ParFSAIData  *fsai_data = (nalu_hypre_ParFSAIData*) data;

   if (!fsai_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   *max_step_size = nalu_hypre_ParFSAIDataMaxStepSize(fsai_data);

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_FSAIGetKapTolerance( void       *data,
                           NALU_HYPRE_Real *kap_tolerance )
{
   nalu_hypre_ParFSAIData  *fsai_data = (nalu_hypre_ParFSAIData*) data;

   if (!fsai_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   *kap_tolerance = nalu_hypre_ParFSAIDataKapTolerance(fsai_data);

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_FSAIGetMaxIterations( void      *data,
                            NALU_HYPRE_Int *max_iterations )
{
   nalu_hypre_ParFSAIData  *fsai_data = (nalu_hypre_ParFSAIData*) data;

   if (!fsai_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   *max_iterations = nalu_hypre_ParFSAIDataMaxIterations(fsai_data);

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_FSAIGetEigMaxIters( void      *data,
                          NALU_HYPRE_Int *eig_max_iters )
{
   nalu_hypre_ParFSAIData  *fsai_data = (nalu_hypre_ParFSAIData*) data;

   if (!fsai_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   *eig_max_iters = nalu_hypre_ParFSAIDataEigMaxIters(fsai_data);

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_FSAIGetZeroGuess( void      *data,
                        NALU_HYPRE_Int *zero_guess )
{
   nalu_hypre_ParFSAIData  *fsai_data = (nalu_hypre_ParFSAIData*) data;

   if (!fsai_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   *zero_guess = nalu_hypre_ParFSAIDataZeroGuess(fsai_data);

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_FSAIGetTolerance( void       *data,
                        NALU_HYPRE_Real *tolerance )
{
   nalu_hypre_ParFSAIData  *fsai_data = (nalu_hypre_ParFSAIData*) data;

   if (!fsai_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   *tolerance = nalu_hypre_ParFSAIDataTolerance(fsai_data);

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_FSAIGetOmega( void       *data,
                    NALU_HYPRE_Real *omega )
{
   nalu_hypre_ParFSAIData  *fsai_data = (nalu_hypre_ParFSAIData*) data;

   if (!fsai_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   *omega = nalu_hypre_ParFSAIDataOmega(fsai_data);

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_FSAIGetLogging( void      *data,
                      NALU_HYPRE_Int *logging )
{
   nalu_hypre_ParFSAIData  *fsai_data = (nalu_hypre_ParFSAIData*) data;

   if (!fsai_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   *logging = nalu_hypre_ParFSAIDataLogging(fsai_data);

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_FSAIGetNumIterations( void      *data,
                            NALU_HYPRE_Int *num_iterations )
{
   nalu_hypre_ParFSAIData  *fsai_data = (nalu_hypre_ParFSAIData*) data;

   if (!fsai_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   *num_iterations = nalu_hypre_ParFSAIDataNumIterations(fsai_data);

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_FSAIGetPrintLevel( void      *data,
                         NALU_HYPRE_Int *print_level )
{
   nalu_hypre_ParFSAIData  *fsai_data = (nalu_hypre_ParFSAIData*) data;

   if (!fsai_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   *print_level = nalu_hypre_ParFSAIDataPrintLevel(fsai_data);

   return nalu_hypre_error_flag;
}
