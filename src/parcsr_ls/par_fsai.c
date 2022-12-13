/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 *******************************************************************************/

#include "_hypre_parcsr_ls.h"
#include "par_fsai.h"

/******************************************************************************
 * NALU_HYPRE_FSAICreate
 ******************************************************************************/

void *
hypre_FSAICreate()
{
   hypre_ParFSAIData    *fsai_data;

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
   fsai_data = hypre_CTAlloc(hypre_ParFSAIData, 1, NALU_HYPRE_MEMORY_HOST);

   /* setup params */
   algo_type = hypre_NumThreads() > 4 ? 2 : 1;
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
    * Create the hypre_ParFSAIData structure and return
    *-----------------------------------------------------------------------*/

   hypre_ParFSAIDataGmat(fsai_data)      = NULL;
   hypre_ParFSAIDataGTmat(fsai_data)     = NULL;
   hypre_ParFSAIDataRWork(fsai_data)     = NULL;
   hypre_ParFSAIDataZWork(fsai_data)     = NULL;
   hypre_ParFSAIDataZeroGuess(fsai_data) = 0;

   hypre_FSAISetAlgoType(fsai_data, algo_type);
   hypre_FSAISetMaxSteps(fsai_data, max_steps);
   hypre_FSAISetMaxStepSize(fsai_data, max_step_size);
   hypre_FSAISetKapTolerance(fsai_data, kap_tolerance);

   hypre_FSAISetMaxIterations(fsai_data, max_iterations);
   hypre_FSAISetEigMaxIters(fsai_data, eig_max_iters);
   hypre_FSAISetTolerance(fsai_data, tolerance);
   hypre_FSAISetOmega(fsai_data, omega);

   hypre_FSAISetLogging(fsai_data, logging);
   hypre_FSAISetNumIterations(fsai_data, num_iterations);

   hypre_FSAISetPrintLevel(fsai_data, print_level);

   NALU_HYPRE_ANNOTATE_FUNC_END;

   return (void *) fsai_data;
}

/******************************************************************************
 * NALU_HYPRE_FSAIDestroy
 ******************************************************************************/

NALU_HYPRE_Int
hypre_FSAIDestroy( void *data )
{
   hypre_ParFSAIData *fsai_data = (hypre_ParFSAIData*)data;

   NALU_HYPRE_ANNOTATE_FUNC_BEGIN;

   if (fsai_data)
   {
      if (hypre_ParFSAIDataGmat(fsai_data))
      {
         hypre_ParCSRMatrixDestroy(hypre_ParFSAIDataGmat(fsai_data));
      }

      if (hypre_ParFSAIDataGTmat(fsai_data))
      {
         hypre_ParCSRMatrixDestroy(hypre_ParFSAIDataGTmat(fsai_data));
      }

      hypre_ParVectorDestroy(hypre_ParFSAIDataRWork(fsai_data));
      hypre_ParVectorDestroy(hypre_ParFSAIDataZWork(fsai_data));

      hypre_TFree(fsai_data, NALU_HYPRE_MEMORY_HOST);
   }

   NALU_HYPRE_ANNOTATE_FUNC_END;

   return hypre_error_flag;
}

/******************************************************************************
 * Routines to SET the setup phase parameters
 ******************************************************************************/

NALU_HYPRE_Int
hypre_FSAISetAlgoType( void      *data,
                       NALU_HYPRE_Int  algo_type )
{
   hypre_ParFSAIData  *fsai_data = (hypre_ParFSAIData*) data;

   if (!fsai_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   if (algo_type < 0)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }

   hypre_ParFSAIDataAlgoType(fsai_data) = algo_type;

   return hypre_error_flag;
}

NALU_HYPRE_Int
hypre_FSAISetMaxSteps( void      *data,
                       NALU_HYPRE_Int  max_steps )
{
   hypre_ParFSAIData  *fsai_data = (hypre_ParFSAIData*) data;

   if (!fsai_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   if (max_steps < 0)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }

   hypre_ParFSAIDataMaxSteps(fsai_data) = max_steps;

   return hypre_error_flag;
}

NALU_HYPRE_Int
hypre_FSAISetMaxStepSize( void      *data,
                          NALU_HYPRE_Int  max_step_size )
{
   hypre_ParFSAIData  *fsai_data = (hypre_ParFSAIData*) data;

   if (!fsai_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   if (max_step_size < 0)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }

   hypre_ParFSAIDataMaxStepSize(fsai_data) = max_step_size;

   return hypre_error_flag;
}

NALU_HYPRE_Int
hypre_FSAISetKapTolerance( void       *data,
                           NALU_HYPRE_Real  kap_tolerance )
{
   hypre_ParFSAIData  *fsai_data = (hypre_ParFSAIData*) data;

   if (!fsai_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   if (kap_tolerance < 0)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }

   hypre_ParFSAIDataKapTolerance(fsai_data) = kap_tolerance;

   return hypre_error_flag;
}

NALU_HYPRE_Int
hypre_FSAISetMaxIterations( void      *data,
                            NALU_HYPRE_Int  max_iterations )
{
   hypre_ParFSAIData  *fsai_data = (hypre_ParFSAIData*) data;

   if (!fsai_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   if (max_iterations < 0)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }

   hypre_ParFSAIDataMaxIterations(fsai_data) = max_iterations;

   return hypre_error_flag;
}

NALU_HYPRE_Int
hypre_FSAISetEigMaxIters( void      *data,
                          NALU_HYPRE_Int  eig_max_iters )
{
   hypre_ParFSAIData  *fsai_data = (hypre_ParFSAIData*) data;

   if (!fsai_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   if (eig_max_iters < 0)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }

   hypre_ParFSAIDataEigMaxIters(fsai_data) = eig_max_iters;

   return hypre_error_flag;
}

NALU_HYPRE_Int
hypre_FSAISetZeroGuess( void     *data,
                        NALU_HYPRE_Int zero_guess )
{
   hypre_ParFSAIData  *fsai_data = (hypre_ParFSAIData*) data;

   if (!fsai_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   if (zero_guess != 0)
   {
      hypre_ParFSAIDataZeroGuess(fsai_data) = 1;
   }
   else
   {
      hypre_ParFSAIDataZeroGuess(fsai_data) = 0;
   }

   return hypre_error_flag;
}

NALU_HYPRE_Int
hypre_FSAISetTolerance( void       *data,
                        NALU_HYPRE_Real  tolerance )
{
   hypre_ParFSAIData  *fsai_data = (hypre_ParFSAIData*) data;

   if (!fsai_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   if (tolerance < 0)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }

   hypre_ParFSAIDataTolerance(fsai_data) = tolerance;

   return hypre_error_flag;
}

NALU_HYPRE_Int
hypre_FSAISetOmega( void       *data,
                    NALU_HYPRE_Real  omega )
{
   hypre_ParFSAIData  *fsai_data = (hypre_ParFSAIData*) data;

   if (!fsai_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   if (omega < 0)
   {
      hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC, "Negative omega not allowed!");
      return hypre_error_flag;
   }

   hypre_ParFSAIDataOmega(fsai_data) = omega;

   return hypre_error_flag;
}

NALU_HYPRE_Int
hypre_FSAISetLogging( void      *data,
                      NALU_HYPRE_Int  logging )
{
   /*   This function should be called before Setup.  Logging changes
    *    may require allocation or freeing of arrays, which is presently
    *    only done there.
    *    It may be possible to support logging changes at other times,
    *    but there is little need.
    */
   hypre_ParFSAIData  *fsai_data = (hypre_ParFSAIData*) data;

   if (!fsai_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   if (logging < 0)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }

   hypre_ParFSAIDataLogging(fsai_data) = logging;

   return hypre_error_flag;
}

NALU_HYPRE_Int
hypre_FSAISetNumIterations( void      *data,
                            NALU_HYPRE_Int  num_iterations )
{
   hypre_ParFSAIData  *fsai_data = (hypre_ParFSAIData*) data;

   if (!fsai_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   if (num_iterations < 0)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }

   hypre_ParFSAIDataNumIterations(fsai_data) = num_iterations;

   return hypre_error_flag;
}

NALU_HYPRE_Int
hypre_FSAISetPrintLevel( void      *data,
                         NALU_HYPRE_Int  print_level )
{
   hypre_ParFSAIData  *fsai_data = (hypre_ParFSAIData*) data;

   if (!fsai_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   if (print_level < 0)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }

   hypre_ParFSAIDataPrintLevel(fsai_data) = print_level;

   return hypre_error_flag;
}

/******************************************************************************
 * Routines to GET the setup phase parameters
 ******************************************************************************/

NALU_HYPRE_Int
hypre_FSAIGetAlgoType( void      *data,
                       NALU_HYPRE_Int *algo_type )
{
   hypre_ParFSAIData  *fsai_data = (hypre_ParFSAIData*) data;

   if (!fsai_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   *algo_type = hypre_ParFSAIDataAlgoType(fsai_data);

   return hypre_error_flag;
}

NALU_HYPRE_Int
hypre_FSAIGetMaxSteps( void      *data,
                       NALU_HYPRE_Int *algo_type )
{
   hypre_ParFSAIData  *fsai_data = (hypre_ParFSAIData*) data;

   if (!fsai_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   *algo_type = hypre_ParFSAIDataMaxSteps(fsai_data);

   return hypre_error_flag;
}

NALU_HYPRE_Int
hypre_FSAIGetMaxStepSize( void      *data,
                          NALU_HYPRE_Int *max_step_size )
{
   hypre_ParFSAIData  *fsai_data = (hypre_ParFSAIData*) data;

   if (!fsai_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   *max_step_size = hypre_ParFSAIDataMaxStepSize(fsai_data);

   return hypre_error_flag;
}

NALU_HYPRE_Int
hypre_FSAIGetKapTolerance( void       *data,
                           NALU_HYPRE_Real *kap_tolerance )
{
   hypre_ParFSAIData  *fsai_data = (hypre_ParFSAIData*) data;

   if (!fsai_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   *kap_tolerance = hypre_ParFSAIDataKapTolerance(fsai_data);

   return hypre_error_flag;
}

NALU_HYPRE_Int
hypre_FSAIGetMaxIterations( void      *data,
                            NALU_HYPRE_Int *max_iterations )
{
   hypre_ParFSAIData  *fsai_data = (hypre_ParFSAIData*) data;

   if (!fsai_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   *max_iterations = hypre_ParFSAIDataMaxIterations(fsai_data);

   return hypre_error_flag;
}

NALU_HYPRE_Int
hypre_FSAIGetEigMaxIters( void      *data,
                          NALU_HYPRE_Int *eig_max_iters )
{
   hypre_ParFSAIData  *fsai_data = (hypre_ParFSAIData*) data;

   if (!fsai_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   *eig_max_iters = hypre_ParFSAIDataEigMaxIters(fsai_data);

   return hypre_error_flag;
}

NALU_HYPRE_Int
hypre_FSAIGetZeroGuess( void      *data,
                        NALU_HYPRE_Int *zero_guess )
{
   hypre_ParFSAIData  *fsai_data = (hypre_ParFSAIData*) data;

   if (!fsai_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   *zero_guess = hypre_ParFSAIDataZeroGuess(fsai_data);

   return hypre_error_flag;
}

NALU_HYPRE_Int
hypre_FSAIGetTolerance( void       *data,
                        NALU_HYPRE_Real *tolerance )
{
   hypre_ParFSAIData  *fsai_data = (hypre_ParFSAIData*) data;

   if (!fsai_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   *tolerance = hypre_ParFSAIDataTolerance(fsai_data);

   return hypre_error_flag;
}

NALU_HYPRE_Int
hypre_FSAIGetOmega( void       *data,
                    NALU_HYPRE_Real *omega )
{
   hypre_ParFSAIData  *fsai_data = (hypre_ParFSAIData*) data;

   if (!fsai_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   *omega = hypre_ParFSAIDataOmega(fsai_data);

   return hypre_error_flag;
}

NALU_HYPRE_Int
hypre_FSAIGetLogging( void      *data,
                      NALU_HYPRE_Int *logging )
{
   hypre_ParFSAIData  *fsai_data = (hypre_ParFSAIData*) data;

   if (!fsai_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   *logging = hypre_ParFSAIDataLogging(fsai_data);

   return hypre_error_flag;
}

NALU_HYPRE_Int
hypre_FSAIGetNumIterations( void      *data,
                            NALU_HYPRE_Int *num_iterations )
{
   hypre_ParFSAIData  *fsai_data = (hypre_ParFSAIData*) data;

   if (!fsai_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   *num_iterations = hypre_ParFSAIDataNumIterations(fsai_data);

   return hypre_error_flag;
}

NALU_HYPRE_Int
hypre_FSAIGetPrintLevel( void      *data,
                         NALU_HYPRE_Int *print_level )
{
   hypre_ParFSAIData  *fsai_data = (hypre_ParFSAIData*) data;

   if (!fsai_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   *print_level = hypre_ParFSAIDataPrintLevel(fsai_data);

   return hypre_error_flag;
}
