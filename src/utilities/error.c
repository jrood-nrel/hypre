/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_nalu_hypre_utilities.h"

NALU_HYPRE_Int nalu_hypre__global_error = 0;

/* Process the error with code ierr raised in the given line of the
   given source file. */
void nalu_hypre_error_handler(const char *filename, NALU_HYPRE_Int line, NALU_HYPRE_Int ierr, const char *msg)
{
   nalu_hypre_error_flag |= ierr;

#ifdef NALU_HYPRE_PRINT_ERRORS
   if (msg)
   {
      nalu_hypre_fprintf(
         stderr, "hypre error in file \"%s\", line %d, error code = %d - %s\n",
         filename, line, ierr, msg);
   }
   else
   {
      nalu_hypre_fprintf(
         stderr, "hypre error in file \"%s\", line %d, error code = %d\n",
         filename, line, ierr);
   }
#endif
}

NALU_HYPRE_Int NALU_HYPRE_GetError()
{
   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int NALU_HYPRE_CheckError(NALU_HYPRE_Int ierr, NALU_HYPRE_Int nalu_hypre_error_code)
{
   return ierr & nalu_hypre_error_code;
}

void NALU_HYPRE_DescribeError(NALU_HYPRE_Int ierr, char *msg)
{
   if (ierr == 0)
   {
      nalu_hypre_sprintf(msg, "[No error] ");
   }

   if (ierr & NALU_HYPRE_ERROR_GENERIC)
   {
      nalu_hypre_sprintf(msg, "[Generic error] ");
   }

   if (ierr & NALU_HYPRE_ERROR_MEMORY)
   {
      nalu_hypre_sprintf(msg, "[Memory error] ");
   }

   if (ierr & NALU_HYPRE_ERROR_ARG)
   {
      nalu_hypre_sprintf(msg, "[Error in argument %d] ", NALU_HYPRE_GetErrorArg());
   }

   if (ierr & NALU_HYPRE_ERROR_CONV)
   {
      nalu_hypre_sprintf(msg, "[Method did not converge] ");
   }
}

NALU_HYPRE_Int NALU_HYPRE_GetErrorArg()
{
   return (nalu_hypre_error_flag >> 3 & 31);
}

NALU_HYPRE_Int NALU_HYPRE_ClearAllErrors()
{
   nalu_hypre_error_flag = 0;
   return (nalu_hypre_error_flag != 0);
}

NALU_HYPRE_Int NALU_HYPRE_ClearError(NALU_HYPRE_Int nalu_hypre_error_code)
{
   nalu_hypre_error_flag &= ~nalu_hypre_error_code;
   return (nalu_hypre_error_flag & nalu_hypre_error_code);
}

