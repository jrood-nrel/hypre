/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_nalu_hypre_utilities.h"

/* Global variable for error handling */
nalu_hypre_Error nalu_hypre__global_error = {0, 0, NULL, 0, 0};

/*--------------------------------------------------------------------------
 * Process the error raised on the given line of the given source file
 *--------------------------------------------------------------------------*/

void nalu_hypre_error_handler(const char *filename, NALU_HYPRE_Int line, NALU_HYPRE_Int ierr, const char *msg)
{
   /* Copy global struct into a short name and copy changes back before exiting */
   nalu_hypre_Error err = nalu_hypre__global_error;

   /* Store the error code */
   err.error_flag |= ierr;

#ifdef NALU_HYPRE_PRINT_ERRORS

   /* Error format strings without and with a message */
   const char fmt_wo[] = "hypre error in file \"%s\", line %d, error code = %d\n";
   const char fmt_wm[] = "hypre error in file \"%s\", line %d, error code = %d - %s\n";

   NALU_HYPRE_Int bufsz = 0;

   /* Print error message to local buffer first */

   if (msg)
   {
      bufsz = nalu_hypre_snprintf(NULL, 0, fmt_wm, filename, line, ierr, msg);
   }
   else
   {
      bufsz = nalu_hypre_snprintf(NULL, 0, fmt_wo, filename, line, ierr);
   }

   bufsz += 1;
   char buffer[bufsz];

   if (msg)
   {
      nalu_hypre_snprintf(buffer, bufsz, fmt_wm, filename, line, ierr, msg);
   }
   else
   {
      nalu_hypre_snprintf(buffer, bufsz, fmt_wo, filename, line, ierr);
   }

   /* Now print buffer to either memory or stderr */

   if (err.print_to_memory)
   {
      NALU_HYPRE_Int  msg_sz = err.msg_sz; /* Store msg_sz for snprintf below */

      /* Make sure there is enough memory for the new message */
      err.msg_sz += bufsz;
      if ( err.msg_sz > err.mem_sz )
      {
         err.mem_sz = err.msg_sz + 1024; /* Add some excess */
         err.memory = nalu_hypre_TReAlloc(err.memory, char, err.mem_sz, NALU_HYPRE_MEMORY_HOST);
      }

      nalu_hypre_snprintf((err.memory + msg_sz), bufsz, "%s", buffer);
   }
   else
   {
      nalu_hypre_fprintf(stderr, "%s", buffer);
   }

#endif

   nalu_hypre__global_error = err;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int NALU_HYPRE_GetError(void)
{
   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int NALU_HYPRE_CheckError(NALU_HYPRE_Int ierr, NALU_HYPRE_Int nalu_hypre_error_code)
{
   return ierr & nalu_hypre_error_code;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

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

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int NALU_HYPRE_GetErrorArg(void)
{
   return (nalu_hypre_error_flag >> 3 & 31);
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int NALU_HYPRE_ClearAllErrors(void)
{
   nalu_hypre_error_flag = 0;
   return (nalu_hypre_error_flag != 0);
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int NALU_HYPRE_ClearError(NALU_HYPRE_Int nalu_hypre_error_code)
{
   nalu_hypre_error_flag &= ~nalu_hypre_error_code;
   return (nalu_hypre_error_flag & nalu_hypre_error_code);
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int NALU_HYPRE_SetPrintErrorMode(NALU_HYPRE_Int mode)
{
   nalu_hypre__global_error.print_to_memory = mode;
   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int NALU_HYPRE_GetErrorMessages(char **buffer, NALU_HYPRE_Int *bufsz)
{
   nalu_hypre_Error err = nalu_hypre__global_error;

   *bufsz  = err.msg_sz;
   *buffer = nalu_hypre_CTAlloc(char, *bufsz, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TMemcpy(*buffer, err.memory, char, *bufsz, NALU_HYPRE_MEMORY_HOST, NALU_HYPRE_MEMORY_HOST);

   nalu_hypre_TFree(err.memory, NALU_HYPRE_MEMORY_HOST);
   err.mem_sz = 0;
   err.msg_sz = 0;

   nalu_hypre__global_error = err;
   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int NALU_HYPRE_PrintErrorMessages(MPI_Comm comm)
{
   nalu_hypre_Error err = nalu_hypre__global_error;

   NALU_HYPRE_Int myid;
   char *msg;

   nalu_hypre_MPI_Barrier(comm);

   nalu_hypre_MPI_Comm_rank(comm, &myid);
   for (msg = err.memory; msg < (err.memory + err.msg_sz); msg += strlen(msg) + 1)
   {
      nalu_hypre_fprintf(stderr, "%d: %s", myid, msg);
   }

   nalu_hypre_TFree(err.memory, NALU_HYPRE_MEMORY_HOST);
   err.mem_sz = 0;
   err.msg_sz = 0;

   nalu_hypre__global_error = err;
   return nalu_hypre_error_flag;
}
