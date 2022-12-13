/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "./NALU_HYPRE_parcsr_ls.h"
#include "../matrix_matrix/NALU_HYPRE_matrix_matrix_protos.h"
#include "_hypre_utilities.h"

/* Must include implementation definition for ParVector since no data access
  functions are publically provided. AJC, 5/99 */
/* Likewise for Vector. AJC, 5/99 */
#include "../seq_mv/vector.h"

/* AB 8/06 - replace header file */
/* #include "../parcsr_mv/par_vector.h" */
#include "../parcsr_mv/_hypre_parcsr_mv.h"

/* These are what we need from Euclid */
#include "distributed_ls/Euclid/_hypre_Euclid.h"
/* #include "../distributed_ls/Euclid/Mem_dh.h" */
/* #include "../distributed_ls/Euclid/io_dh.h" */
/* #include "../distributed_ls/Euclid/TimeLog_dh.h" */
/* #include "../distributed_ls/Euclid/Parser_dh.h" */
/* #include "../distributed_ls/Euclid/Euclid_dh.h" */

/*------------------------------------------------------------------
 * Error checking
 *------------------------------------------------------------------*/

#define NALU_HYPRE_EUCLID_ERRCHKA \
          if (errFlag_dh) {  \
            setError_dh("", __FUNC__, __FILE__, __LINE__); \
            printErrorMsg(stderr);  \
            hypre_MPI_Abort(comm_dh, -1); \
          }

/* What is best to do here?
 * What is HYPRE's error checking strategy?
 * The shadow knows . . .
 *
 * Note: NALU_HYPRE_EUCLID_ERRCHKA macro is only used within this file.
 *
 * Note: "printErrorMsg(stderr)" is O.K. for debugging and
 *        development, possibly not for production.  This
 *        call causes Euclid to print a function call stack
 *        trace that led to the error.  (Potentially, each
 *        MPI task could print a trace.)
 *
 * Note: the __FUNC__ defines at the beginning of the function
 *       calls are used in Euclid's internal error-checking scheme.
 *       The "START_FUNC_DH" and "END_FUNC_VAL" macros are
 *       used for debugging: when "logFuncsToStderr == true"
 *       a function call trace is force-written to stderr;
 *       (useful for debugging over dial-up lines!)  See
 *       src/distributed_ls/Euclid/macros_dh.h and
 *       src/distributed_ls/Euclid/src/globalObjects.c
 *       for further info.
 */


/*--------------------------------------------------------------------------
 * debugging: if ENABLE_EUCLID_LOGGING is defined, each MPI task will open
 * "logFile.id" for writing; also, function-call tracing is operational
 * (ie, you can set logFuncsToFile = true, logFuncsToSterr = true).
 *
 *--------------------------------------------------------------------------*/
#undef ENABLE_EUCLID_LOGGING

#if !defined(ENABLE_EUCLID_LOGGING)
#undef START_FUNC_DH
#undef END_FUNC_VAL
#undef END_FUNC_DH
#define START_FUNC_DH     /**/
#define END_FUNC_DH       /**/
#define END_FUNC_VAL(a)   return(a);
#endif


/*--------------------------------------------------------------------------
 * NALU_HYPRE_EuclidCreate - Return a Euclid "solver".
 *--------------------------------------------------------------------------*/

#undef __FUNC__
#define __FUNC__ "NALU_HYPRE_EuclidCreate"
NALU_HYPRE_Int
NALU_HYPRE_EuclidCreate( MPI_Comm comm,
                    NALU_HYPRE_Solver *solver )
{
#ifdef NALU_HYPRE_MIXEDINT
   hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC, "Euclid cannot be used in mixedint mode!");
   return hypre_error_flag;
#else

   START_FUNC_DH
   Euclid_dh eu;

   /*-----------------------------------------------------------
    * create a few global objects (yuck!) for Euclid's use;
    * these  are all pointers, are initially NULL, and are be set
    * back to NULL in NALU_HYPRE_EuclidDestroy()
    * Global objects are defined in
    * src/distributed_ls/Euclid/src/globalObjects.c
    *-----------------------------------------------------------*/

   comm_dh = comm;
   hypre_MPI_Comm_size(comm_dh, &np_dh);    NALU_HYPRE_EUCLID_ERRCHKA;
   hypre_MPI_Comm_rank(comm_dh, &myid_dh);  NALU_HYPRE_EUCLID_ERRCHKA;

#ifdef ENABLE_EUCLID_LOGGING
   openLogfile_dh(0, NULL); NALU_HYPRE_EUCLID_ERRCHKA;
#endif

   if (mem_dh == NULL)
   {
      Mem_dhCreate(&mem_dh);  NALU_HYPRE_EUCLID_ERRCHKA;
   }

   if (tlog_dh == NULL)
   {
      TimeLog_dhCreate(&tlog_dh); NALU_HYPRE_EUCLID_ERRCHKA;
   }

   if (parser_dh == NULL)
   {
      Parser_dhCreate(&parser_dh); NALU_HYPRE_EUCLID_ERRCHKA;
   }
   Parser_dhInit(parser_dh, 0, NULL); NALU_HYPRE_EUCLID_ERRCHKA;

   /*-----------------------------------------------------------
    * create and return a Euclid object
    *-----------------------------------------------------------*/
   Euclid_dhCreate(&eu); NALU_HYPRE_EUCLID_ERRCHKA;
   *solver = (NALU_HYPRE_Solver) eu;

   END_FUNC_VAL(0)
#endif
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_EuclidDestroy - Destroy a Euclid object.
 *--------------------------------------------------------------------------*/

#undef __FUNC__
#define __FUNC__ "NALU_HYPRE_EuclidDestroy"
NALU_HYPRE_Int
NALU_HYPRE_EuclidDestroy( NALU_HYPRE_Solver solver )
{
#ifdef NALU_HYPRE_MIXEDINT
   hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC, "Euclid cannot be used in mixedint mode!");
   return hypre_error_flag;
#else

   START_FUNC_DH
   Euclid_dh eu = (Euclid_dh)solver;
   bool printMemReport = false;
   bool printStats = false;
   bool logging = eu->logging;

   /*----------------------------------------------------------------
      this block is for printing test data; this is used
      for diffing in autotests.
    *---------------------------------------------------------------- */
   if (Parser_dhHasSwitch(parser_dh, "-printTestData"))
   {
      FILE *fp;

      /* get filename to which to write report */
      char fname[] = "test_data_dh.temp", *fnamePtr = fname;
      Parser_dhReadString(parser_dh, "-printTestData", &fnamePtr); NALU_HYPRE_EUCLID_ERRCHKA;
      if (!strcmp(fnamePtr, "1"))    /* in case usr didn't supply a name! */
      {
         fnamePtr = fname;
      }

      /* print the report */
      fp = openFile_dh(fnamePtr, "w"); NALU_HYPRE_EUCLID_ERRCHKA;
      Euclid_dhPrintTestData(eu, fp); NALU_HYPRE_EUCLID_ERRCHKA;
      closeFile_dh(fp); NALU_HYPRE_EUCLID_ERRCHKA;

      printf_dh("\n@@@@@ Euclid test data was printed to file: %s\n\n", fnamePtr);
   }


   /*----------------------------------------------------------------
      determine which of Euclid's internal reports to print
    *----------------------------------------------------------------*/
   if (logging)
   {
      printStats = true;
      printMemReport = true;
   }
   if (parser_dh != NULL)
   {
      if (Parser_dhHasSwitch(parser_dh, "-eu_stats"))
      {
         printStats = true;
      }
      if (Parser_dhHasSwitch(parser_dh, "-eu_mem"))
      {
         printMemReport = true;
      }
   }

   /*------------------------------------------------------------------
      print Euclid's internal report, then destroy the Euclid object
    *------------------------------------------------------------------ */
   if (printStats)
   {
      Euclid_dhPrintHypreReport(eu, stdout); NALU_HYPRE_EUCLID_ERRCHKA;
   }
   Euclid_dhDestroy(eu); NALU_HYPRE_EUCLID_ERRCHKA;


   /*------------------------------------------------------------------
      destroy all remaining Euclid library objects
      (except the memory object)
    *------------------------------------------------------------------ */
   /*if (parser_dh != NULL) { dah 3/16/06  */
   if (parser_dh != NULL && ref_counter == 0)
   {
      Parser_dhDestroy(parser_dh); NALU_HYPRE_EUCLID_ERRCHKA;
      parser_dh = NULL;
   }

   /*if (tlog_dh != NULL) {  dah 3/16/06  */
   if (tlog_dh != NULL && ref_counter == 0)
   {
      TimeLog_dhDestroy(tlog_dh); NALU_HYPRE_EUCLID_ERRCHKA;
      tlog_dh = NULL;
   }

   /*------------------------------------------------------------------
      optionally print Euclid's memory report,
      then destroy the memory object.
    *------------------------------------------------------------------ */
   /*if (mem_dh != NULL) {  dah 3/16/06  */
   if (mem_dh != NULL && ref_counter == 0)
   {
      if (printMemReport)
      {
         Mem_dhPrint(mem_dh, stdout, false); NALU_HYPRE_EUCLID_ERRCHKA;
      }
      Mem_dhDestroy(mem_dh);  NALU_HYPRE_EUCLID_ERRCHKA;
      mem_dh = NULL;
   }

#ifdef ENABLE_EUCLID_LOGGING
   closeLogfile_dh(); NALU_HYPRE_EUCLID_ERRCHKA;
#endif

   END_FUNC_VAL(0)
#endif
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_EuclidSetup - Set up function for Euclid.
 *--------------------------------------------------------------------------*/

#undef __FUNC__
#define __FUNC__ "NALU_HYPRE_EuclidSetup"
NALU_HYPRE_Int
NALU_HYPRE_EuclidSetup( NALU_HYPRE_Solver solver,
                   NALU_HYPRE_ParCSRMatrix A,
                   NALU_HYPRE_ParVector b,
                   NALU_HYPRE_ParVector x   )
{
#ifdef NALU_HYPRE_MIXEDINT
   hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC, "Euclid cannot be used in mixedint mode!");
   return hypre_error_flag;
#else

   START_FUNC_DH
   Euclid_dh eu = (Euclid_dh)solver;


#if 0

   for testing!
{
   NALU_HYPRE_Int ierr;
   NALU_HYPRE_Int m, n, rs, re, cs, ce;

   NALU_HYPRE_DistributedMatrix mat;
   ierr = NALU_HYPRE_ConvertParCSRMatrixToDistributedMatrix( A, &mat );
      if (ierr) { exit(-1); }

      ierr = NALU_HYPRE_DistributedMatrixGetDims(mat, &m, &n);
      ierr = NALU_HYPRE_DistributedMatrixGetLocalRange(mat, &rs, &re,
                                                  &cs, &ce);

      hypre_printf("\n### [%i] m= %i, n= %i, rs= %i, re= %i, cs= %i, ce= %i\n",
                   myid_dh, m, n, rs, re, cs, ce);

      ierr = NALU_HYPRE_DistributedMatrixDestroy(mat);

      if (ierr) { exit(-1); }
   }
#endif

   Euclid_dhInputHypreMat(eu, A); NALU_HYPRE_EUCLID_ERRCHKA;
   Euclid_dhSetup(eu); NALU_HYPRE_EUCLID_ERRCHKA;

   END_FUNC_VAL(0)
#endif
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_EuclidSolve - Solve function for Euclid.
 *--------------------------------------------------------------------------*/

#undef __FUNC__
#define __FUNC__ "NALU_HYPRE_EuclidSolve"
NALU_HYPRE_Int
NALU_HYPRE_EuclidSolve( NALU_HYPRE_Solver solver,
                   NALU_HYPRE_ParCSRMatrix A,
                   NALU_HYPRE_ParVector bb,
                   NALU_HYPRE_ParVector xx  )
{
#ifdef NALU_HYPRE_MIXEDINT
   hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC, "Euclid cannot be used in mixedint mode!");
   return hypre_error_flag;
#else

   START_FUNC_DH
   Euclid_dh eu = (Euclid_dh)solver;
   NALU_HYPRE_Real *b, *x;

   x = hypre_VectorData(hypre_ParVectorLocalVector((hypre_ParVector *) bb));
   b = hypre_VectorData(hypre_ParVectorLocalVector((hypre_ParVector *) xx));

   Euclid_dhApply(eu, x, b); NALU_HYPRE_EUCLID_ERRCHKA;
   END_FUNC_VAL(0)
#endif
}

/*--------------------------------------------------------------------------
 * Insert command line (flag, value) pairs in Euclid's
 *--------------------------------------------------------------------------*/

#undef __FUNC__
#define __FUNC__ "NALU_HYPRE_EuclidSetParams"
NALU_HYPRE_Int
NALU_HYPRE_EuclidSetParams(NALU_HYPRE_Solver solver,
                      NALU_HYPRE_Int argc,
                      char *argv[] )
{
#ifdef NALU_HYPRE_MIXEDINT
   hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC, "Euclid cannot be used in mixedint mode!");
   return hypre_error_flag;
#else
   START_FUNC_DH
   Parser_dhInit(parser_dh, argc, argv); NALU_HYPRE_EUCLID_ERRCHKA;

   /* maintainers note: even though Parser_dhInit() was called in
      NALU_HYPRE_EuclidCreate(), it's O.K. to call it again.
    */
   END_FUNC_VAL(0)
#endif
}

/*--------------------------------------------------------------------------
 * Insert (flag, value) pairs in Euclid's  database from file
 *--------------------------------------------------------------------------*/

#undef __FUNC__
#define __FUNC__ "NALU_HYPRE_EuclidSetParamsFromFile"
NALU_HYPRE_Int
NALU_HYPRE_EuclidSetParamsFromFile(NALU_HYPRE_Solver solver,
                              char *filename )
{
#ifdef NALU_HYPRE_MIXEDINT
   hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC, "Euclid cannot be used in mixedint mode!");
   return hypre_error_flag;
#else

   START_FUNC_DH
   Parser_dhUpdateFromFile(parser_dh, filename); NALU_HYPRE_EUCLID_ERRCHKA;
   END_FUNC_VAL(0)
#endif
}

NALU_HYPRE_Int
NALU_HYPRE_EuclidSetLevel(NALU_HYPRE_Solver solver,
                     NALU_HYPRE_Int level)
{
#ifdef NALU_HYPRE_MIXEDINT
   hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC, "Euclid cannot be used in mixedint mode!");
   return hypre_error_flag;
#else

   char str_level[8];
   START_FUNC_DH
   hypre_sprintf(str_level, "%d", level);
   Parser_dhInsert(parser_dh, "-level", str_level); NALU_HYPRE_EUCLID_ERRCHKA;
   END_FUNC_VAL(0)
#endif
}

NALU_HYPRE_Int
NALU_HYPRE_EuclidSetBJ(NALU_HYPRE_Solver solver,
                  NALU_HYPRE_Int bj)
{
#ifdef NALU_HYPRE_MIXEDINT
   hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC, "Euclid cannot be used in mixedint mode!");
   return hypre_error_flag;
#else

   char str_bj[8];
   START_FUNC_DH
   hypre_sprintf(str_bj, "%d", bj);
   Parser_dhInsert(parser_dh, "-bj", str_bj); NALU_HYPRE_EUCLID_ERRCHKA;
   END_FUNC_VAL(0)
#endif
}

NALU_HYPRE_Int
NALU_HYPRE_EuclidSetStats(NALU_HYPRE_Solver solver,
                     NALU_HYPRE_Int eu_stats)
{
#ifdef NALU_HYPRE_MIXEDINT
   hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC, "Euclid cannot be used in mixedint mode!");
   return hypre_error_flag;
#else

   char str_eu_stats[8];
   START_FUNC_DH
   hypre_sprintf(str_eu_stats, "%d", eu_stats);
   Parser_dhInsert(parser_dh, "-eu_stats", str_eu_stats); NALU_HYPRE_EUCLID_ERRCHKA;
   END_FUNC_VAL(0)
#endif
}

NALU_HYPRE_Int
NALU_HYPRE_EuclidSetMem(NALU_HYPRE_Solver solver,
                   NALU_HYPRE_Int eu_mem)
{
#ifdef NALU_HYPRE_MIXEDINT
   hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC, "Euclid cannot be used in mixedint mode!");
   return hypre_error_flag;
#else

   char str_eu_mem[8];
   START_FUNC_DH
   hypre_sprintf(str_eu_mem, "%d", eu_mem);
   Parser_dhInsert(parser_dh, "-eu_mem", str_eu_mem); NALU_HYPRE_EUCLID_ERRCHKA;
   END_FUNC_VAL(0)
#endif
}

NALU_HYPRE_Int
NALU_HYPRE_EuclidSetSparseA(NALU_HYPRE_Solver solver,
                       NALU_HYPRE_Real sparse_A)
{
#ifdef NALU_HYPRE_MIXEDINT
   hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC, "Euclid cannot be used in mixedint mode!");
   return hypre_error_flag;
#else

   char str_sparse_A[256];
   START_FUNC_DH
   hypre_sprintf(str_sparse_A, "%f", sparse_A);
   Parser_dhInsert(parser_dh, "-sparseA", str_sparse_A);
   NALU_HYPRE_EUCLID_ERRCHKA;
   END_FUNC_VAL(0)
#endif
}

NALU_HYPRE_Int
NALU_HYPRE_EuclidSetRowScale(NALU_HYPRE_Solver solver,
                        NALU_HYPRE_Int row_scale)
{
#ifdef NALU_HYPRE_MIXEDINT
   hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC, "Euclid cannot be used in mixedint mode!");
   return hypre_error_flag;
#else

   char str_row_scale[8];
   START_FUNC_DH
   hypre_sprintf(str_row_scale, "%d", row_scale);
   Parser_dhInsert(parser_dh, "-rowScale", str_row_scale);
   NALU_HYPRE_EUCLID_ERRCHKA;
   END_FUNC_VAL(0)
#endif
}

NALU_HYPRE_Int
NALU_HYPRE_EuclidSetILUT(NALU_HYPRE_Solver solver,
                    NALU_HYPRE_Real ilut)
{
#ifdef NALU_HYPRE_MIXEDINT
   hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC, "Euclid cannot be used in mixedint mode!");
   return hypre_error_flag;
#else

   char str_ilut[256];
   START_FUNC_DH
   hypre_sprintf(str_ilut, "%f", ilut);
   Parser_dhInsert(parser_dh, "-ilut", str_ilut); NALU_HYPRE_EUCLID_ERRCHKA;
   END_FUNC_VAL(0)
#endif
}

