/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * NALU_HYPRE_ParCSRParaSails interface
 *
 *****************************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "./NALU_HYPRE_parcsr_ls.h"
#include "./_hypre_parcsr_ls.h"

#include "../distributed_matrix/NALU_HYPRE_distributed_matrix_types.h"
#include "../distributed_matrix/NALU_HYPRE_distributed_matrix_protos.h"

#include "../matrix_matrix/NALU_HYPRE_matrix_matrix_protos.h"

#include "../distributed_ls/ParaSails/hypre_ParaSails.h"

/* these includes required for NALU_HYPRE_ParaSailsBuildIJMatrix */
#include "../IJ_mv/NALU_HYPRE_IJ_mv.h"

/* Must include implementation definition for ParVector since no data access
   functions are publically provided. AJC, 5/99 */
/* Likewise for Vector. AJC, 5/99 */
#include "../seq_mv/vector.h"
/* AB 8/06 - replace header file */
/* #include "../parcsr_mv/par_vector.h" */
#include "../parcsr_mv/_hypre_parcsr_mv.h"

/* If code is more mysterious, then it must be good */
typedef struct
{
   hypre_ParaSails obj;
   NALU_HYPRE_Int       sym;
   NALU_HYPRE_Real      thresh;
   NALU_HYPRE_Int       nlevels;
   NALU_HYPRE_Real      filter;
   NALU_HYPRE_Real      loadbal;
   NALU_HYPRE_Int       reuse; /* reuse pattern */
   MPI_Comm        comm;
   NALU_HYPRE_Int       logging;
}
Secret;

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRParaSailsCreate - Return a ParaSails preconditioner object
 * "solver".  The default parameters for the preconditioner are also set,
 * so a call to NALU_HYPRE_ParCSRParaSailsSetParams is not absolutely necessary.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParCSRParaSailsCreate( MPI_Comm comm, NALU_HYPRE_Solver *solver )
{
#ifdef NALU_HYPRE_MIXEDINT
   hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC, "ParaSails not usable in mixedint mode!");
   return hypre_error_flag;
#else

   Secret *secret;

   secret = hypre_TAlloc(Secret, 1, NALU_HYPRE_MEMORY_HOST);

   if (secret == NULL)
   {
      hypre_error(NALU_HYPRE_ERROR_MEMORY);
      return hypre_error_flag;
   }

   secret->sym     = 1;
   secret->thresh  = 0.1;
   secret->nlevels = 1;
   secret->filter  = 0.1;
   secret->loadbal = 0.0;
   secret->reuse   = 0;
   secret->comm    = comm;
   secret->logging = 0;

   hypre_ParaSailsCreate(comm, &secret->obj);

   *solver = (NALU_HYPRE_Solver) secret;

   return hypre_error_flag;
#endif
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRParaSailsDestroy - Destroy a ParaSails object.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParCSRParaSailsDestroy( NALU_HYPRE_Solver solver )
{
#ifdef NALU_HYPRE_MIXEDINT
   hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC, "ParaSails not usable in mixedint mode!");
   return hypre_error_flag;
#else

   Secret *secret;

   secret = (Secret *) solver;
   hypre_ParaSailsDestroy(secret->obj);

   hypre_TFree(secret, NALU_HYPRE_MEMORY_HOST);

   return hypre_error_flag;
#endif
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRParaSailsSetup - Set up function for ParaSails.
 * This function is not called on subsequent times if the preconditioner is
 * being reused.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParCSRParaSailsSetup( NALU_HYPRE_Solver solver,
                            NALU_HYPRE_ParCSRMatrix A,
                            NALU_HYPRE_ParVector b,
                            NALU_HYPRE_ParVector x      )
{
#ifdef NALU_HYPRE_MIXEDINT
   hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC, "ParaSails not usable in mixedint mode!");
   return hypre_error_flag;
#else

   static NALU_HYPRE_Int virgin = 1;
   NALU_HYPRE_DistributedMatrix mat;
   Secret *secret = (Secret *) solver;

   /* The following call will also create the distributed matrix */

   NALU_HYPRE_ConvertParCSRMatrixToDistributedMatrix( A, &mat );
   if (hypre_error_flag) { return hypre_error_flag; }

   if (virgin || secret->reuse == 0) /* call set up at least once */
   {
      virgin = 0;
      hypre_ParaSailsSetup(
         secret->obj, mat, secret->sym, secret->thresh, secret->nlevels,
         secret->filter, secret->loadbal, secret->logging);
      if (hypre_error_flag) { return hypre_error_flag; }
   }
   else /* reuse is true; this is a subsequent call */
   {
      /* reuse pattern: always use filter value of 0 and loadbal of 0 */
      hypre_ParaSailsSetupValues(secret->obj, mat,
                                 0.0, 0.0, secret->logging);
      if (hypre_error_flag) { return hypre_error_flag; }
   }

   NALU_HYPRE_DistributedMatrixDestroy(mat);

   return hypre_error_flag;
#endif
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRParaSailsSolve - Solve function for ParaSails.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParCSRParaSailsSolve( NALU_HYPRE_Solver solver,
                            NALU_HYPRE_ParCSRMatrix A,
                            NALU_HYPRE_ParVector b,
                            NALU_HYPRE_ParVector x     )
{
#ifdef NALU_HYPRE_MIXEDINT
   hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC, "ParaSails not usable in mixedint mode!");
   return hypre_error_flag;
#else

   NALU_HYPRE_Real *rhs, *soln;
   Secret *secret = (Secret *) solver;

   rhs  = hypre_VectorData(hypre_ParVectorLocalVector((hypre_ParVector *) b));
   soln = hypre_VectorData(hypre_ParVectorLocalVector((hypre_ParVector *) x));

   hypre_ParaSailsApply(secret->obj, rhs, soln);

   return hypre_error_flag;
#endif
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRParaSailsSetParams - Set the parameters "thresh" and "nlevels"
 * for a ParaSails object.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParCSRParaSailsSetParams(NALU_HYPRE_Solver solver,
                               NALU_HYPRE_Real   thresh,
                               NALU_HYPRE_Int    nlevels )
{
#ifdef NALU_HYPRE_MIXEDINT
   hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC, "ParaSails not usable in mixedint mode!");
   return hypre_error_flag;
#else

   Secret *secret = (Secret *) solver;

   secret->thresh  = thresh;
   secret->nlevels = nlevels;

   return hypre_error_flag;
#endif
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRParaSailsSetFilter - Set the filter parameter,
 * NALU_HYPRE_ParCSRParaSailsGetFilter
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParCSRParaSailsSetFilter(NALU_HYPRE_Solver solver,
                               NALU_HYPRE_Real   filter  )
{
#ifdef NALU_HYPRE_MIXEDINT
   hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC, "ParaSails not usable in mixedint mode!");
   return hypre_error_flag;
#else

   Secret *secret = (Secret *) solver;

   secret->filter = filter;

   return hypre_error_flag;
#endif
}

NALU_HYPRE_Int
NALU_HYPRE_ParCSRParaSailsGetFilter(NALU_HYPRE_Solver solver,
                               NALU_HYPRE_Real * filter  )
{
#ifdef NALU_HYPRE_MIXEDINT
   hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC, "ParaSails not usable in mixedint mode!");
   return hypre_error_flag;
#else

   Secret *secret = (Secret *) solver;

   *filter = secret->filter;

   return hypre_error_flag;
#endif
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRParaSailsSetSym - Set whether the matrix is symmetric:
 * nonzero = symmetric, 0 = nonsymmetric.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParCSRParaSailsSetSym(NALU_HYPRE_Solver solver,
                            NALU_HYPRE_Int    sym     )
{
#ifdef NALU_HYPRE_MIXEDINT
   hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC, "ParaSails not usable in mixedint mode!");
   return hypre_error_flag;
#else

   Secret *secret = (Secret *) solver;

   secret->sym = sym;

   return hypre_error_flag;
#endif
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRParaSailsSetLoadbal, NALU_HYPRE_ParCSRParaSailsGetLoadbal
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParCSRParaSailsSetLoadbal(NALU_HYPRE_Solver solver,
                                NALU_HYPRE_Real   loadbal )
{
#ifdef NALU_HYPRE_MIXEDINT
   hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC, "ParaSails not usable in mixedint mode!");
   return hypre_error_flag;
#else

   Secret *secret = (Secret *) solver;

   secret->loadbal = loadbal;

   return hypre_error_flag;
#endif
}

NALU_HYPRE_Int
NALU_HYPRE_ParCSRParaSailsGetLoadbal(NALU_HYPRE_Solver solver,
                                NALU_HYPRE_Real * loadbal )
{
#ifdef NALU_HYPRE_MIXEDINT
   hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC, "ParaSails not usable in mixedint mode!");
   return hypre_error_flag;
#else

   Secret *secret = (Secret *) solver;

   *loadbal = secret->loadbal;

   return hypre_error_flag;
#endif
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRParaSailsSetReuse - reuse pattern if "reuse" if nonzero
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParCSRParaSailsSetReuse(NALU_HYPRE_Solver solver,
                              NALU_HYPRE_Int    reuse   )
{
#ifdef NALU_HYPRE_MIXEDINT
   hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC, "ParaSails not usable in mixedint mode!");
   return hypre_error_flag;
#else

   Secret *secret = (Secret *) solver;

   secret->reuse = reuse;

   return hypre_error_flag;
#endif
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRParaSailsSetLogging -
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParCSRParaSailsSetLogging(NALU_HYPRE_Solver solver,
                                NALU_HYPRE_Int    logging )
{
#ifdef NALU_HYPRE_MIXEDINT
   hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC, "ParaSails not usable in mixedint mode!");
   return hypre_error_flag;
#else

   Secret *secret = (Secret *) solver;

   secret->logging = logging;

   return hypre_error_flag;
#endif
}

/******************************************************************************
 *
 * NALU_HYPRE_ParaSails interface
 *
 *****************************************************************************/

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParaSailsCreate - Return a ParaSails preconditioner object
 * "solver".  The default parameters for the preconditioner are also set,
 * so a call to NALU_HYPRE_ParaSailsSetParams is not absolutely necessary.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParaSailsCreate( MPI_Comm comm, NALU_HYPRE_Solver *solver )
{
#ifdef NALU_HYPRE_MIXEDINT
   hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC, "ParaSails not usable in mixedint mode!");
   return hypre_error_flag;
#else

   Secret *secret;

   secret = hypre_TAlloc(Secret, 1, NALU_HYPRE_MEMORY_HOST);

   if (secret == NULL)
   {
      hypre_error(NALU_HYPRE_ERROR_MEMORY);
      return hypre_error_flag;
   }

   secret->sym     = 1;
   secret->thresh  = 0.1;
   secret->nlevels = 1;
   secret->filter  = 0.1;
   secret->loadbal = 0.0;
   secret->reuse   = 0;
   secret->comm    = comm;
   secret->logging = 0;

   hypre_ParaSailsCreate(comm, &secret->obj);

   *solver = (NALU_HYPRE_Solver) secret;

   return hypre_error_flag;
#endif
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParaSailsDestroy - Destroy a ParaSails object.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParaSailsDestroy( NALU_HYPRE_Solver solver )
{
#ifdef NALU_HYPRE_MIXEDINT
   hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC, "ParaSails not usable in mixedint mode!");
   return hypre_error_flag;
#else

   Secret *secret;

   secret = (Secret *) solver;
   hypre_ParaSailsDestroy(secret->obj);

   hypre_TFree(secret, NALU_HYPRE_MEMORY_HOST);

   return hypre_error_flag;
#endif
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParaSailsSetup - Set up function for ParaSails.
 * This function is not called on subsequent times if the preconditioner is
 * being reused.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParaSailsSetup( NALU_HYPRE_Solver solver,
                      NALU_HYPRE_ParCSRMatrix A,
                      NALU_HYPRE_ParVector b,
                      NALU_HYPRE_ParVector x     )
{
#ifdef NALU_HYPRE_MIXEDINT
   hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC, "ParaSails not usable in mixedint mode!");
   return hypre_error_flag;
#else

   static NALU_HYPRE_Int virgin = 1;
   NALU_HYPRE_DistributedMatrix mat;
   Secret *secret = (Secret *) solver;
   NALU_HYPRE_Int ierr;

   /* The following call will also create the distributed matrix */

   ierr = NALU_HYPRE_GetError(); NALU_HYPRE_ClearAllErrors();
   NALU_HYPRE_ConvertParCSRMatrixToDistributedMatrix( A, &mat );
   if (hypre_error_flag) { return hypre_error_flag |= ierr; }

   if (virgin || secret->reuse == 0) /* call set up at least once */
   {
      virgin = 0;
      hypre_ParaSailsSetup(
         secret->obj, mat, secret->sym, secret->thresh, secret->nlevels,
         secret->filter, secret->loadbal, secret->logging);
      if (hypre_error_flag) { return hypre_error_flag |= ierr; }
   }
   else /* reuse is true; this is a subsequent call */
   {
      /* reuse pattern: always use filter value of 0 and loadbal of 0 */
      hypre_ParaSailsSetupValues(secret->obj, mat,
                                 0.0, 0.0, secret->logging);
      if (hypre_error_flag) { return hypre_error_flag |= ierr; }
   }

   NALU_HYPRE_DistributedMatrixDestroy(mat);

   return hypre_error_flag;
#endif
}
/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParaSailsSolve - Solve function for ParaSails.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParaSailsSolve( NALU_HYPRE_Solver solver,
                      NALU_HYPRE_ParCSRMatrix A,
                      NALU_HYPRE_ParVector b,
                      NALU_HYPRE_ParVector x     )
{
#ifdef NALU_HYPRE_MIXEDINT
   hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC, "ParaSails not usable in mixedint mode!");
   return hypre_error_flag;
#else

   NALU_HYPRE_Real *rhs, *soln;
   Secret *secret = (Secret *) solver;

   rhs  = hypre_VectorData(hypre_ParVectorLocalVector((hypre_ParVector *) b));
   soln = hypre_VectorData(hypre_ParVectorLocalVector((hypre_ParVector *) x));

   hypre_ParaSailsApply(secret->obj, rhs, soln);

   return hypre_error_flag;
#endif
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParaSailsSetParams - Set the parameters "thresh" and "nlevels"
 * for a ParaSails object.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParaSailsSetParams(NALU_HYPRE_Solver solver,
                         NALU_HYPRE_Real   thresh,
                         NALU_HYPRE_Int    nlevels )
{
#ifdef NALU_HYPRE_MIXEDINT
   hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC, "ParaSails not usable in mixedint mode!");
   return hypre_error_flag;
#else

   Secret *secret = (Secret *) solver;

   secret->thresh  = thresh;
   secret->nlevels = nlevels;

   return hypre_error_flag;
#endif
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParaSailsSetThresh - Set the "thresh" parameter only
 * for a ParaSails object.
 * NALU_HYPRE_ParaSailsGetThresh
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParaSailsSetThresh( NALU_HYPRE_Solver solver,
                          NALU_HYPRE_Real   thresh )
{
#ifdef NALU_HYPRE_MIXEDINT
   hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC, "ParaSails not usable in mixedint mode!");
   return hypre_error_flag;
#else

   Secret *secret = (Secret *) solver;

   secret->thresh  = thresh;

   return hypre_error_flag;
#endif
}

NALU_HYPRE_Int
NALU_HYPRE_ParaSailsGetThresh( NALU_HYPRE_Solver solver,
                          NALU_HYPRE_Real * thresh )
{
#ifdef NALU_HYPRE_MIXEDINT
   hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC, "ParaSails not usable in mixedint mode!");
   return hypre_error_flag;
#else

   Secret *secret = (Secret *) solver;

   *thresh = secret->thresh;

   return hypre_error_flag;
#endif
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParaSailsSetNlevels - Set the "nlevels" parameter only
 * for a ParaSails object.
 * NALU_HYPRE_ParaSailsGetNlevels
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParaSailsSetNlevels( NALU_HYPRE_Solver solver,
                           NALU_HYPRE_Int    nlevels )
{
#ifdef NALU_HYPRE_MIXEDINT
   hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC, "ParaSails not usable in mixedint mode!");
   return hypre_error_flag;
#else

   Secret *secret = (Secret *) solver;

   secret->nlevels  = nlevels;

   return hypre_error_flag;
#endif
}

NALU_HYPRE_Int
NALU_HYPRE_ParaSailsGetNlevels( NALU_HYPRE_Solver solver,
                           NALU_HYPRE_Int  * nlevels )
{
#ifdef NALU_HYPRE_MIXEDINT
   hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC, "ParaSails not usable in mixedint mode!");
   return hypre_error_flag;
#else

   Secret *secret = (Secret *) solver;

   *nlevels = secret->nlevels;

   return hypre_error_flag;
#endif
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParaSailsSetFilter - Set the filter parameter.
 * NALU_HYPRE_ParaSailsGetFilter
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParaSailsSetFilter(NALU_HYPRE_Solver solver,
                         NALU_HYPRE_Real   filter  )
{
#ifdef NALU_HYPRE_MIXEDINT
   hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC, "ParaSails not usable in mixedint mode!");
   return hypre_error_flag;
#else

   Secret *secret = (Secret *) solver;

   secret->filter = filter;

   return hypre_error_flag;
#endif
}

NALU_HYPRE_Int
NALU_HYPRE_ParaSailsGetFilter(NALU_HYPRE_Solver solver,
                         NALU_HYPRE_Real * filter  )
{
#ifdef NALU_HYPRE_MIXEDINT
   hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC, "ParaSails not usable in mixedint mode!");
   return hypre_error_flag;
#else

   Secret *secret = (Secret *) solver;

   *filter = secret->filter;

   return hypre_error_flag;
#endif
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParaSailsSetSym - Set whether the matrix is symmetric:
 * nonzero = symmetric, 0 = nonsymmetric.
 * NALU_HYPRE_ParaSailsGetSym
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParaSailsSetSym(NALU_HYPRE_Solver solver,
                      NALU_HYPRE_Int    sym     )
{
#ifdef NALU_HYPRE_MIXEDINT
   hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC, "ParaSails not usable in mixedint mode!");
   return hypre_error_flag;
#else

   Secret *secret = (Secret *) solver;

   secret->sym = sym;

   return hypre_error_flag;
#endif
}

NALU_HYPRE_Int
NALU_HYPRE_ParaSailsGetSym(NALU_HYPRE_Solver solver,
                      NALU_HYPRE_Int  * sym     )
{
#ifdef NALU_HYPRE_MIXEDINT
   hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC, "ParaSails not usable in mixedint mode!");
   return hypre_error_flag;
#else

   Secret *secret = (Secret *) solver;

   *sym = secret->sym;

   return hypre_error_flag;
#endif
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParaSailsSetLoadbal, NALU_HYPRE_ParaSailsGetLoadbal
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParaSailsSetLoadbal(NALU_HYPRE_Solver solver,
                          NALU_HYPRE_Real   loadbal )
{
#ifdef NALU_HYPRE_MIXEDINT
   hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC, "ParaSails not usable in mixedint mode!");
   return hypre_error_flag;
#else

   Secret *secret = (Secret *) solver;

   secret->loadbal = loadbal;

   return hypre_error_flag;
#endif
}

NALU_HYPRE_Int
NALU_HYPRE_ParaSailsGetLoadbal(NALU_HYPRE_Solver solver,
                          NALU_HYPRE_Real * loadbal )
{
#ifdef NALU_HYPRE_MIXEDINT
   hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC, "ParaSails not usable in mixedint mode!");
   return hypre_error_flag;
#else

   Secret *secret = (Secret *) solver;

   *loadbal = secret->loadbal;

   return hypre_error_flag;
#endif
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParaSailsSetReuse - reuse pattern if "reuse" if nonzero
 * NALU_HYPRE_ParaSailsGetReuse
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParaSailsSetReuse(NALU_HYPRE_Solver solver,
                        NALU_HYPRE_Int    reuse   )
{
#ifdef NALU_HYPRE_MIXEDINT
   hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC, "ParaSails not usable in mixedint mode!");
   return hypre_error_flag;
#else

   Secret *secret = (Secret *) solver;

   secret->reuse = reuse;

   return hypre_error_flag;
#endif
}

NALU_HYPRE_Int
NALU_HYPRE_ParaSailsGetReuse(NALU_HYPRE_Solver solver,
                        NALU_HYPRE_Int  * reuse   )
{
#ifdef NALU_HYPRE_MIXEDINT
   hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC, "ParaSails not usable in mixedint mode!");
   return hypre_error_flag;
#else

   Secret *secret = (Secret *) solver;

   *reuse = secret->reuse;

   return hypre_error_flag;
#endif
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParaSailsSetLogging, NALU_HYPRE_ParaSailsGetLogging
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParaSailsSetLogging(NALU_HYPRE_Solver solver,
                          NALU_HYPRE_Int    logging )
{
#ifdef NALU_HYPRE_MIXEDINT
   hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC, "ParaSails not usable in mixedint mode!");
   return hypre_error_flag;
#else

   Secret *secret = (Secret *) solver;

   secret->logging = logging;

   return hypre_error_flag;
#endif
}

NALU_HYPRE_Int
NALU_HYPRE_ParaSailsGetLogging(NALU_HYPRE_Solver solver,
                          NALU_HYPRE_Int  * logging )
{
#ifdef NALU_HYPRE_MIXEDINT
   hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC, "ParaSails not usable in mixedint mode!");
   return hypre_error_flag;
#else

   Secret *secret = (Secret *) solver;

   *logging = secret->logging;

   return hypre_error_flag;
#endif
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParaSailsBuildIJMatrix -
 *--------------------------------------------------------------------------*/
NALU_HYPRE_Int
NALU_HYPRE_ParaSailsBuildIJMatrix(NALU_HYPRE_Solver solver, NALU_HYPRE_IJMatrix *pij_A)
{
#ifdef NALU_HYPRE_MIXEDINT
   hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC, "ParaSails not usable in mixedint mode!");
   return hypre_error_flag;
#else

   Secret *secret = (Secret *) solver;

   hypre_ParaSailsBuildIJMatrix(secret->obj, pij_A);

   return hypre_error_flag;
#endif
}
