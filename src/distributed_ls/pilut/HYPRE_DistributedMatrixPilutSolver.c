/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/* Include headers for problem and solver data structure */
#include "./DistributedMatrixPilutSolver.h"


/*--------------------------------------------------------------------------
 * NALU_HYPRE_NewDistributedMatrixPilutSolver
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int  NALU_HYPRE_NewDistributedMatrixPilutSolver(
                                  MPI_Comm comm,
                                  NALU_HYPRE_DistributedMatrix matrix,
                                  NALU_HYPRE_DistributedMatrixPilutSolver *new_solver )
     /* Allocates and Initializes solver structure */
{

   hypre_DistributedMatrixPilutSolver     *solver;
   hypre_PilutSolverGlobals *globals;
   NALU_HYPRE_Int            nprocs, myid;
   FactorMatType *ldu;

   /* Allocate structure for holding solver data */
   solver = (hypre_DistributedMatrixPilutSolver *)
            hypre_CTAlloc( hypre_DistributedMatrixPilutSolver,  1, NALU_HYPRE_MEMORY_HOST);

   /* Initialize components of solver */
   hypre_DistributedMatrixPilutSolverComm(solver) = comm;
   hypre_DistributedMatrixPilutSolverDataDist(solver) =
         (DataDistType *) hypre_CTAlloc( DataDistType,  1 , NALU_HYPRE_MEMORY_HOST);

   /* Structure for holding "global variables"; makes code thread safe(r) */
   globals = hypre_DistributedMatrixPilutSolverGlobals(solver) =
       (hypre_PilutSolverGlobals *) hypre_CTAlloc( hypre_PilutSolverGlobals,  1 , NALU_HYPRE_MEMORY_HOST);

   jr = NULL;
   hypre_lr = NULL;
   jw = NULL;
   w  = NULL;

   globals->logging = 0;

   /* Set some variables in the "global variables" section */
   pilut_comm = comm;

   hypre_MPI_Comm_size( comm, &nprocs );
   npes = nprocs;

   hypre_MPI_Comm_rank( comm, &myid );
   mype = myid;

#ifdef NALU_HYPRE_TIMING
   globals->CCI_timer = hypre_InitializeTiming( "hypre_ComputeCommInfo" );
   globals->SS_timer = hypre_InitializeTiming( "hypre_SelectSet" );
   globals->SFR_timer = hypre_InitializeTiming( "hypre_SendFactoredRows" );
   globals->CR_timer = hypre_InitializeTiming( "hypre_ComputeRmat" );
   globals->FL_timer = hypre_InitializeTiming( "hypre_FactorLocal" );
   globals->SLUD_timer = hypre_InitializeTiming( "SeparateLU_byDIAG" );
   globals->SLUM_timer = hypre_InitializeTiming( "SeparateLU_byMIS" );
   globals->UL_timer = hypre_InitializeTiming( "hypre_UpdateL" );
   globals->FNR_timer = hypre_InitializeTiming( "hypre_FormNRmat" );

   globals->Ll_timer = hypre_InitializeTiming( "Local part of front solve" );
   globals->Lp_timer = hypre_InitializeTiming( "Parallel part of front solve" );
   globals->Up_timer = hypre_InitializeTiming( "Parallel part of back solve" );
   globals->Ul_timer = hypre_InitializeTiming( "Local part of back solve" );
#endif

   /* Data distribution structure */
   DataDistTypeRowdist(hypre_DistributedMatrixPilutSolverDataDist(solver))
       = (NALU_HYPRE_Int *) hypre_CTAlloc( NALU_HYPRE_Int,  nprocs+1 , NALU_HYPRE_MEMORY_HOST);

   hypre_DistributedMatrixPilutSolverFactorMat(solver) =
          (FactorMatType *) hypre_CTAlloc( FactorMatType,  1 , NALU_HYPRE_MEMORY_HOST);

   ldu = hypre_DistributedMatrixPilutSolverFactorMat(solver);

   ldu->lsrowptr = NULL;
   ldu->lerowptr = NULL;
   ldu->lcolind  = NULL;
   ldu->lvalues  = NULL;
   ldu->usrowptr = NULL;
   ldu->uerowptr = NULL;
   ldu->ucolind  = NULL;
   ldu->uvalues  = NULL;
   ldu->dvalues  = NULL;
   ldu->nrm2s    = NULL;
   ldu->perm     = NULL;
   ldu->iperm    = NULL;

   /* Note that because we allow matrix to be NULL at this point so that it can
      be set later with a SetMatrix call, we do nothing with matrix except insert
      it into the structure */
   hypre_DistributedMatrixPilutSolverMatrix(solver) = matrix;

   /* Defaults for Parameters controlling the incomplete factorization */
   hypre_DistributedMatrixPilutSolverGmaxnz(solver)   = 20;     /* Maximum nonzeroes per row of factor */
   hypre_DistributedMatrixPilutSolverTol(solver)   = 0.000001;  /* Drop tolerance for factor */

   /* Return created structure to calling routine */
   *new_solver = ( (NALU_HYPRE_DistributedMatrixPilutSolver) solver );

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_FreeDistributedMatrixPilutSolver
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int NALU_HYPRE_FreeDistributedMatrixPilutSolver (
                  NALU_HYPRE_DistributedMatrixPilutSolver in_ptr )
{
  FactorMatType *ldu;

   hypre_DistributedMatrixPilutSolver *solver =
      (hypre_DistributedMatrixPilutSolver *) in_ptr;

#ifdef NALU_HYPRE_TIMING
   hypre_PilutSolverGlobals *globals;
  globals = hypre_DistributedMatrixPilutSolverGlobals(solver);
#endif

  hypre_TFree( DataDistTypeRowdist(hypre_DistributedMatrixPilutSolverDataDist(solver)), NALU_HYPRE_MEMORY_HOST);
  hypre_TFree( hypre_DistributedMatrixPilutSolverDataDist(solver) , NALU_HYPRE_MEMORY_HOST);

  /* Free malloced members of the FactorMat member */
  ldu = hypre_DistributedMatrixPilutSolverFactorMat(solver);

  hypre_TFree( ldu->lcolind , NALU_HYPRE_MEMORY_HOST);
  hypre_TFree( ldu->ucolind , NALU_HYPRE_MEMORY_HOST);

  hypre_TFree( ldu->lvalues , NALU_HYPRE_MEMORY_HOST);
  hypre_TFree( ldu->uvalues , NALU_HYPRE_MEMORY_HOST);

  hypre_TFree( ldu->lrowptr , NALU_HYPRE_MEMORY_HOST);
  hypre_TFree( ldu->urowptr , NALU_HYPRE_MEMORY_HOST);

  hypre_TFree( ldu->dvalues , NALU_HYPRE_MEMORY_HOST);
  hypre_TFree( ldu->nrm2s , NALU_HYPRE_MEMORY_HOST);
  hypre_TFree( ldu->perm , NALU_HYPRE_MEMORY_HOST);
  hypre_TFree( ldu->iperm , NALU_HYPRE_MEMORY_HOST);

  hypre_TFree( ldu->gatherbuf , NALU_HYPRE_MEMORY_HOST);

  hypre_TFree( ldu->lx , NALU_HYPRE_MEMORY_HOST);
  hypre_TFree( ldu->ux , NALU_HYPRE_MEMORY_HOST);

    /* Beginning of TriSolveCommType freeing */
    hypre_TFree( ldu->lcomm.raddr , NALU_HYPRE_MEMORY_HOST);
    hypre_TFree( ldu->ucomm.raddr , NALU_HYPRE_MEMORY_HOST);

    hypre_TFree( ldu->lcomm.spes , NALU_HYPRE_MEMORY_HOST);
    hypre_TFree( ldu->ucomm.spes , NALU_HYPRE_MEMORY_HOST);

    hypre_TFree( ldu->lcomm.sptr , NALU_HYPRE_MEMORY_HOST);
    hypre_TFree( ldu->ucomm.sptr , NALU_HYPRE_MEMORY_HOST);

    hypre_TFree( ldu->lcomm.sindex , NALU_HYPRE_MEMORY_HOST);
    hypre_TFree( ldu->ucomm.sindex , NALU_HYPRE_MEMORY_HOST);

    hypre_TFree( ldu->lcomm.auxsptr , NALU_HYPRE_MEMORY_HOST);
    hypre_TFree( ldu->ucomm.auxsptr , NALU_HYPRE_MEMORY_HOST);

    hypre_TFree( ldu->lcomm.rpes , NALU_HYPRE_MEMORY_HOST);
    hypre_TFree( ldu->ucomm.rpes , NALU_HYPRE_MEMORY_HOST);

    hypre_TFree( ldu->lcomm.rdone , NALU_HYPRE_MEMORY_HOST);
    hypre_TFree( ldu->ucomm.rdone , NALU_HYPRE_MEMORY_HOST);

    hypre_TFree( ldu->lcomm.rnum , NALU_HYPRE_MEMORY_HOST);
    hypre_TFree( ldu->ucomm.rnum , NALU_HYPRE_MEMORY_HOST);

    /* End of TriSolveCommType freeing */

  hypre_TFree( hypre_DistributedMatrixPilutSolverFactorMat(solver) , NALU_HYPRE_MEMORY_HOST);
  /* End of FactorMat member */

#ifdef NALU_HYPRE_TIMING
  hypre_FinalizeTiming( globals->CCI_timer );
  hypre_FinalizeTiming( globals->SS_timer  );
  hypre_FinalizeTiming( globals->SFR_timer );
  hypre_FinalizeTiming( globals->CR_timer );
  hypre_FinalizeTiming( globals->FL_timer  );
  hypre_FinalizeTiming( globals->SLUD_timer  );
  hypre_FinalizeTiming( globals->SLUM_timer );
  hypre_FinalizeTiming( globals->UL_timer  );
  hypre_FinalizeTiming( globals->FNR_timer  );

  hypre_FinalizeTiming( globals->Ll_timer  );
  hypre_FinalizeTiming( globals->Lp_timer );
  hypre_FinalizeTiming( globals->Up_timer );
  hypre_FinalizeTiming( globals->Ul_timer );
#endif

  hypre_TFree( hypre_DistributedMatrixPilutSolverGlobals(solver) , NALU_HYPRE_MEMORY_HOST);

  hypre_TFree(solver, NALU_HYPRE_MEMORY_HOST);

  return(0);
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_DistributedMatrixPilutSolverInitialize
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int NALU_HYPRE_DistributedMatrixPilutSolverInitialize (
                  NALU_HYPRE_DistributedMatrixPilutSolver solver )
{

   return(0);
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_DistributedMatrixPilutSolverSetMatrix
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int NALU_HYPRE_DistributedMatrixPilutSolverSetMatrix(
                  NALU_HYPRE_DistributedMatrixPilutSolver in_ptr,
                  NALU_HYPRE_DistributedMatrix matrix )
{
  hypre_DistributedMatrixPilutSolver *solver =
      (hypre_DistributedMatrixPilutSolver *) in_ptr;

  hypre_DistributedMatrixPilutSolverMatrix( solver ) = matrix;
  return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_DistributedMatrixPilutSolverGetMatrix
 *--------------------------------------------------------------------------*/

NALU_HYPRE_DistributedMatrix
   NALU_HYPRE_DistributedMatrixPilutSolverGetMatrix(
                  NALU_HYPRE_DistributedMatrixPilutSolver in_ptr )
{
  hypre_DistributedMatrixPilutSolver *solver =
      (hypre_DistributedMatrixPilutSolver *) in_ptr;

  return( hypre_DistributedMatrixPilutSolverMatrix( solver ) );

}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_DistributedMatrixPilutSolverSetFirstLocalRow
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int NALU_HYPRE_DistributedMatrixPilutSolverSetNumLocalRow(
                  NALU_HYPRE_DistributedMatrixPilutSolver in_ptr,
                  NALU_HYPRE_Int FirstLocalRow )
{
  hypre_DistributedMatrixPilutSolver *solver =
      (hypre_DistributedMatrixPilutSolver *) in_ptr;
   hypre_PilutSolverGlobals *globals = hypre_DistributedMatrixPilutSolverGlobals(solver);

  DataDistTypeRowdist(hypre_DistributedMatrixPilutSolverDataDist( solver ))[mype] =
     FirstLocalRow;

  return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_DistributedMatrixPilutSolverSetFactorRowSize
 *   Sets the maximum number of entries to be kept in the incomplete factors
 *   This number applies both to the row of L, and also separately to the
 *   row of U.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int NALU_HYPRE_DistributedMatrixPilutSolverSetFactorRowSize(
                  NALU_HYPRE_DistributedMatrixPilutSolver in_ptr,
                  NALU_HYPRE_Int size )
{
  hypre_DistributedMatrixPilutSolver *solver =
      (hypre_DistributedMatrixPilutSolver *) in_ptr;

  hypre_DistributedMatrixPilutSolverGmaxnz( solver ) = size;

  return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_DistributedMatrixPilutSolverSetDropTolerance
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int NALU_HYPRE_DistributedMatrixPilutSolverSetDropTolerance(
                  NALU_HYPRE_DistributedMatrixPilutSolver in_ptr,
                  NALU_HYPRE_Real tolerance )
{
  hypre_DistributedMatrixPilutSolver *solver =
      (hypre_DistributedMatrixPilutSolver *) in_ptr;

  hypre_DistributedMatrixPilutSolverTol( solver ) = tolerance;

  return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_DistributedMatrixPilutSolverSetMaxIts
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int NALU_HYPRE_DistributedMatrixPilutSolverSetMaxIts(
                  NALU_HYPRE_DistributedMatrixPilutSolver in_ptr,
                  NALU_HYPRE_Int its )
{
  hypre_DistributedMatrixPilutSolver *solver =
      (hypre_DistributedMatrixPilutSolver *) in_ptr;

  hypre_DistributedMatrixPilutSolverMaxIts( solver ) = its;

  return hypre_error_flag;
}

NALU_HYPRE_Int NALU_HYPRE_DistributedMatrixPilutSolverSetLogging(
                  NALU_HYPRE_DistributedMatrixPilutSolver in_ptr,
                  NALU_HYPRE_Int logging )
{
  hypre_DistributedMatrixPilutSolver *solver =
      (hypre_DistributedMatrixPilutSolver *) in_ptr;
   hypre_PilutSolverGlobals *globals = hypre_DistributedMatrixPilutSolverGlobals(solver);

   if (globals)
   {
      globals->logging = logging;
   }

  return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_DistributedMatrixPilutSolverSetup
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int NALU_HYPRE_DistributedMatrixPilutSolverSetup( NALU_HYPRE_DistributedMatrixPilutSolver in_ptr )
{
   NALU_HYPRE_Int m, n, nprocs, start, end, *rowdist, col0, coln, ierr;
   hypre_DistributedMatrixPilutSolver *solver =
      (hypre_DistributedMatrixPilutSolver *) in_ptr;
   hypre_PilutSolverGlobals *globals = hypre_DistributedMatrixPilutSolverGlobals(solver);


   if(hypre_DistributedMatrixPilutSolverMatrix(solver) == NULL )
   {
       hypre_error_in_arg(1);
      /* hypre_printf("Cannot call setup to solver until matrix has been set\n");*/
      /* return hypre_error_flag; */
   }

   /* Set up the DataDist structure */

   NALU_HYPRE_DistributedMatrixGetDims(
      hypre_DistributedMatrixPilutSolverMatrix(solver), &m, &n);

   DataDistTypeNrows( hypre_DistributedMatrixPilutSolverDataDist( solver ) ) = m;

   NALU_HYPRE_DistributedMatrixGetLocalRange(
      hypre_DistributedMatrixPilutSolverMatrix(solver), &start, &end, &col0, &coln);

   DataDistTypeLnrows(hypre_DistributedMatrixPilutSolverDataDist( solver )) =
      end - start + 1;

   /* Set up DataDist entry in distributed_solver */
   /* This requires that each processor know which rows are owned by each proc */
   nprocs = npes;

   rowdist = DataDistTypeRowdist( hypre_DistributedMatrixPilutSolverDataDist( solver ) );

   hypre_MPI_Allgather( &start, 1, NALU_HYPRE_MPI_INT, rowdist, 1, NALU_HYPRE_MPI_INT,
      hypre_DistributedMatrixPilutSolverComm(solver) );

   rowdist[ nprocs ] = n;

#ifdef NALU_HYPRE_TIMING
   {
   NALU_HYPRE_Int ilut_timer;

   ilut_timer = hypre_InitializeTiming( "hypre_ILUT factorization" );

   hypre_BeginTiming( ilut_timer );
#endif

   /* Perform approximate factorization */
   ierr = hypre_ILUT( hypre_DistributedMatrixPilutSolverDataDist (solver),
         hypre_DistributedMatrixPilutSolverMatrix (solver),
         hypre_DistributedMatrixPilutSolverFactorMat (solver),
         hypre_DistributedMatrixPilutSolverGmaxnz (solver),
         hypre_DistributedMatrixPilutSolverTol (solver),
         hypre_DistributedMatrixPilutSolverGlobals (solver)
       );

#ifdef NALU_HYPRE_TIMING
   hypre_EndTiming( ilut_timer );
   /* hypre_FinalizeTiming( ilut_timer ); */
   }
#endif

   if (ierr)
   {
       hypre_error(NALU_HYPRE_ERROR_GENERIC);
       /* return hypre_error_flag; */
   }

#ifdef NALU_HYPRE_TIMING
   {
   NALU_HYPRE_Int Setup_timer;

   Setup_timer = hypre_InitializeTiming( "hypre_SetUpLUFactor: setup for triangular solvers");

   hypre_BeginTiming( Setup_timer );
#endif

   ierr = hypre_SetUpLUFactor( hypre_DistributedMatrixPilutSolverDataDist (solver),
               hypre_DistributedMatrixPilutSolverFactorMat (solver),
               hypre_DistributedMatrixPilutSolverGmaxnz (solver),
               hypre_DistributedMatrixPilutSolverGlobals (solver) );

#ifdef NALU_HYPRE_TIMING
   hypre_EndTiming( Setup_timer );
   /* hypre_FinalizeTiming( Setup_timer ); */
   }
#endif

   if (ierr)
   {
       hypre_error(NALU_HYPRE_ERROR_GENERIC);
       /* return hypre_error_flag; */
   }

#ifdef NALU_HYPRE_DEBUG
   NALU_HYPRE_Int logging = globals ? globals->logging : 0;

   if (logging)
   {
      fflush(stdout);
      hypre_printf("Nlevels: %d\n",
            hypre_DistributedMatrixPilutSolverFactorMat (solver)->nlevels);
   }
#endif

   return hypre_error_flag;
}


/*--------------------------------------------------------------------------
 * NALU_HYPRE_DistributedMatrixPilutSolverSolve
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int NALU_HYPRE_DistributedMatrixPilutSolverSolve( NALU_HYPRE_DistributedMatrixPilutSolver in_ptr,
                                           NALU_HYPRE_Real *x, NALU_HYPRE_Real *b )
{

   hypre_DistributedMatrixPilutSolver *solver =
      (hypre_DistributedMatrixPilutSolver *) in_ptr;

   /******** NOTE: Since I am using this currently as a preconditioner, I am only
     doing a single front and back solve. To be a general-purpose solver, this
     call should really be in a loop checking convergence and counting iterations.
     AC - 2/12/98
   */
   /* It should be obvious, but the current treatment of vectors is pretty
      insufficient. -AC 2/12/98
   */
#ifdef NALU_HYPRE_TIMING
{
   NALU_HYPRE_Int LDUSolve_timer;

   LDUSolve_timer = hypre_InitializeTiming( "hypre_ILUT application" );

   hypre_BeginTiming( LDUSolve_timer );
#endif

   hypre_LDUSolve( hypre_DistributedMatrixPilutSolverDataDist (solver),
         hypre_DistributedMatrixPilutSolverFactorMat (solver),
         x,
         b,
         hypre_DistributedMatrixPilutSolverGlobals (solver)
       );
#ifdef NALU_HYPRE_TIMING
   hypre_EndTiming( LDUSolve_timer );
   /* hypre_FinalizeTiming ( LDUSolve_timer ); */
}
#endif


  return hypre_error_flag;
}
