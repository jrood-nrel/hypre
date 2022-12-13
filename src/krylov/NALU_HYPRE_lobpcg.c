/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * NALU_HYPRE_LOBPCG interface
 *
 *****************************************************************************/

#include "_hypre_utilities.h"

#include "NALU_HYPRE_config.h"

#include "NALU_HYPRE_lobpcg.h"
#include "lobpcg.h"

#include "interpreter.h"
#include "NALU_HYPRE_MatvecFunctions.h"

#include "_hypre_lapack.h"

typedef struct
{
   NALU_HYPRE_Int    (*Precond)(void*, void*, void*, void*);
   NALU_HYPRE_Int    (*PrecondSetup)(void*, void*, void*, void*);

} hypre_LOBPCGPrecond;

typedef struct
{
   lobpcg_Tolerance              tolerance;
   NALU_HYPRE_Int                           maxIterations;
   NALU_HYPRE_Int                           verbosityLevel;
   NALU_HYPRE_Int                           precondUsageMode;
   NALU_HYPRE_Int                           iterationNumber;
   utilities_FortranMatrix*      eigenvaluesHistory;
   utilities_FortranMatrix*      residualNorms;
   utilities_FortranMatrix*      residualNormsHistory;

} lobpcg_Data;

#define lobpcg_tolerance(data)            ((data).tolerance)
#define lobpcg_absoluteTolerance(data)    ((data).tolerance.absolute)
#define lobpcg_relativeTolerance(data)    ((data).tolerance.relative)
#define lobpcg_maxIterations(data)        ((data).maxIterations)
#define lobpcg_verbosityLevel(data)       ((data).verbosityLevel)
#define lobpcg_precondUsageMode(data)     ((data).precondUsageMode)
#define lobpcg_iterationNumber(data)      ((data).iterationNumber)
#define lobpcg_eigenvaluesHistory(data)   ((data).eigenvaluesHistory)
#define lobpcg_residualNorms(data)        ((data).residualNorms)
#define lobpcg_residualNormsHistory(data) ((data).residualNormsHistory)

typedef struct
{

   lobpcg_Data                   lobpcgData;

   mv_InterfaceInterpreter*      interpreter;

   void*                         A;
   void*                         matvecData;
   void*                         precondData;

   void*                         B;
   void*                         matvecDataB;
   void*                         T;
   void*                         matvecDataT;

   hypre_LOBPCGPrecond           precondFunctions;

   NALU_HYPRE_MatvecFunctions*        matvecFunctions;

} hypre_LOBPCGData;

static NALU_HYPRE_Int dsygv_interface (NALU_HYPRE_Int *itype, char *jobz, char *uplo, NALU_HYPRE_Int *
                                  n, NALU_HYPRE_Real *a, NALU_HYPRE_Int *lda, NALU_HYPRE_Real *b, NALU_HYPRE_Int *ldb,
                                  NALU_HYPRE_Real *w, NALU_HYPRE_Real *work, NALU_HYPRE_Int *lwork, NALU_HYPRE_Int *info)
{
   hypre_dsygv(itype, jobz, uplo, n, a, lda, b, ldb, w, work, lwork, info);
   return 0;
}

static NALU_HYPRE_Int dpotrf_interface (const char *uplo, NALU_HYPRE_Int *n, NALU_HYPRE_Real *a, NALU_HYPRE_Int *
                                   lda, NALU_HYPRE_Int *info)
{
   hypre_dpotrf(uplo, n, a, lda, info);
   return 0;
}


NALU_HYPRE_Int
lobpcg_initialize( lobpcg_Data* data )
{
   (data->tolerance).absolute    = 1.0e-06;
   (data->tolerance).relative    = 1.0e-06;
   (data->maxIterations)         = 500;
   (data->precondUsageMode)      = 0;
   (data->verbosityLevel)        = 0;
   (data->eigenvaluesHistory)    = utilities_FortranMatrixCreate();
   (data->residualNorms)         = utilities_FortranMatrixCreate();
   (data->residualNormsHistory)  = utilities_FortranMatrixCreate();

   return 0;
}

NALU_HYPRE_Int
lobpcg_clean( lobpcg_Data* data )
{
   utilities_FortranMatrixDestroy( data->eigenvaluesHistory );
   utilities_FortranMatrixDestroy( data->residualNorms );
   utilities_FortranMatrixDestroy( data->residualNormsHistory );

   return 0;
}

NALU_HYPRE_Int
hypre_LOBPCGDestroy( void *pcg_vdata )
{
   hypre_LOBPCGData      *pcg_data      = (hypre_LOBPCGData*)pcg_vdata;

   if (pcg_data)
   {
      NALU_HYPRE_MatvecFunctions * mv = pcg_data->matvecFunctions;
      if ( pcg_data->matvecData != NULL )
      {
         (*(mv->MatvecDestroy))(pcg_data->matvecData);
         pcg_data->matvecData = NULL;
      }
      if ( pcg_data->matvecDataB != NULL )
      {
         (*(mv->MatvecDestroy))(pcg_data->matvecDataB);
         pcg_data->matvecDataB = NULL;
      }
      if ( pcg_data->matvecDataT != NULL )
      {
         (*(mv->MatvecDestroy))(pcg_data->matvecDataT);
         pcg_data->matvecDataT = NULL;
      }

      lobpcg_clean( &(pcg_data->lobpcgData) );

      hypre_TFree( pcg_vdata, NALU_HYPRE_MEMORY_HOST);
   }

   return hypre_error_flag;
}

NALU_HYPRE_Int
hypre_LOBPCGSetup( void *pcg_vdata, void *A, void *b, void *x )
{
   hypre_LOBPCGData *pcg_data = (hypre_LOBPCGData*)pcg_vdata;
   NALU_HYPRE_MatvecFunctions * mv = pcg_data->matvecFunctions;
   NALU_HYPRE_Int  (*precond_setup)(void*, void*, void*, void*) = (pcg_data->precondFunctions).PrecondSetup;
   void *precond_data = (pcg_data->precondData);

   (pcg_data->A) = A;

   if ( pcg_data->matvecData != NULL )
   {
      (*(mv->MatvecDestroy))(pcg_data->matvecData);
   }
   (pcg_data->matvecData) = (*(mv->MatvecCreate))(A, x);

   if ( precond_setup != NULL )
   {
      if ( pcg_data->T == NULL )
      {
         precond_setup(precond_data, A, b, x);
      }
      else
      {
         precond_setup(precond_data, pcg_data->T, b, x);
      }
   }

   return hypre_error_flag;
}

NALU_HYPRE_Int
hypre_LOBPCGSetupB( void *pcg_vdata, void *B, void *x )
{
   hypre_LOBPCGData *pcg_data = (hypre_LOBPCGData*)pcg_vdata;
   NALU_HYPRE_MatvecFunctions * mv = pcg_data->matvecFunctions;

   (pcg_data->B) = B;

   if ( pcg_data->matvecDataB != NULL )
   {
      (*(mv->MatvecDestroy))(pcg_data -> matvecDataB);
   }
   (pcg_data->matvecDataB) = (*(mv->MatvecCreate))(B, x);
   if ( B != NULL )
   {
      (pcg_data->matvecDataB) = (*(mv->MatvecCreate))(B, x);
   }
   else
   {
      (pcg_data->matvecDataB) = NULL;
   }

   return hypre_error_flag;
}

NALU_HYPRE_Int
hypre_LOBPCGSetupT( void *pcg_vdata, void *T, void *x )
{
   hypre_LOBPCGData *pcg_data = (hypre_LOBPCGData*)pcg_vdata;
   NALU_HYPRE_MatvecFunctions * mv = pcg_data->matvecFunctions;

   (pcg_data -> T) = T;

   if ( pcg_data->matvecDataT != NULL )
   {
      (*(mv->MatvecDestroy))(pcg_data->matvecDataT);
   }
   if ( T != NULL )
   {
      (pcg_data->matvecDataT) = (*(mv->MatvecCreate))(T, x);
   }
   else
   {
      (pcg_data->matvecDataT) = NULL;
   }

   return hypre_error_flag;
}

NALU_HYPRE_Int
hypre_LOBPCGSetTol( void* pcg_vdata, NALU_HYPRE_Real tol )
{
   hypre_LOBPCGData *pcg_data = (hypre_LOBPCGData*)pcg_vdata;

   lobpcg_absoluteTolerance(pcg_data->lobpcgData) = tol;

   return hypre_error_flag;
}

NALU_HYPRE_Int
hypre_LOBPCGSetRTol( void* pcg_vdata, NALU_HYPRE_Real tol )
{
   hypre_LOBPCGData *pcg_data = (hypre_LOBPCGData*) pcg_vdata;

   lobpcg_relativeTolerance(pcg_data->lobpcgData) = tol;

   return hypre_error_flag;
}

NALU_HYPRE_Int
hypre_LOBPCGSetMaxIter( void* pcg_vdata, NALU_HYPRE_Int max_iter  )
{
   hypre_LOBPCGData *pcg_data = (hypre_LOBPCGData*)pcg_vdata;

   lobpcg_maxIterations(pcg_data->lobpcgData) = max_iter;

   return hypre_error_flag;
}

NALU_HYPRE_Int
hypre_LOBPCGSetPrecondUsageMode( void* pcg_vdata, NALU_HYPRE_Int mode  )
{
   hypre_LOBPCGData *pcg_data = (hypre_LOBPCGData*)pcg_vdata;

   lobpcg_precondUsageMode(pcg_data->lobpcgData) = mode;

   return hypre_error_flag;
}

NALU_HYPRE_Int
hypre_LOBPCGGetPrecond( void         *pcg_vdata,
                        NALU_HYPRE_Solver *precond_data_ptr )
{
   hypre_LOBPCGData* pcg_data = (hypre_LOBPCGData*)pcg_vdata;

   *precond_data_ptr = (NALU_HYPRE_Solver)(pcg_data -> precondData);

   return hypre_error_flag;
}

NALU_HYPRE_Int
hypre_LOBPCGSetPrecond( void  *pcg_vdata,
                        NALU_HYPRE_Int  (*precond)(void*, void*, void*, void*),
                        NALU_HYPRE_Int  (*precond_setup)(void*, void*, void*, void*),
                        void  *precond_data )
{
   hypre_LOBPCGData* pcg_data = (hypre_LOBPCGData*)pcg_vdata;

   (pcg_data->precondFunctions).Precond      = precond;
   (pcg_data->precondFunctions).PrecondSetup = precond_setup;
   (pcg_data->precondData)                   = precond_data;

   return hypre_error_flag;
}

NALU_HYPRE_Int
hypre_LOBPCGSetPrintLevel( void *pcg_vdata, NALU_HYPRE_Int level )
{
   hypre_LOBPCGData *pcg_data = (hypre_LOBPCGData*)pcg_vdata;

   lobpcg_verbosityLevel(pcg_data->lobpcgData) = level;

   return hypre_error_flag;
}

void
hypre_LOBPCGPreconditioner( void *vdata, void* x, void* y )
{
   hypre_LOBPCGData *data = (hypre_LOBPCGData*)vdata;
   mv_InterfaceInterpreter* ii = data->interpreter;
   NALU_HYPRE_Int (*precond)(void*, void*, void*, void*) = (data->precondFunctions).Precond;

   if ( precond == NULL )
   {
      (*(ii->CopyVector))(x, y);
      return;
   }

   if ( lobpcg_precondUsageMode(data->lobpcgData) == 0 )
   {
      (*(ii->ClearVector))(y);
   }
   else
   {
      (*(ii->CopyVector))(x, y);
   }

   if ( data->T == NULL )
   {
      precond(data->precondData, data->A, x, y);
   }
   else
   {
      precond(data->precondData, data->T, x, y);
   }
}

void
hypre_LOBPCGOperatorA( void *pcg_vdata, void* x, void* y )
{
   hypre_LOBPCGData*           pcg_data    = (hypre_LOBPCGData*)pcg_vdata;
   NALU_HYPRE_MatvecFunctions * mv = pcg_data->matvecFunctions;
   void*                      matvec_data = (pcg_data -> matvecData);

   (*(mv->Matvec))(matvec_data, 1.0, pcg_data->A, x, 0.0, y);
}

void
hypre_LOBPCGOperatorB( void *pcg_vdata, void* x, void* y )
{
   hypre_LOBPCGData*           pcg_data    = (hypre_LOBPCGData*)pcg_vdata;
   mv_InterfaceInterpreter* ii          = pcg_data->interpreter;
   NALU_HYPRE_MatvecFunctions * mv = pcg_data->matvecFunctions;
   void*                       matvec_data = (pcg_data -> matvecDataB);

   if ( pcg_data->B == NULL )
   {
      (*(ii->CopyVector))(x, y);

      /* a test */
      /*
        (*(ii->ScaleVector))(2.0, y);
      */

      return;
   }

   (*(mv->Matvec))(matvec_data, 1.0, pcg_data->B, x, 0.0, y);
}

void
hypre_LOBPCGMultiPreconditioner( void *data, void * x, void*  y )
{
   hypre_LOBPCGData *pcg_data = (hypre_LOBPCGData*)data;
   mv_InterfaceInterpreter* ii = pcg_data->interpreter;

   ii->Eval( hypre_LOBPCGPreconditioner, data, x, y );
}

void
hypre_LOBPCGMultiOperatorA( void *data, void * x, void*  y )
{
   hypre_LOBPCGData *pcg_data = (hypre_LOBPCGData*)data;
   mv_InterfaceInterpreter* ii = pcg_data->interpreter;

   ii->Eval( hypre_LOBPCGOperatorA, data, x, y );
}

void
hypre_LOBPCGMultiOperatorB( void *data, void * x, void*  y )
{
   hypre_LOBPCGData *pcg_data = (hypre_LOBPCGData*)data;
   mv_InterfaceInterpreter* ii = pcg_data->interpreter;

   ii->Eval( hypre_LOBPCGOperatorB, data, x, y );
}

NALU_HYPRE_Int
hypre_LOBPCGSolve( void *vdata,
                   mv_MultiVectorPtr con,
                   mv_MultiVectorPtr vec,
                   NALU_HYPRE_Real* val )
{
   hypre_LOBPCGData* data = (hypre_LOBPCGData*)vdata;
   NALU_HYPRE_Int (*precond)(void*, void*, void*, void*) = (data->precondFunctions).Precond;
   void* opB = data->B;

   void (*prec)( void*, void*, void* );
   void (*operatorA)( void*, void*, void* );
   void (*operatorB)( void*, void*, void* );

   NALU_HYPRE_Int maxit = lobpcg_maxIterations(data->lobpcgData);
   NALU_HYPRE_Int verb  = lobpcg_verbosityLevel(data->lobpcgData);

   NALU_HYPRE_Int n = mv_MultiVectorWidth( vec );
   lobpcg_BLASLAPACKFunctions blap_fn;

   utilities_FortranMatrix* lambdaHistory;
   utilities_FortranMatrix* residuals;
   utilities_FortranMatrix* residualsHistory;

   lambdaHistory  = lobpcg_eigenvaluesHistory(data->lobpcgData);
   residuals = lobpcg_residualNorms(data->lobpcgData);
   residualsHistory = lobpcg_residualNormsHistory(data->lobpcgData);

   utilities_FortranMatrixAllocateData( n, maxit + 1, lambdaHistory );
   utilities_FortranMatrixAllocateData( n, 1,      residuals );
   utilities_FortranMatrixAllocateData( n, maxit + 1, residualsHistory );

   if ( precond != NULL )
   {
      prec = hypre_LOBPCGMultiPreconditioner;
   }
   else
   {
      prec = NULL;
   }

   operatorA = hypre_LOBPCGMultiOperatorA;

   if ( opB != NULL )
   {
      operatorB = hypre_LOBPCGMultiOperatorB;
   }
   else
   {
      operatorB = NULL;
   }

   blap_fn.dsygv = dsygv_interface;
   blap_fn.dpotrf = dpotrf_interface;

   lobpcg_solve( vec,
                 vdata, operatorA,
                 vdata, operatorB,
                 vdata, prec,
                 con,
                 blap_fn,
                 lobpcg_tolerance(data->lobpcgData), maxit, verb,
                 &(lobpcg_iterationNumber(data->lobpcgData)),
                 val,
                 utilities_FortranMatrixValues(lambdaHistory),
                 utilities_FortranMatrixGlobalHeight(lambdaHistory),
                 utilities_FortranMatrixValues(residuals),
                 utilities_FortranMatrixValues(residualsHistory),
                 utilities_FortranMatrixGlobalHeight(residualsHistory)
               );

   return hypre_error_flag;
}

utilities_FortranMatrix*
hypre_LOBPCGResidualNorms( void *vdata )
{
   hypre_LOBPCGData *data = (hypre_LOBPCGData*)vdata;
   return (lobpcg_residualNorms(data->lobpcgData));
}

utilities_FortranMatrix*
hypre_LOBPCGResidualNormsHistory( void *vdata )
{
   hypre_LOBPCGData *data = (hypre_LOBPCGData*)vdata;
   return (lobpcg_residualNormsHistory(data->lobpcgData));
}

utilities_FortranMatrix*
hypre_LOBPCGEigenvaluesHistory( void *vdata )
{
   hypre_LOBPCGData *data = (hypre_LOBPCGData*)vdata;
   return (lobpcg_eigenvaluesHistory(data->lobpcgData));
}

NALU_HYPRE_Int
hypre_LOBPCGIterations( void* vdata )
{
   hypre_LOBPCGData *data = (hypre_LOBPCGData*)vdata;
   return (lobpcg_iterationNumber(data->lobpcgData));
}


NALU_HYPRE_Int
NALU_HYPRE_LOBPCGCreate( mv_InterfaceInterpreter* ii, NALU_HYPRE_MatvecFunctions* mv,
                    NALU_HYPRE_Solver* solver )
{
   hypre_LOBPCGData *pcg_data;

   pcg_data = hypre_CTAlloc(hypre_LOBPCGData, 1, NALU_HYPRE_MEMORY_HOST);

   (pcg_data->precondFunctions).Precond = NULL;
   (pcg_data->precondFunctions).PrecondSetup = NULL;

   /* set defaults */

   (pcg_data->interpreter)               = ii;
   pcg_data->matvecFunctions             = mv;

   (pcg_data->matvecData)           = NULL;
   (pcg_data->B)                 = NULL;
   (pcg_data->matvecDataB)          = NULL;
   (pcg_data->T)                 = NULL;
   (pcg_data->matvecDataT)          = NULL;
   (pcg_data->precondData)          = NULL;

   lobpcg_initialize( &(pcg_data->lobpcgData) );

   *solver = (NALU_HYPRE_Solver)pcg_data;

   return hypre_error_flag;
}

NALU_HYPRE_Int
NALU_HYPRE_LOBPCGDestroy( NALU_HYPRE_Solver solver )
{
   return ( hypre_LOBPCGDestroy( (void *) solver ) );
}

NALU_HYPRE_Int
NALU_HYPRE_LOBPCGSetup( NALU_HYPRE_Solver solver,
                   NALU_HYPRE_Matrix A,
                   NALU_HYPRE_Vector b,
                   NALU_HYPRE_Vector x      )
{
   return ( hypre_LOBPCGSetup( solver, A, b, x ) );
}

NALU_HYPRE_Int
NALU_HYPRE_LOBPCGSetupB( NALU_HYPRE_Solver solver,
                    NALU_HYPRE_Matrix B,
                    NALU_HYPRE_Vector x      )
{
   return ( hypre_LOBPCGSetupB( solver, B, x ) );
}

NALU_HYPRE_Int
NALU_HYPRE_LOBPCGSetupT( NALU_HYPRE_Solver solver,
                    NALU_HYPRE_Matrix T,
                    NALU_HYPRE_Vector x      )
{
   return ( hypre_LOBPCGSetupT( solver, T, x ) );
}

NALU_HYPRE_Int
NALU_HYPRE_LOBPCGSolve( NALU_HYPRE_Solver solver, mv_MultiVectorPtr con,
                   mv_MultiVectorPtr vec, NALU_HYPRE_Real* val )
{
   return ( hypre_LOBPCGSolve( (void *) solver, con, vec, val ) );
}

NALU_HYPRE_Int
NALU_HYPRE_LOBPCGSetTol( NALU_HYPRE_Solver solver, NALU_HYPRE_Real tol )
{
   return ( hypre_LOBPCGSetTol( (void *) solver, tol ) );
}

NALU_HYPRE_Int
NALU_HYPRE_LOBPCGSetRTol( NALU_HYPRE_Solver solver, NALU_HYPRE_Real tol )
{
   return ( hypre_LOBPCGSetRTol( (void *) solver, tol ) );
}

NALU_HYPRE_Int
NALU_HYPRE_LOBPCGSetMaxIter( NALU_HYPRE_Solver solver, NALU_HYPRE_Int max_iter )
{
   return ( hypre_LOBPCGSetMaxIter( (void *) solver, max_iter ) );
}

NALU_HYPRE_Int
NALU_HYPRE_LOBPCGSetPrecondUsageMode( NALU_HYPRE_Solver solver, NALU_HYPRE_Int mode )
{
   return ( hypre_LOBPCGSetPrecondUsageMode( (void *) solver, mode ) );
}

NALU_HYPRE_Int
NALU_HYPRE_LOBPCGSetPrecond( NALU_HYPRE_Solver         solver,
                        NALU_HYPRE_PtrToSolverFcn precond,
                        NALU_HYPRE_PtrToSolverFcn precond_setup,
                        NALU_HYPRE_Solver         precond_solver )
{
   return ( hypre_LOBPCGSetPrecond( (void *) solver,
                                    (NALU_HYPRE_Int (*)(void*, void*, void*, void*))precond,
                                    (NALU_HYPRE_Int (*)(void*, void*, void*, void*))precond_setup,
                                    (void *) precond_solver ) );
}

NALU_HYPRE_Int
NALU_HYPRE_LOBPCGGetPrecond( NALU_HYPRE_Solver  solver,
                        NALU_HYPRE_Solver *precond_data_ptr )
{
   return ( hypre_LOBPCGGetPrecond( (void *)     solver,
                                    (NALU_HYPRE_Solver *) precond_data_ptr ) );
}

NALU_HYPRE_Int
NALU_HYPRE_LOBPCGSetPrintLevel( NALU_HYPRE_Solver solver, NALU_HYPRE_Int level )
{
   return ( hypre_LOBPCGSetPrintLevel( (void*)solver, level ) );
}

utilities_FortranMatrix*
NALU_HYPRE_LOBPCGResidualNorms( NALU_HYPRE_Solver solver )
{
   return ( hypre_LOBPCGResidualNorms( (void*)solver ) );
}

utilities_FortranMatrix*
NALU_HYPRE_LOBPCGResidualNormsHistory( NALU_HYPRE_Solver solver )
{
   return ( hypre_LOBPCGResidualNormsHistory( (void*)solver ) );
}

utilities_FortranMatrix*
NALU_HYPRE_LOBPCGEigenvaluesHistory( NALU_HYPRE_Solver solver )
{
   return ( hypre_LOBPCGEigenvaluesHistory( (void*)solver ) );
}

NALU_HYPRE_Int
NALU_HYPRE_LOBPCGIterations( NALU_HYPRE_Solver solver )
{
   return ( hypre_LOBPCGIterations( (void*)solver ) );
}

void
lobpcg_MultiVectorByMultiVector( mv_MultiVectorPtr x,
                                 mv_MultiVectorPtr y,
                                 utilities_FortranMatrix* xy )
{
   mv_MultiVectorByMultiVector( x, y,
                                utilities_FortranMatrixGlobalHeight( xy ),
                                utilities_FortranMatrixHeight( xy ),
                                utilities_FortranMatrixWidth( xy ),
                                utilities_FortranMatrixValues( xy ) );
}

