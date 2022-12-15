/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * NALU_HYPRE_SStructSplit solver interface
 *
 * This solver does the following iteration:
 *
 *    x_{k+1} = M^{-1} (b + N x_k) ,
 *
 * where A = M - N is a splitting of A, and M is the block-diagonal
 * matrix of structured intra-variable couplings.
 *
 *****************************************************************************/

#include "_nalu_hypre_sstruct_ls.h"

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

typedef struct nalu_hypre_SStructSolver_struct
{
   nalu_hypre_SStructVector     *y;

   NALU_HYPRE_Int                nparts;
   NALU_HYPRE_Int               *nvars;

   void                 ****smatvec_data;

   NALU_HYPRE_Int            (***ssolver_solve)();
   NALU_HYPRE_Int            (***ssolver_destroy)();
   void                  ***ssolver_data;

   NALU_HYPRE_Real               tol;
   NALU_HYPRE_Int                max_iter;
   NALU_HYPRE_Int                zero_guess;
   NALU_HYPRE_Int                num_iterations;
   NALU_HYPRE_Real               rel_norm;
   NALU_HYPRE_Int                ssolver;

   void                    *matvec_data;

} nalu_hypre_SStructSolver;

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructSplitCreate( MPI_Comm             comm,
                          NALU_HYPRE_SStructSolver *solver_ptr )
{
   nalu_hypre_SStructSolver *solver;

   solver = nalu_hypre_TAlloc(nalu_hypre_SStructSolver,  1, NALU_HYPRE_MEMORY_HOST);

   (solver -> y)               = NULL;
   (solver -> nparts)          = 0;
   (solver -> nvars)           = 0;
   (solver -> smatvec_data)    = NULL;
   (solver -> ssolver_solve)   = NULL;
   (solver -> ssolver_destroy) = NULL;
   (solver -> ssolver_data)    = NULL;
   (solver -> tol)             = 1.0e-06;
   (solver -> max_iter)        = 200;
   (solver -> zero_guess)      = 0;
   (solver -> num_iterations)  = 0;
   (solver -> rel_norm)        = 0;
   (solver -> ssolver)         = NALU_HYPRE_SMG;
   (solver -> matvec_data)     = NULL;

   *solver_ptr = solver;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructSplitDestroy( NALU_HYPRE_SStructSolver solver )
{
   nalu_hypre_SStructVector     *y;
   NALU_HYPRE_Int                nparts;
   NALU_HYPRE_Int               *nvars;
   void                 ****smatvec_data;
   NALU_HYPRE_Int            (***ssolver_solve)();
   NALU_HYPRE_Int            (***ssolver_destroy)();
   void                  ***ssolver_data;

   NALU_HYPRE_Int              (*sdestroy)(void *);
   void                    *sdata;

   NALU_HYPRE_Int                part, vi, vj;

   if (solver)
   {
      y               = (solver -> y);
      nparts          = (solver -> nparts);
      nvars           = (solver -> nvars);
      smatvec_data    = (solver -> smatvec_data);
      ssolver_solve   = (solver -> ssolver_solve);
      ssolver_destroy = (solver -> ssolver_destroy);
      ssolver_data    = (solver -> ssolver_data);

      NALU_HYPRE_SStructVectorDestroy(y);
      for (part = 0; part < nparts; part++)
      {
         for (vi = 0; vi < nvars[part]; vi++)
         {
            for (vj = 0; vj < nvars[part]; vj++)
            {
               if (smatvec_data[part][vi][vj] != NULL)
               {
                  nalu_hypre_StructMatvecDestroy(smatvec_data[part][vi][vj]);
               }
            }
            nalu_hypre_TFree(smatvec_data[part][vi], NALU_HYPRE_MEMORY_HOST);
            sdestroy = (NALU_HYPRE_Int (*)(void *))ssolver_destroy[part][vi];
            sdata = ssolver_data[part][vi];
            sdestroy(sdata);
         }
         nalu_hypre_TFree(smatvec_data[part], NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(ssolver_solve[part], NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(ssolver_destroy[part], NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(ssolver_data[part], NALU_HYPRE_MEMORY_HOST);
      }
      nalu_hypre_TFree(nvars, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(smatvec_data, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(ssolver_solve, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(ssolver_destroy, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(ssolver_data, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_SStructMatvecDestroy(solver -> matvec_data);
      nalu_hypre_TFree(solver, NALU_HYPRE_MEMORY_HOST);
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructSplitSetup( NALU_HYPRE_SStructSolver solver,
                         NALU_HYPRE_SStructMatrix A,
                         NALU_HYPRE_SStructVector b,
                         NALU_HYPRE_SStructVector x )
{
   nalu_hypre_SStructVector     *y;
   NALU_HYPRE_Int                nparts;
   NALU_HYPRE_Int               *nvars;
   void                 ****smatvec_data;
   NALU_HYPRE_Int            (***ssolver_solve)();
   NALU_HYPRE_Int            (***ssolver_destroy)();
   void                  ***ssolver_data;
   NALU_HYPRE_Int                ssolver          = (solver -> ssolver);

   MPI_Comm                 comm;
   nalu_hypre_SStructGrid       *grid;
   nalu_hypre_SStructPMatrix    *pA;
   nalu_hypre_SStructPVector    *px;
   nalu_hypre_SStructPVector    *py;
   nalu_hypre_StructMatrix      *sA;
   nalu_hypre_StructVector      *sx;
   nalu_hypre_StructVector      *sy;
   NALU_HYPRE_StructMatrix      sAH;
   NALU_HYPRE_StructVector      sxH;
   NALU_HYPRE_StructVector      syH;
   NALU_HYPRE_Int              (*ssolve)();
   NALU_HYPRE_Int              (*sdestroy)();
   void                    *sdata;

   NALU_HYPRE_Int                part, vi, vj;

   comm = nalu_hypre_SStructVectorComm(b);
   grid = nalu_hypre_SStructVectorGrid(b);
   NALU_HYPRE_SStructVectorCreate(comm, grid, &y);
   NALU_HYPRE_SStructVectorInitialize(y);
   NALU_HYPRE_SStructVectorAssemble(y);

   nparts = nalu_hypre_SStructMatrixNParts(A);
   nvars = nalu_hypre_TAlloc(NALU_HYPRE_Int,  nparts, NALU_HYPRE_MEMORY_HOST);
   smatvec_data    = nalu_hypre_TAlloc(void ***,  nparts, NALU_HYPRE_MEMORY_HOST);

   // RL: TODO TAlloc?
   ssolver_solve   = (NALU_HYPRE_Int (***)()) nalu_hypre_MAlloc((sizeof(NALU_HYPRE_Int (**)()) * nparts),
                                                      NALU_HYPRE_MEMORY_HOST);
   ssolver_destroy = (NALU_HYPRE_Int (***)()) nalu_hypre_MAlloc((sizeof(NALU_HYPRE_Int (**)()) * nparts),
                                                      NALU_HYPRE_MEMORY_HOST);
#if defined(NALU_HYPRE_USING_MEMORY_TRACKER)
   nalu_hypre_MemoryTrackerInsert1("malloc", ssolver_solve, sizeof(NALU_HYPRE_Int (**)()) * nparts,
                              nalu_hypre_GetActualMemLocation(NALU_HYPRE_MEMORY_HOST), __FILE__, __func__, __LINE__);
   nalu_hypre_MemoryTrackerInsert1("malloc", ssolver_destroy, sizeof(NALU_HYPRE_Int (**)()) * nparts,
                              nalu_hypre_GetActualMemLocation(NALU_HYPRE_MEMORY_HOST), __FILE__, __func__, __LINE__);
#endif

   ssolver_data    = nalu_hypre_TAlloc(void **,  nparts, NALU_HYPRE_MEMORY_HOST);
   for (part = 0; part < nparts; part++)
   {
      pA = nalu_hypre_SStructMatrixPMatrix(A, part);
      px = nalu_hypre_SStructVectorPVector(x, part);
      py = nalu_hypre_SStructVectorPVector(y, part);
      nvars[part] = nalu_hypre_SStructPMatrixNVars(pA);

      smatvec_data[part]    = nalu_hypre_TAlloc(void **,  nvars[part], NALU_HYPRE_MEMORY_HOST);

      // RL: TODO TAlloc?
      ssolver_solve[part]   =
         (NALU_HYPRE_Int (**)()) nalu_hypre_MAlloc((sizeof(NALU_HYPRE_Int (*)()) * nvars[part]), NALU_HYPRE_MEMORY_HOST);
      ssolver_destroy[part] =
         (NALU_HYPRE_Int (**)()) nalu_hypre_MAlloc((sizeof(NALU_HYPRE_Int (*)()) * nvars[part]), NALU_HYPRE_MEMORY_HOST);
#if defined(NALU_HYPRE_USING_MEMORY_TRACKER)
      nalu_hypre_MemoryTrackerInsert1("malloc", ssolver_solve[part], sizeof(NALU_HYPRE_Int (*)()) * nvars[part],
                                 nalu_hypre_GetActualMemLocation(NALU_HYPRE_MEMORY_HOST), __FILE__, __func__, __LINE__);
      nalu_hypre_MemoryTrackerInsert1("malloc", ssolver_destroy[part], sizeof(NALU_HYPRE_Int (*)()) * nvars[part],
                                 nalu_hypre_GetActualMemLocation(NALU_HYPRE_MEMORY_HOST), __FILE__, __func__, __LINE__);
#endif

      ssolver_data[part]    = nalu_hypre_TAlloc(void *,  nvars[part], NALU_HYPRE_MEMORY_HOST);
      for (vi = 0; vi < nvars[part]; vi++)
      {
         smatvec_data[part][vi] = nalu_hypre_TAlloc(void *,  nvars[part], NALU_HYPRE_MEMORY_HOST);
         for (vj = 0; vj < nvars[part]; vj++)
         {
            sA = nalu_hypre_SStructPMatrixSMatrix(pA, vi, vj);
            sx = nalu_hypre_SStructPVectorSVector(px, vj);
            smatvec_data[part][vi][vj] = NULL;
            if (sA != NULL)
            {
               smatvec_data[part][vi][vj] = nalu_hypre_StructMatvecCreate();
               nalu_hypre_StructMatvecSetup(smatvec_data[part][vi][vj], sA, sx);
            }
         }

         sA = nalu_hypre_SStructPMatrixSMatrix(pA, vi, vi);
         sx = nalu_hypre_SStructPVectorSVector(px, vi);
         sy = nalu_hypre_SStructPVectorSVector(py, vi);
         sAH = (NALU_HYPRE_StructMatrix) sA;
         sxH = (NALU_HYPRE_StructVector) sx;
         syH = (NALU_HYPRE_StructVector) sy;
         switch (ssolver)
         {
            default:
               /* If no solver is matched, use Jacobi, but throw and error */
               if (ssolver != NALU_HYPRE_Jacobi)
               {
                  nalu_hypre_error(NALU_HYPRE_ERROR_GENERIC);
               } /* don't break */
            case NALU_HYPRE_Jacobi:
               NALU_HYPRE_StructJacobiCreate(comm, (NALU_HYPRE_StructSolver *)&sdata);
               NALU_HYPRE_StructJacobiSetMaxIter((NALU_HYPRE_StructSolver)sdata, 1);
               NALU_HYPRE_StructJacobiSetTol((NALU_HYPRE_StructSolver)sdata, 0.0);
               if (solver -> zero_guess)
               {
                  NALU_HYPRE_StructJacobiSetZeroGuess((NALU_HYPRE_StructSolver)sdata);
               }
               NALU_HYPRE_StructJacobiSetup((NALU_HYPRE_StructSolver)sdata, sAH, syH, sxH);
               ssolve = (NALU_HYPRE_Int (*)())NALU_HYPRE_StructJacobiSolve;
               sdestroy = (NALU_HYPRE_Int (*)())NALU_HYPRE_StructJacobiDestroy;
               break;
            case NALU_HYPRE_SMG:
               NALU_HYPRE_StructSMGCreate(comm, (NALU_HYPRE_StructSolver *)&sdata);
               NALU_HYPRE_StructSMGSetMemoryUse((NALU_HYPRE_StructSolver)sdata, 0);
               NALU_HYPRE_StructSMGSetMaxIter((NALU_HYPRE_StructSolver)sdata, 1);
               NALU_HYPRE_StructSMGSetTol((NALU_HYPRE_StructSolver)sdata, 0.0);
               if (solver -> zero_guess)
               {
                  NALU_HYPRE_StructSMGSetZeroGuess((NALU_HYPRE_StructSolver)sdata);
               }
               NALU_HYPRE_StructSMGSetNumPreRelax((NALU_HYPRE_StructSolver)sdata, 1);
               NALU_HYPRE_StructSMGSetNumPostRelax((NALU_HYPRE_StructSolver)sdata, 1);
               NALU_HYPRE_StructSMGSetLogging((NALU_HYPRE_StructSolver)sdata, 0);
               NALU_HYPRE_StructSMGSetPrintLevel((NALU_HYPRE_StructSolver)sdata, 0);
               NALU_HYPRE_StructSMGSetup((NALU_HYPRE_StructSolver)sdata, sAH, syH, sxH);
               ssolve = (NALU_HYPRE_Int (*)())NALU_HYPRE_StructSMGSolve;
               sdestroy = (NALU_HYPRE_Int (*)())NALU_HYPRE_StructSMGDestroy;
               break;
            case NALU_HYPRE_PFMG:
               NALU_HYPRE_StructPFMGCreate(comm, (NALU_HYPRE_StructSolver *)&sdata);
               NALU_HYPRE_StructPFMGSetMaxIter((NALU_HYPRE_StructSolver)sdata, 1);
               NALU_HYPRE_StructPFMGSetTol((NALU_HYPRE_StructSolver)sdata, 0.0);
               if (solver -> zero_guess)
               {
                  NALU_HYPRE_StructPFMGSetZeroGuess((NALU_HYPRE_StructSolver)sdata);
               }
               NALU_HYPRE_StructPFMGSetRelaxType((NALU_HYPRE_StructSolver)sdata, 1);
               NALU_HYPRE_StructPFMGSetNumPreRelax((NALU_HYPRE_StructSolver)sdata, 1);
               NALU_HYPRE_StructPFMGSetNumPostRelax((NALU_HYPRE_StructSolver)sdata, 1);
               NALU_HYPRE_StructPFMGSetLogging((NALU_HYPRE_StructSolver)sdata, 0);
               NALU_HYPRE_StructPFMGSetPrintLevel((NALU_HYPRE_StructSolver)sdata, 0);
               NALU_HYPRE_StructPFMGSetup((NALU_HYPRE_StructSolver)sdata, sAH, syH, sxH);
               ssolve = (NALU_HYPRE_Int (*)())NALU_HYPRE_StructPFMGSolve;
               sdestroy = (NALU_HYPRE_Int (*)())NALU_HYPRE_StructPFMGDestroy;
               break;
         }
         ssolver_solve[part][vi]   = ssolve;
         ssolver_destroy[part][vi] = sdestroy;
         ssolver_data[part][vi]    = sdata;
      }
   }

   (solver -> y)               = y;
   (solver -> nparts)          = nparts;
   (solver -> nvars)           = nvars;
   (solver -> smatvec_data)    = smatvec_data;
   (solver -> ssolver_solve)   = ssolver_solve;
   (solver -> ssolver_destroy) = ssolver_destroy;
   (solver -> ssolver_data)    = ssolver_data;
   if ((solver -> tol) > 0.0)
   {
      nalu_hypre_SStructMatvecCreate(&(solver -> matvec_data));
      nalu_hypre_SStructMatvecSetup((solver -> matvec_data), A, x);
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructSplitSolve( NALU_HYPRE_SStructSolver solver,
                         NALU_HYPRE_SStructMatrix A,
                         NALU_HYPRE_SStructVector b,
                         NALU_HYPRE_SStructVector x )
{
   nalu_hypre_SStructVector     *y                = (solver -> y);
   NALU_HYPRE_Int                nparts           = (solver -> nparts);
   NALU_HYPRE_Int               *nvars            = (solver -> nvars);
   void                 ****smatvec_data     = (solver -> smatvec_data);
   NALU_HYPRE_Int            (***ssolver_solve)() = (solver -> ssolver_solve);
   void                  ***ssolver_data     = (solver -> ssolver_data);
   NALU_HYPRE_Real               tol              = (solver -> tol);
   NALU_HYPRE_Int                max_iter         = (solver -> max_iter);
   NALU_HYPRE_Int                zero_guess       = (solver -> zero_guess);
   void                    *matvec_data      = (solver -> matvec_data);

   nalu_hypre_SStructPMatrix    *pA;
   nalu_hypre_SStructPVector    *px;
   nalu_hypre_SStructPVector    *py;
   nalu_hypre_StructMatrix      *sA;
   nalu_hypre_StructVector      *sx;
   nalu_hypre_StructVector      *sy;
   NALU_HYPRE_Int              (*ssolve)(void*, nalu_hypre_StructMatrix*, nalu_hypre_StructVector*,
                                    nalu_hypre_StructVector*);
   void                    *sdata;
   nalu_hypre_ParCSRMatrix      *parcsrA;
   nalu_hypre_ParVector         *parx;
   nalu_hypre_ParVector         *pary;

   NALU_HYPRE_Int                iter, part, vi, vj;
   NALU_HYPRE_Real               b_dot_b = 0, r_dot_r;



   /* part of convergence check */
   if (tol > 0.0)
   {
      /* eps = (tol^2) */
      nalu_hypre_SStructInnerProd(b, b, &b_dot_b);

      /* if rhs is zero, return a zero solution */
      if (b_dot_b == 0.0)
      {
         nalu_hypre_SStructVectorSetConstantValues(x, 0.0);
         (solver -> rel_norm) = 0.0;

         return nalu_hypre_error_flag;
      }
   }

   for (iter = 0; iter < max_iter; iter++)
   {
      /* convergence check */
      if (tol > 0.0)
      {
         /* compute fine grid residual (b - Ax) */
         nalu_hypre_SStructCopy(b, y);
         nalu_hypre_SStructMatvecCompute(matvec_data, -1.0, A, x, 1.0, y);
         nalu_hypre_SStructInnerProd(y, y, &r_dot_r);
         (solver -> rel_norm) = sqrt(r_dot_r / b_dot_b);

         if ((solver -> rel_norm) < tol)
         {
            break;
         }
      }

      /* copy b into y */
      nalu_hypre_SStructCopy(b, y);

      /* compute y = y + Nx */
      if (!zero_guess || (iter > 0))
      {
         for (part = 0; part < nparts; part++)
         {
            pA = nalu_hypre_SStructMatrixPMatrix(A, part);
            px = nalu_hypre_SStructVectorPVector(x, part);
            py = nalu_hypre_SStructVectorPVector(y, part);
            for (vi = 0; vi < nvars[part]; vi++)
            {
               for (vj = 0; vj < nvars[part]; vj++)
               {
                  sdata = smatvec_data[part][vi][vj];
                  sy = nalu_hypre_SStructPVectorSVector(py, vi);
                  if ((sdata != NULL) && (vj != vi))
                  {
                     sA = nalu_hypre_SStructPMatrixSMatrix(pA, vi, vj);
                     sx = nalu_hypre_SStructPVectorSVector(px, vj);
                     nalu_hypre_StructMatvecCompute(sdata, -1.0, sA, sx, 1.0, sy);
                  }
               }
            }
         }
         parcsrA = nalu_hypre_SStructMatrixParCSRMatrix(A);
         nalu_hypre_SStructVectorConvert(x, &parx);
         nalu_hypre_SStructVectorConvert(y, &pary);
         nalu_hypre_ParCSRMatrixMatvec(-1.0, parcsrA, parx, 1.0, pary);
         nalu_hypre_SStructVectorRestore(x, NULL);
         nalu_hypre_SStructVectorRestore(y, pary);
      }

      /* compute x = M^{-1} y */
      for (part = 0; part < nparts; part++)
      {
         pA = nalu_hypre_SStructMatrixPMatrix(A, part);
         px = nalu_hypre_SStructVectorPVector(x, part);
         py = nalu_hypre_SStructVectorPVector(y, part);
         for (vi = 0; vi < nvars[part]; vi++)
         {
            ssolve = (NALU_HYPRE_Int (*)(void *, nalu_hypre_StructMatrix *, nalu_hypre_StructVector *,
                                    nalu_hypre_StructVector *))ssolver_solve[part][vi];
            sdata  = ssolver_data[part][vi];
            sA = nalu_hypre_SStructPMatrixSMatrix(pA, vi, vi);
            sx = nalu_hypre_SStructPVectorSVector(px, vi);
            sy = nalu_hypre_SStructPVectorSVector(py, vi);
            ssolve(sdata, sA, sy, sx);
         }
      }
   }

   (solver -> num_iterations) = iter;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructSplitSetTol( NALU_HYPRE_SStructSolver solver,
                          NALU_HYPRE_Real          tol )
{
   (solver -> tol) = tol;
   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructSplitSetMaxIter( NALU_HYPRE_SStructSolver solver,
                              NALU_HYPRE_Int           max_iter )
{
   (solver -> max_iter) = max_iter;
   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructSplitSetZeroGuess( NALU_HYPRE_SStructSolver solver )
{
   (solver -> zero_guess) = 1;
   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructSplitSetNonZeroGuess( NALU_HYPRE_SStructSolver solver )
{
   (solver -> zero_guess) = 0;
   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructSplitSetStructSolver( NALU_HYPRE_SStructSolver solver,
                                   NALU_HYPRE_Int           ssolver )
{
   (solver -> ssolver) = ssolver;
   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructSplitGetNumIterations( NALU_HYPRE_SStructSolver  solver,
                                    NALU_HYPRE_Int           *num_iterations )
{
   *num_iterations = (solver -> num_iterations);
   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructSplitGetFinalRelativeResidualNorm( NALU_HYPRE_SStructSolver  solver,
                                                NALU_HYPRE_Real          *norm )
{
   *norm = (solver -> rel_norm);
   return nalu_hypre_error_flag;
}
