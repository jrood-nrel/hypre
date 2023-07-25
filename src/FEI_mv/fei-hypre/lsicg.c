/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * LSICG 
 *
 *****************************************************************************/

#include "utilities/_nalu_hypre_utilities.h"
#include "NALU_HYPRE.h"
#include "parcsr_ls/_nalu_hypre_parcsr_ls.h"
#include "parcsr_mv/_nalu_hypre_parcsr_mv.h"
#include "seq_mv/seq_mv.h"

/*--------------------------------------------------------------------------
 * nalu_hypre_LSICGData
 *--------------------------------------------------------------------------*/

typedef struct
{
   int    max_iter;
   int    stop_crit;
   double tol;
   double rel_residual_norm;

   void   *A;
   void   *r;
   void   *ap;
   void   *p;
   void   *z;

   void   *matvec_data;

   int    (*precond)(void*, void*, void*, void*);
   int    (*precond_setup)(void*, void*, void*, void*);
   void   *precond_data;

   int     num_iterations;
 
   int     logging;

} nalu_hypre_LSICGData;

/*--------------------------------------------------------------------------
 * nalu_hypre_LSICGCreate
 *--------------------------------------------------------------------------*/
 
void *nalu_hypre_LSICGCreate( )
{
   nalu_hypre_LSICGData *lsicg_data;
 
   lsicg_data = nalu_hypre_CTAlloc(nalu_hypre_LSICGData,  1, NALU_HYPRE_MEMORY_HOST);
 
   /* set defaults */
   (lsicg_data -> tol)            = 1.0e-06;
   (lsicg_data -> max_iter)       = 1000;
   (lsicg_data -> stop_crit)      = 0; /* rel. residual norm */
   (lsicg_data -> precond)        = nalu_hypre_ParKrylovIdentity;
   (lsicg_data -> precond_setup)  = nalu_hypre_ParKrylovIdentitySetup;
   (lsicg_data -> precond_data)   = NULL;
   (lsicg_data -> logging)        = 0;
   (lsicg_data -> r)              = NULL;
   (lsicg_data -> p)              = NULL;
   (lsicg_data -> ap)             = NULL;
   (lsicg_data -> z)              = NULL;
   (lsicg_data -> matvec_data)    = NULL;
 
   return (void *) lsicg_data;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_LSICGDestroy
 *--------------------------------------------------------------------------*/
 
int nalu_hypre_LSICGDestroy( void *lsicg_vdata )
{
	nalu_hypre_LSICGData *lsicg_data = (nalu_hypre_LSICGData *) lsicg_vdata;
   int             ierr = 0;
 
   if (lsicg_data)
   {
      nalu_hypre_ParKrylovMatvecDestroy(lsicg_data -> matvec_data);
      nalu_hypre_ParKrylovDestroyVector(lsicg_data -> r);
      nalu_hypre_ParKrylovDestroyVector(lsicg_data -> p);
      nalu_hypre_ParKrylovDestroyVector(lsicg_data -> ap);
      nalu_hypre_ParKrylovDestroyVector(lsicg_data -> z);
      nalu_hypre_TFree(lsicg_data, NALU_HYPRE_MEMORY_HOST);
   }
   return(ierr);
}

/*--------------------------------------------------------------------------
 * nalu_hypre_LSICGSetup
 *--------------------------------------------------------------------------*/
 
int nalu_hypre_LSICGSetup( void *lsicg_vdata, void *A, void *b, void *x         )
{
	nalu_hypre_LSICGData *lsicg_data       = (nalu_hypre_LSICGData *) lsicg_vdata;
   int            (*precond_setup)(void*, void*, void*, void*) = (lsicg_data -> precond_setup);
   void           *precond_data      = (lsicg_data -> precond_data);
   int            ierr = 0;
 
   (lsicg_data -> A) = A;
 
   /*--------------------------------------------------
    * The arguments for NewVector are important to
    * maintain consistency between the setup and
    * compute phases of matvec and the preconditioner.
    *--------------------------------------------------*/
 
   if ((lsicg_data -> r) == NULL)
      (lsicg_data -> r) = nalu_hypre_ParKrylovCreateVector(b);
   if ((lsicg_data -> p) == NULL)
      (lsicg_data -> p) = nalu_hypre_ParKrylovCreateVector(b);
   if ((lsicg_data -> z) == NULL)
      (lsicg_data -> z) = nalu_hypre_ParKrylovCreateVector(b);
   if ((lsicg_data -> ap) == NULL)
      (lsicg_data -> ap) = nalu_hypre_ParKrylovCreateVector(b);
   if ((lsicg_data -> matvec_data) == NULL)
      (lsicg_data -> matvec_data) = nalu_hypre_ParKrylovMatvecCreate(A, x);
 
   ierr = precond_setup(precond_data, A, b, x);
 
   return ierr;
}
 
/*--------------------------------------------------------------------------
 * nalu_hypre_LSICGSolve
 *-------------------------------------------------------------------------*/

int nalu_hypre_LSICGSolve(void  *lsicg_vdata, void  *A, void  *b, void  *x)
{
   int               ierr=0, mypid, nprocs, iter, converged=0;
   double            rhom1, rho, r_norm, b_norm, epsilon;
   double            sigma, alpha, beta, dArray[2], dArray2[2];
   nalu_hypre_Vector     *r_local, *z_local;
   MPI_Comm          comm;

   nalu_hypre_LSICGData  *lsicg_data    = (nalu_hypre_LSICGData *) lsicg_vdata;
   int 		     max_iter      = (lsicg_data -> max_iter);
   int 		     stop_crit     = (lsicg_data -> stop_crit);
   double 	     accuracy      = (lsicg_data -> tol);
   void             *matvec_data   = (lsicg_data -> matvec_data);
   void             *r             = (lsicg_data -> r);
   void             *p             = (lsicg_data -> p);
   void             *z             = (lsicg_data -> z);
   void             *ap            = (lsicg_data -> ap);
   int 	           (*precond)(void*, void*, void*, void*)    = (lsicg_data -> precond);
   int 	            *precond_data  = (int*)(lsicg_data -> precond_data);
   int               logging       = (lsicg_data -> logging);

   /* compute initial residual */

   r_local = nalu_hypre_ParVectorLocalVector((nalu_hypre_ParVector *) r);
   z_local = nalu_hypre_ParVectorLocalVector((nalu_hypre_ParVector *) z);
   comm    = nalu_hypre_ParCSRMatrixComm((nalu_hypre_ParCSRMatrix *) A);
   nalu_hypre_ParKrylovCommInfo(A,&mypid,&nprocs);
   nalu_hypre_ParKrylovCopyVector(b,r);
   nalu_hypre_ParKrylovMatvec(matvec_data,-1.0, A, x, 1.0, r);
   r_norm = sqrt(nalu_hypre_ParKrylovInnerProd(r,r));
   b_norm = sqrt(nalu_hypre_ParKrylovInnerProd(b,b));
   if (logging > 0)
   {
      if (mypid == 0)
      {
  	 printf("LSICG : L2 norm of b = %e\n", b_norm);
         if (b_norm == 0.0)
            printf("Rel_resid_norm actually contains the residual norm\n");
         printf("LSICG : Initial L2 norm of residual = %e\n", r_norm);
      }
   }

   /* set convergence criterion */

   if (b_norm > 0.0) epsilon = accuracy * b_norm;
   else              epsilon = accuracy * r_norm;
   if ( stop_crit )  epsilon = accuracy;

   iter = 0;
   nalu_hypre_ParKrylovClearVector(p);

   while ( converged == 0 )
   {
      while ( r_norm > epsilon && iter < max_iter )
      {
         iter++;
         if ( iter == 1 )
         {
            precond(precond_data, A, r, z);
            rhom1 = rho;
            rho   = nalu_hypre_ParKrylovInnerProd(r,z);
            beta = 0.0;
         }
         else beta = rho / rhom1;
         nalu_hypre_ParKrylovScaleVector( beta, p );
         nalu_hypre_ParKrylovAxpy(1.0e0, z, p);
         nalu_hypre_ParKrylovMatvec(matvec_data,1.0e0,A,p,0.0,ap);
         sigma = nalu_hypre_ParKrylovInnerProd(p,ap);
         alpha  = rho / sigma;
         if ( sigma == 0.0 )
         {
            printf("NALU_HYPRE::LSICG ERROR - sigma = 0.0.\n");
            ierr = 2;
            return ierr;
         }
         nalu_hypre_ParKrylovAxpy(alpha, p, x);
         nalu_hypre_ParKrylovAxpy(-alpha, ap, r);
         dArray[0] = nalu_hypre_SeqVectorInnerProd( r_local, r_local );
         precond(precond_data, A, r, z);
         rhom1 = rho;
         dArray[1] = nalu_hypre_SeqVectorInnerProd( r_local, z_local );
         MPI_Allreduce(dArray, dArray2, 2, MPI_DOUBLE, MPI_SUM, comm);
         rho = dArray2[1];
         r_norm = sqrt( dArray2[0] );
         if ( iter % 1 == 0 && mypid == 0 )
            printf("LSICG : iteration %d - residual norm = %e (%e)\n",
                   iter, r_norm, epsilon);
      }
      nalu_hypre_ParKrylovCopyVector(b,r);
      nalu_hypre_ParKrylovMatvec(matvec_data,-1.0, A, x, 1.0, r);
      r_norm = sqrt(nalu_hypre_ParKrylovInnerProd(r,r));
      if ( logging >= 1 && mypid == 0 )
         printf("LSICG actual residual norm = %e \n",r_norm);
      if ( r_norm < epsilon || iter >= max_iter ) converged = 1;
   }
   if ( iter >= max_iter ) ierr = 1;
   lsicg_data->rel_residual_norm = r_norm;
   lsicg_data->num_iterations    = iter;
   if ( logging >= 1 && mypid == 0 )
      printf("LSICG : total number of iterations = %d \n",iter);

   return ierr;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_LSICGSetTol
 *--------------------------------------------------------------------------*/
 
int nalu_hypre_LSICGSetTol( void *lsicg_vdata, double tol )
{
	nalu_hypre_LSICGData *lsicg_data = (nalu_hypre_LSICGData *) lsicg_vdata;
   (lsicg_data -> tol) = tol;
   return 0;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_LSICGSetMaxIter
 *--------------------------------------------------------------------------*/
 
int nalu_hypre_LSICGSetMaxIter( void *lsicg_vdata, int max_iter )
{
	nalu_hypre_LSICGData *lsicg_data = (nalu_hypre_LSICGData *) lsicg_vdata;
   (lsicg_data -> max_iter) = max_iter;
   return 0;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_LSICGSetStopCrit
 *--------------------------------------------------------------------------*/
 
int nalu_hypre_LSICGSetStopCrit( void *lsicg_vdata, double stop_crit )
{
	nalu_hypre_LSICGData *lsicg_data = (nalu_hypre_LSICGData *) lsicg_vdata;
   (lsicg_data -> stop_crit) = stop_crit;
   return 0;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_LSICGSetPrecond
 *--------------------------------------------------------------------------*/
 
int nalu_hypre_LSICGSetPrecond( void  *lsicg_vdata, int  (*precond)(void*,void*,void*,void*),
						   int  (*precond_setup)(void*,void*,void*,void*), void  *precond_data )
{
	nalu_hypre_LSICGData *lsicg_data = (nalu_hypre_LSICGData *) lsicg_vdata;
   (lsicg_data -> precond)        = precond;
   (lsicg_data -> precond_setup)  = precond_setup;
   (lsicg_data -> precond_data)   = precond_data;
   return 0;
}
 
/*--------------------------------------------------------------------------
 * nalu_hypre_LSICGSetLogging
 *--------------------------------------------------------------------------*/
 
int nalu_hypre_LSICGSetLogging( void *lsicg_vdata, int logging)
{
	nalu_hypre_LSICGData *lsicg_data = (nalu_hypre_LSICGData *) lsicg_vdata;
   (lsicg_data -> logging) = logging;
   return 0;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_LSICGGetNumIterations
 *--------------------------------------------------------------------------*/
 
int nalu_hypre_LSICGGetNumIterations(void *lsicg_vdata,int  *num_iterations)
{
	nalu_hypre_LSICGData *lsicg_data = (nalu_hypre_LSICGData *) lsicg_vdata;
   *num_iterations = (lsicg_data -> num_iterations);
   return 0;
}
 
/*--------------------------------------------------------------------------
 * nalu_hypre_LSICGGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/
 
int nalu_hypre_LSICGGetFinalRelativeResidualNorm(void *lsicg_vdata,
                                            double *relative_residual_norm)
{
	nalu_hypre_LSICGData *lsicg_data = (nalu_hypre_LSICGData *) lsicg_vdata;
   *relative_residual_norm = (lsicg_data -> rel_residual_norm);
   return 0;
} 

