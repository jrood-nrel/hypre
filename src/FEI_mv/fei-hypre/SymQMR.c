/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Symmetric QMR 
 *
 *****************************************************************************/

#include "utilities/_nalu_hypre_utilities.h"
#include "NALU_HYPRE.h"
#include "IJ_mv/NALU_HYPRE_IJ_mv.h"
#include "parcsr_mv/NALU_HYPRE_parcsr_mv.h"
#include "parcsr_mv/_nalu_hypre_parcsr_mv.h"
#include "parcsr_ls/_nalu_hypre_parcsr_ls.h"
#include "parcsr_ls/NALU_HYPRE_parcsr_ls.h"

/*--------------------------------------------------------------------------
 * nalu_hypre_SymQMRData
 *--------------------------------------------------------------------------*/

typedef struct
{
   int      max_iter;
   int      stop_crit;
   double   tol;
   double   rel_residual_norm;

   void  *A;
   void  *r;
   void  *q;
   void  *u;
   void  *d;
   void  *t;
   void  *rq;

   void  *matvec_data;

	int    (*precond)(void*,void*,void*,void*);
	int    (*precond_setup)(void*,void*,void*,void*);
   void    *precond_data;

   /* log info (always logged) */
   int      num_iterations;
 
   /* additional log info (logged when `logging' > 0) */
   int      logging;
   double  *norms;
   char    *log_file_name;

} nalu_hypre_SymQMRData;

/*--------------------------------------------------------------------------
 * nalu_hypre_SymQMRCreate
 *--------------------------------------------------------------------------*/
 
void * nalu_hypre_SymQMRCreate( )
{
   nalu_hypre_SymQMRData *symqmr_data;
 
   symqmr_data = nalu_hypre_CTAlloc(nalu_hypre_SymQMRData,  1, NALU_HYPRE_MEMORY_HOST);
 
   /* set defaults */
   (symqmr_data -> tol)            = 1.0e-06;
   (symqmr_data -> max_iter)       = 1000;
   (symqmr_data -> stop_crit)      = 0; /* rel. residual norm */
   (symqmr_data -> precond)        = nalu_hypre_ParKrylovIdentity;
   (symqmr_data -> precond_setup)  = nalu_hypre_ParKrylovIdentitySetup;
   (symqmr_data -> precond_data)   = NULL;
   (symqmr_data -> logging)        = 0;
   (symqmr_data -> r)              = NULL;
   (symqmr_data -> q)              = NULL;
   (symqmr_data -> u)              = NULL;
   (symqmr_data -> d)              = NULL;
   (symqmr_data -> t)              = NULL;
   (symqmr_data -> rq)             = NULL;
   (symqmr_data -> matvec_data)    = NULL;
   (symqmr_data -> norms)          = NULL;
   (symqmr_data -> log_file_name)  = NULL;
 
   return (void *) symqmr_data;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_SymQMRDestroy
 *--------------------------------------------------------------------------*/
 
int nalu_hypre_SymQMRDestroy( void *symqmr_vdata )
{
	nalu_hypre_SymQMRData *symqmr_data = (nalu_hypre_SymQMRData*) symqmr_vdata;
   int ierr = 0;
 
   if (symqmr_data)
   {
      if ((symqmr_data -> logging) > 0)
      {
         nalu_hypre_TFree(symqmr_data -> norms, NALU_HYPRE_MEMORY_HOST);
      }
 
      nalu_hypre_ParKrylovMatvecDestroy(symqmr_data -> matvec_data);
 
      nalu_hypre_ParKrylovDestroyVector(symqmr_data -> r);
      nalu_hypre_ParKrylovDestroyVector(symqmr_data -> q);
      nalu_hypre_ParKrylovDestroyVector(symqmr_data -> u);
      nalu_hypre_ParKrylovDestroyVector(symqmr_data -> d);
      nalu_hypre_ParKrylovDestroyVector(symqmr_data -> t);
      nalu_hypre_ParKrylovDestroyVector(symqmr_data -> rq);
 
      nalu_hypre_TFree(symqmr_data, NALU_HYPRE_MEMORY_HOST);
   }
 
   return(ierr);
}

/*--------------------------------------------------------------------------
 * nalu_hypre_SymQMRSetup
 *--------------------------------------------------------------------------*/
 
int nalu_hypre_SymQMRSetup( void *symqmr_vdata, void *A, void *b, void *x         )
{
	nalu_hypre_SymQMRData *symqmr_data   = (nalu_hypre_SymQMRData*) symqmr_vdata;
   int            max_iter         = (symqmr_data -> max_iter);
   int          (*precond_setup)(void*, void*, void*, void*) = (symqmr_data -> precond_setup);
   void          *precond_data     = (symqmr_data -> precond_data);
   int            ierr = 0;
 
   (symqmr_data -> A) = A;
 
   /*--------------------------------------------------
    * The arguments for NewVector are important to
    * maintain consistency between the setup and
    * compute phases of matvec and the preconditioner.
    *--------------------------------------------------*/
 
   if ((symqmr_data -> r) == NULL)
      (symqmr_data -> r) = nalu_hypre_ParKrylovCreateVector(b);
   if ((symqmr_data -> q) == NULL)
      (symqmr_data -> q) = nalu_hypre_ParKrylovCreateVector(b);
   if ((symqmr_data -> u) == NULL)
      (symqmr_data -> u) = nalu_hypre_ParKrylovCreateVector(b);
   if ((symqmr_data -> d) == NULL)
      (symqmr_data -> d) = nalu_hypre_ParKrylovCreateVector(b);
   if ((symqmr_data -> t) == NULL)
      (symqmr_data -> t) = nalu_hypre_ParKrylovCreateVector(b);
   if ((symqmr_data -> rq) == NULL)
      (symqmr_data -> rq) = nalu_hypre_ParKrylovCreateVector(b);
   if ((symqmr_data -> matvec_data) == NULL)
      (symqmr_data -> matvec_data) = nalu_hypre_ParKrylovMatvecCreate(A, x);
 
   ierr = precond_setup(precond_data, A, b, x);
 
   /*-----------------------------------------------------
    * Allocate space for log info
    *-----------------------------------------------------*/
 
   if ((symqmr_data -> logging) > 0)
   {
      if ((symqmr_data -> norms) == NULL)
         (symqmr_data -> norms) = nalu_hypre_CTAlloc(double,  max_iter + 1, NALU_HYPRE_MEMORY_HOST);
      if ((symqmr_data -> log_file_name) == NULL)
		  (symqmr_data -> log_file_name) = (char*)"symqmr.out.log";
   }
 
   return ierr;
}
 
/*--------------------------------------------------------------------------
 * nalu_hypre_SymQMRSolve
 *-------------------------------------------------------------------------*/

int nalu_hypre_SymQMRSolve(void  *symqmr_vdata, void  *A, void  *b, void  *x)
{
	nalu_hypre_SymQMRData  *symqmr_data    = (nalu_hypre_SymQMRData*) symqmr_vdata;
   int 		     max_iter      = (symqmr_data -> max_iter);
   int 		     stop_crit     = (symqmr_data -> stop_crit);
   double 	     accuracy      = (symqmr_data -> tol);
   void             *matvec_data   = (symqmr_data -> matvec_data);
 
   void             *r             = (symqmr_data -> r);
   void             *q             = (symqmr_data -> q);
   void             *u             = (symqmr_data -> u);
   void             *d             = (symqmr_data -> d);
   void             *t             = (symqmr_data -> t);
   void             *rq            = (symqmr_data -> rq);
   int 	           (*precond)(void*, void*, void*, void*)    = (symqmr_data -> precond);
   int 	            *precond_data  = (int*)(symqmr_data -> precond_data);

   /* logging variables */
   int               logging       = (symqmr_data -> logging);
   double           *norms         = (symqmr_data -> norms);
   
   int               ierr=0, my_id, num_procs, iter;
   double            theta, tau, rhom1, rho, dtmp, r_norm;
   double            thetam1, c, epsilon; 
   double            sigma, alpha, beta;

   nalu_hypre_ParKrylovCommInfo(A,&my_id,&num_procs);
   if (logging > 0)
   {
      norms          = (symqmr_data -> norms);
   }

   /* initialize work arrays */

   nalu_hypre_ParKrylovCopyVector(b,r);

   /* compute initial residual */

   nalu_hypre_ParKrylovMatvec(matvec_data,-1.0, A, x, 1.0, r);
   r_norm = sqrt(nalu_hypre_ParKrylovInnerProd(r,r));
   if (logging > 0)
   {
      norms[0] = r_norm;
      if (my_id == 0)
         printf("SymQMR : Initial L2 norm of residual = %e\n", r_norm);
   }
   iter = 0;
   epsilon = accuracy * r_norm;

   /* convergence criterion |r_i| <= accuracy , absolute residual norm*/
   if (stop_crit) epsilon = accuracy;

   while ( iter < max_iter && r_norm > epsilon )
   {
      if ( my_id == 0 && iter > 0 && logging ) printf("SymQMR restart... \n");

      tau = r_norm;
      precond(precond_data, A, r, q);
      rho = nalu_hypre_ParKrylovInnerProd(r,q);
      theta = 0.0;
      nalu_hypre_ParKrylovClearVector(d);
      nalu_hypre_ParKrylovCopyVector(r,rq);

      while ( iter < max_iter && r_norm > epsilon )
      {
         iter++;

         nalu_hypre_ParKrylovMatvec(matvec_data,1.0,A,q,0.0,t);
         sigma = nalu_hypre_ParKrylovInnerProd(q,t);
         if ( sigma == 0.0 )
         {
            printf("SymQMR ERROR : sigma = 0.0\n");
            exit(1);
         }
         alpha = rho / sigma;
         dtmp = - alpha;
         nalu_hypre_ParKrylovAxpy(dtmp,t,r);
         thetam1 = theta;
         theta = sqrt(nalu_hypre_ParKrylovInnerProd(r,r)) / tau;
         c = 1.0 / sqrt(1.0 + theta * theta );
         tau = tau * theta * c;
         dtmp = c * c * thetam1 * thetam1;
         nalu_hypre_ParKrylovScaleVector(dtmp,d);
         dtmp = c * c * alpha;
         nalu_hypre_ParKrylovAxpy(dtmp,q,d);
         dtmp = 1.0;
         nalu_hypre_ParKrylovAxpy(dtmp,d,x);

         precond(precond_data, A, r, u);
         rhom1 = rho;
         rho = nalu_hypre_ParKrylovInnerProd(r,u);
         beta = rho / rhom1;
         nalu_hypre_ParKrylovScaleVector(beta,q);
         dtmp = 1.0;
         nalu_hypre_ParKrylovAxpy(dtmp,u,q);

         dtmp = 1.0 - c * c;
         nalu_hypre_ParKrylovScaleVector(dtmp,rq);
         dtmp = c * c;
         nalu_hypre_ParKrylovAxpy(dtmp,r,rq);
         r_norm = sqrt(nalu_hypre_ParKrylovInnerProd(rq,rq));
         norms[iter] = r_norm;

         if ( my_id == 0 && logging )
            printf(" SymQMR : iteration %4d - residual norm = %e \n", 
                   iter, r_norm);
      }

      /* compute true residual */

      nalu_hypre_ParKrylovCopyVector(b,r);
      nalu_hypre_ParKrylovMatvec(matvec_data,-1.0, A, x, 1.0, r);
      r_norm = sqrt(nalu_hypre_ParKrylovInnerProd(r,r));
   }

   (symqmr_data -> num_iterations)    = iter;
   (symqmr_data -> rel_residual_norm) = r_norm;

   if (iter >= max_iter && r_norm > epsilon) ierr = 1;

   return ierr;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_SymQMRSetTol
 *--------------------------------------------------------------------------*/
 
int nalu_hypre_SymQMRSetTol( void *symqmr_vdata, double tol )
{
	nalu_hypre_SymQMRData *symqmr_data = (nalu_hypre_SymQMRData*) symqmr_vdata;
   int            ierr = 0;
 
   (symqmr_data -> tol) = tol;
 
   return ierr;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_SymQMRSetMaxIter
 *--------------------------------------------------------------------------*/
 
int nalu_hypre_SymQMRSetMaxIter( void *symqmr_vdata, int max_iter )
{
	nalu_hypre_SymQMRData *symqmr_data = (nalu_hypre_SymQMRData*) symqmr_vdata;
   int              ierr = 0;
 
   (symqmr_data -> max_iter) = max_iter;
 
   return ierr;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_SymQMRSetStopCrit
 *--------------------------------------------------------------------------*/
 
int nalu_hypre_SymQMRSetStopCrit( void *symqmr_vdata, double stop_crit )
{
	nalu_hypre_SymQMRData *symqmr_data = (nalu_hypre_SymQMRData*) symqmr_vdata;
   int            ierr = 0;
 
   (symqmr_data -> stop_crit) = stop_crit;
 
   return ierr;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_SymQMRSetPrecond
 *--------------------------------------------------------------------------*/
 
int nalu_hypre_SymQMRSetPrecond( void  *symqmr_vdata, int  (*precond)(void*,void*,void*,void*),
							int  (*precond_setup)(void*,void*,void*,void*), void  *precond_data )
{
	nalu_hypre_SymQMRData *symqmr_data = (nalu_hypre_SymQMRData*) symqmr_vdata;
   int              ierr = 0;
 
   (symqmr_data -> precond)        = precond;
   (symqmr_data -> precond_setup)  = precond_setup;
   (symqmr_data -> precond_data)   = precond_data;
 
   return ierr;
}
 
/*--------------------------------------------------------------------------
 * nalu_hypre_SymQMRSetLogging
 *--------------------------------------------------------------------------*/
 
int nalu_hypre_SymQMRSetLogging( void *symqmr_vdata, int logging)
{
	nalu_hypre_SymQMRData *symqmr_data = (nalu_hypre_SymQMRData*) symqmr_vdata;
   int              ierr = 0;
 
   (symqmr_data -> logging) = logging;
 
   return ierr;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_SymQMRGetNumIterations
 *--------------------------------------------------------------------------*/
 
int nalu_hypre_SymQMRGetNumIterations(void *symqmr_vdata,int  *num_iterations)
{
	nalu_hypre_SymQMRData *symqmr_data = (nalu_hypre_SymQMRData*) symqmr_vdata;
   int              ierr = 0;
 
   *num_iterations = (symqmr_data -> num_iterations);
 
   return ierr;
}
 
/*--------------------------------------------------------------------------
 * nalu_hypre_SymQMRGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/
 
int nalu_hypre_SymQMRGetFinalRelativeResidualNorm( void   *symqmr_vdata,
                                         double *relative_residual_norm )
{
	nalu_hypre_SymQMRData *symqmr_data = (nalu_hypre_SymQMRData*) symqmr_vdata;
   int 		ierr = 0;
 
   *relative_residual_norm = (symqmr_data -> rel_residual_norm);
   
   return ierr;
} 

