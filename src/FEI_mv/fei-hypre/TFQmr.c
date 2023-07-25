/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * TFQmr 
 *
 *****************************************************************************/

#include "utilities/_nalu_hypre_utilities.h"
#include "NALU_HYPRE.h"
#include "IJ_mv/NALU_HYPRE_IJ_mv.h"
#include "parcsr_mv/NALU_HYPRE_parcsr_mv.h"
#include "parcsr_mv/_nalu_hypre_parcsr_mv.h"
#include "parcsr_ls/_nalu_hypre_parcsr_ls.h"
#include "parcsr_ls/NALU_HYPRE_parcsr_ls.h"

#include "_nalu_hypre_FEI.h"

/*--------------------------------------------------------------------------
 * nalu_hypre_TFQmrData
 *--------------------------------------------------------------------------*/

typedef struct
{
   int      max_iter;
   int      stop_crit;
   double   tol;
   double   rel_residual_norm;

   void  *A;
   void  *r;
   void  *tr;
   void  *yo;
   void  *ye;
   void  *t1;
   void  *t2;
   void  *w;
   void  *v;
   void  *d;
   void  *t3;

   void  *matvec_data;

   int    (*precond)(void*, void*, void*, void*);
   int    (*precond_setup)(void*, void*, void*, void*);
   void    *precond_data;

   /* log info (always logged) */
   int      num_iterations;
 
   /* additional log info (logged when `logging' > 0) */
   int      logging;
   double  *norms;
   char    *log_file_name;

} nalu_hypre_TFQmrData;

/*--------------------------------------------------------------------------
 * nalu_hypre_TFQmrCreate
 *--------------------------------------------------------------------------*/
 
void * nalu_hypre_TFQmrCreate( )
{
   nalu_hypre_TFQmrData *tfqmr_data;
 
   tfqmr_data = nalu_hypre_CTAlloc(nalu_hypre_TFQmrData,  1, NALU_HYPRE_MEMORY_HOST);
 
   /* set defaults */
   (tfqmr_data -> tol)            = 1.0e-06;
   (tfqmr_data -> max_iter)       = 1000;
   (tfqmr_data -> stop_crit)      = 0; /* rel. residual norm */
   (tfqmr_data -> precond)        = nalu_hypre_ParKrylovIdentity;
   (tfqmr_data -> precond_setup)  = nalu_hypre_ParKrylovIdentitySetup;
   (tfqmr_data -> precond_data)   = NULL;
   (tfqmr_data -> logging)        = 0;
   (tfqmr_data -> r)              = NULL;
   (tfqmr_data -> tr)             = NULL;
   (tfqmr_data -> yo)             = NULL;
   (tfqmr_data -> ye)             = NULL;
   (tfqmr_data -> t1)             = NULL;
   (tfqmr_data -> t2)             = NULL;
   (tfqmr_data -> w)              = NULL;
   (tfqmr_data -> v)              = NULL;
   (tfqmr_data -> d)              = NULL;
   (tfqmr_data -> t3)             = NULL;
   (tfqmr_data -> matvec_data)    = NULL;
   (tfqmr_data -> norms)          = NULL;
   (tfqmr_data -> log_file_name)  = NULL;
 
   return (void *) tfqmr_data;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_TFQmrDestroy
 *--------------------------------------------------------------------------*/
 
int nalu_hypre_TFQmrDestroy( void *tfqmr_vdata )
{
	nalu_hypre_TFQmrData *tfqmr_data = (nalu_hypre_TFQmrData *) tfqmr_vdata;
   int ierr = 0;
 
   if (tfqmr_data)
   {
      if ((tfqmr_data -> logging) > 0)
      {
         nalu_hypre_TFree(tfqmr_data -> norms, NALU_HYPRE_MEMORY_HOST);
      }
 
      nalu_hypre_ParKrylovMatvecDestroy(tfqmr_data -> matvec_data);
 
      nalu_hypre_ParKrylovDestroyVector(tfqmr_data -> r);
      nalu_hypre_ParKrylovDestroyVector(tfqmr_data -> tr);
      nalu_hypre_ParKrylovDestroyVector(tfqmr_data -> yo);
      nalu_hypre_ParKrylovDestroyVector(tfqmr_data -> ye);
      nalu_hypre_ParKrylovDestroyVector(tfqmr_data -> t1);
      nalu_hypre_ParKrylovDestroyVector(tfqmr_data -> t2);
      nalu_hypre_ParKrylovDestroyVector(tfqmr_data -> w);
      nalu_hypre_ParKrylovDestroyVector(tfqmr_data -> v);
      nalu_hypre_ParKrylovDestroyVector(tfqmr_data -> d);
      nalu_hypre_ParKrylovDestroyVector(tfqmr_data -> t3);
 
      nalu_hypre_TFree(tfqmr_data, NALU_HYPRE_MEMORY_HOST);
   }
 
   return(ierr);
}

/*--------------------------------------------------------------------------
 * nalu_hypre_TFQmrSetup
 *--------------------------------------------------------------------------*/
 
int nalu_hypre_TFQmrSetup( void *tfqmr_vdata, void *A, void *b, void *x         )
{
	nalu_hypre_TFQmrData *tfqmr_data     = (nalu_hypre_TFQmrData *) tfqmr_vdata;
   int            max_iter         = (tfqmr_data -> max_iter);
   int          (*precond_setup)(void*, void*, void*, void*) = (tfqmr_data -> precond_setup);
   void          *precond_data     = (tfqmr_data -> precond_data);
   int            ierr = 0;
 
   (tfqmr_data -> A) = A;
 
   /*--------------------------------------------------
    * The arguments for NewVector are important to
    * maintain consistency between the setup and
    * compute phases of matvec and the preconditioner.
    *--------------------------------------------------*/
 
   if ((tfqmr_data -> r) == NULL)
      (tfqmr_data -> r) = nalu_hypre_ParKrylovCreateVector(b);
   if ((tfqmr_data -> tr) == NULL)
      (tfqmr_data -> tr) = nalu_hypre_ParKrylovCreateVector(b);
   if ((tfqmr_data -> yo) == NULL)
      (tfqmr_data -> yo) = nalu_hypre_ParKrylovCreateVector(b);
   if ((tfqmr_data -> ye) == NULL)
      (tfqmr_data -> ye) = nalu_hypre_ParKrylovCreateVector(b);
   if ((tfqmr_data -> t1) == NULL)
      (tfqmr_data -> t1) = nalu_hypre_ParKrylovCreateVector(b);
   if ((tfqmr_data -> t2) == NULL)
      (tfqmr_data -> t2) = nalu_hypre_ParKrylovCreateVector(b);
   if ((tfqmr_data -> w) == NULL)
      (tfqmr_data -> w) = nalu_hypre_ParKrylovCreateVector(b);
   if ((tfqmr_data -> v) == NULL)
      (tfqmr_data -> v) = nalu_hypre_ParKrylovCreateVector(b);
   if ((tfqmr_data -> d) == NULL)
      (tfqmr_data -> d) = nalu_hypre_ParKrylovCreateVector(b);
   if ((tfqmr_data -> t3) == NULL)
      (tfqmr_data -> t3) = nalu_hypre_ParKrylovCreateVector(b);
   if ((tfqmr_data -> matvec_data) == NULL)
      (tfqmr_data -> matvec_data) = nalu_hypre_ParKrylovMatvecCreate(A, x);
 
   ierr = precond_setup(precond_data, A, b, x);
 
   /*-----------------------------------------------------
    * Allocate space for log info
    *-----------------------------------------------------*/
 
   if ((tfqmr_data -> logging) > 0)
   {
      if ((tfqmr_data -> norms) == NULL)
         (tfqmr_data -> norms) = nalu_hypre_CTAlloc(double,  max_iter + 1, NALU_HYPRE_MEMORY_HOST);
      if ((tfqmr_data -> log_file_name) == NULL)
		  (tfqmr_data -> log_file_name) = (char*)"tfqmr.out.log";
   }
 
   return ierr;
}
 
/*--------------------------------------------------------------------------
 * nalu_hypre_TFQmrSolve
 *-------------------------------------------------------------------------*/

int nalu_hypre_TFQmrSolve(void  *tfqmr_vdata, void  *A, void  *b, void  *x)
{
	nalu_hypre_TFQmrData  *tfqmr_data    = (nalu_hypre_TFQmrData *) tfqmr_vdata;
   int 		     max_iter      = (tfqmr_data -> max_iter);
   int 		     stop_crit     = (tfqmr_data -> stop_crit);
   double 	     accuracy      = (tfqmr_data -> tol);
   void             *matvec_data   = (tfqmr_data -> matvec_data);
 
   void             *r             = (tfqmr_data -> r);
   void             *tr            = (tfqmr_data -> tr);
   void             *yo            = (tfqmr_data -> yo);
   void             *ye            = (tfqmr_data -> ye);
   void             *t1            = (tfqmr_data -> t1);
   void             *t2            = (tfqmr_data -> t2);
   void             *w             = (tfqmr_data -> w);
   void             *v             = (tfqmr_data -> v);
   void             *d             = (tfqmr_data -> d);
   void             *t3            = (tfqmr_data -> t3);
   int 	           (*precond)(void*, void*, void*, void*)    = (tfqmr_data -> precond);
   int 	            *precond_data  = (int*)(tfqmr_data -> precond_data);

   /* logging variables */
   int               logging       = (tfqmr_data -> logging);
   double           *norms         = (tfqmr_data -> norms);
   
   int               ierr=0, my_id, num_procs, iter;
   double            eta, theta, tau, rhom1, rho, dtmp, r_norm, b_norm;
   double            rnbnd, etam1, thetam1, c, epsilon; 
   double            sigma, alpha, beta;

   nalu_hypre_ParKrylovCommInfo(A,&my_id,&num_procs);
   if (logging > 0)
   {
      norms          = (tfqmr_data -> norms);
   }

   /* initialize work arrays */

   nalu_hypre_ParKrylovCopyVector(b,r);

   /* compute initial residual */

   nalu_hypre_ParKrylovMatvec(matvec_data,-1.0, A, x, 1.0, r);
   r_norm = sqrt(nalu_hypre_ParKrylovInnerProd(r,r));
   b_norm = sqrt(nalu_hypre_ParKrylovInnerProd(b,b));
   if (logging > 0)
   {
      norms[0] = r_norm;
      if (my_id == 0)
      {
  	 printf("TFQmr : L2 norm of b = %e\n", b_norm);
         if (b_norm == 0.0)
            printf("Rel_resid_norm actually contains the residual norm\n");
         printf("TFQmr : Initial L2 norm of residual = %e\n", r_norm);
      }
      
   }
   iter = 0;

   if (b_norm > 0.0)
   {
      /* convergence criterion |r_i| <= accuracy*|b| if |b| > 0 */
      epsilon = accuracy * b_norm;
   }
   else
   {
      /* convergence criterion |r_i| <= accuracy*|r0| if |b| = 0 */
      epsilon = accuracy * r_norm;
   };

   /* convergence criterion |r_i| <= accuracy , absolute residual norm*/
   if (stop_crit) epsilon = accuracy;

   nalu_hypre_ParKrylovCopyVector(r,tr);
   nalu_hypre_ParKrylovCopyVector(r,yo);
   nalu_hypre_ParKrylovCopyVector(r,w);
   nalu_hypre_ParKrylovClearVector(d);
   nalu_hypre_ParKrylovClearVector(v);
   precond(precond_data, A, yo, t3);
   nalu_hypre_ParKrylovMatvec(matvec_data,1.0,A,t3,0.0,v);
   nalu_hypre_ParKrylovCopyVector(v,t1);

   tau   = r_norm;
   theta = 0.0;
   eta   = 0.0;
   rho   = r_norm * r_norm;
   
   while ( iter < max_iter && r_norm > epsilon )
   {
      iter++;

      sigma = nalu_hypre_ParKrylovInnerProd(tr,v);
      alpha = rho / sigma;
      nalu_hypre_ParKrylovCopyVector(yo,ye);
      dtmp = - alpha;
      nalu_hypre_ParKrylovAxpy(dtmp,v,ye);
      nalu_hypre_ParKrylovAxpy(dtmp,t1,w);

      thetam1 = theta;
      theta = sqrt(nalu_hypre_ParKrylovInnerProd(w,w)) / tau;
      c = 1.0 / sqrt(1.0 + theta * theta );
      tau = tau * theta * c;
      etam1 = eta;
      eta = c * c * alpha;

      dtmp = thetam1 * thetam1 * etam1 / alpha;
      nalu_hypre_ParKrylovCopyVector(d,t3);
      nalu_hypre_ParKrylovCopyVector(yo,d);
      nalu_hypre_ParKrylovAxpy(dtmp,t3,d);

      nalu_hypre_ParKrylovAxpy(eta,d,x);
      dtmp = 2.0 * iter;
      rnbnd = tau * sqrt( dtmp );

      precond(precond_data, A, ye, t3);
      nalu_hypre_ParKrylovMatvec(matvec_data,1.0,A,t3,0.0,t2);
      dtmp = - alpha;
      nalu_hypre_ParKrylovAxpy(dtmp,t2,w);

      thetam1 = theta;
      theta = sqrt(nalu_hypre_ParKrylovInnerProd(w,w)) / tau;
      c = 1.0 / sqrt(1.0 + theta * theta );
      tau = tau * theta * c;
      etam1 = eta;
      eta = c * c * alpha;
  
      dtmp = thetam1 * thetam1 * etam1 / alpha;
      nalu_hypre_ParKrylovCopyVector(d,t3);
      nalu_hypre_ParKrylovCopyVector(ye,d);
      nalu_hypre_ParKrylovAxpy(dtmp,t3,d);

      nalu_hypre_ParKrylovAxpy(eta,d,x);
      dtmp = 2.0 * iter + 1.0;
      rnbnd = tau * sqrt( dtmp );

      /* r_norm = theta * tau; */
      r_norm = rnbnd;

      if ( my_id == 0 && logging )
         printf(" TFQmr : iter %4d - res. norm = %e \n", iter, r_norm);

      rhom1 = rho;
      rho = nalu_hypre_ParKrylovInnerProd(tr,w);
      beta = rho / rhom1;

      nalu_hypre_ParKrylovCopyVector(w,yo);
      nalu_hypre_ParKrylovAxpy(beta,ye,yo);
     
      precond(precond_data, A, yo, t3);
      nalu_hypre_ParKrylovMatvec(matvec_data,1.0,A,t3,0.0,t1);
      nalu_hypre_ParKrylovCopyVector(t2,t3);
      nalu_hypre_ParKrylovAxpy(beta,v,t3);
      nalu_hypre_ParKrylovCopyVector(t1,v);
      nalu_hypre_ParKrylovAxpy(beta,t3,v);
   }
   precond(precond_data, A, x, t3);
   nalu_hypre_ParKrylovCopyVector(t3,x);

   (tfqmr_data -> num_iterations) = iter;
   if (b_norm > 0.0)
      (tfqmr_data -> rel_residual_norm) = r_norm/b_norm;
   if (b_norm == 0.0)
      (tfqmr_data -> rel_residual_norm) = r_norm;

   if (iter >= max_iter && r_norm > epsilon) ierr = 1;

   return ierr;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_TFQmrSetTol
 *--------------------------------------------------------------------------*/
 
int nalu_hypre_TFQmrSetTol( void *tfqmr_vdata, double tol )
{
	nalu_hypre_TFQmrData *tfqmr_data = (nalu_hypre_TFQmrData *) tfqmr_vdata;
   int            ierr = 0;
 
   (tfqmr_data -> tol) = tol;
 
   return ierr;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_TFQmrSetMaxIter
 *--------------------------------------------------------------------------*/
 
int nalu_hypre_TFQmrSetMaxIter( void *tfqmr_vdata, int max_iter )
{
	nalu_hypre_TFQmrData *tfqmr_data = (nalu_hypre_TFQmrData *) tfqmr_vdata;
   int              ierr = 0;
 
   (tfqmr_data -> max_iter) = max_iter;
 
   return ierr;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_TFQmrSetStopCrit
 *--------------------------------------------------------------------------*/ 
 
int nalu_hypre_TFQmrSetStopCrit( void *tfqmr_vdata, double stop_crit )
{
	nalu_hypre_TFQmrData *tfqmr_data = (nalu_hypre_TFQmrData *) tfqmr_vdata;
   int            ierr = 0;
 
   (tfqmr_data -> stop_crit) = stop_crit;
 
   return ierr;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_TFQmrSetPrecond
 *--------------------------------------------------------------------------*/
 
int nalu_hypre_TFQmrSetPrecond( void  *tfqmr_vdata, int  (*precond)(void*,void*,void*,void*),
						   int  (*precond_setup)(void*,void*,void*,void*), void  *precond_data )
{
	nalu_hypre_TFQmrData *tfqmr_data = (nalu_hypre_TFQmrData *) tfqmr_vdata;
   int              ierr = 0;
 
   (tfqmr_data -> precond)        = precond;
   (tfqmr_data -> precond_setup)  = precond_setup;
   (tfqmr_data -> precond_data)   = precond_data;
 
   return ierr;
}
 
/*--------------------------------------------------------------------------
 * nalu_hypre_TFQmrSetLogging
 *--------------------------------------------------------------------------*/
 
int nalu_hypre_TFQmrSetLogging( void *tfqmr_vdata, int logging)
{
	nalu_hypre_TFQmrData *tfqmr_data = (nalu_hypre_TFQmrData *) tfqmr_vdata;
   int              ierr = 0;
 
   (tfqmr_data -> logging) = logging;
 
   return ierr;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_TFQmrGetNumIterations
 *--------------------------------------------------------------------------*/
 
int nalu_hypre_TFQmrGetNumIterations(void *tfqmr_vdata,int  *num_iterations)
{
	nalu_hypre_TFQmrData *tfqmr_data = (nalu_hypre_TFQmrData *) tfqmr_vdata;
   int              ierr = 0;
 
   *num_iterations = (tfqmr_data -> num_iterations);
 
   return ierr;
}
 
/*--------------------------------------------------------------------------
 * nalu_hypre_TFQmrGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/
 
int nalu_hypre_TFQmrGetFinalRelativeResidualNorm( void   *tfqmr_vdata,
                                         double *relative_residual_norm )
{
	nalu_hypre_TFQmrData *tfqmr_data = (nalu_hypre_TFQmrData *) tfqmr_vdata;
   int 		ierr = 0;
 
   *relative_residual_norm = (tfqmr_data -> rel_residual_norm);
   
   return ierr;
} 

