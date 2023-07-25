/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_nalu_hypre_FEI.h"

/******************************************************************************
 *
 * BiCGS 
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
 * nalu_hypre_BiCGSData
 *--------------------------------------------------------------------------*/

typedef struct
{
   int      max_iter;
   int      stop_crit;
   double   tol;
   double   rel_residual_norm;

   void  *A;
   void  *r;
   void  *p;
   void  *v;
   void  *q;
   void  *rh;
   void  *u;
   void  *t1;
   void  *t2;

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

} nalu_hypre_BiCGSData;

/*--------------------------------------------------------------------------
 * nalu_hypre_BiCGSCreate
 *--------------------------------------------------------------------------*/
 
void * nalu_hypre_BiCGSCreate( )
{
   nalu_hypre_BiCGSData *bicgs_data;
 
   bicgs_data = nalu_hypre_CTAlloc(nalu_hypre_BiCGSData,  1, NALU_HYPRE_MEMORY_HOST);
 
   /* set defaults */
   (bicgs_data -> tol)            = 1.0e-06;
   (bicgs_data -> max_iter)       = 1000;
   (bicgs_data -> stop_crit)      = 0; /* rel. residual norm */
   (bicgs_data -> precond)        = nalu_hypre_ParKrylovIdentity;
   (bicgs_data -> precond_setup)  = nalu_hypre_ParKrylovIdentitySetup;
   (bicgs_data -> precond_data)   = NULL;
   (bicgs_data -> logging)        = 0;
   (bicgs_data -> r)              = NULL;
   (bicgs_data -> rh)             = NULL;
   (bicgs_data -> p)              = NULL;
   (bicgs_data -> v)              = NULL;
   (bicgs_data -> q)              = NULL;
   (bicgs_data -> u)              = NULL;
   (bicgs_data -> t1)             = NULL;
   (bicgs_data -> t2)             = NULL;
   (bicgs_data -> matvec_data)    = NULL;
   (bicgs_data -> norms)          = NULL;
   (bicgs_data -> log_file_name)  = NULL;
 
   return (void *) bicgs_data;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_BiCGSDestroy
 *--------------------------------------------------------------------------*/
 
int nalu_hypre_BiCGSDestroy( void *bicgs_vdata )
{
	nalu_hypre_BiCGSData *bicgs_data = (nalu_hypre_BiCGSData *) bicgs_vdata;
   int ierr = 0;
 
   if (bicgs_data)
   {
      if ((bicgs_data -> logging) > 0)
      {
         nalu_hypre_TFree(bicgs_data -> norms, NALU_HYPRE_MEMORY_HOST);
      }
 
      nalu_hypre_ParKrylovMatvecDestroy(bicgs_data -> matvec_data);
 
      nalu_hypre_ParKrylovDestroyVector(bicgs_data -> r);
      nalu_hypre_ParKrylovDestroyVector(bicgs_data -> rh);
      nalu_hypre_ParKrylovDestroyVector(bicgs_data -> v);
      nalu_hypre_ParKrylovDestroyVector(bicgs_data -> p);
      nalu_hypre_ParKrylovDestroyVector(bicgs_data -> q);
      nalu_hypre_ParKrylovDestroyVector(bicgs_data -> u);
      nalu_hypre_ParKrylovDestroyVector(bicgs_data -> t1);
      nalu_hypre_ParKrylovDestroyVector(bicgs_data -> t2);
 
      nalu_hypre_TFree(bicgs_data, NALU_HYPRE_MEMORY_HOST);
   }
 
   return(ierr);
}

/*--------------------------------------------------------------------------
 * nalu_hypre_BiCGSSetup
 *--------------------------------------------------------------------------*/
 
int nalu_hypre_BiCGSSetup( void *bicgs_vdata, void *A, void *b, void *x         )
{
	nalu_hypre_BiCGSData *bicgs_data     = (nalu_hypre_BiCGSData *) bicgs_vdata;
   int            max_iter         = (bicgs_data -> max_iter);
   int          (*precond_setup)(void*, void*, void*, void*) = (bicgs_data -> precond_setup);
   void          *precond_data     = (bicgs_data -> precond_data);
   int            ierr = 0;
 
   (bicgs_data -> A) = A;
 
   /*--------------------------------------------------
    * The arguments for NewVector are important to
    * maintain consistency between the setup and
    * compute phases of matvec and the preconditioner.
    *--------------------------------------------------*/
 
   if ((bicgs_data -> r) == NULL)
      (bicgs_data -> r) = nalu_hypre_ParKrylovCreateVector(b);
   if ((bicgs_data -> rh) == NULL)
      (bicgs_data -> rh) = nalu_hypre_ParKrylovCreateVector(b);
   if ((bicgs_data -> v) == NULL)
      (bicgs_data -> v) = nalu_hypre_ParKrylovCreateVector(b);
   if ((bicgs_data -> p) == NULL)
      (bicgs_data -> p) = nalu_hypre_ParKrylovCreateVector(b);
   if ((bicgs_data -> q) == NULL)
      (bicgs_data -> q) = nalu_hypre_ParKrylovCreateVector(b);
   if ((bicgs_data -> u) == NULL)
      (bicgs_data -> u) = nalu_hypre_ParKrylovCreateVector(b);
   if ((bicgs_data -> t1) == NULL)
      (bicgs_data -> t1) = nalu_hypre_ParKrylovCreateVector(b);
   if ((bicgs_data -> t2) == NULL)
      (bicgs_data -> t2) = nalu_hypre_ParKrylovCreateVector(b);
   if ((bicgs_data -> matvec_data) == NULL)
      (bicgs_data -> matvec_data) = nalu_hypre_ParKrylovMatvecCreate(A, x);
 
   ierr = precond_setup(precond_data, A, b, x);
 
   /*-----------------------------------------------------
    * Allocate space for log info
    *-----------------------------------------------------*/
 
   if ((bicgs_data -> logging) > 0)
   {
      if ((bicgs_data -> norms) == NULL)
         (bicgs_data -> norms) = nalu_hypre_CTAlloc(double,  max_iter + 1, NALU_HYPRE_MEMORY_HOST);
      if ((bicgs_data -> log_file_name) == NULL)
		  (bicgs_data -> log_file_name) = (char*)"bicgs.out.log";
   }
 
   return ierr;
}
 
/*--------------------------------------------------------------------------
 * nalu_hypre_BiCGSSolve
 *-------------------------------------------------------------------------*/

int nalu_hypre_BiCGSSolve(void  *bicgs_vdata, void  *A, void  *b, void  *x)
{
	nalu_hypre_BiCGSData  *bicgs_data    = (nalu_hypre_BiCGSData *) bicgs_vdata;
   int 		     max_iter      = (bicgs_data -> max_iter);
   int 		     stop_crit     = (bicgs_data -> stop_crit);
   double 	     accuracy      = (bicgs_data -> tol);
   void             *matvec_data   = (bicgs_data -> matvec_data);
 
   void             *r             = (bicgs_data -> r);
   void             *rh            = (bicgs_data -> rh);
   void             *v             = (bicgs_data -> v);
   void             *p             = (bicgs_data -> p);
   void             *q             = (bicgs_data -> q);
   void             *u             = (bicgs_data -> u);
   void             *t1            = (bicgs_data -> t1);
   void             *t2            = (bicgs_data -> t2);
   int 	           (*precond)(void*, void*, void*, void*)    = (bicgs_data -> precond);
   int 	            *precond_data  = (int*)(bicgs_data -> precond_data);

   /* logging variables */
   int               logging       = (bicgs_data -> logging);
   double           *norms         = (bicgs_data -> norms);
   
   int               ierr=0, my_id, num_procs, iter;
   double            rho1, rho2, sigma, alpha, dtmp, r_norm, b_norm;
   double            beta, epsilon; 

   nalu_hypre_ParKrylovCommInfo(A,&my_id,&num_procs);
   if (logging > 0)
   {
      norms          = (bicgs_data -> norms);
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
  	 printf("BiCGS : L2 norm of b = %e\n", b_norm);
         if (b_norm == 0.0)
            printf("Rel_resid_norm actually contains the residual norm\n");
         printf("BiCGS : Initial L2 norm of residual = %e\n", r_norm);
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

   nalu_hypre_ParKrylovCopyVector(r,rh);
   nalu_hypre_ParKrylovClearVector(p);
   nalu_hypre_ParKrylovClearVector(q);
   rho2 = r_norm * r_norm;
   beta = rho2;

   while ( iter < max_iter && r_norm > epsilon )
   {
      iter++;

      rho1 = rho2;
      nalu_hypre_ParKrylovCopyVector(r,u);
      nalu_hypre_ParKrylovAxpy(beta,q,u);

      nalu_hypre_ParKrylovCopyVector(q,t1);
      nalu_hypre_ParKrylovAxpy(beta,p,t1);
      nalu_hypre_ParKrylovCopyVector(u,p);
      nalu_hypre_ParKrylovAxpy(beta,t1,p);

      precond(precond_data, A, p, t1);
      nalu_hypre_ParKrylovMatvec(matvec_data,1.0,A,t1,0.0,v);

      sigma = nalu_hypre_ParKrylovInnerProd(rh,v);
      alpha = rho1 / sigma;

      nalu_hypre_ParKrylovCopyVector(u,q);
      dtmp = - alpha;
      nalu_hypre_ParKrylovAxpy(dtmp,v,q);

      dtmp = 1.0;
      nalu_hypre_ParKrylovAxpy(dtmp,q,u);

      precond(precond_data, A, u, t1);
      nalu_hypre_ParKrylovAxpy(alpha,t1,x);

      nalu_hypre_ParKrylovMatvec(matvec_data,1.0,A,t1,0.0,t2);

      dtmp = - alpha;
      nalu_hypre_ParKrylovAxpy(dtmp,t2,r);

      rho2 = nalu_hypre_ParKrylovInnerProd(r,rh);
      beta = rho2 / rho1;

      r_norm = sqrt(nalu_hypre_ParKrylovInnerProd(r,r));

      if ( my_id == 0 && logging )
         printf(" BiCGS : iter %4d - res. norm = %e \n", iter, r_norm);
   }

   (bicgs_data -> num_iterations) = iter;
   if (b_norm > 0.0)
      (bicgs_data -> rel_residual_norm) = r_norm/b_norm;
   if (b_norm == 0.0)
      (bicgs_data -> rel_residual_norm) = r_norm;

   if (iter >= max_iter && r_norm > epsilon) ierr = 1;

   return ierr;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_BiCGSSetTol
 *--------------------------------------------------------------------------*/
 
int nalu_hypre_BiCGSSetTol( void *bicgs_vdata, double tol )
{
	nalu_hypre_BiCGSData *bicgs_data = (nalu_hypre_BiCGSData *) bicgs_vdata;
   int            ierr = 0;
 
   (bicgs_data -> tol) = tol;
 
   return ierr;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_BiCGSSetMaxIter
 *--------------------------------------------------------------------------*/
 
int nalu_hypre_BiCGSSetMaxIter( void *bicgs_vdata, int max_iter )
{
	nalu_hypre_BiCGSData *bicgs_data = (nalu_hypre_BiCGSData *) bicgs_vdata;
   int              ierr = 0;
 
   (bicgs_data -> max_iter) = max_iter;
 
   return ierr;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_BiCGSSetStopCrit
 *--------------------------------------------------------------------------*/
 
int nalu_hypre_BiCGSSetStopCrit( void *bicgs_vdata, double stop_crit )
{
	nalu_hypre_BiCGSData *bicgs_data = (nalu_hypre_BiCGSData *) bicgs_vdata;
   int            ierr = 0;
 
   (bicgs_data -> stop_crit) = stop_crit;
 
   return ierr;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_BiCGSSetPrecond
 *--------------------------------------------------------------------------*/
 
int nalu_hypre_BiCGSSetPrecond( void  *bicgs_vdata, int  (*precond)(void*,void*,void*,void*),
						   int  (*precond_setup)(void*,void*,void*,void*), void  *precond_data )
{
	nalu_hypre_BiCGSData *bicgs_data = (nalu_hypre_BiCGSData *) bicgs_vdata;
   int              ierr = 0;
 
   (bicgs_data -> precond)        = precond;
   (bicgs_data -> precond_setup)  = precond_setup;
   (bicgs_data -> precond_data)   = precond_data;
 
   return ierr;
}
 
/*--------------------------------------------------------------------------
 * nalu_hypre_BiCGSSetLogging
 *--------------------------------------------------------------------------*/
 
int nalu_hypre_BiCGSSetLogging( void *bicgs_vdata, int logging)
{
	nalu_hypre_BiCGSData *bicgs_data = (nalu_hypre_BiCGSData *) bicgs_vdata;
   int              ierr = 0;
 
   (bicgs_data -> logging) = logging;
 
   return ierr;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_BiCGSGetNumIterations
 *--------------------------------------------------------------------------*/
 
int nalu_hypre_BiCGSGetNumIterations(void *bicgs_vdata,int  *num_iterations)
{
	nalu_hypre_BiCGSData *bicgs_data = (nalu_hypre_BiCGSData *) bicgs_vdata;
   int              ierr = 0;
 
   *num_iterations = (bicgs_data -> num_iterations);
 
   return ierr;
}
 
/*--------------------------------------------------------------------------
 * nalu_hypre_BiCGSGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/
 
int nalu_hypre_BiCGSGetFinalRelativeResidualNorm( void   *bicgs_vdata,
                                         double *relative_residual_norm )
{
	nalu_hypre_BiCGSData *bicgs_data = (nalu_hypre_BiCGSData *) bicgs_vdata;
   int 		ierr = 0;
 
   *relative_residual_norm = (bicgs_data -> rel_residual_norm);
   
   return ierr;
} 

