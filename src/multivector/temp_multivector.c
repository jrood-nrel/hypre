/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include <math.h>
#include <stdlib.h>

#include "temp_multivector.h"
#include "interpreter.h"
#include "_nalu_hypre_utilities.h"

static void
mv_collectVectorPtr( NALU_HYPRE_Int* mask, mv_TempMultiVector* x, void** px )
{

   NALU_HYPRE_Int ix, jx;

   if ( mask != NULL )
   {
      for ( ix = 0, jx = 0; ix < x->numVectors; ix++ )
         if ( mask[ix] )
         {
            px[jx++] = x->vector[ix];
         }
   }
   else
      for ( ix = 0; ix < x->numVectors; ix++ )
      {
         px[ix] = x->vector[ix];
      }

}

static NALU_HYPRE_Int
aux_maskCount( NALU_HYPRE_Int n, NALU_HYPRE_Int* mask )
{

   NALU_HYPRE_Int i, m;

   if ( mask == NULL )
   {
      return n;
   }

   for ( i = m = 0; i < n; i++ )
      if ( mask[i] )
      {
         m++;
      }

   return m;
}

static void
aux_indexFromMask( NALU_HYPRE_Int n, NALU_HYPRE_Int* mask, NALU_HYPRE_Int* index )
{

   NALU_HYPRE_Int i, j;

   if ( mask != NULL )
   {
      for ( i = 0, j = 0; i < n; i++ )
         if ( mask[i] )
         {
            index[j++] = i + 1;
         }
   }
   else
      for ( i = 0; i < n; i++ )
      {
         index[i] = i + 1;
      }

}

/* ------- here goes simple random number generator --------- */

static nalu_hypre_ulongint next = 1;

/* RAND_MAX assumed to be 32767 */
static NALU_HYPRE_Int myrand(void)
{
   next = next * 1103515245 + 12345;
   return ((unsigned)(next / 65536) % 32768);
}

static void mysrand(unsigned seed)
{
   next = seed;
}


void*
mv_TempMultiVectorCreateFromSampleVector( void* ii_, NALU_HYPRE_Int n, void* sample )
{

   NALU_HYPRE_Int i;
   mv_TempMultiVector* x;
   mv_InterfaceInterpreter* ii = (mv_InterfaceInterpreter*)ii_;

   x = nalu_hypre_TAlloc(mv_TempMultiVector, 1, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_assert( x != NULL );

   x->interpreter = ii;
   x->numVectors = n;

   x->vector = nalu_hypre_CTAlloc(void*,  n, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_assert( x->vector != NULL );

   x->ownsVectors = 1;
   x->mask = NULL;
   x->ownsMask = 0;

   for ( i = 0; i < n; i++ )
   {
      x->vector[i] = (ii->CreateVector)(sample);
   }

   return x;

}

void*
mv_TempMultiVectorCreateCopy( void* src_, NALU_HYPRE_Int copyValues )
{

   NALU_HYPRE_Int i, n;

   mv_TempMultiVector* src;
   mv_TempMultiVector* dest;

   src = (mv_TempMultiVector*)src_;
   nalu_hypre_assert( src != NULL );

   n = src->numVectors;

   dest = (mv_TempMultiVector*)mv_TempMultiVectorCreateFromSampleVector( src->interpreter,
                                                                         n, src->vector[0] );
   if ( copyValues )
      for ( i = 0; i < n; i++ )
      {
         (dest->interpreter->CopyVector)(src->vector[i], dest->vector[i]);
      }

   return dest;
}

void
mv_TempMultiVectorDestroy( void* x_ )
{

   NALU_HYPRE_Int i;
   mv_TempMultiVector* x = (mv_TempMultiVector*)x_;

   if ( x == NULL )
   {
      return;
   }

   if ( x->ownsVectors && x->vector != NULL )
   {
      for ( i = 0; i < x->numVectors; i++ )
      {
         (x->interpreter->DestroyVector)(x->vector[i]);
      }
      nalu_hypre_TFree(x->vector, NALU_HYPRE_MEMORY_HOST);
   }
   if ( x->mask && x->ownsMask )
   {
      nalu_hypre_TFree(x->mask, NALU_HYPRE_MEMORY_HOST);
   }
   nalu_hypre_TFree(x, NALU_HYPRE_MEMORY_HOST);
}

NALU_HYPRE_Int
mv_TempMultiVectorWidth( void* x_ )
{

   mv_TempMultiVector* x = (mv_TempMultiVector*)x_;

   if ( x == NULL )
   {
      return 0;
   }

   return x->numVectors;
}

NALU_HYPRE_Int
mv_TempMultiVectorHeight( void* x_ )
{

   mv_TempMultiVector* x = (mv_TempMultiVector*)x_;

   if ( x == NULL )
   {
      return 0;
   }

   return (x->interpreter->VectorSize)(x->vector[0]);
}

/* this shallow copy of the mask is convenient but not safe;
   a proper copy should be considered */
void
mv_TempMultiVectorSetMask( void* x_, NALU_HYPRE_Int* mask )
{

   mv_TempMultiVector* x = (mv_TempMultiVector*)x_;

   nalu_hypre_assert( x != NULL );
   x->mask = mask;
   x->ownsMask = 0;
}

void
mv_TempMultiVectorClear( void* x_ )
{

   NALU_HYPRE_Int i;
   mv_TempMultiVector* x = (mv_TempMultiVector*)x_;

   nalu_hypre_assert( x != NULL );

   for ( i = 0; i < x->numVectors; i++ )
      if ( x->mask == NULL || (x->mask)[i] )
      {
         (x->interpreter->ClearVector)(x->vector[i]);
      }
}

void
mv_TempMultiVectorSetRandom( void* x_, NALU_HYPRE_Int seed )
{

   NALU_HYPRE_Int i;
   mv_TempMultiVector* x = (mv_TempMultiVector*)x_;

   nalu_hypre_assert( x != NULL );

   mysrand(seed);

   for ( i = 0; i < x->numVectors; i++ )
   {
      if ( x->mask == NULL || (x->mask)[i] )
      {
         seed = myrand();
         (x->interpreter->SetRandomValues)(x->vector[i], seed);
      }
   }
}



void
mv_TempMultiVectorCopy( void* src_, void* dest_ )
{

   NALU_HYPRE_Int i, ms, md;
   void** ps;
   void** pd;
   mv_TempMultiVector* src = (mv_TempMultiVector*)src_;
   mv_TempMultiVector* dest = (mv_TempMultiVector*)dest_;

   nalu_hypre_assert( src != NULL && dest != NULL );

   ms = aux_maskCount( src->numVectors, src->mask );
   md = aux_maskCount( dest->numVectors, dest->mask );
   nalu_hypre_assert( ms == md );

   ps = nalu_hypre_CTAlloc(void*,  ms, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_assert( ps != NULL );
   pd = nalu_hypre_CTAlloc(void*,  md, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_assert( pd != NULL );

   mv_collectVectorPtr( src->mask, src, ps );
   mv_collectVectorPtr( dest->mask, dest, pd );

   for ( i = 0; i < ms; i++ )
   {
      (src->interpreter->CopyVector)(ps[i], pd[i]);
   }

   nalu_hypre_TFree(ps, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(pd, NALU_HYPRE_MEMORY_HOST);
}

void
mv_TempMultiVectorAxpy( NALU_HYPRE_Complex a, void* x_, void* y_ )
{

   NALU_HYPRE_Int i, mx, my;
   void** px;
   void** py;
   mv_TempMultiVector* x;
   mv_TempMultiVector* y;

   x = (mv_TempMultiVector*)x_;
   y = (mv_TempMultiVector*)y_;
   nalu_hypre_assert( x != NULL && y != NULL );

   mx = aux_maskCount( x->numVectors, x->mask );
   my = aux_maskCount( y->numVectors, y->mask );
   nalu_hypre_assert( mx == my );

   px = nalu_hypre_CTAlloc(void*,  mx, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_assert( px != NULL );
   py = nalu_hypre_CTAlloc(void*,  my, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_assert( py != NULL );

   mv_collectVectorPtr( x->mask, x, px );
   mv_collectVectorPtr( y->mask, y, py );

   for ( i = 0; i < mx; i++ )
   {
      (x->interpreter->Axpy)(a, px[i], py[i]);
   }

   nalu_hypre_TFree(px, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(py, NALU_HYPRE_MEMORY_HOST);
}

void
mv_TempMultiVectorByMultiVector( void* x_, void* y_,
                                 NALU_HYPRE_Int xyGHeight, NALU_HYPRE_Int xyHeight,
                                 NALU_HYPRE_Int xyWidth, NALU_HYPRE_Complex* xyVal )
{
   /* xy = x'*y */

   NALU_HYPRE_Int ix, iy, mx, my, jxy;
   NALU_HYPRE_Complex* p;
   void** px;
   void** py;
   mv_TempMultiVector* x;
   mv_TempMultiVector* y;

   x = (mv_TempMultiVector*)x_;
   y = (mv_TempMultiVector*)y_;
   nalu_hypre_assert( x != NULL && y != NULL );

   mx = aux_maskCount( x->numVectors, x->mask );
   nalu_hypre_assert( mx == xyHeight );

   my = aux_maskCount( y->numVectors, y->mask );
   nalu_hypre_assert( my == xyWidth );

   px = nalu_hypre_CTAlloc(void*,  mx, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_assert( px != NULL );
   py = nalu_hypre_CTAlloc(void*,  my, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_assert( py != NULL );

   mv_collectVectorPtr( x->mask, x, px );
   mv_collectVectorPtr( y->mask, y, py );

   jxy = xyGHeight - xyHeight;
   for ( iy = 0, p = xyVal; iy < my; iy++ )
   {
      for ( ix = 0; ix < mx; ix++, p++ )
      {
         *p = (x->interpreter->InnerProd)(px[ix], py[iy]);
      }
      p += jxy;
   }

   nalu_hypre_TFree(px, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(py, NALU_HYPRE_MEMORY_HOST);

}

void
mv_TempMultiVectorByMultiVectorDiag( void* x_, void* y_,
                                     NALU_HYPRE_Int* mask, NALU_HYPRE_Int n, NALU_HYPRE_Complex* diag )
{
   /* diag = diag(x'*y) */

   NALU_HYPRE_Int i, mx, my, m;
   void** px;
   void** py;
   NALU_HYPRE_Int* index;
   mv_TempMultiVector* x;
   mv_TempMultiVector* y;

   x = (mv_TempMultiVector*)x_;
   y = (mv_TempMultiVector*)y_;
   nalu_hypre_assert( x != NULL && y != NULL );

   mx = aux_maskCount( x->numVectors, x->mask );
   my = aux_maskCount( y->numVectors, y->mask );
   m = aux_maskCount( n, mask );
   nalu_hypre_assert( mx == my && mx == m );

   px = nalu_hypre_CTAlloc(void*,  mx, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_assert( px != NULL );
   py = nalu_hypre_CTAlloc(void*,  my, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_assert( py != NULL );

   mv_collectVectorPtr( x->mask, x, px );
   mv_collectVectorPtr( y->mask, y, py );

   index = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  m, NALU_HYPRE_MEMORY_HOST);
   aux_indexFromMask( n, mask, index );

   for ( i = 0; i < m; i++ )
   {
      *(diag + index[i] - 1) = (x->interpreter->InnerProd)(px[i], py[i]);
   }

   nalu_hypre_TFree(index, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(px, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(py, NALU_HYPRE_MEMORY_HOST);

}

void
mv_TempMultiVectorByMatrix( void* x_,
                            NALU_HYPRE_Int rGHeight, NALU_HYPRE_Int rHeight,
                            NALU_HYPRE_Int rWidth, NALU_HYPRE_Complex* rVal,
                            void* y_ )
{

   NALU_HYPRE_Int i, j, jump;
   NALU_HYPRE_Int mx, my;
   NALU_HYPRE_Complex* p;
   void** px;
   void** py;
   mv_TempMultiVector* x;
   mv_TempMultiVector* y;

   x = (mv_TempMultiVector*)x_;
   y = (mv_TempMultiVector*)y_;
   nalu_hypre_assert( x != NULL && y != NULL );

   mx = aux_maskCount( x->numVectors, x->mask );
   my = aux_maskCount( y->numVectors, y->mask );

   nalu_hypre_assert( mx == rHeight && my == rWidth );

   px = nalu_hypre_CTAlloc(void*,  mx, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_assert( px != NULL );
   py = nalu_hypre_CTAlloc(void*,  my, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_assert( py != NULL );

   mv_collectVectorPtr( x->mask, x, px );
   mv_collectVectorPtr( y->mask, y, py );

   jump = rGHeight - rHeight;
   for ( j = 0, p = rVal; j < my; j++ )
   {
      (x->interpreter->ClearVector)( py[j] );
      for ( i = 0; i < mx; i++, p++ )
      {
         (x->interpreter->Axpy)(*p, px[i], py[j]);
      }
      p += jump;
   }

   nalu_hypre_TFree(px, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(py, NALU_HYPRE_MEMORY_HOST);
}

void
mv_TempMultiVectorXapy( void* x_,
                        NALU_HYPRE_Int rGHeight, NALU_HYPRE_Int rHeight,
                        NALU_HYPRE_Int rWidth, NALU_HYPRE_Complex* rVal,
                        void* y_ )
{

   NALU_HYPRE_Int i, j, jump;
   NALU_HYPRE_Int mx, my;
   NALU_HYPRE_Complex* p;
   void** px;
   void** py;
   mv_TempMultiVector* x;
   mv_TempMultiVector* y;

   x = (mv_TempMultiVector*)x_;
   y = (mv_TempMultiVector*)y_;
   nalu_hypre_assert( x != NULL && y != NULL );

   mx = aux_maskCount( x->numVectors, x->mask );
   my = aux_maskCount( y->numVectors, y->mask );

   nalu_hypre_assert( mx == rHeight && my == rWidth );

   px = nalu_hypre_CTAlloc(void*,  mx, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_assert( px != NULL );
   py = nalu_hypre_CTAlloc(void*,  my, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_assert( py != NULL );

   mv_collectVectorPtr( x->mask, x, px );
   mv_collectVectorPtr( y->mask, y, py );

   jump = rGHeight - rHeight;
   for ( j = 0, p = rVal; j < my; j++ )
   {
      for ( i = 0; i < mx; i++, p++ )
      {
         (x->interpreter->Axpy)(*p, px[i], py[j]);
      }
      p += jump;
   }

   nalu_hypre_TFree(px, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(py, NALU_HYPRE_MEMORY_HOST);
}

void
mv_TempMultiVectorByDiagonal( void* x_,
                              NALU_HYPRE_Int* mask, NALU_HYPRE_Int n, NALU_HYPRE_Complex* diag,
                              void* y_ )
{

   NALU_HYPRE_Int j;
   NALU_HYPRE_Int mx, my, m;
   void** px;
   void** py;
   NALU_HYPRE_Int* index;
   mv_TempMultiVector* x;
   mv_TempMultiVector* y;

   x = (mv_TempMultiVector*)x_;
   y = (mv_TempMultiVector*)y_;
   nalu_hypre_assert( x != NULL && y != NULL );

   mx = aux_maskCount( x->numVectors, x->mask );
   my = aux_maskCount( y->numVectors, y->mask );
   m = aux_maskCount( n, mask );

   nalu_hypre_assert( mx == m && my == m );

   if ( m < 1 )
   {
      return;
   }

   px = nalu_hypre_CTAlloc(void*,  mx, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_assert( px != NULL );
   py = nalu_hypre_CTAlloc(void*,  my, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_assert( py != NULL );

   index = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  m, NALU_HYPRE_MEMORY_HOST);
   aux_indexFromMask( n, mask, index );

   mv_collectVectorPtr( x->mask, x, px );
   mv_collectVectorPtr( y->mask, y, py );

   for ( j = 0; j < my; j++ )
   {
      (x->interpreter->ClearVector)(py[j]);
      (x->interpreter->Axpy)(diag[index[j] - 1], px[j], py[j]);
   }

   nalu_hypre_TFree(px, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(py, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree( index, NALU_HYPRE_MEMORY_HOST);
}

void
mv_TempMultiVectorEval( void (*f)( void*, void*, void* ), void* par,
                        void* x_, void* y_ )
{

   NALU_HYPRE_Int i, mx, my;
   void** px;
   void** py;
   mv_TempMultiVector* x;
   mv_TempMultiVector* y;

   x = (mv_TempMultiVector*)x_;
   y = (mv_TempMultiVector*)y_;
   nalu_hypre_assert( x != NULL && y != NULL );

   if ( f == NULL )
   {
      mv_TempMultiVectorCopy( x, y );
      return;
   }

   mx = aux_maskCount( x->numVectors, x->mask );
   my = aux_maskCount( y->numVectors, y->mask );
   nalu_hypre_assert( mx == my );

   px = nalu_hypre_CTAlloc(void*,  mx, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_assert( px != NULL );
   py = nalu_hypre_CTAlloc(void*,  my, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_assert( py != NULL );

   mv_collectVectorPtr( x->mask, x, px );
   mv_collectVectorPtr( y->mask, y, py );

   for ( i = 0; i < mx; i++ )
   {
      f( par, (void*)px[i], (void*)py[i] );
   }

   nalu_hypre_TFree(px, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(py, NALU_HYPRE_MEMORY_HOST);
}
