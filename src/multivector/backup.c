/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include <math.h>

#include "temp_multivector.h"

void*
hypre_TempMultiVectorCreateFromSampleVector( void* ii_, NALU_HYPRE_Int n, void* sample )
{

   NALU_HYPRE_Int i;
   hypre_TempMultiVector* data;
   NALU_HYPRE_InterfaceInterpreter* ii = (NALU_HYPRE_InterfaceInterpreter*)ii_;

   data = hypre_TAlloc(hypre_TempMultiVector, 1, NALU_HYPRE_MEMORY_HOST);
   hypre_assert( data != NULL );

   data->interpreter = ii;
   data->numVectors = n;

   data->vector = hypre_CTAlloc(void*,  n, NALU_HYPRE_MEMORY_HOST);
   hypre_assert( data->vector != NULL );

   data->ownsVectors = 1;
   data->mask = NULL;
   data->ownsMask = 0;

   for ( i = 0; i < n; i++ )
   {
      data->vector[i] = (ii->CreateVector)(sample);
   }

   return data;

}

void*
hypre_TempMultiVectorCreateCopy( void* src_, NALU_HYPRE_Int copyValues )
{

   NALU_HYPRE_Int i, n;

   hypre_TempMultiVector* src;
   hypre_TempMultiVector* dest;

   src = (hypre_TempMultiVector*)src_;
   hypre_assert( src != NULL );

   n = src->numVectors;

   dest = hypre_TempMultiVectorCreateFromSampleVector( src->interpreter,
                                                       n, src->vector[0] );
   if ( copyValues )
      for ( i = 0; i < n; i++ )
      {
         (dest->interpreter->CopyVector)(src->vector[i], dest->vector[i]);
      }

   return dest;
}

void
hypre_TempMultiVectorDestroy( void* v_ )
{

   NALU_HYPRE_Int i;
   hypre_TempMultiVector* data = (hypre_TempMultiVector*)v_;

   if ( data == NULL )
   {
      return;
   }

   if ( data->ownsVectors && data->vector != NULL )
   {
      for ( i = 0; i < data->numVectors; i++ )
      {
         (data->interpreter->DestroyVector)(data->vector[i]);
      }
      hypre_TFree(data->vector, NALU_HYPRE_MEMORY_HOST);
   }
   if ( data->mask && data->ownsMask )
   {
      hypre_TFree(data->mask, NALU_HYPRE_MEMORY_HOST);
   }
   hypre_TFree(data, NALU_HYPRE_MEMORY_HOST);
}

NALU_HYPRE_Int
hypre_TempMultiVectorWidth( void* v )
{

   hypre_TempMultiVector* data = (hypre_TempMultiVector*)v;

   if ( data == NULL )
   {
      return 0;
   }

   return data->numVectors;
}

NALU_HYPRE_Int
hypre_TempMultiVectorHeight( void* v )
{

   return 0;
}

void
hypre_TempMultiVectorSetMask( void* v, NALU_HYPRE_Int* mask )
{

   hypre_TempMultiVector* data = (hypre_TempMultiVector*)v;

   hypre_assert( data != NULL );
   data->mask = mask;
   data->ownsMask = 0;
}

void
hypre_TempMultiVectorClear( void* v )
{

   NALU_HYPRE_Int i;
   hypre_TempMultiVector* data = (hypre_TempMultiVector*)v;

   hypre_assert( data != NULL );

   for ( i = 0; i < data->numVectors; i++ )
      if ( data->mask == NULL || (data->mask)[i] )
      {
         (data->interpreter->ClearVector)(data->vector[i]);
      }
}

void
hypre_TempMultiVectorSetRandom( void* v, NALU_HYPRE_Int seed )
{

   NALU_HYPRE_Int i;
   hypre_TempMultiVector* data = (hypre_TempMultiVector*)v;

   hypre_assert( data != NULL );

   hypre_SeedRand( seed );
   for ( i = 0; i < data->numVectors; i++ )
   {
      if ( data->mask == NULL || (data->mask)[i] )
      {
         seed = hypre_RandI();
         (data->interpreter->SetRandomValues)(data->vector[i], seed);
      }
   }
}

void
hypre_collectVectorPtr( NALU_HYPRE_Int* mask, hypre_TempMultiVector* x, void** px )
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

void
hypre_TempMultiVectorCopy( void* src, void* dest )
{

   NALU_HYPRE_Int i, ms, md;
   void** ps;
   void** pd;
   hypre_TempMultiVector* srcData = (hypre_TempMultiVector*)src;
   hypre_TempMultiVector* destData = (hypre_TempMultiVector*)dest;

   hypre_assert( srcData != NULL && destData != NULL );

   ms = aux_maskCount( srcData->numVectors, srcData->mask );
   md = aux_maskCount( destData->numVectors, destData->mask );
   hypre_assert( ms == md );

   ps = hypre_CTAlloc(void*,  ms, NALU_HYPRE_MEMORY_HOST);
   hypre_assert( ps != NULL );
   pd = hypre_CTAlloc(void*,  md, NALU_HYPRE_MEMORY_HOST);
   hypre_assert( pd != NULL );

   hypre_collectVectorPtr( srcData->mask, srcData, ps );
   hypre_collectVectorPtr( destData->mask, destData, pd );

   for ( i = 0; i < ms; i++ )
   {
      (srcData->interpreter->CopyVector)(ps[i], pd[i]);
   }

   hypre_TFree(ps, NALU_HYPRE_MEMORY_HOST);
   hypre_TFree(pd, NALU_HYPRE_MEMORY_HOST);
}

void
hypre_TempMultiVectorAxpy( NALU_HYPRE_Complex a, void* x_, void* y_ )
{

   NALU_HYPRE_Int i, mx, my;
   void** px;
   void** py;
   hypre_TempMultiVector* xData;
   hypre_TempMultiVector* yData;

   xData = (hypre_TempMultiVector*)x_;
   yData = (hypre_TempMultiVector*)y_;
   hypre_assert( xData != NULL && yData != NULL );

   mx = aux_maskCount( xData->numVectors, xData->mask );
   my = aux_maskCount( yData->numVectors, yData->mask );
   hypre_assert( mx == my );

   px = hypre_CTAlloc(void*,  mx, NALU_HYPRE_MEMORY_HOST);
   hypre_assert( px != NULL );
   py = hypre_CTAlloc(void*,  my, NALU_HYPRE_MEMORY_HOST);
   hypre_assert( py != NULL );

   hypre_collectVectorPtr( xData->mask, xData, px );
   hypre_collectVectorPtr( yData->mask, yData, py );

   for ( i = 0; i < mx; i++ )
   {
      (xData->interpreter->Axpy)(a, px[i], py[i]);
   }

   hypre_TFree(px, NALU_HYPRE_MEMORY_HOST);
   hypre_TFree(py, NALU_HYPRE_MEMORY_HOST);
}

void
hypre_TempMultiVectorByMultiVector( void* x_, void* y_,
                                    NALU_HYPRE_Int xyGHeight, NALU_HYPRE_Int xyHeight,
                                    NALU_HYPRE_Int xyWidth, NALU_HYPRE_Complex* xyVal )
{
   /* xy = x'*y */

   NALU_HYPRE_Int ix, iy, mx, my, jxy;
   NALU_HYPRE_Complex* p;
   void** px;
   void** py;
   hypre_TempMultiVector* xData;
   hypre_TempMultiVector* yData;

   xData = (hypre_TempMultiVector*)x_;
   yData = (hypre_TempMultiVector*)y_;
   hypre_assert( xData != NULL && yData != NULL );

   mx = aux_maskCount( xData->numVectors, xData->mask );
   hypre_assert( mx == xyHeight );

   my = aux_maskCount( yData->numVectors, yData->mask );
   hypre_assert( my == xyWidth );

   px = hypre_CTAlloc(void*,  mx, NALU_HYPRE_MEMORY_HOST);
   hypre_assert( px != NULL );
   py = hypre_CTAlloc(void*,  my, NALU_HYPRE_MEMORY_HOST);
   hypre_assert( py != NULL );

   hypre_collectVectorPtr( xData->mask, xData, px );
   hypre_collectVectorPtr( yData->mask, yData, py );

   jxy = xyGHeight - xyHeight;
   for ( iy = 0, p = xyVal; iy < my; iy++ )
   {
      for ( ix = 0; ix < mx; ix++, p++ )
      {
         *p = (xData->interpreter->InnerProd)(px[ix], py[iy]);
      }
      p += jxy;
   }

   hypre_TFree(px, NALU_HYPRE_MEMORY_HOST);
   hypre_TFree(py, NALU_HYPRE_MEMORY_HOST);
}

void
hypre_TempMultiVectorByMultiVectorDiag( void* x_, void* y_,
                                        NALU_HYPRE_Int* mask, NALU_HYPRE_Int n, NALU_HYPRE_Complex* diag )
{
   /* diag = diag(x'*y) */

   NALU_HYPRE_Int i, mx, my, m;
   void** px;
   void** py;
   NALU_HYPRE_Int* index;
   hypre_TempMultiVector* xData;
   hypre_TempMultiVector* yData;

   xData = (hypre_TempMultiVector*)x_;
   yData = (hypre_TempMultiVector*)y_;
   hypre_assert( xData != NULL && yData != NULL );

   mx = aux_maskCount( xData->numVectors, xData->mask );
   my = aux_maskCount( yData->numVectors, yData->mask );
   m = aux_maskCount( n, mask );
   hypre_assert( mx == my && mx == m );

   px = hypre_CTAlloc(void*,  mx, NALU_HYPRE_MEMORY_HOST);
   hypre_assert( px != NULL );
   py = hypre_CTAlloc(void*,  my, NALU_HYPRE_MEMORY_HOST);
   hypre_assert( py != NULL );

   hypre_collectVectorPtr( xData->mask, xData, px );
   hypre_collectVectorPtr( yData->mask, yData, py );

   index = hypre_CTAlloc(NALU_HYPRE_Int,  m, NALU_HYPRE_MEMORY_HOST);
   aux_indexFromMask( n, mask, index );

   for ( i = 0; i < m; i++ )
   {
      *(diag + index[i] - 1) = (xData->interpreter->InnerProd)(px[i], py[i]);
   }

   hypre_TFree(index, NALU_HYPRE_MEMORY_HOST);
   hypre_TFree(px, NALU_HYPRE_MEMORY_HOST);
   hypre_TFree(py, NALU_HYPRE_MEMORY_HOST);
}

void
hypre_TempMultiVectorByMatrix( void* x_,
                               NALU_HYPRE_Int rGHeight, NALU_HYPRE_Int rHeight,
                               NALU_HYPRE_Int rWidth, NALU_HYPRE_Complex* rVal,
                               void* y_ )
{

   NALU_HYPRE_Int i, j, jump;
   NALU_HYPRE_Int mx, my;
   NALU_HYPRE_Complex* p;
   void** px;
   void** py;
   hypre_TempMultiVector* xData;
   hypre_TempMultiVector* yData;

   xData = (hypre_TempMultiVector*)x_;
   yData = (hypre_TempMultiVector*)y_;
   hypre_assert( xData != NULL && yData != NULL );

   mx = aux_maskCount( xData->numVectors, xData->mask );
   my = aux_maskCount( yData->numVectors, yData->mask );

   hypre_assert( mx == rHeight && my == rWidth );

   px = hypre_CTAlloc(void*,  mx, NALU_HYPRE_MEMORY_HOST);
   hypre_assert( px != NULL );
   py = hypre_CTAlloc(void*,  my, NALU_HYPRE_MEMORY_HOST);
   hypre_assert( py != NULL );

   hypre_collectVectorPtr( xData->mask, xData, px );
   hypre_collectVectorPtr( yData->mask, yData, py );

   jump = rGHeight - rHeight;
   for ( j = 0, p = rVal; j < my; j++ )
   {
      (xData->interpreter->ClearVector)( py[j] );
      for ( i = 0; i < mx; i++, p++ )
      {
         (xData->interpreter->Axpy)(*p, px[i], py[j]);
      }
      p += jump;
   }

   hypre_TFree(px, NALU_HYPRE_MEMORY_HOST);
   hypre_TFree(py, NALU_HYPRE_MEMORY_HOST);
}

void
hypre_TempMultiVectorXapy( void* x_,
                           NALU_HYPRE_Int rGHeight, NALU_HYPRE_Int rHeight,
                           NALU_HYPRE_Int rWidth, NALU_HYPRE_Complex* rVal,
                           void* y_ )
{

   NALU_HYPRE_Int i, j, jump;
   NALU_HYPRE_Int mx, my;
   NALU_HYPRE_Complex* p;
   void** px;
   void** py;
   hypre_TempMultiVector* xData;
   hypre_TempMultiVector* yData;

   xData = (hypre_TempMultiVector*)x_;
   yData = (hypre_TempMultiVector*)y_;
   hypre_assert( xData != NULL && yData != NULL );

   mx = aux_maskCount( xData->numVectors, xData->mask );
   my = aux_maskCount( yData->numVectors, yData->mask );

   hypre_assert( mx == rHeight && my == rWidth );

   px = hypre_CTAlloc(void*,  mx, NALU_HYPRE_MEMORY_HOST);
   hypre_assert( px != NULL );
   py = hypre_CTAlloc(void*,  my, NALU_HYPRE_MEMORY_HOST);
   hypre_assert( py != NULL );

   hypre_collectVectorPtr( xData->mask, xData, px );
   hypre_collectVectorPtr( yData->mask, yData, py );

   jump = rGHeight - rHeight;
   for ( j = 0, p = rVal; j < my; j++ )
   {
      for ( i = 0; i < mx; i++, p++ )
      {
         (xData->interpreter->Axpy)(*p, px[i], py[j]);
      }
      p += jump;
   }

   hypre_TFree(px, NALU_HYPRE_MEMORY_HOST);
   hypre_TFree(py, NALU_HYPRE_MEMORY_HOST);
}

void
hypre_TempMultiVectorByDiagonal( void* x_,
                                 NALU_HYPRE_Int* mask, NALU_HYPRE_Int n, NALU_HYPRE_Complex* diag,
                                 void* y_ )
{

   NALU_HYPRE_Int j;
   NALU_HYPRE_Int mx, my, m;
   void** px;
   void** py;
   NALU_HYPRE_Int* index;
   hypre_TempMultiVector* xData;
   hypre_TempMultiVector* yData;

   xData = (hypre_TempMultiVector*)x_;
   yData = (hypre_TempMultiVector*)y_;
   hypre_assert( xData != NULL && yData != NULL );

   mx = aux_maskCount( xData->numVectors, xData->mask );
   my = aux_maskCount( yData->numVectors, yData->mask );
   m = aux_maskCount( n, mask );

   hypre_assert( mx == m && my == m );

   if ( m < 1 )
   {
      return;
   }

   px = hypre_CTAlloc(void*,  mx, NALU_HYPRE_MEMORY_HOST);
   hypre_assert( px != NULL );
   py = hypre_CTAlloc(void*,  my, NALU_HYPRE_MEMORY_HOST);
   hypre_assert( py != NULL );

   index = hypre_CTAlloc(NALU_HYPRE_Int,  m, NALU_HYPRE_MEMORY_HOST);
   aux_indexFromMask( n, mask, index );

   hypre_collectVectorPtr( xData->mask, xData, px );
   hypre_collectVectorPtr( yData->mask, yData, py );

   for ( j = 0; j < my; j++ )
   {
      (xData->interpreter->ClearVector)(py[j]);
      (xData->interpreter->Axpy)(diag[index[j] - 1], px[j], py[j]);
   }

   hypre_TFree(px, NALU_HYPRE_MEMORY_HOST);
   hypre_TFree(py, NALU_HYPRE_MEMORY_HOST);
   hypre_TFree(index, NALU_HYPRE_MEMORY_HOST);
}

void
hypre_TempMultiVectorEval( void (*f)( void*, void*, void* ), void* par,
                           void* x_, void* y_ )
{

   NALU_HYPRE_Int i, mx, my;
   void** px;
   void** py;
   hypre_TempMultiVector* x;
   hypre_TempMultiVector* y;

   x = (hypre_TempMultiVector*)x_;
   y = (hypre_TempMultiVector*)y_;
   hypre_assert( x != NULL && y != NULL );

   if ( f == NULL )
   {
      hypre_TempMultiVectorCopy( x, y );
      return;
   }

   mx = aux_maskCount( x->numVectors, x->mask );
   my = aux_maskCount( y->numVectors, y->mask );
   hypre_assert( mx == my );

   px = hypre_CTAlloc(void*,  mx, NALU_HYPRE_MEMORY_HOST);
   hypre_assert( px != NULL );
   py = hypre_CTAlloc(void*,  my, NALU_HYPRE_MEMORY_HOST);
   hypre_assert( py != NULL );

   hypre_collectVectorPtr( x->mask, x, px );
   hypre_collectVectorPtr( y->mask, y, py );

   for ( i = 0; i < mx; i++ )
   {
      f( par, (void*)px[i], (void*)py[i] );
   }

   hypre_TFree(px, NALU_HYPRE_MEMORY_HOST);
   hypre_TFree(py, NALU_HYPRE_MEMORY_HOST);
}

NALU_HYPRE_Int
hypre_TempMultiVectorPrint( void* x_, const char* fileName )
{

   NALU_HYPRE_Int i, ierr;
   hypre_TempMultiVector* x;
   char fullName[128];

   x = (hypre_TempMultiVector*)x_;
   hypre_assert( x != NULL );
   if ( x->interpreter->PrintVector == NULL )
   {
      return 1;
   }

   ierr = 0;
   for ( i = 0; i < x->numVectors; i++ )
   {
      hypre_sprintf( fullName, "%s.%d", fileName, i );
      ierr = ierr ||
             (x->interpreter->PrintVector)( x->vector[i], fullName );
   }
   return ierr;
}

void*
hypre_TempMultiVectorRead( MPI_Comm comm, void* ii_, const char* fileName )
{

   NALU_HYPRE_Int i, n, id;
   FILE* fp;
   char fullName[128];
   hypre_TempMultiVector* x;
   NALU_HYPRE_InterfaceInterpreter* ii = (NALU_HYPRE_InterfaceInterpreter*)ii_;

   if ( ii->ReadVector == NULL )
   {
      return NULL;
   }

   hypre_MPI_Comm_rank( comm, &id );

   n = 0;
   do
   {
      hypre_sprintf( fullName, "%s.%d.%d", fileName, n, id );
      if ( (fp = fopen(fullName, "r")) )
      {
         n++;
         fclose( fp );
      }
   }
   while ( fp );

   x = hypre_TAlloc(hypre_TempMultiVector, 1, NALU_HYPRE_MEMORY_HOST);
   hypre_assert( x != NULL );

   x->interpreter = ii;

   x->numVectors = n;

   x->vector = hypre_CTAlloc(void*,  n, NALU_HYPRE_MEMORY_HOST);
   hypre_assert( x->vector != NULL );

   x->ownsVectors = 1;

   for ( i = 0; i < n; i++ )
   {
      hypre_sprintf( fullName, "%s.%d", fileName, i );
      x->vector[i] = (ii->ReadVector)( comm, fullName );
   }

   x->mask = NULL;
   x->ownsMask = 0;

   return x;
}

NALU_HYPRE_Int
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

void
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


