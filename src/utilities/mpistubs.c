/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_nalu_hypre_utilities.h"

/******************************************************************************
 * This routine is the same in both the sequential and normal cases
 *
 * The 'comm' argument for MPI_Comm_f2c is MPI_Fint, which is always the size of
 * a Fortran integer and hence usually the size of nalu_hypre_int.
 ****************************************************************************/

nalu_hypre_MPI_Comm
nalu_hypre_MPI_Comm_f2c( nalu_hypre_int comm )
{
#ifdef NALU_HYPRE_HAVE_MPI_COMM_F2C
   return (nalu_hypre_MPI_Comm) MPI_Comm_f2c(comm);
#else
   return (nalu_hypre_MPI_Comm) (size_t)comm;
#endif
}

/******************************************************************************
 * MPI stubs to generate serial codes without mpi
 *****************************************************************************/

#ifdef NALU_HYPRE_SEQUENTIAL

NALU_HYPRE_Int
nalu_hypre_MPI_Init( nalu_hypre_int   *argc,
                char      ***argv )
{
   return (0);
}

NALU_HYPRE_Int
nalu_hypre_MPI_Finalize( void )
{
   return (0);
}

NALU_HYPRE_Int
nalu_hypre_MPI_Abort( nalu_hypre_MPI_Comm comm,
                 NALU_HYPRE_Int      errorcode )
{
   return (0);
}

NALU_HYPRE_Real
nalu_hypre_MPI_Wtime( void )
{
   return (0.0);
}

NALU_HYPRE_Real
nalu_hypre_MPI_Wtick( void )
{
   return (0.0);
}

NALU_HYPRE_Int
nalu_hypre_MPI_Barrier( nalu_hypre_MPI_Comm comm )
{
   return (0);
}

NALU_HYPRE_Int
nalu_hypre_MPI_Comm_create( nalu_hypre_MPI_Comm   comm,
                       nalu_hypre_MPI_Group  group,
                       nalu_hypre_MPI_Comm  *newcomm )
{
   *newcomm = nalu_hypre_MPI_COMM_NULL;
   return (0);
}

NALU_HYPRE_Int
nalu_hypre_MPI_Comm_dup( nalu_hypre_MPI_Comm  comm,
                    nalu_hypre_MPI_Comm *newcomm )
{
   *newcomm = comm;
   return (0);
}

NALU_HYPRE_Int
nalu_hypre_MPI_Comm_size( nalu_hypre_MPI_Comm  comm,
                     NALU_HYPRE_Int      *size )
{
   *size = 1;
   return (0);
}

NALU_HYPRE_Int
nalu_hypre_MPI_Comm_rank( nalu_hypre_MPI_Comm  comm,
                     NALU_HYPRE_Int      *rank )
{
   *rank = 0;
   return (0);
}

NALU_HYPRE_Int
nalu_hypre_MPI_Comm_free( nalu_hypre_MPI_Comm *comm )
{
   return 0;
}

NALU_HYPRE_Int
nalu_hypre_MPI_Comm_group( nalu_hypre_MPI_Comm   comm,
                      nalu_hypre_MPI_Group *group )
{
   return (0);
}

NALU_HYPRE_Int
nalu_hypre_MPI_Comm_split( nalu_hypre_MPI_Comm  comm,
                      NALU_HYPRE_Int       n,
                      NALU_HYPRE_Int       m,
                      nalu_hypre_MPI_Comm *comms )
{
   return (0);
}

NALU_HYPRE_Int
nalu_hypre_MPI_Group_incl( nalu_hypre_MPI_Group  group,
                      NALU_HYPRE_Int        n,
                      NALU_HYPRE_Int       *ranks,
                      nalu_hypre_MPI_Group *newgroup )
{
   return (0);
}

NALU_HYPRE_Int
nalu_hypre_MPI_Group_free( nalu_hypre_MPI_Group *group )
{
   return 0;
}

NALU_HYPRE_Int
nalu_hypre_MPI_Address( void           *location,
                   nalu_hypre_MPI_Aint *address )
{
   return (0);
}

NALU_HYPRE_Int
nalu_hypre_MPI_Get_count( nalu_hypre_MPI_Status   *status,
                     nalu_hypre_MPI_Datatype  datatype,
                     NALU_HYPRE_Int          *count )
{
   return (0);
}

NALU_HYPRE_Int
nalu_hypre_MPI_Alltoall( void               *sendbuf,
                    NALU_HYPRE_Int           sendcount,
                    nalu_hypre_MPI_Datatype  sendtype,
                    void               *recvbuf,
                    NALU_HYPRE_Int           recvcount,
                    nalu_hypre_MPI_Datatype  recvtype,
                    nalu_hypre_MPI_Comm      comm )
{
   return (0);
}

NALU_HYPRE_Int
nalu_hypre_MPI_Allgather( void               *sendbuf,
                     NALU_HYPRE_Int           sendcount,
                     nalu_hypre_MPI_Datatype  sendtype,
                     void               *recvbuf,
                     NALU_HYPRE_Int           recvcount,
                     nalu_hypre_MPI_Datatype  recvtype,
                     nalu_hypre_MPI_Comm      comm )
{
   NALU_HYPRE_Int i;

   switch (sendtype)
   {
      case nalu_hypre_MPI_INT:
      {
         NALU_HYPRE_Int *crecvbuf = (NALU_HYPRE_Int *)recvbuf;
         NALU_HYPRE_Int *csendbuf = (NALU_HYPRE_Int *)sendbuf;
         for (i = 0; i < sendcount; i++)
         {
            crecvbuf[i] = csendbuf[i];
         }
      }
      break;

      case nalu_hypre_MPI_LONG_LONG_INT:
      {
         NALU_HYPRE_BigInt *crecvbuf = (NALU_HYPRE_BigInt *)recvbuf;
         NALU_HYPRE_BigInt *csendbuf = (NALU_HYPRE_BigInt *)sendbuf;
         for (i = 0; i < sendcount; i++)
         {
            crecvbuf[i] = csendbuf[i];
         }
      }
      break;

      case nalu_hypre_MPI_FLOAT:
      {
         float *crecvbuf = (float *)recvbuf;
         float *csendbuf = (float *)sendbuf;
         for (i = 0; i < sendcount; i++)
         {
            crecvbuf[i] = csendbuf[i];
         }
      }
      break;

      case nalu_hypre_MPI_DOUBLE:
      {
         double *crecvbuf = (double *)recvbuf;
         double *csendbuf = (double *)sendbuf;
         for (i = 0; i < sendcount; i++)
         {
            crecvbuf[i] = csendbuf[i];
         }
      }
      break;

      case nalu_hypre_MPI_LONG_DOUBLE:
      {
         long double *crecvbuf = (long double *)recvbuf;
         long double *csendbuf = (long double *)sendbuf;
         for (i = 0; i < sendcount; i++)
         {
            crecvbuf[i] = csendbuf[i];
         }
      }
      break;

      case nalu_hypre_MPI_CHAR:
      {
         char *crecvbuf = (char *)recvbuf;
         char *csendbuf = (char *)sendbuf;
         for (i = 0; i < sendcount; i++)
         {
            crecvbuf[i] = csendbuf[i];
         }
      }
      break;

      case nalu_hypre_MPI_LONG:
      {
         nalu_hypre_longint *crecvbuf = (nalu_hypre_longint *)recvbuf;
         nalu_hypre_longint *csendbuf = (nalu_hypre_longint *)sendbuf;
         for (i = 0; i < sendcount; i++)
         {
            crecvbuf[i] = csendbuf[i];
         }
      }
      break;

      case nalu_hypre_MPI_BYTE:
      {
         nalu_hypre_TMemcpy(recvbuf, sendbuf, char, sendcount, NALU_HYPRE_MEMORY_HOST, NALU_HYPRE_MEMORY_HOST);
      }
      break;

      case nalu_hypre_MPI_REAL:
      {
         NALU_HYPRE_Real *crecvbuf = (NALU_HYPRE_Real *)recvbuf;
         NALU_HYPRE_Real *csendbuf = (NALU_HYPRE_Real *)sendbuf;
         for (i = 0; i < sendcount; i++)
         {
            crecvbuf[i] = csendbuf[i];
         }
      }
      break;

      case nalu_hypre_MPI_COMPLEX:
      {
         NALU_HYPRE_Complex *crecvbuf = (NALU_HYPRE_Complex *)recvbuf;
         NALU_HYPRE_Complex *csendbuf = (NALU_HYPRE_Complex *)sendbuf;
         for (i = 0; i < sendcount; i++)
         {
            crecvbuf[i] = csendbuf[i];
         }
      }
      break;
   }

   return (0);
}

NALU_HYPRE_Int
nalu_hypre_MPI_Allgatherv( void               *sendbuf,
                      NALU_HYPRE_Int           sendcount,
                      nalu_hypre_MPI_Datatype  sendtype,
                      void               *recvbuf,
                      NALU_HYPRE_Int          *recvcounts,
                      NALU_HYPRE_Int          *displs,
                      nalu_hypre_MPI_Datatype  recvtype,
                      nalu_hypre_MPI_Comm      comm )
{
   return ( nalu_hypre_MPI_Allgather(sendbuf, sendcount, sendtype,
                                recvbuf, *recvcounts, recvtype, comm) );
}

NALU_HYPRE_Int
nalu_hypre_MPI_Gather( void               *sendbuf,
                  NALU_HYPRE_Int           sendcount,
                  nalu_hypre_MPI_Datatype  sendtype,
                  void               *recvbuf,
                  NALU_HYPRE_Int           recvcount,
                  nalu_hypre_MPI_Datatype  recvtype,
                  NALU_HYPRE_Int           root,
                  nalu_hypre_MPI_Comm      comm )
{
   return ( nalu_hypre_MPI_Allgather(sendbuf, sendcount, sendtype,
                                recvbuf, recvcount, recvtype, comm) );
}

NALU_HYPRE_Int
nalu_hypre_MPI_Gatherv( void              *sendbuf,
                   NALU_HYPRE_Int           sendcount,
                   nalu_hypre_MPI_Datatype  sendtype,
                   void               *recvbuf,
                   NALU_HYPRE_Int          *recvcounts,
                   NALU_HYPRE_Int          *displs,
                   nalu_hypre_MPI_Datatype  recvtype,
                   NALU_HYPRE_Int           root,
                   nalu_hypre_MPI_Comm      comm )
{
   return ( nalu_hypre_MPI_Allgather(sendbuf, sendcount, sendtype,
                                recvbuf, *recvcounts, recvtype, comm) );
}

NALU_HYPRE_Int
nalu_hypre_MPI_Scatter( void               *sendbuf,
                   NALU_HYPRE_Int           sendcount,
                   nalu_hypre_MPI_Datatype  sendtype,
                   void               *recvbuf,
                   NALU_HYPRE_Int           recvcount,
                   nalu_hypre_MPI_Datatype  recvtype,
                   NALU_HYPRE_Int           root,
                   nalu_hypre_MPI_Comm      comm )
{
   return ( nalu_hypre_MPI_Allgather(sendbuf, sendcount, sendtype,
                                recvbuf, recvcount, recvtype, comm) );
}

NALU_HYPRE_Int
nalu_hypre_MPI_Scatterv( void               *sendbuf,
                    NALU_HYPRE_Int           *sendcounts,
                    NALU_HYPRE_Int           *displs,
                    nalu_hypre_MPI_Datatype   sendtype,
                    void                *recvbuf,
                    NALU_HYPRE_Int            recvcount,
                    nalu_hypre_MPI_Datatype   recvtype,
                    NALU_HYPRE_Int            root,
                    nalu_hypre_MPI_Comm       comm )
{
   return ( nalu_hypre_MPI_Allgather(sendbuf, *sendcounts, sendtype,
                                recvbuf, recvcount, recvtype, comm) );
}

NALU_HYPRE_Int
nalu_hypre_MPI_Bcast( void               *buffer,
                 NALU_HYPRE_Int           count,
                 nalu_hypre_MPI_Datatype  datatype,
                 NALU_HYPRE_Int           root,
                 nalu_hypre_MPI_Comm      comm )
{
   return (0);
}

NALU_HYPRE_Int
nalu_hypre_MPI_Send( void               *buf,
                NALU_HYPRE_Int           count,
                nalu_hypre_MPI_Datatype  datatype,
                NALU_HYPRE_Int           dest,
                NALU_HYPRE_Int           tag,
                nalu_hypre_MPI_Comm      comm )
{
   return (0);
}

NALU_HYPRE_Int
nalu_hypre_MPI_Recv( void               *buf,
                NALU_HYPRE_Int           count,
                nalu_hypre_MPI_Datatype  datatype,
                NALU_HYPRE_Int           source,
                NALU_HYPRE_Int           tag,
                nalu_hypre_MPI_Comm      comm,
                nalu_hypre_MPI_Status   *status )
{
   return (0);
}

NALU_HYPRE_Int
nalu_hypre_MPI_Isend( void               *buf,
                 NALU_HYPRE_Int           count,
                 nalu_hypre_MPI_Datatype  datatype,
                 NALU_HYPRE_Int           dest,
                 NALU_HYPRE_Int           tag,
                 nalu_hypre_MPI_Comm      comm,
                 nalu_hypre_MPI_Request  *request )
{
   return (0);
}

NALU_HYPRE_Int
nalu_hypre_MPI_Irecv( void               *buf,
                 NALU_HYPRE_Int           count,
                 nalu_hypre_MPI_Datatype  datatype,
                 NALU_HYPRE_Int           source,
                 NALU_HYPRE_Int           tag,
                 nalu_hypre_MPI_Comm      comm,
                 nalu_hypre_MPI_Request  *request )
{
   return (0);
}

NALU_HYPRE_Int
nalu_hypre_MPI_Send_init( void               *buf,
                     NALU_HYPRE_Int           count,
                     nalu_hypre_MPI_Datatype  datatype,
                     NALU_HYPRE_Int           dest,
                     NALU_HYPRE_Int           tag,
                     nalu_hypre_MPI_Comm      comm,
                     nalu_hypre_MPI_Request  *request )
{
   return 0;
}

NALU_HYPRE_Int
nalu_hypre_MPI_Recv_init( void               *buf,
                     NALU_HYPRE_Int           count,
                     nalu_hypre_MPI_Datatype  datatype,
                     NALU_HYPRE_Int           dest,
                     NALU_HYPRE_Int           tag,
                     nalu_hypre_MPI_Comm      comm,
                     nalu_hypre_MPI_Request  *request )
{
   return 0;
}

NALU_HYPRE_Int
nalu_hypre_MPI_Irsend( void               *buf,
                  NALU_HYPRE_Int           count,
                  nalu_hypre_MPI_Datatype  datatype,
                  NALU_HYPRE_Int           dest,
                  NALU_HYPRE_Int           tag,
                  nalu_hypre_MPI_Comm      comm,
                  nalu_hypre_MPI_Request  *request )
{
   return 0;
}

NALU_HYPRE_Int
nalu_hypre_MPI_Startall( NALU_HYPRE_Int          count,
                    nalu_hypre_MPI_Request *array_of_requests )
{
   return 0;
}

NALU_HYPRE_Int
nalu_hypre_MPI_Probe( NALU_HYPRE_Int         source,
                 NALU_HYPRE_Int         tag,
                 nalu_hypre_MPI_Comm    comm,
                 nalu_hypre_MPI_Status *status )
{
   return 0;
}

NALU_HYPRE_Int
nalu_hypre_MPI_Iprobe( NALU_HYPRE_Int         source,
                  NALU_HYPRE_Int         tag,
                  nalu_hypre_MPI_Comm    comm,
                  NALU_HYPRE_Int        *flag,
                  nalu_hypre_MPI_Status *status )
{
   return 0;
}

NALU_HYPRE_Int
nalu_hypre_MPI_Test( nalu_hypre_MPI_Request *request,
                NALU_HYPRE_Int         *flag,
                nalu_hypre_MPI_Status  *status )
{
   *flag = 1;
   return (0);
}

NALU_HYPRE_Int
nalu_hypre_MPI_Testall( NALU_HYPRE_Int          count,
                   nalu_hypre_MPI_Request *array_of_requests,
                   NALU_HYPRE_Int         *flag,
                   nalu_hypre_MPI_Status  *array_of_statuses )
{
   *flag = 1;
   return (0);
}

NALU_HYPRE_Int
nalu_hypre_MPI_Wait( nalu_hypre_MPI_Request *request,
                nalu_hypre_MPI_Status  *status )
{
   return (0);
}

NALU_HYPRE_Int
nalu_hypre_MPI_Waitall( NALU_HYPRE_Int          count,
                   nalu_hypre_MPI_Request *array_of_requests,
                   nalu_hypre_MPI_Status  *array_of_statuses )
{
   return (0);
}

NALU_HYPRE_Int
nalu_hypre_MPI_Waitany( NALU_HYPRE_Int          count,
                   nalu_hypre_MPI_Request *array_of_requests,
                   NALU_HYPRE_Int         *index,
                   nalu_hypre_MPI_Status  *status )
{
   return (0);
}

NALU_HYPRE_Int
nalu_hypre_MPI_Allreduce( void              *sendbuf,
                     void              *recvbuf,
                     NALU_HYPRE_Int          count,
                     nalu_hypre_MPI_Datatype datatype,
                     nalu_hypre_MPI_Op       op,
                     nalu_hypre_MPI_Comm     comm )
{
   NALU_HYPRE_Int i;

   switch (datatype)
   {
      case nalu_hypre_MPI_INT:
      {
         NALU_HYPRE_Int *crecvbuf = (NALU_HYPRE_Int *)recvbuf;
         NALU_HYPRE_Int *csendbuf = (NALU_HYPRE_Int *)sendbuf;
         for (i = 0; i < count; i++)
         {
            crecvbuf[i] = csendbuf[i];
         }
      }
      break;

      case nalu_hypre_MPI_LONG_LONG_INT:
      {
         NALU_HYPRE_BigInt *crecvbuf = (NALU_HYPRE_BigInt *)recvbuf;
         NALU_HYPRE_BigInt *csendbuf = (NALU_HYPRE_BigInt *)sendbuf;
         for (i = 0; i < count; i++)
         {
            crecvbuf[i] = csendbuf[i];
         }
      }
      break;

      case nalu_hypre_MPI_FLOAT:
      {
         float *crecvbuf = (float *)recvbuf;
         float *csendbuf = (float *)sendbuf;
         for (i = 0; i < count; i++)
         {
            crecvbuf[i] = csendbuf[i];
         }
      }
      break;

      case nalu_hypre_MPI_DOUBLE:
      {
         double *crecvbuf = (double *)recvbuf;
         double *csendbuf = (double *)sendbuf;
         for (i = 0; i < count; i++)
         {
            crecvbuf[i] = csendbuf[i];
         }
      }
      break;

      case nalu_hypre_MPI_LONG_DOUBLE:
      {
         long double *crecvbuf = (long double *)recvbuf;
         long double *csendbuf = (long double *)sendbuf;
         for (i = 0; i < count; i++)
         {
            crecvbuf[i] = csendbuf[i];
         }
      }
      break;

      case nalu_hypre_MPI_CHAR:
      {
         char *crecvbuf = (char *)recvbuf;
         char *csendbuf = (char *)sendbuf;
         for (i = 0; i < count; i++)
         {
            crecvbuf[i] = csendbuf[i];
         }
      }
      break;

      case nalu_hypre_MPI_LONG:
      {
         nalu_hypre_longint *crecvbuf = (nalu_hypre_longint *)recvbuf;
         nalu_hypre_longint *csendbuf = (nalu_hypre_longint *)sendbuf;
         for (i = 0; i < count; i++)
         {
            crecvbuf[i] = csendbuf[i];
         }
      }
      break;

      case nalu_hypre_MPI_BYTE:
      {
         nalu_hypre_TMemcpy(recvbuf, sendbuf, char, count, NALU_HYPRE_MEMORY_HOST, NALU_HYPRE_MEMORY_HOST);
      }
      break;

      case nalu_hypre_MPI_REAL:
      {
         NALU_HYPRE_Real *crecvbuf = (NALU_HYPRE_Real *)recvbuf;
         NALU_HYPRE_Real *csendbuf = (NALU_HYPRE_Real *)sendbuf;
         for (i = 0; i < count; i++)
         {
            crecvbuf[i] = csendbuf[i];
         }
      }
      break;

      case nalu_hypre_MPI_COMPLEX:
      {
         NALU_HYPRE_Complex *crecvbuf = (NALU_HYPRE_Complex *)recvbuf;
         NALU_HYPRE_Complex *csendbuf = (NALU_HYPRE_Complex *)sendbuf;
         for (i = 0; i < count; i++)
         {
            crecvbuf[i] = csendbuf[i];
         }
      }
      break;
   }

   return 0;
}

NALU_HYPRE_Int
nalu_hypre_MPI_Reduce( void               *sendbuf,
                  void               *recvbuf,
                  NALU_HYPRE_Int           count,
                  nalu_hypre_MPI_Datatype  datatype,
                  nalu_hypre_MPI_Op        op,
                  NALU_HYPRE_Int           root,
                  nalu_hypre_MPI_Comm      comm )
{
   nalu_hypre_MPI_Allreduce(sendbuf, recvbuf, count, datatype, op, comm);
   return 0;
}

NALU_HYPRE_Int
nalu_hypre_MPI_Scan( void               *sendbuf,
                void               *recvbuf,
                NALU_HYPRE_Int           count,
                nalu_hypre_MPI_Datatype  datatype,
                nalu_hypre_MPI_Op        op,
                nalu_hypre_MPI_Comm      comm )
{
   nalu_hypre_MPI_Allreduce(sendbuf, recvbuf, count, datatype, op, comm);
   return 0;
}

NALU_HYPRE_Int
nalu_hypre_MPI_Request_free( nalu_hypre_MPI_Request *request )
{
   return 0;
}

NALU_HYPRE_Int
nalu_hypre_MPI_Type_contiguous( NALU_HYPRE_Int           count,
                           nalu_hypre_MPI_Datatype  oldtype,
                           nalu_hypre_MPI_Datatype *newtype )
{
   return (0);
}

NALU_HYPRE_Int
nalu_hypre_MPI_Type_vector( NALU_HYPRE_Int           count,
                       NALU_HYPRE_Int           blocklength,
                       NALU_HYPRE_Int           stride,
                       nalu_hypre_MPI_Datatype  oldtype,
                       nalu_hypre_MPI_Datatype *newtype )
{
   return (0);
}

NALU_HYPRE_Int
nalu_hypre_MPI_Type_hvector( NALU_HYPRE_Int           count,
                        NALU_HYPRE_Int           blocklength,
                        nalu_hypre_MPI_Aint      stride,
                        nalu_hypre_MPI_Datatype  oldtype,
                        nalu_hypre_MPI_Datatype *newtype )
{
   return (0);
}

NALU_HYPRE_Int
nalu_hypre_MPI_Type_struct( NALU_HYPRE_Int           count,
                       NALU_HYPRE_Int          *array_of_blocklengths,
                       nalu_hypre_MPI_Aint     *array_of_displacements,
                       nalu_hypre_MPI_Datatype *array_of_types,
                       nalu_hypre_MPI_Datatype *newtype )
{
   return (0);
}

NALU_HYPRE_Int
nalu_hypre_MPI_Type_commit( nalu_hypre_MPI_Datatype *datatype )
{
   return (0);
}

NALU_HYPRE_Int
nalu_hypre_MPI_Type_free( nalu_hypre_MPI_Datatype *datatype )
{
   return (0);
}

NALU_HYPRE_Int
nalu_hypre_MPI_Op_create( nalu_hypre_MPI_User_function *function, nalu_hypre_int commute, nalu_hypre_MPI_Op *op )
{
   return (0);
}

NALU_HYPRE_Int
nalu_hypre_MPI_Op_free( nalu_hypre_MPI_Op *op )
{
   return (0);
}

#if defined(NALU_HYPRE_USING_GPU)
NALU_HYPRE_Int nalu_hypre_MPI_Comm_split_type( nalu_hypre_MPI_Comm comm, NALU_HYPRE_Int split_type, NALU_HYPRE_Int key,
                                     nalu_hypre_MPI_Info info, nalu_hypre_MPI_Comm *newcomm )
{
   return (0);
}

NALU_HYPRE_Int nalu_hypre_MPI_Info_create( nalu_hypre_MPI_Info *info )
{
   return (0);
}

NALU_HYPRE_Int nalu_hypre_MPI_Info_free( nalu_hypre_MPI_Info *info )
{
   return (0);
}
#endif

/******************************************************************************
 * MPI stubs to do casting of NALU_HYPRE_Int and nalu_hypre_int correctly
 *****************************************************************************/

#else

NALU_HYPRE_Int
nalu_hypre_MPI_Init( nalu_hypre_int   *argc,
                char      ***argv )
{
   return (NALU_HYPRE_Int) MPI_Init(argc, argv);
}

NALU_HYPRE_Int
nalu_hypre_MPI_Finalize( void )
{
   return (NALU_HYPRE_Int) MPI_Finalize();
}

NALU_HYPRE_Int
nalu_hypre_MPI_Abort( nalu_hypre_MPI_Comm comm,
                 NALU_HYPRE_Int      errorcode )
{
   return (NALU_HYPRE_Int) MPI_Abort(comm, (nalu_hypre_int)errorcode);
}

NALU_HYPRE_Real
nalu_hypre_MPI_Wtime( void )
{
   return (NALU_HYPRE_Real)MPI_Wtime();
}

NALU_HYPRE_Real
nalu_hypre_MPI_Wtick( void )
{
   return (NALU_HYPRE_Real)MPI_Wtick();
}

NALU_HYPRE_Int
nalu_hypre_MPI_Barrier( nalu_hypre_MPI_Comm comm )
{
   return (NALU_HYPRE_Int) MPI_Barrier(comm);
}

NALU_HYPRE_Int
nalu_hypre_MPI_Comm_create( nalu_hypre_MPI_Comm   comm,
                       nalu_hypre_MPI_Group  group,
                       nalu_hypre_MPI_Comm  *newcomm )
{
   return (NALU_HYPRE_Int) MPI_Comm_create(comm, group, newcomm);
}

NALU_HYPRE_Int
nalu_hypre_MPI_Comm_dup( nalu_hypre_MPI_Comm  comm,
                    nalu_hypre_MPI_Comm *newcomm )
{
   return (NALU_HYPRE_Int) MPI_Comm_dup(comm, newcomm);
}

NALU_HYPRE_Int
nalu_hypre_MPI_Comm_size( nalu_hypre_MPI_Comm  comm,
                     NALU_HYPRE_Int      *size )
{
   nalu_hypre_int mpi_size;
   NALU_HYPRE_Int ierr;
   ierr = (NALU_HYPRE_Int) MPI_Comm_size(comm, &mpi_size);
   *size = (NALU_HYPRE_Int) mpi_size;
   return ierr;
}

NALU_HYPRE_Int
nalu_hypre_MPI_Comm_rank( nalu_hypre_MPI_Comm  comm,
                     NALU_HYPRE_Int      *rank )
{
   nalu_hypre_int mpi_rank;
   NALU_HYPRE_Int ierr;
   ierr = (NALU_HYPRE_Int) MPI_Comm_rank(comm, &mpi_rank);
   *rank = (NALU_HYPRE_Int) mpi_rank;
   return ierr;
}

NALU_HYPRE_Int
nalu_hypre_MPI_Comm_free( nalu_hypre_MPI_Comm *comm )
{
   return (NALU_HYPRE_Int) MPI_Comm_free(comm);
}

NALU_HYPRE_Int
nalu_hypre_MPI_Comm_group( nalu_hypre_MPI_Comm   comm,
                      nalu_hypre_MPI_Group *group )
{
   return (NALU_HYPRE_Int) MPI_Comm_group(comm, group);
}

NALU_HYPRE_Int
nalu_hypre_MPI_Comm_split( nalu_hypre_MPI_Comm  comm,
                      NALU_HYPRE_Int       n,
                      NALU_HYPRE_Int       m,
                      nalu_hypre_MPI_Comm *comms )
{
   return (NALU_HYPRE_Int) MPI_Comm_split(comm, (nalu_hypre_int)n, (nalu_hypre_int)m, comms);
}

NALU_HYPRE_Int
nalu_hypre_MPI_Group_incl( nalu_hypre_MPI_Group  group,
                      NALU_HYPRE_Int        n,
                      NALU_HYPRE_Int       *ranks,
                      nalu_hypre_MPI_Group *newgroup )
{
   nalu_hypre_int *mpi_ranks;
   NALU_HYPRE_Int  i;
   NALU_HYPRE_Int  ierr;

   mpi_ranks = nalu_hypre_TAlloc(nalu_hypre_int,  n, NALU_HYPRE_MEMORY_HOST);
   for (i = 0; i < n; i++)
   {
      mpi_ranks[i] = (nalu_hypre_int) ranks[i];
   }
   ierr = (NALU_HYPRE_Int) MPI_Group_incl(group, (nalu_hypre_int)n, mpi_ranks, newgroup);
   nalu_hypre_TFree(mpi_ranks, NALU_HYPRE_MEMORY_HOST);

   return ierr;
}

NALU_HYPRE_Int
nalu_hypre_MPI_Group_free( nalu_hypre_MPI_Group *group )
{
   return (NALU_HYPRE_Int) MPI_Group_free(group);
}

NALU_HYPRE_Int
nalu_hypre_MPI_Address( void           *location,
                   nalu_hypre_MPI_Aint *address )
{
#if MPI_VERSION > 1
   return (NALU_HYPRE_Int) MPI_Get_address(location, address);
#else
   return (NALU_HYPRE_Int) MPI_Address(location, address);
#endif
}

NALU_HYPRE_Int
nalu_hypre_MPI_Get_count( nalu_hypre_MPI_Status   *status,
                     nalu_hypre_MPI_Datatype  datatype,
                     NALU_HYPRE_Int          *count )
{
   nalu_hypre_int mpi_count;
   NALU_HYPRE_Int ierr;
   ierr = (NALU_HYPRE_Int) MPI_Get_count(status, datatype, &mpi_count);
   *count = (NALU_HYPRE_Int) mpi_count;
   return ierr;
}

NALU_HYPRE_Int
nalu_hypre_MPI_Alltoall( void               *sendbuf,
                    NALU_HYPRE_Int           sendcount,
                    nalu_hypre_MPI_Datatype  sendtype,
                    void               *recvbuf,
                    NALU_HYPRE_Int           recvcount,
                    nalu_hypre_MPI_Datatype  recvtype,
                    nalu_hypre_MPI_Comm      comm )
{
   return (NALU_HYPRE_Int) MPI_Alltoall(sendbuf, (nalu_hypre_int)sendcount, sendtype,
                                   recvbuf, (nalu_hypre_int)recvcount, recvtype, comm);
}

NALU_HYPRE_Int
nalu_hypre_MPI_Allgather( void               *sendbuf,
                     NALU_HYPRE_Int           sendcount,
                     nalu_hypre_MPI_Datatype  sendtype,
                     void               *recvbuf,
                     NALU_HYPRE_Int           recvcount,
                     nalu_hypre_MPI_Datatype  recvtype,
                     nalu_hypre_MPI_Comm      comm )
{
   return (NALU_HYPRE_Int) MPI_Allgather(sendbuf, (nalu_hypre_int)sendcount, sendtype,
                                    recvbuf, (nalu_hypre_int)recvcount, recvtype, comm);
}

NALU_HYPRE_Int
nalu_hypre_MPI_Allgatherv( void               *sendbuf,
                      NALU_HYPRE_Int           sendcount,
                      nalu_hypre_MPI_Datatype  sendtype,
                      void               *recvbuf,
                      NALU_HYPRE_Int          *recvcounts,
                      NALU_HYPRE_Int          *displs,
                      nalu_hypre_MPI_Datatype  recvtype,
                      nalu_hypre_MPI_Comm      comm )
{
   nalu_hypre_int *mpi_recvcounts, *mpi_displs, csize;
   NALU_HYPRE_Int  i;
   NALU_HYPRE_Int  ierr;

   MPI_Comm_size(comm, &csize);
   mpi_recvcounts = nalu_hypre_TAlloc(nalu_hypre_int, csize, NALU_HYPRE_MEMORY_HOST);
   mpi_displs = nalu_hypre_TAlloc(nalu_hypre_int, csize, NALU_HYPRE_MEMORY_HOST);
   for (i = 0; i < csize; i++)
   {
      mpi_recvcounts[i] = (nalu_hypre_int) recvcounts[i];
      mpi_displs[i] = (nalu_hypre_int) displs[i];
   }
   ierr = (NALU_HYPRE_Int) MPI_Allgatherv(sendbuf, (nalu_hypre_int)sendcount, sendtype,
                                     recvbuf, mpi_recvcounts, mpi_displs,
                                     recvtype, comm);
   nalu_hypre_TFree(mpi_recvcounts, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(mpi_displs, NALU_HYPRE_MEMORY_HOST);

   return ierr;
}

NALU_HYPRE_Int
nalu_hypre_MPI_Gather( void               *sendbuf,
                  NALU_HYPRE_Int           sendcount,
                  nalu_hypre_MPI_Datatype  sendtype,
                  void               *recvbuf,
                  NALU_HYPRE_Int           recvcount,
                  nalu_hypre_MPI_Datatype  recvtype,
                  NALU_HYPRE_Int           root,
                  nalu_hypre_MPI_Comm      comm )
{
   return (NALU_HYPRE_Int) MPI_Gather(sendbuf, (nalu_hypre_int) sendcount, sendtype,
                                 recvbuf, (nalu_hypre_int) recvcount, recvtype,
                                 (nalu_hypre_int)root, comm);
}

NALU_HYPRE_Int
nalu_hypre_MPI_Gatherv(void               *sendbuf,
                  NALU_HYPRE_Int           sendcount,
                  nalu_hypre_MPI_Datatype  sendtype,
                  void               *recvbuf,
                  NALU_HYPRE_Int          *recvcounts,
                  NALU_HYPRE_Int          *displs,
                  nalu_hypre_MPI_Datatype  recvtype,
                  NALU_HYPRE_Int           root,
                  nalu_hypre_MPI_Comm      comm )
{
   nalu_hypre_int *mpi_recvcounts = NULL;
   nalu_hypre_int *mpi_displs = NULL;
   nalu_hypre_int csize, croot;
   NALU_HYPRE_Int  i;
   NALU_HYPRE_Int  ierr;

   MPI_Comm_size(comm, &csize);
   MPI_Comm_rank(comm, &croot);
   if (croot == (nalu_hypre_int) root)
   {
      mpi_recvcounts = nalu_hypre_TAlloc(nalu_hypre_int,  csize, NALU_HYPRE_MEMORY_HOST);
      mpi_displs = nalu_hypre_TAlloc(nalu_hypre_int,  csize, NALU_HYPRE_MEMORY_HOST);
      for (i = 0; i < csize; i++)
      {
         mpi_recvcounts[i] = (nalu_hypre_int) recvcounts[i];
         mpi_displs[i] = (nalu_hypre_int) displs[i];
      }
   }
   ierr = (NALU_HYPRE_Int) MPI_Gatherv(sendbuf, (nalu_hypre_int)sendcount, sendtype,
                                  recvbuf, mpi_recvcounts, mpi_displs,
                                  recvtype, (nalu_hypre_int) root, comm);
   nalu_hypre_TFree(mpi_recvcounts, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(mpi_displs, NALU_HYPRE_MEMORY_HOST);

   return ierr;
}

NALU_HYPRE_Int
nalu_hypre_MPI_Scatter( void               *sendbuf,
                   NALU_HYPRE_Int           sendcount,
                   nalu_hypre_MPI_Datatype  sendtype,
                   void               *recvbuf,
                   NALU_HYPRE_Int           recvcount,
                   nalu_hypre_MPI_Datatype  recvtype,
                   NALU_HYPRE_Int           root,
                   nalu_hypre_MPI_Comm      comm )
{
   return (NALU_HYPRE_Int) MPI_Scatter(sendbuf, (nalu_hypre_int)sendcount, sendtype,
                                  recvbuf, (nalu_hypre_int)recvcount, recvtype,
                                  (nalu_hypre_int)root, comm);
}

NALU_HYPRE_Int
nalu_hypre_MPI_Scatterv(void               *sendbuf,
                   NALU_HYPRE_Int          *sendcounts,
                   NALU_HYPRE_Int          *displs,
                   nalu_hypre_MPI_Datatype  sendtype,
                   void               *recvbuf,
                   NALU_HYPRE_Int           recvcount,
                   nalu_hypre_MPI_Datatype  recvtype,
                   NALU_HYPRE_Int           root,
                   nalu_hypre_MPI_Comm      comm )
{
   nalu_hypre_int *mpi_sendcounts = NULL;
   nalu_hypre_int *mpi_displs = NULL;
   nalu_hypre_int csize, croot;
   NALU_HYPRE_Int  i;
   NALU_HYPRE_Int  ierr;

   MPI_Comm_size(comm, &csize);
   MPI_Comm_rank(comm, &croot);
   if (croot == (nalu_hypre_int) root)
   {
      mpi_sendcounts = nalu_hypre_TAlloc(nalu_hypre_int,  csize, NALU_HYPRE_MEMORY_HOST);
      mpi_displs = nalu_hypre_TAlloc(nalu_hypre_int,  csize, NALU_HYPRE_MEMORY_HOST);
      for (i = 0; i < csize; i++)
      {
         mpi_sendcounts[i] = (nalu_hypre_int) sendcounts[i];
         mpi_displs[i] = (nalu_hypre_int) displs[i];
      }
   }
   ierr = (NALU_HYPRE_Int) MPI_Scatterv(sendbuf, mpi_sendcounts, mpi_displs, sendtype,
                                   recvbuf, (nalu_hypre_int) recvcount,
                                   recvtype, (nalu_hypre_int) root, comm);
   nalu_hypre_TFree(mpi_sendcounts, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(mpi_displs, NALU_HYPRE_MEMORY_HOST);

   return ierr;
}

NALU_HYPRE_Int
nalu_hypre_MPI_Bcast( void               *buffer,
                 NALU_HYPRE_Int           count,
                 nalu_hypre_MPI_Datatype  datatype,
                 NALU_HYPRE_Int           root,
                 nalu_hypre_MPI_Comm      comm )
{
   return (NALU_HYPRE_Int) MPI_Bcast(buffer, (nalu_hypre_int)count, datatype,
                                (nalu_hypre_int)root, comm);
}

NALU_HYPRE_Int
nalu_hypre_MPI_Send( void               *buf,
                NALU_HYPRE_Int           count,
                nalu_hypre_MPI_Datatype  datatype,
                NALU_HYPRE_Int           dest,
                NALU_HYPRE_Int           tag,
                nalu_hypre_MPI_Comm      comm )
{
   return (NALU_HYPRE_Int) MPI_Send(buf, (nalu_hypre_int)count, datatype,
                               (nalu_hypre_int)dest, (nalu_hypre_int)tag, comm);
}

NALU_HYPRE_Int
nalu_hypre_MPI_Recv( void               *buf,
                NALU_HYPRE_Int           count,
                nalu_hypre_MPI_Datatype  datatype,
                NALU_HYPRE_Int           source,
                NALU_HYPRE_Int           tag,
                nalu_hypre_MPI_Comm      comm,
                nalu_hypre_MPI_Status   *status )
{
   return (NALU_HYPRE_Int) MPI_Recv(buf, (nalu_hypre_int)count, datatype,
                               (nalu_hypre_int)source, (nalu_hypre_int)tag, comm, status);
}

NALU_HYPRE_Int
nalu_hypre_MPI_Isend( void               *buf,
                 NALU_HYPRE_Int           count,
                 nalu_hypre_MPI_Datatype  datatype,
                 NALU_HYPRE_Int           dest,
                 NALU_HYPRE_Int           tag,
                 nalu_hypre_MPI_Comm      comm,
                 nalu_hypre_MPI_Request  *request )
{
   return (NALU_HYPRE_Int) MPI_Isend(buf, (nalu_hypre_int)count, datatype,
                                (nalu_hypre_int)dest, (nalu_hypre_int)tag, comm, request);
}

NALU_HYPRE_Int
nalu_hypre_MPI_Irecv( void               *buf,
                 NALU_HYPRE_Int           count,
                 nalu_hypre_MPI_Datatype  datatype,
                 NALU_HYPRE_Int           source,
                 NALU_HYPRE_Int           tag,
                 nalu_hypre_MPI_Comm      comm,
                 nalu_hypre_MPI_Request  *request )
{
   return (NALU_HYPRE_Int) MPI_Irecv(buf, (nalu_hypre_int)count, datatype,
                                (nalu_hypre_int)source, (nalu_hypre_int)tag, comm, request);
}

NALU_HYPRE_Int
nalu_hypre_MPI_Send_init( void               *buf,
                     NALU_HYPRE_Int           count,
                     nalu_hypre_MPI_Datatype  datatype,
                     NALU_HYPRE_Int           dest,
                     NALU_HYPRE_Int           tag,
                     nalu_hypre_MPI_Comm      comm,
                     nalu_hypre_MPI_Request  *request )
{
   return (NALU_HYPRE_Int) MPI_Send_init(buf, (nalu_hypre_int)count, datatype,
                                    (nalu_hypre_int)dest, (nalu_hypre_int)tag,
                                    comm, request);
}

NALU_HYPRE_Int
nalu_hypre_MPI_Recv_init( void               *buf,
                     NALU_HYPRE_Int           count,
                     nalu_hypre_MPI_Datatype  datatype,
                     NALU_HYPRE_Int           dest,
                     NALU_HYPRE_Int           tag,
                     nalu_hypre_MPI_Comm      comm,
                     nalu_hypre_MPI_Request  *request )
{
   return (NALU_HYPRE_Int) MPI_Recv_init(buf, (nalu_hypre_int)count, datatype,
                                    (nalu_hypre_int)dest, (nalu_hypre_int)tag,
                                    comm, request);
}

NALU_HYPRE_Int
nalu_hypre_MPI_Irsend( void               *buf,
                  NALU_HYPRE_Int           count,
                  nalu_hypre_MPI_Datatype  datatype,
                  NALU_HYPRE_Int           dest,
                  NALU_HYPRE_Int           tag,
                  nalu_hypre_MPI_Comm      comm,
                  nalu_hypre_MPI_Request  *request )
{
   return (NALU_HYPRE_Int) MPI_Irsend(buf, (nalu_hypre_int)count, datatype,
                                 (nalu_hypre_int)dest, (nalu_hypre_int)tag, comm, request);
}

NALU_HYPRE_Int
nalu_hypre_MPI_Startall( NALU_HYPRE_Int          count,
                    nalu_hypre_MPI_Request *array_of_requests )
{
   return (NALU_HYPRE_Int) MPI_Startall((nalu_hypre_int)count, array_of_requests);
}

NALU_HYPRE_Int
nalu_hypre_MPI_Probe( NALU_HYPRE_Int         source,
                 NALU_HYPRE_Int         tag,
                 nalu_hypre_MPI_Comm    comm,
                 nalu_hypre_MPI_Status *status )
{
   return (NALU_HYPRE_Int) MPI_Probe((nalu_hypre_int)source, (nalu_hypre_int)tag, comm, status);
}

NALU_HYPRE_Int
nalu_hypre_MPI_Iprobe( NALU_HYPRE_Int         source,
                  NALU_HYPRE_Int         tag,
                  nalu_hypre_MPI_Comm    comm,
                  NALU_HYPRE_Int        *flag,
                  nalu_hypre_MPI_Status *status )
{
   nalu_hypre_int mpi_flag;
   NALU_HYPRE_Int ierr;
   ierr = (NALU_HYPRE_Int) MPI_Iprobe((nalu_hypre_int)source, (nalu_hypre_int)tag, comm,
                                 &mpi_flag, status);
   *flag = (NALU_HYPRE_Int) mpi_flag;
   return ierr;
}

NALU_HYPRE_Int
nalu_hypre_MPI_Test( nalu_hypre_MPI_Request *request,
                NALU_HYPRE_Int         *flag,
                nalu_hypre_MPI_Status  *status )
{
   nalu_hypre_int mpi_flag;
   NALU_HYPRE_Int ierr;
   ierr = (NALU_HYPRE_Int) MPI_Test(request, &mpi_flag, status);
   *flag = (NALU_HYPRE_Int) mpi_flag;
   return ierr;
}

NALU_HYPRE_Int
nalu_hypre_MPI_Testall( NALU_HYPRE_Int          count,
                   nalu_hypre_MPI_Request *array_of_requests,
                   NALU_HYPRE_Int         *flag,
                   nalu_hypre_MPI_Status  *array_of_statuses )
{
   nalu_hypre_int mpi_flag;
   NALU_HYPRE_Int ierr;
   ierr = (NALU_HYPRE_Int) MPI_Testall((nalu_hypre_int)count, array_of_requests,
                                  &mpi_flag, array_of_statuses);
   *flag = (NALU_HYPRE_Int) mpi_flag;
   return ierr;
}

NALU_HYPRE_Int
nalu_hypre_MPI_Wait( nalu_hypre_MPI_Request *request,
                nalu_hypre_MPI_Status  *status )
{
   return (NALU_HYPRE_Int) MPI_Wait(request, status);
}

NALU_HYPRE_Int
nalu_hypre_MPI_Waitall( NALU_HYPRE_Int          count,
                   nalu_hypre_MPI_Request *array_of_requests,
                   nalu_hypre_MPI_Status  *array_of_statuses )
{
   return (NALU_HYPRE_Int) MPI_Waitall((nalu_hypre_int)count,
                                  array_of_requests, array_of_statuses);
}

NALU_HYPRE_Int
nalu_hypre_MPI_Waitany( NALU_HYPRE_Int          count,
                   nalu_hypre_MPI_Request *array_of_requests,
                   NALU_HYPRE_Int         *index,
                   nalu_hypre_MPI_Status  *status )
{
   nalu_hypre_int mpi_index;
   NALU_HYPRE_Int ierr;
   ierr = (NALU_HYPRE_Int) MPI_Waitany((nalu_hypre_int)count, array_of_requests,
                                  &mpi_index, status);
   *index = (NALU_HYPRE_Int) mpi_index;
   return ierr;
}

NALU_HYPRE_Int
nalu_hypre_MPI_Allreduce( void              *sendbuf,
                     void              *recvbuf,
                     NALU_HYPRE_Int          count,
                     nalu_hypre_MPI_Datatype datatype,
                     nalu_hypre_MPI_Op       op,
                     nalu_hypre_MPI_Comm     comm )
{
   nalu_hypre_GpuProfilingPushRange("MPI_Allreduce");

   NALU_HYPRE_Int result = MPI_Allreduce(sendbuf, recvbuf, (nalu_hypre_int)count,
                                    datatype, op, comm);

   nalu_hypre_GpuProfilingPopRange();

   return result;
}

NALU_HYPRE_Int
nalu_hypre_MPI_Reduce( void               *sendbuf,
                  void               *recvbuf,
                  NALU_HYPRE_Int           count,
                  nalu_hypre_MPI_Datatype  datatype,
                  nalu_hypre_MPI_Op        op,
                  NALU_HYPRE_Int           root,
                  nalu_hypre_MPI_Comm      comm )
{
   return (NALU_HYPRE_Int) MPI_Reduce(sendbuf, recvbuf, (nalu_hypre_int)count,
                                 datatype, op, (nalu_hypre_int)root, comm);
}

NALU_HYPRE_Int
nalu_hypre_MPI_Scan( void               *sendbuf,
                void               *recvbuf,
                NALU_HYPRE_Int           count,
                nalu_hypre_MPI_Datatype  datatype,
                nalu_hypre_MPI_Op        op,
                nalu_hypre_MPI_Comm      comm )
{
   return (NALU_HYPRE_Int) MPI_Scan(sendbuf, recvbuf, (nalu_hypre_int)count,
                               datatype, op, comm);
}

NALU_HYPRE_Int
nalu_hypre_MPI_Request_free( nalu_hypre_MPI_Request *request )
{
   return (NALU_HYPRE_Int) MPI_Request_free(request);
}

NALU_HYPRE_Int
nalu_hypre_MPI_Type_contiguous( NALU_HYPRE_Int           count,
                           nalu_hypre_MPI_Datatype  oldtype,
                           nalu_hypre_MPI_Datatype *newtype )
{
   return (NALU_HYPRE_Int) MPI_Type_contiguous((nalu_hypre_int)count, oldtype, newtype);
}

NALU_HYPRE_Int
nalu_hypre_MPI_Type_vector( NALU_HYPRE_Int           count,
                       NALU_HYPRE_Int           blocklength,
                       NALU_HYPRE_Int           stride,
                       nalu_hypre_MPI_Datatype  oldtype,
                       nalu_hypre_MPI_Datatype *newtype )
{
   return (NALU_HYPRE_Int) MPI_Type_vector((nalu_hypre_int)count, (nalu_hypre_int)blocklength,
                                      (nalu_hypre_int)stride, oldtype, newtype);
}

NALU_HYPRE_Int
nalu_hypre_MPI_Type_hvector( NALU_HYPRE_Int           count,
                        NALU_HYPRE_Int           blocklength,
                        nalu_hypre_MPI_Aint      stride,
                        nalu_hypre_MPI_Datatype  oldtype,
                        nalu_hypre_MPI_Datatype *newtype )
{
#if MPI_VERSION > 1
   return (NALU_HYPRE_Int) MPI_Type_create_hvector((nalu_hypre_int)count, (nalu_hypre_int)blocklength,
                                              stride, oldtype, newtype);
#else
   return (NALU_HYPRE_Int) MPI_Type_hvector((nalu_hypre_int)count, (nalu_hypre_int)blocklength,
                                       stride, oldtype, newtype);
#endif
}

NALU_HYPRE_Int
nalu_hypre_MPI_Type_struct( NALU_HYPRE_Int           count,
                       NALU_HYPRE_Int          *array_of_blocklengths,
                       nalu_hypre_MPI_Aint     *array_of_displacements,
                       nalu_hypre_MPI_Datatype *array_of_types,
                       nalu_hypre_MPI_Datatype *newtype )
{
   nalu_hypre_int *mpi_array_of_blocklengths;
   NALU_HYPRE_Int  i;
   NALU_HYPRE_Int  ierr;

   mpi_array_of_blocklengths = nalu_hypre_TAlloc(nalu_hypre_int,  count, NALU_HYPRE_MEMORY_HOST);
   for (i = 0; i < count; i++)
   {
      mpi_array_of_blocklengths[i] = (nalu_hypre_int) array_of_blocklengths[i];
   }

#if MPI_VERSION > 1
   ierr = (NALU_HYPRE_Int) MPI_Type_create_struct((nalu_hypre_int)count, mpi_array_of_blocklengths,
                                             array_of_displacements, array_of_types,
                                             newtype);
#else
   ierr = (NALU_HYPRE_Int) MPI_Type_struct((nalu_hypre_int)count, mpi_array_of_blocklengths,
                                      array_of_displacements, array_of_types,
                                      newtype);
#endif

   nalu_hypre_TFree(mpi_array_of_blocklengths, NALU_HYPRE_MEMORY_HOST);

   return ierr;
}

NALU_HYPRE_Int
nalu_hypre_MPI_Type_commit( nalu_hypre_MPI_Datatype *datatype )
{
   return (NALU_HYPRE_Int) MPI_Type_commit(datatype);
}

NALU_HYPRE_Int
nalu_hypre_MPI_Type_free( nalu_hypre_MPI_Datatype *datatype )
{
   return (NALU_HYPRE_Int) MPI_Type_free(datatype);
}

NALU_HYPRE_Int
nalu_hypre_MPI_Op_free( nalu_hypre_MPI_Op *op )
{
   return (NALU_HYPRE_Int) MPI_Op_free(op);
}

NALU_HYPRE_Int
nalu_hypre_MPI_Op_create( nalu_hypre_MPI_User_function *function, nalu_hypre_int commute, nalu_hypre_MPI_Op *op )
{
   return (NALU_HYPRE_Int) MPI_Op_create(function, commute, op);
}

#if defined(NALU_HYPRE_USING_GPU) || defined(NALU_HYPRE_USING_DEVICE_OPENMP)
NALU_HYPRE_Int
nalu_hypre_MPI_Comm_split_type( nalu_hypre_MPI_Comm comm, NALU_HYPRE_Int split_type, NALU_HYPRE_Int key,
                           nalu_hypre_MPI_Info info, nalu_hypre_MPI_Comm *newcomm )
{
   return (NALU_HYPRE_Int) MPI_Comm_split_type(comm, split_type, key, info, newcomm );
}

NALU_HYPRE_Int
nalu_hypre_MPI_Info_create( nalu_hypre_MPI_Info *info )
{
   return (NALU_HYPRE_Int) MPI_Info_create(info);
}

NALU_HYPRE_Int
nalu_hypre_MPI_Info_free( nalu_hypre_MPI_Info *info )
{
   return (NALU_HYPRE_Int) MPI_Info_free(info);
}
#endif

#endif
