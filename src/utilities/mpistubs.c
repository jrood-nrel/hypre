/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_utilities.h"

/******************************************************************************
 * This routine is the same in both the sequential and normal cases
 *
 * The 'comm' argument for MPI_Comm_f2c is MPI_Fint, which is always the size of
 * a Fortran integer and hence usually the size of hypre_int.
 ****************************************************************************/

hypre_MPI_Comm
hypre_MPI_Comm_f2c( hypre_int comm )
{
#ifdef NALU_HYPRE_HAVE_MPI_COMM_F2C
   return (hypre_MPI_Comm) MPI_Comm_f2c(comm);
#else
   return (hypre_MPI_Comm) (size_t)comm;
#endif
}

/******************************************************************************
 * MPI stubs to generate serial codes without mpi
 *****************************************************************************/

#ifdef NALU_HYPRE_SEQUENTIAL

NALU_HYPRE_Int
hypre_MPI_Init( hypre_int   *argc,
                char      ***argv )
{
   return (0);
}

NALU_HYPRE_Int
hypre_MPI_Finalize( )
{
   return (0);
}

NALU_HYPRE_Int
hypre_MPI_Abort( hypre_MPI_Comm comm,
                 NALU_HYPRE_Int      errorcode )
{
   return (0);
}

NALU_HYPRE_Real
hypre_MPI_Wtime( )
{
   return (0.0);
}

NALU_HYPRE_Real
hypre_MPI_Wtick( )
{
   return (0.0);
}

NALU_HYPRE_Int
hypre_MPI_Barrier( hypre_MPI_Comm comm )
{
   return (0);
}

NALU_HYPRE_Int
hypre_MPI_Comm_create( hypre_MPI_Comm   comm,
                       hypre_MPI_Group  group,
                       hypre_MPI_Comm  *newcomm )
{
   *newcomm = hypre_MPI_COMM_NULL;
   return (0);
}

NALU_HYPRE_Int
hypre_MPI_Comm_dup( hypre_MPI_Comm  comm,
                    hypre_MPI_Comm *newcomm )
{
   *newcomm = comm;
   return (0);
}

NALU_HYPRE_Int
hypre_MPI_Comm_size( hypre_MPI_Comm  comm,
                     NALU_HYPRE_Int      *size )
{
   *size = 1;
   return (0);
}

NALU_HYPRE_Int
hypre_MPI_Comm_rank( hypre_MPI_Comm  comm,
                     NALU_HYPRE_Int      *rank )
{
   *rank = 0;
   return (0);
}

NALU_HYPRE_Int
hypre_MPI_Comm_free( hypre_MPI_Comm *comm )
{
   return 0;
}

NALU_HYPRE_Int
hypre_MPI_Comm_group( hypre_MPI_Comm   comm,
                      hypre_MPI_Group *group )
{
   return (0);
}

NALU_HYPRE_Int
hypre_MPI_Comm_split( hypre_MPI_Comm  comm,
                      NALU_HYPRE_Int       n,
                      NALU_HYPRE_Int       m,
                      hypre_MPI_Comm *comms )
{
   return (0);
}

NALU_HYPRE_Int
hypre_MPI_Group_incl( hypre_MPI_Group  group,
                      NALU_HYPRE_Int        n,
                      NALU_HYPRE_Int       *ranks,
                      hypre_MPI_Group *newgroup )
{
   return (0);
}

NALU_HYPRE_Int
hypre_MPI_Group_free( hypre_MPI_Group *group )
{
   return 0;
}

NALU_HYPRE_Int
hypre_MPI_Address( void           *location,
                   hypre_MPI_Aint *address )
{
   return (0);
}

NALU_HYPRE_Int
hypre_MPI_Get_count( hypre_MPI_Status   *status,
                     hypre_MPI_Datatype  datatype,
                     NALU_HYPRE_Int          *count )
{
   return (0);
}

NALU_HYPRE_Int
hypre_MPI_Alltoall( void               *sendbuf,
                    NALU_HYPRE_Int           sendcount,
                    hypre_MPI_Datatype  sendtype,
                    void               *recvbuf,
                    NALU_HYPRE_Int           recvcount,
                    hypre_MPI_Datatype  recvtype,
                    hypre_MPI_Comm      comm )
{
   return (0);
}

NALU_HYPRE_Int
hypre_MPI_Allgather( void               *sendbuf,
                     NALU_HYPRE_Int           sendcount,
                     hypre_MPI_Datatype  sendtype,
                     void               *recvbuf,
                     NALU_HYPRE_Int           recvcount,
                     hypre_MPI_Datatype  recvtype,
                     hypre_MPI_Comm      comm )
{
   NALU_HYPRE_Int i;

   switch (sendtype)
   {
      case hypre_MPI_INT:
      {
         NALU_HYPRE_Int *crecvbuf = (NALU_HYPRE_Int *)recvbuf;
         NALU_HYPRE_Int *csendbuf = (NALU_HYPRE_Int *)sendbuf;
         for (i = 0; i < sendcount; i++)
         {
            crecvbuf[i] = csendbuf[i];
         }
      }
      break;

      case hypre_MPI_LONG_LONG_INT:
      {
         NALU_HYPRE_BigInt *crecvbuf = (NALU_HYPRE_BigInt *)recvbuf;
         NALU_HYPRE_BigInt *csendbuf = (NALU_HYPRE_BigInt *)sendbuf;
         for (i = 0; i < sendcount; i++)
         {
            crecvbuf[i] = csendbuf[i];
         }
      }
      break;

      case hypre_MPI_FLOAT:
      {
         float *crecvbuf = (float *)recvbuf;
         float *csendbuf = (float *)sendbuf;
         for (i = 0; i < sendcount; i++)
         {
            crecvbuf[i] = csendbuf[i];
         }
      }
      break;

      case hypre_MPI_DOUBLE:
      {
         double *crecvbuf = (double *)recvbuf;
         double *csendbuf = (double *)sendbuf;
         for (i = 0; i < sendcount; i++)
         {
            crecvbuf[i] = csendbuf[i];
         }
      }
      break;

      case hypre_MPI_LONG_DOUBLE:
      {
         long double *crecvbuf = (long double *)recvbuf;
         long double *csendbuf = (long double *)sendbuf;
         for (i = 0; i < sendcount; i++)
         {
            crecvbuf[i] = csendbuf[i];
         }
      }
      break;

      case hypre_MPI_CHAR:
      {
         char *crecvbuf = (char *)recvbuf;
         char *csendbuf = (char *)sendbuf;
         for (i = 0; i < sendcount; i++)
         {
            crecvbuf[i] = csendbuf[i];
         }
      }
      break;

      case hypre_MPI_LONG:
      {
         hypre_longint *crecvbuf = (hypre_longint *)recvbuf;
         hypre_longint *csendbuf = (hypre_longint *)sendbuf;
         for (i = 0; i < sendcount; i++)
         {
            crecvbuf[i] = csendbuf[i];
         }
      }
      break;

      case hypre_MPI_BYTE:
      {
         hypre_TMemcpy(recvbuf, sendbuf, char, sendcount, NALU_HYPRE_MEMORY_HOST, NALU_HYPRE_MEMORY_HOST);
      }
      break;

      case hypre_MPI_REAL:
      {
         NALU_HYPRE_Real *crecvbuf = (NALU_HYPRE_Real *)recvbuf;
         NALU_HYPRE_Real *csendbuf = (NALU_HYPRE_Real *)sendbuf;
         for (i = 0; i < sendcount; i++)
         {
            crecvbuf[i] = csendbuf[i];
         }
      }
      break;

      case hypre_MPI_COMPLEX:
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
hypre_MPI_Allgatherv( void               *sendbuf,
                      NALU_HYPRE_Int           sendcount,
                      hypre_MPI_Datatype  sendtype,
                      void               *recvbuf,
                      NALU_HYPRE_Int          *recvcounts,
                      NALU_HYPRE_Int          *displs,
                      hypre_MPI_Datatype  recvtype,
                      hypre_MPI_Comm      comm )
{
   return ( hypre_MPI_Allgather(sendbuf, sendcount, sendtype,
                                recvbuf, *recvcounts, recvtype, comm) );
}

NALU_HYPRE_Int
hypre_MPI_Gather( void               *sendbuf,
                  NALU_HYPRE_Int           sendcount,
                  hypre_MPI_Datatype  sendtype,
                  void               *recvbuf,
                  NALU_HYPRE_Int           recvcount,
                  hypre_MPI_Datatype  recvtype,
                  NALU_HYPRE_Int           root,
                  hypre_MPI_Comm      comm )
{
   return ( hypre_MPI_Allgather(sendbuf, sendcount, sendtype,
                                recvbuf, recvcount, recvtype, comm) );
}

NALU_HYPRE_Int
hypre_MPI_Gatherv( void              *sendbuf,
                   NALU_HYPRE_Int           sendcount,
                   hypre_MPI_Datatype  sendtype,
                   void               *recvbuf,
                   NALU_HYPRE_Int          *recvcounts,
                   NALU_HYPRE_Int          *displs,
                   hypre_MPI_Datatype  recvtype,
                   NALU_HYPRE_Int           root,
                   hypre_MPI_Comm      comm )
{
   return ( hypre_MPI_Allgather(sendbuf, sendcount, sendtype,
                                recvbuf, *recvcounts, recvtype, comm) );
}

NALU_HYPRE_Int
hypre_MPI_Scatter( void               *sendbuf,
                   NALU_HYPRE_Int           sendcount,
                   hypre_MPI_Datatype  sendtype,
                   void               *recvbuf,
                   NALU_HYPRE_Int           recvcount,
                   hypre_MPI_Datatype  recvtype,
                   NALU_HYPRE_Int           root,
                   hypre_MPI_Comm      comm )
{
   return ( hypre_MPI_Allgather(sendbuf, sendcount, sendtype,
                                recvbuf, recvcount, recvtype, comm) );
}

NALU_HYPRE_Int
hypre_MPI_Scatterv( void               *sendbuf,
                    NALU_HYPRE_Int           *sendcounts,
                    NALU_HYPRE_Int           *displs,
                    hypre_MPI_Datatype   sendtype,
                    void                *recvbuf,
                    NALU_HYPRE_Int            recvcount,
                    hypre_MPI_Datatype   recvtype,
                    NALU_HYPRE_Int            root,
                    hypre_MPI_Comm       comm )
{
   return ( hypre_MPI_Allgather(sendbuf, *sendcounts, sendtype,
                                recvbuf, recvcount, recvtype, comm) );
}

NALU_HYPRE_Int
hypre_MPI_Bcast( void               *buffer,
                 NALU_HYPRE_Int           count,
                 hypre_MPI_Datatype  datatype,
                 NALU_HYPRE_Int           root,
                 hypre_MPI_Comm      comm )
{
   return (0);
}

NALU_HYPRE_Int
hypre_MPI_Send( void               *buf,
                NALU_HYPRE_Int           count,
                hypre_MPI_Datatype  datatype,
                NALU_HYPRE_Int           dest,
                NALU_HYPRE_Int           tag,
                hypre_MPI_Comm      comm )
{
   return (0);
}

NALU_HYPRE_Int
hypre_MPI_Recv( void               *buf,
                NALU_HYPRE_Int           count,
                hypre_MPI_Datatype  datatype,
                NALU_HYPRE_Int           source,
                NALU_HYPRE_Int           tag,
                hypre_MPI_Comm      comm,
                hypre_MPI_Status   *status )
{
   return (0);
}

NALU_HYPRE_Int
hypre_MPI_Isend( void               *buf,
                 NALU_HYPRE_Int           count,
                 hypre_MPI_Datatype  datatype,
                 NALU_HYPRE_Int           dest,
                 NALU_HYPRE_Int           tag,
                 hypre_MPI_Comm      comm,
                 hypre_MPI_Request  *request )
{
   return (0);
}

NALU_HYPRE_Int
hypre_MPI_Irecv( void               *buf,
                 NALU_HYPRE_Int           count,
                 hypre_MPI_Datatype  datatype,
                 NALU_HYPRE_Int           source,
                 NALU_HYPRE_Int           tag,
                 hypre_MPI_Comm      comm,
                 hypre_MPI_Request  *request )
{
   return (0);
}

NALU_HYPRE_Int
hypre_MPI_Send_init( void               *buf,
                     NALU_HYPRE_Int           count,
                     hypre_MPI_Datatype  datatype,
                     NALU_HYPRE_Int           dest,
                     NALU_HYPRE_Int           tag,
                     hypre_MPI_Comm      comm,
                     hypre_MPI_Request  *request )
{
   return 0;
}

NALU_HYPRE_Int
hypre_MPI_Recv_init( void               *buf,
                     NALU_HYPRE_Int           count,
                     hypre_MPI_Datatype  datatype,
                     NALU_HYPRE_Int           dest,
                     NALU_HYPRE_Int           tag,
                     hypre_MPI_Comm      comm,
                     hypre_MPI_Request  *request )
{
   return 0;
}

NALU_HYPRE_Int
hypre_MPI_Irsend( void               *buf,
                  NALU_HYPRE_Int           count,
                  hypre_MPI_Datatype  datatype,
                  NALU_HYPRE_Int           dest,
                  NALU_HYPRE_Int           tag,
                  hypre_MPI_Comm      comm,
                  hypre_MPI_Request  *request )
{
   return 0;
}

NALU_HYPRE_Int
hypre_MPI_Startall( NALU_HYPRE_Int          count,
                    hypre_MPI_Request *array_of_requests )
{
   return 0;
}

NALU_HYPRE_Int
hypre_MPI_Probe( NALU_HYPRE_Int         source,
                 NALU_HYPRE_Int         tag,
                 hypre_MPI_Comm    comm,
                 hypre_MPI_Status *status )
{
   return 0;
}

NALU_HYPRE_Int
hypre_MPI_Iprobe( NALU_HYPRE_Int         source,
                  NALU_HYPRE_Int         tag,
                  hypre_MPI_Comm    comm,
                  NALU_HYPRE_Int        *flag,
                  hypre_MPI_Status *status )
{
   return 0;
}

NALU_HYPRE_Int
hypre_MPI_Test( hypre_MPI_Request *request,
                NALU_HYPRE_Int         *flag,
                hypre_MPI_Status  *status )
{
   *flag = 1;
   return (0);
}

NALU_HYPRE_Int
hypre_MPI_Testall( NALU_HYPRE_Int          count,
                   hypre_MPI_Request *array_of_requests,
                   NALU_HYPRE_Int         *flag,
                   hypre_MPI_Status  *array_of_statuses )
{
   *flag = 1;
   return (0);
}

NALU_HYPRE_Int
hypre_MPI_Wait( hypre_MPI_Request *request,
                hypre_MPI_Status  *status )
{
   return (0);
}

NALU_HYPRE_Int
hypre_MPI_Waitall( NALU_HYPRE_Int          count,
                   hypre_MPI_Request *array_of_requests,
                   hypre_MPI_Status  *array_of_statuses )
{
   return (0);
}

NALU_HYPRE_Int
hypre_MPI_Waitany( NALU_HYPRE_Int          count,
                   hypre_MPI_Request *array_of_requests,
                   NALU_HYPRE_Int         *index,
                   hypre_MPI_Status  *status )
{
   return (0);
}

NALU_HYPRE_Int
hypre_MPI_Allreduce( void              *sendbuf,
                     void              *recvbuf,
                     NALU_HYPRE_Int          count,
                     hypre_MPI_Datatype datatype,
                     hypre_MPI_Op       op,
                     hypre_MPI_Comm     comm )
{
   NALU_HYPRE_Int i;

   switch (datatype)
   {
      case hypre_MPI_INT:
      {
         NALU_HYPRE_Int *crecvbuf = (NALU_HYPRE_Int *)recvbuf;
         NALU_HYPRE_Int *csendbuf = (NALU_HYPRE_Int *)sendbuf;
         for (i = 0; i < count; i++)
         {
            crecvbuf[i] = csendbuf[i];
         }
      }
      break;

      case hypre_MPI_LONG_LONG_INT:
      {
         NALU_HYPRE_BigInt *crecvbuf = (NALU_HYPRE_BigInt *)recvbuf;
         NALU_HYPRE_BigInt *csendbuf = (NALU_HYPRE_BigInt *)sendbuf;
         for (i = 0; i < count; i++)
         {
            crecvbuf[i] = csendbuf[i];
         }
      }
      break;

      case hypre_MPI_FLOAT:
      {
         float *crecvbuf = (float *)recvbuf;
         float *csendbuf = (float *)sendbuf;
         for (i = 0; i < count; i++)
         {
            crecvbuf[i] = csendbuf[i];
         }
      }
      break;

      case hypre_MPI_DOUBLE:
      {
         double *crecvbuf = (double *)recvbuf;
         double *csendbuf = (double *)sendbuf;
         for (i = 0; i < count; i++)
         {
            crecvbuf[i] = csendbuf[i];
         }
      }
      break;

      case hypre_MPI_LONG_DOUBLE:
      {
         long double *crecvbuf = (long double *)recvbuf;
         long double *csendbuf = (long double *)sendbuf;
         for (i = 0; i < count; i++)
         {
            crecvbuf[i] = csendbuf[i];
         }
      }
      break;

      case hypre_MPI_CHAR:
      {
         char *crecvbuf = (char *)recvbuf;
         char *csendbuf = (char *)sendbuf;
         for (i = 0; i < count; i++)
         {
            crecvbuf[i] = csendbuf[i];
         }
      }
      break;

      case hypre_MPI_LONG:
      {
         hypre_longint *crecvbuf = (hypre_longint *)recvbuf;
         hypre_longint *csendbuf = (hypre_longint *)sendbuf;
         for (i = 0; i < count; i++)
         {
            crecvbuf[i] = csendbuf[i];
         }
      }
      break;

      case hypre_MPI_BYTE:
      {
         hypre_TMemcpy(recvbuf, sendbuf, char, count, NALU_HYPRE_MEMORY_HOST, NALU_HYPRE_MEMORY_HOST);
      }
      break;

      case hypre_MPI_REAL:
      {
         NALU_HYPRE_Real *crecvbuf = (NALU_HYPRE_Real *)recvbuf;
         NALU_HYPRE_Real *csendbuf = (NALU_HYPRE_Real *)sendbuf;
         for (i = 0; i < count; i++)
         {
            crecvbuf[i] = csendbuf[i];
         }
      }
      break;

      case hypre_MPI_COMPLEX:
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
hypre_MPI_Reduce( void               *sendbuf,
                  void               *recvbuf,
                  NALU_HYPRE_Int           count,
                  hypre_MPI_Datatype  datatype,
                  hypre_MPI_Op        op,
                  NALU_HYPRE_Int           root,
                  hypre_MPI_Comm      comm )
{
   hypre_MPI_Allreduce(sendbuf, recvbuf, count, datatype, op, comm);
   return 0;
}

NALU_HYPRE_Int
hypre_MPI_Scan( void               *sendbuf,
                void               *recvbuf,
                NALU_HYPRE_Int           count,
                hypre_MPI_Datatype  datatype,
                hypre_MPI_Op        op,
                hypre_MPI_Comm      comm )
{
   hypre_MPI_Allreduce(sendbuf, recvbuf, count, datatype, op, comm);
   return 0;
}

NALU_HYPRE_Int
hypre_MPI_Request_free( hypre_MPI_Request *request )
{
   return 0;
}

NALU_HYPRE_Int
hypre_MPI_Type_contiguous( NALU_HYPRE_Int           count,
                           hypre_MPI_Datatype  oldtype,
                           hypre_MPI_Datatype *newtype )
{
   return (0);
}

NALU_HYPRE_Int
hypre_MPI_Type_vector( NALU_HYPRE_Int           count,
                       NALU_HYPRE_Int           blocklength,
                       NALU_HYPRE_Int           stride,
                       hypre_MPI_Datatype  oldtype,
                       hypre_MPI_Datatype *newtype )
{
   return (0);
}

NALU_HYPRE_Int
hypre_MPI_Type_hvector( NALU_HYPRE_Int           count,
                        NALU_HYPRE_Int           blocklength,
                        hypre_MPI_Aint      stride,
                        hypre_MPI_Datatype  oldtype,
                        hypre_MPI_Datatype *newtype )
{
   return (0);
}

NALU_HYPRE_Int
hypre_MPI_Type_struct( NALU_HYPRE_Int           count,
                       NALU_HYPRE_Int          *array_of_blocklengths,
                       hypre_MPI_Aint     *array_of_displacements,
                       hypre_MPI_Datatype *array_of_types,
                       hypre_MPI_Datatype *newtype )
{
   return (0);
}

NALU_HYPRE_Int
hypre_MPI_Type_commit( hypre_MPI_Datatype *datatype )
{
   return (0);
}

NALU_HYPRE_Int
hypre_MPI_Type_free( hypre_MPI_Datatype *datatype )
{
   return (0);
}

NALU_HYPRE_Int
hypre_MPI_Op_create( hypre_MPI_User_function *function, hypre_int commute, hypre_MPI_Op *op )
{
   return (0);
}

NALU_HYPRE_Int
hypre_MPI_Op_free( hypre_MPI_Op *op )
{
   return (0);
}

#if defined(NALU_HYPRE_USING_GPU)
NALU_HYPRE_Int hypre_MPI_Comm_split_type( hypre_MPI_Comm comm, NALU_HYPRE_Int split_type, NALU_HYPRE_Int key,
                                     hypre_MPI_Info info, hypre_MPI_Comm *newcomm )
{
   return (0);
}

NALU_HYPRE_Int hypre_MPI_Info_create( hypre_MPI_Info *info )
{
   return (0);
}

NALU_HYPRE_Int hypre_MPI_Info_free( hypre_MPI_Info *info )
{
   return (0);
}
#endif

/******************************************************************************
 * MPI stubs to do casting of NALU_HYPRE_Int and hypre_int correctly
 *****************************************************************************/

#else

NALU_HYPRE_Int
hypre_MPI_Init( hypre_int   *argc,
                char      ***argv )
{
   return (NALU_HYPRE_Int) MPI_Init(argc, argv);
}

NALU_HYPRE_Int
hypre_MPI_Finalize( )
{
   return (NALU_HYPRE_Int) MPI_Finalize();
}

NALU_HYPRE_Int
hypre_MPI_Abort( hypre_MPI_Comm comm,
                 NALU_HYPRE_Int      errorcode )
{
   return (NALU_HYPRE_Int) MPI_Abort(comm, (hypre_int)errorcode);
}

NALU_HYPRE_Real
hypre_MPI_Wtime( )
{
   return MPI_Wtime();
}

NALU_HYPRE_Real
hypre_MPI_Wtick( )
{
   return MPI_Wtick();
}

NALU_HYPRE_Int
hypre_MPI_Barrier( hypre_MPI_Comm comm )
{
   return (NALU_HYPRE_Int) MPI_Barrier(comm);
}

NALU_HYPRE_Int
hypre_MPI_Comm_create( hypre_MPI_Comm   comm,
                       hypre_MPI_Group  group,
                       hypre_MPI_Comm  *newcomm )
{
   return (NALU_HYPRE_Int) MPI_Comm_create(comm, group, newcomm);
}

NALU_HYPRE_Int
hypre_MPI_Comm_dup( hypre_MPI_Comm  comm,
                    hypre_MPI_Comm *newcomm )
{
   return (NALU_HYPRE_Int) MPI_Comm_dup(comm, newcomm);
}

NALU_HYPRE_Int
hypre_MPI_Comm_size( hypre_MPI_Comm  comm,
                     NALU_HYPRE_Int      *size )
{
   hypre_int mpi_size;
   NALU_HYPRE_Int ierr;
   ierr = (NALU_HYPRE_Int) MPI_Comm_size(comm, &mpi_size);
   *size = (NALU_HYPRE_Int) mpi_size;
   return ierr;
}

NALU_HYPRE_Int
hypre_MPI_Comm_rank( hypre_MPI_Comm  comm,
                     NALU_HYPRE_Int      *rank )
{
   hypre_int mpi_rank;
   NALU_HYPRE_Int ierr;
   ierr = (NALU_HYPRE_Int) MPI_Comm_rank(comm, &mpi_rank);
   *rank = (NALU_HYPRE_Int) mpi_rank;
   return ierr;
}

NALU_HYPRE_Int
hypre_MPI_Comm_free( hypre_MPI_Comm *comm )
{
   return (NALU_HYPRE_Int) MPI_Comm_free(comm);
}

NALU_HYPRE_Int
hypre_MPI_Comm_group( hypre_MPI_Comm   comm,
                      hypre_MPI_Group *group )
{
   return (NALU_HYPRE_Int) MPI_Comm_group(comm, group);
}

NALU_HYPRE_Int
hypre_MPI_Comm_split( hypre_MPI_Comm  comm,
                      NALU_HYPRE_Int       n,
                      NALU_HYPRE_Int       m,
                      hypre_MPI_Comm *comms )
{
   return (NALU_HYPRE_Int) MPI_Comm_split(comm, (hypre_int)n, (hypre_int)m, comms);
}

NALU_HYPRE_Int
hypre_MPI_Group_incl( hypre_MPI_Group  group,
                      NALU_HYPRE_Int        n,
                      NALU_HYPRE_Int       *ranks,
                      hypre_MPI_Group *newgroup )
{
   hypre_int *mpi_ranks;
   NALU_HYPRE_Int  i;
   NALU_HYPRE_Int  ierr;

   mpi_ranks = hypre_TAlloc(hypre_int,  n, NALU_HYPRE_MEMORY_HOST);
   for (i = 0; i < n; i++)
   {
      mpi_ranks[i] = (hypre_int) ranks[i];
   }
   ierr = (NALU_HYPRE_Int) MPI_Group_incl(group, (hypre_int)n, mpi_ranks, newgroup);
   hypre_TFree(mpi_ranks, NALU_HYPRE_MEMORY_HOST);

   return ierr;
}

NALU_HYPRE_Int
hypre_MPI_Group_free( hypre_MPI_Group *group )
{
   return (NALU_HYPRE_Int) MPI_Group_free(group);
}

NALU_HYPRE_Int
hypre_MPI_Address( void           *location,
                   hypre_MPI_Aint *address )
{
#if MPI_VERSION > 1
   return (NALU_HYPRE_Int) MPI_Get_address(location, address);
#else
   return (NALU_HYPRE_Int) MPI_Address(location, address);
#endif
}

NALU_HYPRE_Int
hypre_MPI_Get_count( hypre_MPI_Status   *status,
                     hypre_MPI_Datatype  datatype,
                     NALU_HYPRE_Int          *count )
{
   hypre_int mpi_count;
   NALU_HYPRE_Int ierr;
   ierr = (NALU_HYPRE_Int) MPI_Get_count(status, datatype, &mpi_count);
   *count = (NALU_HYPRE_Int) mpi_count;
   return ierr;
}

NALU_HYPRE_Int
hypre_MPI_Alltoall( void               *sendbuf,
                    NALU_HYPRE_Int           sendcount,
                    hypre_MPI_Datatype  sendtype,
                    void               *recvbuf,
                    NALU_HYPRE_Int           recvcount,
                    hypre_MPI_Datatype  recvtype,
                    hypre_MPI_Comm      comm )
{
   return (NALU_HYPRE_Int) MPI_Alltoall(sendbuf, (hypre_int)sendcount, sendtype,
                                   recvbuf, (hypre_int)recvcount, recvtype, comm);
}

NALU_HYPRE_Int
hypre_MPI_Allgather( void               *sendbuf,
                     NALU_HYPRE_Int           sendcount,
                     hypre_MPI_Datatype  sendtype,
                     void               *recvbuf,
                     NALU_HYPRE_Int           recvcount,
                     hypre_MPI_Datatype  recvtype,
                     hypre_MPI_Comm      comm )
{
   return (NALU_HYPRE_Int) MPI_Allgather(sendbuf, (hypre_int)sendcount, sendtype,
                                    recvbuf, (hypre_int)recvcount, recvtype, comm);
}

NALU_HYPRE_Int
hypre_MPI_Allgatherv( void               *sendbuf,
                      NALU_HYPRE_Int           sendcount,
                      hypre_MPI_Datatype  sendtype,
                      void               *recvbuf,
                      NALU_HYPRE_Int          *recvcounts,
                      NALU_HYPRE_Int          *displs,
                      hypre_MPI_Datatype  recvtype,
                      hypre_MPI_Comm      comm )
{
   hypre_int *mpi_recvcounts, *mpi_displs, csize;
   NALU_HYPRE_Int  i;
   NALU_HYPRE_Int  ierr;

   MPI_Comm_size(comm, &csize);
   mpi_recvcounts = hypre_TAlloc(hypre_int, csize, NALU_HYPRE_MEMORY_HOST);
   mpi_displs = hypre_TAlloc(hypre_int, csize, NALU_HYPRE_MEMORY_HOST);
   for (i = 0; i < csize; i++)
   {
      mpi_recvcounts[i] = (hypre_int) recvcounts[i];
      mpi_displs[i] = (hypre_int) displs[i];
   }
   ierr = (NALU_HYPRE_Int) MPI_Allgatherv(sendbuf, (hypre_int)sendcount, sendtype,
                                     recvbuf, mpi_recvcounts, mpi_displs,
                                     recvtype, comm);
   hypre_TFree(mpi_recvcounts, NALU_HYPRE_MEMORY_HOST);
   hypre_TFree(mpi_displs, NALU_HYPRE_MEMORY_HOST);

   return ierr;
}

NALU_HYPRE_Int
hypre_MPI_Gather( void               *sendbuf,
                  NALU_HYPRE_Int           sendcount,
                  hypre_MPI_Datatype  sendtype,
                  void               *recvbuf,
                  NALU_HYPRE_Int           recvcount,
                  hypre_MPI_Datatype  recvtype,
                  NALU_HYPRE_Int           root,
                  hypre_MPI_Comm      comm )
{
   return (NALU_HYPRE_Int) MPI_Gather(sendbuf, (hypre_int) sendcount, sendtype,
                                 recvbuf, (hypre_int) recvcount, recvtype,
                                 (hypre_int)root, comm);
}

NALU_HYPRE_Int
hypre_MPI_Gatherv(void               *sendbuf,
                  NALU_HYPRE_Int           sendcount,
                  hypre_MPI_Datatype  sendtype,
                  void               *recvbuf,
                  NALU_HYPRE_Int          *recvcounts,
                  NALU_HYPRE_Int          *displs,
                  hypre_MPI_Datatype  recvtype,
                  NALU_HYPRE_Int           root,
                  hypre_MPI_Comm      comm )
{
   hypre_int *mpi_recvcounts = NULL;
   hypre_int *mpi_displs = NULL;
   hypre_int csize, croot;
   NALU_HYPRE_Int  i;
   NALU_HYPRE_Int  ierr;

   MPI_Comm_size(comm, &csize);
   MPI_Comm_rank(comm, &croot);
   if (croot == (hypre_int) root)
   {
      mpi_recvcounts = hypre_TAlloc(hypre_int,  csize, NALU_HYPRE_MEMORY_HOST);
      mpi_displs = hypre_TAlloc(hypre_int,  csize, NALU_HYPRE_MEMORY_HOST);
      for (i = 0; i < csize; i++)
      {
         mpi_recvcounts[i] = (hypre_int) recvcounts[i];
         mpi_displs[i] = (hypre_int) displs[i];
      }
   }
   ierr = (NALU_HYPRE_Int) MPI_Gatherv(sendbuf, (hypre_int)sendcount, sendtype,
                                  recvbuf, mpi_recvcounts, mpi_displs,
                                  recvtype, (hypre_int) root, comm);
   hypre_TFree(mpi_recvcounts, NALU_HYPRE_MEMORY_HOST);
   hypre_TFree(mpi_displs, NALU_HYPRE_MEMORY_HOST);

   return ierr;
}

NALU_HYPRE_Int
hypre_MPI_Scatter( void               *sendbuf,
                   NALU_HYPRE_Int           sendcount,
                   hypre_MPI_Datatype  sendtype,
                   void               *recvbuf,
                   NALU_HYPRE_Int           recvcount,
                   hypre_MPI_Datatype  recvtype,
                   NALU_HYPRE_Int           root,
                   hypre_MPI_Comm      comm )
{
   return (NALU_HYPRE_Int) MPI_Scatter(sendbuf, (hypre_int)sendcount, sendtype,
                                  recvbuf, (hypre_int)recvcount, recvtype,
                                  (hypre_int)root, comm);
}

NALU_HYPRE_Int
hypre_MPI_Scatterv(void               *sendbuf,
                   NALU_HYPRE_Int          *sendcounts,
                   NALU_HYPRE_Int          *displs,
                   hypre_MPI_Datatype  sendtype,
                   void               *recvbuf,
                   NALU_HYPRE_Int           recvcount,
                   hypre_MPI_Datatype  recvtype,
                   NALU_HYPRE_Int           root,
                   hypre_MPI_Comm      comm )
{
   hypre_int *mpi_sendcounts = NULL;
   hypre_int *mpi_displs = NULL;
   hypre_int csize, croot;
   NALU_HYPRE_Int  i;
   NALU_HYPRE_Int  ierr;

   MPI_Comm_size(comm, &csize);
   MPI_Comm_rank(comm, &croot);
   if (croot == (hypre_int) root)
   {
      mpi_sendcounts = hypre_TAlloc(hypre_int,  csize, NALU_HYPRE_MEMORY_HOST);
      mpi_displs = hypre_TAlloc(hypre_int,  csize, NALU_HYPRE_MEMORY_HOST);
      for (i = 0; i < csize; i++)
      {
         mpi_sendcounts[i] = (hypre_int) sendcounts[i];
         mpi_displs[i] = (hypre_int) displs[i];
      }
   }
   ierr = (NALU_HYPRE_Int) MPI_Scatterv(sendbuf, mpi_sendcounts, mpi_displs, sendtype,
                                   recvbuf, (hypre_int) recvcount,
                                   recvtype, (hypre_int) root, comm);
   hypre_TFree(mpi_sendcounts, NALU_HYPRE_MEMORY_HOST);
   hypre_TFree(mpi_displs, NALU_HYPRE_MEMORY_HOST);

   return ierr;
}

NALU_HYPRE_Int
hypre_MPI_Bcast( void               *buffer,
                 NALU_HYPRE_Int           count,
                 hypre_MPI_Datatype  datatype,
                 NALU_HYPRE_Int           root,
                 hypre_MPI_Comm      comm )
{
   return (NALU_HYPRE_Int) MPI_Bcast(buffer, (hypre_int)count, datatype,
                                (hypre_int)root, comm);
}

NALU_HYPRE_Int
hypre_MPI_Send( void               *buf,
                NALU_HYPRE_Int           count,
                hypre_MPI_Datatype  datatype,
                NALU_HYPRE_Int           dest,
                NALU_HYPRE_Int           tag,
                hypre_MPI_Comm      comm )
{
   return (NALU_HYPRE_Int) MPI_Send(buf, (hypre_int)count, datatype,
                               (hypre_int)dest, (hypre_int)tag, comm);
}

NALU_HYPRE_Int
hypre_MPI_Recv( void               *buf,
                NALU_HYPRE_Int           count,
                hypre_MPI_Datatype  datatype,
                NALU_HYPRE_Int           source,
                NALU_HYPRE_Int           tag,
                hypre_MPI_Comm      comm,
                hypre_MPI_Status   *status )
{
   return (NALU_HYPRE_Int) MPI_Recv(buf, (hypre_int)count, datatype,
                               (hypre_int)source, (hypre_int)tag, comm, status);
}

NALU_HYPRE_Int
hypre_MPI_Isend( void               *buf,
                 NALU_HYPRE_Int           count,
                 hypre_MPI_Datatype  datatype,
                 NALU_HYPRE_Int           dest,
                 NALU_HYPRE_Int           tag,
                 hypre_MPI_Comm      comm,
                 hypre_MPI_Request  *request )
{
   return (NALU_HYPRE_Int) MPI_Isend(buf, (hypre_int)count, datatype,
                                (hypre_int)dest, (hypre_int)tag, comm, request);
}

NALU_HYPRE_Int
hypre_MPI_Irecv( void               *buf,
                 NALU_HYPRE_Int           count,
                 hypre_MPI_Datatype  datatype,
                 NALU_HYPRE_Int           source,
                 NALU_HYPRE_Int           tag,
                 hypre_MPI_Comm      comm,
                 hypre_MPI_Request  *request )
{
   return (NALU_HYPRE_Int) MPI_Irecv(buf, (hypre_int)count, datatype,
                                (hypre_int)source, (hypre_int)tag, comm, request);
}

NALU_HYPRE_Int
hypre_MPI_Send_init( void               *buf,
                     NALU_HYPRE_Int           count,
                     hypre_MPI_Datatype  datatype,
                     NALU_HYPRE_Int           dest,
                     NALU_HYPRE_Int           tag,
                     hypre_MPI_Comm      comm,
                     hypre_MPI_Request  *request )
{
   return (NALU_HYPRE_Int) MPI_Send_init(buf, (hypre_int)count, datatype,
                                    (hypre_int)dest, (hypre_int)tag,
                                    comm, request);
}

NALU_HYPRE_Int
hypre_MPI_Recv_init( void               *buf,
                     NALU_HYPRE_Int           count,
                     hypre_MPI_Datatype  datatype,
                     NALU_HYPRE_Int           dest,
                     NALU_HYPRE_Int           tag,
                     hypre_MPI_Comm      comm,
                     hypre_MPI_Request  *request )
{
   return (NALU_HYPRE_Int) MPI_Recv_init(buf, (hypre_int)count, datatype,
                                    (hypre_int)dest, (hypre_int)tag,
                                    comm, request);
}

NALU_HYPRE_Int
hypre_MPI_Irsend( void               *buf,
                  NALU_HYPRE_Int           count,
                  hypre_MPI_Datatype  datatype,
                  NALU_HYPRE_Int           dest,
                  NALU_HYPRE_Int           tag,
                  hypre_MPI_Comm      comm,
                  hypre_MPI_Request  *request )
{
   return (NALU_HYPRE_Int) MPI_Irsend(buf, (hypre_int)count, datatype,
                                 (hypre_int)dest, (hypre_int)tag, comm, request);
}

NALU_HYPRE_Int
hypre_MPI_Startall( NALU_HYPRE_Int          count,
                    hypre_MPI_Request *array_of_requests )
{
   return (NALU_HYPRE_Int) MPI_Startall((hypre_int)count, array_of_requests);
}

NALU_HYPRE_Int
hypre_MPI_Probe( NALU_HYPRE_Int         source,
                 NALU_HYPRE_Int         tag,
                 hypre_MPI_Comm    comm,
                 hypre_MPI_Status *status )
{
   return (NALU_HYPRE_Int) MPI_Probe((hypre_int)source, (hypre_int)tag, comm, status);
}

NALU_HYPRE_Int
hypre_MPI_Iprobe( NALU_HYPRE_Int         source,
                  NALU_HYPRE_Int         tag,
                  hypre_MPI_Comm    comm,
                  NALU_HYPRE_Int        *flag,
                  hypre_MPI_Status *status )
{
   hypre_int mpi_flag;
   NALU_HYPRE_Int ierr;
   ierr = (NALU_HYPRE_Int) MPI_Iprobe((hypre_int)source, (hypre_int)tag, comm,
                                 &mpi_flag, status);
   *flag = (NALU_HYPRE_Int) mpi_flag;
   return ierr;
}

NALU_HYPRE_Int
hypre_MPI_Test( hypre_MPI_Request *request,
                NALU_HYPRE_Int         *flag,
                hypre_MPI_Status  *status )
{
   hypre_int mpi_flag;
   NALU_HYPRE_Int ierr;
   ierr = (NALU_HYPRE_Int) MPI_Test(request, &mpi_flag, status);
   *flag = (NALU_HYPRE_Int) mpi_flag;
   return ierr;
}

NALU_HYPRE_Int
hypre_MPI_Testall( NALU_HYPRE_Int          count,
                   hypre_MPI_Request *array_of_requests,
                   NALU_HYPRE_Int         *flag,
                   hypre_MPI_Status  *array_of_statuses )
{
   hypre_int mpi_flag;
   NALU_HYPRE_Int ierr;
   ierr = (NALU_HYPRE_Int) MPI_Testall((hypre_int)count, array_of_requests,
                                  &mpi_flag, array_of_statuses);
   *flag = (NALU_HYPRE_Int) mpi_flag;
   return ierr;
}

NALU_HYPRE_Int
hypre_MPI_Wait( hypre_MPI_Request *request,
                hypre_MPI_Status  *status )
{
   return (NALU_HYPRE_Int) MPI_Wait(request, status);
}

NALU_HYPRE_Int
hypre_MPI_Waitall( NALU_HYPRE_Int          count,
                   hypre_MPI_Request *array_of_requests,
                   hypre_MPI_Status  *array_of_statuses )
{
   return (NALU_HYPRE_Int) MPI_Waitall((hypre_int)count,
                                  array_of_requests, array_of_statuses);
}

NALU_HYPRE_Int
hypre_MPI_Waitany( NALU_HYPRE_Int          count,
                   hypre_MPI_Request *array_of_requests,
                   NALU_HYPRE_Int         *index,
                   hypre_MPI_Status  *status )
{
   hypre_int mpi_index;
   NALU_HYPRE_Int ierr;
   ierr = (NALU_HYPRE_Int) MPI_Waitany((hypre_int)count, array_of_requests,
                                  &mpi_index, status);
   *index = (NALU_HYPRE_Int) mpi_index;
   return ierr;
}

NALU_HYPRE_Int
hypre_MPI_Allreduce( void              *sendbuf,
                     void              *recvbuf,
                     NALU_HYPRE_Int          count,
                     hypre_MPI_Datatype datatype,
                     hypre_MPI_Op       op,
                     hypre_MPI_Comm     comm )
{
#if defined(NALU_HYPRE_USING_NVTX)
   hypre_GpuProfilingPushRange("MPI_Allreduce");
#endif

   NALU_HYPRE_Int result = MPI_Allreduce(sendbuf, recvbuf, (hypre_int)count,
                                    datatype, op, comm);

#if defined(NALU_HYPRE_USING_NVTX)
   hypre_GpuProfilingPopRange();
#endif

   return result;
}

NALU_HYPRE_Int
hypre_MPI_Reduce( void               *sendbuf,
                  void               *recvbuf,
                  NALU_HYPRE_Int           count,
                  hypre_MPI_Datatype  datatype,
                  hypre_MPI_Op        op,
                  NALU_HYPRE_Int           root,
                  hypre_MPI_Comm      comm )
{
   return (NALU_HYPRE_Int) MPI_Reduce(sendbuf, recvbuf, (hypre_int)count,
                                 datatype, op, (hypre_int)root, comm);
}

NALU_HYPRE_Int
hypre_MPI_Scan( void               *sendbuf,
                void               *recvbuf,
                NALU_HYPRE_Int           count,
                hypre_MPI_Datatype  datatype,
                hypre_MPI_Op        op,
                hypre_MPI_Comm      comm )
{
   return (NALU_HYPRE_Int) MPI_Scan(sendbuf, recvbuf, (hypre_int)count,
                               datatype, op, comm);
}

NALU_HYPRE_Int
hypre_MPI_Request_free( hypre_MPI_Request *request )
{
   return (NALU_HYPRE_Int) MPI_Request_free(request);
}

NALU_HYPRE_Int
hypre_MPI_Type_contiguous( NALU_HYPRE_Int           count,
                           hypre_MPI_Datatype  oldtype,
                           hypre_MPI_Datatype *newtype )
{
   return (NALU_HYPRE_Int) MPI_Type_contiguous((hypre_int)count, oldtype, newtype);
}

NALU_HYPRE_Int
hypre_MPI_Type_vector( NALU_HYPRE_Int           count,
                       NALU_HYPRE_Int           blocklength,
                       NALU_HYPRE_Int           stride,
                       hypre_MPI_Datatype  oldtype,
                       hypre_MPI_Datatype *newtype )
{
   return (NALU_HYPRE_Int) MPI_Type_vector((hypre_int)count, (hypre_int)blocklength,
                                      (hypre_int)stride, oldtype, newtype);
}

NALU_HYPRE_Int
hypre_MPI_Type_hvector( NALU_HYPRE_Int           count,
                        NALU_HYPRE_Int           blocklength,
                        hypre_MPI_Aint      stride,
                        hypre_MPI_Datatype  oldtype,
                        hypre_MPI_Datatype *newtype )
{
#if MPI_VERSION > 1
   return (NALU_HYPRE_Int) MPI_Type_create_hvector((hypre_int)count, (hypre_int)blocklength,
                                              stride, oldtype, newtype);
#else
   return (NALU_HYPRE_Int) MPI_Type_hvector((hypre_int)count, (hypre_int)blocklength,
                                       stride, oldtype, newtype);
#endif
}

NALU_HYPRE_Int
hypre_MPI_Type_struct( NALU_HYPRE_Int           count,
                       NALU_HYPRE_Int          *array_of_blocklengths,
                       hypre_MPI_Aint     *array_of_displacements,
                       hypre_MPI_Datatype *array_of_types,
                       hypre_MPI_Datatype *newtype )
{
   hypre_int *mpi_array_of_blocklengths;
   NALU_HYPRE_Int  i;
   NALU_HYPRE_Int  ierr;

   mpi_array_of_blocklengths = hypre_TAlloc(hypre_int,  count, NALU_HYPRE_MEMORY_HOST);
   for (i = 0; i < count; i++)
   {
      mpi_array_of_blocklengths[i] = (hypre_int) array_of_blocklengths[i];
   }

#if MPI_VERSION > 1
   ierr = (NALU_HYPRE_Int) MPI_Type_create_struct((hypre_int)count, mpi_array_of_blocklengths,
                                             array_of_displacements, array_of_types,
                                             newtype);
#else
   ierr = (NALU_HYPRE_Int) MPI_Type_struct((hypre_int)count, mpi_array_of_blocklengths,
                                      array_of_displacements, array_of_types,
                                      newtype);
#endif

   hypre_TFree(mpi_array_of_blocklengths, NALU_HYPRE_MEMORY_HOST);

   return ierr;
}

NALU_HYPRE_Int
hypre_MPI_Type_commit( hypre_MPI_Datatype *datatype )
{
   return (NALU_HYPRE_Int) MPI_Type_commit(datatype);
}

NALU_HYPRE_Int
hypre_MPI_Type_free( hypre_MPI_Datatype *datatype )
{
   return (NALU_HYPRE_Int) MPI_Type_free(datatype);
}

NALU_HYPRE_Int
hypre_MPI_Op_free( hypre_MPI_Op *op )
{
   return (NALU_HYPRE_Int) MPI_Op_free(op);
}

NALU_HYPRE_Int
hypre_MPI_Op_create( hypre_MPI_User_function *function, hypre_int commute, hypre_MPI_Op *op )
{
   return (NALU_HYPRE_Int) MPI_Op_create(function, commute, op);
}

#if defined(NALU_HYPRE_USING_GPU)
NALU_HYPRE_Int
hypre_MPI_Comm_split_type( hypre_MPI_Comm comm, NALU_HYPRE_Int split_type, NALU_HYPRE_Int key,
                           hypre_MPI_Info info, hypre_MPI_Comm *newcomm )
{
   return (NALU_HYPRE_Int) MPI_Comm_split_type(comm, split_type, key, info, newcomm );
}

NALU_HYPRE_Int
hypre_MPI_Info_create( hypre_MPI_Info *info )
{
   return (NALU_HYPRE_Int) MPI_Info_create(info);
}

NALU_HYPRE_Int
hypre_MPI_Info_free( hypre_MPI_Info *info )
{
   return (NALU_HYPRE_Int) MPI_Info_free(info);
}
#endif

#endif
