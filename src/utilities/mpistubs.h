/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 *  Fake mpi stubs to generate serial codes without mpi
 *
 *****************************************************************************/

#ifndef nalu_hypre_MPISTUBS
#define nalu_hypre_MPISTUBS

#ifdef __cplusplus
extern "C" {
#endif

#ifdef NALU_HYPRE_SEQUENTIAL

/******************************************************************************
 * MPI stubs to generate serial codes without mpi
 *****************************************************************************/

/*--------------------------------------------------------------------------
 * Change all MPI names to nalu_hypre_MPI names to avoid link conflicts.
 *
 * NOTE: MPI_Comm is the only MPI symbol in the HYPRE user interface,
 * and is defined in `NALU_HYPRE_utilities.h'.
 *--------------------------------------------------------------------------*/

#define MPI_Comm            nalu_hypre_MPI_Comm
#define MPI_Group           nalu_hypre_MPI_Group
#define MPI_Request         nalu_hypre_MPI_Request
#define MPI_Datatype        nalu_hypre_MPI_Datatype
#define MPI_Status          nalu_hypre_MPI_Status
#define MPI_Op              nalu_hypre_MPI_Op
#define MPI_Aint            nalu_hypre_MPI_Aint
#define MPI_Info            nalu_hypre_MPI_Info

#define MPI_COMM_WORLD       nalu_hypre_MPI_COMM_WORLD
#define MPI_COMM_NULL        nalu_hypre_MPI_COMM_NULL
#define MPI_COMM_SELF        nalu_hypre_MPI_COMM_SELF
#define MPI_COMM_TYPE_SHARED nalu_hypre_MPI_COMM_TYPE_SHARED

#define MPI_BOTTOM          nalu_hypre_MPI_BOTTOM

#define MPI_FLOAT           nalu_hypre_MPI_FLOAT
#define MPI_DOUBLE          nalu_hypre_MPI_DOUBLE
#define MPI_LONG_DOUBLE     nalu_hypre_MPI_LONG_DOUBLE
#define MPI_INT             nalu_hypre_MPI_INT
#define MPI_LONG_LONG_INT   nalu_hypre_MPI_LONG_LONG_INT
#define MPI_CHAR            nalu_hypre_MPI_CHAR
#define MPI_LONG            nalu_hypre_MPI_LONG
#define MPI_BYTE            nalu_hypre_MPI_BYTE
#define MPI_C_DOUBLE_COMPLEX nalu_hypre_MPI_COMPLEX

#define MPI_SUM             nalu_hypre_MPI_SUM
#define MPI_MIN             nalu_hypre_MPI_MIN
#define MPI_MAX             nalu_hypre_MPI_MAX
#define MPI_LOR             nalu_hypre_MPI_LOR
#define MPI_LAND            nalu_hypre_MPI_LAND
#define MPI_SUCCESS         nalu_hypre_MPI_SUCCESS
#define MPI_STATUSES_IGNORE nalu_hypre_MPI_STATUSES_IGNORE

#define MPI_UNDEFINED       nalu_hypre_MPI_UNDEFINED
#define MPI_REQUEST_NULL    nalu_hypre_MPI_REQUEST_NULL
#define MPI_INFO_NULL       nalu_hypre_MPI_INFO_NULL
#define MPI_ANY_SOURCE      nalu_hypre_MPI_ANY_SOURCE
#define MPI_ANY_TAG         nalu_hypre_MPI_ANY_TAG
#define MPI_SOURCE          nalu_hypre_MPI_SOURCE
#define MPI_TAG             nalu_hypre_MPI_TAG

#define MPI_Init            nalu_hypre_MPI_Init
#define MPI_Finalize        nalu_hypre_MPI_Finalize
#define MPI_Abort           nalu_hypre_MPI_Abort
#define MPI_Wtime           nalu_hypre_MPI_Wtime
#define MPI_Wtick           nalu_hypre_MPI_Wtick
#define MPI_Barrier         nalu_hypre_MPI_Barrier
#define MPI_Comm_create     nalu_hypre_MPI_Comm_create
#define MPI_Comm_dup        nalu_hypre_MPI_Comm_dup
#define MPI_Comm_f2c        nalu_hypre_MPI_Comm_f2c
#define MPI_Comm_group      nalu_hypre_MPI_Comm_group
#define MPI_Comm_size       nalu_hypre_MPI_Comm_size
#define MPI_Comm_rank       nalu_hypre_MPI_Comm_rank
#define MPI_Comm_free       nalu_hypre_MPI_Comm_free
#define MPI_Comm_split      nalu_hypre_MPI_Comm_split
#define MPI_Comm_split_type nalu_hypre_MPI_Comm_split_type
#define MPI_Group_incl      nalu_hypre_MPI_Group_incl
#define MPI_Group_free      nalu_hypre_MPI_Group_free
#define MPI_Address         nalu_hypre_MPI_Address
#define MPI_Get_count       nalu_hypre_MPI_Get_count
#define MPI_Alltoall        nalu_hypre_MPI_Alltoall
#define MPI_Allgather       nalu_hypre_MPI_Allgather
#define MPI_Allgatherv      nalu_hypre_MPI_Allgatherv
#define MPI_Gather          nalu_hypre_MPI_Gather
#define MPI_Gatherv         nalu_hypre_MPI_Gatherv
#define MPI_Scatter         nalu_hypre_MPI_Scatter
#define MPI_Scatterv        nalu_hypre_MPI_Scatterv
#define MPI_Bcast           nalu_hypre_MPI_Bcast
#define MPI_Send            nalu_hypre_MPI_Send
#define MPI_Recv            nalu_hypre_MPI_Recv
#define MPI_Isend           nalu_hypre_MPI_Isend
#define MPI_Irecv           nalu_hypre_MPI_Irecv
#define MPI_Send_init       nalu_hypre_MPI_Send_init
#define MPI_Recv_init       nalu_hypre_MPI_Recv_init
#define MPI_Irsend          nalu_hypre_MPI_Irsend
#define MPI_Startall        nalu_hypre_MPI_Startall
#define MPI_Probe           nalu_hypre_MPI_Probe
#define MPI_Iprobe          nalu_hypre_MPI_Iprobe
#define MPI_Test            nalu_hypre_MPI_Test
#define MPI_Testall         nalu_hypre_MPI_Testall
#define MPI_Wait            nalu_hypre_MPI_Wait
#define MPI_Waitall         nalu_hypre_MPI_Waitall
#define MPI_Waitany         nalu_hypre_MPI_Waitany
#define MPI_Allreduce       nalu_hypre_MPI_Allreduce
#define MPI_Reduce          nalu_hypre_MPI_Reduce
#define MPI_Scan            nalu_hypre_MPI_Scan
#define MPI_Request_free    nalu_hypre_MPI_Request_free
#define MPI_Type_contiguous nalu_hypre_MPI_Type_contiguous
#define MPI_Type_vector     nalu_hypre_MPI_Type_vector
#define MPI_Type_hvector    nalu_hypre_MPI_Type_hvector
#define MPI_Type_struct     nalu_hypre_MPI_Type_struct
#define MPI_Type_commit     nalu_hypre_MPI_Type_commit
#define MPI_Type_free       nalu_hypre_MPI_Type_free
#define MPI_Op_free         nalu_hypre_MPI_Op_free
#define MPI_Op_create       nalu_hypre_MPI_Op_create
#define MPI_User_function   nalu_hypre_MPI_User_function
#define MPI_Info_create     nalu_hypre_MPI_Info_create

/*--------------------------------------------------------------------------
 * Types, etc.
 *--------------------------------------------------------------------------*/

/* These types have associated creation and destruction routines */
typedef NALU_HYPRE_Int nalu_hypre_MPI_Comm;
typedef NALU_HYPRE_Int nalu_hypre_MPI_Group;
typedef NALU_HYPRE_Int nalu_hypre_MPI_Request;
typedef NALU_HYPRE_Int nalu_hypre_MPI_Datatype;
typedef void (nalu_hypre_MPI_User_function) ();

typedef struct
{
   NALU_HYPRE_Int nalu_hypre_MPI_SOURCE;
   NALU_HYPRE_Int nalu_hypre_MPI_TAG;
} nalu_hypre_MPI_Status;

typedef NALU_HYPRE_Int  nalu_hypre_MPI_Op;
typedef NALU_HYPRE_Int  nalu_hypre_MPI_Aint;
typedef NALU_HYPRE_Int  nalu_hypre_MPI_Info;

#define  nalu_hypre_MPI_COMM_SELF   1
#define  nalu_hypre_MPI_COMM_WORLD  0
#define  nalu_hypre_MPI_COMM_NULL  -1

#define  nalu_hypre_MPI_COMM_TYPE_SHARED 0

#define  nalu_hypre_MPI_BOTTOM  0x0

#define  nalu_hypre_MPI_FLOAT 0
#define  nalu_hypre_MPI_DOUBLE 1
#define  nalu_hypre_MPI_LONG_DOUBLE 2
#define  nalu_hypre_MPI_INT 3
#define  nalu_hypre_MPI_CHAR 4
#define  nalu_hypre_MPI_LONG 5
#define  nalu_hypre_MPI_BYTE 6
#define  nalu_hypre_MPI_REAL 7
#define  nalu_hypre_MPI_COMPLEX 8
#define  nalu_hypre_MPI_LONG_LONG_INT 9

#define  nalu_hypre_MPI_SUM 0
#define  nalu_hypre_MPI_MIN 1
#define  nalu_hypre_MPI_MAX 2
#define  nalu_hypre_MPI_LOR 3
#define  nalu_hypre_MPI_LAND 4
#define  nalu_hypre_MPI_SUCCESS 0
#define  nalu_hypre_MPI_STATUSES_IGNORE 0

#define  nalu_hypre_MPI_UNDEFINED -9999
#define  nalu_hypre_MPI_REQUEST_NULL  0
#define  nalu_hypre_MPI_INFO_NULL     0
#define  nalu_hypre_MPI_ANY_SOURCE    1
#define  nalu_hypre_MPI_ANY_TAG       1

#else

/******************************************************************************
 * MPI stubs to do casting of NALU_HYPRE_Int and nalu_hypre_int correctly
 *****************************************************************************/

typedef MPI_Comm     nalu_hypre_MPI_Comm;
typedef MPI_Group    nalu_hypre_MPI_Group;
typedef MPI_Request  nalu_hypre_MPI_Request;
typedef MPI_Datatype nalu_hypre_MPI_Datatype;
typedef MPI_Status   nalu_hypre_MPI_Status;
typedef MPI_Op       nalu_hypre_MPI_Op;
typedef MPI_Aint     nalu_hypre_MPI_Aint;
typedef MPI_Info     nalu_hypre_MPI_Info;
typedef MPI_User_function    nalu_hypre_MPI_User_function;

#define  nalu_hypre_MPI_COMM_WORLD         MPI_COMM_WORLD
#define  nalu_hypre_MPI_COMM_NULL          MPI_COMM_NULL
#define  nalu_hypre_MPI_BOTTOM             MPI_BOTTOM
#define  nalu_hypre_MPI_COMM_SELF          MPI_COMM_SELF
#define  nalu_hypre_MPI_COMM_TYPE_SHARED   MPI_COMM_TYPE_SHARED

#define  nalu_hypre_MPI_FLOAT   MPI_FLOAT
#define  nalu_hypre_MPI_DOUBLE  MPI_DOUBLE
#define  nalu_hypre_MPI_LONG_DOUBLE  MPI_LONG_DOUBLE
/* NALU_HYPRE_MPI_INT is defined in NALU_HYPRE_utilities.h */
#define  nalu_hypre_MPI_INT     NALU_HYPRE_MPI_INT
#define  nalu_hypre_MPI_CHAR    MPI_CHAR
#define  nalu_hypre_MPI_LONG    MPI_LONG
#define  nalu_hypre_MPI_BYTE    MPI_BYTE
/* NALU_HYPRE_MPI_REAL is defined in NALU_HYPRE_utilities.h */
#define  nalu_hypre_MPI_REAL    NALU_HYPRE_MPI_REAL
/* NALU_HYPRE_MPI_COMPLEX is defined in NALU_HYPRE_utilities.h */
#define  nalu_hypre_MPI_COMPLEX NALU_HYPRE_MPI_COMPLEX

#define  nalu_hypre_MPI_SUM MPI_SUM
#define  nalu_hypre_MPI_MIN MPI_MIN
#define  nalu_hypre_MPI_MAX MPI_MAX
#define  nalu_hypre_MPI_LOR MPI_LOR
#define  nalu_hypre_MPI_SUCCESS MPI_SUCCESS
#define  nalu_hypre_MPI_STATUSES_IGNORE MPI_STATUSES_IGNORE

#define  nalu_hypre_MPI_UNDEFINED       MPI_UNDEFINED
#define  nalu_hypre_MPI_REQUEST_NULL    MPI_REQUEST_NULL
#define  nalu_hypre_MPI_INFO_NULL       MPI_INFO_NULL
#define  nalu_hypre_MPI_ANY_SOURCE      MPI_ANY_SOURCE
#define  nalu_hypre_MPI_ANY_TAG         MPI_ANY_TAG
#define  nalu_hypre_MPI_SOURCE          MPI_SOURCE
#define  nalu_hypre_MPI_TAG             MPI_TAG
#define  nalu_hypre_MPI_LAND            MPI_LAND

#endif

/******************************************************************************
 * Everything below this applies to both ifdef cases above
 *****************************************************************************/

/*--------------------------------------------------------------------------
 * Prototypes
 *--------------------------------------------------------------------------*/

/* mpistubs.c */
NALU_HYPRE_Int nalu_hypre_MPI_Init( nalu_hypre_int *argc, char ***argv );
NALU_HYPRE_Int nalu_hypre_MPI_Finalize( void );
NALU_HYPRE_Int nalu_hypre_MPI_Abort( nalu_hypre_MPI_Comm comm, NALU_HYPRE_Int errorcode );
NALU_HYPRE_Real nalu_hypre_MPI_Wtime( void );
NALU_HYPRE_Real nalu_hypre_MPI_Wtick( void );
NALU_HYPRE_Int nalu_hypre_MPI_Barrier( nalu_hypre_MPI_Comm comm );
NALU_HYPRE_Int nalu_hypre_MPI_Comm_create( nalu_hypre_MPI_Comm comm, nalu_hypre_MPI_Group group,
                                 nalu_hypre_MPI_Comm *newcomm );
NALU_HYPRE_Int nalu_hypre_MPI_Comm_dup( nalu_hypre_MPI_Comm comm, nalu_hypre_MPI_Comm *newcomm );
nalu_hypre_MPI_Comm nalu_hypre_MPI_Comm_f2c( nalu_hypre_int comm );
NALU_HYPRE_Int nalu_hypre_MPI_Comm_size( nalu_hypre_MPI_Comm comm, NALU_HYPRE_Int *size );
NALU_HYPRE_Int nalu_hypre_MPI_Comm_rank( nalu_hypre_MPI_Comm comm, NALU_HYPRE_Int *rank );
NALU_HYPRE_Int nalu_hypre_MPI_Comm_free( nalu_hypre_MPI_Comm *comm );
NALU_HYPRE_Int nalu_hypre_MPI_Comm_group( nalu_hypre_MPI_Comm comm, nalu_hypre_MPI_Group *group );
NALU_HYPRE_Int nalu_hypre_MPI_Comm_split( nalu_hypre_MPI_Comm comm, NALU_HYPRE_Int n, NALU_HYPRE_Int m,
                                nalu_hypre_MPI_Comm * comms );
NALU_HYPRE_Int nalu_hypre_MPI_Group_incl( nalu_hypre_MPI_Group group, NALU_HYPRE_Int n, NALU_HYPRE_Int *ranks,
                                nalu_hypre_MPI_Group *newgroup );
NALU_HYPRE_Int nalu_hypre_MPI_Group_free( nalu_hypre_MPI_Group *group );
NALU_HYPRE_Int nalu_hypre_MPI_Address( void *location, nalu_hypre_MPI_Aint *address );
NALU_HYPRE_Int nalu_hypre_MPI_Get_count( nalu_hypre_MPI_Status *status, nalu_hypre_MPI_Datatype datatype,
                               NALU_HYPRE_Int *count );
NALU_HYPRE_Int nalu_hypre_MPI_Alltoall( void *sendbuf, NALU_HYPRE_Int sendcount, nalu_hypre_MPI_Datatype sendtype,
                              void *recvbuf, NALU_HYPRE_Int recvcount, nalu_hypre_MPI_Datatype recvtype, nalu_hypre_MPI_Comm comm );
NALU_HYPRE_Int nalu_hypre_MPI_Allgather( void *sendbuf, NALU_HYPRE_Int sendcount, nalu_hypre_MPI_Datatype sendtype,
                               void *recvbuf, NALU_HYPRE_Int recvcount, nalu_hypre_MPI_Datatype recvtype, nalu_hypre_MPI_Comm comm );
NALU_HYPRE_Int nalu_hypre_MPI_Allgatherv( void *sendbuf, NALU_HYPRE_Int sendcount, nalu_hypre_MPI_Datatype sendtype,
                                void *recvbuf, NALU_HYPRE_Int *recvcounts, NALU_HYPRE_Int *displs, nalu_hypre_MPI_Datatype recvtype,
                                nalu_hypre_MPI_Comm comm );
NALU_HYPRE_Int nalu_hypre_MPI_Gather( void *sendbuf, NALU_HYPRE_Int sendcount, nalu_hypre_MPI_Datatype sendtype,
                            void *recvbuf, NALU_HYPRE_Int recvcount, nalu_hypre_MPI_Datatype recvtype, NALU_HYPRE_Int root,
                            nalu_hypre_MPI_Comm comm );
NALU_HYPRE_Int nalu_hypre_MPI_Gatherv( void *sendbuf, NALU_HYPRE_Int sendcount, nalu_hypre_MPI_Datatype sendtype,
                             void *recvbuf, NALU_HYPRE_Int *recvcounts, NALU_HYPRE_Int *displs, nalu_hypre_MPI_Datatype recvtype,
                             NALU_HYPRE_Int root, nalu_hypre_MPI_Comm comm );
NALU_HYPRE_Int nalu_hypre_MPI_Scatter( void *sendbuf, NALU_HYPRE_Int sendcount, nalu_hypre_MPI_Datatype sendtype,
                             void *recvbuf, NALU_HYPRE_Int recvcount, nalu_hypre_MPI_Datatype recvtype, NALU_HYPRE_Int root,
                             nalu_hypre_MPI_Comm comm );
NALU_HYPRE_Int nalu_hypre_MPI_Scatterv( void *sendbuf, NALU_HYPRE_Int *sendcounts, NALU_HYPRE_Int *displs,
                              nalu_hypre_MPI_Datatype sendtype, void *recvbuf, NALU_HYPRE_Int recvcount, nalu_hypre_MPI_Datatype recvtype,
                              NALU_HYPRE_Int root, nalu_hypre_MPI_Comm comm );
NALU_HYPRE_Int nalu_hypre_MPI_Bcast( void *buffer, NALU_HYPRE_Int count, nalu_hypre_MPI_Datatype datatype,
                           NALU_HYPRE_Int root, nalu_hypre_MPI_Comm comm );
NALU_HYPRE_Int nalu_hypre_MPI_Send( void *buf, NALU_HYPRE_Int count, nalu_hypre_MPI_Datatype datatype, NALU_HYPRE_Int dest,
                          NALU_HYPRE_Int tag, nalu_hypre_MPI_Comm comm );
NALU_HYPRE_Int nalu_hypre_MPI_Recv( void *buf, NALU_HYPRE_Int count, nalu_hypre_MPI_Datatype datatype, NALU_HYPRE_Int source,
                          NALU_HYPRE_Int tag, nalu_hypre_MPI_Comm comm, nalu_hypre_MPI_Status *status );
NALU_HYPRE_Int nalu_hypre_MPI_Isend( void *buf, NALU_HYPRE_Int count, nalu_hypre_MPI_Datatype datatype, NALU_HYPRE_Int dest,
                           NALU_HYPRE_Int tag, nalu_hypre_MPI_Comm comm, nalu_hypre_MPI_Request *request );
NALU_HYPRE_Int nalu_hypre_MPI_Irecv( void *buf, NALU_HYPRE_Int count, nalu_hypre_MPI_Datatype datatype,
                           NALU_HYPRE_Int source, NALU_HYPRE_Int tag, nalu_hypre_MPI_Comm comm, nalu_hypre_MPI_Request *request );
NALU_HYPRE_Int nalu_hypre_MPI_Send_init( void *buf, NALU_HYPRE_Int count, nalu_hypre_MPI_Datatype datatype,
                               NALU_HYPRE_Int dest, NALU_HYPRE_Int tag, nalu_hypre_MPI_Comm comm, nalu_hypre_MPI_Request *request );
NALU_HYPRE_Int nalu_hypre_MPI_Recv_init( void *buf, NALU_HYPRE_Int count, nalu_hypre_MPI_Datatype datatype,
                               NALU_HYPRE_Int dest, NALU_HYPRE_Int tag, nalu_hypre_MPI_Comm comm, nalu_hypre_MPI_Request *request );
NALU_HYPRE_Int nalu_hypre_MPI_Irsend( void *buf, NALU_HYPRE_Int count, nalu_hypre_MPI_Datatype datatype, NALU_HYPRE_Int dest,
                            NALU_HYPRE_Int tag, nalu_hypre_MPI_Comm comm, nalu_hypre_MPI_Request *request );
NALU_HYPRE_Int nalu_hypre_MPI_Startall( NALU_HYPRE_Int count, nalu_hypre_MPI_Request *array_of_requests );
NALU_HYPRE_Int nalu_hypre_MPI_Probe( NALU_HYPRE_Int source, NALU_HYPRE_Int tag, nalu_hypre_MPI_Comm comm,
                           nalu_hypre_MPI_Status *status );
NALU_HYPRE_Int nalu_hypre_MPI_Iprobe( NALU_HYPRE_Int source, NALU_HYPRE_Int tag, nalu_hypre_MPI_Comm comm, NALU_HYPRE_Int *flag,
                            nalu_hypre_MPI_Status *status );
NALU_HYPRE_Int nalu_hypre_MPI_Test( nalu_hypre_MPI_Request *request, NALU_HYPRE_Int *flag, nalu_hypre_MPI_Status *status );
NALU_HYPRE_Int nalu_hypre_MPI_Testall( NALU_HYPRE_Int count, nalu_hypre_MPI_Request *array_of_requests, NALU_HYPRE_Int *flag,
                             nalu_hypre_MPI_Status *array_of_statuses );
NALU_HYPRE_Int nalu_hypre_MPI_Wait( nalu_hypre_MPI_Request *request, nalu_hypre_MPI_Status *status );
NALU_HYPRE_Int nalu_hypre_MPI_Waitall( NALU_HYPRE_Int count, nalu_hypre_MPI_Request *array_of_requests,
                             nalu_hypre_MPI_Status *array_of_statuses );
NALU_HYPRE_Int nalu_hypre_MPI_Waitany( NALU_HYPRE_Int count, nalu_hypre_MPI_Request *array_of_requests,
                             NALU_HYPRE_Int *index, nalu_hypre_MPI_Status *status );
NALU_HYPRE_Int nalu_hypre_MPI_Allreduce( void *sendbuf, void *recvbuf, NALU_HYPRE_Int count,
                               nalu_hypre_MPI_Datatype datatype, nalu_hypre_MPI_Op op, nalu_hypre_MPI_Comm comm );
NALU_HYPRE_Int nalu_hypre_MPI_Reduce( void *sendbuf, void *recvbuf, NALU_HYPRE_Int count,
                            nalu_hypre_MPI_Datatype datatype, nalu_hypre_MPI_Op op, NALU_HYPRE_Int root, nalu_hypre_MPI_Comm comm );
NALU_HYPRE_Int nalu_hypre_MPI_Scan( void *sendbuf, void *recvbuf, NALU_HYPRE_Int count,
                          nalu_hypre_MPI_Datatype datatype, nalu_hypre_MPI_Op op, nalu_hypre_MPI_Comm comm );
NALU_HYPRE_Int nalu_hypre_MPI_Request_free( nalu_hypre_MPI_Request *request );
NALU_HYPRE_Int nalu_hypre_MPI_Type_contiguous( NALU_HYPRE_Int count, nalu_hypre_MPI_Datatype oldtype,
                                     nalu_hypre_MPI_Datatype *newtype );
NALU_HYPRE_Int nalu_hypre_MPI_Type_vector( NALU_HYPRE_Int count, NALU_HYPRE_Int blocklength, NALU_HYPRE_Int stride,
                                 nalu_hypre_MPI_Datatype oldtype, nalu_hypre_MPI_Datatype *newtype );
NALU_HYPRE_Int nalu_hypre_MPI_Type_hvector( NALU_HYPRE_Int count, NALU_HYPRE_Int blocklength, nalu_hypre_MPI_Aint stride,
                                  nalu_hypre_MPI_Datatype oldtype, nalu_hypre_MPI_Datatype *newtype );
NALU_HYPRE_Int nalu_hypre_MPI_Type_struct( NALU_HYPRE_Int count, NALU_HYPRE_Int *array_of_blocklengths,
                                 nalu_hypre_MPI_Aint *array_of_displacements, nalu_hypre_MPI_Datatype *array_of_types,
                                 nalu_hypre_MPI_Datatype *newtype );
NALU_HYPRE_Int nalu_hypre_MPI_Type_commit( nalu_hypre_MPI_Datatype *datatype );
NALU_HYPRE_Int nalu_hypre_MPI_Type_free( nalu_hypre_MPI_Datatype *datatype );
NALU_HYPRE_Int nalu_hypre_MPI_Op_free( nalu_hypre_MPI_Op *op );
NALU_HYPRE_Int nalu_hypre_MPI_Op_create( nalu_hypre_MPI_User_function *function, nalu_hypre_int commute,
                               nalu_hypre_MPI_Op *op );
#if defined(NALU_HYPRE_USING_GPU)
NALU_HYPRE_Int nalu_hypre_MPI_Comm_split_type(nalu_hypre_MPI_Comm comm, NALU_HYPRE_Int split_type, NALU_HYPRE_Int key,
                                    nalu_hypre_MPI_Info info, nalu_hypre_MPI_Comm *newcomm);
NALU_HYPRE_Int nalu_hypre_MPI_Info_create(nalu_hypre_MPI_Info *info);
NALU_HYPRE_Int nalu_hypre_MPI_Info_free( nalu_hypre_MPI_Info *info );
#endif

#ifdef __cplusplus
}
#endif

#endif
