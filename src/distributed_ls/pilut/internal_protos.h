/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/* NALU_HYPRE_DistributedMatrixPilutSolver.c */
NALU_HYPRE_Int NALU_HYPRE_NewDistributedMatrixPilutSolver( MPI_Comm comm , NALU_HYPRE_DistributedMatrix matrix , NALU_HYPRE_DistributedMatrixPilutSolver *new_solver );
NALU_HYPRE_Int NALU_HYPRE_FreeDistributedMatrixPilutSolver( NALU_HYPRE_DistributedMatrixPilutSolver in_ptr );
NALU_HYPRE_Int NALU_HYPRE_DistributedMatrixPilutSolverInitialize( NALU_HYPRE_DistributedMatrixPilutSolver solver );
NALU_HYPRE_Int NALU_HYPRE_DistributedMatrixPilutSolverSetMatrix( NALU_HYPRE_DistributedMatrixPilutSolver in_ptr , NALU_HYPRE_DistributedMatrix matrix );
NALU_HYPRE_DistributedMatrix NALU_HYPRE_DistributedMatrixPilutSolverGetMatrix( NALU_HYPRE_DistributedMatrixPilutSolver in_ptr );
NALU_HYPRE_Int NALU_HYPRE_DistributedMatrixPilutSolverSetNumLocalRow( NALU_HYPRE_DistributedMatrixPilutSolver in_ptr , NALU_HYPRE_Int FirstLocalRow );
NALU_HYPRE_Int NALU_HYPRE_DistributedMatrixPilutSolverSetFactorRowSize( NALU_HYPRE_DistributedMatrixPilutSolver in_ptr , NALU_HYPRE_Int size );
NALU_HYPRE_Int NALU_HYPRE_DistributedMatrixPilutSolverSetDropTolerance( NALU_HYPRE_DistributedMatrixPilutSolver in_ptr , NALU_HYPRE_Real tolerance );
NALU_HYPRE_Int NALU_HYPRE_DistributedMatrixPilutSolverSetMaxIts( NALU_HYPRE_DistributedMatrixPilutSolver in_ptr , NALU_HYPRE_Int its );
NALU_HYPRE_Int NALU_HYPRE_DistributedMatrixPilutSolverSetup( NALU_HYPRE_DistributedMatrixPilutSolver in_ptr );
NALU_HYPRE_Int NALU_HYPRE_DistributedMatrixPilutSolverSolve( NALU_HYPRE_DistributedMatrixPilutSolver in_ptr , NALU_HYPRE_Real *x , NALU_HYPRE_Real *b );

/* comm.c */
NALU_HYPRE_Int nalu_hypre_GlobalSEMax( NALU_HYPRE_Int value , MPI_Comm nalu_hypre_MPI_Context );
NALU_HYPRE_Int nalu_hypre_GlobalSEMin( NALU_HYPRE_Int value , MPI_Comm nalu_hypre_MPI_Context );
NALU_HYPRE_Int nalu_hypre_GlobalSESum( NALU_HYPRE_Int value , MPI_Comm nalu_hypre_MPI_Context );
NALU_HYPRE_Real nalu_hypre_GlobalSEMaxDouble( NALU_HYPRE_Real value , MPI_Comm nalu_hypre_MPI_Context );
NALU_HYPRE_Real nalu_hypre_GlobalSEMinDouble( NALU_HYPRE_Real value , MPI_Comm nalu_hypre_MPI_Context );
NALU_HYPRE_Real nalu_hypre_GlobalSESumDouble( NALU_HYPRE_Real value , MPI_Comm nalu_hypre_MPI_Context );

/* debug.c */
void nalu_hypre_PrintLine( const char *str , nalu_hypre_PilutSolverGlobals *globals );
void nalu_hypre_CheckBounds( NALU_HYPRE_Int low , NALU_HYPRE_Int i , NALU_HYPRE_Int up , nalu_hypre_PilutSolverGlobals *globals );
nalu_hypre_longint nalu_hypre_IDX_Checksum( const NALU_HYPRE_Int *v , NALU_HYPRE_Int len , const char *msg , NALU_HYPRE_Int tag , nalu_hypre_PilutSolverGlobals *globals );
nalu_hypre_longint nalu_hypre_INT_Checksum( const NALU_HYPRE_Int *v , NALU_HYPRE_Int len , const char *msg , NALU_HYPRE_Int tag , nalu_hypre_PilutSolverGlobals *globals );
nalu_hypre_longint nalu_hypre_FP_Checksum( const NALU_HYPRE_Real *v , NALU_HYPRE_Int len , const char *msg , NALU_HYPRE_Int tag , nalu_hypre_PilutSolverGlobals *globals );
nalu_hypre_longint nalu_hypre_RMat_Checksum( const ReduceMatType *rmat , nalu_hypre_PilutSolverGlobals *globals );
nalu_hypre_longint nalu_hypre_LDU_Checksum( const FactorMatType *ldu , nalu_hypre_PilutSolverGlobals *globals );
void nalu_hypre_PrintVector( NALU_HYPRE_Int *v , NALU_HYPRE_Int n , char *msg , nalu_hypre_PilutSolverGlobals *globals );

/* hypre.c */

/* ilut.c */
NALU_HYPRE_Int nalu_hypre_ILUT( DataDistType *ddist , NALU_HYPRE_DistributedMatrix matrix , FactorMatType *ldu , NALU_HYPRE_Int maxnz , NALU_HYPRE_Real tol , nalu_hypre_PilutSolverGlobals *globals );
void nalu_hypre_ComputeAdd2Nrms( NALU_HYPRE_Int num_rows , NALU_HYPRE_Int *rowptr , NALU_HYPRE_Real *values , NALU_HYPRE_Real *nrm2s );

/* parilut.c */
void nalu_hypre_ParILUT( DataDistType *ddist , FactorMatType *ldu , ReduceMatType *rmat , NALU_HYPRE_Int gmaxnz , NALU_HYPRE_Real tol , nalu_hypre_PilutSolverGlobals *globals );
void nalu_hypre_ComputeCommInfo( ReduceMatType *rmat , CommInfoType *cinfo , NALU_HYPRE_Int *rowdist , nalu_hypre_PilutSolverGlobals *globals );
NALU_HYPRE_Int nalu_hypre_Idx2PE( NALU_HYPRE_Int idx , nalu_hypre_PilutSolverGlobals *globals );
NALU_HYPRE_Int nalu_hypre_SelectSet( ReduceMatType *rmat , CommInfoType *cinfo , NALU_HYPRE_Int *perm , NALU_HYPRE_Int *iperm , NALU_HYPRE_Int *newperm , NALU_HYPRE_Int *newiperm , nalu_hypre_PilutSolverGlobals *globals );
void nalu_hypre_SendFactoredRows( FactorMatType *ldu , CommInfoType *cinfo , NALU_HYPRE_Int *newperm , NALU_HYPRE_Int nmis , nalu_hypre_PilutSolverGlobals *globals );
void nalu_hypre_ComputeRmat( FactorMatType *ldu , ReduceMatType *rmat , ReduceMatType *nrmat , CommInfoType *cinfo , NALU_HYPRE_Int *perm , NALU_HYPRE_Int *iperm , NALU_HYPRE_Int *newperm , NALU_HYPRE_Int *newiperm , NALU_HYPRE_Int nmis , NALU_HYPRE_Real tol , nalu_hypre_PilutSolverGlobals *globals );
void nalu_hypre_FactorLocal( FactorMatType *ldu , ReduceMatType *rmat , ReduceMatType *nrmat , CommInfoType *cinfo , NALU_HYPRE_Int *perm , NALU_HYPRE_Int *iperm , NALU_HYPRE_Int *newperm , NALU_HYPRE_Int *newiperm , NALU_HYPRE_Int nmis , NALU_HYPRE_Real tol , nalu_hypre_PilutSolverGlobals *globals );
void nalu_hypre_SecondDropSmall( NALU_HYPRE_Real rtol , nalu_hypre_PilutSolverGlobals *globals );
NALU_HYPRE_Int nalu_hypre_SeperateLU_byDIAG( NALU_HYPRE_Int diag , NALU_HYPRE_Int *newiperm , nalu_hypre_PilutSolverGlobals *globals );
NALU_HYPRE_Int nalu_hypre_SeperateLU_byMIS( nalu_hypre_PilutSolverGlobals *globals );
void nalu_hypre_UpdateL( NALU_HYPRE_Int lrow , NALU_HYPRE_Int last , FactorMatType *ldu , nalu_hypre_PilutSolverGlobals *globals );
void nalu_hypre_FormNRmat( NALU_HYPRE_Int rrow , NALU_HYPRE_Int first , ReduceMatType *nrmat , NALU_HYPRE_Int max_rowlen , NALU_HYPRE_Int in_rowlen , NALU_HYPRE_Int *in_colind , NALU_HYPRE_Real *in_values , nalu_hypre_PilutSolverGlobals *globals );
void nalu_hypre_FormDU( NALU_HYPRE_Int lrow , NALU_HYPRE_Int first , FactorMatType *ldu , NALU_HYPRE_Int *rcolind , NALU_HYPRE_Real *rvalues , NALU_HYPRE_Real tol , nalu_hypre_PilutSolverGlobals *globals );
void nalu_hypre_EraseMap( CommInfoType *cinfo , NALU_HYPRE_Int *newperm , NALU_HYPRE_Int nmis , nalu_hypre_PilutSolverGlobals *globals );
void nalu_hypre_ParINIT( ReduceMatType *nrmat , CommInfoType *cinfo , NALU_HYPRE_Int *rowdist , nalu_hypre_PilutSolverGlobals *globals );

/* parutil.c */
void nalu_hypre_errexit(const char *f_str , ...);
void nalu_hypre_my_abort( NALU_HYPRE_Int inSignal , nalu_hypre_PilutSolverGlobals *globals );
NALU_HYPRE_Int *nalu_hypre_idx_malloc( NALU_HYPRE_Int n ,const char *msg );
NALU_HYPRE_Int *nalu_hypre_idx_malloc_init( NALU_HYPRE_Int n , NALU_HYPRE_Int ival ,const char *msg );
NALU_HYPRE_Real *nalu_hypre_fp_malloc( NALU_HYPRE_Int n ,const char *msg );
NALU_HYPRE_Real *nalu_hypre_fp_malloc_init( NALU_HYPRE_Int n , NALU_HYPRE_Real ival ,const char *msg );
void *nalu_hypre_mymalloc( NALU_HYPRE_Int nbytes ,const char *msg );
void nalu_hypre_free_multi( void *ptr1 , ...);
void nalu_hypre_memcpy_int( NALU_HYPRE_Int *dest , const NALU_HYPRE_Int *src , size_t n );
void nalu_hypre_memcpy_idx( NALU_HYPRE_Int *dest , const NALU_HYPRE_Int *src , size_t n );
void nalu_hypre_memcpy_fp( NALU_HYPRE_Real *dest , const NALU_HYPRE_Real *src , size_t n );

/* pblas1.c */
NALU_HYPRE_Real nalu_hypre_p_dnrm2( DataDistType *ddist , NALU_HYPRE_Real *x , nalu_hypre_PilutSolverGlobals *globals );
NALU_HYPRE_Real nalu_hypre_p_ddot( DataDistType *ddist , NALU_HYPRE_Real *x , NALU_HYPRE_Real *y , nalu_hypre_PilutSolverGlobals *globals );
void nalu_hypre_p_daxy( DataDistType *ddist , NALU_HYPRE_Real alpha , NALU_HYPRE_Real *x , NALU_HYPRE_Real *y );
void nalu_hypre_p_daxpy( DataDistType *ddist , NALU_HYPRE_Real alpha , NALU_HYPRE_Real *x , NALU_HYPRE_Real *y );
void nalu_hypre_p_daxbyz( DataDistType *ddist , NALU_HYPRE_Real alpha , NALU_HYPRE_Real *x , NALU_HYPRE_Real beta , NALU_HYPRE_Real *y , NALU_HYPRE_Real *z );
NALU_HYPRE_Int nalu_hypre_p_vprintf( DataDistType *ddist , NALU_HYPRE_Real *x , nalu_hypre_PilutSolverGlobals *globals );

/* distributed_qsort.c */
void nalu_hypre_tex_qsort(char *base, NALU_HYPRE_Int n, NALU_HYPRE_Int size, NALU_HYPRE_Int (*compar) (char*,char*));
/* distributed_qsort_si.c */
void nalu_hypre_sincsort_fast( NALU_HYPRE_Int n , NALU_HYPRE_Int *base );
void nalu_hypre_sdecsort_fast( NALU_HYPRE_Int n , NALU_HYPRE_Int *base );

/* serilut.c */
NALU_HYPRE_Int nalu_hypre_SerILUT( DataDistType *ddist , NALU_HYPRE_DistributedMatrix matrix , FactorMatType *ldu , ReduceMatType *rmat , NALU_HYPRE_Int maxnz , NALU_HYPRE_Real tol , nalu_hypre_PilutSolverGlobals *globals );
NALU_HYPRE_Int nalu_hypre_SelectInterior( NALU_HYPRE_Int local_num_rows , NALU_HYPRE_DistributedMatrix matrix , NALU_HYPRE_Int *external_rows , NALU_HYPRE_Int *newperm , NALU_HYPRE_Int *newiperm , nalu_hypre_PilutSolverGlobals *globals );
NALU_HYPRE_Int nalu_hypre_FindStructuralUnion( NALU_HYPRE_DistributedMatrix matrix , NALU_HYPRE_Int **structural_union , nalu_hypre_PilutSolverGlobals *globals );
NALU_HYPRE_Int nalu_hypre_ExchangeStructuralUnions( DataDistType *ddist , NALU_HYPRE_Int **structural_union , nalu_hypre_PilutSolverGlobals *globals );
void nalu_hypre_SecondDrop( NALU_HYPRE_Int maxnz , NALU_HYPRE_Real tol , NALU_HYPRE_Int row , NALU_HYPRE_Int *perm , NALU_HYPRE_Int *iperm , FactorMatType *ldu , nalu_hypre_PilutSolverGlobals *globals );
void nalu_hypre_SecondDropUpdate( NALU_HYPRE_Int maxnz , NALU_HYPRE_Int maxnzkeep , NALU_HYPRE_Real tol , NALU_HYPRE_Int row , NALU_HYPRE_Int nlocal , NALU_HYPRE_Int *perm , NALU_HYPRE_Int *iperm , FactorMatType *ldu , ReduceMatType *rmat , nalu_hypre_PilutSolverGlobals *globals );

/* trifactor.c */
void nalu_hypre_LDUSolve( DataDistType *ddist , FactorMatType *ldu , NALU_HYPRE_Real *x , NALU_HYPRE_Real *b , nalu_hypre_PilutSolverGlobals *globals );
NALU_HYPRE_Int nalu_hypre_SetUpLUFactor( DataDistType *ddist , FactorMatType *ldu , NALU_HYPRE_Int maxnz , nalu_hypre_PilutSolverGlobals *globals );
void nalu_hypre_SetUpFactor( DataDistType *ddist , FactorMatType *ldu , NALU_HYPRE_Int maxnz , NALU_HYPRE_Int *petotal , NALU_HYPRE_Int *rind , NALU_HYPRE_Int *imap , NALU_HYPRE_Int *maxsendP , NALU_HYPRE_Int DoingL , nalu_hypre_PilutSolverGlobals *globals );

/* util.c */
NALU_HYPRE_Int nalu_hypre_ExtractMinLR( nalu_hypre_PilutSolverGlobals *globals );
void nalu_hypre_IdxIncSort( NALU_HYPRE_Int n , NALU_HYPRE_Int *idx , NALU_HYPRE_Real *val );
void nalu_hypre_ValDecSort( NALU_HYPRE_Int n , NALU_HYPRE_Int *idx , NALU_HYPRE_Real *val );
NALU_HYPRE_Int nalu_hypre_CompactIdx( NALU_HYPRE_Int n , NALU_HYPRE_Int *idx , NALU_HYPRE_Real *val );
void nalu_hypre_PrintIdxVal( NALU_HYPRE_Int n , NALU_HYPRE_Int *idx , NALU_HYPRE_Real *val );
NALU_HYPRE_Int nalu_hypre_DecKeyValueCmp( const void *v1 , const void *v2 );
void nalu_hypre_SortKeyValueNodesDec( KeyValueType *nodes , NALU_HYPRE_Int n );
NALU_HYPRE_Int nalu_hypre_sasum( NALU_HYPRE_Int n , NALU_HYPRE_Int *x );
void nalu_hypre_sincsort( NALU_HYPRE_Int n , NALU_HYPRE_Int *a );
void nalu_hypre_sdecsort( NALU_HYPRE_Int n , NALU_HYPRE_Int *a );

