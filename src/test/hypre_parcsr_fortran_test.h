/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#ifdef __cplusplus
extern "C" {
#endif

/******************************************************************************
 * Definitions of ParCSR Fortran interface routines
 *****************************************************************************/

#define NALU_HYPRE_ParCSRMatrixCreate  \
        hypre_F90_NAME(fhypre_parcsrmatrixcreate, FNALU_HYPRE_PARCSRMATRIXCREATE)
extern void hypre_F90_NAME(fhypre_parcsrmatrixcreate, FNALU_HYPRE_PARCSRMATRIXCREATE)
(NALU_HYPRE_Int *, NALU_HYPRE_Int *, NALU_HYPRE_Int *, NALU_HYPRE_Int *, NALU_HYPRE_Int *, NALU_HYPRE_Int *, NALU_HYPRE_Int *,
 NALU_HYPRE_Int *, hypre_F90_Obj *);

#define NALU_HYPRE_ParCSRMatrixDestroy  \
        hypre_F90_NAME(fhypre_parcsrmatrixdestroy, FNALU_HYPRE_PARCSRMATRIXDESTROY)
extern void hypre_F90_NAME(fhypre_parcsrmatrixdestroy, FNALU_HYPRE_PARCSRMATRIXDESTROY)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_ParCSRMatrixInitialize  \
        hypre_F90_NAME(fhypre_parcsrmatrixinitialize, FNALU_HYPRE_PARCSRMATRIXINITIALIZE)
extern void hypre_F90_NAME(fhypre_parcsrmatrixinitialize, FNALU_HYPRE_PARCSRMATRIXINITIALIZE)
(hypre_F90_Obj *);

#define NALU_HYPRE_ParCSRMatrixRead  \
        hypre_F90_NAME(fhypre_parcsrmatrixread, FNALU_HYPRE_PARCSRMATRIXREAD)
extern void hypre_F90_NAME(fhypre_parcsrmatrixread, FNALU_HYPRE_PARCSRMATRIXREAD)
(NALU_HYPRE_Int *, char *, hypre_F90_Obj *);

#define NALU_HYPRE_ParCSRMatrixPrint  \
        hypre_F90_NAME(fhypre_parcsrmatrixprint, FNALU_HYPRE_PARCSRMATRIXPRINT)
extern void hypre_F90_NAME(fhypre_parcsrmatrixprint, FNALU_HYPRE_PARCSRMATRIXPRINT)
(hypre_F90_Obj *, char *, NALU_HYPRE_Int *);

#define NALU_HYPRE_ParCSRMatrixGetComm  \
        hypre_F90_NAME(fhypre_parcsrmatrixgetcomm, FNALU_HYPRE_PARCSRMATRIXGETCOMM)
extern void hypre_F90_NAME(fhypre_parcsrmatrixgetcomm, FNALU_HYPRE_PARCSRMATRIXGETCOMM)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_ParCSRMatrixGetDims  \
        hypre_F90_NAME(fhypre_parcsrmatrixgetdims, FNALU_HYPRE_PARCSRMATRIXGETDIMS)
extern void hypre_F90_NAME(fhypre_parcsrmatrixgetdims, FNALU_HYPRE_PARCSRMATRIXGETDIMS)
(hypre_F90_Obj *, NALU_HYPRE_Int *, NALU_HYPRE_Int *);

#define NALU_HYPRE_ParCSRMatrixGetRowPartitioning  \
        hypre_F90_NAME(fhypre_parcsrmatrixgetrowpartit, FNALU_HYPRE_PARCSRMATRIXGETROWPARTIT)
extern void hypre_F90_NAME(fhypre_parcsrmatrixgetrowpartit, FNALU_HYPRE_PARCSRMATRIXGETROWPARTIT)
(hypre_F90_Obj *, hypre_F90_Obj *);

#define NALU_HYPRE_ParCSRMatrixGetColPartitioning  \
        hypre_F90_NAME(fhypre_parcsrmatrixgetcolpartit, FNALU_HYPRE_PARCSRMATRIXGETCOLPARTIT)
extern void hypre_F90_NAME(fhypre_parcsrmatrixgetcolpartit, FNALU_HYPRE_PARCSRMATRIXGETCOLPARTIT)
(hypre_F90_Obj *, hypre_F90_Obj *);

#define NALU_HYPRE_ParCSRMatrixGetLocalRange  \
        hypre_F90_NAME(fhypre_parcsrmatrixgetlocalrang, FNALU_HYPRE_PARCSRMATRIXGETLOCALRANG)
extern void hypre_F90_NAME(fhypre_parcsrmatrixgetlocalrang, FNALU_HYPRE_PARCSRMATRIXGETLOCALRANG)
(hypre_F90_Obj *, NALU_HYPRE_Int *, NALU_HYPRE_Int *, NALU_HYPRE_Int *, NALU_HYPRE_Int *);

#define NALU_HYPRE_ParCSRMatrixGetRow  \
        hypre_F90_NAME(fhypre_parcsrmatrixgetrow, FNALU_HYPRE_PARCSRMATRIXGETROW)
extern void hypre_F90_NAME(fhypre_parcsrmatrixgetrow, FNALU_HYPRE_PARCSRMATRIXGETROW)
(hypre_F90_Obj *, NALU_HYPRE_Int *, NALU_HYPRE_Int *, hypre_F90_Obj *, hypre_F90_Obj *);

#define NALU_HYPRE_ParCSRMatrixRestoreRow  \
        hypre_F90_NAME(fhypre_parcsrmatrixrestorerow, FNALU_HYPRE_PARCSRMATRIXRESTOREROW)
extern void hypre_F90_NAME(fhypre_parcsrmatrixrestorerow, FNALU_HYPRE_PARCSRMATRIXRESTOREROW)
(hypre_F90_Obj *, NALU_HYPRE_Int *, NALU_HYPRE_Int *, hypre_F90_Obj *, hypre_F90_Obj *);

#define NALU_HYPRE_CSRMatrixtoParCSRMatrix  \
        hypre_F90_NAME(fhypre_csrmatrixtoparcsrmatrix, FNALU_HYPRE_CSRMATRIXTOPARCSRMATRIX)
extern void hypre_F90_NAME(fhypre_csrmatrixtoparcsrmatrix, FNALU_HYPRE_CSRMATRIXTOPARCSRMATRIX)
(NALU_HYPRE_Int *, hypre_F90_Obj *, NALU_HYPRE_Int *, NALU_HYPRE_Int *, hypre_F90_Obj *);

#define NALU_HYPRE_ParCSRMatrixMatvec  \
        hypre_F90_NAME(fhypre_parcsrmatrixmatvec, FNALU_HYPRE_PARCSRMATRIXMATVEC)
extern void hypre_F90_NAME(fhypre_parcsrmatrixmatvec, FNALU_HYPRE_PARCSRMATRIXMATVEC)
(NALU_HYPRE_Real *, hypre_F90_Obj *, hypre_F90_Obj *, NALU_HYPRE_Real *, hypre_F90_Obj *);

#define NALU_HYPRE_ParCSRMatrixMatvecT  \
        hypre_F90_NAME(fhypre_parcsrmatrixmatvect, FNALU_HYPRE_PARCSRMATRIXMATVECT)
extern void hypre_F90_NAME(fhypre_parcsrmatrixmatvect, FNALU_HYPRE_PARCSRMATRIXMATVECT)
(NALU_HYPRE_Real *, hypre_F90_Obj *, hypre_F90_Obj *, NALU_HYPRE_Real *, hypre_F90_Obj *);



#define NALU_HYPRE_ParVectorCreate  \
        hypre_F90_NAME(fhypre_parvectorcreate, FNALU_HYPRE_PARVECTORCREATE)
extern void hypre_F90_NAME(fhypre_parvectorcreate, FNALU_HYPRE_PARVECTORCREATE)
(NALU_HYPRE_Int *, NALU_HYPRE_Int *, hypre_F90_Obj *, hypre_F90_Obj *);

#define NALU_HYPRE_ParMultiVectorCreate  \
        hypre_F90_NAME(fhypre_parmultivectorcreate, FNALU_HYPRE_PARMULTIVECTORCREATE)
extern void hypre_F90_NAME(fhypre_parmultivectorcreate, FNALU_HYPRE_PARMULTIVECTORCREATE)
(NALU_HYPRE_Int *, NALU_HYPRE_Int *, hypre_F90_Obj *, NALU_HYPRE_Int *, hypre_F90_Obj *);

#define NALU_HYPRE_ParVectorDestroy  \
        hypre_F90_NAME(fhypre_parvectordestroy, FNALU_HYPRE_PARVECTORDESTROY)
extern void hypre_F90_NAME(fhypre_parvectordestroy, FNALU_HYPRE_PARVECTORDESTROY)
(hypre_F90_Obj *);

#define NALU_HYPRE_ParVectorInitialize  \
        hypre_F90_NAME(fhypre_parvectorinitialize, FNALU_HYPRE_PARVECTORINITIALIZE)
extern void hypre_F90_NAME(fhypre_parvectorinitialize, FNALU_HYPRE_PARVECTORINITIALIZE)
(hypre_F90_Obj *);

#define NALU_HYPRE_ParVectorRead  \
        hypre_F90_NAME(fhypre_parvectorread, FNALU_HYPRE_PARVECTORREAD)
extern void hypre_F90_NAME(fhypre_parvectorread, FNALU_HYPRE_PARVECTORREAD)
(NALU_HYPRE_Int *, hypre_F90_Obj *, char *);

#define NALU_HYPRE_ParVectorPrint  \
        hypre_F90_NAME(fhypre_parvectorprint, FNALU_HYPRE_PARVECTORPRINT)
extern void hypre_F90_NAME(fhypre_parvectorprint, FNALU_HYPRE_PARVECTORPRINT)
(hypre_F90_Obj *, char *, NALU_HYPRE_Int *);

#define NALU_HYPRE_ParVectorSetConstantValues  \
        hypre_F90_NAME(fhypre_parvectorsetconstantvalu, FNALU_HYPRE_PARVECTORSETCONSTANTVALU)
extern void hypre_F90_NAME(fhypre_parvectorsetconstantvalu, FNALU_HYPRE_PARVECTORSETCONSTANTVALU)
(hypre_F90_Obj *, NALU_HYPRE_Real *);

#define NALU_HYPRE_ParVectorSetRandomValues  \
        hypre_F90_NAME(fhypre_parvectorsetrandomvalues, FNALU_HYPRE_PARVECTORSETRANDOMVALUES)
extern void hypre_F90_NAME(fhypre_parvectorsetrandomvalues, FNALU_HYPRE_PARVECTORSETRANDOMVALUES)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_ParVectorCopy  \
        hypre_F90_NAME(fhypre_parvectorcopy, FNALU_HYPRE_PARVECTORCOPY)
extern void hypre_F90_NAME(fhypre_parvectorcopy, FNALU_HYPRE_PARVECTORCOPY)
(hypre_F90_Obj *, hypre_F90_Obj *);

#define NALU_HYPRE_ParVectorCloneShallow  \
        hypre_F90_NAME(fhypre_parvectorcloneshallow, FNALU_HYPRE_PARVECTORCLONESHALLOW)
extern void hypre_F90_NAME(fhypre_parvectorcloneshallow, FNALU_HYPRE_PARVECTORCLONESHALLOW)
(hypre_F90_Obj *);

#define NALU_HYPRE_ParVectorScale  \
        hypre_F90_NAME(fhypre_parvectorscale, FNALU_HYPRE_PARVECTORSCALE)
extern void hypre_F90_NAME(fhypre_parvectorscale, FNALU_HYPRE_PARVECTORSCALE)
(hypre_F90_Obj *, NALU_HYPRE_Real *);

#define NALU_HYPRE_ParVectorAxpy  \
        hypre_F90_NAME(fhypre_parvectoraxpy, FNALU_HYPRE_PARVECTORAXPY)
extern void hypre_F90_NAME(fhypre_parvectoraxpy, FNALU_HYPRE_PARVECTORAXPY)
(NALU_HYPRE_Real *, hypre_F90_Obj *, hypre_F90_Obj *);

#define NALU_HYPRE_ParVectorInnerProd  \
        hypre_F90_NAME(fhypre_parvectorinnerprod, FNALU_HYPRE_PARVECTORINNERPROD)
extern void hypre_F90_NAME(fhypre_parvectorinnerprod, FNALU_HYPRE_PARVECTORINNERPROD)
(hypre_F90_Obj *, hypre_F90_Obj *, NALU_HYPRE_Real *);

#define hypre_ParCSRMatrixGlobalNumRows  \
        hypre_F90_NAME(fhypre_parcsrmatrixglobalnumrow, FNALU_HYPRE_PARCSRMATRIXGLOBALNUMROW)
extern void hypre_F90_NAME(fhypre_parcsrmatrixglobalnumrow, FNALU_HYPRE_PARCSRMATRIXGLOBALNUMROW)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define hypre_ParCSRMatrixRowStarts  \
        hypre_F90_NAME(fhypre_parcsrmatrixrowstarts, FNALU_HYPRE_PARCSRMATRIXROWSTARTS)
extern void hypre_F90_NAME(fhypre_parcsrmatrixrowstarts, FNALU_HYPRE_PARCSRMATRIXROWSTARTS)
(hypre_F90_Obj *, hypre_F90_Obj *);

#define hypre_ParVectorSetDataOwner  \
        hypre_F90_NAME(fhypre_setparvectordataowner, FNALU_HYPRE_SETPARVECTORDATAOWNER)
extern void hypre_F90_NAME(fhypre_setparvectordataowner, FNALU_HYPRE_SETPARVECTORDATAOWNER)
(hypre_F90_Obj *, NALU_HYPRE_Int *);



#define GenerateLaplacian  \
        hypre_F90_NAME(fgeneratelaplacian, FNALU_HYPRE_GENERATELAPLACIAN)
extern void hypre_F90_NAME(fgeneratelaplacian, FNALU_HYPRE_GENERATELAPLACIAN)
(NALU_HYPRE_Int *, NALU_HYPRE_Int *, NALU_HYPRE_Int *, NALU_HYPRE_Int *, NALU_HYPRE_Int *, NALU_HYPRE_Int *, NALU_HYPRE_Int *,
 NALU_HYPRE_Int *, NALU_HYPRE_Int *, NALU_HYPRE_Int *, NALU_HYPRE_Real *, hypre_F90_Obj *);



#define NALU_HYPRE_BoomerAMGCreate  \
        hypre_F90_NAME(fhypre_boomeramgcreate, FNALU_HYPRE_BOOMERAMGCREATE)
extern void hypre_F90_NAME(fhypre_boomeramgcreate, FNALU_HYPRE_BOOMERAMGCREATE)
(hypre_F90_Obj *);

#define NALU_HYPRE_BoomerAMGDestroy  \
        hypre_F90_NAME(fhypre_boomeramgdestroy, FNALU_HYPRE_BOOMERAMGDESTROY)
extern void hypre_F90_NAME(fhypre_boomeramgdestroy, FNALU_HYPRE_BOOMERAMGDESTROY)
(hypre_F90_Obj *);

#define NALU_HYPRE_BoomerAMGSetup  \
        hypre_F90_NAME(fhypre_boomeramgsetup, FNALU_HYPRE_BOOMERAMGSETUP)
extern void hypre_F90_NAME(fhypre_boomeramgsetup, FNALU_HYPRE_BOOMERAMGSETUP)
(hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *);

#define NALU_HYPRE_BoomerAMGSolve  \
        hypre_F90_NAME(fhypre_boomeramgsolve, FNALU_HYPRE_BOOMERAMGSOLVE)
extern void hypre_F90_NAME(fhypre_boomeramgsolve, FNALU_HYPRE_BOOMERAMGSOLVE)
(hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *);

#define NALU_HYPRE_BoomerAMGSolveT  \
        hypre_F90_NAME(fhypre_boomeramgsolvet, FNALU_HYPRE_BOOMERAMGSOLVET)
extern void hypre_F90_NAME(fhypre_boomeramgsolvet, FNALU_HYPRE_BOOMERAMGSOLVET)
(hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *);

#define NALU_HYPRE_BoomerAMGSetRestriction  \
        hypre_F90_NAME(fhypre_boomeramgsetrestriction, FNALU_HYPRE_BOOMERAMGSETRESTRICTION)
extern void hypre_F90_NAME(fhypre_boomeramgsetrestriction, FNALU_HYPRE_BOOMERAMGSETRESTRICTION)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_BoomerAMGSetMaxLevels  \
        hypre_F90_NAME(fhypre_boomeramgsetmaxlevels, FNALU_HYPRE_BOOMERAMGSETMAXLEVELS)
extern void hypre_F90_NAME(fhypre_boomeramgsetmaxlevels, FNALU_HYPRE_BOOMERAMGSETMAXLEVELS)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_BoomerAMGGetMaxLevels  \
        hypre_F90_NAME(fhypre_boomeramggetmaxlevels, FNALU_HYPRE_BOOMERAMGGETMAXLEVELS)
extern void hypre_F90_NAME(fhypre_boomeramggetmaxlevels, FNALU_HYPRE_BOOMERAMGGETMAXLEVELS)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_BoomerAMGSetCoarsenCutFactor  \
        hypre_F90_NAME(fhypre_boomeramgsetcoarsencutfa, FNALU_HYPRE_BOOMERAMGSETCOARSENCUTFAC)
extern void hypre_F90_NAME(fhypre_boomeramgsetcoarsencutfa, FNALU_HYPRE_BOOMERAMGSETCOARSENCUTFAC)
(hypre_F90_Obj *, NALU_HYPRE_Real *);

#define NALU_HYPRE_BoomerAMGGetCoarsenCutFactor  \
        hypre_F90_NAME(fhypre_boomeramggetcoarsencutfa, FNALU_HYPRE_BOOMERAMGGETCOARSENCUTFAC)
extern void hypre_F90_NAME(fhypre_boomeramggetcoarsencutfa, FNALU_HYPRE_BOOMERAMGGETCOARSENCUTFAC)
(hypre_F90_Obj *, NALU_HYPRE_Real *);

#define NALU_HYPRE_BoomerAMGSetStrongThreshold  \
        hypre_F90_NAME(fhypre_boomeramgsetstrongthrshl, FNALU_HYPRE_BOOMERAMGSETSTRONGTHRSHL)
extern void hypre_F90_NAME(fhypre_boomeramgsetstrongthrshl, FNALU_HYPRE_BOOMERAMGSETSTRONGTHRSHL)
(hypre_F90_Obj *, NALU_HYPRE_Real *);

#define NALU_HYPRE_BoomerAMGGetStrongThreshold  \
        hypre_F90_NAME(fhypre_boomeramggetstrongthrshl, FNALU_HYPRE_BOOMERAMGGETSTRONGTHRSHL)
extern void hypre_F90_NAME(fhypre_boomeramggetstrongthrshl, FNALU_HYPRE_BOOMERAMGGETSTRONGTHRSHL)
(hypre_F90_Obj *, NALU_HYPRE_Real *);

#define NALU_HYPRE_BoomerAMGSetMaxRowSum  \
        hypre_F90_NAME(fhypre_boomeramgsetmaxrowsum, FNALU_HYPRE_BOOMERAMGSETMAXROWSUM)
extern void hypre_F90_NAME(fhypre_boomeramgsetmaxrowsum, FNALU_HYPRE_BOOMERAMGSETMAXROWSUM)
(hypre_F90_Obj *, NALU_HYPRE_Real *);

#define NALU_HYPRE_BoomerAMGGetMaxRowSum  \
        hypre_F90_NAME(fhypre_boomeramggetmaxrowsum, FNALU_HYPRE_BOOMERAMGGETMAXROWSUM)
extern void hypre_F90_NAME(fhypre_boomeramggetmaxrowsum, FNALU_HYPRE_BOOMERAMGGETMAXROWSUM)
(hypre_F90_Obj *, NALU_HYPRE_Real *);

#define NALU_HYPRE_BoomerAMGSetTruncFactor  \
        hypre_F90_NAME(fhypre_boomeramgsettruncfactor, FNALU_HYPRE_BOOMERAMGSETTRUNCFACTOR)
extern void hypre_F90_NAME(fhypre_boomeramgsettruncfactor, FNALU_HYPRE_BOOMERAMGSETTRUNCFACTOR)
(hypre_F90_Obj *, NALU_HYPRE_Real *);

#define NALU_HYPRE_BoomerAMGGetTruncFactor  \
        hypre_F90_NAME(fhypre_boomeramggettruncfactor, FNALU_HYPRE_BOOMERAMGGETTRUNCFACTOR)
extern void hypre_F90_NAME(fhypre_boomeramggettruncfactor, FNALU_HYPRE_BOOMERAMGGETTRUNCFACTOR)
(hypre_F90_Obj *, NALU_HYPRE_Real *);

#define NALU_HYPRE_BoomerAMGSetSCommPkgSwitch  \
        hypre_F90_NAME(fhypre_boomeramgsetscommpkgswit, FNALU_HYPRE_BOOMERAMGSETSCOMMPKGSWIT)
extern void hypre_F90_NAME(fhypre_boomeramgsetscommpkgswit, FNALU_HYPRE_BOOMERAMGSETSCOMMPKGSWIT)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_BoomerAMGSetInterpType  \
        hypre_F90_NAME(fhypre_boomeramgsetinterptype, FNALU_HYPRE_BOOMERAMGSETINTERPTYPE)
extern void hypre_F90_NAME(fhypre_boomeramgsetinterptype, FNALU_HYPRE_BOOMERAMGSETINTERPTYPE)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_BoomerAMGSetMinIter  \
        hypre_F90_NAME(fhypre_boomeramgsetminiter, FNALU_HYPRE_BOOMERAMGSETMINITER)
extern void hypre_F90_NAME(fhypre_boomeramgsetminiter, FNALU_HYPRE_BOOMERAMGSETMINITER)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_BoomerAMGSetMaxIter  \
        hypre_F90_NAME(fhypre_boomeramgsetmaxiter, FNALU_HYPRE_BOOMERAMGSETMAXITER)
extern void hypre_F90_NAME(fhypre_boomeramgsetmaxiter, FNALU_HYPRE_BOOMERAMGSETMAXITER)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_BoomerAMGGetMaxIter  \
        hypre_F90_NAME(fhypre_boomeramggetmaxiter, FNALU_HYPRE_BOOMERAMGGETMAXITER)
extern void hypre_F90_NAME(fhypre_boomeramggetmaxiter, FNALU_HYPRE_BOOMERAMGGETMAXITER)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_BoomerAMGSetCoarsenType  \
        hypre_F90_NAME(fhypre_boomeramgsetcoarsentype, FNALU_HYPRE_BOOMERAMGSETCOARSENTYPE)
extern void hypre_F90_NAME(fhypre_boomeramgsetcoarsentype, FNALU_HYPRE_BOOMERAMGSETCOARSENTYPE)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_BoomerAMGGetCoarsenType  \
        hypre_F90_NAME(fhypre_boomeramggetcoarsentype, FNALU_HYPRE_BOOMERAMGGETCOARSENTYPE)
extern void hypre_F90_NAME(fhypre_boomeramggetcoarsentype, FNALU_HYPRE_BOOMERAMGGETCOARSENTYPE)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_BoomerAMGSetMeasureType  \
        hypre_F90_NAME(fhypre_boomeramgsetmeasuretype, FNALU_HYPRE_BOOMERAMGSETMEASURETYPE)
extern void hypre_F90_NAME(fhypre_boomeramgsetmeasuretype, FNALU_HYPRE_BOOMERAMGSETMEASURETYPE)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_BoomerAMGGetMeasureType  \
        hypre_F90_NAME(fhypre_boomeramggetmeasuretype, FNALU_HYPRE_BOOMERAMGGETMEASURETYPE)
extern void hypre_F90_NAME(fhypre_boomeramggetmeasuretype, FNALU_HYPRE_BOOMERAMGGETMEASURETYPE)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_BoomerAMGSetSetupType  \
        hypre_F90_NAME(fhypre_boomeramgsetsetuptype, FNALU_HYPRE_BOOMERAMGSETSETUPTYPE)
extern void hypre_F90_NAME(fhypre_boomeramgsetsetuptype, FNALU_HYPRE_BOOMERAMGSETSETUPTYPE)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_BoomerAMGSetCycleType  \
        hypre_F90_NAME(fhypre_boomeramgsetcycletype, FNALU_HYPRE_BOOMERAMGSETCYCLETYPE)
extern void hypre_F90_NAME(fhypre_boomeramgsetcycletype, FNALU_HYPRE_BOOMERAMGSETCYCLETYPE)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_BoomerAMGGetCycleType  \
        hypre_F90_NAME(fhypre_boomeramggetcycletype, FNALU_HYPRE_BOOMERAMGGETCYCLETYPE)
extern void hypre_F90_NAME(fhypre_boomeramggetcycletype, FNALU_HYPRE_BOOMERAMGGETCYCLETYPE)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_BoomerAMGSetTol  \
        hypre_F90_NAME(fhypre_boomeramgsettol, FNALU_HYPRE_BOOMERAMGSETTOL)
extern void hypre_F90_NAME(fhypre_boomeramgsettol, FNALU_HYPRE_BOOMERAMGSETTOL)
(hypre_F90_Obj *, NALU_HYPRE_Real *);

#define NALU_HYPRE_BoomerAMGGetTol  \
        hypre_F90_NAME(fhypre_boomeramggettol, FNALU_HYPRE_BOOMERAMGGETTOL)
extern void hypre_F90_NAME(fhypre_boomeramggettol, FNALU_HYPRE_BOOMERAMGGETTOL)
(hypre_F90_Obj *, NALU_HYPRE_Real *);

#define NALU_HYPRE_BoomerAMGSetNumSweeps  \
        hypre_F90_NAME(fhypre_boomeramgsetnumsweeps, FNALU_HYPRE_BOOMERAMGSETNUMSWEEPS)
extern void hypre_F90_NAME(fhypre_boomeramgsetnumsweeps, FNALU_HYPRE_BOOMERAMGSETNUMSWEEPS)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_BoomerAMGSetCycleNumSweeps  \
        hypre_F90_NAME(fhypre_boomeramgsetcyclenumswee, FNALU_HYPRE_BOOMERAMGSETCYCLENUMSWEE)
extern void hypre_F90_NAME(fhypre_boomeramgsetcyclenumswee, FNALU_HYPRE_BOOMERAMGSETCYCLENUMSWEE)
(hypre_F90_Obj *, NALU_HYPRE_Int *, NALU_HYPRE_Int *);

#define NALU_HYPRE_BoomerAMGGetCycleNumSweeps  \
        hypre_F90_NAME(fhypre_boomeramggetcyclenumswee, FNALU_HYPRE_BOOMERAMGGETCYCLENUMSWEE)
extern void hypre_F90_NAME(fhypre_boomeramggetcyclenumswee, FNALU_HYPRE_BOOMERAMGGETCYCLENUMSWEE)
(hypre_F90_Obj *, NALU_HYPRE_Int *, NALU_HYPRE_Int *);

#define NALU_HYPRE_BoomerAMGInitGridRelaxation  \
        hypre_F90_NAME(fhypre_boomeramginitgridrelaxat, FNALU_HYPRE_BOOMERAMGINITGRIDRELAXAT)
extern void hypre_F90_NAME(fhypre_boomeramginitgridrelaxat, FNALU_HYPRE_BOOMERAMGINITGRIDRELAXAT)
(hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *, NALU_HYPRE_Int *, hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_BoomerAMGFinalizeGridRelaxation  \
        hypre_F90_NAME(fhypre_boomeramgfingridrelaxatn, FNALU_HYPRE_BOOMERAMGFINGRIDRELAXATN)
extern void hypre_F90_NAME(fhypre_boomeramgfingridrelaxatn, FNALU_HYPRE_BOOMERAMGFINGRIDRELAXATN)
(hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *);

#define NALU_HYPRE_BoomerAMGSetRelaxType  \
        hypre_F90_NAME(fhypre_boomeramgsetrelaxtype, FNALU_HYPRE_BOOMERAMGSETRELAXTYPE)
extern void hypre_F90_NAME(fhypre_boomeramgsetrelaxtype, FNALU_HYPRE_BOOMERAMGSETRELAXTYPE)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_BoomerAMGSetRelaxType  \
        hypre_F90_NAME(fhypre_boomeramgsetrelaxtype, FNALU_HYPRE_BOOMERAMGSETRELAXTYPE)
extern void hypre_F90_NAME(fhypre_boomeramgsetrelaxtype, FNALU_HYPRE_BOOMERAMGSETRELAXTYPE)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_BoomerAMGSetRelaxOrder  \
        hypre_F90_NAME(fhypre_boomeramgsetrelaxorder, FNALU_HYPRE_BOOMERAMGSETRELAXORDER)
extern void hypre_F90_NAME(fhypre_boomeramgsetrelaxorder, FNALU_HYPRE_BOOMERAMGSETRELAXORDER)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_BoomerAMGSetRelaxWeight  \
        hypre_F90_NAME(fhypre_boomeramgsetrelaxweight, FNALU_HYPRE_BOOMERAMGSETRELAXWEIGHT)
extern void hypre_F90_NAME(fhypre_boomeramgsetrelaxweight, FNALU_HYPRE_BOOMERAMGSETRELAXWEIGHT)
(hypre_F90_Obj *, hypre_F90_Obj *);

#define NALU_HYPRE_BoomerAMGSetRelaxWt  \
        hypre_F90_NAME(fhypre_boomeramgsetrelaxwt, FNALU_HYPRE_BOOMERAMGSETRELAXWT)
extern void hypre_F90_NAME(fhypre_boomeramgsetrelaxwt, FNALU_HYPRE_BOOMERAMGSETRELAXWT)
(hypre_F90_Obj *, NALU_HYPRE_Real *);

#define NALU_HYPRE_BoomerAMGSetLevelRelaxWt  \
        hypre_F90_NAME(fhypre_boomeramgsetlevelrelaxwt, FNALU_HYPRE_BOOMERAMGSETLEVELRELAXWT)
extern void hypre_F90_NAME(fhypre_boomeramgsetlevelrelaxwt, FNALU_HYPRE_BOOMERAMGSETLEVELRELAXWT)
(hypre_F90_Obj *, NALU_HYPRE_Real *, NALU_HYPRE_Int *);

#define NALU_HYPRE_BoomerAMGSetOuterWt  \
        hypre_F90_NAME(fhypre_boomeramgsetouterwt, FNALU_HYPRE_BOOMERAMGSETOUTERWT)
extern void hypre_F90_NAME(fhypre_boomeramgsetouterwt, FNALU_HYPRE_BOOMERAMGSETOUTERWT)
(hypre_F90_Obj *, NALU_HYPRE_Real *);

#define NALU_HYPRE_BoomerAMGSetLevelOuterWt  \
        hypre_F90_NAME(fhypre_boomeramgsetlevelouterwt, FNALU_HYPRE_BOOMERAMGSETLEVELOUTERWT)
extern void hypre_F90_NAME(fhypre_boomeramgsetlevelouterwt, FNALU_HYPRE_BOOMERAMGSETLEVELOUTERWT)
(hypre_F90_Obj *, NALU_HYPRE_Real *);

#define NALU_HYPRE_BoomerAMGSetSmoothType  \
        hypre_F90_NAME(fhypre_boomeramgsetsmoothtype, FNALU_HYPRE_BOOMERAMGSETSMOOTHTYPE)
extern void hypre_F90_NAME(fhypre_boomeramgsetsmoothtype, FNALU_HYPRE_BOOMERAMGSETSMOOTHTYPE)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_BoomerAMGGetSmoothType  \
        hypre_F90_NAME(fhypre_boomeramggetsmoothtype, FNALU_HYPRE_BOOMERAMGGETSMOOTHTYPE)
extern void hypre_F90_NAME(fhypre_boomeramggetsmoothtype, FNALU_HYPRE_BOOMERAMGGETSMOOTHTYPE)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_BoomerAMGSetSmoothNumLvls  \
        hypre_F90_NAME(fhypre_boomeramgsetsmoothnumlvl, FNALU_HYPRE_BOOMERAMGSETSMOOTHNUMLVL)
extern void hypre_F90_NAME(fhypre_boomeramgsetsmoothnumlvl, FNALU_HYPRE_BOOMERAMGSETSMOOTHNUMLVL)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_BoomerAMGGetSmoothNumLvls  \
        hypre_F90_NAME(fhypre_boomeramggetsmoothnumlvl, FNALU_HYPRE_BOOMERAMGGETSMOOTHNUMLVL)
extern void hypre_F90_NAME(fhypre_boomeramggetsmoothnumlvl, FNALU_HYPRE_BOOMERAMGGETSMOOTHNUMLVL)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_BoomerAMGSetSmoothNumSwps  \
        hypre_F90_NAME(fhypre_boomeramgsetsmoothnumswp, FNALU_HYPRE_BOOMERAMGSETSMOOTHNUMSWP)
extern void hypre_F90_NAME(fhypre_boomeramgsetsmoothnumswp, FNALU_HYPRE_BOOMERAMGSETSMOOTHNUMSWP)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_BoomerAMGGetSmoothNumSwps  \
        hypre_F90_NAME(fhypre_boomeramggetsmoothnumswp, FNALU_HYPRE_BOOMERAMGGETSMOOTHNUMSWP)
extern void hypre_F90_NAME(fhypre_boomeramggetsmoothnumswp, FNALU_HYPRE_BOOMERAMGGETSMOOTHNUMSWP)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_BoomerAMGSetLogging  \
        hypre_F90_NAME(fhypre_boomeramgsetlogging, FNALU_HYPRE_BOOMERAMGSETLOGGING)
extern void hypre_F90_NAME(fhypre_boomeramgsetlogging, FNALU_HYPRE_BOOMERAMGSETLOGGING)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_BoomerAMGGetLogging  \
        hypre_F90_NAME(fhypre_boomeramggetlogging, FNALU_HYPRE_BOOMERAMGGETLOGGING)
extern void hypre_F90_NAME(fhypre_boomeramggetlogging, FNALU_HYPRE_BOOMERAMGGETLOGGING)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_BoomerAMGSetPrintLevel  \
        hypre_F90_NAME(fhypre_boomeramgsetprintlevel, FNALU_HYPRE_BOOMERAMGSETPRINTLEVEL)
extern void hypre_F90_NAME(fhypre_boomeramgsetprintlevel, FNALU_HYPRE_BOOMERAMGSETPRINTLEVEL)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_BoomerAMGGetPrintLevel  \
        hypre_F90_NAME(fhypre_boomeramggetprintlevel, FNALU_HYPRE_BOOMERAMGGETPRINTLEVEL)
extern void hypre_F90_NAME(fhypre_boomeramggetprintlevel, FNALU_HYPRE_BOOMERAMGGETPRINTLEVEL)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_BoomerAMGSetPrintFileName  \
        hypre_F90_NAME(fhypre_boomeramgsetprintfilenam, FNALU_HYPRE_BOOMERAMGSETPRINTFILENAM)
extern void hypre_F90_NAME(fhypre_boomeramgsetprintfilenam, FNALU_HYPRE_BOOMERAMGSETPRINTFILENAM)
(hypre_F90_Obj *, char *);

#define NALU_HYPRE_BoomerAMGSetDebugFlag  \
        hypre_F90_NAME(fhypre_boomeramgsetdebugflag, FNALU_HYPRE_BOOMERAMGSETDEBUGFLAG)
extern void hypre_F90_NAME(fhypre_boomeramgsetdebugflag, FNALU_HYPRE_BOOMERAMGSETDEBUGFLAG)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_BoomerAMGGetDebugFlag  \
        hypre_F90_NAME(fhypre_boomeramggetdebugflag, FNALU_HYPRE_BOOMERAMGGETDEBUGFLAG)
extern void hypre_F90_NAME(fhypre_boomeramggetdebugflag, FNALU_HYPRE_BOOMERAMGGETDEBUGFLAG)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_BoomerAMGGetNumIterations  \
        hypre_F90_NAME(fhypre_boomeramggetnumiteration, FNALU_HYPRE_BOOMERAMGGETNUMITERATION)
extern void hypre_F90_NAME(fhypre_boomeramggetnumiteration, FNALU_HYPRE_BOOMERAMGGETNUMITERATION)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_BoomerAMGGetCumNumIterations  \
        hypre_F90_NAME(fhypre_boomeramggetcumnumiterat, FNALU_HYPRE_BOOMERAMGGETCUMNUMITERAT)
extern void hypre_F90_NAME(fhypre_boomeramggetcumnumiterat, FNALU_HYPRE_BOOMERAMGGETCUMNUMITERAT)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_BoomerAMGGetResidual  \
        hypre_F90_NAME(fhypre_boomeramggetresidual, FNALU_HYPRE_BOOMERAMGGETRESIDUAL)
extern void hypre_F90_NAME(fhypre_boomeramggetresidual, FNALU_HYPRE_BOOMERAMGGETRESIDUAL)
(hypre_F90_Obj *, NALU_HYPRE_Real *);

#define NALU_HYPRE_BoomerAMGGetFinalRelativeResidualNorm  \
        hypre_F90_NAME(fhypre_boomeramggetfinalreltvre, FNALU_HYPRE_BOOMERAMGGETFINALRELTVRE)
extern void hypre_F90_NAME(fhypre_boomeramggetfinalreltvre, FNALU_HYPRE_BOOMERAMGGETFINALRELTVRE)
(hypre_F90_Obj *, NALU_HYPRE_Real *);

#define NALU_HYPRE_BoomerAMGSetVariant  \
        hypre_F90_NAME(fhypre_boomeramgsetvariant, FNALU_HYPRE_BOOMERAMGSETVARIANT)
extern void hypre_F90_NAME(fhypre_boomeramgsetvariant, FNALU_HYPRE_BOOMERAMGSETVARIANT)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_BoomerAMGGetVariant  \
        hypre_F90_NAME(fhypre_boomeramggetvariant, FNALU_HYPRE_BOOMERAMGGETVARIANT)
extern void hypre_F90_NAME(fhypre_boomeramggetvariant, FNALU_HYPRE_BOOMERAMGGETVARIANT)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_BoomerAMGSetOverlap  \
        hypre_F90_NAME(fhypre_boomeramgsetoverlap, FNALU_HYPRE_BOOMERAMGSETOVERLAP)
extern void hypre_F90_NAME(fhypre_boomeramgsetoverlap, FNALU_HYPRE_BOOMERAMGSETOVERLAP)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_BoomerAMGGetOverlap  \
        hypre_F90_NAME(fhypre_boomeramggetoverlap, FNALU_HYPRE_BOOMERAMGGETOVERLAP)
extern void hypre_F90_NAME(fhypre_boomeramggetoverlap, FNALU_HYPRE_BOOMERAMGGETOVERLAP)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_BoomerAMGSetDomainType  \
        hypre_F90_NAME(fhypre_boomeramgsetdomaintype, FNALU_HYPRE_BOOMERAMGSETDOMAINTYPE)
extern void hypre_F90_NAME(fhypre_boomeramgsetdomaintype, FNALU_HYPRE_BOOMERAMGSETDOMAINTYPE)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_BoomerAMGGetDomainType  \
        hypre_F90_NAME(fhypre_boomeramggetdomaintype, FNALU_HYPRE_BOOMERAMGGETDOMAINTYPE)
extern void hypre_F90_NAME(fhypre_boomeramggetdomaintype, FNALU_HYPRE_BOOMERAMGGETDOMAINTYPE)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_BoomerAMGSetSchwarzRlxWt  \
        hypre_F90_NAME(fhypre_boomeramgsetschwarzrlxwt, FNALU_HYPRE_BOOMERAMGSETSCHWARZRLXWT)
extern void hypre_F90_NAME(fhypre_boomeramgsetschwarzrlxwt, FNALU_HYPRE_BOOMERAMGSETSCHWARZRLXWT)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_BoomerAMGGetSchwarzRlxWt  \
        hypre_F90_NAME(fhypre_boomeramggetschwarzrlxwt, FNALU_HYPRE_BOOMERAMGGETSCHWARZRLXWT)
extern void hypre_F90_NAME(fhypre_boomeramggetschwarzrlxwt, FNALU_HYPRE_BOOMERAMGGETSCHWARZRLXWT)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_BoomerAMGSetSym  \
        hypre_F90_NAME(fhypre_boomeramgsetsym, FNALU_HYPRE_BOOMERAMGSETSYM)
extern void hypre_F90_NAME(fhypre_boomeramgsetsym, FNALU_HYPRE_BOOMERAMGSETSYM)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_BoomerAMGSetLevel  \
        hypre_F90_NAME(fhypre_boomeramgsetlevel, FNALU_HYPRE_BOOMERAMGSETLEVEL)
extern void hypre_F90_NAME(fhypre_boomeramgsetlevel, FNALU_HYPRE_BOOMERAMGSETLEVEL)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_BoomerAMGSetFilter  \
        hypre_F90_NAME(fhypre_boomeramgsetfilter, FNALU_HYPRE_BOOMERAMGSETFILTER)
extern void hypre_F90_NAME(fhypre_boomeramgsetfilter, FNALU_HYPRE_BOOMERAMGSETFILTER)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_BoomerAMGSetDropTol  \
        hypre_F90_NAME(fhypre_boomeramgsetdroptol, FNALU_HYPRE_BOOMERAMGSETDROPTOL)
extern void hypre_F90_NAME(fhypre_boomeramgsetdroptol, FNALU_HYPRE_BOOMERAMGSETDROPTOL)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_BoomerAMGSetMaxNzPerRow  \
        hypre_F90_NAME(fhypre_boomeramgsetmaxnzperrow, FNALU_HYPRE_BOOMERAMGSETMAXNZPERROW)
extern void hypre_F90_NAME(fhypre_boomeramgsetmaxnzperrow, FNALU_HYPRE_BOOMERAMGSETMAXNZPERROW)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_BoomerAMGSetEuclidFile  \
        hypre_F90_NAME(fhypre_boomeramgseteuclidfile, FNALU_HYPRE_BOOMERAMGSETEUCLIDFILE)
extern void hypre_F90_NAME(fhypre_boomeramgseteuclidfile, FNALU_HYPRE_BOOMERAMGSETEUCLIDFILE)
(hypre_F90_Obj *, char *);

#define NALU_HYPRE_BoomerAMGSetNumFunctions  \
        hypre_F90_NAME(fhypre_boomeramgsetnumfunctions, FNALU_HYPRE_BOOMERAMGSETNUMFUNCTIONS)
extern void hypre_F90_NAME(fhypre_boomeramgsetnumfunctions, FNALU_HYPRE_BOOMERAMGSETNUMFUNCTIONS)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_BoomerAMGGetNumFunctions  \
        hypre_F90_NAME(fhypre_boomeramggetnumfunctions, FNALU_HYPRE_BOOMERAMGGETNUMFUNCTIONS)
extern void hypre_F90_NAME(fhypre_boomeramggetnumfunctions, FNALU_HYPRE_BOOMERAMGGETNUMFUNCTIONS)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_BoomerAMGSetNodal  \
        hypre_F90_NAME(fhypre_boomeramgsetnodal, FNALU_HYPRE_BOOMERAMGSETNODAL)
extern void hypre_F90_NAME(fhypre_boomeramgsetnodal, FNALU_HYPRE_BOOMERAMGSETNODAL)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_BoomerAMGSetDofFunc  \
        hypre_F90_NAME(fhypre_boomeramgsetdoffunc, FNALU_HYPRE_BOOMERAMGSETDOFFUNC)
extern void hypre_F90_NAME(fhypre_boomeramgsetdoffunc, FNALU_HYPRE_BOOMERAMGSETDOFFUNC)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_BoomerAMGSetNumPaths  \
        hypre_F90_NAME(fhypre_boomeramgsetnumpaths, FNALU_HYPRE_BOOMERAMGSETNUMPATHS)
extern void hypre_F90_NAME(fhypre_boomeramgsetnumpaths, FNALU_HYPRE_BOOMERAMGSETNUMPATHS)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_BoomerAMGSetAggNumLevels  \
        hypre_F90_NAME(fhypre_boomeramgsetaggnumlevels, FNALU_HYPRE_BOOMERAMGSETAGGNUMLEVELS)
extern void hypre_F90_NAME(fhypre_boomeramgsetaggnumlevels, FNALU_HYPRE_BOOMERAMGSETAGGNUMLEVELS)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_BoomerAMGSetGSMG  \
        hypre_F90_NAME(fhypre_boomeramgsetgsmg, FNALU_HYPRE_BOOMERAMGSETGSMG)
extern void hypre_F90_NAME(fhypre_boomeramgsetgsmg, FNALU_HYPRE_BOOMERAMGSETGSMG)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_BoomerAMGSetNumSamples  \
        hypre_F90_NAME(fhypre_boomeramgsetnumsamples, FNALU_HYPRE_BOOMERAMGSETNUMSAMPLES)
extern void hypre_F90_NAME(fhypre_boomeramgsetsamples, FNALU_HYPRE_BOOMERAMGSETNUMSAMPLES)
(hypre_F90_Obj *, NALU_HYPRE_Int *);



#define NALU_HYPRE_ParCSRBiCGSTABCreate  \
        hypre_F90_NAME(fhypre_parcsrbicgstabcreate, FNALU_HYPRE_PARCSRBICGSTABCREATE)
extern void hypre_F90_NAME(fhypre_parcsrbicgstabcreate, FNALU_HYPRE_PARCSRBICGSTABCREATE)
(NALU_HYPRE_Int *, hypre_F90_Obj *);

#define NALU_HYPRE_ParCSRBiCGSTABDestroy  \
        hypre_F90_NAME(fhypre_parcsrbicgstabdestroy, FNALU_HYPRE_PARCSRBICGSTABDESTROY)
extern void hypre_F90_NAME(fhypre_parcsrbicgstabdestroy, FNALU_HYPRE_PARCSRBICGSTABDESTROY)
(hypre_F90_Obj *);

#define NALU_HYPRE_ParCSRBiCGSTABSetup  \
        hypre_F90_NAME(fhypre_parcsrbicgstabsetup, FNALU_HYPRE_PARCSRBICGSTABSETUP)
extern void hypre_F90_NAME(fhypre_parcsrbicgstabsetup, FNALU_HYPRE_PARCSRBICGSTABSETUP)
(hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *);

#define NALU_HYPRE_ParCSRBiCGSTABSolve  \
        hypre_F90_NAME(fhypre_parcsrbicgstabsolve, FNALU_HYPRE_PARCSRBICGSTABSOLVE)
extern void hypre_F90_NAME(fhypre_parcsrbicgstabsolve, FNALU_HYPRE_PARCSRBICGSTABSOLVE)
(hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *);

#define NALU_HYPRE_ParCSRBiCGSTABSetTol  \
        hypre_F90_NAME(fhypre_parcsrbicgstabsettol, FNALU_HYPRE_PARCSRBICGSTABSETTOL)
extern void hypre_F90_NAME(fhypre_parcsrbicgstabsettol, FNALU_HYPRE_PARCSRBICGSTABSETTOL)
(hypre_F90_Obj *, NALU_HYPRE_Real *);

#define NALU_HYPRE_ParCSRBiCGSTABSetMinIter  \
        hypre_F90_NAME(fhypre_parcsrbicgstabsetminiter, FNALU_HYPRE_PARCSRBICGSTABSETMINITER)
extern void hypre_F90_NAME(fhypre_parcsrbicgstabsetminiter, FNALU_HYPRE_PARCSRBICGSTABSETMINITER)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_ParCSRBiCGSTABSetMaxIter  \
        hypre_F90_NAME(fhypre_parcsrbicgstabsetmaxiter, FNALU_HYPRE_PARCSRBICGSTABSETMAXITER)
extern void hypre_F90_NAME(fhypre_parcsrbicgstabsetmaxiter, FNALU_HYPRE_PARCSRBICGSTABSETMAXITER)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_ParCSRBiCGSTABSetStopCrit  \
        hypre_F90_NAME(fhypre_parcsrbicgstabsetstopcri, FNALU_HYPRE_PARCSRBICGSTABSETSTOPCRI)
extern void hypre_F90_NAME(fhypre_parcsrbicgstabsetstopcri, FNALU_HYPRE_PARCSRBICGSTABSETSTOPCRI)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_ParCSRBiCGSTABSetPrecond  \
        hypre_F90_NAME(fhypre_parcsrbicgstabsetprecond, FNALU_HYPRE_PARCSRBICGSTABSETPRECOND)
extern void hypre_F90_NAME(fhypre_parcsrbicgstabsetprecond, FNALU_HYPRE_PARCSRBICGSTABSETPRECOND)
(hypre_F90_Obj *, NALU_HYPRE_Int *, hypre_F90_Obj *);

#define NALU_HYPRE_ParCSRBiCGSTABGetPrecond  \
        hypre_F90_NAME(fhypre_parcsrbicgstabgetprecond, FNALU_HYPRE_PARCSRBICGSTABGETPRECOND)
extern void hypre_F90_NAME(fhypre_parcsrbicgstabgetprecond, FNALU_HYPRE_PARCSRBICGSTABGETPRECOND)
(hypre_F90_Obj *, hypre_F90_Obj *);

#define NALU_HYPRE_ParCSRBiCGSTABSetLogging  \
        hypre_F90_NAME(fhypre_parcsrbicgstabsetlogging, FNALU_HYPRE_PARCSRBICGSTABSETLOGGING)
extern void hypre_F90_NAME(fhypre_parcsrbicgstabsetlogging, FNALU_HYPRE_PARCSRBICGSTABSETLOGGING)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_ParCSRBiCGSTABSetPrintLevel  \
        hypre_F90_NAME(fhypre_parcsrbicgstabsetprintle, FNALU_HYPRE_PARCSRBICGSTABSETPRINTLE)
extern void hypre_F90_NAME(fhypre_parcsrbicgstabsetprintle, FNALU_HYPRE_PARCSRBICGSTABSETPRINTLE)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_ParCSRBiCGSTABGetNumIter  \
        hypre_F90_NAME(fhypre_parcsrbicgstabgetnumiter, FNALU_HYPRE_PARCSRBICGSTABGETNUMITER)
extern void hypre_F90_NAME(fhypre_parcsrbicgstabgetnumiter, FNALU_HYPRE_PARCSRBICGSTABGETNUMITER)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_ParCSRBiCGSTABGetFinalRel  \
        hypre_F90_NAME(fhypre_parcsrbicgstabgetfinalre, FNALU_HYPRE_PARCSRBICGSTABGETFINALRE)
extern void hypre_F90_NAME(fhypre_parcsrbicgstabgetfinalre, FNALU_HYPRE_PARCSRBICGSTABGETFINALRE)
(hypre_F90_Obj *, NALU_HYPRE_Real *);



#define NALU_HYPRE_BlockTridiagCreate  \
   hypre_F90_NAME(fhypre_blocktridiagcreate, FNALU_HYPRE_BLOCKTRIDIAGCREATE)
extern void hypre_F90_NAME(fhypre_blocktridiagcreate, FNALU_HYPRE_BLOCKTRIDIAGCREATE)
(hypre_F90_Obj *);

#define NALU_HYPRE_BlockTridiagDestroy  \
   hypre_F90_NAME(fhypre_blocktridiagdestroy, FNALU_HYPRE_BLOCKTRIDIAGDESTROY)
extern void hypre_F90_NAME(fhypre_blocktridiagdestroy, FNALU_HYPRE_BLOCKTRIDIAGDESTROY)
(hypre_F90_Obj *);

#define NALU_HYPRE_BlockTridiagSetup  \
   hypre_F90_NAME(fhypre_blocktridiagsetup, FNALU_HYPRE_BLOCKTRIDIAGSETUP)
extern void hypre_F90_NAME(fhypre_blocktridiagsetup, FNALU_HYPRE_BLOCKTRIDIAGSETUP)
(hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *);

#define NALU_HYPRE_BlockTridiagSolve  \
   hypre_F90_NAME(fhypre_blocktridiagsolve, FNALU_HYPRE_BLOCKTRIDIAGSOLVE)
extern void hypre_F90_NAME(fhypre_blocktridiagsolve, FNALU_HYPRE_BLOCKTRIDIAGSOLVE)
(hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *);

#define NALU_HYPRE_BlockTridiagSetIndexSet  \
   hypre_F90_NAME(fhypre_blocktridiagsetindexset, FNALU_HYPRE_BLOCKTRIDIAGSETINDEXSET)
extern void hypre_F90_NAME(fhypre_blocktridiagsetindexset, FNALU_HYPRE_BLOCKTRIDIAGSETINDEXSET)
(hypre_F90_Obj *, NALU_HYPRE_Int *, NALU_HYPRE_Int *);

#define NALU_HYPRE_BlockTridiagSetAMGStrengthThreshold  \
   hypre_F90_NAME(fhypre_blocktridiagsetamgstreng, FNALU_HYPRE_BLOCKTRIDIAGSETAMGSTRENG)
extern void hypre_F90_NAME(fhypre_blocktridiagsetamgstreng, FNALU_HYPRE_BLOCKTRIDIAGSETAMGSTRENG)
(hypre_F90_Obj *, NALU_HYPRE_Real *);

#define NALU_HYPRE_BlockTridiagSetAMGNumSweeps  \
   hypre_F90_NAME(fhypre_blocktridiagsetamgnumswe, FNALU_HYPRE_BLOCKTRIDIAGSETAMGNUMSWE)
extern void hypre_F90_NAME(fhypre_blocktridiagsetamgnumswe, FNALU_HYPRE_BLOCKTRIDIAGSETAMGNUMSWE)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_BlockTridiagSetAMGRelaxType  \
   hypre_F90_NAME(fhypre_blocktridiagsetamgrelaxt, FNALU_HYPRE_BLOCKTRIDIAGSETAMGRELAXT)
extern void hypre_F90_NAME(fhypre_blocktridiagsetamgrelaxt, FNALU_HYPRE_BLOCKTRIDIAGSETAMGRELAXT)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_BlockTridiagSetPrintLevel  \
   hypre_F90_NAME(fhypre_blocktridiagsetprintleve, FNALU_HYPRE_BLOCKTRIDIAGSETPRINTLEVE)
extern void hypre_F90_NAME(fhypre_blocktridiagsetprintleve, FNALU_HYPRE_BLOCKTRIDIAGSETPRINTLEVE)
(hypre_F90_Obj *, NALU_HYPRE_Int *);



#define NALU_HYPRE_ParCSRCGNRCreate  \
   hypre_F90_NAME(fhypre_parcsrcgnrcreate, FNALU_HYPRE_PARCSRCGNRCREATE)
extern void hypre_F90_NAME(fhypre_parcsrcgnrcreate, FNALU_HYPRE_PARCSRCGNRCREATE)
(NALU_HYPRE_Int *, hypre_F90_Obj *);

#define NALU_HYPRE_ParCSRCGNRDestroy  \
        hypre_F90_NAME(fhypre_parcsrcgnrdestroy, FNALU_HYPRE_PARCSRCGNRDESTROY)
extern void hypre_F90_NAME(fhypre_parcsrcgnrdestroy, FNALU_HYPRE_PARCSRCGNRDESTROY)
(hypre_F90_Obj *);

#define NALU_HYPRE_ParCSRCGNRSetup  \
        hypre_F90_NAME(fhypre_parcsrcgnrsetup, FNALU_HYPRE_PARCSRCGNRSETUP)
extern void hypre_F90_NAME(fhypre_parcsrcgnrsetup, FNALU_HYPRE_PARCSRCGNRSETUP)
(hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *);

#define NALU_HYPRE_ParCSRCGNRSolve  \
        hypre_F90_NAME(fhypre_parcsrcgnrsolve, FNALU_HYPRE_PARCSRCGNRSOLVE)
extern void hypre_F90_NAME(fhypre_parcsrcgnrsolve, FNALU_HYPRE_PARCSRCGNRSOLVE)
(hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *);

#define NALU_HYPRE_ParCSRCGNRSetTol  \
        hypre_F90_NAME(fhypre_parcsrcgnrsettol, FNALU_HYPRE_PARCSRCGNRSETTOL)
extern void hypre_F90_NAME(fhypre_parcsrcgnrsettol, FNALU_HYPRE_PARCSRCGNRSETTOL)
(hypre_F90_Obj *, NALU_HYPRE_Real *);

#define NALU_HYPRE_ParCSRCGNRSetMinIter  \
        hypre_F90_NAME(fhypre_parcsrcgnrsetminiter, FNALU_HYPRE_PARCSRCGNRSETMINITER)
extern void hypre_F90_NAME(fhypre_parcsrcgnrsetminiter, FNALU_HYPRE_PARCSRCGNRSETMINITER)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_ParCSRCGNRSetMaxIter  \
        hypre_F90_NAME(fhypre_parcsrcgnrsetmaxiter, FNALU_HYPRE_PARCSRCGNRSETMAXITER)
extern void hypre_F90_NAME(fhypre_parcsrcgnrsetmaxiter, FNALU_HYPRE_PARCSRCGNRSETMAXITER)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_ParCSRCGNRSetStopCrit  \
        hypre_F90_NAME(fhypre_parcsrcgnrsetstopcri, FNALU_HYPRE_PARCSRCGNRSETSTOPCRI)
extern void hypre_F90_NAME(fhypre_parcsrcgnrsetstopcri, FNALU_HYPRE_PARCSRCGNRSETSTOPCRI)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_ParCSRCGNRSetPrecond  \
        hypre_F90_NAME(fhypre_parcsrcgnrsetprecond, FNALU_HYPRE_PARCSRCGNRSETPRECOND)
extern void hypre_F90_NAME(fhypre_parcsrcgnrsetprecond, FNALU_HYPRE_PARCSRCGNRSETPRECOND)
(hypre_F90_Obj *, NALU_HYPRE_Int *, hypre_F90_Obj *);

#define NALU_HYPRE_ParCSRCGNRGetPrecond  \
        hypre_F90_NAME(fhypre_parcsrcgnrgetprecond, FNALU_HYPRE_PARCSRCGNRGETPRECOND)
extern void hypre_F90_NAME(fhypre_parcsrcgnrgetprecond, FNALU_HYPRE_PARCSRCGNRGETPRECOND)
(hypre_F90_Obj *, hypre_F90_Obj *);

#define NALU_HYPRE_ParCSRCGNRSetLogging  \
        hypre_F90_NAME(fhypre_parcsrcgnrsetlogging, FNALU_HYPRE_PARCSRCGNRSETLOGGING)
extern void hypre_F90_NAME(fhypre_parcsrcgnrsetlogging, FNALU_HYPRE_PARCSRCGNRSETLOGGING)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_ParCSRCGNRGetNumIteration  \
        hypre_F90_NAME(fhypre_parcsrcgnrgetnumiteratio, FNALU_HYPRE_PARCSRCGNRGETNUMITERATIO)
extern void hypre_F90_NAME(fhypre_parcsrcgnrgetnumiteratio, FNALU_HYPRE_PARCSRCGNRGETNUMITERATIO)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_ParCSRCGNRGetFinalRelativeResidualNorm  \
        hypre_F90_NAME(fhypre_parcsrcgnrgetfinalrelati, FNALU_HYPRE_PARCSRCGNRGETFINALRELATI)
extern void hypre_F90_NAME(fhypre_parcsrcgnrgetfinalrelati, FNALU_HYPRE_PARCSRCGNRGETFINALRELATI)
(hypre_F90_Obj *, NALU_HYPRE_Real *);



#define NALU_HYPRE_EuclidCreate  \
        hypre_F90_NAME(fhypre_euclidcreate, FNALU_HYPRE_EUCLIDCREATE)
extern void hypre_F90_NAME(fhypre_euclidcreate, FNALU_HYPRE_EUCLIDCREATE)
(NALU_HYPRE_Int *, hypre_F90_Obj *);

#define NALU_HYPRE_EuclidDestroy  \
        hypre_F90_NAME(fhypre_eucliddestroy, FNALU_HYPRE_EUCLIDDESTROY)
extern void hypre_F90_NAME(fhypre_eucliddestroy, FNALU_HYPRE_EUCLIDDESTROY)
(hypre_F90_Obj *);

#define NALU_HYPRE_EuclidSetup  \
        hypre_F90_NAME(fhypre_euclidsetup, FNALU_HYPRE_EUCLIDSETUP)
extern void hypre_F90_NAME(fhypre_euclidsetup, FNALU_HYPRE_EUCLIDSETUP)
(hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *);

#define NALU_HYPRE_EuclidSolve  \
        hypre_F90_NAME(fhypre_euclidsolve, FNALU_HYPRE_EUCLIDSOLVE)
extern void hypre_F90_NAME(fhypre_euclidsolve, FNALU_HYPRE_EUCLIDSOLVE)
(hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *);

#define NALU_HYPRE_EuclidSetParams  \
        hypre_F90_NAME(fhypre_euclidsetparams, FNALU_HYPRE_EUCLIDSETPARAMS)
extern void hypre_F90_NAME(fhypre_euclidsetparams, FNALU_HYPRE_EUCLIDSETPARAMS)
(hypre_F90_Obj *, NALU_HYPRE_Int *, char *);

#define NALU_HYPRE_EuclidSetParamsFromFile  \
        hypre_F90_NAME(fhypre_euclidsetparamsfromfile, FNALU_HYPRE_EUCLIDSETPARAMSFROMFILE)
extern void hypre_F90_NAME(fhypre_euclidsetparamsfromfile, FNALU_HYPRE_EUCLIDSETPARAMSFROMFILE)
(hypre_F90_Obj *, char *);



#define NALU_HYPRE_ParCSRGMRESCreate  \
        hypre_F90_NAME(fhypre_parcsrgmrescreate, FNALU_HYPRE_PARCSRGMRESCREATE)
extern void hypre_F90_NAME(fhypre_parcsrgmrescreate, FNALU_HYPRE_PARCSRGMRESCREATE)
(NALU_HYPRE_Int *, hypre_F90_Obj *);

#define NALU_HYPRE_ParCSRGMRESDestroy  \
        hypre_F90_NAME(fhypre_parcsrgmresdestroy, FNALU_HYPRE_PARCSRGMRESDESTROY)
extern void hypre_F90_NAME(fhypre_parcsrgmresdestroy, FNALU_HYPRE_PARCSRGMRESDESTROY)
(hypre_F90_Obj *);

#define NALU_HYPRE_ParCSRGMRESSetup  \
        hypre_F90_NAME(fhypre_parcsrgmressetup, FNALU_HYPRE_PARCSRGMRESSETUP)
extern void hypre_F90_NAME(fhypre_parcsrgmressetup, FNALU_HYPRE_PARCSRGMRESSETUP)
(hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *);

#define NALU_HYPRE_ParCSRGMRESSolve  \
        hypre_F90_NAME(fhypre_parcsrgmressolve, FNALU_HYPRE_PARCSRGMRESSOLVE)
extern void hypre_F90_NAME(fhypre_parcsrgmressolve, FNALU_HYPRE_PARCSRGMRESSOLVE)
(hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *);

#define NALU_HYPRE_ParCSRGMRESSetKDim  \
        hypre_F90_NAME(fhypre_parcsrgmressetkdim, FNALU_HYPRE_PARCSRGMRESSETKDIM)
extern void hypre_F90_NAME(fhypre_parcsrgmressetkdim, FNALU_HYPRE_PARCSRGMRESSETKDIM)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_ParCSRGMRESSetTol  \
        hypre_F90_NAME(fhypre_parcsrgmressettol, FNALU_HYPRE_PARCSRGMRESSETTOL)
extern void hypre_F90_NAME(fhypre_parcsrgmressettol, FNALU_HYPRE_PARCSRGMRESSETTOL)
(hypre_F90_Obj *, NALU_HYPRE_Real *);

#define NALU_HYPRE_ParCSRGMRESSetMinIter  \
        hypre_F90_NAME(fhypre_parcsrgmressetminiter, FNALU_HYPRE_PARCSRGMRESSETMINITER)
extern void hypre_F90_NAME(fhypre_parcsrgmressetminiter, FNALU_HYPRE_PARCSRGMRESSETMINITER)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_ParCSRGMRESSetMaxIter  \
        hypre_F90_NAME(fhypre_parcsrgmressetmaxiter, FNALU_HYPRE_PARCSRGMRESSETMAXITER)
extern void hypre_F90_NAME(fhypre_parcsrgmressetmaxiter, FNALU_HYPRE_PARCSRGMRESSETMAXITER)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_ParCSRGMRESSetPrecond  \
        hypre_F90_NAME(fhypre_parcsrgmressetprecond, FNALU_HYPRE_PARCSRGMRESSETPRECOND)
extern void hypre_F90_NAME(fhypre_parcsrgmressetprecond, FNALU_HYPRE_PARCSRGMRESSETPRECOND)
(hypre_F90_Obj *, NALU_HYPRE_Int *, hypre_F90_Obj *);

#define NALU_HYPRE_ParCSRGMRESGetPrecond  \
        hypre_F90_NAME(fhypre_parcsrgmresgetprecond, FNALU_HYPRE_PARCSRGMRESGETPRECOND)
extern void hypre_F90_NAME(fhypre_parcsrgmresgetprecond, FNALU_HYPRE_PARCSRGMRESGETPRECOND)
(hypre_F90_Obj *, hypre_F90_Obj *);

#define NALU_HYPRE_ParCSRGMRESSetLogging  \
        hypre_F90_NAME(fhypre_parcsrgmressetlogging, FNALU_HYPRE_PARCSRGMRESSETLOGGING)
extern void hypre_F90_NAME(fhypre_parcsrgmressetlogging, FNALU_HYPRE_PARCSRGMRESSETLOGGING)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_ParCSRGMRESSetPrintLevel  \
        hypre_F90_NAME(fhypre_parcsrgmressetprintlevel, FNALU_HYPRE_PARCSRGMRESSETPRINTLEVEL)
extern void hypre_F90_NAME(fhypre_parcsrgmressetprintlevel, FNALU_HYPRE_PARCSRGMRESSETPRINTLEVEL)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_ParCSRGMRESGetNumIterations  \
        hypre_F90_NAME(fhypre_parcsrgmresgetnumiterati, FNALU_HYPRE_PARCSRGMRESGETNUMITERATI)
extern void hypre_F90_NAME(fhypre_parcsrgmresgetnumiterati, FNALU_HYPRE_PARCSRGMRESGETNUMITERATI)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_ParCSRGMRESGetFinalRelativeResidualNorm  \
        hypre_F90_NAME(fhypre_parcsrgmresgetfinalrelat, FNALU_HYPRE_PARCSRGMRESGETFINALRELAT)
extern void hypre_F90_NAME(fhypre_parcsrgmresgetfinalrelat, FNALU_HYPRE_PARCSRGMRESGETFINALRELAT)
(hypre_F90_Obj *, NALU_HYPRE_Real *);



#define NALU_HYPRE_ParCSRCOGMRESCreate  \
        hypre_F90_NAME(fhypre_parcsrcogmrescreate, FNALU_HYPRE_PARCSRCOGMRESCREATE)
extern void hypre_F90_NAME(fhypre_parcsrcogmrescreate, FNALU_HYPRE_PARCSRCOGMRESCREATE)
(NALU_HYPRE_Int *, hypre_F90_Obj *);

#define NALU_HYPRE_ParCSRCOGMRESDestroy  \
        hypre_F90_NAME(fhypre_parcsrcogmresdestroy, FNALU_HYPRE_PARCSRCOGMRESDESTROY)
extern void hypre_F90_NAME(fhypre_parcsrcogmresdestroy, FNALU_HYPRE_PARCSRCOGMRESDESTROY)
(hypre_F90_Obj *);

#define NALU_HYPRE_ParCSRCOGMRESSetup  \
        hypre_F90_NAME(fhypre_parcsrcogmressetup, FNALU_HYPRE_PARCSRCOGMRESSETUP)
extern void hypre_F90_NAME(fhypre_parcsrcogmressetup, FNALU_HYPRE_PARCSRCOGMRESSETUP)
(hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *);

#define NALU_HYPRE_ParCSRCOGMRESSolve  \
        hypre_F90_NAME(fhypre_parcsrcogmressolve, FNALU_HYPRE_PARCSRCOGMRESSOLVE)
extern void hypre_F90_NAME(fhypre_parcsrcogmressolve, FNALU_HYPRE_PARCSRCOGMRESSOLVE)
(hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *);

#define NALU_HYPRE_ParCSRCOGMRESSetKDim  \
        hypre_F90_NAME(fhypre_parcsrcogmressetkdim, FNALU_HYPRE_PARCSRCOGMRESSETKDIM)
extern void hypre_F90_NAME(fhypre_parcsrcogmressetkdim, FNALU_HYPRE_PARCSRCOGMRESSETKDIM)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_ParCSRCOGMRESSetUnroll  \
        hypre_F90_NAME(fhypre_parcsrcogmressetunroll, FNALU_HYPRE_PARCSRCOGMRESSETUNROLL)
extern void hypre_F90_NAME(fhypre_parcsrcogmressetunroll, FNALU_HYPRE_PARCSRCOGMRESSETUNROLL)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_ParCSRCOGMRESSetCGS  \
        hypre_F90_NAME(fhypre_parcsrcogmressetcgs, FNALU_HYPRE_PARCSRCOGMRESSETCGS)
extern void hypre_F90_NAME(fhypre_parcsrcogmressetcgs, FNALU_HYPRE_PARCSRCOGMRESSETCGS)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_ParCSRCOGMRESSetTol  \
        hypre_F90_NAME(fhypre_parcsrcogmressettol, FNALU_HYPRE_PARCSRCOGMRESSETTOL)
extern void hypre_F90_NAME(fhypre_parcsrcogmressettol, FNALU_HYPRE_PARCSRCOGMRESSETTOL)
(hypre_F90_Obj *, NALU_HYPRE_Real *);

#define NALU_HYPRE_ParCSRCOGMRESSetAbsoluteTol  \
        hypre_F90_NAME(fhypre_parcsrcogmressetabsolutet, FNALU_HYPRE_PARCSRCOGMRESSETABSOLUTET)
extern void hypre_F90_NAME(fhypre_parcsrcogmressetabsolutet, FNALU_HYPRE_PARCSRCOGMRESSETABSOLUTET)
(hypre_F90_Obj *, NALU_HYPRE_Real *);

#define NALU_HYPRE_ParCSRCOGMRESSetMinIter  \
        hypre_F90_NAME(fhypre_parcsrcogmressetminiter, FNALU_HYPRE_PARCSRCOGMRESSETMINITER)
extern void hypre_F90_NAME(fhypre_parcsrcogmressetminiter, FNALU_HYPRE_PARCSRCOGMRESSETMINITER)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_ParCSRCOGMRESSetMaxIter  \
        hypre_F90_NAME(fhypre_parcsrcogmressetmaxiter, FNALU_HYPRE_PARCSRCOGMRESSETMAXITER)
extern void hypre_F90_NAME(fhypre_parcsrcogmressetmaxiter, FNALU_HYPRE_PARCSRCOGMRESSETMAXITER)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_ParCSRCOGMRESSetPrecond  \
        hypre_F90_NAME(fhypre_parcsrcogmressetprecond, FNALU_HYPRE_PARCSRCOGMRESSETPRECOND)
extern void hypre_F90_NAME(fhypre_parcsrcogmressetprecond, FNALU_HYPRE_PARCSRCOGMRESSETPRECOND)
(hypre_F90_Obj *, NALU_HYPRE_Int *, hypre_F90_Obj *);

#define NALU_HYPRE_ParCSRCOGMRESGetPrecond  \
        hypre_F90_NAME(fhypre_parcsrcogmresgetprecond, FNALU_HYPRE_PARCSRCOGMRESGETPRECOND)
extern void hypre_F90_NAME(fhypre_parcsrcogmresgetprecond, FNALU_HYPRE_PARCSRCOGMRESGETPRECOND)
(hypre_F90_Obj *, hypre_F90_Obj *);

#define NALU_HYPRE_ParCSRCOGMRESSetLogging  \
        hypre_F90_NAME(fhypre_parcsrcogmressetlogging, FNALU_HYPRE_PARCSRCOGMRESSETLOGGING)
extern void hypre_F90_NAME(fhypre_parcsrcogmressetlogging, FNALU_HYPRE_PARCSRCOGMRESSETLOGGING)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_ParCSRCOGMRESSetPrintLevel  \
        hypre_F90_NAME(fhypre_parcsrcogmressetprintlevel, FNALU_HYPRE_PARCSRCOGMRESSETPRINTLEVEL)
extern void hypre_F90_NAME(fhypre_parcsrcogmressetprintlevel, FNALU_HYPRE_PARCSRCOGMRESSETPRINTLEVEL)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_ParCSRCOGMRESGetNumIterations  \
        hypre_F90_NAME(fhypre_parcsrcogmresgetnumiterat, FNALU_HYPRE_PARCSRCOGMRESGETNUMITERAT)
extern void hypre_F90_NAME(fhypre_parcsrcogmresgetnumiterat, FNALU_HYPRE_PARCSRCOGMRESGETNUMITERAT)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_ParCSRCOGMRESGetFinalRelativeResidualNorm  \
        hypre_F90_NAME(fhypre_parcsrcogmresgetfinalrela, FNALU_HYPRE_PARCSRCOGMRESGETFINALRELA)
extern void hypre_F90_NAME(fhypre_parcsrcogmresgetfinalrela, FNALU_HYPRE_PARCSRCOGMRESGETFINALRELA)
(hypre_F90_Obj *, NALU_HYPRE_Real *);



#define NALU_HYPRE_ParCSRHybridCreate \
        hypre_F90_NAME(fhypre_parcsrhybridcreate, FNALU_HYPRE_PARCSRHYBRIDCREATE)
extern void hypre_F90_NAME(fhypre_parcsrhybridcreate, FNALU_HYPRE_PARCSRHYBRIDCREATE)
(hypre_F90_Obj *);

#define NALU_HYPRE_ParCSRHybridDestroy \
        hypre_F90_NAME(fhypre_parcsrhybriddestroy, FNALU_HYPRE_PARCSRHYBRIDDESTROY)
extern void hypre_F90_NAME(fhypre_parcsrhybriddestroy, FNALU_HYPRE_PARCSRHYBRIDDESTROY)
(hypre_F90_Obj *);

#define NALU_HYPRE_ParCSRHybridSetup \
        hypre_F90_NAME(fhypre_parcsrhybridsetup, FNALU_HYPRE_PARCSRHYBRIDSETUP)
extern void hypre_F90_NAME(fhypre_parcsrhybridsetup, FNALU_HYPRE_PARCSRHYBRIDSETUP)
(hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *);

#define NALU_HYPRE_ParCSRHybridSolve \
        hypre_F90_NAME(fhypre_parcsrhybridsolve, FNALU_HYPRE_PARCSRHYBRIDSOLVE)
extern void hypre_F90_NAME(fhypre_parcsrhybridsolve, FNALU_HYPRE_PARCSRHYBRIDSOLVE)
(hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *);

#define NALU_HYPRE_ParCSRHybridSetTol \
        hypre_F90_NAME(fhypre_parcsrhybridsettol, FNALU_HYPRE_PARCSRHYBRIDSETTOL)
extern void hypre_F90_NAME(fhypre_parcsrhybridsettol, FNALU_HYPRE_PARCSRHYBRIDSETTOL)
(hypre_F90_Obj *, NALU_HYPRE_Real *);

#define NALU_HYPRE_ParCSRHybridSetConvergenceTol \
        hypre_F90_NAME(fhypre_parcsrhybridsetconvergen, FNALU_HYPRE_PARCSRHYBRIDSETCONVERGEN)
extern void hypre_F90_NAME(fhypre_parcsrhybridsetconvergen, FNALU_HYPRE_PARCSRHYBRIDSETCONVERGEN)
(hypre_F90_Obj *, NALU_HYPRE_Real *);

#define NALU_HYPRE_ParCSRHybridSetDSCGMaxIter \
        hypre_F90_NAME(fhypre_parcsrhybridsetdscgmaxit, FNALU_HYPRE_PARCSRHYBRIDSETDSCGMAXIT)
extern void hypre_F90_NAME(fhypre_parcsrhybridsetdscgmaxit, FNALU_HYPRE_PARCSRHYBRIDSETDSCGMAXIT)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_ParCSRHybridSetPCGMaxIter \
        hypre_F90_NAME(fhypre_parcsrhybridsetpcgmaxite, FNALU_HYPRE_PARCSRHYBRIDSETPCGMAXITE)
extern void hypre_F90_NAME(fhypre_parcsrhybridsetpcgmaxite, FNALU_HYPRE_PARCSRHYBRIDSETPCGMAXITE)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_ParCSRHybridSetSolverType \
        hypre_F90_NAME(fhypre_parcsrhybridsetsolvertyp, FNALU_HYPRE_PARCSRHYBRIDSETSOLVERTYP)
extern void hypre_F90_NAME(fhypre_parcsrhybridsetsolvertyp, FNALU_HYPRE_PARCSRHYBRIDSETSOLVERTYP)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_ParCSRHybridSetKDim \
        hypre_F90_NAME(fhypre_parcsrhybridsetkdim, FNALU_HYPRE_PARCSRHYBRIDSETKDIM)
extern void hypre_F90_NAME(fhypre_parcsrhybridsetkdim, FNALU_HYPRE_PARCSRHYBRIDSETKDIM)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_ParCSRHybridSetTwoNorm \
        hypre_F90_NAME(fhypre_parcsrhybridsettwonorm, FNALU_HYPRE_PARCSRHYBRIDSETTWONORM)
extern void hypre_F90_NAME(fhypre_parcsrhybridsettwonorm, FNALU_HYPRE_PARCSRHYBRIDSETTWONORM)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_ParCSRHybridSetStopCrit \
        hypre_F90_NAME(fhypre_parcsrhybridsetstopcrit, FNALU_HYPRE_PARCSRSETSTOPCRIT)
extern void hypre_F90_NAME(fhypre_parcsrhybridsetstopcrit, FNALU_HYPRE_PARCSRSETSTOPCRIT)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_ParCSRHybridSetRelChange \
        hypre_F90_NAME(fhypre_parcsrhybridsetrelchange, FNALU_HYPRE_PARCSRHYBRIDSETRELCHANGE)
extern void hypre_F90_NAME(fhypre_parcsrhybridsetrelchange, FNALU_HYPRE_PARCSRHYBRIDSETRELCHANGE)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_ParCSRHybridSetPrecond \
        hypre_F90_NAME(fhypre_parcsrhybridsetprecond, FNALU_HYPRE_PARCSRHYBRIDSETPRECOND)
extern void hypre_F90_NAME(fhypre_parcsrhybridsetprecond, FNALU_HYPRE_PARCSRHYBRIDSETPRECOND)
(hypre_F90_Obj *, NALU_HYPRE_Int *, hypre_F90_Obj *);

#define NALU_HYPRE_ParCSRHybridSetLogging \
        hypre_F90_NAME(fhypre_parcsrhybridsetlogging, FNALU_HYPRE_PARCSRHYBRIDSETLOGGING)
extern void hypre_F90_NAME(fhypre_parcsrhybridsetlogging, FNALU_HYPRE_PARCSRHYBRIDSETLOGGING)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_ParCSRHybridSetPrintLevel \
        hypre_F90_NAME(fhypre_parcsrhybridsetprintleve, FNALU_HYPRE_PARCSRHYBRIDSETPRINTLEVE)
extern void hypre_F90_NAME(fhypre_parcsrhybridsetprintleve, FNALU_HYPRE_PARCSRHYBRIDSETPRINTLEVE)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_ParCSRHybridSetStrongThreshold \
        hypre_F90_NAME(fhypre_parcsrhybridsetstrongthr, FNALU_HYPRE_PARCSRHYBRIDSETSTRONGTHR)
extern void hypre_F90_NAME(fhypre_parcsrhybridsetstrongthr, FNALU_HYPRE_PARCSRHYBRIDSETSTRONGTHR)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_ParCSRHybridSetMaxRowSum \
        hypre_F90_NAME(fhypre_parcsrhybridsetmaxrowsum, FNALU_HYPRE_PARCSRHYBRIDSETMAXROWSUM)
extern void hypre_F90_NAME(fhypre_parcsrhybridsetmaxrowsum, FNALU_HYPRE_PARCSRHYBRIDSETMAXROWSUM)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_ParCSRHybridSetTruncFactor \
        hypre_F90_NAME(fhypre_parcsrhybridsettruncfact, FNALU_HYPRE_PARCSRHYBRIDSETTRUNCFACT)
extern void hypre_F90_NAME(fhypre_parcsrhybridsettruncfact, FNALU_HYPRE_PARCSRHYBRIDSETTRUNCFACT)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_ParCSRHybridSetMaxLevels \
        hypre_F90_NAME(fhypre_parcsrhybridsetmaxlevels, FNALU_HYPRE_PARCSRHYBRIDSETMAXLEVELS)
extern void hypre_F90_NAME(fhypre_parcsrhybridsetmaxlevels, FNALU_HYPRE_PARCSRHYBRIDSETMAXLEVELS)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_ParCSRHybridSetMeasureType \
        hypre_F90_NAME(fhypre_parcsrhybridsetmeasurety, FNALU_HYPRE_PARCSRHYBRIDSETMEASURETY)
extern void hypre_F90_NAME(fhypre_parcsrhybridsetmeasurety, FNALU_HYPRE_PARCSRHYBRIDSETMEASURETY)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_ParCSRHybridSetCoarsenType \
        hypre_F90_NAME(fhypre_parcsrhybridsetcoarsenty, FNALU_HYPRE_PARCSRHYBRIDSETCOARSENTY)
extern void hypre_F90_NAME(fhypre_parcsrhybridsetcoarsenty, FNALU_HYPRE_PARCSRHYBRIDSETCOARSENTY)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_ParCSRHybridSetCycleType \
        hypre_F90_NAME(fhypre_parcsrhybridsetcycletype, FNALU_HYPRE_PARCSRHYBRIDSETCYCLETYPE)
extern void hypre_F90_NAME(fhypre_parcsrhybridsetcycletype, FNALU_HYPRE_PARCSRHYBRIDSETCYCLETYPE)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_ParCSRHybridSetNumGridSweeps \
        hypre_F90_NAME(fhypre_parcsrhybridsetnumgridsw, FNALU_HYPRE_PARCSRHYBRIDSETNUMGRIDSW)
extern void hypre_F90_NAME(fhypre_parcsrhybridsetnumgridsw, FNALU_HYPRE_PARCSRHYBRIDSETNUMGRIDSW)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_ParCSRHybridSetGridRelaxType \
        hypre_F90_NAME(fhypre_parcsrhybridsetgridrlxty, FNALU_HYPRE_PARCSRHYBRIDSETGRIDRLXTY)
extern void hypre_F90_NAME(fhypre_parcsrhybridsetgridrlxty, FNALU_HYPRE_PARCSRHYBRIDSETGRIDRLXTY)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_ParCSRHybridSetGridRelaxPoints \
        hypre_F90_NAME(fhypre_parcsrhybridsetgridrlxpt, FNALU_HYPRE_PARCSRHYBRIDSETGRIDRLXPT)
extern void hypre_F90_NAME(fhypre_parcsrhybridsetgridrlxpt, FNALU_HYPRE_PARCSRHYBRIDSETGRIDRLXPT)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_ParCSRHybridSetNumSweeps \
        hypre_F90_NAME(fhypre_parcsrhybridsetnumsweeps, FNALU_HYPRE_PARCSRHYBRIDSETNUMSWEEPS)
extern void hypre_F90_NAME(fhypre_parcsrhybridsetnumsweeps, FNALU_HYPRE_PARCSRHYBRIDSETNUMSWEEPS)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_ParCSRHybridSetCycleNumSweeps \
        hypre_F90_NAME(fhypre_parcsrhybridsetcyclenums, FNALU_HYPRE_PARCSRHYBRIDSETCYCLENUMS)
extern void hypre_F90_NAME(fhypre_parcsrhybridsetcyclenums, FNALU_HYPRE_PARCSRHYBRIDSETCYCLENUMS)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_ParCSRHybridSetRelaxType \
        hypre_F90_NAME(fhypre_parcsrhybridsetrelaxtype, FNALU_HYPRE_PARCSRHYBRIDSETRELAXTYPE)
extern void hypre_F90_NAME(fhypre_parcsrhybridsetrelaxtype, FNALU_HYPRE_PARCSRHYBRIDSETRELAXTYPE)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_ParCSRHybridSetCycleRelaxType \
        hypre_F90_NAME(fhypre_parcsrhybridsetcyclerela, FNALU_HYPRE_PARCSRHYBRIDSETCYCLERELA)
extern void hypre_F90_NAME(fhypre_parcsrhybridsetcyclerela, FNALU_HYPRE_PARCSRHYBRIDSETCYCLERELA)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_ParCSRHybridSetRelaxOrder \
        hypre_F90_NAME(fhypre_parcsrhybridsetrelaxorde, FNALU_HYPRE_PARCSRHYBRIDSETRELAXORDE)
extern void hypre_F90_NAME(fhypre_parcsrhybridsetrelaxorde, FNALU_HYPRE_PARCSRHYBRIDSETRELAXORDE)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_ParCSRHybridSetRelaxWt \
        hypre_F90_NAME(fhypre_parcsrhybridsetrelaxwt, FNALU_HYPRE_PARCSRHYBRIDSETRELAXWT)
extern void hypre_F90_NAME(fhypre_parcsrhybridsetrelaxwt, FNALU_HYPRE_PARCSRHYBRIDSETRELAXWT)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_ParCSRHybridSetLevelRelaxWt \
        hypre_F90_NAME(fhypre_parcsrhybridsetlevelrela, FNALU_HYPRE_PARCSRHYBRIDSETLEVELRELA)
extern void hypre_F90_NAME(fhypre_parcsrhybridsetlevelrela, FNALU_HYPRE_PARCSRHYBRIDSETLEVELRELA)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_ParCSRHybridSetOuterWt \
        hypre_F90_NAME(fhypre_parcsrhybridsetouterwt, FNALU_HYPRE_PARCSRHYBRIDSETOUTERWT)
extern void hypre_F90_NAME(fhypre_parcsrhybridsetouterwt, FNALU_HYPRE_PARCSRHYBRIDSETOUTERWT)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_ParCSRHybridSetLevelOuterWt \
        hypre_F90_NAME(fhypre_parcsrhybridsetleveloute, FNALU_HYPRE_PARCSRHYBRIDSETLEVELOUTE)
extern void hypre_F90_NAME(fhypre_parcsrhybridsetleveloute, FNALU_HYPRE_PARCSRHYBRIDSETLEVELOUTE)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_ParCSRHybridSetRelaxWeight \
        hypre_F90_NAME(fhypre_parcsrhybridsetrelaxweig, FNALU_HYPRE_PARCSRHYBRIDSETRELAXWEIG)
extern void hypre_F90_NAME(fhypre_parcsrhybridsetrelaxweig, FNALU_HYPRE_PARCSRHYBRIDSETRELAXWEIG)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_ParCSRHybridSetOmega \
        hypre_F90_NAME(fhypre_parcsrhybridsetomega, FNALU_HYPRE_PARCSRHYBRIDSETOMEGA)
extern void hypre_F90_NAME(fhypre_parcsrhybridsetomega, FNALU_HYPRE_PARCSRHYBRIDSETOMEGA)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_ParCSRHybridGetNumIterations \
        hypre_F90_NAME(fhypre_parcsrhybridgetnumiterat, FNALU_HYPRE_PARCSRHYBRIDGETNUMITERAT)
extern void hypre_F90_NAME(fhypre_parcsrhybridgetnumiterat, FNALU_HYPRE_PARCSRHYBRIDGETNUMITERAT)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_ParCSRHybridGetDSCGNumIterations \
        hypre_F90_NAME(fhypre_parcsrhybridgetdscgnumit, FNALU_HYPRE_PARCSRHYBRIDGETDSCGNUMIT)
extern void hypre_F90_NAME(fhypre_parcsrhybridgetdscgnumit, FNALU_HYPRE_PARCSRHYBRIDGETDSCGNUMIT)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_ParCSRHybridGetPCGNumIterations \
        hypre_F90_NAME(fhypre_parcsrhybridgetpcgnumite, FNALU_HYPRE_PARCSRHYBRIDGETPCGNUMITE)
extern void hypre_F90_NAME(fhypre_parcsrhybridgetpcgnumite, FNALU_HYPRE_PARCSRHYBRIDGETPCGNUMITE)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_ParCSRHybridGetFinalRelativeResidualNorm \
        hypre_F90_NAME(fhypre_parcsrhybridgetfinalrela, FNALU_HYPRE_PARCSRHYBRIDGETFINALRELA)
extern void hypre_F90_NAME(fhypre_parcsrhybridgetfinalrela, FNALU_HYPRE_PARCSRHYBRIDGETFINALRELA)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_ParSetRandomValues \
        hypre_F90_NAME(fhypre_parsetrandomvalues, FNALU_HYPRE_PARSETRANDOMVALUES)
extern void hypre_F90_NAME(fhypre_parsetrandomvalues, FNALU_HYPRE_PARSETRANDOMVALUES)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_ParPrintVector \
        hypre_F90_NAME(fhypre_parprintvector, FNALU_HYPRE_PARPRINTVECTOR)
extern void hypre_F90_NAME(fhypre_parprintvector, FNALU_HYPRE_PARPRINTVECTOR)
(hypre_F90_Obj *, char *);

#define NALU_HYPRE_ParReadVector \
        hypre_F90_NAME(fhypre_parreadvector, FNALU_HYPRE_PARREADVECTOR)
extern void hypre_F90_NAME(fhypre_parreadvector, FNALU_HYPRE_PARREADVECTOR)
(NALU_HYPRE_Int *, char *);

#define NALU_HYPRE_ParVectorSize \
        hypre_F90_NAME(fhypre_parvectorsize, FNALU_HYPRE_PARVECTORSIZE)
extern void hypre_F90_NAME(fhypre_parvectorsize, FNALU_HYPRE_PARVECTORSIZE)
(NALU_HYPRE_Int *);

#define NALU_HYPRE_ParCSRMultiVectorPrint \
        hypre_F90_NAME(fhypre_parcsrmultivectorprint, FNALU_HYPRE_PARCSRMULTIVECTORPRINT)
extern void hypre_F90_NAME(fhypre_parcsrmultivectorprint, FNALU_HYPRE_PARCSRMULTIVECTORPRINT)
(NALU_HYPRE_Int *, char *);

#define NALU_HYPRE_ParCSRMultiVectorRead \
        hypre_F90_NAME(fhypre_parcsrmultivectorread, FNALU_HYPRE_PARCSRMULTIVECTORREAD)
extern void hypre_F90_NAME(fhypre_parcsrmultivectorread, FNALU_HYPRE_PARCSRMULTIVECTORREAD)
(NALU_HYPRE_Int *, hypre_F90_Obj *, char *);

#define aux_maskCount \
        hypre_F90_NAME(fhypre_aux_maskcount, FNALU_HYPRE_AUX_MASKCOUNT)
extern void hypre_F90_NAME(fhypre_aux_maskcount, FNALU_HYPRE_AUX_MASKCOUNT)
(NALU_HYPRE_Int *, NALU_HYPRE_Int *);

#define aux_indexFromMask \
        hypre_F90_NAME(fhypre_auxindexfrommask, FNALU_HYPRE_AUXINDEXFROMMASK)
extern void hypre_F90_NAME(fhypre_auxindexfrommask, FNALU_HYPRE_AUXINDEXFROMMASK)
(NALU_HYPRE_Int *, NALU_HYPRE_Int *, NALU_HYPRE_Int *);

#define NALU_HYPRE_TempParCSRSetupInterpreter \
        hypre_F90_NAME(fhypre_tempparcsrsetupinterpret, FNALU_HYPRE_TEMPPARCSRSETUPINTERPRET)
extern void hypre_F90_NAME(fhypre_tempparcsrsetupinterpret, FNALU_HYPRE_TEMPPARCSRSETUPINTERPRET)
(hypre_F90_Obj *);

#define NALU_HYPRE_ParCSRSetupInterpreter \
        hypre_F90_NAME(fhypre_parcsrsetupinterpreter, FNALU_HYPRE_PARCSRSETUPINTERPRETER)
extern void hypre_F90_NAME(fhypre_parcsrsetupinterpreter, FNALU_HYPRE_PARCSRSETUPINTERPRETER)
(hypre_F90_Obj *);

#define NALU_HYPRE_ParCSRSetupMatvec \
        hypre_F90_NAME(fhypre_parcsrsetupmatvec, FNALU_HYPRE_PARCSRSETUPMATVEC)
extern void hypre_F90_NAME(fhypre_parcsrsetupmatvec, FNALU_HYPRE_PARCSRSETUPMATVEC)
(hypre_F90_Obj *);



#define NALU_HYPRE_ParaSailsCreate  \
        hypre_F90_NAME(fhypre_parasailscreate, FNALU_HYPRE_PARASAILSCREATE)
extern void hypre_F90_NAME(fhypre_parasailscreate, FNALU_HYPRE_PARASAILSCREATE)
(NALU_HYPRE_Int *, hypre_F90_Obj *);

#define NALU_HYPRE_ParaSailsDestroy  \
        hypre_F90_NAME(fhypre_parasailsdestroy, FNALU_HYPRE_PARASAILSDESTROY)
extern void hypre_F90_NAME(fhypre_parasailsdestroy, FNALU_HYPRE_PARASAILSDESTROY)
(hypre_F90_Obj *);

#define NALU_HYPRE_ParaSailsSetup  \
        hypre_F90_NAME(fhypre_parasailssetup, FNALU_HYPRE_PARASAILSSETUP)
extern void hypre_F90_NAME(fhypre_parasailssetup, FNALU_HYPRE_PARASAILSSETUP)
(hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *);

#define NALU_HYPRE_ParaSailsSolve  \
        hypre_F90_NAME(fhypre_parasailssolve, FNALU_HYPRE_PARASAILSSOLVE)
extern void hypre_F90_NAME(fhypre_parasailssolve, FNALU_HYPRE_PARASAILSSOLVE)
(hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *);

#define NALU_HYPRE_ParaSailsSetParams  \
        hypre_F90_NAME(fhypre_parasailssetparams, FNALU_HYPRE_PARASAILSSETPARAMS)
extern void hypre_F90_NAME(fhypre_parasailssetparams, FNALU_HYPRE_PARASAILSSETPARAMS)
(hypre_F90_Obj *, NALU_HYPRE_Real *, NALU_HYPRE_Int *);

#define NALU_HYPRE_ParaSailsSetThresh  \
        hypre_F90_NAME(fhypre_parasailssetthresh, FNALU_HYPRE_PARASAILSSETTHRESH)
extern void hypre_F90_NAME(fhypre_parasailssetthresh, FNALU_HYPRE_PARASAILSSETTHRESH)
(hypre_F90_Obj *, NALU_HYPRE_Real *);

#define NALU_HYPRE_ParaSailsGetThresh  \
        hypre_F90_NAME(fhypre_parasailsgetthresh, FNALU_HYPRE_PARASAILSGETTHRESH)
extern void hypre_F90_NAME(fhypre_parasailsgetthresh, FNALU_HYPRE_PARASAILSGETTHRESH)
(hypre_F90_Obj *, NALU_HYPRE_Real *);

#define NALU_HYPRE_ParaSailsSetNlevels  \
        hypre_F90_NAME(fhypre_parasailssetnlevels, FNALU_HYPRE_PARASAILSSETNLEVELS)
extern void hypre_F90_NAME(fhypre_parasailssetnlevels, FNALU_HYPRE_PARASAILSSETNLEVELS)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_ParaSailsGetNlevels  \
        hypre_F90_NAME(fhypre_parasailsgetnlevels, FNALU_HYPRE_PARASAILSGETNLEVELS)
extern void hypre_F90_NAME(fhypre_parasailsgetnlevels, FNALU_HYPRE_PARASAILSGETNLEVELS)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_ParaSailsSetFilter  \
        hypre_F90_NAME(fhypre_parasailssetfilter, FNALU_HYPRE_PARASAILSSETFILTER)
extern void hypre_F90_NAME(fhypre_parasailssetfilter, FNALU_HYPRE_PARASAILSSETFILTER)
(hypre_F90_Obj *, NALU_HYPRE_Real *);

#define NALU_HYPRE_ParaSailsGetFilter  \
        hypre_F90_NAME(fhypre_parasailsgetfilter, FNALU_HYPRE_PARASAILSGETFILTER)
extern void hypre_F90_NAME(fhypre_parasailsgetfilter, FNALU_HYPRE_PARASAILSGETFILTER)
(hypre_F90_Obj *, NALU_HYPRE_Real *);

#define NALU_HYPRE_ParaSailsSetSym  \
        hypre_F90_NAME(fhypre_parasailssetsym, FNALU_HYPRE_PARASAILSSETSYM)
extern void hypre_F90_NAME(fhypre_parasailssetsym, FNALU_HYPRE_PARASAILSSETSYM)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_ParaSailsGetSym  \
        hypre_F90_NAME(fhypre_parasailsgetsym, FNALU_HYPRE_PARASAILSGETSYM)
extern void hypre_F90_NAME(fhypre_parasailsgetsym, FNALU_HYPRE_PARASAILSGETSYM)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_ParaSailsSetLoadbal  \
        hypre_F90_NAME(fhypre_parasailssetloadbal, FNALU_HYPRE_PARASAILSSETLOADBAL)
extern void hypre_F90_NAME(fhypre_parasailssetloadbal, FNALU_HYPRE_PARASAILSSETLOADBAL)
(hypre_F90_Obj *, NALU_HYPRE_Real *);

#define NALU_HYPRE_ParaSailsGetLoadbal  \
        hypre_F90_NAME(fhypre_parasailsgetloadbal, FNALU_HYPRE_PARASAILSGETLOADBAL)
extern void hypre_F90_NAME(fhypre_parasailsgetloadbal, FNALU_HYPRE_PARASAILSGETLOADBAL)
(hypre_F90_Obj *, NALU_HYPRE_Real *);

#define NALU_HYPRE_ParaSailsSetReuse  \
        hypre_F90_NAME(fhypre_parasailssetreuse, FNALU_HYPRE_PARASAILSSETREUSE)
extern void hypre_F90_NAME(fhypre_parasailssetreuse, FNALU_HYPRE_PARASAILSSETREUSE)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_ParaSailsGetReuse  \
        hypre_F90_NAME(fhypre_parasailsgetreuse, FNALU_HYPRE_PARASAILSGETREUSE)
extern void hypre_F90_NAME(fhypre_parasailsgetreuse, FNALU_HYPRE_PARASAILSGETREUSE)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_ParaSailsSetLogging  \
        hypre_F90_NAME(fhypre_parasailssetlogging, FNALU_HYPRE_PARASAILSSETLOGGING)
extern void hypre_F90_NAME(fhypre_parasailssetlogging, FNALU_HYPRE_PARASAILSSETLOGGING)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_ParaSailsGetLogging  \
        hypre_F90_NAME(fhypre_parasailsgetlogging, FNALU_HYPRE_PARASAILSGETLOGGING)
extern void hypre_F90_NAME(fhypre_parasailsgetlogging, FNALU_HYPRE_PARASAILSGETLOGGING)
(hypre_F90_Obj *, NALU_HYPRE_Int *);



#define NALU_HYPRE_ParCSRPCGCreate  \
        hypre_F90_NAME(fhypre_parcsrpcgcreate, FNALU_HYPRE_PARCSRPCGCREATE)
extern void hypre_F90_NAME(fhypre_parcsrpcgcreate, FNALU_HYPRE_PARCSRPCGCREATE)
(NALU_HYPRE_Int *, hypre_F90_Obj *);

#define NALU_HYPRE_ParCSRPCGDestroy  \
        hypre_F90_NAME(fhypre_parcsrpcgdestroy, FNALU_HYPRE_PARCSRPCGDESTROY)
extern void hypre_F90_NAME(fhypre_parcsrpcgdestroy, FNALU_HYPRE_PARCSRPCGDESTROY)
(hypre_F90_Obj *);

#define NALU_HYPRE_ParCSRPCGSetup  \
        hypre_F90_NAME(fhypre_parcsrpcgsetup, FNALU_HYPRE_PARCSRPCGSETUP)
extern void hypre_F90_NAME(fhypre_parcsrpcgsetup, FNALU_HYPRE_PARCSRPCGSETUP)
(hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *);

#define NALU_HYPRE_ParCSRPCGSolve  \
        hypre_F90_NAME(fhypre_parcsrpcgsolve, FNALU_HYPRE_PARCSRPCGSOLVE)
extern void hypre_F90_NAME(fhypre_parcsrpcgsolve, FNALU_HYPRE_PARCSRPCGSOLVE)
(hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *);

#define NALU_HYPRE_ParCSRPCGSetTol  \
        hypre_F90_NAME(fhypre_parcsrpcgsettol, FNALU_HYPRE_PARCSRPCGSETTOL)
extern void hypre_F90_NAME(fhypre_parcsrpcgsettol, FNALU_HYPRE_PARCSRPCGSETTOL)
(hypre_F90_Obj *, NALU_HYPRE_Real *);

#define NALU_HYPRE_ParCSRPCGSetMaxIter  \
        hypre_F90_NAME(fhypre_parcsrpcgsetmaxiter, FNALU_HYPRE_PARCSRPCGSETMAXITER)
extern void hypre_F90_NAME(fhypre_parcsrpcgsetmaxiter, FNALU_HYPRE_PARCSRPCGSETMAXITER)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_ParCSRPCGSetStopCrit  \
        hypre_F90_NAME(fhypre_parcsrpcgsetstopcrit, FNALU_HYPRE_PARCSRPCGSETSTOPCRIT)
extern void hypre_F90_NAME(fhypre_parcsrpcgsetstopcrit, FNALU_HYPRE_PARCSRPCGSETSTOPCRIT)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_ParCSRPCGSetTwoNorm  \
        hypre_F90_NAME(fhypre_parcsrpcgsettwonorm, FNALU_HYPRE_PARCSRPCGSETTWONORM)
extern void hypre_F90_NAME(fhypre_parcsrpcgsettwonorm, FNALU_HYPRE_PARCSRPCGSETTWONORM)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_ParCSRPCGSetRelChange  \
        hypre_F90_NAME(fhypre_parcsrpcgsetrelchange, FNALU_HYPRE_PARCSRPCGSETRELCHANGE)
extern void hypre_F90_NAME(fhypre_parcsrpcgsetrelchange, FNALU_HYPRE_PARCSRPCGSETRELCHANGE)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_ParCSRPCGSetPrecond  \
        hypre_F90_NAME(fhypre_parcsrpcgsetprecond, FNALU_HYPRE_PARCSRPCGSETPRECOND)
extern void hypre_F90_NAME(fhypre_parcsrpcgsetprecond, FNALU_HYPRE_PARCSRPCGSETPRECOND)
(hypre_F90_Obj *, NALU_HYPRE_Int *, hypre_F90_Obj *);

#define NALU_HYPRE_ParCSRPCGGetPrecond  \
        hypre_F90_NAME(fhypre_parcsrpcggetprecond, FNALU_HYPRE_PARCSRPCGGETPRECOND)
extern void hypre_F90_NAME(fhypre_parcsrpcggetprecond, FNALU_HYPRE_PARCSRPCGGETPRECOND)
(hypre_F90_Obj *, hypre_F90_Obj *);

#define NALU_HYPRE_ParCSRPCGSetPrintLevel  \
        hypre_F90_NAME(fhypre_parcsrpcgsetprintlevel, FNALU_HYPRE_PARCSRPCGSETPRINTLEVEL)
extern void hypre_F90_NAME(fhypre_parcsrpcgsetprintlevel, FNALU_HYPRE_PARCSRPCGSETPRINTLEVEL)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_ParCSRPCGGetNumIterations  \
        hypre_F90_NAME(fhypre_parcsrpcggetnumiteration, FNALU_HYPRE_PARCSRPCGGETNUMITERATION)
extern void hypre_F90_NAME(fhypre_parcsrpcggetnumiteration, FNALU_HYPRE_PARCSRPCGGETNUMITERATION)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_ParCSRPCGGetFinalRelativeResidualNorm  \
        hypre_F90_NAME(fhypre_parcsrpcggetfinalrelativ, FNALU_HYPRE_PARCSRPCGGETFINALRELATIV)
extern void hypre_F90_NAME(fhypre_parcsrpcggetfinalrelativ, FNALU_HYPRE_PARCSRPCGGETFINALRELATIV)
(hypre_F90_Obj *, NALU_HYPRE_Real *);



#define NALU_HYPRE_ParCSRDiagScaleSetup  \
        hypre_F90_NAME(fhypre_parcsrdiagscalesetup, FNALU_HYPRE_PARCSRDIAGSCALESETUP)
extern void hypre_F90_NAME(fhypre_parcsrdiagscalesetup, FNALU_HYPRE_PARCSRDIAGSCALESETUP)
(hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *);

#define NALU_HYPRE_ParCSRDiagScale  \
        hypre_F90_NAME(fhypre_parcsrdiagscale, FNALU_HYPRE_PARCSRDIAGSCALE)
extern void hypre_F90_NAME(fhypre_parcsrdiagscale, FNALU_HYPRE_PARCSRDIAGSCALE)
(hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *);



#define NALU_HYPRE_ParCSRPilutCreate  \
        hypre_F90_NAME(fhypre_parcsrpilutcreate, FNALU_HYPRE_PARCSRPILUTCREATE)
extern void hypre_F90_NAME(fhypre_parcsrpilutcreate, FNALU_HYPRE_PARCSRPILUTCREATE)
(NALU_HYPRE_Int *, hypre_F90_Obj *);

#define NALU_HYPRE_ParCSRPilutDestroy  \
        hypre_F90_NAME(fhypre_parcsrpilutdestroy, FNALU_HYPRE_PARCSRPILUTDESTROY)
extern void hypre_F90_NAME(fhypre_parcsrpilutdestroy, FNALU_HYPRE_PARCSRPILUTDESTROY)
(hypre_F90_Obj *);

#define NALU_HYPRE_ParCSRPilutSetup  \
        hypre_F90_NAME(fhypre_parcsrpilutsetup, FNALU_HYPRE_PARCSRPILUTSETUP)
extern void hypre_F90_NAME(fhypre_parcsrpilutsetup, FNALU_HYPRE_PARCSRPILUTSETUP)
(hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *);

#define NALU_HYPRE_ParCSRPilutSolve  \
        hypre_F90_NAME(fhypre_parcsrpilutsolve, FNALU_HYPRE_PARCSRPILUTSOLVE)
extern void hypre_F90_NAME(fhypre_parcsrpilutsolve, FNALU_HYPRE_PARCSRPILUTSOLVE)
(hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *);

#define NALU_HYPRE_ParCSRPilutSetMaxIter  \
        hypre_F90_NAME(fhypre_parcsrpilutsetmaxiter, FNALU_HYPRE_PARCSRPILUTSETMAXITER)
extern void hypre_F90_NAME(fhypre_parcsrpilutsetmaxiter, FNALU_HYPRE_PARCSRPILUTSETMAXITER)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_ParCSRPilutSetDropToleran  \
        hypre_F90_NAME(fhypre_parcsrpilutsetdroptolera, FNALU_HYPRE_PARCSRPILUTSETDROPTOLERA)
extern void hypre_F90_NAME(fhypre_parcsrpilutsetdroptolera, FNALU_HYPRE_PARCSRPILUTSETDROPTOLERA)
(hypre_F90_Obj *, NALU_HYPRE_Real *);

#define NALU_HYPRE_ParCSRPilutSetFacRowSize  \
        hypre_F90_NAME(fhypre_parcsrpilutsetfacrowsize, FNALU_HYPRE_PARCSRPILUTSETFACROWSIZE)
extern void hypre_F90_NAME(fhypre_parcsrpilutsetfacrowsize, FNALU_HYPRE_PARCSRPILUTSETFACROWSIZE)
(hypre_F90_Obj *, NALU_HYPRE_Int *);



#define NALU_HYPRE_SchwarzCreate \
        hypre_F90_NAME(fhypre_schwarzcreate, FNALU_HYPRE_SCHWARZCREATE)
extern void hypre_F90_NAME(fhypre_schwarzcreate, FNALU_HYPRE_SCHWARZCREATE)
(hypre_F90_Obj *);

#define NALU_HYPRE_SchwarzDestroy \
        hypre_F90_NAME(fhypre_schwarzdestroy, FNALU_HYPRE_SCHWARZDESTROY)
extern void hypre_F90_NAME(fhypre_schwarzdestroy, FNALU_HYPRE_SCHWARZDESTROY)
(hypre_F90_Obj *);

#define NALU_HYPRE_SchwarzSetup \
        hypre_F90_NAME(fhypre_schwarzsetup, FNALU_HYPRE_SCHWARZSETUP)
extern void hypre_F90_NAME(fhypre_schwarzsetup, FNALU_HYPRE_SCHWARZSETUP)
(hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj*);

#define NALU_HYPRE_SchwarzSolve \
        hypre_F90_NAME(fhypre_schwarzsolve, FNALU_HYPRE_SCHWARZSOLVE)
extern void hypre_F90_NAME(fhypre_schwarzsolve, FNALU_HYPRE_SCHWARZSOLVE)
(hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj*);

#define NALU_HYPRE_SchwarzSetVariant \
        hypre_F90_NAME(fhypre_schwarzsetvariant, FNALU_HYPRE_SCHWARZVARIANT)
extern void hypre_F90_NAME(fhypre_schwarzsetvariant, FNALU_HYPRE_SCHWARZVARIANT)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_SchwarzSetOverlap \
        hypre_F90_NAME(fhypre_schwarzsetoverlap, FNALU_HYPRE_SCHWARZOVERLAP)
extern void hypre_F90_NAME(fhypre_schwarzsetoverlap, FNALU_HYPRE_SCHWARZOVERLAP)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_SchwarzSetDomainType \
        hypre_F90_NAME(fhypre_schwarzsetdomaintype, FNALU_HYPRE_SVHWARZSETDOMAINTYPE)
extern void hypre_F90_NAME(fhypre_schwarzsetdomaintype, FNALU_HYPRE_SVHWARZSETDOMAINTYPE)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_SchwarzSetDomainStructure \
        hypre_F90_NAME(fhypre_schwarzsetdomainstructur, FNALU_HYPRE_SCHWARZSETDOMAINSTRUCTUR)
extern void hypre_F90_NAME(fhypre_schwarzsetdomainstructur, FNALU_HYPRE_SCHWARZSETDOMAINSTRUCTUR)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_SchwarzSetNumFunctions \
        hypre_F90_NAME(fhypre_schwarzsetnumfunctions, FNALU_HYPRE_SCHWARZSETNUMFUNCTIONS)
extern void hypre_F90_NAME(fhypre_schwarzsetnumfunctions, FNALU_HYPRE_SCHWARZSETNUMFUNCTIONS)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_SchwarzSetRelaxWeight \
        hypre_F90_NAME(fhypre_schwarzsetrelaxweight, FNALU_HYPRE_SCHWARZSETRELAXWEIGHT)
extern void hypre_F90_NAME(fhypre_schwarzsetrelaxweight, FNALU_HYPRE_SCHWARZSETRELAXWEIGHT)
(hypre_F90_Obj *, NALU_HYPRE_Real *);

#define NALU_HYPRE_SchwarzSetDofFunc \
        hypre_F90_NAME(fhypre_schwarzsetdoffunc, FNALU_HYPRE_SCHWARZSETDOFFUNC)
extern void hypre_F90_NAME(fhypre_schwarzsetdoffunc, FNALU_HYPRE_SCHWARZSETDOFFUNC)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#ifdef __cplusplus
}
#endif
