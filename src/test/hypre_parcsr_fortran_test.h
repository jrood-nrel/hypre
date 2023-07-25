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
        nalu_hypre_F90_NAME(fnalu_hypre_parcsrmatrixcreate, FNALU_HYPRE_PARCSRMATRIXCREATE)
extern void nalu_hypre_F90_NAME(fnalu_hypre_parcsrmatrixcreate, FNALU_HYPRE_PARCSRMATRIXCREATE)
(NALU_HYPRE_Int *, NALU_HYPRE_Int *, NALU_HYPRE_Int *, NALU_HYPRE_Int *, NALU_HYPRE_Int *, NALU_HYPRE_Int *, NALU_HYPRE_Int *,
 NALU_HYPRE_Int *, nalu_hypre_F90_Obj *);

#define NALU_HYPRE_ParCSRMatrixDestroy  \
        nalu_hypre_F90_NAME(fnalu_hypre_parcsrmatrixdestroy, FNALU_HYPRE_PARCSRMATRIXDESTROY)
extern void nalu_hypre_F90_NAME(fnalu_hypre_parcsrmatrixdestroy, FNALU_HYPRE_PARCSRMATRIXDESTROY)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_ParCSRMatrixInitialize  \
        nalu_hypre_F90_NAME(fnalu_hypre_parcsrmatrixinitialize, FNALU_HYPRE_PARCSRMATRIXINITIALIZE)
extern void nalu_hypre_F90_NAME(fnalu_hypre_parcsrmatrixinitialize, FNALU_HYPRE_PARCSRMATRIXINITIALIZE)
(nalu_hypre_F90_Obj *);

#define NALU_HYPRE_ParCSRMatrixRead  \
        nalu_hypre_F90_NAME(fnalu_hypre_parcsrmatrixread, FNALU_HYPRE_PARCSRMATRIXREAD)
extern void nalu_hypre_F90_NAME(fnalu_hypre_parcsrmatrixread, FNALU_HYPRE_PARCSRMATRIXREAD)
(NALU_HYPRE_Int *, char *, nalu_hypre_F90_Obj *);

#define NALU_HYPRE_ParCSRMatrixPrint  \
        nalu_hypre_F90_NAME(fnalu_hypre_parcsrmatrixprint, FNALU_HYPRE_PARCSRMATRIXPRINT)
extern void nalu_hypre_F90_NAME(fnalu_hypre_parcsrmatrixprint, FNALU_HYPRE_PARCSRMATRIXPRINT)
(nalu_hypre_F90_Obj *, char *, NALU_HYPRE_Int *);

#define NALU_HYPRE_ParCSRMatrixGetComm  \
        nalu_hypre_F90_NAME(fnalu_hypre_parcsrmatrixgetcomm, FNALU_HYPRE_PARCSRMATRIXGETCOMM)
extern void nalu_hypre_F90_NAME(fnalu_hypre_parcsrmatrixgetcomm, FNALU_HYPRE_PARCSRMATRIXGETCOMM)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_ParCSRMatrixGetDims  \
        nalu_hypre_F90_NAME(fnalu_hypre_parcsrmatrixgetdims, FNALU_HYPRE_PARCSRMATRIXGETDIMS)
extern void nalu_hypre_F90_NAME(fnalu_hypre_parcsrmatrixgetdims, FNALU_HYPRE_PARCSRMATRIXGETDIMS)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *, NALU_HYPRE_Int *);

#define NALU_HYPRE_ParCSRMatrixGetRowPartitioning  \
        nalu_hypre_F90_NAME(fnalu_hypre_parcsrmatrixgetrowpartit, FNALU_HYPRE_PARCSRMATRIXGETROWPARTIT)
extern void nalu_hypre_F90_NAME(fnalu_hypre_parcsrmatrixgetrowpartit, FNALU_HYPRE_PARCSRMATRIXGETROWPARTIT)
(nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *);

#define NALU_HYPRE_ParCSRMatrixGetColPartitioning  \
        nalu_hypre_F90_NAME(fnalu_hypre_parcsrmatrixgetcolpartit, FNALU_HYPRE_PARCSRMATRIXGETCOLPARTIT)
extern void nalu_hypre_F90_NAME(fnalu_hypre_parcsrmatrixgetcolpartit, FNALU_HYPRE_PARCSRMATRIXGETCOLPARTIT)
(nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *);

#define NALU_HYPRE_ParCSRMatrixGetLocalRange  \
        nalu_hypre_F90_NAME(fnalu_hypre_parcsrmatrixgetlocalrang, FNALU_HYPRE_PARCSRMATRIXGETLOCALRANG)
extern void nalu_hypre_F90_NAME(fnalu_hypre_parcsrmatrixgetlocalrang, FNALU_HYPRE_PARCSRMATRIXGETLOCALRANG)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *, NALU_HYPRE_Int *, NALU_HYPRE_Int *, NALU_HYPRE_Int *);

#define NALU_HYPRE_ParCSRMatrixGetRow  \
        nalu_hypre_F90_NAME(fnalu_hypre_parcsrmatrixgetrow, FNALU_HYPRE_PARCSRMATRIXGETROW)
extern void nalu_hypre_F90_NAME(fnalu_hypre_parcsrmatrixgetrow, FNALU_HYPRE_PARCSRMATRIXGETROW)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *, NALU_HYPRE_Int *, nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *);

#define NALU_HYPRE_ParCSRMatrixRestoreRow  \
        nalu_hypre_F90_NAME(fnalu_hypre_parcsrmatrixrestorerow, FNALU_HYPRE_PARCSRMATRIXRESTOREROW)
extern void nalu_hypre_F90_NAME(fnalu_hypre_parcsrmatrixrestorerow, FNALU_HYPRE_PARCSRMATRIXRESTOREROW)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *, NALU_HYPRE_Int *, nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *);

#define NALU_HYPRE_CSRMatrixtoParCSRMatrix  \
        nalu_hypre_F90_NAME(fnalu_hypre_csrmatrixtoparcsrmatrix, FNALU_HYPRE_CSRMATRIXTOPARCSRMATRIX)
extern void nalu_hypre_F90_NAME(fnalu_hypre_csrmatrixtoparcsrmatrix, FNALU_HYPRE_CSRMATRIXTOPARCSRMATRIX)
(NALU_HYPRE_Int *, nalu_hypre_F90_Obj *, NALU_HYPRE_Int *, NALU_HYPRE_Int *, nalu_hypre_F90_Obj *);

#define NALU_HYPRE_ParCSRMatrixMatvec  \
        nalu_hypre_F90_NAME(fnalu_hypre_parcsrmatrixmatvec, FNALU_HYPRE_PARCSRMATRIXMATVEC)
extern void nalu_hypre_F90_NAME(fnalu_hypre_parcsrmatrixmatvec, FNALU_HYPRE_PARCSRMATRIXMATVEC)
(NALU_HYPRE_Real *, nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *, NALU_HYPRE_Real *, nalu_hypre_F90_Obj *);

#define NALU_HYPRE_ParCSRMatrixMatvecT  \
        nalu_hypre_F90_NAME(fnalu_hypre_parcsrmatrixmatvect, FNALU_HYPRE_PARCSRMATRIXMATVECT)
extern void nalu_hypre_F90_NAME(fnalu_hypre_parcsrmatrixmatvect, FNALU_HYPRE_PARCSRMATRIXMATVECT)
(NALU_HYPRE_Real *, nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *, NALU_HYPRE_Real *, nalu_hypre_F90_Obj *);



#define NALU_HYPRE_ParVectorCreate  \
        nalu_hypre_F90_NAME(fnalu_hypre_parvectorcreate, FNALU_HYPRE_PARVECTORCREATE)
extern void nalu_hypre_F90_NAME(fnalu_hypre_parvectorcreate, FNALU_HYPRE_PARVECTORCREATE)
(NALU_HYPRE_Int *, NALU_HYPRE_Int *, nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *);

#define NALU_HYPRE_ParMultiVectorCreate  \
        nalu_hypre_F90_NAME(fnalu_hypre_parmultivectorcreate, FNALU_HYPRE_PARMULTIVECTORCREATE)
extern void nalu_hypre_F90_NAME(fnalu_hypre_parmultivectorcreate, FNALU_HYPRE_PARMULTIVECTORCREATE)
(NALU_HYPRE_Int *, NALU_HYPRE_Int *, nalu_hypre_F90_Obj *, NALU_HYPRE_Int *, nalu_hypre_F90_Obj *);

#define NALU_HYPRE_ParVectorDestroy  \
        nalu_hypre_F90_NAME(fnalu_hypre_parvectordestroy, FNALU_HYPRE_PARVECTORDESTROY)
extern void nalu_hypre_F90_NAME(fnalu_hypre_parvectordestroy, FNALU_HYPRE_PARVECTORDESTROY)
(nalu_hypre_F90_Obj *);

#define NALU_HYPRE_ParVectorInitialize  \
        nalu_hypre_F90_NAME(fnalu_hypre_parvectorinitialize, FNALU_HYPRE_PARVECTORINITIALIZE)
extern void nalu_hypre_F90_NAME(fnalu_hypre_parvectorinitialize, FNALU_HYPRE_PARVECTORINITIALIZE)
(nalu_hypre_F90_Obj *);

#define NALU_HYPRE_ParVectorRead  \
        nalu_hypre_F90_NAME(fnalu_hypre_parvectorread, FNALU_HYPRE_PARVECTORREAD)
extern void nalu_hypre_F90_NAME(fnalu_hypre_parvectorread, FNALU_HYPRE_PARVECTORREAD)
(NALU_HYPRE_Int *, nalu_hypre_F90_Obj *, char *);

#define NALU_HYPRE_ParVectorPrint  \
        nalu_hypre_F90_NAME(fnalu_hypre_parvectorprint, FNALU_HYPRE_PARVECTORPRINT)
extern void nalu_hypre_F90_NAME(fnalu_hypre_parvectorprint, FNALU_HYPRE_PARVECTORPRINT)
(nalu_hypre_F90_Obj *, char *, NALU_HYPRE_Int *);

#define NALU_HYPRE_ParVectorSetConstantValues  \
        nalu_hypre_F90_NAME(fnalu_hypre_parvectorsetconstantvalu, FNALU_HYPRE_PARVECTORSETCONSTANTVALU)
extern void nalu_hypre_F90_NAME(fnalu_hypre_parvectorsetconstantvalu, FNALU_HYPRE_PARVECTORSETCONSTANTVALU)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Real *);

#define NALU_HYPRE_ParVectorSetRandomValues  \
        nalu_hypre_F90_NAME(fnalu_hypre_parvectorsetrandomvalues, FNALU_HYPRE_PARVECTORSETRANDOMVALUES)
extern void nalu_hypre_F90_NAME(fnalu_hypre_parvectorsetrandomvalues, FNALU_HYPRE_PARVECTORSETRANDOMVALUES)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_ParVectorCopy  \
        nalu_hypre_F90_NAME(fnalu_hypre_parvectorcopy, FNALU_HYPRE_PARVECTORCOPY)
extern void nalu_hypre_F90_NAME(fnalu_hypre_parvectorcopy, FNALU_HYPRE_PARVECTORCOPY)
(nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *);

#define NALU_HYPRE_ParVectorCloneShallow  \
        nalu_hypre_F90_NAME(fnalu_hypre_parvectorcloneshallow, FNALU_HYPRE_PARVECTORCLONESHALLOW)
extern void nalu_hypre_F90_NAME(fnalu_hypre_parvectorcloneshallow, FNALU_HYPRE_PARVECTORCLONESHALLOW)
(nalu_hypre_F90_Obj *);

#define NALU_HYPRE_ParVectorScale  \
        nalu_hypre_F90_NAME(fnalu_hypre_parvectorscale, FNALU_HYPRE_PARVECTORSCALE)
extern void nalu_hypre_F90_NAME(fnalu_hypre_parvectorscale, FNALU_HYPRE_PARVECTORSCALE)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Real *);

#define NALU_HYPRE_ParVectorAxpy  \
        nalu_hypre_F90_NAME(fnalu_hypre_parvectoraxpy, FNALU_HYPRE_PARVECTORAXPY)
extern void nalu_hypre_F90_NAME(fnalu_hypre_parvectoraxpy, FNALU_HYPRE_PARVECTORAXPY)
(NALU_HYPRE_Real *, nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *);

#define NALU_HYPRE_ParVectorInnerProd  \
        nalu_hypre_F90_NAME(fnalu_hypre_parvectorinnerprod, FNALU_HYPRE_PARVECTORINNERPROD)
extern void nalu_hypre_F90_NAME(fnalu_hypre_parvectorinnerprod, FNALU_HYPRE_PARVECTORINNERPROD)
(nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *, NALU_HYPRE_Real *);

#define nalu_hypre_ParCSRMatrixGlobalNumRows  \
        nalu_hypre_F90_NAME(fnalu_hypre_parcsrmatrixglobalnumrow, FNALU_HYPRE_PARCSRMATRIXGLOBALNUMROW)
extern void nalu_hypre_F90_NAME(fnalu_hypre_parcsrmatrixglobalnumrow, FNALU_HYPRE_PARCSRMATRIXGLOBALNUMROW)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define nalu_hypre_ParCSRMatrixRowStarts  \
        nalu_hypre_F90_NAME(fnalu_hypre_parcsrmatrixrowstarts, FNALU_HYPRE_PARCSRMATRIXROWSTARTS)
extern void nalu_hypre_F90_NAME(fnalu_hypre_parcsrmatrixrowstarts, FNALU_HYPRE_PARCSRMATRIXROWSTARTS)
(nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *);

#define nalu_hypre_ParVectorSetDataOwner  \
        nalu_hypre_F90_NAME(fnalu_hypre_setparvectordataowner, FNALU_HYPRE_SETPARVECTORDATAOWNER)
extern void nalu_hypre_F90_NAME(fnalu_hypre_setparvectordataowner, FNALU_HYPRE_SETPARVECTORDATAOWNER)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);



#define GenerateLaplacian  \
        nalu_hypre_F90_NAME(fgeneratelaplacian, FNALU_HYPRE_GENERATELAPLACIAN)
extern void nalu_hypre_F90_NAME(fgeneratelaplacian, FNALU_HYPRE_GENERATELAPLACIAN)
(NALU_HYPRE_Int *, NALU_HYPRE_Int *, NALU_HYPRE_Int *, NALU_HYPRE_Int *, NALU_HYPRE_Int *, NALU_HYPRE_Int *, NALU_HYPRE_Int *,
 NALU_HYPRE_Int *, NALU_HYPRE_Int *, NALU_HYPRE_Int *, NALU_HYPRE_Real *, nalu_hypre_F90_Obj *);



#define NALU_HYPRE_BoomerAMGCreate  \
        nalu_hypre_F90_NAME(fnalu_hypre_boomeramgcreate, FNALU_HYPRE_BOOMERAMGCREATE)
extern void nalu_hypre_F90_NAME(fnalu_hypre_boomeramgcreate, FNALU_HYPRE_BOOMERAMGCREATE)
(nalu_hypre_F90_Obj *);

#define NALU_HYPRE_BoomerAMGDestroy  \
        nalu_hypre_F90_NAME(fnalu_hypre_boomeramgdestroy, FNALU_HYPRE_BOOMERAMGDESTROY)
extern void nalu_hypre_F90_NAME(fnalu_hypre_boomeramgdestroy, FNALU_HYPRE_BOOMERAMGDESTROY)
(nalu_hypre_F90_Obj *);

#define NALU_HYPRE_BoomerAMGSetup  \
        nalu_hypre_F90_NAME(fnalu_hypre_boomeramgsetup, FNALU_HYPRE_BOOMERAMGSETUP)
extern void nalu_hypre_F90_NAME(fnalu_hypre_boomeramgsetup, FNALU_HYPRE_BOOMERAMGSETUP)
(nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *);

#define NALU_HYPRE_BoomerAMGSolve  \
        nalu_hypre_F90_NAME(fnalu_hypre_boomeramgsolve, FNALU_HYPRE_BOOMERAMGSOLVE)
extern void nalu_hypre_F90_NAME(fnalu_hypre_boomeramgsolve, FNALU_HYPRE_BOOMERAMGSOLVE)
(nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *);

#define NALU_HYPRE_BoomerAMGSolveT  \
        nalu_hypre_F90_NAME(fnalu_hypre_boomeramgsolvet, FNALU_HYPRE_BOOMERAMGSOLVET)
extern void nalu_hypre_F90_NAME(fnalu_hypre_boomeramgsolvet, FNALU_HYPRE_BOOMERAMGSOLVET)
(nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *);

#define NALU_HYPRE_BoomerAMGSetRestriction  \
        nalu_hypre_F90_NAME(fnalu_hypre_boomeramgsetrestriction, FNALU_HYPRE_BOOMERAMGSETRESTRICTION)
extern void nalu_hypre_F90_NAME(fnalu_hypre_boomeramgsetrestriction, FNALU_HYPRE_BOOMERAMGSETRESTRICTION)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_BoomerAMGSetMaxLevels  \
        nalu_hypre_F90_NAME(fnalu_hypre_boomeramgsetmaxlevels, FNALU_HYPRE_BOOMERAMGSETMAXLEVELS)
extern void nalu_hypre_F90_NAME(fnalu_hypre_boomeramgsetmaxlevels, FNALU_HYPRE_BOOMERAMGSETMAXLEVELS)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_BoomerAMGGetMaxLevels  \
        nalu_hypre_F90_NAME(fnalu_hypre_boomeramggetmaxlevels, FNALU_HYPRE_BOOMERAMGGETMAXLEVELS)
extern void nalu_hypre_F90_NAME(fnalu_hypre_boomeramggetmaxlevels, FNALU_HYPRE_BOOMERAMGGETMAXLEVELS)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_BoomerAMGSetCoarsenCutFactor  \
        nalu_hypre_F90_NAME(fnalu_hypre_boomeramgsetcoarsencutfa, FNALU_HYPRE_BOOMERAMGSETCOARSENCUTFAC)
extern void nalu_hypre_F90_NAME(fnalu_hypre_boomeramgsetcoarsencutfa, FNALU_HYPRE_BOOMERAMGSETCOARSENCUTFAC)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Real *);

#define NALU_HYPRE_BoomerAMGGetCoarsenCutFactor  \
        nalu_hypre_F90_NAME(fnalu_hypre_boomeramggetcoarsencutfa, FNALU_HYPRE_BOOMERAMGGETCOARSENCUTFAC)
extern void nalu_hypre_F90_NAME(fnalu_hypre_boomeramggetcoarsencutfa, FNALU_HYPRE_BOOMERAMGGETCOARSENCUTFAC)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Real *);

#define NALU_HYPRE_BoomerAMGSetStrongThreshold  \
        nalu_hypre_F90_NAME(fnalu_hypre_boomeramgsetstrongthrshl, FNALU_HYPRE_BOOMERAMGSETSTRONGTHRSHL)
extern void nalu_hypre_F90_NAME(fnalu_hypre_boomeramgsetstrongthrshl, FNALU_HYPRE_BOOMERAMGSETSTRONGTHRSHL)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Real *);

#define NALU_HYPRE_BoomerAMGGetStrongThreshold  \
        nalu_hypre_F90_NAME(fnalu_hypre_boomeramggetstrongthrshl, FNALU_HYPRE_BOOMERAMGGETSTRONGTHRSHL)
extern void nalu_hypre_F90_NAME(fnalu_hypre_boomeramggetstrongthrshl, FNALU_HYPRE_BOOMERAMGGETSTRONGTHRSHL)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Real *);

#define NALU_HYPRE_BoomerAMGSetMaxRowSum  \
        nalu_hypre_F90_NAME(fnalu_hypre_boomeramgsetmaxrowsum, FNALU_HYPRE_BOOMERAMGSETMAXROWSUM)
extern void nalu_hypre_F90_NAME(fnalu_hypre_boomeramgsetmaxrowsum, FNALU_HYPRE_BOOMERAMGSETMAXROWSUM)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Real *);

#define NALU_HYPRE_BoomerAMGGetMaxRowSum  \
        nalu_hypre_F90_NAME(fnalu_hypre_boomeramggetmaxrowsum, FNALU_HYPRE_BOOMERAMGGETMAXROWSUM)
extern void nalu_hypre_F90_NAME(fnalu_hypre_boomeramggetmaxrowsum, FNALU_HYPRE_BOOMERAMGGETMAXROWSUM)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Real *);

#define NALU_HYPRE_BoomerAMGSetTruncFactor  \
        nalu_hypre_F90_NAME(fnalu_hypre_boomeramgsettruncfactor, FNALU_HYPRE_BOOMERAMGSETTRUNCFACTOR)
extern void nalu_hypre_F90_NAME(fnalu_hypre_boomeramgsettruncfactor, FNALU_HYPRE_BOOMERAMGSETTRUNCFACTOR)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Real *);

#define NALU_HYPRE_BoomerAMGGetTruncFactor  \
        nalu_hypre_F90_NAME(fnalu_hypre_boomeramggettruncfactor, FNALU_HYPRE_BOOMERAMGGETTRUNCFACTOR)
extern void nalu_hypre_F90_NAME(fnalu_hypre_boomeramggettruncfactor, FNALU_HYPRE_BOOMERAMGGETTRUNCFACTOR)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Real *);

#define NALU_HYPRE_BoomerAMGSetSCommPkgSwitch  \
        nalu_hypre_F90_NAME(fnalu_hypre_boomeramgsetscommpkgswit, FNALU_HYPRE_BOOMERAMGSETSCOMMPKGSWIT)
extern void nalu_hypre_F90_NAME(fnalu_hypre_boomeramgsetscommpkgswit, FNALU_HYPRE_BOOMERAMGSETSCOMMPKGSWIT)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_BoomerAMGSetInterpType  \
        nalu_hypre_F90_NAME(fnalu_hypre_boomeramgsetinterptype, FNALU_HYPRE_BOOMERAMGSETINTERPTYPE)
extern void nalu_hypre_F90_NAME(fnalu_hypre_boomeramgsetinterptype, FNALU_HYPRE_BOOMERAMGSETINTERPTYPE)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_BoomerAMGSetMinIter  \
        nalu_hypre_F90_NAME(fnalu_hypre_boomeramgsetminiter, FNALU_HYPRE_BOOMERAMGSETMINITER)
extern void nalu_hypre_F90_NAME(fnalu_hypre_boomeramgsetminiter, FNALU_HYPRE_BOOMERAMGSETMINITER)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_BoomerAMGSetMaxIter  \
        nalu_hypre_F90_NAME(fnalu_hypre_boomeramgsetmaxiter, FNALU_HYPRE_BOOMERAMGSETMAXITER)
extern void nalu_hypre_F90_NAME(fnalu_hypre_boomeramgsetmaxiter, FNALU_HYPRE_BOOMERAMGSETMAXITER)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_BoomerAMGGetMaxIter  \
        nalu_hypre_F90_NAME(fnalu_hypre_boomeramggetmaxiter, FNALU_HYPRE_BOOMERAMGGETMAXITER)
extern void nalu_hypre_F90_NAME(fnalu_hypre_boomeramggetmaxiter, FNALU_HYPRE_BOOMERAMGGETMAXITER)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_BoomerAMGSetCoarsenType  \
        nalu_hypre_F90_NAME(fnalu_hypre_boomeramgsetcoarsentype, FNALU_HYPRE_BOOMERAMGSETCOARSENTYPE)
extern void nalu_hypre_F90_NAME(fnalu_hypre_boomeramgsetcoarsentype, FNALU_HYPRE_BOOMERAMGSETCOARSENTYPE)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_BoomerAMGGetCoarsenType  \
        nalu_hypre_F90_NAME(fnalu_hypre_boomeramggetcoarsentype, FNALU_HYPRE_BOOMERAMGGETCOARSENTYPE)
extern void nalu_hypre_F90_NAME(fnalu_hypre_boomeramggetcoarsentype, FNALU_HYPRE_BOOMERAMGGETCOARSENTYPE)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_BoomerAMGSetMeasureType  \
        nalu_hypre_F90_NAME(fnalu_hypre_boomeramgsetmeasuretype, FNALU_HYPRE_BOOMERAMGSETMEASURETYPE)
extern void nalu_hypre_F90_NAME(fnalu_hypre_boomeramgsetmeasuretype, FNALU_HYPRE_BOOMERAMGSETMEASURETYPE)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_BoomerAMGGetMeasureType  \
        nalu_hypre_F90_NAME(fnalu_hypre_boomeramggetmeasuretype, FNALU_HYPRE_BOOMERAMGGETMEASURETYPE)
extern void nalu_hypre_F90_NAME(fnalu_hypre_boomeramggetmeasuretype, FNALU_HYPRE_BOOMERAMGGETMEASURETYPE)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_BoomerAMGSetSetupType  \
        nalu_hypre_F90_NAME(fnalu_hypre_boomeramgsetsetuptype, FNALU_HYPRE_BOOMERAMGSETSETUPTYPE)
extern void nalu_hypre_F90_NAME(fnalu_hypre_boomeramgsetsetuptype, FNALU_HYPRE_BOOMERAMGSETSETUPTYPE)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_BoomerAMGSetCycleType  \
        nalu_hypre_F90_NAME(fnalu_hypre_boomeramgsetcycletype, FNALU_HYPRE_BOOMERAMGSETCYCLETYPE)
extern void nalu_hypre_F90_NAME(fnalu_hypre_boomeramgsetcycletype, FNALU_HYPRE_BOOMERAMGSETCYCLETYPE)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_BoomerAMGGetCycleType  \
        nalu_hypre_F90_NAME(fnalu_hypre_boomeramggetcycletype, FNALU_HYPRE_BOOMERAMGGETCYCLETYPE)
extern void nalu_hypre_F90_NAME(fnalu_hypre_boomeramggetcycletype, FNALU_HYPRE_BOOMERAMGGETCYCLETYPE)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_BoomerAMGSetTol  \
        nalu_hypre_F90_NAME(fnalu_hypre_boomeramgsettol, FNALU_HYPRE_BOOMERAMGSETTOL)
extern void nalu_hypre_F90_NAME(fnalu_hypre_boomeramgsettol, FNALU_HYPRE_BOOMERAMGSETTOL)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Real *);

#define NALU_HYPRE_BoomerAMGGetTol  \
        nalu_hypre_F90_NAME(fnalu_hypre_boomeramggettol, FNALU_HYPRE_BOOMERAMGGETTOL)
extern void nalu_hypre_F90_NAME(fnalu_hypre_boomeramggettol, FNALU_HYPRE_BOOMERAMGGETTOL)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Real *);

#define NALU_HYPRE_BoomerAMGSetNumSweeps  \
        nalu_hypre_F90_NAME(fnalu_hypre_boomeramgsetnumsweeps, FNALU_HYPRE_BOOMERAMGSETNUMSWEEPS)
extern void nalu_hypre_F90_NAME(fnalu_hypre_boomeramgsetnumsweeps, FNALU_HYPRE_BOOMERAMGSETNUMSWEEPS)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_BoomerAMGSetCycleNumSweeps  \
        nalu_hypre_F90_NAME(fnalu_hypre_boomeramgsetcyclenumswee, FNALU_HYPRE_BOOMERAMGSETCYCLENUMSWEE)
extern void nalu_hypre_F90_NAME(fnalu_hypre_boomeramgsetcyclenumswee, FNALU_HYPRE_BOOMERAMGSETCYCLENUMSWEE)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *, NALU_HYPRE_Int *);

#define NALU_HYPRE_BoomerAMGGetCycleNumSweeps  \
        nalu_hypre_F90_NAME(fnalu_hypre_boomeramggetcyclenumswee, FNALU_HYPRE_BOOMERAMGGETCYCLENUMSWEE)
extern void nalu_hypre_F90_NAME(fnalu_hypre_boomeramggetcyclenumswee, FNALU_HYPRE_BOOMERAMGGETCYCLENUMSWEE)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *, NALU_HYPRE_Int *);

#define NALU_HYPRE_BoomerAMGInitGridRelaxation  \
        nalu_hypre_F90_NAME(fnalu_hypre_boomeramginitgridrelaxat, FNALU_HYPRE_BOOMERAMGINITGRIDRELAXAT)
extern void nalu_hypre_F90_NAME(fnalu_hypre_boomeramginitgridrelaxat, FNALU_HYPRE_BOOMERAMGINITGRIDRELAXAT)
(nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *, NALU_HYPRE_Int *, nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_BoomerAMGFinalizeGridRelaxation  \
        nalu_hypre_F90_NAME(fnalu_hypre_boomeramgfingridrelaxatn, FNALU_HYPRE_BOOMERAMGFINGRIDRELAXATN)
extern void nalu_hypre_F90_NAME(fnalu_hypre_boomeramgfingridrelaxatn, FNALU_HYPRE_BOOMERAMGFINGRIDRELAXATN)
(nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *);

#define NALU_HYPRE_BoomerAMGSetRelaxType  \
        nalu_hypre_F90_NAME(fnalu_hypre_boomeramgsetrelaxtype, FNALU_HYPRE_BOOMERAMGSETRELAXTYPE)
extern void nalu_hypre_F90_NAME(fnalu_hypre_boomeramgsetrelaxtype, FNALU_HYPRE_BOOMERAMGSETRELAXTYPE)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_BoomerAMGSetRelaxType  \
        nalu_hypre_F90_NAME(fnalu_hypre_boomeramgsetrelaxtype, FNALU_HYPRE_BOOMERAMGSETRELAXTYPE)
extern void nalu_hypre_F90_NAME(fnalu_hypre_boomeramgsetrelaxtype, FNALU_HYPRE_BOOMERAMGSETRELAXTYPE)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_BoomerAMGSetRelaxOrder  \
        nalu_hypre_F90_NAME(fnalu_hypre_boomeramgsetrelaxorder, FNALU_HYPRE_BOOMERAMGSETRELAXORDER)
extern void nalu_hypre_F90_NAME(fnalu_hypre_boomeramgsetrelaxorder, FNALU_HYPRE_BOOMERAMGSETRELAXORDER)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_BoomerAMGSetRelaxWeight  \
        nalu_hypre_F90_NAME(fnalu_hypre_boomeramgsetrelaxweight, FNALU_HYPRE_BOOMERAMGSETRELAXWEIGHT)
extern void nalu_hypre_F90_NAME(fnalu_hypre_boomeramgsetrelaxweight, FNALU_HYPRE_BOOMERAMGSETRELAXWEIGHT)
(nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *);

#define NALU_HYPRE_BoomerAMGSetRelaxWt  \
        nalu_hypre_F90_NAME(fnalu_hypre_boomeramgsetrelaxwt, FNALU_HYPRE_BOOMERAMGSETRELAXWT)
extern void nalu_hypre_F90_NAME(fnalu_hypre_boomeramgsetrelaxwt, FNALU_HYPRE_BOOMERAMGSETRELAXWT)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Real *);

#define NALU_HYPRE_BoomerAMGSetLevelRelaxWt  \
        nalu_hypre_F90_NAME(fnalu_hypre_boomeramgsetlevelrelaxwt, FNALU_HYPRE_BOOMERAMGSETLEVELRELAXWT)
extern void nalu_hypre_F90_NAME(fnalu_hypre_boomeramgsetlevelrelaxwt, FNALU_HYPRE_BOOMERAMGSETLEVELRELAXWT)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Real *, NALU_HYPRE_Int *);

#define NALU_HYPRE_BoomerAMGSetOuterWt  \
        nalu_hypre_F90_NAME(fnalu_hypre_boomeramgsetouterwt, FNALU_HYPRE_BOOMERAMGSETOUTERWT)
extern void nalu_hypre_F90_NAME(fnalu_hypre_boomeramgsetouterwt, FNALU_HYPRE_BOOMERAMGSETOUTERWT)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Real *);

#define NALU_HYPRE_BoomerAMGSetLevelOuterWt  \
        nalu_hypre_F90_NAME(fnalu_hypre_boomeramgsetlevelouterwt, FNALU_HYPRE_BOOMERAMGSETLEVELOUTERWT)
extern void nalu_hypre_F90_NAME(fnalu_hypre_boomeramgsetlevelouterwt, FNALU_HYPRE_BOOMERAMGSETLEVELOUTERWT)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Real *);

#define NALU_HYPRE_BoomerAMGSetSmoothType  \
        nalu_hypre_F90_NAME(fnalu_hypre_boomeramgsetsmoothtype, FNALU_HYPRE_BOOMERAMGSETSMOOTHTYPE)
extern void nalu_hypre_F90_NAME(fnalu_hypre_boomeramgsetsmoothtype, FNALU_HYPRE_BOOMERAMGSETSMOOTHTYPE)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_BoomerAMGGetSmoothType  \
        nalu_hypre_F90_NAME(fnalu_hypre_boomeramggetsmoothtype, FNALU_HYPRE_BOOMERAMGGETSMOOTHTYPE)
extern void nalu_hypre_F90_NAME(fnalu_hypre_boomeramggetsmoothtype, FNALU_HYPRE_BOOMERAMGGETSMOOTHTYPE)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_BoomerAMGSetSmoothNumLvls  \
        nalu_hypre_F90_NAME(fnalu_hypre_boomeramgsetsmoothnumlvl, FNALU_HYPRE_BOOMERAMGSETSMOOTHNUMLVL)
extern void nalu_hypre_F90_NAME(fnalu_hypre_boomeramgsetsmoothnumlvl, FNALU_HYPRE_BOOMERAMGSETSMOOTHNUMLVL)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_BoomerAMGGetSmoothNumLvls  \
        nalu_hypre_F90_NAME(fnalu_hypre_boomeramggetsmoothnumlvl, FNALU_HYPRE_BOOMERAMGGETSMOOTHNUMLVL)
extern void nalu_hypre_F90_NAME(fnalu_hypre_boomeramggetsmoothnumlvl, FNALU_HYPRE_BOOMERAMGGETSMOOTHNUMLVL)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_BoomerAMGSetSmoothNumSwps  \
        nalu_hypre_F90_NAME(fnalu_hypre_boomeramgsetsmoothnumswp, FNALU_HYPRE_BOOMERAMGSETSMOOTHNUMSWP)
extern void nalu_hypre_F90_NAME(fnalu_hypre_boomeramgsetsmoothnumswp, FNALU_HYPRE_BOOMERAMGSETSMOOTHNUMSWP)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_BoomerAMGGetSmoothNumSwps  \
        nalu_hypre_F90_NAME(fnalu_hypre_boomeramggetsmoothnumswp, FNALU_HYPRE_BOOMERAMGGETSMOOTHNUMSWP)
extern void nalu_hypre_F90_NAME(fnalu_hypre_boomeramggetsmoothnumswp, FNALU_HYPRE_BOOMERAMGGETSMOOTHNUMSWP)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_BoomerAMGSetLogging  \
        nalu_hypre_F90_NAME(fnalu_hypre_boomeramgsetlogging, FNALU_HYPRE_BOOMERAMGSETLOGGING)
extern void nalu_hypre_F90_NAME(fnalu_hypre_boomeramgsetlogging, FNALU_HYPRE_BOOMERAMGSETLOGGING)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_BoomerAMGGetLogging  \
        nalu_hypre_F90_NAME(fnalu_hypre_boomeramggetlogging, FNALU_HYPRE_BOOMERAMGGETLOGGING)
extern void nalu_hypre_F90_NAME(fnalu_hypre_boomeramggetlogging, FNALU_HYPRE_BOOMERAMGGETLOGGING)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_BoomerAMGSetPrintLevel  \
        nalu_hypre_F90_NAME(fnalu_hypre_boomeramgsetprintlevel, FNALU_HYPRE_BOOMERAMGSETPRINTLEVEL)
extern void nalu_hypre_F90_NAME(fnalu_hypre_boomeramgsetprintlevel, FNALU_HYPRE_BOOMERAMGSETPRINTLEVEL)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_BoomerAMGGetPrintLevel  \
        nalu_hypre_F90_NAME(fnalu_hypre_boomeramggetprintlevel, FNALU_HYPRE_BOOMERAMGGETPRINTLEVEL)
extern void nalu_hypre_F90_NAME(fnalu_hypre_boomeramggetprintlevel, FNALU_HYPRE_BOOMERAMGGETPRINTLEVEL)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_BoomerAMGSetPrintFileName  \
        nalu_hypre_F90_NAME(fnalu_hypre_boomeramgsetprintfilenam, FNALU_HYPRE_BOOMERAMGSETPRINTFILENAM)
extern void nalu_hypre_F90_NAME(fnalu_hypre_boomeramgsetprintfilenam, FNALU_HYPRE_BOOMERAMGSETPRINTFILENAM)
(nalu_hypre_F90_Obj *, char *);

#define NALU_HYPRE_BoomerAMGSetDebugFlag  \
        nalu_hypre_F90_NAME(fnalu_hypre_boomeramgsetdebugflag, FNALU_HYPRE_BOOMERAMGSETDEBUGFLAG)
extern void nalu_hypre_F90_NAME(fnalu_hypre_boomeramgsetdebugflag, FNALU_HYPRE_BOOMERAMGSETDEBUGFLAG)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_BoomerAMGGetDebugFlag  \
        nalu_hypre_F90_NAME(fnalu_hypre_boomeramggetdebugflag, FNALU_HYPRE_BOOMERAMGGETDEBUGFLAG)
extern void nalu_hypre_F90_NAME(fnalu_hypre_boomeramggetdebugflag, FNALU_HYPRE_BOOMERAMGGETDEBUGFLAG)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_BoomerAMGGetNumIterations  \
        nalu_hypre_F90_NAME(fnalu_hypre_boomeramggetnumiteration, FNALU_HYPRE_BOOMERAMGGETNUMITERATION)
extern void nalu_hypre_F90_NAME(fnalu_hypre_boomeramggetnumiteration, FNALU_HYPRE_BOOMERAMGGETNUMITERATION)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_BoomerAMGGetCumNumIterations  \
        nalu_hypre_F90_NAME(fnalu_hypre_boomeramggetcumnumiterat, FNALU_HYPRE_BOOMERAMGGETCUMNUMITERAT)
extern void nalu_hypre_F90_NAME(fnalu_hypre_boomeramggetcumnumiterat, FNALU_HYPRE_BOOMERAMGGETCUMNUMITERAT)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_BoomerAMGGetResidual  \
        nalu_hypre_F90_NAME(fnalu_hypre_boomeramggetresidual, FNALU_HYPRE_BOOMERAMGGETRESIDUAL)
extern void nalu_hypre_F90_NAME(fnalu_hypre_boomeramggetresidual, FNALU_HYPRE_BOOMERAMGGETRESIDUAL)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Real *);

#define NALU_HYPRE_BoomerAMGGetFinalRelativeResidualNorm  \
        nalu_hypre_F90_NAME(fnalu_hypre_boomeramggetfinalreltvre, FNALU_HYPRE_BOOMERAMGGETFINALRELTVRE)
extern void nalu_hypre_F90_NAME(fnalu_hypre_boomeramggetfinalreltvre, FNALU_HYPRE_BOOMERAMGGETFINALRELTVRE)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Real *);

#define NALU_HYPRE_BoomerAMGSetVariant  \
        nalu_hypre_F90_NAME(fnalu_hypre_boomeramgsetvariant, FNALU_HYPRE_BOOMERAMGSETVARIANT)
extern void nalu_hypre_F90_NAME(fnalu_hypre_boomeramgsetvariant, FNALU_HYPRE_BOOMERAMGSETVARIANT)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_BoomerAMGGetVariant  \
        nalu_hypre_F90_NAME(fnalu_hypre_boomeramggetvariant, FNALU_HYPRE_BOOMERAMGGETVARIANT)
extern void nalu_hypre_F90_NAME(fnalu_hypre_boomeramggetvariant, FNALU_HYPRE_BOOMERAMGGETVARIANT)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_BoomerAMGSetOverlap  \
        nalu_hypre_F90_NAME(fnalu_hypre_boomeramgsetoverlap, FNALU_HYPRE_BOOMERAMGSETOVERLAP)
extern void nalu_hypre_F90_NAME(fnalu_hypre_boomeramgsetoverlap, FNALU_HYPRE_BOOMERAMGSETOVERLAP)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_BoomerAMGGetOverlap  \
        nalu_hypre_F90_NAME(fnalu_hypre_boomeramggetoverlap, FNALU_HYPRE_BOOMERAMGGETOVERLAP)
extern void nalu_hypre_F90_NAME(fnalu_hypre_boomeramggetoverlap, FNALU_HYPRE_BOOMERAMGGETOVERLAP)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_BoomerAMGSetDomainType  \
        nalu_hypre_F90_NAME(fnalu_hypre_boomeramgsetdomaintype, FNALU_HYPRE_BOOMERAMGSETDOMAINTYPE)
extern void nalu_hypre_F90_NAME(fnalu_hypre_boomeramgsetdomaintype, FNALU_HYPRE_BOOMERAMGSETDOMAINTYPE)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_BoomerAMGGetDomainType  \
        nalu_hypre_F90_NAME(fnalu_hypre_boomeramggetdomaintype, FNALU_HYPRE_BOOMERAMGGETDOMAINTYPE)
extern void nalu_hypre_F90_NAME(fnalu_hypre_boomeramggetdomaintype, FNALU_HYPRE_BOOMERAMGGETDOMAINTYPE)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_BoomerAMGSetSchwarzRlxWt  \
        nalu_hypre_F90_NAME(fnalu_hypre_boomeramgsetschwarzrlxwt, FNALU_HYPRE_BOOMERAMGSETSCHWARZRLXWT)
extern void nalu_hypre_F90_NAME(fnalu_hypre_boomeramgsetschwarzrlxwt, FNALU_HYPRE_BOOMERAMGSETSCHWARZRLXWT)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_BoomerAMGGetSchwarzRlxWt  \
        nalu_hypre_F90_NAME(fnalu_hypre_boomeramggetschwarzrlxwt, FNALU_HYPRE_BOOMERAMGGETSCHWARZRLXWT)
extern void nalu_hypre_F90_NAME(fnalu_hypre_boomeramggetschwarzrlxwt, FNALU_HYPRE_BOOMERAMGGETSCHWARZRLXWT)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_BoomerAMGSetSym  \
        nalu_hypre_F90_NAME(fnalu_hypre_boomeramgsetsym, FNALU_HYPRE_BOOMERAMGSETSYM)
extern void nalu_hypre_F90_NAME(fnalu_hypre_boomeramgsetsym, FNALU_HYPRE_BOOMERAMGSETSYM)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_BoomerAMGSetLevel  \
        nalu_hypre_F90_NAME(fnalu_hypre_boomeramgsetlevel, FNALU_HYPRE_BOOMERAMGSETLEVEL)
extern void nalu_hypre_F90_NAME(fnalu_hypre_boomeramgsetlevel, FNALU_HYPRE_BOOMERAMGSETLEVEL)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_BoomerAMGSetFilter  \
        nalu_hypre_F90_NAME(fnalu_hypre_boomeramgsetfilter, FNALU_HYPRE_BOOMERAMGSETFILTER)
extern void nalu_hypre_F90_NAME(fnalu_hypre_boomeramgsetfilter, FNALU_HYPRE_BOOMERAMGSETFILTER)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_BoomerAMGSetDropTol  \
        nalu_hypre_F90_NAME(fnalu_hypre_boomeramgsetdroptol, FNALU_HYPRE_BOOMERAMGSETDROPTOL)
extern void nalu_hypre_F90_NAME(fnalu_hypre_boomeramgsetdroptol, FNALU_HYPRE_BOOMERAMGSETDROPTOL)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_BoomerAMGSetMaxNzPerRow  \
        nalu_hypre_F90_NAME(fnalu_hypre_boomeramgsetmaxnzperrow, FNALU_HYPRE_BOOMERAMGSETMAXNZPERROW)
extern void nalu_hypre_F90_NAME(fnalu_hypre_boomeramgsetmaxnzperrow, FNALU_HYPRE_BOOMERAMGSETMAXNZPERROW)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_BoomerAMGSetEuclidFile  \
        nalu_hypre_F90_NAME(fnalu_hypre_boomeramgseteuclidfile, FNALU_HYPRE_BOOMERAMGSETEUCLIDFILE)
extern void nalu_hypre_F90_NAME(fnalu_hypre_boomeramgseteuclidfile, FNALU_HYPRE_BOOMERAMGSETEUCLIDFILE)
(nalu_hypre_F90_Obj *, char *);

#define NALU_HYPRE_BoomerAMGSetNumFunctions  \
        nalu_hypre_F90_NAME(fnalu_hypre_boomeramgsetnumfunctions, FNALU_HYPRE_BOOMERAMGSETNUMFUNCTIONS)
extern void nalu_hypre_F90_NAME(fnalu_hypre_boomeramgsetnumfunctions, FNALU_HYPRE_BOOMERAMGSETNUMFUNCTIONS)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_BoomerAMGGetNumFunctions  \
        nalu_hypre_F90_NAME(fnalu_hypre_boomeramggetnumfunctions, FNALU_HYPRE_BOOMERAMGGETNUMFUNCTIONS)
extern void nalu_hypre_F90_NAME(fnalu_hypre_boomeramggetnumfunctions, FNALU_HYPRE_BOOMERAMGGETNUMFUNCTIONS)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_BoomerAMGSetNodal  \
        nalu_hypre_F90_NAME(fnalu_hypre_boomeramgsetnodal, FNALU_HYPRE_BOOMERAMGSETNODAL)
extern void nalu_hypre_F90_NAME(fnalu_hypre_boomeramgsetnodal, FNALU_HYPRE_BOOMERAMGSETNODAL)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_BoomerAMGSetDofFunc  \
        nalu_hypre_F90_NAME(fnalu_hypre_boomeramgsetdoffunc, FNALU_HYPRE_BOOMERAMGSETDOFFUNC)
extern void nalu_hypre_F90_NAME(fnalu_hypre_boomeramgsetdoffunc, FNALU_HYPRE_BOOMERAMGSETDOFFUNC)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_BoomerAMGSetNumPaths  \
        nalu_hypre_F90_NAME(fnalu_hypre_boomeramgsetnumpaths, FNALU_HYPRE_BOOMERAMGSETNUMPATHS)
extern void nalu_hypre_F90_NAME(fnalu_hypre_boomeramgsetnumpaths, FNALU_HYPRE_BOOMERAMGSETNUMPATHS)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_BoomerAMGSetAggNumLevels  \
        nalu_hypre_F90_NAME(fnalu_hypre_boomeramgsetaggnumlevels, FNALU_HYPRE_BOOMERAMGSETAGGNUMLEVELS)
extern void nalu_hypre_F90_NAME(fnalu_hypre_boomeramgsetaggnumlevels, FNALU_HYPRE_BOOMERAMGSETAGGNUMLEVELS)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_BoomerAMGSetGSMG  \
        nalu_hypre_F90_NAME(fnalu_hypre_boomeramgsetgsmg, FNALU_HYPRE_BOOMERAMGSETGSMG)
extern void nalu_hypre_F90_NAME(fnalu_hypre_boomeramgsetgsmg, FNALU_HYPRE_BOOMERAMGSETGSMG)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_BoomerAMGSetNumSamples  \
        nalu_hypre_F90_NAME(fnalu_hypre_boomeramgsetnumsamples, FNALU_HYPRE_BOOMERAMGSETNUMSAMPLES)
extern void nalu_hypre_F90_NAME(fnalu_hypre_boomeramgsetsamples, FNALU_HYPRE_BOOMERAMGSETNUMSAMPLES)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);



#define NALU_HYPRE_ParCSRBiCGSTABCreate  \
        nalu_hypre_F90_NAME(fnalu_hypre_parcsrbicgstabcreate, FNALU_HYPRE_PARCSRBICGSTABCREATE)
extern void nalu_hypre_F90_NAME(fnalu_hypre_parcsrbicgstabcreate, FNALU_HYPRE_PARCSRBICGSTABCREATE)
(NALU_HYPRE_Int *, nalu_hypre_F90_Obj *);

#define NALU_HYPRE_ParCSRBiCGSTABDestroy  \
        nalu_hypre_F90_NAME(fnalu_hypre_parcsrbicgstabdestroy, FNALU_HYPRE_PARCSRBICGSTABDESTROY)
extern void nalu_hypre_F90_NAME(fnalu_hypre_parcsrbicgstabdestroy, FNALU_HYPRE_PARCSRBICGSTABDESTROY)
(nalu_hypre_F90_Obj *);

#define NALU_HYPRE_ParCSRBiCGSTABSetup  \
        nalu_hypre_F90_NAME(fnalu_hypre_parcsrbicgstabsetup, FNALU_HYPRE_PARCSRBICGSTABSETUP)
extern void nalu_hypre_F90_NAME(fnalu_hypre_parcsrbicgstabsetup, FNALU_HYPRE_PARCSRBICGSTABSETUP)
(nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *);

#define NALU_HYPRE_ParCSRBiCGSTABSolve  \
        nalu_hypre_F90_NAME(fnalu_hypre_parcsrbicgstabsolve, FNALU_HYPRE_PARCSRBICGSTABSOLVE)
extern void nalu_hypre_F90_NAME(fnalu_hypre_parcsrbicgstabsolve, FNALU_HYPRE_PARCSRBICGSTABSOLVE)
(nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *);

#define NALU_HYPRE_ParCSRBiCGSTABSetTol  \
        nalu_hypre_F90_NAME(fnalu_hypre_parcsrbicgstabsettol, FNALU_HYPRE_PARCSRBICGSTABSETTOL)
extern void nalu_hypre_F90_NAME(fnalu_hypre_parcsrbicgstabsettol, FNALU_HYPRE_PARCSRBICGSTABSETTOL)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Real *);

#define NALU_HYPRE_ParCSRBiCGSTABSetMinIter  \
        nalu_hypre_F90_NAME(fnalu_hypre_parcsrbicgstabsetminiter, FNALU_HYPRE_PARCSRBICGSTABSETMINITER)
extern void nalu_hypre_F90_NAME(fnalu_hypre_parcsrbicgstabsetminiter, FNALU_HYPRE_PARCSRBICGSTABSETMINITER)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_ParCSRBiCGSTABSetMaxIter  \
        nalu_hypre_F90_NAME(fnalu_hypre_parcsrbicgstabsetmaxiter, FNALU_HYPRE_PARCSRBICGSTABSETMAXITER)
extern void nalu_hypre_F90_NAME(fnalu_hypre_parcsrbicgstabsetmaxiter, FNALU_HYPRE_PARCSRBICGSTABSETMAXITER)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_ParCSRBiCGSTABSetStopCrit  \
        nalu_hypre_F90_NAME(fnalu_hypre_parcsrbicgstabsetstopcri, FNALU_HYPRE_PARCSRBICGSTABSETSTOPCRI)
extern void nalu_hypre_F90_NAME(fnalu_hypre_parcsrbicgstabsetstopcri, FNALU_HYPRE_PARCSRBICGSTABSETSTOPCRI)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_ParCSRBiCGSTABSetPrecond  \
        nalu_hypre_F90_NAME(fnalu_hypre_parcsrbicgstabsetprecond, FNALU_HYPRE_PARCSRBICGSTABSETPRECOND)
extern void nalu_hypre_F90_NAME(fnalu_hypre_parcsrbicgstabsetprecond, FNALU_HYPRE_PARCSRBICGSTABSETPRECOND)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *, nalu_hypre_F90_Obj *);

#define NALU_HYPRE_ParCSRBiCGSTABGetPrecond  \
        nalu_hypre_F90_NAME(fnalu_hypre_parcsrbicgstabgetprecond, FNALU_HYPRE_PARCSRBICGSTABGETPRECOND)
extern void nalu_hypre_F90_NAME(fnalu_hypre_parcsrbicgstabgetprecond, FNALU_HYPRE_PARCSRBICGSTABGETPRECOND)
(nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *);

#define NALU_HYPRE_ParCSRBiCGSTABSetLogging  \
        nalu_hypre_F90_NAME(fnalu_hypre_parcsrbicgstabsetlogging, FNALU_HYPRE_PARCSRBICGSTABSETLOGGING)
extern void nalu_hypre_F90_NAME(fnalu_hypre_parcsrbicgstabsetlogging, FNALU_HYPRE_PARCSRBICGSTABSETLOGGING)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_ParCSRBiCGSTABSetPrintLevel  \
        nalu_hypre_F90_NAME(fnalu_hypre_parcsrbicgstabsetprintle, FNALU_HYPRE_PARCSRBICGSTABSETPRINTLE)
extern void nalu_hypre_F90_NAME(fnalu_hypre_parcsrbicgstabsetprintle, FNALU_HYPRE_PARCSRBICGSTABSETPRINTLE)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_ParCSRBiCGSTABGetNumIter  \
        nalu_hypre_F90_NAME(fnalu_hypre_parcsrbicgstabgetnumiter, FNALU_HYPRE_PARCSRBICGSTABGETNUMITER)
extern void nalu_hypre_F90_NAME(fnalu_hypre_parcsrbicgstabgetnumiter, FNALU_HYPRE_PARCSRBICGSTABGETNUMITER)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_ParCSRBiCGSTABGetFinalRel  \
        nalu_hypre_F90_NAME(fnalu_hypre_parcsrbicgstabgetfinalre, FNALU_HYPRE_PARCSRBICGSTABGETFINALRE)
extern void nalu_hypre_F90_NAME(fnalu_hypre_parcsrbicgstabgetfinalre, FNALU_HYPRE_PARCSRBICGSTABGETFINALRE)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Real *);



#define NALU_HYPRE_BlockTridiagCreate  \
   nalu_hypre_F90_NAME(fnalu_hypre_blocktridiagcreate, FNALU_HYPRE_BLOCKTRIDIAGCREATE)
extern void nalu_hypre_F90_NAME(fnalu_hypre_blocktridiagcreate, FNALU_HYPRE_BLOCKTRIDIAGCREATE)
(nalu_hypre_F90_Obj *);

#define NALU_HYPRE_BlockTridiagDestroy  \
   nalu_hypre_F90_NAME(fnalu_hypre_blocktridiagdestroy, FNALU_HYPRE_BLOCKTRIDIAGDESTROY)
extern void nalu_hypre_F90_NAME(fnalu_hypre_blocktridiagdestroy, FNALU_HYPRE_BLOCKTRIDIAGDESTROY)
(nalu_hypre_F90_Obj *);

#define NALU_HYPRE_BlockTridiagSetup  \
   nalu_hypre_F90_NAME(fnalu_hypre_blocktridiagsetup, FNALU_HYPRE_BLOCKTRIDIAGSETUP)
extern void nalu_hypre_F90_NAME(fnalu_hypre_blocktridiagsetup, FNALU_HYPRE_BLOCKTRIDIAGSETUP)
(nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *);

#define NALU_HYPRE_BlockTridiagSolve  \
   nalu_hypre_F90_NAME(fnalu_hypre_blocktridiagsolve, FNALU_HYPRE_BLOCKTRIDIAGSOLVE)
extern void nalu_hypre_F90_NAME(fnalu_hypre_blocktridiagsolve, FNALU_HYPRE_BLOCKTRIDIAGSOLVE)
(nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *);

#define NALU_HYPRE_BlockTridiagSetIndexSet  \
   nalu_hypre_F90_NAME(fnalu_hypre_blocktridiagsetindexset, FNALU_HYPRE_BLOCKTRIDIAGSETINDEXSET)
extern void nalu_hypre_F90_NAME(fnalu_hypre_blocktridiagsetindexset, FNALU_HYPRE_BLOCKTRIDIAGSETINDEXSET)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *, NALU_HYPRE_Int *);

#define NALU_HYPRE_BlockTridiagSetAMGStrengthThreshold  \
   nalu_hypre_F90_NAME(fnalu_hypre_blocktridiagsetamgstreng, FNALU_HYPRE_BLOCKTRIDIAGSETAMGSTRENG)
extern void nalu_hypre_F90_NAME(fnalu_hypre_blocktridiagsetamgstreng, FNALU_HYPRE_BLOCKTRIDIAGSETAMGSTRENG)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Real *);

#define NALU_HYPRE_BlockTridiagSetAMGNumSweeps  \
   nalu_hypre_F90_NAME(fnalu_hypre_blocktridiagsetamgnumswe, FNALU_HYPRE_BLOCKTRIDIAGSETAMGNUMSWE)
extern void nalu_hypre_F90_NAME(fnalu_hypre_blocktridiagsetamgnumswe, FNALU_HYPRE_BLOCKTRIDIAGSETAMGNUMSWE)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_BlockTridiagSetAMGRelaxType  \
   nalu_hypre_F90_NAME(fnalu_hypre_blocktridiagsetamgrelaxt, FNALU_HYPRE_BLOCKTRIDIAGSETAMGRELAXT)
extern void nalu_hypre_F90_NAME(fnalu_hypre_blocktridiagsetamgrelaxt, FNALU_HYPRE_BLOCKTRIDIAGSETAMGRELAXT)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_BlockTridiagSetPrintLevel  \
   nalu_hypre_F90_NAME(fnalu_hypre_blocktridiagsetprintleve, FNALU_HYPRE_BLOCKTRIDIAGSETPRINTLEVE)
extern void nalu_hypre_F90_NAME(fnalu_hypre_blocktridiagsetprintleve, FNALU_HYPRE_BLOCKTRIDIAGSETPRINTLEVE)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);



#define NALU_HYPRE_ParCSRCGNRCreate  \
   nalu_hypre_F90_NAME(fnalu_hypre_parcsrcgnrcreate, FNALU_HYPRE_PARCSRCGNRCREATE)
extern void nalu_hypre_F90_NAME(fnalu_hypre_parcsrcgnrcreate, FNALU_HYPRE_PARCSRCGNRCREATE)
(NALU_HYPRE_Int *, nalu_hypre_F90_Obj *);

#define NALU_HYPRE_ParCSRCGNRDestroy  \
        nalu_hypre_F90_NAME(fnalu_hypre_parcsrcgnrdestroy, FNALU_HYPRE_PARCSRCGNRDESTROY)
extern void nalu_hypre_F90_NAME(fnalu_hypre_parcsrcgnrdestroy, FNALU_HYPRE_PARCSRCGNRDESTROY)
(nalu_hypre_F90_Obj *);

#define NALU_HYPRE_ParCSRCGNRSetup  \
        nalu_hypre_F90_NAME(fnalu_hypre_parcsrcgnrsetup, FNALU_HYPRE_PARCSRCGNRSETUP)
extern void nalu_hypre_F90_NAME(fnalu_hypre_parcsrcgnrsetup, FNALU_HYPRE_PARCSRCGNRSETUP)
(nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *);

#define NALU_HYPRE_ParCSRCGNRSolve  \
        nalu_hypre_F90_NAME(fnalu_hypre_parcsrcgnrsolve, FNALU_HYPRE_PARCSRCGNRSOLVE)
extern void nalu_hypre_F90_NAME(fnalu_hypre_parcsrcgnrsolve, FNALU_HYPRE_PARCSRCGNRSOLVE)
(nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *);

#define NALU_HYPRE_ParCSRCGNRSetTol  \
        nalu_hypre_F90_NAME(fnalu_hypre_parcsrcgnrsettol, FNALU_HYPRE_PARCSRCGNRSETTOL)
extern void nalu_hypre_F90_NAME(fnalu_hypre_parcsrcgnrsettol, FNALU_HYPRE_PARCSRCGNRSETTOL)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Real *);

#define NALU_HYPRE_ParCSRCGNRSetMinIter  \
        nalu_hypre_F90_NAME(fnalu_hypre_parcsrcgnrsetminiter, FNALU_HYPRE_PARCSRCGNRSETMINITER)
extern void nalu_hypre_F90_NAME(fnalu_hypre_parcsrcgnrsetminiter, FNALU_HYPRE_PARCSRCGNRSETMINITER)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_ParCSRCGNRSetMaxIter  \
        nalu_hypre_F90_NAME(fnalu_hypre_parcsrcgnrsetmaxiter, FNALU_HYPRE_PARCSRCGNRSETMAXITER)
extern void nalu_hypre_F90_NAME(fnalu_hypre_parcsrcgnrsetmaxiter, FNALU_HYPRE_PARCSRCGNRSETMAXITER)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_ParCSRCGNRSetStopCrit  \
        nalu_hypre_F90_NAME(fnalu_hypre_parcsrcgnrsetstopcri, FNALU_HYPRE_PARCSRCGNRSETSTOPCRI)
extern void nalu_hypre_F90_NAME(fnalu_hypre_parcsrcgnrsetstopcri, FNALU_HYPRE_PARCSRCGNRSETSTOPCRI)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_ParCSRCGNRSetPrecond  \
        nalu_hypre_F90_NAME(fnalu_hypre_parcsrcgnrsetprecond, FNALU_HYPRE_PARCSRCGNRSETPRECOND)
extern void nalu_hypre_F90_NAME(fnalu_hypre_parcsrcgnrsetprecond, FNALU_HYPRE_PARCSRCGNRSETPRECOND)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *, nalu_hypre_F90_Obj *);

#define NALU_HYPRE_ParCSRCGNRGetPrecond  \
        nalu_hypre_F90_NAME(fnalu_hypre_parcsrcgnrgetprecond, FNALU_HYPRE_PARCSRCGNRGETPRECOND)
extern void nalu_hypre_F90_NAME(fnalu_hypre_parcsrcgnrgetprecond, FNALU_HYPRE_PARCSRCGNRGETPRECOND)
(nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *);

#define NALU_HYPRE_ParCSRCGNRSetLogging  \
        nalu_hypre_F90_NAME(fnalu_hypre_parcsrcgnrsetlogging, FNALU_HYPRE_PARCSRCGNRSETLOGGING)
extern void nalu_hypre_F90_NAME(fnalu_hypre_parcsrcgnrsetlogging, FNALU_HYPRE_PARCSRCGNRSETLOGGING)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_ParCSRCGNRGetNumIteration  \
        nalu_hypre_F90_NAME(fnalu_hypre_parcsrcgnrgetnumiteratio, FNALU_HYPRE_PARCSRCGNRGETNUMITERATIO)
extern void nalu_hypre_F90_NAME(fnalu_hypre_parcsrcgnrgetnumiteratio, FNALU_HYPRE_PARCSRCGNRGETNUMITERATIO)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_ParCSRCGNRGetFinalRelativeResidualNorm  \
        nalu_hypre_F90_NAME(fnalu_hypre_parcsrcgnrgetfinalrelati, FNALU_HYPRE_PARCSRCGNRGETFINALRELATI)
extern void nalu_hypre_F90_NAME(fnalu_hypre_parcsrcgnrgetfinalrelati, FNALU_HYPRE_PARCSRCGNRGETFINALRELATI)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Real *);



#define NALU_HYPRE_EuclidCreate  \
        nalu_hypre_F90_NAME(fnalu_hypre_euclidcreate, FNALU_HYPRE_EUCLIDCREATE)
extern void nalu_hypre_F90_NAME(fnalu_hypre_euclidcreate, FNALU_HYPRE_EUCLIDCREATE)
(NALU_HYPRE_Int *, nalu_hypre_F90_Obj *);

#define NALU_HYPRE_EuclidDestroy  \
        nalu_hypre_F90_NAME(fnalu_hypre_eucliddestroy, FNALU_HYPRE_EUCLIDDESTROY)
extern void nalu_hypre_F90_NAME(fnalu_hypre_eucliddestroy, FNALU_HYPRE_EUCLIDDESTROY)
(nalu_hypre_F90_Obj *);

#define NALU_HYPRE_EuclidSetup  \
        nalu_hypre_F90_NAME(fnalu_hypre_euclidsetup, FNALU_HYPRE_EUCLIDSETUP)
extern void nalu_hypre_F90_NAME(fnalu_hypre_euclidsetup, FNALU_HYPRE_EUCLIDSETUP)
(nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *);

#define NALU_HYPRE_EuclidSolve  \
        nalu_hypre_F90_NAME(fnalu_hypre_euclidsolve, FNALU_HYPRE_EUCLIDSOLVE)
extern void nalu_hypre_F90_NAME(fnalu_hypre_euclidsolve, FNALU_HYPRE_EUCLIDSOLVE)
(nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *);

#define NALU_HYPRE_EuclidSetParams  \
        nalu_hypre_F90_NAME(fnalu_hypre_euclidsetparams, FNALU_HYPRE_EUCLIDSETPARAMS)
extern void nalu_hypre_F90_NAME(fnalu_hypre_euclidsetparams, FNALU_HYPRE_EUCLIDSETPARAMS)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *, char *);

#define NALU_HYPRE_EuclidSetParamsFromFile  \
        nalu_hypre_F90_NAME(fnalu_hypre_euclidsetparamsfromfile, FNALU_HYPRE_EUCLIDSETPARAMSFROMFILE)
extern void nalu_hypre_F90_NAME(fnalu_hypre_euclidsetparamsfromfile, FNALU_HYPRE_EUCLIDSETPARAMSFROMFILE)
(nalu_hypre_F90_Obj *, char *);



#define NALU_HYPRE_ParCSRGMRESCreate  \
        nalu_hypre_F90_NAME(fnalu_hypre_parcsrgmrescreate, FNALU_HYPRE_PARCSRGMRESCREATE)
extern void nalu_hypre_F90_NAME(fnalu_hypre_parcsrgmrescreate, FNALU_HYPRE_PARCSRGMRESCREATE)
(NALU_HYPRE_Int *, nalu_hypre_F90_Obj *);

#define NALU_HYPRE_ParCSRGMRESDestroy  \
        nalu_hypre_F90_NAME(fnalu_hypre_parcsrgmresdestroy, FNALU_HYPRE_PARCSRGMRESDESTROY)
extern void nalu_hypre_F90_NAME(fnalu_hypre_parcsrgmresdestroy, FNALU_HYPRE_PARCSRGMRESDESTROY)
(nalu_hypre_F90_Obj *);

#define NALU_HYPRE_ParCSRGMRESSetup  \
        nalu_hypre_F90_NAME(fnalu_hypre_parcsrgmressetup, FNALU_HYPRE_PARCSRGMRESSETUP)
extern void nalu_hypre_F90_NAME(fnalu_hypre_parcsrgmressetup, FNALU_HYPRE_PARCSRGMRESSETUP)
(nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *);

#define NALU_HYPRE_ParCSRGMRESSolve  \
        nalu_hypre_F90_NAME(fnalu_hypre_parcsrgmressolve, FNALU_HYPRE_PARCSRGMRESSOLVE)
extern void nalu_hypre_F90_NAME(fnalu_hypre_parcsrgmressolve, FNALU_HYPRE_PARCSRGMRESSOLVE)
(nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *);

#define NALU_HYPRE_ParCSRGMRESSetKDim  \
        nalu_hypre_F90_NAME(fnalu_hypre_parcsrgmressetkdim, FNALU_HYPRE_PARCSRGMRESSETKDIM)
extern void nalu_hypre_F90_NAME(fnalu_hypre_parcsrgmressetkdim, FNALU_HYPRE_PARCSRGMRESSETKDIM)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_ParCSRGMRESSetTol  \
        nalu_hypre_F90_NAME(fnalu_hypre_parcsrgmressettol, FNALU_HYPRE_PARCSRGMRESSETTOL)
extern void nalu_hypre_F90_NAME(fnalu_hypre_parcsrgmressettol, FNALU_HYPRE_PARCSRGMRESSETTOL)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Real *);

#define NALU_HYPRE_ParCSRGMRESSetMinIter  \
        nalu_hypre_F90_NAME(fnalu_hypre_parcsrgmressetminiter, FNALU_HYPRE_PARCSRGMRESSETMINITER)
extern void nalu_hypre_F90_NAME(fnalu_hypre_parcsrgmressetminiter, FNALU_HYPRE_PARCSRGMRESSETMINITER)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_ParCSRGMRESSetMaxIter  \
        nalu_hypre_F90_NAME(fnalu_hypre_parcsrgmressetmaxiter, FNALU_HYPRE_PARCSRGMRESSETMAXITER)
extern void nalu_hypre_F90_NAME(fnalu_hypre_parcsrgmressetmaxiter, FNALU_HYPRE_PARCSRGMRESSETMAXITER)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_ParCSRGMRESSetPrecond  \
        nalu_hypre_F90_NAME(fnalu_hypre_parcsrgmressetprecond, FNALU_HYPRE_PARCSRGMRESSETPRECOND)
extern void nalu_hypre_F90_NAME(fnalu_hypre_parcsrgmressetprecond, FNALU_HYPRE_PARCSRGMRESSETPRECOND)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *, nalu_hypre_F90_Obj *);

#define NALU_HYPRE_ParCSRGMRESGetPrecond  \
        nalu_hypre_F90_NAME(fnalu_hypre_parcsrgmresgetprecond, FNALU_HYPRE_PARCSRGMRESGETPRECOND)
extern void nalu_hypre_F90_NAME(fnalu_hypre_parcsrgmresgetprecond, FNALU_HYPRE_PARCSRGMRESGETPRECOND)
(nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *);

#define NALU_HYPRE_ParCSRGMRESSetLogging  \
        nalu_hypre_F90_NAME(fnalu_hypre_parcsrgmressetlogging, FNALU_HYPRE_PARCSRGMRESSETLOGGING)
extern void nalu_hypre_F90_NAME(fnalu_hypre_parcsrgmressetlogging, FNALU_HYPRE_PARCSRGMRESSETLOGGING)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_ParCSRGMRESSetPrintLevel  \
        nalu_hypre_F90_NAME(fnalu_hypre_parcsrgmressetprintlevel, FNALU_HYPRE_PARCSRGMRESSETPRINTLEVEL)
extern void nalu_hypre_F90_NAME(fnalu_hypre_parcsrgmressetprintlevel, FNALU_HYPRE_PARCSRGMRESSETPRINTLEVEL)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_ParCSRGMRESGetNumIterations  \
        nalu_hypre_F90_NAME(fnalu_hypre_parcsrgmresgetnumiterati, FNALU_HYPRE_PARCSRGMRESGETNUMITERATI)
extern void nalu_hypre_F90_NAME(fnalu_hypre_parcsrgmresgetnumiterati, FNALU_HYPRE_PARCSRGMRESGETNUMITERATI)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_ParCSRGMRESGetFinalRelativeResidualNorm  \
        nalu_hypre_F90_NAME(fnalu_hypre_parcsrgmresgetfinalrelat, FNALU_HYPRE_PARCSRGMRESGETFINALRELAT)
extern void nalu_hypre_F90_NAME(fnalu_hypre_parcsrgmresgetfinalrelat, FNALU_HYPRE_PARCSRGMRESGETFINALRELAT)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Real *);



#define NALU_HYPRE_ParCSRCOGMRESCreate  \
        nalu_hypre_F90_NAME(fnalu_hypre_parcsrcogmrescreate, FNALU_HYPRE_PARCSRCOGMRESCREATE)
extern void nalu_hypre_F90_NAME(fnalu_hypre_parcsrcogmrescreate, FNALU_HYPRE_PARCSRCOGMRESCREATE)
(NALU_HYPRE_Int *, nalu_hypre_F90_Obj *);

#define NALU_HYPRE_ParCSRCOGMRESDestroy  \
        nalu_hypre_F90_NAME(fnalu_hypre_parcsrcogmresdestroy, FNALU_HYPRE_PARCSRCOGMRESDESTROY)
extern void nalu_hypre_F90_NAME(fnalu_hypre_parcsrcogmresdestroy, FNALU_HYPRE_PARCSRCOGMRESDESTROY)
(nalu_hypre_F90_Obj *);

#define NALU_HYPRE_ParCSRCOGMRESSetup  \
        nalu_hypre_F90_NAME(fnalu_hypre_parcsrcogmressetup, FNALU_HYPRE_PARCSRCOGMRESSETUP)
extern void nalu_hypre_F90_NAME(fnalu_hypre_parcsrcogmressetup, FNALU_HYPRE_PARCSRCOGMRESSETUP)
(nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *);

#define NALU_HYPRE_ParCSRCOGMRESSolve  \
        nalu_hypre_F90_NAME(fnalu_hypre_parcsrcogmressolve, FNALU_HYPRE_PARCSRCOGMRESSOLVE)
extern void nalu_hypre_F90_NAME(fnalu_hypre_parcsrcogmressolve, FNALU_HYPRE_PARCSRCOGMRESSOLVE)
(nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *);

#define NALU_HYPRE_ParCSRCOGMRESSetKDim  \
        nalu_hypre_F90_NAME(fnalu_hypre_parcsrcogmressetkdim, FNALU_HYPRE_PARCSRCOGMRESSETKDIM)
extern void nalu_hypre_F90_NAME(fnalu_hypre_parcsrcogmressetkdim, FNALU_HYPRE_PARCSRCOGMRESSETKDIM)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_ParCSRCOGMRESSetUnroll  \
        nalu_hypre_F90_NAME(fnalu_hypre_parcsrcogmressetunroll, FNALU_HYPRE_PARCSRCOGMRESSETUNROLL)
extern void nalu_hypre_F90_NAME(fnalu_hypre_parcsrcogmressetunroll, FNALU_HYPRE_PARCSRCOGMRESSETUNROLL)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_ParCSRCOGMRESSetCGS  \
        nalu_hypre_F90_NAME(fnalu_hypre_parcsrcogmressetcgs, FNALU_HYPRE_PARCSRCOGMRESSETCGS)
extern void nalu_hypre_F90_NAME(fnalu_hypre_parcsrcogmressetcgs, FNALU_HYPRE_PARCSRCOGMRESSETCGS)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_ParCSRCOGMRESSetTol  \
        nalu_hypre_F90_NAME(fnalu_hypre_parcsrcogmressettol, FNALU_HYPRE_PARCSRCOGMRESSETTOL)
extern void nalu_hypre_F90_NAME(fnalu_hypre_parcsrcogmressettol, FNALU_HYPRE_PARCSRCOGMRESSETTOL)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Real *);

#define NALU_HYPRE_ParCSRCOGMRESSetAbsoluteTol  \
        nalu_hypre_F90_NAME(fnalu_hypre_parcsrcogmressetabsolutet, FNALU_HYPRE_PARCSRCOGMRESSETABSOLUTET)
extern void nalu_hypre_F90_NAME(fnalu_hypre_parcsrcogmressetabsolutet, FNALU_HYPRE_PARCSRCOGMRESSETABSOLUTET)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Real *);

#define NALU_HYPRE_ParCSRCOGMRESSetMinIter  \
        nalu_hypre_F90_NAME(fnalu_hypre_parcsrcogmressetminiter, FNALU_HYPRE_PARCSRCOGMRESSETMINITER)
extern void nalu_hypre_F90_NAME(fnalu_hypre_parcsrcogmressetminiter, FNALU_HYPRE_PARCSRCOGMRESSETMINITER)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_ParCSRCOGMRESSetMaxIter  \
        nalu_hypre_F90_NAME(fnalu_hypre_parcsrcogmressetmaxiter, FNALU_HYPRE_PARCSRCOGMRESSETMAXITER)
extern void nalu_hypre_F90_NAME(fnalu_hypre_parcsrcogmressetmaxiter, FNALU_HYPRE_PARCSRCOGMRESSETMAXITER)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_ParCSRCOGMRESSetPrecond  \
        nalu_hypre_F90_NAME(fnalu_hypre_parcsrcogmressetprecond, FNALU_HYPRE_PARCSRCOGMRESSETPRECOND)
extern void nalu_hypre_F90_NAME(fnalu_hypre_parcsrcogmressetprecond, FNALU_HYPRE_PARCSRCOGMRESSETPRECOND)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *, nalu_hypre_F90_Obj *);

#define NALU_HYPRE_ParCSRCOGMRESGetPrecond  \
        nalu_hypre_F90_NAME(fnalu_hypre_parcsrcogmresgetprecond, FNALU_HYPRE_PARCSRCOGMRESGETPRECOND)
extern void nalu_hypre_F90_NAME(fnalu_hypre_parcsrcogmresgetprecond, FNALU_HYPRE_PARCSRCOGMRESGETPRECOND)
(nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *);

#define NALU_HYPRE_ParCSRCOGMRESSetLogging  \
        nalu_hypre_F90_NAME(fnalu_hypre_parcsrcogmressetlogging, FNALU_HYPRE_PARCSRCOGMRESSETLOGGING)
extern void nalu_hypre_F90_NAME(fnalu_hypre_parcsrcogmressetlogging, FNALU_HYPRE_PARCSRCOGMRESSETLOGGING)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_ParCSRCOGMRESSetPrintLevel  \
        nalu_hypre_F90_NAME(fnalu_hypre_parcsrcogmressetprintlevel, FNALU_HYPRE_PARCSRCOGMRESSETPRINTLEVEL)
extern void nalu_hypre_F90_NAME(fnalu_hypre_parcsrcogmressetprintlevel, FNALU_HYPRE_PARCSRCOGMRESSETPRINTLEVEL)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_ParCSRCOGMRESGetNumIterations  \
        nalu_hypre_F90_NAME(fnalu_hypre_parcsrcogmresgetnumiterat, FNALU_HYPRE_PARCSRCOGMRESGETNUMITERAT)
extern void nalu_hypre_F90_NAME(fnalu_hypre_parcsrcogmresgetnumiterat, FNALU_HYPRE_PARCSRCOGMRESGETNUMITERAT)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_ParCSRCOGMRESGetFinalRelativeResidualNorm  \
        nalu_hypre_F90_NAME(fnalu_hypre_parcsrcogmresgetfinalrela, FNALU_HYPRE_PARCSRCOGMRESGETFINALRELA)
extern void nalu_hypre_F90_NAME(fnalu_hypre_parcsrcogmresgetfinalrela, FNALU_HYPRE_PARCSRCOGMRESGETFINALRELA)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Real *);



#define NALU_HYPRE_ParCSRHybridCreate \
        nalu_hypre_F90_NAME(fnalu_hypre_parcsrhybridcreate, FNALU_HYPRE_PARCSRHYBRIDCREATE)
extern void nalu_hypre_F90_NAME(fnalu_hypre_parcsrhybridcreate, FNALU_HYPRE_PARCSRHYBRIDCREATE)
(nalu_hypre_F90_Obj *);

#define NALU_HYPRE_ParCSRHybridDestroy \
        nalu_hypre_F90_NAME(fnalu_hypre_parcsrhybriddestroy, FNALU_HYPRE_PARCSRHYBRIDDESTROY)
extern void nalu_hypre_F90_NAME(fnalu_hypre_parcsrhybriddestroy, FNALU_HYPRE_PARCSRHYBRIDDESTROY)
(nalu_hypre_F90_Obj *);

#define NALU_HYPRE_ParCSRHybridSetup \
        nalu_hypre_F90_NAME(fnalu_hypre_parcsrhybridsetup, FNALU_HYPRE_PARCSRHYBRIDSETUP)
extern void nalu_hypre_F90_NAME(fnalu_hypre_parcsrhybridsetup, FNALU_HYPRE_PARCSRHYBRIDSETUP)
(nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *);

#define NALU_HYPRE_ParCSRHybridSolve \
        nalu_hypre_F90_NAME(fnalu_hypre_parcsrhybridsolve, FNALU_HYPRE_PARCSRHYBRIDSOLVE)
extern void nalu_hypre_F90_NAME(fnalu_hypre_parcsrhybridsolve, FNALU_HYPRE_PARCSRHYBRIDSOLVE)
(nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *);

#define NALU_HYPRE_ParCSRHybridSetTol \
        nalu_hypre_F90_NAME(fnalu_hypre_parcsrhybridsettol, FNALU_HYPRE_PARCSRHYBRIDSETTOL)
extern void nalu_hypre_F90_NAME(fnalu_hypre_parcsrhybridsettol, FNALU_HYPRE_PARCSRHYBRIDSETTOL)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Real *);

#define NALU_HYPRE_ParCSRHybridSetConvergenceTol \
        nalu_hypre_F90_NAME(fnalu_hypre_parcsrhybridsetconvergen, FNALU_HYPRE_PARCSRHYBRIDSETCONVERGEN)
extern void nalu_hypre_F90_NAME(fnalu_hypre_parcsrhybridsetconvergen, FNALU_HYPRE_PARCSRHYBRIDSETCONVERGEN)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Real *);

#define NALU_HYPRE_ParCSRHybridSetDSCGMaxIter \
        nalu_hypre_F90_NAME(fnalu_hypre_parcsrhybridsetdscgmaxit, FNALU_HYPRE_PARCSRHYBRIDSETDSCGMAXIT)
extern void nalu_hypre_F90_NAME(fnalu_hypre_parcsrhybridsetdscgmaxit, FNALU_HYPRE_PARCSRHYBRIDSETDSCGMAXIT)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_ParCSRHybridSetPCGMaxIter \
        nalu_hypre_F90_NAME(fnalu_hypre_parcsrhybridsetpcgmaxite, FNALU_HYPRE_PARCSRHYBRIDSETPCGMAXITE)
extern void nalu_hypre_F90_NAME(fnalu_hypre_parcsrhybridsetpcgmaxite, FNALU_HYPRE_PARCSRHYBRIDSETPCGMAXITE)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_ParCSRHybridSetSolverType \
        nalu_hypre_F90_NAME(fnalu_hypre_parcsrhybridsetsolvertyp, FNALU_HYPRE_PARCSRHYBRIDSETSOLVERTYP)
extern void nalu_hypre_F90_NAME(fnalu_hypre_parcsrhybridsetsolvertyp, FNALU_HYPRE_PARCSRHYBRIDSETSOLVERTYP)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_ParCSRHybridSetKDim \
        nalu_hypre_F90_NAME(fnalu_hypre_parcsrhybridsetkdim, FNALU_HYPRE_PARCSRHYBRIDSETKDIM)
extern void nalu_hypre_F90_NAME(fnalu_hypre_parcsrhybridsetkdim, FNALU_HYPRE_PARCSRHYBRIDSETKDIM)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_ParCSRHybridSetTwoNorm \
        nalu_hypre_F90_NAME(fnalu_hypre_parcsrhybridsettwonorm, FNALU_HYPRE_PARCSRHYBRIDSETTWONORM)
extern void nalu_hypre_F90_NAME(fnalu_hypre_parcsrhybridsettwonorm, FNALU_HYPRE_PARCSRHYBRIDSETTWONORM)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_ParCSRHybridSetStopCrit \
        nalu_hypre_F90_NAME(fnalu_hypre_parcsrhybridsetstopcrit, FNALU_HYPRE_PARCSRSETSTOPCRIT)
extern void nalu_hypre_F90_NAME(fnalu_hypre_parcsrhybridsetstopcrit, FNALU_HYPRE_PARCSRSETSTOPCRIT)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_ParCSRHybridSetRelChange \
        nalu_hypre_F90_NAME(fnalu_hypre_parcsrhybridsetrelchange, FNALU_HYPRE_PARCSRHYBRIDSETRELCHANGE)
extern void nalu_hypre_F90_NAME(fnalu_hypre_parcsrhybridsetrelchange, FNALU_HYPRE_PARCSRHYBRIDSETRELCHANGE)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_ParCSRHybridSetPrecond \
        nalu_hypre_F90_NAME(fnalu_hypre_parcsrhybridsetprecond, FNALU_HYPRE_PARCSRHYBRIDSETPRECOND)
extern void nalu_hypre_F90_NAME(fnalu_hypre_parcsrhybridsetprecond, FNALU_HYPRE_PARCSRHYBRIDSETPRECOND)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *, nalu_hypre_F90_Obj *);

#define NALU_HYPRE_ParCSRHybridSetLogging \
        nalu_hypre_F90_NAME(fnalu_hypre_parcsrhybridsetlogging, FNALU_HYPRE_PARCSRHYBRIDSETLOGGING)
extern void nalu_hypre_F90_NAME(fnalu_hypre_parcsrhybridsetlogging, FNALU_HYPRE_PARCSRHYBRIDSETLOGGING)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_ParCSRHybridSetPrintLevel \
        nalu_hypre_F90_NAME(fnalu_hypre_parcsrhybridsetprintleve, FNALU_HYPRE_PARCSRHYBRIDSETPRINTLEVE)
extern void nalu_hypre_F90_NAME(fnalu_hypre_parcsrhybridsetprintleve, FNALU_HYPRE_PARCSRHYBRIDSETPRINTLEVE)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_ParCSRHybridSetStrongThreshold \
        nalu_hypre_F90_NAME(fnalu_hypre_parcsrhybridsetstrongthr, FNALU_HYPRE_PARCSRHYBRIDSETSTRONGTHR)
extern void nalu_hypre_F90_NAME(fnalu_hypre_parcsrhybridsetstrongthr, FNALU_HYPRE_PARCSRHYBRIDSETSTRONGTHR)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_ParCSRHybridSetMaxRowSum \
        nalu_hypre_F90_NAME(fnalu_hypre_parcsrhybridsetmaxrowsum, FNALU_HYPRE_PARCSRHYBRIDSETMAXROWSUM)
extern void nalu_hypre_F90_NAME(fnalu_hypre_parcsrhybridsetmaxrowsum, FNALU_HYPRE_PARCSRHYBRIDSETMAXROWSUM)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_ParCSRHybridSetTruncFactor \
        nalu_hypre_F90_NAME(fnalu_hypre_parcsrhybridsettruncfact, FNALU_HYPRE_PARCSRHYBRIDSETTRUNCFACT)
extern void nalu_hypre_F90_NAME(fnalu_hypre_parcsrhybridsettruncfact, FNALU_HYPRE_PARCSRHYBRIDSETTRUNCFACT)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_ParCSRHybridSetMaxLevels \
        nalu_hypre_F90_NAME(fnalu_hypre_parcsrhybridsetmaxlevels, FNALU_HYPRE_PARCSRHYBRIDSETMAXLEVELS)
extern void nalu_hypre_F90_NAME(fnalu_hypre_parcsrhybridsetmaxlevels, FNALU_HYPRE_PARCSRHYBRIDSETMAXLEVELS)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_ParCSRHybridSetMeasureType \
        nalu_hypre_F90_NAME(fnalu_hypre_parcsrhybridsetmeasurety, FNALU_HYPRE_PARCSRHYBRIDSETMEASURETY)
extern void nalu_hypre_F90_NAME(fnalu_hypre_parcsrhybridsetmeasurety, FNALU_HYPRE_PARCSRHYBRIDSETMEASURETY)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_ParCSRHybridSetCoarsenType \
        nalu_hypre_F90_NAME(fnalu_hypre_parcsrhybridsetcoarsenty, FNALU_HYPRE_PARCSRHYBRIDSETCOARSENTY)
extern void nalu_hypre_F90_NAME(fnalu_hypre_parcsrhybridsetcoarsenty, FNALU_HYPRE_PARCSRHYBRIDSETCOARSENTY)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_ParCSRHybridSetCycleType \
        nalu_hypre_F90_NAME(fnalu_hypre_parcsrhybridsetcycletype, FNALU_HYPRE_PARCSRHYBRIDSETCYCLETYPE)
extern void nalu_hypre_F90_NAME(fnalu_hypre_parcsrhybridsetcycletype, FNALU_HYPRE_PARCSRHYBRIDSETCYCLETYPE)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_ParCSRHybridSetNumGridSweeps \
        nalu_hypre_F90_NAME(fnalu_hypre_parcsrhybridsetnumgridsw, FNALU_HYPRE_PARCSRHYBRIDSETNUMGRIDSW)
extern void nalu_hypre_F90_NAME(fnalu_hypre_parcsrhybridsetnumgridsw, FNALU_HYPRE_PARCSRHYBRIDSETNUMGRIDSW)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_ParCSRHybridSetGridRelaxType \
        nalu_hypre_F90_NAME(fnalu_hypre_parcsrhybridsetgridrlxty, FNALU_HYPRE_PARCSRHYBRIDSETGRIDRLXTY)
extern void nalu_hypre_F90_NAME(fnalu_hypre_parcsrhybridsetgridrlxty, FNALU_HYPRE_PARCSRHYBRIDSETGRIDRLXTY)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_ParCSRHybridSetGridRelaxPoints \
        nalu_hypre_F90_NAME(fnalu_hypre_parcsrhybridsetgridrlxpt, FNALU_HYPRE_PARCSRHYBRIDSETGRIDRLXPT)
extern void nalu_hypre_F90_NAME(fnalu_hypre_parcsrhybridsetgridrlxpt, FNALU_HYPRE_PARCSRHYBRIDSETGRIDRLXPT)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_ParCSRHybridSetNumSweeps \
        nalu_hypre_F90_NAME(fnalu_hypre_parcsrhybridsetnumsweeps, FNALU_HYPRE_PARCSRHYBRIDSETNUMSWEEPS)
extern void nalu_hypre_F90_NAME(fnalu_hypre_parcsrhybridsetnumsweeps, FNALU_HYPRE_PARCSRHYBRIDSETNUMSWEEPS)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_ParCSRHybridSetCycleNumSweeps \
        nalu_hypre_F90_NAME(fnalu_hypre_parcsrhybridsetcyclenums, FNALU_HYPRE_PARCSRHYBRIDSETCYCLENUMS)
extern void nalu_hypre_F90_NAME(fnalu_hypre_parcsrhybridsetcyclenums, FNALU_HYPRE_PARCSRHYBRIDSETCYCLENUMS)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_ParCSRHybridSetRelaxType \
        nalu_hypre_F90_NAME(fnalu_hypre_parcsrhybridsetrelaxtype, FNALU_HYPRE_PARCSRHYBRIDSETRELAXTYPE)
extern void nalu_hypre_F90_NAME(fnalu_hypre_parcsrhybridsetrelaxtype, FNALU_HYPRE_PARCSRHYBRIDSETRELAXTYPE)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_ParCSRHybridSetCycleRelaxType \
        nalu_hypre_F90_NAME(fnalu_hypre_parcsrhybridsetcyclerela, FNALU_HYPRE_PARCSRHYBRIDSETCYCLERELA)
extern void nalu_hypre_F90_NAME(fnalu_hypre_parcsrhybridsetcyclerela, FNALU_HYPRE_PARCSRHYBRIDSETCYCLERELA)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_ParCSRHybridSetRelaxOrder \
        nalu_hypre_F90_NAME(fnalu_hypre_parcsrhybridsetrelaxorde, FNALU_HYPRE_PARCSRHYBRIDSETRELAXORDE)
extern void nalu_hypre_F90_NAME(fnalu_hypre_parcsrhybridsetrelaxorde, FNALU_HYPRE_PARCSRHYBRIDSETRELAXORDE)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_ParCSRHybridSetRelaxWt \
        nalu_hypre_F90_NAME(fnalu_hypre_parcsrhybridsetrelaxwt, FNALU_HYPRE_PARCSRHYBRIDSETRELAXWT)
extern void nalu_hypre_F90_NAME(fnalu_hypre_parcsrhybridsetrelaxwt, FNALU_HYPRE_PARCSRHYBRIDSETRELAXWT)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_ParCSRHybridSetLevelRelaxWt \
        nalu_hypre_F90_NAME(fnalu_hypre_parcsrhybridsetlevelrela, FNALU_HYPRE_PARCSRHYBRIDSETLEVELRELA)
extern void nalu_hypre_F90_NAME(fnalu_hypre_parcsrhybridsetlevelrela, FNALU_HYPRE_PARCSRHYBRIDSETLEVELRELA)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_ParCSRHybridSetOuterWt \
        nalu_hypre_F90_NAME(fnalu_hypre_parcsrhybridsetouterwt, FNALU_HYPRE_PARCSRHYBRIDSETOUTERWT)
extern void nalu_hypre_F90_NAME(fnalu_hypre_parcsrhybridsetouterwt, FNALU_HYPRE_PARCSRHYBRIDSETOUTERWT)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_ParCSRHybridSetLevelOuterWt \
        nalu_hypre_F90_NAME(fnalu_hypre_parcsrhybridsetleveloute, FNALU_HYPRE_PARCSRHYBRIDSETLEVELOUTE)
extern void nalu_hypre_F90_NAME(fnalu_hypre_parcsrhybridsetleveloute, FNALU_HYPRE_PARCSRHYBRIDSETLEVELOUTE)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_ParCSRHybridSetRelaxWeight \
        nalu_hypre_F90_NAME(fnalu_hypre_parcsrhybridsetrelaxweig, FNALU_HYPRE_PARCSRHYBRIDSETRELAXWEIG)
extern void nalu_hypre_F90_NAME(fnalu_hypre_parcsrhybridsetrelaxweig, FNALU_HYPRE_PARCSRHYBRIDSETRELAXWEIG)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_ParCSRHybridSetOmega \
        nalu_hypre_F90_NAME(fnalu_hypre_parcsrhybridsetomega, FNALU_HYPRE_PARCSRHYBRIDSETOMEGA)
extern void nalu_hypre_F90_NAME(fnalu_hypre_parcsrhybridsetomega, FNALU_HYPRE_PARCSRHYBRIDSETOMEGA)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_ParCSRHybridGetNumIterations \
        nalu_hypre_F90_NAME(fnalu_hypre_parcsrhybridgetnumiterat, FNALU_HYPRE_PARCSRHYBRIDGETNUMITERAT)
extern void nalu_hypre_F90_NAME(fnalu_hypre_parcsrhybridgetnumiterat, FNALU_HYPRE_PARCSRHYBRIDGETNUMITERAT)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_ParCSRHybridGetDSCGNumIterations \
        nalu_hypre_F90_NAME(fnalu_hypre_parcsrhybridgetdscgnumit, FNALU_HYPRE_PARCSRHYBRIDGETDSCGNUMIT)
extern void nalu_hypre_F90_NAME(fnalu_hypre_parcsrhybridgetdscgnumit, FNALU_HYPRE_PARCSRHYBRIDGETDSCGNUMIT)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_ParCSRHybridGetPCGNumIterations \
        nalu_hypre_F90_NAME(fnalu_hypre_parcsrhybridgetpcgnumite, FNALU_HYPRE_PARCSRHYBRIDGETPCGNUMITE)
extern void nalu_hypre_F90_NAME(fnalu_hypre_parcsrhybridgetpcgnumite, FNALU_HYPRE_PARCSRHYBRIDGETPCGNUMITE)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_ParCSRHybridGetFinalRelativeResidualNorm \
        nalu_hypre_F90_NAME(fnalu_hypre_parcsrhybridgetfinalrela, FNALU_HYPRE_PARCSRHYBRIDGETFINALRELA)
extern void nalu_hypre_F90_NAME(fnalu_hypre_parcsrhybridgetfinalrela, FNALU_HYPRE_PARCSRHYBRIDGETFINALRELA)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_ParSetRandomValues \
        nalu_hypre_F90_NAME(fnalu_hypre_parsetrandomvalues, FNALU_HYPRE_PARSETRANDOMVALUES)
extern void nalu_hypre_F90_NAME(fnalu_hypre_parsetrandomvalues, FNALU_HYPRE_PARSETRANDOMVALUES)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_ParPrintVector \
        nalu_hypre_F90_NAME(fnalu_hypre_parprintvector, FNALU_HYPRE_PARPRINTVECTOR)
extern void nalu_hypre_F90_NAME(fnalu_hypre_parprintvector, FNALU_HYPRE_PARPRINTVECTOR)
(nalu_hypre_F90_Obj *, char *);

#define NALU_HYPRE_ParReadVector \
        nalu_hypre_F90_NAME(fnalu_hypre_parreadvector, FNALU_HYPRE_PARREADVECTOR)
extern void nalu_hypre_F90_NAME(fnalu_hypre_parreadvector, FNALU_HYPRE_PARREADVECTOR)
(NALU_HYPRE_Int *, char *);

#define NALU_HYPRE_ParVectorSize \
        nalu_hypre_F90_NAME(fnalu_hypre_parvectorsize, FNALU_HYPRE_PARVECTORSIZE)
extern void nalu_hypre_F90_NAME(fnalu_hypre_parvectorsize, FNALU_HYPRE_PARVECTORSIZE)
(NALU_HYPRE_Int *);

#define NALU_HYPRE_ParCSRMultiVectorPrint \
        nalu_hypre_F90_NAME(fnalu_hypre_parcsrmultivectorprint, FNALU_HYPRE_PARCSRMULTIVECTORPRINT)
extern void nalu_hypre_F90_NAME(fnalu_hypre_parcsrmultivectorprint, FNALU_HYPRE_PARCSRMULTIVECTORPRINT)
(NALU_HYPRE_Int *, char *);

#define NALU_HYPRE_ParCSRMultiVectorRead \
        nalu_hypre_F90_NAME(fnalu_hypre_parcsrmultivectorread, FNALU_HYPRE_PARCSRMULTIVECTORREAD)
extern void nalu_hypre_F90_NAME(fnalu_hypre_parcsrmultivectorread, FNALU_HYPRE_PARCSRMULTIVECTORREAD)
(NALU_HYPRE_Int *, nalu_hypre_F90_Obj *, char *);

#define aux_maskCount \
        nalu_hypre_F90_NAME(fnalu_hypre_aux_maskcount, FNALU_HYPRE_AUX_MASKCOUNT)
extern void nalu_hypre_F90_NAME(fnalu_hypre_aux_maskcount, FNALU_HYPRE_AUX_MASKCOUNT)
(NALU_HYPRE_Int *, NALU_HYPRE_Int *);

#define aux_indexFromMask \
        nalu_hypre_F90_NAME(fnalu_hypre_auxindexfrommask, FNALU_HYPRE_AUXINDEXFROMMASK)
extern void nalu_hypre_F90_NAME(fnalu_hypre_auxindexfrommask, FNALU_HYPRE_AUXINDEXFROMMASK)
(NALU_HYPRE_Int *, NALU_HYPRE_Int *, NALU_HYPRE_Int *);

#define NALU_HYPRE_TempParCSRSetupInterpreter \
        nalu_hypre_F90_NAME(fnalu_hypre_tempparcsrsetupinterpret, FNALU_HYPRE_TEMPPARCSRSETUPINTERPRET)
extern void nalu_hypre_F90_NAME(fnalu_hypre_tempparcsrsetupinterpret, FNALU_HYPRE_TEMPPARCSRSETUPINTERPRET)
(nalu_hypre_F90_Obj *);

#define NALU_HYPRE_ParCSRSetupInterpreter \
        nalu_hypre_F90_NAME(fnalu_hypre_parcsrsetupinterpreter, FNALU_HYPRE_PARCSRSETUPINTERPRETER)
extern void nalu_hypre_F90_NAME(fnalu_hypre_parcsrsetupinterpreter, FNALU_HYPRE_PARCSRSETUPINTERPRETER)
(nalu_hypre_F90_Obj *);

#define NALU_HYPRE_ParCSRSetupMatvec \
        nalu_hypre_F90_NAME(fnalu_hypre_parcsrsetupmatvec, FNALU_HYPRE_PARCSRSETUPMATVEC)
extern void nalu_hypre_F90_NAME(fnalu_hypre_parcsrsetupmatvec, FNALU_HYPRE_PARCSRSETUPMATVEC)
(nalu_hypre_F90_Obj *);



#define NALU_HYPRE_ParaSailsCreate  \
        nalu_hypre_F90_NAME(fnalu_hypre_parasailscreate, FNALU_HYPRE_PARASAILSCREATE)
extern void nalu_hypre_F90_NAME(fnalu_hypre_parasailscreate, FNALU_HYPRE_PARASAILSCREATE)
(NALU_HYPRE_Int *, nalu_hypre_F90_Obj *);

#define NALU_HYPRE_ParaSailsDestroy  \
        nalu_hypre_F90_NAME(fnalu_hypre_parasailsdestroy, FNALU_HYPRE_PARASAILSDESTROY)
extern void nalu_hypre_F90_NAME(fnalu_hypre_parasailsdestroy, FNALU_HYPRE_PARASAILSDESTROY)
(nalu_hypre_F90_Obj *);

#define NALU_HYPRE_ParaSailsSetup  \
        nalu_hypre_F90_NAME(fnalu_hypre_parasailssetup, FNALU_HYPRE_PARASAILSSETUP)
extern void nalu_hypre_F90_NAME(fnalu_hypre_parasailssetup, FNALU_HYPRE_PARASAILSSETUP)
(nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *);

#define NALU_HYPRE_ParaSailsSolve  \
        nalu_hypre_F90_NAME(fnalu_hypre_parasailssolve, FNALU_HYPRE_PARASAILSSOLVE)
extern void nalu_hypre_F90_NAME(fnalu_hypre_parasailssolve, FNALU_HYPRE_PARASAILSSOLVE)
(nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *);

#define NALU_HYPRE_ParaSailsSetParams  \
        nalu_hypre_F90_NAME(fnalu_hypre_parasailssetparams, FNALU_HYPRE_PARASAILSSETPARAMS)
extern void nalu_hypre_F90_NAME(fnalu_hypre_parasailssetparams, FNALU_HYPRE_PARASAILSSETPARAMS)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Real *, NALU_HYPRE_Int *);

#define NALU_HYPRE_ParaSailsSetThresh  \
        nalu_hypre_F90_NAME(fnalu_hypre_parasailssetthresh, FNALU_HYPRE_PARASAILSSETTHRESH)
extern void nalu_hypre_F90_NAME(fnalu_hypre_parasailssetthresh, FNALU_HYPRE_PARASAILSSETTHRESH)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Real *);

#define NALU_HYPRE_ParaSailsGetThresh  \
        nalu_hypre_F90_NAME(fnalu_hypre_parasailsgetthresh, FNALU_HYPRE_PARASAILSGETTHRESH)
extern void nalu_hypre_F90_NAME(fnalu_hypre_parasailsgetthresh, FNALU_HYPRE_PARASAILSGETTHRESH)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Real *);

#define NALU_HYPRE_ParaSailsSetNlevels  \
        nalu_hypre_F90_NAME(fnalu_hypre_parasailssetnlevels, FNALU_HYPRE_PARASAILSSETNLEVELS)
extern void nalu_hypre_F90_NAME(fnalu_hypre_parasailssetnlevels, FNALU_HYPRE_PARASAILSSETNLEVELS)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_ParaSailsGetNlevels  \
        nalu_hypre_F90_NAME(fnalu_hypre_parasailsgetnlevels, FNALU_HYPRE_PARASAILSGETNLEVELS)
extern void nalu_hypre_F90_NAME(fnalu_hypre_parasailsgetnlevels, FNALU_HYPRE_PARASAILSGETNLEVELS)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_ParaSailsSetFilter  \
        nalu_hypre_F90_NAME(fnalu_hypre_parasailssetfilter, FNALU_HYPRE_PARASAILSSETFILTER)
extern void nalu_hypre_F90_NAME(fnalu_hypre_parasailssetfilter, FNALU_HYPRE_PARASAILSSETFILTER)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Real *);

#define NALU_HYPRE_ParaSailsGetFilter  \
        nalu_hypre_F90_NAME(fnalu_hypre_parasailsgetfilter, FNALU_HYPRE_PARASAILSGETFILTER)
extern void nalu_hypre_F90_NAME(fnalu_hypre_parasailsgetfilter, FNALU_HYPRE_PARASAILSGETFILTER)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Real *);

#define NALU_HYPRE_ParaSailsSetSym  \
        nalu_hypre_F90_NAME(fnalu_hypre_parasailssetsym, FNALU_HYPRE_PARASAILSSETSYM)
extern void nalu_hypre_F90_NAME(fnalu_hypre_parasailssetsym, FNALU_HYPRE_PARASAILSSETSYM)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_ParaSailsGetSym  \
        nalu_hypre_F90_NAME(fnalu_hypre_parasailsgetsym, FNALU_HYPRE_PARASAILSGETSYM)
extern void nalu_hypre_F90_NAME(fnalu_hypre_parasailsgetsym, FNALU_HYPRE_PARASAILSGETSYM)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_ParaSailsSetLoadbal  \
        nalu_hypre_F90_NAME(fnalu_hypre_parasailssetloadbal, FNALU_HYPRE_PARASAILSSETLOADBAL)
extern void nalu_hypre_F90_NAME(fnalu_hypre_parasailssetloadbal, FNALU_HYPRE_PARASAILSSETLOADBAL)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Real *);

#define NALU_HYPRE_ParaSailsGetLoadbal  \
        nalu_hypre_F90_NAME(fnalu_hypre_parasailsgetloadbal, FNALU_HYPRE_PARASAILSGETLOADBAL)
extern void nalu_hypre_F90_NAME(fnalu_hypre_parasailsgetloadbal, FNALU_HYPRE_PARASAILSGETLOADBAL)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Real *);

#define NALU_HYPRE_ParaSailsSetReuse  \
        nalu_hypre_F90_NAME(fnalu_hypre_parasailssetreuse, FNALU_HYPRE_PARASAILSSETREUSE)
extern void nalu_hypre_F90_NAME(fnalu_hypre_parasailssetreuse, FNALU_HYPRE_PARASAILSSETREUSE)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_ParaSailsGetReuse  \
        nalu_hypre_F90_NAME(fnalu_hypre_parasailsgetreuse, FNALU_HYPRE_PARASAILSGETREUSE)
extern void nalu_hypre_F90_NAME(fnalu_hypre_parasailsgetreuse, FNALU_HYPRE_PARASAILSGETREUSE)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_ParaSailsSetLogging  \
        nalu_hypre_F90_NAME(fnalu_hypre_parasailssetlogging, FNALU_HYPRE_PARASAILSSETLOGGING)
extern void nalu_hypre_F90_NAME(fnalu_hypre_parasailssetlogging, FNALU_HYPRE_PARASAILSSETLOGGING)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_ParaSailsGetLogging  \
        nalu_hypre_F90_NAME(fnalu_hypre_parasailsgetlogging, FNALU_HYPRE_PARASAILSGETLOGGING)
extern void nalu_hypre_F90_NAME(fnalu_hypre_parasailsgetlogging, FNALU_HYPRE_PARASAILSGETLOGGING)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);



#define NALU_HYPRE_ParCSRPCGCreate  \
        nalu_hypre_F90_NAME(fnalu_hypre_parcsrpcgcreate, FNALU_HYPRE_PARCSRPCGCREATE)
extern void nalu_hypre_F90_NAME(fnalu_hypre_parcsrpcgcreate, FNALU_HYPRE_PARCSRPCGCREATE)
(NALU_HYPRE_Int *, nalu_hypre_F90_Obj *);

#define NALU_HYPRE_ParCSRPCGDestroy  \
        nalu_hypre_F90_NAME(fnalu_hypre_parcsrpcgdestroy, FNALU_HYPRE_PARCSRPCGDESTROY)
extern void nalu_hypre_F90_NAME(fnalu_hypre_parcsrpcgdestroy, FNALU_HYPRE_PARCSRPCGDESTROY)
(nalu_hypre_F90_Obj *);

#define NALU_HYPRE_ParCSRPCGSetup  \
        nalu_hypre_F90_NAME(fnalu_hypre_parcsrpcgsetup, FNALU_HYPRE_PARCSRPCGSETUP)
extern void nalu_hypre_F90_NAME(fnalu_hypre_parcsrpcgsetup, FNALU_HYPRE_PARCSRPCGSETUP)
(nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *);

#define NALU_HYPRE_ParCSRPCGSolve  \
        nalu_hypre_F90_NAME(fnalu_hypre_parcsrpcgsolve, FNALU_HYPRE_PARCSRPCGSOLVE)
extern void nalu_hypre_F90_NAME(fnalu_hypre_parcsrpcgsolve, FNALU_HYPRE_PARCSRPCGSOLVE)
(nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *);

#define NALU_HYPRE_ParCSRPCGSetTol  \
        nalu_hypre_F90_NAME(fnalu_hypre_parcsrpcgsettol, FNALU_HYPRE_PARCSRPCGSETTOL)
extern void nalu_hypre_F90_NAME(fnalu_hypre_parcsrpcgsettol, FNALU_HYPRE_PARCSRPCGSETTOL)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Real *);

#define NALU_HYPRE_ParCSRPCGSetMaxIter  \
        nalu_hypre_F90_NAME(fnalu_hypre_parcsrpcgsetmaxiter, FNALU_HYPRE_PARCSRPCGSETMAXITER)
extern void nalu_hypre_F90_NAME(fnalu_hypre_parcsrpcgsetmaxiter, FNALU_HYPRE_PARCSRPCGSETMAXITER)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_ParCSRPCGSetStopCrit  \
        nalu_hypre_F90_NAME(fnalu_hypre_parcsrpcgsetstopcrit, FNALU_HYPRE_PARCSRPCGSETSTOPCRIT)
extern void nalu_hypre_F90_NAME(fnalu_hypre_parcsrpcgsetstopcrit, FNALU_HYPRE_PARCSRPCGSETSTOPCRIT)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_ParCSRPCGSetTwoNorm  \
        nalu_hypre_F90_NAME(fnalu_hypre_parcsrpcgsettwonorm, FNALU_HYPRE_PARCSRPCGSETTWONORM)
extern void nalu_hypre_F90_NAME(fnalu_hypre_parcsrpcgsettwonorm, FNALU_HYPRE_PARCSRPCGSETTWONORM)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_ParCSRPCGSetRelChange  \
        nalu_hypre_F90_NAME(fnalu_hypre_parcsrpcgsetrelchange, FNALU_HYPRE_PARCSRPCGSETRELCHANGE)
extern void nalu_hypre_F90_NAME(fnalu_hypre_parcsrpcgsetrelchange, FNALU_HYPRE_PARCSRPCGSETRELCHANGE)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_ParCSRPCGSetPrecond  \
        nalu_hypre_F90_NAME(fnalu_hypre_parcsrpcgsetprecond, FNALU_HYPRE_PARCSRPCGSETPRECOND)
extern void nalu_hypre_F90_NAME(fnalu_hypre_parcsrpcgsetprecond, FNALU_HYPRE_PARCSRPCGSETPRECOND)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *, nalu_hypre_F90_Obj *);

#define NALU_HYPRE_ParCSRPCGGetPrecond  \
        nalu_hypre_F90_NAME(fnalu_hypre_parcsrpcggetprecond, FNALU_HYPRE_PARCSRPCGGETPRECOND)
extern void nalu_hypre_F90_NAME(fnalu_hypre_parcsrpcggetprecond, FNALU_HYPRE_PARCSRPCGGETPRECOND)
(nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *);

#define NALU_HYPRE_ParCSRPCGSetPrintLevel  \
        nalu_hypre_F90_NAME(fnalu_hypre_parcsrpcgsetprintlevel, FNALU_HYPRE_PARCSRPCGSETPRINTLEVEL)
extern void nalu_hypre_F90_NAME(fnalu_hypre_parcsrpcgsetprintlevel, FNALU_HYPRE_PARCSRPCGSETPRINTLEVEL)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_ParCSRPCGGetNumIterations  \
        nalu_hypre_F90_NAME(fnalu_hypre_parcsrpcggetnumiteration, FNALU_HYPRE_PARCSRPCGGETNUMITERATION)
extern void nalu_hypre_F90_NAME(fnalu_hypre_parcsrpcggetnumiteration, FNALU_HYPRE_PARCSRPCGGETNUMITERATION)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_ParCSRPCGGetFinalRelativeResidualNorm  \
        nalu_hypre_F90_NAME(fnalu_hypre_parcsrpcggetfinalrelativ, FNALU_HYPRE_PARCSRPCGGETFINALRELATIV)
extern void nalu_hypre_F90_NAME(fnalu_hypre_parcsrpcggetfinalrelativ, FNALU_HYPRE_PARCSRPCGGETFINALRELATIV)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Real *);



#define NALU_HYPRE_ParCSRDiagScaleSetup  \
        nalu_hypre_F90_NAME(fnalu_hypre_parcsrdiagscalesetup, FNALU_HYPRE_PARCSRDIAGSCALESETUP)
extern void nalu_hypre_F90_NAME(fnalu_hypre_parcsrdiagscalesetup, FNALU_HYPRE_PARCSRDIAGSCALESETUP)
(nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *);

#define NALU_HYPRE_ParCSRDiagScale  \
        nalu_hypre_F90_NAME(fnalu_hypre_parcsrdiagscale, FNALU_HYPRE_PARCSRDIAGSCALE)
extern void nalu_hypre_F90_NAME(fnalu_hypre_parcsrdiagscale, FNALU_HYPRE_PARCSRDIAGSCALE)
(nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *);



#define NALU_HYPRE_ParCSRPilutCreate  \
        nalu_hypre_F90_NAME(fnalu_hypre_parcsrpilutcreate, FNALU_HYPRE_PARCSRPILUTCREATE)
extern void nalu_hypre_F90_NAME(fnalu_hypre_parcsrpilutcreate, FNALU_HYPRE_PARCSRPILUTCREATE)
(NALU_HYPRE_Int *, nalu_hypre_F90_Obj *);

#define NALU_HYPRE_ParCSRPilutDestroy  \
        nalu_hypre_F90_NAME(fnalu_hypre_parcsrpilutdestroy, FNALU_HYPRE_PARCSRPILUTDESTROY)
extern void nalu_hypre_F90_NAME(fnalu_hypre_parcsrpilutdestroy, FNALU_HYPRE_PARCSRPILUTDESTROY)
(nalu_hypre_F90_Obj *);

#define NALU_HYPRE_ParCSRPilutSetup  \
        nalu_hypre_F90_NAME(fnalu_hypre_parcsrpilutsetup, FNALU_HYPRE_PARCSRPILUTSETUP)
extern void nalu_hypre_F90_NAME(fnalu_hypre_parcsrpilutsetup, FNALU_HYPRE_PARCSRPILUTSETUP)
(nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *);

#define NALU_HYPRE_ParCSRPilutSolve  \
        nalu_hypre_F90_NAME(fnalu_hypre_parcsrpilutsolve, FNALU_HYPRE_PARCSRPILUTSOLVE)
extern void nalu_hypre_F90_NAME(fnalu_hypre_parcsrpilutsolve, FNALU_HYPRE_PARCSRPILUTSOLVE)
(nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *);

#define NALU_HYPRE_ParCSRPilutSetMaxIter  \
        nalu_hypre_F90_NAME(fnalu_hypre_parcsrpilutsetmaxiter, FNALU_HYPRE_PARCSRPILUTSETMAXITER)
extern void nalu_hypre_F90_NAME(fnalu_hypre_parcsrpilutsetmaxiter, FNALU_HYPRE_PARCSRPILUTSETMAXITER)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_ParCSRPilutSetDropToleran  \
        nalu_hypre_F90_NAME(fnalu_hypre_parcsrpilutsetdroptolera, FNALU_HYPRE_PARCSRPILUTSETDROPTOLERA)
extern void nalu_hypre_F90_NAME(fnalu_hypre_parcsrpilutsetdroptolera, FNALU_HYPRE_PARCSRPILUTSETDROPTOLERA)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Real *);

#define NALU_HYPRE_ParCSRPilutSetFacRowSize  \
        nalu_hypre_F90_NAME(fnalu_hypre_parcsrpilutsetfacrowsize, FNALU_HYPRE_PARCSRPILUTSETFACROWSIZE)
extern void nalu_hypre_F90_NAME(fnalu_hypre_parcsrpilutsetfacrowsize, FNALU_HYPRE_PARCSRPILUTSETFACROWSIZE)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);



#define NALU_HYPRE_SchwarzCreate \
        nalu_hypre_F90_NAME(fnalu_hypre_schwarzcreate, FNALU_HYPRE_SCHWARZCREATE)
extern void nalu_hypre_F90_NAME(fnalu_hypre_schwarzcreate, FNALU_HYPRE_SCHWARZCREATE)
(nalu_hypre_F90_Obj *);

#define NALU_HYPRE_SchwarzDestroy \
        nalu_hypre_F90_NAME(fnalu_hypre_schwarzdestroy, FNALU_HYPRE_SCHWARZDESTROY)
extern void nalu_hypre_F90_NAME(fnalu_hypre_schwarzdestroy, FNALU_HYPRE_SCHWARZDESTROY)
(nalu_hypre_F90_Obj *);

#define NALU_HYPRE_SchwarzSetup \
        nalu_hypre_F90_NAME(fnalu_hypre_schwarzsetup, FNALU_HYPRE_SCHWARZSETUP)
extern void nalu_hypre_F90_NAME(fnalu_hypre_schwarzsetup, FNALU_HYPRE_SCHWARZSETUP)
(nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj*);

#define NALU_HYPRE_SchwarzSolve \
        nalu_hypre_F90_NAME(fnalu_hypre_schwarzsolve, FNALU_HYPRE_SCHWARZSOLVE)
extern void nalu_hypre_F90_NAME(fnalu_hypre_schwarzsolve, FNALU_HYPRE_SCHWARZSOLVE)
(nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj*);

#define NALU_HYPRE_SchwarzSetVariant \
        nalu_hypre_F90_NAME(fnalu_hypre_schwarzsetvariant, FNALU_HYPRE_SCHWARZVARIANT)
extern void nalu_hypre_F90_NAME(fnalu_hypre_schwarzsetvariant, FNALU_HYPRE_SCHWARZVARIANT)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_SchwarzSetOverlap \
        nalu_hypre_F90_NAME(fnalu_hypre_schwarzsetoverlap, FNALU_HYPRE_SCHWARZOVERLAP)
extern void nalu_hypre_F90_NAME(fnalu_hypre_schwarzsetoverlap, FNALU_HYPRE_SCHWARZOVERLAP)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_SchwarzSetDomainType \
        nalu_hypre_F90_NAME(fnalu_hypre_schwarzsetdomaintype, FNALU_HYPRE_SVHWARZSETDOMAINTYPE)
extern void nalu_hypre_F90_NAME(fnalu_hypre_schwarzsetdomaintype, FNALU_HYPRE_SVHWARZSETDOMAINTYPE)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_SchwarzSetDomainStructure \
        nalu_hypre_F90_NAME(fnalu_hypre_schwarzsetdomainstructur, FNALU_HYPRE_SCHWARZSETDOMAINSTRUCTUR)
extern void nalu_hypre_F90_NAME(fnalu_hypre_schwarzsetdomainstructur, FNALU_HYPRE_SCHWARZSETDOMAINSTRUCTUR)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_SchwarzSetNumFunctions \
        nalu_hypre_F90_NAME(fnalu_hypre_schwarzsetnumfunctions, FNALU_HYPRE_SCHWARZSETNUMFUNCTIONS)
extern void nalu_hypre_F90_NAME(fnalu_hypre_schwarzsetnumfunctions, FNALU_HYPRE_SCHWARZSETNUMFUNCTIONS)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_SchwarzSetRelaxWeight \
        nalu_hypre_F90_NAME(fnalu_hypre_schwarzsetrelaxweight, FNALU_HYPRE_SCHWARZSETRELAXWEIGHT)
extern void nalu_hypre_F90_NAME(fnalu_hypre_schwarzsetrelaxweight, FNALU_HYPRE_SCHWARZSETRELAXWEIGHT)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Real *);

#define NALU_HYPRE_SchwarzSetDofFunc \
        nalu_hypre_F90_NAME(fnalu_hypre_schwarzsetdoffunc, FNALU_HYPRE_SCHWARZSETDOFFUNC)
extern void nalu_hypre_F90_NAME(fnalu_hypre_schwarzsetdoffunc, FNALU_HYPRE_SCHWARZSETDOFFUNC)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#ifdef __cplusplus
}
#endif
