/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#ifdef __cplusplus
extern "C" {
#endif

/******************************************************************************
 * Definitions of IJMatrix Fortran interface routines
 *****************************************************************************/

#define NALU_HYPRE_IJMatrixCreate \
        nalu_hypre_F90_NAME(fnalu_hypre_ijmatrixcreate, FNALU_HYPRE_IJMATRIXCREATE)
extern void nalu_hypre_F90_NAME(fnalu_hypre_ijmatrixcreate, FNALU_HYPRE_IJMATRIXCREATE)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *, NALU_HYPRE_Int *, NALU_HYPRE_Int *, NALU_HYPRE_Int *, nalu_hypre_F90_Obj*);

#define NALU_HYPRE_IJMatrixDestroy \
        nalu_hypre_F90_NAME(fnalu_hypre_ijmatrixdestroy, FNALU_HYPRE_IJMATRIXDESTROY)
extern void nalu_hypre_F90_NAME(fnalu_hypre_ijmatrixdestroy, FNALU_HYPRE_IJMATRIXDESTROY)
(nalu_hypre_F90_Obj *);

#define NALU_HYPRE_IJMatrixInitialize \
        nalu_hypre_F90_NAME(fnalu_hypre_ijmatrixinitialize, FNALU_HYPRE_IJMATRIXINITIALIZE)
extern void nalu_hypre_F90_NAME(fnalu_hypre_ijmatrixinitialize, FNALU_HYPRE_IJMATRIXINITIALIZE)
(nalu_hypre_F90_Obj *);

#define NALU_HYPRE_IJMatrixAssemble \
        nalu_hypre_F90_NAME(fnalu_hypre_ijmatrixassemble, FNALU_HYPRE_IJMATRIXASSEMBLE)
extern void nalu_hypre_F90_NAME(fnalu_hypre_ijmatrixassemble, FNALU_HYPRE_IJMATRIXASSEMBLE)
(nalu_hypre_F90_Obj *);

#define NALU_HYPRE_IJMatrixSetRowSizes \
        nalu_hypre_F90_NAME(fnalu_hypre_ijmatrixsetrowsizes, FNALU_HYPRE_IJMATRIXSETROWSIZES)
extern void nalu_hypre_F90_NAME(fnalu_hypre_ijmatrixsetrowsizes, FNALU_HYPRE_IJMATRIXSETROWSIZES)
(nalu_hypre_F90_Obj *, const NALU_HYPRE_Int *);

#define NALU_HYPRE_IJMatrixSetDiagOffdSizes \
        nalu_hypre_F90_NAME(fnalu_hypre_ijmatrixsetdiagoffdsizes, FNALU_HYPRE_IJMATRIXSETDIAGOFFDSIZES)
extern void nalu_hypre_F90_NAME(fnalu_hypre_ijmatrixsetdiagoffdsizes, FNALU_HYPRE_IJMATRIXSETDIAGOFFDSIZES)
(nalu_hypre_F90_Obj *, const NALU_HYPRE_Int *, const NALU_HYPRE_Int *);

#define NALU_HYPRE_IJMatrixSetValues \
        nalu_hypre_F90_NAME(fnalu_hypre_ijmatrixsetvalues, FNALU_HYPRE_IJMATRIXSETVALUES)
extern void nalu_hypre_F90_NAME(fnalu_hypre_ijmatrixsetvalues, FNALU_HYPRE_IJMATRIXSETVALUES)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *, NALU_HYPRE_Int *, const NALU_HYPRE_Int *, const NALU_HYPRE_Int *,
 const NALU_HYPRE_Real *);

#define NALU_HYPRE_IJMatrixAddToValues \
        nalu_hypre_F90_NAME(fnalu_hypre_ijmatrixaddtovalues, FNALU_HYPRE_IJMATRIXADDTOVALUES)
extern void nalu_hypre_F90_NAME(fnalu_hypre_ijmatrixaddtovalues, FNALU_HYPRE_IJMATRIXADDTOVALUES)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *, NALU_HYPRE_Int *, const NALU_HYPRE_Int *, const NALU_HYPRE_Int *,
 const NALU_HYPRE_Real *);

#define NALU_HYPRE_IJMatrixSetObjectType \
        nalu_hypre_F90_NAME(fnalu_hypre_ijmatrixsetobjecttype, FNALU_HYPRE_IJMATRIXSETOBJECTTYPE)
extern void nalu_hypre_F90_NAME(fnalu_hypre_ijmatrixsetobjecttype, FNALU_HYPRE_IJMATRIXSETOBJECTTYPE)
(nalu_hypre_F90_Obj *, const NALU_HYPRE_Int *);

#define NALU_HYPRE_IJMatrixGetObjectType \
        nalu_hypre_F90_NAME(fnalu_hypre_ijmatrixgetobjecttype, FNALU_HYPRE_IJMATRIXGETOBJECTTYPE)
extern void nalu_hypre_F90_NAME(fnalu_hypre_ijmatrixgetobjecttype, FNALU_HYPRE_IJMATRIXGETOBJECTTYPE)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_IJMatrixGetObject \
        nalu_hypre_F90_NAME(fnalu_hypre_ijmatrixgetobject, FNALU_HYPRE_IJMATRIXGETOBJECT)
extern void nalu_hypre_F90_NAME(fnalu_hypre_ijmatrixgetobject, FNALU_HYPRE_IJMATRIXGETOBJECT)
(nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *);

#define NALU_HYPRE_IJMatrixRead \
        nalu_hypre_F90_NAME(fnalu_hypre_ijmatrixread, FNALU_HYPRE_IJMATRIXREAD)
extern void nalu_hypre_F90_NAME(fnalu_hypre_ijmatrixread, FNALU_HYPRE_IJMATRIXREAD)
(char *, nalu_hypre_F90_Obj *, NALU_HYPRE_Int *, nalu_hypre_F90_Obj *);

#define NALU_HYPRE_IJMatrixPrint \
        nalu_hypre_F90_NAME(fnalu_hypre_ijmatrixprint, FNALU_HYPRE_IJMATRIXPRINT)
extern void nalu_hypre_F90_NAME(fnalu_hypre_ijmatrixprint, FNALU_HYPRE_IJMATRIXPRINT)
(nalu_hypre_F90_Obj *, char *);



#define nalu_hypre_IJMatrixSetObject \
        nalu_hypre_F90_NAME(fnalu_hypre_ijmatrixsetobject, FNALU_HYPRE_IJMATRIXSETOBJECT)
extern void nalu_hypre_F90_NAME(fnalu_hypre_ijmatrixsetobject, FNALU_HYPRE_IJMATRIXSETOBJECT)
(nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *);



#define NALU_HYPRE_IJVectorCreate \
        nalu_hypre_F90_NAME(fnalu_hypre_ijvectorcreate, FNALU_HYPRE_IJVECTORCREATE)
extern void nalu_hypre_F90_NAME(fnalu_hypre_ijvectorcreate, FNALU_HYPRE_IJVECTORCREATE)
(NALU_HYPRE_Int *, NALU_HYPRE_Int *, NALU_HYPRE_Int *, nalu_hypre_F90_Obj *);

#define NALU_HYPRE_IJVectorDestroy \
        nalu_hypre_F90_NAME(fnalu_hypre_ijvectordestroy, FNALU_HYPRE_IJVECTORDESTROY)
extern void nalu_hypre_F90_NAME(fnalu_hypre_ijvectordestroy, FNALU_HYPRE_IJVECTORDESTROY)
(nalu_hypre_F90_Obj *);

#define NALU_HYPRE_IJVectorInitialize \
        nalu_hypre_F90_NAME(fnalu_hypre_ijvectorinitialize, FNALU_HYPRE_IJVECTORINITIALIZE)
extern void nalu_hypre_F90_NAME(fnalu_hypre_ijvectorinitialize, FNALU_HYPRE_IJVECTORINITIALIZE)
(nalu_hypre_F90_Obj *);

#define NALU_HYPRE_IJVectorSetValues \
        nalu_hypre_F90_NAME(fnalu_hypre_ijvectorsetvalues, FNALU_HYPRE_IJVECTORSETVALUES)
extern void nalu_hypre_F90_NAME(fnalu_hypre_ijvectorsetvalues, FNALU_HYPRE_IJVECTORSETVALUES)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *, NALU_HYPRE_Int *, NALU_HYPRE_Real *);

#define NALU_HYPRE_IJVectorAddToValues \
        nalu_hypre_F90_NAME(fnalu_hypre_ijvectoraddtovalues, FNALU_HYPRE_IJVECTORADDTOVALUES)
extern void nalu_hypre_F90_NAME(fnalu_hypre_ijvectoraddtovalues, FNALU_HYPRE_IJVECTORADDTOVALUES)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *, NALU_HYPRE_Int *, NALU_HYPRE_Real *);

#define NALU_HYPRE_IJVectorAssemble \
        nalu_hypre_F90_NAME(fnalu_hypre_ijvectorassemble, FNALU_HYPRE_IJVECTORASSEMBLE)
extern void nalu_hypre_F90_NAME(fnalu_hypre_ijvectorassemble, FNALU_HYPRE_IJVECTORASSEMBLE)
(nalu_hypre_F90_Obj *);

#define NALU_HYPRE_IJVectorGetValues \
        nalu_hypre_F90_NAME(fnalu_hypre_ijvectorgetvalues, FNALU_HYPRE_IJVECTORGETVALUES)
extern void nalu_hypre_F90_NAME(fnalu_hypre_ijvectorgetvalues, FNALU_HYPRE_IJVECTORGETVALUES)
(nalu_hypre_F90_Obj *, const NALU_HYPRE_Int *, const NALU_HYPRE_Int *, NALU_HYPRE_Real *);

#define NALU_HYPRE_IJVectorSetObjectType \
        nalu_hypre_F90_NAME(fnalu_hypre_ijvectorsetobjecttype, FNALU_HYPRE_IJVECTORSETOBJECTTYPE)
extern void nalu_hypre_F90_NAME(fnalu_hypre_ijvectorsetobjecttype, FNALU_HYPRE_IJVECTORSETOBJECTTYPE)
(nalu_hypre_F90_Obj *, const NALU_HYPRE_Int *);

#define NALU_HYPRE_IJVectorGetObjectType \
        nalu_hypre_F90_NAME(fnalu_hypre_ijvectorgetobjecttype, FNALU_HYPRE_IJVECTORGETOBJECTTYPE)
extern void nalu_hypre_F90_NAME(fnalu_hypre_ijvectorgetobjecttype, FNALU_HYPRE_IJVECTORGETOBJECTTYPE)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_IJVectorGetObject \
        nalu_hypre_F90_NAME(fnalu_hypre_ijvectorgetobject, FNALU_HYPRE_IJVECTORGETOBJECT)
extern void nalu_hypre_F90_NAME(fnalu_hypre_ijvectorgetobject, FNALU_HYPRE_IJVECTORGETOBJECT)
(nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *);

#define NALU_HYPRE_IJVectorRead \
        nalu_hypre_F90_NAME(fnalu_hypre_ijvectorread, FNALU_HYPRE_IJVECTORREAD)
extern void nalu_hypre_F90_NAME(fnalu_hypre_ijvectorread, FNALU_HYPRE_IJVECTORREAD)
(char *, nalu_hypre_F90_Obj *, NALU_HYPRE_Int *, nalu_hypre_F90_Obj *);

#define NALU_HYPRE_IJVectorPrint \
        nalu_hypre_F90_NAME(fnalu_hypre_ijvectorprint, FNALU_HYPRE_IJVECTORPRINT)
extern void nalu_hypre_F90_NAME(fnalu_hypre_ijvectorprint, FNALU_HYPRE_IJVECTORPRINT)
(nalu_hypre_F90_Obj *, const char *);

#ifdef __cplusplus
}
#endif
