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
 * Definitions of IJMatrix Fortran interface routines
 *****************************************************************************/

#define NALU_HYPRE_IJMatrixCreate \
        hypre_F90_NAME(fhypre_ijmatrixcreate, FNALU_HYPRE_IJMATRIXCREATE)
extern void hypre_F90_NAME(fhypre_ijmatrixcreate, FNALU_HYPRE_IJMATRIXCREATE)
(hypre_F90_Obj *, NALU_HYPRE_Int *, NALU_HYPRE_Int *, NALU_HYPRE_Int *, NALU_HYPRE_Int *, hypre_F90_Obj*);

#define NALU_HYPRE_IJMatrixDestroy \
        hypre_F90_NAME(fhypre_ijmatrixdestroy, FNALU_HYPRE_IJMATRIXDESTROY)
extern void hypre_F90_NAME(fhypre_ijmatrixdestroy, FNALU_HYPRE_IJMATRIXDESTROY)
(hypre_F90_Obj *);

#define NALU_HYPRE_IJMatrixInitialize \
        hypre_F90_NAME(fhypre_ijmatrixinitialize, FNALU_HYPRE_IJMATRIXINITIALIZE)
extern void hypre_F90_NAME(fhypre_ijmatrixinitialize, FNALU_HYPRE_IJMATRIXINITIALIZE)
(hypre_F90_Obj *);

#define NALU_HYPRE_IJMatrixAssemble \
        hypre_F90_NAME(fhypre_ijmatrixassemble, FNALU_HYPRE_IJMATRIXASSEMBLE)
extern void hypre_F90_NAME(fhypre_ijmatrixassemble, FNALU_HYPRE_IJMATRIXASSEMBLE)
(hypre_F90_Obj *);

#define NALU_HYPRE_IJMatrixSetRowSizes \
        hypre_F90_NAME(fhypre_ijmatrixsetrowsizes, FNALU_HYPRE_IJMATRIXSETROWSIZES)
extern void hypre_F90_NAME(fhypre_ijmatrixsetrowsizes, FNALU_HYPRE_IJMATRIXSETROWSIZES)
(hypre_F90_Obj *, const NALU_HYPRE_Int *);

#define NALU_HYPRE_IJMatrixSetDiagOffdSizes \
        hypre_F90_NAME(fhypre_ijmatrixsetdiagoffdsizes, FNALU_HYPRE_IJMATRIXSETDIAGOFFDSIZES)
extern void hypre_F90_NAME(fhypre_ijmatrixsetdiagoffdsizes, FNALU_HYPRE_IJMATRIXSETDIAGOFFDSIZES)
(hypre_F90_Obj *, const NALU_HYPRE_Int *, const NALU_HYPRE_Int *);

#define NALU_HYPRE_IJMatrixSetValues \
        hypre_F90_NAME(fhypre_ijmatrixsetvalues, FNALU_HYPRE_IJMATRIXSETVALUES)
extern void hypre_F90_NAME(fhypre_ijmatrixsetvalues, FNALU_HYPRE_IJMATRIXSETVALUES)
(hypre_F90_Obj *, NALU_HYPRE_Int *, NALU_HYPRE_Int *, const NALU_HYPRE_Int *, const NALU_HYPRE_Int *,
 const NALU_HYPRE_Real *);

#define NALU_HYPRE_IJMatrixAddToValues \
        hypre_F90_NAME(fhypre_ijmatrixaddtovalues, FNALU_HYPRE_IJMATRIXADDTOVALUES)
extern void hypre_F90_NAME(fhypre_ijmatrixaddtovalues, FNALU_HYPRE_IJMATRIXADDTOVALUES)
(hypre_F90_Obj *, NALU_HYPRE_Int *, NALU_HYPRE_Int *, const NALU_HYPRE_Int *, const NALU_HYPRE_Int *,
 const NALU_HYPRE_Real *);

#define NALU_HYPRE_IJMatrixSetObjectType \
        hypre_F90_NAME(fhypre_ijmatrixsetobjecttype, FNALU_HYPRE_IJMATRIXSETOBJECTTYPE)
extern void hypre_F90_NAME(fhypre_ijmatrixsetobjecttype, FNALU_HYPRE_IJMATRIXSETOBJECTTYPE)
(hypre_F90_Obj *, const NALU_HYPRE_Int *);

#define NALU_HYPRE_IJMatrixGetObjectType \
        hypre_F90_NAME(fhypre_ijmatrixgetobjecttype, FNALU_HYPRE_IJMATRIXGETOBJECTTYPE)
extern void hypre_F90_NAME(fhypre_ijmatrixgetobjecttype, FNALU_HYPRE_IJMATRIXGETOBJECTTYPE)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_IJMatrixGetObject \
        hypre_F90_NAME(fhypre_ijmatrixgetobject, FNALU_HYPRE_IJMATRIXGETOBJECT)
extern void hypre_F90_NAME(fhypre_ijmatrixgetobject, FNALU_HYPRE_IJMATRIXGETOBJECT)
(hypre_F90_Obj *, hypre_F90_Obj *);

#define NALU_HYPRE_IJMatrixRead \
        hypre_F90_NAME(fhypre_ijmatrixread, FNALU_HYPRE_IJMATRIXREAD)
extern void hypre_F90_NAME(fhypre_ijmatrixread, FNALU_HYPRE_IJMATRIXREAD)
(char *, hypre_F90_Obj *, NALU_HYPRE_Int *, hypre_F90_Obj *);

#define NALU_HYPRE_IJMatrixPrint \
        hypre_F90_NAME(fhypre_ijmatrixprint, FNALU_HYPRE_IJMATRIXPRINT)
extern void hypre_F90_NAME(fhypre_ijmatrixprint, FNALU_HYPRE_IJMATRIXPRINT)
(hypre_F90_Obj *, char *);



#define hypre_IJMatrixSetObject \
        hypre_F90_NAME(fhypre_ijmatrixsetobject, FNALU_HYPRE_IJMATRIXSETOBJECT)
extern void hypre_F90_NAME(fhypre_ijmatrixsetobject, FNALU_HYPRE_IJMATRIXSETOBJECT)
(hypre_F90_Obj *, hypre_F90_Obj *);



#define NALU_HYPRE_IJVectorCreate \
        hypre_F90_NAME(fhypre_ijvectorcreate, FNALU_HYPRE_IJVECTORCREATE)
extern void hypre_F90_NAME(fhypre_ijvectorcreate, FNALU_HYPRE_IJVECTORCREATE)
(NALU_HYPRE_Int *, NALU_HYPRE_Int *, NALU_HYPRE_Int *, hypre_F90_Obj *);

#define NALU_HYPRE_IJVectorDestroy \
        hypre_F90_NAME(fhypre_ijvectordestroy, FNALU_HYPRE_IJVECTORDESTROY)
extern void hypre_F90_NAME(fhypre_ijvectordestroy, FNALU_HYPRE_IJVECTORDESTROY)
(hypre_F90_Obj *);

#define NALU_HYPRE_IJVectorInitialize \
        hypre_F90_NAME(fhypre_ijvectorinitialize, FNALU_HYPRE_IJVECTORINITIALIZE)
extern void hypre_F90_NAME(fhypre_ijvectorinitialize, FNALU_HYPRE_IJVECTORINITIALIZE)
(hypre_F90_Obj *);

#define NALU_HYPRE_IJVectorSetValues \
        hypre_F90_NAME(fhypre_ijvectorsetvalues, FNALU_HYPRE_IJVECTORSETVALUES)
extern void hypre_F90_NAME(fhypre_ijvectorsetvalues, FNALU_HYPRE_IJVECTORSETVALUES)
(hypre_F90_Obj *, NALU_HYPRE_Int *, NALU_HYPRE_Int *, NALU_HYPRE_Real *);

#define NALU_HYPRE_IJVectorAddToValues \
        hypre_F90_NAME(fhypre_ijvectoraddtovalues, FNALU_HYPRE_IJVECTORADDTOVALUES)
extern void hypre_F90_NAME(fhypre_ijvectoraddtovalues, FNALU_HYPRE_IJVECTORADDTOVALUES)
(hypre_F90_Obj *, NALU_HYPRE_Int *, NALU_HYPRE_Int *, NALU_HYPRE_Real *);

#define NALU_HYPRE_IJVectorAssemble \
        hypre_F90_NAME(fhypre_ijvectorassemble, FNALU_HYPRE_IJVECTORASSEMBLE)
extern void hypre_F90_NAME(fhypre_ijvectorassemble, FNALU_HYPRE_IJVECTORASSEMBLE)
(hypre_F90_Obj *);

#define NALU_HYPRE_IJVectorGetValues \
        hypre_F90_NAME(fhypre_ijvectorgetvalues, FNALU_HYPRE_IJVECTORGETVALUES)
extern void hypre_F90_NAME(fhypre_ijvectorgetvalues, FNALU_HYPRE_IJVECTORGETVALUES)
(hypre_F90_Obj *, const NALU_HYPRE_Int *, const NALU_HYPRE_Int *, NALU_HYPRE_Real *);

#define NALU_HYPRE_IJVectorSetObjectType \
        hypre_F90_NAME(fhypre_ijvectorsetobjecttype, FNALU_HYPRE_IJVECTORSETOBJECTTYPE)
extern void hypre_F90_NAME(fhypre_ijvectorsetobjecttype, FNALU_HYPRE_IJVECTORSETOBJECTTYPE)
(hypre_F90_Obj *, const NALU_HYPRE_Int *);

#define NALU_HYPRE_IJVectorGetObjectType \
        hypre_F90_NAME(fhypre_ijvectorgetobjecttype, FNALU_HYPRE_IJVECTORGETOBJECTTYPE)
extern void hypre_F90_NAME(fhypre_ijvectorgetobjecttype, FNALU_HYPRE_IJVECTORGETOBJECTTYPE)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_IJVectorGetObject \
        hypre_F90_NAME(fhypre_ijvectorgetobject, FNALU_HYPRE_IJVECTORGETOBJECT)
extern void hypre_F90_NAME(fhypre_ijvectorgetobject, FNALU_HYPRE_IJVECTORGETOBJECT)
(hypre_F90_Obj *, hypre_F90_Obj *);

#define NALU_HYPRE_IJVectorRead \
        hypre_F90_NAME(fhypre_ijvectorread, FNALU_HYPRE_IJVECTORREAD)
extern void hypre_F90_NAME(fhypre_ijvectorread, FNALU_HYPRE_IJVECTORREAD)
(char *, hypre_F90_Obj *, NALU_HYPRE_Int *, hypre_F90_Obj *);

#define NALU_HYPRE_IJVectorPrint \
        hypre_F90_NAME(fhypre_ijvectorprint, FNALU_HYPRE_IJVECTORPRINT)
extern void hypre_F90_NAME(fhypre_ijvectorprint, FNALU_HYPRE_IJVECTORPRINT)
(hypre_F90_Obj *, const char *);

#ifdef __cplusplus
}
#endif
