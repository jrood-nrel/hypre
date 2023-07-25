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
 *  Definitions of sstruct fortran interface routines
 *****************************************************************************/

#define NALU_HYPRE_SStructGraphCreate \
        nalu_hypre_F90_NAME(fnalu_hypre_sstructgraphcreate, FNALU_HYPRE_SSTRUCTGRAPHCREATE)
extern void nalu_hypre_F90_NAME(fnalu_hypre_sstructgraphcreate, FNALU_HYPRE_SSTRUCTGRAPHCREATE)
(NALU_HYPRE_Int *, nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *);

#define NALU_HYPRE_SStructGraphDestroy \
        nalu_hypre_F90_NAME(fnalu_hypre_sstructgraphdestroy, FNALU_HYPRE_SSTRUCTGRAPHDESTROY)
extern void nalu_hypre_F90_NAME(fnalu_hypre_sstructgraphdestroy, FNALU_HYPRE_SSTRUCTGRAPHDESTROY)
(nalu_hypre_F90_Obj *);

#define NALU_HYPRE_SStructGraphSetStencil \
        nalu_hypre_F90_NAME(fnalu_hypre_sstructgraphsetstencil, FNALU_HYPRE_SSTRUCTGRAPHSETSTENCIL)
extern void nalu_hypre_F90_NAME(fnalu_hypre_sstructgraphsetstencil, FNALU_HYPRE_SSTRUCTGRAPHSETSTENCIL)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *, NALU_HYPRE_Int *, nalu_hypre_F90_Obj *);

#define NALU_HYPRE_SStructGraphAddEntries \
        nalu_hypre_F90_NAME(fnalu_hypre_sstructgraphaddentries, FNALU_HYPRE_SSTRUCTGRAPHADDENTRIES)
extern void nalu_hypre_F90_NAME(fnalu_hypre_sstructgraphaddentries, FNALU_HYPRE_SSTRUCTGRAPHADDENTRIES)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *, NALU_HYPRE_Int *, NALU_HYPRE_Int *, NALU_HYPRE_Int *, NALU_HYPRE_Int *, NALU_HYPRE_Int *);

#define NALU_HYPRE_SStructGraphAssemble \
        nalu_hypre_F90_NAME(fnalu_hypre_sstructgraphassemble, FNALU_HYPRE_SSTRUCTGRAPHASSEMBLE)
extern void nalu_hypre_F90_NAME(fnalu_hypre_sstructgraphassemble, FNALU_HYPRE_SSTRUCTGRAPHASSEMBLE)
(nalu_hypre_F90_Obj *);

#define NALU_HYPRE_SStructGraphSetObjectType \
        nalu_hypre_F90_NAME(fnalu_hypre_sstructgraphsetobjecttyp, FNALU_HYPRE_SSTRUCTGRAPHSETOBJECTTYP)

extern void nalu_hypre_F90_NAME(fnalu_hypre_sstructgraphsetobjecttyp, FNALU_HYPRE_SSTRUCTGRAPHSETOBJECTTYP)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);



#define NALU_HYPRE_SStructGridCreate \
        nalu_hypre_F90_NAME(fnalu_hypre_sstructgridcreate, FNALU_HYPRE_SSTRUCTGRIDCREATE)
extern void nalu_hypre_F90_NAME(fnalu_hypre_sstructgridcreate, FNALU_HYPRE_SSTRUCTGRIDCREATE)
(NALU_HYPRE_Int *, NALU_HYPRE_Int *, NALU_HYPRE_Int *, nalu_hypre_F90_Obj *);

#define NALU_HYPRE_SStructGridDestroy \
        nalu_hypre_F90_NAME(fnalu_hypre_sstructgriddestroy, FNALU_HYPRE_SSTRUCTGRIDDESTROY)
extern void nalu_hypre_F90_NAME(fnalu_hypre_sstructgriddestroy, FNALU_HYPRE_SSTRUCTGRIDDESTROY)
(nalu_hypre_F90_Obj *);

#define NALU_HYPRE_SStructGridSetExtents \
        nalu_hypre_F90_NAME(fnalu_hypre_sstructgridsetextents, FNALU_HYPRE_SSTRUCTGRIDSETEXTENTS)
extern void nalu_hypre_F90_NAME(fnalu_hypre_sstructgridsetextents, FNALU_HYPRE_SSTRUCTGRIDSETEXTENTS)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *, NALU_HYPRE_Int *, NALU_HYPRE_Int *);

#define NALU_HYPRE_SStructGridSetVariables \
        nalu_hypre_F90_NAME(fnalu_hypre_sstructgridsetvariables, FNALU_HYPRE_SSTRUCTGRIDSETVARIABLES)
extern void nalu_hypre_F90_NAME(fnalu_hypre_sstructgridsetvariables, FNALU_HYPRE_SSTRUCTGRIDSETVARIABLES)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *, NALU_HYPRE_Int *, nalu_hypre_F90_Obj *);

#define NALU_HYPRE_SStructGridAddVariables \
        nalu_hypre_F90_NAME(fnalu_hypre_sstructgridaddvariables, FNALU_HYPRE_SSTRUCTGRIDADDVARIABLES)
extern void nalu_hypre_F90_NAME(fnalu_hypre_sstructgridaddvariables, FNALU_HYPRE_SSTRUCTGRIDADDVARIABLES)
(nalu_hypre_F90_Obj  *, NALU_HYPRE_Int *, NALU_HYPRE_Int *, NALU_HYPRE_Int *, nalu_hypre_F90_Obj *);

#define NALU_HYPRE_SStructGridSetNeighborBox \
        nalu_hypre_F90_NAME(fnalu_hypre_sstructgridsetneighborbo, FNALU_HYPRE_SSTRUCTGRIDSETNEIGHBORBO)
extern void nalu_hypre_F90_NAME(fnalu_hypre_sstructgridsetneighborbo, FNALU_HYPRE_SSTRUCTGRIDSETNEIGHBORBO)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *, NALU_HYPRE_Int *, NALU_HYPRE_Int *, NALU_HYPRE_Int *, NALU_HYPRE_Int *, NALU_HYPRE_Int *,
 NALU_HYPRE_Int *);

#define NALU_HYPRE_SStructGridAssemble \
        nalu_hypre_F90_NAME(fnalu_hypre_sstructgridassemble, FNALU_HYPRE_SSTRUCTGRIDASSEMBLE)
extern void nalu_hypre_F90_NAME(fnalu_hypre_sstructgridassemble, FNALU_HYPRE_SSTRUCTGRIDASSEMBLE)
(nalu_hypre_F90_Obj *);

#define NALU_HYPRE_SStructGridSetPeriodic \
        nalu_hypre_F90_NAME(fnalu_hypre_sstructgridsetperiodic, FNALU_HYPRE_SSTRUCTGRIDSETPERIODIC)
extern void nalu_hypre_F90_NAME(fnalu_hypre_sstructgridsetperiodic, FNALU_HYPRE_SSTRUCTGRIDSETPERIODIC)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *, NALU_HYPRE_Int *);

#define NALU_HYPRE_SStructGridSetNumGhost \
        nalu_hypre_F90_NAME(fnalu_hypre_sstructgridsetnumghost, FNALU_HYPRE_SSTRUCTGRIDSETNUMGHOST)
extern void nalu_hypre_F90_NAME(fnalu_hypre_sstructgridsetnumghost, FNALU_HYPRE_SSTRUCTGRIDSETNUMGHOST)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);



#define NALU_HYPRE_SStructMatrixCreate \
        nalu_hypre_F90_NAME(fnalu_hypre_sstructmatrixcreate, FNALU_HYPRE_SSTRUCTMATRIXCREATE)
extern void nalu_hypre_F90_NAME(fnalu_hypre_sstructmatrixcreate, FNALU_HYPRE_SSTRUCTMATRIXCREATE)
(NALU_HYPRE_Int *, nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *);

#define NALU_HYPRE_SStructMatrixDestroy \
        nalu_hypre_F90_NAME(fnalu_hypre_sstructmatrixdestroy, FNALU_HYPRE_SSTRUCTMATRIXDESTROY)
extern void nalu_hypre_F90_NAME(fnalu_hypre_sstructmatrixdestroy, FNALU_HYPRE_SSTRUCTMATRIXDESTROY)
(nalu_hypre_F90_Obj *);

#define NALU_HYPRE_SStructMatrixInitialize \
        nalu_hypre_F90_NAME(fnalu_hypre_sstructmatrixinitialize, FNALU_HYPRE_SSTRUCTMATRIXINITIALIZE)
extern void nalu_hypre_F90_NAME(fnalu_hypre_sstructmatrixinitialize, FNALU_HYPRE_SSTRUCTMATRIXINITIALIZE)
(nalu_hypre_F90_Obj *);

#define NALU_HYPRE_SStructMatrixSetValues \
        nalu_hypre_F90_NAME(fnalu_hypre_sstructmatrixsetvalues, FNALU_HYPRE_SSTRUCTMATRIXSETVALUES)
extern void nalu_hypre_F90_NAME(fnalu_hypre_sstructmatrixsetvalues, FNALU_HYPRE_SSTRUCTMATRIXSETVALUES)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *, NALU_HYPRE_Int *, NALU_HYPRE_Int *, NALU_HYPRE_Int *, NALU_HYPRE_Int *, NALU_HYPRE_Real *);

#define NALU_HYPRE_SStructMatrixSetBoxValues \
        nalu_hypre_F90_NAME(fnalu_hypre_sstructmatrixsetboxvalue, FNALU_HYPRE_SSTRUCTMATRIXSETBOXVALUE)
extern void nalu_hypre_F90_NAME(fnalu_hypre_sstructmatrixsetboxvalue, FNALU_HYPRE_SSTRUCTMATRIXSETBOXVALUE)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *, NALU_HYPRE_Int *, NALU_HYPRE_Int *, NALU_HYPRE_Int *, NALU_HYPRE_Int *, NALU_HYPRE_Int *,
 NALU_HYPRE_Real *);

#define NALU_HYPRE_SStructMatrixGetValues \
        nalu_hypre_F90_NAME(fnalu_hypre_sstructmatrixgetvalues, FNALU_HYPRE_SSTRUCTMATRIXGETVALUES)
extern void nalu_hypre_F90_NAME(fnalu_hypre_sstructmatrixgetvalues, FNALU_HYPRE_SSTRUCTMATRIXGETVALUES)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *, NALU_HYPRE_Int *, NALU_HYPRE_Int *, NALU_HYPRE_Int *, NALU_HYPRE_Int *, NALU_HYPRE_Real *);

#define NALU_HYPRE_SStructMatrixGetBoxValues \
        nalu_hypre_F90_NAME(fnalu_hypre_sstructmatrixgetboxvalue, FNALU_HYPRE_SSTRUCTMATRIXGETBOXVALUE)
extern void nalu_hypre_F90_NAME(fnalu_hypre_sstructmatrixgetboxvalue, FNALU_HYPRE_SSTRUCTMATRIXGETBOXVALUE)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *, NALU_HYPRE_Int *, NALU_HYPRE_Int *, NALU_HYPRE_Int *, NALU_HYPRE_Int *, NALU_HYPRE_Int *,
 NALU_HYPRE_Real *);

#define NALU_HYPRE_SStructMatrixAddToValues \
        nalu_hypre_F90_NAME(fnalu_hypre_sstructmatrixaddtovalues, FNALU_HYPRE_SSTRUCTMATRIXADDTOVALUES)
extern void nalu_hypre_F90_NAME(fnalu_hypre_sstructmatrixaddtovalues, FNALU_HYPRE_SSTRUCTMATRIXADDTOVALUES)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *, NALU_HYPRE_Int *, NALU_HYPRE_Int *, NALU_HYPRE_Int *, NALU_HYPRE_Int *, NALU_HYPRE_Real *);

#define NALU_HYPRE_SStructMatrixAddToBoxValues \
        nalu_hypre_F90_NAME(fnalu_hypre_sstructmatrixaddtoboxval, FNALU_HYPRE_SSTRUCTMATRIXADDTOBOXVAL)
extern void nalu_hypre_F90_NAME(fnalu_hypre_sstructmatrixaddtoboxval, FNALU_HYPRE_SSTRUCTMATRIXADDTOBOXVAL)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *, NALU_HYPRE_Int *, NALU_HYPRE_Int *, NALU_HYPRE_Int *, NALU_HYPRE_Int *, NALU_HYPRE_Int *,
 NALU_HYPRE_Real *);

#define NALU_HYPRE_SStructMatrixAssemble \
        nalu_hypre_F90_NAME(fnalu_hypre_sstructmatrixassemble, FNALU_HYPRE_SSTRUCTMATRIXASSEMBLE)
extern void nalu_hypre_F90_NAME(fnalu_hypre_sstructmatrixassemble, FNALU_HYPRE_SSTRUCTMATRIXASSEMBLE)
(nalu_hypre_F90_Obj *);

#define NALU_HYPRE_SStructMatrixSetSymmetric \
        nalu_hypre_F90_NAME(fnalu_hypre_sstructmatrixsetsymmetri, FNALU_HYPRE_SSTRUCTMATRIXSETSYMMETRI)
extern void nalu_hypre_F90_NAME(fnalu_hypre_sstructmatrixsetsymmetri, FNALU_HYPRE_SSTRUCTMATRIXSETSYMMETRI)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *, NALU_HYPRE_Int *, NALU_HYPRE_Int *, NALU_HYPRE_Int *);

#define NALU_HYPRE_SStructMatrixSetNSSymmetric \
        nalu_hypre_F90_NAME(fnalu_hypre_sstructmatrixsetnssymmet, FNALU_HYPRE_SSTRUCTMATRIXSETNSSYMMET)
extern void nalu_hypre_F90_NAME(fnalu_hypre_sstructmatrixsetnssymmet, FNALU_HYPRE_SSTRUCTMATRIXSETNSSYMMET)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_SStructMatrixSetObjectType \
        nalu_hypre_F90_NAME(fnalu_hypre_sstructmatrixsetobjectty, FNALU_HYPRE_SSTRUCTMATRIXSETOBJECTTY)
extern void nalu_hypre_F90_NAME(fnalu_hypre_sstructmatrixsetobjectty, FNALU_HYPRE_SSTRUCTMATRIXSETOBJECTTY)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_SStructMatrixGetObject \
        nalu_hypre_F90_NAME(fnalu_hypre_sstructmatrixgetobject, FNALU_HYPRE_SSTRUCTMATRIXGETOBJECT)
extern void nalu_hypre_F90_NAME(fnalu_hypre_sstructmatrixgetobject, FNALU_HYPRE_SSTRUCTMATRIXGETOBJECT)
(nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *);

#define NALU_HYPRE_SStructMatrixPrint \
        nalu_hypre_F90_NAME(fnalu_hypre_sstructmatrixprint, FNALU_HYPRE_SSTRUCTMATRIXPRINT)
extern void nalu_hypre_F90_NAME(fnalu_hypre_sstructmatrixprint, FNALU_HYPRE_SSTRUCTMATRIXPRINT)
(const char *, nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);



#define NALU_HYPRE_SStructStencilCreate \
        nalu_hypre_F90_NAME(fnalu_hypre_sstructstencilcreate, FNALU_HYPRE_SSTRUCTSTENCILCREATE)
extern void nalu_hypre_F90_NAME(fnalu_hypre_sstructstencilcreate, FNALU_HYPRE_SSTRUCTSTENCILCREATE)
(NALU_HYPRE_Int *, NALU_HYPRE_Int *, nalu_hypre_F90_Obj *);

#define NALU_HYPRE_SStructStencilDestroy \
        nalu_hypre_F90_NAME(fnalu_hypre_sstructstencildestroy, FNALU_HYPRE_SSTRUCTSTENCILDESTROY)
extern void nalu_hypre_F90_NAME(fnalu_hypre_sstructstencildestroy, FNALU_HYPRE_SSTRUCTSTENCILDESTROY)
(nalu_hypre_F90_Obj *);

#define NALU_HYPRE_SStructStencilSetEntry \
        nalu_hypre_F90_NAME(fnalu_hypre_sstructstencilsetentry, FNALU_HYPRE_SSTRUCTSTENCILSETENTRY)
extern void nalu_hypre_F90_NAME(fnalu_hypre_sstructstencilsetentry, FNALU_HYPRE_SSTRUCTSTENCILSETENTRY)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *, NALU_HYPRE_Int *, NALU_HYPRE_Int *);



#define NALU_HYPRE_SStructVectorCreate \
        nalu_hypre_F90_NAME(fnalu_hypre_sstructvectorcreate, FNALU_HYPRE_SSTRUCTVECTORCREATE)
extern void nalu_hypre_F90_NAME(fnalu_hypre_sstructvectorcreate, FNALU_HYPRE_SSTRUCTVECTORCREATE)
(NALU_HYPRE_Int *, nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *);

#define NALU_HYPRE_SStructVectorDestroy \
        nalu_hypre_F90_NAME(fnalu_hypre_sstructvectordestroy, FNALU_HYPRE_SSTRUCTVECTORDESTROY)
extern void nalu_hypre_F90_NAME(fnalu_hypre_sstructvectordestroy, FNALU_HYPRE_SSTRUCTVECTORDESTROY)
(nalu_hypre_F90_Obj *);

#define NALU_HYPRE_SStructVectorInitialize \
        nalu_hypre_F90_NAME(fnalu_hypre_sstructvectorinitialize, FNALU_HYPRE_SSTRUCTVECTORINITIALIZE)
extern void nalu_hypre_F90_NAME(fnalu_hypre_sstructvectorinitialize, FNALU_HYPRE_SSTRUCTVECTORINITIALIZE)
(nalu_hypre_F90_Obj *);

#define NALU_HYPRE_SStructVectorSetValues \
        nalu_hypre_F90_NAME(fnalu_hypre_sstructvectorsetvalues, FNALU_HYPRE_SSTRUCTVECTORSETVALUES)
extern void nalu_hypre_F90_NAME(fnalu_hypre_sstructvectorsetvalues, FNALU_HYPRE_SSTRUCTVECTORSETVALUES)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *, NALU_HYPRE_Int *, NALU_HYPRE_Int *, NALU_HYPRE_Real *);

#define NALU_HYPRE_SStructVectorSetBoxValues \
        nalu_hypre_F90_NAME(fnalu_hypre_sstructvectorsetboxvalue, FNALU_HYPRE_SSTRUCTVECTORSETBOXVALUE)
extern void nalu_hypre_F90_NAME(fnalu_hypre_sstructvectorsetboxvalue, FNALU_HYPRE_SSTRUCTVECTORSETBOXVALUE)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *, NALU_HYPRE_Int *, NALU_HYPRE_Int *, NALU_HYPRE_Int *, NALU_HYPRE_Real *);

#define NALU_HYPRE_SStructVectorAddToValues \
        nalu_hypre_F90_NAME(fnalu_hypre_sstructvectoraddtovalues, FNALU_HYPRE_SSTRUCTVECTORADDTOVALUES)
extern void nalu_hypre_F90_NAME(fnalu_hypre_sstructvectoraddtovalues, FNALU_HYPRE_SSTRUCTVECTORADDTOVALUES)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *, NALU_HYPRE_Int *, NALU_HYPRE_Int *, NALU_HYPRE_Real *);

#define NALU_HYPRE_SStructVectorAddToBoxValues \
        nalu_hypre_F90_NAME(fnalu_hypre_sstructvectoraddtoboxval, FNALU_HYPRE_SSTRUCTVECTORADDTOBOXVAL)
extern void nalu_hypre_F90_NAME(fnalu_hypre_sstructvectoraddtoboxval, FNALU_HYPRE_SSTRUCTVECTORADDTOBOXVAL)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *, NALU_HYPRE_Int *, NALU_HYPRE_Int *, NALU_HYPRE_Int *, NALU_HYPRE_Real *);

#define NALU_HYPRE_SStructVectorAssemble \
        nalu_hypre_F90_NAME(fnalu_hypre_sstructvectorassemble, FNALU_HYPRE_SSTRUCTVECTORASSEMBLE)
extern void nalu_hypre_F90_NAME(fnalu_hypre_sstructvectorassemble, FNALU_HYPRE_SSTRUCTVECTORASSEMBLE)
(nalu_hypre_F90_Obj *);

#define NALU_HYPRE_SStructVectorGather \
        nalu_hypre_F90_NAME(fnalu_hypre_sstructvectorgather, FNALU_HYPRE_SSTRUCTVECTORGATHER)
extern void nalu_hypre_F90_NAME(fnalu_hypre_sstructvectorgather, FNALU_HYPRE_SSTRUCTVECTORGATHER)
(nalu_hypre_F90_Obj *);

#define NALU_HYPRE_SStructVectorGetValues \
        nalu_hypre_F90_NAME(fnalu_hypre_sstructvectorgetvalues, FNALU_HYPRE_SSTRUCTVECTORGETVALUES)
extern void nalu_hypre_F90_NAME(fnalu_hypre_sstructvectorgetvalues, FNALU_HYPRE_SSTRUCTVECTORGETVALUES)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *, NALU_HYPRE_Int *, NALU_HYPRE_Int *, NALU_HYPRE_Real *);

#define NALU_HYPRE_SStructVectorGetBoxValues \
        nalu_hypre_F90_NAME(fnalu_hypre_sstructvectorgetboxvalue, FNALU_HYPRE_SSTRUCTVECTORGETBOXVALUE)
extern void nalu_hypre_F90_NAME(fnalu_hypre_sstructvectorgetboxvalue, FNALU_HYPRE_SSTRUCTVECTORGETBOXVALUE)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *, NALU_HYPRE_Int *, NALU_HYPRE_Int *, NALU_HYPRE_Int *, NALU_HYPRE_Real *);

#define NALU_HYPRE_SStructVectorSetObjectType \
        nalu_hypre_F90_NAME(fnalu_hypre_sstructvectorsetobjectty, FNALU_HYPRE_SSTRUCTVECTORSETOBJECTTY)
extern void nalu_hypre_F90_NAME(fnalu_hypre_sstructvectorsetobjectty, FNALU_HYPRE_SSTRUCTVECTORSETOBJECTTY)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_SStructVectorGetObject \
        nalu_hypre_F90_NAME(fnalu_hypre_sstructvectorgetobject, FNALU_HYPRE_SSTRUCTVECTORGETOBJECT)
extern void nalu_hypre_F90_NAME(fnalu_hypre_sstructvectorgetobject, FNALU_HYPRE_SSTRUCTVECTORGETOBJECT)
(nalu_hypre_F90_Obj *, void *);

#define NALU_HYPRE_SStructVectorPrint \
        nalu_hypre_F90_NAME(fnalu_hypre_sstructvectorprint, FNALU_HYPRE_SSTRUCTVECTORPRINT)
extern void nalu_hypre_F90_NAME(fnalu_hypre_sstructvectorprint, FNALU_HYPRE_SSTRUCTVECTORPRINT)
(const char *, nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);



#define NALU_HYPRE_SStructBiCGSTABCreate \
        nalu_hypre_F90_NAME(fnalu_hypre_sstructbicgstabcreate, FNALU_HYPRE_SSTRUCTBICGSTABCREATE)
extern void nalu_hypre_F90_NAME(fnalu_hypre_sstructbicgstabcreate, FNALU_HYPRE_SSTRUCTBICGSTABCREATE)
(NALU_HYPRE_Int *, nalu_hypre_F90_Obj *);

#define NALU_HYPRE_SStructBiCGSTABDestroy \
        nalu_hypre_F90_NAME(fnalu_hypre_sstructbicgstabdestroy, FNALU_HYPRE_SSTRUCTBICGSTABDESTROY)
extern void nalu_hypre_F90_NAME(fnalu_hypre_sstructbicgstabdestroy, FNALU_HYPRE_SSTRUCTBICGSTABDESTROY)
(nalu_hypre_F90_Obj *);

#define NALU_HYPRE_SStructBiCGSTABSetup \
        nalu_hypre_F90_NAME(fnalu_hypre_sstructbicgstabsetup, FNALU_HYPRE_SSTRUCTBICGSTABSETUP)
extern void nalu_hypre_F90_NAME(fnalu_hypre_sstructbicgstabsetup, FNALU_HYPRE_SSTRUCTBICGSTABSETUP)
(nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *);

#define NALU_HYPRE_SStructBiCGSTABSolve \
        nalu_hypre_F90_NAME(fnalu_hypre_sstructbicgstabsolve, FNALU_HYPRE_SSTRUCTBICGSTABSOLVE)
extern void nalu_hypre_F90_NAME(fnalu_hypre_sstructbicgstabsolve, FNALU_HYPRE_SSTRUCTBICGSTABSOLVE)
(nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *);

#define NALU_HYPRE_SStructBiCGSTABSetTol \
        nalu_hypre_F90_NAME(fnalu_hypre_sstructbicgstabsettol, FNALU_HYPRE_SSTRUCTBICGSTABSETTOL)
extern void nalu_hypre_F90_NAME(fnalu_hypre_sstructbicgstabsettol, FNALU_HYPRE_SSTRUCTBICGSTABSETTOL)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Real *);

#define NALU_HYPRE_SStructBiCGSTABSetMinIter \
        nalu_hypre_F90_NAME(fnalu_hypre_sstructbicgstabsetminite, FNALU_HYPRE_SSTRUCTBICGSTABSETMINITE)
extern void nalu_hypre_F90_NAME(fnalu_hypre_sstructbicgstabsetminite, FNALU_HYPRE_SSTRUCTBICGSTABSETMINITE)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_SStructBiCGSTABSetMaxIter \
        nalu_hypre_F90_NAME(fnalu_hypre_sstructbicgstabsetmaxite, FNALU_HYPRE_SSTRUCTBICGSTABSETMAXITE)
extern void nalu_hypre_F90_NAME(fnalu_hypre_sstructbicgstabsetmaxite, FNALU_HYPRE_SSTRUCTBICGSTABSETMAXITE)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_SStructBiCGSTABSetStopCrit \
        nalu_hypre_F90_NAME(fnalu_hypre_sstructbicgstabsetstopcr, FNALU_HYPRE_SSTRUCTBICGSTABSETSTOPCR)
extern void nalu_hypre_F90_NAME(fnalu_hypre_sstructbicgstabsetstopcr, FNALU_HYPRE_SSTRUCTBICGSTABSETSTOPCR)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_SStructBiCGSTABSetPrecond \
        nalu_hypre_F90_NAME(fnalu_hypre_sstructbicgstabsetprecon, FNALU_HYPRE_SSTRUCTBICGSTABSETPRECON)
extern void nalu_hypre_F90_NAME(fnalu_hypre_sstructbicgstabsetprecon, FNALU_HYPRE_SSTRUCTBICGSTABSETPRECON)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *, nalu_hypre_F90_Obj *);

#define NALU_HYPRE_SStructBiCGSTABSetLogging \
        nalu_hypre_F90_NAME(fnalu_hypre_sstructbicgstabsetloggin, FNALU_HYPRE_SSTRUCTBICGSTABSETLOGGIN)
extern void nalu_hypre_F90_NAME(fnalu_hypre_sstructbicgstabsetloggin, FNALU_HYPRE_SSTRUCTBICGSTABSETLOGGIN)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_SStructBiCGSTABSetPrintLevel \
        nalu_hypre_F90_NAME(fnalu_hypre_sstructbicgstabsetprintl, FNALU_HYPRE_SSTRUCTBICGSTABSETPRINTL)
extern void nalu_hypre_F90_NAME(fnalu_hypre_sstructbicgstabsetprintl, FNALU_HYPRE_SSTRUCTBICGSTABSETPRINTL)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_SStructBiCGSTABGetNumIterations \
        nalu_hypre_F90_NAME(fnalu_hypre_sstructbicgstabgetnumite, FNALU_HYPRE_SSTRUCTBICGSTABGETNUMITE)
extern void nalu_hypre_F90_NAME(fnalu_hypre_sstructbicgstabgetnumite, FNALU_HYPRE_SSTRUCTBICGSTABGETNUMITE)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_SStructBiCGSTABGetFinalRelativeResidualNorm \
        nalu_hypre_F90_NAME(fnalu_hypre_sstructbicgstabgetfinalr, FNALU_HYPRE_SSTRUCTBICGSTABGETFINALR)
extern void nalu_hypre_F90_NAME(fnalu_hypre_sstructbicgstabgetfinalr, FNALU_HYPRE_SSTRUCTBICGSTABGETFINALR)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Real *);

#define NALU_HYPRE_SStructBiCGSTABGetResidual \
        nalu_hypre_F90_NAME(fnalu_hypre_sstructbicgstabgetresidu, FNALU_HYPRE_SSTRUCTBICGSTABGETRESIDU)
extern void nalu_hypre_F90_NAME(fnalu_hypre_sstructbicgstabgetresidu, FNALU_HYPRE_SSTRUCTBICGSTABGETRESIDU)
(nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *);



#define NALU_HYPRE_SStructGMRESCreate \
        nalu_hypre_F90_NAME(fnalu_hypre_sstructgmrescreate, FNALU_HYPRE_SSTRUCTGMRESCREATE)
extern void nalu_hypre_F90_NAME(fnalu_hypre_sstructgmrescreate, FNALU_HYPRE_SSTRUCTGMRESCREATE)
(nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *);

#define NALU_HYPRE_SStructGMRESDestroy \
        nalu_hypre_F90_NAME(fnalu_hypre_sstructgmresdestroy, FNALU_HYPRE_SSTRUCTGMRESDESTROY)
extern void nalu_hypre_F90_NAME(fnalu_hypre_sstructgmresdestroy, FNALU_HYPRE_SSTRUCTGMRESDESTROY)
(nalu_hypre_F90_Obj *);

#define NALU_HYPRE_SStructGMRESSetup \
        nalu_hypre_F90_NAME(fnalu_hypre_sstructgmressetup, FNALU_HYPRE_SSTRUCTGMRESSETUP)
extern void nalu_hypre_F90_NAME(fnalu_hypre_sstructgmressetup, FNALU_HYPRE_SSTRUCTGMRESSETUP)
(nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *);

#define NALU_HYPRE_SStructGMRESSolve \
        nalu_hypre_F90_NAME(fnalu_hypre_sstructgmressolve, FNALU_HYPRE_SSTRUCTGMRESSOLVE)
extern void nalu_hypre_F90_NAME(fnalu_hypre_sstructgmressolve, FNALU_HYPRE_SSTRUCTGMRESSOLVE)
(nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *);

#define NALU_HYPRE_SStructGMRESSetKDim \
        nalu_hypre_F90_NAME(fnalu_hypre_sstructgmressetkdim, FNALU_HYPRE_SSTRUCTGMRESSETKDIM)
extern void nalu_hypre_F90_NAME(fnalu_hypre_sstructgmressetkdim, FNALU_HYPRE_SSTRUCTGMRESSETKDIM)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_SStructGMRESSetTol \
        nalu_hypre_F90_NAME(fnalu_hypre_sstructgmressettol, FNALU_HYPRE_SSTRUCTGMRESSETTOL)
extern void nalu_hypre_F90_NAME(fnalu_hypre_sstructgmressettol, FNALU_HYPRE_SSTRUCTGMRESSETTOL)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Real *);

#define NALU_HYPRE_SStructGMRESSetMinIter \
        nalu_hypre_F90_NAME(fnalu_hypre_sstructgmressetminiter, FNALU_HYPRE_SSTRUCTGMRESSETMINITER)
extern void nalu_hypre_F90_NAME(fnalu_hypre_sstructgmressetminiter, FNALU_HYPRE_SSTRUCTGMRESSETMINITER)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_SStructGMRESSetMaxIter \
        nalu_hypre_F90_NAME(fnalu_hypre_sstructgmressetmaxiter, FNALU_HYPRE_SSTRUCTGMRESSETMAXITER)
extern void nalu_hypre_F90_NAME(fnalu_hypre_sstructgmressetmaxiter, FNALU_HYPRE_SSTRUCTGMRESSETMAXITER)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_SStructGMRESSetStopCrit \
        nalu_hypre_F90_NAME(fnalu_hypre_sstructgmressetstopcrit, FNALU_HYPRE_SSTRUCTGMRESSETSTOPCRIT)
extern void nalu_hypre_F90_NAME(fnalu_hypre_sstructgmressetstopcrit, FNALU_HYPRE_SSTRUCTGMRESSETSTOPCRIT)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_SStructGMRESSetPrecond \
        nalu_hypre_F90_NAME(fnalu_hypre_sstructgmressetprecond, FNALU_HYPRE_SSTRUCTGMRESSETPRECOND)
extern void nalu_hypre_F90_NAME(fnalu_hypre_sstructgmressetprecond, FNALU_HYPRE_SSTRUCTGMRESSETPRECOND)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *, nalu_hypre_F90_Obj *);


#define NALU_HYPRE_SStructGMRESSetLogging \
        nalu_hypre_F90_NAME(fnalu_hypre_sstructgmressetlogging, FNALU_HYPRE_SSTRUCTGMRESSETLOGGING)
extern void nalu_hypre_F90_NAME(fnalu_hypre_sstructgmressetlogging, FNALU_HYPRE_SSTRUCTGMRESSETLOGGING)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_SStructGMRESSetPrintLevel \
        nalu_hypre_F90_NAME(fnalu_hypre_sstructgmressetprintleve, FNALU_HYPRE_SSTRUCTGMRESSETPRINTLEVE)
extern void nalu_hypre_F90_NAME(fnalu_hypre_sstructgmressetprintleve, FNALU_HYPRE_SSTRUCTGMRESSETPRINTLEVE)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_SStructGMRESGetNumIterations \
      nalu_hypre_F90_NAME(fnalu_hypre_sstructgmresgetnumiterati, FNALU_HYPRE_SSTRUCTGMRESGETNUMITERATI)
extern void nalu_hypre_F90_NAME(fnalu_hypre_sstructgmresgetnumiterati, FNALU_HYPRE_SSTRUCTGMRESGETNUMITERATI)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_SStructGMRESGetFinalRelativeResidualNorm \
        nalu_hypre_F90_NAME(fnalu_hypre_sstructgmresgetfinalrela, FNALU_HYPRE_SSTRUCTGMRESGETFINALRELA)
extern void nalu_hypre_F90_NAME(fnalu_hypre_sstructgmresgetfinalrela, FNALU_HYPRE_SSTRUCTGMRESGETFINALRELA)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Real  *);

#define NALU_HYPRE_SStructGMRESGetResidual \
        nalu_hypre_F90_NAME(fnalu_hypre_sstructgmresgetresidual, FNALU_HYPRE_SSTRUCTGMRESGETRESIDUAL)
extern void nalu_hypre_F90_NAME(fnalu_hypre_sstructgmresgetresidual, FNALU_HYPRE_SSTRUCTGMRESGETRESIDUAL)
(nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *);



#define NALU_HYPRE_SStructPCGCreate \
        nalu_hypre_F90_NAME(fnalu_hypre_sstructpcgcreate, FNALU_HYPRE_SSTRUCTPCGCREATE)
extern void nalu_hypre_F90_NAME(fnalu_hypre_sstructpcgcreate, FNALU_HYPRE_SSTRUCTPCGCREATE)
(nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *);

#define NALU_HYPRE_SStructPCGDestroy \
        nalu_hypre_F90_NAME(fnalu_hypre_sstructpcgdestroy, FNALU_HYPRE_SSTRUCTPCGDESTROY)
extern void nalu_hypre_F90_NAME(fnalu_hypre_sstructpcgdestroy, FNALU_HYPRE_SSTRUCTPCGDESTROY)
(nalu_hypre_F90_Obj *);

#define NALU_HYPRE_SStructPCGSetup \
        nalu_hypre_F90_NAME(fnalu_hypre_sstructpcgsetup, FNALU_HYPRE_SSTRUCTPCGDESTROY)
extern void nalu_hypre_F90_NAME(fnalu_hypre_sstructpcgsetup, FNALU_HYPRE_SSTRUCTPCGDESTROY)
(nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *);

#define NALU_HYPRE_SStructPCGSolve \
        nalu_hypre_F90_NAME(fnalu_hypre_sstructpcgsolve, FNALU_HYPRE_SSTRUCTPCGSOLVE)
extern void nalu_hypre_F90_NAME(fnalu_hypre_sstructpcgsolve, FNALU_HYPRE_SSTRUCTPCGSOLVE)
(nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *);

#define NALU_HYPRE_SStructPCGSetTol \
        nalu_hypre_F90_NAME(fnalu_hypre_sstructpcgsettol, FNALU_HYPRE_SSTRUCTPCGSETTOL)
extern void nalu_hypre_F90_NAME(fnalu_hypre_sstructpcgsettol, FNALU_HYPRE_SSTRUCTPCGSETTOL)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Real *);

#define NALU_HYPRE_SStructPCGSetMaxIter \
        nalu_hypre_F90_NAME(fnalu_hypre_sstructpcgsetmaxiter, FNALU_HYPRE_SSTRUCTPCGSETMAXITER)
extern void nalu_hypre_F90_NAME(fnalu_hypre_sstructpcgsetmaxiter, FNALU_HYPRE_SSTRUCTPCGSETMAXITER)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int  *);

#define NALU_HYPRE_SStructPCGSetTwoNorm \
        nalu_hypre_F90_NAME(fnalu_hypre_sstructpcgsettwonorm, FNALU_HYPRE_SSTRUCTPCGSETTWONORM)
extern void nalu_hypre_F90_NAME(fnalu_hypre_sstructpcgsettwonorm, FNALU_HYPRE_SSTRUCTPCGSETTWONORM)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int  *);

#define NALU_HYPRE_SStructPCGSetRelChange \
        nalu_hypre_F90_NAME(fnalu_hypre_sstructpcgsetrelchange, FNALU_HYPRE_SSTRUCTPCGSETRELCHANGE)
extern void nalu_hypre_F90_NAME(fnalu_hypre_sstructpcgsetrelchange, FNALU_HYPRE_SSTRUCTPCGSETRELCHANGE)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int  *);

#define NALU_HYPRE_SStructPCGSetPrecond \
        nalu_hypre_F90_NAME(fnalu_hypre_sstructpcgsetprecond, FNALU_HYPRE_SSTRUCTPCGSETPRECOND)
extern void nalu_hypre_F90_NAME(fnalu_hypre_sstructpcgsetprecond, FNALU_HYPRE_SSTRUCTPCGSETPRECOND)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int  *, nalu_hypre_F90_Obj *);


#define NALU_HYPRE_SStructPCGSetLogging \
        nalu_hypre_F90_NAME(fnalu_hypre_sstructpcgsetlogging, FNALU_HYPRE_SSTRUCTPCGSETLOGGING)
extern void nalu_hypre_F90_NAME(fnalu_hypre_sstructpcgsetlogging, FNALU_HYPRE_SSTRUCTPCGSETLOGGING)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int  *);

#define NALU_HYPRE_SStructPCGSetPrintLevel \
        nalu_hypre_F90_NAME(fnalu_hypre_sstructpcgsetprintlevel, FNALU_HYPRE_SSTRUCTPCGSETPRINTLEVEL)
extern void nalu_hypre_F90_NAME(fnalu_hypre_sstructpcgsetprintlevel, FNALU_HYPRE_SSTRUCTPCGSETPRINTLEVEL)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int  *);

#define NALU_HYPRE_SStructPCGGetNumIterations \
        nalu_hypre_F90_NAME(fnalu_hypre_sstructpcggetnumiteratio, FNALU_HYPRE_SSTRUCTPCGGETNUMITERATIO)
extern void nalu_hypre_F90_NAME(fnalu_hypre_sstructpcggetnumiteratio, FNALU_HYPRE_SSTRUCTPCGGETNUMITERATIO)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int  *);

#define NALU_HYPRE_SStructPCGGetFinalRelativeResidualNorm \
        nalu_hypre_F90_NAME(fnalu_hypre_sstructpcggetfinalrelati, FNALU_HYPRE_SSTRUCTPCGGETFINALRELATI)
extern void nalu_hypre_F90_NAME(fnalu_hypre_sstructpcggetfinalrelati, FNALU_HYPRE_SSTRUCTPCGGETFINALRELATI)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Real *);

#define NALU_HYPRE_SStructPCGGetResidual \
        nalu_hypre_F90_NAME(fnalu_hypre_sstructpcggetresidual, FNALU_HYPRE_SSTRUCTPCGGETRESIDUAL)
extern void nalu_hypre_F90_NAME(fnalu_hypre_sstructpcggetresidual, FNALU_HYPRE_SSTRUCTPCGGETRESIDUAL)
(nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *);

#define NALU_HYPRE_SStructDiagScaleSetup \
        nalu_hypre_F90_NAME(fnalu_hypre_sstructdiagscalesetup, FNALU_HYPRE_SSTRUCTDIAGSCALESETUP)
extern void nalu_hypre_F90_NAME(fnalu_hypre_sstructdiagscalesetup, FNALU_HYPRE_SSTRUCTDIAGSCALESETUP)
(nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *);

#define NALU_HYPRE_SStructDiagScale \
        nalu_hypre_F90_NAME(fnalu_hypre_sstructdiagscale, FNALU_HYPRE_SSTRUCTDIAGSCALE)
extern void nalu_hypre_F90_NAME(fnalu_hypre_sstructdiagscale, FNALU_HYPRE_SSTRUCTDIAGSCALE)
(nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *);


#define NALU_HYPRE_SStructSplitCreate \
        nalu_hypre_F90_NAME(fnalu_hypre_sstructsplitcreate, FNALU_HYPRE_SSTRUCTSPLITCREATE)
extern void nalu_hypre_F90_NAME(fnalu_hypre_sstructsplitcreate, FNALU_HYPRE_SSTRUCTSPLITCREATE)
(nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *);

#define NALU_HYPRE_SStructSplitDestroy \
        nalu_hypre_F90_NAME(fnalu_hypre_sstructsplitdestroy, FNALU_HYPRE_SSTRUCTSPLITDESTROY)
extern void nalu_hypre_F90_NAME(fnalu_hypre_sstructsplitdestroy, FNALU_HYPRE_SSTRUCTSPLITDESTROY)
(nalu_hypre_F90_Obj *);

#define NALU_HYPRE_SStructSplitSetup \
        nalu_hypre_F90_NAME(fnalu_hypre_sstructsplitsetup, FNALU_HYPRE_SSTRUCTSPLITSETUP)
extern void nalu_hypre_F90_NAME(fnalu_hypre_sstructsplitsetup, FNALU_HYPRE_SSTRUCTSPLITSETUP)
(nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *);

#define NALU_HYPRE_SStructSplitSolve \
        nalu_hypre_F90_NAME(fnalu_hypre_sstructsplitsolve, FNALU_HYPRE_SSTRUCTSPLITSOLVE)
extern void nalu_hypre_F90_NAME(fnalu_hypre_sstructsplitsolve, FNALU_HYPRE_SSTRUCTSPLITSOLVE)
(nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *);

#define NALU_HYPRE_SStructSplitSetTol \
        nalu_hypre_F90_NAME(fnalu_hypre_sstructsplitsettol, FNALU_HYPRE_SSTRUCTSPLITSETTOL)
extern void nalu_hypre_F90_NAME(fnalu_hypre_sstructsplitsettol, FNALU_HYPRE_SSTRUCTSPLITSETTOL)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Real *);

#define NALU_HYPRE_SStructSplitSetMaxIter \
        nalu_hypre_F90_NAME(fnalu_hypre_sstructsplitsetmaxiter, FNALU_HYPRE_SSTRUCTSPLITSETMAXITER)
extern void nalu_hypre_F90_NAME(fnalu_hypre_sstructsplitsetmaxiter, FNALU_HYPRE_SSTRUCTSPLITSETMAXITER)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int  *);

#define NALU_HYPRE_SStructSplitSetZeroGuess \
        nalu_hypre_F90_NAME(fnalu_hypre_sstructsplitsetzeroguess, FNALU_HYPRE_SSTRUCTSPLITSETZEROGUESS)
extern void nalu_hypre_F90_NAME(fnalu_hypre_sstructsplitsetzeroguess, FNALU_HYPRE_SSTRUCTSPLITSETZEROGUESS)
(nalu_hypre_F90_Obj *);

#define NALU_HYPRE_SStructSplitSetNonZeroGuess \
        nalu_hypre_F90_NAME(fnalu_hypre_sstructsplitsetnonzerogu, FNALU_HYPRE_SSTRUCTSPLITSETNONZEROGU)
extern void nalu_hypre_F90_NAME(fnalu_hypre_sstructsplitsetnonzerogu, FNALU_HYPRE_SSTRUCTSPLITSETNONZEROGU)
(nalu_hypre_F90_Obj *);

#define NALU_HYPRE_SStructSplitSetStructSolver \
        nalu_hypre_F90_NAME(fnalu_hypre_sstructsplitsetstructsol, FNALU_HYPRE_SSTRUCTSPLITSETSTRUCTSOL)
extern void nalu_hypre_F90_NAME(fnalu_hypre_sstructsplitsetstructsol, FNALU_HYPRE_SSTRUCTSPLITSETSTRUCTSOL)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int  *);

#define NALU_HYPRE_SStructSplitGetNumIterations \
        nalu_hypre_F90_NAME(fnalu_hypre_sstructsplitgetnumiterat, FNALU_HYPRE_SSTRUCTSPLITGETNUMITERAT)
extern void nalu_hypre_F90_NAME(fnalu_hypre_sstructsplitgetnumiterat, FNALU_HYPRE_SSTRUCTSPLITGETNUMITERAT)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int  *);

#define NALU_HYPRE_SStructSplitGetFinalRelativeResidualNorm \
        nalu_hypre_F90_NAME(fnalu_hypre_sstructsplitgetfinalrela, FNALU_HYPRE_SSTRUCTSPLITGETFINALRELA)
extern void nalu_hypre_F90_NAME(fnalu_hypre_sstructsplitgetfinalrela, FNALU_HYPRE_SSTRUCTSPLITGETFINALRELA)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Real *);



#define NALU_HYPRE_SStructSysPFMGCreate \
        nalu_hypre_F90_NAME(fnalu_hypre_sstructsyspfmgcreate, FNALU_HYPRE_SSTRUCTSYSPFMGCREATE)
extern void nalu_hypre_F90_NAME(fnalu_hypre_sstructsyspfmgcreate, FNALU_HYPRE_SSTRUCTSYSPFMGCREATE)
(nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *);

#define NALU_HYPRE_SStructSysPFMGDestroy \
        nalu_hypre_F90_NAME(fnalu_hypre_sstructsyspfmgdestroy, FNALU_HYPRE_SSTRUCTSYSPFMGDESTROY)
extern void nalu_hypre_F90_NAME(fnalu_hypre_sstructsyspfmgdestroy, FNALU_HYPRE_SSTRUCTSYSPFMGDESTROY)
(nalu_hypre_F90_Obj *);

#define NALU_HYPRE_SStructSysPFMGSetup \
        nalu_hypre_F90_NAME(fnalu_hypre_sstructsyspfmgsetup, FNALU_HYPRE_SSTRUCTSYSPFMGSETUP)
extern void nalu_hypre_F90_NAME(fnalu_hypre_sstructsyspfmgsetup, FNALU_HYPRE_SSTRUCTSYSPFMGSETUP)
(nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *);

#define NALU_HYPRE_SStructSysPFMGSolve \
        nalu_hypre_F90_NAME(fnalu_hypre_sstructsyspfmgsolve, FNALU_HYPRE_SSTRUCTSYSPFMGSOLVE)
extern void nalu_hypre_F90_NAME(fnalu_hypre_sstructsyspfmgsolve, FNALU_HYPRE_SSTRUCTSYSPFMGSOLVE)
(nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *);

#define NALU_HYPRE_SStructSysPFMGSetTol \
        nalu_hypre_F90_NAME(fnalu_hypre_sstructsyspfmgsettol, FNALU_HYPRE_SSTRUCTSYSPFMGSETTOL)
extern void nalu_hypre_F90_NAME(fnalu_hypre_sstructsyspfmgsettol, FNALU_HYPRE_SSTRUCTSYSPFMGSETTOL)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Real *);

#define NALU_HYPRE_SStructSysPFMGSetMaxIter \
        nalu_hypre_F90_NAME(fnalu_hypre_sstructsyspfmgsetmaxiter, FNALU_HYPRE_SSTRUCTSYSPFMGSETMAXITER)
extern void nalu_hypre_F90_NAME(fnalu_hypre_sstructsyspfmgsetmaxiter, FNALU_HYPRE_SSTRUCTSYSPFMGSETMAXITER)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_SStructSysPFMGSetRelChange \
        nalu_hypre_F90_NAME(fnalu_hypre_sstructsyspfmgsetrelchan, FNALU_HYPRE_SSTRUCTSYSPFMGSETRELCHAN)
extern void nalu_hypre_F90_NAME(fnalu_hypre_sstructsyspfmgsetrelchan, FNALU_HYPRE_SSTRUCTSYSPFMGSETRELCHAN)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_SStructSysPFMGSetZeroGuess \
        nalu_hypre_F90_NAME(fnalu_hypre_sstructsyspfmgsetzerogue, FNALU_HYPRE_SSTRUCTSYSPFMGSETZEROGUE)
extern void nalu_hypre_F90_NAME(fnalu_hypre_sstructsyspfmgsetzerogue, FNALU_HYPRE_SSTRUCTSYSPFMGSETZEROGUE)
(nalu_hypre_F90_Obj *);

#define NALU_HYPRE_SStructSysPFMGSetNonZeroGuess \
        nalu_hypre_F90_NAME(fnalu_hypre_sstructsyspfmgsetnonzero, FNALU_HYPRE_SSTRUCTSYSPFMGSETNONZERO)
extern void nalu_hypre_F90_NAME(fnalu_hypre_sstructsyspfmgsetnonzero, FNALU_HYPRE_SSTRUCTSYSPFMGSETNONZERO)
(nalu_hypre_F90_Obj *);

#define NALU_HYPRE_SStructSysPFMGSetRelaxType \
        nalu_hypre_F90_NAME(fnalu_hypre_sstructsyspfmgsetrelaxty, FNALU_HYPRE_SSTRUCTSYSPFMGSETRELAXTY)
extern void nalu_hypre_F90_NAME(fnalu_hypre_sstructsyspfmgsetrelaxty, FNALU_HYPRE_SSTRUCTSYSPFMGSETRELAXTY)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_SStructSysPFMGSetNumPreRelax \
        nalu_hypre_F90_NAME(fnalu_hypre_sstructsyspfmgsetnumprer, FNALU_HYPRE_SSTRUCTSYSPFMGSETNUMPRER)
extern void nalu_hypre_F90_NAME(fnalu_hypre_sstructsyspfmgsetnumprer, FNALU_HYPRE_SSTRUCTSYSPFMGSETNUMPRER)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_SStructSysPFMGSetNumPostRelax \
        nalu_hypre_F90_NAME(fnalu_hypre_sstructsyspfmgsetnumpost, FNALU_HYPRE_SSTRUCTSYSPFMGSETNUMPOST)
extern void nalu_hypre_F90_NAME(fnalu_hypre_sstructsyspfmgsetnumpost, FNALU_HYPRE_SSTRUCTSYSPFMGSETNUMPOST)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);


#define NALU_HYPRE_SStructSysPFMGSetSkipRelax \
        nalu_hypre_F90_NAME(fnalu_hypre_sstructsyspfmgsetskiprel, FNALU_HYPRE_SSTRUCTSYSPFMGSETSKIPREL)
extern void nalu_hypre_F90_NAME(fnalu_hypre_sstructsyspfmgsetskiprel, FNALU_HYPRE_SSTRUCTSYSPFMGSETSKIPREL)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_SStructSysPFMGSetDxyz \
        nalu_hypre_F90_NAME(fnalu_hypre_sstructsyspfmgsetdxyz, FNALU_HYPRE_SSTRUCTSYSPFMGSETDXYZ)
extern void nalu_hypre_F90_NAME(fnalu_hypre_sstructsyspfmgsetdxyz, FNALU_HYPRE_SSTRUCTSYSPFMGSETDXYZ)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Real *);

#define NALU_HYPRE_SStructSysPFMGSetLogging \
        nalu_hypre_F90_NAME(fnalu_hypre_sstructsyspfmgsetlogging, FNALU_HYPRE_SSTRUCTSYSPFMGSETLOGGING)
extern void nalu_hypre_F90_NAME(fnalu_hypre_sstructsyspfmgsetlogging, FNALU_HYPRE_SSTRUCTSYSPFMGSETLOGGING)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_SStructSysPFMGSetPrintLevel \
        nalu_hypre_F90_NAME(fnalu_hypre_sstructsyspfmgsetprintle, FNALU_HYPRE_SSTRUCTSYSPFMGSETPRINTLE)
extern void nalu_hypre_F90_NAME(fnalu_hypre_sstructsyspfmgsetprintle, FNALU_HYPRE_SSTRUCTSYSPFMGSETPRINTLE)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_SStructSysPFMGGetNumIterations \
        nalu_hypre_F90_NAME(fnalu_hypre_sstructsyspfmggetnumiter, FNALU_HYPRE_SSTRUCTSYSPFMGGETNUMITER)
extern void nalu_hypre_F90_NAME(fnalu_hypre_sstructsyspfmggetnumiter, FNALU_HYPRE_SSTRUCTSYSPFMGGETNUMITER)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);


#define NALU_HYPRE_SStructSysPFMGGetFinalRelativeResidualNorm \
        nalu_hypre_F90_NAME(fnalu_hypre_sstructsyspfmggetfinalre, FNALU_HYPRE_SSTRUCTSYSPFMGGETFINALRE)
extern void nalu_hypre_F90_NAME(fnalu_hypre_sstructsyspfmggetfinalre, FNALU_HYPRE_SSTRUCTSYSPFMGGETFINALRE)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Real *);



#define NALU_HYPRE_SStructMaxwellCreate \
        nalu_hypre_F90_NAME(fnalu_hypre_sstructmaxwellcreate, FNALU_HYPRE_SSTRUCTMAXWELLCREATE)
extern void nalu_hypre_F90_NAME(fnalu_hypre_sstructmaxwellcreate, FNALU_HYPRE_SSTRUCTMAXWELLCREATE)
(nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *);

#define NALU_HYPRE_SStructMaxwellDestroy \
        nalu_hypre_F90_NAME(fnalu_hypre_sstructmaxwelldestroy, FNALU_HYPRE_SSTRUCTMAXWELLDESTROY)
extern void nalu_hypre_F90_NAME(fnalu_hypre_sstructmaxwelldestroy, FNALU_HYPRE_SSTRUCTMAXWELLDESTROY)
(nalu_hypre_F90_Obj *);

#define NALU_HYPRE_SStructMaxwellSetup \
        nalu_hypre_F90_NAME(fnalu_hypre_sstructmaxwellsetup, FNALU_HYPRE_SSTRUCTMAXWELLSETUP)
extern void nalu_hypre_F90_NAME(fnalu_hypre_sstructmaxwellsetup, FNALU_HYPRE_SSTRUCTMAXWELLSETUP)
(nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *);

#define NALU_HYPRE_SStructMaxwellSolve \
        nalu_hypre_F90_NAME(fnalu_hypre_sstructmaxwellsolve, FNALU_HYPRE_SSTRUCTMAXWELLSOLVE)
extern void nalu_hypre_F90_NAME(fnalu_hypre_sstructmaxwellsolve, FNALU_HYPRE_SSTRUCTMAXWELLSOLVE)
(nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *);

#define NALU_HYPRE_SStructMaxwellSolve2 \
        nalu_hypre_F90_NAME(fnalu_hypre_sstructmaxwellsolve2, FNALU_HYPRE_SSTRUCTMAXWELLSOLVE2)
extern void nalu_hypre_F90_NAME(fnalu_hypre_sstructmaxwellsolve2, FNALU_HYPRE_SSTRUCTMAXWELLSOLVE2)
(nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *);

#define NALU_HYPRE_MaxwellGrad \
        nalu_hypre_F90_NAME(fnalu_hypre_maxwellgrad, FNALU_HYPRE_MAXWELLGRAD)
extern void nalu_hypre_F90_NAME(fnalu_hypre_maxwellgrad, FNALU_HYPRE_MAXWELLGRAD)
(nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *);

#define NALU_HYPRE_SStructMaxwellSetGrad \
        nalu_hypre_F90_NAME(fnalu_hypre_sstructmaxwellsetgrad, FNALU_HYPRE_SSTRUCTMAXWELLSETGRAD)
extern void nalu_hypre_F90_NAME(fnalu_hypre_sstructmaxwellsetgrad, FNALU_HYPRE_SSTRUCTMAXWELLSETGRAD)
(nalu_hypre_F90_Obj *, nalu_hypre_F90_Obj *);

#define NALU_HYPRE_SStructMaxwellSetRfactors \
        nalu_hypre_F90_NAME(fnalu_hypre_sstructmaxwellsetrfactor, FNALU_HYPRE_SSTRUCTMAXWELLSETRFACTOR)
extern void nalu_hypre_F90_NAME(fnalu_hypre_sstructmaxwellsetrfactor, FNALU_HYPRE_SSTRUCTMAXWELLSETRFACTOR)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_SStructMaxwellSetTol \
        nalu_hypre_F90_NAME(fnalu_hypre_sstructmaxwellsettol, FNALU_HYPRE_SSTRUCTMAXWELLSETTOL)
extern void nalu_hypre_F90_NAME(fnalu_hypre_sstructmaxwellsettol, FNALU_HYPRE_SSTRUCTMAXWELLSETTOL)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Real *);

#define NALU_HYPRE_SStructMaxwellSetConstantCoef \
        nalu_hypre_F90_NAME(fnalu_hypre_sstructmaxwellsetconstan, FNALU_HYPRE_SSTRUCTMAXWELLSETCONSTAN)
extern void nalu_hypre_F90_NAME(fnalu_hypre_sstructmaxwellsetconstan, FNALU_HYPRE_SSTRUCTMAXWELLSETCONSTAN)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_SStructMaxwellSetMaxIter \
        nalu_hypre_F90_NAME(fnalu_hypre_sstructmaxwellsetmaxiter, FNALU_HYPRE_SSTRUCTMAXWELLSETMAXITER)
extern void nalu_hypre_F90_NAME(fnalu_hypre_sstructmaxwellsetmaxiter, FNALU_HYPRE_SSTRUCTMAXWELLSETMAXITER)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_SStructMaxwellSetRelChange \
        nalu_hypre_F90_NAME(fnalu_hypre_sstructmaxwellsetrelchan, FNALU_HYPRE_SSTRUCTMAXWELLSETRELCHAN)
extern void nalu_hypre_F90_NAME(fnalu_hypre_sstructmaxwellsetrelchan, FNALU_HYPRE_SSTRUCTMAXWELLSETRELCHAN)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_SStructMaxwellSetNumPreRelax \
        nalu_hypre_F90_NAME(fnalu_hypre_sstructmaxwellsetnumprer, FNALU_HYPRE_SSTRUCTMAXWELLSETNUMPRER)
extern void nalu_hypre_F90_NAME(fnalu_hypre_sstructmaxwellsetnumprer, FNALU_HYPRE_SSTRUCTMAXWELLSETNUMPRER)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_SStructMaxwellSetNumPostRelax \
        nalu_hypre_F90_NAME(fnalu_hypre_sstructmaxwellsetnumpost, FNALU_HYPRE_SSTRUCTMAXWELLSETNUMPOST)
extern void nalu_hypre_F90_NAME(fnalu_hypre_sstructmaxwellsetnumpost, FNALU_HYPRE_SSTRUCTMAXWELLSETNUMPOST)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_SStructMaxwellSetLogging \
        nalu_hypre_F90_NAME(fnalu_hypre_sstructmaxwellsetlogging, FNALU_HYPRE_SSTRUCTMAXWELLSETLOGGING)
extern void nalu_hypre_F90_NAME(fnalu_hypre_sstructmaxwellsetlogging, FNALU_HYPRE_SSTRUCTMAXWELLSETLOGGING)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_SStructMaxwellSetPrintLevel \
        nalu_hypre_F90_NAME(fnalu_hypre_sstructmaxwellsetprintle, FNALU_HYPRE_SSTRUCTMAXWELLSETPRINTLE)
extern void nalu_hypre_F90_NAME(fnalu_hypre_sstructmaxwellsetprintle, FNALU_HYPRE_SSTRUCTMAXWELLSETPRINTLE)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_SStructMaxwellPrintLogging \
        nalu_hypre_F90_NAME(fnalu_hypre_sstructmaxwellprintloggi, FNALU_HYPRE_SSTRUCTMAXWELLPRINTLOGGI)
extern void nalu_hypre_F90_NAME(fnalu_hypre_sstructmaxwellprintloggi, FNALU_HYPRE_SSTRUCTMAXWELLPRINTLOGGI)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_SStructMaxwellGetNumIterations \
        nalu_hypre_F90_NAME(fnalu_hypre_sstructmaxwellgetnumiter, FNALU_HYPRE_SSTRUCTMAXWELLGETNUMITER)
extern void nalu_hypre_F90_NAME(fnalu_hypre_sstructmaxwellgetnumiter, FNALU_HYPRE_SSTRUCTMAXWELLGETNUMITER)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_SStructMaxwellGetFinalRelativeResidualNorm \
        nalu_hypre_F90_NAME(fnalu_hypre_sstructmaxwellgetfinalre, FNALU_HYPRE_SSTRUCTMAXWELLGETFINALRE)
extern void nalu_hypre_F90_NAME(fnalu_hypre_sstructmaxwellgetfinalre, FNALU_HYPRE_SSTRUCTMAXWELLGETFINALRE)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Real *);

#define NALU_HYPRE_SStructMaxwellPhysBdy \
        nalu_hypre_F90_NAME(fnalu_hypre_sstructmaxwellphysbdy, FNALU_HYPRE_SSTRUCTMAXWELLPHYSBDY)
extern void nalu_hypre_F90_NAME(fnalu_hypre_sstructmaxwellphysbdy, FNALU_HYPRE_SSTRUCTMAXWELLPHYSBDY)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *, NALU_HYPRE_Int *, NALU_HYPRE_Int *, NALU_HYPRE_Int *);

#define NALU_HYPRE_SStructMaxwellEliminateRowsCols \
        nalu_hypre_F90_NAME(fnalu_hypre_sstructmaxwelleliminater, FNALU_HYPRE_SSTRUCTMAXWELLELIMINATER)
extern void nalu_hypre_F90_NAME(fnalu_hypre_sstructmaxwelleliminater, FNALU_HYPRE_SSTRUCTMAXWELLELIMINATER)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *, NALU_HYPRE_Int *);

#define NALU_HYPRE_SStructMaxwellZeroVector \
        nalu_hypre_F90_NAME(fnalu_hypre_sstructmaxwellzerovector, FNALU_HYPRE_SSTRUCTMAXWELLZEROVECTOR)
extern void nalu_hypre_F90_NAME(fnalu_hypre_sstructmaxwellzerovector, FNALU_HYPRE_SSTRUCTMAXWELLZEROVECTOR)
(nalu_hypre_F90_Obj *, NALU_HYPRE_Int *, NALU_HYPRE_Int *);

#ifdef __cplusplus
}
#endif
