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
 *  Definitions of sstruct fortran interface routines
 *****************************************************************************/

#define NALU_HYPRE_SStructGraphCreate \
        hypre_F90_NAME(fhypre_sstructgraphcreate, FNALU_HYPRE_SSTRUCTGRAPHCREATE)
extern void hypre_F90_NAME(fhypre_sstructgraphcreate, FNALU_HYPRE_SSTRUCTGRAPHCREATE)
(NALU_HYPRE_Int *, hypre_F90_Obj *, hypre_F90_Obj *);

#define NALU_HYPRE_SStructGraphDestroy \
        hypre_F90_NAME(fhypre_sstructgraphdestroy, FNALU_HYPRE_SSTRUCTGRAPHDESTROY)
extern void hypre_F90_NAME(fhypre_sstructgraphdestroy, FNALU_HYPRE_SSTRUCTGRAPHDESTROY)
(hypre_F90_Obj *);

#define NALU_HYPRE_SStructGraphSetStencil \
        hypre_F90_NAME(fhypre_sstructgraphsetstencil, FNALU_HYPRE_SSTRUCTGRAPHSETSTENCIL)
extern void hypre_F90_NAME(fhypre_sstructgraphsetstencil, FNALU_HYPRE_SSTRUCTGRAPHSETSTENCIL)
(hypre_F90_Obj *, NALU_HYPRE_Int *, NALU_HYPRE_Int *, hypre_F90_Obj *);

#define NALU_HYPRE_SStructGraphAddEntries \
        hypre_F90_NAME(fhypre_sstructgraphaddentries, FNALU_HYPRE_SSTRUCTGRAPHADDENTRIES)
extern void hypre_F90_NAME(fhypre_sstructgraphaddentries, FNALU_HYPRE_SSTRUCTGRAPHADDENTRIES)
(hypre_F90_Obj *, NALU_HYPRE_Int *, NALU_HYPRE_Int *, NALU_HYPRE_Int *, NALU_HYPRE_Int *, NALU_HYPRE_Int *, NALU_HYPRE_Int *);

#define NALU_HYPRE_SStructGraphAssemble \
        hypre_F90_NAME(fhypre_sstructgraphassemble, FNALU_HYPRE_SSTRUCTGRAPHASSEMBLE)
extern void hypre_F90_NAME(fhypre_sstructgraphassemble, FNALU_HYPRE_SSTRUCTGRAPHASSEMBLE)
(hypre_F90_Obj *);

#define NALU_HYPRE_SStructGraphSetObjectType \
        hypre_F90_NAME(fhypre_sstructgraphsetobjecttyp, FNALU_HYPRE_SSTRUCTGRAPHSETOBJECTTYP)

extern void hypre_F90_NAME(fhypre_sstructgraphsetobjecttyp, FNALU_HYPRE_SSTRUCTGRAPHSETOBJECTTYP)
(hypre_F90_Obj *, NALU_HYPRE_Int *);



#define NALU_HYPRE_SStructGridCreate \
        hypre_F90_NAME(fhypre_sstructgridcreate, FNALU_HYPRE_SSTRUCTGRIDCREATE)
extern void hypre_F90_NAME(fhypre_sstructgridcreate, FNALU_HYPRE_SSTRUCTGRIDCREATE)
(NALU_HYPRE_Int *, NALU_HYPRE_Int *, NALU_HYPRE_Int *, hypre_F90_Obj *);

#define NALU_HYPRE_SStructGridDestroy \
        hypre_F90_NAME(fhypre_sstructgriddestroy, FNALU_HYPRE_SSTRUCTGRIDDESTROY)
extern void hypre_F90_NAME(fhypre_sstructgriddestroy, FNALU_HYPRE_SSTRUCTGRIDDESTROY)
(hypre_F90_Obj *);

#define NALU_HYPRE_SStructGridSetExtents \
        hypre_F90_NAME(fhypre_sstructgridsetextents, FNALU_HYPRE_SSTRUCTGRIDSETEXTENTS)
extern void hypre_F90_NAME(fhypre_sstructgridsetextents, FNALU_HYPRE_SSTRUCTGRIDSETEXTENTS)
(hypre_F90_Obj *, NALU_HYPRE_Int *, NALU_HYPRE_Int *, NALU_HYPRE_Int *);

#define NALU_HYPRE_SStructGridSetVariables \
        hypre_F90_NAME(fhypre_sstructgridsetvariables, FNALU_HYPRE_SSTRUCTGRIDSETVARIABLES)
extern void hypre_F90_NAME(fhypre_sstructgridsetvariables, FNALU_HYPRE_SSTRUCTGRIDSETVARIABLES)
(hypre_F90_Obj *, NALU_HYPRE_Int *, NALU_HYPRE_Int *, hypre_F90_Obj *);

#define NALU_HYPRE_SStructGridAddVariables \
        hypre_F90_NAME(fhypre_sstructgridaddvariables, FNALU_HYPRE_SSTRUCTGRIDADDVARIABLES)
extern void hypre_F90_NAME(fhypre_sstructgridaddvariables, FNALU_HYPRE_SSTRUCTGRIDADDVARIABLES)
(hypre_F90_Obj  *, NALU_HYPRE_Int *, NALU_HYPRE_Int *, NALU_HYPRE_Int *, hypre_F90_Obj *);

#define NALU_HYPRE_SStructGridSetNeighborBox \
        hypre_F90_NAME(fhypre_sstructgridsetneighborbo, FNALU_HYPRE_SSTRUCTGRIDSETNEIGHBORBO)
extern void hypre_F90_NAME(fhypre_sstructgridsetneighborbo, FNALU_HYPRE_SSTRUCTGRIDSETNEIGHBORBO)
(hypre_F90_Obj *, NALU_HYPRE_Int *, NALU_HYPRE_Int *, NALU_HYPRE_Int *, NALU_HYPRE_Int *, NALU_HYPRE_Int *, NALU_HYPRE_Int *,
 NALU_HYPRE_Int *);

#define NALU_HYPRE_SStructGridAssemble \
        hypre_F90_NAME(fhypre_sstructgridassemble, FNALU_HYPRE_SSTRUCTGRIDASSEMBLE)
extern void hypre_F90_NAME(fhypre_sstructgridassemble, FNALU_HYPRE_SSTRUCTGRIDASSEMBLE)
(hypre_F90_Obj *);

#define NALU_HYPRE_SStructGridSetPeriodic \
        hypre_F90_NAME(fhypre_sstructgridsetperiodic, FNALU_HYPRE_SSTRUCTGRIDSETPERIODIC)
extern void hypre_F90_NAME(fhypre_sstructgridsetperiodic, FNALU_HYPRE_SSTRUCTGRIDSETPERIODIC)
(hypre_F90_Obj *, NALU_HYPRE_Int *, NALU_HYPRE_Int *);

#define NALU_HYPRE_SStructGridSetNumGhost \
        hypre_F90_NAME(fhypre_sstructgridsetnumghost, FNALU_HYPRE_SSTRUCTGRIDSETNUMGHOST)
extern void hypre_F90_NAME(fhypre_sstructgridsetnumghost, FNALU_HYPRE_SSTRUCTGRIDSETNUMGHOST)
(hypre_F90_Obj *, NALU_HYPRE_Int *);



#define NALU_HYPRE_SStructMatrixCreate \
        hypre_F90_NAME(fhypre_sstructmatrixcreate, FNALU_HYPRE_SSTRUCTMATRIXCREATE)
extern void hypre_F90_NAME(fhypre_sstructmatrixcreate, FNALU_HYPRE_SSTRUCTMATRIXCREATE)
(NALU_HYPRE_Int *, hypre_F90_Obj *, hypre_F90_Obj *);

#define NALU_HYPRE_SStructMatrixDestroy \
        hypre_F90_NAME(fhypre_sstructmatrixdestroy, FNALU_HYPRE_SSTRUCTMATRIXDESTROY)
extern void hypre_F90_NAME(fhypre_sstructmatrixdestroy, FNALU_HYPRE_SSTRUCTMATRIXDESTROY)
(hypre_F90_Obj *);

#define NALU_HYPRE_SStructMatrixInitialize \
        hypre_F90_NAME(fhypre_sstructmatrixinitialize, FNALU_HYPRE_SSTRUCTMATRIXINITIALIZE)
extern void hypre_F90_NAME(fhypre_sstructmatrixinitialize, FNALU_HYPRE_SSTRUCTMATRIXINITIALIZE)
(hypre_F90_Obj *);

#define NALU_HYPRE_SStructMatrixSetValues \
        hypre_F90_NAME(fhypre_sstructmatrixsetvalues, FNALU_HYPRE_SSTRUCTMATRIXSETVALUES)
extern void hypre_F90_NAME(fhypre_sstructmatrixsetvalues, FNALU_HYPRE_SSTRUCTMATRIXSETVALUES)
(hypre_F90_Obj *, NALU_HYPRE_Int *, NALU_HYPRE_Int *, NALU_HYPRE_Int *, NALU_HYPRE_Int *, NALU_HYPRE_Int *, NALU_HYPRE_Real *);

#define NALU_HYPRE_SStructMatrixSetBoxValues \
        hypre_F90_NAME(fhypre_sstructmatrixsetboxvalue, FNALU_HYPRE_SSTRUCTMATRIXSETBOXVALUE)
extern void hypre_F90_NAME(fhypre_sstructmatrixsetboxvalue, FNALU_HYPRE_SSTRUCTMATRIXSETBOXVALUE)
(hypre_F90_Obj *, NALU_HYPRE_Int *, NALU_HYPRE_Int *, NALU_HYPRE_Int *, NALU_HYPRE_Int *, NALU_HYPRE_Int *, NALU_HYPRE_Int *,
 NALU_HYPRE_Real *);

#define NALU_HYPRE_SStructMatrixGetValues \
        hypre_F90_NAME(fhypre_sstructmatrixgetvalues, FNALU_HYPRE_SSTRUCTMATRIXGETVALUES)
extern void hypre_F90_NAME(fhypre_sstructmatrixgetvalues, FNALU_HYPRE_SSTRUCTMATRIXGETVALUES)
(hypre_F90_Obj *, NALU_HYPRE_Int *, NALU_HYPRE_Int *, NALU_HYPRE_Int *, NALU_HYPRE_Int *, NALU_HYPRE_Int *, NALU_HYPRE_Real *);

#define NALU_HYPRE_SStructMatrixGetBoxValues \
        hypre_F90_NAME(fhypre_sstructmatrixgetboxvalue, FNALU_HYPRE_SSTRUCTMATRIXGETBOXVALUE)
extern void hypre_F90_NAME(fhypre_sstructmatrixgetboxvalue, FNALU_HYPRE_SSTRUCTMATRIXGETBOXVALUE)
(hypre_F90_Obj *, NALU_HYPRE_Int *, NALU_HYPRE_Int *, NALU_HYPRE_Int *, NALU_HYPRE_Int *, NALU_HYPRE_Int *, NALU_HYPRE_Int *,
 NALU_HYPRE_Real *);

#define NALU_HYPRE_SStructMatrixAddToValues \
        hypre_F90_NAME(fhypre_sstructmatrixaddtovalues, FNALU_HYPRE_SSTRUCTMATRIXADDTOVALUES)
extern void hypre_F90_NAME(fhypre_sstructmatrixaddtovalues, FNALU_HYPRE_SSTRUCTMATRIXADDTOVALUES)
(hypre_F90_Obj *, NALU_HYPRE_Int *, NALU_HYPRE_Int *, NALU_HYPRE_Int *, NALU_HYPRE_Int *, NALU_HYPRE_Int *, NALU_HYPRE_Real *);

#define NALU_HYPRE_SStructMatrixAddToBoxValues \
        hypre_F90_NAME(fhypre_sstructmatrixaddtoboxval, FNALU_HYPRE_SSTRUCTMATRIXADDTOBOXVAL)
extern void hypre_F90_NAME(fhypre_sstructmatrixaddtoboxval, FNALU_HYPRE_SSTRUCTMATRIXADDTOBOXVAL)
(hypre_F90_Obj *, NALU_HYPRE_Int *, NALU_HYPRE_Int *, NALU_HYPRE_Int *, NALU_HYPRE_Int *, NALU_HYPRE_Int *, NALU_HYPRE_Int *,
 NALU_HYPRE_Real *);

#define NALU_HYPRE_SStructMatrixAssemble \
        hypre_F90_NAME(fhypre_sstructmatrixassemble, FNALU_HYPRE_SSTRUCTMATRIXASSEMBLE)
extern void hypre_F90_NAME(fhypre_sstructmatrixassemble, FNALU_HYPRE_SSTRUCTMATRIXASSEMBLE)
(hypre_F90_Obj *);

#define NALU_HYPRE_SStructMatrixSetSymmetric \
        hypre_F90_NAME(fhypre_sstructmatrixsetsymmetri, FNALU_HYPRE_SSTRUCTMATRIXSETSYMMETRI)
extern void hypre_F90_NAME(fhypre_sstructmatrixsetsymmetri, FNALU_HYPRE_SSTRUCTMATRIXSETSYMMETRI)
(hypre_F90_Obj *, NALU_HYPRE_Int *, NALU_HYPRE_Int *, NALU_HYPRE_Int *, NALU_HYPRE_Int *);

#define NALU_HYPRE_SStructMatrixSetNSSymmetric \
        hypre_F90_NAME(fhypre_sstructmatrixsetnssymmet, FNALU_HYPRE_SSTRUCTMATRIXSETNSSYMMET)
extern void hypre_F90_NAME(fhypre_sstructmatrixsetnssymmet, FNALU_HYPRE_SSTRUCTMATRIXSETNSSYMMET)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_SStructMatrixSetObjectType \
        hypre_F90_NAME(fhypre_sstructmatrixsetobjectty, FNALU_HYPRE_SSTRUCTMATRIXSETOBJECTTY)
extern void hypre_F90_NAME(fhypre_sstructmatrixsetobjectty, FNALU_HYPRE_SSTRUCTMATRIXSETOBJECTTY)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_SStructMatrixGetObject \
        hypre_F90_NAME(fhypre_sstructmatrixgetobject, FNALU_HYPRE_SSTRUCTMATRIXGETOBJECT)
extern void hypre_F90_NAME(fhypre_sstructmatrixgetobject, FNALU_HYPRE_SSTRUCTMATRIXGETOBJECT)
(hypre_F90_Obj *, hypre_F90_Obj *);

#define NALU_HYPRE_SStructMatrixPrint \
        hypre_F90_NAME(fhypre_sstructmatrixprint, FNALU_HYPRE_SSTRUCTMATRIXPRINT)
extern void hypre_F90_NAME(fhypre_sstructmatrixprint, FNALU_HYPRE_SSTRUCTMATRIXPRINT)
(const char *, hypre_F90_Obj *, NALU_HYPRE_Int *);



#define NALU_HYPRE_SStructStencilCreate \
        hypre_F90_NAME(fhypre_sstructstencilcreate, FNALU_HYPRE_SSTRUCTSTENCILCREATE)
extern void hypre_F90_NAME(fhypre_sstructstencilcreate, FNALU_HYPRE_SSTRUCTSTENCILCREATE)
(NALU_HYPRE_Int *, NALU_HYPRE_Int *, hypre_F90_Obj *);

#define NALU_HYPRE_SStructStencilDestroy \
        hypre_F90_NAME(fhypre_sstructstencildestroy, FNALU_HYPRE_SSTRUCTSTENCILDESTROY)
extern void hypre_F90_NAME(fhypre_sstructstencildestroy, FNALU_HYPRE_SSTRUCTSTENCILDESTROY)
(hypre_F90_Obj *);

#define NALU_HYPRE_SStructStencilSetEntry \
        hypre_F90_NAME(fhypre_sstructstencilsetentry, FNALU_HYPRE_SSTRUCTSTENCILSETENTRY)
extern void hypre_F90_NAME(fhypre_sstructstencilsetentry, FNALU_HYPRE_SSTRUCTSTENCILSETENTRY)
(hypre_F90_Obj *, NALU_HYPRE_Int *, NALU_HYPRE_Int *, NALU_HYPRE_Int *);



#define NALU_HYPRE_SStructVectorCreate \
        hypre_F90_NAME(fhypre_sstructvectorcreate, FNALU_HYPRE_SSTRUCTVECTORCREATE)
extern void hypre_F90_NAME(fhypre_sstructvectorcreate, FNALU_HYPRE_SSTRUCTVECTORCREATE)
(NALU_HYPRE_Int *, hypre_F90_Obj *, hypre_F90_Obj *);

#define NALU_HYPRE_SStructVectorDestroy \
        hypre_F90_NAME(fhypre_sstructvectordestroy, FNALU_HYPRE_SSTRUCTVECTORDESTROY)
extern void hypre_F90_NAME(fhypre_sstructvectordestroy, FNALU_HYPRE_SSTRUCTVECTORDESTROY)
(hypre_F90_Obj *);

#define NALU_HYPRE_SStructVectorInitialize \
        hypre_F90_NAME(fhypre_sstructvectorinitialize, FNALU_HYPRE_SSTRUCTVECTORINITIALIZE)
extern void hypre_F90_NAME(fhypre_sstructvectorinitialize, FNALU_HYPRE_SSTRUCTVECTORINITIALIZE)
(hypre_F90_Obj *);

#define NALU_HYPRE_SStructVectorSetValues \
        hypre_F90_NAME(fhypre_sstructvectorsetvalues, FNALU_HYPRE_SSTRUCTVECTORSETVALUES)
extern void hypre_F90_NAME(fhypre_sstructvectorsetvalues, FNALU_HYPRE_SSTRUCTVECTORSETVALUES)
(hypre_F90_Obj *, NALU_HYPRE_Int *, NALU_HYPRE_Int *, NALU_HYPRE_Int *, NALU_HYPRE_Real *);

#define NALU_HYPRE_SStructVectorSetBoxValues \
        hypre_F90_NAME(fhypre_sstructvectorsetboxvalue, FNALU_HYPRE_SSTRUCTVECTORSETBOXVALUE)
extern void hypre_F90_NAME(fhypre_sstructvectorsetboxvalue, FNALU_HYPRE_SSTRUCTVECTORSETBOXVALUE)
(hypre_F90_Obj *, NALU_HYPRE_Int *, NALU_HYPRE_Int *, NALU_HYPRE_Int *, NALU_HYPRE_Int *, NALU_HYPRE_Real *);

#define NALU_HYPRE_SStructVectorAddToValues \
        hypre_F90_NAME(fhypre_sstructvectoraddtovalues, FNALU_HYPRE_SSTRUCTVECTORADDTOVALUES)
extern void hypre_F90_NAME(fhypre_sstructvectoraddtovalues, FNALU_HYPRE_SSTRUCTVECTORADDTOVALUES)
(hypre_F90_Obj *, NALU_HYPRE_Int *, NALU_HYPRE_Int *, NALU_HYPRE_Int *, NALU_HYPRE_Real *);

#define NALU_HYPRE_SStructVectorAddToBoxValues \
        hypre_F90_NAME(fhypre_sstructvectoraddtoboxval, FNALU_HYPRE_SSTRUCTVECTORADDTOBOXVAL)
extern void hypre_F90_NAME(fhypre_sstructvectoraddtoboxval, FNALU_HYPRE_SSTRUCTVECTORADDTOBOXVAL)
(hypre_F90_Obj *, NALU_HYPRE_Int *, NALU_HYPRE_Int *, NALU_HYPRE_Int *, NALU_HYPRE_Int *, NALU_HYPRE_Real *);

#define NALU_HYPRE_SStructVectorAssemble \
        hypre_F90_NAME(fhypre_sstructvectorassemble, FNALU_HYPRE_SSTRUCTVECTORASSEMBLE)
extern void hypre_F90_NAME(fhypre_sstructvectorassemble, FNALU_HYPRE_SSTRUCTVECTORASSEMBLE)
(hypre_F90_Obj *);

#define NALU_HYPRE_SStructVectorGather \
        hypre_F90_NAME(fhypre_sstructvectorgather, FNALU_HYPRE_SSTRUCTVECTORGATHER)
extern void hypre_F90_NAME(fhypre_sstructvectorgather, FNALU_HYPRE_SSTRUCTVECTORGATHER)
(hypre_F90_Obj *);

#define NALU_HYPRE_SStructVectorGetValues \
        hypre_F90_NAME(fhypre_sstructvectorgetvalues, FNALU_HYPRE_SSTRUCTVECTORGETVALUES)
extern void hypre_F90_NAME(fhypre_sstructvectorgetvalues, FNALU_HYPRE_SSTRUCTVECTORGETVALUES)
(hypre_F90_Obj *, NALU_HYPRE_Int *, NALU_HYPRE_Int *, NALU_HYPRE_Int *, NALU_HYPRE_Real *);

#define NALU_HYPRE_SStructVectorGetBoxValues \
        hypre_F90_NAME(fhypre_sstructvectorgetboxvalue, FNALU_HYPRE_SSTRUCTVECTORGETBOXVALUE)
extern void hypre_F90_NAME(fhypre_sstructvectorgetboxvalue, FNALU_HYPRE_SSTRUCTVECTORGETBOXVALUE)
(hypre_F90_Obj *, NALU_HYPRE_Int *, NALU_HYPRE_Int *, NALU_HYPRE_Int *, NALU_HYPRE_Int *, NALU_HYPRE_Real *);

#define NALU_HYPRE_SStructVectorSetObjectType \
        hypre_F90_NAME(fhypre_sstructvectorsetobjectty, FNALU_HYPRE_SSTRUCTVECTORSETOBJECTTY)
extern void hypre_F90_NAME(fhypre_sstructvectorsetobjectty, FNALU_HYPRE_SSTRUCTVECTORSETOBJECTTY)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_SStructVectorGetObject \
        hypre_F90_NAME(fhypre_sstructvectorgetobject, FNALU_HYPRE_SSTRUCTVECTORGETOBJECT)
extern void hypre_F90_NAME(fhypre_sstructvectorgetobject, FNALU_HYPRE_SSTRUCTVECTORGETOBJECT)
(hypre_F90_Obj *, void *);

#define NALU_HYPRE_SStructVectorPrint \
        hypre_F90_NAME(fhypre_sstructvectorprint, FNALU_HYPRE_SSTRUCTVECTORPRINT)
extern void hypre_F90_NAME(fhypre_sstructvectorprint, FNALU_HYPRE_SSTRUCTVECTORPRINT)
(const char *, hypre_F90_Obj *, NALU_HYPRE_Int *);



#define NALU_HYPRE_SStructBiCGSTABCreate \
        hypre_F90_NAME(fhypre_sstructbicgstabcreate, FNALU_HYPRE_SSTRUCTBICGSTABCREATE)
extern void hypre_F90_NAME(fhypre_sstructbicgstabcreate, FNALU_HYPRE_SSTRUCTBICGSTABCREATE)
(NALU_HYPRE_Int *, hypre_F90_Obj *);

#define NALU_HYPRE_SStructBiCGSTABDestroy \
        hypre_F90_NAME(fhypre_sstructbicgstabdestroy, FNALU_HYPRE_SSTRUCTBICGSTABDESTROY)
extern void hypre_F90_NAME(fhypre_sstructbicgstabdestroy, FNALU_HYPRE_SSTRUCTBICGSTABDESTROY)
(hypre_F90_Obj *);

#define NALU_HYPRE_SStructBiCGSTABSetup \
        hypre_F90_NAME(fhypre_sstructbicgstabsetup, FNALU_HYPRE_SSTRUCTBICGSTABSETUP)
extern void hypre_F90_NAME(fhypre_sstructbicgstabsetup, FNALU_HYPRE_SSTRUCTBICGSTABSETUP)
(hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *);

#define NALU_HYPRE_SStructBiCGSTABSolve \
        hypre_F90_NAME(fhypre_sstructbicgstabsolve, FNALU_HYPRE_SSTRUCTBICGSTABSOLVE)
extern void hypre_F90_NAME(fhypre_sstructbicgstabsolve, FNALU_HYPRE_SSTRUCTBICGSTABSOLVE)
(hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *);

#define NALU_HYPRE_SStructBiCGSTABSetTol \
        hypre_F90_NAME(fhypre_sstructbicgstabsettol, FNALU_HYPRE_SSTRUCTBICGSTABSETTOL)
extern void hypre_F90_NAME(fhypre_sstructbicgstabsettol, FNALU_HYPRE_SSTRUCTBICGSTABSETTOL)
(hypre_F90_Obj *, NALU_HYPRE_Real *);

#define NALU_HYPRE_SStructBiCGSTABSetMinIter \
        hypre_F90_NAME(fhypre_sstructbicgstabsetminite, FNALU_HYPRE_SSTRUCTBICGSTABSETMINITE)
extern void hypre_F90_NAME(fhypre_sstructbicgstabsetminite, FNALU_HYPRE_SSTRUCTBICGSTABSETMINITE)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_SStructBiCGSTABSetMaxIter \
        hypre_F90_NAME(fhypre_sstructbicgstabsetmaxite, FNALU_HYPRE_SSTRUCTBICGSTABSETMAXITE)
extern void hypre_F90_NAME(fhypre_sstructbicgstabsetmaxite, FNALU_HYPRE_SSTRUCTBICGSTABSETMAXITE)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_SStructBiCGSTABSetStopCrit \
        hypre_F90_NAME(fhypre_sstructbicgstabsetstopcr, FNALU_HYPRE_SSTRUCTBICGSTABSETSTOPCR)
extern void hypre_F90_NAME(fhypre_sstructbicgstabsetstopcr, FNALU_HYPRE_SSTRUCTBICGSTABSETSTOPCR)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_SStructBiCGSTABSetPrecond \
        hypre_F90_NAME(fhypre_sstructbicgstabsetprecon, FNALU_HYPRE_SSTRUCTBICGSTABSETPRECON)
extern void hypre_F90_NAME(fhypre_sstructbicgstabsetprecon, FNALU_HYPRE_SSTRUCTBICGSTABSETPRECON)
(hypre_F90_Obj *, NALU_HYPRE_Int *, hypre_F90_Obj *);

#define NALU_HYPRE_SStructBiCGSTABSetLogging \
        hypre_F90_NAME(fhypre_sstructbicgstabsetloggin, FNALU_HYPRE_SSTRUCTBICGSTABSETLOGGIN)
extern void hypre_F90_NAME(fhypre_sstructbicgstabsetloggin, FNALU_HYPRE_SSTRUCTBICGSTABSETLOGGIN)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_SStructBiCGSTABSetPrintLevel \
        hypre_F90_NAME(fhypre_sstructbicgstabsetprintl, FNALU_HYPRE_SSTRUCTBICGSTABSETPRINTL)
extern void hypre_F90_NAME(fhypre_sstructbicgstabsetprintl, FNALU_HYPRE_SSTRUCTBICGSTABSETPRINTL)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_SStructBiCGSTABGetNumIterations \
        hypre_F90_NAME(fhypre_sstructbicgstabgetnumite, FNALU_HYPRE_SSTRUCTBICGSTABGETNUMITE)
extern void hypre_F90_NAME(fhypre_sstructbicgstabgetnumite, FNALU_HYPRE_SSTRUCTBICGSTABGETNUMITE)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_SStructBiCGSTABGetFinalRelativeResidualNorm \
        hypre_F90_NAME(fhypre_sstructbicgstabgetfinalr, FNALU_HYPRE_SSTRUCTBICGSTABGETFINALR)
extern void hypre_F90_NAME(fhypre_sstructbicgstabgetfinalr, FNALU_HYPRE_SSTRUCTBICGSTABGETFINALR)
(hypre_F90_Obj *, NALU_HYPRE_Real *);

#define NALU_HYPRE_SStructBiCGSTABGetResidual \
        hypre_F90_NAME(fhypre_sstructbicgstabgetresidu, FNALU_HYPRE_SSTRUCTBICGSTABGETRESIDU)
extern void hypre_F90_NAME(fhypre_sstructbicgstabgetresidu, FNALU_HYPRE_SSTRUCTBICGSTABGETRESIDU)
(hypre_F90_Obj *, hypre_F90_Obj *);



#define NALU_HYPRE_SStructGMRESCreate \
        hypre_F90_NAME(fhypre_sstructgmrescreate, FNALU_HYPRE_SSTRUCTGMRESCREATE)
extern void hypre_F90_NAME(fhypre_sstructgmrescreate, FNALU_HYPRE_SSTRUCTGMRESCREATE)
(hypre_F90_Obj *, hypre_F90_Obj *);

#define NALU_HYPRE_SStructGMRESDestroy \
        hypre_F90_NAME(fhypre_sstructgmresdestroy, FNALU_HYPRE_SSTRUCTGMRESDESTROY)
extern void hypre_F90_NAME(fhypre_sstructgmresdestroy, FNALU_HYPRE_SSTRUCTGMRESDESTROY)
(hypre_F90_Obj *);

#define NALU_HYPRE_SStructGMRESSetup \
        hypre_F90_NAME(fhypre_sstructgmressetup, FNALU_HYPRE_SSTRUCTGMRESSETUP)
extern void hypre_F90_NAME(fhypre_sstructgmressetup, FNALU_HYPRE_SSTRUCTGMRESSETUP)
(hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *);

#define NALU_HYPRE_SStructGMRESSolve \
        hypre_F90_NAME(fhypre_sstructgmressolve, FNALU_HYPRE_SSTRUCTGMRESSOLVE)
extern void hypre_F90_NAME(fhypre_sstructgmressolve, FNALU_HYPRE_SSTRUCTGMRESSOLVE)
(hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *);

#define NALU_HYPRE_SStructGMRESSetKDim \
        hypre_F90_NAME(fhypre_sstructgmressetkdim, FNALU_HYPRE_SSTRUCTGMRESSETKDIM)
extern void hypre_F90_NAME(fhypre_sstructgmressetkdim, FNALU_HYPRE_SSTRUCTGMRESSETKDIM)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_SStructGMRESSetTol \
        hypre_F90_NAME(fhypre_sstructgmressettol, FNALU_HYPRE_SSTRUCTGMRESSETTOL)
extern void hypre_F90_NAME(fhypre_sstructgmressettol, FNALU_HYPRE_SSTRUCTGMRESSETTOL)
(hypre_F90_Obj *, NALU_HYPRE_Real *);

#define NALU_HYPRE_SStructGMRESSetMinIter \
        hypre_F90_NAME(fhypre_sstructgmressetminiter, FNALU_HYPRE_SSTRUCTGMRESSETMINITER)
extern void hypre_F90_NAME(fhypre_sstructgmressetminiter, FNALU_HYPRE_SSTRUCTGMRESSETMINITER)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_SStructGMRESSetMaxIter \
        hypre_F90_NAME(fhypre_sstructgmressetmaxiter, FNALU_HYPRE_SSTRUCTGMRESSETMAXITER)
extern void hypre_F90_NAME(fhypre_sstructgmressetmaxiter, FNALU_HYPRE_SSTRUCTGMRESSETMAXITER)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_SStructGMRESSetStopCrit \
        hypre_F90_NAME(fhypre_sstructgmressetstopcrit, FNALU_HYPRE_SSTRUCTGMRESSETSTOPCRIT)
extern void hypre_F90_NAME(fhypre_sstructgmressetstopcrit, FNALU_HYPRE_SSTRUCTGMRESSETSTOPCRIT)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_SStructGMRESSetPrecond \
        hypre_F90_NAME(fhypre_sstructgmressetprecond, FNALU_HYPRE_SSTRUCTGMRESSETPRECOND)
extern void hypre_F90_NAME(fhypre_sstructgmressetprecond, FNALU_HYPRE_SSTRUCTGMRESSETPRECOND)
(hypre_F90_Obj *, NALU_HYPRE_Int *, hypre_F90_Obj *);


#define NALU_HYPRE_SStructGMRESSetLogging \
        hypre_F90_NAME(fhypre_sstructgmressetlogging, FNALU_HYPRE_SSTRUCTGMRESSETLOGGING)
extern void hypre_F90_NAME(fhypre_sstructgmressetlogging, FNALU_HYPRE_SSTRUCTGMRESSETLOGGING)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_SStructGMRESSetPrintLevel \
        hypre_F90_NAME(fhypre_sstructgmressetprintleve, FNALU_HYPRE_SSTRUCTGMRESSETPRINTLEVE)
extern void hypre_F90_NAME(fhypre_sstructgmressetprintleve, FNALU_HYPRE_SSTRUCTGMRESSETPRINTLEVE)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_SStructGMRESGetNumIterations \
      hypre_F90_NAME(fhypre_sstructgmresgetnumiterati, FNALU_HYPRE_SSTRUCTGMRESGETNUMITERATI)
extern void hypre_F90_NAME(fhypre_sstructgmresgetnumiterati, FNALU_HYPRE_SSTRUCTGMRESGETNUMITERATI)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_SStructGMRESGetFinalRelativeResidualNorm \
        hypre_F90_NAME(fhypre_sstructgmresgetfinalrela, FNALU_HYPRE_SSTRUCTGMRESGETFINALRELA)
extern void hypre_F90_NAME(fhypre_sstructgmresgetfinalrela, FNALU_HYPRE_SSTRUCTGMRESGETFINALRELA)
(hypre_F90_Obj *, NALU_HYPRE_Real  *);

#define NALU_HYPRE_SStructGMRESGetResidual \
        hypre_F90_NAME(fhypre_sstructgmresgetresidual, FNALU_HYPRE_SSTRUCTGMRESGETRESIDUAL)
extern void hypre_F90_NAME(fhypre_sstructgmresgetresidual, FNALU_HYPRE_SSTRUCTGMRESGETRESIDUAL)
(hypre_F90_Obj *, hypre_F90_Obj *);



#define NALU_HYPRE_SStructPCGCreate \
        hypre_F90_NAME(fhypre_sstructpcgcreate, FNALU_HYPRE_SSTRUCTPCGCREATE)
extern void hypre_F90_NAME(fhypre_sstructpcgcreate, FNALU_HYPRE_SSTRUCTPCGCREATE)
(hypre_F90_Obj *, hypre_F90_Obj *);

#define NALU_HYPRE_SStructPCGDestroy \
        hypre_F90_NAME(fhypre_sstructpcgdestroy, FNALU_HYPRE_SSTRUCTPCGDESTROY)
extern void hypre_F90_NAME(fhypre_sstructpcgdestroy, FNALU_HYPRE_SSTRUCTPCGDESTROY)
(hypre_F90_Obj *);

#define NALU_HYPRE_SStructPCGSetup \
        hypre_F90_NAME(fhypre_sstructpcgsetup, FNALU_HYPRE_SSTRUCTPCGDESTROY)
extern void hypre_F90_NAME(fhypre_sstructpcgsetup, FNALU_HYPRE_SSTRUCTPCGDESTROY)
(hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *);

#define NALU_HYPRE_SStructPCGSolve \
        hypre_F90_NAME(fhypre_sstructpcgsolve, FNALU_HYPRE_SSTRUCTPCGSOLVE)
extern void hypre_F90_NAME(fhypre_sstructpcgsolve, FNALU_HYPRE_SSTRUCTPCGSOLVE)
(hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *);

#define NALU_HYPRE_SStructPCGSetTol \
        hypre_F90_NAME(fhypre_sstructpcgsettol, FNALU_HYPRE_SSTRUCTPCGSETTOL)
extern void hypre_F90_NAME(fhypre_sstructpcgsettol, FNALU_HYPRE_SSTRUCTPCGSETTOL)
(hypre_F90_Obj *, NALU_HYPRE_Real *);

#define NALU_HYPRE_SStructPCGSetMaxIter \
        hypre_F90_NAME(fhypre_sstructpcgsetmaxiter, FNALU_HYPRE_SSTRUCTPCGSETMAXITER)
extern void hypre_F90_NAME(fhypre_sstructpcgsetmaxiter, FNALU_HYPRE_SSTRUCTPCGSETMAXITER)
(hypre_F90_Obj *, NALU_HYPRE_Int  *);

#define NALU_HYPRE_SStructPCGSetTwoNorm \
        hypre_F90_NAME(fhypre_sstructpcgsettwonorm, FNALU_HYPRE_SSTRUCTPCGSETTWONORM)
extern void hypre_F90_NAME(fhypre_sstructpcgsettwonorm, FNALU_HYPRE_SSTRUCTPCGSETTWONORM)
(hypre_F90_Obj *, NALU_HYPRE_Int  *);

#define NALU_HYPRE_SStructPCGSetRelChange \
        hypre_F90_NAME(fhypre_sstructpcgsetrelchange, FNALU_HYPRE_SSTRUCTPCGSETRELCHANGE)
extern void hypre_F90_NAME(fhypre_sstructpcgsetrelchange, FNALU_HYPRE_SSTRUCTPCGSETRELCHANGE)
(hypre_F90_Obj *, NALU_HYPRE_Int  *);

#define NALU_HYPRE_SStructPCGSetPrecond \
        hypre_F90_NAME(fhypre_sstructpcgsetprecond, FNALU_HYPRE_SSTRUCTPCGSETPRECOND)
extern void hypre_F90_NAME(fhypre_sstructpcgsetprecond, FNALU_HYPRE_SSTRUCTPCGSETPRECOND)
(hypre_F90_Obj *, NALU_HYPRE_Int  *, hypre_F90_Obj *);


#define NALU_HYPRE_SStructPCGSetLogging \
        hypre_F90_NAME(fhypre_sstructpcgsetlogging, FNALU_HYPRE_SSTRUCTPCGSETLOGGING)
extern void hypre_F90_NAME(fhypre_sstructpcgsetlogging, FNALU_HYPRE_SSTRUCTPCGSETLOGGING)
(hypre_F90_Obj *, NALU_HYPRE_Int  *);

#define NALU_HYPRE_SStructPCGSetPrintLevel \
        hypre_F90_NAME(fhypre_sstructpcgsetprintlevel, FNALU_HYPRE_SSTRUCTPCGSETPRINTLEVEL)
extern void hypre_F90_NAME(fhypre_sstructpcgsetprintlevel, FNALU_HYPRE_SSTRUCTPCGSETPRINTLEVEL)
(hypre_F90_Obj *, NALU_HYPRE_Int  *);

#define NALU_HYPRE_SStructPCGGetNumIterations \
        hypre_F90_NAME(fhypre_sstructpcggetnumiteratio, FNALU_HYPRE_SSTRUCTPCGGETNUMITERATIO)
extern void hypre_F90_NAME(fhypre_sstructpcggetnumiteratio, FNALU_HYPRE_SSTRUCTPCGGETNUMITERATIO)
(hypre_F90_Obj *, NALU_HYPRE_Int  *);

#define NALU_HYPRE_SStructPCGGetFinalRelativeResidualNorm \
        hypre_F90_NAME(fhypre_sstructpcggetfinalrelati, FNALU_HYPRE_SSTRUCTPCGGETFINALRELATI)
extern void hypre_F90_NAME(fhypre_sstructpcggetfinalrelati, FNALU_HYPRE_SSTRUCTPCGGETFINALRELATI)
(hypre_F90_Obj *, NALU_HYPRE_Real *);

#define NALU_HYPRE_SStructPCGGetResidual \
        hypre_F90_NAME(fhypre_sstructpcggetresidual, FNALU_HYPRE_SSTRUCTPCGGETRESIDUAL)
extern void hypre_F90_NAME(fhypre_sstructpcggetresidual, FNALU_HYPRE_SSTRUCTPCGGETRESIDUAL)
(hypre_F90_Obj *, hypre_F90_Obj *);

#define NALU_HYPRE_SStructDiagScaleSetup \
        hypre_F90_NAME(fhypre_sstructdiagscalesetup, FNALU_HYPRE_SSTRUCTDIAGSCALESETUP)
extern void hypre_F90_NAME(fhypre_sstructdiagscalesetup, FNALU_HYPRE_SSTRUCTDIAGSCALESETUP)
(hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *);

#define NALU_HYPRE_SStructDiagScale \
        hypre_F90_NAME(fhypre_sstructdiagscale, FNALU_HYPRE_SSTRUCTDIAGSCALE)
extern void hypre_F90_NAME(fhypre_sstructdiagscale, FNALU_HYPRE_SSTRUCTDIAGSCALE)
(hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *);


#define NALU_HYPRE_SStructSplitCreate \
        hypre_F90_NAME(fhypre_sstructsplitcreate, FNALU_HYPRE_SSTRUCTSPLITCREATE)
extern void hypre_F90_NAME(fhypre_sstructsplitcreate, FNALU_HYPRE_SSTRUCTSPLITCREATE)
(hypre_F90_Obj *, hypre_F90_Obj *);

#define NALU_HYPRE_SStructSplitDestroy \
        hypre_F90_NAME(fhypre_sstructsplitdestroy, FNALU_HYPRE_SSTRUCTSPLITDESTROY)
extern void hypre_F90_NAME(fhypre_sstructsplitdestroy, FNALU_HYPRE_SSTRUCTSPLITDESTROY)
(hypre_F90_Obj *);

#define NALU_HYPRE_SStructSplitSetup \
        hypre_F90_NAME(fhypre_sstructsplitsetup, FNALU_HYPRE_SSTRUCTSPLITSETUP)
extern void hypre_F90_NAME(fhypre_sstructsplitsetup, FNALU_HYPRE_SSTRUCTSPLITSETUP)
(hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *);

#define NALU_HYPRE_SStructSplitSolve \
        hypre_F90_NAME(fhypre_sstructsplitsolve, FNALU_HYPRE_SSTRUCTSPLITSOLVE)
extern void hypre_F90_NAME(fhypre_sstructsplitsolve, FNALU_HYPRE_SSTRUCTSPLITSOLVE)
(hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *);

#define NALU_HYPRE_SStructSplitSetTol \
        hypre_F90_NAME(fhypre_sstructsplitsettol, FNALU_HYPRE_SSTRUCTSPLITSETTOL)
extern void hypre_F90_NAME(fhypre_sstructsplitsettol, FNALU_HYPRE_SSTRUCTSPLITSETTOL)
(hypre_F90_Obj *, NALU_HYPRE_Real *);

#define NALU_HYPRE_SStructSplitSetMaxIter \
        hypre_F90_NAME(fhypre_sstructsplitsetmaxiter, FNALU_HYPRE_SSTRUCTSPLITSETMAXITER)
extern void hypre_F90_NAME(fhypre_sstructsplitsetmaxiter, FNALU_HYPRE_SSTRUCTSPLITSETMAXITER)
(hypre_F90_Obj *, NALU_HYPRE_Int  *);

#define NALU_HYPRE_SStructSplitSetZeroGuess \
        hypre_F90_NAME(fhypre_sstructsplitsetzeroguess, FNALU_HYPRE_SSTRUCTSPLITSETZEROGUESS)
extern void hypre_F90_NAME(fhypre_sstructsplitsetzeroguess, FNALU_HYPRE_SSTRUCTSPLITSETZEROGUESS)
(hypre_F90_Obj *);

#define NALU_HYPRE_SStructSplitSetNonZeroGuess \
        hypre_F90_NAME(fhypre_sstructsplitsetnonzerogu, FNALU_HYPRE_SSTRUCTSPLITSETNONZEROGU)
extern void hypre_F90_NAME(fhypre_sstructsplitsetnonzerogu, FNALU_HYPRE_SSTRUCTSPLITSETNONZEROGU)
(hypre_F90_Obj *);

#define NALU_HYPRE_SStructSplitSetStructSolver \
        hypre_F90_NAME(fhypre_sstructsplitsetstructsol, FNALU_HYPRE_SSTRUCTSPLITSETSTRUCTSOL)
extern void hypre_F90_NAME(fhypre_sstructsplitsetstructsol, FNALU_HYPRE_SSTRUCTSPLITSETSTRUCTSOL)
(hypre_F90_Obj *, NALU_HYPRE_Int  *);

#define NALU_HYPRE_SStructSplitGetNumIterations \
        hypre_F90_NAME(fhypre_sstructsplitgetnumiterat, FNALU_HYPRE_SSTRUCTSPLITGETNUMITERAT)
extern void hypre_F90_NAME(fhypre_sstructsplitgetnumiterat, FNALU_HYPRE_SSTRUCTSPLITGETNUMITERAT)
(hypre_F90_Obj *, NALU_HYPRE_Int  *);

#define NALU_HYPRE_SStructSplitGetFinalRelativeResidualNorm \
        hypre_F90_NAME(fhypre_sstructsplitgetfinalrela, FNALU_HYPRE_SSTRUCTSPLITGETFINALRELA)
extern void hypre_F90_NAME(fhypre_sstructsplitgetfinalrela, FNALU_HYPRE_SSTRUCTSPLITGETFINALRELA)
(hypre_F90_Obj *, NALU_HYPRE_Real *);



#define NALU_HYPRE_SStructSysPFMGCreate \
        hypre_F90_NAME(fhypre_sstructsyspfmgcreate, FNALU_HYPRE_SSTRUCTSYSPFMGCREATE)
extern void hypre_F90_NAME(fhypre_sstructsyspfmgcreate, FNALU_HYPRE_SSTRUCTSYSPFMGCREATE)
(hypre_F90_Obj *, hypre_F90_Obj *);

#define NALU_HYPRE_SStructSysPFMGDestroy \
        hypre_F90_NAME(fhypre_sstructsyspfmgdestroy, FNALU_HYPRE_SSTRUCTSYSPFMGDESTROY)
extern void hypre_F90_NAME(fhypre_sstructsyspfmgdestroy, FNALU_HYPRE_SSTRUCTSYSPFMGDESTROY)
(hypre_F90_Obj *);

#define NALU_HYPRE_SStructSysPFMGSetup \
        hypre_F90_NAME(fhypre_sstructsyspfmgsetup, FNALU_HYPRE_SSTRUCTSYSPFMGSETUP)
extern void hypre_F90_NAME(fhypre_sstructsyspfmgsetup, FNALU_HYPRE_SSTRUCTSYSPFMGSETUP)
(hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *);

#define NALU_HYPRE_SStructSysPFMGSolve \
        hypre_F90_NAME(fhypre_sstructsyspfmgsolve, FNALU_HYPRE_SSTRUCTSYSPFMGSOLVE)
extern void hypre_F90_NAME(fhypre_sstructsyspfmgsolve, FNALU_HYPRE_SSTRUCTSYSPFMGSOLVE)
(hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *);

#define NALU_HYPRE_SStructSysPFMGSetTol \
        hypre_F90_NAME(fhypre_sstructsyspfmgsettol, FNALU_HYPRE_SSTRUCTSYSPFMGSETTOL)
extern void hypre_F90_NAME(fhypre_sstructsyspfmgsettol, FNALU_HYPRE_SSTRUCTSYSPFMGSETTOL)
(hypre_F90_Obj *, NALU_HYPRE_Real *);

#define NALU_HYPRE_SStructSysPFMGSetMaxIter \
        hypre_F90_NAME(fhypre_sstructsyspfmgsetmaxiter, FNALU_HYPRE_SSTRUCTSYSPFMGSETMAXITER)
extern void hypre_F90_NAME(fhypre_sstructsyspfmgsetmaxiter, FNALU_HYPRE_SSTRUCTSYSPFMGSETMAXITER)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_SStructSysPFMGSetRelChange \
        hypre_F90_NAME(fhypre_sstructsyspfmgsetrelchan, FNALU_HYPRE_SSTRUCTSYSPFMGSETRELCHAN)
extern void hypre_F90_NAME(fhypre_sstructsyspfmgsetrelchan, FNALU_HYPRE_SSTRUCTSYSPFMGSETRELCHAN)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_SStructSysPFMGSetZeroGuess \
        hypre_F90_NAME(fhypre_sstructsyspfmgsetzerogue, FNALU_HYPRE_SSTRUCTSYSPFMGSETZEROGUE)
extern void hypre_F90_NAME(fhypre_sstructsyspfmgsetzerogue, FNALU_HYPRE_SSTRUCTSYSPFMGSETZEROGUE)
(hypre_F90_Obj *);

#define NALU_HYPRE_SStructSysPFMGSetNonZeroGuess \
        hypre_F90_NAME(fhypre_sstructsyspfmgsetnonzero, FNALU_HYPRE_SSTRUCTSYSPFMGSETNONZERO)
extern void hypre_F90_NAME(fhypre_sstructsyspfmgsetnonzero, FNALU_HYPRE_SSTRUCTSYSPFMGSETNONZERO)
(hypre_F90_Obj *);

#define NALU_HYPRE_SStructSysPFMGSetRelaxType \
        hypre_F90_NAME(fhypre_sstructsyspfmgsetrelaxty, FNALU_HYPRE_SSTRUCTSYSPFMGSETRELAXTY)
extern void hypre_F90_NAME(fhypre_sstructsyspfmgsetrelaxty, FNALU_HYPRE_SSTRUCTSYSPFMGSETRELAXTY)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_SStructSysPFMGSetNumPreRelax \
        hypre_F90_NAME(fhypre_sstructsyspfmgsetnumprer, FNALU_HYPRE_SSTRUCTSYSPFMGSETNUMPRER)
extern void hypre_F90_NAME(fhypre_sstructsyspfmgsetnumprer, FNALU_HYPRE_SSTRUCTSYSPFMGSETNUMPRER)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_SStructSysPFMGSetNumPostRelax \
        hypre_F90_NAME(fhypre_sstructsyspfmgsetnumpost, FNALU_HYPRE_SSTRUCTSYSPFMGSETNUMPOST)
extern void hypre_F90_NAME(fhypre_sstructsyspfmgsetnumpost, FNALU_HYPRE_SSTRUCTSYSPFMGSETNUMPOST)
(hypre_F90_Obj *, NALU_HYPRE_Int *);


#define NALU_HYPRE_SStructSysPFMGSetSkipRelax \
        hypre_F90_NAME(fhypre_sstructsyspfmgsetskiprel, FNALU_HYPRE_SSTRUCTSYSPFMGSETSKIPREL)
extern void hypre_F90_NAME(fhypre_sstructsyspfmgsetskiprel, FNALU_HYPRE_SSTRUCTSYSPFMGSETSKIPREL)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_SStructSysPFMGSetDxyz \
        hypre_F90_NAME(fhypre_sstructsyspfmgsetdxyz, FNALU_HYPRE_SSTRUCTSYSPFMGSETDXYZ)
extern void hypre_F90_NAME(fhypre_sstructsyspfmgsetdxyz, FNALU_HYPRE_SSTRUCTSYSPFMGSETDXYZ)
(hypre_F90_Obj *, NALU_HYPRE_Real *);

#define NALU_HYPRE_SStructSysPFMGSetLogging \
        hypre_F90_NAME(fhypre_sstructsyspfmgsetlogging, FNALU_HYPRE_SSTRUCTSYSPFMGSETLOGGING)
extern void hypre_F90_NAME(fhypre_sstructsyspfmgsetlogging, FNALU_HYPRE_SSTRUCTSYSPFMGSETLOGGING)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_SStructSysPFMGSetPrintLevel \
        hypre_F90_NAME(fhypre_sstructsyspfmgsetprintle, FNALU_HYPRE_SSTRUCTSYSPFMGSETPRINTLE)
extern void hypre_F90_NAME(fhypre_sstructsyspfmgsetprintle, FNALU_HYPRE_SSTRUCTSYSPFMGSETPRINTLE)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_SStructSysPFMGGetNumIterations \
        hypre_F90_NAME(fhypre_sstructsyspfmggetnumiter, FNALU_HYPRE_SSTRUCTSYSPFMGGETNUMITER)
extern void hypre_F90_NAME(fhypre_sstructsyspfmggetnumiter, FNALU_HYPRE_SSTRUCTSYSPFMGGETNUMITER)
(hypre_F90_Obj *, NALU_HYPRE_Int *);


#define NALU_HYPRE_SStructSysPFMGGetFinalRelativeResidualNorm \
        hypre_F90_NAME(fhypre_sstructsyspfmggetfinalre, FNALU_HYPRE_SSTRUCTSYSPFMGGETFINALRE)
extern void hypre_F90_NAME(fhypre_sstructsyspfmggetfinalre, FNALU_HYPRE_SSTRUCTSYSPFMGGETFINALRE)
(hypre_F90_Obj *, NALU_HYPRE_Real *);



#define NALU_HYPRE_SStructMaxwellCreate \
        hypre_F90_NAME(fhypre_sstructmaxwellcreate, FNALU_HYPRE_SSTRUCTMAXWELLCREATE)
extern void hypre_F90_NAME(fhypre_sstructmaxwellcreate, FNALU_HYPRE_SSTRUCTMAXWELLCREATE)
(hypre_F90_Obj *, hypre_F90_Obj *);

#define NALU_HYPRE_SStructMaxwellDestroy \
        hypre_F90_NAME(fhypre_sstructmaxwelldestroy, FNALU_HYPRE_SSTRUCTMAXWELLDESTROY)
extern void hypre_F90_NAME(fhypre_sstructmaxwelldestroy, FNALU_HYPRE_SSTRUCTMAXWELLDESTROY)
(hypre_F90_Obj *);

#define NALU_HYPRE_SStructMaxwellSetup \
        hypre_F90_NAME(fhypre_sstructmaxwellsetup, FNALU_HYPRE_SSTRUCTMAXWELLSETUP)
extern void hypre_F90_NAME(fhypre_sstructmaxwellsetup, FNALU_HYPRE_SSTRUCTMAXWELLSETUP)
(hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *);

#define NALU_HYPRE_SStructMaxwellSolve \
        hypre_F90_NAME(fhypre_sstructmaxwellsolve, FNALU_HYPRE_SSTRUCTMAXWELLSOLVE)
extern void hypre_F90_NAME(fhypre_sstructmaxwellsolve, FNALU_HYPRE_SSTRUCTMAXWELLSOLVE)
(hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *);

#define NALU_HYPRE_SStructMaxwellSolve2 \
        hypre_F90_NAME(fhypre_sstructmaxwellsolve2, FNALU_HYPRE_SSTRUCTMAXWELLSOLVE2)
extern void hypre_F90_NAME(fhypre_sstructmaxwellsolve2, FNALU_HYPRE_SSTRUCTMAXWELLSOLVE2)
(hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *, hypre_F90_Obj *);

#define NALU_HYPRE_MaxwellGrad \
        hypre_F90_NAME(fhypre_maxwellgrad, FNALU_HYPRE_MAXWELLGRAD)
extern void hypre_F90_NAME(fhypre_maxwellgrad, FNALU_HYPRE_MAXWELLGRAD)
(hypre_F90_Obj *, hypre_F90_Obj *);

#define NALU_HYPRE_SStructMaxwellSetGrad \
        hypre_F90_NAME(fhypre_sstructmaxwellsetgrad, FNALU_HYPRE_SSTRUCTMAXWELLSETGRAD)
extern void hypre_F90_NAME(fhypre_sstructmaxwellsetgrad, FNALU_HYPRE_SSTRUCTMAXWELLSETGRAD)
(hypre_F90_Obj *, hypre_F90_Obj *);

#define NALU_HYPRE_SStructMaxwellSetRfactors \
        hypre_F90_NAME(fhypre_sstructmaxwellsetrfactor, FNALU_HYPRE_SSTRUCTMAXWELLSETRFACTOR)
extern void hypre_F90_NAME(fhypre_sstructmaxwellsetrfactor, FNALU_HYPRE_SSTRUCTMAXWELLSETRFACTOR)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_SStructMaxwellSetTol \
        hypre_F90_NAME(fhypre_sstructmaxwellsettol, FNALU_HYPRE_SSTRUCTMAXWELLSETTOL)
extern void hypre_F90_NAME(fhypre_sstructmaxwellsettol, FNALU_HYPRE_SSTRUCTMAXWELLSETTOL)
(hypre_F90_Obj *, NALU_HYPRE_Real *);

#define NALU_HYPRE_SStructMaxwellSetConstantCoef \
        hypre_F90_NAME(fhypre_sstructmaxwellsetconstan, FNALU_HYPRE_SSTRUCTMAXWELLSETCONSTAN)
extern void hypre_F90_NAME(fhypre_sstructmaxwellsetconstan, FNALU_HYPRE_SSTRUCTMAXWELLSETCONSTAN)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_SStructMaxwellSetMaxIter \
        hypre_F90_NAME(fhypre_sstructmaxwellsetmaxiter, FNALU_HYPRE_SSTRUCTMAXWELLSETMAXITER)
extern void hypre_F90_NAME(fhypre_sstructmaxwellsetmaxiter, FNALU_HYPRE_SSTRUCTMAXWELLSETMAXITER)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_SStructMaxwellSetRelChange \
        hypre_F90_NAME(fhypre_sstructmaxwellsetrelchan, FNALU_HYPRE_SSTRUCTMAXWELLSETRELCHAN)
extern void hypre_F90_NAME(fhypre_sstructmaxwellsetrelchan, FNALU_HYPRE_SSTRUCTMAXWELLSETRELCHAN)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_SStructMaxwellSetNumPreRelax \
        hypre_F90_NAME(fhypre_sstructmaxwellsetnumprer, FNALU_HYPRE_SSTRUCTMAXWELLSETNUMPRER)
extern void hypre_F90_NAME(fhypre_sstructmaxwellsetnumprer, FNALU_HYPRE_SSTRUCTMAXWELLSETNUMPRER)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_SStructMaxwellSetNumPostRelax \
        hypre_F90_NAME(fhypre_sstructmaxwellsetnumpost, FNALU_HYPRE_SSTRUCTMAXWELLSETNUMPOST)
extern void hypre_F90_NAME(fhypre_sstructmaxwellsetnumpost, FNALU_HYPRE_SSTRUCTMAXWELLSETNUMPOST)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_SStructMaxwellSetLogging \
        hypre_F90_NAME(fhypre_sstructmaxwellsetlogging, FNALU_HYPRE_SSTRUCTMAXWELLSETLOGGING)
extern void hypre_F90_NAME(fhypre_sstructmaxwellsetlogging, FNALU_HYPRE_SSTRUCTMAXWELLSETLOGGING)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_SStructMaxwellSetPrintLevel \
        hypre_F90_NAME(fhypre_sstructmaxwellsetprintle, FNALU_HYPRE_SSTRUCTMAXWELLSETPRINTLE)
extern void hypre_F90_NAME(fhypre_sstructmaxwellsetprintle, FNALU_HYPRE_SSTRUCTMAXWELLSETPRINTLE)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_SStructMaxwellPrintLogging \
        hypre_F90_NAME(fhypre_sstructmaxwellprintloggi, FNALU_HYPRE_SSTRUCTMAXWELLPRINTLOGGI)
extern void hypre_F90_NAME(fhypre_sstructmaxwellprintloggi, FNALU_HYPRE_SSTRUCTMAXWELLPRINTLOGGI)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_SStructMaxwellGetNumIterations \
        hypre_F90_NAME(fhypre_sstructmaxwellgetnumiter, FNALU_HYPRE_SSTRUCTMAXWELLGETNUMITER)
extern void hypre_F90_NAME(fhypre_sstructmaxwellgetnumiter, FNALU_HYPRE_SSTRUCTMAXWELLGETNUMITER)
(hypre_F90_Obj *, NALU_HYPRE_Int *);

#define NALU_HYPRE_SStructMaxwellGetFinalRelativeResidualNorm \
        hypre_F90_NAME(fhypre_sstructmaxwellgetfinalre, FNALU_HYPRE_SSTRUCTMAXWELLGETFINALRE)
extern void hypre_F90_NAME(fhypre_sstructmaxwellgetfinalre, FNALU_HYPRE_SSTRUCTMAXWELLGETFINALRE)
(hypre_F90_Obj *, NALU_HYPRE_Real *);

#define NALU_HYPRE_SStructMaxwellPhysBdy \
        hypre_F90_NAME(fhypre_sstructmaxwellphysbdy, FNALU_HYPRE_SSTRUCTMAXWELLPHYSBDY)
extern void hypre_F90_NAME(fhypre_sstructmaxwellphysbdy, FNALU_HYPRE_SSTRUCTMAXWELLPHYSBDY)
(hypre_F90_Obj *, NALU_HYPRE_Int *, NALU_HYPRE_Int *, NALU_HYPRE_Int *, NALU_HYPRE_Int *);

#define NALU_HYPRE_SStructMaxwellEliminateRowsCols \
        hypre_F90_NAME(fhypre_sstructmaxwelleliminater, FNALU_HYPRE_SSTRUCTMAXWELLELIMINATER)
extern void hypre_F90_NAME(fhypre_sstructmaxwelleliminater, FNALU_HYPRE_SSTRUCTMAXWELLELIMINATER)
(hypre_F90_Obj *, NALU_HYPRE_Int *, NALU_HYPRE_Int *);

#define NALU_HYPRE_SStructMaxwellZeroVector \
        hypre_F90_NAME(fhypre_sstructmaxwellzerovector, FNALU_HYPRE_SSTRUCTMAXWELLZEROVECTOR)
extern void hypre_F90_NAME(fhypre_sstructmaxwellzerovector, FNALU_HYPRE_SSTRUCTMAXWELLZEROVECTOR)
(hypre_F90_Obj *, NALU_HYPRE_Int *, NALU_HYPRE_Int *);

#ifdef __cplusplus
}
#endif
