!     Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
!     HYPRE Project Developers. See the top-level COPYRIGHT file for details.
!
!     SPDX-License-Identifier: (Apache-2.0 OR MIT)

!--------------------------------------------------------------------------
! GenerateLaplacian
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_generatelaplacian(fcom, fnx, fny, fnz,
     1                                    fcapp, fcapq, fcapr, 
     1                                    fp, fq, fr,
     2                                    fvalue, fmatrix)

      integer ierr
      integer fcomm
      integer fnx, fny, fnz
      integer fcapp, fcapq, fcapr
      integer fp, fq, fr
      double precision fvalue
      integer*8 fmatrix

      call nalu_hypre_GenerateLaplacian(fcomm, fnx, fny, fnz, fcapp, fcapq,
     1                       fcapr, fp, fq, fr, fvalue, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_generatelaplacian error: ', ierr
      endif

      return
      end

!--------------------------------------------------------------------------
! NALU_HYPRE_BoomerAMGCreate
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_boomeramgcreate(fsolver)

      integer ierr
      integer*8 fsolver

      call NALU_HYPRE_BoomerAMGCreate(fsolver, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_boomeramgcreate error: ', ierr
      endif

      return
      end

!--------------------------------------------------------------------------
! NALU_HYPRE_BoomerAMGDestroy
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_boomeramgdestroy(fsolver)

      integer ierr
      integer*8 fsolver

      call NALU_HYPRE_BoomerAMGDestroy(fsolver, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_boomeramgdestroy error: ', ierr
      endif

      return
      end

!--------------------------------------------------------------------------
! NALU_HYPRE_BoomerAMGSetup
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_boomeramgsetup(fsolver, fA, fb, fx)

      integer ierr
      integer*8 fsolver
      integer*8 fA
      integer*8 fb
      integer*8 fx

      call NALU_HYPRE_BoomerAMGSetup(fsolver, fA, fb, fx, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_boomeramgsetup error: ', ierr
      endif

      return
      end

!--------------------------------------------------------------------------
! NALU_HYPRE_BoomerAMGSolve
!-------------------------------------------------------------------------- 
      subroutine fnalu_hypre_boomeramgsolve(fsolver, fA, fb, fx)

      integer ierr
      integer*8 fsolver
      integer*8 fA
      integer*8 fb
      integer*8 fx

      call NALU_HYPRE_BoomerAMGSolve(fsolver, fA, fb, fx, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_boomeramgsolve error: ', ierr
      endif

      return
      end

!--------------------------------------------------------------------------
! NALU_HYPRE_BoomerAMGSolveT
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_boomeramgsolvet(fsolver, fA, fb, fx)

      integer ierr
      integer*8 fsolver
      integer*8 fA
      integer*8 fb
      integer*8 fx

      call NALU_HYPRE_BoomerAMGSolveT(fsolver, fA, fb, fx, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_boomeramgsolvet error: ', ierr
      endif

      return
      end

!--------------------------------------------------------------------------
! NALU_HYPRE_BoomerAMGSetRestriction
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_boomeramgsetrestriction(fsolver, frestr_par)

      integer ierr
      integer frestr_par
      integer*8 fsolver

      call NALU_HYPRE_BoomerAMGSetRestriction(fsolver, frestr_par, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_boomeramgsetrestriction error: ', ierr
      endif

      return
      end

!--------------------------------------------------------------------------
! NALU_HYPRE_BoomerAMGSetMaxLevels
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_boomeramgsetmaxlevels(fsolver, fmaxlvl)

      integer ierr
      integer fmaxlvl
      integer*8 fsolver

      call NALU_HYPRE_BoomerAMGSetMaxLevels(fsolver, fmaxlvl, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_boomeramgsetmaxlevels error: ', ierr
      endif

      return
      end

!--------------------------------------------------------------------------
! NALU_HYPRE_BoomerAMGGetMaxLevels
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_boomeramggetmaxlevels(fsolver, fmaxlvl)

      integer ierr
      integer fmaxlvl
      integer*8 fsolver

      call NALU_HYPRE_BoomerAMGGetMaxLevels(fsolver, fmaxlvl, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_boomeramggetmaxlevels error: ', ierr
      endif

      return
      end

!--------------------------------------------------------------------------
! NALU_HYPRE_BoomerAMGSetStrongThreshold
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_boomeramgsetstrongthrshl(fsolver, fstrong)

      integer ierr
      double precision fstrong
      integer*8 fsolver

      call NALU_HYPRE_BoomerAMGSetStrongThrshld(fsolver, fstrong, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_boomeramgsetstrongthreshold error: ', ierr
      endif

      return
      end

!--------------------------------------------------------------------------
! NALU_HYPRE_BoomerAMGGetStrongThreshold
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_boomeramggetstrongthrshl(fsolver, fstrong)

      integer ierr
      double precision fstrong
      integer*8 fsolver

      call NALU_HYPRE_BoomerAMGGetStrongThrshld(fsolver, fstrong, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_boomeramggetstrongthreshold error: ', ierr
      endif

      return
      end

!--------------------------------------------------------------------------
! NALU_HYPRE_BoomerAMGSetMaxRowSum
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_boomeramgsetmaxrowsum(fsolver, fmaxrowsum)

      integer ierr
      double precision fmaxrowsum
      integer*8 fsolver

      call NALU_HYPRE_BoomerAMGSetMaxRowSum(fsolver, fmaxrowsum, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_boomeramgsetmaxrowsum error: ', ierr
      endif

      return
      end

!--------------------------------------------------------------------------
! NALU_HYPRE_BoomerAMGGetMaxRowSum
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_boomeramggetmaxrowsum(fsolver, fmaxrowsum)

      integer ierr
      double precision fmaxrowsum
      integer*8 fsolver

      call NALU_HYPRE_BoomerAMGGetMaxRowSum(fsolver, fmaxrowsum, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_boomeramggetmaxrowsum error: ', ierr
      endif

      return
      end

!--------------------------------------------------------------------------
! NALU_HYPRE_BoomerAMGSetTruncFactor
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_boomeramgsettruncfactor(fsolver, ftrunc_factor)

      integer ierr
      double precision ftrunc_factor
      integer*8 fsolver

      call NALU_HYPRE_BoomerAMGSetTruncFactor(fsolver, ftrunc_factor, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_boomeramgsettruncfactor error: ', ierr
      endif

      return
      end

!--------------------------------------------------------------------------
! NALU_HYPRE_BoomerAMGGetTruncFactor
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_boomeramggettruncfactor(fsolver, ftrunc_factor)

      integer ierr
      double precision ftrunc_factor
      integer*8 fsolver

      call NALU_HYPRE_BoomerAMGGetTruncFactor(fsolver, ftrunc_factor, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_boomeramggettruncfactor error: ', ierr
      endif

      return
      end

!--------------------------------------------------------------------------
! NALU_HYPRE_BoomerAMGSetSCommPkgSwitch
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_boomeramgsetscommpkgswit(fsolver, fcommswtch)

      integer ierr
      integer fcommswtch
      integer*8 fsolver

      call NALU_HYPRE_BoomerAMGSetSCommPkgSwitc(fsolver, fcommswtch, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_boomeramgsetscommpkgswitch error: ', ierr
      endif

      return
      end

!--------------------------------------------------------------------------
! NALU_HYPRE_BoomerAMGSetInterpType
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_boomeramgsetinterptype(fsolver, finterp)

      integer ierr
      integer finterp
      integer*8 fsolver

      call NALU_HYPRE_BoomerAMGSetInterpType(fsolver, finterp, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_boomeramgsetinterptype error: ', ierr
      endif

      return
      end

!--------------------------------------------------------------------------
! NALU_HYPRE_BoomerAMGSetMinIter
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_boomeramgsetminiter(fsolver, fminiter)

      integer ierr
      integer fminiter  
      integer*8 fsolver

      call NALU_HYPRE_BoomerAMGSetMinIter(fsolver, fminiter, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_boomeramgsetminiter error: ', ierr
      endif

      return
      end

!--------------------------------------------------------------------------
! NALU_HYPRE_BoomerAMGSetMaxIter
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_boomeramgsetmaxiter(fsolver, fmaxiter)

      integer ierr
      integer fmaxiter  
      integer*8 fsolver

      call NALU_HYPRE_BoomerAMGSetMaxIter(fsolver, fmaxiter, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_boomeramgsetmaxiter error: ', ierr
      endif

      return
      end

!--------------------------------------------------------------------------
! NALU_HYPRE_BoomerAMGGetMaxIter
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_boomeramggetmaxiter(fsolver, fmaxiter)

      integer ierr
      integer fmaxiter  
      integer*8 fsolver

      call NALU_HYPRE_BoomerAMGGetMaxIter(fsolver, fmaxiter, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_boomeramggetmaxiter error: ', ierr
      endif

      return
      end

!--------------------------------------------------------------------------
! NALU_HYPRE_BoomerAMGSetCoarsenType
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_boomeramgsetcoarsentype(fsolver, fcoarsen)

      integer ierr
      integer fcoarsen
      integer*8 fsolver

      call NALU_HYPRE_BoomerAMGSetCoarsenType(fsolver, fcoarsen, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_boomeramgsetcoarsentype error: ', ierr
      endif

      return
      end

!--------------------------------------------------------------------------
! NALU_HYPRE_BoomerAMGGetCoarsenType
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_boomeramggetcoarsentype(fsolver, fcoarsen)

      integer ierr
      integer fcoarsen
      integer*8 fsolver

      call NALU_HYPRE_BoomerAMGGetCoarsenType(fsolver, fcoarsen, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_boomeramggetcoarsentype error: ', ierr
      endif

      return
      end

!--------------------------------------------------------------------------
! NALU_HYPRE_BoomerAMGSetMeasureType
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_boomeramgsetmeasuretype(fsolver, fmeasure)

      integer ierr
      integer fmeasure
      integer*8 fsolver

      call NALU_HYPRE_BoomerAMGSetMeasureType(fsolver, fmeasure, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_boomeramgsetmeasuretype error: ', ierr
      endif

      return
      end

!--------------------------------------------------------------------------
! NALU_HYPRE_BoomerAMGGetMeasureType
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_boomeramggetmeasuretype(fsolver, fmeasure)

      integer ierr
      integer fmeasure
      integer*8 fsolver

      call NALU_HYPRE_BoomerAMGGetMeasureType(fsolver, fmeasure, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_boomeramggetmeasuretype error: ', ierr
      endif

      return
      end

!--------------------------------------------------------------------------
! NALU_HYPRE_BoomerAMGSetSetupType
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_boomeramgsetsetuptype(fsolver, fsetup)

      integer ierr
      integer fsetup
      integer*8 fsolver

      call NALU_HYPRE_BoomerAMGSetSetupType(fsolver, fsetup, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_boomeramgsetuptype error: ', ierr
      endif

      return
      end

!--------------------------------------------------------------------------
! NALU_HYPRE_BoomerAMGSetCycleType
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_boomeramgsetcycletype(fsolver, fcycle)

      integer ierr
      integer fcycle
      integer*8 fsolver

      call NALU_HYPRE_BoomerAMGSetCycleType(fsolver, fcycle, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_boomeramgsetcycletype error: ', ierr
      endif

      return
      end

!--------------------------------------------------------------------------
! NALU_HYPRE_BoomerAMGGetCycleType
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_boomeramggetcycletype(fsolver, fcycle)

      integer ierr
      integer fcycle
      integer*8 fsolver

      call NALU_HYPRE_BoomerAMGGetCycleType(fsolver, fcycle, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_boomeramggetcycletype error: ', ierr
      endif

      return
      end

!--------------------------------------------------------------------------
! NALU_HYPRE_BoomerAMGSetTol
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_boomeramgsettol(fsolver, ftol)

      integer ierr
      double precision ftol
      integer*8 fsolver

      call NALU_HYPRE_BoomerAMGSetTol(fsolver, ftol, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_boomeramgsettol error: ', ierr
      endif

      return
      end

!--------------------------------------------------------------------------
! NALU_HYPRE_BoomerAMGGetTol
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_boomeramggettol(fsolver, ftol)

      integer ierr
      double precision ftol
      integer*8 fsolver

      call NALU_HYPRE_BoomerAMGGetTol(fsolver, ftol, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_boomeramggettol error: ', ierr
      endif

      return
      end

!--------------------------------------------------------------------------
! NALU_HYPRE_BoomerAMGSetNumSweeps
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_boomeramgsetnumsweeps(fsolver, fnumsweeps)

      integer ierr
      integer fnumsweeps
      integer*8 fsolver

      call NALU_HYPRE_BoomerAMGSetNumSweeps(fsolver, fnumsweeps, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_boomeramgsetnumsweeps error: ', ierr
      endif

      return
      end

!--------------------------------------------------------------------------
! NALU_HYPRE_BoomerAMGSetCycleNumSweeps
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_boomeramgsetcyclenumswee(fsolver, fnumsweeps,
     1                                             fk)

      integer ierr
      integer fnumsweeps
      integer fk
      integer*8 fsolver

      call NALU_HYPRE_BoomerAMGSetCycleNumSweeps(fsolver, fnumsweeps, fk,
     1                                      ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_boomeramgsetcyclenumsweeps error: ', ierr
      endif

      return
      end

!--------------------------------------------------------------------------
! NALU_HYPRE_BoomerAMGGetCycleNumSweeps
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_boomeramggetcyclenumswee(fsolver, fnumsweeps,
     1                                             fk)

      integer ierr
      integer fnumsweeps
      integer fk
      integer*8 fsolver

      call NALU_HYPRE_BoomerAMGGetCycleNumSweeps(fsolver, fnumsweeps, fk,
     1                                      ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_boomeramggetcyclenumsweeps error: ', ierr
      endif

      return
      end
!--------------------------------------------------------------------------
! NALU_HYPRE_BoomerAMGInitGridRelaxation
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_boomeramginitgridrelaxat(fnumsweeps, fgridtype,
     1                                            fgridrelax, fcoarsen,
     2                                            frelaxwt, fmaxlvl)

      integer ierr
      integer fcoarsen
      integer fmaxlvl
      integer*8 fnumsweeps
      integer*8 fgridtype
      integer*8 fgridrelax
      integer*8 frelaxwt

      call NALU_HYPRE_BoomerAMGInitGridRelaxatn(fnumsweeps, fgridtype, 
     1                                       fgridrelax, fcoarsen, 
     2                                       frelaxwt, fmaxlvl, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_boomeramginitgridrelaxation error: ', ierr
      endif

      return
      end

!--------------------------------------------------------------------------
! NALU_HYPRE_BoomerAMGFinalizeGridRelaxation
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_boomeramgfingridrelaxatn(fnumsweeps, fgridtype,
     1                                            fgridrelax, frelaxwt)

      integer ierr
      integer*8 fnumsweeps
      integer*8 fgridtype
      integer*8 fgridrelax
      integer*8 frelaxwt

!     nalu_hypre_TFree(num_grid_sweeps);
!     nalu_hypre_TFree(grid_relax_type);
!     nalu_hypre_TFree(grid_relax_points);
!     nalu_hypre_TFree(relax_weights);

      ierr = 0

      return
      end

!--------------------------------------------------------------------------
! NALU_HYPRE_BoomerAMGSetRelaxType
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_boomeramgsetrelaxtype(fsolver, frelaxtype)

      integer ierr
      integer frelaxtype
      integer*8 fsolver

      call NALU_HYPRE_BoomerAMGSetRelaxType(fsolver, frelaxtype, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_boomeramgsetrelaxtype error: ', ierr
      endif
 
      return
      end

!--------------------------------------------------------------------------
! NALU_HYPRE_BoomerAMGSetCycleRelaxType
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_boomeramgsetcyclerelaxty(fsolver, frelaxtype,
     1                                             fk)

      integer ierr
      integer frelaxtype
      integer fk
      integer*8 fsolver

      call NALU_HYPRE_BoomerAMGSetCycleRelaxType(fsolver, fk, frelaxtype, 
     1                                      ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_boomeramgsetcyclerelaxtype error: ', ierr
      endif

      return
      end

!--------------------------------------------------------------------------
! NALU_HYPRE_BoomerAMGGetCycleRelaxType
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_boomeramggetcyclerelaxty(fsolver, frelaxtype,
     1                                             fk)

      integer ierr
      integer frelaxtype
      integer fk
      integer*8 fsolver

      call NALU_HYPRE_BoomerAMGGetCycleRelaxType(fsolver, fk, frelaxtype, 
     1                                      ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_boomeramggetcyclerelaxtype error: ', ierr
      endif

      return
      end

!--------------------------------------------------------------------------
! NALU_HYPRE_BoomerAMGSetRelaxOrder
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_boomeramgsetrelaxorder(fsolver, frlxorder)
     
      integer ierr
      integer frlxorder
      integer*8 fsolver

      call NALU_HYPRE_BoomerAMGSetRelaxOrder(fsolver, frlxorder, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_boomeramgsetrelaxorder error: ', ierr
      endif

      return
      end

!--------------------------------------------------------------------------
! NALU_HYPRE_BoomerAMGSetRelaxWt
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_boomeramgsetrelaxwt(fsolver, frelaxwt)
     
      integer ierr
      integer*8 fsolver
      double precision frelaxwt

      call NALU_HYPRE_BoomerAMGSetRelaxWt(fsolver, frelaxwt, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_boomeramgsetrelaxwt error: ', ierr
      endif

      return
      end

!--------------------------------------------------------------------------
! NALU_HYPRE_BoomerAMGSetLevelRelaxWt
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_boomeramgsetlevelrelaxwt(fsolver, frelaxwt, 
     1                                           flevel)

      integer ierr
      integer flevel
      integer*8 fsolver
      double precision frelaxwt

      call NALU_HYPRE_BoomerAMGSetLevelRelaxWt(fsolver, frelaxwt, flevel,
     1                                    ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_boomeramgsetlevelrelaxwt error: ', ierr
      endif

      return
      end

!--------------------------------------------------------------------------
! NALU_HYPRE_BoomerAMGSetOuterWt
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_boomeramgsetouterwt(fsolver, fouterwt)
     
      integer ierr
      integer*8 fsolver
      double precision fouterwt

      call NALU_HYPRE_BoomerAMGSetOuterWt(fsolver, fouterwt, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_boomeramgsetouterwt error: ', ierr
      endif

      return
      end

!--------------------------------------------------------------------------
! NALU_HYPRE_BoomerAMGSetLevelOuterWt
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_boomeramgsetlevelouterwt(fsolver, fouterwt, 
     1                                           flevel)

      integer ierr
      integer flevel
      integer*8 fsolver
      double precision fouterwt

      call NALU_HYPRE_BoomerAMGSetLevelOuterWt(fsolver, fouterwt, flevel,
     1                                    ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_boomeramgsetlevelouterwt error: ', ierr
      endif

      return
      end

!--------------------------------------------------------------------------
! NALU_HYPRE_BoomerAMGSetSmoothType
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_boomeramgsetsmoothtype(fsolver, fsmooth) 

      integer ierr
      integer fsmooth
      integer*8 fsolver

      call NALU_HYPRE_BoomerAMGSetSmoothType(fsolver, fsmooth, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_boomeramgsetsmoothtype error: ', ierr
      endif

      return
      end

!--------------------------------------------------------------------------
! NALU_HYPRE_BoomerAMGGetSmoothType
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_boomeramggetsmoothtype(fsolver, fsmooth) 

      integer ierr
      integer fsmooth
      integer*8 fsolver

      call NALU_HYPRE_BoomerAMGGetSmoothType(fsolver, fsmooth, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_boomeramggetsmoothtype error: ', ierr
      endif

      return
      end

!--------------------------------------------------------------------------
! NALU_HYPRE_BoomerAMGSetSmoothNumLvls
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_boomeramgsetsmoothnumlvl(fsolver, fsmooth) 

      integer ierr
      integer fsmooth
      integer*8 fsolver

      call NALU_HYPRE_BoomerAMGSetSmoothNumLvls(fsolver, fsmooth, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_boomeramgsetsmoothnumlvls error: ', ierr
      endif

      return
      end

!--------------------------------------------------------------------------
! NALU_HYPRE_BoomerAMGGetSmoothNumLvls
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_boomeramggetsmoothnumlvl(fsolver, fsmooth) 

      integer ierr
      integer fsmooth
      integer*8 fsolver

      call NALU_HYPRE_BoomerAMGGetSmoothNumLvls(fsolver, fsmooth, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_boomeramggetsmoothnumlvls error: ', ierr
      endif

      return
      end

!--------------------------------------------------------------------------
! NALU_HYPRE_BoomerAMGSetSmoothNumSwps
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_boomeramgsetsmoothnumswp(fsolver, fsmooth) 

      integer ierr
      integer fsmooth
      integer*8 fsolver

      call NALU_HYPRE_BoomerAMGSetSmoothNumSwps(fsolver, fsmooth, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_boomeramgsetsmoothnumswps error: ', ierr
      endif

      return
      end

!--------------------------------------------------------------------------
! NALU_HYPRE_BoomerAMGGetSmoothNumSwps
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_boomeramggetsmoothnumswp(fsolver, fsmooth) 

      integer ierr
      integer fsmooth
      integer*8 fsolver

      call NALU_HYPRE_BoomerAMGGetSmoothNumSwps(fsolver, fsmooth, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_boomeramggetsmoothnumswps error: ', ierr
      endif

      return
      end

!--------------------------------------------------------------------------
! NALU_HYPRE_BoomerAMGSetLogging
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_boomeramgsetlogging(fsolver, flogging)

      integer ierr
      integer flogging
      integer*8 fsolver

      call NALU_HYPRE_BoomerAMGSetLogging(fsolver, flogging, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_boomeramgsetlogging error: ', ierr
      endif

      return
      end

!--------------------------------------------------------------------------
! NALU_HYPRE_BoomerAMGGetLogging
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_boomeramggetlogging(fsolver, flogging)

      integer ierr
      integer flogging
      integer*8 fsolver

      call NALU_HYPRE_BoomerAMGGetLogging(fsolver, flogging, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_boomeramggetlogging error: ', ierr
      endif

      return
      end

!--------------------------------------------------------------------------
! NALU_HYPRE_BoomerAMGSetPrintLevel
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_boomeramgsetprintlevel(fsolver, fprintlevel)

      integer ierr
      integer fprintlevel
      integer*8 fsolver

      call NALU_HYPRE_BoomerAMGSetPrintLevel(fsolver, fprintlevel, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_boomeramgsetprintlevel error: ', ierr
      endif

      return
      end

!--------------------------------------------------------------------------
! NALU_HYPRE_BoomerAMGGetPrintLevel
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_boomeramggetprintlevel(fsolver, fprintlevel)

      integer ierr
      integer fprintlevel
      integer*8 fsolver

      call NALU_HYPRE_BoomerAMGGetPrintLevel(fsolver, fprintlevel, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_boomeramggetprintlevel error: ', ierr
      endif

      return
      end

!--------------------------------------------------------------------------
! NALU_HYPRE_BoomerAMGSetPrintFileName
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_boomeramgsetprintfilenam(fsolver, fname)

      integer ierr
      integer*8 fsolver
      character*(*) fname

      call NALU_HYPRE_BoomerAMGSetPrintFileName(fsolver, fname, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_boomeramgsetprintfilename error: ', ierr
      endif

      return
      end

!--------------------------------------------------------------------------
! NALU_HYPRE_BoomerAMGSetDebugFlag
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_boomeramgsetdebugflag(fsolver, fdebug)

      integer ierr
      integer fdebug
      integer*8 fsolver

      call NALU_HYPRE_BoomerAMGSetDebugFlag(fsolver, fdebug, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_boomeramgsetdebugflag error: ', ierr
      endif

      return
      end

!--------------------------------------------------------------------------
! NALU_HYPRE_BoomerAMGGetDebugFlag
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_boomeramggetdebugflag(fsolver, fdebug)

      integer ierr
      integer fdebug
      integer*8 fsolver

      call NALU_HYPRE_BoomerAMGGetDebugFlag(fsolver, fdebug, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_boomeramggetdebugflag error: ', ierr
      endif

      return
      end

!--------------------------------------------------------------------------
! NALU_HYPRE_BoomerAMGGetNumIterations
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_boomeramggetnumiteration(fsolver, fnumiter)

      integer ierr
      integer fnumiter
      integer*8 fsolver

      call NALU_HYPRE_BoomerAMGGetNumIterations(fsolver, fnumiter, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_boomeramggetnumiterations error: ', ierr
      endif

      return
      end

!--------------------------------------------------------------------------
! NALU_HYPRE_BoomerAMGGetCumNumIterations
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_boomeramggetcumnumiterat(fsolver, fnumiter)

      integer ierr
      integer fnumiter
      integer*8 fsolver

      call NALU_HYPRE_BoomerAMGGetNumIterations(fsolver, fnumiter, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_boomeramggetnumiterations error: ', ierr
      endif

      return
      end

!--------------------------------------------------------------------------
! NALU_HYPRE_BoomerAMGGetResidual
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_boomeramggetresidual(fsolver, fresid)

      integer ierr
      integer*8 fsolver
      double precision fresid

      call NALU_HYPRE_BoomerAMGGetResidual(fsolver, fresid, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_boomeramggetresidual error: ', ierr
      endif

      return
      end

!--------------------------------------------------------------------------
! NALU_HYPRE_BoomerAMGGetFinalRelativeResidual
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_boomeramggetfinalreltvre(fsolver, frelresid)

      integer ierr
      integer*8 fsolver
      double precision frelresid

      call NALU_HYPRE_BoomerAMGGetFinalReltvRes(fsolver, frelresid, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_boomeramggetfinalrelativeres error: ', ierr
      endif

      return
      end

!--------------------------------------------------------------------------
! NALU_HYPRE_BoomerAMGSetVariant
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_boomeramgsetvariant(fsolver, fvariant)

      integer ierr
      integer fvariant
      integer*8 fsolver

      call NALU_HYPRE_BoomerAMGSetVariant(fsolver, fvariant, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_boomeramgsetvariant error: ', ierr
      endif

      return
      end

!--------------------------------------------------------------------------
! NALU_HYPRE_BoomerAMGGetVariant
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_boomeramggetvariant(fsolver, fvariant)

      integer ierr
      integer fvariant
      integer*8 fsolver

      call NALU_HYPRE_BoomerAMGGetVariant(fsolver, fvariant, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_boomeramggetvariant error: ', ierr
      endif

      return
      end

!--------------------------------------------------------------------------
! NALU_HYPRE_BoomerAMGSetOverlap
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_boomeramgsetoverlap(fsolver, foverlap)

      integer ierr
      integer foverlap
      integer*8 fsolver

      call NALU_HYPRE_BoomerAMGSetOverlap(fsolver, foverlap, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_boomeramgsetoverlap error: ', ierr
      endif

      return
      end

!--------------------------------------------------------------------------
! NALU_HYPRE_BoomerAMGGetOverlap
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_boomeramggetoverlap(fsolver, foverlap)

      integer ierr
      integer foverlap
      integer*8 fsolver

      call NALU_HYPRE_BoomerAMGGetOverlap(fsolver, foverlap, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_boomeramggetoverlap error: ', ierr
      endif

      return
      end

!--------------------------------------------------------------------------
! NALU_HYPRE_BoomerAMGSetDomainType
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_boomeramgsetdomaintype(fsolver, fdomain)

      integer ierr
      integer fdomain
      integer*8 fsolver

      call NALU_HYPRE_BoomerAMGSetDomainType(fsolver, fdomain, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_boomeramgsetdomaintype error: ', ierr
      endif

      return
      end

!--------------------------------------------------------------------------
! NALU_HYPRE_BoomerAMGGetDomainType
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_boomeramggetdomaintype(fsolver, fdomain)

      integer ierr
      integer fdomain
      integer*8 fsolver

      call NALU_HYPRE_BoomerAMGGetDomainType(fsolver, fdomain, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_boomeramggetdomaintype error: ', ierr
      endif

      return
      end

!--------------------------------------------------------------------------
! NALU_HYPRE_BoomerAMGSetSchwarzRlxWt
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_boomeramgsetschwarzrlxwt(fsolver, fschwarz)

      integer ierr
      integer fschwarz
      integer*8 fsolver

      call NALU_HYPRE_BoomerAMGSetSchwarzRlxWt(fsolver, fschwarz, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_boomeramgsetschwarzrlxwt error: ', ierr
      endif

      return
      end

!--------------------------------------------------------------------------
! NALU_HYPRE_BoomerAMGGetSchwarzRlxWt
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_boomeramggetschwarzrlxwt(fsolver, fschwarz)

      integer ierr
      integer fschwarz
      integer*8 fsolver

      call NALU_HYPRE_BoomerAMGGetSchwarzRlxWt(fsolver, fschwarz, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_boomeramggetschwarzrlxwt error: ', ierr
      endif

      return
      end

!--------------------------------------------------------------------------
! NALU_HYPRE_BoomerAMGSetSym
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_boomeramgsetsym(fsolver, fsym)

      integer ierr
      integer fsym
      integer*8 fsolver

      call NALU_HYPRE_BoomerAMGSetSym(fsolver, fsym, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_boomeramgsetsym error: ', ierr
      endif

      return
      end

!--------------------------------------------------------------------------
! NALU_HYPRE_BoomerAMGSetLevel
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_boomeramgsetlevel(fsolver, flevel)

      integer ierr
      integer flevel
      integer*8 fsolver

      call NALU_HYPRE_BoomerAMGSetLevel(fsolver, flevel, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_boomeramgsetlevel error: ', ierr
      endif

      return
      end

!--------------------------------------------------------------------------
! NALU_HYPRE_BoomerAMGSetFilter
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_boomeramgsetfilter(fsolver, ffilter)

      integer ierr
      integer ffilter
      integer*8 fsolver

      call NALU_HYPRE_BoomerAMGSetFilter(fsolver, ffilter, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_boomeramgsetfilter error: ', ierr
      endif

      return
      end

!--------------------------------------------------------------------------
! NALU_HYPRE_BoomerAMGSetDropTol
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_boomeramgsetdroptol(fsolver, fdroptol)

      integer ierr
      integer fdroptol
      integer*8 fsolver

      call NALU_HYPRE_BoomerAMGSetDropTol(fsolver, fdroptol, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_boomeramgsetdroptol error: ', ierr
      endif

      return
      end

!--------------------------------------------------------------------------
! NALU_HYPRE_BoomerAMGSetMaxNzPerRow
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_boomeramgsetmaxnzperrow(fsolver, fmaxnzperrow)

      integer ierr
      integer fmaxnzperrow
      integer*8 fsolver

      call NALU_HYPRE_BoomerAMGSetMaxNzPerRow(fsolver, fmaxnzperrow, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_boomeramgsetmaxnzperrow error: ', ierr
      endif

      return
      end

!--------------------------------------------------------------------------
! NALU_HYPRE_BoomerAMGSetEuclidFile
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_boomeramgseteuclidfile(fsolver, ffile)

      integer ierr
      integer*8 fsolver
      character*(*) ffile

      call NALU_HYPRE_BoomerAMGSetEuclidFile(fsolver, ffile, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_boomeramgseteuclidfile error: ', ierr
      endif

      return
      end

!--------------------------------------------------------------------------
! NALU_HYPRE_BoomerAMGSetNumFunctions
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_boomeramgsetnumfunctions(fsolver, fnfncs)

      integer ierr
      integer fnfncs
      integer*8 fsolver

      call NALU_HYPRE_BoomerAMGSetNumFunctions(fsolver, fnfncs, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_boomeramgsetnumfunctions error: ', ierr
      endif

      return
      end

!--------------------------------------------------------------------------
! NALU_HYPRE_BoomerAMGGetNumFunctions
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_boomeramggetnumfunctions(fsolver, fnfncs)

      integer ierr
      integer fnfncs
      integer*8 fsolver

      call NALU_HYPRE_BoomerAMGGetNumFunctions(fsolver, fnfncs, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_boomeramggetnumfunctions error: ', ierr
      endif

      return
      end

!--------------------------------------------------------------------------
! NALU_HYPRE_BoomerAMGSetNodal
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_boomeramgsetnodal(fsolver, fnodal)

      integer ierr
      integer fnodal
      integer*8 fsolver

      call NALU_HYPRE_BoomerAMGSetNodal(fsolver, fnodal, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_boomeramgsetnodal error: ', ierr
      endif

      return
      end

!--------------------------------------------------------------------------
! NALU_HYPRE_BoomerAMGSetDofFunc
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_boomeramgsetdoffunc(fsolver, fdoffunc)

      integer ierr
      integer fdoffunc
      integer*8 fsolver

      call NALU_HYPRE_BoomerAMGSetDofFunc(fsolver, fdoffunc, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_boomeramgsetdoffunc error: ', ierr
      endif

      return
      end

!--------------------------------------------------------------------------
! NALU_HYPRE_BoomerAMGSetNumPaths
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_boomeramgsetnumpaths(fsolver, fnumpaths)

      integer ierr
      integer fnumpaths
      integer*8 fsolver

      call NALU_HYPRE_BoomerAMGSetNumPaths(fsolver, fnumpaths, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_boomeramgsetnumpaths error: ', ierr
      endif

      return
      end

!--------------------------------------------------------------------------
! NALU_HYPRE_BoomerAMGSetAggNumLevels
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_boomeramgsetaggnumlevels(fsolver, fagglvl)

      integer ierr
      integer fagglvl
      integer*8 fsolver

      call NALU_HYPRE_BoomerAMGSetAggNumLevels(fsolver, fagglvl, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_boomeramgsetaggnumlevels error: ', ierr
      endif

      return
      end

!--------------------------------------------------------------------------
! NALU_HYPRE_BoomerAMGSetGSMG
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_boomeramgsetgsmg(fsolver, fgsmg)

      integer ierr
      integer fgsmg
      integer*8 fsolver

      call NALU_HYPRE_BoomerAMGSetGSMG(fsolver, fgsmg, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_boomeramgsetgsmg error: ', ierr
      endif

      return
      end

!--------------------------------------------------------------------------
! NALU_HYPRE_BoomerAMGSetNumSamples
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_boomeramgsetnumsamples(fsolver, fsamples)

      integer ierr
      integer fsamples
      integer*8 fsolver

      call NALU_HYPRE_BoomerAMGSetNumSamples(fsolver, fsamples, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_boomeramgsetnumsamples error: ', ierr
      endif

      return
      end

!--------------------------------------------------------------------------
! NALU_HYPRE_ParCSRBiCGSTABCreate
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_parcsrbicgstabcreate(fcomm, fsolver)

      integer fcomm
      integer ierr
      integer*8 fsolver

      call NALU_HYPRE_ParCSRBiCGSTABCreate(fcomm, fsolver, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_parcsrbicgstabcreate error: ', ierr
      endif

      return
      end

!--------------------------------------------------------------------------
! NALU_HYPRE_ParCSRBiCGSTABDestroy
!-------------------------------------------------------------------------- 
      subroutine fnalu_hypre_parcsrbicgstabdestroy(fsolver)

      integer ierr
      integer*8 fsolver

      call NALU_HYPRE_ParCSRBiCGSTABDestroy(fsolver, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_parcsrbicgstabdestroy error: ', ierr
      endif

      return
      end

!--------------------------------------------------------------------------
! NALU_HYPRE_ParCSRBiCGSTABSetup
!------------------------------------------------------------------------- 
      subroutine fnalu_hypre_parcsrbicgstabsetup(fsolver, fA, fb, fx)

      integer ierr
      integer*8 fsolver
      integer*8 fA
      integer*8 fb
      integer*8 fx

      call NALU_HYPRE_ParCSRBiCGSTABSetup(fsolver, fA, fb, fx, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_parcsrbicgstabsetup error: ', ierr
      endif

      return
      end

!--------------------------------------------------------------------------
! NALU_HYPRE_ParCSRBiCGSTABSolve
!-------------------------------------------------------------------------- 
      subroutine fnalu_hypre_parcsrbicgstabsolve(fsolver, fA, fb, fx)
     
      integer ierr
      integer*8 fsolver
      integer*8 fA
      integer*8 fb
      integer*8 fx

      call NALU_HYPRE_ParCSRBiCGSTABSolve(fsolver, fA, fb, fx)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_parcsrbicgstabsolve error: ', ierr
      endif

      return
      end

!-------------------------------------------------------------------------
! NALU_HYPRE_ParCSRBiCGSTABSetTol
!-------------------------------------------------------------------------
      subroutine fnalu_hypre_parcsrbicgstabsettol(fsolver, ftol)

      integer ierr
      integer*8 fsolver
      double precision ftol

      call NALU_HYPRE_ParCSRBiCGSTABSetTol(fsolver, ftol, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_parcsrbicgstabsettol error: ', ierr
      endif

      return
      end

!-------------------------------------------------------------------------
! NALU_HYPRE_ParCSRBiCGSTABSetMinIter
!-------------------------------------------------------------------------
      subroutine fnalu_hypre_parcsrbicgstabsetminiter(fsolver, fminiter)

      integer ierr
      integer fminiter
      integer*8 fsolver

      call NALU_HYPRE_ParCSRBiCGSTABSetMinIter(fsolver, fminiter, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_parcsrparcsrbicgstabsetminiter error: ', ierr
      endif

      return
      end

!-------------------------------------------------------------------------
! NALU_HYPRE_ParCSRBiCGSTABSetMaxIter
!-------------------------------------------------------------------------
      subroutine fnalu_hypre_parcsrbicgstabsetmaxiter(fsolver, fmaxiter)

      integer ierr
      integer fmaxiter
      integer*8 fsolver

      call NALU_HYPRE_ParCSRBiCGSTABSetMaxIter(fsolver, fmaxiter, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_parcsrbicgstabsetmaxiter error: ', ierr
      endif

      return
      end

!-------------------------------------------------------------------------
! NALU_HYPRE_ParCSRBiCGSTABSetStopCrit
!-------------------------------------------------------------------------
      subroutine fnalu_hypre_parcsrbicgstabsetstopcrit(fsolver, fstopcrit)

      integer ierr
      integer fstopcrit
      integer*8 fsolver

      call NALU_HYPRE_ParCSRBiCGSTABSetStopCrit(fsolver, fstopcrit, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_parcsrbicgstabsetstopcrit error: ', ierr
      endif

      return
      end

!-------------------------------------------------------------------------
! NALU_HYPRE_ParCSRBiCGSTABSetPrecond
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_parcsrbicgstabsetprecond(fsolver, fprecond_id, 
     1                                           fprecond)       

      integer ierr
      integer fprecond_id
      integer*8 fsolver
      integer*8 fprecond

      call NALU_HYPRE_ParCSRBiCGSTABSetPrecond(fsolver, fprecond_id, 
     1                                    fprecond, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_parcsrbicgstabsetprecond error: ', ierr
      endif

      return
      end

!-------------------------------------------------------------------------
! NALU_HYPRE_ParCSRBiCGSTABGetPrecond
!-------------------------------------------------------------------------
      subroutine fnalu_hypre_parcsrbicgstabgetprecond(fsolver, fprecond)
      
      integer ierr
      integer*8 fsolver
      integer*8 fprecond

      call NALU_HYPRE_ParCSRBiCGSTABGetPrecond(fsolver, fprecond)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_parcsrbicgstabgetprecond error: ', ierr
      endif

      return
      end

!-------------------------------------------------------------------------
! NALU_HYPRE_ParCSRBiCGSTABSetLogging
!-------------------------------------------------------------------------
      subroutine fnalu_hypre_parcsrbicgstabsetlogging(fsolver, flog)

      integer ierr
      integer flog
      integer*8 fsolver

      call NALU_HYPRE_ParCSRBiCGSTABSetLogging(fsolver, flog, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_parcsrbicgstabsetlogging error: ', ierr
      endif

      return
      end

!-------------------------------------------------------------------------
! NALU_HYPRE_ParCSRBiCGSTABSetPrintLevel
!-------------------------------------------------------------------------
      subroutine fnalu_hypre_parcsrbicgstabsetprintle(fsolver, fprntlvl)

      integer ierr
      integer fprntlvl
      integer*8 fsolver

      call NALU_HYPRE_ParCSRBiCGSTABSetPrintLev(fsolver, fprntlvl, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_parcsrbicgstabsetprintlevel error: ', ierr
      endif

      return
      end

!-------------------------------------------------------------------------
! NALU_HYPRE_ParCSRBiCGSTABGetNumIter
!-------------------------------------------------------------------------
      subroutine fnalu_hypre_parcsrbicgstabgetnumiter(fsolver, fnumiter)

      integer ierr
      integer fnumiter
      integer*8 fsolver

      call NALU_HYPRE_ParCSRBiCGSTABGetNumIter(fsolver, fnumiter, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_parcsrbicgstabgetnumiterations error: ', ierr
      endif

      return
      end

!-------------------------------------------------------------------------
! NALU_HYPRE_ParCSRBiCGSTABGetFinalRelativeResidualNorm
!-------------------------------------------------------------------------
      subroutine fnalu_hypre_parcsrbicgstabgetfinalre(fsolver, fnorm)

      integer ierr
      integer*8 fsolver
      double precision fnorm

      call NALU_HYPRE_ParCSRBiCGSTABGetFinalRel(fsolver, fnorm, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_parcsrbicgstabgetfinalrel error: ', ierr
      endif
      return
      end



!-------------------------------------------------------------------------
! NALU_HYPRE_BlockTridiagCreate
!-------------------------------------------------------------------------
      subroutine fnalu_hypre_blocktridiagcreate(fsolver)

      integer ierr
      integer*8 fsolver

      call NALU_HYPRE_BlockTridiagCreate(fsolver, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_blocktridiagcreate error: ', ierr
      endif

      return
      end

!-------------------------------------------------------------------------
! NALU_HYPRE_BlockTridiagDestroy
!-------------------------------------------------------------------------
      subroutine fnalu_hypre_blocktridiagdestroy(fsolver)

      integer ierr
      integer*8 fsolver

      call NALU_HYPRE_BlockTridiagDestroy(fsolver, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_blocktridiagdestroy error: ', ierr
      endif

      return
      end

!-------------------------------------------------------------------------
! NALU_HYPRE_BlockTridiagSetup
!-------------------------------------------------------------------------
      subroutine fnalu_hypre_blocktridiagsetup(fsolver, fA, fb, fx)

      integer ierr
      integer*8 fsolver
      integer*8 fA
      integer*8 fb
      integer*8 fx

      call NALU_HYPRE_BlockTridiagSetup(fsolver, fA, fb, fx, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_blocktridiagsetup error: ', ierr
      endif

      return
      end

!-------------------------------------------------------------------------
! NALU_HYPRE_BlockTridiagSolve
!-------------------------------------------------------------------------
      subroutine fnalu_hypre_blocktridiagsolve(fsolver, fA, fb, fx)

      integer ierr
      integer*8 fsolver
      integer*8 fA
      integer*8 fb
      integer*8 fx

      call NALU_HYPRE_BlockTridiagSolve(fsolver, fA, fb, fx, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_blocktridiagsolve error: ', ierr
      endif

      return
      end

!-------------------------------------------------------------------------
! NALU_HYPRE_BlockTridiagSetIndexSet
!-------------------------------------------------------------------------
      subroutine fnalu_hypre_blocktridiagsetindexset(fsolver, fn, finds)

      integer ierr
      integer fn
      integer finds
      integer*8 fsolver

      call NALU_HYPRE_BlockTridiagSetIndexSet(fsolver, fn, finds, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_blocktridiagsetindexset error: ', ierr
      endif

      return
      end

!-------------------------------------------------------------------------
! NALU_HYPRE_BlockTridiagSetAMGStrengthThreshold
!-------------------------------------------------------------------------
      subroutine fnalu_hypre_blocktridiagsetamgstreng(fsolver, fthresh)

      integer ierr
      integer*8 fsolver
      double precision fthresh

      call NALU_HYPRE_BlockTridiagSetAMGStrengt(fsolver, fthresh,
     1                                     ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_blocktridiagsetamgstrengththreshold error: ',
     1                                            ierr
      endif

      return
      end

!-------------------------------------------------------------------------
! NALU_HYPRE_BlockTridiagSetAMGNumSweeps
!-------------------------------------------------------------------------
      subroutine fnalu_hypre_blocktridiagsetamgnumswe(fsolver, fnumsweep)

      integer ierr
      integer fnumsweep
      integer*8 fsolver

      call NALU_HYPRE_BlockTridiagSetAMGNumSwee(fsolver, fnumsweep, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_blocktridiagsetamgnumsweeps error: ', ierr
      endif

      return
      end

!-------------------------------------------------------------------------
! NALU_HYPRE_BlockTridiagSetAMGRelaxType
!-------------------------------------------------------------------------
      subroutine fnalu_hypre_blocktridiagsetamgrelaxt(fsolver, frlxtyp)

      integer ierr
      integer frlxtyp
      integer*8 fsolver

      call NALU_HYPRE_BlockTridiagSetAMGRelaxTy(fsolver, frlxtyp, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_blocktridiagsetamgrelaxype error: ', ierr
      endif

      return
      end

!-------------------------------------------------------------------------
! NALU_HYPRE_BlockTridiagSetPrintLevel
!-------------------------------------------------------------------------
      subroutine fnalu_hypre_blocktridiagsetprintleve(fsolver, fprntlvl)

      integer ierr
      integer fprntlvl
      integer*8 fsolver

      call NALU_HYPRE_BlockTridiagSetPrintLevel(fsolver, fprntlvl, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_blocktridiagsetprintlevel error: ', ierr
      endif

      return
      end



!-------------------------------------------------------------------------
! NALU_HYPRE_ParCSRCGNRCreate
!-------------------------------------------------------------------------
      subroutine fnalu_hypre_parcsrcgnrcreate(fcomm, fsolver)

      integer ierr
      integer fcomm
      integer*8 fsolver

      call NALU_HYPRE_ParCSRCGNRCreate(fcomm, fsolver, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_parcsrcgnrcreate error: ', ierr
      endif

      return
      end

!-------------------------------------------------------------------------
! NALU_HYPRE_ParCSRCGNRDestroy
!------------------------------------------------------------------------- 
      subroutine fnalu_hypre_parcsrcgnrdestroy(fsolver)

      integer ierr
      integer*8 fsolver

      call NALU_HYPRE_ParCSRCGNRDestroy(fsolver, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_parcsrcgnrdestroy error: ', ierr
      endif

      return
      end

!-------------------------------------------------------------------------
! NALU_HYPRE_ParCSRCGNRSetup
!------------------------------------------------------------------------- 
      subroutine fnalu_hypre_parcsrcgnrsetup(fsolver, fA, fb, fx)

      integer ierr
      integer*8 fA
      integer*8 fb
      integer*8 fx

      call NALU_HYPRE_ParCSRCGNRSetup(fsolver, fA, fb, fx, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_parcsrcgnrsetup error: ', ierr
      endif

      return
      end

!-------------------------------------------------------------------------
! NALU_HYPRE_ParCSRCGNRSolve
!-------------------------------------------------------------------------
      subroutine fnalu_hypre_parcsrcgnrsolve(fsolver, fA, fb, fx)

      integer ierr
      integer*8 fA
      integer*8 fb
      integer*8 fx

      call NALU_HYPRE_ParCSRCGNRSolve(fsolver, fA, fb, fx, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_parcsrcgnrsolve error: ', ierr
      endif

      return
      end

!-------------------------------------------------------------------------
! NALU_HYPRE_ParCSRCGNRSetTol
!-------------------------------------------------------------------------
      subroutine fnalu_hypre_parcsrcgnrsettol(fsolver, ftol)

      integer ierr
      integer*8 fsolver
      double precision ftol

      call NALU_HYPRE_ParCSRCGNRSetTol(fsolver, ftol, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_parcsrcgnrsettol error: ', ierr
      endif

      return
      end

!-------------------------------------------------------------------------
! NALU_HYPRE_ParCSRCGNRSetMinIter
!-------------------------------------------------------------------------
      subroutine fnalu_hypre_parcsrcgnrsetminiter(fsolver, fminiter)
 
      integer ierr
      integer fminiter
      integer*8 fsolver

      call NALU_HYPRE_ParCSRCGNRSetMinIter(fsolver, fminiter, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_parcsrcgnrsetminiter error: ', ierr
      endif

      return
      end

!-------------------------------------------------------------------------
! NALU_HYPRE_ParCSRCGNRSetMaxIter
!-------------------------------------------------------------------------
      subroutine fnalu_hypre_parcsrcgnrsetmaxiter(fsolver, fmaxiter)
 
      integer ierr
      integer fmaxiter
      integer*8 fsolver

      call NALU_HYPRE_ParCSRCGNRSetMaxIter(fsolver, fmaxiter, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_parcsrcgnrsetmaxiter error: ', ierr
      endif

      return
      end

!-------------------------------------------------------------------------
! NALU_HYPRE_ParCSRCGNRSetStopCrit
!-------------------------------------------------------------------------
      subroutine fnalu_hypre_parcsrcgnrsetstopcri(fsolver, fstopcrit)
 
      integer ierr
      integer fstopcrit
      integer*8 fsolver

      call NALU_HYPRE_ParCSRCGNRSetStopCrit(fsolver, fstopcrit, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_parcsrcgnrsetstopcrit error: ', ierr
      endif

      return
      end

!-------------------------------------------------------------------------
! NALU_HYPRE_ParCSRCGNRSetPrecond
!-------------------------------------------------------------------------
      subroutine fnalu_hypre_parcsrcgnrsetprecond(fsolver, fprecond_id, 
     1                                       fprecond)

      integer ierr
      integer fprecond_id
      integer*8 fsolver
      integer*8 fprecond

      call NALU_HYPRE_ParCSRCGNRSetPrecond(fsolver, fprecond_id, fprecond, 
     1                                ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_parcsrcgnrsetprecond error: ', ierr
      endif

      return
      end

!-------------------------------------------------------------------------
! NALU_HYPRE_ParCSRCGNRGetPrecond
!-------------------------------------------------------------------------
      subroutine fnalu_hypre_parcsrcgnrgetprecond(fsolver, fprecond)

      integer ierr
      integer*8 fsolver
      integer*8 fprecond

      call NALU_HYPRE_ParCSRCGNRGetPrecond(fsolver, fprecond, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_parcsrcgnrgetprecond error: ', ierr
      endif

      return
      end

!-------------------------------------------------------------------------
! NALU_HYPRE_ParCSRCGNRSetLogging
!-------------------------------------------------------------------------
      subroutine fnalu_hypre_parcsrcgnrsetlogging(fsolver, flog)

      integer ierr
      integer flog
      integer*8 fsolver

      call NALU_HYPRE_ParCSRCGNRSetLogging(fsolver, flog, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_parcsrcgnrsetlogging error: ', ierr
      endif

      return
      end

!-------------------------------------------------------------------------
! NALU_HYPRE_ParCSRCGNRGetNumIteration
!-------------------------------------------------------------------------
      subroutine fnalu_hypre_parcsrcgnrgetnumiteratio(fsolver, fnumiter)

      integer ierr
      integer fnumiter
      integer*8 fsolver

      call NALU_HYPRE_ParCSRCGNRGetNumIteration(fsolver, fnumiter, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_parcsrcgnrgetnumiterations error: ', ierr
      endif

      return
      end

!-------------------------------------------------------------------------
! NALU_HYPRE_ParCSRCGNRGetFinalRelativeResidualNorm
!-------------------------------------------------------------------------
      subroutine fnalu_hypre_parcsrcgnrgetfinalrelati(fsolver, fnorm)

      integer ierr
      integer*8 fsolver
      double precision fnorm

      call NALU_HYPRE_ParCSRCGNRGetFinalRelativ(fsolver, fnorm, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_parcsrcgnrgetfinalrelativ error: ', ierr
      endif

      return
      end



!-------------------------------------------------------------------------
! NALU_HYPRE_EuclidCreate
!-------------------------------------------------------------------------
      subroutine fnalu_hypre_euclidcreate(fcomm, fsolver)

      integer ierr
      integer fcomm
      integer*8 fsolver

      call NALU_HYPRE_EuclidCreate(fcomm, fsolver, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_euclidcreate error: ', ierr
      endif

      return
      end

!-------------------------------------------------------------------------
! NALU_HYPRE_EuclidDestroy
!-------------------------------------------------------------------------
      subroutine fnalu_hypre_eucliddestroy(fsolver)

      integer ierr
      integer*8 fsolver

      call NALU_HYPRE_EuclidDestroy(fsolver, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_eucliddestroy error: ', ierr
      endif

      return
      end

!-------------------------------------------------------------------------
! NALU_HYPRE_EuclidSetup
!-------------------------------------------------------------------------
      subroutine fnalu_hypre_euclidsetup(fsolver, fA, fb, fx)

      integer ierr
      integer*8 fsolver
      integer*8 fA
      integer*8 fb
      integer*8 fx

      call NALU_HYPRE_EuclidSetup(fsolver, fA, fb, fx, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_euclidsetup error: ', ierr
      endif

      return
      end

!-------------------------------------------------------------------------
! NALU_HYPRE_EuclidSolve
!-------------------------------------------------------------------------
      subroutine fnalu_hypre_euclidsolve(fsolver, fA, fb, fx)

      integer ierr
      integer*8 fsolver
      integer*8 fA
      integer*8 fb
      integer*8 fx

      call NALU_HYPRE_EuclidSolve(fsolver, fA, fb, fx, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_euclidsolve error: ', ierr
      endif

      return
      end

!-------------------------------------------------------------------------
! NALU_HYPRE_EuclidSetParams
!-------------------------------------------------------------------------
      subroutine fnalu_hypre_euclidsetparams(fsolver, fargc, fargv)

      integer ierr
      integer fargc
      integer*8 fsolver
      character*(*) fargv

      call NALU_HYPRE_EuclidSetParams(fsolver, fargc, fargv, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_euclidsetparams error: ', ierr
      endif

      return
      end

!-------------------------------------------------------------------------
! NALU_HYPRE_EuclidSetParamsFromFile
!-------------------------------------------------------------------------
      subroutine fnalu_hypre_euclidsetparamsfromfile(fsolver, ffile)

      integer ierr
      integer*8 fsolver
      character*(*) ffile

      call NALU_HYPRE_EuclidSetParamsFromFile(fsolver, ffile, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_euclidsetparamsfromfile error: ', ierr
      endif

      return
      end



!-------------------------------------------------------------------------
! NALU_HYPRE_ParCSRGMRESCreate
!-------------------------------------------------------------------------
      subroutine fnalu_hypre_parcsrgmrescreate(fcomm, fsolver)

      integer ierr
      integer fcomm
      integer*8 fsolver

      call NALU_HYPRE_ParCSRGMRESCreate(fcomm, fsolver, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_parcsrgmrescreate error: ', ierr
      endif

      return
      end

!-------------------------------------------------------------------------
! NALU_HYPRE_ParCSRGMRESDestroy
!------------------------------------------------------------------------- 
      subroutine fnalu_hypre_parcsrgmresdestroy(fsolver)

      integer ierr
      integer*8 fsolver

      call NALU_HYPRE_ParCSRGMRESDestroy(fsolver)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_parcsrgmresdestroy error: ', ierr
      endif

      return
      end

!-------------------------------------------------------------------------
! NALU_HYPRE_ParCSRGMRESSetup
!------------------------------------------------------------------------- 
      subroutine fnalu_hypre_parcsrgmressetup(fsolver, fA, fb, fx)

      integer ierr
      integer*8 fsolver
      integer*8 fA
      integer*8 fb
      integer*8 fx

      call NALU_HYPRE_ParCSRGMRESSetup(fsolver, fA, fb, fx, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_parcsrgmressetup error: ', ierr
      endif

      return
      end

!-------------------------------------------------------------------------
! NALU_HYPRE_ParCSRGMRESSolve
!-------------------------------------------------------------------------
      subroutine fnalu_hypre_parcsrgmressolve(fsolver, fA, fb, fx)

      integer ierr
      integer*8 fsolver
      integer*8 fA
      integer*8 fb
      integer*8 fx

      call NALU_HYPRE_ParCSRGMRESSolve(fsolver, fA, fb, fx, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_parcsrgmressolve error: ', ierr
      endif

      return
      end

!-------------------------------------------------------------------------
! NALU_HYPRE_ParCSRGMRESSetKDim
!-------------------------------------------------------------------------
      subroutine fnalu_hypre_parcsrgmressetkdim(fsolver, fkdim)

      integer ierr
      integer fkdim
      integer*8 fsolver

      call NALU_HYPRE_ParCSRGMRESSetKDim(fsolver, fkdim, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_parcsrgmressetkdim error: ', ierr
      endif

      return
      end

!-------------------------------------------------------------------------
! NALU_HYPRE_ParCSRGMRESSetTol
!-------------------------------------------------------------------------
      subroutine fnalu_hypre_parcsrgmressettol(fsolver, ftol)

      integer ierr
      integer*8 fsolver
      double precision ftol

      call NALU_HYPRE_ParCSRGMRESSetTol(fsolver, ftol, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_parcsrgmressettol error: ', ierr
      endif

      return
      end

!-------------------------------------------------------------------------
! NALU_HYPRE_ParCSRGMRESSetMinIter
!-------------------------------------------------------------------------
      subroutine fnalu_hypre_parcsrgmressetminiter(fsolver, fminiter)

      integer ierr
      integer fminiter
      integer*8 fsolver

      call NALU_HYPRE_ParCSRGMRESSetMinIter(fsolver, fminiter, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_parcsrgmressetminiter error: ', ierr
      endif

      return
      end

!-------------------------------------------------------------------------
! NALU_HYPRE_ParCSRGMRESSetMaxIter
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_parcsrgmressetmaxiter(fsolver, fmaxiter)

      integer ierr
      integer fmaxiter
      integer*8 fsolver

      call NALU_HYPRE_ParCSRGMRESSetMaxIter(fsolver, fmaxiter, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_parcsrgmressetmaxiter error: ', ierr
      endif

      return
      end

!-------------------------------------------------------------------------
! NALU_HYPRE_ParCSRGMRESSetStopCrit
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_parcsrgmressetstopcrit(fsolver, fstopcrit)

      integer ierr
      integer fstopcrit
      integer*8 fsolver

      call NALU_HYPRE_ParCSRGMRESSetStopCrit(fsolver, fstopcrit, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_parcsrgmressetstopcrit error: ', ierr
      endif

      return
      end

!-------------------------------------------------------------------------
! NALU_HYPRE_ParCSRGMRESSetPrecond
!-------------------------------------------------------------------------
      subroutine fnalu_hypre_parcsrgmressetprecond(fsolver, fprecond_id, 
     1                                        fprecond)

      integer ierr
      integer fprecond_id
      integer*8 fsolver
      integer*8 fprecond

      call NALU_HYPRE_ParCSRGMRESSetPrecond(fsolver, fprecond_id, fprecond,
     1                                 ierr)
     
      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_parcsrgmressetprecond error: ', ierr
      endif

      return
      end

!-------------------------------------------------------------------------
! NALU_HYPRE_ParCSRGMRESGetPrecond
!-------------------------------------------------------------------------
      subroutine fnalu_hypre_parcsrgmresgetprecond(fsolver, fprecond)

      integer ierr
      integer*8 fsolver
      integer*8 fprecond

      call NALU_HYPRE_ParCSRGMRESGetPrecond(fsolver, fprecond, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_parcsrgmresgetprecond error: ', ierr
      endif

      return
      end

!-------------------------------------------------------------------------
! NALU_HYPRE_ParCSRGMRESSetLogging
!-------------------------------------------------------------------------
      subroutine fnalu_hypre_parcsrgmressetlogging(fsolver, flog)

      integer ierr
      integer flog
      integer*8 fsolver

      call NALU_HYPRE_ParCSRGMRESSetLogging(fsolver, flog, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_parcsrgmressetlogging error: ', ierr
      endif

      return
      end

!-------------------------------------------------------------------------
! NALU_HYPRE_ParCSRGMRESSetPrintLevel
!-------------------------------------------------------------------------
      subroutine fnalu_hypre_parcsrgmressetprintlevel(fsolver, fprntlvl)

      integer ierr
      integer fprntlvl
      integer*8 fsolver

      call NALU_HYPRE_ParCSRGMRESSetPrintLevel(fsolver, fprntlvl, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_parcsrgmressetprintlevel error: ', ierr
      endif

      return
      end

!-------------------------------------------------------------------------
! NALU_HYPRE_ParCSRGMRESGetNumIterations
!-------------------------------------------------------------------------
      subroutine fnalu_hypre_parcsrgmresgetnumiterati(fsolver, fnumiter)

      integer ierr
      integer fnumiter
      integer*8 fsolver

      call NALU_HYPRE_ParCSRGMRESGetNumIteratio(fsolver, fnumiter, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_parcsrgmresgetnumiterations error: ', ierr
      endif

      return
      end

!-------------------------------------------------------------------------
! NALU_HYPRE_ParCSRGMRESGetFinalRelativeResidualNorm
!-------------------------------------------------------------------------
      subroutine fnalu_hypre_parcsrgmresgetfinalrelat(fsolver, fnorm)

      integer ierr
      integer*8 fsolver
      double precision fnorm

      call NALU_HYPRE_ParCSRGMRESGetFinalRelati(fsolver, fnorm, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_parcsrgmresgetfinalrelative error: ', ierr
      endif

      return
      end



!-------------------------------------------------------------------------
! NALU_HYPRE_ParCSRHybridCreate
!-------------------------------------------------------------------------
      subroutine fnalu_hypre_parcsrhybridcreate(fsolver)

      integer ierr
      integer*8 fsolver

      call NALU_HYPRE_ParCSRHybridCreate(fsolver, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_parcsrhybridcreate error: ', ierr
      endif

      return
      end

!-------------------------------------------------------------------------
! NALU_HYPRE_ParCSRHybridDestroy
!-------------------------------------------------------------------------
      subroutine fnalu_hypre_parcsrhybriddestroy(fsolver)

      integer ierr
      integer*8 fsolver

      call NALU_HYPRE_ParCSRHybridDestroy(fsolver, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_parcsrhybriddestroy error: ', ierr
      endif

      return
      end

!-------------------------------------------------------------------------
! NALU_HYPRE_ParCSRHybridSetup
!-------------------------------------------------------------------------
      subroutine fnalu_hypre_parcsrhybridsetup(fsolver, fA, fb, fx)

      integer ierr
      integer*8 fsolver
      integer*8 fA
      integer*8 fb
      integer*8 fx

      call NALU_HYPRE_ParCSRHybridSetup(fsolver, fA, fb, fx, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_parcsrhybridsetup error: ', ierr
      endif

      return
      end

!-------------------------------------------------------------------------
! NALU_HYPRE_ParCSRHybridSolve
!-------------------------------------------------------------------------
      subroutine fnalu_hypre_parcsrhybridsolve(fsolver, fA, fb, fx)

      integer ierr
      integer*8 fsolver
      integer*8 fA
      integer*8 fb
      integer*8 fx

      call NALU_HYPRE_ParCSRHybridSolve(fsolver, fA, fb, fx, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_parcsrhybridsolve error: ', ierr
      endif

      return
      end

!-------------------------------------------------------------------------
! NALU_HYPRE_ParCSRHybridSetTol
!-------------------------------------------------------------------------
      subroutine fnalu_hypre_parcsrhybridsettol(fsolver, ftol)

      integer ierr
      integer*8 fsolver
      double precision ftol

      call NALU_HYPRE_ParCSRHybridSetTol(fsolver, ftol, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_parcsrhybridsettol error: ', ierr
      endif

      return
      end

!-------------------------------------------------------------------------
! NALU_HYPRE_ParCSRHybridSetConvergenceTol
!-------------------------------------------------------------------------
      subroutine fnalu_hypre_parcsrhybridsetconvergenc(fsolver, fcftol)

      integer ierr
      integer*8 fsolver
      double precision fcftol

      call NALU_HYPRE_ParCSRHybridSetConvergenc(fsolver, fcftol, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_parcsrhybridsetconvergencetol error: ', ierr
      endif

      return
      end

!-------------------------------------------------------------------------
! NALU_HYPRE_ParCSRHybridSetDSCGMaxIter
!-------------------------------------------------------------------------
      subroutine fnalu_hypre_parcsrhybridsetdscgmaxit(fsolver, fmaxiter)

      integer ierr
      integer fmaxiter
      integer*8 fsolver

      call NALU_HYPRE_ParCSRHybridSetDSCGMaxIte(fsolver, fmaxiter, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_parcsrhybridsetdscgmaxiter error: ', ierr
      endif

      return
      end

!-------------------------------------------------------------------------
! NALU_HYPRE_ParCSRHybridSetPCGMaxIter
!-------------------------------------------------------------------------
      subroutine fnalu_hypre_parcsrhybridsetpcgmaxite(fsolver, fmaxiter)

      integer ierr
      integer fmaxiter
      integer*8 fsolver

      call NALU_HYPRE_ParCSRHybridSetPCGMaxIter(fsolver, fmaxiter, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_parcsrhybridsetpcgmaxiter error: ', ierr
      endif

      return
      end

!-------------------------------------------------------------------------
! NALU_HYPRE_ParCSRHybridSetSolverType
!-------------------------------------------------------------------------
      subroutine fnalu_hypre_parcsrhybridsetsolvertyp(fsolver, ftype)

      integer ierr
      integer ftype
      integer*8 fsolver

      call NALU_HYPRE_ParCSRHybridSetSolverType(fsolver, ftype, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_parcsrhybridsetsolvertype error: ', ierr
      endif

      return
      end

!-------------------------------------------------------------------------
! NALU_HYPRE_ParCSRHybridSetKDim
!-------------------------------------------------------------------------
      subroutine fnalu_hypre_parcsrhybridsetkdim(fsolver, fkdim)

      integer ierr
      integer fkdim
      integer*8 fsolver

      call NALU_HYPRE_ParCSRHybridSetKDim(fsolver, fkdim, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_parcsrhybridsetkdim error: ', ierr
      endif

      return
      end

!-------------------------------------------------------------------------
! NALU_HYPRE_ParCSRHybridSetTwoNorm
!-------------------------------------------------------------------------
      subroutine fnalu_hypre_parcsrhybridsettwonorm(fsolver, f2norm)

      integer ierr
      integer f2norm
      integer*8 fsolver

      call NALU_HYPRE_ParCSRHybridSetTwoNorm(fsolver, f2norm, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_parcsrhybridsettwonorm error: ', ierr
      endif

      return
      end

!-------------------------------------------------------------------------
! NALU_HYPRE_ParCSRHybridSetStopCrit
!-------------------------------------------------------------------------
      subroutine fnalu_hypre_parcsrhybridsetstopcrit(fsolver, fstopcrit)

      integer ierr
      integer fstopcrit
      integer*8 fsolver

      call NALU_HYPRE_ParCSRHybridSetStopCrit(fsolver, fstopcrit, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_parcsrhybridsetstopcrit error: ', ierr
      endif

      return
      end

!-------------------------------------------------------------------------
! NALU_HYPRE_ParCSRHybridSetRelChange
!-------------------------------------------------------------------------
      subroutine fnalu_hypre_parcsrhybridsetrelchange(fsolver, frelchg)

      integer ierr
      integer  frelchg
      integer*8 fsolver

      call NALU_HYPRE_ParCSRHybridSetRelChange(fsolver, frelchg, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_parcsrhybridsetrelchange error: ', ierr
      endif

      return
      end

!-------------------------------------------------------------------------
! NALU_HYPRE_ParCSRHybridSetPrecond
!-------------------------------------------------------------------------
      subroutine fnalu_hypre_parcsrhybridsetprecond(fsolver, fpreid, 
     1                                         fpresolver)

      integer ierr
      integer  fpreid
      integer*8 fsolver
      integer*8 fpresolver

      call NALU_HYPRE_ParCSRHybridSetPrecond(fsolver, fpreid, fpresolver,
     1                                  ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_parcsrhybridsetprecond error: ', ierr
      endif

      return
      end

!-------------------------------------------------------------------------
! NALU_HYPRE_ParCSRHybridSetLogging
!-------------------------------------------------------------------------
      subroutine fnalu_hypre_parcsrhybridsetlogging(fsolver, flogging)

      integer ierr
      integer  flogging
      integer*8 fsolver

      call NALU_HYPRE_ParCSRHybridSetLogging(fsolver, flogging, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_parcsrhybridsetlogging error: ', ierr
      endif

      return
      end

!-------------------------------------------------------------------------
! NALU_HYPRE_ParCSRHybridSetPrintLevel
!-------------------------------------------------------------------------
      subroutine fnalu_hypre_parcsrhybridsetprintlevel(fsolver, fprntlvl)

      integer ierr
      integer  fprntlvl
      integer*8 fsolver

      call NALU_HYPRE_ParCSRHybridSetPrintLevel(fsolver, fprntlvl, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_parcsrhybridsetprintlevel error: ', ierr
      endif

      return
      end

!-------------------------------------------------------------------------
! NALU_HYPRE_ParCSRHybridSetStrongThreshold
!-------------------------------------------------------------------------
      subroutine fnalu_hypre_parcsrhybridsetstrongthr(fsolver, fthresh)

      integer ierr
      integer  fthresh
      integer*8 fsolver

      call NALU_HYPRE_ParCSRHybridSetStrongThre(fsolver, fthresh, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_parcsrhybridsetstrongthreshold error: ', ierr
      endif

      return
      end

!-------------------------------------------------------------------------
! NALU_HYPRE_ParCSRHybridSetMaxRowSum
!-------------------------------------------------------------------------
      subroutine fnalu_hypre_parcsrhybridsetmaxrowsum(fsolver, fsum)

      integer ierr
      integer  fsum
      integer*8 fsolver

      call NALU_HYPRE_ParCSRHybridSetMaxRowSum(fsolver, fsum, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_parcsrhybridsetmaxrowsum error: ', ierr
      endif

      return
      end

!-------------------------------------------------------------------------
! NALU_HYPRE_ParCSRHybridSetTruncFactor
!-------------------------------------------------------------------------
      subroutine fnalu_hypre_parcsrhybridsettruncfact(fsolver, ftfact)

      integer ierr
      integer  ftfact
      integer*8 fsolver

      call NALU_HYPRE_ParCSRHybridSetTruncFacto(fsolver, ftfact, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_parcsrhybridsettruncfactor error: ', ierr
      endif

      return
      end

!-------------------------------------------------------------------------
! NALU_HYPRE_ParCSRHybridSetMaxLevels
!-------------------------------------------------------------------------
      subroutine fnalu_hypre_parcsrhybridsetmaxlevels(fsolver, fmaxlvl)

      integer ierr
      integer  fmaxlvl
      integer*8 fsolver

      call NALU_HYPRE_ParCSRHybridSetMaxLevels(fsolver, fmaxlvl, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_parcsrhybridsetmaxlevels error: ', ierr
      endif

      return
      end

!-------------------------------------------------------------------------
! NALU_HYPRE_ParCSRHybridSetMeasureType
!-------------------------------------------------------------------------
      subroutine fnalu_hypre_parcsrhybridsetmeasurety(fsolver, fmtype)

      integer ierr
      integer  fmtype
      integer*8 fsolver

      call NALU_HYPRE_ParCSRHybridSetMeasureTyp(fsolver, fmtype, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_parcsrhybridsetmeasuretype error: ', ierr
      endif

      return
      end

!-------------------------------------------------------------------------
! NALU_HYPRE_ParCSRHybridSetCoarsenType
!-------------------------------------------------------------------------
      subroutine fnalu_hypre_parcsrhybridsetcoarsenty(fsolver, fcoarse)

      integer ierr
      integer  fcoarse
      integer*8 fsolver

      call NALU_HYPRE_ParCSRHybridSetCoarsenTyp(fsolver, fcoarse, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_parcsrhybridsetcoarsentype error: ', ierr
      endif

      return
      end

!-------------------------------------------------------------------------
! NALU_HYPRE_ParCSRHybridSetCycleType
!-------------------------------------------------------------------------
      subroutine fnalu_hypre_parcsrhybridsetcycletype(fsolver, fcycle)

      integer ierr
      integer  fcycle
      integer*8 fsolver

      call NALU_HYPRE_ParCSRHybridSetCycleType(fsolver, fcycle, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_parcsrhybridsetcycletype error: ', ierr
      endif

      return
      end

!-------------------------------------------------------------------------
! NALU_HYPRE_ParCSRHybridSetNumGridSweeps
!-------------------------------------------------------------------------
      subroutine fnalu_hypre_parcsrhybridsetnumgridsw(fsolver, fsweep)

      integer ierr
      integer  fsweep
      integer*8 fsolver

      call NALU_HYPRE_ParCSRHybridSetNumGridSwe(fsolver, fsweep, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_parcsrhybridsetnumgridsweeps error: ', ierr
      endif

      return
      end

!-------------------------------------------------------------------------
! NALU_HYPRE_ParCSRHybridSetGridRelaxType
!-------------------------------------------------------------------------
      subroutine fnalu_hypre_parcsrhybridsetgridrlxtyp(fsolver, frlxt)

      integer ierr
      integer  frlxt
      integer*8 fsolver

      call NALU_HYPRE_ParCSRHybridSetGridRelaxT(fsolver, frlxt, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_parcsrhybridsetgridrelaxtype error: ', ierr
      endif

      return
      end

!-------------------------------------------------------------------------
! NALU_HYPRE_ParCSRHybridSetGridRelaxPoints
!-------------------------------------------------------------------------
      subroutine fnalu_hypre_parcsrhybridsetgridrlxpts(fsolver, frlxp)

      integer ierr
      integer  frlxp
      integer*8 fsolver

      call NALU_HYPRE_ParCSRHybridSetGridRelaxP(fsolver, frlxp, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_parcsrhybridsetgridrelaxpoints error: ', ierr
      endif

      return
      end

!-------------------------------------------------------------------------
! NALU_HYPRE_ParCSRHybridSetNumSweeps
!-------------------------------------------------------------------------
      subroutine fnalu_hypre_parcsrhybridsetnumsweeps(fsolver, fsweep)

      integer ierr
      integer  fsweep
      integer*8 fsolver

      call NALU_HYPRE_ParCSRHybridSetNumSweeps(fsolver, fsweep, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_parcsrhybridsetnumsweeps error: ', ierr
      endif

      return
      end

!-------------------------------------------------------------------------
! NALU_HYPRE_ParCSRHybridSetCycleNumSweeps
!-------------------------------------------------------------------------
      subroutine fnalu_hypre_parcsrhybridsetcyclenums(fsolver, fsweep)

      integer ierr
      integer  fsweep
      integer*8 fsolver

      call NALU_HYPRE_ParCSRHybridSetCycleNumSw(fsolver, fsweep, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_parcsrhybridsetcyclenumsweeps error: ', ierr
      endif

      return
      end

!-------------------------------------------------------------------------
! NALU_HYPRE_ParCSRHybridSetRelaxType
!-------------------------------------------------------------------------
      subroutine fnalu_hypre_parcsrhybridsetrelaxtype(fsolver, frlxt)

      integer ierr
      integer  frlxt
      integer*8 fsolver

      call NALU_HYPRE_ParCSRHybridSetRelaxType(fsolver, frlxt, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_parcsrhybridsetrelaxtype error: ', ierr
      endif

      return
      end

!-------------------------------------------------------------------------
! NALU_HYPRE_ParCSRHybridSetCycleRelaxType
!-------------------------------------------------------------------------
      subroutine fnalu_hypre_parcsrhybridsetcyclerela(fsolver, frlxt)

      integer ierr
      integer  frlxt
      integer*8 fsolver

      call NALU_HYPRE_ParCSRHybridSetCycleRelax(fsolver, frlxt, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_parcsrhybridsetcyclerelaxtype error: ', ierr
      endif

      return
      end

!-------------------------------------------------------------------------
! NALU_HYPRE_ParCSRHybridSetRelaxOrder
!-------------------------------------------------------------------------
      subroutine fnalu_hypre_parcsrhybridsetrelaxorde(fsolver, frlx)

      integer ierr
      integer  frlx
      integer*8 fsolver

      call NALU_HYPRE_ParCSRHybridSetRelaxOrder(fsolver, frlx, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_parcsrhybridsetrelaxorder error: ', ierr
      endif

      return
      end

!-------------------------------------------------------------------------
! NALU_HYPRE_ParCSRHybridSetRelaxWt
!-------------------------------------------------------------------------
      subroutine fnalu_hypre_parcsrhybridsetrelaxwt(fsolver, frlx)

      integer ierr
      integer  frlx
      integer*8 fsolver

      call NALU_HYPRE_ParCSRHybridSetRelaxWt(fsolver, frlx, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_parcsrhybridsetrelaxwt error: ', ierr
      endif

      return
      end

!-------------------------------------------------------------------------
! NALU_HYPRE_ParCSRHybridSetLevelRelaxWt
!-------------------------------------------------------------------------
      subroutine fnalu_hypre_parcsrhybridsetlevelrela(fsolver, frlx)

      integer ierr
      integer  frlx
      integer*8 fsolver

      call NALU_HYPRE_ParCSRHybridSetLevelRelax(fsolver, frlx, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_parcsrhybridsetlevelrelaxwt error: ', ierr
      endif

      return
      end

!-------------------------------------------------------------------------
! NALU_HYPRE_ParCSRHybridSetOuterWt
!-------------------------------------------------------------------------
      subroutine fnalu_hypre_parcsrhybridsetouterwt(fsolver, fout)

      integer ierr
      integer  fout
      integer*8 fsolver

      call NALU_HYPRE_ParCSRHybridSetOuterWt(fsolver, fout, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_parcsrhybridsetouterwt error: ', ierr
      endif

      return
      end

!-------------------------------------------------------------------------
! NALU_HYPRE_ParCSRHybridSetLevelOuterWt
!-------------------------------------------------------------------------
      subroutine fnalu_hypre_parcsrhybridsetleveloute(fsolver, fout)

      integer ierr
      integer  fout
      integer*8 fsolver

      call NALU_HYPRE_ParCSRHybridSetLevelOuter(fsolver, fout ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_parcsrhybridsetlevelouterwt error: ', ierr
      endif

      return
      end

!-------------------------------------------------------------------------
! NALU_HYPRE_ParCSRHybridSetRelaxWeight
!-------------------------------------------------------------------------
      subroutine fnalu_hypre_parcsrhybridsetrelaxweig(fsolver, frlx)

      integer ierr
      integer  frlx
      integer*8 fsolver

      call NALU_HYPRE_ParCSRHybridSetRelaxWeigh(fsolver, frlx, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_parcsrhybridsetrelaxweight error: ', ierr
      endif

      return
      end

!-------------------------------------------------------------------------
! NALU_HYPRE_ParCSRHybridSetOmega
!-------------------------------------------------------------------------
      subroutine fnalu_hypre_parcsrhybridsetomega(fsolver, fomega)

      integer ierr
      integer  fomega
      integer*8 fsolver

      call NALU_HYPRE_ParCSRHybridSetOmega(fsolver, fomega, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_parcsrhybridsetomega error: ', ierr
      endif

      return
      end

!-------------------------------------------------------------------------
! NALU_HYPRE_ParCSRHybridGetNumIterations
!-------------------------------------------------------------------------
      subroutine fnalu_hypre_parcsrhybridgetnumiterat(fsolver, fiters)

      integer ierr
      integer  fiters
      integer*8 fsolver

      call NALU_HYPRE_ParCSRHybridGetNumIterati(fsolver, fiters, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_parcsrhybridgetnumiterations error: ', ierr
      endif

      return
      end

!-------------------------------------------------------------------------
! NALU_HYPRE_ParCSRHybridGetDSCGNumIterations
!-------------------------------------------------------------------------
      subroutine fnalu_hypre_parcsrhybridgetdscgnumit(fsolver, fiters)

      integer ierr
      integer  fiters
      integer*8 fsolver

      call NALU_HYPRE_ParCSRHybridGetDSCGNumIte(fsolver, fiters, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_parcsrhybridgetdscgnumiterations error: ',
     1                                                    ierr
      endif

      return
      end

!-------------------------------------------------------------------------
! NALU_HYPRE_ParCSRHybridGetPCGNumIterations
!-------------------------------------------------------------------------
      subroutine fnalu_hypre_parcsrhybridgetpcgnumite(fsolver, fiters)

      integer ierr
      integer  fiters
      integer*8 fsolver

      call NALU_HYPRE_ParCSRHybridGetPCGNumIter(fsolver, fiters, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_parcsrhybridgetpcgnumiterations error: ',
     1                                                    ierr
      endif

      return
      end

!-------------------------------------------------------------------------
! NALU_HYPRE_ParCSRHybridGetFinalRelativeResidualNorm
!-------------------------------------------------------------------------
      subroutine fnalu_hypre_parcsrhybridgetfinalrela(fsolver, fnorm)

      integer ierr
      integer  fnorm
      integer*8 fsolver

      call NALU_HYPRE_ParCSRHybridGetFinalRelat(fsolver, 
     1                                     fnorm, ierr)

      if(ierr .ne. 0) then
         print *, 
     1        'fnalu_hypre_parcsrhybridgetfinalrelativeresidualnorm error: ',
     1                                                    ierr
      endif

      return
      end



!-------------------------------------------------------------------------
! NALU_HYPRE_ParSetRandomValues
!-------------------------------------------------------------------------
      subroutine fnalu_hypre_parsetrandomvalues(fv, fseed)

      integer ierr
      integer fseed
      integer*8 fv

      call NALU_HYPRE_ParVectorSetRandomValues(fv, fseed, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_parsetrandomvalues error: ', ierr
      endif

      return
      end

!-------------------------------------------------------------------------
! NALU_HYPRE_ParPrintVector
!-------------------------------------------------------------------------
      subroutine fnalu_hypre_parprintvector(fv, ffile)

      integer ierr
      integer*8 fv
      character*(*) ffile

      call nalu_hypre_ParVectorPrint(fv, ffile, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_parprintvector error: ', ierr
      endif

      return
      end

!-------------------------------------------------------------------------
! NALU_HYPRE_ParReadVector
!-------------------------------------------------------------------------
      subroutine fnalu_hypre_parreadvector(fcomm, ffile)

      integer ierr
      integer fcomm
      character*(*) ffile

      call nalu_hypre_ParReadVector(fcomm, ffile, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_parreadvector error: ', ierr
      endif

      return
      end

!-------------------------------------------------------------------------
! NALU_HYPRE_ParVectorSize
!-------------------------------------------------------------------------
      subroutine fnalu_hypre_parvectorsize(fx)

      integer ierr
      integer*8 fx

      call nalu_hypre_ParVectorSize(fx, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_parvectorsize error: ', ierr
      endif

      return
      end

!-------------------------------------------------------------------------
! NALU_HYPRE_ParCSRMultiVectorPrint
!-------------------------------------------------------------------------
      subroutine fnalu_hypre_parcsrmultivectorprint(fx, ffile)

      integer ierr
      integer*8 fx
      character*(*) ffile

      call nalu_hypre_ParCSRMultiVectorPrint(fx, ffile, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_parcsrmultivectorprint error: ', ierr
      endif

      return
      end

!-------------------------------------------------------------------------
! NALU_HYPRE_ParCSRMultiVectorRead
!-------------------------------------------------------------------------
      subroutine fnalu_hypre_parcsrmultivectorread(fcomm, fii, ffile)

      integer ierr
      integer fcomm
      integer*8 fii
      character*(*) ffile

      call nalu_hypre_ParCSRMultiVectorRead(fcomm, fii, ffile, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_parcsrmultivectorread error: ', ierr
      endif

      return
      end

!-------------------------------------------------------------------------
! aux_maskCount
!-------------------------------------------------------------------------
      subroutine fnalu_hypre_aux_maskcount(fn, fmask)

      integer ierr
      integer fn
      integer fmask

      call aux_maskCount(fn, fmask, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_aux_maskcount error: ', ierr
      endif

      return
      end

!-------------------------------------------------------------------------
! aux_indexFromMask
!-------------------------------------------------------------------------
      subroutine fnalu_hypre_auxindexfrommask(fn, fmask, findex)

      integer ierr
      integer fn
      integer fmask
      integer findex

      call aux_indexFromMask(fn, fmask, findex, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_aux_indexfrommask error: ', ierr
      endif

      return
      end

!-------------------------------------------------------------------------
! NALU_HYPRE_TempParCSRSetupInterpreter
!-------------------------------------------------------------------------
      subroutine fnalu_hypre_tempparcsrsetupinterpret(fi)

      integer ierr
      integer*8 fi

      call NALU_HYPRE_TempParCSRSetupInterprete(fi, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_tempparcsrsetupinterpreter error: ', ierr
      endif

      return
      end

!-------------------------------------------------------------------------
! NALU_HYPRE_ParCSRSetupInterpreter
!-------------------------------------------------------------------------
      subroutine fnalu_hypre_parcsrsetupinterpreter(fi)

      integer ierr
      integer*8 fi

      call NALU_HYPRE_ParCSRSetupInterpreter(fi, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_parcsrsetupinterpreter error: ', ierr
      endif

      return
      end

!-------------------------------------------------------------------------
! NALU_HYPRE_ParCSRSetupMatvec
!-------------------------------------------------------------------------
      subroutine fnalu_hypre_parcsrsetupmatvec(fmv)

      integer ierr
      integer*8 fmv

      call NALU_HYPRE_ParCSRSetupMatvec(fmv, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_parcsrsetupmatvec error: ', ierr
      endif

      return
      end



!-------------------------------------------------------------------------
! NALU_HYPRE_ParaSailsCreate
!-------------------------------------------------------------------------
      subroutine fnalu_hypre_parasailscreate(fcomm, fsolver)

      integer ierr
      integer fcomm
      integer*8 fsolver

      call NALU_HYPRE_ParaSailsCreate(fcomm, fsolver, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_parasailscreate error: ', ierr
      endif

      return
      end

!-------------------------------------------------------------------------
! NALU_HYPRE_ParaSailsDestroy
!-------------------------------------------------------------------------
      subroutine fnalu_hypre_parasailsdestroy(fsolver)

      integer ierr
      integer*8 fsolver

      call NALU_HYPRE_ParaSailsDestroy(fsolver, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_parasailsdestroy error: ', ierr
      endif

      return
      end

!-------------------------------------------------------------------------
! NALU_HYPRE_ParaSailsSetup
!------------------------------------------------------------------------- 
      subroutine fnalu_hypre_parasailssetup(fsolver, fA, fb, fx)

      integer ierr
      integer*8 fsolver
      integer*8 fA
      integer*8 fb
      integer*8 fx

      call NALU_HYPRE_ParaSailsSetup(fsolver, fA, fb, fx, ierr) 

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_parasailssetup error: ', ierr
      endif

      return
      end

!-------------------------------------------------------------------------
! NALU_HYPRE_ParaSailsSolve
!------------------------------------------------------------------------- 
      subroutine fnalu_hypre_parasailssolve(fsolver, fA, fb, fx)

      integer ierr
      integer*8 fsolver
      integer*8 fA
      integer*8 fb
      integer*8 fx

      call NALU_HYPRE_ParaSailsSolve(fsolver, fA, fb, fx, ierr) 

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_parasailssolve error: ', ierr
      endif

      return
      end

!-------------------------------------------------------------------------
! NALU_HYPRE_ParaSailsSetParams
!-------------------------------------------------------------------------
      subroutine fnalu_hypre_parasailssetparams(fsolver, fthresh, fnlevels)

      integer ierr
      integer fnlevels
      integer*8 fsolver
      double precision fthresh

      call NALU_HYPRE_ParaSailsSetParams(fsolver, fthresh, fnlevels, ierr) 

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_parasailssetparams error: ', ierr
      endif

      return
      end

!-------------------------------------------------------------------------
! NALU_HYPRE_ParaSailsSetThresh
!-------------------------------------------------------------------------
      subroutine fnalu_hypre_parasailssetthresh(fsolver, fthresh)

      integer ierr
      integer*8 fsolver
      double precision fthresh

      call NALU_HYPRE_ParaSailsSetThresh(fsolver, fthresh, ierr) 

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_parasailssetthresh error: ', ierr
      endif

      return
      end

!-------------------------------------------------------------------------
! NALU_HYPRE_ParaSailsGetThresh
!-------------------------------------------------------------------------
      subroutine fnalu_hypre_parasailsgetthresh(fsolver, fthresh)

      integer ierr
      integer*8 fsolver
      double precision fthresh

      call NALU_HYPRE_ParaSailsGetThresh(fsolver, fthresh, ierr) 

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_parasailsgetthresh error: ', ierr
      endif

      return
      end

!-------------------------------------------------------------------------
! NALU_HYPRE_ParaSailsSetNlevels
!-------------------------------------------------------------------------
      subroutine fnalu_hypre_parasailssetnlevels(fsolver, fnlevels)

      integer ierr
      integer fnlevels
      integer*8 fsolver

      call NALU_HYPRE_ParaSailsSetNlevels(fsolver, fnlevels, ierr) 

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_parasailssetnlevels error: ', ierr
      endif

      return
      end

!-------------------------------------------------------------------------
! NALU_HYPRE_ParaSailsGetNlevels
!-------------------------------------------------------------------------
      subroutine fnalu_hypre_parasailsgetnlevels(fsolver, fnlevels)

      integer ierr
      integer fnlevels
      integer*8 fsolver

      call NALU_HYPRE_ParaSailsGetNlevels(fsolver, fnlevels, ierr) 

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_parasailsgetnlevels error: ', ierr
      endif

      return
      end

!-------------------------------------------------------------------------
! NALU_HYPRE_ParaSailsSetFilter
!-------------------------------------------------------------------------
      subroutine fnalu_hypre_parasailssetfilter(fsolver, ffilter)

      integer ierr
      integer*8 fsolver
      double precision ffilter

      call NALU_HYPRE_ParaSailsSetFilter(fsolver, ffilter, ierr) 

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_parasailssetfilter error: ', ierr
      endif

      return
      end

!-------------------------------------------------------------------------
! NALU_HYPRE_ParaSailsGetFilter
!-------------------------------------------------------------------------
      subroutine fnalu_hypre_parasailsgetfilter(fsolver, ffilter)

      integer ierr
      integer*8 fsolver
      double precision ffilter

      call NALU_HYPRE_ParaSailsGetFilter(fsolver, ffilter, ierr) 

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_parasailsgetfilter error: ', ierr
      endif

      return
      end

!-------------------------------------------------------------------------
! NALU_HYPRE_ParaSailsSetSym
!-------------------------------------------------------------------------
      subroutine fnalu_hypre_parasailssetsym(fsolver, fsym)

      integer ierr
      integer fsym
      integer*8 fsolver

      call NALU_HYPRE_ParaSailsSetSym(fsolver, fsym, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_parasailssetsym error: ', ierr
      endif

      return
      end

!-------------------------------------------------------------------------
! NALU_HYPRE_ParaSailsGetSym
!-------------------------------------------------------------------------
      subroutine fnalu_hypre_parasailsgetsym(fsolver, fsym)

      integer ierr
      integer fsym
      integer*8 fsolver

      call NALU_HYPRE_ParaSailsGetSym(fsolver, fsym, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_parasailsgetsym error: ', ierr
      endif

      return
      end

!-------------------------------------------------------------------------
! NALU_HYPRE_ParaSailsSetLoadbal
!-------------------------------------------------------------------------
      subroutine fnalu_hypre_parasailssetloadbal(fsolver, floadbal)

      integer ierr
      integer*8 fsolver
      double precision floadbal

      call NALU_HYPRE_ParaSailsSetLoadbal(fsolver, floadbal, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_parasailssetloadbal error: ', ierr
      endif

      return
      end

!-------------------------------------------------------------------------
! NALU_HYPRE_ParaSailsGetLoadbal
!-------------------------------------------------------------------------
      subroutine fnalu_hypre_parasailsgetloadbal(fsolver, floadbal)

      integer ierr
      integer*8 fsolver
      double precision floadbal

      call NALU_HYPRE_ParaSailsGetLoadbal(fsolver, floadbal, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_parasailsgetloadbal error: ', ierr
      endif

      return
      end

!-------------------------------------------------------------------------
! NALU_HYPRE_ParaSailsSetReuse
!-------------------------------------------------------------------------
      subroutine fnalu_hypre_parasailssetreuse(fsolver, freuse)

      integer ierr
      integer freuse
      integer*8 fsolver

      call NALU_HYPRE_ParaSailsSetReuse(fsolver, freuse, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_parasailssetreuse error: ', ierr
      endif

      return
      end

!-------------------------------------------------------------------------
! NALU_HYPRE_ParaSailsGetReuse
!-------------------------------------------------------------------------
      subroutine fnalu_hypre_parasailsgetreuse(fsolver, freuse)

      integer ierr
      integer freuse
      integer*8 fsolver

      call NALU_HYPRE_ParaSailsGetReuse(fsolver, freuse, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_parasailsgetreuse error: ', ierr
      endif

      return
      end

!-------------------------------------------------------------------------
! NALU_HYPRE_ParaSailsSetLogging
!-------------------------------------------------------------------------
      subroutine fnalu_hypre_parasailssetlogging(fsolver, flog)

      integer ierr
      integer flog
      integer*8 fsolver

      call NALU_HYPRE_ParaSailsSetLogging(fsolver, flog, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_parasailssetlogging error: ', ierr
      endif
      
      return
      end

!-------------------------------------------------------------------------
! NALU_HYPRE_ParaSailsGetLogging
!-------------------------------------------------------------------------
      subroutine fnalu_hypre_parasailsgetlogging(fsolver, flog)

      integer ierr
      integer flog
      integer*8 fsolver

      call NALU_HYPRE_ParaSailsGetLogging(fsolver, flog, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_parasailsgetlogging error: ', ierr
      endif
      
      return
      end



!-------------------------------------------------------------------------
! NALU_HYPRE_ParCSRPCGCreate
!-------------------------------------------------------------------------
      subroutine fnalu_hypre_parcsrpcgcreate(fcomm, fsolver)

      integer ierr
      integer fcomm
      integer*8 fsolver

      call NALU_HYPRE_ParCSRPCGCreate(fcomm, fsolver, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_parcsrpcgcreate error: ', ierr
      endif

      return
      end

!-------------------------------------------------------------------------
! NALU_HYPRE_ParCSRPCGDestroy
!-------------------------------------------------------------------------
      subroutine fnalu_hypre_parcsrpcgdestroy(fsolver)

      integer ierr
      integer*8 fsolver

      call NALU_HYPRE_ParCSRPCGDestroy(fsolver)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_parcsrpcgdestroy error: ', ierr
      endif

      return
      end

!-------------------------------------------------------------------------
! NALU_HYPRE_ParCSRPCGSetup
!-------------------------------------------------------------------------
      subroutine fnalu_hypre_parcsrpcgsetup(fsolver, fA, fb, fx)

      integer ierr
      integer*8 fsolver
      integer*8 fA
      integer*8 fb
      integer*8 fx

      call NALU_HYPRE_ParCSRPCGSetup(fsolver, fA, fb, fx, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_parcsrpcgcreate error: ', ierr
      endif

      return
      end

!-------------------------------------------------------------------------
! NALU_HYPRE_ParCSRPCGSolve
!-------------------------------------------------------------------------
      subroutine fnalu_hypre_parcsrpcgsolve(fsolver, fA, fb, fx)

      integer ierr
      integer*8 fsolver
      integer*8 fA
      integer*8 fb
      integer*8 fx

      call NALU_HYPRE_ParCSRPCGSolve(fsolver, fA, fb, fx, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_parcsrpcgsolve error: ', ierr
      endif

      return
      end

!-------------------------------------------------------------------------
! NALU_HYPRE_ParCSRPCGSetTol
!-------------------------------------------------------------------------
      subroutine fnalu_hypre_parcsrpcgsettol(fsolver, ftol)

      integer ierr
      integer*8 fsolver
      double precision ftol

      call NALU_HYPRE_ParCSRPCGSetTol(fsolver, ftol, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_parcsrpcgsettol error: ', ierr
      endif

      return
      end

!-------------------------------------------------------------------------
! NALU_HYPRE_ParCSRPCGSetMaxIter
!-------------------------------------------------------------------------
      subroutine fnalu_hypre_parcsrpcgsetmaxiter(fsolver, fmaxiter)

      integer ierr
      integer fmaxiter
      integer*8 fsolver

      call NALU_HYPRE_ParCSRPCGSetMaxIter(fsolver, fmaxiter, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_parcsrpcgsetmaxiter error: ', ierr
      endif

      return
      end

!-------------------------------------------------------------------------
! NALU_HYPRE_ParCSRPCGSetStopCrit
!-------------------------------------------------------------------------
      subroutine fnalu_hypre_parcsrpcgsetstopcrit(fsolver, ftol)

      integer ierr
      integer*8 fsolver
      double precision ftol

      call NALU_HYPRE_ParCSRPCGSetStopCrit(fsolver, ftol, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_parcsrpcgsetstopcrit error: ', ierr
      endif

      return
      end

!-------------------------------------------------------------------------
! NALU_HYPRE_ParCSRPCGSetTwoNorm
!-------------------------------------------------------------------------
      subroutine fnalu_hypre_parcsrpcgsettwonorm(fsolver, ftwonorm)

      integer ierr
      integer ftwonorm
      integer*8 fsolver

      call NALU_HYPRE_ParCSRPCGSetTwoNorm(fsolver, ftwonorm, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_parcsrpcgsettwonorm error: ', ierr
      endif

      return
      end

!-------------------------------------------------------------------------
! NALU_HYPRE_ParCSRPCGSetRelChange
!-------------------------------------------------------------------------
      subroutine fnalu_hypre_parcsrpcgsetrelchange(fsolver, frelchange)

      integer ierr
      integer frelchange
      integer*8 fsolver

      call NALU_HYPRE_ParCSRPCGSetRelChange(fsolver, frelchange, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_parcsrpcgsetrelchange error: ', ierr
      endif

      return
      end

!-------------------------------------------------------------------------
! NALU_HYPRE_ParCSRPCGSetPrecond
!-------------------------------------------------------------------------
      subroutine fnalu_hypre_parcsrpcgsetprecond(fsolver, fprecond_id, 
     1                                      fprecond)

      integer ierr
      integer fprecond_id
      integer*8 fsolver
      integer*8 fprecond

      call NALU_HYPRE_ParCSRPCGSetPrecond(fsolver, fprecond_id, fprecond, 
     1                              ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_parcsrpcgsetprecond error: ', ierr
      endif

      return
      end

!-------------------------------------------------------------------------
! NALU_HYPRE_ParCSRPCGGetPrecond
!-------------------------------------------------------------------------
      subroutine fnalu_hypre_parcsrpcggetprecond(fsolver, fprecond)

      integer ierr
      integer*8 fsolver
      integer*8 fprecond

      call NALU_HYPRE_ParCSRPCGGetPrecond(fsolver, fprecond, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_parcsrpcggetprecond error: ', ierr
      endif

      return
      end

!-------------------------------------------------------------------------
! NALU_HYPRE_ParCSRPCGSetPrintLevel
!-------------------------------------------------------------------------
      subroutine fnalu_hypre_parcsrpcgsetprintlevel(fsolver, flevel)

      integer ierr
      integer flevel
      integer*8 fsolver

      call NALU_HYPRE_ParCSRPCGSetPrintLevel(fsolver, flevel, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_parcsrpcgsetprintlevel error: ', ierr
      endif

      return
      end

!-------------------------------------------------------------------------
! NALU_HYPRE_ParCSRPCGGetNumIterations
!-------------------------------------------------------------------------
      subroutine fnalu_hypre_parcsrpcggetnumiteration(fsolver, fnumiter)

      integer ierr
      integer fnumiter
      integer*8 fsolver

      call NALU_HYPRE_ParCSRPCGGetNumIterations(fsolver, fnumiter, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_parcsrpcggetnumiteration error: ', ierr
      endif

      return
      end

!-------------------------------------------------------------------------
! NALU_HYPRE_ParCSRPCGGetFinalRelativeResidualNorm
!-------------------------------------------------------------------------
      subroutine fnalu_hypre_parcsrpcggetfinalrelativ(fsolver, fnorm)

      integer ierr
      integer*8 fsolver
      double precision fnorm

      call NALU_HYPRE_ParCSRPCGGetFinalRelative(fsolver, fnorm, ierr)
      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_parcsrpcggetfinalrelative error: ', ierr
      endif

      return
      end

!-------------------------------------------------------------------------
! NALU_HYPRE_ParCSRDiagScaleSetup
!-------------------------------------------------------------------------
      subroutine fnalu_hypre_parcsrdiagscalesetup(fsolver, fA, fb, fx)

      integer ierr
      integer*8 fsolver
      integer*8 fA
      integer*8 fb
      integer*8 fx

      call NALU_HYPRE_ParCSRDiagScaleSetup(fsolver, fA, fb, fx, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_parcsrdiagscalesetup error: ', ierr
      endif

      return
      end

!-------------------------------------------------------------------------
! NALU_HYPRE_ParCSRDiagScale
!-------------------------------------------------------------------------
      subroutine fnalu_hypre_parcsrdiagscale(fsolver, fHA, fHy, fHx)

      integer ierr
      integer*8 fsolver
      integer*8 fHA
      integer*8 fHy
      integer*8 fHx

      call NALU_HYPRE_ParCSRDiagScale(fsolver, fHA, fHy, fHx, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_parcsrdiagscale error: ', ierr
      endif

      return
      end



!-------------------------------------------------------------------------
! NALU_HYPRE_ParCSRPilutCreate
!-------------------------------------------------------------------------
      subroutine fnalu_hypre_parcsrpilutcreate(fcomm, fsolver)

      integer ierr
      integer fcomm
      integer*8 fsolver

      call NALU_HYPRE_ParCSRPilutCreate(fcomm, fsolver, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_parcsrpilutcreate error: ', ierr
      endif

      return
      end

!-------------------------------------------------------------------------
! NALU_HYPRE_ParCSRPilutDestroy
!-------------------------------------------------------------------------
      subroutine fnalu_hypre_parcsrpilutdestroy(fsolver)

      integer ierr
      integer*8 fsolver

      call NALU_HYPRE_ParCSRPilutDestroy(fsolver, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_parcsrpilutdestroy error: ', ierr
      endif

      return
      end

!-------------------------------------------------------------------------
! NALU_HYPRE_ParCSRPilutSetup
!-------------------------------------------------------------------------
      subroutine fnalu_hypre_parcsrpilutsetup(fsolver, fA, fb, fx)

      integer ierr
      integer*8 fsolver
      integer*8 fA
      integer*8 fb
      integer*8 fx

      call NALU_HYPRE_ParCSRPilutSetup(fsolver, fA, fb, fx, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_parcsrpilutsetup error: ', ierr
      endif

      return
      end

!-------------------------------------------------------------------------
! NALU_HYPRE_ParCSRPilutSolve
!------------------------------------------------------------------------- 
      subroutine fnalu_hypre_parcsrpilutsolve(fsolver, fA, fb, fx)

      integer ierr
      integer*8 fsolver
      integer*8 fA
      integer*8 fb
      integer*8 fx

      call NALU_HYPRE_ParCSRPilutSolve(fsolver, fA, fb, fx, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_parcsrpilutsolve error: ', ierr
      endif

      return
      end

!-------------------------------------------------------------------------
! NALU_HYPRE_ParCSRPilutSetMaxIter
!-------------------------------------------------------------------------
      subroutine fnalu_hypre_parcsrpilutsetmaxiter(fsolver, fmaxiter)

      integer ierr
      integer fmaxiter
      integer*8 fsolver

      call NALU_HYPRE_ParCSRPilutSetMaxIter(fsolver, fmaxiter, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_parcsrpilutsetmaxiter error: ', ierr
      endif

      return
      end

!-------------------------------------------------------------------------
! NALU_HYPRE_ParCSRPilutSetDropToleran
!-------------------------------------------------------------------------
      subroutine fnalu_hypre_parcsrpilutsetdroptolera(fsolver, ftol)

      integer ierr
      integer*8 fsolver
      double precision ftol

      call NALU_HYPRE_ParCSRPilutSetDropToleran(fsolver, ftol, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_parcsrpilutsetdroptol error: ', ierr
      endif

      return
      end

!--------------------------------------------------------------------------
! NALU_HYPRE_ParCSRPilutSetFacRowSize
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_parcsrpilutsetfacrowsize(fsolver, fsize)
      
      integer ierr
      integer fsize
      integer*8 fsolver

      call NALU_HYPRE_ParCSRPilutSetFacRowSize(fsolver, fsize, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_parcsrpilutsetfacrowsize error: ', ierr
      endif

      return
      end



!-------------------------------------------------------------------------
! NALU_HYPRE_SchwarzCreate
!-------------------------------------------------------------------------
      subroutine fnalu_hypre_schwarzcreate(fsolver)

      integer ierr
      integer*8 fsolver

      call NALU_HYPRE_SchwarzCreate(fsolver, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_schwarzcreate error: ', ierr
      endif

      return
      end

!-------------------------------------------------------------------------
! NALU_HYPRE_SchwarzDestroy
!-------------------------------------------------------------------------
      subroutine fnalu_hypre_schwarzdestroy(fsolver)

      integer ierr
      integer*8 fsolver

      call NALU_HYPRE_SchwarzDestroy(fsolver, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_schwarzdestroy error: ', ierr
      endif

      return
      end

!-------------------------------------------------------------------------
! NALU_HYPRE_SchwarzSetup
!-------------------------------------------------------------------------
      subroutine fnalu_hypre_schwarzsetup(fsolver, fA, fb, fx)

      integer ierr
      integer*8 fsolver
      integer*8 fA
      integer*8 fb
      integer*8 fx

      call NALU_HYPRE_SchwarzSetup(fsolver, fA, fb, fx, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_schwarzsetup error: ', ierr
      endif

      return
      end

!-------------------------------------------------------------------------
! NALU_HYPRE_SchwarzSolve
!------------------------------------------------------------------------- 
      subroutine fnalu_hypre_schwarzsolve(fsolver, fA, fb, fx)

      integer ierr
      integer*8 fsolver
      integer*8 fA
      integer*8 fb
      integer*8 fx

      call NALU_HYPRE_SchwarzSolve(fsolver, fA, fb, fx, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_schwarzsolve error: ', ierr
      endif

      return
      end

!-------------------------------------------------------------------------
! NALU_HYPRE_SchwarzSetVariant
!------------------------------------------------------------------------- 
      subroutine fnalu_hypre_schwarzsetvariant(fsolver, fvariant)

      integer ierr
      integer fvariant
      integer*8 fsolver

      call NALU_HYPRE_SchwarzSetVariant(fsolver, fvariant, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_schwarzsetvariant error: ', ierr
      endif

      return
      end

!-------------------------------------------------------------------------
! NALU_HYPRE_SchwarzSetOverlap
!------------------------------------------------------------------------- 
      subroutine fnalu_hypre_schwarzsetoverlap(fsolver, foverlap)

      integer ierr
      integer foverlap
      integer*8 fsolver

      call NALU_HYPRE_SchwarzSetOverlap(fsolver, foverlap, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_schwarzsetoverlap error: ', ierr
      endif

      return
      end

!-------------------------------------------------------------------------
! NALU_HYPRE_SchwarzSetDomainType
!------------------------------------------------------------------------- 
      subroutine fnalu_hypre_schwarzsetdomaintype(fsolver, fdomaint)

      integer ierr
      integer fdomaint
      integer*8 fsolver

      call NALU_HYPRE_SchwarzSetDomainType(fsolver, fdomaint, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_schwarzsetdomaintype error: ', ierr
      endif

      return
      end

!-------------------------------------------------------------------------
! NALU_HYPRE_SchwarzSetDomainStructure
!------------------------------------------------------------------------- 
      subroutine fnalu_hypre_schwarzsetdomainstructur(fsolver, fdomains)

      integer ierr
      integer fdomains
      integer*8 fsolver

      call NALU_HYPRE_SchwarzSetDomainStructure(fsolver, fdomains, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_schwarzsetdomainstructure error: ', ierr
      endif

      return
      end

!-------------------------------------------------------------------------
! NALU_HYPRE_SchwarzSetNumFunctions
!------------------------------------------------------------------------- 
      subroutine fnalu_hypre_schwarzsetnumfunctions(fsolver, fnumfncs)

      integer ierr
      integer fnumfncs
      integer*8 fsolver

      call NALU_HYPRE_SchwarzSetNumFunctions(fsolver, fnumfncs, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_schwarzsetnumfunctions error: ', ierr
      endif

      return
      end

!-------------------------------------------------------------------------
! NALU_HYPRE_SchwarzSetRelaxWeight
!------------------------------------------------------------------------- 
      subroutine fnalu_hypre_schwarzsetrelaxweight(fsolver, frlxwt)

      integer ierr
      integer*8 fsolver
      double precision frlxwt

      call NALU_HYPRE_SchwarzSetRelaxWeight(fsolver, frlxwt, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_schwarzsetrelaxweight error: ', ierr
      endif

      return
      end

!-------------------------------------------------------------------------
! NALU_HYPRE_SchwarzSetDofFunc
!------------------------------------------------------------------------- 
      subroutine fnalu_hypre_schwarzsetdoffunc(fsolver, fdofnc)

      integer ierr
      integer fdofnc
      integer*8 fsolver

      call NALU_HYPRE_SchwarzSetDofFunc(fsolver, fdofnc, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_schwarzsetdoffunc error: ', ierr
      endif

      return
      end
