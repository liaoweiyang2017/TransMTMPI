!===============================================================================
! `mt1dfwd` calculates MT response over 1-D layered earth.
!
!===============================================================================
subroutine mt1dfwd(nlayer, resistivity, thickness, nfreq, mtFreq, mtPredData)

	implicit none

	integer :: nlayer, nfreq
	real(8),dimension(nlayer) :: resistivity, thickness
	real(8),dimension(nfreq):: mtFreq
	real(8),dimension(2*nfreq):: mtPredData

	! local variables
	integer :: i, j, k
	real(8) :: muomega, zMTr, zMTi
	complex(8) :: omi, dj, wj, rj, re
	complex(8), dimension(nlayer) :: layerimpedance
	real(8),dimension(nfreq) :: appRes, appPhs
	real(8)    :: pi = 3.14159265358979323846264d0
	real(8)    :: mu0
	complex(8) :: ii = (0d0,1d0)

	!
	k = 1
	mu0 = 4d-7 * pi
	do i = 1, nfreq
		muomega = 2.0 * pi * mtFreq(i) * mu0
		omi = muomega * ii
		j   = nlayer
		layerimpedance(j) = sqrt(omi * resistivity(j))

		do j = nlayer-1, 1, -1
			dj = sqrt(omi / resistivity(j) )
			wj = dj * resistivity(j)
			rj = (wj - layerimpedance(j+1)) / (wj + layerimpedance(j+1) )
			re = rj * exp( - 2.0 * thickness(j) * dj )
			layerimpedance(j) = wj * ((1.0 - re) / (1.0 + re))
		end do

		! get impedance at surface
		zMTr = real(layerimpedance(1))
		zMTi = aimag(layerimpedance(1))

		! get apparent resistivity and phase
		!appRes(i) = abs(layerimpedance(1)) ** 2 / muomega
		!appPhs(i) = atan2(zMTi, zMTr) * 180.0 / pi
		mtPredData(k)   = zMTr
		mtPredData(k+1) = zMTi
		k = k + 2
	end do

end subroutine
