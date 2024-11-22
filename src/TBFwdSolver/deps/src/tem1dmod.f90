!===============================================================================
! module tem1dmod calculates TEM response over 1-D layered earth.
!
!===============================================================================
module tem1dmod

    implicit none

	! math constants
	integer, parameter         :: drPrec = kind(0d0)
    real(drPrec), parameter    :: pi = 3.14159265358979323846264d0
    real(drPrec), parameter    :: mu0 = 4d-7 * pi
    complex(drPrec), parameter :: ii = (0d0,1d0)

    ! TEM survey configuration
    !integer :: nTimePoint, ntemData
    real(drPrec) :: xLength, yLength, recx, recy
    !real(drPrec),dimension(:),allocatable :: timePoint
    !real(drPrec),dimension(:),allocatable :: temResp

    ! frequency range used for transformation
    integer :: ntemFreq
    real(drPrec) :: bfreq, efreq, freqDecimal
    real(drPrec),dimension(:),allocatable ::temFreq
    complex(drPrec),dimension(:),allocatable ::temfreqResp

    ! Gauss integration
    integer :: nGauss, nGaussHalf
    real(drPrec),dimension(:),allocatable :: bp, wf
    real(drPrec),dimension(:),allocatable :: bpHalf, wfHalf

    ! Hankel transformation
    integer :: orderHankel, nHankel
    real(drPrec) :: wy, s
    real(drPrec), dimension(:),allocatable :: xsHankel

    ! loop integration
    integer :: nxLen, nyLen  ! number of interval for each side
    real(drPrec),dimension(:),allocatable :: intx, inty

    ! sine or cosine transformation
    integer :: nTransFilter

    contains

    !===========================================================================
    ! `compFreqResponse` calculates response in frequency domain
    !
    !===========================================================================
    subroutine compFreqResponse(nlayer, rho, thickness)

        integer :: nlayer
        real(drPrec),dimension(nlayer) :: rho, thickness

        ! local variables
        integer :: m, xm, yn, n
        real(drPrec) :: freqnow, B, dconst
        complex(drPrec) :: Bx1, Bx2, By1, By2
        real(drPrec),dimension(nGauss)    :: x, y, R1, R2, lamb1, lamb2
        complex(drPrec),dimension(nGauss) :: rte1, rte2, tx1, tx2, ty1, ty2

        dconst = 1d-7

        ! edge at x-direction
        do m = 1, ntemFreq
            freqnow = temFreq(m)
            Bx1 = 0d0
            Bx2 = 0d0
            do xm = 1, size(intx,1)-1, 1
                x = 0.5 * (intx(xm+1)-intx(xm)) * bp + 0.5 * (intx(xm+1) + intx(xm))
                do n = 1, size(xsHankel), 1
                    B  = 10**(wy + (n-1) * s)
                    R1 = sqrt((x - recx)**2.0 + (recy + yLength / 2)**2.0)
                    R2 = sqrt((x - recx)**2.0 + (recy - yLength / 2)**2.0)
                    lamb1 = B / R1
                    lamb2 = B / R2
                    call getReflectTE(lamb1, freqnow, rho, thickness, nlayer, nGauss, rte1)
                    call getReflectTE(lamb2, freqnow, rho, thickness, nlayer, nGauss, rte2)
                    tx1 = ((recy + yLength/2.0) / R1**2.0) *(lamb1 * (1+rte1))
                    tx2 = ((recy - yLength/2.0) / R2**2.0) *(lamb2 * (1+rte2))
                    Bx1 = Bx1 + 0.5*dconst*(intx(xm+1)-intx(xm)) * xsHankel(n) * sum(tx1*wf)
                    Bx2 = Bx2 - 0.5*dconst*(intx(xm+1)-intx(xm)) * xsHankel(n) * sum(tx2*wf)
                end do
            end do

            ! edge at y-direction
            By1 = 0d0
            By2 = 0d0
            do yn = 1, size(inty,1)-1, 1
                y = 0.5*(inty(yn+1) - inty(yn)) * bp + 0.5*(inty(yn+1) + inty(yn))
                do n = 1, size(xsHankel), 1
                    B  = 10**(wy + (n-1)*s)
                    R1 = sqrt((y - recy)**2.0 + (recx + xLength/2)**2.0)
                    R2 = sqrt((y - recy)**2.0 + (recx - xLength/2)**2.0)
                    lamb1 = B / R1
                    lamb2 = B / R2
                    call getReflectTE(lamb1, freqnow, rho, thickness, nlayer, nGauss, rte1)
                    call getReflectTE(lamb2, freqnow, rho, thickness, nlayer, nGauss, rte2)
                    ty1 = ((recx + xLength/2.0) / R1**2.0) * (lamb1 * (1+rte1))
                    ty2 = ((recx - xLength/2.0) / R2**2.0) * (lamb2 * (1+rte2))
                    By1 = By1 + 0.5 * dconst*(inty(yn+1)-inty(yn)) * xsHankel(n) * sum(ty1*wf)
                    By2 = By2 - 0.5 * dconst*(inty(yn+1)-inty(yn)) * xsHankel(n) * sum(ty2*wf)
                end do
            end do

            temfreqResp(m) = Bx1 + Bx2 + By1 + By2

        end do

    end subroutine


    !===========================================================================
    ! `compFreqResponseDipole` calculate the frequency domain response only use
    ! two dipoles
    !===========================================================================
    subroutine compFreqResponseDipole(nlayer, rho, thickness)

        integer :: nlayer
        real(drPrec),dimension(nlayer) :: rho, thickness

        ! Other
        integer :: m, xm, yn, n
        real(drPrec) :: freqnow, B, dconst
        complex(drPrec) :: Bx1
        real(drPrec),dimension(2) :: x, R1, lamb1
        complex(drPrec),dimension(2) :: rte1, tx1

        ! -----------------------------------------------------
        ! half edge at x-direction
        dconst = 1d-7
        x(1) = -xLength / 8
        x(2) = 3 * ( -xLength / 8)      ! define two dipole locations
        do m = 1, ntemFreq
            freqnow = temFreq(m)
            Bx1 = 0d0
            do n = 1, size(xsHankel), 1

                B = 10**(wy + (n-1) * s)
                R1= sqrt( (x - recx)**2.0 + (recy + yLength / 2)**2.0 )
                lamb1 = B / R1
                call getReflectTENew(lamb1, freqnow, rho, thickness, nlayer, 2, rte1)
                tx1 = ((recy + yLength /2.0 ) / R1**2.0) * (lamb1 * (1 + rte1) )
                Bx1 = Bx1 + (xLength / 4 ) * dconst * xsHankel(n) * sum(tx1)

            end do

            temfreqResp(m) = 8 * Bx1

        end do

    end


    !===========================================================================
    ! `getReflectTENew` calculates reflected coefficient for TE mode
    !
    !===========================================================================
    subroutine getReflectTENew(lamb, frequency, resistivity, thickness, nlayer, num_gauss, rte)

        integer(4) :: nlayer, num_gauss
        real(8) :: frequency
        real(8),dimension(num_gauss) :: lamb, lamb2
        complex(8),dimension(num_gauss) :: rte
        real(8),dimension(nlayer) :: resistivity, thickness
        !
        complex(8) :: z
        complex(8),dimension(nlayer) :: kk
        real(8),dimension(nlayer) :: cond
        complex(8),dimension(nlayer, num_gauss) :: yd, y
        complex(8),dimension(num_gauss) :: u, yd0, tanh_seri
        !
        integer(4) :: m

        real :: tinyeps
        ! set the permittivity very small to prevent oscillation
        ! revised in April, 2020
        tinyeps = 8.854187817d-20

        ! -------------------------------------------------------------
        cond = 1/resistivity
        lamb2 = lamb**2.0
        z = ii*mu0*frequency
        kk = -z*cond + frequency**2.0*mu0*tinyeps
        !
        yd(nlayer,:)=sqrt(lamb2-kk(nlayer))/z
        y(nlayer,:)=yd(nlayer,:)
        !
        do m = (nlayer-1),1,-1
            u  = sqrt(lamb2-kk(m))
            yd(m,:) = u/z
            !tanh_seri = tanh(u*thickness(m))
			tanh_seri = ctanh_(u*thickness(m), num_gauss)
            y(m,:) = yd(m,:)*(y(m+1,:)+yd(m,:)*tanh_seri)&
                    &/(yd(m,:)+y(m+1,:)*tanh_seri)
        end do
        !
        yd0 = lamb/z

        rte = (yd0 - y(1,:))/(yd0 + y(1,:))

    end subroutine


    !===========================================================================
    ! `compCentralFreqResponse` calculates response in frequency domain for central
    ! configuration
    !===========================================================================
    subroutine compCentralFreqResponse(nlayer, rho, thickness)

        integer :: nlayer
        real(drPrec),dimension(nlayer) :: rho, thickness

        ! local variables
        integer :: m, xm, yn, n
        real(drPrec) :: freqnow, B, dconst
        complex(drPrec) :: Bx1
        real(drPrec),dimension(nGaussHalf)    :: x, R1, lamb1
        complex(drPrec),dimension(nGaussHalf) :: rte1, tx1

        dconst = 1d-7
        bpHalf = bp(1:4)
        wfHalf = wf(1:4)

        ! half edge at x-direction
        do m = 1, ntemFreq
            freqnow = temFreq(m)
            Bx1 = 0d0
            do xm = 1, size(intx,1)-1, 1

                x = 0.5 * (intx(xm+1)-intx(xm)) * bpHalf + 0.5 * (intx(xm+1) + intx(xm))
                do n = 1, size(xsHankel), 1
                    B  = 10**(wy + (n-1) * s)
                    R1 = sqrt((x - recx)**2.0 + (recy + yLength / 2)**2.0)
                    lamb1 = B / R1
                    call getReflectTE(lamb1, freqnow, rho, thickness, nlayer, nGaussHalf, rte1)
                    tx1 = ((recy + yLength/2.0) / R1**2.0) *(lamb1 * (1+rte1))
                    Bx1 = Bx1 + 0.5*dconst*(intx(xm+1)-intx(xm)) * xsHankel(n) * sum(tx1*wfHalf)
                end do

            end do
            temfreqResp(m) = Bx1 * 8.0

        enddo

    end subroutine


    !===========================================================================
    ! `getTEMGaussCoeff` returns coefficients for Gauss integration
    !
    !===========================================================================
    subroutine getTEMGaussCoeff(nGauss, bp, wf)

        integer :: nGauss
        real(drPrec),dimension(nGauss),intent(inout) :: bp, wf

        ! local variables
        real(drPrec),dimension(8)  :: gauss8, gaussA8
        real(drPrec),dimension(16) :: gauss16, gaussA16

        data gauss8/-0.9602898564975363d+0,-0.7966664774136268d+0,&
		           &-0.525532409916329d+0 ,-0.1834346424956498d+0,&
                    &0.1834346424956498d+0, 0.525532409916329d+0 ,&
                    &0.7966664774136268d+0, 0.9602898564975363d+0/

		data gaussA8/0.1012285362903768d+0,0.2223810344533745d+0,&
                    &0.3137066458778874d+0,0.362683783378362d+0 ,&
                    &0.362683783378362d+0 ,0.3137066458778874d+0,&
                    &0.2223810344533745d+0,0.1012285362903768d+0/
        !
        data gauss16/-0.9894009349916499d+0 ,-0.9445750230732326d+0 ,&
                    &-0.8656312023878318d+0 ,-0.755404408355003d+0  ,&
                    &-0.6178762444026438d+0 ,-0.4580167776572274d+0 ,&
                    &-0.2816035507792589d+0 ,-0.09501250983763744d+0,&
                    & 0.09501250983763744d+0, 0.2816035507792589d+0 ,&
		            & 0.4580167776572274d+0 , 0.6178762444026438d+0 ,&
                    & 0.755404408355003d+0  , 0.8656312023878318d+0 ,&
                    & 0.9445750230732326d+0 , 0.9894009349916499d+0/

		data gaussA16/0.02715245941175406d+0,0.06225352393864778d+0,&
                    & 0.0951585116824929d+0 ,0.1246289712555339d+0 ,&
                    & 0.1495959888165768d+0 ,0.1691565193950026d+0 ,&
		  	        & 0.1826034150449236d+0 ,0.1894506104550685d+0 ,&
	  	            & 0.1894506104550685d+0 ,0.1826034150449236d+0 ,&
		            & 0.1691565193950026d+0 ,0.1495959888165768d+0 ,&
            		& 0.1246289712555339d+0 ,0.0951585116824929d+0 ,&
            		& 0.06225352393864778d+0,0.02715245941175406d+0/

        if (nGauss == 8) then
            bp = gauss8
            wf = gaussA8
        elseif (nGauss == 16) then
            bp = gauss16
            wf = gaussA16
        endif

    end subroutine


    !===========================================================================
    ! `getReflectTE` calculates reflected coefficient for TE mode
    !
    !===========================================================================
    subroutine getReflectTE(lamb, frequency, resistivity, thickness, nlayer, num_gauss, rte)

        integer :: nlayer, num_gauss
        real(drPrec) :: frequency
        real(drPrec),dimension(num_gauss) :: lamb
        complex(drPrec),dimension(num_gauss) :: rte
        real(drPrec),dimension(nlayer) :: resistivity, thickness
        !
        complex(drPrec) :: z, k
        real(drPrec),dimension(nlayer) :: cond
        complex(drPrec),dimension(nlayer, num_gauss) :: yd, y
        complex(drPrec),dimension(num_gauss) :: u, yd0
        !
        integer :: m
        real :: tinyeps
        ! set the permittivity very small to prevent oscillation
        ! revised in April, 2020
        tinyeps = 8.854187817d-20

        ! -------------------------------------------------------------
        cond = 1 / resistivity
        z = ii * mu0 * frequency
        k = -ii * mu0 * frequency * cond(nlayer) + frequency**2.0 * mu0 * tinyeps
        !
        yd(nlayer,:) = sqrt(lamb**2.0-k)/z
        y(nlayer,:)  = yd(nlayer,:)
        !
        do m = (nlayer-1),1,-1
            k = -ii*mu0*frequency * cond(m) + frequency**2.0 * mu0 * tinyeps
            u = sqrt(lamb**2.0-k)
          !   print *, u
            yd(m,:) = u / z
            y(m,:) = yd(m, :) * (y(m+1, :) + yd(m,:) * tanh(u*thickness(m)))&
                      &/(yd(m,:) + y(m+1,:) * tanh(u*thickness(m)))
        end do
         !
        yd0 = lamb / z

        rte = (yd0 - y(1, :)) / (yd0 + y(1, :))

    end subroutine getReflectTE


    !===========================================================================
    ! `getTEMHankelFilter` returns coefficients for Hankel transformation
    !
    !===========================================================================
    subroutine getTEMHankelFilter(order, nHankel, wy, s, xsHankel)

        integer,intent(in) :: order, nHankel
        real(drPrec),intent(inout) :: wy, s
        real(drPrec),dimension(nHankel),intent(inout) :: xsHankel

        ! local variables
        real(drPrec),dimension(120) :: j0_120
        real(drPrec),dimension(140) :: j1_140

        data j0_120/																	&
        &9.62801364263d-007,-5.02069203805d-006, 1.25268783953d-005,-1.99324417376d-005,&
        &2.29149033546d-005,-2.04737583809d-005, 1.49952002937d-005,-9.37502840980d-006,&
        &5.20156955323d-006,-2.62939890538d-006, 1.26550848081d-006,-5.73156151923d-007,&
        &2.76281274155d-007,-1.09963734387d-007, 7.38038330280d-008,-9.31614600001d-009,&
        &3.87247135578d-008, 2.10303178461d-008, 4.10556513877d-008, 4.13077946246d-008,&
        &5.68828741789d-008, 6.59543638130d-008, 8.40811858728d-008, 1.01532550003d-007,&
        &1.26437360082d-007, 1.54733678097d-007, 1.91218582499d-007, 2.35008851918d-007,&
        &2.89750329490d-007, 3.56550504341d-007, 4.39299297826d-007, 5.40794544880d-007,&
        &6.66136379541d-007, 8.20175040653d-007, 1.01015545059d-006, 1.24384500153d-006,&
        &1.53187399787d-006, 1.88633707689d-006, 2.32307100992d-006, 2.86067883258d-006,&
        &3.52293208580d-006, 4.33827546442d-006, 5.34253613351d-006, 6.57906223200d-006,&
        &8.10198829111d-006, 9.97723263578d-006, 1.22867312381d-005, 1.51305855976d-005,&
        &1.86329431672d-005, 2.29456891669d-005, 2.82570465155d-005, 3.47973610445d-005,&
        &4.28521099371d-005, 5.27705217882d-005, 6.49856943660d-005, 8.00269662180d-005,&
        &9.85515408752d-005, 1.21361571831d-004, 1.49454562334d-004, 1.84045784500d-004,&
        &2.26649641428d-004, 2.79106748890d-004, 3.43716968725d-004, 4.23267056591d-004,&
        &5.21251001943d-004, 6.41886194381d-004, 7.90483105615d-004, 9.73420647376d-004,&
        &1.19877439042d-003, 1.47618560844d-003, 1.81794224454d-003, 2.23860214971d-003,&
        &2.75687537633d-003, 3.39471308297d-003, 4.18062141752d-003, 5.14762977308d-003,&
        &6.33918155348d-003, 7.80480111772d-003, 9.61064602702d-003, 1.18304971234d-002,&
        &1.45647517743d-002, 1.79219149417d-002, 2.20527911163d-002, 2.71124775541d-002,&
        &3.33214363101d-002, 4.08864842127d-002, 5.01074356716d-002, 6.12084049407d-002,&
        &7.45146949048d-002, 9.00780900611d-002, 1.07940155413d-001, 1.27267746478d-001,&
        &1.46676027814d-001, 1.62254276550d-001, 1.68045766353d-001, 1.52383204788d-001,&
        &1.01214136498d-001,-2.44389126667d-003,-1.54078468398d-001,-3.03214415655d-001,&
       &-2.97674373379d-001, 7.93541259524d-003, 4.26273267393d-001, 1.00032384844d-001,&
       &-4.94117404043d-001, 3.92604878741d-001,-1.90111691178d-001, 7.43654896362d-002,&
       &-2.78508428343d-002, 1.09992061155d-002,-4.69798719697d-003, 2.12587632706d-003,&
       &-9.81986734159d-004, 4.44992546836d-004,-1.89983519162d-004, 7.31024164292d-005,&
       &-2.40057837293d-005, 6.23096824846d-006,-1.12363896552d-006, 1.04470606055d-007/

       data j1_140/																	    &
       &-6.76671159511d-014, 3.39808396836d-013,-7.43411889153d-013, 8.93613024469d-013,&
       &-5.47341591896d-013,-5.84920181906d-014, 5.20780672883d-013,-6.92656254606d-013,&
       & 6.88908045074d-013,-6.39910528298d-013, 5.82098912530d-013,-4.84912700478d-013,&
       & 3.54684337858d-013,-2.10855291368d-013, 1.00452749275d-013, 5.58449957721d-015,&
       &-5.67206735175d-014, 1.09107856853d-013,-6.04067500756d-014, 8.84512134731d-014,&
       & 2.22321981827d-014, 8.38072239207d-014, 1.23647835900d-013, 1.44351787234d-013,&
       & 2.94276480713d-013, 3.39965995918d-013, 6.17024672340d-013, 8.25310217692d-013,&
       & 1.32560792613d-012, 1.90949961267d-012, 2.93458179767d-012, 4.33454210095d-012,&
       & 6.55863288798d-012, 9.78324910827d-012, 1.47126365223d-011, 2.20240108708d-011,&
       & 3.30577485691d-011, 4.95377381480d-011, 7.43047574433d-011, 1.11400535181d-010,&
       & 1.67052734516d-010, 2.50470107577d-010, 3.75597211630d-010, 5.63165204681d-010,&
       & 8.44458166896d-010, 1.26621795331d-009, 1.89866561359d-009, 2.84693620927d-009,&
       & 4.26886170263d-009, 6.40104325574d-009, 9.59798498616d-009, 1.43918931885d-008,&
       & 2.15798696769d-008, 3.23584600810d-008, 4.85195105813d-008, 7.27538583183d-008,&
       & 1.09090191748d-007, 1.63577866557d-007, 2.45275193920d-007, 3.67784458730d-007,&
       & 5.51470341585d-007, 8.26916206192d-007, 1.23991037294d-006, 1.85921554669d-006,&
       & 2.78777669034d-006, 4.18019870272d-006, 6.26794044911d-006, 9.39858833064d-006,&
       & 1.40925408889d-005, 2.11312291505d-005, 3.16846342900d-005, 4.75093313246d-005,&
       & 7.12354794719d-005, 1.06810848460d-004, 1.60146590551d-004, 2.40110903628d-004,&
       & 3.59981158972d-004, 5.39658308918d-004, 8.08925141201d-004, 1.21234066243d-003,&
       & 1.81650387595d-003, 2.72068483151d-003, 4.07274689463d-003, 6.09135552241d-003,&
       & 9.09940027636d-003, 1.35660714813d-002, 2.01692550906d-002, 2.98534800308d-002,&
       & 4.39060697220d-002, 6.39211368217d-002, 9.16763946228d-002, 1.28368795114d-001,&
       & 1.73241920046d-001, 2.19830379079d-001, 2.51193131178d-001, 2.32380049895d-001,&
       & 1.17121080205d-001,-1.17252913088d-001,-3.52148528535d-001,-2.71162871370d-001,&
       & 2.91134747110d-001, 3.17192840623d-001,-4.93075681595d-001, 3.11223091821d-001,&
       &-1.36044122543d-001, 5.12141261934d-002,-1.90806300761d-002, 7.57044398633d-003,&
       &-3.25432753751d-003, 1.49774676371d-003,-7.24569558272d-004, 3.62792644965d-004,&
       &-1.85907973641d-004, 9.67201396593d-005,-5.07744171678d-005, 2.67510121456d-005,&
       &-1.40667136728d-005, 7.33363699547d-006,-3.75638767050d-006, 1.86344211280d-006,&
       &-8.71623576811d-007, 3.61028200288d-007,-1.05847108097d-007,-1.51569361490d-008,&
       & 6.67633241420d-008,-8.33741579804d-008, 8.31065906136d-008,-7.53457009758d-008,&
       & 6.48057680299d-008,-5.37558016587d-008, 4.32436265303d-008,-3.37262648712d-008,&
       & 2.53558687098d-008,-1.81287021528d-008, 1.20228328586d-008,-7.10898040664d-009,&
       & 3.53667004588d-009,-1.36030600198d-009, 3.52544249042d-010,-4.53719284366d-011/

       if (order == 0) then
           wy = -8.38850000000e+00
           s  = 9.04226468670e-02
           xsHankel = j0_120
       elseif (order == 1) then
           wy = -7.91001919000e+00
           s = 8.79671439570e-02
           xsHankel = j1_140
        end if


    end subroutine getTEMHankelFilter


    !===========================================================================
    ! `temsineTransform` calculates time domain response (dB/dt) by sine transformation.
    !
    !===========================================================================
    subroutine temsineTransform(nFreq, FreqSerial, FreqResp0, nTime, TimePoint, TimeResp)

        integer :: nFreq, nTime
        real(drPrec),dimension(nFreq) :: FreqSerial
        complex(drPrec),dimension(nFreq) :: FreqResp0
        real(drPrec),dimension(nTime) :: TimePoint, TimeResp

        ! local variables
        integer :: i, j
        real(drPrec),dimension(nTransFilter) :: transBase, cosFilt, sinFilt
        real(drPrec),dimension(nTransFilter) :: Freq_new, FreqResp_new, Grad_new

        !load the transform coefficient
        call getTransformFilter(transBase, cosFilt, sinFilt)

        do i=1,nTime
            Freq_new = transBase / TimePoint(i)
            do j=1,nTransFilter
                if (Freq_new(j) < FreqSerial(1)) then
                    Freq_new(j) = FreqSerial(1)*1.00001
                elseif (Freq_new(j) > FreqSerial(nFreq)) then
                    Freq_new(j) = FreqSerial(nFreq)*0.9999
                end if
            end do
            call spline1D(FreqSerial, aimag(FreqResp0), nFreq, &
            &Freq_new, FreqResp_new, Grad_new, nTransFilter, nFreq-1, nFreq-2)
            FreqResp_new = FreqResp_new*2/pi
            TimeResp(i) = - sum(FreqResp_new*sinFilt)/TimePoint(i)
        end do

    end subroutine temsineTransform


    !===========================================================================
    ! `temcosineTransform` calculates time domain response (dB/dt) by cosine transformation.
    !
    !===========================================================================
    subroutine temcosineTransform(FreqSerial, FreqResp0, TimePoint, TimeResp, nFreq, nTime)

        integer :: nFreq, nTime
        real(drPrec),dimension(nFreq) :: FreqSerial
        complex(drPrec),dimension(nFreq) :: FreqResp0
        real(drPrec),dimension(nTime) :: TimePoint, TimeResp
        !
        integer :: i, j
        real(drPrec),dimension(nTransFilter) :: transBase, cosFilt, sinFilt
        real(drPrec),dimension(nTransFilter) :: Freq_new, FreqResp_new, Grad_new

        ! load the transform coefficient
        call getTransformFilter(transBase,cosFilt,sinFilt)

        do i=1,nTime
            Freq_new = transBase / TimePoint(i)
            do j=1,nTransFilter
                if (Freq_new(j) < FreqSerial(1)) then
                    Freq_new(j) = FreqSerial(1)*1.00001
                elseif (Freq_new(j) > FreqSerial(nFreq)) then
                    Freq_new(j) = FreqSerial(nFreq)*0.9999
                end if
            end do
            call spline1D(FreqSerial, aimag(FreqResp0), nFreq, &
            &Freq_new, FreqResp_new, Grad_new, nTransFilter, nFreq-1, nFreq-2)
            FreqResp_new = FreqResp_new*2 / pi
            TimeResp(i) = sum(FreqResp_new*cosFilt/Freq_new) / TimePoint(i)
        end do

    end subroutine temcosineTransform


    !===========================================================================
    ! Spline function
    ! "x", "y" is the data array; "n" is the length of "x"
    ! "sx", "f", is the data need to calculate, "f1" is the gradient
    ! "m" is the length of "sx", "n1 = n-1", "n2 = n-2"
    !===========================================================================
    subroutine spline( x, y, n, sx, f, f1, m, n1, n2 )

        implicit none
        integer            :: i, j, k, m, n, n1, n2
        integer, parameter :: dp = 8
        Real(dp)           :: x(n), y(n), sx(m), f(m), f1(m)
        Real(dp)           :: s2(n), h(n1), dy(n1), s(n1), e(n2)
        Real(dp)           :: z, h1, h2, h3, h4

        do i = 1, n1
            h(i)  = x(i+1) - x(i)
            dy(i) = ( y(i+1) - y(i) ) / h(i)
        end do

        s2(1) = 0.d0; s2(n) = 0.d0
        do i = 2, n1
            s2(i) = 6.d0 * ( dy(i) - dy(i-1) )
        end do

        z = 0.5d0 / ( h(1) + h(2) )
        s(1) = -h(2) * z
        e(1) = s2(2) * z
        do i = 2, n2
            k    = i - 1
            j    = i + 1
            z    = 1.d0 / ( 2.d0*( h(i)+h(j) ) + h(i)*s(k) )
            s(i) = -h(j) * z
            e(i) = ( s2(j)-h(i)*e(k) ) * z
        end do

        s2(n1) = e(n2)
        do i = n2, 2, -1
            k     = i - 1
            s2(i) = s(k)*s2(i+1) + e(k)
        end do

        do i = 1, n1
            s(i) = ( s2(i+1) - s2(i) ) / h(i)
        end do

        i = 2
        k = 1
        do j = 1, m
            do
                if ( sx(j) > x(i) ) then
                    k = i
                    i = i + 1
                else
                    exit
                end if
            end do
            h1    = sx(j) - x(k)
            h2    = sx(j) - x(i)
            h3    = h1 * h2
            h4    = s2(k) + h1*s(k)
            z     = ( s2(i) + s2(k) + h4 ) / 6.d0
            f(j)  = y(k) + h1*dy(k) + h3*Z
            f1(j) = dy(k) + z*( h1+h2 ) + h3 * s(k) / 6.d0
        end do

    end subroutine spline

    !===========================================================================
    ! `spline1D` performs 1D spline interpolation
    !
    !===========================================================================
    subroutine spline1D( xbase, ybase, n, xfun, yfun, dfun, m, n1, n2 )

        implicit none
        integer  :: m, n, n1, n2
        Real(drPrec) :: xbase(n), ybase(n)
        Real(drPrec) :: xfun(m), yfun(m), dfun(m)

        ! local variables
        integer :: i, j, k
        Real(drPrec) :: zz, d1, d2, d3, d4
        Real(drPrec) :: dx(n1), dy(n1), ss2(n), ss(n1), ee(n2)

        do i = 1, n1
            dx(i)  = xbase(i+1) - xbase(i)
            dy(i) = ( ybase(i+1) - ybase(i) ) / dx(i)
        end do

        ss2(1) = 0.d0
        ss2(n) = 0.d0
        do i = 2, n1
            ss2(i) = 6.d0 * ( dy(i) - dy(i-1) )
        end do

        zz = 0.5d0 / ( dx(1) + dx(2) )
        ss(1) = -dx(2) * zz
        ee(1) = ss2(2) * zz
        do i = 2, n2
            k    = i - 1
            j    = i + 1
            zz    = 1.d0 / ( 2.d0*( dx(i)+dx(j) ) + dx(i)*ss(k) )
            ss(i) = -dx(j) * zz
            ee(i) = ( ss2(j)-dx(i)*ee(k) ) * zz
        end do

        ss2(n1) = ee(n2)
        do i = n2, 2, -1
            k     = i - 1
            ss2(i) = ss(k)*ss2(i+1) + ee(k)
        end do

        do i = 1, n1
            ss(i) = ( ss2(i+1) - ss2(i) ) / dx(i)
        end do

        i = 2
        k = 1
        do j = 1, m
            do
                if ( xfun(j) > xbase(i) ) then
                    k = i
                    i = i + 1
                else
                    exit
                end if
            end do
            d1    = xfun(j) - xbase(k)
            d2    = xfun(j) - xbase(i)
            d3    = d1 * d2
            d4    = ss2(k) + d1*ss(k)
            zz     = ( ss2(i) + ss2(k) + d4 ) / 6.d0
            yfun(j)  = ybase(k) + d1*dy(k) + d3*zz
            dfun(j) = dy(k) + zz*( d1+d2 ) + d3 * ss(k) / 6.d0
        end do

    end subroutine


    !===========================================================================
    ! `getTransformFilter` gets cosine and sine transform coefficient
    !
    !===========================================================================
    subroutine getTransformFilter(transBase, cosFilt, sinFilt)

        real(drPrec),dimension(nTransFilter) :: transBase, cosFilt, sinFilt

        transBase = (/9.1898135790e-07,1.0560236258e-06,1.2135021986e-06,&
                    &1.3944646218e-06,1.6024129035e-06,1.8413712856e-06,&
                    &2.1159641213e-06,2.4315053666e-06,2.7941014160e-06,&
                    &3.2107692750e-06,3.6895723535e-06,4.2397765101e-06,&
                    &4.8720293663e-06,5.5985663608e-06,6.4334475307e-06,&
                    &7.3928296036e-06,8.4952786647e-06,9.7621294499e-06,&
                    &1.1217898218e-05,1.2890757194e-05,1.4813079759e-05,&
                    &1.7022066946e-05,1.9560467360e-05,2.2477404450e-05,&
                    &2.5829327159e-05,2.9681102326e-05,3.4107269997e-05,&
                    &3.9193485939e-05,4.5038179256e-05,5.1754457203e-05,&
                    &5.9472294055e-05,6.8341046381e-05,7.8532343416e-05,&
                    &9.0243408449e-05,1.0370087551e-04,1.1916517525e-04,&
                    &1.3693557476e-04,1.5735596910e-04,1.8082153637e-04,&
                    &2.0778638524e-04,2.3877234294e-04,2.7437905368e-04,&
                    &3.1529558312e-04,3.6231375319e-04,4.1634346556e-04,&
                    &4.7843031015e-04,5.4977579956e-04,6.3176062088e-04,&
                    &7.2597135488e-04,8.3423117981e-04,9.5863515369e-04,&
                    &1.1015907582e-03,1.2658644886e-03,1.4546353912e-03,&
                    &1.6715565847e-03,1.9208259561e-03,2.2072673980e-03,&
                    &2.5364241622e-03,2.9146661326e-03,3.3493130965e-03,&
                    &3.8487763976e-03,4.4227217140e-03,5.0822561092e-03,&
                    &5.8401429775e-03,6.7110490429e-03,7.7118281915e-03,&
                    &8.8618476299e-03,1.0183362682e-02,1.1701947477e-02,&
                    &1.3446989863e-02,1.5452260124e-02,1.7756564508e-02,&
                    &2.0404496209e-02,2.3447298343e-02,2.6943855605e-02,&
                    &3.0961833823e-02,3.5578989427e-02,4.0884674204e-02,&
                    &4.6981564448e-02,5.3987647963e-02,6.2038507377e-02,&
                    &7.1289943956e-02,8.1920992687e-02,9.4137386993e-02,&
                    &1.0817554011e-01,1.2430712017e-01,1.4284430758e-01,&
                    &1.6414583639e-01,1.8862393652e-01,2.1675231129e-01,&
                    &2.4907530463e-01,2.8621843527e-01,3.2890050183e-01,&
                    &3.7794749316e-01,4.3430857292e-01,4.9907444799e-01,&
                    &5.7349847588e-01,6.5902091994e-01,7.5729682151e-01,&
                    &8.7022802846e-01,1.0000000000e+00,1.1491241000e+00,&
                    &1.3204861972e+00,1.5174025129e+00,1.7436837970e+00,&
                    &2.0037090739e+00,2.3025103863e+00,2.6458701754e+00,&
                    &3.0404331840e+00,3.4938350462e+00,4.0148500530e+00,&
                    &4.6135609538e+00,5.3015540788e+00,6.0921435595e+00,&
                    &7.0006289849e+00,8.0445914817e+00,9.2442339463e+00,&
                    &1.0622772014e+01,1.2206883330e+01,1.4027223820e+01,&
                    &1.6119020948e+01,1.8522755440e+01,2.1284944674e+01,&
                    &2.4459042893e+01,2.8106475651e+01,3.2297828537e+01,&
                    &3.7114213149e+01,4.2648836782e+01,4.9008806184e+01,&
                    &5.6317200298e+01,6.4715452107e+01,7.4366085659e+01,&
                    &8.5455861254e+01,9.8199389654e+01,1.1284328526e+02,&
                    &1.2967093861e+02,1.4900800063e+02,1.7122868462e+02,&
                    &1.9676300810e+02,2.2610511460e+02,2.5982283632e+02,&
                    &2.9856868295e+02,3.4309246908e+02,3.9425582475e+02,&
                    &4.5304886979e+02,5.2060937476e+02,5.9824477922e+02,&
                    &6.8745749350e+02,7.8997397351e+02,9.0777813134e+02,&
                    &1.0431497282e+03,1.1987084926e+03,1.3774648177e+03,&
                    &1.5828780189e+03,1.8189232789e+03,2.0901685758e+03,&
                    &2.4018630836e+03,2.7600387542e+03,3.1716270494e+03,&
                    &3.6445930787e+03,4.1880897415e+03,4.8126348549e+03,&
                    &5.5303146963e+03,6.3550178981e+03,7.3027042227e+03,&
                    &8.3917134175e+03,9.6431201283e+03,1.1081141739e+04,&
                    &1.2733607027e+04,1.4632494715e+04,1.6814552320e+04,&
                    &1.9322007302e+04,2.2203384251e+04,2.5514443945e+04,&
                    &2.9319262435e+04,3.3691471059e+04,3.8715681358e+04,&
                    &4.4489122497e+04,5.1123522849e+04,5.8747272183e+04,&
                    &6.7507906275e+04,7.7574962041e+04,8.9143258438e+04,&
                    &1.0243666662e+05,1.1771244234e+05,1.3526620437e+05,&
                    &1.5543765535e+05,1.7861715581e+05,2.0525327842e+05,&
                    &2.3586148884e+05,2.7103412109e+05,3.1145184046e+05,&
                    &3.5789681587e+05,4.1126785643e+05,4.7259780538e+05,&
                    &5.4307352777e+05,6.2405887883e+05,7.1712109749e+05,&
                    &8.2406113574e+05,9.4694851096e+05,1.0881613554e+06 /)

        !
        cosFilt = (/ 4.8963534801e-04,-3.2447354679e-03,1.0952238450e-02,&
                    &-2.5330877509e-02,4.5603964621e-02,-6.8751965019e-02,&
                    &9.1055170868e-02,-1.0955571014e-01,1.2271886286e-01,&
                    &-1.3032033376e-01,1.3300188934e-01,-1.3178555118e-01,&
                    &1.2775414321e-01,-1.2185305954e-01,1.1484395400e-01,&
                    &-1.0727994394e-01,9.9560982839e-02,-9.1939517527e-02,&
                    &8.4590736600e-02,-7.7600019561e-02,7.1031733586e-02,&
                    &-6.4890821187e-02,5.9196016140e-02,-5.3915848643e-02,&
                    &4.9055780941e-02,-4.4567058677e-02,4.0457261282e-02,&
                    &-3.6666941211e-02,3.3213645140e-02,-3.0027761602e-02,&
                    &2.7141909473e-02,-2.4473692903e-02,2.2075499247e-02,&
                    &-1.9847468574e-02,1.7867265715e-02,-1.6011026178e-02,&
                    &1.4388810260e-02,-1.2844078701e-02,1.1528724140e-02,&
                    &-1.0242160187e-02,9.1906684982e-03,-8.1147007848e-03,&
                    &7.2916108124e-03,-6.3832919299e-03,5.7602923199e-03,&
                    &-4.9801561548e-03,4.5359177425e-03,-3.8467889983e-03,&
                    &3.5670373081e-03,-2.9327308046e-03,2.8105929471e-03,&
                    &-2.1944294961e-03,2.2311101318e-03,-1.5941611250e-03,&
                    &1.8000275449e-03,-1.0989791756e-03,1.4951672247e-03,&
                    &-6.7966542661e-04,1.3003578630e-03,-3.0965377717e-04,&
                    &1.2052335926e-03,3.6106423804e-05,1.2052407467e-03,&
                    &3.8238957500e-04,1.3018966394e-03,7.5508986898e-04,&
                    &1.5033588525e-03,1.1828448710e-03,1.8253815900e-03,&
                    &1.6989785802e-03,2.2927582125e-03,2.3438821840e-03,&
                    &2.9413764370e-03,3.1679810467e-03,3.8210431940e-03,&
                    &4.2354668717e-03,4.9992627736e-03,5.6289926876e-03,&
                    &6.5661551724e-03,7.4555038862e-03,8.6406325432e-03,&
                    &9.8532339747e-03,1.1377691235e-02,1.2999446350e-02,&
                    &1.4975945576e-02,1.7117326788e-02,1.9682681678e-02,&
                    &2.2477549054e-02,2.5789277330e-02,2.9383250822e-02,&
                    &3.3599553377e-02,3.8111654528e-02,4.3330483454e-02,&
                    &4.8751291535e-02,5.4854616146e-02,6.0801696081e-02,&
                    &6.7092281901e-02,7.2264672304e-02,7.6684426491e-02,&
                    &7.7748740946e-02,7.5380953371e-02,6.5034557294e-02,&
                    &4.5881721095e-02,1.0745068396e-02,-4.1063368585e-02,&
                    &-1.1762555751e-01,-2.1286042284e-01,-3.2473691854e-01,&
                    &-4.1889667952e-01,-4.5782693031e-01,-3.5587281979e-01,&
                    &-6.6666111276e-02,4.1398275206e-01,8.2532976945e-01,&
                    &7.4155935075e-01,-2.2362177592e-01,-1.2468909758e+00,&
                    &-5.6409324151e-01,1.6313369577e+00,1.3693670121e-01,&
                    &-1.9009211808e+00,2.1011582167e+00,-1.4539662980e+00,&
                    &8.1038085853e-01,-4.1770604354e-01,2.1941406707e-01,&
                    &-1.2468136830e-01,7.8269403872e-02,-5.3876922627e-02,&
                    &3.9875972156e-02,-3.1099121841e-02,2.5148873022e-02,&
                    &-2.0846754019e-02,1.7576952843e-02,-1.4997498937e-02,&
                    &1.2906795836e-02,-1.1178868977e-02,9.7305065956e-03,&
                    &-8.5038781802e-03,7.4569676118e-03,-6.5580949022e-03,&
                    &5.7826495893e-03,-5.1110604028e-03,4.5274795600e-03,&
                    &-4.0188948308e-03,3.5745070845e-03,-3.1852787668e-03,&
                    &2.8435964847e-03,-2.5430124541e-03,2.2780422184e-03,&
                    &-2.0440036869e-03,1.8368872847e-03,-1.6532500255e-03,&
                    &1.4901283117e-03,-1.3449656159e-03,1.2155521334e-03,&
                    &-1.0999741706e-03,9.9657152340e-04,-9.0390146342e-04,&
                    &8.2070821214e-04,-7.4589699233e-04,6.7851191398e-04,&
                    &-6.1771708409e-04,5.6278043568e-04,-5.1305986067e-04,&
                    &4.6799130039e-04,-4.2707850067e-04,3.8988418417e-04,&
                    &-3.5602243104e-04,3.2515209207e-04,-2.9697108796e-04,&
                    &2.7121146955e-04,-2.4763512919e-04,2.2603006942e-04,&
                    &-2.0620715318e-04,1.8799727230e-04,-1.7124887514e-04,&
                    &1.5582580442e-04,-1.4160541246e-04,1.2847693069e-04,&
                    &-1.1634006683e-04,1.0510380008e-04,-9.4685353394e-05,&
                    &8.5009334765e-05,-7.6007041963e-05,6.7615916584e-05,&
                    &-5.9779122421e-05,5.2445208783e-05,-4.5567833381e-05,&
                    &3.9105728752e-05,-3.3023834075e-05,2.7297600235e-05,&
                    &-2.1922306831e-05,1.6926318239e-05,-1.2382788632e-05,&
                    &8.4107903564e-06,-5.1548458967e-06,2.7349863492e-06,&
                    &-1.1775367755e-06,3.6617415171e-07,-6.1920666642e-08 /)

        !
        sinFilt = (/ -5.8602704469e-10,4.8048608866e-09,-1.9771446412e-08,&
                    &5.5269671220e-08,-1.1943308955e-07,2.1463221286e-07,&
                    &-3.3618689168e-07,4.7460375561e-07,-6.2010393775e-07,&
                    &7.6676687292e-07,-9.1382097135e-07,1.0643058622e-06,&
                    &-1.2227765260e-06,1.3937672564e-06,-1.5810499935e-06,&
                    &1.7877429029e-06,-2.0163554733e-06,2.2692361668e-06,&
                    &-2.5484456754e-06,2.8562454180e-06,-3.1946379473e-06,&
                    &3.5660627246e-06,-3.9725086880e-06,4.4166880900e-06,&
                    &-4.9004832484e-06,5.4270061912e-06,-5.9978954164e-06,&
                    &6.6168927621e-06,-7.2851116079e-06,8.0072717151e-06,&
                    &-8.7834623075e-06,9.6200272667e-06,-1.0515203894e-05,&
                    &1.1478097778e-05,-1.2503644638e-05,1.3605711023e-05,&
                    &-1.4773445946e-05,1.6028992588e-05,-1.7351362256e-05,&
                    &1.8777046268e-05,-2.0267260460e-05,2.1883367382e-05,&
                    &-2.3555377917e-05,2.5387811572e-05,-2.7256008695e-05,&
                    &2.9339419633e-05,-3.1417660497e-05,3.3800354252e-05,&
                    &-3.6099620853e-05,3.8851364451e-05,-4.1374814497e-05,&
                    &4.4599458678e-05,-4.7332670414e-05,5.1188885346e-05,&
                    &-5.4081411276e-05,5.8817223326e-05,-6.1748595688e-05,&
                    &6.7759540725e-05,-7.0477668175e-05,7.8405578638e-05,&
                    &-8.0416390944e-05,9.1318327864e-05,-9.1689586953e-05,&
                    &1.0732811344e-04,-1.0434243729e-04,1.2768616501e-04,&
                    &-1.1822963939e-04,1.5431866321e-04,-1.3280640985e-04,&
                    &1.9025153590e-04,-1.4674319163e-04,2.4032678430e-04,&
                    &-1.5722591369e-04,3.1241839075e-04,-1.5869843610e-04,&
                    &4.1950672593e-04,-1.4061987158e-04,5.8323092045e-04,&
                    &-8.3489051753e-05,8.3998754957e-04,4.8166642276e-05,&
                    &1.2514134019e-03,3.1809381316e-04,1.9223958368e-03,&
                    &8.3977376473e-04,3.0319286375e-03,1.8135564849e-03,&
                    &4.8856220018e-03,3.5901028335e-03,8.0038676867e-03,&
                    &6.7764383042e-03,1.3266048500e-02,1.2406252926e-02,&
                    &2.2134422870e-02,2.2191911863e-02,3.6963935749e-02,&
                    &3.8829525302e-02,6.1308824850e-02,6.6140947826e-02,&
                    &9.9797901699e-02,1.0824048046e-01,1.5616018092e-01,&
                    &1.6533327223e-01,2.2564430323e-01,2.2062815144e-01,&
                    &2.7477185256e-01,2.1146243753e-01,2.0743923168e-01,&
                    &2.9273136482e-03,-1.1875521396e-01,-4.8782254186e-01,&
                    &-5.6383218638e-01,-7.0493717239e-01,-4.5783079001e-02,&
                    &4.9572887643e-01,1.2793243233e+00,7.6984445533e-04,&
                    &-1.0809022247e+00,-9.7826142686e-01,2.4412168562e+00,&
                    &-1.5742192934e+00,1.0474779813e-01,6.8117592803e-01,&
                    &-8.1735467757e-01,6.8894183181e-01,-5.2159909809e-01,&
                    &3.8460793473e-01,-2.8421923628e-01,2.1211949230e-01,&
                    &-1.5993330303e-01,1.2163508070e-01,-9.3164674694e-02,&
                    &7.1777818840e-02,-5.5578938643e-02,4.3227895023e-02,&
                    &-3.3758552896e-02,2.6463811012e-02,-2.0820205681e-02,&
                    &1.6436812641e-02,-1.3019659797e-02,1.0346416487e-02,&
                    &-8.2481070074e-03,6.5957434608e-03,-5.2904671538e-03,&
                    &4.2562264096e-03,-3.4343067315e-03,2.7792243882e-03,&
                    &-2.2556298932e-03,1.8359634981e-03,-1.4986732800e-03,&
                    &1.2268558973e-03,-1.0072161413e-03,8.2926783836e-04,&
                    &-6.8471812797e-04,5.6699154764e-04,-4.7086106657e-04,&
                    &3.9216119658e-04,-3.2756429147e-04,2.7440564056e-04,&
                    &-2.3054635200e-04,1.9426558645e-04,-1.6417564894e-04,&
                    &1.3915493002e-04,-1.1829482011e-04,1.0085758995e-04,&
                    &-8.6242897630e-05,7.3961097311e-05,-6.3611923024e-05,&
                    &5.4867429385e-05,-4.7458310810e-05,4.1162907678e-05,&
                    &-3.5798353790e-05,3.1213433791e-05,-2.7282808825e-05,&
                    &2.3902339183e-05,-2.0985288287e-05,1.8459236224e-05,&
                    &-1.6263565819e-05,1.4347411857e-05,-1.2667986114e-05,&
                    &1.1189208625e-05,-9.8805899457e-06,8.7163210287e-06,&
                    &-7.6745372471e-06,6.7367316776e-06,-5.8873002690e-06,&
                    &5.1132081051e-06,-4.4037712527e-06,3.7505516231e-06,&
                    &-3.1473609978e-06,2.5903624227e-06,-2.0782393944e-06,&
                    &1.6123579386e-06,-1.1967503152e-06,8.3762259930e-07,&
                    &-5.4201864582e-07,3.1540800170e-07,-1.5849870095e-07,&
                    &6.4517602921e-08,-1.8937067759e-08,3.0164202266e-09 /)


    end subroutine


    !===========================================================================
    ! `setupTEMModel` initializes staff for tem modeling
    !
    !===========================================================================
    subroutine setupTEMModel(xLenin, yLenin, recxin, recyin)

        !
        real(drPrec),intent(in) :: xLenin, yLenin, recxin, recyin

        ! local variables
        integer :: i

        xLength = xLenin
        yLength = yLenin
        recx    = recxin
        recy    = recyin

        ! loop configuration
        nxLen = 1
        nyLen = 1
        allocate(intx(nxLen+1), inty(nyLen+1))
        !
        intx(1) = - xLength / 2
        do i = 2, nxLen+1, 1
            intx(i) = intx(i-1) + xlength / nxlen
        end do
        !
        inty(1) = -ylength/2
        do i = 2, nylen+1, 1
            inty(i) = inty(i-1) + ylength/nylen
        end do

        ! Gauss integration and Hankel transformation
        nGauss  = 8
        nGaussHalf = 4
        nHankel = 140
        orderHankel = 1
        allocate(bp(nGauss), wf(nGauss), xsHankel(nHankel))
        allocate(bpHalf(nGaussHalf), wfHalf(nGaussHalf))

        !
        call getTEMGaussCoeff(nGauss, bp, wf)
        call getTEMHankelFilter(orderHankel, nHankel, wy, s, xsHankel)

        ! sine or cosine transformation
        nTransFilter = 201
        bfreq = 0.0
        efreq = 7.0
        freqDecimal = 6.0
        ntemFreq = (efreq - bfreq) * freqDecimal + 1
        allocate(temFreq(ntemFreq), temfreqResp(ntemFreq))

        do i = 1, ntemFreq
            temFreq(i) = 2 * pi * 10.0**(bfreq + (i - 1) / freqDecimal)
        end do

    end subroutine setupTEMModel


	!===========================================================================
    ! `initTEMModel` initializes staff for tem modeling
    !
    !===========================================================================
	subroutine initTEMModel(xLenin, yLenin)

		real(drPrec) :: xLenin, yLenin
        real(drPrec) :: recxin, recyin

		!xLenin = 50.0
		!yLenin = 50.0
		recxin = 0.0
		recyin = 0.0

		call setupTEMModel(xLenin, yLenin, recxin, recyin)

	end subroutine


    !===========================================================================
    ! `deallocateTEMFwd` deallocates all arrays allocated
    !
    !===========================================================================
    subroutine deallocateTEMFwd

        deallocate(temFreq, temfreqResp, bp, wf, xsHankel, intx, inty)
        deallocate(bpHalf, wfHalf)

    end subroutine deallocateTEMFwd
	
	!===========================================================================
    ! `ctanh_`, an alternative to the built-in function `tanh`, 
    !  for complex arguments.
    !===========================================================================
	function ctanh_(x, n)
		implicit none
		integer, intent(in) :: n
		complex(8), dimension(n), intent(in) :: x
		complex(8), dimension(n) :: em
		complex(8), dimension(n) :: ctanh_
		
		em = exp(-2*x)

		!ctanh_ = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
		ctanh_ = (1 - em) / (1 + em)

		where(real(em) > huge(1.0d0)) ctanh_ = (-1.d0, 0.0)

		return
    endfunction


end module
