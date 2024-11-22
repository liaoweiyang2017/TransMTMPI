#------------------------------------------------------------------------------
#
# script to build em1dmod shared library
#
#
#------------------------------------------------------------------------------
using BinDeps

@BinDeps.setup

# setup filepath
depsDir = splitdir(Base.source_path())[1]
srcDir  = joinpath(depsDir, "src")
libDir  = joinpath(depsDir, "lib")

printstyled("=== Building EM1DMOD shared library === \n", color=:blue)
println("depsDir = $(depsDir)")
println("srcDir  = $(srcDir)")
println("libDir  = $(libDir)")

if !isdir(libDir)
    printstyled("=== Creating library directory === \n", color=:blue)
    mkdir(libDir)
    if !isdir(libDir)
        error("Creating library directory failed.")
    end
end


if Sys.isunix()
    libName = "em1dmod.so"
elseif Sys.iswindows()
    libName = "em1dmod.dll"
end

src01 = joinpath(srcDir, "tem1dmod.f90")
src02 = joinpath(srcDir, "comptem1d.f90")
#
src03 = joinpath(srcDir, "FilterModules.f90")
src04 = joinpath(srcDir, "Dipole1D.f90")
src05 = joinpath(srcDir, "callDipole1D.f90")
#
src06 = joinpath(srcDir, "mt1dmod.f90")
dirfile = joinpath(libDir, libName)

#@build_steps begin
	printstyled("=== Fortran compilier Info ===\n", color=:blue)
	run(`gfortran --version`)
	run(`gfortran -O3 -march=native -fPIC -shared $(src01) $(src02) $(src03) $(src04) $(src05) $(src06) -o $(dirfile)`)

#end
