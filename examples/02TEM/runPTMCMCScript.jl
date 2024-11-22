#------------------------------------------------------------------------------
# script to perform MCMC sampling with parallel tempering
#
#------------------------------------------------------------------------------
using Distributed 
@everywhere push!(LOAD_PATH, "D:/information/code/TransdEM-main/TransdEM-main/src")
#
@everywhere begin
    using TransdEM.TBUtility
    using TransdEM.TBStruct
    using TransdEM.TBFileIO
    using TransdEM.TBChain
    using TransdEM.TBFwdSolver
    using TransdEM.TBAnalysis
end
# set up temperature ladder for parallel tempering MCMC
nTarget = 4     # number of chaints at target temperature T = 1.0
nLadder = 4     # number of chains at abnormal temperature T > 1.0
maxTemp = 2.5   # maximum temperature
tempTarget = ones(nTarget-1)
tempLadder = 10.0 .^ range(0.0, stop=log10(maxTemp), length=nLadder+1)
tempLadder = vcat(tempTarget, tempLadder)
nT = nTarget + nLadder

# check number of workers
pids = workers()
np   = length(pids)
if nT < np
    rmprocs(collect(nT+2:np+1))
elseif nT > np
    error("The number of workers should be equal or larger than the number of tempered chains!")
end
pids = workers()

#-------------------------------------------------------------------------------
# run MCMC sampler
@everywhere begin
    startup = "startupfile"
    printstyled("Read datafile and startupfile ...\n", color=:cyan)
    (emData, mclimits) = readstartupFile(startup)
end
nsample = mclimits.totalsamples
swapchain  = zeros(Int, 3, nT, nsample)
tempData   = TBTemperature(nT, tempLadder, swapchain)

#
printstyled("Perform transdimensional Bayesian inversion with parallel tempering...\n", color=:cyan)
@time (mcRef,stRef,dfRef,paramRef) = runTemperedMCMC(emData,mclimits,tempData,pids)

#
printstyled("Output ensemble results ...\n", color=:cyan)
filestr = "tem"
nEnsemble = samplePTStatistics(mcRef, tempData, mclimits, filestr)
outputPTchains(mcRef, tempData, filestr)

#
printstyled("Release all darrays ...\n", color=:cyan)
clearPTrefence(mcRef,stRef,dfRef,paramRef)

println("===============================")
