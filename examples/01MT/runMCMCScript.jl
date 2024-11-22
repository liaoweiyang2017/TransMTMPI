#------------------------------------------------------------------------------
# script to perform MCMC sampling
#
#------------------------------------------------------------------------------
push!(LOAD_PATH, pwd())
push!(LOAD_PATH, joinpath(pwd(),"..","..","src"))
#
using TransdEM.TBUtility
using TransdEM.TBStruct
using TransdEM.TBFileIO
using TransdEM.TBChain
using TransdEM.TBFwdSolver
using TransdEM.TBAnalysis

#------------------------------------------------------------------------------
printstyled("read datafile and startupfile ...\n", color=:blue)
workpath = pwd() * "/"
startup = workpath * "startupfile"
(emData, mclimits) = readstartupFile(startup) 
#
printstyled("perform MCMC sampling ...\n", color=:blue)
@time (mcArray, mcstatus) = runMCMC(mclimits, emData) 
#
printstyled("Output ensemble results ...\n", color=:blue)
@time outputMCMCsamples(mcArray) 
@time Misfit_Statistics(mcArray)

printstyled("Output ensemble results ...\n", color=:blue) 
filestr = "mt"
@time sampleStatistics(mcArray, mclimits, filestr) 

#
printstyled("extract model parameters from the ensemble ...\n", color=:blue)
nLayer = 5 
extractModelParamSingle(mcArray, mclimits, nLayer) 

println("===============================")
#
