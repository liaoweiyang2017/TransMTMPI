#------------------------------------------------------------------------------
# script to perform MCMC sampling in parallel
#
#------------------------------------------------------------------------------
#
using Distributed
using DistributedArrays
# addprocs(4)
@everywhere push!(LOAD_PATH, "D:\\julia_study\\TransMTMPI\\src\\")
@everywhere begin
    using TransdEM.TBUtility
    using TransdEM.TBStruct
    using TransdEM.TBFileIO
    using TransdEM.TBChain
    using TransdEM.TBFwdSolver
    using TransdEM.TBAnalysis
end

#------------------------------------------------------------------------------
@everywhere begin
    printstyled("read datafile and startupfile ...\n", color=:blue)
    workpath = "D:\\julia_study\\TransMTMPI\\examples\\01MT\\"
    startup = workpath * "startupfile"
    (emData, mclimits) = readstartupFile(startup)
end

#
printstyled("perform MCMC sampling ...\n", color=:blue)
pids = workers()  
@time (results, status) = parallelMCMCsampling(mclimits, emData, pids)

#
printstyled("Output ensemble results ...\n", color=:blue)
nData = length(emData.obsData)
multipleStatistics(results, mclimits, nData, "mt")

#
printstyled("extract model parameters from the ensemble ...\n", color=:blue)
nLayer = 5
extractModelParam(results, mclimits, nLayer)

println("===============================")
#
