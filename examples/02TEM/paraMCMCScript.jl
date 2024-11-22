#------------------------------------------------------------------------------
# script to perform MCMC sampling in parallel
#
#------------------------------------------------------------------------------
@everywhere push!(LOAD_PATH, "/yourpath/code")
#
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
    startup = "startupfile"
    (emData,mclimits) = readstartupFile(startup)
end

#
printstyled("perform MCMC sampling ...\n", color=:blue)
pids = workers()
(results, status) = parallelMCMCsampling(mclimits, emData, pids)

#
printstyled("Output ensemble results ...\n", color=:blue)
nData   = length(emData.obsData)
filestr = "tem"
multipleStatistics(results, mclimits, filestr)

println("===============================")
#
