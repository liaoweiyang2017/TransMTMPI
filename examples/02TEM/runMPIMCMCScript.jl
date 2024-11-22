#------------------------------------------------------------------------------
# script to perform MCMC sampling in MPI parallel
# (c) Vincentliao Sep. 12, 2023
#------------------------------------------------------------------------------
#
push!(LOAD_PATH, pwd())
push!(LOAD_PATH, joinpath(pwd(),"..","..","src"))
using MPI 
using TransdEM.TBUtility
using TransdEM.TBStruct
using TransdEM.TBFileIO
using TransdEM.TBChain
using TransdEM.TBFwdSolver
using TransdEM.TBAnalysis


#------------------------------------------------------------------------------
function runParallelMPI()

    root = 0
    (comm, rank, nworkers) = initMPI()

    if rank == root

        println("=================================================")
        println("I am a MPI PARALLEL version.")
        nodes = nworkers + 1
        println("Total Number of nodes = $(nodes).\n")
        println("=================================================")
        printstyled("Reading datafile and startupfile ...\n", color=:blue)
        workpath = pwd() * "/"
        startup = workpath * "startupfile"
        (emData, mclimits) = readstartupFile(startup)

        # send emData and mclimits to workers
        println("Worker #$(rank): Start sending emData and mclimits to workers ...")
        for dst in 1 : nworkers
            MPI.send(emData, dst, 1, comm)
            MPI.send(mclimits, dst, 2, comm)
        end

        # receive mcmc result 
        results = Array{Any}(undef, nworkers)
        tagID = 1000
        for rec in 1:nworkers
            results[rec], = MPI.recv(rec, rec+tagID, comm)
        end
        println("Worker #$(rank): Receiving mcmc results from workers completed ...")

        printstyled("Worker #$(rank): Output ensemble results ...\n", color=:blue)
        nData = length(emData.obsData)
        multipleStatistics(results, mclimits, nData, "mt")

        #
        printstyled("Worker #$(rank): Extract model parameters from the ensemble ...\n", color=:blue)
        nLayer = 5
        extractModelParam(results, mclimits, nLayer)

        printstyled("Worker #$(rank): Run MPI parallel MCMC finished ...\n", color=:blue)
        println("=================================================")

    else

        # receive emData and mclimits from root worker
        emData, = MPI.recv(root, 1, comm)
        mclimits, = MPI.recv(root, 2, comm)
        println("Worker #$(rank): Receiving emData and mclimits from root worker completed ...\n")

        # run multiple MPI chains 
        printstyled("Worker #$(rank): Run the current parallel chain ...\n", color=:blue)
        result = Array{Any}(undef, nworkers)
        status = Array{Any}(undef, nworkers)
        @time result[rank], status[rank] = runMCMC(mclimits, emData, rank)

        println("Worker #$(rank): Start sending mcmc results to root worker...")
        tagID = 1000
        MPI.send(result[rank], root, rank+tagID, comm)

    end

    MPI.Finalize()

end

runParallelMPI()

# run command in windows
# mpiexec -np 7 julia .\runMPIMCMCScript.jl > runMPIInfo.txt

# run command in linux
# mpirun -np 7 julia runMPIMCMCScript.jl > runMPIInfo.txt