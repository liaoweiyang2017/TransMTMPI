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
    MPI.Init()  
    comm = MPI.COMM_WORLD  
    rank = MPI.Comm_rank(comm)    # rank starts from 0  
    size = MPI.Comm_size(comm)  
    nworkers = size - 1

    if rank == root

        print("=================================================\n")
        print("I am a MPI PARALLEL version.\n")
        nodes = nworkers + 1
        print("Total Number of nodes = $(nodes).\n")
        print("=================================================\n")
        print("Reading datafile and startupfile ...\n")
        workpath = pwd() * "/"
        startup = workpath * "startupfile"
        (emData, mclimits) = readstartupFile(startup)

        # send emData and mclimits to workers
        print("Worker #$(rank): Start sending emData and mclimits to workers ...\n")
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
        print("Worker #$(rank): Receiving mcmc results from workers completed ...\n")

        print("Worker #$(rank): Output ensemble results ...\n")
        nData = length(emData.obsData)
        multipleStatistics(results, mclimits, nData, "mt")

        #
        print("Worker #$(rank): Extract model parameters from the ensemble ...\n")
        nLayer = 5
        extractModelParam(results, mclimits, nLayer)

        print("Worker #$(rank): Run MPI parallel MCMC finished ...\n")
        print("=================================================\n")

    else

        # receive emData and mclimits from root worker
        emData, = MPI.recv(root, 1, comm)
        mclimits, = MPI.recv(root, 2, comm)
        print("Worker #$(rank): Receiving emData and mclimits from root worker completed ...\n")

        # run multiple MPI chains 
        print("Worker #$(rank): Run the current parallel chain ...\n")
        result = Array{Any}(undef, nworkers)
        status = Array{Any}(undef, nworkers)
        @time result[rank], status[rank] = runMCMC(mclimits, emData, rank)

        print("Worker #$(rank): Start sending mcmc results to root worker...\n")
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