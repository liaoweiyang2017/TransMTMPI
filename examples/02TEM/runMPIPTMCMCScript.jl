#------------------------------------------------------------------------------
# script to perform MCMC temperature sampling in MPI parallel
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
function runParallelMPIPTMCMC()

    root = 0
    (comm, rank, nworkers) = initMPI()

    # set up temperature ladder for MPI parallel tempering MCMC
    nTarget = 3     # number of chaints at target temperature T = 1.0
    nLadder = 3     # number of chains at abnormal temperature T > 1.0
    maxTemp = 2.0   # maximum temperature
    tempTarget = ones(nTarget-1)
    tempLadder = 10.0 .^ range(0.0, stop=log10(maxTemp), length=nLadder+1)
    tempLadder = vcat(tempTarget, tempLadder)
    nT = nTarget + nLadder # total number of temperature values 


    if rank == root

        println("==============================================================================================")
        println("I am a MPI parallel tempered version.")
        nodes = nworkers + 1
        print("Total Number of nodes = $(nodes), working nodes = $(nworkers).\n")
        println("==============================================================================================")

        # check number of workers
        if nT > nworkers
            error("The number of working nodes = $(nworkers) should be equal or larger than $(nT) tempered chains!")
        end

        printstyled("Reading datafile and startupfile ...\n", color=:blue)
        workpath = pwd() * "/"
        startup = workpath * "startupfile"
        (emData, mclimits) = readstartupFile(startup)

        # send emData and mclimits to workers
        println("Worker #$(rank): Start sending emData and mclimits to workers ...\n")
        for dst in 1 : nT
            MPI.send(emData, dst, 1, comm)
            MPI.send(mclimits, dst, 2, comm)
        end

        # receive mcmc result 
        tagID = 1000
        results = Array{Any}(undef, nT)
        for rec in 1:nT
            results[rec], = MPI.recv(rec, rec+tagID, comm)
        end
        println("Worker #$(rank): Receiving mcmc results from workers completed ...\n")

        # receive tempData from workers
        tagID = 2000
        tempData = Array{Any}(undef, nT)
        for rec in 1:nT
            tempData[rec], = MPI.recv(rec, rec+tagID, comm)
        end
        println("Worker #$(rank): Receiving tempData from workers completed ...")

        # extract output information from the ensemble
        printstyled("Worker #$(rank): Output ensemble results ...\n", color=:cyan)
        filestr = "mt"
        sampleMPIPTStatistics(results, tempData[1], mclimits, filestr)
        outputMPIPTchains(results, tempData[1], filestr)

        println("==============================================================================================")

    elseif rank <= nT

        # receive emData and mclimits from root worker
        emData, = MPI.recv(root, 1, comm)
        mclimits, = MPI.recv(root, 2, comm)
        println("Worker #$(rank): Receiving emData and mclimits from root worker completed ...\n")

        # record swap data
        nsample = mclimits.totalsamples
        swapchain  = zeros(Int, 3, nT, nsample) # record swap history
        tempData   = TBTemperature(nT, tempLadder, swapchain)
        
        printstyled("Worker #$(rank): Perform transdimensional Bayesian inversion with MPI parallel tempering...\n", color=:cyan)
        @time (mcWorker, stWorker, dfWorker, paramWorker) = runMPITemperedMCMC(emData, mclimits, tempData, nworkers, rank)

        # send mcWorker to root
        println("Worker #$(rank): Start sending mcmc results to root worker...")
        tagID = 1000
        MPI.send(mcWorker[rank], root, rank+tagID, comm)

        # send tempData to root
        println("Worker #$(rank): Start sending tempData to root worker...")
        tagID = 2000
        MPI.send(tempData, root, rank+tagID, comm)
        
    else 

        print("Worker #$(rank): I am not in the computing nodes.\n")

    end

    MPI.Finalize()

end

runParallelMPIPTMCMC()

# run command in windows
# mpiexec -np 7 julia .\runMPIPTMCMCScript.jl > runMPIPTInfo.txt

# run command in linux
# mpirun -np 7 julia runMPIPTMCMCScript.jl > runMPIPTInfo.txt