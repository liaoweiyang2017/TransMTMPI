#------------------------------------------------------------------------------
# script to perform MCMC temperature sampling in MPI parallel
# (c) Vincentliao Sep. 12, 2023, revised on Feb. 1, 2025
#------------------------------------------------------------------------------
#
push!(LOAD_PATH, pwd())
push!(LOAD_PATH, joinpath(pwd(),"..","..","src"))
using MPI 
using Serialization 
using TransdEM.TBUtility
using TransdEM.TBStruct
using TransdEM.TBFileIO
using TransdEM.TBChain
using TransdEM.TBFwdSolver
using TransdEM.TBAnalysis


#------------------------------------------------------------------------------
function runParallelMPIPTMCMC()

    # Initialize MPI environment  
    MPI.Init()  
    comm = MPI.COMM_WORLD  
    rank = MPI.Comm_rank(comm)    # rank starts from 0  
    size = MPI.Comm_size(comm)  

    # set up temperature ladder for MPI parallel tempering MCMC
    nTarget = 3     # number of chaints at target temperature T = 1.0
    nLadder = 3     # number of chains at abnormal temperature T > 1.0
    maxTemp = 2.0   # maximum temperature
    tempTarget = ones(nTarget-1)
    tempLadder = 10.0 .^ range(0.0, stop=log10(maxTemp), length=nLadder+1)
    tempLadder = vcat(tempTarget, tempLadder)
    nT = nTarget + nLadder # total number of temperature values 

    # Check if we have enough processes  
    if size < nT  
        if rank == root  
            error("Need at least $nT processes, but only $size available")  
        end  
        MPI.Finalize()  
        return  
    end  

    # Process 0 prints information 
    if rank == 0  
        print("=======================================================================================\n")  
        print("I am a MPI parallel tempered version.\n")    
        print("Total Number of nodes = $(size), using $(nT) for computation.\n") # Corrected message
        print("=======================================================================================\n")
    end
    MPI.Barrier(comm)

    # Each process reads its own data to avoid communication overhead  
    workpath = pwd() * "/"  
    startup = workpath * "startupfile"  
    (emData, mclimits) = readstartupFile(startup)  

    # Only first nT processes participate in computation  
    # rank < nT (MPI ranks start from 0)  
    if rank < nT 
        # Initialize swap history recording  
        nsample = mclimits.totalsamples  
        swapchain = zeros(Int, 3, nT, nsample)  
        tempData = TBTemperature(nT, tempLadder, swapchain)  
        
        # Execute parallel tempering MCMC  
        @time (mcResult, stResult, dfResult, paramResult) = runMPITemperedMCMC(  
            emData, mclimits, tempData, nT, rank, comm)  

        # Gather results from all processes using collective communication  
        # Note: rank + 1 for Julia indexing  
        MPI.Barrier(comm)  
        # Gather data to root (rank 0)
        data = mcResult[rank+1]
        results = MPI.gather(data, comm)

        # Process 0 handles output  
        if rank == 0  
            filestr = "mt"  
            sampleMPIPTStatistics(results, tempData, mclimits, filestr)  
            outputMPIPTchains(results, tempData, filestr)  
            print("MPI Parallel tempering MCMC completed.\n")
            print("=======================================================================================\n")
        end  
    end  

    MPI.Finalize()  
end

# the main function to run MPI parallel tempering MCMC
runParallelMPIPTMCMC()

# run command in windows
# mpiexec -np 6 julia .\runMPIPTMCMCScript.jl > runMPIPTInfo.txt

# run command in linux
# mpirun -np 6 julia runMPIPTMCMCScript.jl > runMPIPTInfo.txt