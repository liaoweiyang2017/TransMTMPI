export parallelMCMCsampling
#------------------------------------------------------------------------------
function parallelMCMCsampling(mclimits::MCPrior, emData::EMData, pids::Vector)

    #
    np = length(pids)
    result = Array{Any}(undef, np)
    status = Array{Any}(undef, np)

    # run multiple chains in parallel
    i = 1
    nextidx() = (idx = i; i+=1; idx)
    @sync begin

        for p = pids
            @async begin
                while true
                    idx = nextidx()
                    if idx > np
                        break
                    end
                @time (result[idx],status[idx]) = remotecall_fetch(runMCMC, p,
                                            mclimits, emData)
                end
            end # @async
        end # p

    end # @sync

    return result, status

end