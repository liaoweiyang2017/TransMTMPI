using MPI
using Serialization

export initMPI, allGatherCustom

function initMPI()

    MPI.Init()
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    world_size = MPI.Comm_size(comm)
    nworkers = world_size - 1

    return (comm, rank, nworkers)

end

# 为MChainArray实现序列化方法  
function Serialization.serialize(s::AbstractSerializer, x::MChainArray{Ti,Tv}) where {Ti,Tv}  
    # 注意这里使用serialize_type时需要包含类型参数  
    Serialization.serialize_type(s, typeof(x))  
    serialize(s, x.nsample)  
    serialize(s, x.startModel)  
    serialize(s, x.chainstep)  
    serialize(s, x.chainvalue)  
    serialize(s, x.chainindice)  
    serialize(s, x.chainnlayer)  
    serialize(s, x.chainMisfit)  
    serialize(s, x.residuals)  
end  

# 实现反序列化方法  
function Serialization.deserialize(s::AbstractSerializer, ::Type{MChainArray{Ti,Tv}}) where {Ti,Tv}  
    nsample = deserialize(s)  
    startModel = deserialize(s)  
    chainstep = deserialize(s)  
    chainvalue = deserialize(s)  
    chainindice = deserialize(s)  
    chainnlayer = deserialize(s)  
    chainMisfit = deserialize(s)  
    residuals = deserialize(s)  
    MChainArray{Ti,Tv}(nsample, startModel, chainstep, chainvalue,   
                chainindice, chainnlayer, chainMisfit, residuals)  
end  

# 为TBTemperature实现序列化方法  
function Serialization.serialize(s::AbstractSerializer, x::TBTemperature)  
    Serialization.serialize_type(s, typeof(x))  # 使用typeof(x)而不是TBTemperature  
    serialize(s, x.nT)  
    serialize(s, x.tempLadder)  
    serialize(s, x.swapchain)  
end  

# 实现反序列化方法  
function Serialization.deserialize(s::AbstractSerializer, ::Type{TBTemperature})  
    nT = deserialize(s)  
    tempLadder = deserialize(s)  
    swapchain = deserialize(s)  
    TBTemperature(nT, tempLadder, swapchain)  
end   

"""  
    allGatherCustom(data, comm, rank)  

Gather custom data types from all processes using MPI collective communication.  

Parameters:  
- data: The data to be gathered (must implement serialize/deserialize)  
- comm: MPI communicator  
- rank: Number of processes to gather from  

Returns:  
- Vector containing gathered data from all processes  
"""  
function allGatherCustom(data::T, nT::Int, comm::MPI.Comm, rank::Int) where T  
    # 只让活跃的进程参与通信  
    if rank > nT  
        return Vector{T}(undef, 0)  
    end  

    # 序列化数据  
    io = IOBuffer()  
    serialize(io, data)  
    send_buffer = take!(io)  
    
    # 收集发送缓冲区大小  
    send_count = length(send_buffer)  
    all_counts = MPI.Allgather(Int32[send_count], comm)  # 使用Int32  
    
    # 计算偏移量  
    displs = Int32[0]  # 使用Int32  
    for i in 2:length(all_counts)  
        push!(displs, displs[end] + all_counts[i-1])  
    end  
    
    # 创建接收缓冲区  
    total_size = sum(all_counts)  
    recv_buffer = Vector{UInt8}(undef, total_size)  
    
    # 使用Allgatherv!  
    MPI.Allgatherv!(send_buffer, recv_buffer, all_counts, comm)  
    
    # 反序列化数据  
    results = Vector{T}(undef, nT)  
    current_pos = 1  
    
    # 处理接收到的数据  
    for i in 1:nT  
        if i <= length(all_counts)  
            chunk_size = all_counts[i]  
            try  
                results[i] = deserialize(IOBuffer(recv_buffer[current_pos:current_pos+chunk_size-1]))  
                current_pos += chunk_size  
            catch e  
                if i == rank  
                    results[i] = data  
                else  
                    results[i] = deepcopy(data)  
                end  
            end  
        else  
            if i == rank  
                results[i] = data  
            else  
                results[i] = deepcopy(data)  
            end  
        end  
    end  
    
    return results  
end  

# 为向量类型提供特化版本  
function allGatherCustom(data::Vector{T}, nT::Int, comm::MPI.Comm, rank::Int) where T  
    if rank > nT || rank > length(data)  
        return Vector{T}(undef, 0)  
    end  

    # 序列化当前进程的数据  
    io = IOBuffer()  
    serialize(io, data[rank])  
    send_buffer = take!(io)  
    
    # 收集发送缓冲区大小  
    send_count = length(send_buffer)  
    all_counts = MPI.Allgather(Int32[send_count], comm)  # 使用Int32  
    
    # 计算偏移量  
    displs = Int32[0]  # 使用Int32  
    for i in 2:length(all_counts)  
        push!(displs, displs[end] + all_counts[i-1])  
    end  
    
    # 创建接收缓冲区  
    total_size = sum(all_counts)  
    recv_buffer = Vector{UInt8}(undef, total_size)  
    
    # 使用Allgatherv!  
    MPI.Allgatherv!(send_buffer, recv_buffer, all_counts, comm)  
    
    # 反序列化数据  
    results = Vector{T}(undef, nT)  
    current_pos = 1  
    
    # 处理接收到的数据  
    for i in 1:nT  
        if i <= length(all_counts)  
            chunk_size = all_counts[i]  
            try  
                results[i] = deserialize(IOBuffer(recv_buffer[current_pos:current_pos+chunk_size-1]))  
                current_pos += chunk_size  
            catch e  
                results[i] = data[rank]  
            end  
        else  
            results[i] = data[rank]  
        end  
    end  
    
    return results  
end