export writeTEMData, writeMTData, writeCSEMData

#-------------------------------------------------------------------------------
"""
    writeTEMData(datafile, temData, predData, dataErr)

output predicated EM data.

"""
function writeTEMData(datafile::String, temData::TEMData,
                      predData::Array, dataErr=zeros(0))

    datID = open(datafile, "w")
    @printf(datID, "# %s\n", "file generated in $(Libc.strftime(time()))")

    xLen = temData.loopxLen
    yLen = temData.loopyLen
    @printf(datID, "%-16s %4g %4g\n","LoopWidth(m):", xLen, yLen)
    timePoint = temData.timePoint

    ntemData = length(timePoint)
    # data block
    if isempty(dataErr)
        dataErr = abs.(predData[1:ntemData]) * 0.05
    elseif length(dataErr) .== 1
        dataErr = abs.(predData[1:ntemData]) * dataErr[1]
    end

    @printf(datID,"%-15s %d\n","Data Block:", ntemData)
    for i = 1:ntemData
        @printf(datID,"%13.8e %15.6e %15.6e\n", timePoint[i], predData[i], dataErr[i])
    end

    close(datID)

end


#-------------------------------------------------------------------------------
"""
    writeMTData(datafile, mtData, predData, dataErr)

output predicated MT data.

"""
function writeMTData(datafile::String, mtData::MTData,
                     predData::Array, dataErr=zeros(0))

    datID = open(datafile, "w")

    # data format
    @printf(datID, "%-20s%s\n", "Format:", "MTData_1.0")
    @printf(datID, "# %s\n", "file generated in $(Libc.strftime(time()))")

    # frequencies
    freqs = mtData.freqArray
    nF = length(freqs)
    @printf(datID,"%-20s %3d\n","Frequencies (Hz):",nF)
    for i = 1:nF
        @printf(datID, "%8.4e\n",freqs[i])
    end

    # data type
    dataType = mtData.dataType
    nDT = length(dataType)
    @printf(datID, "%-12s %d\n", "DataType:", nDT)
    for i = 1:nDT
        @printf(datID, "%4s\n", dataType[i])
    end

    # data block
    if isempty(dataErr)
        dataErr = abs.(predData) * 0.05
    elseif length(dataErr) .== 1
        dataErr = abs.(predData) * dataErr[1]
    end

    dataID = reshape(mtData.dataID,nDT,nF)
    nData  = size(predData, 1)
    @printf(datID,"%-15s %d\n","Data Block:", nData)

    # freqID   recID   DataTpye  Data DataError
    @printf(datID,"# %6s %10s %10s %12s\n","FreqNo.","DataType","Value","Error")
    i = 1
    for k = 1:nF
        for j = 1:nDT
            if dataID[j,k]
                @printf(datID,"%5d %8d %15.6e %15.6e\n",k,j,predData[i],dataErr[i])
                i += 1
            end
        end
    end
    close(datID)

end


#-------------------------------------------------------------------------------
"""
    writeCSEMData(datafile, emData, predData, dataErr)

output predicated CSEM data.

"""
function writeCSEMData(datafile::String, emData::CSEMData,
                     predData::Array, dataErr=zeros(0))

    #
    datID = open(datafile, "w")

    # data format
    @printf(datID, "%-15s%s\n", "Format:", "CSEMData_1.0")
    @printf(datID, "# %s\n", "file generated in $(Libc.strftime(time()))")

    # seawater layer
    if !isempty(emData.seawater)
        seawater = emData.seawater
        @printf(datID,"%-15s %8.1f %8g\n", "SeaLayer:", seawater[1], seawater[2])
    end

    # dipole length
    if emData.dpLen > 1e+2
        @printf(datID,"%-20s %6.1f\n", "Dipole Length:", emData.dpLen)
    end

    # source location
    txLoc = emData.txLoc
    ns = size(txLoc, 1)
    @printf(datID,"%-25s %4d\n", "Source Location (m):", ns)
    @printf(datID,"# %5s %5s %5s %5s %5s\n", "X", "Y", "Z", "Azimuth", "Dip")
    for i = 1:ns
        for j = 1:5
            @printf(datID,"%10.1f ",txLoc[i, j])
        end
        @printf(datID,"\n")
    end

    # receiver location
    rxLoc = emData.rxLoc
    nr = size(rxLoc, 1)
    @printf(datID,"%-25s %4d\n", "Receiver Location (m):", nr)
    @printf(datID,"# %5s %5s %5s\n", "X", "Y", "Z")
    for i = 1:nr
        @printf(datID,"%10.1f %10.1f %10.1f\n", rxLoc[i,1], rxLoc[i,2], rxLoc[i,3])
    end

    # frequencies
    freqArray = emData.freqArray
    nF = length(freqArray)
    @printf(datID,"%-20s %3d\n", "Frequencies (Hz):", nF)
    for i = 1:nF
        @printf(datID,"%8.4e\n", freqArray[i])
    end

    # data type
    dataType = emData.dataType
    nDT = length(dataType)
    @printf(datID,"%-15s %d\n", "DataType:", nDT)
    for i = 1:nDT
        @printf(datID,"%4s\n",dataType[i])
    end

    if isempty(dataErr)
        dataErr = abs.(predData) * 0.05
    elseif length(dataErr) .== 1
        dataErr = abs.(predData) * dataErr[1]
    end

    dataID = reshape(emData.dataID,nDT,nr,ns,nF)
    nData  = size(predData, 1)
    @printf(datID,"%-15s %d\n","Data Block:", nData)

    # freqID   scrID   recID   DataTpye  RealPart ImagPart DataError
    @printf(datID,"#%6s %6s %6s %10s %8s %10s\n","FreqNo", "TxNo", "RxNo",
            "DataType", "Value", "Error")
    i = 1
    for k = 1:nF
        for itx = 1:ns
            for irx = 1:nr
                for idt = 1:nDT
                    if dataID[idt,irx,itx,k]
                        @printf(datID,"%5d %5d %6d %6d %15.6e %15.6e\n",
                                k, itx, irx, idt, predData[i], dataErr[i])
                        i += 1
                    end
                end
            end
        end
    end
    close(datID)

end
