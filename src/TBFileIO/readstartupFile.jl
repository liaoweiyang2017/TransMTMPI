export readstartupFile
#输出开始文件
#-------------------------------------------------------------------------------
function readstartupFile(startupfile::String)

    #
    if isfile(startupfile) #判断是否是正常文件
        fid = open(startupfile, "r") #如果是，以读的方式打开文件；
    else
        error("$(startupfile) does not exist, please try again.")
    end

    #
    surveyType = [] #调查类型
    datafile = []  #数据文件
    mclimits = initMCPrior() #初始化mc限制参数, mclimits是可变结构体
    while !eof(fid)
        cline = strip(readline(fid))

        # ignore all comments: empty line, or line preceded with #
        while cline[1] == '#' || isempty(cline)
            cline = strip(readline(fid))
        end
        cline = lowercase(cline)

        # data filename
        if occursin("datafile:", cline)
            tmp = split(cline)
            datafile = string(tmp[end])

        elseif occursin("surveytype:", cline)
            tmp = split(cline)
            surveyType = string(tmp[end])

            # markov chain parameter
        elseif occursin("burninsamples:", cline)
            tmp = split(cline)
            mclimits.burninsamples = parse(Int, tmp[end])#将一个字符串转化成整数：预热采样次数

        elseif occursin("totalsamples:", cline)
            tmp = split(cline)
            mclimits.totalsamples = parse(Int, tmp[end])

            # prior parameter
        elseif occursin("numberoflayer:", cline)
            tmp = split(cline)
            mclimits.nlayermin = parse(Int, tmp[end-1])
            mclimits.nlayermax = parse(Int, tmp[end])

        elseif occursin("zcoordinate(m):", cline)
            tmp = split(cline)
            zmin = parse(Float64, tmp[end-2])
            zmax = parse(Float64, tmp[end-1])
            zstd = parse(Float64, tmp[end]) # z的标准差
            #
            mclimits.zmin = log10(zmin) #最小深度，同时转换为对数域
            mclimits.zmax = log10(zmax) #最大深度，同时转换为对数域
            mclimits.zstd = zstd * (mclimits.zmax - mclimits.zmin) #标准扰动值的大小

        elseif occursin("minthickness(m):", cline)
            tmp = split(cline)
            mclimits.hmin = parse(Float64, tmp[end]) # 最小层厚

        # proposal parameter
        elseif occursin("resistivity:", cline)
            tmp = split(cline)
            rhomin = parse(Float64, tmp[end-2])
            rhomax = parse(Float64, tmp[end-1])
            rhostd = parse(Float64, tmp[end])
            #
            mclimits.rhomin = log10(rhomin) # 限制的电阻率的最小值，同时转换为对数域
            mclimits.rhomax = log10(rhomax) # 限制的电阻率的最大值，同时转换为对数域
            mclimits.rhostd = rhostd * (mclimits.rhomax - mclimits.rhomin) #标准电阻率扰动值的大小

        # parameters for post analysis
        elseif occursin("numberofbins:", cline)
            tmp = split(cline)
            mclimits.nzBins = parse(Float64, tmp[end-1]) #深度的离散数目
            mclimits.npBins = parse(Float64, tmp[end])   #电阻率的离散数目

        elseif occursin("credinterval:", cline)
            tmp = split(cline)
            mclimits.credInterval = parse(Float64, tmp[end]) #读入置信区间

        else
            @warn("$(cline) is not supported!")

        end

    end # while
    close(fid)

    # read observed data 读观测数据
    if surveyType == "tem"
        print("TEM data is loading ...\n")
        emData = readTEMData(datafile)
    elseif surveyType == "mt"
        print("MT data is loading ...\n")
        emData = readMTData(datafile)
    elseif surveyType == "csem"
        print("CSEM data is loading ...\n")
        emData = readCSEMData(datafile)
    end

    return emData, mclimits

end
