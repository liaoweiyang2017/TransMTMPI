# TransMTMPI: Supercharge Your EM Data Inversion with MPI-Powered MCMC! 🚀

Welcome to TransMTMPI - your powerful ally in the world of electromagnetic (EM) data inversion! Built with the lightning-fast Julia programming language, this package brings the power of parallel computing to your geophysical research through state-of-the-art MPI implementation. Whether you're wrestling with magnetotelluric (MT) data or diving deep into transient electromagnetic (TEM) analysis, TransMTMPI has got your back! 💪

## ✨ What Makes TransMTMPI Special?

- 🏃‍♂️ Blazing-fast MPI-based parallel processing
- 🎯 Smart RJMCMC and parallel tempering MCMC implementations
- 🎨 Seamless handling of 1D MT and TEM data inversion
- 📊 Rock-solid uncertainty quantification
- 🚄 High-performance computing that actually performs!

## 📜 License

TransMTMPI is free as a bird! It's distributed under the GNU General Public License. Check out the full details at the [license documentation](http://www.gnu.org/licenses/).

## 🗂️ What's in the Box?

- 📚 **./doc:** Your go-to guide for all things TransMTMPI
- 🎮 **./examples:** Real-world and synthetic examples to get you started
- 💻 **./src:** The beating heart of TransMTMPI - optimized MPI-powered source code

## 🛠️ Getting Started

### Step 1: Get Julia Up and Running

First things first - you'll need Julia v1.0 or newer. Here's how to get it:

#### 🪟 Windows Users
1. Hop over to [Julia's download page](https://julialang.org/downloads/)
2. Grab the Windows installer (.exe)
3. Click-click-done! 

#### 🐧 Linux Users
Pick your favorite flavor:

1. **The Quick Way (Recommended)**
   ```bash
   # Grab Julia and unleash it!
   wget https://julialang-s3.julialang.org/bin/linux/x64/1.10/julia-1.10.1-linux-x64.tar.gz
   tar zxvf julia-1.10.1-linux-x64.tar.gz
   # Let your system know where to find Julia
   export PATH="$PATH:/path/to/julia-1.10.1/bin"
   ```

2. **The Ubuntu Way**
   ```bash
   sudo apt update && sudo apt install julia
   # Easy peasy! 🍋
   ```

### Step 2: MPI - The Secret Sauce 🌶️

You'll need MPI to unlock TransMTMPI's full potential:

#### 🪟 Windows Warriors
1. Download [Microsoft MPI](https://www.microsoft.com/en-us/download/details.aspx?id=57467)
2. Install and you're ready to roll!

#### 🐧 Linux Lovers
```bash
# Ubuntu/Debian fans
sudo apt update && sudo apt install openmpi-bin libopenmpi-dev

# CentOS/RHEL enthusiasts
sudo yum install openmpi openmpi-devel
```

## 🎮 Let's Get This Party Started!

### Setting Up Your Playground

1. Grab TransMTMPI:
```bash
git clone https://github.com/username/TransMTMPI.git
cd TransMTMPI
```

2. Fire up Julia and prep the environment:
```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
```

3. Load up the good stuff:
```julia
]add MPI BenchmarkTools Distributions Statistics LinearAlgebra Printf Random SparseArrays Test
```

### 🚀 Time to Launch!

Tell Julia where to find TransMTMPI:

```julia
# Linux crew
push!(LOAD_PATH, "/path/to/TransMTMPI")

# Windows gang
push!(LOAD_PATH, "D:\\path\\to\\TransMTMPI")
```

### 🏃‍♂️ Running at Full Speed

TransMTMPI offers two turbocharged modes:

1. **Standard RJMCMC with MPI Boost**:
```bash
# Linux style
mpirun -np 7 julia runMPIMCMCScript.jl > runMPIInfo.txt

# Windows flavor
mpiexec -np 7 julia runMPIMCMCScript.jl > runMPIInfo.txt
```

2. **Parallel Tempering MCMC with MPI Magic**:
```bash
# Linux style
mpirun -np 7 julia runMPIPTMCMCScript.jl > runMPIPTInfo.txt

# Windows flavor
mpiexec -np 7 julia runMPIPTMCMCScript.jl > runMPIPTInfo.txt
```

The `-np` flag is your power dial - set it based on your hardware's muscles! 💪

### 📝 Example Scripts to Get You Started

Check out the `./examples` folder for some ready-to-rock scripts:

- `runMPIMCMCScript.jl`: Your MPI-powered RJMCMC adventure
- `runMPIPTMCMCScript.jl`: Parallel tempering with an MPI twist

These scripts are loaded with comments to help you customize them for your specific needs!

## 🏎️ Performance Tips

- MPI implementation leaves Julia's Distributed package in the dust, especially for parallel tempering
- Match your MPI processes to your CPU cores for maximum zoom
- Keep an eye on your system's memory - don't bite off more than you can chew!

## 📚 Show Some Love

If TransMTMPI helps your research, we'd be thrilled if you cited us:

```bibtex
@article{peng2022julia,
  title={A Julia software package for transdimensional Bayesian inversion of electromagnetic data over horizontally stratified media},
  author={Peng, Ronghua and Han, Bo and Liu, Yajun and Hu, Xiangyun},
  journal={Geophysics},
  volume={87},
  number={5},
  pages={F55--F66},
  year={2022},
  publisher={Society of Exploration Geophysicists}
}
@article{liao20223,
  title={3-D joint inversion of MT and CSEM data for imaging a high-temperature geothermal system in Yanggao Region, Shanxi Province, China},
  author={Liao, Weiyang and Peng, Ronghua and Hu, Xiangyun and Zhou, Wenlong and Huang, Guoshu},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  volume={60},
  pages={1--13},
  year={2022},
  publisher={IEEE}
}
```

## 🤝 Need Help?

Got questions? Hit a snag? No worries! Open an issue on our GitHub repository, and we'll help you get back on track. Remember, we're all in this together! 

Happy inverting! 🎉