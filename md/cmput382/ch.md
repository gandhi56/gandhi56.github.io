# Chapter 1: Introduction
## Heterogenous parallel computing
* *multicore* trajectory seeks to maintain the execution speed of sequential programs while moving into multiple cores
* *many-thread* trajectory focuses more on the execution throughput of parallel applications
* design of CPU is optimized for sequential code performance.
    * makes use of sophisticated control logic to allow instructions from a single thread to execute in parallel or even out of their sequential order while maintaining the appearance of sequential execution.
    * large caches reduce data access latencies
    * neither control logic nor cache memories contribute to the peak calculation throughput
* memory bandwith: data transfer rate from memory to processor
    * GPUs have 10x bandwith than CPUs

## Architecture of a modern GPU
* CUDA-capable GPU is organized into an array of highly threaded streaming multiprocessors (SMs).
* multiple SMs form a building block
    * they share control logic and instruction cache
* GPU comes with GBs of a Graphics Double Data Rate (GDDR) Synchronous DRAM (SDRAM), referred to as **global memory**
    * global memory differs from the system DRAMs on the CPU in that they are essentially the frame buffer memory that is used for graphics.
    * they function as very high-bandwith off-chip memory, though with somewhat longer latency than typical system memory
    * for massive parallelism, high bandwith makes up for the longer latency

## Speeding up real applications
* Amdahl's law: the level of speedup that is achievable through parallel execution can be limited by the parallelizable portion of the application


# Chapter 3: Scalable Parallel Execution
* threads are organized into a two-level hierarchy:
    * grid is a 3d array of blocks
    * block is a 3d array of threads
    * all threads in a block share
* max total size of a block is limited to 1024 threads

## Resource assignment
* execution resources organized into SMs, multiple thread blocks can be assigned to an SM
* CUDA runtime system maintains a list of blocks that need to execute and assigns new blocks to SMs as previously assigned blocks complete execution
* an SM resource limitation is the number of threads that can be simultaneously tracked and scheduled.
    * it takes hardware resources for SMs to maintain the thread and block indices and track their execution status

## Thread scheduling and latency tolerance
* a block is further divided into 32 thread units called **warps**
    * a warp is the unit of thread scheduling in SMs
* an SM is designed to execute all threads in a warp to follow SIMD model
    * at any instant in time, one instruction is fetched and executed for all threads in a warp
    * consequently all threads in a warp will always have the same execution timing
* generally, there are fewer hardware Streaming Processors the number of threads assigned to each SM
* why do we need to have so many warps in an SM if it can only execute a small subset of them at any instant?
    * that is how CUDA processors efficiently execute long-latency operations, such as global memory accesses
* when an instruction to be executed by a warp needs to wait for the result of a previously initiated long-latency operation, the warp is not selected for execution.
    * Instead, selects another warp that does not meet the condition
    * priority scheduling to break ties
* warp scheduling is also used for tolerating other types of operation latencies, such as pipelined floating point arithmetic and branch instructions.
* ability to tolerate long-latency operations is the main reason GPUs do not dedicate nearly as much chip area to cache and branch prediction mechanisms as CPUs do

## Exercise
* Assume that a CUDA device allows up to 8 blocks and 1024 threads per SM, whichever becomes a limitation first. Furthermore, it allows up to 512 threads in each block. For image blur, should we use 8 × 8, 16 × 16, or 32 × 32 thread blocks? To answer the question, we can analyze the pros and cons of each choice. If we use 8 × 8 blocks, each block would have only 64 threads. We will need 1024/64 = 12 blocks to fully occupy an SM. However, each SM can only allow up to 8 blocks; thus, we will end up with only 64 × 8 = 512 threads in each SM. This limited number implies that the SM execution resources will likely be underutilized because fewer warps will be available to schedule around long-latency operations.

* The 16 × 16 blocks result in 256 threads per block, implying that each SM can take
1024/256 = 4 blocks. This number is within the 8-block limitation and is a good configuration as it will allow us a full thread capacity in each SM and a maximal number of
warps for scheduling around the long-latency operations. The 32 × 32 blocks would give
1024 threads in each block, which exceeds the 512 threads per block limitation of this
device. Only 16 × 16 blocks allow a maximal number of threads assigned to each SM.



# Chapter 4: Memory and Data Locality
