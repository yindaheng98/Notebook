# Warp

## Warp是什么

>The multiprocessor creates, manages, schedules, and executes threads in groups of 32 parallel threads called warps.

Warp是Nvidia GPU中Thread的调度单位，每个Warp包含32个Thread。

## Thread在Warp中的组织方式

>When a multiprocessor is given one or more thread blocks to execute, it partitions them into warps and each warp gets scheduled by a warp scheduler for execution.

Warp的划分是由处理器自动进行的。

>The way a block is partitioned into warps is always the same; each warp contains threads of consecutive, increasing thread IDs with the first warp containing thread 0.

每个Warp中的Thread ID都是连续的。

所以，用户在程序里可以通过Thread ID判断Thread被划到了哪个Warp里，但无法指定Warp的划分方式。

## Warp对计算的影响

>Individual threads composing a warp start together at the same program address, but they have their own instruction address counter and register state and are therefore free to branch and execute independently.

每个Warp在从相同的指令地址上开始执行，但有各自的指令计数器。

可以理解为多个Thread都在扫描同一个数组，但各自有各自的进度。

>A warp executes one common instruction at a time, so full efficiency is realized when all 32 threads of a warp agree on their execution path.

但Warp中的每个Thread不是各自执行不同的指令，而是全部执行相同的指令。

所以这个运行模式也不是多个“扫描机”在相互独立地扫描同一个数组，而是一个“扫描机”在扫描数组然后把数据发给多个“执行机”。

>If threads of a warp diverge via a data-dependent conditional branch, the warp executes each branch path taken, disabling threads that are not on that path.

在这种模式下，如果Thread里面有if什么的跳过了一些指令，那这个Thread将会被暂时暂停，直到“扫描机”扫到了这个Thread要执行的下一条指令。

## Warp相关操作

>The `__shfl_sync()` intrinsics permit exchanging of a variable between threads within a warp without use of shared memory. The exchange occurs simultaneously for all active threads within the warp (and named in `mask`), moving 4 or 8 bytes of data per thread depending on the type.

>Threads within a warp are referred to as lanes, and may have an index between 0 and `warpSize-1` (inclusive). Four source-lane addressing modes are supported:
>* __shfl_sync()
>   * Direct copy from indexed lane
>* __shfl_up_sync()
>   * Copy from a lane with lower ID relative to caller
>* __shfl_down_sync()
>   * Copy from a lane with higher ID relative to caller
>* __shfl_xor_sync()
>   * Copy from a lane based on bitwise XOR of own lane ID

可以看到主要都是一些传数据的函数。

>Threads may only read data from another thread which is actively participating in the `__shfl_sync()` command. If the target thread is inactive, the retrieved value is undefined.

>```c
>T __shfl_sync(unsigned mask, T var, int srcLane, int width=warpSize);
>T __shfl_up_sync(unsigned mask, T var, unsigned int delta, int width=warpSize);
>T __shfl_down_sync(unsigned mask, T var, unsigned int delta, int width=warpSize);
>T __shfl_xor_sync(unsigned mask, T var, int laneMask, int width=warpSize);
>```
>`T` can be `int`, `unsigned int`, `long`, `unsigned long`, `long long`, `unsigned long long`, `float` or `double`. With the `cuda_fp16.h` header included, `T` can also be `__half` or `__half2`. Similarly, with the cuda_bf16.h header included, T can also be `__nv_bfloat16` or `__nv_bfloat162`.

>`__shfl_sync()` returns the value of `var` held by the thread whose ID is given by `srcLane`. If `width` is less than `warpSize` then each subsection of the warp behaves as a separate entity with a starting logical lane ID of 0. If `srcLane` is outside the range [0:`width-1`], the value returned corresponds to the value of var held by the `srcLane` modulo `width` (i.e. within the same subsection).

所以这些函数的第一个参数`mask`和地四个参数`width`分别通过掩码和范围限定了操作涉及的Thread的ID范围，第二个参数`var`指定了要传哪个变量的值，第三个参数用于指定具体要找哪个Thread。

## 其他

>If applications have warp-synchronous codes, they will need to insert the new `__syncwarp()` warp-wide barrier synchronization instruction between any steps where data is exchanged between threads via global or shared memory. Assumptions that code is executed in lockstep or that reads/writes from separate threads are visible across a warp without synchronization are invalid.

>```c
>void __syncwarp(unsigned mask=0xffffffff);
>```
>will cause the executing thread to wait until all warp lanes named in mask have executed a `__syncwarp()` (with the same mask) before resuming execution. All non-exited threads named in mask must execute a corresponding `__syncwarp()` with the same mask, or the result is undefined.

就是Warp级的同步，类似`__syncthreads()`