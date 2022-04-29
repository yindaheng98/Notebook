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