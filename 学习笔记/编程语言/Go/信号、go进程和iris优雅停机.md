# 信号、go进程和iris优雅停机

## 信号

信号(Signal)是Linux，类Unix和其它**POSIX兼容的操作系统**中用来**进程间通讯的一种方式**。一个信号就是一个异步的通知，发送给某个进程，或者同进程的某个线程，告诉它们某个事件发生了。

当信号发送到某个进程中时，操作系统会**中断该进程的正常流程**，并进入相应的**信号处理函数**执行操作，**完成后再回到中断的地方**继续执行。

如果目标进程先前注册了**某个信号的处理程序**(signal handler)，则此处理程序会被调用，否则缺省的处理程序被调用。

## 可移植操作系统接口 POSIX

POSIX是IEEE为要在各种UNIX操作系统上运行软件，而定义API的一系列互相关联的标准的总称，其正式称呼为IEEE Std 1003，而国际标准名称为ISO/IEC 9945。此标准源于一个大约开始于1985年的项目。POSIX这个名称基本上是Portable Operating System Interface（可移植操作系统接口）的缩写，而X则表明其对Unix API的传承。

在POSIX中定义了多个信号和32个信号值，详见[Linux Signal及Golang中的信号处理](https://colobu.com/2015/10/09/Linux-Signals/)。

当接收到信号时，进程会根据信号的响应动作执行相应的操作，信号的响应动作有以下几种：

* 中止进程(Term)
* 忽略信号(Ign)
* 中止进程并保存内存信息(Core)
* 停止进程(Stop)
* 继续运行进程(Cont)

## Linux中发送信号

在linux系统中，`kill`系统调用(system call)可以用来发送一个特定的信号给进程。

比如，`kill -9 [PID]` 会发送SIGKILL信号给进程。此外，进程的终端敲入特定的组合键也会导致系统发送某个特定的信号给此进程：

* Ctrl-C 发送 INT signal (SIGINT)，通常导致进程结束
* Ctrl-Z 发送 TSTP signal (SIGTSTP); 通常导致进程挂起(suspend)
* Ctrl-\ 发送 QUIT signal (SIGQUIT); 通常导致进程结束 和 dump core.
* Ctrl-T (不是所有的UNIX都支持) 发送INFO signal (SIGINFO); 导致操作系统显示此运行命令的信息

## Go中的信号发送

## Go中的信号处理

Go信号通知机制可以通过往一个channel中发送os.Signal实现。首先需要创建一个`os.Signal` channel，然后使用signal.Notify注册要接收的信号：

```go
sigs := make(chan os.Signal, 1)
signal.Notify(sigs, syscall.SIGINT, syscall.SIGTERM)
```

这样当有SIGINT或者SIGTERM信号发到这个进程时，就可以用`<-sigs`取到这个信号了。比如这里有一个完整的用法：

```go
func main(){
    sigs := make(chan os.Signal, 1)
    signal.Notify(sigs, syscall.SIGINT, syscall.SIGTERM)
    fmt.Println("等待结束信号")
    <-sigs
    fmt.Println("收到结束信号")
}
```

执行此程序，在命令行按Ctrl-C时程序会输出"收到结束信号"然后退出。

## iris优雅停机

当服务器在运行过程中产生了一些子进程，比如在运行过程中调用了其他的系统指令，那么如果在停机时直接kill主进程的话，那些**子进程会残留在系统中**，造成各种问题（比如资源占用），因此当运行过程中创建过子进程时，**在停机时需要手动将子进程kill掉**。这种手动停机释放资源清理门户的操作就叫做“优雅停机”。

### 使用自定义通知关闭服务

iris使用自定义通知的优雅停机操作和上面的那个例子大致相同，只不过还要多一个屏蔽iris默认中断处理程序的操作：

```go
func main(){
    app := iris.New()
    app.Get("/", func(ctx iris.Context) {
        ctx.HTML("<h1>我还在运行</h1>")
    })
    ch := make(chan os.Signal, 1)
    signal.Notify(ch,
        os.Interrupt,
        syscall.SIGINT, // kill -SIGINT PID 或 Ctrl+c
        os.Kill, // os.Kill等同于syscall.Kill
        syscall.SIGKILL, syscall.SIGTERM, // kill -SIGTERM PID
    )
    app.Run(iris.Addr(":8080"), iris.WithoutInterruptHandler)
    go func(){
        <-ch
        timeout := 5 * time.Second//5秒后停机
        ctx, cancel := stdContext.WithTimeout(stdContext.Background(), timeout)
        defer cancel()
        //此处执行停机清理代码
        app.Shutdown(ctx)
    }()
}
```

### 使用默认通知关闭服务

使用默认通知关闭服务的操作就不涉及信号了，而是使用iris内置的信号处理函数：

```go
func main(){
    app := iris.New()
    
    iris.RegisterOnInterrupt(func() {
        timeout := 5 * time.Second//5秒后停机
        ctx, cancel := stdContext.WithTimeout(stdContext.Background(), timeout)
        defer cancel()
        //此处执行停机清理代码
        app.Shutdown(ctx)
    })

    app.Get("/", func(ctx iris.Context) {
        ctx.HTML("<h1>我还在运行</h1>")
    })
    app.Run(iris.Addr(":8080"), iris.WithoutInterruptHandler)
}
```
