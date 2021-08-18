# 并不是从零开始的Python协程学习-如何运行协程

>## 可等待对象
>如果一个对象可以在 `await` 语句中使用，那么它就是 **可等待** 对象。许多 asyncio API 都被设计为接受可等待对象。
>
>*可等待* 对象有三种主要类型: **协程**, **任务** 和 **Future**。
>
>**协程**
>
>Python 协程属于 *可等待* 对象，因此可以在其他协程中被等待:
>```python
>import asyncio
>
>async def nested():
>    return 42
>
>async def main():
>    # Nothing happens if we just call "nested()".
>    # A coroutine object is created but not awaited,
>    # so it *won't run at all*.
>    nested()
>
>    # Let's do it differently now and await it:
>    print(await nested())  # will print "42".
>
>asyncio.run(main())
>```
>
>**重要**: 在本文档中 "协程" 可用来表示两个紧密关联的概念:
>* *协程函数*: 定义形式为 `async def` 的函数;
>* *协程对象*: 调用 *协程函数* 所返回的对象。
>
>**任务**
>
>*任务* 被用来“并行的”调度协程
>
>当一个协程通过 `asyncio.create_task()` 等函数被封装为一个 *任务*，该协程会被自动调度执行:
>```python
>import asyncio
>
>async def nested():
>    return 42
>
>async def main():
>    # Schedule nested() to run soon concurrently
>    # with "main()".
>    task = asyncio.create_task(nested())
>
>    # "task" can now be used to cancel "nested()", or
>    # can simply be awaited to wait until it is complete:
>    await task
>
>asyncio.run(main())
>```
>
>**Future 对象**
>
>`Future` 是一种特殊的 **低层级** 可等待对象，表示一个异步操作的 **最终结果**。
>
>当一个 Future 对象 *被等待*，这意味着协程将保持等待直到该 Future 对象在其他地方操作完毕。
>
>在 asyncio 中需要 Future 对象以便允许通过 async/await 使用基于回调的代码。
>
>通常情况下 **没有必要** 在应用层级的代码中创建 Future 对象。
>
>Future 对象有时会由库和某些 asyncio API 暴露给用户，用作可等待对象。

## 运行 asyncio 程序

```python
asyncio.run(coro, *, debug=False)
```

执行 coroutine coro 并返回结果。

此函数会运行传入的协程，负责管理 asyncio 事件循环，终结异步生成器，并关闭线程池。

当有其他 asyncio 事件循环在同一线程中运行时，此函数不能被调用。

此函数总是会创建一个新的事件循环并在结束时关闭之。它**应当被用作 asyncio 程序的主入口点，理想情况下应当只被调用一次**。

```python
async def main():
    await asyncio.sleep(1)
    print('hello')

asyncio.run(main())
```

## 创建任务

```python
asyncio.create_task(coro, *, name=None)
```

将 `coro` 协程 封装为一个 Task 并调度其执行。返回 Task 对象。

该任务会在 `get_running_loop()` 返回的循环中执行，如果当前线程没有在运行的循环则会引发 `RuntimeError`。

例如：

```python
async def coro():
    ...

# In Python 3.7+
task = asyncio.create_task(coro())
...

```

## 休眠

```python
coroutine asyncio.sleep(delay, result=None, *, loop=None)
```

阻塞 `delay` 指定的秒数。

如果指定了 `result`，则当协程完成时将其返回给调用者。

`sleep()` 总是会挂起当前任务，以允许其他任务运行。

以下协程示例运行 5 秒，每秒显示一次当前日期:

```python
import asyncio
import datetime

async def display_date():
    loop = asyncio.get_running_loop()
    end_time = loop.time() + 5.0
    while True:
        print(datetime.datetime.now())
        if (loop.time() + 1.0) >= end_time:
            break
        await asyncio.sleep(1)

asyncio.run(display_date())
```

## 并发运行任务

```python
awaitable asyncio.gather(*aws, loop=None, return_exceptions=False)
```

并发 运行 aws 序列中的 可等待对象。

如果 aws 中的某个可等待对象为协程，它将自动被作为一个任务调度。

如果所有可等待对象都成功完成，结果将是一个由所有返回值聚合而成的列表。结果值的顺序与 aws 中可等待对象的顺序一致。

如果 return_exceptions 为 False (默认)，所引发的首个异常会立即传播给等待 gather() 的任务。aws 序列中的其他可等待对象 不会被取消 并将继续运行。

如果 return_exceptions 为 True，异常会和成功的结果一样处理，并聚合至结果列表。

```python
import asyncio

async def factorial(name, number):
    f = 1
    for i in range(2, number + 1):
        print(f"Task {name}: Compute factorial({number}), currently i={i}...")
        await asyncio.sleep(1)
        f *= i
    print(f"Task {name}: factorial({number}) = {f}")
    return f

async def main():
    # Schedule three calls *concurrently*:
    L = await asyncio.gather(
        factorial("A", 2),
        factorial("B", 3),
        factorial("C", 4),
    )
    print(L)

asyncio.run(main())

# Expected output:
#
#     Task A: Compute factorial(2), currently i=2...
#     Task B: Compute factorial(3), currently i=2...
#     Task C: Compute factorial(4), currently i=2...
#     Task A: factorial(2) = 2
#     Task B: Compute factorial(3), currently i=3...
#     Task C: Compute factorial(4), currently i=3...
#     Task B: factorial(3) = 6
#     Task C: Compute factorial(4), currently i=4...
#     Task C: factorial(4) = 24
#     [2, 6, 24]
```