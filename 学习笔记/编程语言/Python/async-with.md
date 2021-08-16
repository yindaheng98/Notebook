# 并不是从零开始的Python协程学习-上下文管理器

## `with`上下文管理器

### 上下文管理器的概念

>上下文管理器 是一个对象，它定义了在执行`with`语句时要建立的运行时上下文。上下文管理器处理进入和退出所需运行时上下文以执行代码块。

>上下文管理器的典型用法包括保存和恢复各种全局状态，锁定和解锁资源，关闭打开的文件等等。

>上下文管理器通常使用`with`语句调用，但是也可以通过直接调用它们的方法来使用。

### 上下文管理器的本质

两个函数：

#### `object.__enter__(self)`
进入与此对象相关的运行时上下文。`with`语句将会绑定这个方法的返回值到`as`子句中指定的目标，如果有的话。

#### `object.__exit__(self, exc_type, exc_value, traceback)`
退出关联到此对象的运行时上下文。 各个参数描述了导致上下文退出的异常。 如果上下文是无异常地退出的，三个参数都将为`None`。

如果提供了异常，并且希望方法屏蔽此异常（即避免其被传播），则应当返回真值。 否则的话，异常将在退出此方法时按正常流程处理。

请注意`__exit__()`方法不应该重新引发被传入的异常，这是调用者的责任。

## `async with`异步上下文管理器

异步上下文管理器是上下文管理器的一种，与普通的上下文管理器区别在于能够在其`enter`和`exit`方法中暂停执行。

异步上下文管理器的本质和一般的上下文管理器类似，也是两个函数：`object.__aenter__(self)`和`object.__exit__(self, exc_type, exc_value, traceback)`

`async with`调用异步上下文管理器相当于就是在进入和退出的地方进行异步等待`await object.__aenter__(self)`

### 异步上下文管理器的典型案例

例如一个典型的限制并发数的爬虫：

```python
import asyncio
import httpx
import time


async def req(delay, sem):
    async with sem:
        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.get(f'http://127.0.0.1:8000/sleep/{delay}')
            result = resp.json()
            print(result)


async def main():
    start = time.time()
    delay_list = [3, 6, 1, 8, 2, 4, 5, 2, 7, 3, 9, 8]
    task_list = []
    sem = asyncio.Semaphore(3)
    for delay in delay_list:
        task = asyncio.create_task(req(delay, sem))
        task_list.append(task)
    await asyncio.gather(*task_list)

    end = time.time()
    print(f'一共耗时：{end - start}')

asyncio.run(main())
```

这里面使用了两个典型的异步上下文管理器：
* `asyncio`中常用于控制并发数的信号量类`asyncio.Semaphore`
* `httpx`里的异步HTTP客户端`httpx.AsyncClient`