# Go的错误处理

Go 语言通过内置的错误接口提供了非常简单的错误处理机制。
error类型是一个接口类型，定义如下：

```go
type error interface {
    Error() string
}
```

* 输出错误用内置的`errors.New`

```go
func Sqrt(f float64) (float64, error) {
    if f < 0 {
        return 0, errors.New("不能对负数开方")
    }
    // 实现
}
```

* 捕获错误

```go
result, err:= Sqrt(-1)

if err != nil {
   fmt.Println(err)
}
```

## defer语句

* `defer`关键字后必须是一个函数的调用（必须是`defer func(形参){}(实参)`或者`defer func 函数名(实参)`而不能是`defer func(){}`）。
* `defer`语句会将函数推迟到外层函数返回之后执行。

推迟调用的函数其**输入参数会立即求值**（即使defer之后参数发生修改，对已经defer的函数没有影响），但**直到外层函数返回前该函数都不会被调用**。

推迟的函数调用会被压入一个栈中。当外层函数返回时，被推迟的函数会按照**后进先出**的顺序调用。

### 要点

* 注意，`defer`后面的语句中只有最外层的函数不会被调用，其他的都是立即调用，例如运行下面这个程序：

```go
func defer_a(v string){
   fmt.Println("defer_a("+v+")")
}

func defer_b(v string)string{
   s:="defer_b("+v+")"
   fmt.Println(s)
   return s
}

func fa(){
   fmt.Println("before defer")
   defer defer_a(defer_b(defer_b("hahaha")))
   fmt.Println("after defer")
}

func main() {
  fa()
}
```

将会输出：

```shell
before defer
defer_b(hahaha)
defer_b(defer_b(hahaha))
after defer
defer_a(defer_b(defer_b(hahaha)))
```

即`defer defer_a(defer_b(defer_b("hahaha")))`中的`defer_b(defer_b("hahaha"))`在`defer`处求值，而最外层的`defer_a`则在函数返回后才求值。

* 在函数`return`语句之后`defer`不会导致函数返回后求值，比如：

```go
func fb(v string){
   defer fmt.Println("run "+v+" defer before")
   if v=="b"{
      return
   }
   defer fmt.Println("run "+v+" defer after")
}

func main() {
  fb("a")
  fb("b")
}
```

会返回：

```shell
run a defer after
run a defer before
run b defer before
```

即在输入`b`时函数`fb`的在第二个`defer`前返回，因此第二个`defer`没有执行。这个很好理解，其实`defer`就是一个和递归函数一样的压栈操作，只不过压到一个go语言设计的专用的栈里面，在`defer`之前就退出执行相当于压栈操作没有执行，自然也就不会在`return`后执行这个`defer`里的函数。

### 用途：清理释放资源

由于 defer 的延迟特性，defer 常用在函数调用结束之后清理相关的资源，比如：

```go
func main() {
   f1, _ := os.Open(filename)
   defer f1.Close()
   f2, _ := os.Open(filename)
   defer f2.Close()
}
```

这里的两个关闭文件操作会在函数返回之后按照`f2.Close()`->`f1.Close()`的顺序被调用。这种用法可以直接将释放资源的代码写在创建资源的代码下面，而不必每次都记住有哪些资源要释放。

### 用途：执行 recover

被 defer 的函数在 return 之后执行，这个时机点正好可以捕获函数抛出的 panic，因而 defer 的另一个重要用途就是执行 recover。

### 用途：修改函数的返回值（不常用）

defer还可以用于在return之后修改返回值，这是defer的实现机制赋予的能力。例如，在开头里面说的那个例子中，函数`fa`可以等价于：

```go
func fa(){
   fmt.Println("before defer")
   v:=defer_b(defer_b("hahaha"))
   fmt.Println("after defer")
   defer_a(v)
   return
}
```

即defer的实际执行其实是在其他代码和return之间，与return紧贴。又比如：

```go
func Sum(a, b int) (sum int) {
    defer func(v int)) {
        sum += c
    }(1)
    sum = a + b
}
```

调用`Sum`输出的结果是`a+b+1`，因为它可以写成：

```go
func Sum(a, b int) (sum int) {
   v:=1
   sum = a + b
   sum += c
   return
}
```

## 复杂一点的错误处理

有了上面的常规错误处理和defer的铺垫，接下来可以开始学习`panic`和`recover`机制的错误处理了。

如果是在goroutine中遇见错误时，前面的错误处理方法在一些情况下可能就不好用了，这时就要用到`panic`和`recover`机制：

`panic`相当于在函数返回和`defer`函数开始出栈执行之前加入一个标记然后立即让函数返回并且直接进入`defer`出栈执行阶段，而`defer`里面的`recover()`可以捕获到这个标记，并将他返给某个变量，如果`recover`没有捕获到标记就返回`nil`。例如：

```go
func except() {
   fmt.Println(recover())
}

func test() {
   fmt.Println("before panic")
   defer except()
   panic("test panic")
   fmt.Println("after panic")
}
```

可以写成：

```go
func except() {
   fmt.Println(recover())
}

func test() {
   fmt.Println("before panic")
   defer except()
   异常="test panic"
   return
   fmt.Println("after panic")
}
```

进而可以写成：

```go
func test() {
   fmt.Println("before panic")
   异常="test panic"
   fmt.Println(异常)
   return
   fmt.Println("after panic")
}
```

因此调用`test()`会输出：

```shell
before panic
test panic
```

### 要点

* `panic`不会向`defer`内传递，这点和正常的变量不同，比如：

```go
func test() {
   a:=1
   defer func(){
      defer func(){
         fmt.Println(a)
         fmt.Println(recover())
      }()
      func(){
         fmt.Println(a)
         fmt.Println(recover())
      }()
   }()
   panic("hahaha")
}
func main() {
   test()
}
```

会输出：

```shell
1
<nil>
1
<nil>
panic: hahaha

goroutine 1 [running]:
main.test()
   /tmp/sandbox416292788/prog.go:16 +0x60
main.main()
   /tmp/sandbox416292788/prog.go:19 +0x20
```

这表明那个嵌套的`defer`里面能看到变量`a`而看不到外面的`panic`。因此，`recover`只有在和`panic`在同一函数的第一层`defer`里面才会生效。任何函数的调用都会清除`panic`（**新来的人(panic之后的函数调用)不可能造成在它来之前就存在的错误，因此是无罪的**）。

* 在`defer`里面再抛出`panic`会发生什么？只有最里面的`panic`会被捕获（如果有`recover`在正确的位置的话）。按照上一条规则自己领悟。

* `panic`不仅会导致调用它的函数退出，还会一路连带着上层调用栈一层层全部退出，这时`panic`会沿着上层调用栈里面定义的`defer`一路传播直到遇到一个`recover`。比如上面那个例子中，要捕获那个`panic`除了可以将`recover`写在第一层`func`里面之外，还可以写成这样：

```go
func main() {
   defer func (){
      fmt.Println(recover())//可以捕获
   }()
   test()
   fmt.Println("after panic")//不会执行
}
```

这样会输出：

```go
1
<nil>
1
<nil>
hahaha
```

即`panic`沿着调用栈的`defer`一路传了出来，并且主函数在`test()`处就因为`panic`退出了，后面的`fmt.Println("after panic")`不会执行。（**在错误发生之前就在的人(panic之前的函数调用)都有可能是错误的元凶，都是有罪的**）

* 当`panic`沿着调用栈的`defer`传出来的路上出现了新的`panic`，那么`recover`只捕获到最后的`panic`（离他最近的）。

### 重要用途：发生错误时也能正常释放资源

这个很简单但很重要，看下面这个就知道了：

比如在C语言里面有一个线程锁被这样用了：

```C++
mu.Lock()
//某些操作
throw(/*某些错误*/)
//某些操作
mu.Unlock()
```

那么在锁的中间抛出错误的时候，`mu`不能正常解锁，导致死锁。

但是在go里面可以这么写：

```go
mu.Lock()
defer mu.Unlock()
//某些操作
panic(/*某些错误*/)
//某些操作
```

那么即使`panic`造成函数退出，`defer`定义的解锁操作也能执行，从而避免了死锁。

### 总结

这里引用go官方教程的错误示范做个总结：

```go
package main

import "fmt"

func main() {
   defer func() {
      defer func() {
         fmt.Println("7:", recover())//无效，没有和panic在同一函数的第一层defer里
         }()
   }()
   defer func() {
      func() {
         fmt.Println("6:", recover())//无效，函数的调用清除了panic
      }()
   }()
   func() {
      defer func() {
         fmt.Println("1:", recover())//无效，按照规则，在外层函数退出defer执行的规则，里面这个函数会立即被执行，这时还没有任何panic产生
      }()
   }()
   func() {
      defer fmt.Println("2:", recover())//无效，理由同上
   }()
   func() {
      fmt.Println("3:", recover())//无效，defer都没有，妥妥的立即执行，这时还没有任何panic产生
   }()
   fmt.Println("4:", recover())//无效，理由同上
   defer fmt.Println("5:", recover())//无效，recover()甚至被立即求值了
   panic(789)
   defer func() {
      fmt.Println("0:", recover())
   }()
}
```
