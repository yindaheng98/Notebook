# Go语言

## Go语言的核心思想

### 并发

>Go 语言被设计成一门应用于搭载 Web 服务器，存储集群或类似用途的巨型中央服务器的系统编程语言。
>
>对于高性能分布式系统领域而言，Go 语言无疑比大多数其它语言有着更高的开发效率。它提供了海量并行的支持，这对于游戏服务端的开发而言是再好不过了。

## Go语言的线程模型：CSP线程模型

### 传统的线程模型

### Go语言推荐的线程模型

[Go条件语句select](https://www.runoob.com/go/go-select-statement.html)

CSP communicating sequential processes

## Go的包

在 Go 中，如果一个名字以大写字母开头，那么它就是已导出的。例如，Pizza 就是个已导出名，Pi 也同样，它导出自 math 包。

pizza 和 pi 并未以大写字母开头，所以它们是未导出的。

在导入一个包时，你只能引用其中已导出的名字。任何“未导出”的名字在该包外均无法访问。就算是在一个已导出的类中定义的方法，如果是以小写字母开头，也是不能导出的。

## Go的函数

函数外的每个语句都必须以关键字开始（var, func 等等），因此 := 结构不能在函数外使用。

## Go的面向对象

Go没有面向对象，但是可以通过函数方法实现类似面向对象的功能。

```go
package main

import (
   "fmt"
)

/* 定义结构体 */
type Circle struct {
  radius float64
}
//该 method 属于 Circle 类型对象中的方法
func (c Circle) getArea() float64 {
  //c.radius 即为 Circle 类型对象中的属性
  return 3.14 * c.radius * c.radius
}

func main() {
  var c1 Circle
  c1.radius = 10.00
  fmt.Println("圆的面积 = ", c1.getArea())
}
```

## Go的接口(interface)类型

Go 语言提供了一种接口数据类型，它把所有的具有共性的方法定义在一起，任何其他类型只要实现了这些方法就是实现了这个接口。

* 定义接口

```go
/* 定义接口 */
type my_interface interface {
   method1()
   method2() uint64
   method3(string) float32
   ...
   方法名(参数类型...) [返回值类型]
}
```

* 实现接口

```go
/* 定义结构体 */
type my_struct struct {
   my_id int
   /* variables */
}

/* 实现接口方法 */
func (变量名 *my_struct) method1() {
   变量名.my_id=0
   /* 方法实现 */
}
func (变量名 my_struct) method2() uint64 {
   /* 方法实现*/
}
func (变量名 my_struct) method3(string a) float32 {
   /* 方法实现*/
}
...
func (变量名 my_struct) 方法名(参数类型...) 返回值类型 {
   /* 方法实现*/
}
```

* 实现后即可调用：

```go
func main() {
    var my_variable my_interface
    my_variable = new(my_struct)
    my_variable.methods3("Hello world!")
}
```

### “非侵入式”

何为非侵入式？

在java里面必须要借助`implement`显式地指定接口才能完成接口实现，这样每个类在定义的时候在被它实现的接口“侵入”；而go里面**不需要显式地声明一个类实现了哪个接口**，**只要实现了接口的所有的方法就可以视为完成了实现**从而能赋值给一个接口变量。

### 注意事项

#### 类的注意事项

* 如果类可导出且方法可导出，在包外可以定义类并调用方法
* 如果类可导出而方法不可导出，在包外可以定义类但不能调用方法
* 如果类不可导出而方法可导出，在包外不可以定义类但可以调用方法（只能调用**包内定义的可导出变量的**可导出方法）
* 如果类不可导出而方法不可导出，在包外不可以定义类也不可以调用方法

#### 接口的注意事项

* 如果接口可导出且方法可导出，在包外可以完成接口继承和类型转换且可以定义接口变量
* 如果接口可导出而方法不可导出，在包外不能完成接口继承和类型转换（会报错“接口中有不可导出方法(non-exported methods)”）
* 如果接口不可导出而方法可导出，在包外可以完成接口继承和类型转换但不能定义接口变量（只能向已经定义的变量或者函数传参）
* 如果接口不可导出而方法不可导出，在包外不能完成接口继承和类型转换

### 指针和值

你可以为指针接收者声明方法，定义方式如上文中的类型方法`my_struct.method1`。

若使用值接收者，那么`method1`方法会对原始`my_struct`值的副本进行操作（对于函数的其它参数也是如此）。`method1`方法必须用指针接受者来更改 main 函数中声明的`my_variable`的值。

指针接收者的方法可以修改接收者指向的值（就像 method1 在这做的）。由于方法经常需要修改它的接收者，指针接收者比值接收者更常用。

#### 类和接口的指针

##### 不管类方法定义时是以指针还是值为接收者，调用时接收者都既能为值又能为指针，并且在方法内调用类方法和成员的方式相同

```go
/*my_interface和my_struct定义同上*/

func main() {
    var my_variable1 my_struct = my_struct{1}
    my_variable1.method1()//定义接收者是指针，调用时接收者为值
    my_variable1.method2()//定义接收者是值，调用时接收者为值

    var my_variable2 *my_struct = new(my_struct)//用new
    my_variable2.method1()//定义接收者是指针，调用时接收者为指针
    my_variable2.method2()//定义接收者是值，调用时接收者为指针

    var my_variable3 *my_struct = &my_struct{1}//或者直接赋值也行
}
```

##### 但是接口方法就比较严格了，用值调用和指针调用必须严格

```go
/*my_interface和my_struct定义同上*/

func main() {
    var my_variable1 my_interface//接口实例
    my_variable1 = my_struct{1}
    my_variable1.method1()//my_variable1是接口实例的值类型
    my_variable1.method2()//因此可以直接调用接口方法

    var my_variable2 *my_interface//接口实例指针
    my_variable2 = &my_variable1
    (*my_variable2).method1()//my_variable1是指向一个接口实例的指针类型
    (*my_variable2).method2()//必须严格按照指针的规则来
    my_variable1.method1()//报错
}
```

##### 指针参数的函数必须接受一个指针，这是毫无疑问的

```go
func func1(变量名 *my_struct, 变量名 my_struct) {
   my_struct
}

func main() {
    func1(&my_variable,my_variable)
    func1(my_variable,my_variable)//报错
}
```

##### 类指针不能进行类型转换

```go
type A interface{
   Amethod()
}
type B struct{
}

func (b B)Amethod(){//B类型实现了A接口
   fmt.Println("Amethod")
}

func main(){
   var a A//A类型的变量a
   var b B//B类型的变量b
   b = B{}
   a = b//正确，可以进行值传递

   var pa *A//指向A类型的指针pa
   var pb *B//指向B类型的指针pb
   pa = &a
   pb = &b
   pa = pb//错误，不可以进行指针传递
   pa = (*A)(unsafe.Pointer(pb))//这个是不安全的指针转换，它不会自己判断被转换指针的类型
}
```

##### 为什么要用指针接收者

首先，方法能够修改其接收者指向的值。

其次，这样可以避免在每次调用方法时复制该值。若值的类型为大型结构体时，这样做会更加高效。

## Go的错误处理

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
