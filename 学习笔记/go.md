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

### 指针和值

你可以为指针接收者声明方法，定义方式如上文中的类型方法`my_struct.method1`。

若使用值接收者，那么`method1`方法会对原始`my_struct`值的副本进行操作（对于函数的其它参数也是如此）。`method1`方法必须用指针接受者来更改 main 函数中声明的`my_variable`的值。

指针接收者的方法可以修改接收者指向的值（就像 method1 在这做的）。由于方法经常需要修改它的接收者，指针接收者比值接收者更常用。

#### 注意事项

不管方法定义时是以指针还是值为接收者，调用时接收者都既能为值又能为指针：

```go
/*my_interface和my_struct定义同上*/

func main() {
    var my_variable1 my_interface
    my_variable1 = new(my_struct)
    my_variable1.method1()//定义接收者是指针，调用时接收者为值
    my_variable1.method2()//定义接收者是值，调用时接收者为值

    var my_variable2 *my_interface
    my_variable2 = new(my_struct)
    my_variable2.method1()//定义接收者是指针，调用时接收者为指针
    my_variable2.method2()//定义接收者是值，调用时接收者为指针
}
```

但是指针参数的函数必须接受一个指针：

```go
func func1(变量名 *my_struct, ...) {}

func main() {
    func1(&my_variable)
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

### defer语句

defer 语句会将函数推迟到外层函数返回之后执行。

推迟调用的函数其参数会立即求值，但直到外层函数返回前该函数都不会被调用。

推迟的函数调用会被压入一个栈中。当外层函数返回时，被推迟的函数会按照后进先出的顺序调用。