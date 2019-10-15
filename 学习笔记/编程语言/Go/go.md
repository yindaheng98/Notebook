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

## Go的指针

go的指针用法和C语言基本相同，用`&`取地址、`*`解指针；指针和值修改的各种特性也和C语言相同。此处只介绍几个go语言独特的地方。

### `&Type{}`和`new(Type)`

下面这几个例子是等价的：

```go
a:=&Type{}
```

```go
func f()*int{
   i:=0
   return &i
}

a:=f()
```

```go
a:=new(Type)
```

```go
func f()*Type{
   a:=new(int)
   return a
}

a:=f()

```

看出来什么了吗？没错，在go语言里面`a:=&Type{}`和`a:=new(Type)`都是分配堆空间，每次`a:=&Type{}`和`a:=new(Type)`操作得到的都是不同的地址。比如调用上面两个`f`函数的任一个：

```go
func main() {
   p:=f()
   pp:=f()
   *p=1
   fmt.Println(*pp)
}
```

这里的输出是0。这就和C语言不一样了，因为在C语言里面，函数里的变量都分配的是栈空间，比如下面这个函数：

```C
int* f(){
   int i = 0;
   return &i;
}
```

这里会返回程序中的局部变量在函数栈中的地址，在编译时会报`warning: address of local variable ‘i’ returned [-Wreturn-local-addr]`。

一般来说，go里面`a:=&Type{}`用的比`a:=new(Type)`多，因为`a:=new(Type)`在新建Type实例时只能初始化为0值，而`a:=&Type{}`可以指定初始值。
