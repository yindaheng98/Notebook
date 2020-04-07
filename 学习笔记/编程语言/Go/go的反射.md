# (未完成)go的反射

## go的反射模型

在Golang实现的反射中，`interface{}`变量可以看作一个“pair”，“pair”中记录了实际变量的值和类型：`(value, type)`，`value`记录了变量的值，`type`记录了变量的类型。这个“pair”在这个变量连续赋值的过程中不会发生变化。反射就是用来检测存储在`interface{}`变量内部“pair”对的一种机制。

## 基本功能：ValueOf和TypeOf

既然反射就是用来检测存储在`interface{}`变量内部“pair”对的一种机制，那么显然，任何反射操作的第一步都是将这个内部“pair”对取出来。在Golang的反射包`reflect`中，取出`value`的操作函数是`reflect.ValueOf(...)`，取出`type`的函数是`reflect.TypeOf(...)`。

这两个函数的定义：

```go
// ValueOf returns a new Value initialized to the concrete value
// stored in the interface i.  ValueOf(nil) returns the zero 
func ValueOf(i interface{}) Value {...}
```

>ValueOf用来获取输入参数接口中的数据的值，如果接口为空则返回0

```go
// TypeOf returns the reflection Type that represents the dynamic type of i.
// If i is a nil interface value, TypeOf returns nil.
func TypeOf(i interface{}) Type {...}
```

>TypeOf用来动态获取输入参数接口中的值的类型，如果接口为空则返回nil

## 