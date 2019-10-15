# Go的面向对象

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

## 指针和值

你可以为指针接收者声明方法，定义方式如上文中的类型方法`my_struct.method1`。

若使用值接收者，那么`method1`方法会对原始`my_struct`值的副本进行操作（对于函数的其它参数也是如此）。`method1`方法必须用指针接受者来更改 main 函数中声明的`my_variable`的值。

指针接收者的方法可以修改接收者指向的值（就像 method1 在这做的）。由于方法经常需要修改它的接收者，指针接收者比值接收者更常用。

### 类和接口的指针

#### 不管类方法定义时是以指针还是值为接收者，调用时接收者都既能为值又能为指针，并且在方法内调用类方法和成员的方式相同

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

#### 但是接口方法就比较严格了，用值调用和指针调用必须严格

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

#### 指针参数的函数必须接受一个指针，这是毫无疑问的

```go
func func1(变量名 *my_struct, 变量名 my_struct) {
   my_struct
}

func main() {
    func1(&my_variable,my_variable)
    func1(my_variable,my_variable)//报错
}
```

#### 类指针不能进行类型转换

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

#### 为什么要用指针接收者

首先，方法能够修改其接收者指向的值。

其次，这样可以避免在每次调用方法时复制该值。若值的类型为大型结构体时，这样做会更加高效。
