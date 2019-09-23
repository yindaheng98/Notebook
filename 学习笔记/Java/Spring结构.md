# Spring原理

## Bean和Spring

* Bean：Java类或者JavaBean（就是一种特殊的Java类）
* Spring：生产Bean的工厂

Spring 可以生产**类**的**对象**，也即**实例化**(`new 类名()`);

## IoC/DI 控制反转/依赖注入

* IoC : Inversion of Control

控制反转（IoC）是一个通用的概念，它可以用许多不同的方式去表达，依赖注入仅仅是控制反转的一个具体的例子。

控制反转将原本在类内部进行的实例化操作**反转**到类的外部，依赖注入使在类内部调用的内由外部**注入**，以此减少类之间的依赖性。

* 非IOC/DI程序的类调用：在代码里，一个类中，引用另外一个类，并 new 一个对象
* IOC/DI程序的类调用：在代码里，一个类中，引用另外一个类，但不 new 对象，而是将 new 操作交给 Spring 的 Bean 工厂完成

例：某个非IOC/DI程序的调用：

* 数据访问层

```java
public Class UserDao {

    //其他代码

}
```

* 业务逻辑层，调用了数据访问层

```java
public Class UserService {

    UserDao userDao = new UserDao();

    //其他代码

}
```

把它改为IOC/DI模式：

```java
public Class UserService {

    UserDao userDao;

    //其他代码

    UserService(UserDao userDao) {
        this.userDao = userDao;
    }
}
```

并且在Spring配置文件中写上：

```xml
<bean id="userDao" class="UserDao" scope="prototype"></bean>

<bean id="userService" class="UserService" scope="prototype">
    <property name="userDao" value="userDao"></property>
</bean>
```

Spring就会在启动时完成 `userDao` 的实例化并**通过构造函数注入**到 `userService` 中，全部过程均由Spring完成，无需再进行 `new` 操作。

## AOP 面向方面的程序设计

面向方面的编程需要把程序逻辑分解成不同的部分，这些部分称为“关注点”。跨**一个应用程序**的**多个点的功能**被称为“横切关注点”，这些横切关注点在概念上独立于应用程序的业务逻辑。有各种各样的常见的很好的方面的例子，如**日志记录**、**审计**、**声明式事务**、**安全性**和**缓存**等。Spring AOP 模块提供拦截器来**拦截**一个应用程序，例如，当执行一个方法时，你可以在方法执行之前或之后添加额外的功能。

## Spring RESTful的一般架构

四层：

* DAO层、Model层、数据模型层：直接操作数据库的层
* Service层：将Controller层的数据转化到数据模型层（进行加密解密、格式转换等操作）
* Controller层：封装Service层，通过API与View层沟通
* View层：前端页面
