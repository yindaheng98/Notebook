# nodejs中的Promise对象

`Promise`是一种有三个状态的对象（“执行中`pending`”、“完成`resolve`”、“失败`reject`”），构造`Promise`对象时的构造函数参数是一个有两个参数的函数，这个函数的两个参数分别对应着`Promise`对象到达两个终点状态`resolve`或`reject`时要调用的函数。

一个典型的`Promise`构造如下👇

```javascript
new Promise(function(resolve, reject){
    /*执行各类语句a*/
    if(/*某个表示任务完成的判断条件*/)
        resolve(/*上面的语句中产生的某个变量*/)
    else
        reject(new Error(/*某个错误*/))
})
```

👆当上面这个新建`Promise`语句执行时，`Promise`构造函数中的`function(resolve, reject){}`会立即被执行，当这个函数在执行“`/*执行各类语句a*/`”时，`Promise`对象的状态为“`pending`”；如果判断条件使函数执行到了`resolve`函数，则`Promise`对象状态变为`resolve`；如果判断条件使函数执行到了`reject`函数，则`Promise`对象状态变为`reject`。

为了让`Promise`在我们想要的地方执行，一般把`Promise`加个壳写成这样👇

```javascript
const myPromise = function (some_value) {
    return new Promise(function (resolve, reject) {
        /*执行各类语句，用到some_value*/
        if (/*某个表示任务完成的判断条件*/)
            resolve(/*上面的语句中产生的某个变量v*/)
        else
            reject(new Error(/*某个错误e*/))
    })
}
```

调用`myPromise()`的返回值为一个新建的`Promise`对象。

然后用的时候就这么写👇

```javascript
myPromise(some_value)
```

关于`return`的注意事项👇

## resolve和reject函数从何而来？

### 答曰：来自`then`方法或者`catch/final`方法。

在上面的`Promise`例子中，如何告诉`myPromise`resolve和reject都是什么函数？

#### 正确姿势1👇

```javascript
function resolve(v/*对应上面例子中的变量v*/){/*对v做点什么*/}
function reject(e/*对应上面例子中的错误e*/){/*对e做点什么*/}
myPromise(some_value).then(resolve,reject)
```

👆这样resolve和reject就会在构造myPromise的那个函数里面被执行了。这就相当于`myPromise`执行完成之后把某个结果`v`传递给了`resolve`函数。然后这个`then`的返回值就是那个`function (resolve, reject) {}`的返回值。

#### 正确姿势2👇

```javascript
function resolve(v/*对应上面例子中的变量v*/){/*对v做点什么*/}
function reject(e/*对应上面例子中的错误e*/){/*对e做点什么*/}
myPromise(some_value).then(resolve).catch(reject)
```

👆起始`then`里面可以不用写`reject`用的那个函数，`reject`函数可以写在`.catch`方法里面，就像`try{}catch(){}`错误处理一样，如果删了上面那个`.catch(reject)`，当出错时`new Error(/*某个错误e*/)`会真的作为错误被抛出来。

`myPromise(some_value)`的返回值是一个`Promise`对象，而`myPromise(some_value).then(resolve,reject)`的返回值则是myPromise对象构造时里面那个`function (resolve, reject) {}`的返回值。

在上面这个例子中，`myPromise(some_value).then(resolve,reject)`和`myPromise(some_value).then(resolve).catch(reject)`的返回值都为`undefined`因为`myPromise`里面的`function (resolve, reject) {}`没有返回值。

>注意，调用resolve或reject并不会终结 Promise 的参数函数的执行。

```javascript
new Promise((resolve, reject) => {
  resolve(1);
  console.log(2);
}).then(r => {
  console.log(r);
});
// 2
// 1
```

>上面代码中，调用`resolve(1)`以后，后面的`console.log(2)`还是会执行，并且会首先打印出来。这是因为虽然看起来`resolve(1)`在`console.log(2)`的前面，但是这其实只是告诉了`Promise`当`resolve`时要执行`resolve(1)`，这个`resolve(1)`语句会被保留直到`console.log(2)`执行完并且函数退出后才会触发。
>
>一般来说，调用`resolve`或`reject`以后，`Promise`的使命就完成了，后继操作应该放到`then`方法里面，而不应该直接写在`resolve`或`reject`的后面。所以，最好在它们前面加上`return`语句，这样就不会有意外。

就像这样👇

```javascript
const myPromise = function (some_value) {
    return new Promise(function (resolve, reject) {
        /*执行各类语句，用到some_value*/
        if (/*某个表示任务完成的判断条件*/)
            return resolve(/*上面的语句中产生的某个变量v*/)
        else
            return reject(new Error(/*某个错误e*/))
    })
}
```

这时再调用`myPromise(some_value).then(resolve,reject)`和`myPromise(some_value).then(resolve).catch(reject)`的话就会有返回值了，因为`myPromise`里面的`function (resolve, reject) {}`有了返回值。并且按照上面那个写法，`myPromise(some_value).then(resolve,reject)`和`myPromise(some_value).then(resolve).catch(reject)`的返回值就是`resolve(/*上面的语句中产生的某个变量v*/)`或者`reject(new Error(/*某个错误e*/))`

后面所有的代码都默认`Promise`在`resolve`或`reject`处返回值。

## nodejs高玩的骚操作👇

### `resolve`函数返回一个`Promise`

先来个简单的，让`then`返回一个新的`myPromise`

```javascript
const myPromise = function (some_value) {
    return new Promise(function (resolve, reject) {
        /*执行各类语句，用到some_value*/
        if (/*某个表示任务完成的判断条件*/)
            return resolve(v/*上面的语句中产生的某个变量v*/)
        else
            return reject(new Error(e/*某个错误e*/))
    })
}

myPromise(value1).then(function(v){
    return myPromise(v)
})
```

然后因为第一个`then`的返回值变成一个`Promise`了，它又可以再`then`一次，所以我们就可以进一步这么写👇

```javascript
function resolve(v/*对应上面例子中的变量v*/){/*对v做点什么*/}
function reject(e/*对应上面例子中的错误e*/){/*对e做点什么*/}

myPromise(value1).then(function(v){
    return myPromise(v)
}).then(resolve,reject)
```

或者这么写👇

```javascript
myPromise(value1).then(function(v){
    return myPromise(v)
}).then(resolve).catch(reject)
```

这相当于是把`myPromise(value1)`的结果传递给了又一个`myPromise(v)`，然后再把`myPromise(v)`的结果传递给`resolve(v)`；并且`myPromise(v)`在`myPromise(value1)`到`resolve`状态了之后才会执行。

#### 出错了咋办？

注意到上面两个例子中的第一个`then`没有指定`reject`，这时如果有某一个`myPromise`运行到`reject`了，后面的`then`都不会执行直到这个`reject`碰到了某个`then(resolve,reject)`或者`catch(reject)`。如果后面没有`then(resolve,reject)`或者`catch(reject)`了？那就成为一个被抛出的错误。

#### 骚操作x5

连6个`myPromise`👇

```javascript
myPromise(v1).then(function(v2){
    return myPromise(v2)
}).then(function(v3){
    return myPromise(v3)
}).then(function(v4){
    return myPromise(v4)
}).then(function(v5){
    return myPromise(v5)
}).then(function(v6){
    return myPromise(v6)
}).then(resolve).catch(reject)
```

套4个`myPromise`👇

```javascript
myPromise(v1).then(function(v2){
    return myPromise(v2).then(function(v3){
        return myPromise2(v2,v3).then(function(v4){
            return myPromise3(v2,v3,v4)
        })
    })
}).then(resolve).catch(reject)
```

嵌套和连接不一样的地方就在于，嵌套可以综合前面各个`Promise`的返回值，连接只能获取前面一个。`myPromise2`和`myPromise3`定义如下👇。

```javascript
const myPromise2 = function (v2,v3) {
    return new Promise(function (resolve, reject) {
        /*执行各类语句*/
        if (/*某个表示任务完成的判断条件*/)
            return resolve(v4/*上面的语句中产生的某个变量v4*/)
        else
            return reject(new Error(e/*某个错误e*/))
    })
}

const myPromise2 = function (v2,v3,v4) {
    return new Promise(function (resolve, reject) {
        /*执行各类语句*/
        if (/*某个表示任务完成的判断条件*/)
            return resolve(v5/*上面的语句中产生的某个变量v5*/)
        else
            return reject(new Error(e/*某个错误e*/))
    })
}
```
