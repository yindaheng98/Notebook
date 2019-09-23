# nodejs-express

从nodejs原始的服务器开始👇

```javascript
var http = require('http');
http.createServer(function (req, res) {
    res.writeHead(200, {'Content-Type': 'text/plain'});
    res.end('Hello World\n');
}).listen(3000);
```

到express型的服务器👇

```javascript
var http = require('http');
var app = express();
//中间会有一大段服务器初始化配置代码
http.createServer(app).listen(app.get('port'), function(){
    console.log('Express server listening on port ' + app.get('port'));
});
```

## express服务器设置

### 应用层中间件

前提👇

```javascript
var app = express();
```

设置服务器的参数👇

```javascript
app.set('port', 3000);//设置了服务器监听端口
```

设置中间件👇每个请求到达的时候都会调用定义的一系列中间件

`app.use`设置对所有请求调用定义的中间件

`app.get`和`app.post`分别是对GET和POST请求调用中间件

```javascript
app.use(function (req, res, next) {
  console.log('Time:', Date.now());
  next();
});)//中间件1
app.use('/user', function (req, res, next) {
  console.log('This is a user');
  next();
});//中间件2
app.use('/user/:id', function (req, res, next) {
  console.log('ID:', req.params.id);
  next();
});//中间件2

var cookieParser = require('cookie-parser');
app.use(cookieParser());//cookieParser中间件
```

中间件的调用顺序和设置时的顺序一样，每个中间件都会对req和res进行一些处理，然后用`next()`调用下一个中间件。如果某个中间件没调用`next()`，那么下面的中间件都不会被调用。中间件的第一个参数为`'/user'`样子时表示只对url"xxx.com/user"调用中间件，如果是`'/user/:id'`这种样子，那就是对url"xxx.com/user/12"调用中间件，而且这里调用的中间件里面还可以用`req.params.id`查到id=12。

路由也是一种中间件👇

```javascript
//为请求的链接注册路由
app.use('/', indexRouter);
app.use('/users', usersRouter);
```

### 路由器层中间件

```javascript
var router = express.Router();
```

`router.use`、`router.get`、`router.post`和上面那几个`app.use`、`app.get`、`app.post`用法一样。

不一样的地方在于，路由像这样设置好之后👇

```javascript
app.use('/users', router);
```

链接`/user/abc/def`到了`router`那里就被截成`/abc/def`了，这就叫路由。