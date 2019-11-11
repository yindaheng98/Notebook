# N1QL使用方法

![封面](i/Artboard.svg)

## 基本的选择语句

```N1QL
SELECT 'Hello World' AS Greeting
```

结果：

```json
{
  "results": [
    {
      "Greeting": "Hello World"
    }
  ]
}
```

Couchbase的N1QL选择结果由一个JSON字典给出，字典最外层是`result`字段，字段值是一个JSON数组，存储了所有的结果。

## 基础

一个叫`tutorial`的文档库有如下6个json文档：

```json
{
    "age": 46,
    "children": [
        {"age": 17,"fname": "Aiden","gender": "m"},
        {"age": 2,"fname": "Bill","gender": "f"}
    ],
    "email": "dave@gmail.com",
    "fname": "Dave",
    "hobbies": ["golf","surfing"],
    "lname": "Smith",
    "relation": "friend",
    "title": "Mr.",
    "type": "contact"
}
```

```json
{
    "age": 46,
    "children": [
        {"age": 17,"fname": "Xena","gender": "f"},
        {"age": 2,"fname": "Yuri","gender": "m"}
    ],
    "email": "earl@gmail.com",
    "fname": "Earl",
    "hobbies": ["surfing"],
    "lname": "Johnson",
    "relation": "friend",
    "title": "Mr.",
    "type": "contact"
}
```

```json
{
    "age": 18,
    "children": null,
    "email": "fred@gmail.com",
    "fname": "Fred",
    "hobbies": ["golf","surfing"],
    "lname": "Jackson",
    "relation": "coworker",
    "title": "Mr.",
    "type": "contact"
}
```

```json
{
    "age": 20,
    "email": "harry@yahoo.com",
    "fname": "Harry",
    "lname": "Jackson",
    "relation": "parent",
    "title": "Mr.",
    "type": "contact"
}
```

```json
{
    "age": 56,
    "children": [
        {"age": 17,"fname": "Abama","gender": "m"},
        {"age": 21,"fname": "Bebama","gender": "m"}
    ],
    "email": "ian@gmail.com",
    "fname": "Ian",
    "hobbies": ["golf","surfing"],
    "lname": "Taylor",
    "relation": "cousin",
    "title": "Mr.",
    "type": "contact"
}
```

```json
{
    "age": 40,
    "contacts": [
        {"fname": "Fred"},
        {"fname": "Sheela"}
    ],
    "email": "jane@gmail.com",
    "fname": "Jane",
    "lname": "Edwards",
    "relation": "cousin",
    "title": "Mrs.",
    "type": "contact"
}
```

### 条件选择和从某个文档库中选择

#### WHERE

从`tutorial`文档库中选出所有`fname`为`Ian`的文档内容。

```N1QL
SELECT * FROM tutorial WHERE fname = 'Ian'
```

返回第5个JSON文档：

```json
{
  "results": [
      <第5个JSON文档的内容>
  ]
}
```

从`tutorial`文档库中选出所有`fname`为`Dave`的文档中`children`字段的第一项的`fname`，并命名为`child_name`。

```N1QL
SELECT children[0].fname AS child_name FROM tutorial WHERE fname='Dave'
```

返回结果是第一个JSON文档中`children`字段的第一项的`fname`字段值：

```json
{
  "results": [
    {
      "child_name": "Aiden"
    }
  ]
}
```

#### LIKE

同样的，N1QL中也有`LIKE`语句：

```N1QL
SELECT fname, email FROM tutorial WHERE email LIKE '%@yahoo.com'
```

选出所有用雅虎邮箱的文档的`fname`和`email`。

#### AND

和一般的SQL一样，不多说：

```N1QL
SELECT fname, email, children
    FROM tutorial
        WHERE ARRAY_LENGTH(children) > 0 AND email LIKE '%@gmail.com'
```

### 选择加计算

和一般SQL的计算一样，不用多说，当被计算的字段是数值类型时可用：

```N1QL
SELECT fname AS name_dog, age, age/7 AS age_dog_years FROM tutorial WHERE fname = 'Dave'
```

如果被计算的字段不是数值类型，那就会返回`null`。

### 函数

和一般的SQL函数用法一样，不用多说：

```N1QL
SELECT fname, age, ROUND(age/7) AS age_dog_years FROM tutorial WHERE fname = 'Dave'
```

聚合函数也是一样：

```N1QL
SELECT COUNT(*) AS count FROM tutorial
```

### 分组

说到聚合函数就要说`GROUP BY`分组查询，N1QL和SQL里面的也是一样：

```N1QL
SELECT relation, COUNT(*) AS count FROM tutorial GROUP BY relation
```

返回：

```json
{
  "results": [
    {"count": 1,"relation": "parent"},
    {"count": 2,"relation": "cousin"},
    {"count": 2,"relation": "friend"},
    {"count": 1,"relation": "coworker"}
  ]
}
```

### 选择分组

在分组聚合后取部分查询结果。比如在上面那个查亲属人数的语句基础上加一个`HAVING`子句：

```N1QL
SELECT relation, COUNT(*) AS count FROM tutorial GROUP BY relation HAVING COUNT(*) > 1
```

将只从亲属计数中返回人员总数大于1的查询结果：

```json
{
  "results": [
    {"count": 2,"relation": "cousin"},
    {"count": 2,"relation": "friend"}
  ]
}
```

### 字符串连接

N1QL的字符串不是像SQL中的连接函数，而是用`||`符号：

```N1QL
SELECT fname || " " || lname AS full_name FROM tutorial
```

### DISTINCT

和SQL一样，N1QL也有`DISTINCT`：

```N1QL
SELECT DISTINCT relation FROM tutorial
```

返回所有的`relation`字段：

```json
{
  "results": [
    {"relation": "friend"},
    {"relation": "coworker"},
    {"relation": "parent"},
    {"relation": "cousin"}
  ]
}
```

### 找空值

字段不存在和字段显式地指定为`null`都算空值：

```N1QL
SELECT fname, children FROM tutorial WHERE children IS NULL
```

返回`children`字段不存在或为`null`的文档的`fname`和`children`：

```JSON
{
  "results": [
    {
      "children": null,
      "fname": "Fred"
    }
  ]
}
```

### 排序和指定个数

和一般的SQL中排序一样，不多说：

```N1QL
SELECT fname, age FROM tutorial ORDER BY age LIMIT 2
```

### 跳过

`OFFSET`用于跳过结果，比如上面的N1QL语句加上`OFFSET`之后：

```N1QL
SELECT fname, age FROM tutorial ORDER BY age LIMIT 2 OFFSET 4
```

则返回按`age`排序，跳过前面4个结果，返回第5和6名。

## 进阶

### 返回数据库的元数据

>Document databases such as Couchbase often store meta-data about a document outside of the document.

比如数据库中的文档ID是典型的数据库元数据：

```N1QL
SELECT META(tutorial) AS meta FROM tutorial
```

返回：

```json
{
  "results": [
    {"meta": {"id": "dave"}},
    {"meta": {"id": "earl"}},
    {"meta": {"id": "fred"}},
    {"meta": {"id": "harry"}},
    {"meta": {"id": "ian"}},
    {"meta": {"id": "jane"}}
  ]
}
```

### 复合条件语句

复合条件语句的作用是判断一个JSON数组格式字段的所有值，有两种：

* `ANY [循环变量] IN [数组] SATISFIES [条件] END`
* `EVERY [循环变量] IN [数组] SATISFIES [条件] END`

顾名思义，`ANY`是指只要有一项满足就是`true`，`EVERY`必须要所有条件满足才返回`true`。

#### ANY

选出家里有至少一个孩子在十岁以上的家庭的`fname`：

```N1QL
SELECT fname
    FROM tutorial
        WHERE ANY child IN tutorial.children SATISFIES child.age > 10 END
```

#### EVERY

选出家里有全部孩子都在十岁以上的家庭的`fname`：

```N1QL
SELECT fname
    FROM tutorial
        WHERE EVERY child IN tutorial.children SATISFIES child.age > 10 END
```

### USE KEYS []

这个语句的功能和`WHERE`一样，都是按照某个字段找文档，但是`WHERE`是按照文档内的字段找文档，而`USE KEYS []`使用文档元数据中的文档ID找。显然，找文档ID比找文章内的值快：

```N1QL
SELECT * FROM tutorial USE KEYS ["dave", "ian"]
```

按照前面的`META`的结果，这个语句会返回第一个和第五个文档。

### 数组切片

N1QL中的数组切片和python一样，不多说，看看就懂：

```N1QL
SELECT children[0:2] FROM tutorial
```

返回每个文档的`children`字段的前两个值，如果文档的`children`字段值是`null`，那么返回字段值也是`null`；如果原文档没有`children`字段，则返回空字典`{}`：

```json
{
  "results": [
    {
      "$1": [
        {"age": 17,"fname": "Aiden","gender": "m"},
        {"age": 2,"fname": "Bill","gender": "f"}
      ]
    },
    {
      "$1": [
        {"age": 17,"fname": "Xena","gender": "f"},
        {"age": 2,"fname": "Yuri","gender": "m"}
      ]
    },
    {
      "$1": null
    },
    {},
    {
      "$1": [
        {"age": 17,"fname": "Abama","gender": "m"},
        {"age": 21,"fname": "Bebama","gender": "m"}
      ]
    },
    {}
  ]
}
```

#### IS NOT MISSING

`IS NOT MISSING`用于判断字段值是否存在，如果不存在则不返回。比如当上面的数组切片查询加上的这个子句之后：

```N1QL
SELECT children[0:2] FROM tutorial WHERE children[0:2] IS NOT MISSING
```

返回值中就不会有因为字段不存在而返回的空字典`{}`了。

### `ARRAY`循环生成

N1QL中的`ARRAY`和python中的列表推导式很像：

```N1QL
SELECT fname AS parent_name,
ARRAY child.fname FOR child IN tutorial.children END AS child_names
FROM tutorial WHERE children IS NOT NULL
```

对比python中的列表推导式：

```python
child_names = [child.fname for child in tutorial.children]
```

一看就懂，不多说。

## 高级

### 表连接

和SQL一样，按照结果中的数据进行连接。具体来说，Couchbase的表连接就是在一个文档库中找到特定字段为特定值的文档，然后和另一个文档连在一起：

```N1QL
SELECT * FROM users_with_orders usr
    JOIN orders_with_users orders
        ON KEYS ARRAY s.order_id FOR s IN usr.shipped_order_history END
```

* 连接文档库`users_with_orders`和`orders_with_users`，并分别以`usr`和`orders`作为其别名
* `orders_with_users`文档库中文文档的ID为`users_with_orders`库中文档的`shipped_order_history[i].order_id`的值
* 连接时，对文档库`users_with_orders`每个文档的`shipped_order_history`字段中的每一项，都取其`order_id`字段，以此为ID在`orders_with_users`文档库中查找文档
* 对文档库`users_with_orders`每个文档，将`orders_with_users`文档库中中查找到的对应文档与之组合为结果的一项，其字段名为各自文档库的别名：

```json
{
  "results": [
    {
        "usr": <users_with_orders中对应数据>,
        "orders":<orders_with_users中对应数据>
    },
    ...
  ]
}
```

#### LEFT JOIN

上面那个语句的连接操作没有`LEFT JOIN`子句，这时，如果`users_with_orders`中有某个文档在`orders_with_users`中没有对应的文库可以连接，那么这个文档就不会出现在结果中。而如果用了`LEFT JOIN`子句：

```N1QL
SELECT * FROM users_with_orders usr
    JOIN orders_with_users orders
        ON KEYS ARRAY s.order_id FOR s IN usr.shipped_order_history END
```

那么在`orders_with_users`中没有对应的文库可以连接的`users_with_orders`文档也会出现在结果中，只是没有`orders_with_users`中的字段。

（`LEFT JOIN`**将子句左边文档库的没有连接的项显示在结果中**）

### NEST

`NEST`和`JOIN`的功能完全一样，不一样的只是输出。

对于子句左边的每一项，`JOIN`将子句右边的连接结果与之一一相连，如果左边有一项和右边有3项可以相连，那么结果中就会有3个结果，这3个结果中左边的这个结果会重复3次。比如上面的那句N1QL可能输出：

```json
{
  "results": [
    {
        "usr": <usr数据1>,
        "orders":<orders数据1>
    },
    {
        "usr": <usr数据1>,
        "orders":<orders数据2>
    },
    {
        "usr": <usr数据1>,
        "orders":<orders数据3>
    }
  ]
}
```

而如果改成`NEST`：

```N1QL
SELECT * FROM users_with_orders usr
    NEST orders_with_users orders
        ON KEYS ARRAY s.order_id FOR s IN usr.shipped_order_history END
```

那么右边文档库`orders_with_users`中的那三个数据会被“nest”到一个数组里面，就像这样：

```json
{
  "results": [
    {
        "usr": <usr数据1>,
        "orders":[<orders数据1>,<orders数据2>,<orders数据3>]
    }
  ]
}
```

结果的长度大大减小。

#### LEFT NEST

`NEST`和`LEFT NEST`的区别就像`JOIN`和`LEFT JOIN`的区别一样，都是把左边没有连接的项也放在结果中，不再赘述。
