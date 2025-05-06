# 【转载】模块化规范：ESM与CJS 的差异

[原文在此](https://juejin.cn/post/7459006547051315252)

**ESM（ECMAScript Modules）** 和 **CJS（CommonJS）** 是 JavaScript 中两种不同的模块系统。它们的设计目标、语法和使用场景有所不同。以下是它们的主要差异：

## 语法差异

### ESM

**导出模块**：使用 export 关键字。

```javascript
// 导出变量
export const name = "Alice";

// 导出函数
export function greet() {
  console.log("Hello!");
}

// 默认导出
export default function() {
  console.log("Default export");
}
```

**导入模块**：使用 import 关键字。

```javascript
// 导入命名导出
import { name, greet } from './module.js';

// 导入默认导出
import myFunction from './module.js';
```

### CJS

**导出模块**：使用 `module.exports` 或 `exports`。

```javascript
// 导出变量
exports.name = "Alice";

// 导出函数
module.exports.greet = function() {
  console.log("Hello!");
};

// 默认导出
module.exports = function() {
  console.log("Default export");
};
```

**导入模块**：使用 `require` 函数。

```javascript
// 导入模块
const myModule = require('./module');

// 使用导出的内容
console.log(myModule.name);
myModule.greet();
```

## 加载方式

### ESM

* **静态加载**：ESM 是静态的，模块的依赖关系在代码解析阶段就确定了。
* `import` 语句必须位于模块的顶层，不能动态加载。
* 支持静态分析，便于工具进行优化（如 Tree Shaking）。
* **异步加载**：ESM 支持异步加载模块（通过 `import()` 动态导入）。

```javascript
import('./module.js').then(module => {
module.greet();
});
```

### CJS

* **动态加载**：CJS 是动态的，模块的依赖关系在运行时确定。

* `require` 可以在代码的任何地方调用，甚至可以动态加载模块。
* 不支持静态分析，难以进行 Tree Shaking。

```javascript
if (condition) {
const myModule = require('./module');
myModule.greet();
}
```

## 运行环境

### ESM

* **浏览器**：ESM 是浏览器的原生模块系统，可以直接在浏览器中使用。

```html
<script type="module" src="app.js"></script>
```

* **Node.js**：从 Node.js 12 开始，ESM 得到了原生支持，但需要使用 `.mjs` 文件扩展名或在 `package.json` 中设置 `"type": "module"`。

```json
{
"type": "module"
}
```

### CJS

* **Node.js**：CJS 是 Node.js 的默认模块系统，广泛用于服务器端开发。
* **浏览器**：CJS 不是浏览器的原生模块系统，需要通过打包工具（如 Webpack、Browserify）转换为浏览器可用的代码。

## 性能

### ESM

* **静态分析**：ESM 支持静态分析，便于工具进行优化（如 Tree Shaking），减少最终打包体积。
* **异步加载**：ESM 支持异步加载模块，适合现代 Web 应用的按需加载需求。

### CJS

* **动态加载**：CJS 的动态加载特性使得静态分析和优化变得困难，可能导致打包体积较大。
* **同步加载**：CJS 是同步加载模块的，不适合浏览器环境中的按需加载需求。

## 互操作性

### ESM 导入 CJS

在 ESM 中可以导入 CJS 模块，但需要注意：

* CJS 模块的默认导出需要通过 `default` 属性访问。

```javascript
import cjsModule from './cjs-module.js';
console.log(cjsModule.default);
```

### CJS 导入 ESM

* 在 CJS 中无法直接导入 ESM 模块（Node.js 中需要使用 `import()` 动态导入）。

```javascript
import('./esm-module.mjs').then(module => {
console.log(module.default);
});
```

## 其他差异

### 顶层 `this`

* **ESM**：顶层 `this` 是 `undefined`。
* **CJS**：顶层 `this` 是 `exports` 对象。

### 严格模式

* **ESM**：默认启用严格模式。
* **CJS**：默认不启用严格模式，需要手动添加 `"use strict"`。

### 循环依赖

* **ESM**：支持循环依赖，但行为与 CJS 不同。
* **CJS**：支持循环依赖，但可能导致未完全初始化的模块。

## 总结对比

| 特性 | ESM（ECMAScript Modules） | CJS（CommonJS） |
| --- | --- | --- |
| **语法** | `import` / `export` | `require` / `module.exports` |
| **加载方式** | 静态加载，支持异步动态加载 | 动态加载 |
| **运行环境** | 浏览器原生支持，Node.js 12+ 支持 | Node.js 默认模块系统 |
| **性能** | 支持静态分析，适合 Tree Shaking | 动态加载，难以优化 |
| **互操作性** | 可以导入 CJS 模块 | 无法直接导入 ESM 模块 |
| **顶层 `this`** | `undefined` | `exports` 对象 |
| **严格模式** | 默认启用 | 默认不启用 |
| **循环依赖** | 支持，行为与 CJS 不同 | 支持，可能导致未完全初始化的模块 |

选择建议
----

* **ESM**：适合现代浏览器环境和 Node.js 12+，支持静态分析和异步加载，适合构建现代 Web 应用。
* **CJS**：适合 Node.js 环境，尤其是需要动态加载模块的场景。