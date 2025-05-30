# 【翻译】 Code-splitting for libraries—bundling for npm with Rollup 1.0

原文来自 [Medium](https://levelup.gitconnected.com/code-splitting-for-libraries-bundling-for-npm-with-rollup-1-0-2522c7437697), 由 Deepseek 翻译.

[Lukas Taegert](https://medium.com/@lukastaegert?source=post_page---byline--2522c7437697---------------------------------------)

With the recent [1.0 release](https://github.com/rollup/rollup/releases/tag/v1.0.0) of Rollup, code-splitting has become a first-class feature of the notorious JavaScript module bundler. Here I want to make the case for why this is a game-changer not only for the frontend but for libraries as well. In this article you will learn:

- How to bundle a library with Rollup (and why you would want this), including some best practices, but most importantly
- How Rollup code-splitting works and when and how to use it for libraries

# Our library project

For this example, we will be building a collection of awesome utilities to convert strings to upper and lower case that we want to publish to npm under the name `fancy-case` (note: at the time of this writing, there was no library with that name).

> Write your sources as ES modules

I definitely recommend using [ECMAScript modules](http://exploringjs.com/es6/ch_modules.html) for writing a library:

- Modern bundlers, including but not limited to Rollup itself, will produce more efficient code when including your library
- ES modules can be easily converted to Node’s CommonJS format. Converting CJS to ES modules on the other hand is more difficult, does not provide full feature parity and leads to less efficient code.

We will see below how to convert our project to CJS using Rollup. For now, here are the project files:

<iframe src="https://levelup.gitconnected.com/media/0e7a7550a662e9350918f3e62d94fe14" allowfullscreen frameborder="0" scrolling="no"></iframe>

Following good programming practices, we write small modules containing functions that “do one thing only”.

> Aim for small modules

Small modules not only make it easy to see at a glance what a module is about but will also make our code-splitting build more efficient later because Rollup has more options how to group files into chunks. Admittedly, we are taking this somewhat to the extreme here.

This is what our project contains so far:

- `main.js` is the main entry module exporting the three utility functions `upper`, `lower`, and `upperFirst` that represent the public API of our library.
- The utility functions themselves are defined in separate modules of the same name.
- The remaining modules contain code that is shared between the utility functions.

# Different distribution formats

Once we have decided to make our library available to others via npm, we should take a moment to consider how it should be possible to import it. There are indeed quite a few options today:

## CommonJS module for Node

This is probably the most important target. This allows Node users and legacy bundlers to import your library as a [CommonJS module](http://wiki.commonjs.org/wiki/Modules/1.1.1) via

```
const fancyCase = require('fancy-case');
console.log(fancyCase.upper('some Text'));
```

## Single bundle to be used in a script tag

The “traditional way” of distributing JavaScript may still be interesting for small, hand-crafted sites with minimal setup. The bundle creates a global variable via which its exports can be accessed.

```
<script src="fancy-case.js"></script>
<script>
    console.log(fancyCase.upper('some Text'));
</script>
```

## AMD module to be used with an AMD loader

There are still quite a few [AMD/RequireJS](https://requirejs.org/) based projects out there. We can distribute a file that can itself be used as a dependency of an AMD module.

```
define(['fancy-case.js'], function (fancyCase) {
    console.log(fancyCase.upper('some Text'));
});
```

## ES module for modern bundlers

[ECMAScript modules](http://exploringjs.com/es6/ch_modules.html) are now the official, standardized JavaScript module format.

> Provide an ES module version for optimized browser bundles

ES modules support superior static analysis, which in turn enables bundlers to better optimize the generated code using techniques such as [scope-hoisting](https://medium.com/adobetech/optimizing-javascript-through-scope-hoisting-2259ef7f5994) and [tree-shaking](https://medium.com/@Rich_Harris/tree-shaking-versus-dead-code-elimination-d3765df85c80). For that, they are the preferred format for modern bundlers. Our module can be consumed via

```
import {upper} from 'fancy-case';
console.log(upper('some Text'));
```

## Direct imports for CJS or ESM consumers

An emerging new pattern especially for libraries with many independent utility functions is to allow users to import independent parts of the library from separate files. Node users could write

```
const upper = require('fancy-case/cjs/upper');
console.log(upper('some Text'));
```

while ESM consumers could write

```
import upper from 'fancy-case/esm/upper';
console.log(upper('some Text'));
```

> Provide a way to directly import independent parts of your library

Direct imports have several advantages:

- Node needs to load and parse less code, which leads to quicker startup time and reduced memory consumption.
- Bundlers need to analyze less code, which makes bundling faster.
- You do not need good tree-shaking support in the bundler to avoid dead code. Often, utility libraries apply fancy transformations to their exports which render tree-shaking algorithms useless; this can be easily avoided using this technique.

Note however that this can also lead to more bundled code if some modules import the whole library while others directly import some functions. Below we will show how to use Rollup’s new code-splitting to avoid this in an elegant way.

# Publishing monolithic bundles

For now, we will focus on the first four publishing targets, i.e. CJS, script tag, AMD and ESM. To that end, let us prepare our project for publishing:

```
mkdir fancy-casecd fancy-casenpm init --yesgit clone https://gist.github.com/lukastaegert/e9c6c04b8f96adc562a70c096c3e7705 srcnpm install --save-dev rollup
```

This will create a `package.json` file for our project, put our sample files into a `src` folder and install Rollup. Rollup supports a special output format called a [“Universal Module Definition”](https://github.com/umdjs/umd), which simultaneously supports the CJS, script tag, and ESM use cases. To create it, add a new file called `rollup.config.js` to the root of your project:

```
export default {
    input: 'src/main.js',
    output: {
        file: 'umd/fancy-case.js',
        format: 'umd',
        name: 'fancyCase'
    }
};
```

This instructs Rollup to start with `src/main.js` and bundle it together with all its dependencies into a UMD bundle in `umd/fancy-case.js`. The `name` option tells Rollup which global variable to create when the bundle is used in a script tag, in this case `fancyCase`. This variable will **only** be created if this bundle is not consumed in a Node or AMD context.

Now if you run

```
npx rollup --config
```

from your project’s root, this will pick up our config file and create a new folder named “umd” that contains our UMD bundle. You can check out the result on Rollup’s website: [https://rollupjs.org/repl?gist=e9c6c04b8f96adc562a70c096c3e7705](https://rollupjs.org/repl?gist=e9c6c04b8f96adc562a70c096c3e7705)

A monolithic bundle merges all modules together

If you switch to the UMD tab in the output section of the website and enter the correct `global` variable name, you will see all your files compressed together, surrounded by a wrapper like this

```
(function (global, factory) {
  typeof exports === 'object' && typeof module !== 'undefined' ? factory(exports) :
  typeof define === 'function' && define.amd ? define(['exports'], factory) :
  (global = global || self, factory(global.fancyCase = {}));
}(this, function (exports) { 'use strict';

  // ... all your bundled code  Object.defineProperty(exports, '__esModule', { value: true });
}));
```

This wrapper will analyze the current runtime environment and provide the exports of your module in a convenient way. Note this line:

```
Object.defineProperty(exports, '__esModule', { value: true });
```

When attempting to `import default` our UMD bundle in an ESM context, modern bundlers add interoperability code that checks for the presence of the `__esModule` property. If it is present, then the default import will not provide the whole exports object but just the `default` property of that object.

As we are going to create a dedicated ESM bundle anyway which should be used in these cases, we can consider skipping this line by adding `esModule: false` to the `output` section of the config file. You can also get a more optimized wrapper by creating dedicated builds for the formats “cjs” (Node), “amd” or “iife” (script tag), check out the corresponding tabs on the website.

Note that apart from the wrapper code in contrast to most other bundlers, no dedicated runtime environment to resolve imports is added. Apart from its configurability, this is another reason why Rollup is quite popular with libraries creators that strive to create efficient bundles with minimal overhead.

> UMD bundles should be minified

As especially for the AMD and script tag use, this bundle is meant to be run in the browser unmodified, we should go the whole way and minify it. To do that, I recommend using [TerserJS](https://github.com/terser-js/terser), which is a fork of the more well-known [UglifyJS](http://lisperator.net/uglifyjs/) that supports modern ES2015+ JavaScript code. After installing the necessary dependency

```
npm install --save-dev rollup-plugin-terser
```

you should modify your `rollup.config.js` like this:

```
import {terser} from 'rollup-plugin-terser';export default {
    input: 'src/main.js',
    plugins: [terser()],
    output: {
        file: 'umd/fancy-case.js',
        format: 'umd',
        name: 'fancyCase',
        esModule: false
    }
};
```

As mentioned above, we also want to provide a dedicated ESM bundle. This could be done by adding a second `output` to our configuration but as this bundle is meant to be consumed by other bundlers anyway and does not profit from minification (in fact this will make it harder to hunt for bugs), I rather recommend to forgo this and export two separate configurations:

```
import {terser} from 'rollup-plugin-terser';export default [\
    {\
        input: 'src/main.js',\
        plugins: [terser()],\
        output: {\
            file: 'umd/fancy-case.js',\
            format: 'umd',\
            name: 'fancyCase',\
            esModule: false\
        }\
    },\
    {\
        input: 'src/main.js',\
        output: {\
            file: 'esm/index.js',\
            format: 'esm'\
        }\
    }\
];
```

To publish our module, we need to make sure that importers of our library receive the right file and that this file is built from the latest sources upon publishing. To do that, modify our `packjage.json` file like this:

```
{
  "name": "fancy-case",
  "version": "1.0.0",
  "main": "umd/fancy-case.js",
  "module": "esm/index.js",
  "scripts": {
    "prepare": "rollup --config"
  },
  "files": [\
    "esm/*",\
    "umd/*"\
  ],
  "devDependencies": {
    "rollup": "^1.1.0",
    "rollup-plugin-terser": "^4.0.2"
  }
}
```

> Add both “main” and “module” fields

The `main` field makes sure that Node users using `require` will be served the UMD version. The `module` field is not an official npm feature but a common convention among bundlers to designate how to import an ESM version of our library.

> Use “files” to include your bundles

The `files` field makes sure that besides some default files, only our designated bundles are distributed via npm excluding sources, test files etc. This will keep the `node_modules` folder of your users small and make `npm install` faster. You could also create an `.npmignore` file for a similar effect but with a “deny-listing” instead of an “allow-listing” approach. To my experience, “files” is easier to maintain, though.

> Create a “prepare” script

The `prepare` script is a special script that will be run by npm each time we run `npm install` or `npm publish`. It also makes it possible to directly install branches from Github for testing purposes via

```
npm install <user>/<repository>#<branch>
```

Now we can just run `npm publish` and the first version of our library will be available for everyone via `npm install fancy-case`!

> Do a publishing dry-run with “npm pack”

If you are unsure whether you configured everything correctly, you can run `npm pack` first to get a tarball containing everything that will be sent to npm upon publishing and inspect it. If you are slightly paranoid, you can even use this tarball for automated testing.

# Publishing optimized chunks

As noted above, it can be very beneficial to our users if we provide direct imports for independent parts of our library. One way of doing this could be to just distribute the source files together with our library and instruct our users to import from there.

This can lead to nasty issues, though, if different parts of the users’s code import our library in different ways. Imagine one module imports the `upper` function from `"fancy-case"` while another imports it from `"fancy-case/src/upper"`. Even though it is technically the same code, these are now two very distinct functions and `upper` will end up twice in the user’s code.

This may not sound too problematic but imagine what happens if we store some persistent state in a variable next to the `upper` function (definitely not a recommended practice, but it happens) or if the user relies on comparing references to our `upper` function. Suddenly we are facing a myriad of weird, hard-to-track bugs. Also, the untouched source code did not benefit from any optimizations such as scope-hoisting or tree-shaking or any transformations applied to the code via plugins like [rollup-plugin-babel](https://github.com/rollup/rollup-plugin-babel).

> Mark independent parts of your library as additional entry modules

Rollup 1.0 offers a simple but powerful solution: You can designate the independent parts of your library as additional entry points. Before we change our project, take a look at the result in the REPL: [https://rollupjs.org/repl?gist=e9c6c04b8f96adc562a70c096c3e7705&entry=lower.js,upper.js,upperFirst.js](https://rollupjs.org/repl?gist=e9c6c04b8f96adc562a70c096c3e7705&entry=lower.js%2Cupper.js%2CupperFirst.js)

A code-split bundle will group modules into chunks

You see that our originally eight modules have been reduced to five chunks, one for each entry module and an additional chunk that is imported by several of the other chunks. Depending on the format you look at, the chunks simply `import` or `require` each other without any additional management code added or any code duplication.

To avoid duplications and thus the potential issues with duplicated state or references mentioned above, Rollup applies a “coloring” algorithm that assigns an individual color to each entry module and then traverses the module graph to assign each module the “mixed” color of all entry points that depend on it.

In our example, both the red entry module `upper.js` as well as the blue entry module `lower.js` depend on `constants.js` and `shiftChar.js` so those are assigned to a new purple chunk. `main.js` and `upperFirst.js` only depend on other entry modules and thus do not further change the coloring.

This is how you can change your `rollup.config.js` to produce code-split builds for CJS and ESM consumers:

```
import {terser} from 'rollup-plugin-terser';export default [\
    {\
        input: 'src/main.js',\
        plugins: [terser()],\
        output: {\
            file: 'umd/fancy-case.js',\
            format: 'umd',\
            name: 'fancyCase',\
            esModule: false\
        }\
    },\
    {\
        input: {\
            index: 'src/main.js',\
            upper: 'src/upper.js',\
            lower: 'src/lower.js',\
            upperFirst: 'src/upperFirst.js'\
        },\
        output: [\
            {\
                dir: 'esm',\
                format: 'esm'\
            },\
            {\
                dir: 'cjs',\
                format: 'cjs'\
            }\
        ]\
    }\
];
```

As you can see, we now provide an object as `input` where the properties correspond to the generated entry chunks while the values correspond to their entry modules. Also instead of specifying `file`, we now define an output `dir` where all chunks are placed. If you want to adjust the naming scheme and placement of chunks, take a look a the `entryFileNames` and `chunkFileNames` output options. To inspect the result, just run

```
npm run prepare
```

> Write JavaScript code in your config file for advanced configurations

If you have a lot of entry points, instead of specifying them individually, note that your config file is a JavaScript file that can import anything from `node_modules` as well as the built-in Node libraries. Thus you can for instance `import fs from 'fs'` and then use `fs.readDirSync` to build an object of entry modules from a directory.

Now everything that remains to be done is to adjust your `package.json` file to include the new files and import targets and then publish the result:

```
{
  "name": "fancy-case",
  "version": "1.0.0",
  "main": "cjs/index.js",
  "module": "esm/index.js",
  "scripts": {
    "prepare": "rollup --config"
  },
  "files": [\
    "cjs/*",\
    "esm/*",\
    "umd/*"\
  ],
  "devDependencies": {
    "rollup": "^1.1.0",
    "rollup-plugin-terser": "^4.0.2"
  }
}
```

## Using .mjs

Another emerging pattern is to build dual-mode packages where CJS and ES modules are placed next to each other where the CJS files sport the `.js` extension while the ESM files have an `.mjs` extension. This can be easily achieved by using the same `dir` for both outputs while adding

```
entryFileNames: [name].mjs
```

to the ESM output options. Note however that this can change bundling behaviour in some bundlers when using external dependencies and should be done at your own risk.

## Avoiding the waterfall

One thing you may notice is that Rollup seems to add additional “empty” imports to some chunks. E.g. this is what your `main.js` chunk will look like in the ESM version:

```
import '../chunk-59d826da.js';
export { default as upper } from '../upper.js';
export { default as lower } from '../lower.js';
export { default as upperFirst } from '../upperFirst.js';
```

So why is there an additional import for our shared chunk at the top? Let us take a look what happens when we want to run `main.js` without this addition:

1. load and parse `main.js`
2. after parsing, we know that we also need `upper.js`, `lower.js` and `upperFirst.js` so we load and parse those
3. after parsing any of them, we know that we also need our shared chunk so we load and parse that one as well
4. run everything

With the added import, step 3 no longer needs to wait for step 2 as all dependencies are known once an entry chunk has been parsed. Thus, Node user’s will be able to load your library faster while bundlers will profit from reduced module discovery time.

## External dependencies

Even though our library is self-sufficient, it is easy to add external dependencies as well. Note however that in this case, you should add the `external` option providing an array of all external dependencies, otherwise you will receive warnings from Rollup. To have external dependencies in a UMD or IIFE build, you also need to specify under which global variable names the external dependencies can be found when using the library in a script tag via the `globals` output option.

# Conclusion

Via a single central configuration file, Rollup makes it possible to provide many different bundle formats simultaneously.

Code-splitting with Rollup provides a new way of bundling libraries that can prevent many pit-falls of having several import targets exposed to the user while giving you full configurability and flexibility. Expect Rollup to provide more code optimizations in the future that both code-split and monolithic bundles will profit from.

Of course, you can use Rollup as well for bundling web apps ( [check out code-splitting via dynamic imports!](https://rollupjs.org/repl?shareable=JTdCJTIybW9kdWxlcyUyMiUzQSU1QiU3QiUyMm5hbWUlMjIlM0ElMjJtYWluLmpzJTIyJTJDJTIyY29kZSUyMiUzQSUyMiUyRiolMjBEWU5BTUlDJTIwSU1QT1JUUyU1Q24lMjAlMjAlMjBSb2xsdXAlMjBzdXBwb3J0cyUyMGF1dG9tYXRpYyUyMGNodW5raW5nJTIwYW5kJTIwbGF6eS1sb2FkaW5nJTVDbiUyMCUyMCUyMHZpYSUyMGR5bmFtaWMlMjBpbXBvcnRzJTIwdXRpbGl6aW5nJTIwdGhlJTIwaW1wb3J0JTIwbWVjaGFuaXNtJTVDbiUyMCUyMCUyMG9mJTIwdGhlJTIwaG9zdCUyMHN5c3RlbS4lMjAqJTJGJTVDbmlmJTIwKGRpc3BsYXlNYXRoKSUyMCU3QiU1Q24lNUN0aW1wb3J0KCcuJTJGbWF0aHMuanMnKS50aGVuKGZ1bmN0aW9uJTIwKG1hdGhzKSUyMCU3QiU1Q24lNUN0JTVDdGNvbnNvbGUubG9nKG1hdGhzLnNxdWFyZSg1KSklM0IlNUNuJTVDdCU1Q3Rjb25zb2xlLmxvZyhtYXRocy5jdWJlKDUpKSUzQiU1Q24lNUN0JTdEKSUzQiU1Q24lN0QlMjIlMkMlMjJpc0VudHJ5JTIyJTNBdHJ1ZSU3RCUyQyU3QiUyMm5hbWUlMjIlM0ElMjJtYXRocy5qcyUyMiUyQyUyMmNvZGUlMjIlM0ElMjJpbXBvcnQlMjBzcXVhcmUlMjBmcm9tJTIwJy4lMkZzcXVhcmUuanMnJTNCJTVDbiU1Q25leHBvcnQlMjAlN0JkZWZhdWx0JTIwYXMlMjBzcXVhcmUlN0QlMjBmcm9tJTIwJy4lMkZzcXVhcmUuanMnJTNCJTVDbiU1Q25leHBvcnQlMjBmdW5jdGlvbiUyMGN1YmUlMjAoeCUyMCklMjAlN0IlNUNuJTVDdHJldHVybiUyMHNxdWFyZSh4KSUyMColMjB4JTNCJTVDbiU3RCUyMiUyQyUyMmlzRW50cnklMjIlM0FmYWxzZSU3RCUyQyU3QiUyMm5hbWUlMjIlM0ElMjJzcXVhcmUuanMlMjIlMkMlMjJjb2RlJTIyJTNBJTIyZXhwb3J0JTIwZGVmYXVsdCUyMGZ1bmN0aW9uJTIwc3F1YXJlJTIwKCUyMHglMjApJTIwJTdCJTVDbiU1Q3RyZXR1cm4lMjB4JTIwKiUyMHglM0IlNUNuJTdEJTIyJTJDJTIyaXNFbnRyeSUyMiUzQWZhbHNlJTdEJTVEJTJDJTIyb3B0aW9ucyUyMiUzQSU3QiUyMmZvcm1hdCUyMiUzQSUyMmNqcyUyMiUyQyUyMm5hbWUlMjIlM0ElMjJteUJ1bmRsZSUyMiUyQyUyMmFtZCUyMiUzQSU3QiUyMmlkJTIyJTNBJTIyJTIyJTdEJTJDJTIyZ2xvYmFscyUyMiUzQSU3QiU3RCU3RCUyQyUyMmV4YW1wbGUlMjIlM0ElMjIwMCUyMiU3RA%3D%3D)), but this shall not be the focus of this article.

I hope this provides a nice introduction to the topic, feedback is very much welcome!
