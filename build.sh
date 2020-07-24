#!/bin/bash
git clone -b dev https://github.com/yindaheng98/yindaheng98.github.io.git
rm -rf yindaheng98.github.io/blogs
cp -r 学习笔记 yindaheng98.github.io/blogs
cd yindaheng98.github.io
cp -r blogs .vuepress/public
echo "开始编译"
tree
npm install
npm run build
cd ..