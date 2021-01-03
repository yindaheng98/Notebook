#!/bin/bash
git clone -b master https://github.com/yindaheng98/vuepress-blog.git ./theme
cp -r 学习笔记/. ./theme/blogs
cp -r 学习笔记/. ./theme/blogs/.vuepress/public
cd ./theme
echo "开始编译"
tree
npm install
npm run build
cd ..