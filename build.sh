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

#编译一个Docker镜像
echo "$1" | docker login -u "yindaheng98" --password-stdin
docker build --build-arg 'PAGES=./theme/blogs/.vuepress/public' -t yindaheng98/yindaheng98.github.io:latest .
docker push yindaheng98/yindaheng98.github.io:latest