#!/bin/bash
git clone -b master https://github.com/yindaheng98/simple-blog.git
cp -r 学习笔记/. simple-blog/md
cd simple-blog
echo "开始编译"
tree
echo "$1" | docker login -u "yindaheng98" --password-stdin
docker build -t yindaheng98/simple-blog:latest .
docker push yindaheng98/simple-blog:latest
docker login --username yindaheng98
cd ..
rm -rf simple-blog