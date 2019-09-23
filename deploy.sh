git config user.name "TravisCI"
git config user.email "yindaheng98@163.com"
set -e
git clone https://github.com/yindaheng98/yindaheng98.github.io.git
cp -r 学习笔记 yindaheng98.github.io/source/_post
cd yindaheng98.github.io
git add -A
git commit -m "TravisCI push"
git push -u https://$1@github.com/yindaheng98/yindaheng98.github.io.git HEAD:master --force