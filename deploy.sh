#!/bin/bash
git config user.name "TravisCI"
git config user.email "yindaheng98@163.com"
cd yindaheng98.github.io/public
set -e
git init
git add -A
git commit -m 'TravisCI Deploy'
set -e
git push -u https://$1@github.com/yindaheng98/yindaheng98.github.io.git HEAD:master --force