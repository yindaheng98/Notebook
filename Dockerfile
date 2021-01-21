FROM httpd:alpine

ARG PAGES=./theme/blogs/.vuepress/public
COPY $PAGES /usr/local/apache2/htdocs/