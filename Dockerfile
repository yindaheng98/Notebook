FROM httpd:alpine

ARG PAGES=./theme/public
COPY $PAGES /usr/local/apache2/htdocs/