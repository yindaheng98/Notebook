# Apache2反向代理的配置方法

——基于mattrayner/lamp:latest-1804

查看apache2的错误记录

    cat /var/log/apache2/error.log

## 装mod

    a2enmod proxy
    a2enmod proxy_http

## 改配置文件

    vim /etc/apache2/mods-enabled/proxy.conf

以本地8080端口上的jetty为例，把/database请求转发到本地8080端口的/proxy去

在\<IfModule mod_proxy.c>下面加上👇

    ProxyPass /database http://localhost:8080/proxy
    ProxyPassReverse /database http://localhost:8080/proxy

重启

    service apache2 restart

之后当浏览器浏览👇

    http://[站点host]:80/database/index.jsp

时会收到👇

    http://[站点host]:8080/proxy/index.jsp

的内容