# mattrayner/docker-lamp的1804版的dockerfile解读

```dockerfile
FROM phusion/baseimage
MAINTAINER Matthew Rayner <hello@rayner.io>
```

`FROM`基于哪个docker image

`MAINTAINER`维护者信息

```dockerfile
ENV REFRESHED_AT 2019-03-12

ENV DOCKER_USER_ID 501
ENV DOCKER_USER_GID 20

ENV BOOT2DOCKER_ID 1000
ENV BOOT2DOCKER_GID 50

ENV PHPMYADMIN_VERSION=4.8.5
```

`ENV`是在**镜像系统**内定义变量的指令。通过`ENV`定义的环境变量，会永久的保存到该镜像创建的任何容器中，可以被后面的所有指令中使用，包括后面运行用`ADD`引入到系统的外部sh文件也能看得见这个

```dockerfile
# Tweaks to give Apache/PHP write permissions to the app
RUN usermod -u ${BOOT2DOCKER_ID} www-data && \
    usermod -G staff www-data && \
    useradd -r mysql && \
    usermod -G staff mysql

RUN groupmod -g $(($BOOT2DOCKER_GID + 10000)) $(getent group $BOOT2DOCKER_GID | cut -d: -f1)
RUN groupmod -g ${BOOT2DOCKER_GID} staff
```

`RUN`在容器内运行指令，`$XXX`和`${XXX}`是调用`ENV`系统变量的方式

```dockerfile
# Install packages
ENV DEBIAN_FRONTEND noninteractive
RUN add-apt-repository -y ppa:ondrej/php && \
  apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 4F4EA0AAE5267A6C && \
  apt-get update && \
  apt-get -y upgrade && \
  apt-get -y install supervisor wget git apache2 php-xdebug libapache2-mod-php mysql-server php-mysql pwgen php-apcu php7.1-mcrypt php-gd php-xml php-mbstring php-gettext zip unzip php-zip curl php-curl && \
  apt-get -y autoremove && \
  echo "ServerName localhost" >> /etc/apache2/apache2.conf

# needed for phpMyAdmin
RUN ln -s /etc/php/7.1/mods-available/mcrypt.ini /etc/php/7.3/mods-available/ && \
  phpenmod mcrypt
```

这是在装php

```dockerfile
# Add image configuration and scripts
ADD supporting_files/start-apache2.sh /start-apache2.sh
ADD supporting_files/start-mysqld.sh /start-mysqld.sh
ADD supporting_files/run.sh /run.sh
```

`ADD [把哪个文件] [放到哪]`是把外部文件放进容器中，这里直接放了几个启动脚本在根目录

```dockerfile
RUN chmod 755 /*.sh
```

管理员模式运行刚才放进去那几个启动脚本

```dockerfile
ADD supporting_files/supervisord-apache2.conf /etc/supervisor/conf.d/supervisord-apache2.conf
ADD supporting_files/supervisord-mysqld.conf /etc/supervisor/conf.d/supervisord-mysqld.conf
```

放supervisord的配置文件。supervisord是一个nb的进程管理软件

这些.conf文件里面都写了启动命令的，然后run.sh脚本里面没有明显调用这几个文件而是只有一个`exec supervisord -n`。supervisord启动时会自动搜索配置文件

```dockerfile
ADD supporting_files/mysqld_innodb.cnf /etc/mysql/conf.d/mysqld_innodb.cnf
```

放mysqld的配置文件

```dockerfile
# Allow mysql to bind on 0.0.0.0
RUN sed -i "s/.*bind-address.*/bind-address = 0.0.0.0/" /etc/mysql/my.cnf
```

改MySQL的绑定端口使能外部连接

```dockerfile
# Set PHP timezones to Europe/London
RUN sed -i "s/;date.timezone =/date.timezone = Europe\/London/g" /etc/php/7.3/apache2/php.ini
RUN sed -i "s/;date.timezone =/date.timezone = Europe\/London/g" /etc/php/7.3/cli/php.ini
```

设php时区

```dockerfile
# Remove pre-installed database
RUN rm -rf /var/lib/mysql

# Add MySQL utils
ADD supporting_files/create_mysql_users.sh /create_mysql_users.sh
RUN chmod 755 /*.sh
```

又整了一个创建MySQL用户的脚本

```dockerfile
# Add phpmyadmin
RUN wget -O /tmp/phpmyadmin.tar.gz https://files.phpmyadmin.net/phpMyAdmin/${PHPMYADMIN_VERSION}/phpMyAdmin-${PHPMYADMIN_VERSION}-all-languages.tar.gz
RUN tar xfvz /tmp/phpmyadmin.tar.gz -C /var/www
RUN ln -s /var/www/phpMyAdmin-${PHPMYADMIN_VERSION}-all-languages /var/www/phpmyadmin
RUN mv /var/www/phpmyadmin/config.sample.inc.php /var/www/phpmyadmin/config.inc.php
```

装phpmyadmin

```dockerfile
# Add composer
RUN php -r "copy('https://getcomposer.org/installer', 'composer-setup.php');" && \
    php composer-setup.php && \
    php -r "unlink('composer-setup.php');" && \
    mv composer.phar /usr/local/bin/composer
```

不知道干嘛的，反之就是在容器里面又run了点php有关的东西

```dockerfile
ENV MYSQL_PASS:-$(pwgen -s 12 1)
```

这一条没找到引用在哪里

```dockerfile
# config to enable .htaccess
ADD supporting_files/apache_default /etc/apache2/sites-available/000-default.conf
RUN a2enmod rewrite
```

改apache的配置文件

```dockerfile
# Configure /app folder with sample app
RUN mkdir -p /app && rm -fr /var/www/html && ln -s /app /var/www/html
ADD app/ /app
```

创建/app目录，然后链接到/var/www/html下面，然后把外部文件中的app/文件夹里面的文件（那个欢迎界面index.php）加进去

```dockerfile
#Environment variables to configure php
ENV PHP_UPLOAD_MAX_FILESIZE 10M
ENV PHP_POST_MAX_SIZE 10M
```

这两个在run.sh里面用到了，看字面意思是POST和上传文件时的大小限制

```dockerfile
# Add volumes for the app and MySql
VOLUME  ["/etc/mysql", "/var/lib/mysql", "/app" ]
```

`VOLUME [文件夹A,文件夹B,...]`把指定的文件夹暴露给外部挂载，有了这个指令就可以在`docker run`的时候用`-v`挂载外部目录了。没暴露的文件夹不能挂载

```dockerfile
EXPOSE 80 3306
```

`EXPOSE 端口A 端口B ...`把指定的端口暴露给外部映射，有了这个指令就可以在`docker run`的时候用`-p`指定端口映射了。没暴露的端口不能映射

```dockerfile
CMD ["/run.sh"]
```

又运行了一下方才`ADD`进去的一个sh脚本

`CMD`是在容器内的控制台运行指令，这个会被`docker run [CONTAIN_ID] [指令]`中的[指令]覆盖
