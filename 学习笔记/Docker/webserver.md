# 如何在Ubuntu上整一个LAMP

二话不说先装个vim换源👇（docker上的ubuntu没有预装vim还要自己下😅

    apt-get install vim
    vim /etc/apt/sources.list

在vim里面输这个👇

    :%s/archive.ubuntu.com/mirrors.aliyun.com/g
    :wq!

👆换源就要阿里云，网速杠杠的

完事不忘update👇

    apt-get update

嗯，确实很快

## MySQL

服务端客户端各整一个👇

    apt-get install mysql-server
    apt-get install mysql-client

改一哈配置文件👇把数据库编码弄成utf8

    vim  /etc/mysql/mysql.conf.d/mysqld.cnf

👆在后面加上这一段👇

    [mysqld]
    default-storage-engine=INNODB
    character-set-server=utf8
    collation-server=utf8_general_ci
    [client]
    default-character-set=utf8

启动！👇

    service mysql start

### 如果要从外网连接

MySQL的默认绑定端口是127.0.0.1，从外面连不上，如果要从外面连的话要先要改一下👇

    vim /etc/mysql/mysql.conf.d/mysqld.cnf

👆里面的`bind-address=127.0.0.1`改成`bind-address=0.0.0.0`

然后在MySQL里面搞一个外网能连的root账户

```SQL
GRANT ALL PRIVILEGES ON *.* TO 'root'@'%' IDENTIFIED BY 'password';
flush privileges;
```

就可以用这个root账户从外网连了

## Apache和PHP

装Apache2👇不多bb

    apt-get install apache2

加php软件库👇

    apt-get install software-properties-common
    add-apt-repository ppa:ondrej/php

然后装php7.0👇不多bb

    apt-get update
    apt-get install -y php7.0

装一堆php组件👇不多bb

    apt-get install php7.0-mysql php7.0-curl php7.0-json php7.0-cgi php7.0-xsl php7.0-mbstring php7.0-fpm php7.0-dev

Apache服务器整一个php模式👇不多bb

    apt-get install libapache2-mod-php7.0

然后开个FPM👇

    a2enmod proxy_fcgi setenvif
    a2enconf php7.0-fpm

然后重启

    service apache2 restart

装个phpmyadmin👇

    apt-get install phpmyadmin

把phpmyadmin文件夹连到php默认文件夹里面👇

    ln -s /usr/share/phpmyadmin /var/www/html

## FTP

加个用户👇设个密码

    useradd -d /var/www/html -s /bin/bash yindaheng98
    passwd yindaheng98

给个权限

    chown yindaheng98:yindaheng98 /var/www/html

装FTP服务器👇不多bb

    apt install vsftpd

调设置👇

    vim /etc/vsftpd.conf

在这个文件里面👆改这些东西👇

    write_enable =YES
    local_umask=022
    userlist_enable=YES
    userlist_deny=NO
    userlist_file=/etc/vsftpd.user_list

用户名加入`/etc/vsftpd.user_list`文件👇

    vim /etc/vsftpd.user_list

👆在文件尾加上`yindaheng98`

重启更新设置👇

    systemctl restart vsftpd.service

然后就能在22端口用yindaheng98登FTP了