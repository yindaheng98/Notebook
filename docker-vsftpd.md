# 在mattrayner/docker-lamp的1804版里加vsftpd

## 在Dockerfile里面

### 1804版的Dockerfile的25~30行左右👇

```dockerfile
# Install packages
ENV DEBIAN_FRONTEND noninteractive
RUN add-apt-repository -y ppa:ondrej/php && \
  apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 4F4EA0AAE5267A6C && \
  apt-get update && \
  apt-get -y upgrade && \
  apt-get -y install supervisor wget git apache2 php-xdebug libapache2-mod-php mysql-server php-mysql pwgen php-apcu php7.1-mcrypt php-gd php-xml php-mbstring php-gettext zip unzip php-zip curl php-curl vsftpd && \
  apt-get -y autoremove && \
  echo "ServerName localhost" >> /etc/apache2/apache2.conf
```

👆安装packages的时候加装一个vsftpd

### 40行左右👇

```dockerfile
#needed for vsftpd
RUN mkdir -p /var/run/vsftpd
```

👆建立vsftpd的安全目录`/var/run/vsftpd/empty`

为什么加这个？后面`vsftpd.conf`里面`secure_chroot_dir`有写这个文件夹，运行时用的

👇看原版`vsftpd.conf`对`secure_chroot_dir`的注释

>This option should be the name of a directory which is empty.  Also, the directory should not be writable by the ftp user. This directory is used as a secure chroot() jail at times vsftpd does not require filesystem access.

### 40~50行左右👇

```dockerfile
# Add image configuration and scripts
ADD supporting_files/start-apache2.sh /start-apache2.sh
ADD supporting_files/start-mysqld.sh /start-mysqld.sh
ADD supporting_files/start-vsftpd.sh /start-vsftpd.sh
ADD supporting_files/run.sh /run.sh
RUN chmod 755 /*.sh
ADD supporting_files/supervisord-apache2.conf /etc/supervisor/conf.d/supervisord-apache2.conf
ADD supporting_files/supervisord-mysqld.conf /etc/supervisor/conf.d/supervisord-mysqld.conf
ADD supporting_files/supervisord-vsftpd.conf /etc/supervisor/conf.d/supervisord-vsftpd.conf
ADD supporting_files/mysqld_innodb.cnf /etc/mysql/conf.d/mysqld_innodb.cnf
ADD supporting_files/vsftpd.conf /etc/vsftpd.conf
```

👆加了3个文件👇

* `start-vsftpd.sh`：supervisord里面要用的vsftpd启动脚本
* `supervisord-vsftpd.conf`：supervisord的配置文件，有了这个supervisord能开机启动vsftpd
* `vsftpd.conf`：vsftpd自己的配置文件

👇这几个文件长这样

`start-vsftpd.sh`👇

    exec vsftpd

`supervisord-vsftpd.conf`👇

    [program:vsftpd]
    command=/start-vsftpd.sh
    numprocs=1
    autostart=true
    autorestart=true

`vsftpd.conf`👇

    listen=YES
    listen_ipv6=NO
    anonymous_enable=NO
    local_enable=YES
    write_enable=YES
    local_umask=022
    dirmessage_enable=YES
    use_localtime=YES
    xferlog_enable=YES
    connect_from_port_20=YES
    secure_chroot_dir=/var/run/vsftpd/empty
    pam_service_name=vsftpd
    rsa_cert_file=/etc/ssl/certs/ssl-cert-snakeoil.pem
    rsa_private_key_file=/etc/ssl/private/ssl-cert-snakeoil.key
    ssl_enable=NO
    userlist_enable=YES
    userlist_deny=NO
    userlist_file=/etc/vsftpd.user_list

### 90行左右👇

```dockerfile
ENV FTP_USERNAME yindaheng98
ENV FTP_PASSWORD 8s50-55fn-wzxr
ADD supporting_files/create_vsftpd_users.sh /create_vsftpd_users.sh
RUN chmod 755 /*.sh
```

👆设置FTP用户名和密码，运行一个创建FTP用户的shell

`create_vsftpd_users.sh`内容👇

    useradd -g ftp -d /app -s /bin/bash -p ${FTP_PASSWORD} ${FTP_USERNAME}

👆加个用户到ftp用户组

    chown ftp /app

👆ftp用户组给个权限

    echo "${FTP_USERNAME}">>/etc/vsftpd.user_list

👆用户加准入名单

## TODO:

目前已经可以从外网连接了，输了用户名也能验证成功然后输入密码，但是输入了密码之后会报`530 Login incorrect`目前还不清楚是自己电脑的问题还是镜像没配好