# (持续更新)Spring使用笔记

## application.properties和application.yml

将resource文件夹下原有的application.properties文件删除，创建application.yml和application-dev.yml配置文件，SpringBoot底层会把application.yml文件解析为application.properties。

application.yml

```yml
spring:
  profiles:
    active: dev
```

application-dev.yml

```yml
server:
  port: 8080

spring:
  datasource:
    username: root
    password: 1234
    url: jdbc:mysql://localhost:3306/springboot?useUnicode=true&characterEncoding=utf-8&useSSL=true&serverTimezone=UTC
    driver-class-name: com.mysql.jdbc.Driver

mybatis:
  mapper-locations: classpath:mapping/*Mapper.xml
  type-aliases-package: com.example.entity
#showSql

logging:
  level:
    com:
      example:
        mapper : debug
```

在Spring Boot中多环境配置文件名需要满足application-{profile}.yml的格式，其中{profile}对应你的环境标识，比如：

* application-dev.yml：开发环境
* application-test.yml：测试环境
* application-prod.yml：生产环境

至于哪个具体的配置文件会被加载，需要在application.yml文件中通过spring.profiles.active属性来设置，其值对应{profile}值。
