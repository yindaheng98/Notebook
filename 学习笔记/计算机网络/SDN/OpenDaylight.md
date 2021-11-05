# OpenDaylight 调研

调研项|结果
-|-
核心控制模块使用的语言|Java
开源|是
提供网页以图形化的方式显示网络硬件拓扑结构|是
提供网页以图形化的方式显示网络逻辑拓扑结构|否
支持OpenFlow 1.3中的Meter表限速功能|是
带有网络虚拟化的功能|否

## 如何用虚拟机或Docker进行一键部署

OpenDaylight主要模块使用Java写成，因此只要在虚拟机或Docker内部署Java运行环境即可部署OpenDaylight。

Dockerfile案例：
```Dockerfile
FROM debian:stretch

RUN apt-get update && \
    apt-get install -y wget openjdk-8-jdk-headless procps net-tools && \
    mkdir /opt/opendaylight && \
    wget https://nexus.opendaylight.org/content/repositories/public/org/opendaylight/integration/karaf/0.8.0/karaf-0.8.0.tar.gz && \
    tar -xvzf karaf-0.8.0.tar.gz -C /opt/opendaylight --strip-components 1

WORKDIR /opt/opendaylight

EXPOSE 8181 8101

CMD ./bin/karaf
```

Dockerfile案例：
```Dockerfile
FROM anapsix/alpine-java:8_jdk

MAINTAINER Guillaume Lefevre <gelefevre@octo.com>

RUN mkdir /odl
WORKDIR /odl

RUN apk add --no-cache gcc g++ make libc-dev python-dev openssl && \
    apk add maven --update-cache --repository http://dl-3.alpinelinux.org/alpine/edge/community/ && \
    wget https://nexus.opendaylight.org/content/groups/public/org/opendaylight/integration/distribution-karaf/0.5.0-Boron/distribution-karaf-0.5.0-Boron.tar.gz && \
    tar -xvzf distribution-karaf-0.5.0-Boron.tar.gz && \
    apk del gcc make python-dev libc-dev g++ maven && \
    rm -rf /var/cache/apk/*

EXPOSE 8181 6633 8101

CMD ./distribution-karaf-0.5.0-Boron/bin/karaf
```

## 提供了哪些操作方式

OpenDaylight有两种主要的开发方式：
* 写插件：按照OpenDaylight MD-SAL模块标准编写Java插件，直接调用OpenDaylight Java API
* 调用HTTP API：写脚本调用OpenDaylight提供的REST API

## 如何查看流表

文档地址：https://docs.opendaylight.org/projects/openflowplugin/en/latest/users/operation.html

向OpenDaylight提供的REST API发GET请求：`http://<OpenDaylight HTTP API所在地址>/restconf/operational/opendaylight-inventory:nodes/node/<SDN交换机名称>/table/<流表ID>/flow/<流ID>`

返回值格式和下发流表时PUT数据的格式相同。

## 如何下发流表

文档地址：https://docs.opendaylight.org/projects/openflowplugin/en/latest/users/operation.html

向OpenDaylight提供的REST API发PUT请求：`http://<OpenDaylight HTTP API所在地址>/restconf/config/opendaylight-inventory:nodes/node/<SDN交换机名称>/table/<流表ID>/flow/<流ID>`

其中PUT数据的格式为：
```xml
<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<flow
    xmlns="urn:opendaylight:flow:inventory">
    <strict>true</strict>
    <instructions>
        <instruction>
            <order>设定instruction的执行顺序</order>
            <apply-actions>
                <action>
                  <order>设定action的执行顺序</order>
                    <设定执行何种操作/>
                </action>
            </apply-actions>
        </instruction>
    </instructions>
    <table_id>下发到哪个流表</table_id>
    <id>下发到流表中的哪条流</id>
    <cookie_mask>10</cookie_mask>
    <out_port>10</out_port>
    <installHw>false</installHw>
    <out_group>2</out_group>
    <match>
        匹配条件
    </match>
    <hard-timeout>0</hard-timeout>
    <cookie>10</cookie>
    <idle-timeout>0</idle-timeout>
    <flow-name>给这个流取个名字</flow-name>
    <priority>优先级</priority>
    <barrier>false</barrier>
</flow>
```

例如，下面这个流表表示“将所有来自10.0.0.1/24网段的数据包的包头的标上vlan id为123后洪泛出去”，并且将该规则设置为“2号流表的111号规则”：
```xml
<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<flow
    xmlns="urn:opendaylight:flow:inventory">
    <strict>true</strict>
    <instructions>
        <instruction>
            <order>1</order>
            <apply-actions>
                <action>
                  <order>1</order>
                    <set-field>
                        <vlan-match>
                            <vlan-id>123</vlan-id>
                            <vlan-id-present>123</vlan-id-present>
                        </vlan-match>
                    </set-field>
                </action>
                <action>
                  <order>2</order>
                    <flood-all-action/>
                </action>
            </apply-actions>
        </instruction>
    </instructions>
    <table_id>2</table_id>
    <id>111</id>
    <cookie_mask>10</cookie_mask>
    <out_port>10</out_port>
    <installHw>false</installHw>
    <out_group>2</out_group>
    <match>
        <ethernet-match>
            <ethernet-type>
                <type>2048</type>
            </ethernet-type>
        </ethernet-match>
        <ipv4-destination>10.0.0.1/24</ipv4-destination>
    </match>
    <hard-timeout>0</hard-timeout>
    <cookie>10</cookie>
    <idle-timeout>0</idle-timeout>
    <flow-name>FooXf22</flow-name>
    <priority>2</priority>
    <barrier>false</barrier>
</flow>
```

## 如何查看Meter表

文档地址：https://docs.opendaylight.org/projects/openflowplugin/en/latest/users/operation.html

向OpenDaylight提供的REST API发GET请求：`http://<OpenDaylight HTTP API所在地址>/restconf/operational/opendaylight-inventory:nodes/node/<SDN交换机名称>/meter/<meter表ID>`

返回值格式和下发Meter表时PUT数据的格式类似，但会多返回一个流量统计项`<meter-band-stats>`。例如：
```xml
<?xml version="1.0"?>
<meter xmlns="urn:opendaylight:flow:inventory">
  <meter-id>2</meter-id>
  <flags>meter-kbps</flags>
  <meter-statistics xmlns="urn:opendaylight:meter:statistics">
    <packet-in-count>0</packet-in-count>
    <byte-in-count>0</byte-in-count>
    <meter-band-stats>
      <band-stat>
        <band-id>0</band-id>
        <byte-band-count>0</byte-band-count>
        <packet-band-count>0</packet-band-count>
      </band-stat>
    </meter-band-stats>
    <duration>
      <nanosecond>364000000</nanosecond>
      <second>114</second>
    </duration>
    <meter-id>2</meter-id>
    <flow-count>0</flow-count>
  </meter-statistics>
  <meter-band-headers>
    <meter-band-header>
      <band-id>0</band-id>
      <band-rate>100</band-rate>
      <band-burst-size>0</band-burst-size>
      <meter-band-types>
        <flags>ofpmbt-drop</flags>
      </meter-band-types>
      <drop-burst-size>0</drop-burst-size>
      <drop-rate>100</drop-rate>
    </meter-band-header>
  </meter-band-headers>
</meter>
```

## 如何下发Meter表

文档地址：https://docs.opendaylight.org/projects/openflowplugin/en/latest/users/operation.html

向OpenDaylight提供的REST API发PUT请求：`http://<OpenDaylight HTTP API所在地址>/restconf/config/opendaylight-inventory:nodes/node/<SDN交换机名称>/meter/<meter表ID>`

```xml
<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<meter xmlns="urn:opendaylight:flow:inventory">
    <flags>meter-kbps</flags>
    <meter-band-headers>
        <meter-band-header>
            <band-id>0</band-id>
            <drop-rate>256</drop-rate>
            <drop-burst-size>512</drop-burst-size>
            <meter-band-types>
                <flags>ofpmbt-drop</flags>
            </meter-band-types>
        </meter-band-header>
    </meter-band-headers>
    <meter-id>2</meter-id>
    <meter-name>Foo</meter-name>
</meter>
```

## 如何获取全网实际的拓扑结构（SDN交换机之间及其与终端之间的硬件连接情况）

向其HTTP API发起请求`http://<OpenDaylight HTTP API所在地址>/restconf/operational/network-topology:network-topology`

返回值形如：
```xml
<topology xmlns="urn:TBD:params:xml:ns:yang:network-topology">
    <topology-id>flow: 1</topology-id>
    <node>
        <node-id>openflow: 2</node-id>
        <termination-point>
            <tp-id>openflow: 2: 2</tp-id>
            <inventory-node-connector-ref xmlns="urn:opendaylight:model:topology:inventory" xmlns:a="urn:opendaylight:inventory">/a:nodes/a:node[a:id='openflow: 2']/a:node-connector[a:id='openflow: 2: 2']
        </inventory-node-connector-ref>
        </termination-point>
        <termination-point>
            <tp-id>openflow: 2: 1</tp-id>
            <inventory-node-connector-ref xmlns="urn:opendaylight:model:topology:inventory" xmlns:a="urn:opendaylight:inventory">/a:nodes/a:node[a:id='openflow: 2']/a:node-connector[a:id='openflow: 2: 1']
        </inventory-node-connector-ref>
        </termination-point>
        <termination-point>
            <tp-id>openflow: 2:LOCAL</tp-id>
            <inventory-node-connector-ref xmlns="urn:opendaylight:model:topology:inventory" xmlns:a="urn:opendaylight:inventory">/a:nodes/a:node[a:id='openflow: 2']/a:node-connector[a:id='openflow: 2:LOCAL']
        </inventory-node-connector-ref>
        </termination-point>
        <inventory-node-ref xmlns="urn:opendaylight:model:topology:inventory" xmlns:a="urn:opendaylight:inventory">/a:nodes/a:node[a:id='openflow: 2']
        </inventory-node-ref>
    </node>
    <node>
        <node-id>openflow: 1</node-id>
        <termination-point>
            <tp-id>openflow: 1: 1</tp-id>
            <inventory-node-connector-ref xmlns="urn:opendaylight:model:topology:inventory" xmlns:a="urn:opendaylight:inventory">/a:nodes/a:node[a:id='openflow: 1']/a:node-connector[a:id='openflow: 1: 1']
        </inventory-node-connector-ref>
        </termination-point>
        <termination-point>
            <tp-id>openflow: 1:LOCAL</tp-id>
            <inventory-node-connector-ref xmlns="urn:opendaylight:model:topology:inventory" xmlns:a="urn:opendaylight:inventory">/a:nodes/a:node[a:id='openflow: 1']/a:node-connector[a:id='openflow: 1:LOCAL']
        </inventory-node-connector-ref>
        </termination-point>
        <termination-point>
            <tp-id>openflow: 1: 2</tp-id>
            <inventory-node-connector-ref xmlns="urn:opendaylight:model:topology:inventory" xmlns:a="urn:opendaylight:inventory">/a:nodes/a:node[a:id='openflow: 1']/a:node-connector[a:id='openflow: 1: 2']
        </inventory-node-connector-ref>
        </termination-point>
        <inventory-node-ref xmlns="urn:opendaylight:model:topology:inventory" xmlns:a="urn:opendaylight:inventory">/a:nodes/a:node[a:id='openflow: 1']
    </inventory-node-ref>
    </node>
    <link>
        <link-id>openflow:1:2</link-id>
        <destination>
            <dest-tp>openflow:2:2</dest-tp>
            <dest-node>openflow:2</dest-node>
        </destination>
        <source>
            <source-node>openflow:1</source-node>
            <source-tp>openflow:1:2</source-tp>
        </source>
    </link>
    <link>
        <link-id>openflow:2:2</link-id>
        <destination>
            <dest-tp>openflow:1:2</dest-tp>
            <dest-node>openflow:1</dest-node>
        </destination>
        <source>
            <source-node>openflow:2</source-node>
            <source-tp>openflow:2:2</source-tp>
        </source>
    </link>
</topology>
```

## (网络虚拟化相关)如何获取网络逻辑上的拓扑结构（终端之间的连通性）

不能。OpenDaylight只知道下发流表，不知道下发的流表会不会造成节点之间可达或不可达

## (网络虚拟化相关)如何通过输入一个配置文件（例如邻接矩阵等）自动配置全网的逻辑上的拓扑结构

不能。理由同上。要实现这个功能只能自己写。

## 适配了哪些品牌的SDN交换机

>到2016年10月为止，OpenDaylight项目的白金成员有Brocade、Cisco、爱立信、HP、Intel、RedHat；黄金成员有Inocybe、NEC；白银成员有42家，其中包括中国的阿里巴巴、中国移动、富士通、新华三、华为、腾讯、中兴等。

从OpenDaylight项目成员来看，Brocade、Cisco、NEC、新华三、华为这些都是交换机生产商，它们的产品必有对OpenDaylight进行适配。

是否提供文档齐全的HTTP API
若无HTTP API或文档不完善，那么提供了哪些语言实现的客户端库
官方推荐使用何种调用方式
