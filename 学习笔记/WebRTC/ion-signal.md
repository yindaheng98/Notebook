# ION中的Signal服务

上接[《ION的基本架构》](ION-Arch.md)，Signal应该是个GRPC代理，把外面来的标准GRPC请求转换为`nats-grpc`的请求。

这个[`sig.Director`函数](https://github.com/pion/ion/blob/65dbd12eaad0f0e0a019b4d8ee80742930bcdc28/pkg/node/signal/signal.go)应该就是最核心的东西：
```go
func (s *Signal) Director(ctx context.Context, fullMethodName string) (context.Context, grpc.ClientConnInterface, error) {
	md, ok := metadata.FromIncomingContext(ctx)
	if ok {
		log.Infof("fullMethodName: %v, md %v", fullMethodName, md)
	}

	//Authenticate here.
	authConfig := &s.conf.Signal.JWT
	if authConfig.Enabled {
        ......
	}

	//Find service in neighbor nodes.
	svcConf := s.conf.Signal.SVC
	for _, svc := range svcConf.Services {
		if strings.HasPrefix(fullMethodName, "/"+svc+".") {
			//Using grpc.Metadata as a parameters for ndc.Get.
			var parameters = make(map[string]interface{})
			for key, value := range md {
				parameters[key] = value[0]
			}
			cli, err := s.NewNatsRPCClient(svc, "*", parameters)
			if err != nil {
				log.Errorf("failed to Get service [%v]: %v", svc, err)
				return ctx, nil, status.Errorf(codes.Unavailable, "Service Unavailable: %v", err)
			}
			return ctx, cli, nil
		}
	}

	return ctx, nil, status.Errorf(codes.Unimplemented, "Unknown Service.Method %v", fullMethodName)
}
```
最核心的实际上就是下面这个`for`循环，明显是通过设置项里指定的`signal.svc`这一项找到要调用的GRPC接口名称`fullMethodName`对应的哪个服务，然后根据服务名称调用`NewNatsRPCClient`生成`nats-grpc`客户端，于是把请求转发到了nats里面。

（这查询居然用`for`循环一个个比较？？）

然后从`NewNatsRPCClient`里的逻辑看：
```go
func (n *Node) NewNatsRPCClient(service, peerNID string, parameters map[string]interface{}) (*nrpc.Client, error) {
	var cli *nrpc.Client = nil
	selfNID := n.NID
	for id, node := range n.neighborNodes {
		if node.Service == service && (id == peerNID || peerNID == "*") {
			cli = nrpc.NewClient(n.nc, id, selfNID)
		}
	}

	if cli == nil {
		resp, err := n.ndc.Get(service, parameters)
		if err != nil {
			log.Errorf("failed to Get service [%v]: %v", service, err)
			return nil, err
		}

		if len(resp.Nodes) == 0 {
			err := fmt.Errorf("get service [%v], node cnt == 0", service)
			return nil, err
		}

		cli = nrpc.NewClient(n.nc, resp.Nodes[0].NID, selfNID)
	}

	n.cliLock.Lock()
	defer n.cliLock.Unlock()
	n.clis[util.RandomString(12)] = cli
	return cli, nil
}
```

居然......就只返回了最后一个查找到的服务啊：
```go
	for id, node := range n.neighborNodes {
		if node.Service == service && (id == peerNID || peerNID == "*") {
			cli = nrpc.NewClient(n.nc, id, selfNID)
		}
	}
```

看来这ION里的服务每一种都只能部署一个。

😂我还以为能有多SFU的路径选择或者多个Signal或者Room服务的负载均衡呢，考虑是我高估这东西的完成度了