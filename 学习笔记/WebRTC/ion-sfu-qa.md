# `pion/ion-sfu`Q&A

## `pion/ion-sfu`和ion中的SFU服务之间的区别和联系？

* ion中的SFU服务是在`pion/ion-sfu`的基础上添加了GRPC信令传输功能得来的
* ion中的SFU服务代码主要是传输信令和根据信令调用`pion/ion-sfu`中的函数

## 可以控制`pion/ion-sfu`主动连接其他SFU吗

* `pion/ion-sfu`主要为被动接收连接请求设计，所以不能`CreateOffer`，ion中的SFU服务只有一个信令服务器，想要发起连接只能用`pion/ion-go-sdk`将本地流推送到SFU服务，而不能控制SFU服务主动向其他SFU发起请求
* 但`pion/ion-sfu`中有`OnOffer`，如果hack一下`pion/ion-go-sdk`
* Session相关的代码都在`pion/ion-sfu`里面，ion中的SFU服务的代码中基本没有操作Session的逻辑

## 可以用本地视频文件创建一个没有上行流的SFU服务吗？

## `pion/ion-sfu`中是如何处理新增的Track的？

首先，`pion/ion-sfu`中根据视频流的传输方向抽象出了几种传输控制类：
* [`Publisher`](https://github.com/pion/ion-sfu/blob/68545cc25230220435ee028d5a0af6e768a0a79a/pkg/sfu/publisher.go#L18)和[`PublisherTrack`](https://github.com/pion/ion-sfu/blob/68545cc25230220435ee028d5a0af6e768a0a79a/pkg/sfu/publisher.go#L44)：处理从外面“Publish”到本SFU的流，即上行流
* [`Subscriber`](https://github.com/pion/ion-sfu/blob/68545cc25230220435ee028d5a0af6e768a0a79a/pkg/sfu/subscriber.go#L16)和[`DownTrack`](https://github.com/pion/ion-sfu/blob/68545cc25230220435ee028d5a0af6e768a0a79a/pkg/sfu/downtrack.go#L27)：处理外面“Subscribe”本SFU的流，即下行流

这两种传输控制类分别有各自的PeerConnection，所以`pion/ion-sfu`中是没有双向的PeerConnection的，收和发分别由两个PeerConnection控制。

`Publisher`和`Subscriber`的初始化函数大体相同，都会创建PeerConnection。而在`Publisher`的初始化函数比`Subscriber`的初始化函数多了这么[一段代码](https://github.com/pion/ion-sfu/blob/68545cc25230220435ee028d5a0af6e768a0a79a/pkg/sfu/publisher.go#L77)：
```go
	pc.OnTrack(func(track *webrtc.TrackRemote, receiver *webrtc.RTPReceiver) {
		Logger.V(1).Info("Peer got remote track id",
			"peer_id", p.id,
			"track_id", track.ID(),
			"mediaSSRC", track.SSRC(),
			"rid", track.RID(),
			"stream_id", track.StreamID(),
		)

		r, pub := p.router.AddReceiver(receiver, track, track.ID(), track.StreamID())
		if pub {
			p.session.Publish(p.router, r)
			p.mu.Lock()
			publisherTrack := PublisherTrack{track, r, true}
			p.tracks = append(p.tracks, publisherTrack)
			for _, rp := range p.relayPeers {
				if err = p.createRelayTrack(track, r, rp.peer); err != nil {
					Logger.V(1).Error(err, "Creating relay track.", "peer_id", p.id)
				}
			}
			p.mu.Unlock()
			if handler, ok := p.onPublisherTrack.Load().(func(PublisherTrack)); ok && handler != nil {
				handler(publisherTrack)
			}
		} else {
			p.mu.Lock()
			p.tracks = append(p.tracks, PublisherTrack{track, r, false})
			p.mu.Unlock()
		}
	})

	pc.OnDataChannel(func(dc *webrtc.DataChannel) {
		if dc.Label() == APIChannelLabel {
			// terminate api data channel
			return
		}
		p.session.AddDatachannel(id, dc)
	})
```

可以看到，这段代码分别注册了`OnTrack`和`OnDataChannel`两个函数，在对面有新Track和新DataChannel加进来的时候执行操作，很明显最核心的就是这个`p.session.Publish(p.router, r)`和`p.session.AddDatachannel(id, dc)`。把这两个函数打开看看：

首先是[`Publish`](https://github.com/pion/ion-sfu/blob/68545cc25230220435ee028d5a0af6e768a0a79a/pkg/sfu/session.go#L222)：
```go
func (s *SessionLocal) Publish(router Router, r Receiver) {
	for _, p := range s.Peers() {
		// Don't sub to self
		if router.ID() == p.ID() || p.Subscriber() == nil {
			continue
		}

		Logger.V(0).Info("Publishing track to peer", "peer_id", p.ID())

		if err := router.AddDownTracks(p.Subscriber(), r); err != nil {
			Logger.Error(err, "Error subscribing transport to Router")
			continue
		}
	}
}
```
太明显了，这就是一个循环把Track加进这个Session的所有Peer的`Subscriber`的DownTracks里面。

然后是[`AddDatachannel`](https://github.com/pion/ion-sfu/blob/68545cc25230220435ee028d5a0af6e768a0a79a/pkg/sfu/session.go#L158)：
```go
func (s *SessionLocal) AddDatachannel(owner string, dc *webrtc.DataChannel) {
	label := dc.Label()

	s.mu.Lock()
	for _, lbl := range s.fanOutDCs {
		if label == lbl {
			s.mu.Unlock()
			return
		}
	}
	s.fanOutDCs = append(s.fanOutDCs, label)
	peerOwner := s.peers[owner]
	s.mu.Unlock()
	peers := s.Peers()
	peerOwner.Subscriber().RegisterDatachannel(label, dc)

	dc.OnMessage(func(msg webrtc.DataChannelMessage) {
		s.FanOutMessage(owner, label, msg)
	})

	for _, p := range peers {
		peer := p
		if peer.ID() == owner || peer.Subscriber() == nil {
			continue
		}
		ndc, err := peer.Subscriber().AddDataChannel(label)

		if err != nil {
			Logger.Error(err, "error adding datachannel")
			continue
		}

		if peer.Publisher() != nil && peer.Publisher().Relayed() {
			peer.Publisher().AddRelayFanOutDataChannel(label)
		}

		pid := peer.ID()
		ndc.OnMessage(func(msg webrtc.DataChannelMessage) {
			s.FanOutMessage(pid, label, msg)

			if peer.Publisher().Relayed() {
				for _, rdc := range peer.Publisher().GetRelayedDataChannels(label) {
					if msg.IsString {
						if err = rdc.SendText(string(msg.Data)); err != nil {
							Logger.Error(err, "Sending dc message err")
						}
					} else {
						if err = rdc.Send(msg.Data); err != nil {
							Logger.Error(err, "Sending dc message err")
						}
					}
				}
			}
		})

		peer.Subscriber().negotiate()
	}
}
```
一看就是在搞消息转发，就不细看了

## `pion/ion-sfu`中的`JoinConfig`是如何控制SFU的转发逻辑的？

[`JoinConfig`](https://github.com/pion/ion-sfu/blob/68545cc25230220435ee028d5a0af6e768a0a79a/pkg/sfu/peer.go#L37)长这样：
```go
// JoinConfig allow adding more control to the peers joining a SessionLocal.
type JoinConfig struct {
	// If true the peer will not be allowed to publish tracks to SessionLocal.
	NoPublish bool
	// If true the peer will not be allowed to subscribe to other peers in SessionLocal.
	NoSubscribe bool
	// If true the peer will not automatically subscribe all tracks,
	// and then the peer can use peer.Subscriber().AddDownTrack/RemoveDownTrack
	// to customize the subscrbe stream combination as needed.
	// this parameter depends on NoSubscribe=false.
	NoAutoSubscribe bool
}
```

首先，在[Peer的初始化过程](https://github.com/pion/ion-sfu/blob/68545cc25230220435ee028d5a0af6e768a0a79a/pkg/sfu/peer.go#L107)中有`NoSubscribe`和`NoPublish`发挥作用：

```go
	if !conf.NoSubscribe {
		p.subscriber, err = NewSubscriber(uid, cfg)
		if err != nil {
			return fmt.Errorf("error creating transport: %v", err)
		}

		p.subscriber.noAutoSubscribe = conf.NoAutoSubscribe

		p.subscriber.OnNegotiationNeeded(func() {
			p.Lock()
			defer p.Unlock()

			if p.remoteAnswerPending {
				p.negotiationPending = true
				return
			}

			Logger.V(1).Info("Negotiation needed", "peer_id", p.id)
			offer, err := p.subscriber.CreateOffer()
			if err != nil {
				Logger.Error(err, "CreateOffer error")
				return
			}

			p.remoteAnswerPending = true
			if p.OnOffer != nil && !p.closed.get() {
				Logger.V(0).Info("Send offer", "peer_id", p.id)
				p.OnOffer(&offer)
			}
		})

		p.subscriber.OnICECandidate(func(c *webrtc.ICECandidate) {
			Logger.V(1).Info("On subscriber ice candidate called for peer", "peer_id", p.id)
			if c == nil {
				return
			}

			if p.OnIceCandidate != nil && !p.closed.get() {
				json := c.ToJSON()
				p.OnIceCandidate(&json, subscriber)
			}
		})
	}
```
```go
	if !conf.NoSubscribe {
		p.session.Subscribe(p)
	}
```
显然，这`NoSubscribe`在生成`PeerLocal`时控制`Subscriber`的初始化，如果`NoSubscribe=true`就不会有`Subscriber`生成。从而也就没法`AddDownTrack`向外传出Track。

```go
	if !conf.NoPublish {
		p.publisher, err = NewPublisher(uid, p.session, &cfg)
		if err != nil {
			return fmt.Errorf("error creating transport: %v", err)
		}
		if !conf.NoSubscribe {
			for _, dc := range p.session.GetDCMiddlewares() {
				if err := p.subscriber.AddDatachannel(p, dc); err != nil {
					return fmt.Errorf("setting subscriber default dc datachannel: %w", err)
				}
			}
		}

		p.publisher.OnICECandidate(func(c *webrtc.ICECandidate) {
			Logger.V(1).Info("on publisher ice candidate called for peer", "peer_id", p.id)
			if c == nil {
				return
			}

			if p.OnIceCandidate != nil && !p.closed.get() {
				json := c.ToJSON()
				p.OnIceCandidate(&json, publisher)
			}
		})

		p.publisher.OnICEConnectionStateChange(func(s webrtc.ICEConnectionState) {
			if p.OnICEConnectionStateChange != nil && !p.closed.get() {
				p.OnICEConnectionStateChange(s)
			}
		})
	}
```
显然，这`NoPublish`在生成`PeerLocal`时控制`Publisher`的初始化，如果`NoPublish=true`就不会有`Publisher`生成。根据上一节的分析，`PublisherTrack`增减相关的操作主要就是在`Publisher`的初始化过程中执行的，没有了`Publisher`也就不会有对传入的`PublisherTrack`的那些操作了，从而也就不会接收传入的Track。

此外，我们发现`NoAutoSubscribe`被赋值给了`p.subscriber.noAutoSubscribe`这个值主要在[`AddDownTracks`](https://github.com/pion/ion-sfu/blob/68545cc25230220435ee028d5a0af6e768a0a79a/pkg/sfu/router.go#L210)的里面发挥作用：
```go
func (r *router) AddDownTracks(s *Subscriber, recv Receiver) error {
	r.Lock()
	defer r.Unlock()

	if s.noAutoSubscribe {
		Logger.Info("peer turns off automatic subscription, skip tracks add")
		return nil
	}

	if recv != nil {
		if _, err := r.AddDownTrack(s, recv); err != nil {
			return err
		}
		s.negotiate()
		return nil
	}

	if len(r.receivers) > 0 {
		for _, rcv := range r.receivers {
			if _, err := r.AddDownTrack(s, rcv); err != nil {
				return err
			}
		}
		s.negotiate()
	}
	return nil
}
```
所以，当调用[`Publish`](https://github.com/pion/ion-sfu/blob/68545cc25230220435ee028d5a0af6e768a0a79a/pkg/sfu/session.go#L222)的时候，`NoAutoSubscribe=true`的router不会被调用`AddDownTrack`。根据前面对[`Publisher`的初始化函数](https://github.com/pion/ion-sfu/blob/68545cc25230220435ee028d5a0af6e768a0a79a/pkg/sfu/publisher.go#L77)的分析，`Publisher`有新Track到达的时候会对所有Session里的Peer调用`Publish`，所以`NoAutoSubscribe=true`不调用`AddDownTrack`就意味着新Track到达的时候这个Peer没法`AddDownTrack`，所以达到了“No Auto Subscribe”的目的。