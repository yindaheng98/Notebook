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

这两种传输控制类分别有各自的PeerConnection，所以`pion/ion-sfu`中是没有双向的PeerConnection的，收和发分别由两个PeerConnection控制。两个PeerConnection怎么处理Offer和Answer过程见后文。

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

##  `pion/ion-sfu`中是如何处理关闭track的？


相关操作[在`AddDownTrack`的时候就已经通过`OnCloseHandler`定好了](https://github.com/pion/ion-sfu/blob/68545cc25230220435ee028d5a0af6e768a0a79a/pkg/sfu/router.go#L269)
```go
	downTrack.OnCloseHandler(func() {
		if sub.pc.ConnectionState() != webrtc.PeerConnectionStateClosed {
			if err := sub.pc.RemoveTrack(downTrack.transceiver.Sender()); err != nil {
				if err == webrtc.ErrConnectionClosed {
					return
				}
				Logger.Error(err, "Error closing down track")
			} else {
				sub.RemoveDownTrack(recv.StreamID(), downTrack)
				sub.negotiate()
			}
		}
	})
```
一个`Publisher`里过来的Track可能会通过`AddDownTrack`加到很多个`Subscriber`里，当Publish侧的SDK通过`UnPublish`函数关闭了一个流，`Publisher`里会有流关闭，进而触发所有这些`Subscriber`里的`OnCloseHandler`函数，从而达到删除流的目的。

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

## `Publisher`和`Subscriber`两个PeerConnection怎么处理Offer和Answer的？

从[SDK的代码](https://github.com/pion/ion-sdk-go/blob/12e32a5871b905bf2bdf58bc45c2fdd2741c4f81/rtc.go#L654)上看，信令的传输也会被分类为两种。在SDK侧，接收到的所有Offer都交给[`negotiate`](https://github.com/pion/ion-sdk-go/blob/12e32a5871b905bf2bdf58bc45c2fdd2741c4f81/rtc.go#L364)函数处理，接收到的所有Answer都交给[`setRemoteSDP`](https://github.com/pion/ion-sdk-go/blob/12e32a5871b905bf2bdf58bc45c2fdd2741c4f81/rtc.go#L540)函数处理：
```go
			var sdpType webrtc.SDPType
			if payload.Description.Type == "offer" {
				sdpType = webrtc.SDPTypeOffer
			} else {
				sdpType = webrtc.SDPTypeAnswer
			}
			sdp := webrtc.SessionDescription{
				SDP:  payload.Description.Sdp,
				Type: sdpType,
			}
			if sdp.Type == webrtc.SDPTypeOffer {
				log.Infof("[%v] [description] got offer call s.OnNegotiate sdp=%+v", r.uid, sdp)
				err := r.negotiate(sdp)
				if err != nil {
					log.Errorf("error: %v", err)
				}
			} else if sdp.Type == webrtc.SDPTypeAnswer {
				log.Infof("[%v] [description] got answer call sdp=%+v", r.uid, sdp)
				err = r.setRemoteSDP(sdp)
				if err != nil {
					log.Errorf("[%v] [description] setRemoteSDP err=%s", r.uid, err)
				}
			}
```
并且可以看到[`negotiate`](https://github.com/pion/ion-sdk-go/blob/12e32a5871b905bf2bdf58bc45c2fdd2741c4f81/rtc.go#L364)函数里基本上都是对Subscribe方向的操作、[`setRemoteSDP`](https://github.com/pion/ion-sdk-go/blob/12e32a5871b905bf2bdf58bc45c2fdd2741c4f81/rtc.go#L540)函数里基本上都是对Publish方向的操作。

所以，所有从SFU到SDK的流（即“Subscribe”）都是SFU向SDK发Offer、SDK向SFU回Answer；所有从SDK到SFU的流（即“Publish”）都是SDK向SFU发Offer、SFU向SDK回Answer。

所以：
* 如果在SFU那边收到了Offer，那必然是“Publish”流里的，应该给`Publisher`里的PeerConnection用，并且让`Publisher`里的PeerConnection回复一个Answer。代码位于[这里](https://github.com/pion/ion/blob/65dbd12eaad0f0e0a019b4d8ee80742930bcdc28/pkg/node/sfu/service.go#L338)。
* 如果在SFU那边收到了Answer，那必然是“Subscribe”流里的，应该给`Subscriber`里的PeerConnection用。代码位于[这里](https://github.com/pion/ion/blob/65dbd12eaad0f0e0a019b4d8ee80742930bcdc28/pkg/node/sfu/service.go#L338)。
* 如果在SDK这边收到了Offer，那必然是“Subscribe”流里的，应该给Subscribe方向的PeerConnection用，并且让Subscribe方向的PeerConnection回复一个Answer。代码就是上面介绍的[`negotiate`](https://github.com/pion/ion-sdk-go/blob/12e32a5871b905bf2bdf58bc45c2fdd2741c4f81/rtc.go#L364)。
* 如果在SDK这边收到了Answer，那必然是“Publish”流里的，应该给Publish方向的PeerConnection用。代码就是上面介绍的[`setRemoteSDP`](https://github.com/pion/ion-sdk-go/blob/12e32a5871b905bf2bdf58bc45c2fdd2741c4f81/rtc.go#L540)。

### 那两个方向的Offer都是从哪来的？

“Publish”流的Offer是SDK在[`Join`函数](https://github.com/pion/ion-sdk-go/blob/12e32a5871b905bf2bdf58bc45c2fdd2741c4f81/rtc.go#L195)里发出的：
```go
	offer, err := r.pub.pc.CreateOffer(nil)
	if err != nil {
		return err
	}

	err = r.pub.pc.SetLocalDescription(offer)
	if err != nil {
		return err
	}

	if len(config) > 0 {
		err = r.SendJoin(sid, r.uid, offer, *config[0])
	} else {
		err = r.SendJoin(sid, r.uid, offer, nil)
	}
```
这里的`SendJoin`就是将SDP打包在`rtc.Request_Join`里发出。

“Subscribe”流的Offer是在SFU处理上面这SDK发的`rtc.Request_Join`请求时通过[设置`OnOffer`](https://github.com/pion/ion/blob/65dbd12eaad0f0e0a019b4d8ee80742930bcdc28/pkg/node/sfu/service.go#L164)发出的：
```go
			// Notify user of new offer
			peer.OnOffer = func(o *webrtc.SessionDescription) {
				log.Debugf("[S=>C] peer.OnOffer: %v", o.SDP)
				err = sig.Send(&rtc.Reply{
					Payload: &rtc.Reply_Description{
						Description: &rtc.SessionDescription{
							Target: rtc.Target(rtc.Target_SUBSCRIBER),
							Sdp:    o.SDP,
							Type:   o.Type.String(),
						},
					},
				})
				if err != nil {
					log.Errorf("negotiation error: %v", err)
				}
			}
```
很明显，不用多讲。

进一步，SDK接收“Subscribe”流和SFU接收“Publish”流用的都是`OnTrack`，SFU里的操作前面已经介绍了，SDK里的`OnTrack`在这：