# interceptor寻踪：从`TrackRemote`开始深入挖掘`pion/interceptor`的用法

上接[《interceptor寻踪：`pion/interceptor`在`pion/webrtc`里的用法解析》](./interceptor在pc里.md)，来深入挖掘一下interceptor在`TrackRemote`里的用法

## 在`TrackRemote`里

从[《pion中的`TrackRemote`》](./TrackRemote.md)里的调用链可以看到，最核心的函数就只有一个`Read`：
```go
// Read reads data from the track.
func (t *TrackRemote) Read(b []byte) (n int, attributes interceptor.Attributes, err error) {
	t.mu.RLock()
	r := t.receiver
	peeked := t.peeked != nil
	t.mu.RUnlock()

	......

	n, attributes, err = r.readRTP(b, t)
	if err != nil {
		return
	}

	err = t.checkAndUpdateTrack(b)
	return
}
```
而`Read`里最核心的明显就是这句`r.readRTP(b, t)`，而这个`r`就是来自于上面那句`r := t.receiver`。再去看看`TrackRemote`的定义，可以发现这个`t.receiver`是个`RTPReceiver`：
```go
type TrackRemote struct {
	......

	receiver         *RTPReceiver

	......
}
```
所以这个`TrackRemote`里的interceptor相关操作是在外面定义好了封进`RTPReceiver`传进来的，`TrackRemote`里面本身也不涉及interceptor相关的操作。
