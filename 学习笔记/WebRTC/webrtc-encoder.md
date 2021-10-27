# WebRTC源码中的Encoder

**学习要点：**
1. 所有的编码器都继承自`VideoEncoder`类
2. `LibvpxVp9Encoder`继承自`VP9Encoder`，`VP9Encoder`继承自`VideoEncoder`
3. `LibvpxInterface`类对libvpx里面的那些函数接口进行了封装（libvpx是C写的，没有面向对象）
4. `LibvpxVp8Encoder`和`LibvpxVp9Encoder`都是调用`LibvpxInterface`类进行的实现，而不是直接调用libvpx里面的函数接口
5. 因此，`VideoEncoder`就是沟通底层编码器和上层应用的接口。顺着`VideoEncoder`向上可以找到实现自适应编码的过程、向下可以找到自适应编码修改了编码器的哪些参数。

