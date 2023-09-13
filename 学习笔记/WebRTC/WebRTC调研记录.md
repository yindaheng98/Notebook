## FFmpeg硬件加速

[wiki:HWAccelIntro](https://trac.ffmpeg.org/wiki/HWAccelIntro)

>NVDEC offers decoders for H.264, HEVC, MJPEG, MPEG-1/2/4, VP8/VP9, VC-1. Codec support varies by hardware (see the ​GPU compatibility table).
>
>Note that FFmpeg offers both NVDEC and CUVID hwaccels. They differ in how frames are decoded and forwarded in memory.

## 在Jetson上测试FFmpeg的GPU解码

[jetson-ffmpeg](https://github.com/jocover/jetson-ffmpeg)
