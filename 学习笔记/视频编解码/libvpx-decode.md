# libvpx解码过程解读

上接[《`libvpx`再深入一点》](./libvpx-insight.md)，本篇从上篇最后的`vp9_receive_compressed_data`函数开始解读。

## `vp9_receive_compressed_data`

```c
int vp9_receive_compressed_data(VP9Decoder *pbi, size_t size,
                                const uint8_t **psource) {
  VP9_COMMON *volatile const cm = &pbi->common;
  BufferPool *volatile const pool = cm->buffer_pool;
  RefCntBuffer *volatile const frame_bufs = cm->buffer_pool->frame_bufs;
  const uint8_t *source = *psource;
  int retcode = 0;
  cm->error.error_code = VPX_CODEC_OK;

  if (size == 0) {
    // This is used to signal that we are missing frames.
    // We do not know if the missing frame(s) was supposed to update
    // any of the reference buffers, but we act conservative and
    // mark only the last buffer as corrupted.
    //
    // TODO(jkoleszar): Error concealment is undefined and non-normative
    // at this point, but if it becomes so, [0] may not always be the correct
    // thing to do here.
    if (cm->frame_refs[0].idx > 0) {
      assert(cm->frame_refs[0].buf != NULL);
      cm->frame_refs[0].buf->corrupted = 1;
    }
  }

  pbi->ready_for_new_data = 0;

  // Check if the previous frame was a frame without any references to it.
  if (cm->new_fb_idx >= 0 && frame_bufs[cm->new_fb_idx].ref_count == 0 &&
      !frame_bufs[cm->new_fb_idx].released) {
    pool->release_fb_cb(pool->cb_priv,
                        &frame_bufs[cm->new_fb_idx].raw_frame_buffer);
    frame_bufs[cm->new_fb_idx].released = 1;
  }

  // Find a free frame buffer. Return error if can not find any.
  cm->new_fb_idx = get_free_fb(cm);
  if (cm->new_fb_idx == INVALID_IDX) {
    pbi->ready_for_new_data = 1;
    release_fb_on_decoder_exit(pbi);
    vpx_clear_system_state();
    vpx_internal_error(&cm->error, VPX_CODEC_MEM_ERROR,
                       "Unable to find free frame buffer");
    return cm->error.error_code;
  }

  // Assign a MV array to the frame buffer.
  cm->cur_frame = &pool->frame_bufs[cm->new_fb_idx];

  pbi->hold_ref_buf = 0;
  pbi->cur_buf = &frame_bufs[cm->new_fb_idx];

  if (setjmp(cm->error.jmp)) {
    cm->error.setjmp = 0;
    pbi->ready_for_new_data = 1;
    release_fb_on_decoder_exit(pbi);
    // Release current frame.
    decrease_ref_count(cm->new_fb_idx, frame_bufs, pool);
    vpx_clear_system_state();
    return -1;
  }

  cm->error.setjmp = 1;
  vp9_decode_frame(pbi, source, source + size, psource);

  swap_frame_buffers(pbi);

  vpx_clear_system_state();

  if (!cm->show_existing_frame) {
    cm->last_show_frame = cm->show_frame;
    cm->prev_frame = cm->cur_frame;
    if (cm->seg.enabled) vp9_swap_current_and_last_seg_map(cm);
  }

  if (cm->show_frame) cm->cur_show_frame_fb_idx = cm->new_fb_idx;

  // Update progress in frame parallel decode.
  cm->last_width = cm->width;
  cm->last_height = cm->height;
  if (cm->show_frame) {
    cm->current_video_frame++;
  }

  cm->error.setjmp = 0;
  return retcode;
}
```