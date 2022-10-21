# libvpx中的`decode_partition`

终于要接近最深处了，这是`decode_partition`：

```c
static void decode_partition(TileWorkerData *twd, VP9Decoder *const pbi,
                             int mi_row, int mi_col, BLOCK_SIZE bsize,
                             int n4x4_l2) {
  VP9_COMMON *const cm = &pbi->common;
  const int n8x8_l2 = n4x4_l2 - 1;
  const int num_8x8_wh = 1 << n8x8_l2;
  const int hbs = num_8x8_wh >> 1;
  PARTITION_TYPE partition;
  BLOCK_SIZE subsize;
  const int has_rows = (mi_row + hbs) < cm->mi_rows;
  const int has_cols = (mi_col + hbs) < cm->mi_cols;
  MACROBLOCKD *const xd = &twd->xd;

  if (mi_row >= cm->mi_rows || mi_col >= cm->mi_cols) return;

  partition = read_partition(twd, mi_row, mi_col, has_rows, has_cols, n8x8_l2);
  subsize = subsize_lookup[partition][bsize];  // get_subsize(bsize, partition);
  if (!hbs) {
    // calculate bmode block dimensions (log 2)
    xd->bmode_blocks_wl = 1 >> !!(partition & PARTITION_VERT);
    xd->bmode_blocks_hl = 1 >> !!(partition & PARTITION_HORZ);
    decode_block(twd, pbi, mi_row, mi_col, subsize, 1, 1);
  } else {
    switch (partition) {
      case PARTITION_NONE:
        decode_block(twd, pbi, mi_row, mi_col, subsize, n4x4_l2, n4x4_l2);
        break;
      case PARTITION_HORZ:
        decode_block(twd, pbi, mi_row, mi_col, subsize, n4x4_l2, n8x8_l2);
        if (has_rows)
          decode_block(twd, pbi, mi_row + hbs, mi_col, subsize, n4x4_l2,
                       n8x8_l2);
        break;
      case PARTITION_VERT:
        decode_block(twd, pbi, mi_row, mi_col, subsize, n8x8_l2, n4x4_l2);
        if (has_cols)
          decode_block(twd, pbi, mi_row, mi_col + hbs, subsize, n8x8_l2,
                       n4x4_l2);
        break;
      case PARTITION_SPLIT:
        decode_partition(twd, pbi, mi_row, mi_col, subsize, n8x8_l2);
        decode_partition(twd, pbi, mi_row, mi_col + hbs, subsize, n8x8_l2);
        decode_partition(twd, pbi, mi_row + hbs, mi_col, subsize, n8x8_l2);
        decode_partition(twd, pbi, mi_row + hbs, mi_col + hbs, subsize,
                         n8x8_l2);
        break;
      default: assert(0 && "Invalid partition type");
    }
  }

  // update partition context
  if (bsize >= BLOCK_8X8 &&
      (bsize == BLOCK_8X8 || partition != PARTITION_SPLIT))
    dec_update_partition_context(twd, mi_row, mi_col, subsize, num_8x8_wh);
}
```

可以看见，很短， 核心部分就是一个`switch`下套4个条件，里面有递归的`decode_partition`和一个`decode_block`函数。

这个递归的条件`PARTITION_SPLIT`很明显就是把区域再分四块进行解码，对应编码器里可变大小编码块的知识。

所以再往下一层就是这个`decode_block`了：

```c
static void decode_block(TileWorkerData *twd, VP9Decoder *const pbi, int mi_row,
                         int mi_col, BLOCK_SIZE bsize, int bwl, int bhl) {
  VP9_COMMON *const cm = &pbi->common;
  const int less8x8 = bsize < BLOCK_8X8;
  const int bw = 1 << (bwl - 1);
  const int bh = 1 << (bhl - 1);
  const int x_mis = VPXMIN(bw, cm->mi_cols - mi_col);
  const int y_mis = VPXMIN(bh, cm->mi_rows - mi_row);
  vpx_reader *r = &twd->bit_reader;
  MACROBLOCKD *const xd = &twd->xd;

  MODE_INFO *mi = set_offsets(cm, xd, bsize, mi_row, mi_col, bw, bh, x_mis,
                              y_mis, bwl, bhl);

  if (bsize >= BLOCK_8X8 && (cm->subsampling_x || cm->subsampling_y)) {
    const BLOCK_SIZE uv_subsize =
        ss_size_lookup[bsize][cm->subsampling_x][cm->subsampling_y];
    if (uv_subsize == BLOCK_INVALID)
      vpx_internal_error(xd->error_info, VPX_CODEC_CORRUPT_FRAME,
                         "Invalid block size.");
  }

  vp9_read_mode_info(twd, pbi, mi_row, mi_col, x_mis, y_mis);

  if (mi->skip) {
    dec_reset_skip_context(xd);
  }

  if (!is_inter_block(mi)) {
    int plane;
    for (plane = 0; plane < MAX_MB_PLANE; ++plane) {
      const struct macroblockd_plane *const pd = &xd->plane[plane];
      const TX_SIZE tx_size = plane ? get_uv_tx_size(mi, pd) : mi->tx_size;
      const int num_4x4_w = pd->n4_w;
      const int num_4x4_h = pd->n4_h;
      const int step = (1 << tx_size);
      int row, col;
      const int max_blocks_wide =
          num_4x4_w + (xd->mb_to_right_edge >= 0
                           ? 0
                           : xd->mb_to_right_edge >> (5 + pd->subsampling_x));
      const int max_blocks_high =
          num_4x4_h + (xd->mb_to_bottom_edge >= 0
                           ? 0
                           : xd->mb_to_bottom_edge >> (5 + pd->subsampling_y));

      xd->max_blocks_wide = xd->mb_to_right_edge >= 0 ? 0 : max_blocks_wide;
      xd->max_blocks_high = xd->mb_to_bottom_edge >= 0 ? 0 : max_blocks_high;

      for (row = 0; row < max_blocks_high; row += step)
        for (col = 0; col < max_blocks_wide; col += step)
          predict_and_reconstruct_intra_block(twd, mi, plane, row, col,
                                              tx_size);
    }
  } else {
    // Prediction
    dec_build_inter_predictors_sb(pbi, xd, mi_row, mi_col);

    // Reconstruction
    if (!mi->skip) {
      int eobtotal = 0;
      int plane;

      for (plane = 0; plane < MAX_MB_PLANE; ++plane) {
        const struct macroblockd_plane *const pd = &xd->plane[plane];
        const TX_SIZE tx_size = plane ? get_uv_tx_size(mi, pd) : mi->tx_size;
        const int num_4x4_w = pd->n4_w;
        const int num_4x4_h = pd->n4_h;
        const int step = (1 << tx_size);
        int row, col;
        const int max_blocks_wide =
            num_4x4_w + (xd->mb_to_right_edge >= 0
                             ? 0
                             : xd->mb_to_right_edge >> (5 + pd->subsampling_x));
        const int max_blocks_high =
            num_4x4_h +
            (xd->mb_to_bottom_edge >= 0
                 ? 0
                 : xd->mb_to_bottom_edge >> (5 + pd->subsampling_y));

        xd->max_blocks_wide = xd->mb_to_right_edge >= 0 ? 0 : max_blocks_wide;
        xd->max_blocks_high = xd->mb_to_bottom_edge >= 0 ? 0 : max_blocks_high;

        for (row = 0; row < max_blocks_high; row += step)
          for (col = 0; col < max_blocks_wide; col += step)
            eobtotal +=
                reconstruct_inter_block(twd, mi, plane, row, col, tx_size);
      }

      if (!less8x8 && eobtotal == 0) mi->skip = 1;  // skip loopfilter
    }
  }

  xd->corrupted |= vpx_reader_has_error(r);

  if (cm->lf.filter_level) {
    vp9_build_mask(cm, mi, mi_row, mi_col, bw, bh);
  }
}
```