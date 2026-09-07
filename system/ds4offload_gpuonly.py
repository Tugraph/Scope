pass

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ds4offload import *
import ds4offload as _base
import torch
import torch.nn.functional as F

_mm = _base._model_module

def _gpuonly_prefill_phase1(attn, x, bsz, seqlen):
    dev   = attn.kv_cache.device
    rd    = attn.rope_head_dim
    ratio = attn.compress_ratio
    win   = attn.window_size
    freqs_full = attn.freqs_cis[:seqlen]

    kv_all = attn.kv_norm(attn.wkv(x))
    _mm.apply_rotary_emb(kv_all[..., -rd:], freqs_full)
    kv_all_cpu = kv_all.to('cpu', non_blocking=True)
    del kv_all
    torch.cuda.synchronize()

    if ratio:
        attn.compressor(x, 0)

    if ratio and attn.indexer is not None:
        if attn.indexer.compressor.kv_cache is None:
            attn.indexer.compressor.kv_cache = attn.indexer.kv_cache
        if attn.indexer.compressor.freqs_cis is None:
            attn.indexer.compressor.freqs_cis = attn.indexer.freqs_cis
        attn.indexer.compressor(x, 0)

        if hasattr(attn.indexer, 'cpu_kv_cache'):
            nc = seqlen // attn.indexer.compress_ratio
            if nc > 0:
                attn.indexer.cpu_kv_cache[:bsz, :nc].copy_(
                    attn.indexer.kv_cache[:bsz, :nc].float(), non_blocking=False)

    kv_tail_gpu = kv_all_cpu[:bsz, max(0, seqlen - win):seqlen].to(dev)
    if seqlen <= win:
        attn.kv_cache[:bsz, :seqlen] = kv_tail_gpu
    else:
        cutoff = seqlen % win
        if cutoff == 0:
            attn.kv_cache[:bsz] = kv_tail_gpu
        else:
            attn.kv_cache[:bsz, cutoff:win] = kv_tail_gpu[:, :win - cutoff]
            attn.kv_cache[:bsz, :cutoff]    = kv_tail_gpu[:, win - cutoff:]
    del kv_tail_gpu

    attn._p1_kv_cpu  = kv_all_cpu
    attn._p1_seqlen  = seqlen

def _gpuonly_prefill_full_seq(attn, x, bsz, seqlen):
    _gpuonly_prefill_phase1(attn, x, bsz, seqlen)
    _pcs = getattr(attn, 'prefill_chunk_size', 256)
    out  = torch.empty_like(x)
    for q_s in range(0, seqlen, _pcs):
        q_e = min(q_s + _pcs, seqlen)
        _base._prefill_phase2_chunk(attn, x[:, q_s:q_e], bsz, q_s, q_e, out[:, q_s:q_e])
    del attn._p1_kv_cpu, attn._p1_seqlen
    return out

def _gpuonly_attn_forward(self, x, start_pos):
    bsz, seqlen, _ = x.size()
    win   = self.window_size
    ratio = self.compress_ratio

    _pcs = getattr(self, '_prefill_chunk_size',
                   getattr(self, 'prefill_chunk_size', 256))
    if start_pos == 0 and seqlen > _pcs:
        return _gpuonly_prefill_full_seq(self, x, bsz, seqlen)

    freqs_cis = self.freqs_cis[start_pos:start_pos + seqlen]
    rd = self.rope_head_dim
    if ratio and self.compressor.kv_cache is None:
        self.compressor.kv_cache = self.kv_cache[:, win:]
        self.compressor.freqs_cis = self.freqs_cis
        if self.indexer is not None:
            self.indexer.freqs_cis = self.freqs_cis

    qr = q = self.q_norm(self.wq_a(x))
    q  = self.wq_b(q).unflatten(-1, (self.n_local_heads, self.head_dim))
    q  = (q * torch.rsqrt(q.square().mean(-1, keepdim=True) + self.eps)).to(torch.bfloat16)
    _mm.apply_rotary_emb(q[..., -rd:], freqs_cis)

    kv = self.wkv(x)
    kv = self.kv_norm(kv)
    _mm.apply_rotary_emb(kv[..., -rd:], freqs_cis)

    if start_pos == 0:
        topk_idxs = _mm.get_window_topk_idxs(win, bsz, seqlen, start_pos).to(kv.device)
        if ratio:
            offset = kv.size(1)
            if self.indexer is not None:
                compress_topk_idxs = self.indexer(x, qr, start_pos, offset)
            else:
                compress_topk_idxs = _mm.get_compress_topk_idxs(
                    ratio, bsz, seqlen, start_pos, offset).to(kv.device)
            topk_idxs = torch.cat([topk_idxs, compress_topk_idxs], dim=-1)
        topk_idxs = topk_idxs.int()
        if seqlen <= win:
            self.kv_cache[:bsz, :seqlen] = kv
        else:
            cutoff = seqlen % win
            self.kv_cache[:bsz, cutoff:win], self.kv_cache[:bsz, :cutoff] =\
                kv[:, -win:].split([win - cutoff, cutoff], dim=1)
        if ratio:
            if (kv_compress := self.compressor(x, start_pos)) is not None:
                kv = torch.cat([kv, kv_compress.to(kv.device).to(kv.dtype)], dim=1)
        o = _mm.sparse_attn(q, kv, self.attn_sink, topk_idxs, self.softmax_scale)

    elif seqlen > 1:
        dev     = self.kv_cache.device
        cpu_buf = getattr(self, 'cpu_compress_kv', None)
        n_compress_hist = start_pos // ratio if ratio else 0
        scale = self.softmax_scale
        _pcs2 = getattr(self, '_prefill_chunk_size',
                        getattr(self, 'prefill_chunk_size', 256))

        kv_win = self.kv_cache[:bsz]
        csa_streaming = (cpu_buf is not None and n_compress_hist > 0
                         and cpu_buf.device.type == 'cpu')

        if csa_streaming and n_compress_hist > 0:
            kv_csa_gpu = cpu_buf[:bsz, :n_compress_hist].to(dev)
        else:
            kv_csa_gpu = None

        if not csa_streaming:
            if cpu_buf is not None and n_compress_hist > 0:
                kv_hist = torch.cat([cpu_buf[:bsz, :n_compress_hist], kv_win, kv], dim=1)
            else:
                kv_hist = torch.cat([kv_win, kv], dim=1)
            n_hist = kv_hist.shape[1] - seqlen

        o = torch.empty_like(q)
        for q_s in range(0, seqlen, _pcs2):
            q_e  = min(q_s + _pcs2, seqlen)
            q_cs = q_e - q_s
            q_c  = q[:, q_s:q_e]
            kv_c = kv[:, q_s:q_e]
            n_win_hist   = win
            n_intra_prev = q_s

            if csa_streaming:
                q_perm    = q_c.permute(0, 2, 1, 3).float()
                n_heads_c = q_perm.shape[1]
                kv_win_intra = torch.cat([kv_win, kv[:, :q_e]], dim=1)
                n_wi = kv_win_intra.shape[1]
                ab   = torch.zeros(q_cs, n_wi, device=dev, dtype=torch.float32)
                cmask = torch.ones(q_cs, q_cs, device=dev, dtype=torch.bool).triu(diagonal=1)
                ab[:, n_wi - q_cs:].masked_fill_(cmask, float('-inf'))
                del cmask
                kv_wi_b  = kv_win_intra.unsqueeze(1)
                scores_g = torch.einsum('bnqd,bnkd->bnqk', q_perm, kv_wi_b.float()) * scale + ab
                lse_g    = torch.logsumexp(scores_g, dim=-1)
                exp_g    = torch.exp(scores_g - lse_g.unsqueeze(-1))
                o_gpu    = torch.einsum('bnqk,bnkd->bnqd', exp_g, kv_wi_b.float()).permute(0, 2, 1, 3)
                del scores_g, exp_g, ab, kv_wi_b, kv_win_intra

                _NC_SINGLE_MAX = getattr(self, '_csa_nc_single_max', 2048)
                if n_compress_hist <= _NC_SINGLE_MAX:
                    kv_tile = kv_csa_gpu.unsqueeze(1)
                    sc_tile = torch.einsum('bnqd,bnkd->bnqk', q_perm, kv_tile.float()) * scale
                    lse_run = torch.logsumexp(sc_tile, dim=-1)
                    o_run   = torch.einsum('bnqk,bnkd->bnqd',
                                           torch.exp(sc_tile - lse_run.unsqueeze(-1)),
                                           kv_tile.float())
                    del sc_tile, kv_tile
                else:
                    GROUP    = 2048
                    buf_attr = '_csa_stream_buf2048'
                    if not hasattr(self, buf_attr):
                        bs_max = cpu_buf.shape[0]; hd_c = cpu_buf.shape[-1]
                        setattr(self, buf_attr, [
                            torch.zeros(bs_max, GROUP, hd_c, dtype=torch.bfloat16, device=dev),
                            torch.zeros(bs_max, GROUP, hd_c, dtype=torch.bfloat16, device=dev),
                        ])
                    pp         = getattr(self, buf_attr)
                    stream_h2d = torch.cuda.Stream(device=dev)
                    lse_run = torch.full((bsz, n_heads_c, q_cs), float('-inf'), device=dev, dtype=torch.float32)
                    o_run   = torch.zeros(bsz, n_heads_c, q_cs, cpu_buf.shape[-1], device=dev, dtype=torch.float32)
                    g0e = min(GROUP, n_compress_hist)
                    with torch.cuda.stream(stream_h2d):
                        pp[0][:bsz, :g0e].copy_(kv_csa_gpu[:bsz, :g0e].contiguous())
                    n_groups = (n_compress_hist + GROUP - 1) // GROUP
                    for gi in range(n_groups):
                        g_s = gi * GROUP; g_e = min(g_s + GROUP, n_compress_hist)
                        gs  = g_e - g_s;  cur = gi % 2
                        torch.cuda.current_stream(dev).wait_stream(stream_h2d)
                        if gi + 1 < n_groups:
                            nxt = 1 - cur; ns = (gi+1)*GROUP; ne = min(ns+GROUP, n_compress_hist)
                            with torch.cuda.stream(stream_h2d):
                                pp[nxt][:bsz, :ne-ns].copy_(kv_csa_gpu[:bsz, ns:ne].contiguous())
                        kv_tile  = pp[cur][:bsz, :gs].unsqueeze(1)
                        sc_tile  = torch.einsum('bnqd,bnkd->bnqk', q_perm, kv_tile.float()) * scale
                        lse_new  = torch.logsumexp(sc_tile, dim=-1)
                        norm_new = torch.einsum('bnqk,bnkd->bnqd',
                                                torch.exp(sc_tile - lse_new.unsqueeze(-1)),
                                                kv_tile.float())
                        m_max = torch.maximum(lse_run, lse_new)
                        w_old = torch.exp(lse_run - m_max); w_new = torch.exp(lse_new - m_max)
                        w_sum = (w_old + w_new).clamp(min=1e-9)
                        o_run = (o_run * w_old.unsqueeze(-1) + norm_new * w_new.unsqueeze(-1)) / w_sum.unsqueeze(-1)
                        lse_run = m_max + torch.log(w_sum)
                        del sc_tile, lse_new, norm_new, m_max, w_old, w_new, w_sum, kv_tile

                m   = torch.maximum(lse_g, lse_run)
                w_g = torch.exp(lse_g - m); w_c = torch.exp(lse_run - m)
                o_c = ((o_gpu.permute(0, 2, 1, 3) * w_g.unsqueeze(-1) +
                        o_run * w_c.unsqueeze(-1)) /
                       (w_g + w_c).clamp(min=1e-9).unsqueeze(-1)
                      ).permute(0, 2, 1, 3).to(q.dtype)
                del o_run, o_gpu, lse_run, lse_g, w_g, w_c, m, q_perm
            else:
                kv_slice    = torch.cat([kv_hist[:, :n_hist], kv[:, :q_e]], dim=1)
                sk          = kv_slice.shape[1]
                n_prefix    = sk - q_cs
                causal_mask = torch.zeros(q_cs, sk, device=dev, dtype=q.dtype)
                imask       = torch.ones(q_cs, q_cs, device=dev, dtype=torch.bool).triu(diagonal=1)
                causal_mask[:, n_prefix:].masked_fill_(imask, float('-inf'))
                del imask
                q_perm = q_c.permute(0, 2, 1, 3)
                kv_b   = kv_slice.unsqueeze(1).expand(bsz, q_perm.shape[1], -1, -1)
                o_c    = F.scaled_dot_product_attention(
                    q_perm, kv_b, kv_b, attn_mask=causal_mask, scale=scale
                ).permute(0, 2, 1, 3).to(q.dtype)
                del q_perm, kv_b, kv_slice, causal_mask

            o[:, q_s:q_e] = o_c
            del o_c

        if kv_csa_gpu is not None:
            del kv_csa_gpu
        if not csa_streaming:
            del kv_hist

        start_idx = start_pos % win
        if seqlen >= win:
            cutoff = (start_pos + seqlen) % win
            if cutoff == 0:
                self.kv_cache[:bsz] = kv[:, -win:]
            else:
                self.kv_cache[:bsz, cutoff:win] = kv[:, -win:-win + (win - cutoff)]
                self.kv_cache[:bsz, :cutoff]    = kv[:, -cutoff:]
        elif start_idx + seqlen <= win:
            self.kv_cache[:bsz, start_idx:start_idx + seqlen] = kv
        else:
            first = win - start_idx
            self.kv_cache[:bsz, start_idx:] = kv[:, :first]
            self.kv_cache[:bsz, :seqlen - first] = kv[:, first:]
        if ratio:
            self.compressor(x, start_pos)

            if self.indexer is not None:
                idx = self.indexer
                if idx.compressor.kv_cache is None:
                    idx.compressor.kv_cache  = idx.kv_cache
                    idx.compressor.freqs_cis = idx.freqs_cis
                idx.compressor(x, start_pos)
                if hasattr(idx, 'cpu_kv_cache'):
                    nc_prev = start_pos // idx.compress_ratio
                    nc_now  = (start_pos + seqlen) // idx.compress_ratio
                    if nc_now > nc_prev:
                        idx.cpu_kv_cache[:bsz, nc_prev:nc_now].copy_(
                            idx.kv_cache[:bsz, nc_prev:nc_now].float(), non_blocking=False)

    else:
        dev = self.kv_cache.device
        topk_idxs = _mm.get_window_topk_idxs(win, bsz, seqlen, start_pos).to(dev)
        if ratio:
            if self.indexer is not None:
                idx = self.indexer
                if idx.compressor.kv_cache is None:
                    idx.compressor.kv_cache  = idx.kv_cache
                    idx.compressor.freqs_cis = idx.freqs_cis
                idx.compressor(x, start_pos)
                nc = start_pos // idx.compress_ratio

                if nc > 0 and hasattr(idx, 'cpu_kv_cache'):

                    idx.cpu_kv_cache[:bsz, nc - 1].copy_(
                        idx.kv_cache[:bsz, nc - 1].float(), non_blocking=False)

                    kv_idx_gpu = idx.cpu_kv_cache[:bsz, :nc].to(dev, non_blocking=False)

                    freqs_i = idx.freqs_cis[start_pos:start_pos + seqlen]
                    rd_i    = idx.rope_head_dim
                    q_i     = idx.wq_b(qr).unflatten(-1, (idx.n_local_heads, idx.head_dim)).to(torch.bfloat16)
                    _mm.apply_rotary_emb(q_i[..., -rd_i:], freqs_i)
                    q_i     = _mm.rotate_activation(q_i)
                    _mm.fp4_act_quant(q_i, _mm.fp4_block_size, True)
                    weights_i   = idx.weights_proj(x) * (idx.softmax_scale * idx.n_heads ** -0.5)
                    index_score = torch.einsum('bshd,btd->bsht', q_i, kv_idx_gpu.to(q_i.dtype))
                    index_score = (index_score.relu_() * weights_i.unsqueeze(-1)).sum(dim=2)
                    topk_k      = min(idx.index_topk, nc)
                    compress_topk_idxs = index_score.topk(topk_k, dim=-1)[1] + win
                else:
                    compress_topk_idxs = self.indexer(x, qr, start_pos, win)
            else:
                compress_topk_idxs = _mm.get_compress_topk_idxs(
                    ratio, bsz, seqlen, start_pos, win).to(dev)
            topk_idxs = torch.cat([topk_idxs, compress_topk_idxs], dim=-1)
        topk_idxs = topk_idxs.int()
        self.kv_cache[:bsz, start_pos % win] = kv.squeeze(1)
        if ratio:
            self.compressor(x, start_pos)
            cpu_buf   = getattr(self, 'cpu_compress_kv', None)
            n_compress = start_pos // ratio
            _parallel_thresh = float('inf')
            use_parallel = False

            if cpu_buf is not None and topk_idxs.shape[-1] > win:
                n_ctok = topk_idxs.shape[-1] - win
                if self.indexer is None:

                    compress_gpu = cpu_buf[:bsz, :n_ctok].to(dev, non_blocking=False)
                else:

                    c_idxs_flat_cpu = (topk_idxs[:, :, win:].long() - win).clamp(min=0).view(bsz, n_ctok).cpu()
                    compress_gpu = cpu_buf[:bsz].gather(
                        1, c_idxs_flat_cpu.unsqueeze(-1).expand(-1, -1, cpu_buf.shape[-1])
                    ).to(dev, non_blocking=False)
                kv_full  = torch.cat([self.kv_cache[:bsz], compress_gpu], dim=1)
                new_topk = torch.arange(win + n_ctok, device=dev, dtype=torch.int32).view(
                    1, 1, -1).expand(bsz, seqlen, -1).contiguous()
                o = _mm.sparse_attn(q, kv_full, self.attn_sink, new_topk, self.softmax_scale)
            else:
                o = _mm.sparse_attn(q, self.kv_cache[:bsz], self.attn_sink, topk_idxs, self.softmax_scale)
        else:
            o = _mm.sparse_attn(q, self.kv_cache[:bsz], self.attn_sink, topk_idxs, self.softmax_scale)

    _mm.apply_rotary_emb(o[..., -rd:], freqs_cis, True)
    o    = o.view(bsz, seqlen, self.n_local_groups, -1)
    wo_a = self.wo_a.weight.view(self.n_local_groups, self.o_lora_rank, -1)
    o    = torch.einsum("bsgd,grd->bsgr", o, wo_a)
    return self.wo_b(o.flatten(2))

_base._OrigAttention.forward = _gpuonly_attn_forward

