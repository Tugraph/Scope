pass
import os, sys, time, json, random, queue, threading
import torch
sys.path.insert(0, '/home/yed/software/cpuvision3')
os.environ['MLA_TILE'] = '2048'
from cpu_attn_ext.build_tile import build as _build_tile
_tile_ext = _build_tile(verbose=False)
_tile_ext.set_num_threads(16)
import q53_rerun_ds2 as q53

DEV, nH, CLR, QRD, scale = q53.DEV, q53.nH, q53.CLR, q53.QRD, q53.scale
OLD = '/home/yed/software/cpuvision3/exp53_results/exp53_sweep_chunk.json'
OUT = '/home/yed/software/cpuvision3/exp53_results/ds2_rstar_revalidate.json'
BS = 8
KV_LENS = [2048, 4096, 8192]
ITERS = 10
WARMUP = 3
CHUNK_G = 2048
random.seed(0)

class _Worker:
    def __init__(self):
        self._q = queue.Queue(); self._res = {}; self._lk = threading.Lock()
        self._thr = threading.Thread(target=self._loop, daemon=True); self._thr.start()
    def _loop(self):
        try: os.sched_setaffinity(0, list(range(24, 48)) + list(range(72, 96)))
        except Exception: pass
        _tile_ext.set_num_threads(16)
        while True:
            job = self._q.get()
            if job is None: return
            fn, tag = job
            try: out = fn()
            except Exception as e: out = e
            with self._lk: self._res[tag] = out; self._res['_done'] = tag
    def submit(self, fn, tag='c'):
        with self._lk: self._res['_done'] = None
        self._q.put((fn, tag))
    def wait(self, tag='c'):
        while True:
            with self._lk:
                if self._res.get('_done') == tag:
                    r = self._res.get(tag)
                    if isinstance(r, Exception): raise r
                    return r
            time.sleep(0.0005)

W = _Worker()

def _gpu_stream(qp, qpe, c_chunks, p_chunks, bs, kv_gs):
    n_ch = len(c_chunks)
    sizes = [c.shape[1] for c in c_chunks]
    bufs = [torch.empty(bs, CHUNK_G, CLR, dtype=torch.bfloat16, device=DEV),
            torch.empty(bs, CHUNK_G, CLR, dtype=torch.bfloat16, device=DEV)]
    pbufs = [torch.empty(bs, CHUNK_G, QRD, dtype=torch.bfloat16, device=DEV),
             torch.empty(bs, CHUNK_G, QRD, dtype=torch.bfloat16, device=DEV)]
    oa = torch.zeros(bs * nH, 1, CLR, device=DEV, dtype=torch.float32)
    ma = torch.full((bs * nH, 1, 1), float('-inf'), device=DEV, dtype=torch.float32)
    sa = torch.zeros(bs * nH, 1, 1, device=DEV, dtype=torch.float32)
    q1 = qp.view(bs * nH, 1, CLR)
    q2 = qpe.view(bs * nH, 1, QRD)
    stream = torch.cuda.Stream(device=DEV)
    with torch.cuda.stream(stream):
        bufs[0][:, :sizes[0]].copy_(c_chunks[0], non_blocking=True)
        pbufs[0][:, :sizes[0]].copy_(p_chunks[0], non_blocking=True)
    for ci in range(n_ch):
        csz = sizes[ci]; cur = ci % 2; nxt = 1 - cur
        torch.cuda.current_stream().wait_stream(stream)
        if ci + 1 < n_ch:
            nsz = sizes[ci + 1]
            with torch.cuda.stream(stream):
                bufs[nxt][:, :nsz].copy_(c_chunks[ci + 1], non_blocking=True)
                pbufs[nxt][:, :nsz].copy_(p_chunks[ci + 1], non_blocking=True)
        ce = bufs[cur][:, :csz].repeat_interleave(nH, dim=0)
        pe = pbufs[cur][:, :csz].repeat_interleave(nH, dim=0)
        sc = (torch.bmm(q1, ce.transpose(1, 2)) + torch.bmm(q2, pe.transpose(1, 2))) * scale
        mg = sc.amax(-1, keepdim=True)
        eg = torch.exp(sc - mg)
        sg = eg.sum(-1, keepdim=True)
        og = torch.bmm(eg, ce)
        m_new = torch.maximum(ma, mg)
        w_old = torch.exp(ma - m_new)
        w_new = torch.exp(mg - m_new)
        sa = w_old * sa + w_new * sg
        oa = w_old * oa + w_new * og.float()
        ma = m_new
    return oa, ma, sa

def measure(bs, kv, gs):
    a = dict(
        qp=torch.randn(bs * nH, CLR, dtype=torch.bfloat16, device=DEV),
        qpe=torch.randn(bs * nH, QRD, dtype=torch.bfloat16, device=DEV),
        w=torch.randn(bs * nH, CLR, 128, dtype=torch.bfloat16, device=DEV),
        c_cpu=torch.randn(bs, gs, CLR, dtype=torch.bfloat16).pin_memory() if gs > 0 else None,
        p_cpu=torch.randn(bs, gs, QRD, dtype=torch.bfloat16).pin_memory() if gs > 0 else None,
        c_gpu=torch.randn(bs, kv - gs, CLR, dtype=torch.bfloat16).pin_memory() if kv - gs > 0 else None,
        p_gpu=torch.randn(bs, kv - gs, QRD, dtype=torch.bfloat16).pin_memory() if kv - gs > 0 else None)
    if kv - gs > 0:
        a['c_chunks'] = [a['c_gpu'][:, s:s + CHUNK_G].contiguous().pin_memory()
                         for s in range(0, kv - gs, CHUNK_G)]
        a['p_chunks'] = [a['p_gpu'][:, s:s + CHUNK_G].contiguous().pin_memory()
                         for s in range(0, kv - gs, CHUNK_G)]

    def one():
        def cpu_fn():
            o, m, s = _tile_ext.fused_cpu_attn_mla_compressed(
                a['qp'].view(bs * nH, CLR).cpu(), a['qpe'].view(bs * nH, QRD).cpu(),
                a['c_cpu'], a['p_cpu'], scale)
            return (o.view(bs * nH, 1, CLR).to(DEV),
                    m.view(bs * nH, 1, 1).to(DEV),
                    s.view(bs * nH, 1, 1).to(DEV))
        if gs > 0:
            W.submit(cpu_fn)
        else:
            o_c = torch.zeros(bs * nH, 1, CLR, device=DEV, dtype=torch.bfloat16)
            m_c = torch.full((bs * nH, 1, 1), float('-inf'), device=DEV, dtype=torch.bfloat16)
            s_c = torch.zeros(bs * nH, 1, 1, device=DEV, dtype=torch.bfloat16)
        if kv - gs > 0:
            og, mg, sg = _gpu_stream(a['qp'], a['qpe'], a['c_chunks'], a['p_chunks'], bs, kv - gs)
        else:
            og = torch.zeros(bs * nH, 1, CLR, device=DEV, dtype=torch.float32)
            mg = torch.full((bs * nH, 1, 1), float('-inf'), device=DEV, dtype=torch.float32)
            sg = torch.zeros(bs * nH, 1, 1, device=DEV, dtype=torch.float32)
        torch.cuda.synchronize()
        if gs > 0:
            o_c, m_c, s_c = W.wait()
        M = torch.maximum(m_c, mg)
        wc = torch.exp(m_c - M)
        wg = torch.exp(mg - M)
        om = wc * o_c + wg * og
        sm = wc * s_c + wg * sg
        torch.bmm((om / sm.clamp(min=1e-6)).bfloat16(), a['w'])
        torch.cuda.synchronize()

    for _ in range(WARMUP): one()
    t0 = time.perf_counter()
    for _ in range(ITERS): one()
    ms = (time.perf_counter() - t0) / ITERS * 1000

    for k in list(a.keys()):
        a[k] = None
    torch.cuda.empty_cache()
    return ms

def main():
    old = json.load(open(OLD))
    results = {}
    print(f"_")
    print(f"_")
    print("-" * 56)
    for kv in KV_LENS:
        old_curve = {int(k): v for k, v in old[f'{BS},{kv}']['curve'].items() if v != 'OOM'}

        gs_list = list(old_curve.keys())
        random.shuffle(gs_list)
        new_curve = {}
        for gs in gs_list:
            ms = measure(BS, kv, gs)
            new_curve[gs] = round(ms, 3)
        inflated = 0
        for gs in sorted(old_curve.keys()):
            o = old_curve[gs]
            n = new_curve[gs]
            r = n / o if o > 0 else 0
            flag = ""
            if r < 0.8: flag = "_"; inflated += 1
            elif r < 0.95: flag = "_"
            r_star = gs / kv
            print(f"{kv:>5} {gs:>6} {r_star*100:>5.1f}% | {o:>8.3f} {n:>8.3f} {r:>6.2f} {flag:>8}", flush=True)
        g_opt = min(new_curve, key=new_curve.get)
        old_g_opt = min(old_curve, key=old_curve.get)
        results[f'{BS},{kv}'] = {
            'new_curve': new_curve,
            'new_gs_opt': g_opt, 'new_r_star': round(g_opt/kv, 3),
            'new_ms_opt': new_curve[g_opt],
            'old_gs_opt': old_g_opt, 'old_ms_opt': old_curve[old_g_opt],
            'inflated_points': inflated,
        }
        json.dump(results, open(OUT, 'w'), indent=1)
        print(f"_"
              f"_", flush=True)
    print(f"_")

if __name__ == '__main__':
    main()

