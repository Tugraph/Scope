pass
import os, sys, argparse, threading, queue
import torch
import torch.nn.functional as F

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
import bench_resident_kv as rkv

GiB = 1024 ** 3
DEV = rkv.DEV
CLR, QRD = 512, 64
NH = GROUPS = QHD = VHD = NL = None
CHUNK = rkv.CHUNK
BATCH_SIZES = [4, 8, 16, 32]
KV_LENS_NOMINAL = [2048, 4096, 8192, 16384, 32768]

os.environ['OMP_NUM_THREADS'] = '16'
torch.set_num_threads(16)

class _Worker:
    def __init__(self):
        self._q = queue.Queue(); self._res = {}; self._lk = threading.Lock()
        self._thr = threading.Thread(target=self._loop, daemon=True); self._thr.start()
    def _loop(self):
        try: os.sched_setaffinity(0, rkv._NUMA1_CPUS)
        except Exception: pass
        from cpu_attn_ext.build import build as _b
        self._ext = _b()
        self._ext.set_num_threads(16)
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
import time
WORKER = _Worker()
_CPU_EXT = None

def n_resident_compressed(kv, budget_bytes):
    pass
    per_tok = NL * (CLR + QRD) * 2
    return max(0, int(budget_bytes // per_tok)) if per_tok else 0

def _prep_q(B, n_heads):
    q_gpu = torch.randn(n_heads, GROUPS, QHD, dtype=rkv.dtype, device=DEV)
    return q_gpu

def _prep_compressed(budget_n_res, stream_tokens, cpu_tokens):
    pass
    c_res = torch.randn(budget_n_res, CLR, dtype=rkv.dtype, device=DEV)
    p_res = torch.randn(budget_n_res, QRD, dtype=rkv.dtype, device=DEV)

    c_chunks, p_chunks = [], []
    for s in range(0, stream_tokens, CHUNK):
        n = min(CHUNK, stream_tokens - s)
        c_chunks.append(torch.randn(n, CLR, dtype=rkv.dtype).pin_memory())
        p_chunks.append(torch.randn(n, QRD, dtype=rkv.dtype).pin_memory())
    c_cpu = torch.randn(1, max(cpu_tokens, 0), CLR, dtype=rkv.dtype).pin_memory()
    p_cpu = torch.randn(1, max(cpu_tokens, 0), QRD, dtype=rkv.dtype).pin_memory()
    return c_res, p_res, c_chunks, p_chunks, c_cpu, p_cpu

def _decoupled_gpu(q_proj, q_pe, c, p, scale):
    pass
    sc = (q_proj @ c.t() + q_pe @ p.t()) * scale
    m = sc.amax(dim=-1, keepdim=True)
    e = torch.exp(sc - m)
    s = e.sum(dim=-1, keepdim=True)
    o = e @ c
    return o, m, s

def _cpu_attn_compressed(q_proj, q_pe, c_cpu, p_cpu, scale):
    pass
    def fn():
        o_c, mx_c, sm_c = WORKER._ext.fused_cpu_attn_mla_compressed(
            q_proj.cpu(), q_pe.cpu(), c_cpu, p_cpu, scale)
        return o_c.to(DEV), mx_c.to(DEV), sm_c.to(DEV)
    return WORKER.submit(fn), WORKER

def prep_compressed(B, ekv, n_res, gs, cpu_tokens, w_uk):
    pass
    n_heads = B * NH
    q_gpu = torch.randn(B, NH, QHD, dtype=rkv.dtype, device=DEV)
    q_nope, q_pe = torch.split(q_gpu, [128, 64], dim=-1)
    q_proj = torch.einsum('bhd, hdk -> bhk', q_nope, w_uk).reshape(n_heads, CLR).contiguous()
    q_pe_f = q_pe.reshape(n_heads, QRD).contiguous()
    c_res, p_res, c_chunks, p_chunks, c_cpu, p_cpu = _prep_compressed(n_res, gs, cpu_tokens)
    return q_proj, q_pe_f, c_res, p_res, c_chunks, p_chunks, c_cpu, p_cpu

def run_compressed(B, ekv, n_res, gs, cpu_tokens, use_pingpong, scale, w_uk, w_uv,
                   q_proj, q_pe_f, c_res, p_res, c_chunks, p_chunks, c_cpu, p_cpu):
    pass
    n_heads = B * NH

    cpu_res = {}
    thr = None
    if cpu_tokens > 0:
        def _cpu_fn():
            out_c, mx_c, sm_c = WORKER._ext.fused_cpu_attn_mla_compressed(
                q_proj.cpu(), q_pe_f.cpu(), c_cpu, p_cpu, scale)
            cpu_res.update({'o': out_c.to(DEV),
                            'm': mx_c.view(n_heads, 1).to(DEV),
                            's': sm_c.view(n_heads, 1).to(DEV)})
        WORKER.submit(_cpu_fn, 'cpu')

    if n_res > 0:
        o_r, m_r, s_r = _decoupled_gpu(q_proj, q_pe_f, c_res, p_res, scale)
    else:
        o_r = torch.zeros(n_heads * GROUPS, CLR, device=DEV, dtype=torch.float32)
        m_r = torch.full((n_heads * GROUPS, 1), float('-inf'), device=DEV, dtype=torch.float32)
        s_r = torch.zeros(n_heads * GROUPS, 1, device=DEV, dtype=torch.float32)

    if gs > 0 and use_pingpong:
        o_s = torch.zeros(n_heads * GROUPS, CLR, device=DEV, dtype=torch.float32)
        m_s = torch.full((n_heads * GROUPS, 1), float('-inf'), device=DEV, dtype=torch.float32)
        s_s = torch.zeros(n_heads * GROUPS, 1, device=DEV, dtype=torch.float32)
        for cc, pc in zip(c_chunks, p_chunks):
            c_g = cc.to(DEV); p_g = pc.to(DEV)
            o1, m1, s1 = _decoupled_gpu(q_proj, q_pe_f, c_g, p_g, scale)
            mx = torch.maximum(m_s, m1)
            wa = torch.exp(m_s - mx); wb = torch.exp(m1 - mx)
            o_s = o_s * wa + o1 * wb; s_s = s_s * wa + s1 * wb; m_s = mx
    else:
        o_s = torch.zeros(n_heads * GROUPS, CLR, device=DEV, dtype=torch.float32)
        m_s = torch.full((n_heads * GROUPS, 1), float('-inf'), device=DEV, dtype=torch.float32)
        s_s = torch.zeros(n_heads * GROUPS, 1, device=DEV, dtype=torch.float32)

    mx = torch.maximum(m_r, m_s)
    wa = torch.exp(m_r - mx); wb = torch.exp(m_s - mx)
    o_g = o_r * wa + o_s * wb; s_g = s_r * wa + s_s * wb; m_g = mx

    torch.cuda.synchronize(DEV)
    if cpu_tokens > 0:
        WORKER.wait('cpu')
        mx = torch.maximum(m_g, cpu_res['m'])
        wa = torch.exp(m_g - mx); wb = torch.exp(cpu_res['m'] - mx)
        o_all = o_g * wa + cpu_res['o'] * wb
        s_all = s_g * wa + cpu_res['s'] * wb
        o_all = (o_all / s_all.clamp(min=1e-6))
    else:
        o_all = (o_g / s_g.clamp(min=1e-6))

    torch.bmm(o_all.unsqueeze(1).bfloat16(),
              w_uv.unsqueeze(0).expand(B, NH, CLR, 128).reshape(n_heads, CLR, 128))
    torch.cuda.synchronize(DEV)
    return o_all

BASELINES = {
    'solid':     dict(ratio=0.8, budget=0.6 * GiB, pingpong=True,  cpu=False),
    'scout':     dict(ratio=0.8, budget=0.6 * GiB, pingpong=False, cpu=True),
    'hybridgen': dict(ratio=1.0, budget=0.6 * GiB, pingpong=False, cpu=True),
    'fastdecode': dict(ratio=1.0, budget=0,        pingpong=False, cpu=True),
}

def _load_weights():
    pass
    torch.manual_seed(0)
    w_uk = torch.randn(NH, 128, CLR, dtype=rkv.dtype, device=DEV)
    w_uv = torch.randn(NH, CLR, 128, dtype=rkv.dtype, device=DEV)
    scale = QHD ** -0.5
    return w_uk, w_uv, scale

def bench_baseline(bname, full=False, no_resident=False, ratio_override=None):
    global NH, GROUPS, QHD, VHD, NL
    rkv.set_model('ds2')
    NH, GROUPS, QHD, VHD, NL = rkv.NH, rkv.GROUPS, rkv.QHD, rkv.VHD, rkv.NL
    cfg = dict(BASELINES[bname])
    if ratio_override is not None:
        cfg['ratio'] = ratio_override
    w_uk, w_uv, scale = _load_weights()
    tag = " (no-resident)" if no_resident else ""
    if ratio_override is not None:
        tag += f" ratio={ratio_override}"
    print(f"### DeepSeek-V2 / {bname}-adapt (kvlen, compressed MLA{tag})")
    print(f"  budget={cfg['budget']/GiB:.1f}GiB  ratio={cfg['ratio']}  "
          f"per_tok_compressed={NL * (CLR + QRD) * 2}B")
    print("  BS      KV  eff_KV |   n_res      gs | latency_ms")
    print("─" * 60)
    for B in BATCH_SIZES:
        for kv in KV_LENS_NOMINAL:
            ekv = round(kv * cfg['ratio'])
            n_res = 0 if no_resident else min(ekv, n_resident_compressed(ekv, cfg['budget']))
            if cfg['pingpong']:
                gs = ekv - n_res; cpu_tokens = 0
            else:
                gs = 0; cpu_tokens = ekv - n_res
            prep = prep_compressed(B, ekv, n_res, gs, cpu_tokens, w_uk)
            def _run():
                run_compressed(B, ekv, n_res, gs, cpu_tokens, cfg['pingpong'], scale, w_uk, w_uv, *prep)
            for _ in range(3): _run()
            t0 = time.perf_counter()
            for _ in range(10): _run()
            ms = (time.perf_counter() - t0) / 10 * 1000
            print(f"  {B:>2} {kv:>6} {ekv:>6} | {n_res:>6} {gs:>6} | {ms:9.2f}ms", flush=True)
        print()

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--baseline', choices=list(BASELINES), required=True)
    ap.add_argument('--full', action='store_true')
    ap.add_argument('--no-resident', action='store_true', help="_")
    ap.add_argument('--ratio', type=float, default=None, help="_")
    args = ap.parse_args()
    bench_baseline(args.baseline, args.full, args.no_resident, args.ratio)

