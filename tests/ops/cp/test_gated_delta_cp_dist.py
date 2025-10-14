# test_gated_delta_cp_dist.py
# -*- coding: utf-8 -*-

import os
import socket
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

from fla.ops.gated_delta_rule.chunk import chunk_gated_delta_rule_fwd, chunk_gated_delta_rule_bwd
from fla.utils import assert_close, device as ref_device  # avoid shadowing

# ---------- utilities ----------

def _free_tcp_port() -> str:
    """ Find a free TCP port for distributed setup """
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("", 0))
    port = s.getsockname()[1]
    s.close()
    return str(port)

def _setup(rank: int, world_size: int, master_port: str):
    """ Initialize distributed environment """
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = master_port
    # Optional: quieter NCCL unless debugging
    os.environ.setdefault("NCCL_ASYNC_ERROR_HANDLING", "1")
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

def _cleanup():
    """ Clean up distributed environment """
    dist.destroy_process_group()

def _baton_recv(rank: int, device: torch.device, shape) -> torch.Tensor:
    """ Receive h0 from previous rank; zeros if rank 0 """
    if rank == 0:
        return torch.zeros(*shape, dtype=torch.float32, device=device)
    h0 = torch.empty(*shape, dtype=torch.float32, device=device)
    dist.recv(h0, src=rank - 1)
    return h0

def _baton_send(rank: int, world_size: int, ht: torch.Tensor):
    """ Send ht to next rank; no-op if last rank """
    if rank < world_size - 1:
        dist.send(ht, dst=rank + 1)

def _gradient_baton_recv(rank: int, world_size: int, device: torch.device, shape) -> torch.Tensor:
    """ Receive dht from next rank; zeros if last rank """
    if rank == world_size - 1:
        return torch.zeros(*shape, dtype=torch.float32, device=device)
    dht = torch.empty(*shape, dtype=torch.float32, device=device)
    dist.recv(dht, src=rank + 1)
    return dht

def _gradient_baton_send(rank: int, dht: torch.Tensor):
    """ Send dht to previous rank; no-op if rank 0 """
    if rank > 0:
        dist.send(dht, dst=rank - 1)

# ---------- worker ----------

def _forward_worker(rank: int, world_size: int,
                    q_list, k_list, v_list, g_list, beta_list,
                    scale: float, master_port: str, out_dict):
    """ Distributed forward worker """
    _setup(rank, world_size, master_port)
    try:
        device = torch.device(f"cuda:{rank}")
        q = q_list[rank].to(device)
        k = k_list[rank].to(device)
        v = v_list[rank].to(device)
        g = g_list[rank].to(device)
        beta = beta_list[rank].to(device)

        B, T_local, H, D = q.shape
        assert T_local > 0, "empty shard"
        h0 = _baton_recv(rank, device, (B, H, D, D))

        # local forward
        g_out, o, A, ht = chunk_gated_delta_rule_fwd(
            q=q, k=k, v=v, g=g, beta=beta, scale=scale,
            initial_state=h0, output_final_state=True,
        )

        # pass baton
        _baton_send(rank, world_size, ht)

        # gather o, g_out, A, h0 across sequence shards
        dist.barrier()
        o_list = [torch.empty_like(o) for _ in range(world_size)]
        g_out_list = [torch.empty_like(g_out) for _ in range(world_size)]
        A_list = [torch.empty_like(A) for _ in range(world_size)]
        h0_list = [torch.empty_like(h0) for _ in range(world_size)]

        dist.all_gather(o_list, o)
        dist.all_gather(g_out_list, g_out)
        dist.all_gather(A_list, A)
        dist.all_gather(h0_list, h0)

        if rank == 0:
            out_dict["o_dist"] = torch.cat(o_list, dim=1).detach().cpu()
            out_dict["g_out_dist"] = torch.cat(g_out_list, dim=1).detach().cpu()
            out_dict["A_dist"] = torch.cat(A_list, dim=1).detach().cpu()
            out_dict["h0_dist"] = torch.cat(h0_list, dim=1).detach().cpu()
    finally:
        _cleanup()


def _backward_worker(rank: int, world_size: int,
                     q_list, k_list, v_list, g_list, beta_list,
                     g_out_list, A_list, h0_list,
                     do_list, scale: float, master_port: str, out_dict):
    """ Distributed backward worker """
    
    _setup(rank, world_size, master_port)

    try:
        device = torch.device(f"cuda:{rank}")

        q, k, v, g, beta, g_out, A, h0, do = map(lambda x : x[rank].to(device),
            (q_list, k_list, v_list, g_list, beta_list, g_out_list, A_list, h0_list, do_list)
        )

        B, T_local, H, D = q.shape
        assert T_local > 0, "empty shard"

        # Receive dht from next rank
        dht = _gradient_baton_recv(rank, world_size, device, (B, H, D, D))

        dq, dk, dv, db, dg, dh0 = chunk_gated_delta_rule_bwd(q=q, k=k, v=v,
            g=g_out, beta=beta, A=A, scale=scale,
            initial_state=h0, do=do, dht=dht,
        )

        # Pass dht to previous rank
        _gradient_baton_send(rank, dh0)

        # gather gradients across sequence shards
        dist.barrier()
        dq_list = [torch.empty_like(dq) for _ in range(world_size)]
        dk_list = [torch.empty_like(dk) for _ in range(world_size)]
        dv_list = [torch.empty_like(dv) for _ in range(world_size)]
        db_list = [torch.empty_like(db) for _ in range(world_size)]
        dg_list = [torch.empty_like(dg) for _ in range(world_size)]

        dist.all_gather(dq_list, dq)
        dist.all_gather(dk_list, dk)
        dist.all_gather(dv_list, dv)
        dist.all_gather(db_list, db)
        dist.all_gather(dg_list, dg)

        if rank == 0:
            out_dict["dq_dist"] = torch.cat(dq_list, dim=1).detach().cpu()
            out_dict["dk_dist"] = torch.cat(dk_list, dim=1).detach().cpu()
            out_dict["dv_dist"] = torch.cat(dv_list, dim=1).detach().cpu()
            out_dict["db_dist"] = torch.cat(db_list, dim=1).detach().cpu()
            out_dict["dg_dist"] = torch.cat(dg_list, dim=1).detach().cpu()

    finally:
        _cleanup()

# ---------- tests ----------

class TestDistributedCP:
    """
    Validates distributed Context-Parallel (CP) forward and backward equivalence
    against single-GPU reference for GatedDeltaRule kernels.
    """
    B: int = 1
    T: int = 256
    H: int = 1
    D: int = 64
    scale: float = 1.0
    gate_logit_normalizer: float = 1.0
    mask_p: float = 0.0
    dtype: torch.dtype = torch.bfloat16

    def cp_shard(self, x: torch.Tensor, cp_degree: int, shard_dim: int = 1):
        return list(x.tensor_split(cp_degree, dim=shard_dim))

    @pytest.mark.parametrize("cp_degree", [2, 4, 8])
    def test_cp_forward_matches_reference(self, cp_degree: int):
        torch.manual_seed(42)
        assert torch.cuda.device_count() >= cp_degree, "Need >= cp_degree GPUs"
        assert self.T % cp_degree == 0, "T must be divisible by cp_degree"

        # ---- inputs on reference device (single GPU) ----
        q = torch.rand(self.B, self.T, self.H, self.D, dtype=self.dtype)
        k = torch.rand(self.B, self.T, self.H, self.D, dtype=self.dtype)
        v = torch.rand(self.B, self.T, self.H, self.D, dtype=self.dtype)
        beta = torch.rand(self.B, self.T, self.H, dtype=self.dtype).sigmoid()
        g = F.logsigmoid(torch.rand(self.B, self.T, self.H, dtype=torch.float32))
        g = (g / self.gate_logit_normalizer) * (torch.rand_like(g) > self.mask_p)
        h0 = torch.zeros(self.B, self.H, self.D, self.D, dtype=torch.float32)

        # move once; normalize once; keep consistent with fake CP
        q, k, v, beta, g, h0 = map(lambda x: x.to(ref_device), (q, k, v, beta, g, h0))
     

        qn = F.normalize(q, p=2, dim=-1)
        kn = F.normalize(k, p=2, dim=-1)

        # ---- reference forward (no CP) ----
        _, o_ref, _, _ = chunk_gated_delta_rule_fwd(
            q=qn, k=kn, v=v, g=g, beta=beta, scale=self.scale,
            initial_state=h0, output_final_state=True,
        )
        o_ref_cpu = o_ref.detach().cpu()

        # ---- shard inputs for CP workers ----
        q_list = self.cp_shard(qn, cp_degree)
        k_list = self.cp_shard(kn, cp_degree)
        v_list = self.cp_shard(v,  cp_degree)
        g_list = self.cp_shard(g,  cp_degree)
        b_list = self.cp_shard(beta, cp_degree)

        # ---- spawn workers ----
        port = _free_tcp_port()
        mgr = mp.Manager()
        out_dict = mgr.dict()

        mp.spawn(
            _forward_worker,
            args=(cp_degree, q_list, k_list, v_list, g_list, b_list, self.scale, port, out_dict),
            nprocs=cp_degree,
            join=True,
        )

        # ---- compare ----
        o_dist = out_dict["o_dist"]
        assert_close("o", o_ref_cpu, o_dist, 0.002)

    @pytest.mark.parametrize("cp_degree", [2, 4, 8])
    def test_cp_backward_matches_reference(self, cp_degree: int):
        torch.manual_seed(42)

        assert torch.cuda.device_count() >= cp_degree, "Need >= cp_degree GPUs"
        assert self.T % cp_degree == 0, "T must be divisible by cp_degree"

        # ---- inputs on reference device (single GPU) ----
        q = torch.rand(self.B, self.T, self.H, self.D, dtype=self.dtype)
        k = torch.rand(self.B, self.T, self.H, self.D, dtype=self.dtype)
        v = torch.rand(self.B, self.T, self.H, self.D, dtype=self.dtype)
        beta = torch.rand(self.B, self.T, self.H, dtype=self.dtype).sigmoid()
        g = F.logsigmoid(torch.rand(self.B, self.T, self.H, dtype=torch.float32))
        g = (g / self.gate_logit_normalizer) * (torch.rand_like(g) > self.mask_p)
        h0 = torch.zeros(self.B, self.H, self.D, self.D, dtype=torch.float32)

        # move once; normalize once; keep consistent with fake CP
        q, k, v, beta, g, h0 = map(lambda x: x.to(ref_device), (q, k, v, beta, g, h0))

        # --- Gradient targets ---
        do = torch.rand_like(v)
     
        qn = F.normalize(q, p=2, dim=-1)
        kn = F.normalize(k, p=2, dim=-1)

        # ---- reference forward/backward (no CP) ----

        g_out_ref, o_ref, A_ref, _ = chunk_gated_delta_rule_fwd(
            q=qn, k=kn, v=v, g=g, beta=beta, scale=self.scale,
            initial_state=h0, output_final_state=True,
        )

        dht = torch.zeros_like(h0, dtype=torch.float32)

        dq_ref, dk_ref, dv_ref, db_ref, dg_ref, dh0_ref = chunk_gated_delta_rule_bwd(
            q=qn, k=kn, v=v, g=g_out_ref, beta=beta, A=A_ref, scale=self.scale,
            initial_state=h0, do=do, dht=dht,
        )

        dq_ref_cpu = dq_ref.detach().cpu()
        dk_ref_cpu = dk_ref.detach().cpu()
        dv_ref_cpu = dv_ref.detach().cpu()
        db_ref_cpu = db_ref.detach().cpu()
        dg_ref_cpu = dg_ref.detach().cpu()


        ### ---- shard inputs for CP workers ----
        q_list = self.cp_shard(qn, cp_degree)
        k_list = self.cp_shard(kn, cp_degree)
        v_list = self.cp_shard(v,  cp_degree)
        g_list = self.cp_shard(g,  cp_degree)
        b_list = self.cp_shard(beta, cp_degree)
        do_list = self.cp_shard(do, cp_degree)

        # ---- spawn workers ----
        port = _free_tcp_port()
        mgr = mp.Manager()
        out_dict = mgr.dict()

        mp.spawn(
            _forward_worker,
            args=(cp_degree, q_list, k_list, v_list, g_list, b_list, self.scale, port, out_dict),
            nprocs=cp_degree,
            join=True,
        )

        # ---- shard outputs of forward pass for backward pass ----
        g_out_dist = out_dict["g_out_dist"]
        A_dist = out_dict["A_dist"]
        h0_dist = out_dict["h0_dist"]

        g_out_list = self.cp_shard(g_out_dist, cp_degree)
        A_list = self.cp_shard(A_dist, cp_degree)
        h0_list = self.cp_shard(h0_dist, cp_degree)

        # ---- spawn backward workers ----
        port = _free_tcp_port()
        out_dict = mgr.dict()   
        mp.spawn(
            _backward_worker,
            args=(cp_degree, q_list, k_list, v_list, g_list, b_list,
                  g_out_list, A_list, h0_list, do_list,
                  self.scale, port, out_dict),
            nprocs=cp_degree,
            join=True,
        )

        # ---- compare ----
        dq_dist = out_dict["dq_dist"]
        dk_dist = out_dict["dk_dist"]
        dv_dist = out_dict["dv_dist"]
        db_dist = out_dict["db_dist"]
        dg_dist = out_dict["dg_dist"]

        # needs somewhat looser tolerance with larger cp_degree
        assert_close("dq", dq_ref_cpu, dq_dist, 0.01)
        assert_close("dk", dk_ref_cpu, dk_dist, 0.01) 
        assert_close("dv", dv_ref_cpu, dv_dist, 0.01)  
        assert_close("db", db_ref_cpu, db_dist, 0.01)
        assert_close("dg", dg_ref_cpu, dg_dist, 0.01)








 