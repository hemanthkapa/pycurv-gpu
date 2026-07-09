"""
Delta-stepping SSSP on compact subgraphs (batched).

Drop-in dense solver for sssp_triangle_batch when --sssp delta is selected.
"""

import math
import torch


def _sssp_delta_stepping(edge_src, edge_dst, edge_dist, num_nodes,
                         sources, g_max):
    """
    Batched delta-stepping SSSP on a compact subgraph.

    Processes nodes in distance buckets of width delta. Settled nodes are
    never revisited, reducing total relaxations vs SPFA on mesh graphs.
    """
    device = edge_src.device
    B = sources.shape[0]
    V = num_nodes

    delta = edge_dist.median().item()
    if delta <= 0:
        delta = edge_dist[edge_dist > 0].min().item() if (edge_dist > 0).any() else 1.0

    dist = torch.full((B, V), float('inf'), dtype=torch.float32, device=device)
    dist[torch.arange(B, device=device), sources] = 0.0

    settled = torch.zeros((B, V), dtype=torch.bool, device=device)

    light_mask = edge_dist <= delta
    heavy_mask = ~light_mask

    light_src = edge_src[light_mask]
    light_dst = edge_dst[light_mask]
    light_w = edge_dist[light_mask]
    heavy_src = edge_src[heavy_mask]
    heavy_dst = edge_dst[heavy_mask]
    heavy_w = edge_dist[heavy_mask]

    bucket_lo = 0.0
    max_buckets = int(math.ceil((g_max + 1e-6) / delta)) + 1

    for _ in range(max_buckets):
        bucket_hi = bucket_lo + delta

        in_bucket = (dist >= bucket_lo) & (dist < bucket_hi) & ~settled

        if not in_bucket.any():
            unsettled_valid = ~settled & (dist <= g_max)
            if not unsettled_valid.any():
                break
            masked_dist = dist.clone()
            masked_dist[settled | (dist > g_max)] = float('inf')
            min_remaining = masked_dist.min().item()
            if min_remaining == float('inf'):
                break
            bucket_lo = (min_remaining // delta) * delta
            continue

        for _ in range(V):
            active = in_bucket
            any_active_col = active.any(dim=0)

            if not any_active_col.any():
                break

            src_active = any_active_col[light_src]
            if not src_active.any():
                break

            a_src = light_src[src_active]
            a_dst = light_dst[src_active]
            a_w = light_w[src_active]
            A = a_src.shape[0]

            row_active = active[:, a_src]
            proposal = dist[:, a_src] + a_w.unsqueeze(0)
            proposal[~row_active] = float('inf')
            proposal = proposal.clamp(max=g_max + 1e-6)

            dst_expanded = a_dst.unsqueeze(0).expand(B, A)
            old_at_dst = dist.gather(1, dst_expanded).clone()
            dist.scatter_reduce_(1, dst_expanded, proposal, reduce='amin', include_self=True)

            new_at_dst = dist.gather(1, dst_expanded)
            any_improved = (new_at_dst < old_at_dst - 1e-10).any()

            in_bucket = (dist >= bucket_lo) & (dist < bucket_hi) & ~settled

            if not any_improved:
                break

        to_settle = (dist >= bucket_lo) & (dist < bucket_hi) & ~settled
        settled |= to_settle

        if heavy_src.numel() > 0:
            settled_col = to_settle.any(dim=0)
            src_settled = settled_col[heavy_src]

            if src_settled.any():
                a_src = heavy_src[src_settled]
                a_dst = heavy_dst[src_settled]
                a_w = heavy_w[src_settled]
                A = a_src.shape[0]

                row_settled = to_settle[:, a_src]
                proposal = dist[:, a_src] + a_w.unsqueeze(0)
                proposal[~row_settled] = float('inf')
                proposal = proposal.clamp(max=g_max + 1e-6)

                dst_expanded = a_dst.unsqueeze(0).expand(B, A)
                dist.scatter_reduce_(1, dst_expanded, proposal, reduce='amin', include_self=True)

        bucket_lo = bucket_hi

    return dist
