#pragma once

#include <deep_ep/common/comm.cuh>
#include <deep_ep/common/compiled.cuh>
#include <deep_ep/common/layout.cuh>
#include <deep_ep/common/ptx.cuh>


namespace deep_ep::elastic {

template <int kNumQPs,
          int kNumEntriesPerRank,
          int kNumHiddenBytes,
          int kNumSFPacks,
          int kNumEntriesPerToken,
          int kNumRDMAPeers,
          int kNumRanksPerRDMAPeer,
          size_t kIntraRankStorageStride,
          int kNumThreads,
          typename team_t,
          int kNumWarps = kNumThreads / 32>
__global__ void __launch_bounds__(kNumThreads, 1)
engram_fetch_impl(const ncclDevComm_t nccl_dev_comm, const ncclWindow_t nccl_window,
                  void* storage, void* fetched, int* indices,
                  const sf_pack_t* sf_table, sf_pack_t* fetched_sf,
                  const int sf_token_stride, const int sf_hidden_stride,
                  const int num_tokens) {
    const auto qp_idx = static_cast<int>(blockIdx.x);
    const auto warp_idx = ptx::get_warp_idx();
    const auto global_warp_idx = qp_idx * kNumWarps + warp_idx;
    const auto thread_idx = static_cast<int>(threadIdx.x);

    // Gin handle
    const auto gin = handle::NCCLGin(nccl_dev_comm, nccl_window, qp_idx, NCCL_GIN_RESOURCE_SHARING_CTA);

    __shared__ int num_requests_per_peer[kNumRDMAPeers];
    EP_STATIC_ASSERT(kNumRDMAPeers <= kNumThreads, "Too many RDMA peers");
    if (thread_idx < kNumRDMAPeers)
        num_requests_per_peer[thread_idx] = 0;
    __syncthreads();

    // Issue RDMA
    const auto issue_rdma_get = [=](const int& token_idx, const int& peer_idx,
                                    const int64_t& src_byte_offset, const int& extra_options = 0) {
        gin.get<team_t, ncclCoopThread, ncclGin_SegmentMixed>(math::advance_ptr(storage, src_byte_offset),
                        math::advance_ptr(fetched, static_cast<int64_t>(token_idx) * kNumHiddenBytes),
                        kNumHiddenBytes, peer_idx, extra_options);
    };

    // Each warp fetches one token cooperatively via RDMA gin.get
    // TODO: deal with padded tokens
    if (ptx::elect_one_sync()) {
        #pragma unroll 4
        for (int i = global_warp_idx; i < num_tokens * kNumEntriesPerToken; i += kNumQPs * kNumWarps) {
            const auto global_idx = __ldg(indices + i);
            const auto owner_rank_idx = global_idx / kNumEntriesPerRank;
            const auto local_entry_idx = global_idx % kNumEntriesPerRank;

            // Route owner rank to RDMA peer and intra-peer rank
            const auto peer_idx = owner_rank_idx / kNumRanksPerRDMAPeer;
            const auto intra_peer_rank_idx = owner_rank_idx % kNumRanksPerRDMAPeer;

            // Byte offset into storage layout
            const auto src_byte_offset = static_cast<int64_t>(intra_peer_rank_idx) * kIntraRankStorageStride +
                                         static_cast<int64_t>(local_entry_idx) * kNumHiddenBytes;

            // Issue RDMA get
            const auto request_idx = atomicAdd_block(num_requests_per_peer + peer_idx, 1);
            issue_rdma_get(
                i, peer_idx, src_byte_offset,
                // NOTES: requests may exceed the queue depth, flush if needed
                (request_idx % kGinQPFlushDepth == (kGinQPFlushDepth - 1)) ? 0 : ncclGinOptFlagsAggregateRequests
            );

            // TODO: once NCCL supports ncclCoopWarp gin.get, drop the elect_one_sync and let the whole warp
            // gather SF packs in parallel.
            if constexpr (kNumSFPacks > 0) {
                EP_STATIC_ASSERT(sizeof(sf_pack_t) == sizeof(int), "SF pack must be 4 bytes");
                const auto token_idx = i / kNumEntriesPerToken;
                const auto entry_idx = i % kNumEntriesPerToken;
                const auto* src = reinterpret_cast<const int*>(sf_table) + global_idx * kNumSFPacks;
                auto* dst = reinterpret_cast<int*>(fetched_sf)
                          + token_idx * sf_token_stride + entry_idx * kNumSFPacks * sf_hidden_stride;
                #pragma unroll
                for (int p = 0; p < kNumSFPacks; ++p)
                    dst[p * sf_hidden_stride] = __ldg(src + p);
            }
        }
    }
    __syncthreads();

    // One device-side completion point per QP makes every issued get visible
    // to subsequent work on the caller-owned stream.
    if (thread_idx == 0)
        gin.flush<ncclCoopThread>();
}

} // namespace deep_ep::elastic
