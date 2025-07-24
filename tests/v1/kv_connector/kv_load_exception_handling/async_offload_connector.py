import logging
from typing import Optional
import random
from dataclasses import dataclass
import torch

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1, KVConnectorMetadata, KVConnectorRole)
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.distributed.parallel_state import (
    get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size,
    get_tp_group)

logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)


@dataclass
class AsyncOffloadConnectorMetadata(KVConnectorMetadata):
    req_meta: dict[str, int]

class AsyncOffloadConnector(KVConnectorBase_V1):
    def __init__(self, vllm_config: "VllmConfig", role: KVConnectorRole):
        super().__init__(vllm_config=vllm_config, role=role)

        self.failure_request:list[str] = []
        self._reqs_need_recv:dict[str, int] = {}
        self._finish_load:dict[str, int] = {}
        
        self.chunk_size = 256
        
        if role == KVConnectorRole.WORKER:
            self.tp_rank = get_tensor_model_parallel_rank()
            self.world_size = get_tensor_model_parallel_world_size()
            self.tp_group = get_tp_group()
        
    ############################################################
    # Scheduler Side Methods
    ############################################################

    def get_num_new_matched_tokens(
            self, request: "Request",
            num_computed_tokens: int) -> tuple[int, bool]:
        if request.request_id in self.failure_request:
            self.failure_request.remove(request.request_id)
            return 0, False
        num_external_hit_tokens = request.num_prompt_tokens // self.chunk_size * self.chunk_size
        logger.info(f"request {request.request_id} request.num_prompt_tokens {request.num_prompt_tokens} num_external_hit_tokens {num_external_hit_tokens}")
        return num_external_hit_tokens, True

    def update_state_after_alloc(self, request: "Request",
                                 blocks: "KVCacheBlocks",
                                 num_external_tokens: int):
        if num_external_tokens > 0:
            self._reqs_need_recv[request.request_id] = request.prompt_token_ids[:num_external_tokens]

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> KVConnectorMetadata:
        req_meta= self._reqs_need_recv.copy()
        self._reqs_need_recv.clear()
        return AsyncOffloadConnectorMetadata(req_meta)
    
    def add_failure_request(self, request: "Request"):
        self.failure_request.append(request.request_id)
        
    def start_load_kv(self, forward_context, **kwargs)-> None:
        for request_id, hit_tokens in self._get_connector_metadata().req_meta.items():
            num_actual_load_tokens = self.load_kv(request_id, hit_tokens)
            logger.info(f"request {request_id} hit_tokens {len(hit_tokens)} num_actual_load_tokens {num_actual_load_tokens}")
            self._finish_load[request_id] = num_actual_load_tokens
    
    def wait_for_layer_load(self, layer_name: str) -> None:
        pass

    def save_kv_layer(self, layer_name: str, kv_layer: torch.Tensor,
                      attn_metadata: "AttentionMetadata", **kwargs) -> None:
        pass

    def wait_for_save(self):
        pass
    
    def load_kv(self, request_id, hit_tokens):
        num_actual_load_tokens = random.randint(0, len(hit_tokens))
        return num_actual_load_tokens
    
    def get_finish_loading(self) -> dict[str, int]:
        if not self._finish_load:
            return None
        finish_loading = self._finish_load.copy()
        self._finish_load.clear()
        if self.tp_rank == 0:
            all_rank_finish_loading = [finish_loading]
            for i in range(1, self.world_size):
                all_rank_finish_loading.append(self.tp_group.recv_object(src=i))
            all_request_ids = [set(worker.keys()) for worker in all_rank_finish_loading]
            common_request_ids = set.intersection(*all_request_ids)
            result = {}
            for req_id in common_request_ids:
                min_quantity = min(worker[req_id] for worker in all_rank_finish_loading if req_id in worker)
                result[req_id] = min_quantity
            return result
        else:
            self.tp_group.send_object(finish_loading, dst=0)
            return finish_loading