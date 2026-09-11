# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import nullcontext
from unittest.mock import MagicMock, patch

import pytest

from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorBase_V1
from vllm.distributed.kv_transfer.kv_connector.v1.example_connector import (  # noqa: E501
    ExampleConnectorMetadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import MultiConnector
from vllm.distributed.kv_transfer.kv_connector.v1.simple_cpu_offload_connector import (
    SimpleCPUOffloadConnector,
)
from vllm.distributed.kv_transfer.kv_transfer_state import (
    ensure_kv_transfer_initialized,
    ensure_kv_transfer_shutdown,
    get_kv_transfer_group,
)
from vllm.v1.core.sched.output import CachedRequestData, SchedulerOutput
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.simple_kv_offload.metadata import SimpleCPUOffloadMetadata
from vllm.v1.simple_kv_offload.worker import SimpleCPUOffloadWorker
from vllm.v1.worker.gpu.kv_connector import ActiveKVConnector
from vllm.v1.worker.kv_connector_model_runner_mixin import KVConnectorModelRunnerMixin

# Importing utils registers TestExampleConnector with the factory
from .utils import create_vllm_config


def _make_empty_scheduler_output():
    return SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens={},
        total_num_scheduled_tokens=0,
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[],
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
        kv_connector_metadata=ExampleConnectorMetadata(),
    )


def test_kv_connector_mixin_clears_metadata():
    vllm_config = create_vllm_config(
        kv_connector="TestExampleConnector",
        kv_role="kv_both",
        kv_connector_extra_config={"name": "unit"},
    )

    kv_cache_config = KVCacheConfig(
        num_blocks=0, kv_cache_tensors=[], kv_cache_groups=[]
    )
    # Initialize the global connector instance.
    # kv_transfer init now syncs engine_id across TP, so unit tests need
    # a minimal mocked TP group.
    mock_tp_group = MagicMock()
    mock_tp_group.broadcast_object.side_effect = lambda value, src=0: value

    with patch(
        "vllm.distributed.parallel_state.get_tp_group",
        return_value=mock_tp_group,
    ):
        ensure_kv_transfer_initialized(vllm_config, kv_cache_config)

    try:
        # Minimal scheduler output with empty metadata; mixin should still
        # bind/clear metadata even if no loads happen
        scheduler_output = _make_empty_scheduler_output()

        # Invoke the no-forward path which uses the mixin context manager
        KVConnectorModelRunnerMixin.kv_connector_no_forward(
            scheduler_output, vllm_config
        )

        # Verify clear_connector_metadata was called on the connector
        connector = get_kv_transfer_group()
        assert connector._connector_metadata is None
        # Test connector wrapper records method calls
        assert connector.call_record.get("bind_connector_metadata", 0) == 1
        assert connector.call_record.get("clear_connector_metadata", 0) == 1
    finally:
        # Ensure we clean up the global connector between tests
        ensure_kv_transfer_shutdown()


@pytest.fixture
def store_connector(monkeypatch):
    connector = SimpleCPUOffloadConnector.__new__(SimpleCPUOffloadConnector)
    connector.worker_handler = SimpleCPUOffloadWorker(None, None, 0)
    connector.scheduler_manager = None
    connector.worker_handler._backend = MagicMock()
    monkeypatch.setattr("torch.Event", MagicMock())
    monkeypatch.setattr("torch.cuda.current_stream", MagicMock())
    prefix = "vllm.v1.worker.kv_connector_model_runner_mixin."
    monkeypatch.setattr(prefix + "get_kv_transfer_group", lambda: connector)
    monkeypatch.setattr(prefix + "has_kv_transfer_group", lambda: True)
    monkeypatch.setattr(prefix + "get_forward_context", MagicMock())
    monkeypatch.setattr(prefix + "set_forward_context", lambda *a, **kw: nullcontext())
    monkeypatch.setattr(
        "vllm.v1.worker.gpu.kv_connector.is_forward_context_available", lambda: True
    )
    monkeypatch.setattr(
        "vllm.v1.worker.gpu.kv_connector.get_forward_context", MagicMock()
    )
    return connector


def _make_store_scheduler_output():
    output = _make_empty_scheduler_output()
    output.kv_connector_metadata = SimpleCPUOffloadMetadata(
        store_event=1, store_gpu_blocks=[1], store_cpu_blocks=[2]
    )
    return output


@pytest.mark.parametrize("runner_v2", [False, True])
@pytest.mark.parametrize("opt_in", [False, True])
def test_idle_runner_saves_only_opted_in_connectors(
    store_connector, runner_v2, opt_in, monkeypatch
):
    connector = store_connector
    if not opt_in:
        monkeypatch.setattr(
            SimpleCPUOffloadConnector,
            "save_kv_no_forward",
            KVConnectorBase_V1.save_kv_no_forward,
        )
    output = _make_store_scheduler_output()
    if runner_v2:
        runner = ActiveKVConnector.__new__(ActiveKVConnector)
        runner.kv_connector = connector
        runner._disabled = False
        runner._pending_load_start = False
        runner.no_forward(output)
    else:
        KVConnectorModelRunnerMixin.kv_connector_no_forward(output, MagicMock())
    backend = connector.worker_handler._backend
    assert backend.launch_copy.call_count == int(opt_in)
    assert connector.worker_handler._connector_metadata is None


def test_deferred_finalization_submits_store_after_draft(store_connector):
    connector = store_connector
    backend = connector.worker_handler._backend
    with KVConnectorModelRunnerMixin._get_kv_connector_output(
        _make_store_scheduler_output(), defer_finalize=True
    ):
        pass  # Target forward; draft has not run yet.
    backend.launch_copy.assert_not_called()
    assert connector.worker_handler._store_compute_done is None
    assert connector.worker_handler._connector_metadata is not None

    # The runner finalizes only after the draft forward has completed.
    KVConnectorModelRunnerMixin.finalize_kv_connector()
    backend.launch_copy.assert_called_once()
    assert backend.launch_copy.call_args.kwargs["wait_event"] is not None
    assert connector.worker_handler._connector_metadata is None


def test_multi_connector_preserves_idle_save_opt_in(store_connector):
    legacy = MagicMock(spec=KVConnectorBase_V1)
    legacy.save_kv_no_forward.side_effect = (
        lambda: KVConnectorBase_V1.save_kv_no_forward(legacy)
    )
    multi = MultiConnector.__new__(MultiConnector)
    multi._connectors = [legacy, store_connector]
    store_connector.bind_connector_metadata(
        _make_store_scheduler_output().kv_connector_metadata
    )
    multi.save_kv_no_forward()
    legacy.wait_for_save.assert_not_called()
    store_connector.worker_handler._backend.launch_copy.assert_called_once()
