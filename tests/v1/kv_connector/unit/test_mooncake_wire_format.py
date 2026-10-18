# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import msgspec
import pytest

from vllm.distributed.kv_transfer.kv_connector.v1.mooncake import mooncake_connector
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.metadata import (
    MooncakeConnectorMetadata,
    PullReqMeta,
    ReqId,
    SendBlockMeta,
    TransferId,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.protocol import (
    MooncakeXferMetadata,
    MooncakeXferResponse,
    MooncakeXferResponseStatus,
)

# Encoded by the definitions as they stood in mooncake_connector.py before the
# metadata and wire structs moved out. Every field carries a distinct value, so
# a reordered field, a changed default, a changed omit_defaults or a changed
# status value changes these bytes.
_GOLDEN_METADATA_HEX = (
    "8faf72656d6f74655f686f73746e616d65a931302e302e302e3131ab72656d6f74655f706f"
    "7274cd3cc6ae72656d6f74655f74705f73697a6504ae72656d6f74655f74705f72616e6b03"
    "aa7265715f626c6f636b7382a57265712d6192a6786665722d6192930102039107a5726571"
    "2d6292a6786665722d6290b36b765f6361636865735f626173655f6164647293cd1000cd20"
    "00cd4000aa626c6f636b5f6c656e7393cd1000cd1000cd2000ad6b765f626c6f636b5f6c65"
    "6e7393cd0800cd0800cd1000b6726567697374657265645f6c617965725f6e616d657392a7"
    "6c617965722e30a76c617965722e31b8726567697374657265645f6c617965725f696e6469"
    "636573920001b8726567697374657265645f67726f75705f696e6469636573920001bb7265"
    "67697374657265645f7368617265645f67726f75705f69647391920001b672656769737465"
    "7265645f726f775f6f66667365747392ffcd0200ae72656d6f74655f70705f73697a6502a9"
    "6c61796f75745f696407"
)

# The four responses share every field but the status code, so the encoded bytes
# differ only in the status value.
_GOLDEN_RESPONSE_HEX = {
    MooncakeXferResponseStatus.FINISH: (
        "85a673746174757300"
        "a76f6b5f7265717391a57265712d61"
        "a86572725f7265717391a57265712d62"
        "a76572725f6d7367a4626f6f6d"
        "a96c61796f75745f696407"
    ),
    MooncakeXferResponseStatus.CONTINUE: (
        "85a673746174757301"
        "a76f6b5f7265717391a57265712d61"
        "a86572725f7265717391a57265712d62"
        "a76572725f6d7367a4626f6f6d"
        "a96c61796f75745f696407"
    ),
    MooncakeXferResponseStatus.ERROR: (
        "85a673746174757302"
        "a76f6b5f7265717391a57265712d61"
        "a86572725f7265717391a57265712d62"
        "a76572725f6d7367a4626f6f6d"
        "a96c61796f75745f696407"
    ),
    MooncakeXferResponseStatus.LAYOUT_MISS: (
        "85a673746174757303"
        "a76f6b5f7265717391a57265712d61"
        "a86572725f7265717391a57265712d62"
        "a76572725f6d7367a4626f6f6d"
        "a96c61796f75745f696407"
    ),
}


def _metadata() -> MooncakeXferMetadata:
    return MooncakeXferMetadata(
        remote_hostname="10.0.0.11",
        remote_port=15558,
        remote_tp_size=4,
        remote_tp_rank=3,
        req_blocks={"req-a": ("xfer-a", [[1, 2, 3], [7]]), "req-b": ("xfer-b", [])},
        kv_caches_base_addr=[4096, 8192, 16384],
        block_lens=[4096, 4096, 8192],
        kv_block_lens=[2048, 2048, 4096],
        registered_layer_names=["layer.0", "layer.1"],
        registered_layer_indices=[0, 1],
        registered_group_indices=[0, 1],
        registered_shared_group_ids=[[0, 1]],
        registered_row_offsets=[-1, 512],
        remote_pp_size=2,
        layout_id=7,
    )


def _response(status: MooncakeXferResponseStatus) -> MooncakeXferResponse:
    return MooncakeXferResponse(
        status=status,
        ok_reqs=["req-a"],
        err_reqs=["req-b"],
        err_msg="boom",
        layout_id=7,
    )


def test_xfer_metadata_wire_format_is_unchanged():
    assert MooncakeXferMetadata.__struct_fields__ == (
        "remote_hostname",
        "remote_port",
        "remote_tp_size",
        "remote_tp_rank",
        "req_blocks",
        "kv_caches_base_addr",
        "block_lens",
        "kv_block_lens",
        "registered_layer_names",
        "registered_layer_indices",
        "registered_group_indices",
        "registered_shared_group_ids",
        "registered_row_offsets",
        "remote_pp_size",
        "layout_id",
    )
    assert msgspec.msgpack.encode(_metadata()).hex() == _GOLDEN_METADATA_HEX
    decoded = msgspec.msgpack.Decoder(MooncakeXferMetadata).decode(
        bytes.fromhex(_GOLDEN_METADATA_HEX)
    )
    assert decoded == _metadata()


@pytest.mark.parametrize("status", list(MooncakeXferResponseStatus))
def test_xfer_response_wire_format_is_unchanged(status: MooncakeXferResponseStatus):
    assert MooncakeXferResponse.__struct_fields__ == (
        "status",
        "ok_reqs",
        "err_reqs",
        "err_msg",
        "layout_id",
    )
    golden = _GOLDEN_RESPONSE_HEX[status]
    assert msgspec.msgpack.encode(_response(status)).hex() == golden
    decoded = msgspec.msgpack.Decoder(MooncakeXferResponse).decode(
        bytes.fromhex(golden)
    )
    assert decoded == _response(status)


def test_xfer_response_status_values_are_unchanged():
    assert [(status.name, int(status)) for status in MooncakeXferResponseStatus] == [
        ("FINISH", 0),
        ("CONTINUE", 1),
        ("ERROR", 2),
        ("LAYOUT_MISS", 3),
    ]


def test_mooncake_connector_still_exports_the_moved_names():
    assert mooncake_connector.MooncakeXferMetadata is MooncakeXferMetadata
    assert mooncake_connector.MooncakeXferResponse is MooncakeXferResponse
    assert mooncake_connector.MooncakeXferResponseStatus is MooncakeXferResponseStatus
    assert mooncake_connector.MooncakeConnectorMetadata is MooncakeConnectorMetadata
    assert mooncake_connector.PullReqMeta is PullReqMeta
    assert mooncake_connector.SendBlockMeta is SendBlockMeta
    assert mooncake_connector.ReqId is ReqId
    assert mooncake_connector.TransferId is TransferId
