"""Shared host harness: real target layouts/validators with a deferred HIP queue."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROVIDERS = ROOT / 'native/providers/gdn'


def write_target_harness(directory):
    for name in ('sm121_q2_target.h', 'sm121_q2_publication.h', 'sm121_q2_model_weights.h', 'sm121_mtp_model_weights.h'):
        (directory / name).write_text((PROVIDERS / name).read_text().replace('#include <hip/hip_runtime.h>', ''))
    # Preserve actual layer layouts and complete validators. Only device
    # producers are replaced by a deferred queue with identifiable data.
    for kind, marker, signature in (
        ('linear', '// Complete private target layer,',
         'template<class Element> hipError_t launch_linear_layer(const LinearLayerViews<Element>&, const LinearLayerTables&, unsigned, hipStream_t);'),
        ('attention', '// All results remain private,',
         'hipError_t launch_attention_layer(const AttentionLayerViews&, const AttentionLayerTables&, unsigned, hipStream_t);')):
        actual = (PROVIDERS / f'sm121_q2_{kind}_layer.h').read_text().split(marker)[0]
        actual = '\n'.join(line for line in actual.splitlines() if not line.startswith('#include'))
        (directory / f'sm121_q2_{kind}_layer.h').write_text(
            '#include "sm121_q2_' + kind + '_block_layout.h"\n#include "target_test_moe.h"\n'
            + actual + '\n' + signature + '\n}\n')
    moe = (PROVIDERS / 'sm121_mtp_moe.h').read_text()
    declarations = moe[moe.index('struct MoeWeights {'):moe.index('namespace mtp_moe_detail {')]
    (directory / 'target_test_moe.h').write_text('#pragma once\n#include "sm121_mtp_moe_layout.h"\n'
        'namespace qrt_sm121_mtp {\n' + declarations
        + 'hipError_t launch_residual_normalize(const uint16_t*,const uint16_t*,const uint16_t*,const unsigned char*,unsigned,uint16_t*,uint16_t*,hipStream_t);\n}\n')
    (directory / 'sm121_q2_head.h').write_text('#pragma once\n'
        'namespace qrt_sm121_mtp {constexpr unsigned head_vocabulary=248320u;}\n'
        'namespace qrt_sm121_q2 {hipError_t launch_target_head(const uint16_t*,const uint16_t*,uint16_t*,uint32_t*,float*,uint32_t*,unsigned,hipStream_t);}\n')
