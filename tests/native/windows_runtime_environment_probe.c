/* Link this object with windows_runtime_environment_probe.rs. The probe
 * calls the production Rust setter and the actual existing C-core reader. */
#include "../../native/src/qrt.c"
#include "../../native/src/qrt_server_bridge.c"

int probe_native_early_prefill_enabled(void) {
    return qrt_qwen36_whole_provider_early_prefill_stream_enabled();
}

const char *probe_get_crt_early_prefill(void) {
    return getenv("QRT_QWEN36_WHOLE_PROVIDER_EARLY_PREFILL_STREAM_CALLBACK");
}

int probe_crt_wide_matches(const uint16_t *name, const uint16_t *expected) {
    const wchar_t *actual = _wgetenv((const wchar_t *)name);
    size_t index;
    if (expected == NULL) return actual == NULL;
    if (actual == NULL) return 0;
    for (index = 0u; actual[index] != 0 || expected[index] != 0; ++index) {
        if ((uint16_t)actual[index] != expected[index]) return 0;
    }
    return 1;
}

int probe_invalid_native_arguments(void) {
    const uint16_t good[] = {'Q', 'R', 'T', '_', 'T', 'E', 'S', 'T', 0};
    const uint16_t bad[] = {'Q', '=', 'T', 0};
    const uint16_t empty[] = {0};
    const uint16_t one[] = {'1', 0};
    return qrt_server_set_environment_utf16_v1(NULL, one) == QRT_STATUS_INVALID_ARGUMENT &&
        qrt_server_set_environment_utf16_v1(good, NULL) == QRT_STATUS_INVALID_ARGUMENT &&
        qrt_server_set_environment_utf16_v1(empty, one) == QRT_STATUS_INVALID_ARGUMENT &&
        qrt_server_set_environment_utf16_v1(bad, one) == QRT_STATUS_INVALID_ARGUMENT;
}
