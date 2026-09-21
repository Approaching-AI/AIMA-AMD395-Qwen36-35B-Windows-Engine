// Compare the actual provider submission wrapper with the independent retained
// kernels, original captured boundaries, all intermediate tensors and guards.
// Keep the implementation in this translation unit. Windows hipcc rewrites a
// .cpp argument after -include as a source file, breaking that command form.
#include "../../native/providers/gdn/blackwell_cooperative.cpp"
#define QRT_GDN_HYBRID_RETRY
#define QRT_GDN_PROVIDER_REPLAY
#include "separate_state_gdn_selftest.cpp"
