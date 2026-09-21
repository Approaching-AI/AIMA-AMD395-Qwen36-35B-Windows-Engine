// Compare the actual provider submission wrapper with the independent retained
// kernels, original captured boundaries, all intermediate tensors and guards.
#define QRT_GDN_HYBRID_RETRY
#define QRT_GDN_PROVIDER_REPLAY
#include "separate_state_gdn_selftest.cpp"
