"""Compose independent observers; model outputs remain the original tensors."""
from capture_explicit_dense_layers import OrderedDenseCapture
from capture_explicit_routed_layers import OrderedRoutedCapture
from capture_explicit_attention_layers import OrderedAttentionCapture

class ExplicitQ8192Capture(OrderedDenseCapture, OrderedRoutedCapture, OrderedAttentionCapture):
    pass
