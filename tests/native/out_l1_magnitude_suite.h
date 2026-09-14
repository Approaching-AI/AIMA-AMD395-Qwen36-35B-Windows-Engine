#pragma once

namespace projection_safety_test {
// Independent exact dyadic sums span every output of the real OUT dimensions.
// The producer and magnitude views are the ones used by the read-only audit.
unsigned run_out_l1_magnitude_suite() {
#if defined(QRT_ENABLE_HIPBLASLT_RESIDENT_MATRIX_PROVIDER)
    namespace ap = absolute_product_bound_test;
    constexpr unsigned rows = 2048u, tokens = 8192u, k = 4096u, cells = rows * tokens;
    std::vector<uint16_t> weights(size_t(rows) * k + 2u * ap::guard, ap::operand_guard);
    std::vector<uint16_t> inputs(size_t(tokens) * k + 2u * ap::guard, ap::operand_guard);
    std::vector<unsigned> wf(rows + 2u * ap::guard, ap::flag_guard), xf(tokens + 2u * ap::guard, ap::flag_guard);
    for (unsigned r = 0u; r < rows; ++r) for (unsigned j = 0u; j < k; ++j)
        weights[ap::guard + size_t(r) * k + j] = bf16(weight_value(r, j));
    for (unsigned t = 0u; t < tokens; ++t) for (unsigned j = 0u; j < k; ++j)
        inputs[ap::guard + size_t(t) * k + j] = bf16(input_value(t, j));
    double reference[7][11]{};
    for (unsigned t = 0u; t < 7u; ++t) for (unsigned r = 0u; r < 11u; ++r) {
        int64_t numerator = 0;
        for (unsigned j = 0u; j < k; ++j)
            numerator += std::abs(int64_t(int((r * 7u + j * 3u) % 11u) - 5) *
                (int((t * 3u + j) % 7u) - 3));
        reference[t][r] = double(numerator) / 128.0;
        double independent = 0.0;
        for (unsigned j = 0u; j < k; ++j)
            independent += std::abs(ap::bf16(weights[ap::guard + size_t(r) * k + j]) *
                ap::bf16(inputs[ap::guard + size_t(t) * k + j]));
        require(independent == reference[t][r], "independent L1 dyadic classes");
    }
    ap::Buffer<uint16_t> dw(weights), dx(inputs);
    ap::Buffer<unsigned> dwf(wf), dxf(xf);
    auto magnitudes_w = weights, magnitudes_x = inputs;
    std::fill(magnitudes_w.begin(), magnitudes_w.end(), ap::operand_guard);
    std::fill(magnitudes_x.begin(), magnitudes_x.end(), ap::operand_guard);
    ap::Buffer<uint16_t> mw(magnitudes_w), mx(magnitudes_x);
    std::vector<float> output(cells + 2u * ap::guard, ap::output_guard);
    ap::Buffer<float> out(output);
    hipLaunchKernelGGL(qrt_sm121_scalar_projection::eligible_rows_kernel, dim3(rows), dim3(256u),
        0u, nullptr, dw.data(), dwf.data(), rows, k);
    ap::check(hipGetLastError());
    hipLaunchKernelGGL(qrt_sm121_scalar_projection::eligible_rows_kernel, dim3(tokens), dim3(256u),
        0u, nullptr, dx.data(), dxf.data(), tokens, k);
    ap::check(hipGetLastError());
    hipLaunchKernelGGL(qrt_bf16_absolute_product_views::prepare_rows_kernel, dim3(rows), dim3(256u),
        0u, nullptr, dw.data(), dwf.data(), mw.data(), rows, k);
    ap::check(hipGetLastError());
    hipLaunchKernelGGL(qrt_bf16_absolute_product_views::prepare_rows_kernel, dim3(tokens), dim3(256u),
        0u, nullptr, dx.data(), dxf.data(), mx.data(), tokens, k);
    ap::check(hipGetLastError()); ap::complete();
    for (unsigned which = 0u; which < 2u; ++which) {
        auto& flags = which ? xf : wf;
        auto& magnitudes = which ? magnitudes_x : magnitudes_w;
        const auto& original = which ? inputs : weights;
        (which ? dxf : dwf).read(flags);
        (which ? mx : mw).read(magnitudes);
        for (unsigned i = 0u; i < flags.size(); ++i)
            require(flags[i] == (i >= ap::guard && i < flags.size() - ap::guard ? 1u : ap::flag_guard), "L1 eligibility and guards");
        for (size_t i = 0u; i < magnitudes.size(); ++i)
            require(magnitudes[i] == (i >= ap::guard && i < magnitudes.size() - ap::guard
                ? uint16_t(original[i] & 0x7fffu) : ap::operand_guard), "L1 magnitudes and guards");
    }
    std::string stage, failure;
    require(resident_bf16_matrix_matmul_f32_output_with_heuristic_index(mw.data(), mx.data(), out.data(),
        rows, k, tokens, 4u, nullptr, "out_l1_full_shape", &stage, &failure), failure.c_str());
    ap::complete();
    out.read(output);
    unsigned underestimates = 0u;
    double maximum_ratio = 0.0;
    for (unsigned index = 0u; index < cells; ++index) {
        const float raw = output[ap::guard + index];
        require(std::isfinite(raw) && raw >= 0.0f, "invalid or unwritten full OUT magnitude");
        const double expected = reference[(index / rows) % 7u][(index % rows) % 11u];
        const float upper = qrt_bf16_positive_sum_bound::finish(raw, k);
        underestimates += double(upper) < expected;
        maximum_ratio = (std::max)(maximum_ratio, double(upper) / expected);
    }
    for (unsigned j = 0u; j < ap::guard; ++j)
        require(output[j] == ap::output_guard && output[ap::guard + cells + j] == ap::output_guard, "OUT magnitude matrix guards");
    dw.unchanged(weights); dx.unchanged(inputs); mw.unchanged(magnitudes_w); mx.unchanged(magnitudes_x);
    dwf.unchanged(wf); dxf.unchanged(xf);
    require(!underestimates, "OUT magnitude underestimated independent sum");
    std::printf("{\"type\":\"out_l1_magnitude\",\"rows\":%u,\"tokens\":%u,\"k\":%u,\"algorithm\":4,\"cells\":%u,\"independent_cpu_classes\":77,\"underestimates\":%u,\"maximum_ratio\":%.9g,\"flags_checked\":%u,\"redzones_pass\":true,\"immutable_inputs\":true,\"inference_acceptance\":false,\"performance_acceptance\":false}\n",
        rows, tokens, k, cells, underestimates, maximum_ratio, rows + tokens);
    return 1u;
#else
    throw std::runtime_error("OUT magnitude requires hipBLASLt");
#endif
}
} // namespace projection_safety_test
