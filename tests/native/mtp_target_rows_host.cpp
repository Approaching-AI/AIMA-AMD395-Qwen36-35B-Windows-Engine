#include "mtp_target_rows.h"
#include "mtp_target_rows_trace.h"
#include <cassert>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <thread>

using namespace qrt_mtp_target_rows;

static float from_bits(uint32_t bits) {
    float value;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

static void complete(PrefillRows &batch, const std::vector<float> &values, uint32_t token) {
    assert(batch.stage(batch.local_rows(), values, token));
    assert(batch.staged() && !batch.published());
    assert(batch.hidden().empty() && batch.shifted_tokens().empty());
    assert(batch.sampled_token() == UINT32_MAX);
    assert(!batch.publish(token + 1u));
    assert(batch.publish(token));
    assert(!batch.publish(token));
    assert(!batch.stage(batch.local_rows(), values, token));
    assert(batch.sampled_token() == token);
}

static void identities_and_chunk_tails() {
    uint32_t prompt[] = {10, 11, 12, 13, 14};
    std::vector<float> values(3u * hidden_width, 1.0f);
    PrefillRows first(prompt, 5, 0, 3);
    assert(first.valid() && first.matches_input(prompt, 3));
    assert(!first.matches_input(prompt + 1, 3) && !first.matches_input(prompt, 2));
    assert(!first.matches_input(nullptr, 3));
    prompt[4] = 99; // The owner retained the complete request identity.
    complete(first, values, 700);
    assert(first.discarded_prefill() && first.first_position() == 0 && first.prompt_tokens() == 5);
    assert((first.shifted_tokens() == std::vector<uint32_t>{11, 12, 14}));
    assert(first.hidden().size() == 3u * hidden_width && first.hidden().back() == 0x3f80);
    prompt[4] = 14;
    PrefillRows last(prompt, 5, 3, 2);
    assert(last.matches_input(prompt + 3, 2) && !last.discarded_prefill());
    values.resize(2u * hidden_width);
    complete(last, values, 701);
    assert((last.shifted_tokens() == std::vector<uint32_t>{14, 701}));
    PrefillRows one(prompt, 5, 4, 1);
    values.resize(hidden_width);
    complete(one, values, 702);
    assert(one.shifted_tokens() == std::vector<uint32_t>{702});
}

static void request_bounds() {
    uint32_t prompt[] = {10, 11, 12, 13, 14};
    const auto bad = [&](const uint32_t *p, size_t count, size_t first, size_t rows) {
        PrefillRows batch(p, count, first, rows);
        assert(!batch.valid() && !batch.published() && !batch.staged());
        assert(batch.hidden().empty() && batch.shifted_tokens().empty());
    };
    bad(nullptr, 5, 0, 3); bad(prompt, 0, 0, 1); bad(prompt, 262145, 0, 1);
    bad(prompt, 5, 0, 0); bad(prompt, 5, 0, 8193); bad(prompt, 5, 5, 1);
    bad(prompt, 5, 4, 2); bad(prompt, 5, std::numeric_limits<size_t>::max(), 3);
    for (size_t i : {size_t(0), size_t(2), size_t(4)}) {
        const uint32_t save = prompt[i]; prompt[i] = vocabulary;
        bad(prompt, 5, 0, 3); prompt[i] = save;
    }
    // Exercise the real maximum bounded allocation and last prompt chunk.
    std::vector<uint32_t> large(262144, 23);
    PrefillRows batch(large.data(), large.size(), 253952, 8192);
    std::vector<float> values(8192u * hidden_width, -2.0f);
    complete(batch, values, 17);
    assert(batch.hidden().size() == 8192u * hidden_width);
    assert(batch.hidden().front() == 0xc000 && batch.hidden().back() == 0xc000);
    assert(batch.shifted_tokens().front() == 23 && batch.shifted_tokens().back() == 17);
}

static void row_and_number_failures() {
    uint32_t prompt[] = {10, 11};
    std::vector<float> values(2u * hidden_width, 1.0f);
    PrefillRows batch(prompt, 2, 0, 2);
    for (const auto &positions : {std::vector<unsigned>{}, {1}, {0, 0}, {1, 0}, {0, 2}})
        assert(!batch.stage(positions, values, 3));
    assert(!batch.stage(batch.local_rows(), values, vocabulary));
    values.pop_back(); assert(!batch.stage(batch.local_rows(), values, 3));
    values.push_back(1.0f);
    for (uint32_t bits : {0x7f800000u, 0xff800000u, 0x7fc00001u, 0x7f7fffffu, 0xff7fffffu}) {
        values.back() = from_bits(bits);
        assert(!batch.stage(batch.local_rows(), values, 3));
        assert(!batch.staged() && !batch.published() && batch.hidden().empty());
    }
    values.back() = 1.0f;
    const uint32_t input[] = {0x3f808000u, 0x3f818000u, 0xbf808000u, 0xbf818000u,
        0u, 0x80000000u, 0x00008000u, 0x00018000u, 0x7f7f0000u};
    const uint16_t expected[] = {0x3f80, 0x3f82, 0xbf80, 0xbf82, 0, 0x8000, 0, 2, 0x7f7f};
    for (size_t i = 0; i < sizeof(input) / sizeof(*input); ++i) values[i] = from_bits(input[i]);
    complete(batch, values, 3);
    assert(std::equal(std::begin(expected), std::end(expected), batch.hidden().begin()));
}

static void failed_provider_publication() {
    const uint32_t prompt[] = {10, 11};
    const std::vector<float> values(2u * hidden_width, 1.0f);
    for (bool publish_before_failure : {false, true}) {
        PrefillRows batch(prompt, 2, 0, 2);
        {
            Publication guard(&batch);
            assert(batch.stage(batch.local_rows(), values, 3));
            if (publish_before_failure) assert(batch.publish(3));
            // A later provider check or trace write failed: no complete().
        }
        assert(!batch.valid() && !batch.published() && !batch.staged());
        assert(batch.hidden().empty() && batch.shifted_tokens().empty());
        assert(!batch.publish(3));
    }
    PrefillRows exception(prompt, 2, 0, 2);
    try {
        Publication guard(&exception);
        complete(exception, values, 3);
        throw std::runtime_error("later provider failure");
    } catch (const std::runtime_error &) {}
    assert(!exception.published() && exception.hidden().empty());
    PrefillRows accepted(prompt, 2, 0, 2);
    {
        Publication guard(&accepted);
        complete(accepted, values, 3);
        guard.complete();
    }
    assert(accepted.published() && accepted.hidden().size() == values.size());
}

static void scopes_and_head_selection() {
    const uint32_t prompt[] = {10, 11, 12};
    PrefillRows outer(prompt, 3, 0, 3), inner(prompt, 3, 2, 1);
    assert(!Scope::active && !Scope::requested(3));
    {
        Scope scope(&outer);
        assert(Scope::requested(3) && !Scope::requested(2));
        std::thread other([] { assert(!Scope::active && !Scope::requested(3)); });
        other.join();
        try {
            Scope nested(&inner);
            assert(Scope::requested(1));
            { Scope suspended(nullptr); assert(!Scope::active); }
            assert(Scope::active == &inner);
            throw 3;
        } catch (int) {}
        assert(Scope::active == &outer);
    }
    assert(!Scope::active);
    std::vector<float> values(3u * hidden_width);
    for (size_t row = 0; row < 3; ++row)
        std::fill_n(values.data() + row * hidden_width, hidden_width, float(row + 1));
    std::vector<float> selected;
    assert(select_head_rows(outer.local_rows(), values, {2}, &selected));
    assert(selected.size() == hidden_width && selected.front() == 3 && selected.back() == 3);
    assert(select_head_rows(outer.local_rows(), values, {2, 0, 2}, &selected));
    assert(selected[0] == 3 && selected[hidden_width] == 1 && selected[2u * hidden_width] == 3);
    const auto previous = selected;
    assert(!select_head_rows(outer.local_rows(), values, {3}, &selected) && selected == previous);
    assert(!select_head_rows({1, 0, 2}, values, {2}, &selected) && selected == previous);
    assert(!select_head_rows(outer.local_rows(), values, {}, &selected));
    // Output aliasing must not invalidate source iterators during selection.
    assert(select_head_rows(outer.local_rows(), values, {2, 0}, &values));
    assert(values.size() == 2u * hidden_width && values.front() == 3 && values.back() == 1);
}

int main(int argc, char **argv) {
    assert(argc == 2);
    identities_and_chunk_tails();
    request_bounds();
    row_and_number_failures();
    failed_provider_publication();
    scopes_and_head_selection();
    const uint32_t prompt[] = {10, 11};
    PrefillRows traced(prompt, 2, 0, 2);
    const std::string prefix = std::string(argv[1]) + "/actual";
    std::string failure;
    assert(!write_trace(traced, prefix, &failure));
    std::vector<float> values(2u * hidden_width, -2.0f);
    complete(traced, values, 701);
    assert(!write_trace(traced, prefix, nullptr));
    assert(!write_trace(traced, std::string(argv[1]) + "/missing/actual", &failure));
    assert(write_trace(traced, prefix, &failure));
    assert(!write_trace(traced, prefix, &failure));
    // Detect an existing metadata file before touching either binary output.
    const std::string occupied = std::string(argv[1]) + "/occupied";
    { std::ofstream file(occupied + ".json"); file << "original"; }
    assert(!write_trace(traced, occupied, &failure));
    assert(!std::filesystem::exists(occupied + ".hidden.bf16.bin"));
}
