#include "../../native/providers/ck_fmha/blackwell_attention.h"
#include "../../native/providers/moe_accumulator/sm121_crt_integer_core.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <thread>
#include <vector>

namespace {
namespace core = qrt_sm121_crt_integer;
namespace original = qrt_blackwell_attention;
constexpr unsigned guard = 64u, tiles = 8192u;
struct Row { int high[4]{}, low[4]{}; };
struct Cell { int integer[4],residues[4];float approximate[2];int accepted[2];int64_t recovered[2]; };
void check(hipError_t status) { if (status != hipSuccess) throw std::runtime_error(hipGetErrorString(status)); }
void finish() {
    hipEvent_t event; check(hipEventCreate(&event)); check(hipEventRecord(event));
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    for (;;) {
        const auto status = hipEventQuery(event);
        if (status == hipSuccess) break;
        if (status != hipErrorNotReady) check(status);
        if (std::chrono::steady_clock::now() >= deadline) throw std::runtime_error("CRT integer core completion deadline");
        std::this_thread::yield();
    }
    check(hipEventDestroy(event));
}
template<class T> struct Guarded {
    T* base = nullptr; size_t size;
    explicit Guarded(size_t n) : size(n) {
        check(hipMalloc(reinterpret_cast<void**>(&base), (n + 2u * guard) * sizeof(T)));
        reset();
    }
    ~Guarded() { if (base && hipFree(base) != hipSuccess) std::abort(); }
    T* data() { return base + guard; }
    void reset() { check(hipMemset(base, 0xa5, (size + 2u * guard) * sizeof(T))); }
    void verify_guards() {
        std::vector<unsigned char> low(guard * sizeof(T)), high(low.size());
        check(hipMemcpy(low.data(), base, low.size(), hipMemcpyDeviceToHost));
        check(hipMemcpy(high.data(), data() + size, high.size(), hipMemcpyDeviceToHost));
        if (std::any_of(low.begin(), low.end(), [](unsigned char x) { return x != 0xa5u; }) ||
            std::any_of(high.begin(), high.end(), [](unsigned char x) { return x != 0xa5u; }))
            throw std::runtime_error("CRT integer core memory guard changed");
    }
    std::vector<T> read() {
        std::vector<T> result(size); check(hipMemcpy(result.data(), data(), size * sizeof(T), hipMemcpyDeviceToHost)); return result;
    }
    void unchanged(const std::vector<T>& expected) {
        if (expected.size() != size) throw std::runtime_error("input length changed");
        const auto actual = read();
        if (std::memcmp(actual.data(), expected.data(), size * sizeof(T))) throw std::runtime_error("CRT integer core input changed");
        verify_guards();
    }
};
__global__ void prepare_rows(const Row* input, core::Prepared* output, unsigned count) {
    const unsigned row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < count) output[row] = core::prepare(input[row]);
}
__global__ void compare(const Row* rows, const core::Prepared* prepared, Cell* output) {
    const unsigned lane = threadIdx.x, base = blockIdx.x * 32u;
    const auto integer = original::blackwell_integer_prepared_products(rows[base + lane % 16u], rows[base + 16u + lane % 16u]);
    const auto a=core::prepare(rows[base+lane%16u]),b=core::prepare(rows[base+16u+lane%16u]);
    const auto inline_parts=core::products(a,b);
    const auto pa=prepared[base+lane%16u],pb=prepared[base+16u+lane%16u];
    const auto prepared_parts=core::products(pa,pb);
#pragma unroll
    for (unsigned i = 0u; i < 8u; ++i) {
        Cell cell{};
#pragma unroll
        for (unsigned part = 0u; part < 4u; ++part) cell.integer[part] = integer.value[part][i];
        cell.approximate[0]=inline_parts.approximate[i];cell.approximate[1]=prepared_parts.approximate[i];
        cell.residues[0]=inline_parts.residue255[i];cell.residues[1]=inline_parts.residue256[i];
        cell.residues[2]=prepared_parts.residue255[i];cell.residues[3]=prepared_parts.residue256[i];
        cell.recovered[0]=cell.recovered[1]=INT64_MIN;
        cell.accepted[0]=core::recover(cell.approximate[0],uint32_t(cell.residues[0]),uint32_t(cell.residues[1]),&cell.recovered[0]);
        cell.accepted[1]=core::recover(cell.approximate[1],uint32_t(cell.residues[2]),uint32_t(cell.residues[3]),&cell.recovered[1]);
        output[blockIdx.x * 256u + (2u * i + lane / 16u) * 16u + lane % 16u] = cell;
    }
}
template<unsigned Variant>
__global__ void sequence(const Row* rows, const core::Prepared* prepared, uint64_t* output, unsigned groups) {
    const unsigned lane = threadIdx.x;
    uint64_t totals[8]{};
    for (unsigned group = 0u; group < groups; ++group) {
        const unsigned base = ((blockIdx.x * groups + group) % tiles) * 32u;
        if constexpr (Variant == 0u) {
            const auto parts = original::blackwell_integer_prepared_products(rows[base + lane % 16u], rows[base + 16u + lane % 16u]);
#pragma unroll
            for (unsigned i = 0u; i < 8u; ++i)
                totals[i] += uint64_t(int64_t(parts.value[0][i]) * 65536 +
                    (int64_t(parts.value[1][i]) + parts.value[2][i]) * 256 + parts.value[3][i]);
        } else {
            core::Prepared a,b;
            if constexpr(Variant==1u){a=core::prepare(rows[base+lane%16u]);b=core::prepare(rows[base+16u+lane%16u]);}
            else{a=prepared[base+lane%16u];b=prepared[base+16u+lane%16u];}
            const auto parts=core::products(a,b);
#pragma unroll
            for(unsigned i=0u;i<8u;++i){
                int64_t recovered=INT64_MIN;
                core::recover(parts.approximate[i],uint32_t(parts.residue255[i]),uint32_t(parts.residue256[i]),&recovered);
                totals[i]+=uint64_t(recovered);
            }
        }
    }
#pragma unroll
    for (unsigned i = 0u; i < 8u; ++i)
        output[blockIdx.x * 256u + (2u * i + lane / 16u) * 16u + lane % 16u] = totals[i];
}
int decoded(unsigned value) { return value & 32768u ? int(value) - 65536 : int(value); }
float decode_half(uint16_t value) {
    const unsigned exponent = (value >> 10u) & 31u;
    return exponent ? ((value & 32768u) ? -1.0f : 1.0f) * std::ldexp(float(1024u + (value & 1023u)), int(exponent) - 25) : 0.0f;
}
double elapsed(std::chrono::steady_clock::time_point begin) { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count(); }
void run() {
    uint32_t state = 0x3958192u;
    auto random = [&]() { state ^= state << 13u; state ^= state >> 17u; state ^= state << 5u; return state; };
    std::vector<uint16_t> values(tiles * 512u);
    std::vector<Row> rows(tiles * 32u);
    std::vector<core::Prepared> expected_prepared(rows.size());
    std::vector<bool> encodings(65536u,false);
    std::vector<uint16_t> domain;
    for(unsigned x=0u;x<65536u;++x){
        const int value=decoded(x);bool eligible=false;
        for(unsigned shift=0u;shift<8u;++shift)for(unsigned m=0u;m<256u;++m)
            eligible |= value==int(m<<shift) || value==-int(m<<shift);
        if(eligible)domain.push_back(uint16_t(x));
    }
    if(domain.size()!=2303u)throw std::runtime_error("independent H7 domain count changed");
    const int extremes[]={0,1,-1,127,-127,128,-128,255,-255,256,-256,32640,-32640};
    for(unsigned i=0u;i<values.size();++i){
        const unsigned mode=i/512u%8u;
        const uint16_t value=i<domain.size()?domain[i]:mode==0u?uint16_t(extremes[random()%13u]):
            mode==1u?uint16_t(i&1u?32640:-32640):mode==2u?uint16_t(i%16u?0u:domain[random()%domain.size()]):
            mode==3u?uint16_t(int((128u+(random()&127u))<<(random()%8u))*(random()&1u?1:-1)):
            mode==4u?uint16_t(-32640):mode==5u?uint16_t(i%512u<256u?32640:-32640):domain[random()%domain.size()];
        values[i]=value;encodings[value]=true;
        const unsigned row=i/16u,word=i%16u/4u,shift=i%4u*8u;
        rows[row].high[word]=int(uint32_t(rows[row].high[word])|uint32_t(value>>8u)<<shift);
        rows[row].low[word]=int(uint32_t(rows[row].low[word])|uint32_t(value&255u)<<shift);
    }
    for(uint16_t value:domain)if(!encodings[value])throw std::runtime_error("missing H7 encoding");
    for(unsigned row=0u;row<rows.size();++row){
        expected_prepared[row]=core::prepare(rows[row]);
        for(unsigned i=0u;i<16u;++i){
            const int value=decoded(values[row*16u+i]);const auto& prepared=expected_prepared[row];
            int residue=value%255;if(residue<0)residue+=255;
            if(decode_half(prepared.values[i])!=float(value) ||
                ((prepared.residue255[i/4u]>>(i%4u*8u))&255u)!=unsigned(residue) ||
                ((prepared.residue256[i/4u]>>(i%4u*8u))&255u)!=(uint32_t(value)&255u))
                throw std::runtime_error("prepared H7 encoding changed");
        }
    }
    Guarded<Row> input(rows.size()); Guarded<core::Prepared> prepared(rows.size()); Guarded<Cell> output(tiles * 256u);
    check(hipMemcpy(input.data(), rows.data(), rows.size() * sizeof(Row), hipMemcpyHostToDevice)); finish();
    auto begin = std::chrono::steady_clock::now();
    hipLaunchKernelGGL(prepare_rows, dim3((rows.size() + 255u) / 256u), dim3(256u), 0u, nullptr, input.data(), prepared.data(), unsigned(rows.size()));
    check(hipGetLastError()); finish(); const double encoding_ms = elapsed(begin);
    prepared.unchanged(expected_prepared);
    hipLaunchKernelGGL(compare, dim3(tiles), dim3(32u), 0u, nullptr, input.data(), prepared.data(), output.data());
    check(hipGetLastError()); finish(); const auto actual = output.read();
    std::vector<int64_t> mathematical(actual.size());
    size_t integer_bad=0u,residue_bad[2]{},raw_differences[2]{},recovery_bad[2]{},reconstruction_bad[2]{};
    double maximum_error[2]{};
    for(unsigned cell=0u;cell<actual.size();++cell){
        const unsigned tile=cell/256u,row=cell%256u/16u,column=cell%16u;
        int integer[4]{},residues[2]{};int64_t total=0;
        for(unsigned k=0u;k<16u;++k){
            const unsigned a=values[tile*512u+row*16u+k],b=values[tile*512u+256u+column*16u+k];
            const int ah=int(a>>8u)-((a&32768u)?256:0),al=int(a&255u);
            const int bh=int(b>>8u)-((b&32768u)?256:0),bl=int(b&255u);
            integer[0]+=ah*bh;integer[1]+=ah*bl;integer[2]+=al*bh;integer[3]+=al*bl;
            int am=decoded(a)%255,bm=decoded(b)%255;if(am<0)am+=255;if(bm<0)bm+=255;
            residues[0]+=am*bm;residues[1]+=al*bl;total+=int64_t(decoded(a))*decoded(b);
        }
        mathematical[cell]=total;
        for(unsigned part=0u;part<4u;++part)integer_bad+=unsigned(actual[cell].integer[part]!=integer[part]);
        for(unsigned variant=0u;variant<2u;++variant){
            for(unsigned part=0u;part<2u;++part)residue_bad[variant]+=unsigned(actual[cell].residues[variant*2u+part]!=residues[part]);
            const float approximate=actual[cell].approximate[variant];
            const double error=std::abs(double(approximate)-double(total));
            raw_differences[variant]+=unsigned(error!=0.0);
            if(!std::isfinite(approximate) || error>3920.0)throw std::runtime_error("H7 conditional native error bound failed");
            maximum_error[variant]=std::max(maximum_error[variant],error);
            recovery_bad[variant]+=unsigned(!actual[cell].accepted[variant]);
            reconstruction_bad[variant]+=unsigned(actual[cell].recovered[variant]!=total);
        }
    }
    input.unchanged(rows);prepared.unchanged(expected_prepared);output.verify_guards();
    for(unsigned variant=0u;variant<2u;++variant)
        std::printf("{\"kind\":\"crt_integer_core\",\"variant\":%u,\"tiles\":%u,\"cells\":%zu,\"core_encodings\":2303,\"original_integer_partial_mismatches\":%zu,\"residue_partial_mismatches\":%zu,\"native_approximate_differences\":%zu,\"recovery_failures\":%zu,\"reconstruction_mismatches\":%zu,\"maximum_native_absolute_error\":%.12g,\"conditional_error_bound\":3920,\"crt_modulus\":65280,\"prepared_encoding_ms\":%.6f,\"prepared_bytes\":%zu,\"redzones_pass\":true,\"immutable_inputs\":true,\"hardware_premise_proven\":false,\"inference_acceptance\":false}\n",variant+1u,tiles,actual.size(),integer_bad,residue_bad[variant],raw_differences[variant],recovery_bad[variant],reconstruction_bad[variant],maximum_error[variant],encoding_ms,expected_prepared.size()*sizeof(core::Prepared));
    std::fflush(stdout);
    if(integer_bad || residue_bad[0] || residue_bad[1] || recovery_bad[0] || recovery_bad[1] || reconstruction_bad[0] || reconstruction_bad[1])throw std::runtime_error("CRT core differs from independent integer oracle");
    constexpr unsigned blocks = 1024u;
    Guarded<uint64_t> sequence_output(blocks * 256u);
    for (unsigned groups : {16u, 64u, 256u}) {
        std::vector<uint64_t> expected(blocks * 256u);
        for (unsigned block = 0u; block < blocks; ++block) for (unsigned cell = 0u; cell < 256u; ++cell)
            for (unsigned group = 0u; group < groups; ++group)
                expected[block * 256u + cell] += uint64_t(mathematical[((block * groups + group) % tiles) * 256u + cell]);
        double timings[3][3]{};
        for (unsigned attempt = 0u; attempt < 4u; ++attempt) for (unsigned position = 0u; position < 3u; ++position) {
            const unsigned variant = (attempt + position) % 3u;
            sequence_output.reset(); finish(); begin = std::chrono::steady_clock::now();
#define QRT_CORE_SEQUENCE(V) case V: hipLaunchKernelGGL((sequence<V>), dim3(blocks), dim3(32u), 0u, nullptr, input.data(), prepared.data(), sequence_output.data(), groups); break
            switch (variant) { QRT_CORE_SEQUENCE(0u); QRT_CORE_SEQUENCE(1u); QRT_CORE_SEQUENCE(2u); }
#undef QRT_CORE_SEQUENCE
            check(hipGetLastError()); finish(); const double ms = elapsed(begin);
            if (attempt) timings[variant][attempt - 1u] = ms;
            sequence_output.unchanged(expected);
        }
        input.unchanged(rows); prepared.unchanged(expected_prepared);
        for (unsigned variant = 0u; variant < 3u; ++variant) {
            double ordered[] = {timings[variant][0], timings[variant][1], timings[variant][2]}; std::sort(ordered, ordered + 3u);
            std::printf("{\"kind\":\"crt_integer_sequence\",\"variant\":%u,\"groups\":%u,\"blocks\":%u,\"median_ms\":%.6f,\"samples_ms\":[%.6f,%.6f,%.6f],\"prepared_encoding_ms_excluded\":%.6f,\"cpu_integer_sequence_output_cells\":%u,\"all_attempts_verified\":true,\"mismatches\":0,\"redzones_pass\":true,\"immutable_inputs\":true,\"canonical_carry_and_exceptions_included\":false,\"performance_acceptance\":false,\"inference_acceptance\":false}\n", variant, groups, blocks, ordered[1], timings[variant][0], timings[variant][1], timings[variant][2], variant == 2u ? encoding_ms : 0.0, blocks * 256u * 4u);
        }
        std::fflush(stdout);
    }
}
}
int main() try {
    hipDeviceProp_t device{}; check(hipGetDeviceProperties(&device, 0));
    if (std::strncmp(device.gcnArchName, "gfx1151", 7u)) throw std::runtime_error("requires gfx1151");
    run(); return 0;
} catch (const std::exception& error) { std::fprintf(stderr, "%s\n", error.what()); return 1; }
