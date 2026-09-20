// Windows original-weight replay. Expected tensors stay on the host.
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <bcrypt.h>
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cstring>
#include <climits>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include "native/providers/moe_accumulator/sm121_packed_dense.h"
#include "native/providers/moe_accumulator/sm121_q1_moe.h"

void check(hipError_t s) { if (s != hipSuccess) throw std::runtime_error(hipGetErrorString(s)); }
std::string sha256(const void* data, size_t bytes) {
    if (bytes > ULONG_MAX) throw std::runtime_error("hash extent");
    BCRYPT_ALG_HANDLE algorithm = nullptr; BCRYPT_HASH_HANDLE hash = nullptr;
    DWORD length = 0, copied = 0; unsigned char digest[32];
    std::vector<unsigned char> object;
    const auto ok = [](NTSTATUS s) { if (s < 0) throw std::runtime_error("BCrypt SHA256"); };
    try {
        ok(BCryptOpenAlgorithmProvider(&algorithm, BCRYPT_SHA256_ALGORITHM, nullptr, 0));
        ok(BCryptGetProperty(algorithm, BCRYPT_OBJECT_LENGTH, reinterpret_cast<PUCHAR>(&length), sizeof(length), &copied, 0));
        object.resize(length);
        ok(BCryptCreateHash(algorithm, &hash, object.data(), length, nullptr, 0, 0));
        ok(BCryptHashData(hash, const_cast<PUCHAR>(static_cast<const unsigned char*>(data)), static_cast<ULONG>(bytes), 0));
        ok(BCryptFinishHash(hash, digest, sizeof(digest), 0));
        BCryptDestroyHash(hash); hash = nullptr;
        BCryptCloseAlgorithmProvider(algorithm, 0); algorithm = nullptr;
    } catch (...) {
        if (hash) BCryptDestroyHash(hash);
        if (algorithm) BCryptCloseAlgorithmProvider(algorithm, 0);
        throw;
    }
    std::ostringstream text;
    for (unsigned char c : digest) text << std::hex << std::setfill('0') << std::setw(2) << unsigned(c);
    return text.str();
}
std::vector<uint16_t> read(const std::string& path, uint64_t offset, size_t count,
                           const std::string& digest, bool entire = false) {
    const size_t bytes = count * sizeof(uint16_t);
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file || file.tellg() <= 0 || uint64_t(file.tellg()) > (uint64_t(8) << 30u) ||
        !count || bytes > (64u << 20u) || offset > uint64_t(file.tellg()) ||
        bytes > uint64_t(file.tellg()) - offset || (entire && (offset || bytes != uint64_t(file.tellg()))))
        throw std::runtime_error("input extent: " + path);
    std::vector<uint16_t> values(count); file.seekg(static_cast<std::streamoff>(offset));
    file.read(reinterpret_cast<char*>(values.data()), static_cast<std::streamsize>(bytes));
    if (!file || sha256(values.data(), bytes) != digest) throw std::runtime_error("input hash: " + path);
    return values;
}
uint64_t number(const std::string& s, uint64_t maximum) {
    if (s.empty() || s.find_first_not_of("0123456789") != std::string::npos) throw std::runtime_error("integer");
    size_t used = 0; const auto value = std::stoull(s, &used);
    if (used != s.size() || value > maximum) throw std::runtime_error("integer bound");
    return value;
}
struct Buffer {
    unsigned char* raw = nullptr; size_t bytes = 0;
    std::vector<unsigned char> original;
    explicit Buffer(size_t size) : bytes(size) {
        check(hipMalloc(reinterpret_cast<void**>(&raw), bytes + 512u)); reset();
    }
    Buffer(const Buffer&) = delete; Buffer& operator=(const Buffer&) = delete;
    ~Buffer() { if (raw && hipDeviceSynchronize() == hipSuccess) (void)hipFree(raw); }
    void* data() const { return raw + 256u; }
    void reset() { check(hipMemset(raw, 0xa5, bytes + 512u)); }
    template<class T> void upload(const std::vector<T>& values) {
        if (values.size() * sizeof(T) != bytes) throw std::runtime_error("upload extent");
        original.resize(bytes + 512u, 0xa5);
        std::memcpy(original.data() + 256u, values.data(), bytes);
        check(hipMemcpy(raw, original.data(), original.size(), hipMemcpyHostToDevice));
    }
    std::vector<unsigned char> copy() const {
        std::vector<unsigned char> host(bytes + 512u);
        check(hipMemcpy(host.data(), raw, host.size(), hipMemcpyDeviceToHost)); return host;
    }
    size_t changed() const {
        if (original.empty()) throw std::runtime_error("missing immutable input");
        const auto after = copy(); size_t bad = 0;
        for (size_t i = 0; i < after.size(); ++i) bad += after[i] != original[i];
        return bad;
    }
    size_t guards(const std::vector<unsigned char>& host, size_t used) const {
        size_t bad = 0; if (used > bytes) throw std::runtime_error("output extent");
        for (size_t i = 0; i < 256u; ++i) bad += host[i] != 0xa5u;
        for (size_t i = 256u + used; i < host.size(); ++i) bad += host[i] != 0xa5u;
        return bad;
    }
};
template<unsigned K, class Input, class Output>
void packed(const void* x, const void* w, void* y, unsigned n, unsigned queries, unsigned cap) {
    for (unsigned q = 0; q < queries; ++q) for (unsigned first = 0; first < n; first += cap) {
        const unsigned count = std::min(cap, n - first);
        constexpr unsigned outputs_per_block = 256u / qrt_sm121_packed_dense::lanes<K>;
        hipLaunchKernelGGL((qrt_sm121_packed_dense::projection<K,Input,Output>),
            dim3((count + outputs_per_block - 1u) / outputs_per_block), dim3(256u), 0, nullptr,
            static_cast<const Input*>(x) + size_t(q) * K,
            static_cast<const uint16_t*>(w) + size_t(first) * K,
            static_cast<Output*>(y) + size_t(q) * n + first, count, 1u);
        check(hipGetLastError());
    }
}
template<unsigned K> void launch(bool original_packed, bool input_float, bool output_float,
    const void* x, const void* w, void* y, unsigned n, unsigned queries, unsigned cap) {
    if (original_packed) {
        if (input_float && output_float) packed<K,float,float>(x,w,y,n,queries,cap);
        else if (input_float) packed<K,float,uint16_t>(x,w,y,n,queries,cap);
        else if (output_float) packed<K,uint16_t,float>(x,w,y,n,queries,cap);
        else packed<K,uint16_t,uint16_t>(x,w,y,n,queries,cap);
    } else {
        if (input_float || output_float) throw std::runtime_error("K16 carrier");
        for (unsigned q = 0; q < queries; ++q) for (unsigned first = 0; first < n; first += cap) {
            const unsigned count = std::min(cap, n - first);
            hipLaunchKernelGGL((qrt_sm121_q1_moe::projection<K>), dim3((count + 15u) / 16u), dim3(256u), 0, nullptr,
                static_cast<const uint16_t*>(x) + size_t(q) * K,
                static_cast<const uint16_t*>(w) + size_t(first) * K,
                static_cast<uint16_t*>(y) + size_t(q) * n + first, count);
            check(hipGetLastError());
        }
    }
}
int main(int argc, char** argv) try {
    if (argc != 2) throw std::runtime_error("bound tab-separated replay plan");
    std::ifstream plan(argv[1]); if (!plan) throw std::runtime_error("plan missing");
    std::string line; unsigned cases = 0; bool all_pass = true;
    while (std::getline(plan, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        std::istringstream parts(line); std::vector<std::string> f; std::string field;
        while (std::getline(parts, field, '\t')) f.push_back(field);
        if (f.size() != 20u || ++cases > 512u) throw std::runtime_error("plan fields or case bound");
        const auto ordinal = number(f[0], 511u); const auto& kind = f[1];
        const bool is_packed = number(f[2], 1u) != 0;
        const unsigned queries = unsigned(number(f[3], 2u)), k = unsigned(number(f[4], 4096u));
        const unsigned n = unsigned(number(f[5], 8192u)), stride = unsigned(number(f[10], 16384u));
        const unsigned offset = unsigned(number(f[11], 16384u));
        if (!queries || !n || (k != 512u && k != 2048u && k != 4096u) || offset + n > stride ||
            (kind != "projection" && kind != "shared_gate" && kind != "shared_activation") ||
            (kind == "shared_gate" && (k != 2048u || n != 1u)) ||
            (kind == "shared_activation" && (k != 2048u || n != 512u))) throw std::runtime_error("case shape");
        const auto input = read(f[6],0,size_t(queries)*k,f[7],true);
        const auto expected = read(f[8],0,size_t(queries)*stride,f[9],true);
        const auto weights = read(f[12],number(f[13],uint64_t(8)<<30u),size_t(n)*k,f[14]);
        Buffer x(input.size()*2u), w(weights.size()*2u), y(size_t(queries)*n*4u);
        x.upload(input); w.upload(weights);
        std::vector<float> widened(input.size());
        for (size_t i=0;i<input.size();++i) widened[i]=qrt_sm121_q1::widen(input[i]);
        Buffer xf(widened.size()*4u); xf.upload(widened);
        size_t mismatches=0, compared=0, guards=0, immutable=0, first_index=0;
        uint32_t first_actual=0, first_expected=0; unsigned configurations=0;
        const auto compare = [&](bool floats) {
            check(hipDeviceSynchronize()); const auto host=y.copy();
            guards += y.guards(host,size_t(queries)*n*(floats?4u:2u));
            for (unsigned q=0;q<queries;++q) for (unsigned col=0;col<n;++col) {
                const size_t i=size_t(q)*n+col; uint32_t actual=0;
                std::memcpy(&actual,host.data()+256u+i*(floats?4u:2u),floats?4u:2u);
                const uint32_t want=uint32_t(expected[size_t(q)*stride+offset+col]) << (floats?16u:0u);
                ++compared; if (actual!=want) {
                    if (!mismatches) { first_index=i;first_actual=actual;first_expected=want; }
                    ++mismatches;
                }
            }
            ++configurations;
        };
        if (kind == "projection") {
            for (unsigned carrier=0;carrier<(is_packed?4u:1u);++carrier) for (unsigned cap : {111u,8192u}) {
                y.reset();const bool fi=(carrier&1u)!=0,fo=(carrier&2u)!=0;
                const auto invoke = [&](auto width) {
                    launch<decltype(width)::value>(is_packed,fi,fo,fi?xf.data():x.data(),w.data(),y.data(),n,queries,cap);
                };
                if(k==512u)invoke(std::integral_constant<unsigned,512u>{});
                else if(k==2048u)invoke(std::integral_constant<unsigned,2048u>{});
                else invoke(std::integral_constant<unsigned,4096u>{});
                compare(fo);
            }
        } else if (kind == "shared_gate") {
            y.reset();
            for(unsigned q=0;q<queries;++q) {
                hipLaunchKernelGGL(qrt_sm121_q1_moe::shared_gate,dim3(1u),dim3(16u),0,nullptr,
                    static_cast<const uint16_t*>(x.data())+size_t(q)*k,static_cast<const uint16_t*>(w.data()),
                    static_cast<uint16_t*>(y.data())+q);check(hipGetLastError());
            }
            compare(false);
        } else {
            const auto up=read(f[15],number(f[16],uint64_t(8)<<30u),size_t(n)*k,f[17]);
            auto table=read(f[18],0,65536u+12u,f[19],true);
            if(std::memcmp(table.data(),"QRTSBF1\0",8u))throw std::runtime_error("SiLU table magic");
            table.erase(table.begin(),table.begin()+12u);
            Buffer u(up.size()*2u),s(table.size()*2u);u.upload(up);s.upload(table);y.reset();
            for(unsigned q=0;q<queries;++q) {
                const auto* in=static_cast<const uint16_t*>(x.data())+size_t(q)*k;
                auto* out=static_cast<uint16_t*>(y.data())+size_t(q)*n;
                if(is_packed) {
                    hipLaunchKernelGGL(qrt_sm121_packed_dense::shared_activation,dim3(32u),dim3(256u),0,nullptr,
                        in,static_cast<const uint16_t*>(w.data()),static_cast<const uint16_t*>(u.data()),static_cast<const uint16_t*>(s.data()),out);
                } else {
                    hipLaunchKernelGGL(qrt_sm121_q1_moe::shared_activation,dim3(32u),dim3(256u),0,nullptr,
                        in,static_cast<const uint16_t*>(w.data()),static_cast<const uint16_t*>(u.data()),static_cast<const uint16_t*>(s.data()),out);
                }
                check(hipGetLastError());
            }
            compare(false);immutable+=u.changed()+s.changed();
        }
        immutable+=x.changed()+xf.changed()+w.changed();
        const bool passed=!mismatches&&!guards&&!immutable;all_pass&=passed;
        std::cout << "{\"ordinal\":" << ordinal << ",\"kind\":\"" << kind << "\",\"packed\":" << (is_packed?"true":"false")
            << ",\"queries\":" << queries << ",\"input_width\":" << k << ",\"output_width\":" << n
            << ",\"configurations\":" << configurations << ",\"compared_elements\":" << compared
            << ",\"bit_mismatches\":" << mismatches << ",\"guard_errors\":" << guards << ",\"immutable_input_errors\":" << immutable
            << ",\"first_difference\":[" << first_index << ',' << first_actual << ',' << first_expected << ']'
            << ",\"passed\":" << (passed?"true":"false") << ",\"native_execution\":true,\"model_loaded\":false,\"inference_acceptance\":false}\n" << std::flush;
    }
    if(!cases)throw std::runtime_error("empty plan");
    return all_pass?0:1;
} catch(const std::exception& e) {std::cerr << e.what() << '\n';return 2;}
