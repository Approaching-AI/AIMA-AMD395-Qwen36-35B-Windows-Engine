// Compare the complete native MoE chain with original captured intermediates.
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>
#include "native/providers/gdn/sm121_mtp_moe.h"

void check(hipError_t status) {
    if (status != hipSuccess) throw std::runtime_error(hipGetErrorString(status));
}
std::vector<unsigned char> region(const std::string& path, uint64_t offset, size_t bytes) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file || file.tellg() <= 0 || uint64_t(file.tellg()) > (uint64_t(4) << 30u) ||
        offset > uint64_t(file.tellg()) || !bytes || bytes > (size_t(1) << 30u) ||
        bytes > uint64_t(file.tellg()) - offset) throw std::runtime_error("input extent: " + path);
    std::vector<unsigned char> data(bytes);
    file.seekg(static_cast<std::streamoff>(offset));
    file.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(bytes));
    if (!file) throw std::runtime_error("short read: " + path);
    return data;
}
std::vector<unsigned char> whole(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file || file.tellg() <= 0 || file.tellg() > (16u << 20u))
        throw std::runtime_error("captured file extent: " + path);
    return region(path, 0u, static_cast<size_t>(file.tellg()));
}
struct Input {
    unsigned char* raw = nullptr;
    unsigned char* data = nullptr;
    size_t bytes = 0;
    std::string path;
    uint64_t offset = 0;
};
struct Scratch {
    std::vector<void*> allocations;
    std::vector<Input> inputs;
    ~Scratch() {
        if (hipDeviceSynchronize() != hipSuccess) return;
        for (void* pointer : allocations) (void)hipFree(pointer);
    }
    unsigned char* allocate(size_t bytes) {
        unsigned char* pointer = nullptr;
        check(hipMalloc(reinterpret_cast<void**>(&pointer), bytes));
        allocations.push_back(pointer);
        return pointer;
    }
    void* upload(const std::string& path, uint64_t offset, size_t bytes) {
        auto host = region(path, offset, bytes);
        auto* raw = allocate(bytes + 512u);
        check(hipMemset(raw, 0xa5, 256u));
        check(hipMemset(raw + 256u + bytes, 0xa5, 256u));
        check(hipMemcpy(raw + 256u, host.data(), bytes, hipMemcpyHostToDevice));
        inputs.push_back({raw, raw + 256u, bytes, path, offset});
        return raw + 256u;
    }
    size_t immutable_mismatches() const {
        size_t bad = 0;
        for (const auto& input : inputs) {
            for (size_t first = 0; first < input.bytes; first += 4u << 20u) {
                const size_t count = std::min(size_t(4u << 20u), input.bytes - first);
                const auto expected = region(input.path, input.offset + first, count);
                auto actual = expected;
                check(hipMemcpy(actual.data(), input.data + first, count, hipMemcpyDeviceToHost));
                for (size_t i = 0; i < count; ++i) bad += actual[i] != expected[i];
            }
            unsigned char guard[512];
            check(hipMemcpy(guard, input.raw, 256u, hipMemcpyDeviceToHost));
            check(hipMemcpy(guard + 256u, input.data + input.bytes, 256u, hipMemcpyDeviceToHost));
            for (unsigned char value : guard) bad += value != 0xa5u;
        }
        return bad;
    }
};
struct Comparison {
    size_t elements = 0, mismatches = 0, first_row = 0;
    unsigned first_column = 0, first_actual = 0, first_expected = 0;
    void add(uint32_t actual, uint32_t expected, size_t row, unsigned column) {
        ++elements;
        if (actual == expected) return;
        if (!mismatches) { first_row = row; first_column = column; first_actual = actual; first_expected = expected; }
        ++mismatches;
    }
};

int main(int argc, char** argv) try {
    if (argc != 13) throw std::runtime_error(
        "directory router shared_gate shared_gate_up shared_down routed_gate_up_shard offset routed_down_shard offset silu sigmoid router_exp");
    using namespace qrt_sm121_mtp;
    const std::string root = argv[1];
    const auto input = whole(root + "/input.bin");
    const unsigned rows = static_cast<unsigned>(input.size() / 4096u);
    if (!rows || rows > 128u || input.size() != size_t(rows) * 4096u) throw std::runtime_error("input shape");
    Scratch scratch;
    const auto bf = [&](const char* path, uint64_t offset, size_t elements) {
        return static_cast<const uint16_t*>(scratch.upload(path, offset, elements * 2u));
    };
    const auto* device_input = bf((root + "/input.bin").c_str(), 0u, input.size() / 2u);
    MoeWeights weights{bf(argv[2], 0u, 256u * 2048u), bf(argv[3], 0u, 2048u),
        bf(argv[4], 0u, 1024u * 2048u), bf(argv[5], 0u, 2048u * 512u),
        bf(argv[6], std::stoull(argv[7]), size_t(256u) * 1024u * 2048u),
        bf(argv[8], std::stoull(argv[9]), size_t(256u) * 2048u * 512u)};
    MoeTables tables{bf(argv[10], 0u, 65536u + 12u) + 12u, bf(argv[11], 0u, 65536u),
        static_cast<const uint32_t*>(scratch.upload(argv[12], 0u, 8388608u * 4u))};
    struct Stage { const char* label; unsigned width, element_bytes; void* data; };
    const auto stages = [](const MoeBuffers& b) {
        return std::vector<Stage>{{"router",256,2,b.router},{"shared-gate",1,2,b.shared_gate},
            {"shared-gate-up",1024,2,b.shared_gate_up},{"shared-activated",512,2,b.shared_activated},
            {"shared-down",2048,2,b.shared_down},{"shared",2048,2,b.shared},
            {"routed-gate-up",8192,2,b.routed_gate_up},{"routed-activated",4096,2,b.routed_activated},
            {"routed-weighted",16384,2,b.routed_weighted},{"expert-part-1",2048,2,b.routed},
            {"moe-output",2048,2,b.output},{"topk-ids",8,4,b.topk_ids},{"topk-weights",8,4,b.topk_weights}};
    };
    std::map<std::string,std::vector<unsigned char>> expected;
    for (const auto& stage : stages({})) {
        auto data = whole(root + '/' + stage.label + ".bin");
        if (data.size() != size_t(rows) * stage.width * stage.element_bytes) throw std::runtime_error("stage shape");
        expected.emplace(stage.label, std::move(data));
    }
    const size_t capacity = moe_workspace_bytes(2u);
    auto* raw = scratch.allocate(capacity + 512u);auto* workspace = raw + 256u;
    check(hipMemset(raw, 0xa5, capacity + 512u));
    unsigned invalid_launch_cases = 0;
    const auto reject = [&](hipError_t status) {
        if (status != hipErrorInvalidValue) throw std::runtime_error("invalid MoE launch accepted");
        ++invalid_launch_cases;
    };
    reject(launch_moe(device_input,weights,tables,workspace,capacity,0u));
    reject(launch_moe(device_input,weights,tables,workspace,capacity,3u));
    reject(launch_moe(device_input,weights,tables,workspace,capacity,1u,0u));
    reject(launch_moe(device_input,weights,tables,workspace,capacity,1u,4097u));
    reject(launch_moe(nullptr,weights,tables,workspace,capacity,1u));
    reject(launch_moe(device_input,weights,{},workspace,capacity,1u));
    reject(launch_moe(device_input,weights,tables,workspace+1u,capacity,1u));
    reject(launch_moe(device_input,weights,tables,workspace,moe_workspace_bytes(1u)-1u,1u));
    reject(launch_moe(reinterpret_cast<uint16_t*>(workspace),weights,tables,workspace,capacity,1u));
    check(hipDeviceSynchronize());
    std::vector<unsigned char> actual(capacity + 512u);
    check(hipMemcpy(actual.data(),raw,actual.size(),hipMemcpyDeviceToHost));
    if (std::any_of(actual.begin(),actual.end(),[](unsigned char x){return x!=0xa5u;}))
        throw std::runtime_error("rejected launch modified workspace");
    std::map<std::string,Comparison> comparisons;
    size_t guard_bad = 0, padding_bad = 0, invalid_flags = 0;unsigned calls = 0;
    for (unsigned blocks : {7u,1024u}) for (unsigned maximum_rows : {1u,2u})
        for (unsigned first = 0; first < rows;) {
            const unsigned count = std::min(maximum_rows, rows - first);
            MoeBuffers buffers;
            if (!bind_moe_buffers(workspace,capacity,count,&buffers)) throw std::runtime_error("workspace view");
            check(hipMemset(raw,0xa5,capacity+512u));
            check(launch_moe(device_input+size_t(first)*2048u,weights,tables,workspace,capacity,count,blocks));
            ++calls;check(hipDeviceSynchronize());
            check(hipMemcpy(actual.data(),raw,actual.size(),hipMemcpyDeviceToHost));
            std::vector<bool> used(capacity,false);
            for (const auto& stage : stages(buffers)) {
                const size_t offset = reinterpret_cast<uintptr_t>(stage.data) - reinterpret_cast<uintptr_t>(workspace);
                const size_t bytes = size_t(count) * stage.width * stage.element_bytes;
                if (offset > capacity || bytes > capacity-offset) throw std::runtime_error("stage extent");
                std::fill_n(used.begin()+offset,bytes,true);
                for (unsigned row=0;row<count;++row) for(unsigned column=0;column<stage.width;++column) {
                    uint32_t observed=0,wanted=0;
                    std::memcpy(&observed,actual.data()+256u+offset+(size_t(row)*stage.width+column)*stage.element_bytes,stage.element_bytes);
                    std::memcpy(&wanted,expected.at(stage.label).data()+(size_t(first+row)*stage.width+column)*stage.element_bytes,stage.element_bytes);
                    comparisons[stage.label].add(observed,wanted,first+row,column);
                }
            }
            const size_t flag_offset = reinterpret_cast<uintptr_t>(buffers.invalid) - reinterpret_cast<uintptr_t>(workspace);
            uint32_t flag=0;std::memcpy(&flag,actual.data()+256u+flag_offset,4u);invalid_flags+=flag!=0;
            std::fill_n(used.begin()+flag_offset,4u,true);
            for(size_t i=0;i<capacity;++i)padding_bad+=!used[i] && actual[256u+i]!=0xa5u;
            for(size_t i=0;i<256u;++i){guard_bad+=actual[i]!=0xa5u;guard_bad+=actual[actual.size()-1u-i]!=0xa5u;}
            first+=count;
        }
    const size_t input_bad = scratch.immutable_mismatches();size_t mismatches=0;
    std::cout<<"{\"kind\":\"original_mtp_moe_native_chain\",\"rows\":"<<rows
        <<",\"row_batch_configurations\":[1,2],\"maximum_blocks\":[7,1024],\"configurations\":4,\"calls\":"<<calls
        <<",\"workspace_bytes\":"<<capacity<<",\"checks\":{";
    bool comma=false;
    for(const auto& entry:comparisons) {
        if(comma)std::cout<<',';comma=true;const auto& c=entry.second;mismatches+=c.mismatches;
        std::cout<<'"'<<entry.first<<"\":{\"elements\":"<<c.elements<<",\"mismatches\":"<<c.mismatches<<",\"first_difference\":";
        if(c.mismatches)std::cout<<"{\"row\":"<<c.first_row<<",\"column\":"<<c.first_column
            <<",\"actual_bits\":"<<c.first_actual<<",\"expected_bits\":"<<c.first_expected<<'}';
        else std::cout<<"null";std::cout<<'}';
    }
    const bool passed=!(mismatches||guard_bad||padding_bad||invalid_flags||input_bad);
    std::cout<<"},\"guard_mismatches\":"<<guard_bad<<",\"padding_mismatches\":"<<padding_bad
        <<",\"invalid_device_flags\":"<<invalid_flags<<",\"immutable_input_byte_mismatches\":"<<input_bad
        <<",\"invalid_launch_cases\":"<<invalid_launch_cases<<",\"passed\":"<<(passed?"true":"false")
        <<",\"inference_acceptance\":false,\"performance_acceptance\":false}\n";
    return passed?0:1;
} catch(const std::exception& error) {std::cerr<<error.what()<<'\n';return 2;}
