#include <array>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include "ck_fmha/query_lds_layout.h"
template<unsigned Layout> void check() {
    constexpr unsigned guard=17u,n=qrt_query_lds_layout::words<Layout>();
    std::array<uint32_t,n+2u*guard> data{};
    std::array<unsigned,n> writes{};
    data.fill(0xa5a5a5a5u);
    // Original 256-thread cooperative load order, followed by the actual
    // query/key-thread consumption order for all eight K16 groups.
    for(unsigned thread=0;thread<256;++thread)
        for(unsigned cell=thread;cell<2048;cell+=256) {
            const unsigned at=qrt_query_lds_layout::index<Layout>(cell/128u,cell%128u);
            assert(at<n && ++writes[at]==1u);
            data[guard+at]=0x3f800000u+cell;
        }
    for(unsigned thread=0;thread<256;++thread)
        for(unsigned base=0;base<128;base+=16)
            for(unsigned i=0;i<16;++i) {
                const unsigned at=qrt_query_lds_layout::index<Layout>(thread/16u,base+i);
                assert(at<n && data[guard+at]==0x3f800000u+(thread/16u)*128u+base+i);
            }
    unsigned live=0;
    for(unsigned i=0;i<n;++i){live+=writes[i];if(!writes[i])assert(data[guard+i]==0xa5a5a5a5u);}
    assert(live==2048u);
    for(unsigned i=0;i<guard;++i)assert(data[i]==0xa5a5a5a5u && data[guard+n+i]==0xa5a5a5a5u);
}
int main(){
    check<0>();check<1>();check<2>();check<3>();
    std::puts("query_lds_layout_host_pass layouts=4 loaded_cells=8192 consumed_values=131072 guards_pass=1 native_kernel_executed=0");
}
