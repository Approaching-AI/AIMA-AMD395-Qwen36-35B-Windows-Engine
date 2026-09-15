#include "../../native/providers/moe_accumulator/sm121_slab_half_layout.h"
#include <cassert>
#include <cstdio>
#include <initializer_list>
#include <vector>
namespace layout=qrt_sm121_slab_half_layout;
template<unsigned Rows> size_t verify() {
    size_t checked=0u;
    for (unsigned rows:{1u,15u,16u,17u,63u,64u,65u,129u,255u,256u,257u,513u})
        for (unsigned width:{16u,32u,48u,272u,2048u,4096u,4112u,8192u}) {
            const size_t count=layout::records<Rows>(rows,width);
            assert(count && layout::words<Rows>(rows,width)==count*9u);
            std::vector<unsigned char> seen(size_t(rows)*(width/16u),0u);
            size_t index=0u;
            for (unsigned first_row=0u;first_row<rows;first_row+=Rows)
                for (unsigned first_group=0u;first_group<width/16u;first_group+=2u)
                    for (unsigned r=0u;r<Rows;++r) for (unsigned g=0u;g<2u;++g,++index) {
                        const unsigned row=first_row+r,group=first_group+g;
                        assert(layout::row<Rows>(width,index)==row && layout::group<Rows>(width,index)==group);
                        if (row<rows && group<width/16u) {
                            assert(layout::offset<Rows>(rows,width,row,group)==index);
                            assert(!seen[size_t(row)*(width/16u)+group]++);
                        } else assert(layout::offset<Rows>(rows,width,row,group)==layout::invalid);
                        ++checked;
                    }
            assert(index==count);
            for (auto x:seen) assert(x==1u);
        }
    for (unsigned rows:{1u,65536u,524288u}) for (unsigned width:{16u,272u,8192u}) {
        const size_t last=layout::offset<Rows>(rows,width,rows-1u,width/16u-1u);
        assert(last<layout::records<Rows>(rows,width));
        assert(layout::row<Rows>(width,last)==rows-1u && layout::group<Rows>(width,last)==width/16u-1u);
    }
    for (unsigned rows:{0u,524289u,UINT32_MAX}) assert(!layout::words<Rows>(rows,2048u));
    for (unsigned width:{0u,1u,15u,17u,8193u,UINT32_MAX}) assert(!layout::words<Rows>(257u,width));
    return checked;
}
int main() {
    const size_t checked=verify<16u>()+verify<64u>()+verify<256u>();
    std::printf("{\"kind\":\"slab_half_layout_host\",\"independent_record_mappings\":%zu,\"payload_and_control_planes\":true,\"padding_and_large_boundaries_pass\":true,\"mismatches\":0,\"native_execution\":false}\n",checked);
}
