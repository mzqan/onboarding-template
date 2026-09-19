#pragma once

#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <array>
#include <new>

// can access Grid data using its shape and strides without copying it
template <typename T, std::size_t N>
struct View {
    T* data;
    std::array<std::size_t, N> shape;
    std::array<std::size_t, N> strides;

    // accessed like v(a, b, c, ..)
    template <typename... Indices>
    T& operator()(Indices... indices) const {
        static_assert(sizeof...(Indices) == N);

        const std::array<std::size_t, N> index{static_cast<std::size_t>(indices)...};

        std::size_t offset{0};

        for (std::size_t i = 0; i < N; ++i){
            offset += index[i] * strides[i];
        }

        return data[offset];
    }
};

// evaluator targets AVX2 (-march=x86-64-v3): 256-bit/32-byte hardware register
static constexpr std::size_t kRowAlign = 32;

// round up to next multiple of align (must be power of 2)
static constexpr std::size_t round_up(std::size_t n, std::size_t align) noexcept {
    // add (align - 1) to "overshoot"
    // truncate down by zeroing bits to remain a multiple of align 
    return (n + align - 1u) & ~(align - 1u);
}

// pad each row so its start is kRowAlign-byte aligned
static std::size_t padded_stride(std::size_t cols) noexcept {
    return round_up(cols * sizeof(double), kRowAlign) / sizeof(double);
}

// RAII: resource acquisition is initialization, ties resource lifetime to object
// when object is destroyed, resource is automatically released
struct AlignedFree { void operator()(double* p) const noexcept { std::free(p); } };
// unique ptr is one implementation of RAII, requires instance of deleter (AlignedFree)
using AlignedBuffer = std::unique_ptr<double[], AlignedFree>;


static AlignedBuffer make_aligned_buffer(std::size_t count) {
    if (count == 0) return {};
    
    const std::size_t bytes = round_up(count * sizeof(double), kRowAlign);
    
    void* raw = std::aligned_alloc(kRowAlign, bytes);
    if (!raw) throw std::bad_alloc{};
    
    // zero-initialize
    std::memset(raw, 0, bytes);

    return AlignedBuffer{static_cast<double*>(raw)};
}

class Grid {
private:
    std::array<std::size_t, 2> shape_;
    std::array<std::size_t, 2> strides_;
    AlignedBuffer data_;

public:
    Grid(std::size_t rows, std::size_t cols)
        : shape_{rows, cols}
        , strides_{padded_stride(cols), 1}
        , data_{make_aligned_buffer(rows * strides_[0])} {}

    Grid(const Grid&) = delete;
    Grid& operator=(const Grid&) = delete;
    Grid(Grid&&) = default;
    Grid& operator=(Grid&&) = default;

    // r/w
    double& operator()(std::size_t row, std::size_t col){
        return data_[row * strides_[0] + col];
    }

    // read-only
    const double& operator()(std::size_t row, std::size_t col) const{
        return data_[row * strides_[0] + col];
    }

    // # of elements per dimension
    const std::array<std::size_t, 2>& shape() const{
        return shape_;
    }

    // # of elements to move in memory by dimension
    const std::array<std::size_t, 2>& strides() const{
        return strides_;
    }

    // r/w
    View<double, 2> view(){
        return {data_.get(), shape_, strides_};
    }

    // read-only
    View<const double, 2> view() const{
        return {data_.get(), shape_, strides_};
    }
};

// copy top and bottom boundary rows into new_view (includes corners)
inline void copy_boundaries(View<const double, 2> old_view, View<double, 2> new_view, std::size_t rows, std::size_t cols, std::size_t stride) {
    // memcpy copies a contiguous block in one call
    // avoids per-element stride arithmetic and View overhead
    std::memcpy(new_view.data, old_view.data, cols * sizeof(double));
    std::memcpy(new_view.data + (rows-1)*stride,  old_view.data + (rows-1)*stride,  cols * sizeof(double));
}

// Apply the five-point stencil over all interior points, copying the boundary
// values unchanged from old_grid to new_grid
void apply_stencil(const Grid& old_grid, Grid& new_grid){
    View<const double, 2> old_view {old_grid.view()};
    View<double, 2> new_view {new_grid.view()};

    // tells compiler that input and output arrays do not overlap in memory => more aggressive optimizations possible
    const double* __restrict old_data {old_view.data};
    double* __restrict new_data {new_view.data};

    const auto& [rows, cols] = old_grid.shape();
    const std::size_t stride {old_grid.strides()[0]};

    copy_boundaries(old_view, new_view, rows, cols, stride);

    // parallelize rows (outputs independent from one another)
    #pragma omp parallel for
    for (std::size_t i = 1; i < rows - 1; ++i){
        // tells compiler to use AVX2's vmovdqa (32-byte aligned) over vmovdqu (unaligned) load instruction => faster
        const double* row_cur   = static_cast<const double*>(__builtin_assume_aligned(old_data +  i      * stride, kRowAlign));
        const double* row_above = static_cast<const double*>(__builtin_assume_aligned(old_data + (i - 1) * stride, kRowAlign));
        const double* row_below = static_cast<const double*>(__builtin_assume_aligned(old_data + (i + 1) * stride, kRowAlign));
        double*       row_dst   = static_cast<double*>      (__builtin_assume_aligned(new_data +  i      * stride, kRowAlign));

        // left and right boundaries copied while row i is already in cache
        row_dst[0] = row_cur[0];
        row_dst[cols - 1] = row_cur[cols - 1];

        for (std::size_t j = 1; j < cols - 1; ++j){
            row_dst[j] =
                0.5 * row_cur[j]
                + 0.125 * (
                    row_above[j]
                    + row_below[j]
                    + row_cur[j - 1]
                    + row_cur[j + 1]
                );
        }
    }
}
