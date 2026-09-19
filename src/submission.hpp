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

// cache line is 64B on the x86-64 evaluator
static constexpr std::size_t kCacheLine = 64;
static constexpr std::size_t kLaneElements = kCacheLine / sizeof(double);

// interior loop starts at col 1, so prefix each row by (lane - 1) doubles such that col 1 is on a 64-byte boundary => aligned loads
static constexpr std::size_t kRowPrefix = kLaneElements - 1;
static constexpr std::size_t kPrefixBytes = kRowPrefix * sizeof(double);

// round up to next multiple of align (must be power of 2)
static constexpr std::size_t round_up(std::size_t n, std::size_t align) noexcept {
    // add (align - 1) to "overshoot"
    // truncate down by zeroing bits to remain a multiple of align 
    return (n + align - 1u) & ~(align - 1u);
}

// pad each row to whole cache lines so every row's col 1 keeps the same 64-byte alignment
static std::size_t padded_stride(std::size_t cols) noexcept {
    return round_up((cols + kRowPrefix) * sizeof(double), kCacheLine) / sizeof(double);
}

// RAII: resource acquisition is initialization, ties resource lifetime to object
// when object is destroyed, resource is automatically released
struct AlignedFree { void operator()(double* p) const noexcept { std::free(p); } };
// unique ptr is one implementation of RAII, requires instance of deleter (AlignedFree)
using AlignedBuffer = std::unique_ptr<double[], AlignedFree>;

static AlignedBuffer make_aligned_buffer(std::size_t count) {
    if (count == 0) return {};
    
    const std::size_t bytes = round_up(count * sizeof(double), kCacheLine);
    
    void* raw = std::aligned_alloc(kCacheLine, bytes);
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
    //points at cell (0,0) by skipping kRowPrefix doubles
    double* base_{nullptr};
public:
    Grid(std::size_t rows, std::size_t cols)
        : shape_{rows, cols}
        , strides_{padded_stride(cols), 1}
        , data_{make_aligned_buffer(kRowPrefix + rows * strides_[0])}
        , base_{data_.get() ? data_.get() + kRowPrefix : nullptr} {}

    Grid(const Grid&) = delete;
    Grid& operator=(const Grid&) = delete;
    Grid(Grid&&) = default;
    Grid& operator=(Grid&&) = default;

    // r/w
    double& operator()(std::size_t row, std::size_t col){
        return base_[row * strides_[0] + col];
    }

    // read-only
    const double& operator()(std::size_t row, std::size_t col) const{
        return base_[row * strides_[0] + col];
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
        return {base_, shape_, strides_};
    }

    // read-only
    View<const double, 2> view() const{
        return {base_, shape_, strides_};
    }
};

// Copy the top and bottom boundary rows from old_grid to new_grid
// Returns true if there is an interior to stencil,
// false if the boundary covers the entire grid
inline bool copy_boundaries(const double* __restrict old_data, double* __restrict new_data, std::size_t rows, std::size_t cols, std::size_t stride) noexcept {
    if (rows == 0 || cols == 0) return false;

    // memcpy avoids per-element stride arithmetic and View overhead (vs. for loop + assignment)
    std::memcpy(new_data, old_data, cols * sizeof(double));
    if (rows > 1)
        std::memcpy(new_data + (rows-1) * stride, old_data + (rows-1) * stride, cols * sizeof(double));

    // everything is a boundary for 1 or 2 col. grids
    if (cols < 3) {
        for (std::size_t i = 1; i + 1 < rows; ++i)
            std::memcpy(new_data + i * stride, old_data + i * stride, cols * sizeof(double));
        return false;
    }

    return rows >= 3;
}

// Applies five-point stencil over all interior rows [1, rows-1).
inline void apply_stencil_interior(const double* __restrict old_data, double* __restrict new_data, std::size_t rows, std::size_t cols, std::size_t stride) noexcept {
    const std::size_t interior_cols = cols - 2;

    // parallelism: rows are independent and cost the same (reads old_grid), schedule(static) splits them in equal chunks without sync nor load-balancing
    #pragma omp parallel for schedule(static)
    for (std::size_t i = 1; i < rows - 1; ++i){
        // each row's col 0 sits kPrefixBytes into a cache line (col 1 is aligned)
        // tells compiler to emit vmovapd (aligned) over vmovupd load instruction => faster
        const double* row_cur   = static_cast<const double*>(__builtin_assume_aligned(old_data +  i * stride, kCacheLine, kPrefixBytes));
        const double* row_above = static_cast<const double*>(__builtin_assume_aligned(old_data + (i - 1) * stride, kCacheLine, kPrefixBytes));
        const double* row_below = static_cast<const double*>(__builtin_assume_aligned(old_data + (i + 1) * stride, kCacheLine, kPrefixBytes));
        double* row_dst = static_cast<double*>(__builtin_assume_aligned(new_data +  i * stride, kCacheLine, kPrefixBytes));

        // left and right boundaries copied while row i is already in cache
        row_dst[0] = row_cur[0];
        row_dst[cols - 1] = row_cur[cols - 1];

        // vectorization: compiler packs 4 doubles per AVX2 instruction cleanly (re-indexed to 0 instead of handling a leading partial chunk)
        // hot path: writes start at col 1, kRowPrefix ensures a 64-byte boundary 
        #pragma omp simd
        for (std::size_t j = 0; j < interior_cols; ++j){
            row_dst[j + 1] =
                0.5  * row_cur[j + 1]
                + 0.125 * (
                    row_above[j + 1]    // up
                    + row_below[j + 1]  // down
                    + row_cur[j]        // left
                    + row_cur[j + 2]    // right
                );
        }
    }
}

// Apply the five-point stencil over all interior points
// Keep boundary values unchanged from old_grid to new_grid
void apply_stencil(const Grid& old_grid, Grid& new_grid){
    const auto& [rows, cols] = old_grid.shape();
    const std::size_t stride = old_grid.strides()[0];

    const double* __restrict old_data = old_grid.view().data;
    double* __restrict new_data = new_grid.view().data;

    if (!copy_boundaries(old_data, new_data, rows, cols, stride)) return;

    apply_stencil_interior(old_data, new_data, rows, cols, stride);
}
