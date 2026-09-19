#pragma once

#include <cstddef>
#include <vector>
#include <array>

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

class Grid {
private:
    std::array<std::size_t, 2> shape_;
    std::array<std::size_t, 2> strides_;
    std::vector<double> data_;

public:
    Grid(std::size_t rows, std::size_t cols)
        : shape_{rows, cols}
        , strides_{cols, 1}
        , data_(rows * cols) {}

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
        return {data_.data(), shape_, strides_};
    }

    // read-only
    View<const double, 2> view() const{
        return {data_.data(), shape_, strides_};
    }
};

// copy top and bottom boundary rows into new_view (includes corners)
inline void copy_boundaries(View<const double, 2> old_view, View<double, 2> new_view, std::size_t rows, std::size_t cols) {
    for (std::size_t i = 0; i < cols; ++i){
        new_view(0, i) = old_view(0, i);
        new_view(rows - 1, i) = old_view(rows - 1, i);
    }
}

// Apply the five-point stencil over all interior points, copying the boundary
// values unchanged from old_grid to new_grid
void apply_stencil(const Grid& old_grid, Grid& new_grid){
    View<const double, 2> old_view {old_grid.view()};
    View<double, 2> new_view {new_grid.view()};

    // confirms to compiler that input and output arrays do not overlap in memory
    const double* __restrict old_data {old_view.data};
    double* __restrict new_data {new_view.data};

    const auto& [rows, cols] = old_grid.shape();
    const std::size_t stride {old_grid.strides()[0]};

    copy_boundaries(old_view, new_view, rows, cols);

    // parallelize rows (outputs independent from one another)
    #pragma omp parallel for
    for (std::size_t i = 1; i < rows - 1; ++i){

        const std::size_t row_start {i * stride};

        // left and right boundaries copied while row i is already in cache
        new_data[row_start]            = old_data[row_start];
        new_data[row_start + cols - 1] = old_data[row_start + cols - 1];

        // vectorize inner loop (cols are contiguous in memory)
        #pragma omp simd
        for (std::size_t j = 1; j < cols - 1; ++j){

            const std::size_t index {row_start + j};

            new_data[index] =
                0.5 * old_data[index]
                + 0.125 * (
                    old_data[index - stride]
                    + old_data[index + stride]
                    + old_data[index - 1]
                    + old_data[index + 1]
                );
        }
    }
}
