#pragma once

#include <cstddef>
#include <vector>
#include <array>

// brute force v1: {"runtime_ms": 331.281, "memory_mb": 16.777, "score": 0.695}
// open mp v2 (ignore): {"runtime_ms": 340.116, "memory_mb": 16.777, "score": 0.725}
// flat vector v3: {"runtime_ms": 287.349, "memory_mb": 16.777, "score": 0.817}
// boundary vs interior v4: {"runtime_ms": 223.805, "memory_mb": 16.777, "score": 1.003}
// views & simd v5: {"runtime_ms": 213.517, "memory_mb": 16.777, "score": 1.032}
// open mp v6: {"runtime_ms": 116.867, "memory_mb": 16.777, "score": 2.174}
// templated view v7: {"runtime_ms": 119.108, "memory_mb": 16.777, "score": 2.391}
// n dimensional view + restrict v8: {"runtime_ms": 113.251, "memory_mb": 16.777, "score": 3.009}

template <typename T, std::size_t N>
struct View {
    T* data;
    std::array<std::size_t, N> shape;
    std::array<std::size_t, N> strides;

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

    // # of elements b/w 
    const std::array<std::size_t, 2>& strides() const{
        return strides_;
    }

    View<double, 2> view(){
        return {data_.data(), shape_, strides_};
    }

    View<const double, 2> view() const{
        return {data_.data(), shape_, strides_};
    }
};

// Apply the five-point stencil over all interior points, copying the boundary
// values unchanged from old_grid to new_grid
void apply_stencil(const Grid& old_grid, Grid& new_grid){
    View<const double, 2> old_view {old_grid.view()};
    View<double, 2> new_view {new_grid.view()};

    // confirms to compiler that input and output arrays do not overlap in memory
    const double* __restrict old_data {old_view.data};
    double* __restrict new_data {new_view.data};

    const std::size_t rows {old_grid.shape()[0]};
    const std::size_t cols {old_grid.shape()[1]};
    const std::size_t stride {old_grid.strides()[0]};

    // top-bottom boundary unchanged
    for (std::size_t i = 0; i < cols; ++i){
        new_view(0, i) = old_view(0, i);
        new_view(rows - 1, i) = old_view(rows - 1, i);
    }

    //left-right boundary unchanged
    for (std::size_t i = 1; i < rows - 1; ++i){
        new_view(i, 0) = old_view(i, 0);
        new_view(i, cols - 1) = old_view(i, cols - 1);
    }

    // parallelize rows (outputs independent from one another)
    #pragma omp parallel for
    for (std::size_t i = 1; i < rows - 1; ++i){

        const std::size_t row_start {i * stride};

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
