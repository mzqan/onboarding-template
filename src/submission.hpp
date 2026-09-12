#pragma once

#include <cstddef>
#include <vector>
// brute force v1: {"runtime_ms": 331.281, "memory_mb": 16.777, "score": 0.695}
// open mp v2: {"runtime_ms": 340.116, "memory_mb": 16.777, "score": 0.725}
// flat vector v3: {"runtime_ms": 287.349, "memory_mb": 16.777, "score": 0.817}
// boundary vs interior v4: {"runtime_ms": 223.805, "memory_mb": 16.777, "score": 1.003}
// views & simd v5: {"runtime_ms": 213.517, "memory_mb": 16.777, "score": 1.032}

struct GridView {
  double* data;
  std::size_t rows;
  std::size_t cols;
  std::size_t stride;
  
  double& operator()(std::size_t i, std::size_t j) const{
    return data[i * stride + j];
  }
};

struct ReadOnlyGridView {
  const double* data;
  std::size_t rows;
  std::size_t cols;
  std::size_t stride;
  
  double operator()(std::size_t i, std::size_t j) const{
    return data[i * stride + j];
  }
};

class Grid {
private:
  std::size_t rows_;
  std::size_t cols_;
  std::size_t stride_;
  std::vector<double> data_;

public:
  Grid(std::size_t rows, std::size_t cols)
    : rows_(rows)
    , cols_(cols)
    , stride_(cols)
    , data_(rows * stride_) {}

  // returns reference to cell, allows r/w
  double& operator()(std::size_t i, std::size_t j){
    return data_[i * stride_ + j];
  }

  // returns copy to cell, read-only
  double operator()(std::size_t i, std::size_t j) const{
    return data_[i * stride_ + j];
  }

  std::size_t rows() const{
    return rows_;
  }

  std::size_t cols() const{
    return cols_;
  }

  std::size_t stride() const{
    return stride_;
  }

  GridView view() {
    return {data_.data(), rows_, cols_, stride_};
  }

  ReadOnlyGridView view() const{
    return {data_.data(), rows_, cols_, stride_};
  }
};  

// Apply the five-point stencil over all interior points, copying the boundary
// values unchanged from old_grid to new_grid
void apply_stencil(const Grid& old_grid, Grid& new_grid){
  ReadOnlyGridView old_view {old_grid.view()};
  GridView new_view {new_grid.view()};

  const std::size_t rows {old_grid.rows()};
  const std::size_t cols {old_grid.cols()};

  // top-bottom boundary unchanged
  for (std::size_t i {0}; i < cols; ++i){
      new_view(0, i) = old_view(0, i);
      new_view(rows - 1, i) = old_view(rows - 1, i);
  }

  //left-right boundary unchanged
  for (std::size_t i {1}; i < rows - 1; ++i){
      new_view(i, 0) = old_view(i, 0);
      new_view(i, cols - 1) = old_view(i, cols - 1);
  }

  // parallelize rows (outputs independent from one another)
  #pragma openmp parallel for
  for (std::size_t i{1}; i < rows - 1; ++i){
      const std::size_t row_start{i * old_view.stride};

      // vectorize inner loop (cols are contiguous in memory)
      #pragma omp simd
      for (std::size_t j{1}; j < cols - 1; ++j){
          const std::size_t index{row_start + j};

          new_view.data[index] =
              0.5 * old_view.data[index]
              + 0.125 * (
                  old_view.data[index - old_view.stride]
                  + old_view.data[index + old_view.stride]
                  + old_view.data[index - 1]
                  + old_view.data[index + 1]
              );
      }
  }
}
