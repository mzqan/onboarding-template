#pragma once

#include <cstddef>
#include <vector>
// brute force v1: {"runtime_ms": 331.281, "memory_mb": 16.777, "score": 0.695}
// open mp v2: {"runtime_ms": 340.116, "memory_mb": 16.777, "score": 0.725}
// flat vector v3: {"runtime_ms": 287.349, "memory_mb": 16.777, "score": 0.817}

class Grid {
private:
  std::size_t rows_;
  std::size_t cols_;
  std::vector<double> data_;

public:
  Grid(std::size_t rows, std::size_t cols)
    : rows_(rows)
    , cols_(cols)
    , data_(rows * cols) {}

  // returns reference to cell, allows r/w
  double& operator()(std::size_t i, std::size_t j){
    return data_[i * cols_ + j];
  }

  // returns copy to cell, read-only
  double operator()(std::size_t i, std::size_t j) const{
    return data_[i * cols_ + j];
  }

  std::size_t rows() const {
    return rows_;
  }

  std::size_t cols() const{
    return cols_;
  }
};  

// Apply the five-point stencil over all interior points, copying the boundary
// values unchanged from old_grid to new_grid
void apply_stencil(const Grid& old_grid, Grid& new_grid){
  const std::size_t rows {old_grid.rows()};
  const std::size_t cols {old_grid.cols()};

  #pragma openmp parallel for
  for (std::size_t i{0}; i < rows; ++i){
    for (std::size_t j{0}; j < cols; ++j){
      // boundary points unchanged
      if (i == 0 || i == rows - 1 || 
          j == 0 || j == cols - 1) {
            new_grid(i, j) = old_grid(i, j);
      }
      else {
        // weighted avg
        new_grid(i, j) = 0.5 * old_grid(i, j) + 0.125 * (old_grid(i - 1, j) + old_grid(i + 1, j) + old_grid(i, j - 1) + old_grid(i, j + 1));
      }
    }
  }
}
