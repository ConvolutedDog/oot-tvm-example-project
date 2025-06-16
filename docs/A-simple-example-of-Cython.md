---
title: A simple example of Cython
tags: Cython
abbrlink: ec8e075c
date: 2025-06-14 23:39:13

---

# Tutorial: Binding C++ Functions and Classes to Python using Cython

<!-- more -->

## Table of Contents

1. [Introduction](#introduction)
2. [Project Structure](#project-structure)
3. [C++ Implementation](#cpp-implementation)
4. [Cython Binding Files](#cython-binding-files)
5. [Building the Extension](#building-the-extension)
6. [Using the Python Module](#using-the-python-module)
7. [Best Practices](#best-practices)

## Introduction

Cython is a powerful tool that allows you to write C extensions for Python. It's particularly useful when you want to:

- Speed up Python code by converting it to C
- Interface with existing C/C++ code
- Create Python bindings for C++ classes and functions

This tutorial demonstrates how to create Python bindings for C++ functions and classes using a real-world example.

## Project Structure

A typical Cython project for C++ bindings consists of the following files:

```
src/cc/
├── functions.hpp      # C++ header file
├── functions.cc       # C++ implementation
├── functions.pxd      # Cython declarations
├── functions.pyx      # Cython implementation
└── setup.py          # Build configuration
```

You can find the code [🔗here](https://github.com/BenkangPeng/toyRPC/tree/main/src/cc).

## C++ Implementation

### Header File (functions.hpp)

```cpp
#ifndef CC_FUNCTIONS_HPP
#define CC_FUNCTIONS_HPP

#include <vector>

// Function declarations
std::vector<int> prim(int n);
std::vector<int> fib(int n);
std::vector<std::vector<int>> matmul(std::vector<std::vector<int>> &A,
                                    std::vector<std::vector<int>> &B);

// Class declaration
class matrix {
public:
    matrix();
    std::vector<std::vector<int>> mul(std::vector<std::vector<int>> &A,
                                     std::vector<std::vector<int>> &B);
    std::vector<std::vector<int>> transpose(std::vector<std::vector<int>> &A);
};

#endif
```

### Implementation File(functions.cc)

```cpp
#include "functions.hpp"
#include <stdexcept>
std::vector<int> prim(int n) {
  if (n <= 0)
    return {};
  if (n == 1)
    return {2};

  // calculate the upper bound of the n-th prime numbers
  int upper = static_cast<int>(n * (log(n) + log(log(n)))) + 10;

  std::vector<bool> sieve(upper + 1, true);
  sieve[0] = sieve[1] = false;

  for (int p = 2; p * p <= upper; p++) {
    if (sieve[p]) {
      for (int i = p * p; i <= upper; i += p) {
        sieve[i] = false;
      }
    }
  }

  std::vector<int> primes;
  primes.reserve(n);
  for (int i = 2; primes.size() < static_cast<size_t>(n) && i <= upper; i++) {
    if (sieve[i]) {
      primes.push_back(i);
    }
  }

  return primes;
}

std::vector<int> fib(int n) {
  std::vector<int> res(n);
  res[0] = 1;
  res[1] = 1;
  for (int i = 2; i < n; ++i) {
    res[i] = res[i - 1] + res[i - 2];
  }

  return res;
}

std::vector<std::vector<int>> matmul(std::vector<std::vector<int>> &A,
                                     std::vector<std::vector<int>> &B) {
  if (A.empty() || B.empty() || A[0].size() != B.size()) {
    throw std::invalid_argument(
        "Invalid matrix dimensions for multiplication.");
  }
  std::vector<std::vector<int>> res(A.size(), std::vector<int>(B[0].size(), 0));

  for (int i = 0; i < A.size(); ++i) {
    for (int j = 0; j < B[0].size(); ++j) {
      for (int k = 0; k < B.size(); ++k) {
        res[i][j] += A[i][k] * B[k][j];
      }
    }
  }
  return res;
}

std::vector<std::vector<int>> matrix::mul(std::vector<std::vector<int>> &A,
                                          std::vector<std::vector<int>> &B) {
  if (A.empty() || B.empty() || A[0].size() != B.size()) {
    throw std::invalid_argument(
        "Invalid matrix dimensions for multiplication.");
  }
  std::vector<std::vector<int>> res(A.size(), std::vector<int>(B[0].size(), 0));

  for (int i = 0; i < A.size(); ++i) {
    for (int j = 0; j < B[0].size(); ++j) {
      for (int k = 0; k < B.size(); ++k) {
        res[i][j] += A[i][k] * B[k][j];
      }
    }
  }

  return res;
}

matrix::matrix() {}

std::vector<std::vector<int>>
matrix::transpose(std::vector<std::vector<int>> &A) {
  if (A.empty()) {
    throw std::invalid_argument("Invalid matrix dimensions for transpose.");
  }
  int row = A.size();
  int col = A[0].size();
  std::vector<std::vector<int>> res(col, std::vector<int>(row, 0));

  for (int i = 0; i < row; ++i) {
    for (int j = 0; j < col; ++j) {
      res[j][i] = A[i][j];
    }
  }

  return res;
}
```

## Cython Binding Files

### 1. PXD File (functions.pxd)

The PXD file contains declarations that tell Cython about the C++ types and functions:

```python
from libcpp.vector cimport vector

cdef extern from "functions.cc":
    # Function declarations
    cdef vector[int] prim(int n)
    cdef vector[int] fib(int n)
    cdef vector[vector[int]] matmul(vector[vector[int]] A, vector[vector[int]] B)

    # Class declaration
    cdef cppclass matrix:
        matrix()
        vector[vector[int]] mul(vector[vector[int]] A, vector[vector[int]] B)
        vector[vector[int]] transpose(vector[vector[int]] A)
```

❌ Intuitively, it should be `cdef extern from "functions.hpp"` in `functions.pxd`, and the TVM project also includes header files (`.hpp`) in its `.pxi`. However, this causes errors on my Windows computer. Changing it to `.cc` resolves the issue.

### 2. PYX File (functions.pyx)

The PYX file contains the Python interface implementation:

```python
# distutils: language=c++

from libcpp.vector cimport vector

# import the functions and class from functions.pxd
from functions cimport prim, fib, matmul, matrix

# Function wrappers
def getNPrimes(int n):
    """
    Get the first n prime numbers.
    """
    cdef vector[int] primes = prim(n)
    return primes

def getNFibonacci(int n):
    """
    Get the first n Fibonacci numbers.
    """
    cdef vector[int] res = fib(n)
    return res

def matMul(vector[vector[int]] A, vector[vector[int]] B):
    """
    Multiply two matrices.
    """
    cdef vector[vector[int]] res = matmul(A, B)
    return res

# Class wrapper
cdef class PyMatrix:
    cdef matrix _matrix # wrap a C++ class `matrix` in python class `PyMatrix`

    def __init__(self):
        self._matrix = matrix()

    def mul(self, vector[vector[int]] A, vector[vector[int]] B):
        return self._matrix.mul(A, B)
    
    def transpose(self, vector[vector[int]] A):
        return self._matrix.transpose(A)
```

## Building the Extension

Create a `setup.py` file to build the Cython extension:

```python
from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize("functions.pyx")
)
```

To build the extension, run:

```bash
python setup.py build_ext --inplace
```

The command above will generate a `.so` file(`.pyd` in Windows), for example, `functions.so`.



## Using the Python Module

After building, you can use the module in Python:

```python
# import the functions from functions.so
from functions import getNPrimes, getNFibonacci, matMul, PyMatrix

# Using functions
primes = getNPrimes(10)
fibonacci = getNFibonacci(10)

# Using the matrix class
matrix = PyMatrix()
A = [[1, 2], [3, 4]]
B = [[5, 6], [7, 8]]
result = matrix.mul(A, B)
transposed = matrix.transpose(A)
```

