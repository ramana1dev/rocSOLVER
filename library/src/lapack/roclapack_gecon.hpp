/************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.9.0) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     November 2019
 * Copyright (C) 2019-2024 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 * *************************************************************************/

#pragma once

#include "rocblas.hpp"
#include "rocsolver/rocsolver.h"

ROCSOLVER_BEGIN_NAMESPACE

// Forward declarations used by the .cpp
template <typename T>
void rocsolver_gecon_getMemorySize(rocblas_int n,
                                   size_t* size_scalars,
                                   size_t* size_work_workArr,
                                   size_t* size_temp);

template <typename T, typename U, typename V>
rocblas_status rocsolver_gecon_template(rocblas_handle handle,
                                        rocblas_norm norm,
                                        rocblas_int n,
                                        U A,
                                        rocblas_int lda,
                                        const V* anorm,
                                        V* rcond,
                                        T* work,
                                        rocblas_int* info,
                                        T* scalars,
                                        void* work_workArr,
                                        T* temp);

// Real A (float/double): Twork = Treal, U = Treal*, V = Treal
template <typename T>
rocblas_status rocsolver_gecon_impl(rocblas_handle handle,
                                    rocblas_norm norm,
                                    rocblas_int n,
                                    T* A,
                                    rocblas_int lda,
                                    const T* anorm,
                                    T* rcond,
                                    T* work,
                                    rocblas_int* info);

// Complex A (rocblas_complex_num<T>): Twork = Cx<T>, U = Cx<T>*, V = T (real)
template <typename T>
rocblas_status rocsolver_gecon_impl(rocblas_handle handle,
                                    rocblas_norm norm,
                                    rocblas_int n,
                                    rocblas_complex_num<T>* A,
                                    rocblas_int lda,
                                    const T* anorm,
                                    T* rcond,
                                    rocblas_complex_num<T>* work,
                                    rocblas_int* info);

ROCSOLVER_END_NAMESPACE
