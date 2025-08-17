/* **************************************************************************
 * Copyright (C) 2019-2025 Advanced Micro Devices, Inc.
 * *************************************************************************/

#include "roclapack_gecon.hpp"

ROCSOLVER_BEGIN_NAMESPACE

// -----------------------------
// Real A (float/double)
// -----------------------------
template <typename T>
rocblas_status rocsolver_gecon_impl(rocblas_handle handle,
                                    rocblas_norm norm,
                                    rocblas_int n,
                                    T* A,
                                    rocblas_int lda,
                                    const T* anorm,
                                    T* rcond,
                                    T* work,
                                    rocblas_int* info)
{
    ROCSOLVER_ENTER_TOP("gecon", "--norm", norm, "-n", n, "--lda", lda);

    if(!handle)
        return rocblas_status_invalid_handle;

    // basic arg checks (you can tighten later to rocSOLVER style helpers)
    if(n < 0 || lda < n || !A || !anorm || !rcond || !info)
        return rocblas_status_invalid_size;

    // Workspace sizes for Twork = T (real)
    size_t size_scalars, size_work_workArr, size_temp;
    rocsolver_gecon_getMemorySize<T>(n, &size_scalars, &size_work_workArr, &size_temp);

    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_set_optimal_device_memory_size(
            handle, size_scalars, size_work_workArr, size_temp);

    // Allocate device workspace
    void *scalars = nullptr, *work_workArr = nullptr, *temp = nullptr;
    rocblas_device_malloc mem(handle, size_scalars, size_work_workArr, size_temp);
    if(!mem)
        return rocblas_status_memory_error;

    scalars      = mem[0];
    work_workArr = mem[1];
    temp         = mem[2];

    if(size_scalars > 0)
        init_scalars(handle, (T*)scalars);

    // Call the shared template
    return rocsolver_gecon_template<T, T*, T>(handle,
                                              norm,
                                              n,
                                              A,
                                              lda,
                                              anorm,
                                              rcond,
                                              work,
                                              info,
                                              (T*)scalars,
                                              work_workArr,
                                              (T*)temp);
}

// -----------------------------
// Complex A (complex float/double)
// -----------------------------
template <typename T>
rocblas_status rocsolver_gecon_impl(rocblas_handle handle,
                                    rocblas_norm norm,
                                    rocblas_int n,
                                    rocblas_complex_num<T>* A,
                                    rocblas_int lda,
                                    const T* anorm,
                                    T* rcond,
                                    rocblas_complex_num<T>* work,
                                    rocblas_int* info)
{
    using Cx = rocblas_complex_num<T>; // workspace/scalars/temp type

    ROCSOLVER_ENTER_TOP("gecon", "--norm", norm, "-n", n, "--lda", lda);

    if(!handle)
        return rocblas_status_invalid_handle;

    if(n < 0 || lda < n || !A || !anorm || !rcond || !info)
        return rocblas_status_invalid_size;

    // Workspace sizes for Twork = complex
    size_t size_scalars, size_work_workArr, size_temp;
    rocsolver_gecon_getMemorySize<Cx>(n, &size_scalars, &size_work_workArr, &size_temp);

    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_set_optimal_device_memory_size(
            handle, size_scalars, size_work_workArr, size_temp);

    // Allocate device workspace
    void *scalars = nullptr, *work_workArr = nullptr, *temp = nullptr;
    rocblas_device_malloc mem(handle, size_scalars, size_work_workArr, size_temp);
    if(!mem)
        return rocblas_status_memory_error;

    scalars      = mem[0];
    work_workArr = mem[1];
    temp         = mem[2];

    if(size_scalars > 0)
        init_scalars(handle, (Cx*)scalars);

    // IMPORTANT: first template parameter = complex type (matches scalars/temp/work type)
    return rocsolver_gecon_template<Cx, Cx*, T>(handle,
                                                norm,
                                                n,
                                                A,
                                                lda,
                                                anorm,
                                                rcond,
                                                work,
                                                info,
                                                (Cx*)scalars,
                                                work_workArr,
                                                (Cx*)temp);
}

ROCSOLVER_END_NAMESPACE

/*
 * ===========================================================================
 *    C wrappers (must match rocsolver-functions.h exactly)
 * ===========================================================================
 */
extern "C" {

rocblas_status rocsolver_sgecon(rocblas_handle handle,
                                const rocblas_norm norm_type,
                                const rocblas_int n,
                                float* A,
                                const rocblas_int lda,
                                const float* anorm,
                                float* rcond,
                                float* work,
                                rocblas_int* info)
{
    return rocsolver::rocsolver_gecon_impl<float>(
        handle, norm_type, n, A, lda, anorm, rcond, work, info);
}

rocblas_status rocsolver_dgecon(rocblas_handle handle,
                                const rocblas_norm norm_type,
                                const rocblas_int n,
                                double* A,
                                const rocblas_int lda,
                                const double* anorm,
                                double* rcond,
                                double* work,
                                rocblas_int* info)
{
    return rocsolver::rocsolver_gecon_impl<double>(
        handle, norm_type, n, A, lda, anorm, rcond, work, info);
}

rocblas_status rocsolver_cgecon(rocblas_handle handle,
                                const rocblas_norm norm_type,
                                const rocblas_int n,
                                rocblas_float_complex* A,
                                const rocblas_int lda,
                                const float* anorm,
                                float* rcond,
                                rocblas_float_complex* work,
                                rocblas_int* info)
{
    return rocsolver::rocsolver_gecon_impl<float>(
        handle, norm_type, n, A, lda, anorm, rcond, work, info);
}

rocblas_status rocsolver_zgecon(rocblas_handle handle,
                                const rocblas_norm norm_type,
                                const rocblas_int n,
                                rocblas_double_complex* A,
                                const rocblas_int lda,
                                const double* anorm,
                                double* rcond,
                                rocblas_double_complex* work,
                                rocblas_int* info)
{
    return rocsolver::rocsolver_gecon_impl<double>(
        handle, norm_type, n, A, lda, anorm, rcond, work, info);
}

} // extern "C"
