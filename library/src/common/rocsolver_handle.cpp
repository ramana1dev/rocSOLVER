/* **************************************************************************
 * Copyright (C) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "rocsolver_handle.hpp"
#include "rocblas.hpp"

#include <memory>
#include <unordered_map>
#include <mutex>

ROCSOLVER_BEGIN_NAMESPACE

// Thread-safe map to store rocsolver handle data per rocblas_handle
struct rocsolver_handle_wrapper
{
    std::shared_ptr<rocsolver_handle_data_> data;
};

static std::unordered_map<rocblas_handle, rocsolver_handle_wrapper> handle_map;
static std::mutex handle_map_mutex;

rocblas_status rocsolver_set_alg_mode_impl(rocblas_handle handle,
                                           const rocsolver_function func,
                                           const rocsolver_alg_mode mode)
{
    if(!handle)
        return rocblas_status_invalid_handle;
    if(mode == rocsolver_alg_mode_mixed)
        return rocblas_status_invalid_value;

    std::lock_guard<std::mutex> lock(handle_map_mutex);

    auto& wrapper = handle_map[handle];
    if(!wrapper.data)
        wrapper.data = std::make_shared<rocsolver_handle_data_>();

    rocsolver_handle_data handle_data = wrapper.data.get();
    handle_data->checksum = sizeof(rocsolver_handle_data_);

    switch(func)
    {
    case rocsolver_function_gesvd:
    case rocsolver_function_bdsqr:
        if(mode == rocsolver_alg_mode_gpu || mode == rocsolver_alg_mode_hybrid)
        {
            handle_data->bdsqr_mode = mode;
            return rocblas_status_success;
        }
        break;
    case rocsolver_function_sterf:
        if(mode == rocsolver_alg_mode_gpu || mode == rocsolver_alg_mode_hybrid)
        {
            handle_data->sterf_mode = mode;
            return rocblas_status_success;
        }
        break;
    case rocsolver_function_steqr:
        if(mode == rocsolver_alg_mode_gpu || mode == rocsolver_alg_mode_hybrid)
        {
            handle_data->steqr_mode = mode;
            return rocblas_status_success;
        }
        break;
    case rocsolver_function_syev_heev:
        if(mode == rocsolver_alg_mode_gpu || mode == rocsolver_alg_mode_hybrid)
        {
            handle_data->sterf_mode = mode;
            handle_data->steqr_mode = mode;
            return rocblas_status_success;
        }
        break;
    default:
        return rocblas_status_invalid_value;
    }

    return rocblas_status_invalid_value;
}

rocblas_status rocsolver_get_alg_mode_impl(rocblas_handle handle,
                                           const rocsolver_function func,
                                           rocsolver_alg_mode* mode)
{
    if(!handle || !mode)
        return rocblas_status_invalid_handle;

    std::lock_guard<std::mutex> lock(handle_map_mutex);

    auto it = handle_map.find(handle);
    if(it == handle_map.end() || !it->second.data)
    {
        *mode = rocsolver_alg_mode_gpu; // default
        return rocblas_status_success;
    }

    rocsolver_handle_data handle_data = it->second.data.get();

    if(handle_data->checksum != sizeof(rocsolver_handle_data_))
        return rocblas_status_internal_error;

    switch(func)
    {
    case rocsolver_function_gesvd:
    case rocsolver_function_bdsqr:
        *mode = handle_data->bdsqr_mode;
        break;
    case rocsolver_function_sterf:
        *mode = handle_data->sterf_mode;
        break;
    case rocsolver_function_steqr:
        *mode = handle_data->steqr_mode;
        break;
    case rocsolver_function_syev_heev:
        if(handle_data->sterf_mode == handle_data->steqr_mode)
            *mode = handle_data->sterf_mode;
        else
            *mode = rocsolver_alg_mode_mixed;
        break;
    default:
        return rocblas_status_invalid_value;
    }

    return rocblas_status_success;
}

ROCSOLVER_END_NAMESPACE

extern "C" {

rocblas_status rocsolver_set_alg_mode(rocblas_handle handle,
                                      const rocsolver_function func,
                                      const rocsolver_alg_mode mode)
try
{
    return rocsolver::rocsolver_set_alg_mode_impl(handle, func, mode);
}
catch(...)
{
    return rocsolver::exception_to_rocblas_status();
}

rocblas_status rocsolver_get_alg_mode(rocblas_handle handle,
                                      const rocsolver_function func,
                                      rocsolver_alg_mode* mode)
try
{
    return rocsolver::rocsolver_get_alg_mode_impl(handle, func, mode);
}
catch(...)
{
    return rocsolver::exception_to_rocblas_status();
}

} // extern "C"
