#pragma once

#include "optix7.hpp"

// std
#include <vector>
#include <assert.h>

/**
*	Credits to Ingo Wald and his SIGGRAPH introduction course to OptiX 7.
*/

namespace mcrt {
	// Wrapper for managing a device-side buffer
	struct CUDABuffer {
		// Returns pointer to device-side memory 
		inline CUdeviceptr d_pointer() const { return (CUdeviceptr)d_ptr; }


		// Resize buffer to given number of bytes
		void resize(size_t size)
		{
			if (d_ptr) free();
			alloc(size);
		}

		// Allocate given number of bytes
		void alloc(size_t size)
		{
			assert(d_ptr == nullptr);
			this->sizeInBytes = size;
			CUDA_CHECK(Malloc((void**)&d_ptr, sizeInBytes));
		}

		// Free allocated memory
		void free()
		{
			CUDA_CHECK(Free(d_ptr));
			d_ptr = nullptr;
			sizeInBytes = 0;
		}

		template<typename T>
		void alloc_and_upload(const std::vector<T>& vt)
		{
			alloc(vt.size() * sizeof(T));
			upload((const T*)vt.data(), vt.size());
		}

		// Copy host memory to device
		template<typename T>
		void upload(const T* t, size_t count)
		{
			assert(d_ptr != nullptr);
			assert(sizeInBytes == count * sizeof(T));
			CUDA_CHECK(Memcpy(d_ptr, (void*)t,
				count * sizeof(T), cudaMemcpyHostToDevice));
		}

		// Copy device memory to host
		template<typename T>
		void download(T* t, size_t count)
		{
			assert(d_ptr != nullptr);
			assert(sizeInBytes == count * sizeof(T));
			CUDA_CHECK(Memcpy((void*)t, d_ptr,
				count * sizeof(T), cudaMemcpyDeviceToHost));
		}

		// Copy device memory section to host
		template<typename T>
		void download_with_offset(T* t, size_t count, int offset)
		{
			assert(d_ptr != nullptr);
			//assert(sizeInBytes == count * sizeof(T));
			//assert(sizeInBytes <= offset * sizeof(T) + count * sizeof(T))
			CUDA_CHECK(Memcpy((void*)t, (T*)d_ptr + offset,
				count * sizeof(T), cudaMemcpyDeviceToHost));
		}

		size_t sizeInBytes{ 0 };
		void* d_ptr{ nullptr };
	};
}