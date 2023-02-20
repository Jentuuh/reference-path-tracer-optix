#pragma once
#include <optix_device.h>
#include "LaunchParams.hpp"

#define PI 3.14159265358979323846f

using namespace mcrt;
// ==========================================================
// https://www.sjbrown.co.uk/posts/spherical-harmonic-basis/
// ==========================================================
namespace mcrt {
	// This file defines functions for the first 3 orders of basis functions 
	// of spherical harmonics.

	// TODO: optimizations: we can precompute the cosines, sines and divisions beforehand and reuse them!
	static __forceinline__ __device__ double Y_0_0()
	{
		return (0.5f) * sqrtf((1.0f / PI));
	}

	//static __forceinline__ __device__ double Y_min1_1(double phi, double theta)
	//{
	//	return clamp((0.5f) * sqrtf((3.0f / PI)) * sin(phi) * sin(theta), 0.0f, 100.0f);
	//}

	//static __forceinline__ __device__ double Y_0_1(double phi, double theta)
	//{
	//	return clamp((0.5f) * sqrtf((3.0f / PI)) * cos(theta), 0.0f, 100.0f);
	//}

	//static __forceinline__ __device__ double Y_1_1(double phi, double theta)
	//{
	//	return clamp((0.5f) * sqrtf((3.0f / PI)) * cos(phi) * sin(theta), 0.0f, 100.0f);
	//}

	//static __forceinline__ __device__ double Y_min2_2(double phi, double theta)
	//{
	//	return clamp((0.5f) * sqrtf((15.0f / PI)) * sin(phi) * cos(phi) * sin(theta) * sin(theta), 0.0f, 100.0f);
	//}

	//static __forceinline__ __device__ double Y_min1_2(double phi, double theta)
	//{
	//	return clamp((0.5f) * sqrtf((15.0f / PI)) * sin(phi) * sin(theta) * cos(theta), 0.0f, 100.0f);
	//}

	//static __forceinline__ __device__ double Y_0_2(double phi, double theta)
	//{
	//	return clamp((0.25f) * sqrtf((5.0f / PI)) * ((3 * cos(theta) * cos(theta)) - 1), 0.0f, 100.0f);
	//}

	//static __forceinline__ __device__ double Y_1_2(double phi, double theta)
	//{
	//	return clamp((0.5f) * sqrtf((15.0f / PI)) * cos(phi) * sin(theta) * cos(theta), 0.0f, 100.0f);
	//}

	//static __forceinline__ __device__ double Y_2_2(double phi, double theta)
	//{
	//	return clamp((0.25f) * sqrtf((15.0f / PI)) * ((cos(phi) * cos(phi)) - (sin(phi) * sin(phi))) * sin(theta) * sin(theta), 0.0f, 100.0f);
	//}

	static __forceinline__ __device__ double Y_min1_1(double phi, double theta)
	{
		return (0.5f) * sqrtf((3.0f / PI)) * sin(phi) * sin(theta);
	}

	static __forceinline__ __device__ double Y_0_1(double phi, double theta)
	{
		return (0.5f) * sqrtf((3.0f / PI)) * cos(theta);
	}

	static __forceinline__ __device__ double Y_1_1(double phi, double theta)
	{
		return (0.5f) * sqrtf((3.0f / PI)) * cos(phi) * sin(theta);
	}

	static __forceinline__ __device__ double Y_min2_2(double phi, double theta)
	{
		return (0.5f) * sqrtf((15.0f / PI)) * sin(phi) * cos(phi) * sin(theta) * sin(theta);
	}

	static __forceinline__ __device__ double Y_min1_2(double phi, double theta)
	{
		return (0.5f) * sqrtf((15.0f / PI)) * sin(phi) * sin(theta) * cos(theta);
	}

	static __forceinline__ __device__ double Y_0_2(double phi, double theta)
	{
		return (0.25f) * sqrtf((5.0f / PI)) * ((3 * cos(theta) * cos(theta)) - 1);
	}

	static __forceinline__ __device__ double Y_1_2(double phi, double theta)
	{
		return (0.5f) * sqrtf((15.0f / PI)) * cos(phi) * sin(theta) * cos(theta);
	}

	static __forceinline__ __device__ double Y_2_2(double phi, double theta)
	{
		return (0.25f) * sqrtf((15.0f / PI)) * ((cos(phi) * cos(phi)) - (sin(phi) * sin(phi))) * sin(theta) * sin(theta);
	}


	static __forceinline__ __device__ double Y_min1_1(float x, float y, float z)
	{
		return (0.5f) * sqrtf((3.0f / PI)) * y;
	}

	static __forceinline__ __device__ double Y_0_1(float x, float y, float z)
	{
		return (0.5f) * sqrtf((3.0f / PI)) * z;
	}

	static __forceinline__ __device__ double Y_1_1(float x, float y, float z)
	{
		return (0.5f) * sqrtf((3.0f / PI)) * x;
	}

	static __forceinline__ __device__ double Y_min2_2(float x, float y, float z)
	{
		return (0.5f) * sqrtf((15.0f / PI)) * x * y;
	}

	static __forceinline__ __device__ double Y_min1_2(float x, float y, float z)
	{
		return (0.5f) * sqrtf((15.0f / PI)) * y * z;
	}

	static __forceinline__ __device__ double Y_0_2(float x, float y, float z)
	{
		return (0.25f) * sqrtf((5.0f / PI)) * ((2*z*z) - (x * x) - (y * y));
	}

	static __forceinline__ __device__ double Y_1_2(float x, float y, float z)
	{
		return (0.5f) * sqrtf((15.0f / PI)) * x * z;
	}

	static __forceinline__ __device__ double Y_2_2(float x, float y, float z)
	{
		return (0.25f) * sqrtf((15.0f / PI)) * ((x * x) - (y * y));
	}
}