#pragma once
#include "CUDABuffer.hpp"
#include "LaunchParams.hpp"
#include "camera.hpp"
#include "image.hpp"
#include "scene.hpp"
#include "helpers.hpp"
#include "default_pipeline.hpp"

#include <stb/stb_image.h>

namespace mcrt {
	
	enum CAMERA_MODE {
		TARGET,
		FREE_ROAM
	};

	class Renderer {
	public:
		/*! Constructor : performs setup, including initializing OptiX, creation of module
		 pipelines, programs, SBT etc. */
		Renderer(Scene& scene, const Camera& camera, CAMERA_MODE camMode);

		void render(glm::ivec2 fbSize);

		void resize(const glm::ivec2& newSize, GameObject* viewerObject);

		// Download rendered color buffer from device
		void downloadPixels(uint32_t h_pixels[]);

		// Update camera to render from
		void updateCamera(GameObject* viewerObject);
		void updateCameraInCircle(GameObject* viewerObject, float dt);

	private:
		void writeToImage(std::string fileName, int resX, int resY, void* data);
		void initLightingTextures(int size);

	protected:
		// ------------------
		//	Internal helpers
		// ------------------

		// OptiX initialization, checking for errors
		void initOptix();

		// Creation + configuration of OptiX device context
		void createContext();

		// Fill geometryBuffers with scene geometry
		void fillGeometryBuffers();

		// Upload textures and create CUDA texture objects for them
		void createTextures();

	protected:
		// CUDA device context + stream that OptiX pipeline will run on,
		// and device properties of the device
		CUcontext			cudaContext;
		CUstream			stream;
		cudaDeviceProp		deviceProperties;

		// OptiX context that pipeline will run in
		OptixDeviceContext	optixContext;

		// Pipelines
		std::unique_ptr<DefaultPipeline> tutorialPipeline;


		CUDABuffer lightSourceTexture; // UV map with direct light source (to test the SH projection)
		CUDABuffer colorBuffer;	// Framebuffer we will write to
		CUDABuffer lightDataBuffer;	// In this buffer we'll store our light source data

		Camera renderCamera;
		CAMERA_MODE camMode;
		std::vector<glm::vec3> viewTargets;

		// Scene we are tracing rays against
		Scene& scene;

		// Device-side buffers (one buffer per input mesh!)
		std::vector<CUDABuffer> vertexBuffers;	
		std::vector<CUDABuffer> indexBuffers;
		std::vector<CUDABuffer> normalBuffers;
		std::vector<CUDABuffer> texcoordBuffers;
		std::vector<int> amountVertices;
		std::vector<int> amountIndices;

		std::vector<cudaArray_t>         textureArrays;
		std::vector<cudaTextureObject_t> textureObjects;

	private:
		float CURRENT_TIME = 0.0f;
		float ROTATE_TOTAL_TIME = 5.0f;
	};
}