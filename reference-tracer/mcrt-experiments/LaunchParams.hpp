#pragma once

#include "optix7.hpp"
#include "glm/glm.hpp"
#include "area_light.hpp"
#include <vector>

namespace mcrt {

	enum { RADIANCE_RAY_TYPE = 0, SHADOW_RAY_TYPE, RAY_TYPE_COUNT };

	/**
	* ================
	*	HELPER TYPES 
	* ================
	*/
	struct PixelBuffer {
		float* colorBuffer;
		int size;
	};

	struct UVWorldData {
		glm::vec3 worldPosition;
		glm::vec3 worldNormal;
		glm::vec3 diffuseColor;
	};



	/**
	* =======================
	*	  TUTORIAL PASS
	* =======================
	*/

	struct MeshSBTData {
		glm::vec3 color;
		glm::vec3 *vertex;
		glm::vec3* normal;
		glm::vec2* texcoord;
		glm::ivec3* index;

		bool hasTexture;
		cudaTextureObject_t texture;

		int objectType;
	};

	struct referenceTracerPRD {
		glm::vec3    emitted;
		glm::vec3    radiance;
		glm::vec3    attenuation;
		glm::vec3    origin;
		glm::vec3    direction;
		unsigned int seed;
		int          countEmitted;
		int          done;
		int          pad;
	};

	struct LaunchParamsTutorial
	{
		int frameID = 0;

		struct {
			uint32_t* colorBuffer;
			glm::ivec2 size;
		} frame;

		struct {
			glm::vec3 position;
			glm::vec3 direction;
			glm::vec3 horizontal;
			glm::vec3 vertical;
		} camera;

		int w;
		int h;
		int samplesPerPixel;
		int stratifyResX;
		int stratifyResY;
		int amountLights;
		LightData* lights;

		OptixTraversableHandle traversable;
	};
}