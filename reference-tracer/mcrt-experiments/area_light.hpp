#pragma once
#include <glm/glm.hpp>

namespace mcrt {
	struct LightData {
		glm::vec3 origin;
		glm::vec3 du;
		glm::vec3 dv;
		glm::vec3 normal;
		glm::vec3 power;	// Power in r g b channels respectively;
		float width;
		float height;
	};

	class AreaLight
	{
	public:
		AreaLight(bool twoSided, LightData initData);

		LightData lightProps;
	private:
		bool twoSided = false;
	};
}


