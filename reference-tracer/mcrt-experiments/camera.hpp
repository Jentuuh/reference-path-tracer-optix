#pragma once

#include "glm/glm.hpp"

namespace mcrt {
	class Camera
	{
	public:
		Camera(glm::vec3 pos, glm::vec3 target, glm::vec3 up);

		glm::vec3 position;
		glm::vec3 up;
		glm::vec3 target;
	private:

	};
}
