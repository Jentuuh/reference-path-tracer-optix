#pragma once
#include <glm/glm.hpp>

namespace mcrt {
	inline glm::vec3 generateRandomColor()
	{
		glm::vec3 color;
		for (int i = 0; i < 3; i++)
		{
			color[i] = (float)rand() / RAND_MAX;
		}
		return color;
	}
}
