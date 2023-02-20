#pragma once
#include "glm/glm.hpp"
#include <string>
#include <vector>
#include <memory>


namespace mcrt {
	struct AABB {
		glm::vec3 min = { 1.0f, 1.0f, 1.0f }; // (1,1,1) is the max coordinate that is possible in the normalized scene
		glm::vec3 max = { 0.0f, 0.0f, 0.0f }; // (0,0,0) is the min coordinate that is possible in the normalized scene
	};

	struct TriangleMesh {
		std::vector<glm::vec3> vertices;
		std::vector<glm::ivec3> indices;
		std::vector<glm::vec2> texCoords;
		std::vector<glm::vec3> normals;

		// Mesh AABB
		AABB boundingBox;

		// material data
		glm::vec3 diffuse;
		int diffuseTextureID{ -1 };
	};


	class Model
	{
	public:
		Model();

		void loadModel();
		std::shared_ptr<TriangleMesh> mesh;
	};
}

