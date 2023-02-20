#pragma once
#include "game_object.hpp"
#include "voxelizer.hpp"

#include <vector>

namespace mcrt {
	class RadianceCell
	{
	public:
		RadianceCell(glm::ivec3 coord, glm::ivec3 res, float scale);

		void addObject(std::shared_ptr<GameObject> obj);
		void removeObject(std::shared_ptr<GameObject> obj);

		bool addUVIfInside(glm::vec2 uv, glm::vec3 uvWorldCoord);
		int amountObjects() { return objectsInside.size(); };

		bool intersects(Voxel v);
		bool contains(glm::vec3 coord);
		
		glm::ivec3 getCellCoords() { return coord; };
		std::vector<glm::vec3>& getVertices() { return vertices; };
		std::vector<glm::ivec3>& getIndices() { return indices; };
		glm::vec3 getCenter() { return min + 0.5f * (max - min); };
		std::vector<glm::vec3> getNormals();
		std::vector<glm::vec2>& getUVsInside() { return uvsInside; };
		int getAmountUVsInside() { return uvsInside.size(); };

		glm::vec3 min;
		glm::vec3 max;
	private:
		glm::ivec3 coord;
		std::vector<std::shared_ptr<GameObject>> objectsInside;
		std::vector<glm::vec2> uvsInside;

		std::vector<glm::vec3> vertices;
		std::vector<glm::ivec3> indices;
	};
}


