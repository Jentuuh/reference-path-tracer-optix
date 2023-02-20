#pragma once
#include "radiance_cell.hpp"
#include "voxelizer.hpp"

#include "glm/glm.hpp"

#include <vector>

namespace mcrt {
	struct NonEmptyCells {
		std::vector<std::shared_ptr<RadianceCell>> nonEmptyCells;
		std::vector<int> nonEmptyCellIndices;
	};

	class RadianceGrid
	{
	public:
		RadianceGrid();
		void init(float cellSize);
		void assignObjectsToCells(std::vector<std::shared_ptr<Voxelizer>>& voxelizers);
		void assignUVToCells(glm::vec2 uv, glm::vec3 UVWorldCoord);
		NonEmptyCells getNonEmptyCells();

		int getAmountCells() { return grid.size(); };
		int getCellIndex(std::shared_ptr<RadianceCell> cell);
		float getCellSize() { return cellSize; };
		RadianceCell& getCell(glm::ivec3 coord);
		RadianceCell& getCell(int index) { return grid[index]; };
		std::vector<glm::vec3>& getVertices();
		std::vector<glm::ivec3>& getIndices();

		glm::ivec3 resolution;

	private:
		float cellSize = 1.0f;
		std::vector<RadianceCell> grid;
	};
}


