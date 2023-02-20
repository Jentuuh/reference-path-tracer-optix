#include "radiance_grid.hpp"
#include "glm/gtx/string_cast.hpp"
#include <iostream>
namespace mcrt {

	RadianceGrid::RadianceGrid(){}

	void RadianceGrid::init(float cellSize)
	{
		this->cellSize = cellSize;
		// Assuming the scene is normalized and shifted into the positive quadrant
		resolution.x = 1.0f / cellSize;
		resolution.y = 1.0f / cellSize;
		resolution.z = 1.0f / cellSize;


		for (int z = 0; z < resolution.z; z++)
		{
			for (int y = 0; y < resolution.y; y++)
			{
				for (int x = 0; x < resolution.x; x++)
				{
					grid.push_back(RadianceCell{ glm::ivec3{x,y,z}, resolution, cellSize });
				}
			}
		}
	}

	void RadianceGrid::assignObjectsToCells(std::vector<std::shared_ptr<Voxelizer>>& voxelizers)
	{
 		for (auto v: voxelizers)
		{
			AABB objectWorldAABB = v->voxelizedObject->getWorldAABB();
			// floor(minCoords / cellSize)
			glm::ivec3 minRadianceCoords = { std::floorf(objectWorldAABB.min.x / cellSize), std::floorf(objectWorldAABB.min.y / cellSize), std::floorf(objectWorldAABB.min.z / cellSize) };
			glm::ivec3 maxRadianceCoords = { std::floorf(objectWorldAABB.max.x / cellSize), std::floorf(objectWorldAABB.max.y / cellSize), std::floorf(objectWorldAABB.max.z / cellSize) };
			
			// TODO: ALSO MAKE THE POSSIBILITY TO USE THE ACTUAL TRIANGLE INTERSECTIONS INSTEAD OF INTERSECTIONS WITH VOXELS (MORE COSTLY BUT MORE ACCURATE)

			for (int x = minRadianceCoords.x; x < maxRadianceCoords.x; x++)
			{
				for (int y = minRadianceCoords.y; y < maxRadianceCoords.y; y++)
				{
					for (int z = minRadianceCoords.z; z < maxRadianceCoords.z; z++)
					{
						RadianceCell& currentCell = getCell({ x,y,z });
						for (auto& vox : v->resultVoxelGrid)
						{
							// If the cell intersects with the voxel proxy we can add the game object to this cell
							if (currentCell.intersects(vox)) {
								currentCell.addObject(v->voxelizedObject);
							}
						}
					}
				}
			}
		}

		// Print amount of objects for each cell
		int idx = 0;
		for (auto& c : grid)
		{
			std::cout << "Cell " << idx << ": " << c.amountObjects() << " objects." << std::endl;
			idx++;
		}
	}

	void RadianceGrid::assignUVToCells(glm::vec2 uv, glm::vec3 UVWorldCoord)
	{
		for (auto& c : grid)
		{
			// We can return once the UV has been added to a cell, because it cannot be part of multiple cells
			if(c.addUVIfInside(uv, UVWorldCoord))
				return;
		}
	}

	NonEmptyCells RadianceGrid::getNonEmptyCells()
	{
		NonEmptyCells returnStruct;

		// Get all non empty cells
		for (int i = 0; i < grid.size(); i++)
		{
			if (grid[i].getAmountUVsInside() > 0)
			{
				returnStruct.nonEmptyCellIndices.push_back(i);
				returnStruct.nonEmptyCells.push_back(std::make_shared<RadianceCell>(grid[i]));
			}
		}

		return returnStruct;
	}


	RadianceCell& RadianceGrid::getCell(glm::ivec3 coord)
	{
	/*	if((coord.y * resolution.x) + (coord.z * resolution.x * resolution.y) + coord.x < grid.size())*/
			return grid[(coord.y * resolution.x) + (coord.z * resolution.x * resolution.y) + coord.x];
	}

	std::vector<glm::vec3>& RadianceGrid::getVertices()
	{
		std::vector<glm::vec3> allVertices;
		for (auto& r : grid)
		{
			allVertices.insert(allVertices.end(), r.getVertices().begin(), r.getVertices().end());
		}
		return allVertices;
	}


	std::vector<glm::ivec3>& RadianceGrid::getIndices()
	{
		std::vector<glm::ivec3> allIndices;
		for (auto& r : grid)
		{
			allIndices.insert(allIndices.end(), r.getIndices().begin(), r.getIndices().end());
		}
		return allIndices;
	}
}