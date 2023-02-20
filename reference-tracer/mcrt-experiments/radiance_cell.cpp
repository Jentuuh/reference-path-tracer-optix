#include "radiance_cell.hpp"
#include "glm/gtc/matrix_transform.hpp"
#include "glm/gtx/string_cast.hpp"
#include <iostream>

namespace mcrt {
	RadianceCell::RadianceCell(glm::ivec3 coord, glm::ivec3 res, float scale): coord{coord}
	{
        int order = (coord.y * res.x) + (coord.z * res.x * res.y) + coord.x;

        glm::mat4 scaleM = glm::scale(glm::mat4(1.0f), glm::vec3{ scale, scale, scale });
        glm::mat4 transM = glm::translate(glm::mat4(1.0f), glm::vec3{ scale * coord.x, scale * coord.y, scale * coord.z });

        glm::mat4 transform = transM * scaleM;

        // Min and max world coordinate of this cell
        min = transform * glm::vec4{ 0.0f, 0.0f, 0.0f, 1.0f };
        max = transform * glm::vec4{ 1.0f, 1.0f, 1.0f, 1.0f };

        // Transform the cell's vertices into the right position of the grid
        vertices.push_back(transform * glm::vec4{ 0.0f, 0.0f, 0.0f, 1.0f });
        vertices.push_back(transform * glm::vec4{ 1.0f, 0.0f, 0.0f, 1.0f });
        vertices.push_back(transform * glm::vec4{ 0.0f, 1.0f, 0.0f, 1.0f });
        vertices.push_back(transform * glm::vec4{ 1.0f, 1.0f, 0.0f, 1.0f });
        vertices.push_back(transform * glm::vec4{ 0.0f, 0.0f, 1.0f, 1.0f });
        vertices.push_back(transform * glm::vec4{ 1.0f, 0.0f, 1.0f, 1.0f });
        vertices.push_back(transform * glm::vec4{ 0.0f, 1.0f, 1.0f, 1.0f });
        vertices.push_back(transform * glm::vec4{ 1.0f, 1.0f, 1.0f, 1.0f });

        int indicesCube[] = { 0,1,3, 2,3,0,
                         5,7,6, 5,6,4,
                         0,4,5, 0,5,1,
                         2,3,7, 2,7,6,
                         1,5,7, 1,7,3,
                         4,0,2, 4,2,6
        };

        for (int i = 0; i < 12; i++)
            indices.push_back(glm::ivec3(indicesCube[3 * i + 0],
                indicesCube[3 * i + 1],
                indicesCube[3 * i + 2]));
	}

	void RadianceCell::addObject(std::shared_ptr<GameObject> obj) 
	{
        // If the object is not in this cell already
        if(std::find(objectsInside.begin(), objectsInside.end(), obj) == objectsInside.end())
		    objectsInside.push_back(obj);
	}

	void RadianceCell::removeObject(std::shared_ptr<GameObject> obj)
	{
		remove(objectsInside.begin(), objectsInside.end(), obj);
	}

    bool RadianceCell::addUVIfInside(glm::vec2 uv, glm::vec3 uvWorldCoord)
    {
        if (contains(uvWorldCoord))
        {
            uvsInside.push_back(uv);
            return true;
        }
        return false;
    }


    std::vector<glm::vec3> RadianceCell::getNormals()
    {
        std::vector<glm::vec3> normals;
        normals.push_back({ -1.0f, 0.0f, 0.0f });   // LEFT
        normals.push_back({ 1.0f, 0.0f, 0.0f });    // RIGHT
        normals.push_back({ 0.0f, 1.0f, 0.0f });    // UP
        normals.push_back({ 0.0f, -1.0f, 0.0f });   // DOWN
        normals.push_back({ 0.0f, 0.0f, -1.0f });   // FRONT
        normals.push_back({ 0.0f, 0.0f, 1.0f });    // BACK

        return normals;
    }

    bool RadianceCell::intersects(Voxel v)
    {
        // Cells/voxels with no volume
        if (v.min.x == v.max.x || v.min.y == v.max.y || v.min.z == v.max.z || min.x == max.x || min.y == max.y || min.z == max.z)
            return false;

        // Do not overlap on x-axis
        if (v.min.x > max.x || min.x > v.max.x)
            return false;

        // Do not overlap on y-axis
        if (v.min.y > max.y || min.y > v.max.y)
            return false;

        // Do not overlap on z-axis
        if (v.min.z > max.z || min.z > v.max.z)
            return false;
        
        return true;
    }


    bool RadianceCell::contains(glm::vec3 coord)
    {
        //std::cout << coord.x << " " << coord.y << " " << coord.z << std::endl;
        //std::cout << min.x << min.y << min.z << max.x << max.y << max.z << std::endl;
        return coord.x > min.x && coord.x < max.x && coord.y > min.y && coord.y < max.y && coord.z > min.z && coord.z < max.z;
    }

}