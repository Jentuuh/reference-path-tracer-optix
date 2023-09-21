#pragma once

#include "glm/glm.hpp"

namespace mcrt {

	struct BasisAxis {
		glm::vec3 u;
		glm::vec3 v;
		glm::vec3 w;
	};

	class Camera
	{
	public:
		Camera(glm::vec3 pos, glm::vec3 target, glm::vec3 up);

		void setOrthographicProjection(float left, float right, float top, float bottom, float near, float far);
		void setPerspectiveProjection(float fov_vert, float aspect, float near, float far);

		void setViewDirection(glm::vec3 position, glm::vec3 direction, glm::vec3 up = glm::vec3{ 0.f, -1.f, 0.f });
		void setViewTarget(glm::vec3 position, glm::vec3 target, glm::vec3 up = glm::vec3{ 0.f, -1.f, 0.f });
		void setViewYXZ(glm::vec3 position, glm::vec3 rotation);

		glm::vec3 getViewDirection(glm::vec3 position, glm::vec3 rotation);
		glm::vec3 getViewRight(glm::vec3 position, glm::vec3 rotation);
		glm::vec3 getViewUp(glm::vec3 position, glm::vec3 rotation);

		BasisAxis getTargetBasisAxis(glm::vec3 position, glm::vec3 target, glm::vec3 up);

		const glm::mat4& getProjection() const { return projectionMatrix; };
		const glm::mat4& getView() const { return viewMatrix; };

		glm::vec3 position;
		glm::vec3 up;
		glm::vec3 target;
	private:
		glm::mat4 projectionMatrix{ 1.f };
		glm::mat4 viewMatrix{ 1.f };
	};
}
