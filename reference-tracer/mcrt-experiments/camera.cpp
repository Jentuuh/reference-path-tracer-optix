#include "camera.hpp"

namespace mcrt {
	Camera::Camera(glm::vec3 pos, glm::vec3 target, glm::vec3 up) :position{ pos }, target{ target }, up{ up } {}

}