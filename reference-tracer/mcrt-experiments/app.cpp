#include "app.hpp"
#include "glm/gtx/string_cast.hpp"


namespace mcrt {
	App::App()
	{
		Camera camera = Camera{ glm::vec3{ -10.f,5.f,-12.f}, glm::vec3{0.f,0.f,0.f}, glm::vec3{0.0f, 1.0f, 0.0f } };
		scene = Scene{};
		loadScene();

		// something approximating the scale of the world, so the
		// camera knows how much to move for any given user interaction:
		const float worldScale = 10.f;

		window = std::make_unique<MCRTWindow>("Memory Coherent Ray Tracing", scene, camera, worldScale);
	}

	void App::run()
	{
		window->run();

		/* Renderer sample{model};
		sample.setCamera(camera);

		const glm::ivec2 fbSize(glm::ivec2(1200, 1024));
		sample.resize(fbSize);
		sample.render();
		std::vector<uint32_t> pixels(fbSize.x * fbSize.y);
		sample.downloadPixels(pixels.data());

		const std::string fileName = "mcrt_test.png";
		stbi_write_png(fileName.c_str(), fbSize.x, fbSize.y, 4,
		pixels.data(), fbSize.x * sizeof(uint32_t));*/
	}


	void App::loadScene()
	{
		//scene.loadModelFromOBJ("../models/sponza/sponza.obj");
		scene.loadModelFromOBJ("../models/crytek-sponza/sponza.obj");
		//scene.loadModelFromOBJ("../models/sponza/sponza.obj");
		std::cout << "Loaded scene: " << scene.amountVertices() << " vertices. Scene Max: " << glm::to_string(scene.maxCoord()) << " Scene Min: " << glm::to_string(scene.minCoord()) << std::endl;
		
		// Normalize scene to be contained within [0;1] in each dimension
		scene.normalize();

		// Create light sources
		scene.loadLights();
	}

}