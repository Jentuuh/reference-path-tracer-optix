#include "app.hpp"
#include "glm/gtx/string_cast.hpp"

#include <random>

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
		generateViewPositions();
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
		//scene.loadModelFromOBJ("../models/rungholt/tile_1.obj");
		//scene.loadModelFromOBJ("../models/rungholt/tile_2.obj");
		//scene.loadModelFromOBJ("../models/rungholt/tile_3.obj");
		//scene.loadModelFromOBJ("../models/rungholt/tile_4.obj");
		//scene.loadModelFromOBJ("../models/rungholt/tile_5.obj");
		//scene.loadModelFromOBJ("../models/rungholt/tile_6.obj");
		//scene.loadModelFromOBJ("../models/rungholt/tile_7.obj");
		//scene.loadModelFromOBJ("../models/rungholt/tile_8.obj");
		//scene.loadModelFromOBJ("../models/rungholt/tile_9.obj");
		//scene.loadModelFromOBJ("../models/rungholt/tile_10.obj");
		//scene.loadModelFromOBJ("../models/rungholt/tile_11.obj");
		//scene.loadModelFromOBJ("../models/rungholt/tile_12.obj");
		//scene.loadModelFromOBJ("../models/rungholt/tile_13.obj");
		//scene.loadModelFromOBJ("../models/rungholt/tile_14.obj");
		//scene.loadModelFromOBJ("../models/rungholt/tile_15.obj");


		//scene.loadModelFromOBJ("../models/san-miguel/san-miguel-low-poly.obj");

		std::cout << "Loaded scene: " << scene.amountVertices() << " vertices. Scene Max: " << glm::to_string(scene.maxCoord()) << " Scene Min: " << glm::to_string(scene.minCoord()) << std::endl;
		
		// Normalize scene to be contained within [0;1] in each dimension
		scene.normalize();
		std::cout << "Scene center: " << glm::to_string((scene.maxCoord() - scene.minCoord()) / 2.0f) << std::endl;

		// Create light sources
		scene.loadLights();
	}


	void App::generateViewPositions()
	{
		std::random_device rd;
		std::mt19937 gen(rd());

		//glm::vec3 min = scene.minCoord() + glm::vec3 {0.13f, 0.03f, 0.13f};
		//glm::vec3 max = scene.maxCoord() - glm::vec3{ 0.13f, 0.05f, 0.13f };

		glm::vec3 min = scene.minCoord() + glm::vec3{ 0.05f, 0.01f, 0.05f };
		glm::vec3 max = scene.maxCoord() + glm::vec3{ 0.05f, 0.1f, 0.05f };

		std::uniform_real_distribution<> distX(min.x, max.x);
		std::uniform_real_distribution<> distY(min.y, max.y);
		std::uniform_real_distribution<> distZ(min.z, max.z);

		for (int i = 0; i < 200; i++)
		{
			float randomX = distX(gen);
			float randomY = distY(gen);
			float randomZ = distZ(gen);

			window->viewPositions.push_back(glm::vec3{randomX, randomY, randomZ});
		}
	}

}