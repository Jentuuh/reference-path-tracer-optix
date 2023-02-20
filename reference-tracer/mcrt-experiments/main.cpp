#include "app.hpp"

// std
#include <iostream>


namespace mcrt {
   

	// Main entry point
	extern "C" int main(int argc, char* argv[]) {
        try {

            App app{};
            app.run();
        }
        catch (std::runtime_error& e) {
            std::cout << "FATAL ERROR: " << e.what() << std::endl;
            exit(1);
        }
        return 0;
	}
}
