#include "area_light.hpp"


namespace mcrt {
	AreaLight::AreaLight(bool twoSided, LightData initData) : twoSided{twoSided}, lightProps{initData}{}
}