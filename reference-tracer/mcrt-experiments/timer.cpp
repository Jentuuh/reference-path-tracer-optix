#include "timer.hpp"
#include <iostream>
#include <fstream>
namespace mcrt {


	Timer::Timer()
	{}

	void Timer::startTimedEvent(std::string eventName)
	{
		ongoingTimings[eventName] = high_resolution_clock::now();
	}


	void Timer::stopTimedEvent(std::string eventName)
	{
		if (ongoingTimings.count(eventName))
		{
			auto end = high_resolution_clock::now();
			auto duration = duration_cast<milliseconds>(end - ongoingTimings[eventName]);
			timedEventsInMs[eventName] = duration.count();
		}
		else {
			std::cout << "TIMER ERROR: Given key '" << eventName << "' is not an ongoing timed event. Cannot terminate it." << std::endl;
		}

	}

	void Timer::printSummary()
	{
		double totalTime = 0; 
		std::cout << std::endl;
		std::cout << "============= TIMER SUMMARY ==============" << std::endl;
		for (auto const& [key, val] : timedEventsInMs)
		{
			std::cout << "==========================================" << std::endl;
			std::cout << key << ':' << val << " ms." << std::endl;
			totalTime += val;
		}
		std::cout << "==========================================" << std::endl;
		std::cout << "Total time: " << totalTime << " ms." << std::endl;
		std::cout << std::endl;
	}

	void Timer::printSummaryToTXT()
	{
		std::ofstream newFile("../debug_output/timings.txt");
		for (auto const& [key, val] : timedEventsInMs)
		{
			newFile << val << std::endl;
		}
		newFile.close();
	}

}