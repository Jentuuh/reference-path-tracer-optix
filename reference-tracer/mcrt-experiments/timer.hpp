#pragma once
#include <map>
#include <string>
#include <chrono>

using namespace std::chrono;

namespace mcrt {
	class Timer
	{
	public:
		Timer();

		void startTimedEvent(std::string eventName);
		void stopTimedEvent(std::string eventName);
		void printSummary();
		void printSummaryToTXT();

	private:
		std::map<std::string, time_point<high_resolution_clock>> ongoingTimings;
		std::map<std::string, double> timedEventsInMs;


	};

}

