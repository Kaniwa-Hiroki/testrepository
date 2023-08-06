#ifndef TEST_HPP
#define TEST_HPP

struct InitialTargetDates {
    int initialYear;
    int initialMonth;
    int initialDay;
    bool isOneDay;
    bool isAfterDay;
    int durationDays;
    int targetYear;
    int targetMonth;
    int targetDay;
    int targetWeekDay;
};

#ifdef DEBUG

#include <fstream>
#include <stdexcept>

#define LOG_FILE_NAME "logfile.csv"

class Logger {
public:
    static void log(const std::string& message) {
        instance().file << message << std::endl;
    }

    static void logInitialTargetDates(const InitialTargetDates& initialTargetDates) {
        instance().file
            << initialTargetDates.initialYear << ','
            << initialTargetDates.initialMonth << ','
            << initialTargetDates.initialDay << ','
            << initialTargetDates.isOneDay << ','
            << initialTargetDates.isAfterDay << ','
            << initialTargetDates.durationDays << ','
            << initialTargetDates.targetYear << ','
            << initialTargetDates.targetMonth << ','
            << initialTargetDates.targetDay << ','
            << initialTargetDates.targetWeekDay << std::endl;
    }

private:
    Logger(const std::string& fileName)
    {
        file.open(fileName, std::ios::out);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open log file.");
        }
    }
    
    ~Logger()
    {
        file.close();
    }

    Logger(const Logger&) = delete;
    Logger& operator=(const Logger&) = delete;

    static Logger& instance()
    {
        static Logger logger(LOG_FILE_NAME);
        return logger;
    }

    std::ofstream file;
};

#define logITD(itd) Logger::logInitialTargetDates(itd)

#else // DEFINE

#define logITD(itd)

#endif // DEFINE

#endif // TEST_HPP
