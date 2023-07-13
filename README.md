#include <fstream>
#include <iostream>
#include <ostream>

class Logger {
public:
    static Logger& getInstance() {
        static Logger instance;
        return instance;
    }

    void openLogFile(const std::string& filename) {
        logFile.open(filename, std::ios::app);
    }

    void closeLogFile() {
        logFile.close();
    }

    std::ostream& getLogFile() {
        return logFile;
    }

    template <typename T>
    void log(const T& message) {
        logFile << message << std::endl;
    }

private:
    Logger() {}
    ~Logger() {}

    std::ofstream logFile;
};

class CommunicationSender {
public:
    void sendData() {
        Logger::getInstance().log("Data sent.");

        // 送信処理
    }
};

class CommunicationReceiver {
public:
    void receiveData() {
        Logger::getInstance().log("Data received.");

        // 受信処理
    }
};

int main() {
    Logger::getInstance().openLogFile("communication.log");

    CommunicationSender sender;
    CommunicationReceiver receiver;

    sender.sendData();
    receiver.receiveData();

    Logger::getInstance().closeLogFile();

    return 0;
}
