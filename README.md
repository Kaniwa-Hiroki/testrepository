#include <fstream>
#include <string>
#include <vector>
#include <sstream>

class FileReader {
public:
    FileReader(const std::string& filename, char delimiter) : inputFile(filename), delimiter(delimiter) {}

    bool openFile() {
        inputFile.open(filename);
        return inputFile.is_open();
    }

    void closeFile() {
        if (inputFile.is_open()) {
            inputFile.close();
        }
    }

    std::string getNextRow() {
        std::string line;
        if (std::getline(inputFile, line)) {
            return line;
        }
        return "";
    }

    std::vector<std::string> splitRow(const std::string& row) {
        std::vector<std::string> result;
        std::stringstream ss(row);
        std::string item;
        while (std::getline(ss, item, delimiter)) {
            result.push_back(item);
        }
        return result;
    }

private:
    std::string filename;
    std::ifstream inputFile;
    char delimiter;
};

int main() {
    FileReader reader("data.csv", ',');  // データが格納されたCSVファイル名を指定して FileReader オブジェクトを作成

    if (reader.openFile()) {
        std::string row;
        while ((row = reader.getNextRow()) != "") {
            // 取得した行データを利用する処理
            std::vector<std::string> columns = reader.splitRow(row);
            for (const auto& item : columns) {
                // データを利用する処理
            }
        }

        reader.closeFile();
    } else {
        // ファイルオープン失敗の処理
    }

    return 0;
}
