#include <iostream>
#include <fstream>
#include <vector>
#include <charconv>
#include <algorithm>
#include <cmath>
#include <random>
#include <tuple>
#include <chrono>

std::vector<std::string_view> split(const std::string_view& str, const char s){
    std::vector<std::string_view> result;
    int i=0, j=0;
    for(; j<str.size(); j++){
        if(str[j] == s){
            result.emplace_back(str.substr(i, j-i));
            i = j+1;
        }
    }
    result.emplace_back(str.substr(i, j-i));
    return result;
}

template<typename T> using vec_t = std::vector<T>;
template<typename T> using mat_t = std::vector<std::vector<T>>;
template<typename T>
std::ostream& saveCsv(std::ostream& f, const mat_t<T>& scores){
    for(auto userScore: scores){
        for(size_t i=0; i<userScore.size(); i++){
            f << userScore[i];
            if(i+1!=userScore.size()){
                f << ',';
            } else{
                f << '\n';
            }
        }
    }
    return f;
}

class ItemCF{
private:
    const int mNSimilar, mNRecommend;
    mat_t<int> mData, mFullLines, mBlankLines;
    mat_t<int> mTrainSet, mTestSet;
    mat_t<double> mTrainSimilarMatrix;

public:
    ItemCF(const std::string& dataPath, int nSimilar, int nRecommend):
        mNSimilar(nSimilar), mNRecommend(nRecommend){
        //load csv
        std::ifstream f(dataPath, std::ios::in);
        if(!f.good()){
            throw std::runtime_error("open file '"+dataPath+"' error");
        }
        std::string line;
        while(getline(f, line)){
            auto csvItem = split(line, ',');
            vec_t<int> result;
            result.reserve(csvItem.size());
            for(auto& numStr: csvItem){
                int num;
                auto err = std::from_chars(numStr.data(), numStr.data()+numStr.size(), num);
                if(err.ec == std::errc::invalid_argument){
                    f.close();
                    throw std::runtime_error("csv parse error");
                }
                result.emplace_back(num);
            }
            mData.emplace_back(std::move(result));
            if(mData.back().size() != mData.front().size()){
                f.close();
                throw std::runtime_error("broken csv");
            }
        }
        mFullLines = mat_t<int>(mData.cbegin(), mData.cbegin() + 4100);
        mBlankLines = mat_t<int>(mData.cbegin() + 4100, mData.cend());
        f.close();
        std::tie(mTrainSet, mTestSet) = divideDataset(mFullLines, 75);
    }

    static std::tuple<mat_t<int>, mat_t<int>> divideDataset[[nodiscard]](const mat_t<int>& data, int pivot){
        if(pivot < 0 || pivot > 100){
            throw std::runtime_error("pivot should in [0, 100]");
        }
        const size_t nItems = data.front().size();
        const size_t nUsers = data.size();
        std::default_random_engine randomEngine;
        std::uniform_int_distribution<int> distribution(0, 99);
        mat_t<int> trainSet, testSet;
        trainSet.reserve(nUsers);
        testSet.reserve(nUsers);

        for(const auto& d: data){
            vec_t<int> trainLine, testLine;
            trainLine.reserve(nItems);
            testSet.reserve(nItems);
            for(auto score: d){
                bool belongTrain = distribution(randomEngine) < pivot;
                if(belongTrain){
                    trainLine.emplace_back(score);
                    testLine.emplace_back(0);
                } else {
                    trainLine.emplace_back(0);
                    testLine.emplace_back(score);
                }
            }
            trainSet.emplace_back(std::move(trainLine));
            testSet.emplace_back(std::move(testLine));
        }
        return {trainSet, testSet};
    }

    static mat_t<double> calculateSimilarity(const mat_t<int>& data){
        if(data.empty()){
            throw std::runtime_error("empty data");
        }
        size_t nItems = data.front().size();
        vec_t<int> popularItems(nItems, 0);
        mat_t<double> similarityMatrix(nItems);
        for(auto& m: similarityMatrix) m.resize(nItems, 0);
        for(auto& user: data){
            if(user.size() != nItems){
                throw std::runtime_error("invalid data");
            }
            std::vector<size_t> scoredFilms;
            for(size_t film=0; film < nItems; film++){
                if(user[film] != 0){
                    popularItems[film] += 1;
                    scoredFilms.emplace_back(film);
                }
            }
            for(auto film1: scoredFilms){
                for(auto film2: scoredFilms){
                    if(film1 != film2){
                        similarityMatrix[film1][film2] += 1;
                    }
                }
            }
        }
        for(int i=0; i<nItems; i++){
            for(int j=0; j<nItems; j++){
                if(similarityMatrix[i][j] != 0){
                    similarityMatrix[i][j] /= std::sqrt(double(popularItems[i]*popularItems[j]));
                }
            }
        }
        return similarityMatrix;
    }

    mat_t<std::pair<double, size_t>> getKSimilarMatrix [[nodiscard]](const mat_t<double>& similarMatrix) const{
        const size_t nFilms = similarMatrix.size();
        mat_t<std::pair<double, size_t>> result;
        result.reserve(nFilms);
        for(auto& simFilms: similarMatrix){
            std::vector<std::pair<double, size_t>> kSimFilms;
            kSimFilms.reserve(nFilms);
            for(size_t sfilm=0; sfilm<nFilms; sfilm++){
                kSimFilms.emplace_back(simFilms[sfilm], sfilm);
            }
            std::sort(kSimFilms.begin(), kSimFilms.end(), std::greater<>());
            kSimFilms.resize(mNSimilar);
            result.emplace_back(std::move(kSimFilms));
        }
        return result;
    }

    std::vector<std::pair<double, size_t>> recommend [[nodiscard]](
            const mat_t<std::pair<double, size_t>>& kSimilarMatrix, const vec_t<int>& userScore) const {
        const size_t nFilms = userScore.size();
        std::vector<size_t> watchedFilms;
        for(size_t film=0; film<nFilms; film++){
            if(userScore[film] != 0){
                watchedFilms.emplace_back(film);
            }
        }
        std::vector<std::pair<double, size_t>> relatedFilms;
        relatedFilms.reserve(nFilms);
        for(size_t film=0; film<nFilms; film++){
            relatedFilms.emplace_back(0, film);
        }
        for(auto film: watchedFilms){
            auto rating = userScore[film];
            const auto& kSimFilms = kSimilarMatrix[film];
            for(auto [w, sfilm]: kSimFilms){
                if(std::binary_search(watchedFilms.cbegin(), watchedFilms.cend(), sfilm))
                    continue;
                relatedFilms[sfilm].first += (w*rating);
            }
        }
        std::sort(relatedFilms.begin(), relatedFilms.end(), std::greater<>());
        relatedFilms.resize(mNRecommend);
        return relatedFilms;
    }

    double test[[nodiscard]]() const {
        const size_t nUsers = mTrainSet.size();
        size_t hit = 0;
        const auto similarityMatrix = calculateSimilarity(mTrainSet);
        const auto kSimilarMatrix = getKSimilarMatrix(similarityMatrix);
        for(size_t user=0; user<nUsers; user++){
            auto recommendFilms = recommend(kSimilarMatrix, mTrainSet[user]);
            for(auto [w, film]: recommendFilms){
                if(mTestSet[user][film] != 0){
                    hit += 1;
                }
            }
        }
        size_t recommendCount = mNRecommend * nUsers;
        double hitRate = double(hit) / double(recommendCount);
        return hitRate;
    }

    mat_t<int> scoring[[nodiscard]]() const {
        const size_t nUsers = mBlankLines.size();
        const auto similarityMatrix = calculateSimilarity(mFullLines);
        auto scoreOneItem =
        [&similarityMatrix](size_t itemId, const vec_t<std::pair<int, size_t>>& scoredItems)->double {
            const auto& similarItems = similarityMatrix[itemId];
            double scoreNum = 0, scoreDen = 0;
            for(auto [score, item]: scoredItems){
                double similarity = similarItems[item];
                scoreNum += similarity * score;
                scoreDen += similarity;
            }
            if(scoreDen == 0) return 0;
            return scoreNum / scoreDen;
        };
        mat_t<int> result;
        result.reserve(nUsers);
        for(size_t user=0; user<nUsers; user++){
            const auto& userScores = mBlankLines[user];
            vec_t<std::pair<int, size_t>> scoredItems;
            vec_t<int> predictedScores;
            predictedScores.reserve(userScores.size());
            for(size_t item=0; item<2700; item++){
                int score = userScores[item];
                if(score != 0){
                    scoredItems.emplace_back(score, item);
                }
                predictedScores.emplace_back(score);
            }
            for(size_t item=2700; item<userScores.size(); item++){
                double score = scoreOneItem(item, scoredItems);
                predictedScores.emplace_back(std::round(score));
            }
            result.emplace_back(std::move(predictedScores));
        }
        return result;
    }

    void fillBlank(const std::string& fileName){
        auto predictedScores = scoring();
        std::ofstream f(fileName, std::ios::out);
        if(!f.good()){
            throw std::runtime_error("open file '"+fileName+"' error");
        }
        saveCsv(f, mFullLines);
        saveCsv(f, predictedScores);
    }
};

int main(){
    auto t0 = std::chrono::system_clock::now();
    ItemCF itemCf("./col_matrix.csv", 20, 10);
    auto t1 = std::chrono::system_clock::now();
    std::cout << "hit-rate = " << itemCf.test() << '\n';
    auto t2 = std::chrono::system_clock::now();
    std::string outputFile = "test_prediction.csv";
    itemCf.fillBlank(outputFile);
    auto t3 = std::chrono::system_clock::now();
    std::cout << "output to '" << outputFile << "'\n";
    auto dt1 = double(std::chrono::duration_cast<std::chrono::milliseconds>(t1-t0).count()) / 1000;
    auto dt2 = double(std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count()) / 1000;
    auto dt3 = double(std::chrono::duration_cast<std::chrono::milliseconds>(t3-t2).count()) / 1000;
    std::cout << dt1 << "s for loading data\n" <<
                 dt2 << "s for testing hit rate\n" <<
                 dt3 << "s for scoring and saving\n";
    return 0;
}