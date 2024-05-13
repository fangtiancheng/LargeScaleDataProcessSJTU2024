#include <fstream>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>
#include <stdexcept>
#include <charconv>
#include <map>
#include <algorithm>
#include <chrono>

using scalar_t = double;

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

std::vector<std::pair<int, int>> load_csv[[maybe_unused]](const std::string& file_name){
    std::ifstream f(file_name, std::ios::in);
    if(!f.good()){
        throw std::runtime_error("open file '"+file_name+"' error");
    }
    std::cout << "loading... " << std::flush;
    std::string header, line;
    std::getline(f, header); // line 0
    std::vector<std::pair<int, int>> result;
    while(std::getline(f, line)){
        if(line.empty()) continue;
        auto line_split = split(line, ',');
        if(line_split.size() != 2){
            throw std::runtime_error("broken csv");
        }
        const auto& from_str = line_split[0];
        const auto& to_str = line_split[1];
        int from_node, to_node;
        auto err_from  = std::from_chars(from_str.data(), from_str.data()+from_str.size(), from_node);
        auto err_to = std::from_chars(to_str.data(), to_str.data()+to_str.size(), to_node);
        if(err_from.ec == std::errc::invalid_argument || err_to.ec == std::errc::invalid_argument){
            throw std::runtime_error("parse int error");
        }
        result.emplace_back(from_node, to_node);
    }
    std::cout << "done" << std::endl;
    return result;
}

void save_csv[[maybe_unused]](const std::string& file_name, const std::map<int, scalar_t>& csv){
    std::ofstream f(file_name, std::ios::out);
    if(!f.good()){
        throw std::runtime_error("open file '"+file_name+"' error");
    }
    std::cout << "saving... " << std::flush;
    f << "BideUd,PageRank_Value\n";
    for(auto& p: csv){
        f << p.first << ',' << p.second << '\n';
    }
    std::cout << "done" << std::endl;
}

void save_csv[[maybe_unused]](const std::string& file_name, const std::vector<std::pair<int, scalar_t>>& csv){
    std::ofstream f(file_name, std::ios::out);
    if(!f.good()){
        throw std::runtime_error("open file '"+file_name+"' error");
    }
    std::cout << "saving... " << std::flush;
    f << "BideUd,PageRank_Value\n";
    for(auto& p: csv){
        f << p.first << ',' << p.second << "\n";
    }
    std::cout << "done" << std::endl;
}

class PageRank{
protected:
    static const size_t N_ITER = 15;
    std::vector<std::vector<size_t>> mGraph;
    std::map<size_t , int> mIdxToName;
    std::map<int, size_t> mNameToIdx;
    size_t registerIdx[[nodiscard]](int name){
        auto iter = mNameToIdx.find(name);
        if(iter == mNameToIdx.end()){
            size_t idx = mNameToIdx.size();
            mNameToIdx.emplace(name, idx);
            mIdxToName.emplace(idx, name);
            mGraph.emplace_back();
            return idx;
        } else {
            return iter->second;
        }
    }
public:
    explicit PageRank(const std::vector<std::pair<int, int>>& originGraph): mGraph{}{
        std::cout << "preprocessing... " << std::flush;
        // make graph
        for(auto edge: originGraph){
            auto idxFrom = registerIdx(edge.first);
            auto idxTo = registerIdx(edge.second);
            mGraph[idxFrom].emplace_back(idxTo);
        }
        std::cout << "done" << std::endl;
    }

    std::map<int, scalar_t> calcPageRank[[nodiscard]](scalar_t beta) const {
        std::cout << "iterating... " << std::flush;
        const size_t n = mGraph.size();
        std::vector<scalar_t> pr(n, 1/(scalar_t)n);

        for(int i=0; i<N_ITER; i++){
            auto prevPr = pr;
            std::fill(pr.begin(), pr.end(), beta/(scalar_t)n);
            #pragma omp parallel for
            for(size_t idx=0; idx<n; idx++){
                const auto& toNodes = mGraph[idx];
                scalar_t delta_pr = (1-beta) * prevPr[idx] / (scalar_t)toNodes.size();
                for(size_t toNode: toNodes){
                    #pragma omp atomic
                    pr[toNode] += delta_pr;
                }
            }
        }
        std::map<int, scalar_t> result;
        for(size_t idx=0; idx<n; idx++){
            result.emplace(mIdxToName.at(idx), pr[idx]);
        }
        std::cout << "done" << std::endl;
        return result;
    }
    static std::vector<std::pair<int, scalar_t>> selectTopK(const std::map<int, scalar_t>& pr, size_t k){
        std::cout << "selecting top k... " << std::flush;
        std::vector<std::pair<int, scalar_t>> result(pr.cbegin(), pr.cend());
        using v_t = std::pair<int, scalar_t>;
        std::sort(result.begin(), result.end(), [](const v_t& x, const v_t& y){return y.second < x.second;});
        result.resize(k);
        std::cout << "done" << std::endl;
        return result;
    }
};

int main(){
    auto time0 = std::chrono::system_clock::now();
    auto graph = load_csv("./web_links.csv");
    PageRank pageRank(graph);
    auto pr = pageRank.calcPageRank(0.15);
    save_csv("test_prediction.csv", PageRank::selectTopK(pr, 1000));
    auto time1 = std::chrono::system_clock::now();
    std::cout << "takes "
        << (double)std::chrono::duration_cast<std::chrono::microseconds>(time1-time0).count() / 1e6
        << " seconds" << std::endl;
    return 0;
}