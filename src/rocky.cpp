#include <iostream>
#include <vector>
#include <string>
#include <functional>
#include <random>
#include <chrono>
#include <iomanip>
#include <numeric>

#include <cassert>
#include <cmath>

#include "cblas.h"

#include "mpi.h"

// MPI variables
constexpr int ROOT_NODE_RANK = 0;
struct NodeInfo {
    int rank;

    std::string appendRank(const std::string& line)
    {
        return std::string("[rank: " + std::to_string(rank) + "]: ").append(line);
    }
};

// numerical auxillary types
struct point {
    double x{0}, y{0}, z{0};
};

// sampler: random distribution + random engine
struct sampler {
    // uniform[0,1]
    std::uniform_real_distribution<double> unif{0.0, 1.0};
    std::default_random_engine re{};
};
// create sampler via seed
sampler makeSampler(const std::size_t seed)
{
    sampler s;
    s.re.seed(seed);
    return s;
}

bool unitCircleContains(const point p)
{
    // contained in circle of radius 1?
    const auto x = p.x;
    const auto y = p.y;
    return (x*x + y*y < 1.0);

}

int main(int argc, char* argv[])
{
    // Initialize MPI
    // @TODO: wrap NodeInfo
    int ierr = MPI_Init(&argc, &argv);

    NodeInfo nodeInfo{};
    ierr = MPI_Comm_rank(MPI_COMM_WORLD, &nodeInfo.rank);

    // start timer
    const auto start = std::chrono::steady_clock::now();

    // seed sampler with node rank
    auto point_sampler{ makeSampler(nodeInfo.rank) };

    // compute local estimate of PI
    const std::size_t count_points{ 1 << 28 };
    std::size_t totalContains{ 0 };
    for(int i = 0; i < count_points; ++i) {
        // generate point uniform[0,1] x uniform[0,1]
        const auto px = point_sampler.unif(point_sampler.re);
        const auto py = point_sampler.unif(point_sampler.re);
        const point p{px, py, 0};

        //std::cout << nodeInfo.appendRank("points[" + std::to_string(i) + "] = " 
        //    + std::to_string(px) + "," + std::to_string(py) + ")\n"
        //);
        if(unitCircleContains(p)) ++totalContains;
    }
    double average = totalContains; 
    average /= count_points;

    // root node: gather estimate of pi from each node into buffer of doubles, report time elapsed
    int count_nodes{ 0 };
    MPI_Comm_size(MPI_COMM_WORLD, &count_nodes);
    std::vector<double> buffer = std::vector<double>(count_nodes);
    double piEstimate = 4 * average;
    std::cout << nodeInfo.appendRank("pi ~= " + std::to_string(piEstimate) + "\n");
    MPI_Gather(&piEstimate, 1, MPI_DOUBLE, buffer.data(), 1, MPI_DOUBLE, ROOT_NODE_RANK, MPI_COMM_WORLD);
    if(nodeInfo.rank == ROOT_NODE_RANK) {
        double aggregatePiEstimate{ 0 };
        for(std::size_t i = 0; i < count_nodes; ++i) {
            aggregatePiEstimate += buffer[i];
        }
        aggregatePiEstimate /= count_nodes;
        std::cout << nodeInfo.appendRank("(end) pi ~= " + std::to_string(aggregatePiEstimate) + "\n");
        const auto end = std::chrono::steady_clock::now();
        const std::chrono::duration<double> diff = end - start;
        std::cout << std::fixed << std::setprecision(9) << "elapsed: " << diff << "\n";
    }

    // Finalize MPI
    MPI_Finalize();

    return 0;
}