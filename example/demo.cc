#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "cubic_spline.h"

std::pair<std::vector<double>, std::vector<double>> loadXYfromCSV(const std::string &fname) {

  std::ifstream fin(fname);
  if (!fin) throw std::runtime_error("cannot open " + fname);

  std::string line;
  std::getline(fin, line);
  std::vector<double> xs, ys;
  xs.reserve(1024);
  ys.reserve(1024);

  while (std::getline(fin, line)) {
    if (line.empty()) continue;
    std::stringstream ss(line);
    std::string cell;

    if (!std::getline(ss, cell, ',')) break;
    xs.push_back(std::stod(cell));

    if (!std::getline(ss, cell, ',')) break;
    ys.push_back(std::stod(cell));
  }

  if (xs.size() != ys.size()) throw std::runtime_error("loadXYfromCSV: x/y size mismatch");

  return {xs, ys};
}

void saveXYtoCSV(const std::string &fname,
                 const std::vector<double> &xs,
                 const std::vector<double> &ys,
                 const std::string &header = "x,y") {
  if (xs.size() != ys.size()) throw std::runtime_error("saveXYtoCSV: x/y size mismatch");
  

  std::ofstream fout(fname, std::ios::trunc);
  if (!fout) throw std::runtime_error("cannot open " + fname);

  fout << header << '\n';

  fout << std::setprecision(15) << std::scientific;
  for (std::size_t i = 0; i < xs.size(); ++i) fout << xs[i] << ',' << ys[i] << '\n';

  if (!fout.good()) throw std::runtime_error("failed to write " + fname);
}

int main() {
  auto [csx, csy] = loadXYfromCSV("./scripts/xy.csv");
  double ds = 0.01;
  const unsigned long len = csx.size();

  std::unique_ptr<trajectory_smooth::CubicSpline2D> c2;
  c2 = std::make_unique<trajectory_smooth::CubicSpline2D>(csx.data(), csy.data(), len);


  auto start = std::chrono::high_resolution_clock::now();

  auto N = c2->getSamplePointsCount(ds);
  std::cout << "points count=" << N << '\n';
  std::cout<<"s_max="<<c2->getSmax()<<"\n";
  std::vector<double> cx(N, 0.0);
  std::vector<double> cy(N, 0.0);
  c2->cubicSpline(cx.data(), cy.data(), ds);
  auto end = std::chrono::high_resolution_clock::now();
  auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

  auto us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

  auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

  double fms = std::chrono::duration<double, std::milli>(end - start).count();

  std::cout << "millis: " << ms << " ms\n"
            << "micros: " << us << " μs\n"
            << "nanos : " << ns << " ns\n"
            << "float ms: " << fms << " ms\n";

  saveXYtoCSV("./scripts/xy_interpolation.csv", cx, cy);
  return 0;
}
