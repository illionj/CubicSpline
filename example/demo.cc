#include <Eigen/Dense>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <list>
#include <type_traits>

#include "Eigen/Core"  // IWYU pragma: export
#include "cubic_spline.h"
std::pair<Eigen::VectorXd, Eigen::VectorXd> loadXYfromCSV(const std::string &fname) {
  std::ifstream fin(fname);
  if (!fin) throw std::runtime_error("cannot open " + fname);

  std::string line;
  std::getline(fin, line);  // ① 读掉表头

  std::vector<double> xs, ys;
  xs.reserve(1024);
  ys.reserve(1024);

  while (std::getline(fin, line)) {
    if (line.empty()) continue;  // 忽略空行
    std::stringstream ss(line);
    std::string cell;

    // ② 解析第一列 X
    if (!std::getline(ss, cell, ',')) break;
    xs.push_back(std::stod(cell));

    // ③ 解析第二列 Y
    if (!std::getline(ss, cell, ',')) break;
    ys.push_back(std::stod(cell));
  }

  if (xs.size() != ys.size()) throw std::runtime_error("loadXYfromCSV: x/y size mismatch");

  // ④ 转为 Eigen 向量（零拷贝构造 + move）
  Eigen::VectorXd x = Eigen::Map<Eigen::VectorXd>(xs.data(), xs.size());
  Eigen::VectorXd y = Eigen::Map<Eigen::VectorXd>(ys.data(), ys.size());
  // std::cout << "cx_uniform =\n" << x << "\n";
  // exit(0);
  return {std::move(x), std::move(y)};
}

void saveXYasCSV(const std::string &fname, const Eigen::VectorXd &x, const Eigen::VectorXd &y) {
  if (x.size() != y.size()) {
    throw std::runtime_error("saveXYasCSV: x/y size mismatch");
  }

  std::ofstream fout(fname);
  if (!fout) throw std::runtime_error("cannot open " + fname);

  fout << "X" << '\t' << "Y" << '\n';
  // for (Eigen::Index i = 0; i < x.size(); ++i) {
  //   fout << x(i) << ',' << y(i) << '\n';
  // }

  Eigen::IOFormat csvFmt(Eigen::FullPrecision, Eigen::DontAlignCols, "\t", "\n");
  for (Eigen::Index i = 0; i < x.size(); ++i) {
    fout << Eigen::Vector2d{x(i), y(i)}.transpose().format(csvFmt) << '\n';
  }
}

int main() {
  //   x = [0.0, 30.0, 6.0, 20.0, 35.0,10.0, 0.0, 0.0]
  // y = [0.0, 0.0, 20.0, 35.0, 20.0,30.0, 5.0, 0.0]
  // const unsigned long len = 8;
  // double x[len] = {0.0, 30.0, 6.0, 20.0, 35.0, 10.0, 0.0, 0.0};
  // double y[len] = {0.0, 0.0, 20.0, 35.0, 20.0, 30.0, 5.0, 0.0};
  // double out_x[len];
  // double out_y[len];
  double ds = 0.01;

  // Eigen::Map<Eigen::VectorXd> ex(x, len);
  // Eigen::Map<Eigen::VectorXd> ey(y, len);

  auto [ex, ey] = loadXYfromCSV("trace.csv");
  
  auto start = std::chrono::high_resolution_clock::now();
  auto x=ex.data();
  auto y=ey.data();
  auto len=ex.size();
  auto c2=trajectory_smooth::CubicSpline2D(x,y,len);
  auto N=c2.getSamplePointsCount(ds);
  // std::cout<<"N:"<<N<<'\n';
  std::vector<double> out_x(N,0.0);
  std::vector<double> out_y(N,0.0);
  c2.cubicSpline( out_x.data(), out_y.data(),ds);

  auto end = std::chrono::high_resolution_clock::now();

  const Eigen::Map<const Eigen::VectorXd> cx_uniform(out_x.data(), N);
  const Eigen::Map<const Eigen::VectorXd> cy_uniform(out_y.data(), N);
  // xy2s(ex, ey, s);
  // auto cx = CubicSpline1D(s, ex);
  // auto cy = CubicSpline1D(s, ey);

  // double stop = s(s.size() - 1);
  // std::ptrdiff_t N = static_cast<std::ptrdiff_t>(std::floor(stop / ds));
  // Eigen::VectorXd s_uniform;
  // if (N != 0) {
  //   s_uniform = Eigen::VectorXd::LinSpaced(N, 0.0, (N - 1) * ds);
  // }

  // auto cx_uniform = cx.sample(s_uniform);
  // auto cy_uniform = cy.sample(s_uniform);

  auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

  auto us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

  auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

  double fms = std::chrono::duration<double, std::milli>(end - start).count();

  std::cout  << "millis: " << ms << " ms\n"
            << "micros: " << us << " μs\n"
            << "nanos : " << ns << " ns\n"
            << "float ms: " << fms << " ms\n";

  saveXYasCSV("xy.csv", cx_uniform, cy_uniform);

  // std::cout << "cx_uniform =\n" << cx_uniform << "\n";
  // std::cout << "s_uniform =\n" << s_uniform << "\n";
  // cubicSpline(x, y, len,out_x,out_y,ds);
  return 0;
}
