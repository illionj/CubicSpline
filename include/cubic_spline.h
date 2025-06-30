#ifndef TRAJECTORY_SMOOTH_CUBIC
#define TRAJECTORY_SMOOTH_CUBIC

#pragma once
#if defined _WIN32 || defined __CYGWIN__
  #ifdef TS_BUILD_DLL              // 编库时由 CMake 定义
    #define TS_EXPORT __declspec(dllexport)
  #else
    #define TS_EXPORT __declspec(dllimport)
  #endif
#elif defined __GNUC__ || defined __clang__
  #define TS_EXPORT __attribute__((visibility("default")))
#else
  #define TS_EXPORT                // 其他平台留空
#endif

#include <memory>



namespace trajectory_smooth {


struct TS_EXPORT ClosestState {
  double s;        // 投影弧长
  double x, y;     // 投影点坐标
  double heading;  // 轨迹切线方向
  double lat_err;  // 横向误差（带符号）
  double dist2;    // 距离平方
};


class TS_EXPORT CubicSpline2D {
 public:
  CubicSpline2D(const double *x, const double *y, unsigned long len);
  ~CubicSpline2D();

  CubicSpline2D(CubicSpline2D &&) noexcept;
  CubicSpline2D &operator=(CubicSpline2D &&) noexcept;

  CubicSpline2D(const CubicSpline2D &) = delete;
  CubicSpline2D &operator=(const CubicSpline2D &) = delete;

  ClosestState continuousClosest(double state_x, double state_y);
  void findTargetPoint(const ClosestState &cs, double ld, double &target_x, double &target_y);
  size_t getSamplePointsCount(double ds);
  void cubicSpline(double *out_x, double *out_y, double ds);

 private:
  struct CubicSpline2DImpl;  // 完全隐藏
  std::unique_ptr<CubicSpline2DImpl> p_;
};

}  // namespace trajectory_smooth
#endif