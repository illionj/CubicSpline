#ifndef TRAJECTORY_SMOOTH_CUBIC
#define TRAJECTORY_SMOOTH_CUBIC

#pragma once
#if defined _WIN32 || defined __CYGWIN__
#ifdef TS_BUILD_DLL
#define TS_EXPORT __declspec(dllexport)
#else
#define TS_EXPORT __declspec(dllimport)
#endif
#elif defined __GNUC__ || defined __clang__
#define TS_EXPORT __attribute__((visibility("default")))
#else
#define TS_EXPORT
#endif

#include <memory>

namespace trajectory_smooth {

struct TS_EXPORT ClosestState {
  double s;        // 投影弧长
  double x, y;     // 投影点坐标
  double heading;  // 轨迹切线方向
  double lat_err;  // 横向误差(带符号)
  double dist2;    // 距离平方
};

/**
 * @brief  三次样条插值的对外接口
 */
class TS_EXPORT CubicSpline2D {
 public:
  /**
   * @brief  Construct a new Cubic Spline 2 D object
   * @param  x:
   * @param  y:
   * @param  len: x和y长度必须一样
   */
  CubicSpline2D(const double *x, const double *y, unsigned long len);
  ~CubicSpline2D();

  CubicSpline2D(CubicSpline2D &&) noexcept;
  CubicSpline2D &operator=(CubicSpline2D &&) noexcept;

  CubicSpline2D(const CubicSpline2D &) = delete;
  CubicSpline2D &operator=(const CubicSpline2D &) = delete;

  /**
   * @brief  给定某点,寻找当前插值曲线上的最近点,支持复用结果
   * @param  state_x:
   * @param  state_y:
   * @return ClosestState:
   */
  ClosestState continuousClosest(double state_x, double state_y);

  /**
   * @brief  根据弧长计算目标点
   * @param  s: 
   * @param  target_x: 
   * @param  target_y: 
   */
  void pointAt(double s, double &target_x, double &target_y);
  /**
   * @brief  专供pure_pursuit的接口,寻找目标点
   * @param  cs: 
   * @param  ld: 
   * @param  target_x: 
   * @param  target_y: 
   */
  void findTargetPoint(const ClosestState &cs, double ld, double &target_x, double &target_y);

  /**
   * @brief  Get the Sample Points Count object
   * @param  ds: 采样间距
   * @return size_t: 以此采样间距下的采样数
   */
  size_t getSamplePointsCount(double ds);

  /**
   * @brief  调用getSamplePointsCount获取ds下的最小存储空间,外部申请out_x,out_y内存.此接口生成三次样条采样数据
   * @param  out_x:
   * @param  out_y:
   * @param  ds:
   */
  void cubicSpline(double *out_x, double *out_y, double ds);

 private:
  struct CubicSpline2DImpl;  // 完全隐藏
  std::unique_ptr<CubicSpline2DImpl> p_;
};

}  // namespace trajectory_smooth
#endif