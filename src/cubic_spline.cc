
#include "cubic_spline.h"

#include <Eigen/Dense>
#include <Eigen/QR>
#include <algorithm>  // std::lower_bound
#include <cstddef>
#include <unsupported/Eigen/CXX11/Tensor>
#include <unsupported/Eigen/Polynomials>

#include "Eigen/Core"  // IWYU pragma: export
namespace trajectory_smooth {

class CubicSpline1D {
 public:
  CubicSpline1D(const Eigen::VectorXd &s, const Eigen::VectorXd &x)
      : s_(s),
        a_(x),
        nx_(static_cast<size_t>(std::min(s_.size(), a_.size()))) {
    // s_.conservativeResize(nx_);
    // a_.conservativeResize(nx_);

    h_ = s_.segment(1, nx_ - 1) - s_.segment(0, nx_ - 1);
    // std::cout<<"h="<<h_<<"\n";

    buildSystem();

    // c_.resize(nx_ - 1);
    // c_ = A_.triangularView<Eigen::UnitLower>().solve(B_);
    // c_ = A_.colPivHouseholderQr().solve(B_);

    // Thomas 算法
    Eigen::VectorXd lower(nx_ - 1), diag(nx_), upper(nx_ - 1), rhs(nx_);

    diag.setOnes();
    rhs.setZero();

    for (size_t i = 1; i < nx_ - 1; ++i) {
      lower(i - 1) = h_(i - 1);
      diag(i) = 2.0 * (h_(i - 1) + h_(i));
      upper(i) = h_(i);
      rhs(i) = 3.0 * ((a_(i + 1) - a_(i)) / h_(i) - (a_(i) - a_(i - 1)) / h_(i - 1));
    }
    upper(nx_ - 2) = h_(nx_ - 2);

    for (size_t i = 1; i < nx_; ++i) {
      double w = lower(i - 1) / diag(i - 1);
      diag(i) -= w * upper(i - 1);
      rhs(i) -= w * rhs(i - 1);
    }

    c_.resize(nx_);
    c_(nx_ - 1) = rhs(nx_ - 1) / diag(nx_ - 1);

    for (int i = int(nx_) - 2; i >= 0; --i) {
      c_(i) = (rhs(i) - upper(i) * c_(i + 1)) / diag(i);
    }

    b_.resize(nx_ - 1);
    d_.resize(nx_ - 1);
    for (size_t i = 0; i < nx_ - 1; ++i) {
      d_(i) = (c_(i + 1) - c_(i)) / (3.0 * h_(i));
      b_(i) = (a_(i + 1) - a_(i)) / h_(i) - (h_(i) / 3.0) * (2.0 * c_(i) + c_(i + 1));
    }
  }

  size_t findInterval(double s_query) const {
    return static_cast<size_t>(std::lower_bound(s_.data(), s_.data() + nx_, s_query) - s_.data()) - 1;
  }

  double evaluate(double s_query) const {
    size_t i = findInterval(s_query);
    double dx = s_query - s_(i);
    return a_(i) + b_(i) * dx + c_(i) * dx * dx + d_(i) * dx * dx * dx;
  }

  double firstDerivative(double s_query) const {
    size_t i = findInterval(s_query);
    double dx = s_query - s_(i);
    return b_(i) + 2.0 * c_(i) * dx + 3.0 * d_(i) * dx * dx;
  }

  double secondDerivative(double s_query) const {
    size_t i = findInterval(s_query);
    double dx = s_query - s_(i);
    return 2.0 * c_(i) + 6.0 * d_(i) * dx;
  }

  Eigen::VectorXd sample(const Eigen::VectorXd &s_uniform) const {
    const Eigen::Index M = s_uniform.size();
    Eigen::VectorXd y(M);

    size_t i = 0;
    for (Eigen::Index k = 0; k < M; ++k) {
      const double s_val = s_uniform(k);

      while (i + 1 < nx_ && s_val > s_(i + 1)) ++i;

      const double dx = s_val - s_(i);
      y(k) = a_(i) + b_(i) * dx + c_(i) * dx * dx + d_(i) * dx * dx * dx;
    }
    return y;
  }

  void sample(const Eigen::VectorXd &s_uniform, double *out_ptr) const {
    const Eigen::Index M = s_uniform.size();
    // Eigen::Map<Eigen::VectorXd, Eigen::Unaligned> y(out_ptr, M);

    std::size_t i = 0;
    for (Eigen::Index k = 0; k < M; ++k) {
      const double s_val = s_uniform(k);

      while (i + 1 < nx_ && s_val > s_(i + 1)) ++i;

      const double dx = s_val - s_(i);

      auto ans = a_(i) + b_(i) * dx + c_(i) * dx * dx + d_(i) * dx * dx * dx;

      out_ptr[k] = ans;
    }
  }

  void sample(const Eigen::VectorXd &s_uniform,
              double *out_ptr,   // f(s)
              double *out_ptr1,  // f'(s)
              double *out_ptr2)  // f''(s)
      const {
    const Eigen::Index M = s_uniform.size();
    std::size_t i = 0;

    for (Eigen::Index k = 0; k < M; ++k) {
      const double s_val = s_uniform(k);

      while (i + 1 < nx_ && s_val > s_(i + 1)) ++i;

      const double dx = s_val - s_(i);
      const double dx2 = dx * dx;

      const double f = a_(i) + b_(i) * dx + c_(i) * dx2 + d_(i) * dx2 * dx;

      const double fp = b_(i) + 2.0 * c_(i) * dx + 3.0 * d_(i) * dx2;

      const double fpp = 2.0 * c_(i) + 6.0 * d_(i) * dx;

      out_ptr[k] = f;
      out_ptr1[k] = fp;
      out_ptr2[k] = fpp;
    }
  }

  double operator()(double s_query) const {
    size_t i = findInterval(s_query);
    double dx = s_query - s_(i);
    return a_(i) + b_(i) * dx + c_(i) * dx * dx + d_(i) * dx * dx * dx;
  }

  /// 段数 = 节点数 − 1
  size_t segmentCount() const { return nx_ - 1; }

  /// 第 i 段左端弧长（= 节点 s_i）
  double knot(size_t i) const { return s_(i); }

  /// 第 i 段弧长 h_i  (= s_{i+1} − s_i)
  double segmentLength(size_t i) const { return h_(i); }

  /// 第 i 段四个系数 a,b,c,d  →  [a0,a1,a2,a3]
  Eigen::Vector4d coeff(size_t i) const { return {a_(i), b_(i), c_(i), d_(i)}; }

 private:
  void buildSystem() {
    A_.setZero(nx_, nx_);
    B_.setZero(nx_);

    // 自然边界条件：c0 = cn = 0
    A_(0, 0) = 1.0;
    A_(nx_ - 1, nx_ - 1) = 1.0;

    // 内点：三对角
    for (size_t i = 1; i < nx_ - 1; ++i) {
      A_(i, i - 1) = h_(i - 1);
      A_(i, i) = 2.0 * (h_(i - 1) + h_(i));
      A_(i, i + 1) = h_(i);

      B_(i) = 3.0 * ((a_(i + 1) - a_(i)) / h_(i) - (a_(i) - a_(i - 1)) / h_(i - 1));
    }
  }

 private:
  const Eigen::VectorXd &s_;
  const Eigen::VectorXd &a_;
  Eigen::VectorXd b_, c_, d_;
  Eigen::VectorXd h_;

  Eigen::MatrixXd A_;
  Eigen::VectorXd B_;

  size_t nx_{0};
};

Eigen::VectorXd xy2s(const Eigen::VectorXd &x, const Eigen::VectorXd &y) {
  const Eigen::Index n = x.size();
  if (n < 2)  // 0 点或 1 点时，弧长恒为 0
    return Eigen::VectorXd::Zero(n);

  /* ------- 逐段长度 -------- */
  Eigen::ArrayXd dx = x.tail(n - 1) - x.head(n - 1);
  Eigen::ArrayXd dy = y.tail(n - 1) - y.head(n - 1);
  Eigen::ArrayXd seg = (dx.square() + dy.square()).sqrt();  // 每段弧长

  /* ------- 累加得到弧长参数 s -------- */
  Eigen::VectorXd s(n);
  s(0) = 0.0;

#if defined(EIGEN_CXX11_TENSOR_MODULE)     // <-- 只要打开 unsupported/Tensor 就有 cumsum
  s.tail(n - 1) = seg.cumsum(0).matrix();  // 0 轴（唯一轴）做前缀和 :contentReference[oaicite:0]{index=0}
#else
  // 低版本或不想启用 Tensor 时：手写一行循环，但仍然只操作 Eigen 对象
  s.tail(n - 1) = seg.matrix();
  for (Eigen::Index i = 1; i < n - 1; ++i) s(i + 1) += s(i);  // 等价于 inclusive_scan
#endif

  return s;
}

inline double curvature(double dx, double dy, double ddx, double ddy) noexcept {
  const double num = std::abs(dx * ddy - dy * ddx);
  const double denom = std::pow(dx * dx + dy * dy, 1.5);
  return denom > 0.0 ? num / denom : 0.0;  // 防止 0 除
}

inline double curvatureAt(double s, const CubicSpline1D &sx, const CubicSpline1D &sy) {
  const double dx = sx.firstDerivative(s);
  const double dy = sy.firstDerivative(s);
  const double ddx = sx.secondDerivative(s);
  const double ddy = sy.secondDerivative(s);
  return curvature(dx, dy, ddx, ddy);  // 调用上面的标量公式
}

void computeCurvature(const Eigen::VectorXd &s_uniform,
                      const CubicSpline1D &spline_x,
                      const CubicSpline1D &spline_y,
                      double *out_kappa) {
  const Eigen::Index M = s_uniform.size();
  for (Eigen::Index k = 0; k < M; ++k) {
    double s = s_uniform(k);
    double dx = spline_x.firstDerivative(s);
    double dy = spline_y.firstDerivative(s);
    double ddx = spline_x.secondDerivative(s);
    double ddy = spline_y.secondDerivative(s);
    out_kappa[k] = curvature(dx, dy, ddx, ddy);
  }
}

struct CubicSpline2D::CubicSpline2DImpl {
 public:
  // const Eigen::Map<const Eigen::VectorXd> ex;
  // const Eigen::Map<const Eigen::VectorXd> ey;
  Eigen::VectorXd ex;
  Eigen::VectorXd ey;
  Eigen::VectorXd es;
  CubicSpline1D cx;
  CubicSpline1D cy;
  std::optional<double> s_prev_opt;

  CubicSpline2DImpl(const double *x, const double *y, unsigned long len)
      : ex(Eigen::Map<const Eigen::VectorXd>(x, len)),
        ey(Eigen::Map<const Eigen::VectorXd>(y, len)),
        es(xy2s(ex, ey)),
        cx(es, ex),
        cy(es, ey),
        s_prev_opt(std::nullopt) {}

  size_t getSamplePointsCount(double ds) {
    double stop = es(es.size() - 1);
    // std::ptrdiff
    return static_cast<std::size_t>(std::floor(stop / ds));
  }

  Eigen::VectorXd getSamplePoints(double ds) {
    double stop = es(es.size() - 1);
    // std::ptrdiff
    auto N = getSamplePointsCount(ds);
    return Eigen::VectorXd::LinSpaced(N, 0.0, (N - 1) * ds);
  }

  void cubicSpline(double *out_x, double *out_y, double ds) {
    auto s_uniform = getSamplePoints(ds);
    cx.sample(s_uniform, out_x);
    cy.sample(s_uniform, out_y);
  }

  void computeCurvature(double *out_kappa, double ds) {
    auto s_uniform = getSamplePoints(ds);
    const Eigen::Index M = s_uniform.size();
    for (Eigen::Index k = 0; k < M; ++k) {
      out_kappa[k] = curvatureAt(s_uniform[k], cx, cy);
    }
  }

  std::pair<double, double> pointAt(double s) { return std::pair<double, double>(cx.evaluate(s), cy.evaluate(s)); }

  // ----------------------------------------------
  // 查询某 s 处的斜率   （dy/dx 形式）
  // ----------------------------------------------
  double slopeAt(double s) const {
    double dx = cx.firstDerivative(s);
    double dy = cy.firstDerivative(s);

    // 防止分母过小导致数值不稳
    constexpr double kEps = 1e-12;
    if (std::abs(dx) < kEps) {
      return (dy > 0.0 ? std::numeric_limits<double>::infinity() : -std::numeric_limits<double>::infinity());
    }
    return dy / dx;  // 斜率 dy/dx
    // return std::atan2(dy, dx);   // 航向角（弧度）
  }

  // ClosestState continuousClosest(double x0,
  //                                double y0,
  //                                double s_max
  //                                ) {
  //   // ---- (1) warm start + 稀疏采样 --------------------------------------------------
  //   const double safety = 1e-4;
  //   double best_s = 0.0;
  //   auto dist2_at = [&](double s) {
  //     double dx = cx.evaluate(s) - x0;
  //     double dy = cy.evaluate(s) - y0;
  //     return dx * dx + dy * dy;
  //   };

  //   double best_d2 = std::numeric_limits<double>::infinity();
  //   if (0) {
  //     best_s = std::clamp(s_prev_opt.value(), safety, s_max - safety);
  //     best_d2 = dist2_at(best_s);
  //   } else {
  //     const int coarse_N = 256;
  //     for (int i = 0; i <= coarse_N; ++i) {
  //       double s_c = s_max * i / coarse_N;
  //       double d2 = dist2_at(s_c);
  //       if (d2 < best_d2) {
  //         best_d2 = d2;
  //         best_s = s_c;
  //       }
  //     }
  //   }
  //   std::cout.setf(std::ios::fixed);
  //   // std::cout << std::setprecision(8);

  //   // 打印 (x0,y0) 到样条每个节点 (s_i) 的距离平方
  //   for (size_t i = 0; i < cx.segmentCount() + 1; ++i) {
  //     double dx = cx.evaluate(cx.knot(i)) - x0;
  //     double dy = cy.evaluate(cx.knot(i)) - y0;
  //     std::cout << "node[" << i << "]  s=" << cx.knot(i) << "  d2=" << dx * dx + dy * dy << '\n';
  //   }
  //   // ---- (2) 对每段解析求垂足 (D'(u)=0 的五次多项式) -------------------------------
  //   const size_t Nseg = cx.segmentCount();
  //   for (size_t k = 0; k < Nseg; ++k) {
  //     const double s_left = cx.knot(k);
  //     const double h = cx.segmentLength(k);
  //     // std::cout<<"s_left\n";

  //     // 取系数： x(s) = a0 + a1 u + a2 u² + a3 u³,   u ∈ [0,h]
  //     const auto ax = cx.coeff(k);
  //     const auto ay = cy.coeff(k);

  //     // 平移到 (x0,y0) 并展开五次多项式系数
  //     Eigen::Matrix<double, 6, 1> c;
  //     c.setZero();
  //     const double a0 = ax[0] - x0, a1 = ax[1], a2 = ax[2], a3 = ax[3];
  //     const double b0 = ay[0] - y0, b1 = ay[1], b2 = ay[2], b3 = ay[3];

  //     c[5] = 10 * (a3 * a3 + b3 * b3);
  //     c[4] = 8 * (a2 * a3 + b2 * b3);
  //     c[3] = 6 * (a1 * a3 + a2 * a2 + b1 * b3 + b2 * b2);
  //     c[2] = 4 * (a0 * a3 + a1 * a2 + b0 * b3 + b1 * b2);
  //     c[1] = 2 * (a0 * a2 + a1 * a1 + b0 * b2 + b1 * b1);
  //     c[0] = 2 * (a0 * a1 + b0 * b1);

  //     // -- 求根
  //     Eigen::PolynomialSolver<double, Eigen::Dynamic> solver;
  //     solver.compute(c);
  //     const auto &roots = solver.roots();

  //     auto check = [&](double u) {
  //       if (u < 0.0 || u > h) return;
  //       const double uu = u * u, u3 = uu * u;
  //       const double x = ax[0] + ax[1] * u + ax[2] * uu + ax[3] * u3;
  //       const double y = ay[0] + ay[1] * u + ay[2] * uu + ay[3] * u3;
  //       const double d2 = (x - x0) * (x - x0) + (y - y0) * (y - y0);
  //       if (d2 < best_d2) {
  //         best_d2 = d2;
  //         best_s = s_left + u;
  //       }
  //     };

  //     for (int i = 0; i < roots.size(); ++i)
  //       if (roots[i].imag() == 0.0) check(roots[i].real());

  //     check(0.0);  // 左端
  //     check(h);    // 右端
  //   }

  //   // ---- (3) 汇总 ---------------------------------------------------------------
  //   ClosestState st;
  //   st.s = best_s;
  //   st.x = cx.evaluate(best_s);
  //   st.y = cy.evaluate(best_s);
  //   st.dist2 = best_d2;
  //   s_prev_opt = best_s;  // 下帧 warm-start
  //   return st;
  // }

  // struct ClosestState {
  //   double s;      // 匹配的弧长
  //   double x, y;   // 曲线上坐标
  //   double dist2;  // 距离平方
  // };
  ClosestState continuousClosest(double x0, double y0, double s_max, int max_iter ) {
    const double safety = 1e-4 * s_max;  // 距端点 ≥ 0.01 %
    double s = 0.0;

    // if (s_prev_opt.has_value()) {
      if (0) {
      s = std::clamp(s_prev_opt.value(), safety, s_max - safety);
    } else {
      /* 在稀疏采样里粗找最近点；采样多一点避免落到端点   */
      const int coarse_N = 256;
      double best_d2 = std::numeric_limits<double>::infinity();
      for (int i = 0; i <= coarse_N; ++i) {
        double s_c = s_max * i / coarse_N;
        double dx = cx.evaluate(s_c) - x0;
        double dy = cy.evaluate(s_c) - y0;
        double d2 = dx * dx + dy * dy;
        if (d2 < best_d2) {
          best_d2 = d2;
          s = s_c;
        }
      }
      s = std::clamp(s, safety, s_max - safety);
    }

    /* ---------- (2)  迭代求最近点 ------------------------------ */
    constexpr double eps_s = 1e-10;     // s 收敛阈
    constexpr double eps_G = 1e-10;     // |G| 收敛阈
    constexpr double eps_Gp = 1e-12;    // “二阶导退化” 判据
    constexpr double step_limit = 0.1;  // trust-region

    for (int it = 0; it < max_iter; ++it) {
      double xs = cx.evaluate(s);
      double ys = cy.evaluate(s);
      double dxs = cx.firstDerivative(s);
      double dys = cy.firstDerivative(s);
      double ddxs = cx.secondDerivative(s);
      double ddys = cy.secondDerivative(s);

      double rx = xs - x0;
      double ry = ys - y0;

      double G = rx * dxs + ry * dys;
      double Gp = dxs * dxs + dys * dys + rx * ddxs + ry * ddys;

      /* ---- 收敛判据：梯度近零 且 二阶导不再显著 --------- */
      if (std::abs(G) < eps_G || std::abs(Gp) < eps_Gp) break;

      /* ---- 处理导数退化：|r′|² 太小 → 小步侧移 ---------- */
      double grad2 = dxs * dxs + dys * dys;
      if (grad2 < 1e-8) {
        double h = 0.01 * s_max;  // 1 % 的小步
        double s1 = std::min(s + h, s_max - safety);
        double s2 = std::max(s - h, safety);

        auto dist2 = [&](double sp) {
          double dx = cx.evaluate(sp) - x0;
          double dy = cy.evaluate(sp) - y0;
          return dx * dx + dy * dy;
        };
        s = (dist2(s1) < dist2(s2)) ? s1 : s2;
        continue;  // 重新迭代
      }

      /* ---- Newton + trust-region  --------------------------- */
      double delta;
      if (std::abs(Gp) >= eps_Gp) {
        delta = -G / Gp;
        delta = std::clamp(delta, -step_limit, step_limit);
      } else {
        /* 二阶导≈0：退化成梯度下降 */
        delta = -G / (std::sqrt(grad2) + 1e-12) * 0.05;  // 再×5 %
      }

      double s_new = std::clamp(s + delta, safety, s_max - safety);
      if (std::abs(s_new - s) < eps_s) {
        s = s_new;
        break;
      }

      s = s_new;
    }

    /* ---------- (3)  汇总结果 --------------------------------- */
    ClosestState st;
    st.s = s;
    s_prev_opt = s;  // 供下一次 warm-start
    st.x = cx.evaluate(s);
    st.y = cy.evaluate(s);
    double dx = st.x - x0;
    double dy = st.y - y0;
    st.dist2 = dx * dx + dy * dy;
    double dxs = cx.firstDerivative(s);
    double dys = cy.firstDerivative(s);
    double tang_mag = std::hypot(dxs, dys);

    // 横向误差（带符号）
    st.lat_err = (-dx * dys + dy * dxs) / (tang_mag + 1e-12);
    return st;
  }

  // ClosestState continuousClosest(double x0, double y0, double s_max, int max_iter = 6) {
  //   /* ---------- (1) 初值 ------------------------------ */
  //   double s = 0.0;
  //   std::cout << "x0=" << x0 << "  y0=" << y0 << '\n';
  //   if (s_prev_opt.has_value()) {
  //     s = std::clamp(s_prev_opt.value(), 0.0, s_max);

  //   } else {
  //     // ▸ 第一次没有 warm-start：在稀疏采样里粗找最近点
  //     constexpr int coarse_N = 64;  // 64 个点足够
  //     double best_d2 = std::numeric_limits<double>::max();
  //     for (int i = 0; i <= coarse_N; ++i) {
  //       double s_c = s_max * i / coarse_N;
  //       double rx = cx.evaluate(s_c) - x0;
  //       double ry = cy.evaluate(s_c) - y0;
  //       double dist2 = rx * rx + ry * ry;
  //       if (dist2 < best_d2) {
  //         best_d2 = dist2;
  //         s = s_c;
  //       }
  //     }
  //   }

  //   const double eps = 1e-8;  // 收敛阈值
  //   for (int it = 0; it < max_iter; ++it) {
  //     const double xs = cx.evaluate(s);
  //     const double ys = cy.evaluate(s);
  //     const double dxs = cx.firstDerivative(s);
  //     const double dys = cy.firstDerivative(s);
  //     const double ddxs = cx.secondDerivative(s);
  //     const double ddys = cy.secondDerivative(s);

  //     const double rx = xs - x0;
  //     const double ry = ys - y0;

  //     const double G = rx * dxs + ry * dys;                             // F′
  //     const double Gp = dxs * dxs + dys * dys + rx * ddxs + ry * ddys;  // F″
  //     std::cout << "G=" << G << "  GP=" << Gp << '\n';
  //     const double epsG = 1e-10;      // 对 G 的收敛要求
  //     const double epsGp = 1e-12;     // 对 G' 的退化判断
  //     const double step_limit = 0.5;  // 牛顿步长上限（防飞）
  //     // 1) 只有同时满足 |G| 和 |Gp| 都小才算收敛
  //     if (std::abs(G) < epsG && std::abs(Gp) < epsGp) break;

  //     double delta;  // 下一步的尝试位移
  //     if (std::abs(Gp) >= epsGp) {
  //       // 正常牛顿
  //       delta = -G / Gp;
  //       // 2) 给牛顿步加个 trust-region
  //       delta = std::clamp(delta, -step_limit, step_limit);
  //     } else {
  //       // 二阶导退化，退回到梯度下降
  //       double grad_norm = std::hypot(dxs, dys) + 1e-12;
  //       delta = -G / grad_norm * 0.05;  // 小步走
  //     }

  //     double s_new = std::clamp(s + delta, 0.0, s_max);

  //     if (std::abs(s_new - s) < eps) {
  //       s = s_new;
  //       break;
  //     }
  //     s = s_new;

  //     // if (std::abs(Gp) < 1e-12) break;                                  // 近直线段：提前结束

  //     // double s_new = s - G / Gp;              // Newton 步
  //     // s_new = std::clamp(s_new, 0.0, s_max);  // 保证落在合法区间

  //     std::cout <<"i="<<it<< "  s_new=" << s_new << "  s_max=" << s_max <<"rx="<<rx<<" ry="<<ry<< '\n';
  //     // if (std::abs(s_new - s) < eps) {  // 收敛判据
  //     //   s = s_new;
  //     //   break;
  //     // }
  //     // s = s_new;  // 继续迭代
  //   }

  //   /* ---------- (3) 汇总结果 -------------------------- */
  //   ClosestState st;
  //   st.s = s;
  //   s_prev_opt = st.s;
  //   st.x = cx.evaluate(s);
  //   st.y = cy.evaluate(s);
  //   st.dist2 = std::pow(st.x - x0, 2) + std::pow(st.y - y0, 2);
  //   return st;
  // }
};

CubicSpline2D::CubicSpline2D(const double *x, const double *y, unsigned long len)
    : p_(std::make_unique<CubicSpline2DImpl>(x, y, len)) {}

ClosestState CubicSpline2D::continuousClosest(double state_x, double state_y) {
  auto res = p_->continuousClosest(state_x, state_y, p_->es[p_->es.size() - 1],10);
  // return ClosestState{};
  return res;
}

void CubicSpline2D::findTargetPoint(const ClosestState &cs, double ld, double &target_x, double &target_y) {
  auto target_s = ld + cs.s;
  std::tie(target_x, target_y) = p_->pointAt(target_s);
}

size_t CubicSpline2D::getSamplePointsCount(double ds) { return p_->getSamplePointsCount(ds); }

void CubicSpline2D::cubicSpline(double *out_x, double *out_y, double ds) { p_->cubicSpline(out_x, out_y, ds); }
CubicSpline2D::~CubicSpline2D() = default;
CubicSpline2D::CubicSpline2D(CubicSpline2D &&) noexcept = default;
CubicSpline2D &CubicSpline2D::operator=(CubicSpline2D &&) noexcept = default;

}  // namespace trajectory_smooth
