
#include "cubic_spline.h"

#include <Eigen/Dense>
#include <Eigen/QR>
#include <algorithm>  // std::lower_bound
#include <cstddef>
#include <cstring>  // std::strncpy, std::memset
#include <iomanip>
#include <sstream>
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

    // buildSystem();

    // c_.resize(nx_ - 1);
    // c_ = A_.triangularView<Eigen::UnitLower>().solve(B_);
    // c_ = A_.colPivHouseholderQr().solve(B_);

    // Thomas 三对角求解
    Eigen::VectorXd lower(nx_ - 1), diag(nx_), upper(nx_ - 1), rhs(nx_);
    lower.setZero();
    upper.setZero();

    diag.setOnes();
    rhs.setZero();

    for (size_t i = 1; i < nx_ - 1; ++i) {
      lower(i - 1) = h_(i - 1);
      diag(i) = 2.0 * (h_(i - 1) + h_(i));
      upper(i) = h_(i);
      rhs(i) = 3.0 * ((a_(i + 1) - a_(i)) / h_(i) - (a_(i) - a_(i - 1)) / h_(i - 1));
    }
    upper(nx_ - 2) = h_(nx_ - 2);  // 最后一行上对角
    // lower(nx_-2) 已在 setZero() 中正确置 0（自然边界）

    for (size_t i = 1; i < nx_; ++i) {
      double w = lower(i - 1) / diag(i - 1);
      diag(i) -= w * upper(i - 1);
      rhs(i) -= w * rhs(i - 1);
    }

    c_.resize(nx_);
    c_(nx_ - 1) = rhs(nx_ - 1) / diag(nx_ - 1);
    for (int i = static_cast<int>(nx_) - 2; i >= 0; --i) {
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
    if (s_query <= s_(0)) return 0;
    if (s_query >= s_(nx_ - 1)) return nx_ - 2;
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

  size_t segmentCount() const { return nx_ - 1; }

  double knot(size_t i) const { return s_(i); }

  double segmentLength(size_t i) const { return h_(i); }

  Eigen::Vector4d coeff(size_t i) const { return {a_(i), b_(i), c_(i), d_(i)}; }

  const Eigen::MatrixXd &A() const noexcept { return A_; }
  const Eigen::VectorXd &B() const noexcept { return B_; }

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

// Eigen::VectorXd xy2s(const Eigen::VectorXd &x, const Eigen::VectorXd &y) {
//   const Eigen::Index n = x.size();
//   if (n < 2)  // 0 点或 1 点时，弧长恒为 0
//     return Eigen::VectorXd::Zero(n);

//   Eigen::ArrayXd dx = x.tail(n - 1) - x.head(n - 1);
//   Eigen::ArrayXd dy = y.tail(n - 1) - y.head(n - 1);
//   Eigen::ArrayXd seg = (dx.square() + dy.square()).sqrt();  // 每段弧长

//   Eigen::VectorXd s(n);
//   s(0) = 0.0;

//   s.tail(n - 1) = seg.matrix();
//   for (Eigen::Index i = 1; i < n - 1; ++i) s(i + 1) += s(i);

//   return s;
// }

Eigen::VectorXd xy2s(const Eigen::VectorXd &x, const Eigen::VectorXd &y) {
  const Eigen::Index n = x.size();
  Eigen::VectorXd s(n);

  if (n == 0) return s;
  if (n == 1) {
    s.setZero();
    return s;
  }

  Eigen::ArrayXd dx = x.tail(n - 1) - x.head(n - 1);
  Eigen::ArrayXd dy = y.tail(n - 1) - y.head(n - 1);
  Eigen::ArrayXd seg = (dx.square() + dy.square()).sqrt();  // 每段长

  if ((!dx.isFinite()).any() || (!dy.isFinite()).any()) throw std::runtime_error("xy2s(): x/y contains NaN or Inf");
  if ((!seg.isFinite()).any()) throw std::runtime_error("xy2s(): segment length NaN/Inf");
  if ((seg <= 0).any()) throw std::runtime_error("xy2s(): duplicated consecutive points");

  s.setZero();
  s.tail(n - 1) = seg.matrix();

  s[0] = 0.0;
  for (Eigen::Index i = 1; i < n; ++i) s[i] = s[i - 1] + seg[i - 1];

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
  double _s_max;
  double safety;
  CubicSpline1D cx;
  CubicSpline1D cy;
  std::optional<double> s_prev_opt;

  CubicSpline2DImpl(const double *x, const double *y, unsigned long len)
      : ex(Eigen::Map<const Eigen::VectorXd>(x, len)),
        ey(Eigen::Map<const Eigen::VectorXd>(y, len)),
        es(xy2s(ex, ey)),
        cx(es, ex),
        cy(es, ey),
        s_prev_opt(std::nullopt) {
    _s_max = es(es.size() - 1);
    safety = 1e-4 * _s_max;
  }

  double getSmax()
  {
    return _s_max-safety;
  }

  size_t getSamplePointsCount(double ds) {
    // std::ptrdiff
    return static_cast<std::size_t>(std::floor(_s_max / ds));
  }

  Eigen::VectorXd getSamplePoints(double ds) {
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

  std::pair<double, double> pointAt(double s) { 
    s=std::clamp(s, 0.0,getSmax());
    return std::pair<double, double>(cx.evaluate(s), cy.evaluate(s));
  }

  double slopeAt(double s) const {
    double dx = cx.firstDerivative(s);
    double dy = cy.firstDerivative(s);

    constexpr double kEps = 1e-12;
    if (std::abs(dx) < kEps) {
      return (dy > 0.0 ? std::numeric_limits<double>::infinity() : -std::numeric_limits<double>::infinity());
    }
    return dy / dx;  // 斜率 dy/dx
    // return std::atan2(dy, dx);   // 航向角（弧度）
  }

  void debug(char *info, size_t len) {
    if (!info || len == 0) return;
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(6);

    /* -------- 基本信息 -------- */
    oss << "======== CubicSpline2D Debug ========\n";
    oss << "Control  points (N) : " << ex.size() << '\n';
    oss << "Segment  count      : " << cx.segmentCount() << '\n';
    oss << "Total arc length    : " << _s_max << "\n\n";

    /* -------- 每段系数 -------- */
    for (size_t i = 0; i < cx.segmentCount(); ++i) {
      const double s0 = cx.knot(i);
      const double s1 = cx.knot(i + 1);
      auto cx_coef = cx.coeff(i);
      auto cy_coef = cy.coeff(i);

      oss << "---- Segment " << i << "  (s in [" << s0 << ", " << s1 << "]) ----\n"
          << "X(s) = " << cx_coef[0] << " + " << cx_coef[1] << "*ds + " << cx_coef[2] << "*ds^2 + " << cx_coef[3]
          << "*ds^3\n"
          << "Y(s) = " << cy_coef[0] << " + " << cy_coef[1] << "*ds + " << cy_coef[2] << "*ds^2 + " << cy_coef[3]
          << "*ds^3\n\n";
    }

    // 调用buildsystem后才能使用
    // Eigen::IOFormat mat_fmt(6, 0, " ", "\n", "[", "]");
    // oss << "===== Ax = B for X-dimension spline =====\n";
    // oss << "A_x =\n" << cx.A().format(mat_fmt) << "\n\n";
    // oss << "B_x =\n" << cx.B().transpose().format(mat_fmt) << "\n\n";

    // oss << "===== Ay = B for Y-dimension spline =====\n";
    // oss << "A_y =\n" << cy.A().format(mat_fmt) << "\n\n";
    // oss << "B_y =\n" << cy.B().transpose().format(mat_fmt) << "\n\n";

    const std::string str = oss.str();
    std::size_t copy_len = std::min(len - 1, str.size());
    std::memcpy(info, str.data(), copy_len);
    info[copy_len] = '\0';
  }

  ClosestState continuousClosest(double x0, double y0, double s_max, int max_iter) {
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

    constexpr double eps_s = 1e-10;
    constexpr double eps_G = 1e-10;
    constexpr double eps_Gp = 1e-12;
    constexpr double step_limit = 0.1;

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

      if (std::abs(G) < eps_G || std::abs(Gp) < eps_Gp) break;

      double grad2 = dxs * dxs + dys * dys;
      if (grad2 < 1e-8) {
        double h = 0.01 * s_max;
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

      double delta;
      if (std::abs(Gp) >= eps_Gp) {
        delta = -G / Gp;
        delta = std::clamp(delta, -step_limit, step_limit);
      } else {
        delta = -G / (std::sqrt(grad2) + 1e-12) * 0.05;  // 再×5 %
      }

      double s_new = std::clamp(s + delta, safety, s_max - safety);
      if (std::abs(s_new - s) < eps_s) {
        s = s_new;
        break;
      }

      s = s_new;
    }

    ClosestState st;
    st.s = s;
    s_prev_opt = s;
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
};

CubicSpline2D::CubicSpline2D(const double *x, const double *y, unsigned long len)
    : p_(std::make_unique<CubicSpline2DImpl>(x, y, len)) {}

ClosestState CubicSpline2D::continuousClosest(double state_x, double state_y) {
  auto res = p_->continuousClosest(state_x, state_y, p_->_s_max, 10);
  return res;
}

void CubicSpline2D::findTargetPoint(const ClosestState &cs, double ld, double &target_x, double &target_y) {
  auto target_s = ld + cs.s;
  std::tie(target_x, target_y) = p_->pointAt(target_s);
}

void CubicSpline2D::pointAt(double s, double &target_x, double &target_y) {
  std::tie(target_x, target_y) = p_->pointAt(s);
}

double CubicSpline2D ::getSmax(){return p_->getSmax();}
size_t CubicSpline2D::getSamplePointsCount(double ds) { return p_->getSamplePointsCount(ds); }
void CubicSpline2D::debug(char *info, size_t len) { return p_->debug(info, len); }

void CubicSpline2D::cubicSpline(double *out_x, double *out_y, double ds) { p_->cubicSpline(out_x, out_y, ds); }
CubicSpline2D::~CubicSpline2D() = default;
CubicSpline2D::CubicSpline2D(CubicSpline2D &&) noexcept = default;
CubicSpline2D &CubicSpline2D::operator=(CubicSpline2D &&) noexcept = default;

}  // namespace trajectory_smooth
