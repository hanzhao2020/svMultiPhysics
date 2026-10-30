/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject
 * to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
 * IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
 * TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
 * PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER
 * OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

//------------------------------------------------------------------------------
// Unit tests for the finite-element interpolation used by the immersed
// boundary / immersed FEM coupling in ib.cpp.
//
// The immersed FSI coupling operator is C_lambda,AB = N_A^f(phi(X_B)), i.e. the
// background fluid shape functions evaluated at immersed solid node locations
// (see immersed_fsi_eq14_algorithm.md). Those weights are produced by
// nn::get_nnx, which inverse-maps a physical point into a background element and
// evaluates the shape functions there. Every correctness property of the IB
// interpolation therefore reduces to a property of get_nnx on a single element.
//
// These tests exercise get_nnx directly (no ComMod, no ib.cpp edits required)
// and check the finite-element analogues of Peskin's discrete-delta conditions:
//
//   1. Partition of unity  : sum_A N_A(x) = 1        (Peskin zeroth moment,
//                                                      sum of delta = 1).
//                                                      => constant reproduction,
//                                                      the "identity" check.
//   2. Polynomial (linear) reproduction:
//        - first moment     : sum_A N_A(x) x_A = x    (Peskin first moment = 0)
//        - a linear field prescribed at the element nodes is interpolated
//          exactly at any interior point.
//   3. Localization         : a point outside the element is rejected
//                             (get_nnx throws), so spreading/interpolation never
//                             uses an element that does not contain the point.
//
// The tests cover a simplex family (TET4, TRI3) where the map is affine and the
// reproduction is exact to round-off, and a tensor-product family (HEX8, QUD4)
// where get_xi must invert a (tri/bi)linear map.
//------------------------------------------------------------------------------

#include "nn.h"
#include "FE/Common/FEException.h"
#include "consts.h"
#include "ComMod.h"    // pulls in Array<T> / Vector<T>
#include "gtest/gtest.h"

#include <cmath>
#include <stdexcept>
#include <vector>

using namespace consts;

namespace {

// A single reference element described by its type, spatial dimension, number
// of nodes, physical nodal coordinates, and a starting parametric guess for the
// Newton inverse map inside get_nnx.
struct ElementFixture {
  ElementType eType;
  int nsd;
  int eNoN;
  Array<double> xl;      // (nsd, eNoN) physical nodal coordinates
  Vector<double> xi0;    // (nsd) initial parametric guess for get_xi
};

// Build the physical coordinate at parametric weights {w} (a convex combination
// of the element nodes, w_a >= 0, sum w_a = 1). Guarantees an interior point.
Vector<double> physical_point(const ElementFixture& el, const std::vector<double>& w)
{
  Vector<double> x(el.nsd);
  x = 0.0;
  for (int a = 0; a < el.eNoN; a++) {
    for (int i = 0; i < el.nsd; i++) {
      x(i) += w[a] * el.xl(i, a);
    }
  }
  return x;
}

// Call get_nnx and return the interpolation weights N at physical point xp.
Vector<double> interpolate(const ElementFixture& el, const Vector<double>& xp)
{
  Array<double> xib(2, el.nsd);
  Array<double> Nb(2, el.eNoN);
  nn::get_nn_bnds(el.nsd, el.eType, el.eNoN, xib, Nb);

  Vector<double> xi = el.xi0;
  Vector<double> N(el.eNoN);
  Array<double> Nx(el.nsd, el.eNoN);
  nn::get_nnx(el.nsd, el.eType, el.eNoN, el.xl, xib, Nb, xp, xi, N, Nx);
  return N;
}

// ------------------------- Element definitions ---------------------------

// TET4: N0=xi0, N1=xi1, N2=xi2, N3=1-xi0-xi1-xi2. Any non-degenerate tet works
// because the reproduction properties are independent of node ordering.
ElementFixture make_tet4()
{
  ElementFixture el;
  el.eType = ElementType::TET4;
  el.nsd = 3;
  el.eNoN = 4;
  el.xl.resize(3, 4);
  const double v[4][3] = {
    {0.10, 0.20, 0.00},
    {1.20, 0.10, 0.30},
    {0.20, 1.10, 0.10},
    {0.00, 0.30, 1.20}};
  for (int a = 0; a < 4; a++)
    for (int i = 0; i < 3; i++) el.xl(i, a) = v[a][i];
  el.xi0.resize(3);
  el.xi0 = 0.25;   // parametric centroid of the reference tet
  return el;
}

// TRI3: N0=xi0, N1=xi1, N2=1-xi0-xi1.
ElementFixture make_tri3()
{
  ElementFixture el;
  el.eType = ElementType::TRI3;
  el.nsd = 2;
  el.eNoN = 3;
  el.xl.resize(2, 3);
  const double v[3][2] = {
    {0.00, 0.00},
    {1.30, 0.20},
    {0.10, 1.10}};
  for (int a = 0; a < 3; a++)
    for (int i = 0; i < 2; i++) el.xl(i, a) = v[a][i];
  el.xi0.resize(2);
  el.xi0 = 1.0 / 3.0;
  return el;
}

// HEX8: reference corners at (-1,-1,-1),(+1,-1,-1),(+1,+1,-1),(-1,+1,-1),
// (-1,-1,+1),(+1,-1,+1),(+1,+1,+1),(-1,+1,+1). Physical corners obtained by an
// affine map plus a small perturbation of one corner so the map is genuinely
// non-affine and get_xi's Newton iteration is exercised.
ElementFixture make_hex8()
{
  ElementFixture el;
  el.eType = ElementType::HEX8;
  el.nsd = 3;
  el.eNoN = 8;
  el.xl.resize(3, 8);
  const double r[8][3] = {
    {-1,-1,-1},{ 1,-1,-1},{ 1, 1,-1},{-1, 1,-1},
    {-1,-1, 1},{ 1,-1, 1},{ 1, 1, 1},{-1, 1, 1}};
  for (int a = 0; a < 8; a++) {
    // affine map: scale, shear, translate  (r in [-1,1] -> physical)
    double X = r[a][0], Y = r[a][1], Z = r[a][2];
    el.xl(0, a) = 0.5 + 1.0*X + 0.15*Y;
    el.xl(1, a) = 0.3 + 0.8*Y + 0.10*Z;
    el.xl(2, a) = 0.2 + 1.2*Z;
  }
  // Perturb corner 6 to make the map non-affine.
  el.xl(0, 6) += 0.12;
  el.xl(1, 6) += 0.08;
  el.xl(2, 6) += 0.05;
  el.xi0.resize(3);
  el.xi0 = 0.0;    // parametric center of the reference hex
  return el;
}

// QUD4: reference corners (-1,-1),(+1,-1),(+1,+1),(-1,+1).
ElementFixture make_qud4()
{
  ElementFixture el;
  el.eType = ElementType::QUD4;
  el.nsd = 2;
  el.eNoN = 4;
  el.xl.resize(2, 4);
  const double r[4][2] = {{-1,-1},{1,-1},{1,1},{-1,1}};
  for (int a = 0; a < 4; a++) {
    double X = r[a][0], Y = r[a][1];
    el.xl(0, a) = 0.4 + 1.0*X + 0.20*Y;
    el.xl(1, a) = 0.1 + 0.9*Y;
  }
  el.xl(0, 2) += 0.10;   // perturb one corner -> non-affine bilinear map
  el.xl(1, 2) += 0.06;
  el.xi0.resize(2);
  el.xi0 = 0.0;
  return el;
}

// A set of strictly-interior sampling weights for an element with eNoN nodes.
std::vector<std::vector<double>> interior_weights(int eNoN)
{
  std::vector<std::vector<double>> ws;
  // centroid
  ws.push_back(std::vector<double>(eNoN, 1.0 / eNoN));
  // a few asymmetric interior points
  for (int shift = 0; shift < 2; shift++) {
    std::vector<double> w(eNoN);
    double s = 0.0;
    for (int a = 0; a < eNoN; a++) { w[a] = 1.0 + (a + shift) % eNoN; s += w[a]; }
    for (int a = 0; a < eNoN; a++) w[a] /= s;
    ws.push_back(w);
  }
  return ws;
}

// ------------------------------- Tests -----------------------------------

// Property 1 + 2: for every element type and several interior points, check
// partition of unity and first-moment (isoparametric) reproduction.
void check_partition_and_reproduction(const ElementFixture& el, double tol)
{
  for (const auto& w : interior_weights(el.eNoN)) {
    Vector<double> xp = physical_point(el, w);
    Vector<double> N = interpolate(el, xp);

    // 1. Partition of unity: sum_A N_A = 1  (Peskin identity / constant reproduction).
    double sumN = 0.0;
    for (int a = 0; a < el.eNoN; a++) sumN += N(a);
    EXPECT_NEAR(sumN, 1.0, tol) << "partition of unity failed";

    // 2. First moment: sum_A N_A x_A = xp  (Peskin first moment vanishes).
    for (int i = 0; i < el.nsd; i++) {
      double xi_interp = 0.0;
      for (int a = 0; a < el.eNoN; a++) xi_interp += N(a) * el.xl(i, a);
      EXPECT_NEAR(xi_interp, xp(i), tol) << "first-moment reproduction failed, dim " << i;
    }
  }
}

// Property 2 (explicit): a linear scalar field f(x) = alpha + beta . x prescribed
// at the nodes is interpolated exactly at any interior point.
void check_linear_field(const ElementFixture& el, double tol)
{
  const double alpha = -0.7;
  std::vector<double> beta = {1.3, -0.4, 0.9};   // uses first nsd entries

  for (const auto& w : interior_weights(el.eNoN)) {
    Vector<double> xp = physical_point(el, w);
    Vector<double> N = interpolate(el, xp);

    double f_interp = 0.0;
    for (int a = 0; a < el.eNoN; a++) {
      double f_a = alpha;
      for (int i = 0; i < el.nsd; i++) f_a += beta[i] * el.xl(i, a);
      f_interp += N(a) * f_a;
    }
    double f_exact = alpha;
    for (int i = 0; i < el.nsd; i++) f_exact += beta[i] * xp(i);

    EXPECT_NEAR(f_interp, f_exact, tol) << "linear field not reproduced";
  }
}

// Same reproduction check as above, but for a vector field. This matches the
// actual IFEM velocity transfer, where each spatial component is interpolated
// with the same scalar row of C_lambda.
void check_linear_vector_field(const ElementFixture& el, double tol)
{
  const double alpha[3] = {0.5, -0.3, 1.1};
  const double grad[3][3] = {
    { 1.0,  0.2, -0.1},
    { 0.3, -0.5,  0.4},
    {-0.2,  0.1,  0.7}};

  for (const auto& w : interior_weights(el.eNoN)) {
    Vector<double> xp = physical_point(el, w);
    Vector<double> N = interpolate(el, xp);

    for (int i = 0; i < el.nsd; i++) {
      double vi_interp = 0.0;
      for (int a = 0; a < el.eNoN; a++) {
        double vi_a = alpha[i];
        for (int j = 0; j < el.nsd; j++) {
          vi_a += grad[i][j] * el.xl(j, a);
        }
        vi_interp += N(a) * vi_a;
      }

      double vi_exact = alpha[i];
      for (int j = 0; j < el.nsd; j++) {
        vi_exact += grad[i][j] * xp(j);
      }
      EXPECT_NEAR(vi_interp, vi_exact, tol)
          << "linear vector field not reproduced, component " << i;
    }
  }
}

void check_tet4_boundary_points(double tol)
{
  const ElementFixture el = make_tet4();

  std::vector<std::vector<double>> samples = {
    {1.0, 0.0, 0.0, 0.0},
    {0.0, 1.0, 0.0, 0.0},
    {0.0, 0.0, 1.0, 0.0},
    {0.0, 0.0, 0.0, 1.0},
    {0.5, 0.5, 0.0, 0.0},
    {0.0, 0.5, 0.5, 0.0},
    {0.0, 0.0, 0.5, 0.5},
    {0.5, 0.0, 0.0, 0.5},
    {1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 0.0},
    {1.0 / 3.0, 0.0, 1.0 / 3.0, 1.0 / 3.0}};

  for (const auto& w : samples) {
    Vector<double> xp = physical_point(el, w);
    Vector<double> N = interpolate(el, xp);

    double sumN = 0.0;
    for (int a = 0; a < el.eNoN; a++) {
      sumN += N(a);
      EXPECT_GE(N(a), -tol) << "TET4 boundary shape value below tolerance";
    }
    EXPECT_NEAR(sumN, 1.0, tol) << "TET4 boundary partition failed";

    for (int i = 0; i < el.nsd; i++) {
      double xi_interp = 0.0;
      for (int a = 0; a < el.eNoN; a++) {
        xi_interp += N(a) * el.xl(i, a);
      }
      EXPECT_NEAR(xi_interp, xp(i), tol)
          << "TET4 boundary first-moment reproduction failed, dim " << i;
    }
  }
}

TEST(IBInterpolation, PartitionOfUnity_TET4)  { check_partition_and_reproduction(make_tet4(), 1e-10); }
TEST(IBInterpolation, PartitionOfUnity_TRI3)  { check_partition_and_reproduction(make_tri3(), 1e-10); }
TEST(IBInterpolation, PartitionOfUnity_HEX8)  { check_partition_and_reproduction(make_hex8(), 1e-7);  }
TEST(IBInterpolation, PartitionOfUnity_QUD4)  { check_partition_and_reproduction(make_qud4(), 1e-7);  }

TEST(IBInterpolation, LinearReproduction_TET4) { check_linear_field(make_tet4(), 1e-10); }
TEST(IBInterpolation, LinearReproduction_TRI3) { check_linear_field(make_tri3(), 1e-10); }
TEST(IBInterpolation, LinearReproduction_HEX8) { check_linear_field(make_hex8(), 1e-7);  }
TEST(IBInterpolation, LinearReproduction_QUD4) { check_linear_field(make_qud4(), 1e-7);  }

TEST(IBInterpolation, LinearVectorReproduction_TET4) { check_linear_vector_field(make_tet4(), 1e-10); }
TEST(IBInterpolation, LinearVectorReproduction_HEX8) { check_linear_vector_field(make_hex8(), 1e-7);  }

TEST(IBInterpolation, BoundaryPointReproduction_TET4) { check_tet4_boundary_points(1e-10); }

// Property 3: a point well outside the element must be rejected by get_nnx.
// This is what lets locate_background_fluid_trace reject a wrong element and
// move on, so a solid node is only ever coupled to the element that contains it.
TEST(IBInterpolation, LocalizationRejectsOutsidePoint)
{
  ElementFixture el = make_tet4();
  Vector<double> xp(3);
  xp(0) = 5.0; xp(1) = 5.0; xp(2) = 5.0;   // far outside the tet

  EXPECT_THROW(interpolate(el, xp), svmp::FE::InvalidArgumentException)
      << "get_nnx accepted a point outside the element";
}

} // namespace
