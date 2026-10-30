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
// Operator-level unit tests for the immersed FSI interpolation matrix.
//
// The kernel tests (test_ib_interpolation.cpp) verify nn::get_nnx on a single
// element. These tests verify the *assembled* coupling operator
//     C_lambda,AB = N_A^f(phi(X_B))
// built by ib::build_coupling_rows over a real (multi-element) background mesh,
// and the interpolation apply ib::project_fluid_velocity_to_solid (which is the
// transpose action C_lambda^T of that operator on a fluid velocity field).
//
// Fixture: a background fluid mesh of two HEX8 elements sharing a face
//   element 0 = [0,1]x[0,1]x[0,1],  element 1 = [1,2]x[0,1]x[0,1]
// and four immersed solid nodes, two inside each element. No displacement is
// applied (Ubk = 0), so each solid node's current position equals its reference
// position.
//
// Checks:
//   1. Localization      : every solid node is matched to the element that
//                          actually contains it, with the correct fluid nodes.
//   2. Row partition of unity : each assembled row sum_A N_A = 1 (Peskin
//                          identity after real localization).
//   3. Constant reproduction  : a uniform fluid velocity interpolates to the
//                          same constant at every solid node.
//   4. Linear reproduction    : a linear fluid field V(x)=a+G.x interpolates
//                          to a+G.phi(X_B) exactly (first-moment condition).
//   5. Momentum conservation  : spreading a solid nodal force through C_lambda
//                          preserves the total force (follows from row-sum=1).
//   6. Operator consistency   : project_fluid_velocity_to_solid reproduces the
//                          same result as applying the stored rows by hand, i.e.
//                          the production interpolation path and the assembled
//                          operator agree (the transpose reuse behind the
//                          adjoint/duality identity V_f.(C_lambda f_s) =
//                          (C_lambda^T V_f).f_s).
//------------------------------------------------------------------------------

#include "ib.h"
#include "nn.h"
#include "consts.h"
#include "ComMod.h"
#include "SolutionStates.h"
#include "gtest/gtest.h"

#include <array>
#include <cmath>
#include <set>
#include <stdexcept>
#include <vector>

using namespace consts;

namespace {

constexpr int NSD = 3;
constexpr int N_FLUID = 12;   // fluid nodes (global ids 0..11)
constexpr int N_SOLID = 4;    // immersed solid nodes (global ids 12..15)
constexpr int TNNO = N_FLUID + N_SOLID;
constexpr int FLUID_DOMAIN = 0;

// Linear test field V_i(x) = a_i + sum_j G_ij x_j.
const double A_[NSD] = {0.5, -0.3, 1.1};
const double G_[NSD][NSD] = {
  { 1.0,  0.2, -0.1},
  { 0.3, -0.5,  0.4},
  {-0.2,  0.1,  0.7}};

double linear_field(int i, const double* x)
{
  return A_[i] + G_[i][0]*x[0] + G_[i][1]*x[1] + G_[i][2]*x[2];
}

// Fluid nodal coordinates.
const double FX[N_FLUID][NSD] = {
  {0,0,0},{1,0,0},{1,1,0},{0,1,0},{0,0,1},{1,0,1},{1,1,1},{0,1,1},   // elem 0 corners
  {2,0,0},{2,1,0},{2,0,1},{2,1,1}};                                  // extra corners for elem 1
// HEX8 connectivity (standard corner ordering used by nn_elem_gnn.h).
const int IEN0[8] = {0,1,2,3,4,5,6,7};
const int IEN1[8] = {1,8,9,2,5,10,11,6};

// Immersed solid nodal coordinates and the element each one sits in.
const double SX[N_SOLID][NSD] = {
  {0.5,0.5,0.5},   // -> elem 0
  {0.3,0.7,0.2},   // -> elem 0
  {1.5,0.5,0.5},   // -> elem 1
  {1.7,0.3,0.8}};  // -> elem 1
const int EXPECT_ELEM[N_SOLID] = {0, 0, 1, 1};

class IBOperator : public ::testing::Test {
protected:
  ComMod com_mod;

  void SetUp() override
  {
    com_mod.cm.nProcs = 1;
    com_mod.nsd = NSD;
    com_mod.nMsh = 1;
    com_mod.tnNo = TNNO;

    // Global coordinate array (fluid nodes then solid nodes).
    com_mod.x.resize(NSD, TNNO);
    for (int n = 0; n < N_FLUID; n++)
      for (int i = 0; i < NSD; i++) com_mod.x(i, n) = FX[n][i];
    for (int n = 0; n < N_SOLID; n++)
      for (int i = 0; i < NSD; i++) com_mod.x(i, N_FLUID + n) = SX[n][i];

    // Background fluid mesh: two HEX8 elements.
    mshType mesh;
    mesh.eType = ElementType::HEX8;
    mesh.eNoN = 8;
    mesh.nEl = 2;
    mesh.IEN.resize(8, 2);
    for (int a = 0; a < 8; a++) { mesh.IEN(a, 0) = IEN0[a]; mesh.IEN(a, 1) = IEN1[a]; }
    mesh.eId.resize(2);
    mesh.eId(0) = (1 << FLUID_DOMAIN);   // mark both elements as fluid-domain elements
    mesh.eId(1) = (1 << FLUID_DOMAIN);
    mesh.xib.resize(2, NSD);
    mesh.Nb.resize(2, mesh.eNoN);
    nn::get_nn_bnds(NSD, mesh.eType, mesh.eNoN, mesh.xib, mesh.Nb);
    // mesh.xi left empty -> get_nnx starts Newton from the element center (xi=0).
    com_mod.msh.push_back(mesh);

    // Immersed data.
    auto& ib = com_mod.ib;
    ib.tnNo = N_SOLID;
    ib.x.resize(NSD, N_SOLID);
    ib.Ubk.resize(NSD, N_SOLID);
    ib.Ubk = 0.0;
    ib.Yb.resize(NSD, N_SOLID);
    ib.gN.resize(N_SOLID);
    for (int a = 0; a < N_SOLID; a++) {
      for (int i = 0; i < NSD; i++) ib.x(i, a) = SX[a][i];
      ib.gN(a) = N_FLUID + a;   // solid node a maps to global id 12+a
    }

    // Assemble the coupling/interpolation operator (localization + rows).
    ASSERT_NO_THROW(ib::build_coupling_rows(com_mod, FLUID_DOMAIN, "unit_test"));
    ASSERT_EQ(static_cast<int>(com_mod.ib.ifemCoupling.size()), N_SOLID);
  }

  // Interpolate a nodal fluid field V (NSD x TNNO) through project_..._to_solid;
  // returns the solid velocities read back from the global velocity array.
  std::vector<std::array<double, NSD>> project(const std::vector<std::array<double, NSD>>& fluidV)
  {
    SolutionStates sol;
    Array<double>& V = sol.current.get_velocity();
    V.resize(NSD, TNNO);
    V = 0.0;
    for (int n = 0; n < N_FLUID; n++)
      for (int i = 0; i < NSD; i++) V(i, n) = fluidV[n][i];

    ib::project_fluid_velocity_to_solid(com_mod, sol);

    std::vector<std::array<double, NSD>> out(N_SOLID);
    for (int a = 0; a < N_SOLID; a++)
      for (int i = 0; i < NSD; i++) out[a][i] = V(i, com_mod.ib.gN(a));
    return out;
  }
};

TEST_F(IBOperator, CrossingCellFaceRejectsCachedExtrapolation)
{
  // Move the first HEX8 coupling point just across x=1, inside the generic
  // inverse-map margin of its previous cell but outside that physical cell.
  com_mod.ib.Ubk(0,0) = 0.500001;
  ib::build_coupling_rows(com_mod, FLUID_DOMAIN, "hex_crossing_test");
  const auto& row = com_mod.ib.ifemCoupling[0];
  EXPECT_EQ(row.fluidElem,1);
  for (int a = 0; a < row.N.size(); a++) EXPECT_GE(row.N(a),-1e-12);
}

// 1. Localization: correct host element and fluid node set for every solid node.
TEST_F(IBOperator, LocalizationFindsContainingElement)
{
  const std::set<int> nodes0(IEN0, IEN0 + 8);
  const std::set<int> nodes1(IEN1, IEN1 + 8);

  for (int a = 0; a < N_SOLID; a++) {
    const auto& row = com_mod.ib.ifemCoupling[a];
    EXPECT_EQ(row.ibNode, a);
    EXPECT_EQ(row.fluidElem, EXPECT_ELEM[a]) << "solid node " << a << " located in wrong element";

    std::set<int> got;
    for (int k = 0; k < row.fluidNodes.size(); k++) got.insert(row.fluidNodes(k));
    EXPECT_EQ(got, (EXPECT_ELEM[a] == 0 ? nodes0 : nodes1)) << "wrong fluid nodes for solid node " << a;
  }
}

// 2. Partition of unity of each assembled row (Peskin identity after localization).
TEST_F(IBOperator, RowPartitionOfUnity)
{
  for (int a = 0; a < N_SOLID; a++) {
    const auto& row = com_mod.ib.ifemCoupling[a];
    double s = 0.0;
    for (int k = 0; k < row.N.size(); k++) s += row.N(k);
    EXPECT_NEAR(s, 1.0, 1e-9) << "row " << a << " weights do not sum to 1";
  }
}

// 3. Constant reproduction: uniform fluid velocity -> same constant at solid nodes.
TEST_F(IBOperator, ConstantReproduction)
{
  const std::array<double, NSD> c = {2.0, -1.0, 3.5};
  std::vector<std::array<double, NSD>> fluidV(N_FLUID, c);

  auto solidV = project(fluidV);
  for (int a = 0; a < N_SOLID; a++)
    for (int i = 0; i < NSD; i++)
      EXPECT_NEAR(solidV[a][i], c[i], 1e-9) << "constant not reproduced at solid node " << a;
}

// 4. Linear reproduction: V(x)=a+G.x -> a+G.phi(X_B) at each solid node.
TEST_F(IBOperator, LinearReproduction)
{
  std::vector<std::array<double, NSD>> fluidV(N_FLUID);
  for (int n = 0; n < N_FLUID; n++)
    for (int i = 0; i < NSD; i++) fluidV[n][i] = linear_field(i, FX[n]);

  auto solidV = project(fluidV);
  for (int a = 0; a < N_SOLID; a++)
    for (int i = 0; i < NSD; i++) {
      double expected = linear_field(i, SX[a]);
      EXPECT_NEAR(solidV[a][i], expected, 1e-7) << "linear field not reproduced at solid node " << a;
    }
}

// 5. Momentum conservation: spreading a solid force through C_lambda preserves
//    the total force. Uses the assembled rows directly (spreading is the
//    transpose action of interpolation and reuses the same weights).
TEST_F(IBOperator, ForceSpreadingConservesMomentum)
{
  const double fs[N_SOLID][NSD] = {
    { 1.0, -2.0,  0.5},
    {-0.7,  0.3,  1.2},
    { 0.4,  0.9, -1.1},
    { 2.1, -0.6,  0.2}};

  // Spread: R_f(i, fluidNode) += N_a * fs(i, ibNode).
  std::vector<std::array<double, NSD>> Rf(N_FLUID, {0.0, 0.0, 0.0});
  for (int a = 0; a < N_SOLID; a++) {
    const auto& row = com_mod.ib.ifemCoupling[a];
    for (int k = 0; k < row.fluidNodes.size(); k++) {
      int Ac = row.fluidNodes(k);
      for (int i = 0; i < NSD; i++) Rf[Ac][i] += row.N(k) * fs[a][i];
    }
  }

  for (int i = 0; i < NSD; i++) {
    double total_spread = 0.0, total_applied = 0.0;
    for (int n = 0; n < N_FLUID; n++) total_spread += Rf[n][i];
    for (int a = 0; a < N_SOLID; a++) total_applied += fs[a][i];
    EXPECT_NEAR(total_spread, total_applied, 1e-9) << "momentum not conserved, component " << i;
  }
}

// 6. Operator consistency: the production interpolation path matches a direct
//    application of the stored rows (confirms C_lambda^T uses the same weights
//    that C_lambda would use for spreading).
TEST_F(IBOperator, ProjectionMatchesStoredRows)
{
  std::vector<std::array<double, NSD>> fluidV(N_FLUID);
  for (int n = 0; n < N_FLUID; n++)
    for (int i = 0; i < NSD; i++) fluidV[n][i] = linear_field(i, FX[n]);

  auto solidV = project(fluidV);

  for (int a = 0; a < N_SOLID; a++) {
    const auto& row = com_mod.ib.ifemCoupling[a];
    std::array<double, NSD> manual = {0.0, 0.0, 0.0};
    for (int k = 0; k < row.fluidNodes.size(); k++) {
      int Ac = row.fluidNodes(k);
      for (int i = 0; i < NSD; i++) manual[i] += row.N(k) * fluidV[Ac][i];
    }
    for (int i = 0; i < NSD; i++)
      EXPECT_NEAR(solidV[a][i], manual[i], 1e-12) << "projection disagrees with stored rows, node " << a;
  }
}

// 7. Work conservation / transpose identity:
//      V_f . (C_lambda R_s) == (C_lambda^T V_f) . R_s
//    This is the strongest cheap verification that interpolation and spreading
//    are using one assembled 3D operator as true transpose actions.
TEST_F(IBOperator, InterpolationAndSpreadingAreTransposeActions)
{
  std::vector<std::array<double, NSD>> fluidV(N_FLUID);
  for (int n = 0; n < N_FLUID; n++) {
    for (int i = 0; i < NSD; i++) {
      fluidV[n][i] = linear_field(i, FX[n]);
    }
  }

  const double solidR[N_SOLID][NSD] = {
    { 1.0, -2.0,  0.5},
    {-0.7,  0.3,  1.2},
    { 0.4,  0.9, -1.1},
    { 2.1, -0.6,  0.2}};

  std::vector<std::array<double, NSD>> spread(N_FLUID, {0.0, 0.0, 0.0});
  for (int a = 0; a < N_SOLID; a++) {
    const auto& row = com_mod.ib.ifemCoupling[a];
    for (int k = 0; k < row.fluidNodes.size(); k++) {
      const int Ac = row.fluidNodes(k);
      for (int i = 0; i < NSD; i++) {
        spread[Ac][i] += row.N(k) * solidR[a][i];
      }
    }
  }

  const auto solidV = project(fluidV);
  double fluid_work = 0.0;
  double solid_work = 0.0;

  for (int n = 0; n < N_FLUID; n++) {
    for (int i = 0; i < NSD; i++) {
      fluid_work += fluidV[n][i] * spread[n][i];
    }
  }
  for (int a = 0; a < N_SOLID; a++) {
    for (int i = 0; i < NSD; i++) {
      solid_work += solidV[a][i] * solidR[a][i];
    }
  }

  EXPECT_NEAR(fluid_work, solid_work, 1e-10);
}

} // namespace
