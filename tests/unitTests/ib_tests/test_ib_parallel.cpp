// SPDX-FileCopyrightText: Copyright (c) Stanford University, The Regents of the University of California, and others.
// SPDX-License-Identifier: BSD-3-Clause

#include "ib.h"
#include "nn.h"
#include "gtest/gtest.h"

namespace {

// The regular unit-test executable uses gtest_main, which does not initialize
// MPI. Initialize it once when these tests run and finalize only if we own it.
void initialize_test_mpi()
{
  struct Runtime {
    bool owned = false;
    Runtime() {
      int initialized = 0;
      MPI_Initialized(&initialized);
      owned = !initialized;
      if (owned) MPI_Init(nullptr, nullptr);
    }
    ~Runtime() {
      if (owned) MPI_Finalize();
    }
  };
  static Runtime runtime;
}

mshType quad_mesh(const int count)
{
  mshType mesh;
  mesh.eType = consts::ElementType::QUD4;
  mesh.eNoN = 4;
  mesh.nEl = count;
  mesh.IEN.resize(4,count);
  mesh.eId.resize(count);
  for (int e = 0; e < count; e++) mesh.eId(e) = 1;
  mesh.xib.resize(2,2);
  mesh.Nb.resize(2,4);
  nn::get_nn_bnds(2,mesh.eType,4,mesh.xib,mesh.Nb);
  return mesh;
}

void set_solid_node(ComMod& com, int local_node, double x, double y)
{
  com.ib.tnNo = 1;
  com.ib.gN.resize(1);
  com.ib.gN(0) = local_node;
  com.ib.x.resize(2,1);
  com.ib.x(0,0) = x;
  com.ib.x(1,0) = y;
  com.ib.Ubk.resize(2,1);
  com.ib.Ubk = 0.0;
  com.ib.Yb.resize(2,1);
}

TEST(IBPointLocation, MovingAcrossFaceDoesNotRetainExtrapolatingTrace)
{
  ComMod com;
  com.cm.nProcs = 1;
  com.nsd = 2;
  com.nMsh = 1;
  com.tnNo = 7;
  const double points[6][2] = {{0,0},{1,0},{1,1},{0,1},{2,0},{2,1}};
  com.x.resize(2,7);
  for (int a = 0; a < 6; a++)
    for (int i = 0; i < 2; i++) com.x(i,a) = points[a][i];
  auto mesh = quad_mesh(2);
  const int cells[2][4] = {{0,1,2,3},{1,4,5,2}};
  for (int e = 0; e < 2; e++)
    for (int a = 0; a < 4; a++) mesh.IEN(a,e) = cells[e][a];
  com.msh.push_back(mesh);
  set_solid_node(com,6,0.5,0.5);
  ib::build_coupling_rows(com,0,"crossing_test");
  EXPECT_EQ(com.ib.ifemCoupling[0].fluidElem,0);
  // This lies inside get_nnx's general acceptance margin of the left cell,
  // but IFEM must select the right cell rather than extrapolate from the left.
  com.ib.Ubk(0,0) = 0.500001;
  ib::build_coupling_rows(com,0,"crossing_test");
  const auto& row = com.ib.ifemCoupling[0];
  EXPECT_EQ(row.fluidElem,1);
  EXPECT_EQ(row.ibNode,0);
  EXPECT_EQ(row.ibGlobalNode,6);
  for (int a = 0; a < row.N.size(); a++) EXPECT_GE(row.N(a),-1e-12);
}

TEST(IBParallel, RemoteVelocityAndIncrementUseSelectedTrace)
{
  initialize_test_mpi();
  ComMod com;
  com.cm.new_cm(MPI_COMM_WORLD);
  if (com.cm.np() < 2) GTEST_SKIP() << "Run with mpirun -np 2 or more";
  const int rank = com.cm.idcm();
  com.nsd = 2;
  com.dof = 3;
  com.tnNo = rank < 2 ? 3 : 0;
  com.ib.tnNo = 0;
  if (rank < 2) {
    set_solid_node(com,2,0.5,0.5);
    ifemCouplingType row;
    row.ibNode = 0;
    row.ibGlobalNode = 9;
    row.fluidElemOwnerRank = 0;
    row.fluidElemGlobal = 42;
    row.fluidNodes.resize(2);
    row.N.resize(2);
    row.fluidNodes(0) = 0;
    row.fluidNodes(1) = 1;
    row.N(0) = row.N(1) = 0.5;
    com.ib.ifemCoupling.push_back(row);
    if (rank == 0) {
      com.ib.ifemRemoteCoupling.push_back(row);
      row.fluidElemGlobal = 43;
      row.N(0) = 1.0;
      row.N(1) = 0.0;
      com.ib.ifemRemoteCoupling.push_back(row); // valid but unselected response
    }
  } else {
    com.ib.Yb.resize(2,0);
  }
  SolutionStates sol;
  auto& velocity = sol.current.get_velocity();
  velocity.resize(2,com.tnNo);
  velocity = -99.0;
  com.R.resize(3,com.tnNo);
  com.R = 0.0;
  if (rank < 2) {
    for (int i = 0; i < 2; i++) {
      velocity(i,0) = 2.0 + i;
      velocity(i,1) = 4.0 + i;
      com.R(i,0) = 6.0 + i;
      com.R(i,1) = 8.0 + i;
    }
  }
  ib::project_fluid_velocity_to_solid(com,sol);
  ib::project_fluid_increment_to_solid(com);
  if (rank < 2) {
    for (int i = 0; i < 2; i++) {
      EXPECT_DOUBLE_EQ(velocity(i,2),3.0 + i);
      EXPECT_DOUBLE_EQ(com.R(i,2),7.0 + i);
    }
  }
}

TEST(IBParallel, OnlySelectedPointLocationResponseIsRetained)
{
  initialize_test_mpi();
  ComMod com;
  com.cm.new_cm(MPI_COMM_WORLD);
  if (com.cm.np() < 3) GTEST_SKIP() << "Run with mpirun -np 3 or more";
  const int rank = com.cm.idcm();
  com.nsd = 2;
  com.nMsh = 1;
  com.tnNo = rank < 2 ? 4 : (rank == 2 ? 1 : 0);
  com.x.resize(2,com.tnNo);
  com.ltg.resize(com.tnNo);
  auto mesh = quad_mesh(rank < 2 ? 1 : 0);
  if (rank < 2) {
    const double points[4][2] = {{0,0},{1,0},{1,1},{0,1}};
    const int global_nodes[2][4] = {{0,1,2,3},{1,4,5,2}};
    for (int a = 0; a < 4; a++) {
      com.x(0,a) = points[a][0] + rank;
      com.x(1,a) = points[a][1];
      mesh.IEN(a,0) = a;
      com.ltg(a) = global_nodes[rank][a];
    }
  }
  com.msh.push_back(mesh);
  if (rank == 2) {
    com.ltg(0) = 6;
    set_solid_node(com,0,1.0,0.5); // both fluid ranks can locate this point
  } else {
    com.ib.tnNo = 0;
  }
  ib::build_coupling_rows(com,0,"multiple_response_test");
  int local = static_cast<int>(com.ib.ifemRemoteCoupling.size());
  int global = 0;
  MPI_Allreduce(&local,&global,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
  EXPECT_EQ(global,1);
  if (rank == 2) EXPECT_EQ(com.ib.ifemCoupling[0].fluidElemOwnerRank,0);

  // Both ranks number their only fluid cell as local element zero. Moving
  // across the partition changes the owner even though the local index stays
  // zero, and must invalidate the projected tangent's sparse graph.
  eqType equation;
  equation.immersed_method = true;
  equation.immersed_fluid_domain = 0;
  com.eq.push_back(equation);
  SolutionStates sol;
  auto& displacement = sol.intermediate.get_displacement();
  displacement.resize(2,com.tnNo);
  displacement = 0.0;
  if (rank == 2) displacement(0,0) = 0.1;
  EXPECT_TRUE(ib::build_ifem_coupling_operator(com,sol));
  if (rank == 2) EXPECT_EQ(com.ib.ifemCoupling[0].fluidElemOwnerRank,1);

}

} // namespace
