// SPDX-FileCopyrightText: Copyright (c) Stanford University, The Regents of the University of California, and others.
// SPDX-License-Identifier: BSD-3-Clause

#include "ib.h"

#include "all_fun.h"
#include "lhsa.h"
#include "nn.h"
#include "utils.h"

#include "fluid.h"
#include "sv_struct.h"

#include <algorithm>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace ib {

constexpr int IFEM_GRAPH_ELEMENT_RING_LAYERS = 1;

namespace {

/// @brief Check whether a mesh has any element in the given domain.
bool mesh_has_domain(const mshType& mesh, const int domain_id) {
  for (int e = 0; e < mesh.nEl; e++) {
    if (mesh.eId.size() != 0 && utils::btest(mesh.eId(e), domain_id)) {
      return true;
    }
  }
  return false;
}


/// @brief Locate an immersed point in the background fluid mesh.
bool locate_background_fluid_trace(const ComMod& com_mod, const Vector<double>& x,
    const int fluid_domain, ifemCouplingType& trace) {
  const int nsd = com_mod.nsd;

  // Most immersed nodes move little between Newton iterations, so try the
  // previous containing element before scanning the full background mesh.
  if (trace.fluidMesh >= 0 && trace.fluidMesh < com_mod.nMsh) {
    const auto& mesh = com_mod.msh[trace.fluidMesh];
    const int e = trace.fluidElem;

    if (e >= 0 && e < mesh.nEl && mesh.eId.size() != 0 && utils::btest(mesh.eId(e), fluid_domain)) {
      const int eNoN = mesh.eNoN;
      Array<double> xl(nsd, eNoN);
      for (int a = 0; a < eNoN; a++) {
        const int Ac = mesh.IEN(a,e);
        for (int i = 0; i < nsd; i++) {
          xl(i,a) = com_mod.x(i,Ac);
        }
      }

      Vector<double> xi(nsd);
      xi = 0.0;
      if (mesh.xi.ncols() != 0) {
        for (int g = 0; g < mesh.xi.ncols(); g++) {
          for (int i = 0; i < nsd; i++) {
            xi(i) += mesh.xi(i,g);
          }
        }
        for (int i = 0; i < nsd; i++) {
          xi(i) /= static_cast<double>(mesh.xi.ncols());
        }
      }

      Vector<double> N(eNoN);
      Array<double> Nx(nsd, eNoN);
      try {
        nn::get_nnx(nsd, mesh.eType, eNoN, xl, mesh.xib, mesh.Nb, x, xi, N, Nx);
        trace.fluidNodes.resize(eNoN);
        trace.N.resize(eNoN);
        for (int a = 0; a < eNoN; a++) {
          trace.fluidNodes(a) = mesh.IEN(a,e);
          trace.N(a) = N(a);
        }
        return true;
      } catch (const std::exception&) {
      }
    }
  }

  for (int iM = 0; iM < com_mod.nMsh; iM++) {
    const auto& mesh = com_mod.msh[iM];
    const int eNoN = mesh.eNoN;
    Array<double> xl(nsd, eNoN);

    for (int e = 0; e < mesh.nEl; e++) {
      if (mesh.eId.size() == 0 || !utils::btest(mesh.eId(e), fluid_domain)) {
        continue;
      }

      for (int a = 0; a < eNoN; a++) {
        const int Ac = mesh.IEN(a,e);
        for (int i = 0; i < nsd; i++) {
          xl(i,a) = com_mod.x(i,Ac);
        }
      }

      Vector<double> xi(nsd);
      xi = 0.0;
      if (mesh.xi.ncols() != 0) {
        for (int g = 0; g < mesh.xi.ncols(); g++) {
          for (int i = 0; i < nsd; i++) {
            xi(i) += mesh.xi(i,g);
          }
        }
        for (int i = 0; i < nsd; i++) {
          xi(i) /= static_cast<double>(mesh.xi.ncols());
        }
      }
      Vector<double> N(eNoN);
      Array<double> Nx(nsd, eNoN);

      try {
        nn::get_nnx(nsd, mesh.eType, eNoN, xl, mesh.xib, mesh.Nb, x, xi, N, Nx);
      } catch (const std::exception&) {
        continue;
      }

      trace.fluidMesh = iM;
      trace.fluidElem = e;
      trace.fluidNodes.resize(eNoN);
      trace.N.resize(eNoN);
      for (int a = 0; a < eNoN; a++) {
        trace.fluidNodes(a) = mesh.IEN(a,e);
        trace.N(a) = N(a);
      }
      return true;
    }
  }

  return false;
}

}


/// @brief Replace immersed solid rows in the reduced linear system with identity rows.
void apply_ifem_reduced_solid_rows(ComMod& com_mod)
{
  const auto& ib_data = com_mod.ib;
  const int dof = com_mod.dof;
  auto& R = com_mod.R;
  auto& Val = com_mod.Val;

  // Solid increments are not solved directly; they are recovered from the
  // background-fluid increment after the linear solve.
  for (int a = 0; a < ib_data.tnNo; a++) {
    const int row_node = ib_data.gN(a);

    for (int i = 0; i < dof; i++) {
      R(i,row_node) = 0.0;
    }

    bool found_diagonal = false;
    for (int ptr = com_mod.rowPtr(row_node); ptr < com_mod.rowPtr(row_node + 1); ptr++) {
      const bool diagonal = com_mod.colPtr(ptr) == row_node;
      found_diagonal = found_diagonal || diagonal;

      for (int j = 0; j < dof; j++) {
        for (int i = 0; i < dof; i++) {
          Val(i + j*dof,ptr) = 0.0;
        }
      }

      if (diagonal) {
        for (int i = 0; i < dof; i++) {
          Val(i + i*dof,ptr) = 1.0;
        }
      }
    }

    if (!found_diagonal) {
      throw std::runtime_error("[ib::apply_ifem_reduced_solid_rows] Missing diagonal sparse entry for immersed solid row " +
          std::to_string(row_node) + ".");
    }
  }
}


/// @brief Project the solved background-fluid increment to immersed solid nodes.
void project_fluid_increment_to_solid(ComMod& com_mod)
{
  auto& ib_data = com_mod.ib;
  const int nsd = com_mod.nsd;
  const int dof = com_mod.dof;
  const Array<double> fluid_increment = com_mod.R;

  for (const auto& row : ib_data.ifemCoupling) {
    const int solid_node = ib_data.gN(row.ibNode);

    for (int i = 0; i < dof; i++) {
      com_mod.R(i,solid_node) = 0.0;
    }

    for (int a = 0; a < row.fluidNodes.size(); a++) {
      const int fluid_node = row.fluidNodes(a);
      const double Na = row.N(a);
      for (int i = 0; i < nsd; i++) {
        com_mod.R(i,solid_node) += Na * fluid_increment(i,fluid_node);
      }
    }
  }
}


/// @brief Build background-fluid interpolation rows for all immersed solid nodes.
void build_coupling_rows(ComMod& com_mod, const int fluid_domain, const std::string& caller)
{
  auto& ib_data = com_mod.ib;
  ib_data.ifemCoupling.resize(ib_data.tnNo);

  int missed_nodes = 0;
  int first_missed_node = -1;

  for (int a = 0; a < ib_data.tnNo; a++) {
    auto& row = ib_data.ifemCoupling[a];
    row.ibNode = a;

    Vector<double> x(com_mod.nsd);
    for (int i = 0; i < com_mod.nsd; i++) {
      x(i) = ib_data.x(i,a) + ib_data.Ubk(i,a);
    }
    if (locate_background_fluid_trace(com_mod, x, fluid_domain, row)) {
      continue;
    }

    missed_nodes++;
    if (first_missed_node < 0) {
      first_missed_node = a;
    }
  }

  if (missed_nodes != 0) {
    const int ib_node = first_missed_node;
    const auto& row = ib_data.ifemCoupling[ib_node];

    Vector<double> x(com_mod.nsd);
    for (int i = 0; i < com_mod.nsd; i++) {
      x(i) = ib_data.x(i,ib_node) + ib_data.Ubk(i,ib_node);
    }

    std::vector<double> fluid_min(com_mod.nsd, std::numeric_limits<double>::max());
    std::vector<double> fluid_max(com_mod.nsd, -std::numeric_limits<double>::max());
    int fluid_elements = 0;
    int nearest_mesh = -1;
    int nearest_elem = -1;
    double nearest_dist2 = std::numeric_limits<double>::max();
    std::vector<double> nearest_min(com_mod.nsd, 0.0);
    std::vector<double> nearest_max(com_mod.nsd, 0.0);
    int bbox_hits = 0;
    constexpr double bbox_tol = 1.0e-8;

    for (int iM = 0; iM < com_mod.nMsh; iM++) {
      const auto& mesh = com_mod.msh[iM];

      for (int e = 0; e < mesh.nEl; e++) {
        if (mesh.eId.size() == 0 || !utils::btest(mesh.eId(e), fluid_domain)) {
          continue;
        }

        fluid_elements++;
        std::vector<double> elem_min(com_mod.nsd, std::numeric_limits<double>::max());
        std::vector<double> elem_max(com_mod.nsd, -std::numeric_limits<double>::max());

        for (int a = 0; a < mesh.eNoN; a++) {
          const int Ac = mesh.IEN(a,e);
          for (int i = 0; i < com_mod.nsd; i++) {
            const double xi = com_mod.x(i,Ac);
            elem_min[i] = std::min(elem_min[i], xi);
            elem_max[i] = std::max(elem_max[i], xi);
            fluid_min[i] = std::min(fluid_min[i], xi);
            fluid_max[i] = std::max(fluid_max[i], xi);
          }
        }

        bool inside_bbox = true;
        double dist2 = 0.0;
        for (int i = 0; i < com_mod.nsd; i++) {
          if (x(i) < elem_min[i] - bbox_tol || x(i) > elem_max[i] + bbox_tol) {
            inside_bbox = false;
          }

          double di = 0.0;
          if (x(i) < elem_min[i]) {
            di = elem_min[i] - x(i);
          } else if (x(i) > elem_max[i]) {
            di = x(i) - elem_max[i];
          }
          dist2 += di * di;
        }

        if (inside_bbox) {
          bbox_hits++;
        }

        if (dist2 < nearest_dist2) {
          nearest_dist2 = dist2;
          nearest_mesh = iM;
          nearest_elem = e;
          nearest_min = elem_min;
          nearest_max = elem_max;
        }
      }
    }

    std::ostringstream msg;
    msg << "[ib::" << caller << "] Could not locate " << missed_nodes
        << " immersed nodes in the background fluid mesh.\n"
        << "  First missed IB node: " << ib_node << "\n"
        << "  Original global node id: " << ib_data.gN(ib_node) << "\n"
        << "  Current position x = (";
    for (int i = 0; i < com_mod.nsd; i++) {
      msg << x(i) << (i + 1 == com_mod.nsd ? "" : ", ");
    }
    msg << ")\n  Reference position X = (";
    for (int i = 0; i < com_mod.nsd; i++) {
      msg << ib_data.x(i,ib_node) << (i + 1 == com_mod.nsd ? "" : ", ");
    }
    msg << ")\n  Displacement U = (";
    for (int i = 0; i < com_mod.nsd; i++) {
      msg << ib_data.Ubk(i,ib_node) << (i + 1 == com_mod.nsd ? "" : ", ");
    }
    msg << ")\n  Previous fluid mesh/element = (" << row.fluidMesh << ", "
        << row.fluidElem << ")\n"
        << "  Fluid domain id = " << fluid_domain << "\n"
        << "  Fluid domain element count scanned = " << fluid_elements << "\n"
        << "  Fluid domain bounds min = (";
    for (int i = 0; i < com_mod.nsd; i++) {
      msg << fluid_min[i] << (i + 1 == com_mod.nsd ? "" : ", ");
    }
    msg << "), max = (";
    for (int i = 0; i < com_mod.nsd; i++) {
      msg << fluid_max[i] << (i + 1 == com_mod.nsd ? "" : ", ");
    }
    msg << ")\n  Number of fluid element bounding boxes containing x = "
        << bbox_hits << "\n"
        << "  Nearest fluid mesh/element by bounding box = (" << nearest_mesh
        << ", " << nearest_elem << ")\n"
        << "  Nearest element bbox min = (";
    for (int i = 0; i < com_mod.nsd; i++) {
      msg << nearest_min[i] << (i + 1 == com_mod.nsd ? "" : ", ");
    }
    msg << "), max = (";
    for (int i = 0; i < com_mod.nsd; i++) {
      msg << nearest_max[i] << (i + 1 == com_mod.nsd ? "" : ", ");
    }
    msg << ")\n  Squared distance to nearest bbox = " << nearest_dist2 << "\n"
        << "  Interpretation: bbox_hits > 0 usually means nn::get_nnx rejected"
        << " the point by natural-coordinate/shape-function tolerance; "
        << "bbox_hits == 0 means the current point is outside all fluid element"
        << " bounding boxes.";

    throw std::runtime_error(msg.str());
  }
}

/// @brief Return the local VMS stabilization multiplier near the immersed surface.
double ib_vms_stabilization_s(const ComMod& com_mod, const Vector<double>& x, const double h)
{
  if (!com_mod.ibFlag) {
    return 1.0;
  }

  const auto& eq = com_mod.eq[com_mod.cEq];
  const double s_shell = eq.immersed_vms_stabilization_s;
  const double width = eq.immersed_vms_stabilization_width;
  if (s_shell <= 1.0 || width <= 0.0 || h <= 0.0) {
    return 1.0;
  }

  const auto& ib_data = com_mod.ib;
  if (ib_data.tnNo == 0 || ib_data.x.ncols() != ib_data.tnNo) {
    return 1.0;
  }

  const int nsd = com_mod.nsd;
  const double radius = width * h;
  const double radius2 = radius * radius;

  for (int a = 0; a < ib_data.tnNo; a++) {
    double dist2 = 0.0;

    for (int i = 0; i < nsd; i++) {
      const double xs = ib_data.x(i,a) + ib_data.Ubk(i,a);
      const double dx = x(i) - xs;
      dist2 += dx * dx;
    }

    if (dist2 <= radius2) {
      return s_shell;
    }
  }

  return 1.0;
}


/// @brief Build the immersed-solid mesh view from the configured solid domain.
void initialize_immersed_meshes(ComMod& com_mod) {
  int solid_domain = -1;
  for (const auto& eq : com_mod.eq) {
    if (eq.immersed_method) {
      solid_domain = eq.immersed_solid_domain;
      break;
    }
  }

  if (solid_domain < 0) {
    return;
  }

  const int nsd = com_mod.nsd;
  auto& ib_data = com_mod.ib;
  ib_data.nMsh = 0;
  ib_data.tnNo = 0;
  ib_data.msh.clear();

  std::vector<int> immersed_mesh_indices;
  std::unordered_map<int,int> global_to_ib;
  std::vector<int> ib_global_nodes;

  for (int iM = 0; iM < com_mod.nMsh; iM++) {
    const auto& mesh = com_mod.msh[iM];
    if (!mesh_has_domain(mesh, solid_domain)) {
      continue;
    }

    immersed_mesh_indices.push_back(iM);
    for (int a = 0; a < mesh.nNo; a++) {
      const int Ac = mesh.gN(a);
      if (global_to_ib.find(Ac) == global_to_ib.end()) {
        const int ib_node = static_cast<int>(ib_global_nodes.size());
        global_to_ib[Ac] = ib_node;
        ib_global_nodes.push_back(Ac);
      }
    }
  }


  if (immersed_mesh_indices.empty()) {
    throw std::runtime_error("[ib::initialize_immersed_meshes] No mesh elements were found for immersed solid domain " +
        std::to_string(solid_domain) + ".");
  }

  ib_data.nMsh = static_cast<int>(immersed_mesh_indices.size());
  ib_data.tnNo = static_cast<int>(ib_global_nodes.size());
  ib_data.dmnID.resize(ib_data.nMsh);
  ib_data.gN.resize(ib_data.tnNo);
  ib_data.x.resize(nsd, ib_data.tnNo);

  for (int a = 0; a < ib_data.tnNo; a++) {
    const int Ac = ib_global_nodes[a];
    ib_data.gN(a) = Ac;
    for (int i = 0; i < nsd; i++) {
      ib_data.x(i,a) = com_mod.x(i,Ac);
    }
  }

  ib_data.msh.reserve(ib_data.nMsh);
  for (int local_mesh = 0; local_mesh < ib_data.nMsh; local_mesh++) {
    const int iM = immersed_mesh_indices[local_mesh];
    const auto& src = com_mod.msh[iM];
    mshType dst = src;
    dst.iGC.clear();
    dst.gN.resize(src.nNo);
    dst.gnNo = src.nNo;
    dst.gnEl = src.nEl;
    dst.nNo = src.nNo;

    for (int a = 0; a < src.nNo; a++) {
      dst.gN(a) = global_to_ib.at(src.gN(a));
    }

    dst.gIEN.clear();
    dst.gpN.clear();
    dst.otnIEN.clear();
    for (int e = 0; e < dst.IEN.ncols(); e++) {
      for (int a = 0; a < dst.IEN.nrows(); a++) {
        dst.IEN(a,e) = global_to_ib.at(dst.IEN(a,e));
      }
    }

    dst.lN.resize(ib_data.tnNo);
    dst.lN = -1;
    for (int a = 0; a < dst.gN.size(); a++) {
      dst.lN(dst.gN(a)) = a;
    }

    for (int iFa = 0; iFa < dst.nFa; iFa++) {
      auto& face = dst.fa[iFa];
      face.iM = local_mesh;
      for (int a = 0; a < face.gN.size(); a++) {
        face.gN(a) = global_to_ib.at(face.gN(a));
      }
      for (int e = 0; e < face.IEN.ncols(); e++) {
        for (int a = 0; a < face.IEN.nrows(); a++) {
          face.IEN(a,e) = global_to_ib.at(face.IEN(a,e));
        }
      }
      face.lN.resize(ib_data.tnNo);
      face.lN = -1;
      for (int a = 0; a < face.gN.size(); a++) {
        face.lN(face.gN(a)) = a;
      }
    }

    ib_data.dmnID(local_mesh) = solid_domain;
    ib_data.msh.push_back(std::move(dst));
  }

  ib_data.Yb.resize(nsd, ib_data.tnNo);
  ib_data.Auo.resize(nsd, ib_data.tnNo);
  ib_data.Aun.resize(nsd, ib_data.tnNo);
  ib_data.Auk.resize(nsd, ib_data.tnNo);
  ib_data.Ubo.resize(nsd, ib_data.tnNo);
  ib_data.Ubn.resize(nsd, ib_data.tnNo);
  ib_data.Ubk.resize(nsd, ib_data.tnNo);
  ib_data.R.resize(nsd, ib_data.tnNo);

  ib_data.Yb = 0.0;
  ib_data.Auo = 0.0;
  ib_data.Aun = 0.0;
  ib_data.Auk = 0.0;
  ib_data.Ubo = 0.0;
  ib_data.Ubn = 0.0;
  ib_data.Ubk = 0.0;
  ib_data.R = 0.0;
}


/// @brief Update IFEM interpolation rows and report whether the sparse graph changed.
bool build_ifem_coupling_operator(ComMod& com_mod, const SolutionStates& solutions) {
  auto& ib_data = com_mod.ib;
  int fluid_domain = -1;
  for (const auto& eq : com_mod.eq) {
    if (eq.immersed_method) {
      fluid_domain = eq.immersed_fluid_domain;
      break;
    }
  }
  const auto& displacement = solutions.intermediate.get_displacement();

  std::vector<int> old_fluid_mesh(ib_data.tnNo, -1);
  std::vector<int> old_fluid_elem(ib_data.tnNo, -1);
  const bool previous_coupling_exists = ib_data.ifemCoupling.size() == static_cast<std::size_t>(ib_data.tnNo);
  if (previous_coupling_exists) {
    for (int a = 0; a < ib_data.tnNo; a++) {
      old_fluid_mesh[a] = ib_data.ifemCoupling[a].fluidMesh;
      old_fluid_elem[a] = ib_data.ifemCoupling[a].fluidElem;
    }
  }

  for (int a = 0; a < ib_data.tnNo; a++) {
    const int Ac = ib_data.gN(a);
    for (int i = 0; i < com_mod.nsd; i++) {
      ib_data.Ubk(i,a) = displacement(i,Ac);
    }
  }

  build_coupling_rows(com_mod, fluid_domain, "build_ifem_coupling_operator");

  if (!previous_coupling_exists) {
    return true;
  }
  for (int a = 0; a < ib_data.tnNo; a++) {
    if (old_fluid_mesh[a] != ib_data.ifemCoupling[a].fluidMesh ||
        old_fluid_elem[a] != ib_data.ifemCoupling[a].fluidElem) {
      return true;
    }
  }
  return false;
}


/// @brief Add sparse graph entries required by the projected immersed-solid tangent.
void add_ifem_coupling_to_lhs_graph(ComMod& com_mod, int& mnnzeic, Array<int>& uInd)
{
  auto& ib_data = com_mod.ib;
  int fluid_domain = -1;
  int solid_domain = -1;
  for (const auto& eq : com_mod.eq) {
    if (eq.immersed_method) {
      fluid_domain = eq.immersed_fluid_domain;
      solid_domain = eq.immersed_solid_domain;
      break;
    }
  }
  if (ib_data.ifemCoupling.size() != static_cast<std::size_t>(ib_data.tnNo)) {
    build_coupling_rows(com_mod, fluid_domain, "add_ifem_coupling_to_lhs_graph");
  }

  std::vector<std::vector<std::vector<int>>> fluid_mesh_neighbors(com_mod.nMsh);
  std::vector<int> fluid_mesh_neighbors_built(com_mod.nMsh, 0);
  std::vector<std::vector<int>> graph_nodes(ib_data.tnNo);

  // The projected solid tangent couples nearby fluid interpolation supports, so
  // these graph entries must exist before element assembly writes matrix values.
  for (int ib_node = 0; ib_node < ib_data.tnNo; ib_node++) {
    const auto& row = ib_data.ifemCoupling[ib_node];
    const int fluid_mesh_id = row.fluidMesh;
    const auto& fluid_mesh = com_mod.msh[fluid_mesh_id];

    if (fluid_mesh_neighbors_built[fluid_mesh_id] == 0) {
      std::vector<std::vector<int>> node_to_elements(com_mod.tnNo);
      fluid_mesh_neighbors[fluid_mesh_id].resize(fluid_mesh.nEl);

      for (int e = 0; e < fluid_mesh.nEl; e++) {
        if (fluid_mesh.eId.size() == 0 || !utils::btest(fluid_mesh.eId(e), fluid_domain)) {
          continue;
        }
        for (int a = 0; a < fluid_mesh.eNoN; a++) {
          node_to_elements[fluid_mesh.IEN(a,e)].push_back(e);
        }
      }

      for (int e = 0; e < fluid_mesh.nEl; e++) {
        if (fluid_mesh.eId.size() == 0 || !utils::btest(fluid_mesh.eId(e), fluid_domain)) {
          continue;
        }
        for (int a = 0; a < fluid_mesh.eNoN; a++) {
          const int fluid_node = fluid_mesh.IEN(a,e);
          for (const int neighbor : node_to_elements[fluid_node]) {
            if (neighbor != e &&
                std::find(fluid_mesh_neighbors[fluid_mesh_id][e].begin(),
                    fluid_mesh_neighbors[fluid_mesh_id][e].end(), neighbor) ==
                    fluid_mesh_neighbors[fluid_mesh_id][e].end()) {
              fluid_mesh_neighbors[fluid_mesh_id][e].push_back(neighbor);
            }
          }
        }
      }

      fluid_mesh_neighbors_built[fluid_mesh_id] = 1;
    }

    std::vector<int> frontier{row.fluidElem};
    std::vector<int> visited(fluid_mesh.nEl, 0);
    visited[row.fluidElem] = 1;

    for (int layer = 0; layer <= IFEM_GRAPH_ELEMENT_RING_LAYERS && !frontier.empty(); layer++) {
      std::vector<int> next_frontier;

      for (const int e : frontier) {
        for (int a = 0; a < fluid_mesh.eNoN; a++) {
          const int fluid_node = fluid_mesh.IEN(a,e);
          if (std::find(graph_nodes[ib_node].begin(), graph_nodes[ib_node].end(), fluid_node) ==
              graph_nodes[ib_node].end()) {
            graph_nodes[ib_node].push_back(fluid_node);
          }
        }

        if (layer == IFEM_GRAPH_ELEMENT_RING_LAYERS) {
          continue;
        }

        for (const int neighbor : fluid_mesh_neighbors[fluid_mesh_id][e]) {
          if (visited[neighbor] == 0) {
            visited[neighbor] = 1;
            next_frontier.push_back(neighbor);
          }
        }
      }

      frontier = std::move(next_frontier);
    }
  }

  for (const auto& ib_mesh : ib_data.msh) {
    if (!mesh_has_domain(ib_mesh, solid_domain)) {
      continue;
    }

    for (int e = 0; e < ib_mesh.nEl; e++) {
      for (int a = 0; a < ib_mesh.eNoN; a++) {
        const int ib_a = ib_mesh.IEN(a,e);
        for (const int fluid_A : graph_nodes[ib_a]) {

          for (int b = 0; b < ib_mesh.eNoN; b++) {
            const int ib_b = ib_mesh.IEN(b,e);
            for (const int fluid_B : graph_nodes[ib_b]) {
              lhsa_ns::add_col(com_mod.tnNo, fluid_A, fluid_B, mnnzeic, uInd);
            }
          }
        }
      }
    }
  }
}


/// @brief Interpolate background-fluid velocity to immersed solid nodes.
void project_fluid_velocity_to_solid(ComMod& com_mod, SolutionStates& solutions) {
  const int nsd = com_mod.nsd;
  auto& ib_data = com_mod.ib;
  auto& velocity = solutions.current.get_velocity();

  ib_data.Yb = 0.0;

  for (const auto& row : ib_data.ifemCoupling) {
    for (int a = 0; a < row.fluidNodes.size(); a++) {
      const int Ac = row.fluidNodes(a);
      const double Na = row.N(a);
      for (int i = 0; i < nsd; i++) {
        ib_data.Yb(i,row.ibNode) += Na * velocity(i,Ac);
      }
    }

    const int solid_node = ib_data.gN(row.ibNode);
    for (int i = 0; i < nsd; i++) {
      velocity(i,solid_node) = ib_data.Yb(i,row.ibNode);
    }
  }
}


/// @brief Assemble immersed FSI by projecting the solid residual/tangent into the fluid system.
void construct_immersed_fsi(ComMod& com_mod, CepMod& cep_mod, const mshType& lM, const SolutionStates& solutions)
{
  const auto& Ag = solutions.intermediate.get_acceleration();
  const auto& Yg = solutions.intermediate.get_velocity();
  const auto& Dg = solutions.intermediate.get_displacement();

  using namespace consts;

  int fluid_domain = -1;
  int solid_domain = -1;
  for (const auto& eq : com_mod.eq) {
    if (eq.immersed_method) {
      fluid_domain = eq.immersed_fluid_domain;
      solid_domain = eq.immersed_solid_domain;
      break;
    }
  }

  if (fluid_domain < 0 || solid_domain < 0) {
    throw std::runtime_error("[ib::construct_immersed_fsi] Immersed fluid/solid domains are not initialized.");
  }

  if (!mesh_has_domain(lM, fluid_domain)) {
    return;
  }

  fluid::construct_fluid(com_mod, lM, solutions);

  int current_mesh = -1;
  int first_fluid_mesh = -1;
  for (int iM = 0; iM < com_mod.nMsh; iM++) {
    const auto& mesh = com_mod.msh[iM];
    if (&mesh == &lM || (mesh.dname == lM.dname && mesh.nEl == lM.nEl && mesh.nNo == lM.nNo)) {
      current_mesh = iM;
    }
    if (first_fluid_mesh < 0 && mesh_has_domain(mesh, fluid_domain)) {
      first_fluid_mesh = iM;
    }
  }
  // Fluid assembly is done for each fluid mesh, but the immersed solid force
  // projection is global and should be added only once.
  if (current_mesh != first_fluid_mesh) {
    return;
  }

  const int nsd = com_mod.nsd;
  const int tDof = com_mod.tDof;
  const int dof = com_mod.dof;
  const int cEq = com_mod.cEq;
  const auto& eq = com_mod.eq[cEq];
  auto& cDmn = com_mod.cDmn;
  const int nsymd = com_mod.nsymd;
  auto& pS0 = com_mod.pS0;
  auto& pSn = com_mod.pSn;
  auto& pSa = com_mod.pSa;
  bool pstEq = com_mod.pstEq;

  auto& ib_data = com_mod.ib;

  ib_data.R = 0.0;
  ib_data.Auk = 0.0;
  ib_data.Yb = 0.0;

  for (const auto& row : ib_data.ifemCoupling) {
    for (int a = 0; a < row.fluidNodes.size(); a++) {
      const int Ac = row.fluidNodes(a);
      const double Na = row.N(a);
      for (int i = 0; i < nsd; i++) {
        ib_data.Auk(i,row.ibNode) += Na * Ag(i,Ac);
        ib_data.Yb(i,row.ibNode) += Na * Yg(i,Ac);
      }
    }
  }

  for (int a = 0; a < ib_data.tnNo; a++) {
    const int Ac = ib_data.gN(a);
    for (int i = 0; i < nsd; i++) {
      ib_data.Ubk(i,a) = Dg(i,Ac);
    }
  }

  auto& cem = cep_mod.cem;

  for (const auto& ib_mesh : ib_data.msh) {
    if (!mesh_has_domain(ib_mesh, solid_domain)) {
      continue;
    }

    const int eNoN = ib_mesh.eNoN;
    int nFn = ib_mesh.nFn;
    if (nFn == 0) {
      nFn = 1;
    }

    Vector<int> ptr(eNoN);
    Vector<double> pSl(nsymd), ya_l_f(eNoN), ya_l_s(eNoN), ya_l_n(eNoN), N(eNoN);
    Array<double> xl(nsd,eNoN), al(tDof,eNoN), yl(tDof,eNoN), dl(tDof,eNoN),
        bfl(nsd,eNoN), fN(nsd,nFn), pS0l(nsymd,eNoN), Nx(nsd,eNoN), lR(dof,eNoN);
    Array3<double> lK(dof*dof,eNoN,eNoN);

    for (int e = 0; e < ib_mesh.nEl; e++) {
      cDmn = all_fun::domain(com_mod, ib_mesh, cEq, e);
      auto cPhys = eq.dmn[cDmn].phys;
      if (cPhys != EquationType::phys_struct) {
        continue;
      }

      fN = 0.0;
      pS0l = 0.0;
      ya_l_f = 0.0;
      ya_l_s = 0.0;
      ya_l_n = 0.0;

      for (int a = 0; a < eNoN; a++) {
        const int ib_Ac = ib_mesh.IEN(a,e);
        const int global_Ac = ib_data.gN(ib_Ac);
        ptr(a) = ib_Ac;

        for (int i = 0; i < nsd; i++) {
          xl(i,a) = ib_data.x(i,ib_Ac);
          bfl(i,a) = com_mod.Bf(i,global_Ac);
        }

        for (int i = 0; i < tDof; i++) {
          al(i,a) = 0.0;
          yl(i,a) = 0.0;
          dl(i,a) = 0.0;
        }
        for (int i = 0; i < nsd; i++) {
          al(i,a) = ib_data.Auk(i,ib_Ac);
          yl(i,a) = ib_data.Yb(i,ib_Ac);
          dl(i,a) = ib_data.Ubk(i,ib_Ac);
        }

        if (ib_mesh.fN.size() != 0) {
          for (int iFn = 0; iFn < nFn; iFn++) {
            for (int i = 0; i < nsd; i++) {
              fN(i,iFn) = ib_mesh.fN(i+nsd*iFn,e);
            }
          }
        }

        if (pS0.size() != 0) {
          pS0l.set_col(a, pS0.col(global_Ac));
        }

        if (cem.cpld) {
          ya_l_f(a) = cem.Ya_f[global_Ac];
          ya_l_s(a) = cem.Ya_s[global_Ac];
          ya_l_n(a) = cem.Ya_n[global_Ac];
        }
      }

      lR = 0.0;
      lK = 0.0;

      double Jac{0.0};
      Array<double> ksix(nsd,nsd);

      for (int g = 0; g < ib_mesh.nG; g++) {
        if (g == 0 || !ib_mesh.lShpF) {
          auto Nx_g = ib_mesh.Nx.slice(g);
          nn::gnn(eNoN, nsd, nsd, Nx_g, xl, Nx, Jac, ksix);
          if (utils::is_zero(Jac)) {
            throw std::runtime_error("[ib::construct_immersed_fsi] Jacobian for immersed solid element " +
                std::to_string(e) + " is < 0.");
          }
        }

        const double w = ib_mesh.w(g) * Jac;
        N = ib_mesh.N.col(g);
        pSl = 0.0;

        if (nsd == 3) {
          struct_ns::struct_3d(com_mod, cep_mod, eNoN, nFn, w, N, Nx, al, yl, dl, bfl, fN, pS0l, pSl,
                               ya_l_f, ya_l_s, ya_l_n, lR, lK);
        } else if (nsd == 2) {
          struct_ns::struct_2d(com_mod, cep_mod, eNoN, nFn, w, N, Nx, al, yl, dl, bfl, fN, pS0l, pSl,
                               ya_l_f, ya_l_s, ya_l_n, lR, lK);
        }

        if (pstEq) {
          for (int a = 0; a < eNoN; a++) {
            const int global_Ac = ib_data.gN(ptr(a));
            pSa(global_Ac) = pSa(global_Ac) + w*N(a);
            for (int i = 0; i < pSn.nrows(); i++) {
              pSn(i,global_Ac) = pSn(i,global_Ac) + w*N(a)*pSl(i);
            }
          }
        }
      }

      for (int a = 0; a < eNoN; a++) {
        const int ib_a = ptr(a);
        const auto& row_a = ib_data.ifemCoupling[ib_a];

        for (int i = 0; i < nsd; i++) {
          ib_data.R(i,ib_a) += lR(i,a);
        }

        for (int A = 0; A < row_a.fluidNodes.size(); A++) {
          const int fluid_A = row_a.fluidNodes(A);
          const double NA = row_a.N(A);

          // Spread solid residual/tangent with the IFEM operator: C^T R_s and
          // C^T K_s C.
          for (int i = 0; i < nsd; i++) {
            com_mod.R(i,fluid_A) += NA * lR(i,a);
          }

          for (int b = 0; b < eNoN; b++) {
            const int ib_b = ptr(b);
            const auto& row_b = ib_data.ifemCoupling[ib_b];

            for (int B = 0; B < row_b.fluidNodes.size(); B++) {
              const int fluid_B = row_b.fluidNodes(B);
              const double NB = row_b.N(B);

              int matrix_ptr = -1;
              int left = com_mod.rowPtr(fluid_A);
              int right = com_mod.rowPtr(fluid_A + 1) - 1;
              while (left <= right) {
                const int candidate = (left + right) / 2;
                if (com_mod.colPtr(candidate) == fluid_B) {
                  matrix_ptr = candidate;
                  break;
                }
                if (com_mod.colPtr(candidate) < fluid_B) {
                  left = candidate + 1;
                } else {
                  right = candidate - 1;
                }
              }

              if (matrix_ptr < 0) {
                throw std::runtime_error("[ib::construct_immersed_fsi] Missing sparse matrix entry for IFEM projected tangent.");
              }

              for (int j = 0; j < nsd; j++) {
                for (int i = 0; i < nsd; i++) {
                  com_mod.Val(i + j*dof,matrix_ptr) += NA * lK(i + j*dof,a,b) * NB;
                }
              }
            }
          }
        }
      }
    }
  }

}

}
