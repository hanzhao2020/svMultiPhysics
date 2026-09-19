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


int local_to_global_node(const ComMod& com_mod, const int local_node)
{
  if (local_node >= 0 && local_node < com_mod.ltg.size()) {
    return com_mod.ltg(local_node);
  }
  return local_node;
}


int global_to_local_node(const ComMod& com_mod, const int global_node)
{
  for (int a = 0; a < com_mod.ltg.size(); a++) {
    if (com_mod.ltg(a) == global_node) {
      return a;
    }
  }
  return -1;
}


int local_to_global_element(const ComMod& com_mod, const mshType& mesh, const int elem)
{
  const int rank = com_mod.cm.idcm();
  if (rank >= 0 && rank + 1 < mesh.eDist.size()) {
    return mesh.eDist(rank) + elem;
  }
  return elem;
}


void set_fluid_trace_nodes(const ComMod& com_mod, const mshType& mesh, const int elem,
    const Vector<double>& N, ifemCouplingType& trace)
{
  const int eNoN = mesh.eNoN;
  trace.fluidElem = elem;
  trace.fluidElemGlobal = local_to_global_element(com_mod, mesh, elem);
  trace.fluidElemOwnerRank = com_mod.cm.idcm();
  trace.fluidNodes.resize(eNoN);
  trace.fluidLocalNodes.resize(eNoN);
  trace.fluidGlobalNodes.resize(eNoN);
  trace.N.resize(eNoN);

  for (int a = 0; a < eNoN; a++) {
    const int local_node = mesh.IEN(a,elem);
    trace.fluidNodes(a) = local_node;
    trace.fluidLocalNodes(a) = local_node;
    trace.fluidGlobalNodes(a) = local_to_global_node(com_mod, local_node);
    trace.N(a) = N(a);
  }
}


void clear_fluid_trace_nodes(ifemCouplingType& trace)
{
  trace.fluidMesh = -1;
  trace.fluidElem = -1;
  trace.fluidElemGlobal = -1;
  trace.fluidElemOwnerRank = -1;
  trace.fluidNodes.clear();
  trace.fluidLocalNodes.clear();
  trace.fluidGlobalNodes.clear();
  trace.N.clear();
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
        set_fluid_trace_nodes(com_mod, mesh, e, N, trace);
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
      set_fluid_trace_nodes(com_mod, mesh, e, N, trace);
      return true;
    }
  }

  return false;
}


void allgatherv_int(const ComMod& com_mod, const std::vector<int>& local,
    std::vector<int>& global)
{
  const int nTasks = com_mod.cm.np();
  std::vector<int> counts(nTasks, 0);
  std::vector<int> displs(nTasks, 0);
  int local_count = static_cast<int>(local.size());

  MPI_Allgather(&local_count, 1, cm_mod::mpint, counts.data(), 1, cm_mod::mpint, com_mod.cm.com());

  int total = 0;
  for (int i = 0; i < nTasks; i++) {
    displs[i] = total;
    total += counts[i];
  }

  global.resize(total);
  MPI_Allgatherv(local.data(), local_count, cm_mod::mpint, global.data(), counts.data(),
      displs.data(), cm_mod::mpint, com_mod.cm.com());
}


void allgatherv_double(const ComMod& com_mod, const std::vector<double>& local,
    std::vector<double>& global)
{
  const int nTasks = com_mod.cm.np();
  std::vector<int> counts(nTasks, 0);
  std::vector<int> displs(nTasks, 0);
  int local_count = static_cast<int>(local.size());

  MPI_Allgather(&local_count, 1, cm_mod::mpint, counts.data(), 1, cm_mod::mpint, com_mod.cm.com());

  int total = 0;
  for (int i = 0; i < nTasks; i++) {
    displs[i] = total;
    total += counts[i];
  }

  global.resize(total);
  MPI_Allgatherv(local.data(), local_count, cm_mod::mpreal, global.data(), counts.data(),
      displs.data(), cm_mod::mpreal, com_mod.cm.com());
}


void update_vms_stabilization_points(ComMod& com_mod)
{
  auto& ib_data = com_mod.ib;
  const int nsd = com_mod.nsd;
  std::vector<double> local_points;
  local_points.reserve(nsd * ib_data.tnNo);

  for (int a = 0; a < ib_data.tnNo; a++) {
    for (int i = 0; i < nsd; i++) {
      local_points.push_back(ib_data.x(i,a) + ib_data.Ubk(i,a));
    }
  }

  std::vector<double> points;
  if (com_mod.cm.seq()) {
    points = std::move(local_points);
  } else {
    allgatherv_double(com_mod, local_points, points);
  }

  const int n_points = static_cast<int>(points.size()) / nsd;
  ib_data.vmsX.resize(nsd, n_points);
  for (int a = 0; a < n_points; a++) {
    for (int i = 0; i < nsd; i++) {
      ib_data.vmsX(i,a) = points[nsd*a + i];
    }
  }
}


void resolve_missed_coupling_rows_parallel(ComMod& com_mod, const int fluid_domain)
{
  auto& ib_data = com_mod.ib;
  const int rank = com_mod.cm.idcm();
  const int nsd = com_mod.nsd;
  ib_data.ifemRemoteCoupling.clear();

  std::vector<int> local_query_meta;
  std::vector<double> local_query_x;

  for (int a = 0; a < ib_data.tnNo; a++) {
    const auto& row = ib_data.ifemCoupling[a];
    if (row.fluidElemOwnerRank >= 0) {
      continue;
    }

    local_query_meta.push_back(rank);
    local_query_meta.push_back(a);
    local_query_meta.push_back(row.ibGlobalNode);

    for (int i = 0; i < nsd; i++) {
      local_query_x.push_back(ib_data.x(i,a) + ib_data.Ubk(i,a));
    }
  }

  std::vector<int> query_meta;
  std::vector<double> query_x;
  allgatherv_int(com_mod, local_query_meta, query_meta);
  allgatherv_double(com_mod, local_query_x, query_x);

  std::vector<int> local_result_meta;
  std::vector<double> local_result_N;
  const int n_queries = static_cast<int>(query_meta.size()) / 3;

  for (int q = 0; q < n_queries; q++) {
    const int owner_rank = query_meta[3*q];
    const int owner_ib_node = query_meta[3*q + 1];
    const int owner_ib_global_node = query_meta[3*q + 2];

    Vector<double> x(nsd);
    for (int i = 0; i < nsd; i++) {
      x(i) = query_x[nsd*q + i];
    }

    ifemCouplingType trace;
    if (!locate_background_fluid_trace(com_mod, x, fluid_domain, trace)) {
      continue;
    }

    const int n_nodes = trace.fluidGlobalNodes.size();
    local_result_meta.push_back(owner_rank);
    local_result_meta.push_back(owner_ib_node);
    local_result_meta.push_back(owner_ib_global_node);
    local_result_meta.push_back(trace.fluidMesh);
    local_result_meta.push_back(trace.fluidElem);
    local_result_meta.push_back(trace.fluidElemGlobal);
    local_result_meta.push_back(trace.fluidElemOwnerRank);
    local_result_meta.push_back(n_nodes);

    for (int a = 0; a < n_nodes; a++) {
      local_result_meta.push_back(trace.fluidLocalNodes(a));
    }
    for (int a = 0; a < n_nodes; a++) {
      local_result_meta.push_back(trace.fluidGlobalNodes(a));
      local_result_N.push_back(trace.N(a));
    }
  }

  std::vector<int> result_meta;
  std::vector<double> result_N;
  allgatherv_int(com_mod, local_result_meta, result_meta);
  allgatherv_double(com_mod, local_result_N, result_N);

  int meta_pos = 0;
  int n_pos = 0;
  while (meta_pos < result_meta.size()) {
    const int owner_rank = result_meta[meta_pos++];
    const int owner_ib_node = result_meta[meta_pos++];
    const int owner_ib_global_node = result_meta[meta_pos++];
    const int fluid_mesh = result_meta[meta_pos++];
    const int fluid_elem = result_meta[meta_pos++];
    const int fluid_elem_global = result_meta[meta_pos++];
    const int fluid_elem_owner = result_meta[meta_pos++];
    const int n_nodes = result_meta[meta_pos++];

    std::vector<int> fluid_local_nodes(n_nodes);
    std::vector<int> fluid_global_nodes(n_nodes);
    for (int a = 0; a < n_nodes; a++) {
      fluid_local_nodes[a] = result_meta[meta_pos++];
    }
    for (int a = 0; a < n_nodes; a++) {
      fluid_global_nodes[a] = result_meta[meta_pos++];
    }

    if (fluid_elem_owner == rank && owner_rank != rank) {
      ifemCouplingType remote_row;
      remote_row.ibNode = owner_ib_node;
      remote_row.ibGlobalNode = owner_ib_global_node;
      remote_row.fluidMesh = fluid_mesh;
      remote_row.fluidElem = fluid_elem;
      remote_row.fluidElemGlobal = fluid_elem_global;
      remote_row.fluidElemOwnerRank = fluid_elem_owner;
      remote_row.fluidNodes.resize(n_nodes);
      remote_row.fluidLocalNodes.resize(n_nodes);
      remote_row.fluidGlobalNodes.resize(n_nodes);
      remote_row.N.resize(n_nodes);

      for (int a = 0; a < n_nodes; a++) {
        remote_row.fluidNodes(a) = fluid_local_nodes[a];
        remote_row.fluidLocalNodes(a) = fluid_local_nodes[a];
        remote_row.fluidGlobalNodes(a) = fluid_global_nodes[a];
        remote_row.N(a) = result_N[n_pos + a];
      }

      ib_data.ifemRemoteCoupling.push_back(std::move(remote_row));
    }

    if (owner_rank != rank || owner_ib_node < 0 || owner_ib_node >= ib_data.tnNo) {
      n_pos += n_nodes;
      continue;
    }

    auto& row = ib_data.ifemCoupling[owner_ib_node];
    if (row.fluidElemOwnerRank >= 0) {
      n_pos += n_nodes;
      continue;
    }

    row.ibGlobalNode = owner_ib_global_node;
    row.fluidMesh = fluid_mesh;
    row.fluidElem = fluid_elem;
    row.fluidElemGlobal = fluid_elem_global;
    row.fluidElemOwnerRank = fluid_elem_owner;
    row.fluidNodes.resize(n_nodes);
    row.fluidLocalNodes.resize(n_nodes);
    row.fluidGlobalNodes.resize(n_nodes);
    row.N.resize(n_nodes);

    for (int a = 0; a < n_nodes; a++) {
      row.fluidNodes(a) = fluid_local_nodes[a];
      row.fluidLocalNodes(a) = fluid_local_nodes[a];
      row.fluidGlobalNodes(a) = fluid_global_nodes[a];
      row.N(a) = result_N[n_pos + a];
    }

    n_pos += n_nodes;
  }
}


std::vector<int> collect_fluid_graph_nodes(ComMod& com_mod, const int fluid_domain,
    const ifemCouplingType& row, std::vector<std::vector<std::vector<int>>>& fluid_mesh_neighbors,
    std::vector<int>& fluid_mesh_neighbors_built)
{
  const int fluid_mesh_id = row.fluidMesh;
  const auto& fluid_mesh = com_mod.msh[fluid_mesh_id];
  std::vector<int> graph_nodes;

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
        if (std::find(graph_nodes.begin(), graph_nodes.end(), fluid_node) == graph_nodes.end()) {
          graph_nodes.push_back(fluid_node);
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

  return graph_nodes;
}


void add_ifem_graph_entries(ComMod& com_mod, const std::vector<std::vector<int>>& supports,
    const std::vector<int>& owned_rows, int& mnnzeic, Array<int>& uInd)
{
  for (int a = 0; a < supports.size(); a++) {
    if (owned_rows[a] == 0) {
      continue;
    }

    const auto& support_A = supports[a];
    for (const int fluid_A : support_A) {
      for (const auto& support_B : supports) {
        for (const int fluid_B : support_B) {
          lhsa_ns::add_col(com_mod.tnNo, fluid_A, fluid_B, mnnzeic, uInd);
        }
      }
    }
  }
}


void add_ifem_graph_entries(ComMod& com_mod, const std::vector<std::vector<int>>& supports,
    int& mnnzeic, Array<int>& uInd)
{
  std::vector<int> owned_rows(supports.size(), 1);
  add_ifem_graph_entries(com_mod, supports, owned_rows, mnnzeic, uInd);
}


std::vector<int> map_support_to_local(const ComMod& com_mod, const ifemCouplingType& row,
    const std::string& caller)
{
  std::vector<int> support;
  support.reserve(row.fluidGlobalNodes.size());

  for (int a = 0; a < row.fluidGlobalNodes.size(); a++) {
    const int local_node = global_to_local_node(com_mod, row.fluidGlobalNodes(a));
    if (local_node < 0) {
      throw std::runtime_error("[ib::" + caller + "] Missing local or ghost fluid node " +
          std::to_string(row.fluidGlobalNodes(a)) + " needed for mixed-owner IFEM assembly.");
    }
    support.push_back(local_node);
  }

  return support;
}


std::vector<int> unique_fluid_owners(const std::vector<const ifemCouplingType*>& rows)
{
  std::vector<int> owners;
  for (const auto* row : rows) {
    if (std::find(owners.begin(), owners.end(), row->fluidElemOwnerRank) == owners.end()) {
      owners.push_back(row->fluidElemOwnerRank);
    }
  }
  return owners;
}


void add_remote_ifem_graph_entries(ComMod& com_mod, const int solid_domain,
    const std::unordered_map<int, std::vector<int>>& remote_graph_nodes,
    int& mnnzeic, Array<int>& uInd)
{
  auto& ib_data = com_mod.ib;
  const int rank = com_mod.cm.idcm();
  std::vector<int> local_records;

  for (const auto& ib_mesh : ib_data.msh) {
    if (!mesh_has_domain(ib_mesh, solid_domain)) {
      continue;
    }

    for (int e = 0; e < ib_mesh.nEl; e++) {
      std::vector<const ifemCouplingType*> rows;
      rows.reserve(ib_mesh.eNoN);

      for (int a = 0; a < ib_mesh.eNoN; a++) {
        const int ib_node = ib_mesh.IEN(a,e);
        rows.push_back(&ib_data.ifemCoupling[ib_node]);
      }

      for (const int fluid_owner : unique_fluid_owners(rows)) {
        if (fluid_owner < 0 || fluid_owner == rank) {
          continue;
        }

        local_records.push_back(fluid_owner);
        local_records.push_back(ib_mesh.eNoN);
        for (const auto* row : rows) {
          local_records.push_back(row->ibGlobalNode);
          local_records.push_back(row->fluidGlobalNodes.size());
          for (int n = 0; n < row->fluidGlobalNodes.size(); n++) {
            local_records.push_back(row->fluidGlobalNodes(n));
          }
        }
      }
    }
  }

  std::vector<int> records;
  allgatherv_int(com_mod, local_records, records);

  int pos = 0;
  while (pos < records.size()) {
    const int fluid_owner = records[pos++];
    const int eNoN = records[pos++];
    std::vector<int> ib_global_nodes(eNoN);
    std::vector<std::vector<int>> global_supports(eNoN);
    for (int a = 0; a < eNoN; a++) {
      ib_global_nodes[a] = records[pos++];
      const int n_support = records[pos++];
      global_supports[a].resize(n_support);
      for (int n = 0; n < n_support; n++) {
        global_supports[a][n] = records[pos++];
      }
    }

    if (fluid_owner != rank) {
      continue;
    }

    std::vector<std::vector<int>> supports;
    std::vector<int> owned_rows(eNoN, 0);
    supports.reserve(eNoN);
    for (int a = 0; a < eNoN; a++) {
      const int ib_global_node = ib_global_nodes[a];
      const auto it = remote_graph_nodes.find(ib_global_node);
      if (it != remote_graph_nodes.end()) {
        owned_rows[a] = 1;
        supports.push_back(it->second);
        continue;
      }

      std::vector<int> support;
      support.reserve(global_supports[a].size());
      for (const int global_node : global_supports[a]) {
        const int local_node = global_to_local_node(com_mod, global_node);
        if (local_node < 0) {
          throw std::runtime_error("[ib::add_ifem_coupling_to_lhs_graph] Missing local or ghost "
              "fluid node " + std::to_string(global_node) +
              " needed for mixed-owner remote IFEM graph assembly.");
        }
        support.push_back(local_node);
      }
      supports.push_back(std::move(support));
    }

    add_ifem_graph_entries(com_mod, supports, owned_rows, mnnzeic, uInd);
  }
}


void project_remote_fluid_state_to_solid(ComMod& com_mod, const Array<double>& Ag,
    const Array<double>& Yg)
{
  auto& ib_data = com_mod.ib;
  const int nsd = com_mod.nsd;
  const int rank = com_mod.cm.idcm();

  std::vector<int> local_meta;
  std::vector<double> local_values;

  for (const auto& row : ib_data.ifemRemoteCoupling) {
    Vector<double> Ab(nsd), Yb(nsd);
    Ab = 0.0;
    Yb = 0.0;

    for (int a = 0; a < row.fluidNodes.size(); a++) {
      const int Ac = row.fluidNodes(a);
      const double Na = row.N(a);
      for (int i = 0; i < nsd; i++) {
        Ab(i) += Na * Ag(i,Ac);
        Yb(i) += Na * Yg(i,Ac);
      }
    }

    local_meta.push_back(row.ibGlobalNode);
    for (int i = 0; i < nsd; i++) {
      local_values.push_back(Ab(i));
    }
    for (int i = 0; i < nsd; i++) {
      local_values.push_back(Yb(i));
    }
  }

  std::vector<int> meta;
  std::vector<double> values;
  allgatherv_int(com_mod, local_meta, meta);
  allgatherv_double(com_mod, local_values, values);

  std::unordered_map<int, int> local_ib_nodes;
  for (int a = 0; a < ib_data.tnNo; a++) {
    local_ib_nodes[ib_data.ifemCoupling[a].ibGlobalNode] = a;
  }

  std::vector<int> remote_state_set(ib_data.tnNo, 0);
  for (int r = 0; r < meta.size(); r++) {
    const auto it = local_ib_nodes.find(meta[r]);
    if (it == local_ib_nodes.end()) {
      continue;
    }

    const int ib_node = it->second;
    for (int i = 0; i < nsd; i++) {
      ib_data.Auk(i,ib_node) = values[(2*nsd)*r + i];
      ib_data.Yb(i,ib_node) = values[(2*nsd)*r + nsd + i];
    }
    remote_state_set[ib_node] = 1;
  }

  int missing_remote_state = 0;
  for (const auto& row : ib_data.ifemCoupling) {
    if (row.fluidElemOwnerRank >= 0 && row.fluidElemOwnerRank != rank &&
        remote_state_set[row.ibNode] == 0) {
      missing_remote_state++;
    }
  }

  if (missing_remote_state != 0) {
    throw std::runtime_error("[ib::construct_immersed_fsi] Missing remote IFEM fluid-state "
        "projection for " + std::to_string(missing_remote_state) + " immersed coupling rows.");
  }
}


int find_sparse_matrix_entry(const ComMod& com_mod, const int row_node, const int col_node)
{
  int left = com_mod.rowPtr(row_node);
  int right = com_mod.rowPtr(row_node + 1) - 1;

  while (left <= right) {
    const int candidate = (left + right) / 2;
    if (com_mod.colPtr(candidate) == col_node) {
      return candidate;
    }
    if (com_mod.colPtr(candidate) < col_node) {
      left = candidate + 1;
    } else {
      right = candidate - 1;
    }
  }

  return -1;
}


void add_ifem_spread_to_fluid(ComMod& com_mod, const std::vector<const ifemCouplingType*>& rows,
    const std::vector<int>& owned_rows, const Array<double>& lR, const Array3<double>& lK)
{
  const int nsd = com_mod.nsd;
  const int dof = com_mod.dof;
  const int eNoN = static_cast<int>(rows.size());

  for (int a = 0; a < eNoN; a++) {
    if (owned_rows[a] == 0) {
      continue;
    }

    const auto& row_a = *rows[a];

    for (int A = 0; A < row_a.fluidNodes.size(); A++) {
      const int fluid_A = row_a.fluidNodes(A);
      const double NA = row_a.N(A);

      for (int i = 0; i < nsd; i++) {
        com_mod.R(i,fluid_A) += NA * lR(i,a);
      }

      for (int b = 0; b < eNoN; b++) {
        const auto& row_b = *rows[b];

        for (int B = 0; B < row_b.fluidNodes.size(); B++) {
          const int fluid_B = row_b.fluidNodes(B);
          const double NB = row_b.N(B);
          const int matrix_ptr = find_sparse_matrix_entry(com_mod, fluid_A, fluid_B);

          if (matrix_ptr < 0) {
            throw std::runtime_error("[ib::construct_immersed_fsi] Missing sparse matrix entry "
                "for IFEM projected tangent.");
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


void add_ifem_spread_to_fluid(ComMod& com_mod, const std::vector<const ifemCouplingType*>& rows,
    const Array<double>& lR, const Array3<double>& lK)
{
  std::vector<int> owned_rows(rows.size(), 1);
  add_ifem_spread_to_fluid(com_mod, rows, owned_rows, lR, lK);
}


ifemCouplingType make_column_row_on_rank(const ComMod& com_mod, const ifemCouplingType& row,
    const std::string& caller)
{
  ifemCouplingType local_row = row;
  const auto support = map_support_to_local(com_mod, row, caller);
  local_row.fluidNodes.resize(support.size());
  local_row.fluidLocalNodes.resize(support.size());

  for (int a = 0; a < support.size(); a++) {
    local_row.fluidNodes(a) = support[a];
    local_row.fluidLocalNodes(a) = support[a];
  }

  return local_row;
}


void add_remote_ifem_spread_to_fluid(ComMod& com_mod, const std::vector<int>& local_meta,
    const std::vector<double>& local_values)
{
  auto& ib_data = com_mod.ib;
  const int rank = com_mod.cm.idcm();
  const int nsd = com_mod.nsd;
  const int dof = com_mod.dof;

  std::unordered_map<int, const ifemCouplingType*> remote_rows;
  for (const auto& row : ib_data.ifemRemoteCoupling) {
    remote_rows[row.ibGlobalNode] = &row;
  }

  std::vector<int> meta;
  std::vector<double> values;
  allgatherv_int(com_mod, local_meta, meta);
  allgatherv_double(com_mod, local_values, values);

  int meta_pos = 0;
  int value_pos = 0;
  while (meta_pos < meta.size()) {
    const int fluid_owner = meta[meta_pos++];
    const int eNoN = meta[meta_pos++];
    std::vector<int> ib_global_nodes(eNoN);
    std::vector<std::vector<int>> global_supports(eNoN);
    int n_shape_values = 0;
    for (int a = 0; a < eNoN; a++) {
      ib_global_nodes[a] = meta[meta_pos++];
      const int n_support = meta[meta_pos++];
      n_shape_values += n_support;
      global_supports[a].resize(n_support);
      for (int n = 0; n < n_support; n++) {
        global_supports[a][n] = meta[meta_pos++];
      }
    }

    const int n_values = n_shape_values + nsd*eNoN + nsd*nsd*eNoN*eNoN;
    if (fluid_owner != rank) {
      value_pos += n_values;
      continue;
    }

    std::vector<std::vector<double>> shape_values(eNoN);
    for (int a = 0; a < eNoN; a++) {
      shape_values[a].resize(global_supports[a].size());
      for (int n = 0; n < global_supports[a].size(); n++) {
        shape_values[a][n] = values[value_pos++];
      }
    }

    std::vector<ifemCouplingType> column_rows(eNoN);
    std::vector<const ifemCouplingType*> rows(eNoN, nullptr);
    std::vector<int> owned_rows(eNoN, 0);
    for (int a = 0; a < eNoN; a++) {
      const int ib_global_node = ib_global_nodes[a];
      const auto it = remote_rows.find(ib_global_node);
      if (it != remote_rows.end()) {
        owned_rows[a] = 1;
        rows[a] = it->second;
        continue;
      }

      auto& row = column_rows[a];
      row.ibGlobalNode = ib_global_node;
      row.fluidNodes.resize(global_supports[a].size());
      row.fluidLocalNodes.resize(global_supports[a].size());
      row.fluidGlobalNodes.resize(global_supports[a].size());
      row.N.resize(global_supports[a].size());
      for (int n = 0; n < global_supports[a].size(); n++) {
        const int local_node = global_to_local_node(com_mod, global_supports[a][n]);
        if (local_node < 0) {
          throw std::runtime_error("[ib::construct_immersed_fsi] Missing local or ghost fluid node " +
              std::to_string(global_supports[a][n]) +
              " needed for mixed-owner remote IFEM tangent assembly.");
        }
        row.fluidNodes(n) = local_node;
        row.fluidLocalNodes(n) = local_node;
        row.fluidGlobalNodes(n) = global_supports[a][n];
        row.N(n) = shape_values[a][n];
      }
      rows[a] = &row;
    }

    Array<double> lR(nsd,eNoN);
    Array3<double> lK(dof*dof,eNoN,eNoN);
    lR = 0.0;
    lK = 0.0;

    for (int a = 0; a < eNoN; a++) {
      for (int i = 0; i < nsd; i++) {
        lR(i,a) = values[value_pos++];
      }
    }
    for (int b = 0; b < eNoN; b++) {
      for (int a = 0; a < eNoN; a++) {
        for (int j = 0; j < nsd; j++) {
          for (int i = 0; i < nsd; i++) {
            lK(i + j*dof,a,b) = values[value_pos++];
          }
        }
      }
    }

    add_ifem_spread_to_fluid(com_mod, rows, owned_rows, lR, lK);
  }
}


void project_remote_fluid_increment_to_solid(ComMod& com_mod, const Array<double>& fluid_increment)
{
  auto& ib_data = com_mod.ib;
  const int nsd = com_mod.nsd;
  const int rank = com_mod.cm.idcm();

  std::vector<int> local_meta;
  std::vector<double> local_values;

  for (const auto& row : ib_data.ifemRemoteCoupling) {
    Vector<double> solid_increment(nsd);
    solid_increment = 0.0;

    for (int a = 0; a < row.fluidNodes.size(); a++) {
      const int fluid_node = row.fluidNodes(a);
      const double Na = row.N(a);
      for (int i = 0; i < nsd; i++) {
        solid_increment(i) += Na * fluid_increment(i,fluid_node);
      }
    }

    local_meta.push_back(row.ibGlobalNode);
    for (int i = 0; i < nsd; i++) {
      local_values.push_back(solid_increment(i));
    }
  }

  std::vector<int> meta;
  std::vector<double> values;
  allgatherv_int(com_mod, local_meta, meta);
  allgatherv_double(com_mod, local_values, values);

  std::unordered_map<int, int> local_ib_nodes;
  for (int a = 0; a < ib_data.tnNo; a++) {
    local_ib_nodes[ib_data.ifemCoupling[a].ibGlobalNode] = a;
  }

  std::vector<int> remote_increment_set(ib_data.tnNo, 0);
  for (int r = 0; r < meta.size(); r++) {
    const auto it = local_ib_nodes.find(meta[r]);
    if (it == local_ib_nodes.end()) {
      continue;
    }

    const int ib_node = it->second;
    const int solid_node = ib_data.gN(ib_node);
    for (int i = 0; i < nsd; i++) {
      com_mod.R(i,solid_node) = values[nsd*r + i];
    }
    remote_increment_set[ib_node] = 1;
  }

  int missing_remote_increment = 0;
  for (const auto& row : ib_data.ifemCoupling) {
    if (row.fluidElemOwnerRank >= 0 && row.fluidElemOwnerRank != rank &&
        remote_increment_set[row.ibNode] == 0) {
      missing_remote_increment++;
    }
  }

  if (missing_remote_increment != 0) {
    throw std::runtime_error("[ib::project_fluid_increment_to_solid] Missing remote IFEM "
        "increment projection for " + std::to_string(missing_remote_increment) +
        " immersed coupling rows.");
  }
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

  for (int a = 0; a < ib_data.tnNo; a++) {
    const int solid_node = ib_data.gN(a);
    for (int i = 0; i < dof; i++) {
      com_mod.R(i,solid_node) = 0.0;
    }
  }

  for (const auto& row : ib_data.ifemCoupling) {
    if (row.fluidElemOwnerRank != com_mod.cm.idcm()) {
      continue;
    }

    const int solid_node = ib_data.gN(row.ibNode);

    for (int a = 0; a < row.fluidNodes.size(); a++) {
      const int fluid_node = row.fluidNodes(a);
      const double Na = row.N(a);
      for (int i = 0; i < nsd; i++) {
        com_mod.R(i,solid_node) += Na * fluid_increment(i,fluid_node);
      }
    }
  }

  if (!com_mod.cm.seq()) {
    project_remote_fluid_increment_to_solid(com_mod, fluid_increment);
  }
}


/// @brief Build background-fluid interpolation rows for all immersed solid nodes.
void build_coupling_rows(ComMod& com_mod, const int fluid_domain, const std::string& caller)
{
  auto& ib_data = com_mod.ib;
  ib_data.ifemCoupling.resize(ib_data.tnNo);
  ib_data.ifemRemoteCoupling.clear();

  int missed_nodes = 0;
  int first_missed_node = -1;

  for (int a = 0; a < ib_data.tnNo; a++) {
    auto& row = ib_data.ifemCoupling[a];
    row.ibNode = a;
    row.ibGlobalNode = local_to_global_node(com_mod, ib_data.gN(a));

    Vector<double> x(com_mod.nsd);
    for (int i = 0; i < com_mod.nsd; i++) {
      x(i) = ib_data.x(i,a) + ib_data.Ubk(i,a);
    }
    if (locate_background_fluid_trace(com_mod, x, fluid_domain, row)) {
      continue;
    }

    clear_fluid_trace_nodes(row);
    missed_nodes++;
    if (first_missed_node < 0) {
      first_missed_node = a;
    }
  }

  if (!com_mod.cm.seq()) {
    int global_missed_nodes = 0;
    MPI_Allreduce(&missed_nodes, &global_missed_nodes, 1, cm_mod::mpint, MPI_SUM, com_mod.cm.com());

    if (global_missed_nodes != 0) {
      resolve_missed_coupling_rows_parallel(com_mod, fluid_domain);
    }

    missed_nodes = 0;
    first_missed_node = -1;
    for (int a = 0; a < ib_data.tnNo; a++) {
      if (ib_data.ifemCoupling[a].fluidElemOwnerRank >= 0) {
        continue;
      }
      missed_nodes++;
      if (first_missed_node < 0) {
        first_missed_node = a;
      }
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
  const double ib_vms_stab_scale = eq.immersed_vms_stabilization_s;
  const double width = eq.immersed_vms_stabilization_width;
  if (ib_vms_stab_scale <= 1.0 || width <= 0.0 || h <= 0.0) {
    return 1.0;
  }

  const auto& ib_data = com_mod.ib;
  if (ib_data.vmsX.ncols() == 0) {
    return 1.0;
  }

  const int nsd = com_mod.nsd;
  const double radius = width * h;
  const double radius2 = radius * radius;

  for (int a = 0; a < ib_data.vmsX.ncols(); a++) {
    double dist2 = 0.0;

    for (int i = 0; i < nsd; i++) {
      const double xs = ib_data.vmsX(i,a);
      const double dx = x(i) - xs;
      dist2 += dx * dx;
    }

    if (dist2 <= radius2) {
      return ib_vms_stab_scale;
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
    if (!com_mod.cm.seq()) {
      ib_data.dmnID.clear();
      ib_data.gN.clear();
      ib_data.x.clear();
      ib_data.Yb.clear();
      ib_data.Auo.clear();
      ib_data.Aun.clear();
      ib_data.Auk.clear();
      ib_data.Ubo.clear();
      ib_data.Ubn.clear();
      ib_data.Ubk.clear();
      ib_data.R.clear();
      ib_data.ifemCoupling.clear();
      ib_data.vmsX.clear();
      return;
    }
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

  if (!com_mod.cm.seq()) {
    return false;
  }

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
  if (!com_mod.cm.seq() || ib_data.ifemCoupling.size() != static_cast<std::size_t>(ib_data.tnNo)) {
    build_coupling_rows(com_mod, fluid_domain, "add_ifem_coupling_to_lhs_graph");
  }

  std::vector<std::vector<std::vector<int>>> fluid_mesh_neighbors(com_mod.nMsh);
  std::vector<int> fluid_mesh_neighbors_built(com_mod.nMsh, 0);
  std::vector<std::vector<int>> graph_nodes(ib_data.tnNo);
  std::unordered_map<int, std::vector<int>> remote_graph_nodes;

  // The projected solid tangent couples nearby fluid interpolation supports, so
  // these graph entries must exist before element assembly writes matrix values.
  for (int ib_node = 0; ib_node < ib_data.tnNo; ib_node++) {
    const auto& row = ib_data.ifemCoupling[ib_node];
    if (row.fluidElemOwnerRank != com_mod.cm.idcm()) {
      continue;
    }

    graph_nodes[ib_node] = collect_fluid_graph_nodes(com_mod, fluid_domain, row,
        fluid_mesh_neighbors, fluid_mesh_neighbors_built);
  }

  for (const auto& row : ib_data.ifemRemoteCoupling) {
    remote_graph_nodes[row.ibGlobalNode] = collect_fluid_graph_nodes(com_mod, fluid_domain, row,
        fluid_mesh_neighbors, fluid_mesh_neighbors_built);
  }

  for (const auto& ib_mesh : ib_data.msh) {
    if (!mesh_has_domain(ib_mesh, solid_domain)) {
      continue;
    }

    for (int e = 0; e < ib_mesh.nEl; e++) {
      std::vector<const ifemCouplingType*> rows;
      rows.reserve(ib_mesh.eNoN);
      for (int a = 0; a < ib_mesh.eNoN; a++) {
        const int ib_a = ib_mesh.IEN(a,e);
        rows.push_back(&ib_data.ifemCoupling[ib_a]);
      }

      bool has_owned_row = false;
      for (const auto* row : rows) {
        if (row->fluidElemOwnerRank == com_mod.cm.idcm()) {
          has_owned_row = true;
          break;
        }
      }
      if (!has_owned_row) {
        continue;
      }

      std::vector<std::vector<int>> supports;
      std::vector<int> owned_rows(ib_mesh.eNoN, 0);
      supports.reserve(ib_mesh.eNoN);
      for (int a = 0; a < ib_mesh.eNoN; a++) {
        const int ib_a = ib_mesh.IEN(a,e);
        const auto& row = *rows[a];
        if (row.fluidElemOwnerRank == com_mod.cm.idcm()) {
          owned_rows[a] = 1;
          supports.push_back(graph_nodes[ib_a]);
        } else {
          supports.push_back(map_support_to_local(com_mod, row, "add_ifem_coupling_to_lhs_graph"));
        }
      }
      add_ifem_graph_entries(com_mod, supports, owned_rows, mnnzeic, uInd);
    }
  }

  if (!com_mod.cm.seq()) {
    add_remote_ifem_graph_entries(com_mod, solid_domain, remote_graph_nodes, mnnzeic, uInd);
  }
}


/// @brief Interpolate background-fluid velocity to immersed solid nodes.
void project_fluid_velocity_to_solid(ComMod& com_mod, SolutionStates& solutions) {
  const int nsd = com_mod.nsd;
  auto& ib_data = com_mod.ib;
  auto& velocity = solutions.current.get_velocity();

  ib_data.Yb = 0.0;

  for (const auto& row : ib_data.ifemCoupling) {
    if (row.fluidElemOwnerRank != com_mod.cm.idcm()) {
      continue;
    }

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

  for (int a = 0; a < ib_data.tnNo; a++) {
    const int Ac = ib_data.gN(a);
    for (int i = 0; i < nsd; i++) {
      ib_data.Ubk(i,a) = Dg(i,Ac);
    }
  }
  update_vms_stabilization_points(com_mod);

  int current_mesh = -1;
  for (int iM = 0; iM < com_mod.nMsh; iM++) {
    const auto& mesh = com_mod.msh[iM];
    if (&mesh == &lM || (mesh.dname == lM.dname && mesh.nEl == lM.nEl && mesh.nNo == lM.nNo)) {
      current_mesh = iM;
    }
  }

  if (mesh_has_domain(lM, fluid_domain)) {
    fluid::construct_fluid(com_mod, lM, solutions);
  }

  // Fluid assembly is done for each fluid mesh, but the immersed solid force
  // projection is global and every rank must enter its collectives at the same
  // mesh index.
  if (current_mesh != 0) {
    return;
  }

  ib_data.R = 0.0;
  ib_data.Auk = 0.0;
  ib_data.Yb = 0.0;

  for (const auto& row : ib_data.ifemCoupling) {
    if (row.fluidElemOwnerRank != com_mod.cm.idcm()) {
      continue;
    }

    for (int a = 0; a < row.fluidNodes.size(); a++) {
      const int Ac = row.fluidNodes(a);
      const double Na = row.N(a);
      for (int i = 0; i < nsd; i++) {
        ib_data.Auk(i,row.ibNode) += Na * Ag(i,Ac);
        ib_data.Yb(i,row.ibNode) += Na * Yg(i,Ac);
      }
    }
  }
  if (!com_mod.cm.seq()) {
    project_remote_fluid_state_to_solid(com_mod, Ag, Yg);
  }

  auto& cem = cep_mod.cem;
  std::vector<int> remote_spread_meta;
  std::vector<double> remote_spread_values;

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
        for (int i = 0; i < nsd; i++) {
          ib_data.R(i,ib_a) += lR(i,a);
        }
      }

      std::vector<const ifemCouplingType*> rows;
      rows.reserve(eNoN);
      for (int a = 0; a < eNoN; a++) {
        const int ib_a = ptr(a);
        const auto& row = ib_data.ifemCoupling[ib_a];
        rows.push_back(&row);
      }

      for (const int fluid_owner : unique_fluid_owners(rows)) {
        if (fluid_owner == com_mod.cm.idcm()) {
          std::vector<ifemCouplingType> column_rows(eNoN);
          std::vector<const ifemCouplingType*> local_rows(eNoN, nullptr);
          std::vector<int> owned_rows(eNoN, 0);

          for (int a = 0; a < eNoN; a++) {
            if (rows[a]->fluidElemOwnerRank == fluid_owner) {
              owned_rows[a] = 1;
              local_rows[a] = rows[a];
            } else {
              column_rows[a] = make_column_row_on_rank(com_mod, *rows[a],
                  "construct_immersed_fsi");
              local_rows[a] = &column_rows[a];
            }
          }

          // Spread solid residual/tangent with the IFEM operator: C^T R_s and
          // C^T K_s C.
          add_ifem_spread_to_fluid(com_mod, local_rows, owned_rows, lR, lK);
          continue;
        }

        remote_spread_meta.push_back(fluid_owner);
        remote_spread_meta.push_back(eNoN);
        for (int a = 0; a < eNoN; a++) {
          const auto& row = *rows[a];
          remote_spread_meta.push_back(row.ibGlobalNode);
          remote_spread_meta.push_back(row.fluidGlobalNodes.size());
          for (int n = 0; n < row.fluidGlobalNodes.size(); n++) {
            remote_spread_meta.push_back(row.fluidGlobalNodes(n));
          }
        }

        for (int a = 0; a < eNoN; a++) {
          const auto& row = *rows[a];
          for (int n = 0; n < row.N.size(); n++) {
            remote_spread_values.push_back(row.N(n));
          }
        }
        for (int a = 0; a < eNoN; a++) {
          for (int i = 0; i < nsd; i++) {
            remote_spread_values.push_back(lR(i,a));
          }
        }
        for (int b = 0; b < eNoN; b++) {
          for (int a = 0; a < eNoN; a++) {
            for (int j = 0; j < nsd; j++) {
              for (int i = 0; i < nsd; i++) {
                remote_spread_values.push_back(lK(i + j*dof,a,b));
              }
            }
          }
        }
      }
    }
  }

  if (!com_mod.cm.seq()) {
    add_remote_ifem_spread_to_fluid(com_mod, remote_spread_meta, remote_spread_values);
  }
}

}
