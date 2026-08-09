// SPDX-FileCopyrightText: Copyright (c) Stanford University, The Regents of the University of California, and others.
// SPDX-License-Identifier: BSD-3-Clause

#ifndef IB_H
#define IB_H

#include "ComMod.h"
#include "SolutionStates.h"
#include "Simulation.h"

namespace ib {

void initialize_immersed_meshes(ComMod& com_mod);

bool build_ifem_coupling_operator(ComMod& com_mod, const SolutionStates& solutions);

void build_coupling_rows(ComMod& com_mod, const int fluid_domain, const std::string& caller);

void add_ifem_coupling_to_lhs_graph(ComMod& com_mod, int& mnnzeic, Array<int>& uInd);

void apply_ifem_reduced_solid_rows(ComMod& com_mod);

void project_fluid_increment_to_solid(ComMod& com_mod);

void project_fluid_velocity_to_solid(ComMod& com_mod, SolutionStates& solutions);

double ib_vms_stabilization_s(const ComMod& com_mod, const Vector<double>& x, double h);

void construct_immersed_fsi(ComMod& com_mod, CepMod& cep_mod, const mshType& lM, const SolutionStates& solutions);

};

#endif
