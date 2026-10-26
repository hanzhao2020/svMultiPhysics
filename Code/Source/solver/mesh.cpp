// SPDX-FileCopyrightText: Copyright (c) Stanford University, The Regents of the University of California, and others.
// SPDX-License-Identifier: BSD-3-Clause

#include "mesh.h"

#include "all_fun.h"
#include "consts.h"
#include "fluid.h"
#include "fs.h"
#include "l_elas.h"
#include "lhsa.h"
#include "nn.h"
#include "sv_struct.h"
#include "utils.h"

#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <math.h>

namespace mesh {

void construct_mesh(ComMod& com_mod, CepMod& cep_mod, const mshType& lM, const SolutionStates& solutions)
{
  const auto& Do = solutions.old.get_displacement();
  const auto& Ag = solutions.intermediate.get_acceleration();
  const auto& Dg = solutions.intermediate.get_displacement();
  #define n_debug_construct_mesh 
  #ifdef debug_construct_mesh
  DebugMsg dmsg(__func__, com_mod.cm.idcm());
  dmsg.banner();
  #endif

  using namespace consts;

  auto& cem = cep_mod.cem;
  const int nsd  = com_mod.nsd;
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

  // Start and end DOF
  int is = nsd + 1;
  int ie = 2*nsd;
  int eNoN = lM.eNoN;
  #ifdef debug_construct_mesh
  dmsg << "cEq: " << cEq;
  dmsg << "is: " << is;
  dmsg << "ie: " << ie;
  dmsg << "cDmn: " << cDmn;
  #endif

  Vector<int> ptr(eNoN);
  Vector<double> pSl(nsymd), ya_l(eNoN), N(eNoN);
  Array<double> xl(nsd,eNoN), al(tDof,eNoN), yl(tDof,eNoN), dl(tDof,eNoN),
                dol(nsd,eNoN), pS0l(nsymd,eNoN), Nx(nsd,eNoN), lR(dof,eNoN);
  Array3<double> lK(dof*dof,eNoN,eNoN);
  Array<double> ksix(nsd,nsd), bfl(nsd,eNoN);
  int n_inverted = 0;

  // Jacobian-based stiffening exponent and the weight of the relaxation toward the
  // original-mesh (total) solution at this time (incremental form only)
  const double chi = eq.mesh_stiff_exp;
  const double alpha = eq.mesh_total_disp ? 0.0 : eq.mesh_relax.value(com_mod.time);
  Array<double> xl0, dl0, lR0;
  Array3<double> lK0;
  if (alpha > 0.0) {
    xl0.resize(nsd,eNoN);
    dl0.resize(tDof,eNoN);
    lR0.resize(dof,eNoN);
    lK0.resize(dof*dof,eNoN,eNoN);
  }

  // Element residual and tangent of the pseudo-elastic mesh problem with reference
  // coordinates x_ref and displacement d_ref (Gauss integration).
  auto integrate = [&](const int e, Array<double>& x_ref, const Array<double>& d_ref,
                       Array<double>& R, Array3<double>& K) {
    R = 0.0;
    K = 0.0;
    double Jac{0.0};

    for (int g = 0; g < lM.nG; g++) {
      if (g == 0 || !lM.lShpF) {
        auto Nx_g = lM.Nx.rslice(g);
        nn::gnn(eNoN, nsd, nsd, Nx_g, x_ref, Nx, Jac, ksix);
        if (utils::is_zero(Jac)) {
          throw std::runtime_error("[construct_mesh] Jacobian for element " + std::to_string(e) + " is < 0.");
        }
      }

      // The weight leaves out Jac (chi = 1), so the stiffness scales with 1/(element size);
      // otherwise w*|Jac|^(1-chi) gives a stiffness ~ (element size)^(-chi).
      double w = lM.w(g);
      if (chi != 1.0) {
        w *= std::pow(std::abs(Jac), 1.0 - chi);
      }
      N = lM.N.col(g);
      pS0l = 0.0;

      if (nsd == 3) {
        l_elas::l_elas_3d(com_mod, eNoN, w, N, Nx, al, d_ref, bfl, pS0l, pSl, R, K);

      } else if (nsd == 2) {
        l_elas::l_elas_2d(com_mod, eNoN, w, N, Nx, al, d_ref, bfl, pS0l, pSl, R, K);
      }
    }
  };

  for (int e = 0; e < lM.nEl; e++) {
    // Update domain and proceed if domain phys and eqn phys match
    cDmn = all_fun::domain(com_mod, lM, cEq, e);
    auto cPhys = eq.dmn[cDmn].phys;
    if (cPhys != EquationType::phys_mesh) {
      continue;
    }

    // Update shape functions for NURBS
    if (lM.eType == ElementType::NRB) {
      //CALL NRBNNX(lM, e)
    }

    // Create local copies
    bfl = 0.0;
    pS0l = 0.0;

    for (int a = 0; a < eNoN; a++) {
      int Ac = lM.IEN(a,e);
      ptr(a) = Ac;

      for (int i = 0; i < nsd; i++) {
        xl(i,a) = com_mod.x(i,Ac);
        dol(i,a) = Do(is+i,Ac);
      }

      for (int i = 0; i < tDof; i++) {
        al(i,a) = Ag(i,Ac);
        dl(i,a) = Dg(i,Ac);
      }
    }

    // Reference configuration of the pseudo-elastic mesh problem:
    //  - default (incremental): the configuration at the beginning of the time
    //    step; the unknown is the displacement increment D - D_old.
    //  - Use_original_mesh_reference (total): the original mesh; the unknown is
    //    the total displacement D. The mesh is then a (linear) function of the
    //    current boundary displacement only, so it does not drift under cyclic
    //    boundary motion.
    //  - Mesh_relaxation_file_path (incremental + relaxation): the incremental and
    //    the total forms blended with the weight alpha(t), see MeshRelaxation.
    if (alpha > 0.0) {
      xl0 = xl;
      dl0 = dl;
    }
    if (!eq.mesh_total_disp) {
      for (int i = 0; i < nsd; i++) {
        for (int j = 0; j < eNoN; j++) {
          xl(i,j) = xl(i,j) + dol(i,j);
          dl(i+is,j) = dl(i+is,j) - dol(i,j);
        }
      }
    } else {
      // The stiffness is integrated on the original mesh, so an inverted
      // deformed element would go unnoticed here. Check the start-of-step
      // configuration at the first Gauss point and count inverted elements
      // (Nx is recomputed from xl in the Gauss loop below).
      Array<double> xcur(nsd,eNoN);
      for (int i = 0; i < nsd; i++) {
        for (int j = 0; j < eNoN; j++) {
          xcur(i,j) = xl(i,j) + dol(i,j);
        }
      }
      double Jcur{0.0};
      auto Nx_c = lM.Nx.rslice(0);
      nn::gnn(eNoN, nsd, nsd, Nx_c, xcur, Nx, Jcur, ksix);
      if (Jcur <= 0.0) {
        n_inverted += 1;
      }
    }

    // Gauss integration
    //
    integrate(e, xl, dl, lR, lK);

    // Relaxation: (1 - alpha) * incremental + alpha * total (original mesh, total D).
    // Both forms have the same unknown D, so the blended tangent is exact.
    if (alpha > 0.0) {
      integrate(e, xl0, dl0, lR0, lK0);
      for (int a = 0; a < eNoN; a++) {
        for (int i = 0; i < dof; i++) {
          lR(i,a) = (1.0 - alpha) * lR(i,a) + alpha * lR0(i,a);
        }
        for (int b = 0; b < eNoN; b++) {
          for (int i = 0; i < dof*dof; i++) {
            lK(i,a,b) = (1.0 - alpha) * lK(i,a,b) + alpha * lK0(i,a,b);
          }
        }
      }
    }

    eq.linear_algebra->assemble(com_mod, eNoN, ptr, lK, lR);
  }

  // Report inverted deformed elements once per time step (total formulation only).
  static int last_reported_step = -1;
  if (n_inverted > 0 && com_mod.cTS != last_reported_step) {
    std::cout << "WARNING [construct_mesh] " << n_inverted
              << " mesh element(s) with non-positive Jacobian in the deformed configuration"
              << " (time step " << com_mod.cTS << ", process " << com_mod.cm.idcm() << ")." << std::endl;
    last_reported_step = com_mod.cTS;
  }
}

};
