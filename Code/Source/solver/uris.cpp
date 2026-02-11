// SPDX-FileCopyrightText: Copyright (c) Stanford University, The Regents of the University of California, and others.
// SPDX-License-Identifier: BSD-3-Clause

#include "uris.h"

#include "all_fun.h"
#include "lhsa.h"
#include "mat_fun.h"
#include "nn.h"
#include "utils.h"
#include "set_bc.h"

#include "load_msh.h"
#include "vtk_xml.h"
#include "read_msh.h"
#include "VtkData.h"

namespace uris { 

/// @brief This subroutine computes the mean pressure and flux on the 
/// immersed surface 
void uris_meanp(ComMod& com_mod, CmMod& cm_mod, const int iUris) {
  #define n_debug_uris_meanp
  #ifdef debug_uris_meanp
  DebugMsg dmsg(__func__, com_mod.cm.idcm());
  dmsg.banner();
  dmsg << "urisFlag: " << com_mod.urisFlag;
  #endif

  using namespace consts;

  auto& cm = com_mod.cm;
  auto& eq = com_mod.eq;
  auto& uris = com_mod.uris;
  auto& msh = com_mod.msh;
  auto& uris_obj = uris[iUris];

  const int nsd = com_mod.nsd;
  auto& Yn = com_mod.Yn;

  // Let's conpute the mean pressure in the two regions of the fluid mesh 
  // For the moment let's define a flag IdSubDmn(size the number of elements)

  // Now we can compute the pressure mean on each subdomain
  // We need to have a sdf array for each mesh
  double Deps = uris_obj.sdf_deps * 2.5;
  double volU = 0.0;
  double volD = 0.0;

  // Let's compute left side 
  Array<double> sUPS(1,com_mod.tnNo);

  sUPS = 0.0;
  for (size_t j = 0; j < sUPS.size(); j++) {
    if (uris_obj.sdf(j) >= 0.0 && uris_obj.sdf(j) <= Deps) {
        sUPS(0,j) = 1.0;
    }
  }

  for (int iM = 0; iM < com_mod.nMsh; iM++) {
    volU += all_fun::integ(com_mod, cm_mod, iM, sUPS);
  }

  // Let's compute right side
  Array<double> sDST(1,com_mod.tnNo);
  sDST = 0.0;
  for (size_t j = 0; j < sDST.size(); j++) {
    if (uris_obj.sdf(j) < 0.0 && uris_obj.sdf(j) >= -Deps) {
        sDST(0,j) = 1.0;
    }
  } 

  for (int iM = 0; iM < com_mod.nMsh; iM++) {
    volD += all_fun::integ(com_mod, cm_mod, iM, sDST);
  }

  // Print volume messages.
  if (cm.mas(cm_mod)) {
    std::cout << "volume upstream " << volU << " for: " << uris_obj.name << std::endl;
    std::cout << "volume downstream " << volD << " for: " << uris_obj.name << std::endl;
  }

  double meanPU = 0.0;
  double meanPD = 0.0;

  int iEq = 0;
  int m = 1;
  int s = eq[iEq].s + nsd;

  Array<double> tmpV(maxNSD, com_mod.tnNo);
  for (int j = 0; j < Yn.ncols(); j++) {
    tmpV(0,j) = Yn(s,j)*sUPS(j);
  }
  for (int iM = 0; iM < com_mod.nMsh; iM++) {
    meanPU += all_fun::integ(com_mod, cm_mod, iM, tmpV);
  }
  meanPU = meanPU / volU;

  for (int j = 0; j < Yn.ncols(); j++) {
    tmpV(0,j) = Yn(s,j)*sDST(j);
  }
  for (int iM = 0; iM < com_mod.nMsh; iM++) {
    meanPD += all_fun::integ(com_mod, cm_mod,iM, tmpV);
  }
  meanPD = meanPD / volD;

  uris_obj.meanPU = uris_obj.relax_factor * meanPU +
                       (1.0 - uris_obj.relax_factor) * uris_obj.meanPU;
  uris_obj.meanPD = uris_obj.relax_factor * meanPD +
                       (1.0 - uris_obj.relax_factor) * uris_obj.meanPD;

  if (cm.mas(cm_mod)) {
    std::cout << "mean P upstream " << meanPU << " " << uris_obj.meanPU
              << " for: " << uris_obj.name << std::endl;
    std::cout << "mean P downstream " << meanPD << " " << uris_obj.meanPD
              << " for: " << uris_obj.name << std::endl;
  }

  //  If the uris has passed the closing state
  int stab_cycle = 1; // The uris is considered to be closed if within this number of states
  if (uris_obj.cnt > stab_cycle*uris_obj.DxClose.nrows() 
      && com_mod.cTS*com_mod.dt > uris_obj.pressurization_time) {
    if (uris_obj.meanPD > uris_obj.meanPU) {
      uris_obj.cnt = 1;
      uris_obj.clsFlg = false;
      com_mod.urisActFlag = true;
      if (cm.mas(cm_mod)) {
        std::cout << "** Set urisCloseFlag to FALSE for: "
                  << uris_obj.name << std::endl;
      }
    }
  }
  if (cm.mas(cm_mod)) {
    std::cout << "urisCloseFlag is: " << uris_obj.clsFlg << " for: "
              << uris_obj.name << std::endl;
  }
  
}

/// @brief This subroutine computes the mean velocity in the fluid elements 
/// near the immersed surface  
void uris_meanv(ComMod& com_mod, CmMod& cm_mod, const int iUris) {
  #define n_debug_uris_meanv
  #ifdef debug_uris_meanv
  DebugMsg dmsg(__func__, com_mod.cm.idcm());
  dmsg.banner();
  dmsg << "urisFlag: " << com_mod.urisFlag;
  #endif

  using namespace consts;

  auto& cm = com_mod.cm;
  auto& eq = com_mod.eq;
  auto& uris = com_mod.uris;
  auto& msh = com_mod.msh;
  auto& uris_obj = uris[iUris];

  const int nsd = com_mod.nsd;
  auto& Yn = com_mod.Yn;

  // Let's compute the neighboring region below the valve normal. When
  // the valve is open, this region should roughly be valve oriface.
  int iEq = 0;

  double Deps = uris_obj.sdf_deps;
  Array<double> sImm(1, com_mod.tnNo);
  sImm = 0.0;
  double volI = 0.0;

  for (int i = 0; i < com_mod.tnNo; i++) {
    if (uris_obj.sdf(i) <= -Deps) {
      sImm(0,i) = 1.0;
    }
  }

  for (int iM = 0; iM < com_mod.nMsh; iM++) {
    volI += all_fun::integ(com_mod, cm_mod, iM, sImm);
  }
  if (cm.mas(cm_mod)) {
    std::cout << "volume inside " << volI << " for: " << uris_obj.name << std::endl;
  }
  
  int m = nsd;
  int s = eq[iEq].s;

  Array<double> tmpV(maxNSD, com_mod.tnNo);
  for (int i = 0; i < nsd; i++) {
    for (int j = 0; j < Yn.ncols(); j++) {
      tmpV(i,j) = Yn(s+i,j)*sImm(0,j);
    }
  }

  Array<double> tmpVNrm(1, com_mod.tnNo);
  for (int i = 0; i < com_mod.tnNo; i++) {
    tmpVNrm(0,i) = tmpV(0,i) * uris_obj.nrm(0) 
                 + tmpV(1,i) * uris_obj.nrm(1) 
                 + tmpV(2,i) * uris_obj.nrm(2);
  }

  double meanV = 0.0;
  for (int iM = 0; iM < com_mod.nMsh; iM++) {
    meanV += all_fun::integ(com_mod, cm_mod, iM, tmpVNrm)/volI;
  }
  
  if (cm.mas(cm_mod)) {
    std::cout << "mean velocity: " << meanV << " for: " << uris_obj.name << std::endl;
  }

  // If the uris has passed the open state
  int stab_cycle = 1; // The uris is considered to be open if within this number of states
  if (uris_obj.cnt > stab_cycle*uris_obj.DxOpen.nrows()
      && com_mod.cTS*com_mod.dt > uris_obj.pressurization_time) {
    if (meanV < 0.0) {
      uris_obj.cnt = 1;
      uris_obj.clsFlg = true;
      com_mod.urisActFlag = true;
      if (cm.mas(cm_mod)) {
        std::cout << "** Set urisCloseFlag to TRUE for: " 
                  << uris_obj.name << std::endl;
      }
    }
  }
  if (cm.mas(cm_mod)) {
    std::cout << "urisCloseFlag is: " << uris_obj.clsFlg << " for: "
              << uris_obj.name << std::endl;
  }

}

/// @brief  This subroutine computes the displacement of the immersed 
/// surface with fem projection
void uris_update_disp(ComMod& com_mod, CmMod& cm_mod) {
  #define n_debug_uris_update_disp 
  #ifdef debug_uris_update_disp
  DebugMsg dmsg(__func__, com_mod.cm.idcm());
  dmsg.banner();
  #endif

  auto& cm = com_mod.cm;
  // auto& eq = com_mod.eq;
  auto& uris = com_mod.uris;
  auto& msh = com_mod.msh;
  int nUris = com_mod.nUris;

  const int nsd = com_mod.nsd;

  // For each point in the immersed surface we need to localize it 
  // = find the fluid element that contains the node
  // Since the fluid element could be on another processor, we need to
  // gather the displacement values at the end      
  // [FK] it's probably better to save the element ids so that we don't
  // have to run the search every time step, only during open or close 

  Array<double> localYd, xl, Nxi;
  Vector<double> N;
  Vector<double> xp(nsd), xi(nsd), d(nsd);
  bool fl;

  for (int iUris = 0; iUris < nUris; iUris++) {
    auto& uris_obj = uris[iUris];
    uris_find_tetra(com_mod, cm_mod, iUris);
    localYd.resize(nsd, uris_obj.tnNo);
    localYd = 0.0;
    for (int nd = 0; nd < uris_obj.tnNo; nd++) {
      int jM = uris_obj.elemId(0, nd);
      auto& mesh = msh[jM];
      // If the fluid mesh element is not on the current proc
      if (jM == -1) {continue;}

      int iEln = uris_obj.elemId(1, nd);
      Vector<double> xp = uris_obj.x.col(nd);
      xl.resize(nsd, mesh.eNoN);
      N.resize(mesh.eNoN);
      Nxi.resize(nsd, mesh.eNoN);

      for (int a = 0; a < mesh.eNoN; a++) {
        int Ac = mesh.IEN(a, iEln);
        for (int i = 0; i < nsd; i++) {
          xl(i, a) = com_mod.x(i, Ac);
        }
      }
      // Get displacement  
      // Localize p inside the parent element
      nn::get_xi(nsd, mesh.eType, mesh.eNoN, xl, xp, xi, fl);
      if (!fl) {
        if (cm.mas(cm_mod)) {
          std::cout << "[WARNING] URIS get_xi not converging!" << std::endl;
        }
      }
      // evaluate N at xi 
      nn::get_gnn(nsd, mesh.eType, mesh.eNoN, xi, N, Nxi);
      // use this to compute disp al node xp
      d = 0.0;
      for (int a = 0; a < mesh.eNoN; a++) {
        int Ac = mesh.IEN(a, iEln);
        //We have to use Do because Dn contains the result coming from the solid 
        d(0) += N(a)*com_mod.Do(nsd+1, Ac);
        d(1) += N(a)*com_mod.Do(nsd+2, Ac);
        d(2) += N(a)*com_mod.Do(nsd+3, Ac);
      }
      // update uris disp  
      localYd.set_col(nd, d);
    }
    MPI_Allreduce(localYd.data(), uris_obj.Yd.data(), uris_obj.tnNo*nsd,
                  cm_mod::mpreal, MPI_SUM, cm.com());

    for (int nd = 0; nd < uris_obj.tnNo; nd++) {
      double divisor = static_cast<double>(std::max(1, uris_obj.elemCounter(nd)));
      for (int i = 0; i < nsd; i++) {
        uris_obj.Yd(i,nd) /= divisor;
      }
    }
  }

}

/// @brief  This subroutine computes the tetrahedral elements
void uris_find_tetra(ComMod& com_mod, CmMod& cm_mod, const int iUris) {
  #define n_debug_uris_find_tetra 
  #ifdef debug_uris_find_tetra
  DebugMsg dmsg(__func__, com_mod.cm.idcm());
  dmsg.banner();
  #endif

  auto& cm = com_mod.cm;
  auto& uris = com_mod.uris;
  auto& uris_obj = uris[iUris];
  const int nsd = com_mod.nsd;

  // We need to check if the valve needs to move
  int cnt;
  if (!uris_obj.clsFlg) {
    cnt = std::min(uris_obj.cnt, uris_obj.DxOpen.nrows());
  } else {
    cnt = std::min(uris_obj.cnt, uris_obj.DxClose.nrows());
  }

  if (uris_obj.elemId.allocated() && cnt < uris_obj.cnt) {
    return;
  }

  // For each point in the immersed surface we need to localize it 
  // = find the fluid element that contains the node
  // Since the fluid element could be on another processor, we need to
  // gather the displacement values at the end      
  // [FK] it's probably better to save the element ids so that we don't
  // have to run the search every time step, only during open or close      

  bool ultra = true;
  if (!uris_obj.elemId.allocated()) {
    uris_obj.elemId.resize(2, uris_obj.tnNo);
  }
  if (!uris_obj.elemCounter.allocated()) {
    uris_obj.elemCounter.resize(uris_obj.tnNo);
  }
  Vector<int> local_counter(uris_obj.tnNo);
  local_counter = 0;
  uris_obj.elemId = -1;
  uris_obj.elemCounter = 0;
  int flag;
  Array<double> xl;
  for (int nd = 0; nd < uris_obj.tnNo; nd++) {
    flag = 0;
    // Check if we were able to find the tetra.
    // [FK] if not, the tetra is on another processor 
    Vector<double> xp = uris_obj.x.col(nd);
    bool found = false;

    for (int jM = 0; jM < com_mod.nMsh && !found; jM++) {
      auto& mesh = com_mod.msh[jM];
      xl.resize(nsd, mesh.eNoN);
      for (int iEln = 0; iEln < mesh.nEl && !found; iEln++) {
        for (int a = 0; a < mesh.eNoN; a++) {
          int Ac = mesh.IEN(a, iEln);
          for(int i = 0; i < nsd; i++) {
            xl(i,a) = com_mod.x(i,Ac);
          }
        }
        inside_tet(com_mod, mesh.eNoN, xp, xl, flag, ultra);
        if (flag == 1) {
          uris_obj.elemId(0, nd) = jM;
          uris_obj.elemId(1, nd) = iEln;
          local_counter(nd) += 1;
          found = true;
        }
      }
    }
  }

  MPI_Allreduce(local_counter.data(), uris_obj.elemCounter.data(), 
                uris_obj.tnNo, cm_mod::mpint, MPI_SUM, cm.com());

}


/// @brief This subroutine check if a node is inside a tetrahedron
void inside_tet(ComMod& com_mod, int& eNoN, Vector<double>& xp, 
                Array<double>& xl, int& flag, bool ext) {
  #define n_debug_inside_tet 
  #ifdef debug_inside_tet
  DebugMsg dmsg(__func__, com_mod.cm.idcm());
  dmsg.banner();
  #endif

  auto& uris = com_mod.uris;
  const int nsd = com_mod.nsd;
  Vector<double> minb(nsd);
  Vector<double> maxb(nsd);

  // Create a bounding box around of the current solid location 
  // [FK]: Hard coded BBox?? This is going to cause problem if scale changes
  for (int i = 0; i < nsd; i++) {
    double min_val = std::numeric_limits<double>::max();
    double max_val = std::numeric_limits<double>::lowest();
    for (int j = 0; j < eNoN; j++) {
      double val_minus = xl(i,j) - 0.1;
      double val_plus = xl(i,j) + 0.1;
      if (val_minus < min_val) {min_val = val_minus;}
      if (val_plus > max_val) {max_val = val_plus;}
    }
    minb(i) = min_val;
    maxb(i) = max_val;
  }

  // Is the node inside the BBox? 
  bool inside = true;
  for (int i = 0; i < nsd; ++i) {
    if (xp(i) < minb(i) || xp(i) > maxb(i)) {
      inside = false;
      break;
    }
  }

  flag = 0;
  if (inside) {
    flag = in_poly(xp, xl, ext);
  }
}


/// @brief Read the URIS mesh separately 
void uris_read_msh(Simulation* simulation) {
  #define n_debug_uris_read_msh 
  #ifdef debug_uris_read_msh
  DebugMsg dmsg(__func__, simulation->com_mod.cm.idcm());
  dmsg.banner();
  #endif

  auto& com_mod = simulation->get_com_mod();
  auto& uris = com_mod.uris;
  const int nsd = com_mod.nsd;

  com_mod.urisFlag = true;
  com_mod.urisActFlag = true;

  int nUris = simulation->parameters.URIS_mesh_parameters.size();
  com_mod.nUris = nUris;

  std::cout << "Number of immersed surfaces for uris: " << nUris << std::endl;
  uris.resize(nUris);

  for (int iUris = 0; iUris < nUris; iUris++) {
    auto param = simulation->parameters.URIS_mesh_parameters[iUris];
    auto& uris_obj = uris[iUris];
    uris_obj.name = param->name();
    std::cout << "** Reading URIS mesh: " << uris_obj.name << std::endl;

    uris_obj.scF = param->mesh_scale_factor();
    uris_obj.nFa = param->URIS_face_parameters.size();
    uris_obj.msh.resize(uris_obj.nFa);
    uris_obj.nrm.resize(nsd);
    Array<double> gX(0,0);

    std::string positive_flow_normal_file_path = param->positive_flow_normal_file_path();

    std::ifstream file_stream;
    file_stream.open(positive_flow_normal_file_path);
    if (!file_stream.is_open()) {
      throw std::runtime_error("Failed to open the open positive flow normal file '" 
                               + positive_flow_normal_file_path + "'.");
    }
    for (int i = 0; i < nsd; i++) {
      file_stream >> uris_obj.nrm(i);
    }
    file_stream.close();

    std::string scaffold_file_path = param->scaffold_file_path();
    if (scaffold_file_path != "") {
      uris_obj.scaffold_flag = true;
      uris_obj.scaffold_mesh.lShl = true;
      vtk_xml::read_vtu(scaffold_file_path, uris_obj.scaffold_mesh);
      nn::select_ele(simulation->com_mod, uris_obj.scaffold_mesh);
      read_msh_ns::check_ien(simulation, uris_obj.scaffold_mesh);

      int b = 0;
      auto& scaffold_mesh = uris_obj.scaffold_mesh;
      scaffold_mesh.nNo = scaffold_mesh.gnNo;
      scaffold_mesh.gN.resize(scaffold_mesh.nNo);
      scaffold_mesh.gN = 0;
      scaffold_mesh.lN.resize(scaffold_mesh.nNo);
      scaffold_mesh.lN = 0;
      for (int a = 0; a < scaffold_mesh.nNo; a++) {
        scaffold_mesh.gN(a) = b;
        scaffold_mesh.lN(b) = a;
        b++;
      }

      // Remap msh%gIEN array
      scaffold_mesh.nEl = scaffold_mesh.gnEl;
      scaffold_mesh.IEN.resize(scaffold_mesh.eNoN, scaffold_mesh.nEl);
      for (int e = 0; e < scaffold_mesh.nEl; e++) {
        for (int a = 0; a < scaffold_mesh.eNoN; a++) {
          int Ac = scaffold_mesh.gIEN(a,e);
          Ac = scaffold_mesh.gN(Ac);
          scaffold_mesh.IEN(a,e) = Ac;
        }
      }
      scaffold_mesh.gIEN.clear();
    }

    uris_obj.sdf_deps = param->thickness();
    uris_obj.sdf_deps_close = param->close_thickness();
    uris_obj.resistance = param->resistance();
    uris_obj.resistance_close = param->resistance_close();
    uris_obj.clsFlg = param->valve_starts_as_closed();
    uris_obj.use_valve_velocity = param->use_valve_velocity();
    uris_obj.reverse_normal = param->reverse_surface_normal();
    uris_obj.pressurization_time = param->pressurization_time();

    for (int iM = 0; iM < uris_obj.nFa; iM++) {
      auto mesh_param = param->URIS_face_parameters[iM];
      auto& mesh = uris_obj.msh[iM];
      mesh.lShl = true;
      mesh.name = mesh_param->name();
      std::cout << "-- Reading URIS face: " << mesh.name << std::endl;

      // Read mesh nodal coordinates and element connectivity.
      uris_read_sv(simulation, mesh, mesh_param);

      // IF (uris(iUris)%msh(iM)%eType .EQ. eType_NA) THEN
      // CALL READCCNE(lPN, uris(iUris)%msh(iM))
      // END IF
      // IF (uris(iUris)%msh(iM)%eType .EQ. eType_NA) THEN
      //     CALL READNRB(lPN, uris(iUris)%msh(iM))
      // END IF
      // IF (uris(iUris)%msh(iM)%eType .EQ. eType_NA) THEN
      //     CALL READGAMBIT(lPN, uris(iUris)%msh(iM))
      // END IF
      // IF (uris(iUris)%msh(iM)%eType .EQ. eType_NA) THEN
      //     err = " Failed to identify format of the uris mesh"
      // END IF

      std::cout << "Number of uris nodes: " << mesh.gnNo << std::endl;
      std::cout << "Number of uris elements: " << mesh.gnEl << std::endl;

      // Read valve motion: note that this motion is defined on the 
      // reference configuration 
      std::string open_motion_file_path = mesh_param->open_motion_file_path();
      // std::ifstream file_stream;
      file_stream.open(open_motion_file_path);
      if (!file_stream.is_open()) {
        throw std::runtime_error("Failed to open the open motion file '" 
                                 + open_motion_file_path + "'.");
      }
      int dispNtOpen, dispNnOpen;
      file_stream >> dispNtOpen >> dispNnOpen;

      if (dispNnOpen != mesh.gnNo) {
        throw std::runtime_error("Mismatch in node numbers between URIS mesh and displacements.");
      }

      Array3<double> dispOpen(dispNtOpen, nsd, dispNnOpen);
      for (int t = 0; t < dispNtOpen; t++) {
        for (int a = 0; a < dispNnOpen; a++) {
          for (int i = 0; i < nsd; i++) {
            file_stream >> dispOpen(t,i,a);
          }
        }
      }
      file_stream.close();

      std::string close_motion_file_path = mesh_param->close_motion_file_path();
      // std::ifstream file_stream;
      file_stream.open(close_motion_file_path);
      if (!file_stream.is_open()) {
        throw std::runtime_error("Failed to open the close motion file '" 
                                 + close_motion_file_path + "'.");
      }
      int dispNtClose, dispNnClose;
      file_stream >> dispNtClose >> dispNnClose;

      if (dispNnClose != mesh.gnNo) {
        throw std::runtime_error("Mismatch in node numbers between URIS mesh and displacements.");
      }

      Array3<double> dispClose(dispNtClose, nsd, dispNnClose);
      for (int t = 0; t < dispNtClose; t++) {
        for (int a = 0; a < dispNnClose; a++) {
          for (int i = 0; i < nsd; i++) {
            file_stream >> dispClose(t,i,a);
          }
        }
      }
      file_stream.close();

      // To scale the mesh, while attaching x to gX
      int a = uris_obj.tnNo + mesh.gnNo;

      if (iM == 0) {
        gX.resize(nsd, a);
        gX = 0.0;
        uris_obj.DxOpen.resize(dispNtOpen, nsd, a);
        uris_obj.DxClose.resize(dispNtClose, nsd, a);
      } else{
        Array<double> tmpX(nsd, uris_obj.tnNo);
        tmpX = gX;
        gX.resize(nsd, a);
        for (int i = 0; i < nsd; i++) {
          for (int j = 0; j < uris_obj.tnNo; j++) {
            gX(i,j) = tmpX(i,j);
          }
        }
        // Move data for open
        Array3<double> tmpDxOpen(dispNtOpen, nsd, uris_obj.tnNo);
        tmpDxOpen = uris_obj.DxOpen;
        uris_obj.DxOpen.resize(dispNtOpen, nsd, a);
        for (int k = 0; k < dispNtOpen; k++) {
          for (int i = 0; i < nsd; i++) {
            for (int j = 0; j < uris_obj.tnNo; j++) {
              uris_obj.DxOpen(k,i,j) = tmpDxOpen(k,i,j);
            }
          }
        }
        // Move data for open
        Array3<double> tmpDxClose(dispNtClose, nsd, uris_obj.tnNo);
        tmpDxClose = uris_obj.DxClose;
        uris_obj.DxClose.resize(dispNtClose, nsd, a);
        for (int k = 0; k < dispNtClose; k++) {
          for (int i = 0; i < nsd; i++) {
            for (int j = 0; j < uris_obj.tnNo; j++) {
              uris_obj.DxClose(k,i,j) = tmpDxClose(k,i,j);
            }
          }
        }
      }

      for (int i = 0; i < nsd; i++) {
        for (int j = uris_obj.tnNo; j < a; j++) {
          gX(i,j) = mesh.x(i,j-uris_obj.tnNo) * uris_obj.scF;
        }
      }
      for (int k = 0; k < dispNtOpen; k++) {
        for (int i = 0; i < nsd; i++) {
          for (int j = uris_obj.tnNo; j < a; j++) {
            uris_obj.DxOpen(k,i,j) = dispOpen(k,i,j-uris_obj.tnNo) * uris_obj.scF;
          }
        }
      }
      for (int k = 0; k < dispNtClose; k++) {
        for (int i = 0; i < nsd; i++) {
          for (int j = uris_obj.tnNo; j < a; j++) {
            uris_obj.DxClose(k,i,j) = dispClose(k,i,j-uris_obj.tnNo) * uris_obj.scF;
          }
        }
      }
      uris_obj.tnNo = a;
    }
    uris_obj.x.resize(nsd, uris_obj.tnNo);
    uris_obj.x_prev.resize(nsd, uris_obj.tnNo);
    if (!uris_obj.clsFlg) {
      int nrowsDxOpen = uris_obj.DxOpen.nrows();
      for (int i = 0; i < uris_obj.x.nrows(); i++) {
        for (int j = 0; j < uris_obj.x.ncols(); j++) {
          uris_obj.x(i,j) = uris_obj.DxOpen(nrowsDxOpen-1,i,j);
          uris_obj.x_prev(i,j) = uris_obj.DxOpen(nrowsDxOpen-1,i,j);
        }
      }
    } else {
      int nrowsDxClose = uris_obj.DxClose.nrows();
      for (int i = 0; i < uris_obj.x.nrows(); i++) {
        for (int j = 0; j < uris_obj.x.ncols(); j++) {
          uris_obj.x(i,j) = uris_obj.DxClose(nrowsDxClose-1,i,j);
          uris_obj.x_prev(i,j) = uris_obj.DxClose(nrowsDxClose-1,i,j);
        }
      }
    }
    uris_obj.v.resize(nsd, uris_obj.tnNo);
    uris_obj.v = 0.0;
    uris_obj.Yd.resize(nsd, uris_obj.tnNo);
    uris_obj.Yd = 0.0;

    // Setting mesh.gN, mesh.lN parameter
    int b = 0;
    for (int iM = 0; iM < uris_obj.nFa; iM++) {
      auto& mesh = uris_obj.msh[iM];
      mesh.nNo = mesh.gnNo;
      mesh.gN.resize(mesh.nNo);
      mesh.gN = 0;
      mesh.lN.resize(uris_obj.tnNo);
      mesh.lN = 0;
      for (int a = 0; a < mesh.nNo; a++) {
        mesh.gN(a) = b;
        mesh.lN(b) = a;
        b++;
      }
    }
    if (b != uris_obj.tnNo) {
      throw std::runtime_error("Mismatch in uris.tnNo. Correction needed.");
    }

    // Remap msh%gIEN array
    for (int iM = 0; iM < uris_obj.nFa; iM++) {
      auto& mesh = uris_obj.msh[iM];
      mesh.nEl = mesh.gnEl;
      mesh.IEN.resize(mesh.eNoN, mesh.nEl);
      for (int e = 0; e < mesh.nEl; e++) {
        for (int a = 0; a < mesh.eNoN; a++) {
          int Ac = mesh.gIEN(a,e);
          Ac = mesh.gN(Ac);
          mesh.IEN(a,e) = Ac;
        }
      }
      mesh.gIEN.clear();
    }

    if (uris_obj.nFa > 0) {
      std::string msg = "Total number of uris nodes: " + std::to_string(uris_obj.tnNo);
      std::cout << msg << std::endl;
      int total_nel = 0;
      for (int iM = 0; iM < uris_obj.nFa; iM++) {
          total_nel += uris_obj.msh[iM].nEl;
      }
      msg = "Total number of uris elements: " + std::to_string(total_nel);
      std::cout << msg << std::endl;
    }
  }
  std::cout << "URIS mesh data imported successfully." << std::endl;

}

/// @brief Write URIS solution to a vtu file
void uris_write_vtus(ComMod& com_mod) {
  #define n_debug_uris_write_vtus 
  #ifdef debug_uris_write_vtus
  DebugMsg dmsg(__func__, simulation->com_mod.cm.idcm());
  dmsg.banner();
  #endif

  using namespace consts;

  auto& uris = com_mod.uris;
  const int nsd = com_mod.nsd;
  const int nUris = com_mod.nUris;

  // we plot coord + displacement + velocity
  int nOut = 3;
  int outDof = nOut * nsd;
  std::vector<std::string> outNames(nOut);
  Vector<int> outS(nOut+1);

  // Prepare all solultions in to dataType d
  std::vector<dataType> d;
  for (int iUris = 0; iUris < nUris; iUris++) {
    auto& uris_obj = uris[iUris];
    d.resize(uris_obj.nFa);
    int nNo = 0;
    int nEl = 0;
    for (int iM = 0; iM < uris_obj.nFa; iM++) {
      auto& mesh = uris_obj.msh[iM];
      int cOut = 0;
      outS(cOut) = 0;
      outS(cOut+1) = nsd;

      if (mesh.eType == ElementType::NRB) {
        throw std::runtime_error("Outputs for NURBS data is under development.");
      }
      d[iM].nNo = mesh.nNo;
      d[iM].nEl = mesh.nEl;
      d[iM].eNoN = mesh.eNoN;
      d[iM].vtkType = mesh.vtkType;
      d[iM].x.resize(outDof, mesh.nNo);
      d[iM].IEN.resize(mesh.eNoN, mesh.nEl);

      for (int a = 0; a < mesh.nNo; a++) {
        int Ac = mesh.gN(a);
        for (int i = 0; i < nsd; i++) {
          // Scale the valve displacement to the unit of the mesh
          d[iM].x(i,a) = uris_obj.x(i,Ac) / uris_obj.scF;
        }
      }
      for (int e = 0; e < mesh.nEl; e++) {
        for (int i = 0; i < mesh.eNoN; i++) {
          d[iM].IEN(i,e) = mesh.IEN(i,e);
        }
      }

      int l = nsd; // l = 3
      int s = 0;
      int e = s + l; // e = 3

      cOut += 1; // cOut = 1
      int is = outS(cOut); // is = outS[1] = 3
      int ie = is + l; // ie = 3 + 3 = 6
      outS(cOut+1) = ie; // outS[2] = 7, outS = [0, 3, 6]

      for (int a = 0; a < mesh.nNo; a++) {
        int Ac = mesh.gN(a);
        for (int i = 0; i < nsd; i++) {
          d[iM].x(is+i,a) = uris_obj.Yd(s+i,Ac) / uris_obj.scF; // [HZ] Need to check this
        }
      }

      // For velocity
      cOut += 1; // cOut = 2
      is = outS(cOut); // is = outS[2] = 6
      ie = is + l; // ie = 6 + 3 = 9
      outS(cOut+1) = ie; // outS[3] = 7, outS = [0, 3, 6, 9]

      for (int a = 0; a < mesh.nNo; a++) {
        int Ac = mesh.gN(a);
        for (int i = 0; i < nsd; i++) {
          d[iM].x(is+i,a) = uris_obj.v(s+i,Ac);
        }
      }

      outNames[0] = "coordinates";
      outNames[1] = "URIS_displacement";
      outNames[2] = "URIS_velocity";

      nNo += mesh.nNo;
      nEl += mesh.nEl;
    }

    // Writing to vtu file (master only)
    char fName_num[100];
    if (com_mod.cTS >= 1000) {
      sprintf(fName_num, "%d", com_mod.cTS);
    } else {
      sprintf(fName_num, "%03d", com_mod.cTS);
    }
    std::string fName = com_mod.saveName + "_uris_" + uris_obj.name + "_" + fName_num + ".vtu";

    auto vtk_writer = VtkData::create_writer(fName);
    // Writing the position data
    int iOut = 0;
    int s = outS(iOut);
    int nSh = 0;
    Array<double> tmpV(maxNSD, nNo);
    tmpV = 0.0;
    for (int iM = 0; iM < uris_obj.nFa; iM++) {
      for (int a = 0; a < d[iM].nNo; a++) {
        for (int i = 0; i < nsd; i++) {
          tmpV(i,a+nSh) = d[iM].x(s+i,a);
        }
      }
      nSh += d[iM].nNo;
    }
    vtk_writer->set_points(tmpV);

    // Writing the connectivity data
    Array<int> tmpI;
    for (int iM = 0; iM < uris_obj.nFa; iM++) {
      tmpI.resize(d[iM].eNoN, d[iM].nEl);
      for (int e = 0; e < d[iM].nEl; e++) {
        for (int i = 0; i < d[iM].eNoN; i++) {
          tmpI(i,e) = d[iM].IEN(i,e);
        }
      }
      vtk_writer->set_connectivity(nsd, tmpI);
    }

    // Writing all solutions
    for (int iOut = 0; iOut < nOut; iOut++) {
      int s = outS(iOut);
      int e = outS(iOut+1);
      int l = e - s;
      tmpV.resize(l, nNo);
      tmpV = 0.0;
      int nSh = 0;
      for (int iM = 0; iM < uris_obj.nFa; iM++) {
        for (int a = 0; a < d[iM].nNo; a++) {
          for (int i = 0; i < nsd; i++) {
            tmpV(i,a+nSh) = d[iM].x(s+i,a);
          }
        }
        nSh += d[iM].nNo;
      }
      vtk_writer->set_point_data(outNames[iOut], tmpV);
    }

    vtk_writer->write();

    delete vtk_writer;

  }

}

/// @brief  Checks if a probe lies inside or outside an immersed boundary
void uris_calc_sdf(ComMod& com_mod) {
  #define n_debug_uris_calc_sdf 
  #ifdef debug_uris_calc_sdf
  DebugMsg dmsg(__func__, simulation->com_mod.cm.idcm());
  dmsg.banner();
  #endif

  using namespace consts;

  int cEq = com_mod.cEq;
  auto& eq = com_mod.eq[cEq];
  auto& cm = com_mod.cm;
  auto& uris = com_mod.uris;
  const int nsd = com_mod.nsd;
  const int nUris = com_mod.nUris;

  Array<double> xXi(nsd, nsd-1);
  Vector<double> minb(nsd), maxb(nsd), extra(nsd), xp(nsd);
  Vector<double> minb_scaf(nsd), maxb_scaf(nsd), extra_scaf(nsd), xp_scaf(nsd);
  Vector<double> xp_plane(nsd), E1(nsd), E2(nsd), v(nsd), u_interp(nsd);
  double extra_val = 0.1, extra_scaf_val = 0.1; // BBox size adjustment

  for (int iUris = 0; iUris < nUris; iUris++) {
    // We need to check if the valve needs to move 
    auto& uris_obj = uris[iUris];

    int cnt = uris_obj.clsFlg ? std::min(uris_obj.cnt, uris_obj.DxClose.nrows()) : 
                                std::min(uris_obj.cnt, uris_obj.DxOpen.nrows());
    Array3<double>& Dx = uris_obj.clsFlg ? uris_obj.DxClose : uris_obj.DxOpen;

    for (int i = 0; i < uris_obj.x.nrows(); i++) {
      for (int j = 0; j < uris_obj.x.ncols(); j++) {
        uris_obj.x(i,j) = Dx(cnt-1,i,j);
      }
    }

    for (int i = 0; i < nsd; i++) {
      for (int j = 0; j < uris_obj.tnNo; j++) {
        if (uris_obj.use_valve_velocity) {
          uris_obj.v(i,j) = (uris_obj.x(i,j) - uris_obj.x_prev(i,j)) / com_mod.dt;
        } else {
          uris_obj.v(i,j) = 0.0;
        }
        uris_obj.x_prev(i,j) = uris_obj.x(i,j);
      }
    }

    if (com_mod.cTS > 1 && cnt < uris_obj.cnt) {
      uris_obj.sdf_t = 0.0;
      if (com_mod.stFileFlag && uris_obj.sdf[0] == -1.0) {
        uris_obj.sdf = uris_obj.sdf_default;
      } else {
        continue;
      }
    } else {
      uris_obj.sdf = uris_obj.sdf_default;
      uris_obj.sdf_t = 0.0;
    }

    int max_eNoN = 0;
    for (int iM = 0; iM < uris_obj.nFa; iM++) {
      auto& mesh = uris_obj.msh[iM];
      if (mesh.eNoN > max_eNoN) {
        max_eNoN = mesh.eNoN;
      }
    }

    Array<double> lX(nsd, max_eNoN);

    if (cm.idcm() == 0) {
      std::cout << "Recomputing SDF for " << uris_obj.name << std::endl;
    }

    // Each time when the URIS moves (open/close), we need to 
    // recompute the signed distance function.
    // Find the bounding box of the valve, the BBox will be 10% larger
    // than the actual valve.
    Vector<double> minb(nsd);
    Vector<double> maxb(nsd);
    Vector<double> extra(nsd);
    for (int i = 0; i < nsd; i++) {
      minb(i) = std::numeric_limits<double>::max();
      maxb(i) = std::numeric_limits<double>::lowest();
    }

    // For each coordinate dimension, find the minimum and maximum in uris_obj.x.
    for (int i = 0; i < nsd; i++) {
      for (int j = 0; j < uris_obj.x.ncols(); j++) {
        double val = uris_obj.x(i,j);
        minb(i) = std::min(minb(i), val);
        maxb(i) = std::max(maxb(i), val);
      }
      extra(i) = (maxb(i) - minb(i)) * extra_val;
    }

    // The SDF is computed on the reference configuration, which
    // means that the valves will be morphed based on the fluid mesh
    // motion. If the fluid mesh stretches near the valve, the valve
    // leaflets will also be streched. Note that
    // this is a simplifying assumption.     
    for (int ca = 0; ca < com_mod.tnNo; ca++) {
    // int mesh_ind = 0;
    // for (int ca = 0; ca < com_mod.msh[mesh_ind].nNo; ca++) {
      double minS = std::numeric_limits<double>::max();
      for (int i = 0; i < nsd; i++) {
        xp(i) = com_mod.x(i,ca);
        // xp(i) = com_mod.x(i,com_mod.msh[mesh_ind].gN(ca));
      }
      // Is the node inside the BBox?
      bool inside = true;
      for (int i = 0; i < nsd; i++) {
        if (xp(i) < (minb(i) - extra(i)) || xp(i) > (maxb(i) + extra(i))) {
          inside = false;
          break;
        }
      }
      if (inside) {
        // This point is inside the BBox
        // Find the closest URIS face centroid
        int Ec = -1;
        int jM = -1;
        Vector<double> xb(nsd);
        for (int iM = 0; iM < uris_obj.nFa; iM++) {
          auto& mesh = uris_obj.msh[iM];
          for (int e = 0; e < mesh.nEl; e++) {
            xb = 0.0;
            for (int a = 0; a < mesh.eNoN; a++) {
              int Ac = mesh.IEN(a,e);
              for (int i = 0; i < nsd; i++) {
                xb(i) += uris_obj.x(i,Ac);
              }
            }
            for (int i = 0; i < nsd; i++) {
              xb(i) /= static_cast<double>(mesh.eNoN);
            }
            double dS = 0.0;
            for (int i = 0; i < nsd; i++) {
              dS += (xp[i] - xb[i]) * (xp[i] - xb[i]);
            }
            dS = std::sqrt(dS);

            if (dS < minS) {
              minS = dS; // closest distance
              Ec = e; // closest URIS element index
              jM = iM; // closest URIS mesh index
            }
          }
        }

        // We also need to compute the sign (above or below the valve).
        // Compute the element normal
        auto& mesh = uris_obj.msh[jM];
        xXi = 0.0;
        lX = 0.0;
        xb = 0.0;
        for (int a = 0; a < mesh.eNoN; a++) {
          int Ac = mesh.IEN(a,Ec);
          for (int i = 0; i < nsd; i++) {
            xb(i) += uris_obj.x(i,Ac);
            lX(i,a) = uris_obj.x(i,Ac);
          }
        }
        for (int i = 0; i < nsd; i++) {
          xb(i) /= static_cast<double>(mesh.eNoN);
        }

        for (int a = 0; a < mesh.eNoN; a++) {
          for (int i = 0; i < nsd - 1; i++) {
            for (int j = 0; j < nsd; j++) {
              xXi(j,i) += lX(j,a) * mesh.Nx(i,a,0);
            } 
          }
        }

        auto nV = utils::cross(xXi);
        auto Jac = sqrt(utils::norm(nV));
        if (uris_obj.reverse_normal) {
          nV = -nV / Jac;
        } else {
          nV = nV / Jac;
        }
        auto dotP = utils::norm(xp-xb, nV);

        // [HZ] Improved implementation for SDF sign
        double sdf_sign = 1.0;
        if (uris_obj.clsFlg) {
          auto dot_nrm = utils::norm(xp-xb, uris_obj.nrm);
          if (dot_nrm < 0.0 && dotP < 0.0) {
            sdf_sign = -1.0;
          } else {
            sdf_sign = 1.0;
          }
        } else {
          if (dotP < 0.0) {
            sdf_sign = -1.0;
          } else {
            sdf_sign = 1.0;
          }
        }

        uris_obj.sdf[ca] = sdf_sign * minS;

        if (uris_obj.use_valve_velocity) {
          // We need to interpolate valve velocity 
          // project xp onto the triangle plane
          for (int i = 0; i < nsd; i++) {
            xp_plane(i) = xp(i) - dotP * nV(i);
          }
          // compute barycentric (\xi, \eta) via the planar solve
          for (int i = 0; i < nsd; i++) {
            E1(i) = lX(i,1) - lX(i,0);
            E2(i) = lX(i,2) - lX(i,0);
            v(i) = xp_plane(i) - lX(i,0);
          }
          auto g11 = utils::norm(E1, E1);
          auto g12 = utils::norm(E1, E2); 
          auto g22 = utils::norm(E2, E2);
          auto b1 = utils::norm(v, E1);
          auto b2 = utils::norm(v, E2);
          double det = g11 * g22 - g12 * g12;

          double xi = (g22 * b1 - g12 * b2) / det;
          double eta = (g11 * b2 - g12 * b1) / det;

          // shape functions:
          double N1 = 1.0 - xi - eta;
          double N2 = xi;
          double N3 = eta;

          // interpolate the valve velocity:
          for (int i = 0; i < nsd; i++) {
            u_interp(i) = N1 * uris_obj.v(i, mesh.IEN(0,Ec)) +
                          N2 * uris_obj.v(i, mesh.IEN(1,Ec)) +
                          N3 * uris_obj.v(i, mesh.IEN(2,Ec));
          }
          
          // Store interpolated velocity
          for (int i = 0; i < nsd; i++) {
            uris_obj.sdf_t(i,ca) = u_interp(i);
          }
        }

      }
    }
  }

  // Compute the SDF for the scaffold mesh
  for (int iUris = 0; iUris < nUris; iUris++) {
    auto& uris_obj = uris[iUris];
    
    if (!uris_obj.scaffold_flag || com_mod.cTS > 1) {
      if (com_mod.stFileFlag && uris_obj.sdf_scaffold[0] == -1.0) {
        uris_obj.sdf_scaffold = uris_obj.sdf_default;
      } else {
        continue;
      }
    }

    auto& scaffold_mesh = uris_obj.scaffold_mesh;
    uris_obj.sdf_scaffold = uris_obj.sdf_default;

    for (int i = 0; i < nsd; i++) {
      minb_scaf(i) = std::numeric_limits<double>::max();
      maxb_scaf(i) = std::numeric_limits<double>::lowest();
    }

    // For each coordinate dimension, find the minimum and maximum in uris_obj.x.
    for (int i = 0; i < nsd; i++) {
      for (int j = 0; j < scaffold_mesh.x.ncols(); j++) {
        double val = scaffold_mesh.x(i,j);
        minb_scaf(i) = std::min(minb_scaf(i), val);
        maxb_scaf(i) = std::max(maxb_scaf(i), val);
      }
      extra_scaf(i) = (maxb_scaf(i) - minb_scaf(i)) * extra_scaf_val;
    }

    // The SDF is computed on the reference configuration, which
    // means that the valves will be morphed based on the fluid mesh
    // motion. If the fluid mesh stretches near the valve, the valve
    // leaflets will also be streched. Note that
    // this is a simplifying assumption. 
    Vector<double> xp_scaf(nsd);
    for (int ca = 0; ca < com_mod.tnNo; ca++) {
      double minS_scaf = std::numeric_limits<double>::max();
      for (int i = 0; i < nsd; i++) {
        xp_scaf(i) = com_mod.x(i,ca);
      }
      // Is the node inside the BBox?
      bool inside = true;
      for (int i = 0; i < nsd; i++) {
        if (xp_scaf(i) < (minb_scaf(i) - extra_scaf(i)) || xp_scaf(i) > (maxb_scaf(i) + extra_scaf(i))) {
          inside = false;
          break;
        }
      }

      if (inside) {
        // This point is inside the BBox
        // Find the closest URIS face centroid
        Vector<double> xb_scaf(nsd);

        for (int e = 0; e < scaffold_mesh.nEl; e++) {
          xb_scaf = 0.0;
          for (int a = 0; a < scaffold_mesh.eNoN; a++) {
            int Ac = scaffold_mesh.IEN(a,e);
            for (int i = 0; i < nsd; i++) {
              xb_scaf(i) += scaffold_mesh.x(i,Ac);
            }
          }
          for (int i = 0; i < nsd; i++) {
            xb_scaf(i) /= static_cast<double>(scaffold_mesh.eNoN);
          }
          double dS_scaf = 0.0;
          for (int i = 0; i < nsd; i++) {
            dS_scaf += (xp_scaf[i] - xb_scaf[i]) * (xp_scaf[i] - xb_scaf[i]);
          }
          dS_scaf = std::sqrt(dS_scaf);

          if (dS_scaf < minS_scaf) {
            minS_scaf = dS_scaf;
          }
        }
        uris_obj.sdf_scaffold[ca] = minS_scaf;
      }
    }
  }

}


/// @brief Create data for a mesh.
///
/// Replicates Fortran READSV subroutine defined in LOADMSH.f.
///
///   SUBROUTINE READSV(list, lM)
//

void uris_read_sv(Simulation* simulation, mshType& mesh, const URISFaceParameters* mesh_param) {
  #define n_dbg_read_sv
  #ifdef dbg_read_sv
    DebugMsg dmsg(__func__, simulation->com_mod.cm.idcm());
    dmsg.banner();
    dmsg << "Mesh name: " << mesh_name;
    dmsg << "Mesh path: " << mesh_path;
    dmsg << "mesh.lShl: " << mesh.lShl;
  #endif

  auto mesh_path = mesh_param->face_file_path();
  auto mesh_name = mesh_param->name();
  // Read in volume mesh.
  vtk_xml::read_vtu(mesh_path, mesh);

  // Check that the input number of spatial dimensions is consistent 
  // with the types of elements defined for the simulation mesh.
  //
  int nsd = simulation->com_mod.nsd;
  int elem_dim = consts::element_dimension.at(mesh.eType);
  auto elem_type = consts::element_type_to_string.at(mesh.eType);

  if (mesh.lShl) { 
    if (nsd == 1) {
      throw std::runtime_error("The number of spatial dimensions (" + std::to_string(nsd) + 
          ") is not consistent with the mesh '" + mesh.name + "' which contains shell elements.");
    }

  } else if (!mesh.lFib) { 
    if (elem_dim != nsd) {
      throw std::runtime_error("The number of spatial dimensions (" + std::to_string(nsd) + 
        ") is not consistent with the mesh '" + mesh.name + "' which contains " + elem_type + " elements.");
    }
  }

  // Set mesh element properites for the input element type.
  nn::select_ele(simulation->com_mod, mesh);

  // Check the mesh element node ordering.
  //
  // Note: This may change element node ordering.
  //
  auto &com_mod = simulation->get_com_mod();
  if (com_mod.ichckIEN) {
      read_msh_ns::check_ien(simulation, mesh);
  }
}


/// @brief This routine gives the distance between two points 

int in_poly(Vector<double>& P, Array<double>& P1, bool ext) {
  int nd = P1.nrows();
  bool flag = true;
  int inpoly = 0;

  Vector<double> N(nd);
  N = 0.0;

  if (nd == 2) {
    for (int i = 0; i <= nd; i++) {
      // compute normal in 2D for P2-P1 P3-P1
      if (i != nd) {
        N(0) = P1(1,i) - P1(1,i+1);
        N(1) = P1(0,i+1) - P1(0,i);
      } else {
        N(0) = P1(1,i) - P1(1,0);
        N(1) = P1(0,0) - P1(0,i);
      }
      // test dot product between P-P1 and the normals
      double dotP = N(0)*(P(0)-P1(0,i)) + N(1)*(P(1)-P1(1,i));
      if (dotP < 0.0) {
        flag = false;
        break;
      }
    }
    if (flag) {
      inpoly = 1;
    }
  } else {
    Vector<double> v1 = P1.col(0);
    Vector<double> v2 = P1.col(1);
    Vector<double> v3 = P1.col(2);
    Vector<double> v4 = P1.col(3);
    int s1 = same_side(v1, v2, v3, v4, P, ext);
    int s2 = same_side(v2, v3, v4, v1, P, ext);
    int s3 = same_side(v3, v4, v1, v2, P, ext);
    int s4 = same_side(v4, v1, v2, v3, P, ext);
    inpoly = (s1 + s2 + s3 + s4) / 4;
  }

  return inpoly;
}


/// @brief Chech if a point is on the same side of anotehr point wrt a triangle in 3D
int same_side(Vector<double>& v1, Vector<double>& v2, Vector<double>& v3,
              Vector<double>& v4, Vector<double>& p, bool ext) {
  #define n_dbg_same_side
  #ifdef dbg_same_side
    DebugMsg dmsg(__func__, simulation->com_mod.cm.idcm());
    dmsg.banner();
    dmsg << "checking same side";
  #endif

  int sameside = 0;
  double eps = 2.0e-4;

  Vector<double> v21 = v2 - v1;
  Vector<double> v31 = v3 - v1;
  Array<double> V(3,2);
  V.set_col(0, v21);
  V.set_col(1, v31);
  Vector<double> v41 = v4 - v1;
  Vector<double> vp1 = p - v1;
  Vector<double> N = utils::cross(V);
  double dotV4 = utils::norm(N, v41);
  double dotP = utils::norm(N, vp1);
  // check if P and P4 are from the same side
  if ((dotP >= 0 && dotV4 >= 0) || (dotP < 0 && dotV4 < 0)) {
    sameside = 1;
  }

  // If it is not, check if it is on any face, it might on the face 
  if (ext && sameside != 1) {
    if (std::fabs(dotP) <= eps) {
      sameside = 1;
    }
  }

  return sameside;
}

}