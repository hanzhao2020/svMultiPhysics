# **Problem Description**
Simulation of a 2D square pipe flow with an immersed 2D block.

The model parameters are specified in the `Immersed_method` sub-section
```
<Add_equation type="immersed_FSI" > 
   <Coupled> true </Coupled>
   <Min_iterations> 3 </Min_iterations>  
   <Max_iterations> 10 </Max_iterations> 
   <Tolerance> 1e-4 </Tolerance> 
   
  <Immersed_method type="IFEM">
    <Background_fluid_domain> 0 </Background_fluid_domain>
    <Immersed_solid_domain> 1 </Immersed_solid_domain>
    <Coupling_method type="monolithic"> </Coupling_method>
    <Interpolation type="FE"> </Interpolation>
    <VMS_stabilization_s> 100.0 </VMS_stabilization_s>
    <VMS_stabilization_width> 1.0 </VMS_stabilization_width>
  </Immersed_method>
```
