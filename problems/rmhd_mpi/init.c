/* ///////////////////////////////////////////////////////////////////// */
/*! 
  \file  
  \brief Contains basic functions for problem initialization.

  The init.c file collects most of the user-supplied functions useful 
  for problem configuration.
  It is automatically searched for by the makefile.

  \author A. Mignone (mignone@ph.unito.it)
  \date   March 5, 2017
*/
/* ///////////////////////////////////////////////////////////////////// */
#include "pluto.h"

/* ********************************************************************* */
void Init (double *v, double x1, double x2, double x3)
/*! 
 * The Init() function can be used to assign initial conditions as
 * as a function of spatial position.
 *
 * \param [out] v   a pointer to a vector of primitive variables
 * \param [in] x1   coordinate point in the 1st dimension
 * \param [in] x2   coordinate point in the 2nd dimension
 * \param [in] x3   coordinate point in the 3rdt dimension
 *
 * The meaning of x1, x2 and x3 depends on the geometry:
 * \f[ \begin{array}{cccl}
 *    x_1  & x_2    & x_3  & \mathrm{Geometry}    \\ \noalign{\medskip}
 *     \hline
 *    x    &   y    &  z   & \mathrm{Cartesian}   \\ \noalign{\medskip}
 *    R    &   z    &  -   & \mathrm{cylindrical} \\ \noalign{\medskip}
 *    R    & \phi   &  z   & \mathrm{polar}       \\ \noalign{\medskip}
 *    r    & \theta & \phi & \mathrm{spherical} 
 *    \end{array}
 *  \f]
 *
 * Variable names are accessed by means of an index v[nv], where
 * nv = RHO is density, nv = PRS is pressure, nv = (VX1, VX2, VX3) are
 * the three components of velocity, and so forth.
 *
 *********************************************************************** 
 * GRID
 * X1-grid    2    10000.0     1500    u    13000.0     5000     s    300000.0
 * X1-grid    2    10000.0     1500    u    13000.0     5000     s    1000000.0
 * X1-grid    2    10000.0     1500    u    13000.0     10000    s    10000000.0 
 * X1-grid    2    10000.0     1500    u    13000.0     3500     s    20000.0    */
{

  //double G, M_ns, rho_in, v_in, v_inf, R_in, R_inf, v_esc;
  //double G, M_ns, rho_in, v_init, v_inf_init, R_in, R_inf, v_esc;
  double rho_in, v_init, v_inf_init, R_in, R_inf;
  double cs0, cs02, theta0, p0;


  //G = 6.674e-8*UNIT_DENSITY*pow(UNIT_LENGTH/UNIT_VELOCITY,2);
  //M_ns = g_inputParam[M_NS]*1.988e33/UNIT_DENSITY/pow(UNIT_LENGTH,3);
  g_gamma = g_inputParam[GAMMA];
  rho_in = g_inputParam[RHO_IN]/UNIT_DENSITY;
  //v_in = g_inputParam[V_IN];
  //v_inf = g_inputParam[V_INF];

  v_init = g_inputParam[V_INIT]; 
  v_inf_init = g_inputParam[V_INF_INIT];
  
  cs0 = g_inputParam[CS_REL_0];
  cs02 = cs0*cs0;

  if (cs02 <= 0.0 || cs02 >= g_gamma - 1.0) {
  printLog("! CS_REL_0 must satisfy 0 < CS_REL_0^2 < GAMMA - 1\n");
  QUIT_PLUTO(1);
  }

  theta0 = cs02/
          (g_gamma - g_gamma*cs02/(g_gamma - 1.0));

  p0 = rho_in*theta0;


  R_in = 12.0;
  R_inf = (v_inf_init/v_init-1.0)*R_in;
  //v_esc = pow(2.0*G*M_ns/R_in,0.5);

  v[RHO] = rho_in*pow(R_in/x1,2)*(1.0+R_inf/x1)/(1.0+R_inf/R_in);
  v[VX1] = v_inf_init/(1.0+R_inf/x1);
  v[VX2] = 0.0;
  v[VX3] = 0.0;
  #if HAVE_ENERGY
  //v[PRS] = (g_gamma-1.0)/2.0/g_gamma*rho_in*pow(R_in/x1,2)*(1.0+R_inf/x1)/(1.0+R_inf/R_in)*(pow(v_inf,2)*R_inf*(2.0*x1+R_inf)/pow(x1+R_inf,2)+pow(v_esc,2)*R_in/x1);
  v[PRS] = p0*pow(v[RHO]/rho_in, g_gamma);
  #endif
  v[TRC] = 0.0;

  //if ((x1 >= R_surf) && (x1 <= R_bullet) && (x2 <= 0.643501)) v[PRS] = P_shock;

  #if PHYSICS == RMHD
  {
  double Bstar_code;

  Bstar_code = g_inputParam[B_SURF]/(UNIT_VELOCITY*sqrt(4.0*CONST_PI*UNIT_DENSITY));

  v[BX1] = Bstar_code*pow(R_in/x1, 2.0);
  v[BX2] = 0.0;
  v[BX3] = 0.0;  
  }
  #endif
}
/* ********************************************************************* */
void InitDomain (Data *d, Grid *grid)
/*! 
 * Assign initial condition by looping over the computational domain.
 * Called after the usual Init() function to assign initial conditions
 * on primitive variables.
 * Value assigned here will overwrite those prescribed during Init().
 *
 *
 *********************************************************************** */
{
}

/* ********************************************************************* */
void Analysis (const Data *d, Grid *grid)
/*! 
 *  Perform runtime data analysis.
 *
 * \param [in] d the PLUTO Data structure
 * \param [in] grid   pointer to array of Grid structures  
 *
 *********************************************************************** */
{

}
#if PHYSICS == MHD
/* ********************************************************************* */
void BackgroundField (double x1, double x2, double x3, double *B0)
/*!
 * Define the component of a static, curl-free background 
 * magnetic field.
 *
 * \param [in] x1  position in the 1st coordinate direction \f$x_1\f$
 * \param [in] x2  position in the 2nd coordinate direction \f$x_2\f$
 * \param [in] x3  position in the 3rd coordinate direction \f$x_3\f$
 * \param [out] B0 array containing the vector componens of the background
 *                 magnetic field
 *********************************************************************** */
{
   B0[0] = 0.0;
   B0[1] = 0.0;
   B0[2] = 0.0;
}
#endif

/* ********************************************************************* */
void UserDefBoundary (const Data *d, RBox *box, int side, Grid *grid)
/*! 
 *  Assign user-defined boundary conditions.
 *
 * \param [in,out] d  pointer to the PLUTO data structure containing
 *                    cell-centered primitive quantities (d->Vc) and 
 *                    staggered magnetic fields (d->Vs, when used) to 
 *                    be filled.
 * \param [in] box    pointer to a RBox structure containing the lower
 *                    and upper indices of the ghost zone-centers/nodes
 *                    or edges at which data values should be assigned.
 * \param [in] side   specifies the boundary side where ghost zones need
 *                    to be filled. It can assume the following 
 *                    pre-definite values: X1_BEG, X1_END,
 *                                         X2_BEG, X2_END, 
 *                                         X3_BEG, X3_END.
 *                    The special value side == 0 is used to control
 *                    a region inside the computational domain.
 * \param [in] grid  pointer to an array of Grid structures.
 *
 *********************************************************************** */
 {
  int   i, j, k, nv;
  double  *x1, *x2, *x3;

  x1 = grid->x[IDIR];
  x2 = grid->x[JDIR];
  x3 = grid->x[KDIR];

 // double G, M_ns, rho_in, v_in, v_inf, R_in, R_inf, v_esc;

// G = 6.674e-8*UNIT_DENSITY*pow(UNIT_LENGTH/UNIT_VELOCITY,2);
/// M_ns = g_inputParam[M_NS]*1.988e33/UNIT_DENSITY/pow(UNIT_LENGTH,3);
//g_gamma = g_inputParam[GAMMA];
// rho_in = g_inputParam[RHO_IN]/UNIT_DENSITY;
//v_in = g_inputParam[V_IN];
//v_inf = g_inputParam[V_INF];
  
//  R_in = 12.0;
//  R_inf = (v_inf/v_in-1.0)*R_in;
//  v_esc = pow(2.0*G*M_ns/R_in,0.5);



  double rho_in, R_in, Bstar_code;
  double cs0, cs02, theta0, p0;
  int i_live;

  g_gamma = g_inputParam[GAMMA];
  rho_in = g_inputParam[RHO_IN]/UNIT_DENSITY;
  R_in = 12.0;

  #if PHYSICS == RMHD
  Bstar_code =
    g_inputParam[B_SURF]/
    (UNIT_VELOCITY*sqrt(4.0*CONST_PI*UNIT_DENSITY));
  #endif
  
  
  cs0 = g_inputParam[CS_REL_0];
  cs02 = cs0*cs0;

  if (cs02 <= 0.0 || cs02 >= g_gamma - 1.0) {
    printLog("! CS_REL_0 must satisfy 0 < CS_REL_0^2 < GAMMA - 1\n");
    QUIT_PLUTO(1);
  }

  theta0 = cs02/
           (g_gamma - g_gamma*cs02/(g_gamma - 1.0));

  p0 = rho_in*theta0;

  if (side == 0) {
    i_live = IBEG;

    while (i_live <= IEND && x1[i_live] <= R_in) {
      i_live++;
    }

    if (i_live > IEND) {
      printLog("! Cannot find a live cell outside R_in\n");
      QUIT_PLUTO(1);
    }

    TOT_LOOP(k,j,i) {
      if (x1[i] <= R_in) {
        double rho_bc, v_base;

        rho_bc = rho_in*pow(R_in/x1[i], 2.0);
        v_base = d->Vc[VX1][k][j][i_live];

        d->Vc[RHO][k][j][i] = rho_bc;
        d->Vc[PRS][k][j][i] = p0*pow(rho_bc/rho_in, g_gamma);

        d->Vc[VX1][k][j][i] = v_base;
        d->Vc[VX2][k][j][i] = 0.0;
        d->Vc[VX3][k][j][i] = 0.0;

        #if PHYSICS == RMHD
        d->Vc[BX1][k][j][i] = Bstar_code*pow(R_in/x1[i], 2.0);
        d->Vc[BX2][k][j][i] = 0.0;
        d->Vc[BX3][k][j][i] = 0.0;
        #endif

        d->flag[k][j][i] |= FLAG_INTERNAL_BOUNDARY;
      }
    }
  }
  
//  if (side == 0) {    /* -- check solution inside domain -- */
 //   TOT_LOOP(k,j,i){
  //    if (x1[i] <= R_in) {
   //     d->Vc[RHO][k][j][i] = rho_in*pow(R_in/x1[i],2)*(1.0+R_inf/x1[i])/(1.0+R_inf/R_in);
    //    d->Vc[VX1][k][j][i] = v_inf/(1.0+R_inf/x1[i]);
     //   d->Vc[VX2][k][j][i] = 0.0;
      //  d->Vc[VX3][k][j][i] = 0.0;
       // d->Vc[PRS][k][j][i] = (g_gamma-1.0)/2.0/g_gamma*rho_in*pow(R_in/x1[i],2)*(1.0+R_inf/x1[i])/(1.0+R_inf/R_in)*(pow(v_inf,2)*R_inf*(2.0*x1[i]+R_inf)/pow(x1[i]+R_inf,2)+pow(v_esc,2)*R_in/x1[i]);
       // d->flag[k][j][i] |= FLAG_INTERNAL_BOUNDARY;
     // }
    //}
  //}

  if (side == X1_BEG){  /* -- X1_BEG boundary -- */
    if (box->vpos == CENTER) {
      BOX_LOOP(box,k,j,i){ }
    }else if (box->vpos == X1FACE){
      BOX_LOOP(box,k,j,i){ }
    }else if (box->vpos == X2FACE){
      BOX_LOOP(box,k,j,i){ }
    }else if (box->vpos == X3FACE){
      BOX_LOOP(box,k,j,i){ }
    }
  }

  if (side == X1_END){  /* -- X1_END boundary -- */
    if (box->vpos == CENTER) {
      BOX_LOOP(box,k,j,i){  }
    }else if (box->vpos == X1FACE){
      BOX_LOOP(box,k,j,i){  }
    }else if (box->vpos == X2FACE){
      BOX_LOOP(box,k,j,i){  }
    }else if (box->vpos == X3FACE){
      BOX_LOOP(box,k,j,i){  }
    }
  }

  if (side == X2_BEG){  /* -- X2_BEG boundary -- */
    if (box->vpos == CENTER) {
      BOX_LOOP(box,k,j,i){  }
    }else if (box->vpos == X1FACE){
      BOX_LOOP(box,k,j,i){  }
    }else if (box->vpos == X2FACE){
      BOX_LOOP(box,k,j,i){  }
    }else if (box->vpos == X3FACE){
      BOX_LOOP(box,k,j,i){  }
    }
  }

  if (side == X2_END){  /* -- X2_END boundary -- */
    if (box->vpos == CENTER) {
      BOX_LOOP(box,k,j,i){  }
    }else if (box->vpos == X1FACE){
      BOX_LOOP(box,k,j,i){  }
    }else if (box->vpos == X2FACE){
      BOX_LOOP(box,k,j,i){  }
    }else if (box->vpos == X3FACE){
      BOX_LOOP(box,k,j,i){  }
    }
  }

  if (side == X3_BEG){  /* -- X3_BEG boundary -- */
    if (box->vpos == CENTER) {
      BOX_LOOP(box,k,j,i){  }
    }else if (box->vpos == X1FACE){
      BOX_LOOP(box,k,j,i){  }
    }else if (box->vpos == X2FACE){
      BOX_LOOP(box,k,j,i){  }
    }else if (box->vpos == X3FACE){
      BOX_LOOP(box,k,j,i){  }
    }
  }

  if (side == X3_END){  /* -- X3_END boundary -- */
    if (box->vpos == CENTER) {
      BOX_LOOP(box,k,j,i){  }
    }else if (box->vpos == X1FACE){
      BOX_LOOP(box,k,j,i){  }
    }else if (box->vpos == X2FACE){
      BOX_LOOP(box,k,j,i){  }
    }else if (box->vpos == X3FACE){
      BOX_LOOP(box,k,j,i){  }
    }
  }
}

#if BODY_FORCE != NO
/* ********************************************************************* */
void BodyForceVector(double *v, double *g, double x1, double x2, double x3)
/*!
 * Prescribe the acceleration vector as a function of the coordinates
 * and the vector of primitive variables *v.
 *
 * \param [in] v  pointer to a cell-centered vector of primitive 
 *                variables
 * \param [out] g acceleration vector
 * \param [in] x1  position in the 1st coordinate direction \f$x_1\f$
 * \param [in] x2  position in the 2nd coordinate direction \f$x_2\f$
 * \param [in] x3  position in the 3rd coordinate direction \f$x_3\f$
 *
 *********************************************************************** */
{
  double G, M_ns;

  G = 6.674e-8*UNIT_DENSITY*pow(UNIT_LENGTH/UNIT_VELOCITY,2);
  M_ns = g_inputParam[M_NS]*1.988e33/UNIT_DENSITY/pow(UNIT_LENGTH,3);

  g[IDIR] = -G*M_ns/pow(x1,2);
  g[JDIR] = 0.0;
  g[KDIR] = 0.0;
}
/* ********************************************************************* */
double BodyForcePotential(double x1, double x2, double x3)
/*!
 * Return the gravitational potential as function of the coordinates.
 *
 * \param [in] x1  position in the 1st coordinate direction \f$x_1\f$
 * \param [in] x2  position in the 2nd coordinate direction \f$x_2\f$
 * \param [in] x3  position in the 3rd coordinate direction \f$x_3\f$
 * 
 * \return The body force potential \f$ \Phi(x_1,x_2,x_3) \f$.
 *
 *********************************************************************** */
{
  return 0.0;
}
#endif