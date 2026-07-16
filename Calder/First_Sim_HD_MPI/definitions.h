#define  PHYSICS                        HD
#define  DIMENSIONS                     1
#define  GEOMETRY                       SPHERICAL
#define  BODY_FORCE                     VECTOR
#define  COOLING                        NO
#define  RECONSTRUCTION                 LINEAR
#define  TIME_STEPPING                  RK2
#define  NTRACER                        0
#define  PARTICLES                      NO
#define  USER_DEF_PARAMETERS            5

/* -- physics dependent declarations -- */

#define  DUST_FLUID                     NO
#define  EOS                            IDEAL
#define  ENTROPY_SWITCH                 ALWAYS
#define  INCLUDE_LES                    NO
#define  THERMAL_CONDUCTION             NO
#define  VISCOSITY                      NO
#define  RADIATION                      NO
#define  ROTATING_FRAME                 NO

/* -- user-defined parameters (labels) -- */

#define  M_NS                           0
#define  GAMMA                          1
#define  RHO_IN                         2
#define  V_IN                           3
#define  V_INF                          4

/* [Beg] user-defined constants (do not change this line) */

#define  UNIT_DENSITY                   1.0e7
#define  UNIT_LENGTH                    1.0e5
#define  UNIT_VELOCITY                  2.998e10
#define  H_MASS_FRAC                    0.2
#define  He_MASS_FRAC                   0.8
#define  INTERNAL_BOUNDARY              YES
#define  INTERNAL_BOUNDARY_CFL          NO

/* [End] user-defined constants (do not change this line) */
