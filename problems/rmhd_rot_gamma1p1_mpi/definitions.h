#define  PHYSICS                        RMHD
#define  DIMENSIONS                     1
#define  GEOMETRY                       SPHERICAL
#define  BODY_FORCE                     VECTOR
#define  COOLING                        NO
#define  RECONSTRUCTION                 LINEAR
#define  TIME_STEPPING                  RK2
#define  NTRACER                        0
#define  PARTICLES                      NO
#define  USER_DEF_PARAMETERS            8

/* -- physics dependent declarations -- */

#define  EOS                            IDEAL
#define  ENTROPY_SWITCH                 SELECTIVE
#define  RADIATION                      NO
#define  DIVB_CONTROL                   NO

/* -- user-defined parameters (labels) -- */

#define  M_NS                           0
#define  GAMMA                          1
#define  RHO_IN                         2
#define  CS_REL_0                       3
#define  V_INIT                         4
#define  V_INF_INIT                     5
#define  B_SURF                         6
#define  P_ROT_MS                       7

/* [Beg] user-defined constants (do not change this line) */

#define  UNIT_DENSITY                   1.0e7
#define  UNIT_LENGTH                    1.0e5
#define  UNIT_VELOCITY                  CONST_c
#define  H_MASS_FRAC                    0.2
#define  He_MASS_FRAC                   0.8
#define  INTERNAL_BOUNDARY              YES
#define  INTERNAL_BOUNDARY_CFL          NO

/* [End] user-defined constants (do not change this line) */
