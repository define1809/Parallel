#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <mpi.h>

#define TRUE (1)
#define FALSE (0)

#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))

enum ProcLocation {
  LOC_NONE = 0,           // 0
  LOC_INNER,              // 1 
  LOC_INNER_BOT,          // 2
  LOC_INNER_TOP,          // 3
  LOC_INNER_LEFT,         // 4
  LOC_INNER_RIGHT,        // 5
  LOC_CORNER_BOTLEFT,     // 6
  LOC_CORNER_BOTRIGHT,    // 7
  LOC_CORNER_TOPLEFT,     // 8
  LOC_CORNER_TOPRIGHT,    // 9
  LOC_CUP,                // 10
  LOC_CAP,                // 11
  LOC_GLOBAL,             // 12
};
typedef enum ProcLocation ProcLocation_t;

// Structure which contains process location info and neighbors
struct ProcInfo {
  // Process rank
  int rank;
  // Process topological coords
  int coords[2];
  // Ranks of the neighbors of the process
  int left, right, up, down;
  // Process local domain size
  size_t m, n;
  // Process global domain start and end point
  // start[0] <= points[i] < end[0]
  // start[1] <= points[j] < end[1]
  size_t start[2], end[2];
  // Procces location type
  ProcLocation_t proc_loc;
};
typedef struct ProcInfo ProcInfo_t;

// There are variables from my variant 2.
// u(x, y) = u_2(x, y) = sqrt(4 + x*y) 
// k(x, y) = k_3(x, y) = 4 + x + y
// q(x, y) = q_2(x, y) = x + y

double u2(double x, double y) {
  return 4.0 + x * y;
}

double u(double x, double y) {
  return sqrt(u2(x, y));
}

double k(double x, double y) {
  return 4.0 + x + y;
}

double q(double x, double y) {
  return x + y;
}

// This is analytically computed right side F(x, y) of equation.

double F(double x, double y) {
  return (k(x, y) * (x * x + y * y) - 2.0 * q(x, y) * u2(x, y)) / (4.0 * sqrt(u2(x, y)) * u2(x,y)) + q(x, y) * u(x, y);
}

// These are the boundary conditions calculated analytically for my variant 2.
// Г_R: phi(x, y) = u(x, y)
// Г_L: phi(x, y) = u(x, y)
// Г_T: phi(x, y) = u(x, y)
// Г_B: psi(x, y) = - (x*k(x, y)) / (2*u(x, y)) + sqrt(k(x, y))

double phi(double x, double y) {
  return u(x, y);
}

double psi(double x, double y) {
  return u(x, y) - (x * k(x, y)) / (2.0 * u(x, y));
}

// Grid coefficients.
// gi and gj means global i and global j in all code
double a(size_t gi, size_t gj, double h1, double h2) {
  return k(gi * h1 - 0.5 * h1, gj * h2);
}

double b(size_t gi, size_t gj, double h1, double h2) {
  return k(gi * h1, gj * h2 - 0.5 * h2);
}

// Right and left difference partial derivatives.
// li and lj means local i and local i in all code

double right_dx(double** v, size_t li, size_t lj, double h1) {
  return (v[li + 1][lj] - v[li][lj]) / h1;
}

double left_dx(double** v, size_t li, size_t lj, double h1) {
  return (v[li][lj] - v[li - 1][lj]) / h1;
}

double right_dy(double** v, size_t li, size_t lj, double h2) {
  return (v[li][lj + 1] - v[li][lj]) / h2;
}

double left_dy(double** v, size_t li, size_t lj, double h2) {
  return (v[li][lj] - v[li][lj - 1]) / h2;
}

// Laplace operator.

double left_delta(double** w, size_t li, size_t lj, size_t gi, size_t gj, double h1, double h2) {
  return (1.0 / h1) * (k(gi * h1 + 0.5 * h1, gj * h2) * right_dx(w, li, lj, h1) - a(gi, gj, h1, h2) * left_dx(w, li, lj, h1));
}

double right_delta(double** w, size_t li, size_t lj, size_t gi, size_t gj, double h1, double h2) {
  return (1.0 / h2) * (k(gi * h1, gj * h2 + 0.5 * h2) * right_dy(w, li, lj, h2) - b(gi, gj, h1, h2) * left_dy(w, li, lj, h2));
}

double laplace_operator(double** w, size_t li, size_t lj, size_t gi, size_t gj, double h1, double h2) {
  return left_delta(w, li, lj, gi, gj, h1, h2) + right_delta(w, li, lj, gi, gj, h1, h2);
}

// Fill B (right part of Aw = B).
// TODO: rename RHS
void fill_B(double** B, double h1, double h2, size_t M, size_t N, ProcInfo_t *info) {
  // Local iteration variables for position in local domain grid of proc
  size_t li, lj;
  // Global iteration variables for position in global domain grid
  size_t gi, gj;
  // Fill domains of B
  switch(info->proc_loc) {
  case LOC_INNER:
    // Internal grid points of inner domain
    for (li = 1; li < info->m + 1; ++li) {
      gi = info->start[0] + li - 1; 
      for (lj = 1; lj < info->n + 1; ++lj) {
        gj = info->start[1] + lj - 1;
        B[li][lj] = F(gi * h1, gj * h2); 
      }
    }
    break;
  case LOC_INNER_BOT:
    // Internal grid points in domain which connected with bottom 
    for (li = 1; li < info->m + 1; ++li) {
      gi = info->start[0] + li - 1;
      for (lj = 2; lj < info->n + 1; ++lj) {
        gj = info->start[1] + lj - 1;
        B[li][lj] = F(gi * h1, gj * h2);
      }
    }  
    // Bottom grid points in domain which connected with bottom
    for (li = 1; li < info->m + 1; ++li) {
      gi = info->start[0] + li - 1;
      B[li][1] = F(gi * h1, 0.0) + (2.0 / h2) * psi(gi * h1, 0.0);
    }
    break;
  case LOC_INNER_TOP:
    // Internal grid points in domain which connected with top
    for (li = 1; li < info->m + 1; ++li) {
      gi = info->start[0] + li - 1;
      for (lj = 1; lj < info->n; ++lj) {
        gj = info->start[1] + lj - 1;
        B[li][lj] = F(gi * h1, gj * h2);
      }
    }
    // Top grid points in domain which connected with top
    for (li = 1; li < info->m + 1; ++li) {
      gi = info->start[0] + li - 1;
      B[li][info->n] = phi(gi * h1, N * h2);
    }
    break;
  case LOC_INNER_LEFT:
    // Internal grid points in domain which connected with left
    for (li = 2; li < info->m + 1; ++li) {
      gi = info->start[0] + li - 1;
      for (lj = 1; lj < info->n + 1; ++lj) {
        gj = info->start[1] + lj - 1;
        B[li][lj] = F(gi * h1, gj * h2);
      }
    }
    // Left grid points in domain which connected with left
    for (lj = 1; lj < info->n + 1; ++lj) {
      gj = info->start[1] + lj - 1;
      B[1][lj] = phi(0.0, gj * h2);
    }
    break;
  case LOC_INNER_RIGHT:
    // Internal grid points in domain which connected with right
    for (li = 1; li < info->m; ++li) {
      gi = info->start[0] + li - 1;
      for (lj = 1; lj < info->n + 1; ++lj) {
        gj = info->start[1] + lj - 1; 
        B[li][lj] = F(gi * h1, gj * h2);
      }
    }
    // Right grid points in domain which connected with right
    for (lj = 1; lj < info->n + 1; ++lj) {
      gj = info->start[1] + lj - 1;
      B[info->m][lj] = phi(M * h1, gj * h2);
    }
    break;
  case LOC_CORNER_BOTLEFT:
    // Internal grid points in corner bot-left domain
    for (li = 2; li < info->m + 1; ++li) {
      gi = info->start[0] + li - 1;
      for (lj = 2; lj < info->n + 1; ++lj) {
        gj = info->start[1] + lj - 1;
        B[li][lj] = F(gi * h1, gj * h2); 
      }
    }
    // Bottom grid points in corner bot-left domain
    for (li = 2; li < info->m + 1; ++li) {
      gi = info->start[0] + li - 1;
      B[li][1] = F(gi * h1, 0.0) + (2.0 / h2) * psi(gi * h1, 0.0);
    }
    // Left grid points in corner bot-left domain
    for (lj = 2; lj < info->n + 1; ++lj) {
      gj = info->start[1] + lj - 1;
      B[1][lj] = phi(0.0, gj * h2);
    }
    // Bot-left corner
    B[1][1] = phi(0.0, 0.0);
    break;
  case LOC_CORNER_BOTRIGHT:
    // Internal grid points in corner bot-right domain
    for (li = 1; li < info->m; ++li) {
      gi = info->start[0] + li - 1;
      for (lj = 2; lj < info->n + 1; ++lj) {
        gj = info->start[1] + lj - 1;
        B[li][lj] = F(gi * h1, gj * h2);
      }
    }
    // Bottom grid points in corner bot-right domain
    for (li = 1; li < info->m; ++li) {
      gi = info->start[0] + li - 1;
      B[li][1] = F(gi * h1, 0.0) + (2.0 / h2) * psi(gi * h1, 0.0);
    }
    // Right grid points in corner bot-right domain
    for (lj = 2; lj < info->n + 1; ++lj) {
      gj = info->start[1] + lj - 1;
      B[info->m][lj] = phi(M * h1, gj * h2);
    }
    // Bot-right corner
    B[info->m][1] = phi(M * h1, 0.0);  
    break;
  case LOC_CORNER_TOPLEFT:
    // Internal grid points in corner top-left domain
    for (li = 2; li < info->m + 1; ++li) {
      gi = info->start[0] + li - 1;
      for (lj = 1; lj < info->n; ++lj) {
        gj = info->start[1] + lj - 1;
        B[li][lj] = F(gi * h1, gj * h2);
      }
    }
    // Top grid points in corner top-left domain
    for (li = 2; li < info->m + 1; ++li) {
      gi = info->start[0] + li - 1;
      B[li][info->n] = phi(gi * h1, N * h2);
    }
    // Left grid points in corner top-left domain
    for (lj = 1; lj < info->n; ++lj) {
      gj = info->start[1] + lj - 1;
      B[1][lj] = phi(0.0, gj * h2);
    } 
    // Top-left corner
    B[1][info->n] = phi(0.0, N * h2);
    break;
  case LOC_CORNER_TOPRIGHT:
    // Internal grid point in corner top-right domain
    for (li = 1; li < info->m; ++li) {
      gi = info->start[0] + li - 1;
      for (lj = 1; lj < info->n; ++lj) {
        gj = info->start[1] + lj - 1;
        B[li][lj] = F(gi * h1, gj * h2);
      }
    }
    // Top grid points in corner top-right domain
    for (li = 1; li < info->m; ++li) {
      gi = info->start[0] + li - 1;
      B[li][info->n] = phi(gi * h1, N * h2);
    }
    // Right grid points in corner top-right domain
    for (lj = 1; lj < info->n; ++lj) {
      gj = info->start[1] + li - 1;
      B[info->m][lj] = phi(M * h1, gj * h2);
    }
    // Top-right corner
    B[info->m][info->n] = phi(M * h1, N * h2);
    break;
  case LOC_CUP:
    // Internal grid points in CUP-shaped domain
    for (li = 2; li < info->m; ++li) {
      gi = info->start[0] + li - 1;
      for (lj = 2; lj < info->n + 1; ++lj) {
        gj = info->start[1] + lj - 1;
        B[li][lj] = F(gi * h1, gj * h2);
      }
    }
    // Bottom grid points in CUP-shaped domain
    for (li = 2; li < info->m; ++li) {
      gi = info->start[0] + li - 1;
      B[li][1] = F(gi * h1, 0.0) + (2.0 / h2) * psi(gi * h1, 0.0);
    }
    // Left and right grid points in CUP-shaped domain
    for (lj = 2; lj < info->n + 1; ++lj) {
      gj = info->start[1] + lj - 1;
      B[1][lj] = phi(0.0, gj * h2);
      B[info->m][lj] = phi(M * h1, gj * h2);
    }
    // Bot-left corner
    B[1][1] = phi(0.0, 0.0);
    // Bot-right corner
    B[info->m][1] = phi(M * h1, 0.0);
    break;
  case LOC_CAP:
    // Internal grid points in CAP-shaped domain
    for (li = 2; li < info->m; ++li) {
      gi = info->start[0] + li - 1;
      for (lj = 1; lj < info->n; ++lj) {
        gj = info->start[1] + lj - 1;
        B[li][lj] = F(gi * h1, gj * h2);
      }
    }
    // Top grid points in CAP-shaped domain
    for (li = 2; li < info->m; ++li) {
      gi = info->start[0] + li - 1;
      B[li][info->n] = phi(gi * h1, N * h2); 
    }
    // Left and right grid points in CAP-shaped domain
    for (lj = 1; lj < info->n; ++lj) {
      gj = info->start[1] + lj - 1;
      B[1][lj] = phi(0.0, gj * h2);
      B[info->m][lj] = phi(M * h1, gj * h2);
    }
    // Top-left corner
    B[1][info->n] = phi(0.0, N * h2);
    // Top-right corner
    B[info->m][info->n] = phi(M * h1, N * h2);
    break;
  case LOC_GLOBAL:
    // Internal grid points in global domain
    for (li = 2; li < info->m; ++li) {
      gi = info->start[0] + li - 1;
      for (lj = 2; lj < info->n; ++lj) {
        gj = info->start[1] + lj - 1;
        B[li][lj] = F(gi * h1, gj * h2);
      }
    }
    // Bottom and top grid points in global domain
    for (li = 2; li < info->m; ++li) {
      gi = info->start[0] + li - 1;
      B[li][1] = F(gi * h1, 0.0) + (2.0 / h2) * psi(gi * h1, 0.0);
      B[li][info->n] = phi(gi * h1, N * h2);
    }
    // Left and right grid points in global domain
    for (lj = 2; lj < info->n; ++lj) {
      gj = info->start[1] + lj - 1;
      B[1][lj] = phi(0.0, lj * h2);
      B[info->m][lj] = phi(M * h1, gj * h2); 
    }
    // Corner grid points
    B[1][1] = phi(0.0, 0.0);
    B[info->m][1] = phi(M * h1, 0.0);
    B[1][info->n] = phi(0.0, N * h2);
    B[info->m][info->n] = phi(M * h1, N * h2);
    break;
  default:
    fprintf(stderr, "[Rank %d]: Cant fill B: unknown location type!\n", info->rank);
    MPI_Finalize();
    exit(EXIT_FAILURE);
  }
}

// Weight functions for dot product.

static double rho_x(size_t gi, size_t M) {
  return (gi >= 1 && gi <= M - 1) ? 1.0 : 0.5;
}

static double rho_y(size_t gj, size_t N) {
  return (gj >= 1 && gj <= N - 1) ? 1.0 : 0.5;
}

static double rho(size_t gi, size_t gj, size_t M, size_t N) {
  return rho_x(gi, M) * rho_y(gj, N);
}

// Dot product (u*v).
// TODO: MPI_Allreduce
double dot_product(double** u, double** v, double h1, double h2, size_t M, size_t N, ProcInfo_t *info) {
  size_t li, lj, gi, gj;
  double sum = 0.0;
  for (li = 1; li < info->m + 1; ++li) {
    gi = info->start[0] + li - 1; 
    double tmp_sum = 0.0;
    for (lj = 1; lj < info->n + 1; ++lj) {
      gj = info->start[1] + lj - 1;  
      tmp_sum += h2 * rho(gi, gj, M, N) * u[li][lj] * v[li][lj];
    }
    sum += h1 * tmp_sum;
  }
  return sum;
}

// Norm (||u|| = sqrt(u*u)).

double norm(double** u, double h1, double h2, size_t M, size_t N, ProcInfo_t *info) {
  return sqrt(dot_product(u, u, h1, h2, M, N, info));
}

// (aw_~x)_ij

double aw(double** w, size_t li, size_t lj, size_t gi, size_t gj, double h1, double h2) {
  return k(gi * h1 - 0.5 * h1, gj * h2) * left_dx(w, li, lj, h1);
}

// (bw_~y)_ij

double bw(double** w, size_t li, size_t lj, size_t gi, size_t gj, double h1, double h2) {
  return k(gi * h1, gj * h2 - 0.5 * h2) * left_dy(w, li, lj, h2);
}

// Fill Aw (left part of Aw = B).
// r = Aw.
// TODO: MPI and rename LHS
void fill_Aw(double** w, double** r, double h1, double h2, size_t M, size_t N, ProcInfo_t *info) {
  // Local iteration variables for position in local domain grid of proc
  size_t li, lj;
  // Global iteration variables for position on global domain grid
  size_t gi, gj;
  // Fill domains of Aw
  switch(info->proc_loc) {
  // TODO: another proc loc types
  case LOC_GLOBAL:
    // Internal grid points in global domain
    for (li = 2; li < info->m; ++li) {
      gi = info->start[0] + li - 1;
      for (lj = 2; lj < info->n; ++lj) {
        gj = info->start[1] + lj - 1;
        r[li][lj] = -laplace_operator(w, li, lj, gi, gj, h1, h2) + q(gi * h1, gj * h2) * w[li][lj];
      }
    }
    // Bottom and top grid points in global domain
    for (li = 2; li < info->m; ++li) {
      gi = info->start[0] + li - 1;
      r[li][1] = -(2.0 / h2) * bw(w, li, 2, gi, 1, h1, h2) + (q(gi * h1, 0.0) + 2.0 / h1) * w[li][1] - left_delta(w, li, 1, gi, 0, h1, h2);
      r[li][info->n] = w[li][info->n];
    }
    // Left and right grid points in global domain
    for (lj = 1; lj < info->n; ++lj) {
      gj = info->start[1] + lj - 1;
      r[1][lj] = w[1][lj];
      r[info->m][lj] = w[info->m][lj];
    }
    // Corner grid points
    r[1][1] = w[1][1];
    r[info->m][1] = w[info->m][1];
    r[1][info->n] = w[1][info->n];
    r[info->m][info->n] = w[info->m][info->n]; 
    break;
  default:
    fprintf(stderr, "[Rank %d]: Cant fill Aw: unkown location type!\n", info->rank);
    MPI_Finalize();
    exit(EXIT_FAILURE);
  }
} 

// Let num = 2^power, that function returns the power
// else return -1

static int get_power(int num) {
  if (num <= 0)
    return -1;
  int power = 0;
  while ((num & 1) == 0) {
    ++power;
    num = num >> 1; 
  } 
  if ((num >> 1) != 0)
    return -1;
  return power;
}

// Returns px, where px is dims[0] = 2^px

static int split(size_t M, size_t N, int power) {
  double m = (double)M;
  double n = (double)N;
  int px = 0;
  for (int i = 0; i < power; ++i) {
    if (m > n) {
      m /= 2.0;
      ++px;
    } else {
      n /= 2.0;
    }
  } 
  return px;
}

void domain_decomposition(size_t M, size_t N, MPI_Comm *GridComm, ProcInfo_t *info) {
  // The number pf processes
  int ProcNum;
  // ProcNum = 2^(power), power = px + py
  int power, px, py;
  // dims[0] = 2^px, dims[1] = 2^py
  int dims[2];
  // 2D topology dimension
  const int ndims = 2;
  // Topologically grid is not closed
  int periods[2] = {FALSE, FALSE};
  // Init MPI lib
  MPI_Comm_size(MPI_COMM_WORLD, &ProcNum);
  MPI_Comm_rank(MPI_COMM_WORLD, &(info->rank));
  // Check that grid size contains positive numbers
/*if ((M <= 0) || (N <= 0)) {
    if (info->rank == 0)  
      fprintf(stderr, "M and N must be positive!\n");
    MPI_Finalize();
    exit(EXIT_FAILURE);
  } */
  // Check that number of processes is a power of 2 and get power
  if ((power = get_power(ProcNum)) < 0) {
    if (info->rank == 0)
      fprintf(stderr, "The number of procs must be a power of 2!\n"); 
    MPI_Finalize();
    exit(EXIT_FAILURE);
  } 
  // Find such px, py that ProcNum = 2^(px+py)
  px = split(M, N, power);
  py = power - px;
  // Find dims[0] = 2^px and dims[1] = 2^py
  dims[0] = (unsigned int)1 << px; dims[1] = (unsigned int)1 << py;
  // Find local domain size: m = M/(2^px), n = N/(2^py) 
  info->m = (M + 1) >> px; info->n = (N+1) >> py;
  int rx = M + 1 - dims[0] * info->m;
  int ry = N + 1 - dims[1] * info->n;
  // Create the cartesian 2D topology
  MPI_Cart_create(MPI_COMM_WORLD, ndims, dims, periods, TRUE, GridComm);
  MPI_Comm_rank(*GridComm, &(info->rank));
  MPI_Cart_coords(*GridComm, info->rank, ndims, info->coords);
  // Get process start and end points, local domain size
  info->start[0] = MIN(rx, info->coords[0]) * (info->m + 1) + MAX(0, (info->coords[0] - rx)) * info->m;
  info->start[1] = MIN(ry, info->coords[1]) * (info->n + 1) + MAX(0, (info->coords[1] - ry)) * info->n;
  info->end[0] = info->start[0] + info->m + (info->coords[0] < rx ? 1 : 0);
  info->end[1] = info->start[1] + info->n + (info->coords[1] < ry ? 1 : 0);
  info->m = info->end[0] - info->start[0];
  info->n = info->end[1] - info->start[1];
  // Find process location type
  if (info->start[0] != 0 && info->end[0] - 1 != M &&
      info->start[1] != 0 && info->end[1] - 1 != N) {
    info->proc_loc = LOC_INNER;
  } else if (info->start[0] != 0 && info->start[1] == 0 &&
             info->end[0] - 1 != M && info->end[1] - 1 != N) {
    info->proc_loc = LOC_INNER_BOT;
  } else if (info->start[0] != 0 && info->start[1] != 0 &&
             info->end[0] - 1 != M && info->end[1] - 1 == N) {
    info->proc_loc = LOC_INNER_TOP;
  } else if (info->start[0] == 0 && info->start[1] != 0 &&
             info->end[0] - 1 != M && info->end[1] - 1 != N) {
    info->proc_loc = LOC_INNER_LEFT;
  } else if (info->start[0] != 0 && info->start[1] != 0 &&
             info->end[0] - 1 == M && info->end[1] - 1!= N) {
    info->proc_loc = LOC_INNER_RIGHT;
  } else if (info->start[0] == 0 && info->start[1] == 0 &&
             info->end[0] - 1 != M && info->end[1] - 1 != N) {
    info->proc_loc = LOC_CORNER_BOTLEFT;
  } else if (info->start[0] != 0 && info->start[1] == 0 &&
             info->end[0] - 1 == M && info->end[1] - 1 != N) {
    info->proc_loc = LOC_CORNER_BOTRIGHT;
  } else if (info->start[0] == 0 && info->start[1] != 0 &&
             info->end[0] - 1 != M && info->end[1] - 1 == N) {
    info->proc_loc = LOC_CORNER_TOPLEFT;
  } else if (info->start[0] != 0 && info->start[1] != 0 &&
             info->end[0] - 1 == M && info->end[1] - 1 == N) {
    info->proc_loc = LOC_CORNER_TOPRIGHT;
  } else if (info->start[0] == 0 && info->start[1] == 0 &&
             info->end[0] - 1 == M && info->end[1] - 1 != N) {
    info->proc_loc = LOC_CUP;
  } else if (info->start[0] == 0 && info->start[1] != 0 &&
             info->end[0] - 1 == M && info->end[1] - 1 == N) {
    info->proc_loc = LOC_CAP;
  } else if (info->start[0] == 0 && info->start[1] == 0 &&
             info->end[0] - 1 == M && info->end[1] - 1 == N) {
    info->proc_loc = LOC_GLOBAL;
  } else {
    info->proc_loc = LOC_NONE;
  }
  if (info->proc_loc == LOC_NONE) {
    fprintf(stderr, "The process with rank = %d cant find its location type!\n", info->rank);
    MPI_Finalize();
    exit(EXIT_FAILURE);
  } 
  // Get process neighbors
  MPI_Cart_shift(*GridComm, 1, -1, &(info->up), &(info->down));
  MPI_Cart_shift(*GridComm, 0, 1, &(info->left), &(info->right));
#ifdef debug_decomp_print
  printf("******************************************************\n"
         "Rank = %d, coords = (%d, %d).\n"
         "Local domain size = %d x %d.\n"
         "Neighbords: left = %d, right = %d, up = %d, down = %d.\n"
         "Start coords = (%d, %d), end coords = (%d, %d).\n"
         "Process location type = %d.\n",
         info->rank, info->coords[0], info->coords[1], 
         info->m, info->n, 
         info->left, info->right, info->up, info->down,
         info->start[0], info->start[1], info->end[0], info->end[1],
         info->proc_loc); 
#endif
}

// Exchange of boundaries between neighboring processes
// TODO: this funtion

void exchange(MPI_Comm *GridComm, ProcInfo_t *info) {
  
}

void print_matrix(double **w, ProcInfo_t *info) {
  for (size_t lj = info->n; lj >= 1; --lj) {
    printf("rank = %d ", info->rank);
    for (size_t li = 1; li < info->m + 1; ++li) {
      printf("%lf ", w[li][lj]);
    }
    printf("\n");
  }  
  printf("\n");
}

void solve(size_t M, size_t N, MPI_Comm *GridComm, ProcInfo_t *info) {
  size_t li, lj, gi, gj;
  size_t iteration = 0;
  const double eps = 1e-6;
  const double h1 = 4.0 / (double) M;
  const double h2 = 3.0 / (double) N;
  double tau = 0.0;
  //TODO: rename LHS, RHS, Solution
  double **B = (double**) malloc((info->m + 2) * sizeof(double*));
  double **w = (double**) malloc((info->m + 2) * sizeof(double*));
  double **tmp_w = (double**)malloc((info->m + 2) * sizeof(double*));
  double **r = (double**) malloc((info->m + 2) * sizeof(double*));
  double **Ar = (double**) malloc((info->m + 2) * sizeof(double*));
  double **u_arr = (double**) malloc((info->m + 2) * sizeof(double*));
  for (size_t i = 0; i < info->m + 2; ++i) { 
    B[i] = (double*) calloc((info->n + 2), sizeof(double));
    w[i] = (double*) calloc((info->n + 2), sizeof(double));
    tmp_w[i] = (double*) calloc((info->n + 2), sizeof(double));
    r[i] = (double*) calloc((info->n + 2), sizeof(double));
    Ar[i] = (double*) calloc((info->n + 2), sizeof(double));
    u_arr[i] = (double*) calloc((info->n + 2) , sizeof(double));
  }
  for (li = 1; li < info->m + 1; ++li) {
    gi = info->start[0] + li - 1;
    for (lj = 1; lj < info->n + 1; ++lj) {
      gj = info->start[1] + lj - 1; 
      u_arr[li][lj] = u(gi * h1, gj * h2);
    }
  }
  fill_B(B, h1, h2, M, N, info);
#ifdef debug_B_print
  print_matrix(B, info); 
#endif
  while (TRUE) {
#ifdef debug_solve_print
    printf("Iteration: %lu\n", iteration++);
#endif
    fill_Aw(w, r, h1, h2, M, N, info);    
    for (li = 1; li < info->m + 1; ++li) {
      for (lj = 1; lj < info->n + 1; ++lj) {
        r[li][lj] -= B[li][lj];
        tmp_w[li][lj] = w[li][lj];
      }
    }  
    fill_Aw(r, Ar, h1, h2, M, N, info);
    tau = dot_product(Ar, r, h1, h2, M, N, info) / pow(norm(Ar, h1, h2, M, N, info), 2.0);
    for (li = 1; li < info->m + 1; ++li) {
      for (lj = 1; lj < info->n + 1; ++lj) {
        w[li][lj] = w[li][lj] - tau * r[li][lj];
      } 
    }
    for (li = 1; li < info->m + 1; ++li) {
      for (lj = 1; lj < info->n + 1; ++lj) {
        tmp_w[li][lj] = w[li][lj] - tmp_w[li][lj];
      }
    }
  double diff = norm(tmp_w, h1, h2, M, N, info);
#ifdef debug_solve_print
  printf("Diff: %lf\n", diff);
#endif
  if (diff < eps)
    break;
  }
  printf("%lu,%lu\n", M, N);
  for (li = 1; li < info->m + 1; ++li) {
    for (lj = 1; lj < info->n + 1; ++lj) {
      printf("%lf,%lf\n", u_arr[li][lj], w[li][lj]);
    }
  }
  for (size_t i = 0; i < info->m + 2; ++i) {
    free(B[i]); 
    free(w[i]);
    free(tmp_w[i]);
    free(r[i]);
    free(Ar[i]);
    free(u_arr[i]);
  }
  free(B);
  free(w);
  free(tmp_w);
  free(r);
  free(Ar);
  free(u_arr);
}

int main(int argc, char **argv) {
  if (argc < 3) {
    fprintf(stderr, "Not enought arguments!\n");
    return 1;
  }
  // Grid:
  // Ox : 0..M
  // Oy : 0..N
  // Where M, N are numbers of internal points
  const size_t M = atoi(argv[1]);
  const size_t N = atoi(argv[2]);
  // MPI handler of grid
  MPI_Comm GridComm;
  // Current process information
  ProcInfo_t info;
  // MPI lib init
  MPI_Init(&argc, &argv); 
  // Performe domain decomposition 
  domain_decomposition(M, N, &GridComm, &info);
  // Solve the task
  solve(M, N, &GridComm, &info);
  MPI_Finalize();
  return 0;
}
