#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <mpi.h>

#define TRUE (1)
#define FALSE (0)

// Structure which contains process location info and neighbors
struct ProcInfo {
  // Process rank
  int rank;
  // Process topological coords
  int coords[2];
  // Ranks of the neighbors of the process
  int left, right, up, down;
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

double a(size_t i, size_t j, double h1, double h2) {
  return k(i * h1 - 0.5 * h1, j * h2);
}

double b(size_t i, size_t j, double h1, double h2) {
  return k(i * h1, j * h2 - 0.5 * h2);
}

// Right and left difference partial derivatives.

double right_dx(double** v, size_t i, size_t j, double h1) {
  return (v[i + 1][j] - v[i][j]) / h1;
}

double left_dx(double** v, size_t i, size_t j, double h1) {
  return (v[i][j] - v[i - 1][j]) / h1;
}

double right_dy(double** v, size_t i, size_t j, double h2) {
  return (v[i][j + 1] - v[i][j]) / h2;
}

double left_dy(double** v, size_t i, size_t j, double h2) {
  return (v[i][j] - v[i][j - 1]) / h2;
}

// Laplace operator.

double left_delta(double** w, size_t i, size_t j, double h1, double h2) {
  return (1.0 / h1) * (k(i * h1 + 0.5 * h1, j * h2) * right_dx(w, i, j, h1) - a(i, j, h1,
      h2) * left_dx(w, i, j, h1));
}

double right_delta(double** w, size_t i, size_t j, double h1, double h2) {
  return (1.0 / h2) * (k(i * h1, j * h2 + 0.5 * h2) * right_dy(w, i, j, h2) - b(i, j,
      h1, h2) * left_dy(w, i, j, h2));
}

double laplace_operator(double** w, size_t i, size_t j, double h1, double h2) {
  return left_delta(w, i, j, h1, h2) + right_delta(w, i, j, h1, h2);
}

// Fill B (right part of Aw = B).

void fill_B(double** B, size_t M, size_t N, double h1, double h2) {
  // Internal grid points. 
  for (size_t i = 1; i < M; ++i) {
      for (size_t j = 1; j < N; ++j) {
          B[i][j] = F(i * h1, j * h2);
      }
  }
  // Bottom and top grid points.
  for (size_t i = 1; i < M; ++i) {
      B[i][0] = F(i * h1, 0) + (2.0 / h2) * psi(i * h1, 0);
      B[i][N] = phi(i * h1, N * h2);
  }
  // Left and right grid points.
  for (size_t j = 1; j < N; ++j) {
      B[0][j] = phi(0, j * h2);
      B[M][j] = phi(M * h1, j * h2);
  }
  // Corner grid points.
  B[0][0] = phi(0, 0);
  B[M][0] = phi(M * h1, 0);
  B[0][N] = phi(0, N * h2);
  B[M][N] = phi(M * h1, N * h2);
}

// Weight functions for dot product.

double rho_x(size_t i, size_t M) {
  return (i >= 1 && i <= M - 1) ? 1.0 : 0.5;
}

double rho_y(size_t j, size_t N) {
  return (j >= 1 && j <= N - 1) ? 1.0 : 0.5;
}

double rho(size_t i, size_t j, size_t M, size_t N) {
  return rho_x(i, M) * rho_y(j, N);
}

// Dot product (u*v).

double dot_product(double** u, double** v, double h1, double h2, size_t M, size_t N) {
  double sum = 0.0;
  for (size_t i = 0; i <= M; ++i) {
      double tmp_sum = 0.0;
      for (size_t j = 0; j <= N; ++j) {
          tmp_sum += h2 * rho(i, j, M, N) * u[i][j] * v[i][j];
      }
      sum += h1 * tmp_sum;
  }
  return sum;
}

// Norm (||u|| = sqrt(u*u)).

double norm(double** u, double h1, double h2, size_t M, size_t N) {
  return sqrt(dot_product(u, u, h1, h2, M, N));
}

// (aw_~x)_ij

double aw(double** w, size_t i, size_t j, double h1, double h2) {
  return k(i * h1 - 0.5 * h1, j * h2) * left_dx(w, i, j, h1);
}

// (bw_~y)_ij

double bw(double** w, size_t i, size_t j, double h1, double h2) {
  return k(i * h1, j * h2 - 0.5 * h2) * left_dy(w, i, j, h2);
}

// Fill Aw (left part of Aw = B).
// r = Aw.

void fill_Aw(double** w, double** r, double h1, double h2, size_t M, size_t N) {
  // Internal grid points.
  for (size_t i = 1; i < M; ++i) {
      for (size_t j = 1; j < N; ++j) {
          r[i][j] = -laplace_operator(w, i, j, h1, h2) + q(i * h1, j * h2) * w[i][j];
      }
  }
  // Bottom and top grid points.
  for (size_t i = 1; i < M; ++i) {
      r[i][0] = -(2.0 / h2) * bw(w, i, 1, h1, h2) + (q(i * h1, 0) + 2.0 / h1) * w[i][0] - left_delta(w, i, 0, h1, h2);
      r[i][N] = w[i][N];
  }
  // Left and right grid points.
  for (size_t j = 1; j < N; ++j) {
      r[0][j] = w[0][j];
      r[M][j] = w[M][j];
  }
  // Corner grid points.
  r[0][0] = w[0][0];
  r[M][0] = w[M][0];
  r[0][N] = w[0][N];
  r[M][N] = w[M][N];
}

void domain_decomposition(MPI_Comm *GridComm, ProcInfo_t *info) {
  int ProcNum;
  int dims[2], periods[2];
  const int ndims = 2;
  // Temp dims = 2
  dims[0] = dims[1] = 2;
  // Topologically grid is not closed
  periods[0] = periods[1] = FALSE;
  // Init MPI lib
  MPI_Comm_size(MPI_COMM_WORLD, &ProcNum);
  MPI_Comm_rank(MPI_COMM_WORLD, &(info->rank));
  // Create the cartesian 2D topology
  MPI_Cart_create(MPI_COMM_WORLD, ndims, dims, periods, TRUE, GridComm);
  MPI_Comm_rank(*GridComm, &(info->rank));
  MPI_Cart_coords(*GridComm, info->rank, ndims, info->coords);
  MPI_Cart_shift(*GridComm, 0, 1, &(info->up), &(info->down));
  MPI_Cart_shift(*GridComm, 1, 1, &(info->left), &(info->right));
  printf("Rank = %d, coords = (%d, %d).\n"
         "Neighbords: left = %d, right = %d, up = %d, down = %d.\n",
         info->rank, info->coords[0], info->coords[1], info->left, info->right, info->up, info->down); 
}

int main(int argc, char **argv) {
  if (argc < 3) {
    fprintf(stderr, "Not enought arguments!\n");
    return 1;
  }
  const size_t M = atoi(argv[1]);
  const size_t N = atoi(argv[2]);
  const double eps = 1e-6;
  const double h1 = 4.0 / (double)M;
  const double h2 = 3.0 / (double)N;
  double tau = 0.0;
  MPI_Comm GridComm;
  ProcInfo_t info;
  MPI_Init(&argc, &argv); 
  domain_decomposition(&GridComm, &info);
/*  double** w = (double**)malloc((M + 1) * sizeof(double*));
  for (size_t i = 0; i <= M; ++i)
      w[i] = (double*)malloc((N + 1) * sizeof(double));
  double** tmp_w = (double**)malloc((M + 1) * sizeof(double*));
  for (size_t i = 0; i <= M; ++i)
      tmp_w[i] = (double*)malloc((N + 1) * sizeof(double));
  double** r = (double**)malloc((M + 1) * sizeof(double*));
  for (size_t i = 0; i <= M; ++i)
    r[i] = (double*)malloc((N + 1) * sizeof(double));
  double** Ar = (double**)malloc((M + 1) * sizeof(double*));
  for (size_t i = 0; i <= M; ++i)
      Ar[i] = (double*)malloc((N + 1) * sizeof(double));
  double** B = (double**)malloc((M + 1) * sizeof(double*));
  for (size_t i = 0; i <= M; ++i)
      B[i] = (double*)malloc((N + 1) * sizeof(double));
  double** u_arr = (double**)malloc((M + 1) * sizeof(double*));
  for (size_t i = 0; i <= M; ++i)
      u_arr[i] = (double*)malloc((N + 1) * sizeof(double));
  // w^0 = 0
  for (size_t i = 0; i <= M; ++i)
      for (size_t j = 0; j <= N; ++j)
          w[i][j] = 0.0;
  // Fill B
  fill_B(B, M, N, h1, h2);
  // Iterations
  while (1) {
      // r^(k) = Aw^(k)
      fill_Aw(w, r, h1, h2, M, N);
      // r^(k) = Aw^(k) - B
      for (size_t i = 0; i <= M; ++i)
          for (size_t j = 0; j <= N; ++j) {
              r[i][j] -= B[i][j];
              tmp_w[i][j] = w[i][j];
          }
      // Ar^(k)
      fill_Aw(r, Ar, h1, h2, M, N);
      // tau^(k+1) = Ar^(k)*r^(k) / ||Ar^(k)||^2
      tau = dot_product(Ar, r, h1, h2, M, N) / pow(norm(Ar, h1, h2, M, N), 2);
      //w^(k+1) = w^(k) - tau^(k+1)r^(k)
      for (size_t i = 0; i <= M; ++i)
          for (size_t j = 0; j <= N; ++j)
              w[i][j] = w[i][j] - tau * r[i][j];
      //tmp_w = w^(k+1) - w^(k)
      for (size_t i = 0; i <= M; ++i)
          for (size_t j = 0; j <= N; ++j)
              tmp_w[i][j] = w[i][j] - tmp_w[i][j];
      double diff = norm(tmp_w, h1, h2, M, N);
      //    printf("%lf\n", diff);
      if (diff < eps)
          break;
  }
  for (size_t i = 0; i <= M; ++i)
      for (size_t j = 0; j <= N; ++j)
          u_arr[i][j] = u(i * h1, j * h2);
  printf("%lu,%lu\n", M, N);
  for (size_t i = 0; i <= M; ++i)
      for (size_t j = 0; j <= N; ++j)
          printf("%lf,%lf\n", u_arr[i][j], w[i][j]); */
  MPI_Finalize();
  return 0;
}
