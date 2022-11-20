#include <stdlib.h>
#include <stdio.h>
#include <math.h>

// There are variables from my variant 2.
// u(x, y) = u_2(x, y) = sqrt(4 + x*y) 
// k(x, y) = k_3(x, y) = 4 + x + y
// q(x, y) = q_2(x, y) = x + y

double u2(double x, double y) {
  return 4.0 + x*y;
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
  return (k(x, y)*(x*x + y*y) - 2.0*q(x, y)*u2(x, y)) / (4.0*sqrt(u2(x, y)*u2(x,
y)*u2(x,y))) + q(x, y)*u(x, y); 
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
  return u(x, y) - (x*k(x, y)) / (2.0*u(x, y));
}

// Grid coefficients.

double a(size_t i, size_t j, double h1, double h2) {
  return k(i*h1 - 0.5*h1, j*h2);
}

double b(size_t i, size_t j, double h1, double h2) {
  return k(i*h1, j*h2 - 0.5*h2);
}

// Right and left difference partial derivatives.

double right_dx(double **v, size_t i, size_t j, double h1) {
  return (v[i + 1][j] - v[i][j]) / h1;
} 

double left_dx(double **v, size_t i, size_t j, double h1) {
  return (v[i][j] - v[i - 1][j]) / h1;
}

double right_dy(double **v, size_t i, size_t j, double h2) {
  return (v[i][j + 1] - v[i][j]) / h2;
}

double left_dy(double **v, size_t i, size_t j, double h2) {
  return (v[i][j] - v[i][j - 1]) / h2;
}

// Laplace operator.

double left_delta(double **w, size_t i, size_t j, double h1, double h2) {
  return (k(i*h1 + 0.5*h1, j*h2)*right_dx(w, i, j, h1) - a(i, j, h1,
h2)*left_dx(w, i, j, h1));
}

double right_delta(double **w, size_t i, size_t j, double h1, double h2) {
  return (k(i*h1, j*h2 + 0.5*h2)*right_dy(w, i, j, h2) - b(i, j,
h1, h2)*left_dy(w, i, j, h2));
}

double laplace_operator(double **w, size_t i, size_t j, double h1, double h2) {
  return (1.0 / h1)*left_delta(w, i, j, h1, h2) + (1.0 / h2)*right_delta(w, i, j, h1, h2);
}

// Fill B (right part of Aw = B).

void fill_B(double **B, size_t M, size_t N, double h1, double h2) {
  // Internal grid points. 
  for (size_t i = 1; i < M; ++i) {
    for (size_t j = 1; j < N; ++j) {
      B[i][j] = F(i*h1, j*h2); 
    }
  }
  // Bottom and top grid points.
  for (size_t i = 1; i < M; ++i) {
    B[i][0] = F(i*h1, 0) + (2.0 / h2)*psi(i*h1, 0);
    B[i][N] = F(i*h1, (N - 1) * h2) + (1.0 / h2*h2)*b(i, N, h1, h2)*phi(i*h1, N*h2); 
  } 
  // Left and right grid points.
  for (size_t j = 1; j < N; ++j) {
    B[0][j] = F(1, j*h2) + (1.0 / h1*h1)*a(1, j, h1, h2)*phi(0, j*h2);
    B[M][j] = F((M - 1)*h1, j*h2) + (1.0 / h1*h1)*a(M, j, h1, h2)*phi(M*h1, j*h2); 
  }
  // Corner grid points.
  B[0][0] = phi(0, 0); 
  B[M][0] = phi(M*h1, 0); 
  B[0][N] = phi(0, N*h2);
  B[M][N] = phi(M*h1, N*h2); 
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

double dot_product(double **u, double **v, double h1, double h2, size_t M,
size_t N) {
  double sum = 0.0;
  for (size_t i = 0; i <= M; ++i) {
    double tmp_sum = 0.0;
    for (size_t j = 0; j <= N; ++j) {
      tmp_sum += h2*rho(i, j, M, N)*u[i][j]*v[i][j];
    } 
    sum += h1*tmp_sum;
  }
  return sum;
}

// Norm (||u|| = sqrt(u*u)).

double norm(double **u, double h1, double h2, size_t M, size_t N) {
  return sqrt(dot_product(u, u, h1, h2, M, N));
}

// Fill Aw (left part of Aw = B).
// r = Aw.

void fill_Aw(double **w, double **r, double h1, double h2, size_t M, size_t N) { 
  // Internal grid points.
  for (size_t i = 1; i < M; ++i) {
    for (size_t j = 1; j < N; ++j) {
      r[i][j] = -laplace_operator(w, i, j, h1, h2) + q(i*h1, j*h2)*w[i][j];
    }
  } 
  // Bottom and top grid points.
  for (size_t i = 1; i < M; ++i) {
    r[i][0] = -(2.0 / h2)*right_delta(w, i, 1, h1, h2) + (q(i*h1, 0) + 2.0 / h1)*w[i][0] - left_delta(w, i, 0, h1, h2);
    r[i][N] = -left_delta(w, i, N - 1, h1, h2) + (1.0 / h2)*(right_delta(w, i, N - 1, h1, h2) + (1.0 / h2)*w[i][N - 1]) + q(i*h1, (N - 1)*h2)*w[i][N-1];
  }
  // Left and right grid points.
  for (size_t j = 1; j < N; ++j) {
    r[0][j] = -(1.0 / h1)*left_delta(w, 2, j, h1, h2) + (1.0 / h1*h1)*left_delta(w, 1, j, h1, h2) - right_delta(w, 1, j, h1, h2) + q(1*h1, j*h2)*w[1][j];
    r[M][j] = (1.0 / h1)*(left_delta(w, M - 1, j, h1, h2) + (1.0 / h1)*a(M, j, h1, h2)*w[M - 1][j]) - right_delta(w, M - 1, j, h1, h2) + q((M - 1)*h1, j*h2)*w[M - 1][j];
  }  
  // Corner grid points.
  r[0][0] = phi(0, 0);
  r[M][0] = phi(M*h1, 0);
  r[0][N] = phi(0, N*h2);
  r[M][N] = phi(M*h1, N*h2);
}

int main(int argc, char **argv) {
  return 0;
}
