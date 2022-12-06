#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#define M (80)
#define N (80)
const double h1 = (4.0 / (double) M);
const double h2 = (3.0 / (double) N);
#pragma dvm array distribute [block][block]
double B[M + 1][N + 1];
#pragma dvm array align([i][j] with B[i][j])
double tmp_w[M + 1][N + 1];
#pragma dvm array align([i][j] with B[i][j]), shadow[1:1][1:1]
double r[M + 1][N + 1];
#pragma dvm array align([i][j] with B[i][j])
double Ar[M + 1][N + 1];
#pragma dvm array align([i][j] with B[i][j]), shadow[1:1][1:1]
double w[M + 1][N + 1];
#pragma dvm array align([i][j] with B[i][j])
double u_arr[M + 1][N + 1];

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

// Fill B (right part of Aw = B).

void fill_B(void) {
    //#pragma dvm region
    //{
    // Internal grid points.
    #pragma dvm parallel([i][j] on B[i][j])
    for (size_t i = 1; i < M; ++i) {
        for (size_t j = 1; j < N; ++j) {
            B[i][j] = F(i * h1, j * h2);
        }
    }
    // Bottom and top grid points.
    #pragma dvm parallel([i] on B[i][])
    for (size_t i = 1; i < M; ++i) {
        B[i][0] = F(i * h1, 0) + (2.0 / h2) * psi(i * h1, 0);
        B[i][N] = phi(i * h1, N * h2);
    }
    // Left and right grid points.
    #pragma dvm parallel([j] on B[][j])
    for (size_t j = 0; j <= N; ++j) {
        B[0][j] = phi(0, j * h2);
        B[M][j] = phi(M * h1, j * h2);
    }
    //}
    // Corner grid points.
    /*
    B[0][0] = phi(0, 0);
    B[M][0] = phi(M * h1, 0);
    B[0][N] = phi(0, N * h2);
    B[M][N] = phi(M * h1, N * h2);
    */
}

// Weight functions for dot product.

double rho_x(size_t i) {
    return (i >= 1 && i <= M - 1) ? 1.0 : 0.5;
}

double rho_y(size_t j) {
    return (j >= 1 && j <= N - 1) ? 1.0 : 0.5;
}

double rho(size_t i, size_t j) {
    return rho_x(i) * rho_y(j);
}

// Dot product (u*v).

double dot_product_Ar_r(void) {
    double s = 0.0;
    //#pragma dvm actual(s)
    //#pragma region
    //{
    #pragma dvm parallel([i][j] on Ar[i][j]) reduction(sum(s))
    for (size_t i = 0; i <= M; ++i) {
        for (size_t j = 0; j <= N; ++j) {
            s += h1 * h2 * rho(i, j) * Ar[i][j] * r[i][j];
        }
    }
    //}
    //#pragma dvm actual(s)
    return s;
}

double dot_product_Ar_Ar(void) {
    double s = 0.0;
    //#pragma dvm actual(s)
    //#pragma region
    //{
    #pragma dvm parallel([i][j] on Ar[i][j]) reduction(sum(s))
    for (size_t i = 0; i <= M; ++i) {
        for (size_t j = 0; j <= N; ++j) {
            s += h1 * h2 * rho(i, j) * Ar[i][j] * Ar[i][j];
        }
    }
    //}
    //#pragma dvm actual(s)
    return s;
}

double dot_product_tmp_w_tmp_w(void) {
    double s = 0.0;
    //#pragma dvm actual(s)
    //#pragma region
    //{
    #pragma dvm parallel([i][j] on tmp_w[i][j]) reduction(sum(s))
    for (size_t i = 0; i <= M; ++i) {
        for (size_t j = 0; j <= N; ++j) {
            s += h1 * h2 * rho(i, j) * tmp_w[i][j] * tmp_w[i][j];
        }
    }
    //}
    //#pragma dvm actual(s)
    return s;
}

double norm_Ar(void) {
    return sqrt(dot_product_Ar_Ar());
}

double norm_tmp_w(void) {
    return sqrt(dot_product_tmp_w_tmp_w());
}

// Laplace operator.

double laplace_operator_w(size_t i, size_t j) {
    return (1.0 / h1) * (k(i * h1 + 0.5 * h1, j * h2) * ((w[i + 1][j] - w[i][j]) / h1) - k(i * h1 - 0.5 * h1, j * h2) * ((w[i][j] - w[i - 1][j]) / h1)) + (1.0 / h2) * (k(i * h1, j * h2 + 0.5 * h2) * ((w[i][j + 1] - w[i][j]) / h2) - k(i * h1, j * h2 - 0.5 * h2) * (w[i][j] - w[i][j - 1]) / h2);
}

double laplace_operator_r(size_t i, size_t j) {
    return (1.0 / h1) * (k(i * h1 + 0.5 * h1, j * h2) * ((r[i + 1][j] - r[i][j]) / h1) - k(i * h1 - 0.5 * h1, j * h2) * ((r[i][j] - r[i - 1][j]) / h1)) + (1.0 / h2) * (k(i * h1, j * h2 + 0.5 * h2) * ((r[i][j + 1] - r[i][j]) / h2) - k(i * h1, j * h2 - 0.5 * h2) * (r[i][j] - r[i][j - 1]) / h2);
}

// Fill Aw (left part of Aw = B).

// fill_Aw(w, r);
void fill_w2r(void) {
    //#pragma region
    //{
    // Internal grid points.
    #pragma dvm parallel([i][j] on r[i][j]) shadow_renew(w)
    for (size_t i = 1; i < M; ++i) {
        for (size_t j = 1; j < N; ++j) {
            r[i][j] = -laplace_operator_w(i, j) + q(i * h1, j * h2) * w[i][j];
        }
    }
    // Bottom and top grid points.
    #pragma dvm parallel([i] on r[i][]) shadow_renew(w)
    for (size_t i = 1; i < M; ++i) {
        r[i][0] = -(2.0 / h2) * (k(i * h1, h2 - 0.5 * h2) * ((w[i][1] - w[i][0]) / h2)) + (q(i * h1, 0) + 2.0 / h1) * w[i][0] - ((1.0 / h1) * (k(i * h1 + 0.5 * h1, 0.0) * ((w[i + 1][0] - w[i][0]) / h1) - k(i * h1 - 0.5 * h1, 0.0) * ((w[i][0] - w[i - 1][0]) / h1)));
        r[i][N] = w[i][N];
    }
    // Left and right grid points.
    #pragma dvm parallel([j] on r[][j])
    for (size_t j = 0; j <= N; ++j) {
        r[0][j] = w[0][j];
        r[M][j] = w[M][j];
    }
    //}
    // Corner grid points.
    /*
    r[0][0] = w[0][0];
    r[M][0] = w[M][0];
    r[0][N] = w[0][N];
    r[M][N] = w[M][N];
    */
}

// fill_Aw(r, Ar);
void fill_r2Ar(void) {
    //#pragma region
    //{
    // Internal grid points.
    #pragma dvm parallel([i][j] on Ar[i][j]) shadow_renew(r)
    for (size_t i = 1; i < M; ++i) {
        for (size_t j = 1; j < N; ++j) {
            Ar[i][j] = -laplace_operator_r(i, j) + q(i * h1, j * h2) * r[i][j];
        }
    }
    // Bottom and top grid points.
    #pragma dvm parallel([i] on Ar[i][]) shadow_renew(r)
    for (size_t i = 1; i < M; ++i) {
        Ar[i][0] = -(2.0 / h2) * (k(i * h1, h2 - 0.5 * h2) * ((r[i][1] - r[i][0]) / h2)) + (q(i * h1, 0) + 2.0 / h1) * r[i][0] - ((1.0 / h1) * (k(i * h1 + 0.5 * h1, 0.0) * ((r[i + 1][0] - r[i][0]) / h1) - k(i * h1 - 0.5 * h1, 0.0) * ((r[i][0] - r[i - 1][0]) / h1)));
        Ar[i][N] = r[i][N];
    }
    // Left and right grid points.
    #pragma dvm parallel([j] on Ar[][j])
    for (size_t j = 0; j <= N; ++j) {
        Ar[0][j] = r[0][j];
        Ar[M][j] = r[M][j];
    }
    //}
    // Corner grid points.
    /*
    Ar[0][0] = r[0][0];
    Ar[M][0] = r[M][0];
    Ar[0][N] = r[0][N];
    Ar[M][N] = r[M][N];
    */
}

int main(int argc, char** argv) {
    const double eps = 1e-6;
    double tau = 0.0;
    double diff = 0.0;
    //#pragma region
    //{
    // w^0 = 2.0
    #pragma dvm parallel([i][j] on w[i][j])
    for (size_t i = 0; i <= M; ++i)
        for (size_t j = 0; j <= N; ++j)
            w[i][j] = 2.0;
    //}
    // Fill B
    fill_B();
    // Iterations
    while (1) {
        // r^(k) = Aw^(k)
        fill_w2r();
        //#pragma region
        //{
        // r^(k) = Aw^(k) - B
        #pragma dvm parallel([i][j] on r[i][j])
        for (size_t i = 0; i <= M; ++i) {
            for (size_t j = 0; j <= N; ++j) {
                r[i][j] = r[i][j] -  B[i][j];
                tmp_w[i][j] = w[i][j];
            }
        }
        // Ar^(k)
        fill_r2Ar();
        // tau^(k+1) = Ar^(k)*r^(k) / ||Ar^(k)||^2
        tau = dot_product_Ar_r() / pow(norm_Ar(), 2);
        //#pragma region
        //{
        //w^(k+1) = w^(k) - tau^(k+1)r^(k)
        #pragma dvm parallel([i][j] on w[i][j])
        for (size_t i = 0; i <= M; ++i)
            for (size_t j = 0; j <= N; ++j)
                w[i][j] = w[i][j] - tau * r[i][j];
        //tmp_w = w^(k+1) - w^(k)
        #pragma dvm parallel([i][j] on tmp_w[i][j])
        for (size_t i = 0; i <= M; ++i)
            for (size_t j = 0; j <= N; ++j)
                tmp_w[i][j] = w[i][j] - tmp_w[i][j];
        //}
        diff = norm_tmp_w();
        if (diff < eps)
            break;
    }
    #pragma dvm parallel([i][j] on u_arr[i][j])
    for (size_t i = 0; i <= M; ++i)
        for (size_t j = 0; j <= N; ++j)
            u_arr[i][j] = u(i * h1, j * h2);
    printf("%u,%u\n", M, N);
    //#pragma dvm actual(w)
    //#pragma dvm parallel([i][j] on w[i][j])
//TODO: print results and time
//    for (size_t i = 0; i <= M; ++i)
//        for (size_t j = 0; j <= N; ++j)
//            printf("%lf,%lf\n", u_arr[i][j], w[i][j]); 
    return 0;
}
