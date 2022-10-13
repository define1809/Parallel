#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include <time.h>

#define EXACT_SOLUTION (4.0*M_PI/3.0)

#define DX (2)

double f(double y, double z)
{
    return sqrt(y*y + z*z);
}

double F(double y, double z)
{
    if (y*y + z*z <= 1)
        return f(y, z);
    return 0;
}

double rand01()
{
    return (double) rand() / (RAND_MAX);
}

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        fprintf(stderr, "eps expected\n");
        return -1;
    } 

    double eps = atof(argv[1]);
    double delta = eps + 1.0; 
    double integral_summ = 0.0;
    double I = 0.0;

    size_t n = 0;
    size_t i = 0;

    srand(time(NULL));

    while (delta > eps)
    {
        n += 10;

        for (; i < n; ++i)
        {
            double y = rand01();
            double z = rand01();

            integral_summ += F(y, z);
        }

        I = 4 * DX * (integral_summ / n);
        delta = fabs(I - EXACT_SOLUTION);
    }  

    printf("I = %lf\ndelta = %lf\nn = %ld\n", I, delta, n);

    return 0; 
}
