#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <mpi.h>

#define EXACT_SOLUTION (4.0*M_PI/3.0)
#define INTEGRATION_CONST (8)
#define PORTION_POINTS (64 * 100)

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
    double reduced_integral_summ = 0.0;
    double I = 0.0;
    int start_idx = 0; 
    MPI_Status Status;
    int flag = 1;

    MPI_Init(&argc, &argv);

    int rank, size;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int portion_points = PORTION_POINTS / (size - 1);

    double *points_y = malloc(portion_points * sizeof(double));
    double *points_z = malloc(portion_points * sizeof(double));

    size_t points = 0;

    if (rank == 0)
    {
        int d = 0;
        double start = MPI_Wtime();
        srand(start);
//        MPI_Request request;        
        while (1)
        {
            points += PORTION_POINTS; 
            for (int rk = 1; rk < size; ++rk)
            {
                for (size_t i = 0; i < portion_points; ++i)
                {
                    points_y[i] = rand01();
                    points_z[i] = rand01(); 
                }
                
                MPI_Send(points_y, 
                         portion_points, 
                         MPI_DOUBLE, 
                         rk, 
                         0,
                         MPI_COMM_WORLD);
                    //     &request); 
                MPI_Send(points_z, 
                         portion_points, 
                         MPI_DOUBLE, 
                         rk, 
                         0,
                         MPI_COMM_WORLD);
                     //    &request);                
            } 
           
          //  MPI_Wait(&request, MPI_STATUS_IGNORE);
 
            MPI_Reduce(&integral_summ, 
                       &reduced_integral_summ, 
                       1, 
                       MPI_DOUBLE, 
                       MPI_SUM, 
                       0,
                       MPI_COMM_WORLD);            

            I = INTEGRATION_CONST * (reduced_integral_summ / points);
            delta = fabs(I - EXACT_SOLUTION);

            if (delta < eps)
            {
                flag = 0;
                for (int rk = 1; rk < size; ++rk)
                    MPI_Send(&flag,
                              1,
                              MPI_INT,
                              rk,
                              0,
                              MPI_COMM_WORLD);
                          //    &request);    
              //  MPI_Wait(&request, MPI_STATUS_IGNORE); 
                break;
            }   
            else
            {
                for (int rk = 1; rk < size; ++rk)
                    MPI_Send(&flag,
                              1,
                              MPI_INT,
                              rk,
                              0,
                              MPI_COMM_WORLD);
                           //   &request);
               // MPI_Wait(&request, MPI_STATUS_IGNORE);
            }
        } 

        double end = MPI_Wtime(); 
       
        printf("I = %lf\ndelta = %lf\npoints = %ld\ntime = %lf\n", I, delta,
points, end - start);

    }
    else
    {
        while (flag)
        {
            MPI_Recv(points_y, 
                     portion_points, 
                     MPI_DOUBLE, 
                     0, 
                     0, 
                     MPI_COMM_WORLD,
                     &Status); 
            MPI_Recv(points_z, 
                     portion_points, 
                     MPI_DOUBLE, 
                     0, 
                     0,
                     MPI_COMM_WORLD,
                     &Status);

            for (size_t i = 0; i < portion_points; ++i)
                integral_summ += F(points_y[i], points_z[i]);
            
            MPI_Reduce(&integral_summ,
                       &reduced_integral_summ,
                       1,
                       MPI_DOUBLE,
                       MPI_SUM,
                       0,
                       MPI_COMM_WORLD);

            MPI_Recv(&flag,
                     1,
                     MPI_INT,
                     0,
                     0,
                     MPI_COMM_WORLD,
                     &Status);
        } 
    }


    free(points_y);
    free(points_z);

    MPI_Finalize(); 
    
    return 0;
}
