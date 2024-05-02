#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <mpi.h>
using namespace std;

vector<int> fermat_factorization(int n)
{
    vector<int> factors;
    if (n % 2 == 0)
    {
        if (n / 2 != 1)
        {
            factors.push_back(2);
            vector<int> temp = fermat_factorization(n / 2);
            factors.insert(factors.end(), temp.begin(), temp.end());
        }
        else
            factors.push_back(2);
        return factors;
    }

    int a = ceil(sqrt(n));
    int b2 = a * a - n;
    while (sqrt(b2) != static_cast<int>(sqrt(b2)))
    {
        a++;
        b2 = a * a - n;
    }

    int factor1 = a + static_cast<int>(sqrt(b2));
    int factor2 = a - static_cast<int>(sqrt(b2));

    if (factor1 == 1)
        factors.push_back(factor2);
    else if (factor2 == 1)
        factors.push_back(factor1);
    else
    {
        vector<int> temp1 = fermat_factorization(factor1);
        vector<int> temp2 = fermat_factorization(factor2);
        factors.insert(factors.end(), temp1.begin(), temp1.end());
        factors.insert(factors.end(), temp2.begin(), temp2.end());
    }

    return factors;
}

vector<int> read_input(const string& FILE)
{
    vector<int> random_numbers;
    ifstream file(FILE);
    if (file.is_open())
    {
        int number;
        while (file >> number)
            random_numbers.push_back(number);
        file.close();
    }
    return random_numbers;
}

int main(int argc, char** argv)
{
    const string FILE_INPUT = "C:\\Users\\unden\\Desktop\\Labs\\Factoriz_Fermat_MPI\\Factoriz_Fermat_MPI\\random_numbers1.txt";

    int ProcNum, ProcRank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &ProcNum);
    MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);

    vector<int> random_numbers;

    if (ProcRank == 0)
        random_numbers = read_input(FILE_INPUT);

    double start_time = MPI_Wtime();

    int num_numbers = random_numbers.size();
    MPI_Bcast(&num_numbers, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (ProcRank != 0)
        random_numbers.resize(num_numbers);

    MPI_Bcast(random_numbers.data(), num_numbers, MPI_INT, 0, MPI_COMM_WORLD);

    int num_per_proc = num_numbers / ProcNum;
    int start = ProcRank * num_per_proc;
    int end = start + num_per_proc;

    if (ProcRank == ProcNum - 1)
        end = num_numbers;

    vector<int> results;
    for (int i = start; i < end; i++)
    {
        int number = random_numbers[i];
        vector<int> factors = fermat_factorization(number);
        results.push_back(ProcRank);
        results.push_back(number);
        results.insert(results.end(), factors.begin(), factors.end());
        results.push_back(-1);
    }

    vector<int> results_counts(ProcNum, 0);
    int local_count = results.size();
    MPI_Gather(&local_count, 1, MPI_INT, results_counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    vector<int> displs(ProcNum, 0);
    displs[0] = 0;
    for (int i = 1; i < ProcNum; i++)
        displs[i] = displs[i - 1] + results_counts[i - 1];

    vector<int> all_results(displs[ProcNum - 1] + results_counts[ProcNum - 1], 0);

    MPI_Gatherv(results.data(), results.size(), MPI_INT,
        all_results.data(), results_counts.data(), displs.data(), MPI_INT,
        0, MPI_COMM_WORLD);

    if (ProcRank == 0)
    {
        int i = 0;
        while (i < all_results.size())
        {
            cout << "Process " << all_results[i] << ": " << all_results[i + 1] << ": ";
            i += 2;
            while (all_results[i] != -1)
            {
                cout << all_results[i] << " ";
                i++;
            }
            cout << "\n";
            i++;
        }
        double end_time = MPI_Wtime();
        double run_time = end_time - start_time;
        cout << "\nTotal run time: " << run_time << " seconds.\n";
    }

    MPI_Finalize();
    return 0;
}