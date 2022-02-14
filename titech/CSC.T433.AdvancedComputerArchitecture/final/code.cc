#include <iostream>

int f1() {
  const int n = 1;
  int sum = 0;
  int i, j;
  for (i = 0; i <= n; i++)
    for (j = 0; j <= n; j++) sum += (j + i);
  return sum;
}

int f2() {
  const int n = 3;
  int A[n];
  int sum = 0;
  int i;
  for (i = 0; i < n; i++) A[i] = i;               /* initialize the array */
  for (i = 1; i < n; i++) A[i] = A[i - 1] + A[i]; /* compute */
  for (i = 0; i < n; i++) sum += A[i];            /* obtain the sum */
  return sum;
}

int main() {
  std::cout << "f1 " << f1() << std::endl;
  std::cout << "f2 " << f2() << std::endl;
  return 0;
}
